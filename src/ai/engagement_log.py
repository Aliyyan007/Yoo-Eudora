"""Stale engagement message management — prevents walls of unanswered bot messages.

The bot sends proactive engagement (chat revives, "anyone up for X?", @here
pings, post-bump messages). Historically it never tracked them, so failed
attempts stacked into walls. This module:

- records every engagement message the bot sends (per channel deque)
- before the next engagement send, sweeps the channel:
  * counts human responses after each tracked message
  * protects messages that generated interaction (any reply/responders)
  * deletes stale unanswered ones (age > stale threshold), oldest first,
    with jittered delays so deletions don't look robotic
  * enforces a soft cap on unanswered engagement messages per channel
- tracks consecutive failed engagement cycles per channel and pauses
  engagement there until humans speak again (kills the self-re-arming loop)

All thresholds are env-tunable; deleting its own messages needs no channel
perms. Zero extra LLM/API usage — pure bookkeeping + occasional history fetch.
"""
import asyncio
import os
import random
import time
from collections import deque
from typing import Dict, Optional

import discord
from loguru import logger


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


class EngagementLog:
    """Per-channel record of the bot's engagement messages + stale sweeper."""

    def __init__(self):
        # ch_id -> deque of {"msg_id", "ts", "kind", "msg"}
        self._tracked: Dict[str, deque] = {}
        # ch_id -> consecutive unanswered engagement cycles
        self._fails: Dict[str, int] = {}
        # ch_id -> ts when channel was paused
        self._paused_since: Dict[str, float] = {}
        # ch_id -> ts of last sweep (don't sweep twice in a row instantly)
        self._last_sweep: Dict[str, float] = {}
        # ch_id -> ts of last engagement send (SHARED cooldown across senders —
        # revive/auto-chat/post-bump/proactive/re-engage each had own timers,
        # which is how walls formed)
        self._last_send: Dict[str, float] = {}

        # ── tunables (env) ──
        self.stale_s = _env_int("ENGAGE_STALE_MINUTES", 15) * 60
        # grace before a message is even eligible for deletion
        self.grace_s = max(300, self.stale_s // 3)
        # soft cap on unanswered engagement messages per channel (not rigid —
        # active conversations can hold more; see sweep scoring)
        self.max_unanswered = _env_int("ENGAGE_MAX_UNANSWERED", 4)
        # consecutive fully-failed cycles before the channel gets paused
        self.pause_after_fails = _env_int("ENGAGE_PAUSE_AFTER_FAILS", 2)
        # how long a pause lasts without human activity (also ends instantly
        # on human interaction via mark_interaction)
        self.pause_s = _env_int("ENGAGE_PAUSE_MINUTES", 120) * 60
        # min gap between ANY two engagement sends in a channel — all senders
        # share this (5 independent timers is how the wall happened)
        self.min_send_gap_s = _env_int("ENGAGE_MIN_GAP_MIN", 8) * 60
        # max deletions per sweep (rate-limit friendly)
        self.max_deletes_per_sweep = _env_int("ENGAGE_MAX_DELETES", 3)
        # min gap between sweeps per channel
        self.sweep_cooldown_s = 120

    # ── recording ────────────────────────────────────────────────────────
    def record(self, ch_id: str, msg: discord.Message, kind: str = "engage") -> None:
        """Track a bot engagement message for later staleness evaluation."""
        if msg is None:
            return
        dq = self._tracked.setdefault(ch_id, deque(maxlen=16))
        dq.append({
            "msg_id": msg.id,
            "ts": msg.created_at.timestamp(),
            "kind": kind,
            "msg": msg,
        })

    def mark_interaction(self, ch_id: str) -> None:
        """A human spoke in the channel — engagement worked / is no longer needed."""
        self._fails[ch_id] = 0
        self._paused_since.pop(ch_id, None)

    def recent_engagement(self, ch_id: str, within_s: float = 240) -> bool:
        """True if the bot posted an engagement message here recently —
        used to boost reply probability to messages that respond to it."""
        dq = self._tracked.get(ch_id)
        return bool(dq) and (time.time() - dq[-1]["ts"]) < within_s

    def unanswered_count(self, ch_id: str, history_msgs=None, bot_user=None) -> int:
        """Best-effort count of tracked msgs with no visible responses —
        plus untracked own msgs still visible in history (post-restart walls)."""
        bot_id = getattr(bot_user, "id", None)
        dq = self._tracked.get(ch_id)
        tracked_ids = {rec["msg_id"] for rec in dq} if dq else set()
        n = sum(1 for rec in dq if not self._had_response(rec, history_msgs, bot_id)) if dq else 0
        if history_msgs and bot_user is not None:
            now = time.time()
            human_tss = sorted(
                m.created_at.timestamp() for m in history_msgs
                if not m.author.bot and m.author.id != bot_id
            )
            for m in history_msgs:
                if m.author.id != bot_user.id or m.id in tracked_ids:
                    continue
                ts = m.created_at.timestamp()
                if now - ts >= self.stale_s and not any(ht > ts for ht in human_tss):
                    n += 1
        return n

    def can_send_engagement(self, ch_id: str, last_human_ts: float = 0.0,
                            min_gap_s: float = None, history_msgs=None,
                            bot_user=None) -> bool:
        """Unified gate every engagement sender must pass: paused? wall full?
        sent too recently? Failing any of these → don't send."""
        if self.is_paused(ch_id, last_human_ts):
            return False
        gap = self.min_send_gap_s if min_gap_s is None else min_gap_s
        now = time.time()
        if now - self._last_send.get(ch_id, 0) < gap:
            return False
        # Already sitting on a stack of unanswered engagement msgs → stop
        # adding to the wall until humans respond or they go stale
        if self.unanswered_count(ch_id, history_msgs, bot_user) >= self.max_unanswered:
            return False
        return True

    def mark_sent(self, ch_id: str) -> None:
        """Record that an engagement message just went out (shared cooldown)."""
        self._last_send[ch_id] = time.time()

    def is_paused(self, ch_id: str, last_human_ts: float = 0.0) -> bool:
        """Channel is in cooldown after repeated failed engagement cycles.
        Un-pauses on human interaction (mark_interaction) or pause expiry."""
        since = self._paused_since.get(ch_id)
        if since is None:
            return False
        if time.time() - since > self.pause_s:
            self._paused_since.pop(ch_id, None)
            self._fails[ch_id] = 0
            return False
        # human activity after the pause started → unpause
        if last_human_ts and last_human_ts > since:
            self._paused_since.pop(ch_id, None)
            self._fails[ch_id] = 0
            return False
        return True

    # ── response detection ───────────────────────────────────────────────
    def _had_response(self, rec: dict, history_msgs=None, bot_id=None) -> bool:
        """Did any human meaningfully respond after this message?
        Uses a fresh channel history slice if provided, else the stored
        message's own reply/reaction data (weak but free).

        NOTE: this is a self-bot — our own messages have author.bot=False,
        so `author.id == bot_id` must be excluded explicitly or our next
        engagement message counts as a "response" to the previous one."""
        ts = rec["ts"]
        if bot_id is None:
            bot_id = getattr(getattr(rec.get("msg"), "author", None), "id", None)
        if history_msgs is not None:
            for m in history_msgs:
                if m.created_at.timestamp() <= ts:
                    continue
                if m.author.bot:
                    continue
                if bot_id is not None and m.author.id == bot_id:
                    continue  # our own message — not a human response
                # direct reply to this specific message
                ref = getattr(m, "reference", None)
                if ref is not None and getattr(ref, "message_id", None) == rec["msg_id"]:
                    return True
                # any human message after it counts as engagement context
                return True
            return False
        # fallback: reactions on the stored object (only real if re-fetched)
        msg = rec.get("msg")
        try:
            if msg and any(r.count > 0 for r in msg.reactions):
                return True
        except Exception:
            pass
        return False

    # ── sweep ────────────────────────────────────────────────────────────
    async def sweep_before_send(self, channel, bot_user, force: bool = False,
                                history_msgs=None) -> int:
        """Clean stale unanswered engagement messages in `channel` before the
        next send. Returns the number of unanswered tracked messages remaining.
        Never deletes messages that generated interaction."""
        ch_id = str(channel.id)
        tracked = self._tracked.get(ch_id)
        now = time.time()
        # No tracked msgs AND nothing to verify in history → nothing to do.
        # NOTE: we still fetch history even with empty tracked — after a
        # restart `tracked` is empty while old wall msgs still exist.
        if not tracked and not force and \
                (now - self._last_sweep.get(ch_id, 0)) < self.sweep_cooldown_s:
            return 0
        if tracked and not force and \
                (now - self._last_sweep.get(ch_id, 0)) < self.sweep_cooldown_s:
            return self.unanswered_count(ch_id)
        self._last_sweep[ch_id] = now

        # One history fetch classifies every tracked message. If it fails we
        # fall back to the client's cached history; if we still can't see the
        # channel, never delete blind.
        hist = None
        try:
            hist = [m async for m in channel.history(limit=60)]
        except Exception as e:
            logger.debug(f"[engage] history fetch failed for sweep: {e}")
        if hist is None and history_msgs:
            hist = list(history_msgs)
        if hist is None:
            return len(tracked) if tracked else 0

        tracked = tracked or deque()

        deleted = 0
        bot_id = getattr(bot_user, "id", None)
        # snapshot BEFORE deletion drains the counts — a "failed cycle" is
        # judged on what was there when we arrived, not what's left after
        tracked_at_start = len(tracked)
        got_any_response = False
        if hist:
            last_human_ts = max(
                (m.created_at.timestamp() for m in hist
                 if not m.author.bot and m.author.id != bot_id),
                default=0.0,
            )
        else:
            last_human_ts = 0.0

        # Partition tracked msgs: responded (protected + untracked), fresh
        # (kept, not deletable), stale candidates (deletion-eligible)
        fresh_count = 0
        stale_all = []  # (ts, tracked_rec_or_None, message)
        for rec in list(tracked):
            age = now - rec["ts"]
            if self._had_response(rec, hist, bot_id):
                # it worked — protect it and stop tracking (keep the wall clean
                # of bookkeeping, keep the message)
                got_any_response = True
                tracked.remove(rec)
                continue
            if age >= self.stale_s:
                stale_all.append((rec["ts"], rec, rec["msg"]))
            else:
                fresh_count += 1

        # Restart resilience: `tracked` is in-memory — messages sent before a
        # restart/deploy are invisible to it (this is how old walls survived).
        # Classify OWN messages straight from the fetched history with the
        # same rule: stale + no human message after it = wall material.
        tracked_ids = {rec["msg_id"] for rec in tracked}
        human_tss = sorted(
            m.created_at.timestamp() for m in hist
            if not m.author.bot and m.author.id != bot_id
        )
        for m in hist:
            if bot_user is None or m.author.id != bot_user.id:
                continue
            if m.id in tracked_ids:
                continue
            ts = m.created_at.timestamp()
            if (now - ts) < self.stale_s:
                continue
            if any(ht > ts for ht in human_tss):
                continue  # a human spoke after it → treated as answered
            stale_all.append((ts, None, m))
        stale_all.sort(key=lambda t: t[0])  # oldest first

        # Delete oldest-first while the surviving pile still exceeds the cap
        # (and anything very old regardless), bounded by max_deletes_per_sweep
        total_unanswered = fresh_count + len(stale_all)
        unanswered = fresh_count
        for ts, rec, msg in stale_all:
            age = now - ts
            very_old = age > self.stale_s * 3
            if deleted < self.max_deletes_per_sweep and \
                    (total_unanswered - deleted > self.max_unanswered or very_old):
                try:
                    await msg.delete()
                    deleted += 1
                    if rec is not None:
                        tracked.remove(rec)
                    logger.info(f"[engage] deleted stale message in #{getattr(channel, 'name', ch_id)} (age {int(age/60)}m)")
                    await asyncio.sleep(random.uniform(2.5, 6.0))
                    continue
                except discord.NotFound:
                    if rec is not None:
                        tracked.remove(rec)
                    continue
                except discord.HTTPException as e:
                    logger.debug(f"[engage] delete failed: {e}")
                    if getattr(e, "status", None) == 429:
                        break  # rate limited — stop sweeping
            unanswered += 1

        # consecutive-failure bookkeeping → channel pause. A cycle counts as
        # failed when the channel held a real batch of unanswered engagement
        # AND humans still didn't talk.
        human_recent = bool(hist) and last_human_ts > (now - self.pause_s)
        if (tracked_at_start >= self.max_unanswered or unanswered + deleted >= self.max_unanswered) \
                and not got_any_response and not human_recent:
            self._fails[ch_id] = self._fails.get(ch_id, 0) + 1
            if self._fails[ch_id] >= self.pause_after_fails and ch_id not in self._paused_since:
                self._paused_since[ch_id] = now
                logger.info(f"[engage] pausing engagement in #{getattr(channel, 'name', ch_id)} — {self._fails[ch_id]} unanswered cycles")
        elif got_any_response or human_recent:
            self._fails[ch_id] = 0

        return unanswered


_log = None


def get_engagement_log() -> EngagementLog:
    global _log
    if _log is None:
        _log = EngagementLog()
    return _log
