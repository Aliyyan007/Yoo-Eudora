"""
Re-engagement and ping control module for the Eudora persona.

Features:
1. When the bot sends a message and no one replies, ping online/active users
2. @here usage: 2-3 times daily (algorithmic)
3. @everyone usage: 2 times weekly (algorithmic)
4. Control chat revive ping frequency (cooldown to prevent irritation)
5. Track ping usage per channel and globally

Algorithmic approach:
- After bot sends a message, wait for a reply (configurable timeout)
- If no reply, ping online users (not offline, unless they're active recently)
- @here: max 3 per day per channel, with 2h cooldown
- @everyone: max 2 per week per channel, with 24h cooldown
- Role ping (chat revive): max 1 per 30 min per channel (was too frequent)
- Track all pings to prevent over-pinging
"""
import os
import json
import time
import random
from pathlib import Path
from typing import Optional, List, Dict, Set
from collections import deque
from loguru import logger
import discord


class PingController:
    """
    Controls all ping usage to prevent irritation.

    Algorithmic limits:
    - @here: 3 per day per channel, 2h cooldown
    - @everyone: 2 per week per channel, 24h cooldown
    - Role ping (chat revive): 1 per 30 min per channel
    - Direct user ping: 1 per 10 min per channel
    - Total daily ping cap: 15 per channel
    """

    def __init__(self):
        # @here usage: channel_id -> list of timestamps
        self._here_pings: Dict[str, deque] = {}
        # @everyone usage: channel_id -> list of timestamps
        self._everyone_pings: Dict[str, deque] = {}
        # Role ping usage: channel_id -> list of timestamps
        self._role_pings: Dict[str, deque] = {}
        # Direct user ping usage: channel_id -> list of timestamps
        self._user_pings: Dict[str, deque] = {}
        # Recently-pinged users: channel_id -> deque[(user_id, ts)] — don't
        # re-ping the same person for a while (looked spammy when it happened)
        self._recently_pinged: Dict[str, deque] = {}
        # Consecutive channel pings (any type) that got NO human reply.
        # A dead channel otherwise gets pinged every cooldown forever —
        # sweep deletes the unanswered msg, next cycle fires again. Streak
        # multiplies cooldowns; ≥2 unanswered silences mass pings until a
        # human actually speaks.
        self._unanswered_pings: Dict[str, int] = {}
        try:
            self._ping_repeat_s = int(os.getenv("PING_REPEAT_COOLDOWN_HOURS", "2")) * 3600
        except ValueError:
            self._ping_repeat_s = 7200

        # Limits — env-tunable
        import os as _os
        def _ei(n, d):
            try:
                return int(_os.getenv(n, str(d)))
            except ValueError:
                return d
        self._here_daily_limit = _ei("PING_HERE_DAILY_LIMIT", 3)
        self._here_cooldown_s = _ei("PING_HERE_COOLDOWN_MIN", 120) * 60
        self._everyone_weekly_limit = _ei("PING_EVERYONE_WEEKLY_LIMIT", 1)  # ~once/week max
        self._everyone_cooldown_s = _ei("PING_EVERYONE_COOLDOWN_HOURS", 168) * 3600
        self._role_cooldown_s = _ei("PING_ROLE_COOLDOWN_MIN", 30) * 60
        self._role_daily_limit = _ei("PING_ROLE_DAILY_LIMIT", 4)
        self._user_cooldown_s = _ei("PING_USER_COOLDOWN_MIN", 10) * 60
        self._total_daily_limit = _ei("PING_TOTAL_DAILY_LIMIT", 15)

        # Persisted ping timelines — cooldowns survive restarts
        self._state_path = Path("data/ping_state.json")
        self._load_state()

    # ── Persistence ──────────────────────────────────────────────────────
    def _load_state(self):
        """Restore ping timestamps from disk so cooldowns survive restarts.
        Drops entries older than 8 days (all windows are shorter)."""
        try:
            if not self._state_path.exists():
                return
            data = json.loads(self._state_path.read_text())
            now = time.time()
            keep = 8 * 24 * 3600
            for name, tracker in [("here", self._here_pings),
                                  ("everyone", self._everyone_pings),
                                  ("role", self._role_pings),
                                  ("user", self._user_pings)]:
                for ch_id, ts_list in (data.get(name) or {}).items():
                    fresh = [float(t) for t in ts_list if now - float(t) < keep]
                    if fresh:
                        tracker[ch_id] = deque(fresh, maxlen=50)
            for ch_id, pairs in (data.get("recently") or {}).items():
                fresh = [(int(u), float(t)) for u, t in pairs
                         if now - float(t) < keep]
                if fresh:
                    self._recently_pinged[ch_id] = deque(fresh, maxlen=30)
            for ch_id, streak in (data.get("unanswered") or {}).items():
                self._unanswered_pings[ch_id] = int(streak)
            logger.debug(f"Ping state restored from {self._state_path}")
        except Exception as e:
            logger.debug(f"Ping state load failed (starting fresh): {e}")
        # Merge D1 state — survives redeploys where data/ is wiped (Render
        # ephemeral FS is exactly how @everyone escaped its weekly cap)
        try:
            from . import d1_memory as _mem
            remote = _mem.load_server_state("ping_state")
            if isinstance(remote, dict):
                self._merge_remote(remote)
        except Exception:
            pass

    def _merge_remote(self, data: dict):
        """Union remote (D1) ping state into local — take the STRICTER view
        (union of timestamps, max streak) so neither a wiped file nor a stale
        file can reset cooldowns."""
        for name, tracker in [("here", self._here_pings),
                              ("everyone", self._everyone_pings),
                              ("role", self._role_pings),
                              ("user", self._user_pings)]:
            for ch_id, ts_list in (data.get(name) or {}).items():
                merged = sorted(set(
                    float(t) for t in
                    list(tracker.get(ch_id, [])) + [float(t) for t in ts_list]
                ))[-50:]
                tracker[ch_id] = deque(merged, maxlen=50)
        for ch_id, pairs in (data.get("recently") or {}).items():
            merged = list(self._recently_pinged.get(ch_id, [])) + [
                (int(u), float(t)) for u, t in pairs]
            self._recently_pinged[ch_id] = deque(merged[-30:], maxlen=30)
        for ch_id, streak in (data.get("unanswered") or {}).items():
            self._unanswered_pings[ch_id] = max(
                self._unanswered_pings.get(ch_id, 0), int(streak))

    def _save_state(self):
        """Atomically persist all ping timelines to disk."""
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "here": {c: list(q) for c, q in self._here_pings.items()},
                "everyone": {c: list(q) for c, q in self._everyone_pings.items()},
                "role": {c: list(q) for c, q in self._role_pings.items()},
                "user": {c: list(q) for c, q in self._user_pings.items()},
                "recently": {c: [[u, t] for u, t in q]
                             for c, q in self._recently_pinged.items()},
                "unanswered": dict(self._unanswered_pings),
            }
            tmp = self._state_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data))
            tmp.replace(self._state_path)
            # Mirror to D1 — best-effort, survives redeploys (ephemeral FS)
            try:
                from . import d1_memory as _mem
                _mem.save_server_state("ping_state", data)
            except Exception:
                pass
        except Exception as e:
            logger.debug(f"Ping state save failed: {e}")

    def _clean_old(self, channel_id: str, tracker: Dict[str, deque], max_age_s: float):
        """Remove timestamps older than max_age_s from a tracker."""
        if channel_id not in tracker:
            return
        now = time.time()
        tracker[channel_id] = deque(
            [ts for ts in tracker[channel_id] if now - ts < max_age_s],
            maxlen=50
        )

    def _count_recent(self, channel_id: str, tracker: Dict[str, deque], max_age_s: float) -> int:
        """Count pings in the last max_age_s seconds."""
        self._clean_old(channel_id, tracker, max_age_s)
        return len(tracker.get(channel_id, []))

    def _last_ping_time(self, channel_id: str, tracker: Dict[str, deque]) -> float:
        """Get the last ping time for a channel."""
        pings = tracker.get(channel_id, deque())
        return pings[-1] if pings else 0

    # ── Unanswered-ping streaks ──────────────────────────────────────────
    # A role/@here/@everyone ping nobody responds to is a warning shot —
    # pinging again on the next cooldown tick is what made the bot feel
    # spammy/dangerous. Each unanswered ping multiplies the cooldown and
    # ≥2 unanswered mutes mass pings until a human actually speaks.

    def _unanswered(self, channel_id: str) -> int:
        return self._unanswered_pings.get(channel_id, 0)

    def note_human_activity(self, channel_id: str):
        """A human spoke in the channel — pings are working; reset streak."""
        if self._unanswered_pings.pop(channel_id, None) is not None:
            self._save_state()

    def _bump_unanswered(self, channel_id: str):
        self._unanswered_pings[channel_id] = self._unanswered(channel_id) + 1

    def _mass_ping_blocked(self, channel_id: str) -> bool:
        """Channel pings muted after 2 consecutive unanswered pings —
        resume only when humans actually talk again."""
        return self._unanswered(channel_id) >= 2

    def _cooldown_left(self, channel_id: str, tracker: Dict[str, deque],
                       base_s: float) -> float:
        """Seconds until the next ping is allowed — streak-scaled."""
        last = self._last_ping_time(channel_id, tracker)
        eff = base_s * (1 + 3 * self._unanswered(channel_id))
        return (last + eff) - time.time()

    def can_ping_here(self, channel_id: str) -> bool:
        """Check if @here can be used in this channel."""
        if self._mass_ping_blocked(channel_id):
            return False
        now = time.time()
        # Check daily limit
        daily_count = self._count_recent(channel_id, self._here_pings, 24 * 3600)
        if daily_count >= self._here_daily_limit:
            return False
        # Check cooldown (streak-scaled)
        if self._cooldown_left(channel_id, self._here_pings, self._here_cooldown_s) > 0:
            return False
        # Check total daily limit
        total = self._get_total_daily_count(channel_id)
        if total >= self._total_daily_limit:
            return False
        return True

    def can_ping_everyone(self, channel_id: str) -> bool:
        """Check if @everyone can be used in this channel."""
        if self._mass_ping_blocked(channel_id):
            return False
        now = time.time()
        # Check weekly limit
        weekly_count = self._count_recent(channel_id, self._everyone_pings, 7 * 24 * 3600)
        if weekly_count >= self._everyone_weekly_limit:
            return False
        # Check cooldown (streak-scaled)
        if self._cooldown_left(channel_id, self._everyone_pings, self._everyone_cooldown_s) > 0:
            return False
        # Check total daily limit
        total = self._get_total_daily_count(channel_id)
        if total >= self._total_daily_limit:
            return False
        return True

    def can_ping_role(self, channel_id: str) -> bool:
        """Check if a role ping can be used in this channel."""
        if self._mass_ping_blocked(channel_id):
            return False
        now = time.time()
        # Check daily limit
        daily_count = self._count_recent(channel_id, self._role_pings, 24 * 3600)
        if daily_count >= self._role_daily_limit:
            return False
        # Check cooldown (streak-scaled: 30m → 2h → muted)
        if self._cooldown_left(channel_id, self._role_pings, self._role_cooldown_s) > 0:
            return False
        # Check total daily limit
        total = self._get_total_daily_count(channel_id)
        if total >= self._total_daily_limit:
            return False
        return True

    def can_ping_user(self, channel_id: str) -> bool:
        """Check if a direct user ping can be used in this channel.
        User pings are lighter (one notification, one person) — they get a
        milder streak block (4 unanswered) and a smaller cooldown scale."""
        if self._unanswered(channel_id) >= 4:
            return False
        now = time.time()
        # Check cooldown (mild streak scale)
        last = self._last_ping_time(channel_id, self._user_pings)
        eff = self._user_cooldown_s * (1 + self._unanswered(channel_id))
        if now - last < eff:
            return False
        # Check total daily limit
        total = self._get_total_daily_count(channel_id)
        if total >= self._total_daily_limit:
            return False
        return True

    def _get_total_daily_count(self, channel_id: str) -> int:
        """Get total ping count in the last 24 hours."""
        now = time.time()
        total = 0
        for tracker in [self._here_pings, self._everyone_pings, self._role_pings, self._user_pings]:
            self._clean_old(channel_id, tracker, 24 * 3600)
            total += len(tracker.get(channel_id, []))
        return total

    def record_here_ping(self, channel_id: str):
        """Record an @here ping."""
        if channel_id not in self._here_pings:
            self._here_pings[channel_id] = deque(maxlen=50)
        self._here_pings[channel_id].append(time.time())
        self._bump_unanswered(channel_id)
        self._save_state()
        logger.info(f"@here ping recorded in #{channel_id}")

    def record_everyone_ping(self, channel_id: str):
        """Record an @everyone ping."""
        if channel_id not in self._everyone_pings:
            self._everyone_pings[channel_id] = deque(maxlen=50)
        self._everyone_pings[channel_id].append(time.time())
        self._bump_unanswered(channel_id)
        self._save_state()
        logger.info(f"@everyone ping recorded in #{channel_id}")

    def record_role_ping(self, channel_id: str):
        """Record a role ping."""
        if channel_id not in self._role_pings:
            self._role_pings[channel_id] = deque(maxlen=50)
        self._role_pings[channel_id].append(time.time())
        self._bump_unanswered(channel_id)
        self._save_state()
        logger.info(f"Role ping recorded in #{channel_id}")

    def record_user_ping(self, channel_id: str, user_id: int = None):
        """Record a direct user ping."""
        if channel_id not in self._user_pings:
            self._user_pings[channel_id] = deque(maxlen=50)
        self._user_pings[channel_id].append(time.time())
        if user_id is not None:
            dq = self._recently_pinged.setdefault(channel_id, deque(maxlen=30))
            dq.append((int(user_id), time.time()))
        self._bump_unanswered(channel_id)
        self._save_state()
        logger.info(f"User ping recorded in #{channel_id}")

    def recently_pinged(self, channel_id: str) -> set:
        """User ids pinged within the repeat-cooldown window — exclude them
        from the next selection so the same person isn't pinged twice in a row."""
        now = time.time()
        return {uid for uid, ts in self._recently_pinged.get(channel_id, ())
                if now - ts < self._ping_repeat_s}

    def reset_daily(self):
        """Reset daily counters (call at midnight)."""
        # Don't clear everything — weekly counters need to persist
        now = time.time()
        for tracker in [self._here_pings, self._role_pings, self._user_pings]:
            for ch_id in list(tracker.keys()):
                tracker[ch_id] = deque(
                    [ts for ts in tracker[ch_id] if now - ts < 24 * 3600],
                    maxlen=50
                )


# Singleton instance
_ping_controller = PingController()


def get_ping_controller() -> PingController:
    """Get the global ping controller."""
    return _ping_controller


# ── Re-engagement logic ──────────────────────────────────────────────────────

class ReEngagementTracker:
    """
    Tracks when the bot sends messages and whether anyone replies.

    Algorithmic behavior:
    - After bot sends a message, record the timestamp
    - If no one replies within the timeout, trigger re-engagement
    - Re-engagement: ping online users, use @here, or use @everyone
    - Escalation: first try direct ping, then @here, then @everyone
    """

    def __init__(self):
        # channel_id -> last bot message timestamp
        self._last_bot_msg: Dict[str, float] = {}
        # channel_id -> whether anyone replied after bot's last message
        self._got_reply: Dict[str, bool] = {}
        # channel_id -> last ping strategy used (variety — don't repeat)
        self._last_action: Dict[str, str] = {}
        # Re-engagement timeout: 10 minutes
        self._timeout_s = 600

    def record_bot_message(self, channel_id: str):
        """Record that the bot sent a message."""
        self._last_bot_msg[channel_id] = time.time()
        self._got_reply[channel_id] = False

    def record_human_reply(self, channel_id: str):
        """Record that a human replied in the channel."""
        self._got_reply[channel_id] = True
        # Humans are talking — reset the unanswered-ping streak so pings
        # are allowed again in this channel
        _ping_controller.note_human_activity(channel_id)

    def needs_re_engagement(self, channel_id: str) -> bool:
        """
        Check if the channel needs re-engagement.

        Returns True if:
        - Bot sent a message
        - No one replied within the timeout
        """
        if channel_id not in self._last_bot_msg:
            return False

        if self._got_reply.get(channel_id, True):
            return False

        last_bot = self._last_bot_msg[channel_id]
        time_since = time.time() - last_bot

        return time_since > self._timeout_s

    def get_re_engagement_action(self, channel_id: str) -> str:
        """
        Algorithmically determine the re-engagement action.

        Escalation:
        1. First: ping a random online user
        2. Second: use @here
        3. Third: use @everyone (rare, weekly limit)

        Returns: "user_ping", "here", "everyone", or "none"
        """
        ping_ctrl = get_ping_controller()

        # Check what pings are available
        can_user = ping_ctrl.can_ping_user(channel_id)
        can_here = ping_ctrl.can_ping_here(channel_id)
        can_everyone = ping_ctrl.can_ping_everyone(channel_id)

        # Algorithmic escalation
        # 60% chance: try user ping first
        # 30% chance: try @here
        # 10% chance: try @everyone (if available)

        if can_everyone and random.random() < 0.10:
            action = "everyone"
        elif can_here and random.random() < 0.30:
            action = "here"
        elif can_user:
            action = "user_ping"
        # Fallback: try whatever is available
        elif can_here:
            action = "here"
        elif can_everyone:
            action = "everyone"
        elif can_user:
            action = "user_ping"
        else:
            action = "none"

        # Strategy variety — don't repeat the same ping strategy back-to-back.
        # If we rolled the same action as last time, 70% chance to pick a
        # different available one instead.
        last = self._last_action.get(channel_id)
        if action != "none" and action == last and random.random() < 0.70:
            alts = [a for a, ok in (("user_ping", can_user), ("here", can_here), ("everyone", can_everyone)) if ok and a != action]
            if alts:
                action = random.choice(alts)
        if action != "none":
            self._last_action[channel_id] = action
        return action

    def clear_channel(self, channel_id: str):
        """Clear re-engagement state for a channel."""
        self._last_bot_msg.pop(channel_id, None)
        self._got_reply.pop(channel_id, None)


# Singleton instance
_re_engagement = ReEngagementTracker()


def get_re_engagement_tracker() -> ReEngagementTracker:
    """Get the global re-engagement tracker."""
    return _re_engagement


# ── Online user selection ────────────────────────────────────────────────────

def select_online_user(
    guild: discord.Guild,
    exclude_ids: Set[int] = None,
    channel: Optional[discord.abc.GuildChannel] = None,
    history_msgs=None,
) -> Optional[discord.Member]:
    """
    Algorithmically select an online user to ping for re-engagement.

    Selection criteria:
    - Must be online or idle (not offline)
    - Must not be a bot
    - Must not be the bot itself
    - Must be able to SEE the channel (if channel is provided) — prevents
      pinging users in private/restricted channels they can't access
    - Prefer users who have been active recently (in the last 24h)
    - 20% chance to select an offline but recently active user
    - Recent chat participants from `history_msgs` get bonus weight — on a
      self-bot `guild.members` is just the thin member cache, so without it
      selection skews toward whichever few members happen to be cached
    """
    if exclude_ids is None:
        exclude_ids = set()

    # Don't re-ping someone pinged recently — falls back to them only if
    # literally everyone else has been pinged too
    recent_ping_ids = _ping_controller.recently_pinged(str(channel.id)) if channel is not None else set()

    online_users = []
    offline_active = []
    fallback_pool = []  # recently-pinged members, used only as last resort
    recent_speakers = []  # deduped humans seen talking in this channel's history
    _seen_speakers = set()

    try:
        for member in guild.members:
            if member.bot or member.id in exclude_ids:
                continue

            if member.id in recent_ping_ids:
                # visible-channel check still applies for the fallback pool
                if channel is not None:
                    try:
                        perms = channel.permissions_for(member)
                        if not perms.read_messages or not perms.view_channel:
                            continue
                    except Exception:
                        continue
                if member.status in (discord.Status.online, discord.Status.idle, discord.Status.dnd):
                    fallback_pool.append(member)
                continue

            # CRITICAL: Check if the user can actually see the channel
            # This prevents pinging users in private channels they can't access
            if channel is not None:
                try:
                    perms = channel.permissions_for(member)
                    if not perms.read_messages or not perms.view_channel:
                        continue  # User can't see this channel — don't ping them
                except Exception:
                    continue  # Can't determine permissions — skip to be safe

            # Check if online/idle/dnd
            if member.status in (discord.Status.online, discord.Status.idle, discord.Status.dnd):
                online_users.append(member)
            elif member.status == discord.Status.offline:
                # Check if they were active recently (joined voice or sent messages)
                # We can't easily check message history here, but we can check
                # if they have a recent activity (e.g., joined in the last month)
                # For now, add to offline_active with lower priority
                offline_active.append(member)
    except Exception:
        pass

    # Humans who actually spoke in this channel recently — far better ping
    # targets than the raw member cache. On a self-bot `guild.members` only
    # holds whoever the gateway happened to cache, which is how one active
    # member ends up pinged over and over.
    if history_msgs:
        for m in reversed(list(history_msgs)):
            author = getattr(m, "author", None)
            if author is None or getattr(author, "bot", False):
                continue
            if author.id in exclude_ids or author.id in _seen_speakers:
                continue
            if author.id in recent_ping_ids:
                continue
            member = guild.get_member(author.id) or author
            if getattr(member, "bot", False):
                continue
            if channel is not None:
                try:
                    perms = channel.permissions_for(member)
                    if not perms.read_messages or not perms.view_channel:
                        continue
                except Exception:
                    continue
            _seen_speakers.add(member.id)
            recent_speakers.append(member)

    # Weighted candidate pool — favors OFFLINE/lurker weight via
    # PING_ONLINE_RATIO (0-1, default 0.40 → ~60% offline pull), with a
    # strong bonus for users recently seen talking in this channel.
    import os as _os
    try:
        online_ratio = float(_os.getenv("PING_ONLINE_RATIO", "0.40"))
    except ValueError:
        online_ratio = 0.40
    online_ratio = max(0.0, min(1.0, online_ratio))

    weighted: Dict[int, list] = {}

    def _add(member, w):
        if w <= 0:
            return
        entry = weighted.setdefault(member.id, [member, 0.0])
        entry[1] += w

    for m in online_users:
        _add(m, online_ratio)
    for m in offline_active:
        _add(m, 1.0 - online_ratio)
    for m in recent_speakers:
        _add(m, 3.0)  # proven talkers in this channel dominate the pool

    if weighted:
        pool = [m for m, _ in weighted.values()]
        wts = [w for _, w in weighted.values()]
        return random.choices(pool, weights=wts, k=1)[0]
    if fallback_pool:
        # every unpinged candidate was exhausted — re-ping is acceptable
        return random.choice(fallback_pool)

    return None
