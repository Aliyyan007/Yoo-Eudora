"""
Proactive messaging — aggressive dead-chat revival and random auto-chat.
Finds dead channels and revives them with natural conversation starters.
Also sends random messages to keep the bot feeling alive even when no one is talking.
"""
import asyncio
import random
import time
from typing import List, Optional, Set
from collections import deque
from loguru import logger
import discord

from .ai import reply as ai_reply
from .ai import output_guard
from .ai import d1_memory as mem  # D1-backed memory (falls back to JSON if D1 unavailable)
from .ai import engagement_topics, human_timing
from .channel_scanner import can_speak_in, is_skip_channel, find_most_active_channel
from .ai.re_engagement import get_ping_controller, get_re_engagement_tracker, select_online_user
from .ai.engagement_log import get_engagement_log


def _persona_id(client) -> str:
    """Active persona id for pool/pace lookups ('' if unset)."""
    return getattr(getattr(client, "persona", None), "id", "") or ""


def _recent_lines(client, ch_id: str, n: int = 6):
    """Last n human message contents from the cached channel history —
    grounding context for the topic engine's channel_callback genre."""
    try:
        own = getattr(getattr(client, "user", None), "id", None)
        out = []
        for m in list(client.history_cache.get(ch_id, ())):
            if m.content and getattr(getattr(m, "author", None), "id", None) != own:
                out.append(m.content)
        return out[-n:]
    except Exception:
        return []


def _fallback_or(msg: Optional[str], persona_id: str, kind: str, ch_id: str) -> str:
    """Return the LLM message if usable, else a persona-voiced canned line
    drawn via shuffle-bag (every line used once before any repeat)."""
    if msg and len(msg) >= 3 and not output_guard.is_degenerate(msg) \
            and not output_guard.leaks_internals(msg):
        return msg
    return engagement_topics.fallback_line(persona_id, kind, ch_id)


class ProactiveMessenger:
    """
    Monitors channel activity and sends proactive messages to keep chat alive.
    Three modes:
    1. Dead chat revival — when a channel has been silent for X minutes
    2. Random auto-chat — periodically sends spontaneous messages to active channels
    3. Post-bump engagement — sends messages after bump batches
    """

    def __init__(
        self,
        client: discord.Client,
        channel_ids: Set[int],
        ping_role_id: Optional[int] = None,
        dead_chat_threshold_min: int = 20,  # Increased from 12 to reduce ping frequency
        check_interval_min: int = 5,
        proactive_interval_min: tuple = (30, 60),
        auto_chat_chance: float = 0.25,  # Reduced from 0.35
    ):
        self.client = client
        self.channel_ids = channel_ids
        self.ping_role_id = ping_role_id
        self.dead_chat_threshold = dead_chat_threshold_min * 60
        self.check_interval = check_interval_min * 60
        self.proactive_interval = proactive_interval_min
        self.auto_chat_chance = auto_chat_chance
        self._running = False
        # Owner-kill-switch for pings: discord_client's stop_pinging command
        # sets this False — revives still go out, just without any
        # role/@here/user ping prefixes.
        self.pings_enabled = True
        self.last_proactive = 0
        self.last_auto_chat = time.time()
        self.last_activity: dict = {}  # channel_id -> timestamp
        # Warmup: no proactive/revive sends for a few minutes after coming
        # online — a person doesn't message the second they connect, and on
        # rotation the new account inherits genuinely-quiet channels.
        self._warmup_until = time.time() + random.uniform(360, 840)

    async def start_monitoring(self):
        """Start the proactive monitoring loop."""
        self._running = True
        logger.info(f"Proactive messaging started for {len(self.channel_ids)} channels")
        logger.info(f"Dead chat threshold: {self.dead_chat_threshold // 60}min | "
                    f"Check interval: {self.check_interval // 60}min | "
                    f"Auto-chat chance: {self.auto_chat_chance:.0%}")
        await self.client.wait_until_ready()

        while self._running and not self.client.is_closed():
            try:
                await self._monitor_loop()
            except Exception as e:
                logger.error(f"Proactive monitor error: {e}")
            # Jittered check interval — a metronome tick is detectable
            await asyncio.sleep(self.check_interval * random.uniform(0.7, 1.6))

    async def stop(self):
        """Stop the proactive messaging loop."""
        self._running = False

    def record_activity(self, channel_id: int):
        """Record that a message was seen in a channel."""
        self.last_activity[str(channel_id)] = time.time()
        # NOTE: called for bot msgs too — human-only interaction marking
        # happens in discord_client's `not author.bot` tracking block

    async def _last_seen(self, channel_id: int) -> float:
        """Real last-activity timestamp for a channel. A fresh client has an
        empty last_activity — without this every channel reads as dead-forever
        and gets revived/pinged seconds after login (looks bot-like, and on
        rotation the new account re-pings a channel that was already pinged)."""
        ch_id = str(channel_id)
        ts = self.last_activity.get(ch_id, 0)
        if ts:
            return ts
        last = 0.0
        try:
            channel = self.client.get_channel(channel_id)
            if channel is not None:
                async for m in channel.history(limit=1):
                    last = m.created_at.timestamp()
        except Exception:
            pass
        # Channel with no history: count from now — a never-active channel
        # earns its silence like any other instead of being instantly "dead".
        ts = last or time.time()
        self.last_activity[ch_id] = ts
        return ts

    async def _monitor_loop(self):
        """
        Check all channels for:
        1. Dead chat (silent for > threshold) — revive with ping role
        2. Inactive chat (silent for > 5 min but < threshold) — random auto-chat
        """
        now = time.time()
        if now < self._warmup_until:
            return  # just came online — lurk before posting

        for channel_id in self.channel_ids:
            ch_id = str(channel_id)
            last_msg = await self._last_seen(channel_id)
            silence_duration = now - last_msg

            # 1. Dead chat revival (silent for > threshold)
            if silence_duration > self.dead_chat_threshold:
                # Only revive if we haven't sent a proactive message in the last 30 min
                # (increased from 5 min to reduce ping irritation)
                if (now - self.last_proactive) > 1800:
                    if await self._revive_dead_chat(channel_id):
                        self.last_proactive = now
                        return  # One revival per cycle

            # 2. Random auto-chat (channel has some activity but is slowing down)
            # Send a random message with auto_chat_chance probability
            # Only if: silence > 5 min, we haven't auto-chatted in 10+ min,
            # AND real humans actually spoke here in the last 30 min —
            # chatting into a bot-only/empty room is pure bot behavior.
            elif silence_duration > 300 and (now - self.last_auto_chat) > 600:
                if random.random() < self.auto_chat_chance:
                    try:
                        from .persona import runtime as _prt
                        own_ids = set(_prt.own_user_ids())
                    except Exception:
                        own_ids = set()
                    humans_30m = sum(
                        1 for m in self.client.history_cache.get(ch_id, ())
                        if not getattr(m.author, "bot", False)
                        and m.author.id not in own_ids
                        and (now - m.created_at.timestamp()) < 1800)
                    if humans_30m >= 2:
                        await self._send_random_chat(channel_id)
                        self.last_auto_chat = now
                    return  # One auto-chat per cycle

    async def _revive_dead_chat(self, channel_id: int) -> bool:
        """Send a message to revive a dead channel (with ping role).
        Uses ping controller to prevent over-pinging.
        Returns True only when a message was actually sent — the caller sets
        last_proactive on it so a gated/blocked revive doesn't burn the
        global 30-min revive window."""
        channel = self.client.get_channel(channel_id)
        if not channel:
            return False

        # Check permissions
        if hasattr(channel, 'guild') and channel.guild:
            if not can_speak_in(channel, channel.guild.me):
                return False

        ch_id = str(channel_id)
        eng_log = get_engagement_log()
        # Sweep stale unanswered engagement messages BEFORE the gate — if the
        # wall is over cap, gating first would return early and the stale
        # messages would never get cleaned (deadlock: blocked AND never drained)
        await eng_log.sweep_before_send(channel, self.client.user,
                                        history_msgs=self.client.history_cache.get(ch_id))
        # Unified gate: paused / shared per-channel send cooldown / wall full
        if not eng_log.can_send_engagement(ch_id, last_human_ts=self.last_activity.get(ch_id, 0),
                                            history_msgs=self.client.history_cache.get(ch_id),
                                            bot_user=self.client.user):
            logger.debug(f"[engage] #{channel.name} gated — skipping revive")
            return False

        ping_ctrl = get_ping_controller()
        loop = asyncio.get_running_loop()

        # Decide the ping target FIRST — the message text is generated with
        # the pinged user's name in context so it never attributes personal
        # facts to them ("got any sketches?" to someone who never sketched).
        ping_prefix = ""
        pinged_user = None
        role = None
        if self.pings_enabled:
            if self.ping_role_id and hasattr(channel, 'guild') and channel.guild:
                role = channel.guild.get_role(self.ping_role_id)
                if role is None:
                    # ID may be stale — fall back to a name match ("chat revive",
                    # "revive", ...) so a bad/stale id never silently disables it
                    for r in channel.guild.roles:
                        if "revive" in (r.name or "").lower():
                            role = r
                            logger.info(f"[engage] revive role id stale — matched '{r.name}' by name")
                            break
                    if role is None:
                        logger.debug(f"[engage] CHAT_REVIVE_PING_ROLE {self.ping_role_id} not found in {channel.guild.name}")
            if role and ping_ctrl.can_ping_role(ch_id):
                ping_prefix = f"<@&{role.id}> "
            elif ping_ctrl.can_ping_here(ch_id) and random.random() < 0.30:
                ping_prefix = "@here "
            elif ping_ctrl.can_ping_user(ch_id) and hasattr(channel, 'guild') and channel.guild:
                pinged_user = select_online_user(
                    channel.guild, exclude_ids={self.client.user.id}, channel=channel,
                    history_msgs=self.client.history_cache.get(ch_id))
                if pinged_user:
                    ping_prefix = f"<@{pinged_user.id}> "

        # Try AI-generated message first (topic-engine driven), fall back to
        # the persona-voiced shuffle-bag pool when the LLM is unavailable
        topic = await loop.run_in_executor(
            None, lambda: mem.get_channel_topic_fresh(ch_id))
        pid = _persona_id(self.client)
        message = await loop.run_in_executor(
            None, lambda: ai_reply.generate_proactive_message(
                topic, for_user=getattr(pinged_user, "display_name", None),
                channel_id=ch_id, recent_lines=_recent_lines(self.client, ch_id),
                persona_id=pid)
        )
        message = ping_prefix + _fallback_or(message, pid, "dead", ch_id)

        if role and ping_prefix.startswith("<@&"):
            await loop.run_in_executor(None, ping_ctrl.record_role_ping, ch_id)
        elif ping_prefix.startswith("@here"):
            await loop.run_in_executor(None, ping_ctrl.record_here_ping, ch_id)
        elif pinged_user:
            await loop.run_in_executor(
                None, lambda: ping_ctrl.record_user_ping(ch_id, user_id=pinged_user.id))
            if hasattr(self.client, 'conversation_tracker'):
                if ch_id not in self.client.conversation_tracker:
                    self.client.conversation_tracker[ch_id] = {}
                self.client.conversation_tracker[ch_id][str(pinged_user.id)] = time.time()

        # If the sweep just deleted stale lines, let the channel breathe a
        # moment — old msgs vanishing the same second a new one appears is a tell
        if eng_log.seconds_since_delete(ch_id) < 45:
            await asyncio.sleep(random.uniform(8, 30))

        # Send with typing simulation (proportional to message length)
        typing_dur = human_timing.typing_seconds(message, persona_id=pid)
        async with channel.typing():
            await asyncio.sleep(typing_dur)
        sent = await channel.send(message)
        eng_log.mark_sent(ch_id)
        eng_log.record(ch_id, sent, kind="revive")
        logger.info(f"Revived dead chat in #{channel.name}: {message}")
        return True

    async def _send_random_chat(self, channel_id: int):
        """Send a spontaneous message to a channel (no ping role).
        LLM-first through the topic engine so the content is fresh each
        time; persona-voiced canned line only when the LLM is unavailable."""
        channel = self.client.get_channel(channel_id)
        if not channel:
            return

        # Check permissions
        if hasattr(channel, 'guild') and channel.guild:
            if not can_speak_in(channel, channel.guild.me):
                return

        ch_id = str(channel_id)
        eng_log = get_engagement_log()
        await eng_log.sweep_before_send(channel, self.client.user,
                                        history_msgs=self.client.history_cache.get(ch_id))
        if not eng_log.can_send_engagement(ch_id, last_human_ts=self.last_activity.get(ch_id, 0),
                                            history_msgs=self.client.history_cache.get(ch_id),
                                            bot_user=self.client.user):
            logger.debug(f"[engage] #{channel.name} gated — skipping auto-chat")
            return

        loop = asyncio.get_running_loop()
        pid = _persona_id(self.client)
        topic = await loop.run_in_executor(
            None, lambda: mem.get_channel_topic_fresh(ch_id))
        message = await loop.run_in_executor(
            None, lambda: ai_reply.generate_proactive_message(
                topic, channel_id=ch_id,
                recent_lines=_recent_lines(self.client, ch_id),
                persona_id=pid)
        )
        message = _fallback_or(message, pid, "chat", ch_id)

        # Settle gap after a sweep delete
        if eng_log.seconds_since_delete(ch_id) < 45:
            await asyncio.sleep(random.uniform(8, 30))

        # Send with typing simulation (proportional)
        typing_dur = human_timing.typing_seconds(message, persona_id=pid)
        async with channel.typing():
            await asyncio.sleep(typing_dur)
        sent = await channel.send(message)
        eng_log.mark_sent(ch_id)
        eng_log.record(ch_id, sent, kind="auto_chat")
        logger.info(f"Random auto-chat in #{channel.name}: {message}")

    async def send_post_bump_messages(self):
        """Send an engagement message after a bump batch to boost chat
        activity. ONE channel, not every channel — an account hopping rooms
        posting the same opener right after each bump is the pattern
        watchers clock. Sometimes skipped entirely — a person doesn't
        celebrate every bump."""
        # ~30% of batches get no chatter at all
        if random.random() < 0.3:
            return

        eng_log = get_engagement_log()
        eligible = []
        for channel_id in self.channel_ids:
            channel = self.client.get_channel(channel_id)
            if not channel:
                continue
            if hasattr(channel, 'guild') and channel.guild:
                if not can_speak_in(channel, channel.guild.me):
                    continue
            ch_id = str(channel_id)
            # Skip channels with no recent human activity — posting into a
            # dead room right after bumping is the most bot-like thing there is
            if self.last_activity.get(ch_id, 0) < time.time() - 1800:
                continue
            eligible.append((channel, ch_id))
        if not eligible:
            return

        channel, ch_id = random.choice(eligible)
        logger.info(f"Sending post-bump message to #{channel.name}")

        # The "finished bumping, wandered back to chat" beat — a real pause,
        # not bump→+10s→engagement on a fixed cadence
        await asyncio.sleep(random.uniform(30, 300))

        # Clean stale unanswered engagement first; unified gate after
        await eng_log.sweep_before_send(channel, self.client.user,
                                        history_msgs=self.client.history_cache.get(ch_id))
        if not eng_log.can_send_engagement(ch_id, last_human_ts=self.last_activity.get(ch_id, 0),
                                            history_msgs=self.client.history_cache.get(ch_id),
                                            bot_user=self.client.user):
            logger.debug(f"[engage] #{channel.name} gated — skipping post-bump")
            return

        loop = asyncio.get_running_loop()
        pid = _persona_id(self.client)
        topic = await loop.run_in_executor(
            None, lambda: mem.get_channel_topic_fresh(ch_id))
        msg = await loop.run_in_executor(
            None, lambda: ai_reply.generate_proactive_message(
                topic, channel_id=ch_id,
                recent_lines=_recent_lines(self.client, ch_id),
                persona_id=pid)
        )
        msg = _fallback_or(msg, pid, "bump", ch_id)

        # Settle gap after a sweep delete
        if eng_log.seconds_since_delete(ch_id) < 45:
            await asyncio.sleep(random.uniform(8, 30))

        try:
            async with channel.typing():
                await asyncio.sleep(human_timing.typing_seconds(msg, persona_id=pid))
            sent = await channel.send(msg)
            eng_log.mark_sent(ch_id)
            eng_log.record(ch_id, sent, kind="post_bump")
            logger.info(f"Post-bump message in #{channel.name}: {msg}")
        except Exception as e:
            logger.warning(f"Failed to send post-bump message: {e}")

    async def proactive_loop(self):
        """
        Background loop: occasionally send a conversation-starting message
        in the most active channel across all guilds.
        This runs separately from the dead-chat monitor and targets channels
        that ARE active (to join ongoing conversations naturally).
        """
        await self.client.wait_until_ready()

        while self._running and not self.client.is_closed():
            # Wait between proactive messages
            wait_secs = random.randint(
                self.proactive_interval[0] * 60,
                self.proactive_interval[1] * 60,
            )
            logger.debug(f"Proactive loop sleeping {wait_secs // 60}min...")
            await asyncio.sleep(wait_secs)

            if time.time() < self._warmup_until:
                continue  # still settling in — no posting right after login

            try:
                # Find the most active channel across all guilds
                best_ch = None
                best_ts = 0.0
                now = time.time()

                for guild in self.client.guilds:
                    ch = await find_most_active_channel(guild, self.last_activity, max_age_hours=2)
                    if ch:
                        ch_id = str(ch.id)
                        ts = self.last_activity.get(ch_id, 0)
                        if ts > best_ts:
                            best_ts = ts
                            best_ch = ch

                if not best_ch:
                    logger.debug("Proactive: no active channels found, skipping")
                    continue

                ch_id = str(best_ch.id)
                loop = asyncio.get_running_loop()
                pid = _persona_id(self.client)
                topic = await loop.run_in_executor(
                    None, lambda: mem.get_channel_topic_fresh(ch_id))
                msg = await loop.run_in_executor(
                    None, lambda: ai_reply.generate_proactive_message(
                        topic, channel_id=ch_id,
                        recent_lines=_recent_lines(self.client, ch_id),
                        persona_id=pid)
                )
                msg = _fallback_or(msg, pid, "chat", ch_id)

                # Clean stale engagement first; unified gate decides
                eng_log = get_engagement_log()
                await eng_log.sweep_before_send(best_ch, self.client.user,
                                                history_msgs=self.client.history_cache.get(ch_id))
                if not eng_log.can_send_engagement(ch_id, last_human_ts=self.last_activity.get(ch_id, 0),
                                            history_msgs=self.client.history_cache.get(ch_id),
                                            bot_user=self.client.user):
                    logger.debug(f"[engage] #{best_ch.name} gated — skipping proactive")
                    continue

                # Settle gap after a sweep delete
                if eng_log.seconds_since_delete(ch_id) < 45:
                    await asyncio.sleep(random.uniform(8, 30))

                # Send with typing simulation (proportional)
                typing_dur = human_timing.typing_seconds(msg, persona_id=pid)
                async with best_ch.typing():
                    await asyncio.sleep(typing_dur)
                sent = await best_ch.send(msg)
                eng_log.mark_sent(ch_id)
                eng_log.record(ch_id, sent, kind="proactive")
                self.last_proactive = time.time()
                logger.info(f"Proactive message in #{best_ch.name}: {msg}")

            except Exception as e:
                logger.warning(f"Proactive loop error: {e}")
