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
from .ai import d1_memory as mem  # D1-backed memory (falls back to JSON if D1 unavailable)
from .channel_scanner import can_speak_in, is_skip_channel, find_most_active_channel
from .ai.re_engagement import get_ping_controller, get_re_engagement_tracker, select_online_user
from .ai.engagement_log import get_engagement_log


# Random messages for post-bump engagement
POST_BUMP_MESSAGES = [
    "yo what's everyone up to",
    "what y'all doing",
    "anyone wanna chat",
    "this server been kinda dead ngl",
    "yo who's active rn",
    "what we talking about",
    "hru everyone",
    "anyone playing anything rn",
    "yo chat is dead today fr",
    "what's good everyone",
    "anyone else bored rn",
    "ngl i'm so bored",
    "yo revive chat",
    "who's still awake",
    "what we doing today",
]

# Random messages for dead chat revival
DEAD_CHAT_MESSAGES = [
    "chat is dead fr",
    "yo anyone there",
    "this server been so quiet",
    "hello?? anyone alive",
    "revive chat y'all",
    "why is it so dead in here",
    "yo chat wake up",
    "anyone wanna talk",
    "i'm bored someone talk to me",
    "ngl this server been dead",
]

# Random "just chatting" messages — sent even when chat isn't fully dead
# to make the bot feel alive and spontaneous
RANDOM_CHAT_MESSAGES = [
    "yo",
    "sup",
    "anyone up",
    "what's good",
    "bored fr",
    "yo anyone wanna talk",
    "ngl i'm bored",
    "what everyone doing",
    "hru",
    "anyone playing anything",
    "yo chat",
    "fr been a minute",
    "what's the move today",
    "anyone else just chilling",
    "yo i just woke up",
    "what we talking about",
    "someone talk to me",
    "i'm so bored rn",
    "yo who's active",
    "anyone wanna vibe",
    "just got on, what'd i miss",
    "yo this server quiet today",
    "anyone else here",
    "what's up everyone",
    "yo hru guys",
    "anyone awake",
    "chat's slow today ngl",
    "yo anyone wanna play smth",
    "i'm back, what's good",
    "who's around",
]


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
        self.last_proactive = 0
        self.last_auto_chat = time.time()
        self.last_activity: dict = {}  # channel_id -> timestamp

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
            await asyncio.sleep(self.check_interval)

    async def stop(self):
        """Stop the proactive messaging loop."""
        self._running = False

    def record_activity(self, channel_id: int):
        """Record that a message was seen in a channel."""
        self.last_activity[str(channel_id)] = time.time()
        # NOTE: called for bot msgs too — human-only interaction marking
        # happens in discord_client's `not author.bot` tracking block

    async def _monitor_loop(self):
        """
        Check all channels for:
        1. Dead chat (silent for > threshold) — revive with ping role
        2. Inactive chat (silent for > 5 min but < threshold) — random auto-chat
        """
        now = time.time()

        for channel_id in self.channel_ids:
            ch_id = str(channel_id)
            last_msg = self.last_activity.get(ch_id, 0)
            silence_duration = now - last_msg if last_msg > 0 else 999999

            # 1. Dead chat revival (silent for > threshold)
            if silence_duration > self.dead_chat_threshold:
                # Only revive if we haven't sent a proactive message in the last 30 min
                # (increased from 5 min to reduce ping irritation)
                if (now - self.last_proactive) > 1800:
                    await self._revive_dead_chat(channel_id)
                    self.last_proactive = now
                    return  # One revival per cycle

            # 2. Random auto-chat (channel has some activity but is slowing down)
            # Send a random message with auto_chat_chance probability
            # Only if: silence > 5 min, and we haven't auto-chatted in 10+ min
            elif silence_duration > 300 and (now - self.last_auto_chat) > 600:
                if random.random() < self.auto_chat_chance:
                    await self._send_random_chat(channel_id)
                    self.last_auto_chat = now
                    return  # One auto-chat per cycle

    async def _revive_dead_chat(self, channel_id: int):
        """Send a message to revive a dead channel (with ping role).
        Uses ping controller to prevent over-pinging."""
        channel = self.client.get_channel(channel_id)
        if not channel:
            return

        # Check permissions
        if hasattr(channel, 'guild') and channel.guild:
            if not can_speak_in(channel, channel.guild.me):
                return

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
            return

        ping_ctrl = get_ping_controller()

        # Decide the ping target FIRST — the message text is generated with
        # the pinged user's name in context so it never attributes personal
        # facts to them ("got any sketches?" to someone who never sketched).
        ping_prefix = ""
        pinged_user = None
        role = None
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

        # Try AI-generated message first, fall back to random
        topic = mem.get_channel_topic(ch_id)
        loop = asyncio.get_running_loop()
        message = await loop.run_in_executor(
            None, lambda: ai_reply.generate_proactive_message(
                topic, for_user=getattr(pinged_user, "display_name", None))
        )
        if not message or len(message) < 3:
            message = random.choice(DEAD_CHAT_MESSAGES)
        message = ping_prefix + message

        if role and ping_prefix.startswith("<@&"):
            ping_ctrl.record_role_ping(ch_id)
        elif ping_prefix.startswith("@here"):
            ping_ctrl.record_here_ping(ch_id)
        elif pinged_user:
            ping_ctrl.record_user_ping(ch_id, user_id=pinged_user.id)
            if hasattr(self.client, 'conversation_tracker'):
                if ch_id not in self.client.conversation_tracker:
                    self.client.conversation_tracker[ch_id] = {}
                self.client.conversation_tracker[ch_id][str(pinged_user.id)] = time.time()

        # Send with typing simulation
        typing_dur = random.uniform(1.5, 3.5)
        async with channel.typing():
            await asyncio.sleep(typing_dur)
        sent = await channel.send(message)
        eng_log.mark_sent(ch_id)
        eng_log.record(ch_id, sent, kind="revive")
        logger.info(f"Revived dead chat in #{channel.name}: {message}")

    async def _send_random_chat(self, channel_id: int):
        """Send a random spontaneous message to a channel (no ping role).
        Uses pre-written messages only — NO AI calls, saves Groq tokens."""
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

        # Always use pre-written messages for auto-chat (saves Groq tokens)
        message = random.choice(RANDOM_CHAT_MESSAGES)

        # Send with typing simulation
        typing_dur = random.uniform(1.0, 3.0)
        async with channel.typing():
            await asyncio.sleep(typing_dur)
        sent = await channel.send(message)
        eng_log.mark_sent(ch_id)
        eng_log.record(ch_id, sent, kind="auto_chat")
        logger.info(f"Random auto-chat in #{channel.name}: {message}")

    async def send_post_bump_messages(self):
        """Send engagement messages after a bump batch to boost chat activity."""
        logger.info(f"Sending post-bump messages to {len(self.channel_ids)} channels")

        eng_log = get_engagement_log()
        for channel_id in self.channel_ids:
            channel = self.client.get_channel(channel_id)
            if not channel:
                continue

            if hasattr(channel, 'guild') and channel.guild:
                if not can_speak_in(channel, channel.guild.me):
                    continue

            # Clean stale unanswered engagement first; unified gate after
            ch_id = str(channel_id)
            await eng_log.sweep_before_send(channel, self.client.user,
                                            history_msgs=self.client.history_cache.get(ch_id))
            if not eng_log.can_send_engagement(ch_id, last_human_ts=self.last_activity.get(ch_id, 0),
                                            history_msgs=self.client.history_cache.get(ch_id),
                                            bot_user=self.client.user):
                logger.debug(f"[engage] #{channel.name} gated — skipping post-bump")
                continue

            # Send 1-2 messages with delays (a 3-message burst reads as spam)
            num_messages = random.randint(1, 2)
            for i in range(num_messages):
                await asyncio.sleep(random.uniform(8, 20))

                # First message: AI-generated or random
                if i == 0:
                    ch_id = str(channel_id)
                    topic = mem.get_channel_topic(ch_id)
                    loop = asyncio.get_running_loop()
                    msg = await loop.run_in_executor(
                        None, lambda: ai_reply.generate_proactive_message(topic)
                    )
                    if not msg or len(msg) < 3:
                        msg = random.choice(POST_BUMP_MESSAGES)
                else:
                    msg = random.choice(POST_BUMP_MESSAGES)

                # Sometimes tag a random recent user — skip anyone pinged recently
                if i > 0 and random.random() < 0.4:
                    try:
                        ping_ctrl = get_ping_controller()
                        recent_ping_ids = ping_ctrl.recently_pinged(ch_id)
                        recent_messages = [m async for m in channel.history(limit=20)]
                        human_authors = [
                            m.author for m in recent_messages
                            if not m.author.bot and m.author != self.client.user
                            and m.author.id not in recent_ping_ids
                        ]
                        if human_authors:
                            target = random.choice(human_authors)
                            msg = f"<@{target.id}> {msg}"
                            ping_ctrl.record_user_ping(ch_id, user_id=target.id)
                    except Exception:
                        pass

                try:
                    async with channel.typing():
                        await asyncio.sleep(random.uniform(1.5, 3.0))
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
                topic = mem.get_channel_topic(ch_id)
                loop = asyncio.get_running_loop()
                msg = await loop.run_in_executor(
                    None, lambda: ai_reply.generate_proactive_message(topic)
                )
                if not msg or len(msg) < 3:
                    msg = random.choice(RANDOM_CHAT_MESSAGES)

                # Clean stale engagement first; unified gate decides
                eng_log = get_engagement_log()
                await eng_log.sweep_before_send(best_ch, self.client.user,
                                                history_msgs=self.client.history_cache.get(ch_id))
                if not eng_log.can_send_engagement(ch_id, last_human_ts=self.last_activity.get(ch_id, 0),
                                            history_msgs=self.client.history_cache.get(ch_id),
                                            bot_user=self.client.user):
                    logger.debug(f"[engage] #{best_ch.name} gated — skipping proactive")
                    continue

                # Send with typing simulation
                typing_dur = random.uniform(1.5, 3.5)
                async with best_ch.typing():
                    await asyncio.sleep(typing_dur)
                sent = await best_ch.send(msg)
                eng_log.mark_sent(ch_id)
                eng_log.record(ch_id, sent, kind="proactive")
                self.last_proactive = time.time()
                logger.info(f"Proactive message in #{best_ch.name}: {msg}")

            except Exception as e:
                logger.warning(f"Proactive loop error: {e}")
