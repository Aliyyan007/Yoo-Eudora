"""Voice manager — manages voice connections, auto-join/leave, silence detection.

Features:
1. Auto-join VC when users are present (with probability)
2. Auto-leave when VC is empty or silent for too long
3. Text VC members when no one is talking (but users are present)
4. Manual join/leave via text commands
5. Voice conversation pipeline management
"""
from __future__ import annotations

import asyncio
import random
import time
from typing import Optional, Dict, Set, Callable, Awaitable
from loguru import logger
import discord

from .pipeline import VoicePipeline
from .tts import TTSConfig
from .proactive import ProactiveEngager


class VoiceManager:
    """Manages all voice channel activity for the bot."""

    def __init__(
        self,
        tts_config: TTSConfig,
        on_transcript: Callable[[int, str], Awaitable[Optional[str]]],
        bot_id: int,
        on_transcript_stream: Optional[Callable] = None,
    ):
        self._tts_config = tts_config
        self._on_transcript = on_transcript
        self._on_transcript_stream = on_transcript_stream
        self._bot_id = bot_id

        # Per-guild state: guild_id -> VoicePipeline
        self._pipelines: Dict[int, VoicePipeline] = {}
        # guild_id -> channel_id (current VC)
        self._current_vcs: Dict[int, int] = {}
        # guild_id -> last speaking timestamp
        self._last_speaking: Dict[int, float] = {}
        # guild_id -> last text nudge timestamp (for silent VC texting)
        self._last_nudge: Dict[int, float] = {}

        # Config
        self._silence_leave_threshold_s = 300  # 5 min silence → leave
        self._silence_nudge_threshold_s = 120  # 2 min silence → text members
        self._nudge_cooldown_s = 600           # 10 min between nudges
        self._empty_check_interval_s = 30      # check every 30s
        self._join_grace_s = 60                # don't empty-leave within 60s of joining
        self._auto_join_chance = 0.15          # ~15% per tick — more engaged joining
        self._min_users_to_join = 1            # even 1 person = someone to talk to

        # Stay-duration + equity bookkeeping
        self._vc_joined_at: Dict[int, float] = {}        # guild_id -> join ts
        self._vc_left_at: Dict[int, tuple] = {}          # guild_id -> (channel_id, leave ts)

        # Track which users we've greeted in VC
        self._greeted_users: Dict[int, Set[int]] = {}  # guild_id -> set of user_ids

        # Proactive engagement — bot asks questions when VC goes quiet
        self._proactive: ProactiveEngager = ProactiveEngager(self)

    def is_in_vc(self, guild_id: int) -> bool:
        return guild_id in self._current_vcs

    def get_current_vc_id(self, guild_id: int) -> Optional[int]:
        return self._current_vcs.get(guild_id)

    async def join_vc(
        self,
        client: discord.Client,
        channel: discord.VoiceChannel,
        loop: asyncio.AbstractEventLoop,
    ) -> bool:
        """Join a voice channel and start listening."""
        guild_id = channel.guild.id

        # Already connected in this guild? Same channel = no-op; different
        # channel = move in place (keeps the VoiceClient + listener alive).
        existing = discord.utils.get(client.voice_clients, guild=channel.guild)
        if existing is not None and getattr(existing, "is_connected", lambda: True)():
            cur_ch = getattr(existing, "channel", None)
            if cur_ch is not None and cur_ch.id == channel.id:
                return True
            try:
                await existing.move_to(channel)
                self._current_vcs[guild_id] = channel.id
                self._last_speaking[guild_id] = time.time()
                self._vc_joined_at[guild_id] = time.time()
                logger.info(f"Moved VC to '{channel.name}' in guild '{channel.guild.name}'")
                return True
            except Exception as e:
                logger.debug(f"move_to failed — reconnecting cleanly: {e}")
                await self.leave_vc(client, channel.guild)

        # Try discord-native-voice first (enables voice receive), fall back to regular VoiceClient
        native_vc = None
        try:
            from discord.ext.native_voice import VoiceClient
            native_vc = VoiceClient
        except ImportError:
            logger.warning("discord-native-voice not installed — voice receive unavailable")

        # Attempt connection with native voice client
        if native_vc is not None:
            try:
                vc = await channel.connect(
                    cls=native_vc,
                    self_mute=False,
                    self_deaf=False,
                    reconnect=False,
                )
                self._current_vcs[guild_id] = channel.id
                self._last_speaking[guild_id] = time.time()
                self._vc_joined_at[guild_id] = time.time()

                # Create pipeline for this guild
                pipeline = VoicePipeline(
                    self._tts_config, self._on_transcript, self._bot_id,
                    on_transcript_stream=self._on_transcript_stream,
                )
                pipeline.set_voice_client(vc, loop)
                pipeline._guild_id = guild_id
                pipeline._relocate_cb = lambda ch, gid=guild_id: self._after_vc_move(client, gid, ch)
                pipeline._leave_cb = lambda g=channel.guild: self.leave_vc(client, g)
                # Outsider action worker — spoken action requests go to the
                # isolated engine (leave stays native via _leave_cb above).
                pipeline._action_cb = lambda uid, txt, g=channel.guild: self._run_voice_action(client, uid, txt, g)
                self._pipelines[guild_id] = pipeline

                # Start comfort noise to keep voice indicator active
                pipeline.start_comfort_noise()

                # Start listening to incoming audio
                packet_count = [0]  # mutable counter for debug logging
                def on_packet(packet):
                    try:
                        packet_count[0] += 1
                        if packet_count[0] <= 5 or packet_count[0] % 100 == 0:
                            logger.info(f"[voice] packet #{packet_count[0]}: media_type={packet.media_type}, user_id={packet.user_id}, payload_len={len(bytes(packet.payload)) if hasattr(packet, 'payload') else 'N/A'}")
                        if packet.media_type != "audio":
                            return
                        user_id = packet.user_id
                        if user_id == self._bot_id:
                            return
                        # MediaPacket.payload contains the encoded audio bytes (Opus)
                        audio_data = bytes(packet.payload) if hasattr(packet, 'payload') else b""
                        if not audio_data:
                            return
                        self._last_speaking[guild_id] = time.time()
                        # Update display name tracking for proactive engagement
                        member = channel.guild.get_member(user_id)
                        if member:
                            display_name = member.display_name or member.name
                            pipeline.set_user_display_name(user_id, display_name)
                        fut = asyncio.run_coroutine_threadsafe(
                            pipeline.process_audio(user_id, audio_data),
                            loop,
                        )
                        fut.add_done_callback(self._log_pipeline_error)
                    except Exception as e:
                        logger.error(f"Voice packet processing error: {e!r}")

                vc.listen(on_packet)
                logger.info(f"Joined VC '{channel.name}' in guild '{channel.guild.name}' with voice receive (is_listening={vc.is_listening()})")

                # Speak a greeting after joining (delay slightly to let connection settle)
                non_bot_members = [m for m in channel.members if not m.bot and m.id != self._bot_id]
                if non_bot_members:
                    async def _delayed_greeting():
                        await asyncio.sleep(1.5)  # Wait for connection to stabilize
                        names = [m.display_name or m.name for m in non_bot_members[:3]]
                        try:
                            await self._speak_join_greeting(pipeline, names)
                        except Exception as e:
                            logger.debug(f"[voice] Join greeting error: {e}")
                    asyncio.create_task(_delayed_greeting())

                return True
            except Exception as e:
                logger.error(f"Native voice client failed for '{channel.name}': {e!r} — falling back to regular VoiceClient")

        # Fallback: join with regular VoiceClient (can speak but not listen)
        try:
            vc = await channel.connect(self_mute=False, self_deaf=False, reconnect=False)
            self._current_vcs[guild_id] = channel.id
            self._last_speaking[guild_id] = time.time()
            logger.info(f"Joined VC '{channel.name}' (no voice receive — fallback mode)")
            return True
        except Exception as e:
            logger.error(f"Failed to join VC '{channel.name}': {e!r}")
            return False

    async def _run_voice_action(self, client: discord.Client, user_id: int,
                                text: str, guild) -> tuple | None:
        """Route a spoken request through the outsider action worker.
        Returns ("info", facts) when a lookup ran and produced an answer, or
        ("exec", None) when an action was queued to run after the spoken
        reply — she says "on it" then does it, like a person. None → not an
        action / failed → normal reply continues. Resolves the LIVE voice
        channel (stays correct across move_to)."""
        try:
            # Leave-vc is native-only — never reaches the worker
            try:
                from .vc_intent import leave_vc_score
                if leave_vc_score(text) >= 0.5:
                    return None
            except Exception:
                pass
            vc = discord.utils.get(client.voice_clients, guild=guild)
            ch = getattr(vc, "channel", None)
            if ch is None:
                cid = self._current_vcs.get(guild.id)
                ch = guild.get_channel(cid) if cid else None
            if ch is None:
                return None
            from ..ai.action_bridge import get_action_worker
            worker = get_action_worker(client)
            kind = await worker.classify_request(text)
            if kind == "info":
                facts = await worker.run_voice_info(user_id, text, ch)
                return ("info", facts) if facts else None
            if kind == "exec":
                if worker.queue_voice_action(user_id, text, ch):
                    return ("exec", None)
            return None
        except Exception as e:
            logger.debug(f"[voice] action worker call failed: {e!r}")
            return None

    @staticmethod
    def _log_pipeline_error(fut) -> None:
        if fut.cancelled():
            return
        exc = fut.exception()
        if exc is not None:
            logger.opt(exception=exc).error(f"Voice pipeline error: {exc!r}")

    async def leave_vc(self, client: discord.Client, guild: discord.Guild) -> str:
        """Leave the current voice channel."""
        guild_id = guild.id
        if guild_id not in self._current_vcs:
            return "i'm not in a vc rn"

        try:
            # Clean up pipeline
            pipeline = self._pipelines.pop(guild_id, None)
            if pipeline:
                await pipeline.cleanup()
                # Clear display name tracking for this guild's users
                pipeline._user_display_names.clear()
                pipeline._user_speech_count.clear()
                pipeline._user_last_speech_time.clear()

            # Disconnect
            voice_client = discord.utils.get(client.voice_clients, guild=guild)
            if voice_client:
                await voice_client.disconnect()
                logger.info(f"Left VC in guild '{guild.name}'")

            # Remember where/when we left — join-requests to a VC the bot
            # just left get declined for a while (equilibrium between users)
            if vc_id := self._current_vcs.pop(guild_id, None):
                self._vc_left_at[guild_id] = (vc_id, time.time())
            self._vc_joined_at.pop(guild_id, None)
            self._last_speaking.pop(guild_id, None)
            self._last_nudge.pop(guild_id, None)
            self._greeted_users.pop(guild_id, None)
            return "left the vc 👋"
        except Exception as e:
            logger.error(f"Failed to leave VC: {e}")
            return "couldn't leave the vc"

    def record_speaking(self, guild_id: int):
        """Record that speaking activity happened in a VC."""
        if guild_id in self._current_vcs:
            self._last_speaking[guild_id] = time.time()

    def get_silence_duration(self, guild_id: int) -> float:
        """Return seconds since last speaking activity."""
        last = self._last_speaking.get(guild_id, 0)
        if last == 0:
            return float("inf")
        return time.time() - last

    async def monitor_vcs(self, client: discord.Client):
        """Background task: auto-join/leave VCs, text silent VC members."""
        logger.info("Voice monitor background task started")
        loop = asyncio.get_event_loop()

        # Start proactive engagement (bot asks questions when VC is quiet)
        self._proactive.start(client)

        while True:
            try:
                await asyncio.sleep(self._empty_check_interval_s)

                for guild in client.guilds:
                    try:
                        if self.is_in_vc(guild.id):
                            await self._check_current_vc(client, guild, loop)
                        else:
                            await self._maybe_auto_join(client, guild, loop)
                    except Exception as e:
                        logger.debug(f"VC monitor error for guild '{guild.name}': {e}")

            except asyncio.CancelledError:
                logger.info("Voice monitor task cancelled")
                break
            except Exception as e:
                logger.debug(f"Voice monitor loop error: {e}")

    async def _check_current_vc(
        self,
        client: discord.Client,
        guild: discord.Guild,
        loop: asyncio.AbstractEventLoop,
    ):
        """Check if bot should leave current VC or text silent members."""
        vc_id = self._current_vcs.get(guild.id)
        if vc_id is None:
            return

        vc_channel = guild.get_channel(vc_id)
        if vc_channel is None:
            await self.leave_vc(client, guild)
            return

        non_bot_members = [m for m in vc_channel.members if not m.bot and m.id != self._bot_id]

        # Leave if empty — but give a join grace window first. Members can
        # lag behind the connect event in the gateway cache (and people need
        # a moment to actually show up after asking us to join).
        if len(non_bot_members) == 0:
            joined_at = self._vc_joined_at.get(guild.id, 0)
            if time.time() - joined_at < self._join_grace_s:
                return
            await self._auto_leave(client, guild, "vc's empty, dipping out 👋")
            return

        # Check silence
        silence = self.get_silence_duration(guild.id)

        # Leave if silent for too long
        if silence > self._silence_leave_threshold_s:
            await self._auto_leave(client, guild, "alright it's gone dead in here — i'm heading off, see you lot")
            return

        # ── Algorithmic stay-duration / VC-state leave scoring ────────────
        # The bot shares its time fairly: ~50min soft cap growing toward a
        # ~2h urge, accelerated when the VC is dry (long silences, nobody
        # engaging) or the room got heated. Not a hard rule — a lively VC
        # keeps it here well past 2h; a dead one gets left early.
        leave_score = self._leave_score(guild.id, non_bot_members, silence)
        if leave_score >= 0.55 and random.random() < leave_score:
            await self._auto_leave(
                client, guild,
                random.choice([
                    "alright, i've been here a while — gonna head off, catch you lot later",
                    "i'm gonna dip now — been here ages innit, see you",
                    "right, i'm heading out — catch you all later",
                ]),
            )
            return
        # Moderate urge → occasionally voice the intent ("i might go soon")
        # so it doesn't come out of nowhere — but usually stays
        if leave_score >= 0.3 and random.random() < leave_score * 0.15:
            pipeline = self._pipelines.get(guild.id)
            if pipeline and not pipeline.is_speaking:
                try:
                    await pipeline._speak(random.choice([
                        "i might head off soon honestly",
                        "been here a while now — might dip in a bit",
                    ]))
                except Exception:
                    pass

        # Text VC members if silent but people are present
        if silence > self._silence_nudge_threshold_s:
            last_nudge = self._last_nudge.get(guild.id, 0)
            if time.time() - last_nudge > self._nudge_cooldown_s:
                await self._nudge_silent_vc_members(client, guild, vc_channel, silence)
                self._last_nudge[guild.id] = time.time()

    async def _nudge_silent_vc_members(
        self,
        client: discord.Client,
        guild: discord.Guild,
        vc_channel: discord.VoiceChannel,
        silence_s: float,
    ):
        """Text VC members when no one is talking but people are present."""
        # Find a text channel to send the nudge
        text_channel = self._find_vc_text_channel(guild, vc_channel)
        if text_channel is None:
            logger.debug(f"No text channel found for VC '{vc_channel.name}'")
            return

        # Check if we can send messages there
        if not text_channel.permissions_for(guild.me).send_messages:
            return

        # Get VC members (non-bot)
        members = [m for m in vc_channel.members if not m.bot and m.id != self._bot_id]
        if not members:
            return

        # Pick a random member to nudge
        target = random.choice(members)
        silence_min = int(silence_s / 60)

        nudges = [
            f"yo {target.mention} u still alive in vc or did u fall asleep 😭 it's been {silence_min} min",
            f"{target.mention} say something innit, {silence_min} mins of silence is mad",
            f"hello?? {target.mention} ?? vc is dead bruv, {silence_min} mins nobody spoke",
            f"{target.mention} bro the vc is silent af, {silence_min} mins already",
            f"oi {target.mention} u good? {silence_min} mins of nothing in vc 💀",
        ]

        message = random.choice(nudges)
        try:
            async with text_channel.typing():
                await asyncio.sleep(random.uniform(1.5, 3.0))
            await text_channel.send(message)
            logger.info(f"Nudged silent VC members in #{text_channel.name}: {message[:60]}")
        except discord.Forbidden:
            logger.debug(f"Can't send nudge in #{text_channel.name} (forbidden)")
        except Exception as e:
            logger.debug(f"Failed to nudge VC members: {e}")

    def _find_vc_text_channel(
        self,
        guild: discord.Guild,
        vc_channel: discord.VoiceChannel,
    ) -> Optional[discord.TextChannel]:
        """Find a text channel associated with a voice channel."""
        vc_name = vc_channel.name.lower().replace(" ", "-")

        # Strategy 1: Same name + "-chat" or "-text"
        for tc in guild.text_channels:
            tc_name = tc.name.lower()
            if tc_name == vc_name + "-chat" or tc_name == vc_name + "-text":
                return tc
            if tc_name == vc_name:
                return tc

        # Strategy 2: Same category
        if vc_channel.category:
            for tc in guild.text_channels:
                if tc.category and tc.category.id == vc_channel.category.id:
                    if any(word in tc.name.lower() for word in vc_channel.name.lower().split()):
                        return tc

        # Strategy 3: General voice-chat channels
        for tc in guild.text_channels:
            tc_name = tc.name.lower()
            if tc_name in ("voice-chat", "vc-chat", "vc-text", "voice-text", "voice"):
                return tc

        # Strategy 4: First text channel in same category
        if vc_channel.category:
            for tc in guild.text_channels:
                if tc.category and tc.category.id == vc_channel.category.id:
                    return tc

        return None

    async def _maybe_auto_join(
        self,
        client: discord.Client,
        guild: discord.Guild,
        loop: asyncio.AbstractEventLoop,
    ):
        """Small chance to join an active VC independently."""
        if random.random() > self._auto_join_chance:
            return

        # Find active VCs with enough members — skip the channel we just left
        # (join_decline_probability keeps it fair; a fresh room is preferred)
        active_vcs = []
        for channel in guild.voice_channels:
            members = [m for m in channel.members if not m.bot and m.id != self._bot_id]
            if len(members) >= self._min_users_to_join:
                if random.random() >= self.join_decline_probability(guild.id, channel.id):
                    active_vcs.append((len(members), channel))

        if not active_vcs:
            return

        # Prefer the most populated VC, but 30% of the time take a random
        # one — algorithmic variety so she doesn't always camp the biggest
        active_vcs.sort(key=lambda x: x[0], reverse=True)
        if len(active_vcs) > 1 and random.random() < 0.3:
            best_vc = random.choice(active_vcs[:3])[1]
        else:
            best_vc = active_vcs[0][1]

        try:
            success = await self.join_vc(client, best_vc, loop)
            if success:
                logger.info(f"Auto-joined VC '{best_vc.name}' in guild '{guild.name}'")
        except Exception as e:
            logger.debug(f"Auto-join failed: {e}")

    async def _auto_leave(self, client: discord.Client, guild: discord.Guild, reason: str):
        """Leave the VC. No text announcement — just a brief spoken farewell
        when someone's still around to hear it, then disconnect."""
        vc_id = self._current_vcs.get(guild.id)
        if vc_id is None:
            return

        vc_channel = guild.get_channel(vc_id)
        members = [
            m for m in (vc_channel.members if vc_channel else [])
            if not m.bot and m.id != self._bot_id
        ]
        pipeline = self._pipelines.get(guild.id)
        if members and pipeline:
            try:
                await pipeline._speak(reason)
            except Exception as e:
                logger.debug(f"[voice] Leave farewell failed: {e}")

        await self.leave_vc(client, guild)

    async def _after_vc_move(self, client: discord.Client, guild_id: int, channel) -> None:
        """vc.move_to fallback — disconnect and rejoin the target channel."""
        guild = client.get_guild(guild_id)
        if guild is None:
            return
        await self.leave_vc(client, guild)
        await self.join_vc(client, channel, asyncio.get_event_loop())

    def _leave_score(self, guild_id: int, members: list, silence: float) -> float:
        """0-1 algorithmic leave urge — combines stay duration, VC dryness
        and room irritation into one score. Evaluated every ~30s tick.

        - 0-50min: essentially 0 (she settles in)
        - 50-120min: grows 0 → ~0.5 (she's given this VC its time)
        - 120min+: ~0.55 base, still not a hard cap — a lively VC survives
          because the dryness/activity terms stay low
        - dryness adds up to +0.35, irritation adds up to +0.5
        """
        score = 0.0
        joined = self._vc_joined_at.get(guild_id)
        if joined:
            elapsed_min = (time.time() - joined) / 60.0
            if elapsed_min > 120:
                score += 0.55
            elif elapsed_min > 50:
                score += 0.10 + (elapsed_min - 50) / 70.0 * 0.40

        # Dryness — long silence with people present means a dead VC
        if silence > 240:
            score += 0.35
        elif silence > 120:
            score += 0.15

        # Irritation — if the room got abusive, the urge to bail rises
        pipeline = self._pipelines.get(guild_id)
        if pipeline:
            urges = [pipeline._irritation.leave_urge(m.id) for m in members]
            score += 0.5 * max(urges, default=0.0)

        return min(1.0, score)

    def join_decline_probability(self, guild_id: int, channel_id: int) -> float:
        """0-1 chance the bot declines a request to rejoin a channel it just
        left — high right after leaving, decays to 0 over ~45min. Algorithmic
        equity between users so one VC can't hog her."""
        left = self._vc_left_at.get(guild_id)
        if not left or left[0] != channel_id:
            return 0.0
        mins_since = (time.time() - left[1]) / 60.0
        if mins_since >= 45:
            return 0.0
        return max(0.0, 1.0 - mins_since / 45.0)

    async def handle_voice_state_update(
        self,
        member: discord.Member,
        before: discord.VoiceState,
        after: discord.VoiceState,
        client: discord.Client,
    ):
        """Handle voice state changes (users joining/leaving VC)."""
        guild_id = member.guild.id

        if not self.is_in_vc(guild_id):
            return

        vc_id = self._current_vcs.get(guild_id)
        if vc_id is None:
            return

        vc_channel = member.guild.get_channel(vc_id)
        if vc_channel is None:
            return

        # User left the VC we're in
        if before.channel and before.channel.id == vc_id and (
            not after.channel or after.channel.id != vc_id
        ):
            if member.id != self._bot_id:
                # Clean up their pipeline resources
                pipeline = self._pipelines.get(guild_id)
                if pipeline:
                    await pipeline.cleanup_user(member.id)
                logger.debug(f"User {member.name} left VC, cleaned up resources")

        # User joined the VC we're in
        if after.channel and after.channel.id == vc_id and (
            not before.channel or before.channel.id != vc_id
        ):
            if member.id != self._bot_id:
                self._last_speaking[guild_id] = time.time()
                logger.debug(f"User {member.name} joined VC '{vc_channel.name}'")

                # Speak a greeting when a user joins the VC
                greeted = self._greeted_users.setdefault(guild_id, set())
                if member.id not in greeted:
                    greeted.add(member.id)
                    pipeline = self._pipelines.get(guild_id)
                    if pipeline and not pipeline._is_speaking:
                        display_name = member.display_name or member.name
                        # Build a greeting using the on_transcript callback's LLM
                        # but with a greeting-specific prompt
                        try:
                            asyncio.create_task(self._speak_greeting(pipeline, display_name))
                        except Exception as e:
                            logger.debug(f"[voice] Greeting task error: {e}")

    def get_status(self) -> dict:
        """Get voice status for debugging."""
        return {
            "current_vcs": dict(self._current_vcs),
            "pipelines": len(self._pipelines),
            "last_speaking": {
                gid: time.time() - ts for gid, ts in self._last_speaking.items()
            },
        }

    async def _speak_greeting(self, pipeline, display_name: str) -> None:
        """Speak a greeting when a user joins the VC.

        Uses the LLM to generate a natural greeting, then speaks it via TTS.
        Falls back to a pre-written greeting if the LLM fails.
        """
        import random

        # Pre-written greetings as fallback and for speed
        greetings = [
            f"hey {display_name}, how are you doing?",
            f"oh hey {display_name}, what's going on?",
            f"hi {display_name}! how's your day been?",
            f"yo {display_name}, what are you up to?",
            f"hey {display_name}, you alright?",
            f"oh, {display_name}'s here. what's happening?",
        ]

        # Try LLM-generated greeting for more variety
        try:
            from ..ai import llm
            from ..ai import prompts as ai_prompts

            system = ai_prompts.VOICE_REPLY_SYSTEM
            user_prompt = (
                f"{display_name} just joined the voice call. "
                f"Greet them naturally — welcome them and ask how they're doing. "
                f"Keep it to 1-2 sentences. Use their name. Be warm but casual. "
                f"Say ONLY what you'd speak out loud."
            )

            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: llm.call_voice(
                    "voice_greeting", system, user_prompt,
                    want_json=False,
                )
            )

            if response and response.strip():
                # Clean up the response
                import re
                response = re.sub(r'\*+([^*]+)\*+', r'\1', response)
                response = response.replace('\n', ' ').strip()
                from ..ai.reply import clean_for_speech
                response = clean_for_speech(response)
                if response:
                    logger.info(f"[voice] LLM greeting for {display_name}: {response[:80]}")
                    await pipeline._speak(response)
                    return
        except Exception as e:
            logger.debug(f"[voice] LLM greeting failed: {e}")

        # Fallback to pre-written greeting
        greeting = random.choice(greetings)
        logger.info(f"[voice] Fallback greeting for {display_name}: {greeting}")
        await pipeline._speak(greeting)

    async def _speak_join_greeting(self, pipeline, names: list) -> None:
        """Speak a greeting when the bot joins a VC with people already in it."""
        import random

        if not names:
            greeting = "hey, just hopped in. what's everyone up to?"
        elif len(names) == 1:
            greetings = [
                f"hey {names[0]}, how are you doing?",
                f"oh hey {names[0]}, what's going on?",
                f"hi {names[0]}! how's your day been?",
                f"yo {names[0]}, what are you up to?",
            ]
            greeting = random.choice(greetings)
        else:
            names_str = ", ".join(names[:3])
            greetings = [
                f"hey everyone, {names_str}. what are we talking about?",
                f"oh hey {names_str}. what's going on?",
                f"hey guys. {names_str}, what are you all up to?",
            ]
            greeting = random.choice(greetings)

        logger.info(f"[voice] Join greeting: {greeting}")
        await pipeline._speak(greeting)
