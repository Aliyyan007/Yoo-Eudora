"""
Voice Channel (VC) management module for the Eudora persona.

Features:
1. Join VC when asked — recognizes "join vc", "hop in vc", "get in voice"
2. Detect VC from member voice states — sees who's in which VC
3. Ask which VC to join if multiple are active (human-like validation)
4. Mention the user who asked and say "I'm shy, can't open mic"
5. Don't repeat the "I'm shy" message to the same user (say "I already told u")
6. Send the shy message in the VC's text chat if allowed, otherwise the channel
   where the request was made

Algorithmic approach:
- Detects VC join requests via pattern matching
- Validates that VCs exist and have users in them
- Asks for clarification if multiple VCs are active
- Tracks which users have already been told about the shyness
- Falls back gracefully if voice connection fails
"""
import re
import time
import asyncio
import random
from typing import Optional, List, Tuple, Dict
from loguru import logger
import discord


# Patterns that indicate a VC join request
VC_JOIN_PATTERNS = [
    r'\b(join|hop\s+in|get\s+in|come\s+to|hop\s+on|get\s+on|come\s+to)\s+(?:the\s+)?(?:vc|voice|call)\b',
    r'\b(join|hop\s+in|get\s+in)\s+(?:the\s+)?vc\b',
    r'\b(vc|voice)\s+(?:join|come|hop)\b',
    r'\b(join|come)\s+(?:vc|voice)\b',
    r'\b(get\s+in\s+(?:the\s+)?(?:vc|voice|call))\b',
    r'\b(hop\s+in\s+(?:the\s+)?(?:vc|voice|call))\b',
    r'\b(i\'?m\s+in\s+(?:the\s+)?vc)\b.*\b(join|come|hop)\b',
    r'\b(come\s+to\s+vc)\b',
    r'\b(vc\s+me)\b',
    r'\b(join\s+my\s+(?:vc|voice|call))\b',
]

# Compiled patterns for performance
_VC_JOIN_REGEXES = [re.compile(p, re.IGNORECASE) for p in VC_JOIN_PATTERNS]


def detect_vc_join_request(text: str) -> bool:
    """
    Algorithmically detect if a message is asking the bot to join a voice channel.
    """
    if not text:
        return False
    for regex in _VC_JOIN_REGEXES:
        if regex.search(text):
            return True
    return False


def find_active_voice_channels(guild: discord.Guild) -> List[Tuple[discord.VoiceChannel, List[discord.Member]]]:
    """
    Find all voice channels in a guild that have at least one non-bot member.

    Returns a list of (channel, members_in_channel) tuples.
    """
    active = []
    for channel in guild.voice_channels:
        members = [m for m in channel.members if not m.bot]
        if members:
            active.append((channel, members))
    return active


def find_user_voice_channel(guild: discord.Guild, user_id: int) -> Optional[discord.VoiceChannel]:
    """
    Find which voice channel a specific user is in.
    Returns the channel or None if the user is not in VC.
    """
    for channel in guild.voice_channels:
        for member in channel.members:
            if member.id == user_id:
                return channel
    return None


def find_vc_by_name(guild: discord.Guild, name: str) -> Optional[discord.VoiceChannel]:
    """
    Find a voice channel by name (case-insensitive, partial match).
    """
    name_lower = name.lower().strip()
    for channel in guild.voice_channels:
        if name_lower in channel.name.lower():
            return channel
    return None


class VCManager:
    """
    Manages voice channel joining, validation, and the "I'm shy" message system.

    Algorithmic behavior:
    - When asked to join VC, first checks which VCs are active
    - If 0 active VCs: says "no one's in vc rn"
    - If 1 active VC: joins it directly
    - If 2+ active VCs: asks "which one?" (human-like validation)
    - After joining, mentions the requester and says "I'm shy, can't open mic"
    - If the same user asks again: says "I already told u I'm shy"
    - Tries to send the shy message in the VC's text chat, falls back to request channel
    """

    def __init__(self):
        # Track which users have been told about shyness (user_id -> timestamp)
        self._told_shy: Dict[int, float] = {}
        # Track current VC connection (guild_id -> channel_id)
        self._current_vc: Dict[int, int] = {}
        # Cooldown for shy message (don't repeat to same user within 30 min)
        self._shy_cooldown_s = 1800  # 30 minutes
        # VC intelligence: silence tracking, request channels, auto-behavior
        self._last_speaking: Dict[int, float] = {}  # guild_id -> timestamp of last speaking
        self._last_request_channel: Dict[int, int] = {}  # guild_id -> channel_id where join was requested
        self._silence_threshold_s = 180  # 3 minutes of silence → leave
        self._empty_check_interval_s = 30  # check every 30 seconds
        self._independent_join_chance = 0.05  # 5% chance to join an active VC independently

    def has_been_told_shy(self, user_id: int) -> bool:
        """Check if a user has already been told about the shyness recently."""
        last_told = self._told_shy.get(user_id, 0)
        return (time.time() - last_told) < self._shy_cooldown_s

    def mark_told_shy(self, user_id: int):
        """Mark that a user has been told about the shyness."""
        self._told_shy[user_id] = time.time()

    def is_in_vc(self, guild_id: int) -> bool:
        """Check if the bot is currently in a VC in this guild."""
        return guild_id in self._current_vc

    async def join_vc(
        self,
        client: discord.Client,
        guild: discord.Guild,
        channel: discord.VoiceChannel,
        requester: discord.Member,
        request_channel: discord.TextChannel,
    ) -> str:
        """
        Join a voice channel and send the "I'm shy" message.

        Returns the response message to send in the text channel.
        """
        try:
            # Join the voice channel (self_mute=True since we can't talk)
            await channel.connect(self_mute=True, self_deaf=False, reconnect=False)
            self._current_vc[guild.id] = channel.id
            self._last_request_channel[guild.id] = request_channel.id
            self._last_speaking[guild.id] = time.time()  # Reset silence timer on join
            logger.info(f"Joined VC '{channel.name}' in guild '{guild.name}' (self_mute=True)")

            # Determine the shy message
            if self.has_been_told_shy(requester.id):
                shy_msg = f"i already told u i'm shy {requester.mention}, can't open mic innit"
            else:
                shy_msg = f"{requester.mention} i'm shy, can't open mic 😳"
                self.mark_told_shy(requester.id)

            # Try to find a text channel associated with the VC
            # (some servers have a text chat linked to the VC)
            vc_text_channel = self._find_vc_text_channel(guild, channel)

            if vc_text_channel:
                # Send in the VC's text chat
                try:
                    await vc_text_channel.send(shy_msg)
                    logger.info(f"Sent shy message in VC text channel #{vc_text_channel.name}")
                    return f"hopping in {channel.name} 🎧"
                except discord.Forbidden:
                    # Can't send in VC text chat, send in request channel
                    await request_channel.send(shy_msg)
                    logger.info(f"Sent shy message in request channel (VC text chat forbidden)")
                    return f"hopping in {channel.name} 🎧"
            else:
                # No VC text chat, send in the request channel
                await request_channel.send(shy_msg)
                logger.info(f"Sent shy message in request channel (no VC text chat)")
                return f"hopping in {channel.name} 🎧"

        except discord.Forbidden:
            logger.warning(f"Failed to join VC '{channel.name}': no permission")
            return "can't join that vc, don't have perms bruv"
        except Exception as e:
            logger.error(f"Failed to join VC '{channel.name}': {e}")
            return f"couldn't join the vc, something went wrong"

    async def leave_vc(self, client: discord.Client, guild: discord.Guild) -> str:
        """Leave the current voice channel."""
        if guild.id not in self._current_vc:
            return "i'm not in a vc rn"

        try:
            # Find the voice client for this guild
            voice_client = discord.utils.get(client.voice_clients, guild=guild)
            if voice_client:
                await voice_client.disconnect()
                logger.info(f"Left VC in guild '{guild.name}'")
            del self._current_vc[guild.id]
            return "left the vc 👋"
        except Exception as e:
            logger.error(f"Failed to leave VC: {e}")
            return "couldn't leave the vc"

    def _find_vc_text_channel(
        self,
        guild: discord.Guild,
        vc_channel: discord.VoiceChannel,
    ) -> Optional[discord.TextChannel]:
        """
        Algorithmically find a text channel associated with a voice channel.
        Discord sometimes has a text chat linked to VC channels.
        We look for:
        1. A channel with the same name + "-chat" or "-text"
        2. A channel that is a child of the same category as the VC
        3. A channel whose topic mentions the VC name
        """
        vc_name = vc_channel.name.lower().replace(" ", "-")

        # Strategy 1: Look for a text channel with similar name
        for tc in guild.text_channels:
            tc_name = tc.name.lower()
            if tc_name == vc_name + "-chat" or tc_name == vc_name + "-text":
                return tc
            if tc_name == vc_name:
                return tc

        # Strategy 2: Look for a text channel in the same category
        if vc_channel.category:
            for tc in guild.text_channels:
                if tc.category and tc.category.id == vc_channel.category.id:
                    # Check if the name seems related
                    if any(word in tc.name.lower() for word in vc_channel.name.lower().split()):
                        return tc

        # Strategy 3: Look for a general "voice-chat" or "vc-chat" channel
        for tc in guild.text_channels:
            tc_name = tc.name.lower()
            if tc_name in ("voice-chat", "vc-chat", "vc-text", "voice-text", "voice"):
                return tc

        return None

    def get_status(self) -> dict:
        """Get VC status for debugging."""
        return {
            "current_vcs": dict(self._current_vc),
            "told_shy_users": len(self._told_shy),
            "last_speaking": dict(self._last_speaking),
            "last_request_channels": dict(self._last_request_channel),
        }

    # ── VC Intelligence ──────────────────────────────────────────────────────

    def should_leave_vc(self, guild: discord.Guild) -> Tuple[bool, str]:
        """
        Check if the bot should leave the current VC.
        Returns (should_leave, reason).

        Leaves if:
        - The bot is in a VC but there are 0 non-bot members (empty VC)
        - The bot is the only one in the VC
        """
        if guild.id not in self._current_vc:
            return (False, "")

        # Find the VC the bot is in
        vc_id = self._current_vc[guild.id]
        vc_channel = guild.get_channel(vc_id)
        if vc_channel is None:
            # Channel no longer exists — should leave (cleanup)
            return (True, "vc's gone, dipping out 👋")

        non_bot_members = [m for m in vc_channel.members if not m.bot]
        if len(non_bot_members) == 0:
            return (True, "vc's empty, dipping out 👋")

        return (False, "")

    def record_speaking(self, guild_id: int, user_id: int):
        """Record that a user spoke in the VC (called from voice state update)."""
        # Only track if the bot is actually in a VC in this guild
        if guild_id in self._current_vc:
            self._last_speaking[guild_id] = time.time()
            logger.debug(f"Recorded speaking in guild {guild_id} by user {user_id}")

    def get_silence_duration(self, guild_id: int) -> float:
        """Return seconds since last speaking activity in the VC."""
        last = self._last_speaking.get(guild_id, 0)
        if last == 0:
            # No speaking recorded yet — treat as infinite silence
            return float("inf")
        return time.time() - last

    def find_best_vc_to_join(self, guild: discord.Guild) -> Optional[discord.VoiceChannel]:
        """
        Find the most active VC to join independently.
        Only suggests VCs with 2+ non-bot members (don't join empty or solo VCs).
        Returns the best VC or None.
        """
        active = find_active_voice_channels(guild)
        candidates = []
        for channel, members in active:
            if len(members) < 2:
                continue  # Don't join solo or empty VCs
            # Score by member count (more members = more interesting)
            score = len(members)
            # Bonus for recent speaking activity
            silence = self.get_silence_duration(guild.id)
            if silence != float("inf") and silence < self._silence_threshold_s:
                # Recent activity — boost score
                score += max(0, int((self._silence_threshold_s - silence) / 60))
            candidates.append((score, channel))

        if not candidates:
            return None

        # Pick the highest-scoring VC
        candidates.sort(key=lambda c: c[0], reverse=True)
        return candidates[0][1]

    def should_ask_reason(self, text: str) -> bool:
        """
        Determine if the bot should ask for a reason before joining VC.
        - If the user already gave a reason in their message → don't ask, just join
        - Otherwise 70% chance of asking for a reason, 30% chance of joining directly
        Returns True if the bot should ask for a reason first.
        """
        if not text:
            return random.random() < 0.7

        # Heuristic: if the message contains reason-indicating words, skip asking
        reason_keywords = [
            "music", "listen", "game", "gaming", "play", "chat", "talk",
            "hang", "hangout", "watch", "stream", "study", "work", "sing",
            "karaoke", "event", "party", "because", "to ", "for ", "so i",
            "so we", "wanna", "want to", "need", "help",
        ]
        text_lower = text.lower()
        if any(kw in text_lower for kw in reason_keywords):
            return False

        return random.random() < 0.7

    def generate_reason_question(self) -> str:
        """Generate a casual question asking why they want the bot to join."""
        variations = [
            "what for?",
            "why, what's happening in there?",
            "what u guys doing in vc?",
            "why tho? what's going on?",
            "what's the vibe in there?",
            "what u lot up to in vc?",
            "why should i hop in?",
            "what's happening in vc then?",
        ]
        return random.choice(variations)

    async def auto_leave(self, client: discord.Client, guild: discord.Guild, reason: str):
        """Leave the VC — silently, never a text announcement."""
        try:

            # Disconnect
            voice_client = discord.utils.get(client.voice_clients, guild=guild)
            if voice_client:
                await voice_client.disconnect()
                logger.info(f"Auto-left VC in guild '{guild.name}': {reason}")

            # Cleanup tracking
            self._current_vc.pop(guild.id, None)
            self._last_speaking.pop(guild.id, None)
            self._last_request_channel.pop(guild.id, None)
        except Exception as e:
            logger.error(f"Failed to auto-leave VC in guild '{guild.name}': {e}")

    async def monitor_vcs(self, client: discord.Client):
        """
        Background task: monitor VCs and auto-leave/join as needed.
        Call this from discord_client.py on_ready() as asyncio.create_task().
        """
        logger.info("VC monitor background task started")
        while True:
            try:
                await asyncio.sleep(self._empty_check_interval_s)
                for guild in client.guilds:
                    try:
                        if self.is_in_vc(guild.id):
                            # Bot is in a VC — check if it should leave
                            should_leave, reason = self.should_leave_vc(guild)
                            if should_leave:
                                await self.auto_leave(client, guild, reason)
                                continue

                            # Check for silence
                            silence = self.get_silence_duration(guild.id)
                            if silence != float("inf") and silence > self._silence_threshold_s:
                                await self.auto_leave(client, guild, "it's dead in here, gonna dip 👋")
                        else:
                            # Bot is NOT in a VC — small chance to join an active one
                            best_vc = self.find_best_vc_to_join(guild)
                            if best_vc is not None and random.random() < self._independent_join_chance:
                                try:
                                    await best_vc.connect(self_mute=True, self_deaf=False, reconnect=False)
                                    self._current_vc[guild.id] = best_vc.id
                                    self._last_speaking[guild.id] = time.time()
                                    logger.info(f"Independently joined VC '{best_vc.name}' in guild '{guild.name}'")
                                except discord.Forbidden:
                                    logger.debug(f"Could not independently join VC '{best_vc.name}' (forbidden)")
                                except Exception as e:
                                    logger.debug(f"Could not independently join VC '{best_vc.name}': {e}")
                    except Exception as e:
                        logger.debug(f"Error monitoring guild '{getattr(guild, 'name', '?')}': {e}")
            except asyncio.CancelledError:
                logger.info("VC monitor background task cancelled")
                break
            except Exception as e:
                logger.debug(f"VC monitor loop error: {e}")


# Singleton instance
_vc_manager = VCManager()


def get_vc_manager() -> VCManager:
    """Get the global VC manager instance."""
    return _vc_manager
