"""
Main Discord client — the heart of the AI persona system.
Handles: message events, reactions, AFK simulation, daily caps,
mood management, memory updates, channel scanning, and more.
"""
import re
import json
import os
import random
import asyncio
import time
from collections import deque
from typing import Optional, Set
from loguru import logger
import discord

from .ai import reply as ai_reply
from .ai import d1_memory as mem  # D1-backed memory (falls back to JSON if D1 unavailable)
from .ai import search as web_search
from .ai.mood import MoodEngine, MOODS
from .ai.engagement import EngagementEngine
from .ai.sentiment import detect_sentiment, get_sentiment_timing_modifier, get_sentiment_context, detect_loneliness
from .ai.commands import detect_command
from .ai.msg_cache import get_cache
from .ai.vc_manager import get_vc_manager, find_active_voice_channels, find_user_voice_channel, detect_vc_join_request, find_vc_by_name
try:
    from .voice import VoiceManager
    from .voice.tts import TTSConfig
except ImportError as _voice_err:
    VoiceManager = None
    TTSConfig = None
    _VOICE_IMPORT_ERROR = _voice_err
else:
    _VOICE_IMPORT_ERROR = None
from .ai.mention_system import get_mention_manager
from .ai.abuse_handler import get_abuse_handler
from .ai.multi_message import determine_message_count, split_into_messages, detect_conversation_nature
from .ai.channel_nature import analyze_channel_nature, get_cached_nature, get_nature_summary
from .ai.reactions import react_to_message, get_tracker
from .ai.stickers_gifs import get_sticker_manager, get_gif_manager, is_gif_request
from .ai.status_manager import get_status_manager
from .ai.user_behavior import should_store_message, extract_potential_facts, get_engagement_tracker
from .ai.unanswered_questions import get_unanswered_tracker, is_question
from .ai.owner_system import get_owner_context, get_non_owner_command_context, detect_command_attempt, get_owner_tracker, is_owner
from .ai.re_engagement import get_ping_controller, get_re_engagement_tracker, select_online_user
from .ai.engagement_log import get_engagement_log
from .ai.conversational_initiative import get_engagement_analyzer, get_conversational_initiative_context
from .ai.welcome_system import get_welcome_system
from .channel_scanner import (
    can_speak_in, is_skip_channel,
    fetch_guild_rules, scan_guild,
)
from .proactive_messaging import ProactiveMessenger


URL_PATTERN = re.compile(r'https?://\S+')

# ── Safety / rate-limit constants ─────────────────────────────────────────────
CHANNEL_COOLDOWN_S = 8      # seconds between replies in same channel
DM_COOLDOWN_S = 12          # seconds between DM replies
DAILY_CAP_PER_CH = 60       # max replies per channel per day
DAILY_CAP_GLOBAL = 400      # max replies globally per day

# ── Greeting detection ────────────────────────────────────────────────────────
# When someone says hi/hello/heya etc. in ANY channel, the bot responds.
# This makes the bot feel welcoming and alive across the whole server.
GREETING_PATTERNS = re.compile(
    r'^(?:'
    r'h+i+\b|h+e+l+l+o+\b|h+e+y+a*\b|h+i+y+a+\b|'
    r'y+o+\b|s+u+p+\b|w+a+s+s+u+p+\b|w+h+a+t+\s*\'?s+\s*u+p+\b|'
    r'h+o+l+a+\b|h+i+y+a+\b|g+o+o+d+\s+m+o+r+n+i+n+g+\b|'
    r'g+o+o+d+\s+a+f+t+e+r+n+o+o+n+\b|g+o+o+d+\s+e+v+e+n+i+n+g+\b|'
    r'h+e+l+l+o+w+\b|h+e+y+y+\b|h+e+y+\b'
    r')[!?.~]*\s*$',
    re.IGNORECASE,
)


def is_greeting(text: str) -> bool:
    """Check if a message is a greeting (hi, hello, heya, yo, sup, etc.)."""
    text = text.strip().lower()
    if not text or len(text) > 30:
        return False
    # Quick check before regex
    first_word = text.split()[0] if text.split() else ""
    quick_greetings = {"hi", "hello", "hey", "heya", "hiya", "yo", "sup",
                       "wassup", "hola", "hey", "heyy", "hii", "hiii", "hiiii",
                       "helloo", "heyaa", "yoo", "yooo"}
    if first_word in quick_greetings:
        return True
    return bool(GREETING_PATTERNS.match(text))


# Bare "wbu?/hru?/wyd?" bursts — fine once, robotic when it follows every reply
_GENERIC_FOLLOWUP_STEMS = {"wbu", "hbu", "hru", "wyd", "u", "you", "wby", "hby"}


def _is_generic_followup(text: str) -> bool:
    """True for short bare follow-up pings like 'wbu?', 'wbu lmao', 'u?'."""
    words = re.sub(r'[^a-z\s?]', '', text.lower()).split()
    return 0 < len(words) <= 4 and words[0].rstrip('?') in _GENERIC_FOLLOWUP_STEMS


def dynamic_reply_chance() -> float:
    """
    Return a reply probability tuned to the time of day.
    Lower chances = bot is more selective, sounds less like an AI that
    responds to everything. Tuned for natural human-like participation.
    """
    import datetime
    utc_now = datetime.datetime.utcnow()
    hour = utc_now.hour
    if 22 <= hour or hour < 2:    # 10 PM - 2 AM (peak)
        return 0.35
    elif 18 <= hour < 22:          # 6 PM - 10 PM (evening)
        return 0.30
    elif 8 <= hour < 18:           # 8 AM - 6 PM (daytime)
        return 0.28
    else:                          # 2 AM - 8 AM (dead)
        return 0.12


def is_ignorable(message: discord.Message) -> bool:
    """Return True for bots, webhooks, or pure URL/emoji-only messages (but NOT images)."""
    if message.author.bot:
        return True
    if message.webhook_id:
        return True
    # If message has image attachments, always process it
    if any(a.content_type and a.content_type.startswith("image/") for a in message.attachments):
        return False
    content = message.content.strip()
    if not content or len(content) < 2:
        return True
    # Strip URLs and custom emoji — if nothing left, skip
    stripped = URL_PATTERN.sub('', content).strip()
    stripped = re.sub(r'<a?:\w+:\d+>', '', stripped).strip()
    if len(stripped) < 2:
        return True
    return False


# Distress signals — we skip these to avoid insensitive AI responses
DISTRESS_SIGNALS = [
    "im a bad person", "i'm a bad person", "feel useless", "i'm useless",
    "im useless", "contribute nothing", "nobody needs me", "want to die",
    "wanna die", "kill myself", "end it", "give up", "can't go on",
    "horrible person", "worthless", "cant get out of bed", "no point",
]


class AIPersonaClient(discord.Client):
    """
    The main AI persona Discord client.
    Simulates a real human user with moods, memory, and human-like behavior.
    """

    def __init__(self, **kwargs):
        # Stealth: don't chunk guilds at startup (minimizes API traffic)
        super().__init__(chunk_guilds_at_startup=False, **kwargs)

        # Mood engine
        self.mood_engine = MoodEngine()

        # Reply tracking
        self.last_replied: dict = {}       # channel_id -> timestamp
        self.last_sent: dict = {}          # channel_id -> last reply text
        self.reply_history: dict = {}      # channel_id -> deque of last 8 replies
        self.processed_msgs = deque(maxlen=200)
        self.history_cache: dict = {}      # channel_id -> deque of last 40 messages
        self.engager_leaves = deque(maxlen=1000)   # action worker: recent leaves feed

        # Side-channel action worker (vendored Engager tool loop) — env knobs:
        # ACTIONS_ENABLED=0 kills the whole feature; ACTIONS_OWNER_ONLY=0
        # lets non-owner directed action requests run (owner-gated tools like
        # schedule/prefs still refuse internally).
        self._actions_enabled = os.getenv("ACTIONS_ENABLED", "1") == "1"
        self._actions_owner_only = os.getenv("ACTIONS_OWNER_ONLY", "1") == "1"

        # Daily message counters
        self.daily_count_ch: dict = {}
        self.daily_count_global: int = 0
        self.daily_reset_day: int = -1

        # AFK simulation
        self.afk_until: dict = {}          # channel_id -> timestamp when AFK ends

        # Backoff: when told to go away / not talked to, stop replying for 10 min
        self.backoff_until: dict = {}      # channel_id -> timestamp when backoff ends

        # Reaction dedup
        self.recent_reactions: dict = {}   # channel_id -> deque of last 3 emoji

        # Channel activity tracking (for proactive messaging)
        self.last_activity: dict = {}      # channel_id -> timestamp

        # Conversation stickiness: channel_id -> timestamp until which we
        # respond to ALL messages in that channel (no ping/reply needed).
        # Set when we reply, lasts 2 minutes. Keeps conversations flowing naturally
        # without creating a feedback loop where the bot answers everything.
        self.sticky_until: dict = {}      # channel_id -> timestamp
        STICKY_DURATION_S = 120           # 2 minutes (was 5 min — too aggressive)

        # Multi-user conversation tracker: channel_id -> {user_id -> last_bot_interaction_ts}
        # Tracks when the bot last interacted with each specific user per channel.
        # Used to ensure the bot replies to users it pinged (even after sticky expires).
        self.conversation_tracker: Dict[str, Dict[str, float]] = {}
        CONVERSATION_TRACKER_EXPIRY_S = 600  # 10 minutes (was 30 min — too long)

        # Rules cache: guild_id -> rules_text
        self.rules_cache: dict = {}

        # Bot's own profile
        self.my_profile_text: str = ""
        self.current_status_text: Optional[str] = None

        # Proactive messenger (set externally)
        self.proactive_messenger: Optional[ProactiveMessenger] = None

        # Bump scheduler (set externally)
        self.bump_scheduler = None

        # Engagement engine (MaiBot-pattern willingness scoring)
        self.engagement = EngagementEngine()

        # Message cache with TTL (3 days general, 2 months important)
        self.msg_cache = get_cache()

        # VC (Voice Channel) manager (legacy — for text-based VC join/leave detection)
        self.vc_manager = get_vc_manager()

        # Voice manager (real-time voice conversation: VAD + ASR + LLM + TTS)
        self.voice_manager = None  # Initialized in on_ready() when we have self.user.id

        # Voice conversation history: guild_id -> deque of (speaker, text) turns.
        # Gives the LLM context across turns in a voice call so it doesn't treat
        # each utterance in isolation (makes replies feel like a real conversation).
        self.voice_history: dict = {}

        # Per-user cross-modal context — links a user's text messages with
        # their voice utterances so the bot connects what they typed and what
        # they said aloud. user_id -> deque[(ts, text)]
        self._user_recent_texts: dict = {}
        self._user_recent_voice: dict = {}

        # Recently-joined members (user_id -> join ts) — their messages get a
        # reply-probability boost so they integrate into normal engagement
        self._recent_joins: dict = {}

        # Designated engagement channels (TEXT_CHANNEL_IDS) — these are the
        # bot's home turf; utility-channel gates never silence them
        self._engage_channel_ids: set = set()
        _tc = os.getenv("TEXT_CHANNEL_IDS", "")
        for _p in _tc.replace(";", ",").split(","):
            _p = _p.strip()
            if _p.isdigit():
                self._engage_channel_ids.add(int(_p))

        # Mention system manager
        self.mention_manager = get_mention_manager()

        # Abuse handler
        self.abuse_handler = get_abuse_handler()

        # Status/bio manager
        self.status_manager = get_status_manager()

        # Unanswered question tracker
        self.unanswered_tracker = get_unanswered_tracker()

        # Speed multiplier (adjustable via "reply faster" / "slow down" commands)
        self._speed_multiplier = 1.0

        # Per-channel send lock (prevent double-send race condition)
        self._sending_in_channel: Set[str] = set()
        # Per-channel message queue (messages that arrive while bot is typing)
        self._pending_messages: Dict[str, deque] = {}

    # ── Daily cap management ───────────────────────────────────────────────

    def _reset_daily_counters_if_needed(self):
        """Reset per-day message counters at the start of each new day."""
        import datetime
        today = datetime.date.today().toordinal()
        if today != self.daily_reset_day:
            self.daily_count_ch.clear()
            self.daily_count_global = 0
            self.daily_reset_day = today
            logger.info("Daily message counters reset")

    def can_send(self, ch_id: str) -> bool:
        """Return True if we are still under daily message caps."""
        self._reset_daily_counters_if_needed()
        if self.daily_count_global >= DAILY_CAP_GLOBAL:
            logger.debug(f"Global daily cap ({DAILY_CAP_GLOBAL}) reached — skipping")
            return False
        if self.daily_count_ch.get(ch_id, 0) >= DAILY_CAP_PER_CH:
            logger.debug(f"Channel daily cap ({DAILY_CAP_PER_CH}) reached for {ch_id}")
            return False
        return True

    def record_send(self, ch_id: str):
        """Increment daily counters after a message is sent."""
        self.daily_count_ch[ch_id] = self.daily_count_ch.get(ch_id, 0) + 1
        self.daily_count_global += 1

    def _get_status_for_mood(self, mood: str) -> str:
        """
        Algorithmically generate a custom Discord status based on the bot's mood.
        Returns a short status text that fits the Eudora persona.
        """
        mood_statuses = {
            "flat": "existing",
            "bored": "so bored innit",
            "giddy": "feeling good today",
            "hyped": "let's gooo",
            "chill": "vibing rn",
            "playful": "in a silly mood",
            "proud": "nailed it",
            "annoyed": "bruv allow it",
            "sad": "not feeling great",
            "angry": "don't even",
            "tired": "so tired rn",
            "curious": "thinking about stuff",
            "nostalgic": "reminiscing",
            "lonely": "anyone up?",
            "excited": "can't wait fr",
        }
        return mood_statuses.get(mood, "just vibing")

    # ── Event handlers ─────────────────────────────────────────────────────

    async def on_ready(self):
        """Called when the bot is logged in and ready."""
        logger.info(f"Logged in as {self.user.name} (ID: {self.user.id})")
        logger.info(f"Mood: {self.mood_engine.current_mood} | Groq keys: {ai_reply.llm.get_key_count()}")

        # Set Discord token for native GIF search
        get_gif_manager().set_discord_token(self.http.token)

        # Start engagement engine
        self.engagement.start()

        # ── Set up persona profile (bio + display name) ───────────────────
        # Update the Discord profile to match the Eudora Edward persona
        try:
            from .persona import load_persona
            import yaml, os
            _cfg_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "config.yaml")
            with open(_cfg_path, "r", encoding="utf-8") as _f:
                _cfg = yaml.safe_load(_f) or {}
            _persona = _cfg.get("persona", {})
            _bio = _persona.get("bio", "")
            _display_name = _persona.get("full_name", _persona.get("name", ""))

            # Update display name (global name) and bio
            if _bio or _display_name:
                try:
                    await self.user.edit(
                        global_name=_display_name if _display_name else ...,
                        bio=_bio if _bio else ...,
                    )
                    logger.info(f"Profile updated: name={_display_name}, bio={len(_bio)} chars")
                except Exception as e:
                    logger.warning(f"Could not edit profile (may need password): {e}")

            # Set custom status based on mood
            mood = self.mood_engine.current_mood
            status_text = self._get_status_for_mood(mood)
            if status_text:
                try:
                    activity = discord.CustomActivity(name=status_text)
                    await self.change_presence(activity=activity, status=discord.Status.online)
                    logger.info(f"Set custom status: {status_text}")
                except Exception as e:
                    logger.warning(f"Could not set custom status: {e}")
        except Exception as e:
            logger.warning(f"Persona profile setup failed: {e}")

        # Fetch our own profile
        try:
            profile = await self.fetch_user_profile(self.user.id)
            bio = getattr(profile, "bio", "") or ""
            pronouns = getattr(profile, "pronouns", "") or ""
            self.my_profile_text = f"My display name: {self.user.display_name}\n"
            self.my_profile_text += f"My name: Eudora Edward (goes by Eudora)\n"
            self.my_profile_text += f"My age: 22\n"
            self.my_profile_text += f"My location: London, England\n"
            self.my_profile_text += f"My heritage: Half-French, British\n"
            if pronouns:
                self.my_profile_text += f"My pronouns: {pronouns}\n"
            if bio:
                self.my_profile_text += f"My bio/About Me: {bio}\n"
            logger.info(f"Fetched own profile (Bio: {len(bio)} chars)")
        except Exception as e:
            logger.warning(f"Failed to fetch own profile: {e}")
            self.my_profile_text = f"My display name: {self.user.display_name}\n"
            self.my_profile_text += f"My name: Eudora Edward (goes by Eudora)\n"

        # Scan all guilds: discover channels, fetch rules, learn styles
        for guild in self.guilds:
            asyncio.create_task(scan_guild(guild, self.rules_cache))

        # Start proactive messaging
        if self.proactive_messenger:
            asyncio.create_task(self.proactive_messenger.start_monitoring())
            asyncio.create_task(self.proactive_messenger.proactive_loop())

        # Action worker: scheduled-message executor (schedule_message tool
        # persistence + revival). Independent of the reply path — failure is
        # isolated.
        try:
            from .actions.core import scheduler as _action_scheduler
            _action_scheduler.init(self)
            asyncio.create_task(_action_scheduler.scheduler.start())
            logger.info("Action worker scheduler started")
        except Exception as e:
            logger.warning(f"Action worker scheduler unavailable: {e}")

        # Start periodic status/bio updates
        asyncio.create_task(self._status_update_loop())

        # Start re-engagement loop (check if no one replied to bot's messages)
        asyncio.create_task(self._re_engagement_loop())

        # ── Initialize voice manager (real-time voice conversation) ────────
        try:
            fish_api_key = os.getenv("FISH_AUDIO_API_KEY", "")
            fish_voice_id = os.getenv("FISH_AUDIO_VOICE_ID", "")
            if fish_api_key and fish_voice_id and VoiceManager is not None:
                tts_config = TTSConfig(
                    api_key=fish_api_key,
                    voice_id=fish_voice_id,
                    fallback_voice_id=os.getenv("FISH_AUDIO_FALLBACK_VOICE_ID", ""),
                    model=os.getenv("FISH_AUDIO_MODEL", "s2.1-pro-free"),
                )
                self.voice_manager = VoiceManager(
                    tts_config=tts_config,
                    on_transcript=self._handle_voice_transcript,
                    bot_id=self.user.id,
                    on_transcript_stream=self._stream_voice_transcript,
                )
                asyncio.create_task(self.voice_manager.monitor_vcs(self))
                logger.info("Voice manager initialized (Fish Audio TTS + VAD + ASR)")
            else:
                if VoiceManager is None:
                    logger.warning(f"Voice dependencies unavailable — voice disabled ({_VOICE_IMPORT_ERROR})")
                else:
                    logger.info("Voice manager not initialized (FISH_AUDIO_API_KEY or FISH_AUDIO_VOICE_ID not set)")
        except Exception as e:
            logger.warning(f"Voice manager init failed: {e}")

        logger.info("Bot is ready and listening for messages.")

    async def _status_update_loop(self):
        """Periodically update Discord status and bio."""
        await asyncio.sleep(60)  # Wait 1 min before first update
        while True:
            try:
                # Update status based on current mood
                await self.status_manager.update_status(self, self.mood_engine.current_mood)
                # Check for monthly bio update
                await self.status_manager.update_bio(self)
            except Exception as e:
                logger.debug(f"Status update loop error: {e}")
            await asyncio.sleep(1800)  # Check every 30 minutes

    # ── Voice conversation handlers ────────────────────────────────────────

    def _get_voice_history(self, guild_id: int) -> deque:
        """Get (or create) the conversation-history deque for a guild's VC."""
        if guild_id not in self.voice_history:
            self.voice_history[guild_id] = deque(maxlen=10)  # last 10 turns
        return self.voice_history[guild_id]

    def _format_voice_history(self, history: deque) -> str:
        """Render the voice conversation history as a readable back-and-forth."""
        if not history:
            return ""
        return "\n".join(f"{speaker}: {text}" for speaker, text in history)

    def _find_voice_guild_id(self, user_id: int) -> Optional[int]:
        """Find which guild's VC the given user is speaking in (with the bot)."""
        for vc in self.voice_clients:
            try:
                if vc.channel and any(m.id == user_id for m in vc.channel.members):
                    return vc.guild.id
            except Exception:
                continue
        return None

    async def _build_voice_prompt(self, user_id: int, transcript: str, hints: dict = None):
        """Shared prep for voice replies: display name, D1 memory lookup
        (capped so a slow remote call can't stall a live turn), mood,
        conversation history, greeting/other-user detection, and the final
        system + user prompts. Returns (username, history, system, user)."""
        from .ai import prompts as ai_prompts

        user = self.get_user(user_id)
        username = user.display_name if user else f"user_{user_id}"

        loop = asyncio.get_running_loop()

        # Kick off the D1 memory lookup in a worker thread FIRST so the
        # remote query overlaps with the (instant) local prep below.
        memory_task = loop.run_in_executor(
            None, mem.get_user_memory_text, str(user_id), username
        )

        # Get current mood for tone matching
        mood = self.mood_engine.current_mood if self.mood_engine else ""

        # Build conversation history so the bot has context across turns.
        # Format BEFORE appending the new transcript — the current line is
        # already included verbatim in the prompt below, so leaving it in
        # the history block too would send it twice (bigger prompt, slower TTFT).
        guild_id = self._find_voice_guild_id(user_id)
        history = self._get_voice_history(guild_id) if guild_id else deque()
        history_text = self._format_voice_history(history)
        # Record what the user just said — deduped because the streaming path
        # may call this again via the non-streaming fallback
        if not history or history[-1] != (username, transcript):
            history.append((username, transcript))

        # Cross-modal memory: remember their recent voice lines so a later
        # text reply can reference what they said in the call
        from collections import deque as _dq
        vq = self._user_recent_voice.setdefault(user_id, _dq(maxlen=5))
        vq.append((time.time(), transcript[:200]))

        # Detect if this is a greeting (hey, hi, yo, what's up, etc.)
        transcript_lower = transcript.lower().strip()
        greeting_words = {"hi", "hey", "hello", "yo", "sup", "hiya", "heya", "what's up", "whats up", "howdy", "morning", "evening"}
        is_greeting = any(transcript_lower == w or transcript_lower.startswith(w + " ") for w in greeting_words)

        # Get other users in the call (for the bot to be aware of multi-user context)
        other_users = ""
        if guild_id:
            for vc in self.voice_clients:
                if vc.guild.id == guild_id and vc.channel:
                    others = [m.display_name for m in vc.channel.members if not m.bot and m.id != user_id]
                    if others:
                        other_users = ", ".join(others[:5])
                    break

        # ── Spoken action request → side-channel worker ───────────────
        # Pipeline flagged this utterance as action-ish (leave/sing/etc are
        # already filtered out upstream). Owner-only by default; the worker
        # runs silently and the reply just acknowledges it in one line.
        if (hints or {}).get("action") and is_owner(str(user_id)):
            try:
                action_dir = await self._run_vc_action(user_id, transcript)
                if action_dir:
                    hints = {**(hints or {}), "directive": (hints.get("directive", "") + " " + action_dir).strip()}
            except Exception as e:
                logger.warning(f"[voice action] worker failed: {e}")

        # Wait for the memory lookup — capped so a slow D1 can never stall
        # a live voice turn. On timeout the reply is just less personalised.
        try:
            user_memory = await asyncio.wait_for(memory_task, timeout=0.9)
        except Exception:
            user_memory = ""

        # Cross-modal context: pull what they recently TYPED in text channels
        # (last ~10min) so the voice reply can connect both
        recent_texts = ""
        texts = self._user_recent_texts.get(user_id)
        if texts:
            fresh = [t for ts, t in texts if time.time() - ts < 600]
            if fresh:
                recent_texts = " / ".join(f'"{t}"' for t in fresh[-3:])

        # Build prompts using the dedicated voice system + builder
        system_prompt = ai_prompts.VOICE_REPLY_SYSTEM
        user_prompt = ai_prompts.build_voice_reply_prompt(
            username=username,
            transcript=transcript,
            user_memory=user_memory,
            conversation_history=history_text,
            mood=mood,
            is_greeting=is_greeting,
            other_users=other_users,
            extra_directive=(hints or {}).get("directive", ""),
            recent_texts=recent_texts,
        )
        return username, history, system_prompt, user_prompt

    async def _stream_voice_transcript(self, user_id: int, transcript: str, hints: dict = None):
        """Async generator: yields reply text pieces as the LLM streams them.

        The pipeline consumes this directly into TTS — each completed sentence
        is pushed to Fish Audio while the model is still generating the next.
        Yields nothing if streaming can't start (caller falls back to
        _handle_voice_transcript for the full response).
        """
        try:
            username, history, system_prompt, user_prompt = await self._build_voice_prompt(user_id, transcript, hints)

            from .ai import llm
            from .ai.reply import clean_for_speech

            loop = asyncio.get_running_loop()

            # Start the stream in a worker — the sync Groq call blocks on the
            # handshake + first token, and returns None if no key could stream
            stream = await loop.run_in_executor(
                None,
                lambda: llm.call_voice_stream("voice_reply", system_prompt, user_prompt),
            )
            if stream is None:
                return

            # Bridge sync deltas (executor thread) → asyncio queue
            q: asyncio.Queue = asyncio.Queue()

            def _pump():
                try:
                    for piece in stream:
                        loop.call_soon_threadsafe(q.put_nowait, piece)
                finally:
                    loop.call_soon_threadsafe(q.put_nowait, None)

            loop.run_in_executor(None, _pump)

            pieces = []
            while True:
                piece = await q.get()
                if piece is None:
                    break
                pieces.append(piece)
                yield piece

            # Record the cleaned full reply in conversation history
            full = clean_for_speech("".join(pieces))
            if full.strip():
                history.append(("Eudora", full))
                logger.info(f"[voice] Streamed reply to {username}: {full[:80]}")
        except Exception as e:
            logger.error(f"Voice stream handler error: {e}")

    async def _handle_voice_transcript(self, user_id: int, transcript: str, hints: dict = None) -> str:
        """Called when a user's speech is transcribed in a voice channel.
        Generates an LLM response using the Groq reply system with a
        voice-optimised prompt (full words, follow-up questions, no text slang).
        Returns the response text to be spoken via TTS.

        Latency notes:
        - The D1 memory lookup is a remote REST call, so it's fired off in a
          worker thread and awaited with a hard cap — a slow lookup only means
          a less-personalised reply, and the fetch still completes in the
          background which warms the cache for the next turn.
        - Replies use llm.call_voice: gpt-oss-20b at reasoning_effort=low with
          a gpt-oss-120b fallback and a bounded key-wait so rate-limit
          saturation fails fast instead of stalling a live voice turn.
        """
        try:
            # Get the user's display name
            username, history, system_prompt, user_prompt = await self._build_voice_prompt(user_id, transcript, hints)

            from .ai import llm
            from .ai.reply import clean_for_speech

            loop = asyncio.get_running_loop()

            # call_voice: gpt-oss-20b (reasoning_effort=low) → gpt-oss-120b
            # fallback on a separate rate-limit pool, bounded wait.
            response = await loop.run_in_executor(
                None,
                lambda: llm.call_voice(
                    "voice_reply", system_prompt, user_prompt,
                    want_json=False,
                )
            )

            if not response:
                # Last resort — a key may have freed since call_voice's chain ran
                logger.warning(f"[voice] call_voice returned empty, falling back to call_smart")
                response = await loop.run_in_executor(
                    None,
                    lambda: llm.call_smart(
                        "voice_reply", system_prompt, user_prompt,
                        max_tokens=500, temperature=0.8, want_json=False,
                        max_wait_s=8,
                    )
                )

            if not response:
                logger.warning(f"[voice] LLM returned empty for transcript: '{transcript[:60]}' (all keys may be rate-limited)")
                return ""

            # Clean up any markdown / JSON wrapping / text abbreviations
            stripped = response.strip()
            if stripped.startswith("{") and "response" in stripped:
                try:
                    parsed = json.loads(stripped)
                    if isinstance(parsed, dict) and "response" in parsed:
                        response = parsed["response"]
                except Exception:
                    pass
            # Remove markdown bold/italic markers
            response = re.sub(r'\*+([^*]+)\*+', r'\1', response)
            # Collapse newlines into spaces (single spoken line)
            response = response.replace('\n', ' ').strip()
            # Expand text abbreviations into full words for natural TTS speech
            response = clean_for_speech(response)

            # If generation was cut off mid-sentence at the token cap, drop the
            # dangling fragment — TTS speaking half a word sounds broken.
            if response and response[-1] not in '.!?"\'':
                last_end = max(response.rfind('.'), response.rfind('!'), response.rfind('?'))
                if last_end > 10:
                    response = response[:last_end + 1].strip()

            if response:
                # Record our reply in the conversation history
                history.append(("Eudora", response))
                logger.info(f"[voice] LLM response to {username}: {response[:80]}")
                return response
            return ""
        except Exception as e:
            logger.error(f"Voice transcript handler error: {e}")
            return ""

    async def _speak_voice_greeting(self, guild_id: int) -> None:
        """Speak a greeting when the bot joins a voice channel."""
        try:
            if not self.voice_manager:
                return
            pipeline = self.voice_manager._pipelines.get(guild_id)
            if not pipeline:
                return

            # Find who's in the call
            other_names = []
            for vc in self.voice_clients:
                if vc.guild.id == guild_id and vc.channel:
                    other_names = [m.display_name for m in vc.channel.members if not m.bot]
                    break

            if not other_names:
                # Generic greeting
                greeting = "hey, just hopped in. what's everyone up to?"
            elif len(other_names) == 1:
                name = other_names[0]
                greetings = [
                    f"hey {name}, how are you doing?",
                    f"oh hey {name}, what's going on?",
                    f"hi {name}! how's your day been?",
                    f"yo {name}, what are you up to?",
                    f"hey {name}, how are you?",
                ]
                import random
                greeting = random.choice(greetings)
            else:
                names_str = ", ".join(other_names[:3])
                greeting = f"hey everyone, {names_str}. what are we talking about?"

            logger.info(f"[voice] Speaking greeting: {greeting}")
            await pipeline._speak(greeting)
        except Exception as e:
            logger.error(f"[voice] Greeting error: {e}")

    async def on_voice_state_update(self, member, before, after):
        """Handle voice state changes — users joining/leaving/muting in VCs."""
        # Skip for bots
        if member.bot:
            return

        # Legacy VC manager: record speaking activity
        try:
            if before.channel or after.channel:
                self.vc_manager.record_speaking(member.guild.id, member.id)
        except Exception:
            pass

        # New voice manager: handle join/leave for pipeline cleanup
        if self.voice_manager:
            try:
                await self.voice_manager.handle_voice_state_update(
                    member, before, after, self
                )
            except Exception as e:
                logger.debug(f"Voice state update error: {e}")

    async def _re_engagement_loop(self):
        """
        Periodically check if the bot's messages got no replies.
        If no one replied, try to re-engage by pinging users.
        """
        await asyncio.sleep(120)  # Wait 2 min before first check
        while True:
            try:
                re_tracker = get_re_engagement_tracker()
                ping_ctrl = get_ping_controller()

                for guild in self.guilds:
                    for channel in guild.text_channels:
                        ch_id = str(channel.id)
                        if not re_tracker.needs_re_engagement(ch_id):
                            continue

                        # Check permissions
                        from .channel_scanner import can_speak_in, is_skip_channel
                        if is_skip_channel(channel.name):
                            continue
                        if not can_speak_in(channel, guild.me):
                            continue

                        # Unified gate: paused / shared send cooldown / wall full
                        eng_log = get_engagement_log()
                        if not eng_log.can_send_engagement(ch_id, last_human_ts=self.last_activity.get(ch_id, 0),
                                                           history_msgs=self.history_cache.get(ch_id),
                                                           bot_user=self.user):
                            continue
                        # Clean stale unanswered engagement msgs before adding more
                        await eng_log.sweep_before_send(channel, self.user,
                                                        history_msgs=self.history_cache.get(ch_id))

                        # Determine re-engagement action
                        action = re_tracker.get_re_engagement_action(ch_id)
                        if action == "none":
                            continue

                        # Generate re-engagement message (in executor to not block)
                        topic = mem.get_channel_topic(ch_id)
                        loop = asyncio.get_running_loop()
                        msg = await loop.run_in_executor(
                            None, lambda: ai_reply.generate_proactive_message(topic)
                        )
                        if not msg or len(msg) < 3:
                            msg = random.choice([
                                "anyone there?", "yo someone talk to me",
                                "chat's dead fr", "hello?? anyone alive",
                            ])

                        # Apply ping based on action
                        if action == "everyone":
                            msg = f"@everyone {msg}"
                            ping_ctrl.record_everyone_ping(ch_id)
                        elif action == "here":
                            msg = f"@here {msg}"
                            ping_ctrl.record_here_ping(ch_id)
                        elif action == "user_ping":
                            user = select_online_user(guild, exclude_ids={self.user.id}, channel=channel)
                            if user:
                                msg = f"<@{user.id}> {msg}"
                                ping_ctrl.record_user_ping(ch_id, user_id=user.id)
                                # Track conversation with this user
                                if ch_id not in self.conversation_tracker:
                                    self.conversation_tracker[ch_id] = {}
                                self.conversation_tracker[ch_id][str(user.id)] = time.time()
                            else:
                                # No user to ping, skip
                                continue

                        # Send with typing simulation
                        typing_dur = random.uniform(1.5, 3.5)
                        async with channel.typing():
                            await asyncio.sleep(typing_dur)
                        sent = await channel.send(msg)
                        eng_log.mark_sent(ch_id)
                        eng_log.record(ch_id, sent, kind="re_engage")
                        logger.info(f"Re-engagement ({action}) in #{channel.name}: {msg}")

                        # Clear re-engagement state
                        re_tracker.clear_channel(ch_id)
                        re_tracker.record_bot_message(ch_id)

                        # Only do one re-engagement per cycle
                        break

            except Exception as e:
                logger.debug(f"Re-engagement loop error: {e}")
            await asyncio.sleep(300)  # Check every 5 minutes

    async def on_guild_join(self, guild: discord.Guild):
        """Called when joining a new server — scan it."""
        logger.info(f"Joined guild '{guild.name}' — scanning...")
        asyncio.create_task(scan_guild(guild, self.rules_cache))

    async def on_member_join(self, member):
        """Greet new members in the most active channel."""
        try:
            # Recently-joined members get a reply boost in plain-text
            # engagement — they're part of the conversation pool now
            self._recent_joins[member.id] = time.time()
            await get_welcome_system().handle_new_member(member, self)
        except Exception as e:
            logger.debug(f"Welcome system error: {e}")

    async def on_message(self, message: discord.Message):
        """Main message handler — decides whether and how to respond."""
        try:
            await self._on_message_impl(message)
        except Exception as e:
            logger.error(f"[on_message] EXCEPTION: {e}", exc_info=True)

    async def _on_message_impl(self, message: discord.Message):
        """Actual message handler implementation (wrapped for error catching)."""
        # Always cache our own messages for history accuracy
        if message.author == self.user:
            ch_id = str(message.channel.id)
            if ch_id not in self.history_cache:
                self.history_cache[ch_id] = deque(maxlen=40)
            self.history_cache[ch_id].append(message)
            return

        # Dedup
        if message.id in self.processed_msgs:
            return
        self.processed_msgs.append(message.id)

        # Define ch_id early (needed by tracking systems below)
        ch_id = str(message.channel.id)

        # ── User behavior tracking (algorithmic) ────────────────────────────
        # Track engagement and classify message value for memory storage
        if not message.author.bot:
            # Cross-modal memory: remember their recent text so a voice reply
            # can reference what they typed (per-user, not per-channel)
            if message.content and message.content.strip():
                from collections import deque as _dq
                dq = self._user_recent_texts.setdefault(message.author.id, _dq(maxlen=5))
                dq.append((time.time(), message.content.strip()[:200]))

            engagement_tracker = get_engagement_tracker()
            engagement_tracker.record_message(str(message.author.id), message.content)
            # If message is high-value, extract potential facts immediately
            if should_store_message(message.content):
                potential_facts = extract_potential_facts(message.content)
                if potential_facts:
                    logger.debug(f"High-value msg from {message.author.name}: {potential_facts}")

            # ── Track unanswered questions ────────────────────────────────
            # If this message contains a question, track it so the bot can
            # answer it later (e.g., if the bot missed it and then gets mentioned)
            if is_question(message.content):
                self.unanswered_tracker.record_question(
                    ch_id, message.content, message.author.display_name, message.id
                )

            # ── Re-engagement tracking ────────────────────────────────────
            # Record that a human replied (so we don't re-engage unnecessarily)
            get_re_engagement_tracker().record_human_reply(ch_id)
            # A human spoke — engagement worked / pause lifted
            get_engagement_log().mark_interaction(ch_id)

            # ── Conversational initiative: track user engagement ──────────
            # Record message for engagement analysis (response time, length, etc.)
            get_engagement_analyzer().record_user_message(str(message.author.id), message.content)

            # ── Update conversation tracker ────────────────────────────────
            # If we have an active conversation with this user, refresh the
            # timestamp so the conversation window extends when they reply.
            # This keeps multi-user conversations alive simultaneously.
            ch_conversations = self.conversation_tracker.get(ch_id, {})
            user_id_str = str(message.author.id)
            if user_id_str in ch_conversations:
                # Only refresh if within the 30 min window (don't revive dead convos)
                conv_age = time.time() - ch_conversations[user_id_str]
                if conv_age < 1800:
                    ch_conversations[user_id_str] = time.time()

            # Detect new participants — users who haven't spoken in this channel recently
            # Note: Message uses __slots__ in discord.py-self 2.2.0a, so we can't set
            # arbitrary attributes on it. Use a dict keyed by message.id instead.
            try:
                cache = self.history_cache.get(ch_id, deque())
                user_msgs_in_cache = sum(1 for m in cache if str(m.author.id) == str(message.author.id) and not m.author.bot)
                active_users_count = self.msg_cache.get_active_user_count(ch_id, window_minutes=10)
                # New participant = hasn't spoken in recent cache AND channel isn't too busy
                is_new_participant = (user_msgs_in_cache <= 1 and active_users_count < 8)
                # Returning user = check if they spoke in the last hour by looking at a wider window
                is_returning_user = is_new_participant  # Simplified: treat all new participants as potentially returning
                if not hasattr(self, '_msg_flags'):
                    self._msg_flags = {}
                self._msg_flags[message.id] = {
                    'is_new_participant': is_new_participant,
                    'is_returning_user': is_returning_user,
                }
            except Exception:
                if not hasattr(self, '_msg_flags'):
                    self._msg_flags = {}
                self._msg_flags[message.id] = {
                    'is_new_participant': False,
                    'is_returning_user': False,
                }

        # Update history cache
        if ch_id not in self.history_cache:
            try:
                hist = [m async for m in message.channel.history(limit=40)]
                self.history_cache[ch_id] = deque(reversed(hist), maxlen=40)
            except Exception:
                self.history_cache[ch_id] = deque(maxlen=40)

        if message not in self.history_cache[ch_id]:
            self.history_cache[ch_id].append(message)

        # Track activity for proactive messaging
        self.last_activity[ch_id] = time.time()
        if self.proactive_messenger:
            self.proactive_messenger.record_activity(message.channel.id)

        # Cache message in the TTL-based message cache (3 days / 2 months)
        self.msg_cache.add_message(
            ch_id=ch_id,
            content=message.clean_content,
            author=message.author.display_name,
            author_id=str(message.author.id),
            timestamp=message.created_at.timestamp(),
            is_bot=(message.author == self.user),
            message_id=message.id,
        )

        # Periodic cleanup of expired messages (every 30 min)
        if self.msg_cache.should_cleanup():
            self.msg_cache.cleanup_expired()

        # Track engagement (activity level per channel)
        self.engagement.update_activity(ch_id)

        # Boost engagement based on message content
        mentions_bot = self.user in message.mentions
        if mentions_bot:
            self.engagement.on_mention(ch_id)
        elif message.reference is not None:
            # Check if it's a reply to us
            cache = self.history_cache.get(ch_id, deque())
            for m in cache:
                if m.id == message.reference.message_id and m.author == self.user:
                    self.engagement.on_reply_to_us(ch_id)
                    break
        elif is_greeting(message.content):
            self.engagement.on_greeting(ch_id)

        # Loneliness detection — "anyone here?", "someone talk", "I'm bored"
        # This ALWAYS triggers a response (the bot's job is to keep chat active)
        if detect_loneliness(message.content):
            self.engagement.on_loneliness_detected(ch_id)

        # Skip ignorable messages
        if is_ignorable(message):
            return

        # Determine if we should respond
        should_respond, respond_reason = self._should_respond(message)
        logger.info(f"[{message.channel}] {message.author.name}: '{message.content[:50]}' -> {respond_reason}")

        if not should_respond:
            return

        # Double-send guard with message queue
        # If we're already sending in this channel, OR there are pending
        # messages from a previous batch, queue this message to maintain
        # FIFO order. Without the pending-check, a new message can jump
        # ahead of queued messages, causing out-of-order responses.
        if ch_id in self._sending_in_channel or self._pending_messages.get(ch_id):
            # Queue this message for processing after the current reply
            if ch_id not in self._pending_messages:
                self._pending_messages[ch_id] = deque(maxlen=5)
            self._pending_messages[ch_id].append(message)
            logger.debug(f"Queued message in #{message.channel} (bot is typing or pending queue) — will reply after current send")
            return
        self._sending_in_channel.add(ch_id)

        try:
            await self._handle_reply(message, ch_id)
            # Process any messages that arrived while we were typing
            await self._process_pending_messages(ch_id)
        finally:
            self._sending_in_channel.discard(ch_id)

    async def _process_pending_messages(self, ch_id: str):
        """Process messages that were queued while the bot was typing/sending."""
        pending = self._pending_messages.get(ch_id, deque())
        while pending:
            msg = pending.popleft()
            # NOTE: We do NOT re-check _should_respond here. These messages
            # were already approved when they arrived. Re-checking would
            # reject them due to cooldowns (the bot just replied, so
            # since_last < 5s), causing messages to be silently skipped.
            logger.info(f"Processing pending message from {msg.author.name}: '{msg.content[:50]}'")
            try:
                await self._handle_reply(msg, ch_id)
            except Exception as e:
                logger.warning(f"Error processing pending message: {e}")
                break

    def _should_respond(self, message: discord.Message) -> tuple:
        """Determine if and why we should respond to a message.

        Uses research-backed patterns from production AI bots:
        - IGNORE_NO_MENTION: Skip if message mentions other users but not us
        - First-mention check: Only treat as addressed to us if first mention is us
        - Reply-reference check: Skip if replying to another user
        - Dismissal detection: Back off if told "not talking to you" / "shut up"
        - Name detection: Lower reply chance if addressing another user by name
        """
        is_dm = isinstance(message.channel, (discord.DMChannel, discord.GroupChannel))
        ch_id = str(message.channel.id)
        now = time.time()
        txt_low = message.content.lower()

        # ── 0. Dismissal detection ─────────────────────────────────────────
        # If someone tells the bot to go away / shut up / not talk to them,
        # set a backoff timer and don't reply for 10 minutes in that channel.
        DISMISSAL_PHRASES = [
            "not talking to you", "wasn't talking to you", "wasnt talking to you",
            "i wasn't talking to you", "i wasnt talking to you",
            "shut up", "stfu", "go away", "leave me alone", "stop talking",
            "nobody asked", "who asked", "who asked you", "did i ask",
            "stop replying", "stop responding", "ignore me",
            "not for you", "wasn't for you", "wasnt for you",
            "i'm not talking to you", "im not talking to you",
            "don't talk to me", "dont talk to me",
            "bro shut up", "can you shut up", "hush",
        ]
        if any(phrase in txt_low for phrase in DISMISSAL_PHRASES):
            # Only back off if the message is directed at us (mention or reply to us)
            # or if we're in a sticky conversation (we were just talking)
            is_directed_at_us = self.user in message.mentions
            is_sticky = self.sticky_until.get(ch_id, 0) > now
            if is_directed_at_us or is_sticky:
                self.backoff_until[ch_id] = now + 600  # 10 min backoff
                # Also clear sticky — conversation is over
                self.sticky_until[ch_id] = 0
                logger.info(f"Dismissal detected in #{message.channel} — backing off for 10 min")
                return False, "dismissed-by-user"

        # ── 0.5. Backoff check ─────────────────────────────────────────────
        # If we're in a backoff period, only respond to direct mentions
        backoff_end = self.backoff_until.get(ch_id, 0)
        if backoff_end > now:
            if self.user not in message.mentions:
                return False, f"backoff ({int(backoff_end - now)}s left)"
            # Mention breaks the backoff
            self.backoff_until[ch_id] = 0
            logger.info(f"Backoff broken by mention in #{message.channel}")

        # AFK check
        afk_end = self.afk_until.get(ch_id, 0)
        if afk_end > now:
            is_mention = self.user in message.mentions
            if not is_mention:
                return False, "AFK — ignoring"
            else:
                self.afk_until[ch_id] = 0
                logger.debug(f"AFK broken by mention in #{message.channel}")

        # ── 1. DMs — only the owner can DM the bot ────────────────────────
        if is_dm:
            if not is_owner(str(message.author.id)):
                return False, "dm-not-owner"
            since_last = now - self.last_replied.get(ch_id, 0)
            if since_last < DM_COOLDOWN_S:
                return False, f"DM cooldown ({int(DM_COOLDOWN_S - since_last)}s left)"
            if not self.can_send(ch_id):
                return False, "daily-cap"
            return True, "DM"

        # ── 2. "Addressed to another user" check (CRITICAL) ────────────────
        # This runs BEFORE sticky and everything else. If the message is
        # clearly directed at another user, we stay out of it — even in
        # a sticky conversation. This prevents the bot from jumping into
        # conversations between other people.
        #
        # Pattern from Hermes Agent: DISCORD_IGNORE_NO_MENTION
        # If message mentions other users but NOT the bot → skip
        mentions_bot = self.user in message.mentions
        mentions_others = [u for u in message.mentions if u != self.user and not u.bot]

        if mentions_others and not mentions_bot:
            # Message is addressed to another user, not us
            return False, "addressed-to-other-user (mention)"

        # Also check: if the message is a reply to another user (not us)
        # We can check this synchronously by looking at the reply target
        # in the cached history
        if message.reference is not None and not mentions_bot:
            # Check if the replied-to message is ours
            ref_id = message.reference.message_id
            replied_is_ours = False
            cache = self.history_cache.get(ch_id, deque())
            for m in cache:
                if m.id == ref_id:
                    replied_is_ours = (m.author == self.user)
                    break
            if not replied_is_ours:
                # The reply is to someone else, not us
                return False, "addressed-to-other-user (reply)"

        # ── 2b. Mentions — always respond ────────────────────────────────
        # A direct ping outranks every gate below (sticky cooldown, perms
        # name-checks, engagement scoring). This MUST run before the sticky
        # block — a mention inside a live sticky window is still a mention,
        # and the sticky gates were swallowing it (e.g. 'sticky-no-perms'
        # in a bump channel whose name is on the skip list).
        if mentions_bot:
            if not self.can_send(ch_id):
                return False, "mention-daily-cap"
            return True, "mention"

        # ── 3. Conversation stickiness ─────────────────────────────────────
        # If we recently replied in this channel, respond to messages for
        # 5 minutes — BUT only if the message isn't addressed to someone else
        # (that check already passed above for @mentions and replies).
        # We ALSO check for name mentions here (e.g. "hey daniel" without @).
        STICKY_COOLDOWN_S = 2
        sticky_end = self.sticky_until.get(ch_id, 0)
        if sticky_end > now:
            since_last = now - self.last_replied.get(ch_id, 0)
            if since_last < STICKY_COOLDOWN_S:
                # Inside a live back-and-forth, only drop pure noise — real
                # content (questions, replies to us, >3 words) always lands
                is_filler = (len(txt_low.split()) <= 3
                             and not is_question(txt_low)
                             and message.reference is None)
                if is_filler:
                    return False, f"sticky-cooldown ({int(STICKY_COOLDOWN_S - since_last)}s left)"
            if not self.can_send(ch_id):
                return False, "sticky-daily-cap"

            # Name detection: even in sticky mode, if the message addresses
            # another user by name (not us), stay silent. This prevents the
            # bot from replying to "hey daniel how are you" in a sticky convo.
            our_name = self.user.display_name.lower()
            our_username = self.user.name.lower()
            msg_has_our_name = (our_name in txt_low or our_username in txt_low or "eudora" in txt_low)
            msg_has_other_name = False
            if hasattr(message.channel, 'guild') and message.channel.guild:
                # Check recent chatters in cache
                cache = self.history_cache.get(ch_id, deque())
                recent_names = set()
                for m in cache:
                    if m.author != self.user and not m.author.bot:
                        recent_names.add(m.author.display_name.lower())
                        recent_names.add(m.author.name.lower())
                # Also check guild members (cached by discord.py-self)
                try:
                    for member in message.channel.guild.members:
                        if member != self.user and not member.bot:
                            recent_names.add(member.display_name.lower())
                            recent_names.add(member.name.lower())
                except Exception:
                    pass  # Member list might not be available

                for name in recent_names:
                    # Use word boundary check to avoid partial matches
                    if len(name) >= 3 and name != our_name and name != our_username and name != "eudora":
                        if re.search(r'\b' + re.escape(name) + r'\b', txt_low):
                            msg_has_other_name = True
                            logger.debug(f"Name match: '{name}' found in message")
                            break

            # Also check for addressing patterns: "hey X", "talking to X",
            # "waiting for X", "do you know X" where X is a capitalized word
            if not msg_has_other_name and not msg_has_our_name:
                addressing_patterns = [
                    r'\b(hey|hi|hello|yo|sup)\s+([A-Z][a-z]{2,})\b',
                    r'\b(talking\s+to|wait(?:ing)?\s+for|looking\s+for|do\s+you\s+know|where\s+is|is\s+here)\s+([A-Z][a-z]{2,})\b',
                    r'\b([A-Z][a-z]{2,})\s+(come\s+in|come\s+on|get\s+in|join|where\s+are\s+you|you\s+there)\b',
                ]
                for pat in addressing_patterns:
                    match = re.search(pat, message.content)
                    if match:
                        # Extract the name and check it's not us
                        name_found = match.group(2) if match.lastindex >= 2 else match.group(1)
                        if name_found.lower() not in (our_name, our_username, "eudora"):
                            msg_has_other_name = True
                            logger.debug(f"Addressing pattern matched: '{name_found}'")
                            break

            if msg_has_other_name and not msg_has_our_name and not mentions_bot:
                return False, "sticky-name-match (addressing another user)"

            # Check permissions — real Discord perms only. The skip-channel
            # name list is for UNSOLICITED participation; once she's already
            # conversing in a channel (that's what sticky means), continuing
            # shouldn't be blocked by the channel being called e.g. 'bump-server'.
            if hasattr(message.channel, 'guild') and message.channel.guild:
                me = message.channel.guild.me
                if not can_speak_in(message.channel, me, include_name_check=False):
                    return False, "sticky-no-perms"
            remaining = int(sticky_end - now)
            return True, f"sticky-convo ({remaining}s left)"

        # ── 5. Direct reply to the bot (via reply button) ──────────────────
        # Reply button = ALWAYS respond (it's a direct response to us)
        # Extended window: respond to replies within 30 min (was 5 min)
        # For late replies (10+ min), the bot will take longer to respond
        if message.reference is not None:
            ref_id = message.reference.message_id
            replied_is_ours = False
            cache = self.history_cache.get(ch_id, deque())
            for m in cache:
                if m.id == ref_id:
                    replied_is_ours = (m.author == self.user)
                    break
            if replied_is_ours:
                since_last = now - self.last_replied.get(ch_id, 0)
                if since_last < 1800:  # Extended to 30 min (was 5 min)
                    if since_last < 15:
                        return False, f"reply-cooldown ({int(15 - since_last)}s left)"
                    if not self.can_send(ch_id):
                        return False, "reply-daily-cap"
                    if since_last > 600:
                        return True, f"reply-to-me (late: {int(since_last // 60)}m ago)"
                    return True, "reply-to-me"

        # ── 5b. Bot name mentioned (without @) — respond if contextual ─────
        # If someone says "eudora" in a message, respond — but ONLY if the
        # message has more context than just the name (avoid replying to
        # standalone name drops which looks too AI-like).
        #
        # ALGORITHM for name detection:
        # 1. Standalone name (1-2 words) → SKIP (too AI-like)
        # 2. Name + context (3+ words) → respond
        # 3. Name as part of a question → respond
        # 4. Name with punctuation like "eudora!" → respond (addressing us)
        our_name_lower = self.user.display_name.lower()
        our_username_lower = self.user.name.lower()
        # Use word boundary matching to avoid false positives
        name_patterns = [our_name_lower, our_username_lower, "eudora"]
        name_patterns = list(set([p for p in name_patterns if len(p) >= 3]))
        for name_pat in name_patterns:
            if re.search(r'\b' + re.escape(name_pat) + r'\b', txt_low):
                # ALGORITHM: Don't reply to standalone name (just "eudora" alone)
                # — that's too AI-like. Only reply if there's additional context.
                words_in_msg = txt_low.strip().split()
                # Check for punctuation (addressing us emphatically)
                has_punct = any(c in txt_low for c in "!?,.")

                # If the message is JUST the name (1 word) with no punctuation, skip
                if len(words_in_msg) <= 1 and not has_punct:
                    return False, "name-standalone (too AI-like, skipping)"

                # 2-word messages: only respond if it's a greeting + name
                # e.g., "hey eudora" or "eudora!" or "yo eudora"
                if len(words_in_msg) == 2:
                    # Check if it's a greeting pattern
                    greetings = {"hey", "hi", "hello", "yo", "sup", "oi", "ayy", "ay", "heya"}
                    has_greeting = any(w in greetings for w in words_in_msg)
                    if not has_greeting and not has_punct:
                        return False, "name-standalone-2words (too AI-like, skipping)"

                since_last = now - self.last_replied.get(ch_id, 0)
                if since_last < 8:  # 8s cooldown for name mentions
                    return False, f"name-cooldown ({int(8 - since_last)}s left)"
                if not self.can_send(ch_id):
                    return False, "name-daily-cap"
                self.engagement.on_name_mentioned(ch_id)
                return True, f"name-mentioned ({name_pat})"

        # ── 5c. Personal questions — ALWAYS respond ────────────────────────
        # If someone asks about themselves ("what's my name", "what do you know
        # about me"), always respond — this tests the bot's memory.
        personal_questions = [
            "what's my name", "what is my name", "whats my name",
            "do you know my name", "who am i", "do you know me",
            "what do you know about me", "what do you remember about me",
            "my name", "remember me",
        ]
        for q in personal_questions:
            if q in txt_low:
                since_last = now - self.last_replied.get(ch_id, 0)
                if since_last < 8:
                    return False, f"personal-q-cooldown ({int(8 - since_last)}s left)"
                if not self.can_send(ch_id):
                    return False, "personal-q-daily-cap"
                self.engagement.boost(ch_id, 2.0, "personal-question")
                return True, f"personal-question ({q})"

        # ── 5d. Loneliness / seeking-chat detection — ALWAYS respond ───────
        # If someone says "anyone here?", "someone talk", "I'm bored", etc.,
        # the bot should ALWAYS respond — its job is to keep chat active.
        if detect_loneliness(txt_low):
            since_last = now - self.last_replied.get(ch_id, 0)
            if since_last < 5:  # 5s cooldown for loneliness responses
                return False, f"loneliness-cooldown ({int(5 - since_last)}s left)"
            if not self.can_send(ch_id):
                return False, "loneliness-daily-cap"
            return True, "loneliness-detected (seeking chat)"

        # ── 5e. Active conversation with this user ──────────────────────────
        # If the bot recently interacted with this specific user (pinged them,
        # replied to them, or was in a conversation with them), respond to
        # their messages — even without a mention or reply button.
        # This handles the case where the bot pings a user, they reply, but
        # the sticky conversation has expired. The bot should still talk to them.
        user_id_str = str(message.author.id)
        ch_conversations = self.conversation_tracker.get(ch_id, {})
        last_interaction = ch_conversations.get(user_id_str, 0)
        if last_interaction > 0:
            conv_age = now - last_interaction
            if conv_age < 1800:  # 30 min window
                since_last = now - self.last_replied.get(ch_id, 0)
                if since_last < 5:  # 5s cooldown
                    return False, f"conversation-cooldown ({int(5 - since_last)}s left)"
                if not self.can_send(ch_id):
                    return False, "conversation-daily-cap"
                # Check if message is addressed to someone else (don't hijack)
                if not mentions_others:
                    age_note = f"{int(conv_age // 60)}m ago" if conv_age > 60 else f"{int(conv_age)}s ago"
                    return True, f"active-conversation (user interacted {age_note})"

        # ── 6. Random server participation ─────────────────────────────────
        if hasattr(message.channel, 'guild') and message.channel.guild is not None:
            since_last = now - self.last_replied.get(ch_id, 0)
            if since_last < CHANNEL_COOLDOWN_S:
                return False, f"cooldown ({int(CHANNEL_COOLDOWN_S - since_last)}s left)"
            if not self.can_send(ch_id):
                return False, "daily-cap"

            # ── Busy channel detection ──────────────────────────────────────
            # If 5+ unique users are actively chatting (within 10 min),
            # the bot should NOT auto-step in. It only responds when called
            # (mention, name, reply, personal question, loneliness).
            # This prevents the bot from spamming in busy conversations.
            active_users = self.msg_cache.get_active_user_count(ch_id, window_minutes=10)
            if active_users >= 8:
                return False, f"busy-channel ({active_users} users active — staying silent)"

            # Check for distress signals — skip
            if any(s in txt_low for s in DISTRESS_SIGNALS):
                return False, "emotional-skip (distress)"

            # Check permissions
            me = message.channel.guild.me
            if not can_speak_in(message.channel, me):
                return False, "no-perms-or-skip-channel"

            # Greeting detection — if someone says hi/hello/heya/yo etc.,
            # always respond (regardless of message length or time of day).
            # Runs BEFORE the utility gates — a greeting is always worth
            # answering even in a channel whose topic mentions xp/levels.
            # But NOT if the greeting is followed by another user's name
            # (e.g. "hey @daniel" or "hi john" — that's for someone else)
            if is_greeting(message.content):
                if mentions_others:
                    return False, "greeting-for-other-user"
                return True, "greeting-detected"

            # Check channel nature — skip random participation in utility channels
            # (rank-check, bot-commands, counting, etc.) even if the name wasn't caught
            # by is_skip_channel. Also check the channel topic for utility keywords.
            channel_nature = get_cached_nature(ch_id)
            channel_topic = getattr(message.channel, 'topic', '') or ''
            topic_lower = channel_topic.lower()
            # Tight utility-topic phrases — a chat channel that merely
            # mentions "level"/"xp" (xp-farming servers) is NOT a utility channel
            utility_topic_keywords = ["bot command", "verify yourself", "counting",
                                      "leaderboard", "rank check", "reaction role",
                                      "commands only", "bot spam"]
            is_utility_ch = (channel_nature == "utility"
                             or any(kw in topic_lower for kw in utility_topic_keywords))
            # Designated chat channels (TEXT_CHANNEL_IDS) are never utility —
            # "chat and level up!" topics shouldn't silence the main channel
            if is_utility_ch and message.channel.id in self._engage_channel_ids:
                is_utility_ch = False

            if is_utility_ch:
                # High-signal bypass — these messages deserve engagement even
                # in utility-ish channels (a channel whose topic mentions
                # "level"/"xp" can still be a real chat channel)
                msg_flags = getattr(self, '_msg_flags', {}).get(message.id, {})
                join_ts = self._recent_joins.get(message.author.id)
                bypass = (
                    msg_flags.get('is_new_participant')
                    or (join_ts and (now - join_ts) < 1800)
                    or get_engagement_log().recent_engagement(ch_id, within_s=240)
                )
                if not bypass:
                    return False, "utility-channel (skipping random participation)"

            # Skip very short messages (non-greetings)
            if len(message.content.strip()) < 4:
                return False, "too-short"

            # Name detection: if the message contains another user's display
            # name but NOT our name, lower the reply chance significantly.
            # This catches "hey daniel what are you doing" type messages.
            our_name = self.user.display_name.lower()
            msg_has_our_name = our_name in txt_low
            msg_has_other_name = False
            if hasattr(message.channel, 'guild') and message.channel.guild:
                # Check if any member's name appears in the message
                # (lightweight check — only check recent speakers in cache)
                cache = self.history_cache.get(ch_id, deque())
                recent_names = set()
                for m in cache:
                    if m.author != self.user and not m.author.bot:
                        recent_names.add(m.author.display_name.lower())
                        recent_names.add(m.author.name.lower())
                for name in recent_names:
                    if len(name) >= 3 and name != our_name and name != self.user.name.lower():
                        # Use word boundary to avoid partial matches
                        if re.search(r'\b' + re.escape(name) + r'\b', txt_low):
                            msg_has_other_name = True
                            break

            if msg_has_other_name and not msg_has_our_name:
                # Message is about another user, not us — much lower chance
                chance = dynamic_reply_chance() * 0.15
                if random.random() < chance:
                    return True, f"random-server-name-match (chance: {chance:.0%})"
                return False, f"random-skip-other-name (chance: {chance:.0%})"

            # Dynamic reply chance based on time of day + engagement score
            time_chance = dynamic_reply_chance()
            engagement_prob = self.engagement.get_engagement_probability(ch_id)
            # Blend: 60% time-of-day, 40% engagement score
            chance = (time_chance * 0.6) + (engagement_prob * 0.4)

            # Boost chance for messages that are questions or have substance
            msg_stripped = message.content.strip()
            if "?" in msg_stripped:
                chance += 0.15  # Questions deserve a reply more often
            if len(msg_stripped) > 30:
                chance += 0.10  # Longer messages show more engagement
            # If only 1-2 users active (not busy), be more responsive
            if active_users <= 2:
                chance += 0.10

            # Boost chance for new participants (someone stepping into the chat)
            msg_flags = getattr(self, '_msg_flags', {}).get(message.id, {})
            is_new_participant = msg_flags.get('is_new_participant', False)
            is_returning_user = msg_flags.get('is_returning_user', False)
            if is_new_participant:
                chance += 0.25  # Much more likely to acknowledge new participants
                if is_returning_user:
                    chance += 0.10  # Extra boost for returning users

            # Their message probably answers the engagement message the bot
            # just sent ("anyone up for a game?" → "hey bro" / "i'm bored af")
            if get_engagement_log().recent_engagement(ch_id, within_s=240):
                chance += 0.35

            # Recently-joined members — part of the conversation pool now
            join_ts = self._recent_joins.get(message.author.id)
            if join_ts and (now - join_ts) < 1800:
                chance += 0.15

            chance = min(chance, 0.85)  # Cap at 85%

            if random.random() < chance:
                self.engagement.boost(ch_id, 0.5, "decided-to-reply")
                return True, f"random-server (chance: {chance:.0%}, engage: {engagement_prob:.0%})"
            self.engagement.on_skipped(ch_id)
            return False, f"random-skip (chance: {chance:.0%}, engage: {engagement_prob:.0%})"

        return False, "unhandled"

    @staticmethod
    def _log_task_error(fut: asyncio.Task) -> None:
        """Log exceptions from fire-and-forget asyncio tasks."""
        if fut.cancelled():
            return
        exc = fut.exception()
        if exc is not None:
            logger.error(f"Background task failed: {exc!r}", exc_info=exc)

    def _extract_named_vc(self, text: str, guild) -> Optional[discord.VoiceChannel]:
        """Pull a voice-channel name out of a join request, if present.

        Handles: join "Your Custom" vc | join 'general' vc | join the dev vc.
        Returns a resolved channel, or None (no name / no match — caller
        falls back to the requester-is-in-VC heuristic)."""
        if not guild:
            return None
        name = None
        # Quoted name first — most precise
        m = re.search(r'["\u201c\']([^"\'\u201d]{1,45})["\'\u201d]\s*(?:vc|voice|call)', text, re.I)
        if m:
            name = m.group(1).strip()
        if not name:
            # "join the X vc" — words between the join verb and the vc keyword
            verb = re.search(r'\b(join|jon|jion)\b', text, re.I)
            if verb:
                tail = text[verb.end():]
                m3 = re.match(
                    r'\s+(?:the\s+)?([A-Za-z0-9][\w\s\-\u2019\']{0,38}?)\s+(?:vc|voice\s+channel|voice|call)\b',
                    tail, re.I)
                if m3:
                    name = m3.group(1).strip()
        if not name or len(name) < 2:
            return None
        # Generic words are never channel names
        if name.lower() in ("vc", "voice", "call", "voice channel", "current",
                            "my", "this", "your", "a", "an", "the", "active"):
            return None
        return find_vc_by_name(guild, name)

    async def _handle_vc_join(self, message: discord.Message, user_id: str, username: str):
        """
        Algorithmically handle a VC join request:
        1. Check if the requesting user is in a VC → join that one directly
        2. If not, find all active VCs in the guild
        3. If 0 active VCs → say "no one's in vc rn"
        4. If 1 active VC → join it
        5. If 2+ active VCs → ask "which one?" (human-like validation)
        Uses the new VoiceManager (with voice receive) if available, falls back to legacy.
        """
        if not hasattr(message.channel, 'guild') or not message.channel.guild:
            await message.channel.send("can't join vc in dms bruv")
            return

        guild = message.channel.guild
        requester = message.author

        # Step 0: explicit named target — "join the 'Your Custom' vc" /
        # "join general vc". A named channel beats the requester-is-in-VC
        # heuristic (they may be asking for a channel they're not in).
        named_vc = self._extract_named_vc(message.content, guild)
        user_vc = named_vc or find_user_voice_channel(guild, requester.id)
        if user_vc:
            # Already in a DIFFERENT VC → permission-ask flow: the bot says
            # it's occupied, then asks the current VC members "can I go?"
            # and listens for a yes/no in the next utterances.
            if self.voice_manager:
                current_vc_id = self.voice_manager.get_current_vc_id(guild.id)
                if current_vc_id and current_vc_id != user_vc.id:
                    pipeline = self.voice_manager._pipelines.get(guild.id)
                    if pipeline:
                        requester_name = requester.display_name or requester.name
                        req_channel = message.channel

                        async def _notify_move(accepted: bool):
                            try:
                                if accepted:
                                    await req_channel.send(f"they said yes — hopping over to {user_vc.name} 🎧")
                                else:
                                    await req_channel.send("they want me to stay — i'll catch you later innit")
                            except Exception:
                                pass

                        await message.channel.send(random.choice([
                            "i'm already in a vc — lemme ask them first",
                            "i'm in a vc rn — hold on, asking them",
                            "already in another vc — lemme check if they're cool with me hopping",
                            "i'm in a call rn — asking them if i can dip",
                            "i'm occupied in another vc — one sec, asking them",
                        ]))
                        await pipeline.request_vc_move(requester_name, user_vc, notify_cb=_notify_move)
                        return
                # Same VC → she's literally already there
                if current_vc_id == user_vc.id:
                    await message.channel.send("i'm literally already in this vc bruv 😭")
                    return
            # Not in a VC (or no voice manager) — direct join path below.
            # If she just left that VC, she may decline (equilibrium).
            if self.voice_manager and random.random() < self.voice_manager.join_decline_probability(guild.id, user_vc.id):
                await message.channel.send("i just left there innit — gimme a bit, i'll come back later")
                return
            # Join the VC the user is in
            logger.info(f"VC: Joining '{user_vc.name}' (user {username} is in it)")
            if self.voice_manager:
                loop = asyncio.get_event_loop()
                success = await self.voice_manager.join_vc(self, user_vc, loop)
                if success:
                    await message.channel.send(f"hopping in {user_vc.name} 🎧 can hear u now")
                else:
                    await message.channel.send("couldn't join the vc, something went wrong")
            else:
                response = await self.vc_manager.join_vc(
                    self, guild, user_vc, requester, message.channel
                )
                await message.channel.send(response)
            return

        # Step 2: Find all active VCs — but never count the bot's OWN vc as
        # a join target (joining the vc she's already in just fails)
        active_vcs = find_active_voice_channels(guild)
        cur_vc_id = self.voice_manager.get_current_vc_id(guild.id) if self.voice_manager else None
        if cur_vc_id:
            active_vcs = [vc for vc in active_vcs if vc[0].id != cur_vc_id]

        if len(active_vcs) == 0:
            if cur_vc_id:
                # She's already in a vc and the requester isn't in any —
                # natural reply instead of a failed join
                await message.channel.send(random.choice([
                    "i'm already in a vc — hop in yours and i'll ask if i can move",
                    "you're not in a vc rn innit — join one and ask me again",
                    "i'm already vibing in one — get in a vc and i'll swing by",
                ]))
            else:
                await message.channel.send("no one's in a vc rn innit")
            logger.info(f"VC: No joinable VCs in guild '{guild.name}' (bot_in_vc={bool(cur_vc_id)})")
            return

        if cur_vc_id:
            # She's occupied and the requester isn't in any VC — tell them
            # to hop into the target first so the permission-ask has a target
            other_names = ", ".join(f"**{ch.name}**" for ch, _ in active_vcs)
            await message.channel.send(random.choice([
                f"i'm already in a vc — if you hop in {other_names} i'll ask if i can move",
                f"join {other_names} first and i'll see about coming over",
            ]))
            return

        if len(active_vcs) == 1:
            # Only one active VC — join it directly
            vc_channel, members = active_vcs[0]
            logger.info(f"VC: Joining '{vc_channel.name}' (only active VC, {len(members)} members)")
            if self.voice_manager:
                loop = asyncio.get_event_loop()
                success = await self.voice_manager.join_vc(self, vc_channel, loop)
                if success:
                    await message.channel.send(f"hopping in {vc_channel.name} 🎧 can hear u now")
                else:
                    await message.channel.send("couldn't join the vc, something went wrong")
            else:
                response = await self.vc_manager.join_vc(
                    self, guild, vc_channel, requester, message.channel
                )
                await message.channel.send(response)
            return

        # Multiple active VCs — ask which one (human-like validation)
        vc_names = [f"**{ch.name}** ({len(members)} ppl)" for ch, members in active_vcs]
        vc_list = ", ".join(vc_names)
        await message.channel.send(f"which vc? there's {len(active_vcs)} active rn: {vc_list}")
        logger.info(f"VC: Asked which VC to join ({len(active_vcs)} active)")

    async def _handle_mention_request(self, message: discord.Message, target_name: str, start_chat: bool, requester_name: str):
        """
        Algorithmically handle a mention request:
        1. Find the target user in the guild by name or ID
        2. If found: mention them and optionally start a chat
        3. If multiple matches: ask for disambiguation
        4. If not found: say "can't find that user"
        """
        if not hasattr(message.channel, 'guild') or not message.channel.guild:
            await message.channel.send("can't mention people in dms bruv")
            return

        guild = message.channel.guild
        result = await self.mention_manager.handle_mention_request(
            guild, target_name, start_chat=start_chat
        )

        if result["found"] and result["mention"]:
            # Found the user — send the mention
            await message.channel.send(result["message"])
            logger.info(f"Mention: {requester_name} asked to mention '{target_name}' → {result['member'].display_name}")
        elif result["multiple_matches"]:
            # Multiple matches — ask for disambiguation
            await message.channel.send(result["message"])
            logger.info(f"Mention: multiple matches for '{target_name}' — asking for disambiguation")
        else:
            # Not found
            await message.channel.send(result["message"])
            logger.info(f"Mention: user '{target_name}' not found (requested by {requester_name})")

    def _wants_action_worker(self, message: discord.Message, text: str, is_dm: bool) -> bool:
        """Should this message go to the side-channel action worker?

        Runs only for messages directed at the bot (mention / reply / DM) —
        or for an owner mid-conversation — and only when the text smells like
        a Discord action request. Non-owners get the worker only when
        ACTIONS_OWNER_ONLY=0.
        """
        if not self._actions_enabled:
            return False
        try:
            from .actions.intent import looks_like_action
            if not looks_like_action(text):
                return False
        except Exception:
            return False
        directed = (
            is_dm
            or self.user in message.mentions
            or message.reference is not None
        )
        owner = is_owner(str(message.author.id))
        if not directed:
            # owner mid-conversation still counts as directed
            sticky = self.sticky_until.get(str(message.channel.id), 0) > time.time()
            if not (owner and sticky):
                return False
        if self._actions_owner_only:
            return owner
        return True

    async def _run_vc_action(self, user_id: int, transcript: str) -> str:
        """Run the side-channel worker for a spoken action request.
        Returns a voice-prompt directive fragment ('' if nothing ran)."""
        if not self._actions_enabled or not self.voice_manager:
            return ""
        guild_id = self._find_voice_guild_id(user_id)
        if not guild_id:
            return ""
        guild = self.get_guild(guild_id)
        vc_id = self.voice_manager.get_current_vc_id(guild_id)
        channel = guild.get_channel(vc_id) if guild and vc_id else None
        if channel is None:
            return ""
        member = guild.get_member(user_id)
        from .actions.worker import get_action_worker
        note = await get_action_worker().run_request(
            self, guild=guild, channel=channel, author_id=user_id,
            author_name=getattr(member, "display_name", None) or str(user_id),
            text=transcript)
        if not note:
            return ""
        # note looks like "[ACTION PERFORMED for X: sent a message — ...]"
        m = re.search(r"for [^:]+:\s*([^—\]]+)", note)
        did = m.group(1).strip() if m else "the thing they asked"
        return (f"You already did it for them ({did}) — acknowledge it in "
                f"ONE short spoken line like it's no big deal. Don't re-ask.")

    async def _run_action_request(self, message: discord.Message, text: str) -> str:
        """Hand a directed action request to the side-channel worker.
        Returns a context note for the prompt, or "" when nothing ran."""
        try:
            from .actions.worker import get_action_worker
            # typing() makes the (intentional) execution delay look human
            async with message.channel.typing():
                note = await get_action_worker().run_request(
                    self,
                    guild=getattr(message.channel, "guild", None),
                    channel=message.channel,
                    author_id=message.author.id,
                    author_name=message.author.name,
                    text=text,
                )
            return note or ""
        except Exception as e:
            logger.warning(f"[action] worker call failed: {e}")
            return ""

    async def _handle_command(self, cmd_type: str, params: dict, message: discord.Message, ch_id: str):
        """Handle a detected command from the user.

        This runs BEFORE the AI reply generation, so:
        - Action commands (bump, stop pinging) are executed immediately
        - Memory commands (set name, remember) store data for future use
        - The AI still generates a conversational reply acknowledging the action
        """
        user_id = str(message.author.id)
        username = message.author.name

        try:
            if cmd_type == "bump_server" or cmd_type == "bump_with":
                # Trigger an immediate bump batch
                if hasattr(self, 'bump_scheduler') and self.bump_scheduler:
                    logger.info(f"Command: bumping servers (requested by {username})")
                    asyncio.create_task(self.bump_scheduler._perform_bump_batch())
                else:
                    logger.warning("Bump scheduler not available for command")

            elif cmd_type == "stop_bumping":
                # Stop the automatic bump scheduler
                if hasattr(self, 'bump_scheduler') and self.bump_scheduler:
                    self.bump_scheduler._running = False
                    logger.info(f"Command: stopped bump scheduler (requested by {username})")
                mem.add_instruction(user_id, "stop bumping servers", ch_id)

            elif cmd_type == "join_vc":
                # Join a voice channel — algorithmic VC detection
                task = asyncio.create_task(self._handle_vc_join(message, user_id, username))
                task.add_done_callback(self._log_task_error)

            elif cmd_type == "leave_vc":
                # Leave the current voice channel — silently (no text
                # announcement; the disconnect itself is the acknowledgment)
                if hasattr(message.channel, 'guild') and message.channel.guild:
                    if self.voice_manager and self.voice_manager.is_in_vc(message.channel.guild.id):
                        await self.voice_manager.leave_vc(self, message.channel.guild)
                    else:
                        await self.vc_manager.leave_vc(self, message.channel.guild)
                    logger.info(f"Command: left VC (requested by {username})")

            elif cmd_type in ("mention_user", "mention_user_id", "start_chat"):
                # Mention a user by name or ID, optionally start chatting
                target_name = params.get("target") or params.get("name") or ""
                start_chat = (cmd_type == "start_chat")
                if target_name:
                    asyncio.create_task(self._handle_mention_request(message, target_name, start_chat, username))
                else:
                    logger.warning(f"Mention command but no target name extracted from: {trigger_text[:50]}")

            elif cmd_type == "stop_pinging":
                # Store instruction to stop pinging
                mem.add_instruction(user_id, "stop pinging chat revive role", ch_id)
                if self.proactive_messenger:
                    self.proactive_messenger.auto_chat_chance = 0
                    logger.info("Command: chat revive pings disabled")

            elif cmd_type == "ping_after_hours":
                hours = params.get("hours", 16)
                mem.add_instruction(user_id, f"only ping after {hours} hours", ch_id)
                if self.proactive_messenger:
                    # Set the dead chat threshold to the specified hours
                    self.proactive_messenger.dead_threshold_mins = hours * 60
                    logger.info(f"Command: ping threshold set to {hours} hours")

            elif cmd_type == "set_name":
                name = params.get("name", "")
                if name:
                    # Get old name before updating (for logging)
                    old_profile = mem.get_user_profile(user_id)
                    old_name = old_profile.get("real_name", "")

                    # Update profile (replaces old name automatically)
                    mem.update_user_profile(user_id, username, "real_name", name)

                    # Update facts (algorithmic replacement removes old name fact)
                    mem.update_user_memory(user_id, username, [f"their real name is {name}"])

                    if old_name and old_name.lower() != name.lower():
                        logger.info(f"Command: name updated '{old_name}' -> '{name}' for {username}")
                    else:
                        logger.info(f"Command: stored name '{name}' for {username}")

            elif cmd_type == "set_hobby":
                hobby = params.get("hobby", "")
                if hobby:
                    mem.update_user_profile(user_id, username, "hobbies", hobby)
                    logger.info(f"Command: stored hobby '{hobby}' for {username}")

            elif cmd_type == "set_personality":
                personality = params.get("personality", "")
                if personality:
                    mem.update_user_profile(user_id, username, "personality", personality)
                    logger.info(f"Command: stored personality for {username}")

            elif cmd_type == "set_location":
                location = params.get("location", "")
                if location:
                    mem.update_user_profile(user_id, username, "real_name", username)  # ensure user exists
                    mem.update_user_memory(user_id, username, [f"from {location}"])
                    logger.info(f"Command: stored location '{location}' for {username}")

            elif cmd_type == "remember":
                fact = params.get("fact", "")
                if fact:
                    mem.update_user_memory(user_id, username, [fact])
                    logger.info(f"Command: stored fact for {username}: {fact[:60]}")

            elif cmd_type == "speed_up":
                # Reduce read delays — store as a runtime preference
                self._speed_multiplier = 0.5  # 50% faster
                logger.info("Command: speed up responses")

            elif cmd_type == "slow_down":
                self._speed_multiplier = 1.5  # 50% slower
                logger.info("Command: slow down responses")

        except Exception as e:
            logger.warning(f"Command handling failed for {cmd_type}: {e}")

    async def _handle_reply(self, message: discord.Message, ch_id: str):
        """Generate and send a reply to a message."""
        is_dm = isinstance(message.channel, (discord.DMChannel, discord.GroupChannel))

        # Update mood based on message content
        self.mood_engine.maybe_drift(trigger_text=message.content)

        # ── Abuse detection — fight back if abused ────────────────────────
        # Algorithmically detect abuse and generate a response BEFORE the AI
        # This ensures fast, deterministic, in-character pushback
        abuse_response = self.abuse_handler.handle_abuse(message.author.id, message.content)
        if abuse_response:
            # Fight back with the abuse response instead of the AI reply
            logger.info(f"Abuse detected from {message.author.name} — fighting back")
            # Simulate a short typing delay
            typing_delay = random.uniform(1.0, 3.0)
            await asyncio.sleep(typing_delay)
            async with message.channel.typing():
                await asyncio.sleep(min(len(abuse_response) * 0.05, 2.0))
            sent = await message.channel.send(abuse_response)
            self.last_replied[ch_id] = time.time()
            self.record_send(ch_id)
            if ch_id not in self.history_cache:
                self.history_cache[ch_id] = deque(maxlen=40)
            self.history_cache[ch_id].append(sent)
            # Update mood to annoyed
            self.mood_engine.force_mood("annoyed")
            return

        # Detect sentiment for emotional intelligence
        sentiment, sentiment_score = detect_sentiment(message.content)
        sentiment_modifier = get_sentiment_timing_modifier(sentiment)

        # Simulate read/notification delay (modulated by sentiment + speed pref)
        msg_len = len(message.content)
        speed_mult = getattr(self, '_speed_multiplier', 1.0)
        combined_modifier = sentiment_modifier * speed_mult

        # ── Late message detection ──────────────────────────────────────
        # If the message is a reply to an old message (10+ min), or if the
        # bot hasn't replied in this channel for 10+ min, simulate a longer
        # "opening Discord and noticing the message" delay.
        late_delay = 0
        msg_age = time.time() - message.created_at.timestamp()
        since_last_reply = time.time() - self.last_replied.get(ch_id, 0)

        if msg_age > 600 or since_last_reply > 600:
            # Message is 10+ min old, or bot hasn't replied in 10+ min
            # Simulate "opening Discord late" — longer notification delay
            late_delay = random.uniform(3.0, 8.0)
            logger.debug(f"Late message detected (age: {int(msg_age)}s, since last reply: {int(since_last_reply)}s) — adding {late_delay:.1f}s delay")

        notif_delay = 0
        if self.user in message.mentions or is_dm or message.reference is not None:
            notif_delay = random.uniform(2.5, 6.0) * combined_modifier  # Opening app on ping

        if msg_len < 20:
            read_delay = random.uniform(0.4, 1.5) * combined_modifier
        elif msg_len < 80:
            read_delay = random.uniform(1.0, 3.0) * combined_modifier
        else:
            read_delay = min(random.uniform(2.0, 4.0) + msg_len * 0.005, 10.0) * combined_modifier

        total_delay = max(notif_delay + read_delay + late_delay, 0.5)  # minimum 0.5s
        logger.debug(f"Simulating {total_delay:.1f}s read delay [sentiment: {sentiment}, speed: {speed_mult}x, late: {late_delay:.1f}s]...")
        await asyncio.sleep(total_delay)

        try:
            # Build transcript from the TTL-based message cache
            # Smart context: 4 messages + all messages from the same user
            # within the last 10 minutes (so the bot sees the full burst flow)
            transcript = self.msg_cache.get_context_text(ch_id, count=8)

            # Also get reactions from the discord history cache (for vibe)
            ctx_msgs = list(self.history_cache.get(ch_id, []))[-10:]
            now = time.time()
            reaction_info = ""
            for m in reversed(ctx_msgs):
                if m.reactions and m.content:
                    age = int(now - m.created_at.timestamp())
                    if age < 60:
                        time_label = f"{age}s ago"
                    elif age < 3600:
                        time_label = f"{age // 60}m ago"
                    else:
                        time_label = f"{age // 3600}h ago"
                    parts = []
                    for rx in m.reactions[:6]:
                        parts.append(f"{rx.emoji}x{rx.count}")
                    rx_str = f" [{', '.join(parts)}]"
                    label = "You" if m.author == self.user else m.author.name
                    reaction_info += f"[{time_label}] {label}: {m.clean_content}{rx_str}\n"

            user_id = str(message.author.id)
            username = message.author.name

            # Check for images
            image_urls = [
                a.url for a in message.attachments
                if a.content_type and a.content_type.startswith("image/")
            ]
            gif_embeds = [
                e.url for e in message.embeds
                if e.url and ("tenor.com" in e.url or "giphy.com" in e.url)
            ]

            trigger_text = message.clean_content
            if not trigger_text and image_urls:
                trigger_text = "[shared an image]"
            if gif_embeds:
                trigger_text = (trigger_text + " [sent a GIF]").strip()

            # ── Command detection ──────────────────────────────────────────
            # Check if the user is giving a command (bump, remember, etc.)
            logger.debug(f"[cmd] trigger_text='{trigger_text[:100]}'")
            cmd_type, cmd_params = detect_command(trigger_text)
            logger.info(f"[cmd] detect_command result: cmd_type={cmd_type}, params={cmd_params}")

            # Fallback: check vc_manager's more comprehensive VC join detection
            if not cmd_type and detect_vc_join_request(trigger_text):
                logger.info(f"[cmd] detect_vc_join_request matched for: '{trigger_text[:80]}'")
                cmd_type = "join_vc"
                cmd_params = {}

            if cmd_type:
                await self._handle_command(cmd_type, cmd_params, message, ch_id)
                # Leave commands produce no text at all — the disconnect is
                # the only acknowledgment. Skip the LLM reply path entirely
                # (otherwise [COMMAND DETECTED] generates a second message).
                if cmd_type == "leave_vc":
                    return

            # ── Side-channel action worker ────────────────────────────────
            # Natural-language Discord actions the regex commands missed:
            # "send him a dm", "react to that", "change your status to dnd",
            # "ping me in 10 mins". Silent execution — the normal reply path
            # acknowledges it in-character via action_note below.
            action_note = ""
            if not cmd_type and self._wants_action_worker(message, trigger_text, is_dm):
                action_note = await self._run_action_request(message, trigger_text)

            # ── Scan recent messages for unexecuted commands ────────────────
            # When the bot is mentioned, check if there were commands in recent
            # messages that the bot missed (e.g., "ping Mr. Alien" sent before
            # the bot was mentioned). This handles the case where the user sends
            # a command as a separate message and then mentions the bot.
            if not cmd_type and (self.user in message.mentions or is_dm):
                cache_msgs = list(self.history_cache.get(ch_id, []))[-5:]
                for recent_msg in reversed(cache_msgs):
                    # Skip bot's own messages and the current message
                    if recent_msg.author == self.user or recent_msg.id == message.id:
                        continue
                    # Only check messages from the same user in the last 2 minutes
                    age = time.time() - recent_msg.created_at.timestamp()
                    if age > 120:
                        continue
                    if recent_msg.author != message.author:
                        continue
                    # Check if this recent message contains a command
                    recent_cmd, recent_params = detect_command(recent_msg.content)
                    if recent_cmd and recent_cmd in ("mention_user", "mention_user_id", "join_vc", "leave_vc", "start_chat"):
                        logger.info(f"Found unexecuted command '{recent_cmd}' in recent message: '{recent_msg.content[:50]}'")
                        await self._handle_command(recent_cmd, recent_params, message, ch_id)
                        cmd_type = recent_cmd  # Mark as handled
                        break

            # Gather context
            rules_text = ""
            if hasattr(message.channel, 'guild') and message.channel.guild:
                rules_text = await fetch_guild_rules(message.channel.guild, self.rules_cache)

            recent_replies = list(self.reply_history.get(ch_id, []))
            channel_topic = mem.get_channel_topic(ch_id)
            channel_style = mem.get_channel_style(ch_id)
            channel_lessons = mem.get_channel_lessons(ch_id)
            ch_name = getattr(message.channel, 'name', '') or ''

            # ── Channel nature analysis (24h cache, 20 msgs first, 5 msgs refresh) ──
            channel_nature = get_cached_nature(ch_id)
            if channel_nature is None and hasattr(message.channel, 'guild') and message.channel.guild:
                # Analyze in background (don't block the reply)
                asyncio.create_task(analyze_channel_nature(message.channel))
                channel_nature = "general"  # Default while analyzing
            if channel_nature and channel_nature != "general":
                nature_summary = get_nature_summary(channel_nature)
                # Add nature to channel topic if not already there
                if nature_summary and nature_summary not in (channel_topic or ""):
                    channel_topic = nature_summary if not channel_topic else f"{channel_topic} ({nature_summary})"
            discord_topic = getattr(message.channel, 'topic', '') or ''

            # ── User memory: everything we know about this user ────────────
            user_memory_text = mem.get_user_memory_text(user_id, username)
            user_facts_detail = mem.get_all_user_facts(user_id)

            # ── Active instructions from users ─────────────────────────────
            instructions_text = mem.get_instructions_text()

            # Build list of mentioned users (excluding the bot) for context
            mentioned_other_names = [
                u.display_name for u in message.mentions
                if u != self.user and not u.bot
            ]

            # Add sentiment context to the transcript for emotional intelligence
            sentiment_ctx = get_sentiment_context(sentiment, sentiment_score)
            if sentiment_ctx:
                transcript = sentiment_ctx + "\n" + transcript

            # Add user memory context to transcript
            if user_facts_detail:
                transcript = f"[USER MEMORY: {user_facts_detail}]\n" + transcript

            # Cross-modal context: if this user has also been talking to the
            # bot in a voice call recently, the text reply should know
            voice_lines = self._user_recent_voice.get(message.author.id)
            if voice_lines:
                fresh_voice = [t for ts, t in voice_lines if time.time() - ts < 600]
                if fresh_voice:
                    said = " / ".join(f'"{t}"' for t in fresh_voice[-3:])
                    transcript = f"[{username} also said in the voice call recently: {said}]\n" + transcript

            # ── 10-message lookahead — multi-part messages / things aimed at us ──
            # Before replying, scan the last ~10 channel messages: same-author
            # messages in the last ~3min ("im asking to u?" then "?") and any
            # message that looks directed at the bot. Prepended so the reply
            # addresses the FULL message train, not just the latest fragment.
            lookahead = []
            try:
                now_ts = time.time()
                our_names = {self.user.display_name.lower(), self.user.name.lower(), "eudora"}
                for m in list(self.history_cache.get(ch_id, []))[-10:]:
                    if m.id == message.id or m.author == self.user or m.author.bot:
                        continue
                    age = now_ts - m.created_at.timestamp()
                    if age > 180:
                        continue
                    txt = (m.content or "").strip()
                    if not txt:
                        continue
                    # same author continuing a thought — or anyone clearly
                    # addressing the bot ("u?", "eudora ...", "@bot", "answer me")
                    same_author = m.author.id == message.author.id
                    aimed_at_us = (
                        any(f"{n} " in txt.lower() or txt.lower().endswith(n) or txt.lower() == n for n in our_names)
                        or self.user in m.mentions
                        or (same_author and "u" == txt.lower().strip("?"))
                        or (same_author and is_question(txt))
                    )
                    if same_author or aimed_at_us:
                        lookahead.append(f'{m.author.display_name}: "{txt[:120]}"')
            except Exception:
                pass
            if lookahead:
                ctx = " | ".join(lookahead[-4:])
                transcript = f"[just before this: {ctx}]\n" + transcript

            # Familiarity context — if we've already been talking to this user
            # in the last ~30min, don't let the reply re-greet them
            convo_ts = self.conversation_tracker.get(ch_id, {}).get(user_id)
            if convo_ts and (time.time() - convo_ts) < 1800:
                transcript = f"[you've already been talking with {username} — don't greet them like it's the first time]\n" + transcript

            # Add instructions context to transcript
            if instructions_text:
                transcript = f"[{instructions_text}]\n" + transcript

            # ── Owner respect system ─────────────────────────────────────────
            # If the user is an owner, add context telling the AI to respect them
            owner_context = get_owner_context(user_id, username)
            if owner_context:
                transcript = owner_context + "\n" + transcript
                # Add any stored owner instructions
                owner_instructions = get_owner_tracker().get_all_owner_instructions()
                if owner_instructions:
                    transcript = owner_instructions + "\n" + transcript
                # Record owner interaction
                get_owner_tracker().record_interaction(user_id)
            elif detect_command_attempt(trigger_text):
                # Non-owner trying to command the bot — add pushback context
                non_owner_context = get_non_owner_command_context(username)
                transcript = non_owner_context + "\n" + transcript
                logger.info(f"Non-owner {username} attempted command: {trigger_text[:50]}")

            # Add command context if a command was detected
            if cmd_type:
                transcript = f"[COMMAND DETECTED: {cmd_type} — already handled, acknowledge naturally]\n" + transcript
            elif action_note:
                transcript = action_note + " — already done silently; acknowledge naturally in your own words, don't ask if they want it done\n" + transcript

            # ── Add unanswered questions context ────────────────────────────
            # If there are unanswered questions from recent messages, prepend them
            # so the AI knows to answer them (e.g., user asked "how old are you?"
            # but the bot missed it, then mentioned the bot)
            unanswered_note = self.unanswered_tracker.get_context_note(ch_id)
            if unanswered_note:
                transcript = unanswered_note + "\n" + transcript
                logger.info(f"Including {len(self.unanswered_tracker.get_unanswered_questions(ch_id))} unanswered question(s) in context")

            # ── Conversational initiative context ────────────────────────────
            # Add engagement analysis and follow-up encouragement
            initiative_context = get_conversational_initiative_context(
                user_id, is_answering_question=is_question(trigger_text), reply_text=""
            )
            if initiative_context:
                transcript = initiative_context + "\n" + transcript

            # Generate reply (in executor to not block event loop)
            loop = asyncio.get_running_loop()
            ai_data = await loop.run_in_executor(
                None,
                lambda: ai_reply.generate_reply(
                    transcript, username, user_id, trigger_text or "[shared something]",
                    self.mood_engine.current_mood, recent_replies, rules_text,
                    channel_style, self.my_profile_text, channel_lessons,
                    channel_topic, ch_name, discord_topic, image_urls,
                    my_name=self.user.display_name,
                    mentioned_users=mentioned_other_names,
                )
            )

            if not ai_data:
                logger.warning("AI returned no data — skipping")
                return

            reply_text = ai_data.get("reply")
            burst_reply = ai_data.get("burst_reply")
            reaction_emoji = ai_data.get("reaction")
            new_status = ai_data.get("new_status")
            new_mood = ai_data.get("new_mood")
            search_query = ai_data.get("search_query")

            # Update mood if AI suggested a change
            if new_mood and str(new_mood).strip() in MOODS:
                self.mood_engine.set_mood(str(new_mood).strip())

            # Update status if AI suggested one
            if new_status and str(new_status).strip() not in ("null", "None", ""):
                if new_status != self.current_status_text:
                    try:
                        await self.change_presence(activity=discord.CustomActivity(name=str(new_status)))
                        self.current_status_text = str(new_status)
                        logger.debug(f"Status -> {new_status}")
                    except Exception as e:
                        logger.debug(f"Status change failed: {e}")

            # Retry with stronger instruction if reply was null — BUT only
            # force a reply when the bot is directly addressed (mention/DM/reply).
            # For random/sticky participation, RESPECT the LLM's null decision
            # (it correctly decided the message wasn't directed at the bot).
            is_directly_addressed = is_dm or (self.user in message.mentions) or (message.reference is not None and message.reference.resolved == self.user)

            if not reply_text or str(reply_text).strip() in ("null", "None", ""):
                if is_directly_addressed:
                    # Direct address — must respond even if LLM returned null
                    logger.debug("LLM returned null for direct address, retrying with stronger instruction")
                    try:
                        retry_transcript = transcript + "\n\nIMPORTANT: You MUST provide a reply text. Do not return null or empty. Always say something."
                        retry_data = await loop.run_in_executor(
                            None,
                            lambda: ai_reply.generate_reply(
                                retry_transcript, username, user_id, trigger_text or "[shared something]",
                                self.mood_engine.current_mood, recent_replies, rules_text,
                                channel_style, self.my_profile_text, channel_lessons,
                                channel_topic, ch_name, discord_topic, image_urls,
                                my_name=self.user.display_name,
                                mentioned_users=mentioned_other_names,
                            )
                        )
                        if retry_data and retry_data.get("reply"):
                            reply_text = retry_data.get("reply")
                            reaction_emoji = retry_data.get("reaction") or reaction_emoji
                            logger.debug(f"Retry succeeded: {str(reply_text)[:60]}")
                    except Exception as e:
                        logger.debug(f"Reply retry failed: {e}")
                else:
                    # LLM decided to stay silent for a non-directed message — respect that
                    logger.debug("LLM returned null for non-directed message — staying silent (human-like)")
                    self.engagement.on_skipped(ch_id)
                    return

            # React with emoji — use algorithmic reaction system
            # (Moved AFTER reply-text check: only react if we actually have a reply)

            # Send main reply
            if reply_text and str(reply_text).strip() not in ("null", "None", ""):
                reply_text = ai_reply.humanize(str(reply_text).strip())[:2000]

                # Dedup check
                last = self.last_sent.get(ch_id, "")
                if ai_reply.is_duplicate(reply_text, last):
                    logger.debug(f"Skipping duplicate reply: '{reply_text[:40]}'")
                    return

                # React with emoji — use algorithmic reaction system
                # First check if AI suggested a reaction, then use our algorithm
                # to decide whether to actually send it (not on every message)
                if reaction_emoji and str(reaction_emoji).strip() not in ("null", "None", ""):
                    emoji_clean = str(reaction_emoji).strip()
                    # Validate emoji
                    is_valid = (
                        re.match(r'^<a?:[a-zA-Z0-9_]+:[0-9]+>$', emoji_clean) is not None
                        or not re.search(r'[a-zA-Z()\[\]{}\\/;:,\.!?\-=_+]', emoji_clean)
                    )
                    if is_valid:
                        # Use algorithm to decide whether to react (cooldown, daily limit)
                        tracker = get_tracker()
                        if tracker.can_react(ch_id) and not tracker.is_recent_emoji(ch_id, emoji_clean):
                            try:
                                await message.add_reaction(emoji_clean)
                                tracker.record_reaction(ch_id, emoji_clean)
                                logger.debug(f"Reacted (AI): {emoji_clean}")
                            except Exception as e:
                                logger.debug(f"Reaction failed: {e}")
                else:
                    # No AI reaction suggested — use our algorithm to potentially react
                    algo_emoji = react_to_message(
                        message.content,
                        is_directed_at_bot=(self.user in message.mentions or message.reference is not None),
                        is_reply_to_bot=message.reference is not None,
                        is_mention=(self.user in message.mentions),
                        mood=self.mood_engine.current_mood,
                        channel_id=ch_id,
                    )
                    if algo_emoji:
                        try:
                            await message.add_reaction(algo_emoji)
                            logger.debug(f"Reacted (algorithm): {algo_emoji}")
                        except Exception as e:
                            logger.debug(f"Algorithm reaction failed: {e}")

                # Split into burst messages for natural sending
                # Algorithmically determine message count based on conversation nature
                convo_nature = detect_conversation_nature(reply_text, trigger_text or "")
                msg_count = determine_message_count(
                    reply_text,
                    mood=self.mood_engine.current_mood,
                    is_question=convo_nature["is_question"],
                    is_story=convo_nature["is_story"],
                    is_excited=convo_nature["is_excited"],
                )
                if msg_count > 1:
                    bursts = split_into_messages(reply_text, msg_count)
                else:
                    bursts = [reply_text]

                # Decide whether to use reply (quote) or plain message.
                # Algorithmic: direct triggers (DM/mention/question) always
                # quote; plain conversational replies quote ~60% so she
                # visibly answers the specific message, plain ~40% feels casual.
                is_dm = isinstance(message.channel, (discord.DMChannel, discord.GroupChannel))
                is_mention = self.user in message.mentions
                is_direct_reply = message.reference is not None
                is_question_msg = is_question(trigger_text)
                use_reply_quote = is_dm or is_mention or is_direct_reply or is_question_msg

                if not use_reply_quote:
                    use_reply_quote = random.random() < 0.60
                elif not is_mention and not is_question_msg and random.random() < 0.20:
                    use_reply_quote = False

                # Send each burst message with natural typing delays
                for i, burst_text in enumerate(bursts):
                    if not burst_text or not burst_text.strip():
                        continue

                    # Typing simulation (proportional to message length)
                    typing_dur = max(min(len(burst_text) / random.uniform(5.5, 8.0), 12.0), 2.0)
                    logger.debug(f"Typing {typing_dur:.1f}s [mood: {self.mood_engine.current_mood}] burst {i+1}/{len(bursts)}")
                    async with message.channel.typing():
                        await asyncio.sleep(typing_dur)

                    try:
                        if use_reply_quote and i == 0:
                            await message.reply(burst_text)
                        else:
                            await message.channel.send(burst_text)
                    except (discord.NotFound, discord.HTTPException):
                        await message.channel.send(burst_text)

                    # Natural delay between bursts (if more to send)
                    if i < len(bursts) - 1:
                        inter_delay = random.uniform(1.5, 4.0)
                        await asyncio.sleep(inter_delay)

                self.last_replied[ch_id] = time.time()
                self.last_sent[ch_id] = reply_text
                self.record_send(ch_id)
                self.engagement.on_sent(ch_id)

                # Mark unanswered questions as answered (we just replied)
                self.unanswered_tracker.mark_answered(ch_id)

                # Record bot message for re-engagement tracking
                get_re_engagement_tracker().record_bot_message(ch_id)

                # Record bot message for engagement analysis (response timing)
                get_engagement_analyzer().record_bot_message_to_user(user_id)

                # Set conversation stickiness: for the next 2 minutes, we'll
                # respond to messages in this channel without needing pings.
                # Shorter window prevents the "answers everything" feedback loop.
                self.sticky_until[ch_id] = time.time() + 120
                logger.debug(f"Sticky convo active in #{message.channel} for 2 min")

                # Track conversation with this specific user (multi-user support)
                # This ensures the bot replies to this user's messages even after
                # the sticky conversation expires (up to 30 min)
                if ch_id not in self.conversation_tracker:
                    self.conversation_tracker[ch_id] = {}
                self.conversation_tracker[ch_id][user_id] = time.time()
                logger.debug(f"Conversation tracked with {username} in #{message.channel}")

                if ch_id not in self.reply_history:
                    self.reply_history[ch_id] = deque(maxlen=8)
                self.reply_history[ch_id].append(reply_text)
                logger.info(f"Sent reply ({len(bursts)} burst(s)): {reply_text[:80]}")

                # ── Sticker/GIF sending (low frequency, algorithmic) ────────
                # Try to send a sticker after the reply (very low chance)
                sticker_mgr = get_sticker_manager()
                if sticker_mgr.should_send_sticker(message.content, ch_id):
                    if hasattr(message.channel, 'guild') and message.channel.guild:
                        sticker = sticker_mgr.select_sticker(
                            message.channel.guild, message.content, ch_id
                        )
                        if sticker:
                            try:
                                await message.channel.send(stickers=[sticker])
                                logger.info(f"Sent sticker: {sticker.name}")
                            except Exception as e:
                                logger.debug(f"Sticker send failed: {e}")

                # Try to send a GIF (low chance, or always if user requested)
                gif_mgr = get_gif_manager()
                force_gif = is_gif_request(trigger_text)
                if gif_mgr.should_send_gif(message.content, ch_id, force=force_gif):
                    gif_query = gif_mgr.get_gif_query(message.content)
                    if not gif_query and force_gif:
                        gif_query = "cool"
                    if gif_query:
                        # Search for a GIF using Discord's native GIF search API (falls back to Tenor)
                        gif_url = await gif_mgr.search_gif(gif_query)
                        if not gif_url:
                            # Fallback: try scraping Tenor search page
                            gif_url = await gif_mgr.search_gif_fallback(gif_query)
                        if gif_url:
                            try:
                                await message.channel.send(gif_url)
                                gif_mgr.record_gif(ch_id)
                                logger.info(f"Sent GIF: {gif_query} -> {gif_url[:60]}")
                            except Exception as e:
                                logger.debug(f"GIF send failed: {e}")
                        elif force_gif:
                            # User requested a GIF but search failed — tell them
                            logger.warning(f"GIF search failed for query '{gif_query}'")
                        else:
                            logger.debug(f"GIF search failed for '{gif_query}'")

                # Burst reply (follow-up message)
                if burst_reply and str(burst_reply).strip() not in ("null", "None", "") and self.can_send(ch_id):
                    burst_text = ai_reply.humanize(str(burst_reply).strip())[:200]
                    # Don't chain bare "wbu?"-style bursts — if a generic
                    # follow-up already went out in the last few bot msgs,
                    # only a *specific* question gets through
                    if _is_generic_followup(burst_text) and any(
                            _is_generic_followup(b)
                            for b in list(self.reply_history.get(ch_id, []))[-4:]):
                        logger.debug(f"Suppressed repeat generic burst: '{burst_text[:40]}'")
                        burst_text = None
                    if burst_text and ai_reply._similarity_ratio(burst_text, reply_text) < 0.5:
                        burst_delay = random.uniform(2.0, 5.0)
                        logger.debug(f"Burst in {burst_delay:.1f}s: '{burst_text[:60]}'")
                        await asyncio.sleep(burst_delay)
                        async with message.channel.typing():
                            await asyncio.sleep(random.uniform(1.5, 3.0))
                        await message.channel.send(burst_text)
                        self.record_send(ch_id)
                        self.reply_history[ch_id].append(burst_text)
                        logger.info(f"Burst sent: {burst_text[:60]}")

                # Background web search if needed
                if search_query and str(search_query).strip() not in ("null", "None", ""):
                    query_clean = str(search_query).strip()[:120]
                    asyncio.create_task(self._do_background_search(
                        message.channel, transcript, username, user_id, query_clean, ch_id
                    ))

                # Background memory/analysis (very low chance to save tokens)
                asyncio.create_task(self._background_analysis(
                    transcript, username, user_id, ch_id
                ))
            else:
                logger.debug("AI decided not to reply (null reply)")

        except Exception as e:
            if "429" in str(e) or "rate_limit" in str(e).lower():
                logger.warning("Rate limit hit in reply handler")
            else:
                logger.error(f"Error processing reply: {e}")

    async def _do_background_search(self, channel, prompt: str, username: str,
                                     user_id: str, query: str, ch_id: str):
        """Run a web search in the background and send an informed follow-up."""
        try:
            await asyncio.sleep(random.uniform(3.0, 8.0))
            loop = asyncio.get_running_loop()
            search_result = await loop.run_in_executor(None, web_search.search, query)
            if not search_result:
                logger.debug(f"BG search for '{query}' returned nothing")
                return

            logger.debug(f"BG search result ({len(search_result)} chars)")
            ai_data = await loop.run_in_executor(
                None, lambda: ai_reply.generate_reply_with_search(prompt, search_result)
            )
            if not ai_data:
                return

            follow_up = ai_data.get("reply")
            if follow_up and str(follow_up).strip() not in ("null", "None", ""):
                follow_up = ai_reply.humanize(str(follow_up).strip())[:300]
                if self.can_send(ch_id):
                    async with channel.typing():
                        await asyncio.sleep(random.uniform(1.0, 2.5))
                    await channel.send(follow_up)
                    self.record_send(ch_id)
                    logger.info(f"BG search follow-up sent: '{follow_up[:60]}'")
        except Exception as e:
            logger.debug(f"BG search error: {e}")

    async def _background_analysis(self, transcript: str, username: str,
                                    user_id: str, ch_id: str):
        """Run memory extraction and channel analysis in background (low chance)."""
        try:
            await asyncio.sleep(random.uniform(5, 12))
            loop = asyncio.get_running_loop()

            # Memory extraction (2% chance) — only for high-value messages
            if random.random() < 0.02:
                # Use user behavior classification to decide if worth extracting
                if should_store_message(transcript):
                    new_facts = await loop.run_in_executor(
                        None, lambda: ai_reply.extract_memory(transcript, username)
                    )
                    if new_facts:
                        mem.update_user_memory(user_id, username, new_facts)

            # Channel topic update (0.5% chance)
            if random.random() < 0.005:
                topic = await loop.run_in_executor(
                    None, lambda: ai_reply.extract_channel_topic(transcript)
                )
                if topic:
                    mem.update_channel_topic(ch_id, topic)

            # Channel style update (0.5% chance)
            if random.random() < 0.005:
                style = await loop.run_in_executor(
                    None, lambda: ai_reply.extract_channel_style(transcript)
                )
                if style:
                    mem.update_channel_style(ch_id, style)

            # Self-reflection (1% chance)
            if random.random() < 0.01:
                lessons = await loop.run_in_executor(
                    None, lambda: ai_reply.self_reflect(transcript)
                )
                if lessons:
                    mem.update_channel_lessons(ch_id, lessons)

        except Exception as e:
            logger.debug(f"BG analysis error: {e}")

    async def on_raw_reaction_add(self, payload: discord.RawReactionActionEvent):
        """Notice when someone reacts to one of our messages."""
        if payload.user_id == self.user.id:
            return
        try:
            channel = self.get_channel(payload.channel_id)
            if channel is None:
                return
            message = await channel.fetch_message(payload.message_id)
            if message.author != self.user:
                return

            emoji = str(payload.emoji)
            ch_id = str(channel.id)
            logger.debug(f"Someone reacted {emoji} to our message in #{channel}")

            POSITIVE_RX = {"👍", "❤️", "😂", "💀", "🔥", "😭", "💯", "W", "✅", "❤", "😍", "🤣", "😆"}
            NEGATIVE_RX = {"👎", "😒", "🙄", "💔", "❌", "🤦"}

            # 20% chance to react back or respond
            if random.random() > 0.20:
                return
            if not self.can_send(ch_id):
                return

            if emoji in POSITIVE_RX:
                if random.random() < 0.30:
                    counter_rx = random.choice(["😭", "💀", "🫡", "🔥"])
                    try:
                        await message.add_reaction(counter_rx)
                        logger.debug(f"Counter-reacted {counter_rx}")
                    except Exception:
                        pass
            elif emoji in NEGATIVE_RX:
                if random.random() < 0.25:
                    replies = ["lol ok", "whatever", "fair enough", "idk man", "noted"]
                    await asyncio.sleep(random.uniform(1.5, 4.0))
                    await channel.send(random.choice(replies))
                    self.record_send(ch_id)
                    logger.debug(f"Responded to negative reaction")
        except Exception as e:
            logger.debug(f"Reaction handler error: {e}")

    async def on_message_edit(self, before, after):
        """Update history cache when a message is edited."""
        ch_id = str(after.channel.id)
        if ch_id in self.history_cache:
            cache = self.history_cache[ch_id]
            for i, m in enumerate(cache):
                if m.id == after.id:
                    cache[i] = after
                    break
