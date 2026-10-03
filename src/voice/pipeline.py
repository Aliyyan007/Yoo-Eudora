"""Voice pipeline — orchestrates VAD → ASR → LLM → TTS for one user utterance.

This is the core of the voice conversation system:
1. Receive audio from Discord voice channel (discord-native-voice)
2. Run VAD to detect speech start/end
3. Feed speech to ASR for transcription
4. Send transcript to LLM (existing Groq) for response
5. Stream response to TTS (Fish Audio) for voice output
6. Play TTS audio back in the voice channel

ARCHITECTURE (improved based on EchoTwin research):
- Pre-roll buffer: 300ms of 48kHz PCM buffered before VAD triggers,
  drained into ASR on speech start to capture the first syllable.
- Wall-clock watchdog: endpoint detection uses a wall-clock timer
  (600ms silence) instead of VAD's utterance_ended alone, which is
  more reliable.
- Two-pass ASR: streaming zipformer for live partials, then
  faster-whisper re-scores the full utterance for much better accuracy.
- Noise suppression: noisereduce cleans audio before the final ASR pass.
- VAD tuning: threshold 0.5, low 0.3, frame_window 3, silence 600ms.
"""
from __future__ import annotations

import asyncio
import os
import queue as _queue
import time
import struct
from typing import Optional, Callable, Awaitable

import numpy as np
import discord
from loguru import logger

from .vad import SileroVAD, WebRTCVAD, VADResult
from .asr import StreamingASR
from .tts import FishAudioTTS, TTSConfig
from .audio_source import OpusAudioSource, ComfortNoiseSource
from .chunker import chunk_text, speakable
from .ogg_demux import OggDemuxer
from .preroll_buffer import PrerollRingBuffer
from .vc_intent import IrritationTracker, farewell_score, leave_vc_score, song_request_score, vc_invite_score, vote


def _add_breath_effects(text: str) -> str:
    """Add natural breath and pause tags to TTS text for more human-like speech.

    Fish Audio S2.1-pro supports [bracket] paralanguage tags:
    - [pause] — short pause (like a comma break)
    - [inhale] — audible breath before a sentence
    - (breath) — softer breath (requires normalize=false)

    This function inserts these tags at natural points:
    - Before sentences that start with "well", "so", "I mean", "honestly" → add [pause]
    - Between sentences (after . ! ?) → add [pause]
    - Occasionally before the start → add [inhale] for a natural breath

    Only applies to text that doesn't already contain bracket tags.
    """
    import re
    import random

    # Don't double-process if tags already present — includes singing/music
    # tags, which shouldn't get speech pauses sprinkled through the lyrics
    if "[" in text and "]" in text and any(
        t in text for t in ["[pause]", "[inhale]", "[breath]", "[sing", "[break]", "[long-break]", "[music"]
    ):
        return text

    result = text

    # Add [pause] between sentences (after . ! ?)
    # This creates a natural pause between thoughts
    result = re.sub(r'([.!?])\s+', r'\1 [pause] ', result)

    # Add [inhale] at the start ~30% of the time (natural breath before speaking)
    if random.random() < 0.3:
        result = f"[inhale] {result}"

    # Add [pause] before conversational fillers (well, so, I mean, honestly, to be fair)
    fillers = ["well", "so", "I mean", "honestly", "to be fair", "like"]
    for filler in fillers:
        # Only add pause if the filler is mid-sentence (not at the very start)
        result = re.sub(
            rf'([a-z,])\s+{filler}\s',
            rf'\1 [pause] {filler} ',
            result,
            count=1  # Only first occurrence to avoid overdoing it
        )

    return result


def _has_repeated_phrase(text: str, n: int = 3) -> bool:
    """Detect ASR hallucination loops — the same n-word phrase repeated
    (e.g. 'I HAVE IN COMMUNITY SHOES AND I HAVE IN COMMUNITY SHOES').
    Natural speech almost never repeats an exact 3-word sequence."""
    if not text:
        return False
    words = text.lower().split()
    if len(words) < n * 2:
        return False
    seen = set()
    for i in range(len(words) - n + 1):
        gram = " ".join(words[i:i + n])
        if gram in seen:
            return True
        seen.add(gram)
    return False


# Two-pass ASR (optional — only if faster-whisper is installed)
try:
    from .whisper_two_pass import WhisperTwoPass
    _HAS_WHISPER = True
except ImportError:
    _HAS_WHISPER = False
    WhisperTwoPass = None  # type: ignore

# Groq Whisper API two-pass (fast, cloud-based — preferred over local Whisper)
try:
    from .groq_whisper import GroqWhisperTwoPass
    _HAS_GROQ_WHISPER = True
except ImportError:
    _HAS_GROQ_WHISPER = False
    GroqWhisperTwoPass = None  # type: ignore

# AI auto-correction of transcripts
try:
    from .transcript_corrector import auto_correct_transcript
    _HAS_CORRECTOR = True
except ImportError:
    _HAS_CORRECTOR = False

# Noise suppression (optional — only if noisereduce is installed)
try:
    from .noise_suppress import NoiseSuppressor
    _HAS_NOISE_SUPPRESS = True
except ImportError:
    _HAS_NOISE_SUPPRESS = False
    NoiseSuppressor = None  # type: ignore

# ── Endpoint watchdog settings ────────────────────────────────────────────
# Wall-clock silence to trigger utterance end. More reliable than VAD alone.
# Reduced to 500ms for fast response — the bot should reply almost
# immediately after the user stops talking. 500ms is enough to detect
# natural sentence endings without cutting mid-sentence pauses.
_ENDPOINT_SILENCE_MS = 450     # 0.45s — fast response, catches sentence endings
_ENDPOINT_SILENCE_MS_SLOW = 900  # feeble/slow speakers get a longer grace window
_MIN_UTTERANCE_MS = 250        # drop utterances shorter than this
_MIN_TRANSCRIPT_WORDS = 1      # accept 1-word utterances

# ── Adaptive gain control (feeble/quiet mics) ──────────────────────────────
# Quiet mics produce low-energy PCM that never crosses the VAD threshold and
# transcribes poorly. Frames are boosted toward a target RMS so feeble speech
# behaves like a normal mic. Noise-floor gated and capped so silence isn't
# amplified into false VAD triggers; per-user EMA smoothing avoids pumping.
_AGC_TARGET_RMS = 2400         # target speech RMS in int16 units (~-22 dBFS)
_AGC_MAX_GAIN = 8.0            # max boost (~18 dB)
_AGC_NOISE_FLOOR = 90          # below this RMS → near-silence, don't amplify
_AGC_SMOOTHING = 0.35          # EMA pull toward the per-frame target gain
_FEEBLE_RMS = 1400             # speech-energy EMA below this → "feeble" speaker
_MAX_UTTERANCE_S = 12.0        # force endpoint if an utterance runs this long
_MAX_UTTERANCE_PCM = 48000 * 2 * 15   # ~15s of buffered 48k audio (whisper input cap)
_PKT_QUEUE_MAX = 150           # ~3s of 20ms packets — beyond this we're already behind

# Streaming sherpa ASR is heavy on weak CPUs — VOICE_STREAM_ASR=0 runs the
# Whisper-only path (Groq API is faster AND more accurate anyway).
_STREAM_ASR_DISABLED = os.getenv("VOICE_STREAM_ASR", "1") == "0"
# noisereduce is a spectral-gating CPU hog — opt-in via VOICE_NOISE_SUPPRESS=1.
# Whisper handles noisy audio well on its own.
_NOISE_SUPPRESS_DISABLED = os.getenv("VOICE_NOISE_SUPPRESS", "0") != "1"
# VAD backend: "webrtc" (~50x cheaper, eats 48kHz directly) or "silero"
_VAD_BACKEND = os.getenv("VOICE_VAD", "silero").lower()
_WEBRTC_AGGRESSIVENESS = int(os.getenv("VOICE_WEBRTC_MODE", "1"))

# Short interjections that are always real speech, never noise
_ALLOWED_SHORT = {"no", "ok", "hi", "yo", "um", "oh", "yeah", "yes", "nah", "lol", "wait"}

# ── VAD settings (tuned for maximum sensitivity without false triggers) ───
# Lowered threshold from 0.4 to 0.3 to catch quieter/whispered speech.
# Discord audio can be quiet, and 0.4 was missing too many speech frames.
_VAD_THRESHOLD = 0.3           # high threshold (voice start) — was 0.4
_VAD_THRESHOLD_LOW = 0.15      # low threshold (voice end, hysteresis) — was 0.25
_VAD_SILENCE_MS = 500          # min silence to end utterance — was 700 (faster endpointing)
_VAD_FRAME_WINDOW = 2          # consecutive voice frames required (noise spike suppression)

# ── Pre-roll buffer ───────────────────────────────────────────────────────
# Increased from 15 to 20 frames (400ms) to capture more of the first syllable
# since we lowered the VAD threshold (slower trigger = need more pre-roll).
_PREROLL_FRAMES = 20           # 20 × 20ms = 400ms of 48kHz PCM


class VoicePipeline:
    """Per-user voice pipeline: VAD + ASR for listening, LLM + TTS for speaking."""

    def __init__(
        self,
        tts_config: TTSConfig,
        on_transcript: Callable[[int, str], Awaitable[Optional[str]]],
        bot_id: int,
        on_transcript_stream: Optional[Callable] = None,
        persona_gender: str = "female",
    ):
        """
        Args:
            tts_config: Fish Audio TTS configuration
            on_transcript: async callback(user_id, text) -> LLM response text or None
            bot_id: The bot's Discord user ID (to skip own audio)
            on_transcript_stream: optional async-generator callback
                (user_id, text) -> yields reply text pieces as the LLM streams
                them. When set, replies are pipelined to TTS sentence-by-
                sentence instead of waiting for the full response.
            persona_gender: 'male'|'female' — picks the hmm/ack sound set so a
                male persona gets gruff acknowledgments, not soft humming.
        """
        self._tts_config = tts_config
        self._on_transcript = on_transcript
        self._on_transcript_stream = on_transcript_stream
        self._bot_id = bot_id
        self._persona_gender = persona_gender

        # Per-user state
        self._user_vads: dict[int, SileroVAD] = {}
        self._user_asrs: dict[int, StreamingASR] = {}
        self._user_speaking: dict[int, bool] = {}
        self._user_prerolls: dict[int, PrerollRingBuffer] = {}
        self._user_in_speech: dict[int, bool] = {}
        self._user_last_voice_time: dict[int, float] = {}
        self._user_utterance_pcm: dict[int, bytearray] = {}  # raw 48k PCM for two-pass
        self._user_watchdog_tasks: dict[int, asyncio.TimerHandle] = {}

        # Per-user DSP state — decoders and resamplers are STATEFUL (frame
        # prediction, filter history) and must never be shared across users
        self._user_decoders: dict = {}        # user_id -> opuslib.Decoder
        self._user_resamplers: dict = {}      # user_id -> Resampler
        self._user_agc_gain: dict[int, float] = {}    # user_id -> smoothed AGC gain
        self._user_rms_ema: dict[int, float] = {}     # user_id -> speech energy EMA
        self._user_last_response: dict[int, float] = {}  # user_id -> last reply ts
        self._user_utt_start: dict[int, float] = {}   # user_id -> utterance start ts
        self._finalizing: set = set()                 # users mid-finalize (no double-entry)

        # Serialized per-user packet processing — packets go on a queue and
        # a worker drains them one at a time (no coroutine pileup, no stale
        # state reads). DSP runs on a dedicated thread pool, not the loop.
        self._user_pkt_queues: dict[int, asyncio.Queue] = {}
        self._user_pkt_workers: dict[int, asyncio.Task] = {}
        self._dsp_pool = None  # lazy ThreadPoolExecutor for decode+VAD work

        # Social-brain state — algorithmic, per-user:
        # irritation score (abuse→angry tone+louder TTS), farewell timestamps
        # (goodbye turns suppress questions + deprioritize proactive pings),
        # and an optional pending "can I move to another VC?" vote
        self._irritation = IrritationTracker()
        self._user_farewell_at: dict[int, float] = {}
        self._pending_vc_move: Optional[dict] = None
        # Turn-taking extras — per-user:
        # _user_last_transcript: dedup key (same text within 15s = hallucination
        #   repeat). Was pipeline-global — user B's "yeah" after user A's "yeah"
        #   got dropped in multi-user calls.
        # _user_ack_at: scheduled mid-utterance backchannel time (inf = none).
        #   While someone monologues, a real listener occasionally goes "mhm".
        # _user_reply_gen: generation counter — a newer utterance endpointing
        #   while a reply is still generating bumps it; stale replies drop
        #   before speaking ("answer what they said LAST").
        # _user_finalize_queued: at most one deferred finalize per user.
        self._user_last_transcript: dict[int, tuple] = {}
        self._user_ack_at: dict[int, float] = {}
        self._user_reply_gen: dict[int, int] = {}
        self._user_finalize_queued: set = set()
        self._relocate_cb = None   # set by manager — leave+join fallback for vc.move_to
        self._leave_cb = None      # set by manager — spoken "leave the vc" → disconnect

        # Two-pass ASR (shared, lazily loaded)
        # Priority: Groq Whisper API (fast, cloud) → local faster-whisper (fallback)
        # Groq whisper-large-v3-turbo processes audio in ~0.1s vs 2-6s for local base.en
        self._whisper = None
        self._whisper_type = None  # "groq" or "local"

        if _HAS_GROQ_WHISPER:
            try:
                self._whisper = GroqWhisperTwoPass(model="whisper-large-v3-turbo")
                self._whisper_type = "groq"
                logger.info("[voice] Using Groq Whisper API for two-pass (whisper-large-v3-turbo)")
            except Exception as e:
                logger.warning(f"[voice] Groq Whisper init failed: {e!r}")

        if self._whisper is None and _HAS_WHISPER:
            try:
                self._whisper = WhisperTwoPass(model_size="base.en")
                self._whisper_type = "local"
                logger.info("[voice] Using local faster-whisper for two-pass (base.en)")
            except Exception as e:
                logger.warning(f"[voice] Local Whisper init failed: {e!r}, trying tiny.en")
                try:
                    self._whisper = WhisperTwoPass(model_size="tiny.en")
                    self._whisper_type = "local"
                except Exception as e2:
                    logger.warning(f"[voice] Whisper tiny.en also failed: {e2!r}")

        # Noise suppressor (shared) — opt-in; spectral gating is too expensive
        # for weak CPUs and whisper-large-v3 handles noise fine without it
        self._noise_suppressor = None
        if _HAS_NOISE_SUPPRESS and not _NOISE_SUPPRESS_DISABLED:
            try:
                self._noise_suppressor = NoiseSuppressor(enabled=True)
            except Exception:
                pass

        # Shared state
        self._is_speaking = False  # Bot is currently speaking
        self._speak_active = False  # True for entire _speak() incl. TTS-open window (barge-in only clears _is_speaking)
        self._speak_deadline = 0.0  # Safety: auto-reset _is_speaking after 35s
        self._play_queue: Optional[_queue.Queue] = None
        self._voice_client = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # Comfort noise — keeps the Discord voice indicator active like a real human
        self._comfort_noise: Optional[ComfortNoiseSource] = None
        self._comfort_noise_active = False

        # User activity tracking — for proactive engagement
        # Tracks when each user last spoke, how many times, and their display names
        self._user_last_speech_time: dict[int, float] = {}   # user_id -> timestamp of last speech
        self._user_speech_count: dict[int, int] = {}          # user_id -> total utterances
        self._user_display_names: dict[int, str] = {}         # user_id -> display name (set by manager)
        self._user_mic_active: dict[int, bool] = {}           # user_id -> mic currently open (speaking)
        self._last_bot_speech_time: float = 0.0               # when bot last finished speaking
        self._last_proactive_time: float = 0.0                # when bot last asked a proactive question

        # Hmm sound player — algorithmic thinking sounds during silence
        self._hmm_player = None  # initialized when VC is joined

        # Speak lock — prevents concurrent _speak calls (causes "Already playing media")
        self._speak_lock = asyncio.Lock()

    def set_voice_client(self, vc, loop: asyncio.AbstractEventLoop):
        """Set the current Discord voice client for audio playback."""
        self._voice_client = vc
        self._loop = loop

        # Start hmm sound player for algorithmic thinking sounds
        if self._hmm_player is None:
            try:
                from .hmm_sounds import HmmSoundPlayer
                self._hmm_player = HmmSoundPlayer(self)
                self._hmm_player.start()
            except Exception as e:
                logger.warning(f"[voice] Hmm sound player init failed: {e}")

    def start_comfort_noise(self) -> None:
        """Start playing comfort noise to keep the voice indicator active.

        This makes the bot look like a real human in the call — the green ring
        flickers slightly as if there's background noise, instead of being
        completely silent like a bot.
        """
        if not self._voice_client or not self._loop:
            return
        if self._comfort_noise_active:
            return
        # A speak attempt is in progress — don't steal the player before
        # _speak calls play() (barge-in schedules us via call_later, which can
        # fire during the TTS-open window → "Already playing media")
        if self._is_speaking or self._speak_active:
            return
        try:
            # Stop any existing playback first
            if self._voice_client.is_playing():
                self._voice_client.stop()
                import time as _t
                _t.sleep(0.1)

            self._comfort_noise = ComfortNoiseSource(noise_level=120)
            self._voice_client.play(self._comfort_noise)
            self._comfort_noise_active = True
            logger.info("[voice] Comfort noise started (voice indicator active)")
        except Exception as e:
            logger.debug(f"[voice] Comfort noise start error: {e}")

    def stop_comfort_noise(self) -> None:
        """Stop comfort noise playback (e.g. before speaking TTS)."""
        if self._comfort_noise:
            try:
                self._comfort_noise.stop()
            except Exception:
                pass
            self._comfort_noise = None
        if self._voice_client and self._voice_client.is_playing():
            try:
                self._voice_client.stop()
            except Exception:
                pass
        self._comfort_noise_active = False

    def _get_vad(self, user_id: int):
        if user_id not in self._user_vads:
            if _VAD_BACKEND == "webrtc":
                try:
                    self._user_vads[user_id] = WebRTCVAD(
                        aggressiveness=_WEBRTC_AGGRESSIVENESS,
                        min_silence_duration_ms=_VAD_SILENCE_MS,
                        frame_window=_VAD_FRAME_WINDOW,
                    )
                except Exception as e:
                    logger.warning(f"[voice] WebRTCVAD unavailable ({e!r}) — falling back to silero")
                    self._user_vads[user_id] = self._make_silero()
            else:
                self._user_vads[user_id] = self._make_silero()
        return self._user_vads[user_id]

    def _make_silero(self) -> SileroVAD:
        return SileroVAD(
            min_silence_duration_ms=_VAD_SILENCE_MS,
            threshold=_VAD_THRESHOLD,
            threshold_low=_VAD_THRESHOLD_LOW,
            frame_window=_VAD_FRAME_WINDOW,
        )

    def _get_preroll(self, user_id: int) -> PrerollRingBuffer:
        if user_id not in self._user_prerolls:
            self._user_prerolls[user_id] = PrerollRingBuffer(max_frames=_PREROLL_FRAMES)
        return self._user_prerolls[user_id]

    async def _get_asr(self, user_id: int) -> StreamingASR:
        # Streaming sherpa ASR can be disabled on weak CPUs — the Whisper
        # two-pass path handles transcription on its own (and wins the
        # stream-vs-whisper decision almost every time anyway).
        if _STREAM_ASR_DISABLED:
            return None
        # Use a sentinel to prevent race condition: without this, the first
        # utterance would create dozens of ASR instances (one per 20ms packet)
        # because open() takes ~7s to download the model on first use.
        if user_id in self._user_asrs:
            asr = self._user_asrs[user_id]
            if asr is not None:
                return asr
            return None  # Previous open failed

        # Mark as "loading" to prevent concurrent creation
        self._user_asrs[user_id] = None  # sentinel
        asr = StreamingASR()
        try:
            await asr.open()
            self._user_asrs[user_id] = asr
        except Exception as e:
            logger.error(f"[voice] ASR open failed for user {user_id}: {e!r}")
            # Leave sentinel as None so we don't retry forever on every packet
            self._user_asrs[user_id] = None
            return None  # type: ignore
        return asr

    async def process_audio(self, user_id: int, opus_packet: bytes) -> None:
        """Enqueue an Opus packet for processing.

        Called from discord-native-voice's listen callback (~50 packets/sec).
        Packets go onto a per-user queue drained serially by a worker task —
        this means pipeline state can never race between packets and heavy
        DSP work runs on a dedicated thread pool instead of the event loop.
        """
        if user_id == self._bot_id:
            return

        q = self._user_pkt_queues.get(user_id)
        if q is None:
            q = asyncio.Queue(maxsize=_PKT_QUEUE_MAX)
            self._user_pkt_queues[user_id] = q
            loop = self._loop or asyncio.get_running_loop()
            self._user_pkt_workers[user_id] = loop.create_task(
                self._pkt_worker(user_id)
            )
        try:
            q.put_nowait(opus_packet)
        except asyncio.QueueFull:
            pass  # already behind — drop stale audio rather than pile up

    async def _pkt_worker(self, user_id: int) -> None:
        """Serially drain a user's packet queue — one packet at a time,
        batching any backlog into a single DSP call."""
        q = self._user_pkt_queues[user_id]
        while True:
            pkt = await q.get()
            pkts = [pkt]
            try:
                while True:
                    pkts.append(q.get_nowait())
            except asyncio.QueueEmpty:
                pass
            try:
                await self._process_packets(user_id, pkts)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"[voice] Packet processing error for {user_id}: {e!r}")

    def _trim_tail_silence(self, audio: np.ndarray) -> np.ndarray:
        """Cut trailing silence from buffered 16k audio. Scans 20ms frames
        backward for the last frame with real energy, keeps a ~160ms tail so
        the last syllable isn't clipped."""
        frame = 320  # 20ms @16kHz
        n = len(audio)
        if n < frame * 10:
            return audio
        abs_a = np.abs(audio)
        i = n
        while i >= frame:
            if float(abs_a[i - frame:i].mean()) > 0.005:
                end = min(n, i + frame * 8)
                trimmed = n - end
                if trimmed > frame:
                    logger.debug(f"[voice] Trimmed {trimmed/16000:.2f}s trailing silence before Whisper")
                return audio[:end]
            i -= frame
        return audio

    def _play_ack(self) -> None:
        """Instant 'mhm/yeah' ack at utterance end — plays pre-generated opus
        frames (~50% of turns). The real reply's _speak() calls vc.stop() if
        this is still playing, so it can't collide."""
        try:
            import random as _rng
            from .hmm_sounds import _sounds_for
            sounds = _sounds_for(self)
            if not sounds:
                return
            vc = self._voice_client
            if not vc or not vc.is_connected():
                return
            if self._is_speaking or self._speak_active:
                return
            if vc.is_playing() and not self._comfort_noise_active:
                # Real playback (a reply/hmm) owns the player — don't steal it
                return
            if _rng.random() > 0.5:
                return
            # Prefer the short acknowledgments; long ones undercut the reply
            acks = [s for s in sounds if len(s) <= 6]
            packets = sounds.get(_rng.choice(acks or list(sounds)), [])
            if not packets:
                return
            self.stop_comfort_noise()
            from .audio_source import CachedOpusSource
            vc.play(CachedOpusSource(packets, volume=0.8), after=lambda e: self._ack_done(e))
        except Exception as e:
            logger.debug(f"[voice] Ack play failed: {e}")

    def _ack_done(self, error) -> None:
        """After the ack finishes, restart comfort noise (a no-op if a real
        reply has already taken the player — start_comfort_noise bails while
        _speak_active is set)."""
        try:
            if self._loop and self._loop.is_running():
                self._loop.call_later(0.3, self.start_comfort_noise)
        except Exception:
            pass

    def _build_turn_hints(self, user_id: int, text: str) -> dict:
        """Algorithmic intent scoring on the final transcript — all regex
        scorers, zero latency, zero extra API calls.

        Returns a hints dict consumed two ways:
        - 'directive' → appended to the LLM user prompt (same-turn tone control)
        - 'vol_db'/'speed_mult' → applied to the TTS prosody for this turn
        - 'farewell' → suppresses follow-up questions this turn
        """
        hints = {"directive": "", "vol_db": 0.0, "speed_mult": 1.0,
                 "farewell": False, "vc_invite": False,
                 "leave_cmd": False, "sing": False}

        # Leave command — 'leave the vc', 'get out of this call' → she actually
        # disconnects (resolved in _finalize_inner before the reply path)
        if leave_vc_score(text) >= 0.6:
            hints["leave_cmd"] = True

        # Song request — 'sing a song', 'sing for me' → lyrics + singing prosody
        if song_request_score(text) >= 0.7:
            hints["sing"] = True
            hints["directive"] = (
                "They're asking you to sing — write a short original song: a verse "
                "and a hook, 4-7 lines max, about whatever they suggested (or "
                "anything fun if they didn't). Output ONLY the lyrics — one line "
                "per line, no commentary, no question. Make them rhyme and flow."
            )

        # Farewell — 'i gotta go', 'goodnight', 'see you later' → goodbye, not
        # another question. Recorded so proactive doesn't ping a leaving user.
        if farewell_score(text) >= 0.6:
            hints["farewell"] = True
            self._user_farewell_at[user_id] = time.time()
            hints["directive"] = (
                "They're saying goodbye — reply with a short warm farewell. "
                "Do NOT ask a question or try to keep them talking."
            )

        # VC-invite — 'come to our vc', 'join my vc' — flagged for the client.
        # If it's spoken, the speaker is in THIS vc — the target channel can't
        # be resolved from audio, so she asks them to message her about it.
        if vc_invite_score(text) >= 0.5:
            hints["vc_invite"] = True
            hints["directive"] = (
                hints["directive"] +
                " They're asking you to join another vc — you can't see that "
                "channel from here, so tell them to message you about it in "
                "text and you'll sort it. Keep it light and short."
            ).strip()

        # Irritation — severity feeds the decaying per-user score; the tier
        # drives BOTH the reply tone (prompt directive) and the voice itself
        # (louder + faster TTS). Algorithmic: builds with repeated abuse,
        # decays when they calm down, drops on apologies.
        score = self._irritation.feed(user_id, text)
        tier_name, tier_directive, vol_db, spd_mult, _ = self._irritation.tier(user_id)
        if tier_name != "calm":
            hints["directive"] = (hints["directive"] + " " + tier_directive).strip()
            hints["vol_db"] = vol_db
            hints["speed_mult"] = spd_mult
            logger.info(f"[voice] Irritation {tier_name} ({score:.0f}) for user {user_id}")
        return hints

    # ── VC transfer requests ("can I go to their vc?") ────────────────────
    # Fallback phrasings if the LLM ask fails — each one reads differently
    _VC_MOVE_ASKS = [
        "hey — {name} is asking me to join their vc. you lot alright if i go?",
        "so {name} wants me in their vc — anyone mind if i head over?",
        "quick one — {name} is calling me to their vc. cool if i dip?",
        "{name} is asking me to come to their vc — you gonna manage without me?",
        "oi — {name} wants me over in their vc. yes or no?",
        "heads up — {name}'s asking me to move to their vc. any objections?",
        "{name} needs me in their vc apparently — can i go or you keeping me?",
    ]

    async def request_vc_move(self, requester_name: str, channel, notify_cb=None) -> None:
        """Someone asked the bot to join a different VC while it's already in
        this one — ask the current VC for permission (LLM-phrased, different
        every time), then listen ~45s for a yes/no answer in the transcripts."""
        self._pending_vc_move = {
            "requester_name": requester_name,
            "channel": channel,
            "asked_at": time.time(),
            "notify_cb": notify_cb,
        }
        ask = await self._gen_vc_move_ask(requester_name)
        await self._speak(ask)

    async def _gen_vc_move_ask(self, requester_name: str) -> str:
        """Phrase the permission-ask with the LLM for variety — falls back to
        the hardcoded pool if anything fails."""
        import random as _r
        try:
            from ..ai import llm
            from ..ai import prompts as ai_prompts
            loop = asyncio.get_event_loop()
            resp = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: llm.call_voice(
                        "vc_move_ask",
                        ai_prompts.VOICE_REPLY_SYSTEM,
                        f"{requester_name} just messaged asking you to come to their voice channel, "
                        f"but you're already in this one. Ask the people here if it's okay for you "
                        f"to go — casually, 1 short sentence, mention {requester_name}. "
                        f"Say ONLY what you'd speak out loud.",
                        want_json=False,
                    ),
                ),
                timeout=8.0,
            )
            if resp and resp.strip():
                import re as _re
                line = _re.sub(r'\*+([^*]+)\*+', r'\1', resp).replace("\n", " ").strip()
                if len(line) > 10:
                    return line
        except Exception as e:
            logger.debug(f"[voice] LLM move-ask failed: {e}")
        return _r.choice(self._VC_MOVE_ASKS).format(name=requester_name)

    async def _resolve_vc_move_vote(self, user_id: int, text: str) -> bool:
        """If a move ask is pending, a decisive yes/no transcript resolves it.
        Returns True if this utterance was consumed as a vote."""
        pending = self._pending_vc_move
        if pending is None:
            return False
        if time.time() - pending["asked_at"] > 45:
            self._pending_vc_move = None  # timed out — stay put
            return False
        v = vote(text)
        if v is None:
            return False
        self._pending_vc_move = None
        import random as _r
        if v == "yes":
            logger.info(f"[voice] VC move approved by {user_id} — moving to '{pending['channel'].name}'")
            await self._speak(_r.choice([
                f"alright, off to {pending['requester_name']}'s vc then — see you lot",
                f"sweet — heading to {pending['requester_name']}'s vc, catch you",
                f"done deal — off to {pending['requester_name']}'s. laters",
            ]))
            await self._do_vc_move(pending["channel"])
        else:
            logger.info(f"[voice] VC move declined by {user_id} — staying")
            await self._speak(_r.choice([
                "they don't want me to go — i'm staying innit",
                "nah they said stay — i'm not going anywhere",
                "vetoed — i'm staying right here",
            ]))
        cb = pending.get("notify_cb")
        if cb:
            try:
                await cb(v == "yes")
            except Exception:
                pass
        return True

    # ── Singing ──────────────────────────────────────────────────────────
    # They asked for a song: songwriter LLM writes a short verse+hook, then
    # it's pushed through a dedicated TTS session tuned for singing
    # ([singing] tag, slower lingering speed, higher temperature, bigger
    # chunks for prosodic continuity). Lyrics are text — one line per
    # push+flush gives the model its phrase breaks.
    def _sing_tts_cfg(self) -> TTSConfig:
        """TTSConfig tuned for singing — slower + more expressive + bigger
        context chunks, low repetition penalty so held notes can sustain."""
        from dataclasses import replace as _dc_replace
        return _dc_replace(
            self._tts_config,
            speed=0.85,           # lingering, sustained delivery
            volume_db=0.5,
            temperature=0.85,     # more pitch/prosody variation (melodic)
            top_p=0.9,
            chunk_length=300,     # more context → smoother lines
            repetition_penalty=1.05,  # let sung notes/vowels sustain
        )

    async def _handle_sing(self, user_id: int, request_text: str) -> bool:
        """Generate lyrics and sing them. Returns False → caller falls back
        to the normal reply path."""
        lyrics = await self._gen_lyrics(user_id, request_text)
        if not lyrics:
            return False
        logger.info(f"[voice] Singing for {user_id}: '{lyrics.splitlines()[0][:60]}...'")
        try:
            await self._speak(lyrics, tts_cfg=self._sing_tts_cfg())
            return True
        except Exception as e:
            logger.error(f"[voice] Sing speak failed: {e!r}")
            return False

    async def _gen_lyrics(self, user_id: int, request_text: str) -> str:
        """Songwriter call — short verse+hook lyrics about whatever they asked
        for. Keeps newlines (each line = one push+flush phrase break)."""
        name = self._user_display_names.get(user_id) or ""
        subject = request_text.strip()
        try:
            from ..ai import llm
            from ..persona.runtime import active as _active_persona
            _p = _active_persona()
            loop = asyncio.get_event_loop()
            resp = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: llm.call_voice(
                        "songwriter",
                        (
                            f"You are {_p.name.capitalize()} — {_p.short_identity} "
                            "You write songs that sound like real people "
                            "singing in a VC — casual, rhyming, a bit silly, full of heart."
                        ),
                        (
                            f"They asked you to sing: \"{subject}\"\n\n"
                            f"Write a short song — one verse + a hook, 5-7 lines total. "
                            f"{'Mention ' + name + ' in it if it flows naturally. ' if name else ''}"
                            "Rules: every line rhymes or flows, keep lines SHORT (4-9 words), "
                            "make it feel like a real improvised song not a poem. "
                            "Output ONLY the lyrics — one line per line. No title, no "
                            "commentary, no questions, no asterisks."
                        ),
                        want_json=False,
                        max_tokens=400,   # reasoning model needs headroom for lyrics
                    ),
                ),
                timeout=10.0,
            )
            if resp and resp.strip():
                import re as _re
                text = _re.sub(r'\*+([^*]+)\*+', r'\1', resp).strip()
                lines = [l.strip() for l in text.splitlines() if l.strip()]
                # Drop chatter lines the model sometimes wraps around lyrics
                lines = [l for l in lines if not _re.match(r'^(here|sure|okay|alright|enjoy|hope)', l.lower())]
                if 2 <= len(lines) <= 10:
                    # [singing] leads the whole thing — tag persists until the
                    # next tag; [break] marks the verse/hook boundary halfway
                    mid = len(lines) // 2
                    lines.insert(mid, "[break]")
                    return "[singing] " + "\n".join(lines)
        except Exception as e:
            logger.debug(f"[voice] Lyric generation failed: {e}")
        # Fallback — a tiny improvised song so the feature never dead-ends
        import random as _r
        fallback = _r.choice([
            ["[singing] You asked me for a song so here I go,",
             "singing in the vc nice and slow,",
             "[break]",
             "la la la, that's how it goes,",
             "that was my song, now you know!"],
            ["[singing] Oi oi, listen up innit,",
             "got a little tune for you, one minute,",
             "[break]",
             "tra la la and a hey hey hey,",
             "singing for you all night and day!"],
        ])
        return "\n".join(fallback)

    async def _do_vc_move(self, channel) -> None:
        """Move to another VC — try move_to first (keeps the connection),
        fall back to leave+join via the manager callback."""
        try:
            if self._voice_client and self._voice_client.is_connected():
                await self._voice_client.move_to(channel)
                return
        except Exception as e:
            logger.debug(f"[voice] vc.move_to failed ({e!r}) — falling back to leave+join")
        if self._relocate_cb:
            try:
                await self._relocate_cb(channel)
            except Exception as e:
                logger.error(f"[voice] VC relocate failed: {e!r}")

    def _dsp_executor(self):
        """Lazily-created thread pool for per-packet DSP work."""
        if self._dsp_pool is None:
            import concurrent.futures
            self._dsp_pool = concurrent.futures.ThreadPoolExecutor(
                max_workers=2, thread_name_prefix="voice-dsp"
            )
        return self._dsp_pool

    def _dsp_frame(self, user_id: int, packets: list):
        """Sync DSP for a packet batch — runs on the voice-dsp thread:
        Opus decode → AGC → resample (only if VAD needs 16k) → VAD."""
        in_speech = self._user_in_speech.get(user_id, False)
        pcms = []
        any_voice_data = False
        for pkt in packets:
            # DTX/comfort-silence packets are ≤8 bytes — skip decoding them
            # outside speech; they carry no voice energy
            if len(pkt) <= 8 and not in_speech:
                pcms.append(b"\x00" * 1920)
                continue
            try:
                pcm = self._decode_opus(user_id, pkt)
            except Exception as e:
                logger.error(f"Opus decode error for user {user_id}: {e!r}")
                continue
            if pcm:
                pcms.append(pcm)
                any_voice_data = True
        if not pcms:
            return None, None
        pcm_48k = b"".join(pcms)
        if not any_voice_data:
            # Whole batch was DTX silence — skip AGC/resample/VAD entirely
            return pcm_48k, None
        # Boost feeble/quiet audio — one boost covers VAD, streaming ASR,
        # and the buffered two-pass Whisper input
        pcm_48k = self._apply_agc(user_id, pcm_48k)
        vad = self._get_vad(user_id)
        if getattr(vad, "input_rate", 16000) == 48000:
            # webrtcvad eats 48kHz directly — no per-packet resample needed
            return pcm_48k, vad.feed(pcm_48k)
        return pcm_48k, vad.feed(self._resample_48k_to_16k(user_id, pcm_48k))

    async def _process_packets(self, user_id: int, packets: list) -> None:
        """Process a batch of incoming Opus packets.

        1. DSP (decode/AGC/resample/VAD) on the voice-dsp executor — keeps
           codec + onnxruntime inference off the event loop on weak CPUs
        2. Push to pre-roll ring buffer (captures first syllable)
        3. On speech start: drain pre-roll into ASR
        4. While in speech: feed ASR + accumulate PCM for the two-pass
        5. On silence/endpoints: finalize ASR + Whisper + LLM
        """
        try:
            pcm_48k, vad_result = await asyncio.get_running_loop().run_in_executor(
                self._dsp_executor(), self._dsp_frame, user_id, packets
            )
        except Exception as e:
            logger.error(f"DSP error for user {user_id}: {e!r}")
            return
        if pcm_48k is None:
            return
        if vad_result is None:
            vad_result = VADResult(is_voice=False, utterance_ended=False, speech_started=False)

        if self._is_speaking:
            # BARGE-IN: If a user starts speaking while the bot is talking,
            # stop the bot's speech so the user can be heard. This mimics
            # natural conversation — you stop talking when someone interrupts you.
            # Safety: if _is_speaking has been stuck for >35s, force reset
            if time.time() > self._speak_deadline:
                logger.warning("[voice] _is_speaking stuck for too long, force-resetting")
                self._is_speaking = False
            else:
                if vad_result.speech_started:
                    logger.info(f"[voice] BARGE-IN: user {user_id} started speaking, interrupting bot")
                    self.stop_speaking()
                    # Fall through to normal pipeline processing below
                else:
                    return  # Not speech — keep bot talking

        # Log first few packets for debugging
        if not hasattr(self, '_pkt_count'):
            self._pkt_count = 0
        self._pkt_count += 1
        if self._pkt_count <= 3 or (self._pkt_count % 500 == 0 and len(packets[0]) > 3):
            logger.info(f"[voice] process_audio #{self._pkt_count}: user={user_id}, batch={len(packets)}, opus_len={len(packets[0])}, opus_hex={packets[0][:10].hex()}")

        # ── Push to pre-roll buffer (ALWAYS — before VAD) ────────────────
        # This captures the first 300ms of audio so when VAD finally triggers,
        # we can feed the pre-roll into ASR and not lose the first syllable.
        preroll = self._get_preroll(user_id)
        preroll.push(pcm_48k)

        in_speech = self._user_in_speech.get(user_id, False)

        # ── Speech start: drain pre-roll into ASR ─────────────────────────
        if vad_result.speech_started and not in_speech:
            self._user_in_speech[user_id] = True
            self._user_speaking[user_id] = True
            self._user_utterance_pcm[user_id] = bytearray()
            self._user_utt_start[user_id] = time.time()
            self._user_last_voice_time[user_id] = time.time()
            self._user_mic_active[user_id] = True
            self._user_last_speech_time[user_id] = time.time()
            # Mid-utterance backchannel — while someone monologues, a real
            # listener occasionally goes "mhm"/"yeah" mid-speech. Decided
            # once at speech start (~40% of utterances); _play_ack uses
            # pre-cached frames so it costs nothing.
            import random as _rng2
            self._user_ack_at[user_id] = (
                time.time() + _rng2.uniform(2.5, 6.0)
                if _rng2.random() < 0.40 else float("inf"))
            logger.info(f"[voice] User {user_id} started speaking (pcm_48k={len(pcm_48k)})")

            # Drain pre-roll buffer into ASR — this is the KEY improvement
            # that captures the first syllable VAD's latency would have missed
            head = preroll.drain()
            if head:
                try:
                    asr = await self._get_asr(user_id)
                    if asr is not None:
                        await asr.feed_audio(head)
                        # Also add pre-roll to utterance PCM for two-pass
                        self._user_utterance_pcm[user_id].extend(head)
                except Exception as e:
                    logger.error(f"[voice] Pre-roll ASR feed error: {e!r}")

            # Start watchdog IMMEDIATELY when speech starts (not on next packet)
            self._restart_watchdog(user_id)

        # ── Feed to ASR while in speech ───────────────────────────────────
        # Once speech has started, feed ALL audio to ASR (not just voice frames).
        # The VAD threshold may miss quiet speech frames, but as long as we're
        # in a speech segment, we should keep feeding ASR. The watchdog will
        # handle endpointing based on wall-clock silence.
        if in_speech:
            # Track last time we received ANY audio packet (for watchdog)
            self._user_last_voice_time[user_id] = time.time()

            # Mid-utterance backchannel fires at its scheduled moment —
            # acknowledges without taking the floor
            if time.time() >= self._user_ack_at.get(user_id, float("inf")):
                self._user_ack_at[user_id] = float("inf")
                self._play_ack()

            # Feed to streaming ASR
            try:
                asr = await self._get_asr(user_id)
                if asr is not None:
                    await asr.feed_audio(pcm_48k)
            except Exception as e:
                logger.error(f"ASR feed error: {e!r}")

            # Accumulate raw 48k PCM for two-pass Whisper (capped — a stuck-open
            # utterance must not grow the whisper input + memory unbounded)
            if user_id in self._user_utterance_pcm:
                if len(self._user_utterance_pcm[user_id]) < _MAX_UTTERANCE_PCM:
                    self._user_utterance_pcm[user_id].extend(pcm_48k)

        # ── VAD utterance ended (endpointing) ────────────────────────────
        if vad_result.utterance_ended and in_speech:
            if self._is_feeble(user_id):
                # Feeble/slow speakers pause mid-sentence — defer to the
                # voice-gated watchdog so a short pause doesn't cut the
                # utterance in half. If speech resumes, ASR keeps feeding
                # because in_speech stays True.
                logger.debug(f"[voice] VAD endpoint for feeble user {user_id} — deferring to watchdog")
            else:
                logger.info(f"[voice] VAD endpoint for user {user_id}")
                await self._finalize_and_respond(user_id)
                return  # Already finalized, don't restart watchdog

        # ── Restart wall-clock watchdog on VOICE frames only ──────────────
        # Gating on is_voice makes the watchdog measure real silence —
        # Discord keeps streaming silence/keep-alive frames which otherwise
        # keep the watchdog alive forever and leave endpointing to VAD alone.
        if in_speech and vad_result.is_voice:
            # Hard cap on utterance length — a held-open VAD (boosted noise,
            # hysteresis, feeble-defer) must not accumulate audio forever
            if time.time() - self._user_utt_start.get(user_id, time.time()) > _MAX_UTTERANCE_S:
                logger.info(f"[voice] Utterance over {_MAX_UTTERANCE_S:.0f}s — forcing endpoint")
                await self._finalize_and_respond(user_id)
                return
            self._restart_watchdog(user_id)

    def _restart_watchdog(self, user_id: int) -> None:
        """Restart the wall-clock silence watchdog via loop.call_later —
        a cancelled-and-recreated Task per voice packet (~50Hz) was burning
        CPU; a timer Handle is ~10x cheaper to churn."""
        old = self._user_watchdog_tasks.get(user_id)
        if old is not None:
            old.cancel()

        # Feeble/slow speakers get a longer silence window so mid-sentence
        # pauses don't cut their utterance in half
        if self._loop and self._loop.is_running():
            delay_ms = _ENDPOINT_SILENCE_MS_SLOW if self._is_feeble(user_id) else _ENDPOINT_SILENCE_MS
            self._user_watchdog_tasks[user_id] = self._loop.call_later(
                delay_ms / 1000.0, self._watchdog_fire, user_id
            )

    def _watchdog_fire(self, user_id: int) -> None:
        """call_later callback — fires ~500-900ms after the last VOICE packet.

        The watchdog is only restarted on is_voice frames, so if it fires,
        that much wall-clock silence has passed — a true speech endpoint.
        """
        try:
            if not self._user_in_speech.get(user_id, False):
                return  # already finalized

            silence_duration = time.time() - self._user_last_voice_time.get(user_id, 0)
            logger.info(f"[voice] Watchdog endpoint for user {user_id} (silence={silence_duration:.2f}s)")
            self._loop.create_task(self._finalize_and_respond(user_id))
        except Exception as e:
            logger.error(f"[voice] Watchdog error for user {user_id}: {e!r}")

    def _decode_opus(self, user_id: int, opus_packet: bytes) -> bytes:
        """Decode Opus packet to 48kHz mono PCM 16-bit.

        Decoders are keyed per user — Opus carries frame-to-frame prediction
        state, so two users talking through one shared decoder corrupts both
        streams."""
        try:
            import opuslib
        except ImportError:
            try:
                import opuslib_next as opuslib
            except ImportError:
                logger.warning("No Opus decoder available (opuslib/opuslib_next)")
                return b""
        decoder = self._user_decoders.get(user_id)
        if decoder is None:
            decoder = opuslib.Decoder(48000, 1)
            self._user_decoders[user_id] = decoder
        # 20ms frame at 48kHz = 960 samples * 2 bytes = 1920 bytes
        return decoder.decode(opus_packet, 960)

    def _apply_agc(self, user_id: int, pcm_48k: bytes) -> bytes:
        """Adaptive gain control — boost quiet/feeble mics toward a target RMS
        so the VAD, streaming ASR, and Whisper two-pass can actually hear them.

        Also maintains a per-user speech-energy EMA (content frames only) for
        feeble-speaker detection, which grants a longer endpoint grace window."""
        samples = np.frombuffer(pcm_48k, dtype=np.int16).astype(np.float32)
        if samples.size == 0:
            return pcm_48k
        rms = float(np.sqrt(np.mean(samples * samples)))

        if rms >= _AGC_NOISE_FLOOR:
            # Content frame — fold into the speech-energy EMA (silence frames
            # are excluded so they don't drag the estimate toward feeble)
            prev_ema = self._user_rms_ema.get(user_id)
            self._user_rms_ema[user_id] = rms if prev_ema is None else prev_ema * 0.7 + rms * 0.3
            target_gain = 1.0 if rms >= _AGC_TARGET_RMS else min(_AGC_TARGET_RMS / rms, _AGC_MAX_GAIN)
        else:
            # Near-silence — pull the gain back toward unity; amplifying the
            # noise floor is how false VAD triggers happen
            target_gain = 1.0

        prev = self._user_agc_gain.get(user_id, 1.0)
        gain = prev + _AGC_SMOOTHING * (target_gain - prev)
        self._user_agc_gain[user_id] = gain
        if gain <= 1.02:
            return pcm_48k
        return np.clip(samples * gain, -32768, 32767).astype(np.int16).tobytes()

    def _is_feeble(self, user_id: int) -> bool:
        """Whether this user's speech energy is consistently low (quiet mic,
        soft-spoken, slow/feeble delivery) → grant extra endpoint grace."""
        ema = self._user_rms_ema.get(user_id)
        return ema is not None and ema < _FEEBLE_RMS

    def _resample_48k_to_16k(self, user_id: int, pcm_48k: bytes) -> bytes:
        """Resample 48kHz to 16kHz using high-quality soxr resampler.
        Falls back to scipy anti-aliased resampling, then simple decimation.

        Resamplers are keyed per user — soxr keeps filter state between
        frames, so sharing one across users cross-contaminates their audio.
        """
        if user_id not in self._user_resamplers:
            try:
                from ._resampler import Resampler
                self._user_resamplers[user_id] = Resampler(48000, 16000)
            except Exception:
                self._user_resamplers[user_id] = None
        resampler = self._user_resamplers[user_id]
        if resampler is not None:
            return resampler.feed(pcm_48k)
        # Fallback: scipy anti-aliased resampling
        try:
            from scipy.signal import resample_poly
            import numpy as np
            samples = np.frombuffer(pcm_48k, dtype=np.int16).astype(np.float32)
            resampled = resample_poly(samples, 1, 3)
            return resampled.astype(np.int16).tobytes()
        except ImportError:
            pass
        # Last resort: simple decimation (causes aliasing)
        import struct
        samples = struct.unpack(f'<{len(pcm_48k)//2}h', pcm_48k)
        decimated = samples[::3]
        return struct.pack(f'<{len(decimated)}h', *decimated)

    def _resample_48k_to_16k_float32(self, user_id: int, pcm_48k: bytes) -> np.ndarray:
        """Resample 48kHz int16 PCM to 16kHz float32 numpy array (for Whisper)."""
        pcm_16k_int16 = self._resample_48k_to_16k(user_id, pcm_48k)
        if not pcm_16k_int16:
            return np.zeros(0, dtype=np.float32)
        return np.frombuffer(pcm_16k_int16, dtype=np.int16).astype(np.float32) / 32768.0

    async def _finalize_and_respond(self, user_id: int) -> None:
        """Guard wrapper — an utterance must never be finalized twice
        concurrently (watchdog + VAD endpoint + length cap can race).

        NEW: if a reply for this user's previous utterance is still
        generating, we don't drop their new words — we bump the reply
        generation (the stale reply drops before speaking) and queue one
        deferred finalize. Humans answer what you said LAST, not first."""
        if user_id in self._finalizing:
            self._user_reply_gen[user_id] = self._user_reply_gen.get(user_id, 0) + 1
            if user_id not in self._user_finalize_queued:
                self._user_finalize_queued.add(user_id)
                try:
                    asyncio.get_running_loop().create_task(
                        self._deferred_finalize(user_id))
                except Exception:
                    self._user_finalize_queued.discard(user_id)
            logger.debug(f"[voice] Finalize superseded — queued re-finalize for user {user_id}")
            return
        self._finalizing.add(user_id)
        try:
            await self._finalize_inner(user_id)
        finally:
            self._finalizing.discard(user_id)

    async def _deferred_finalize(self, user_id: int) -> None:
        """Re-run finalize once the in-flight one frees. The streaming ASR
        kept accumulating while the slot was held, so end_utterance returns
        everything the user said since — the reply covers their LATEST
        words, not the superseded ones."""
        try:
            for _ in range(80):          # ~20s ceiling, then give up
                await asyncio.sleep(0.25)
                if user_id not in self._finalizing:
                    break
            if user_id not in self._finalizing:
                await self._finalize_and_respond(user_id)
        except Exception as e:
            logger.debug(f"[voice] deferred finalize failed for {user_id}: {e!r}")
        finally:
            self._user_finalize_queued.discard(user_id)

    async def _finalize_inner(self, user_id: int) -> None:
        """Get final ASR transcript, send to LLM, speak the response.

        Two-pass ASR:
        1. Get streaming ASR result (fast, lower accuracy)
        2. If Whisper is available, re-score the full utterance (slower, higher accuracy)
        3. Use the Whisper result if available and different, otherwise use streaming result
        """
        # Mark as no longer in speech
        self._user_in_speech[user_id] = False
        self._user_speaking[user_id] = False
        self._user_mic_active[user_id] = False
        self._user_ack_at.pop(user_id, None)

        # Instant acknowledgment — a pre-cached "mhm/yeah" plays ~immediately
        # (~100ms perceived response) while Whisper+LLM+TTS compute the real
        # reply. Zero cost: cached opus frames, no TTS call. Humans do exactly
        # this — it makes the bot feel alive rather than dead-slow.
        self._play_ack()

        # Clear the watchdog task reference — but do NOT cancel it.
        # The watchdog task IS the task that called _finalize_and_respond.
        # Cancelling it would cancel ourselves, raising CancelledError
        # at the next await point (e.g. asr.end_utterance), silently
        # killing the entire finalize pipeline. This was THE root cause
        # of the bot not listening or speaking.
        self._user_watchdog_tasks.pop(user_id, None)

        # Reset the VAD — it may still be mid-utterance (forced endpoint or the
        # deferred-feeble path). A fresh state guarantees the next voice onset
        # produces a new speech_started instead of desyncing into silence.
        vad = self._user_vads.get(user_id)
        if vad is not None:
            try:
                vad.reset()
            except Exception:
                pass

        # Clear pre-roll for next utterance
        preroll = self._get_preroll(user_id)
        preroll.clear()

        # ── Pass 1: Streaming ASR (fast, lower accuracy) ──────────────────
        # Bounded: sherpa decodes share a class-level infer lock + the default
        # executor pool — a starved/hung decode must not hang the whole turn.
        asr = self._user_asrs.get(user_id)
        stream_text = None
        if asr is not None:
            try:
                stream_text = await asyncio.wait_for(asr.end_utterance(), timeout=6.0)
            except asyncio.TimeoutError:
                logger.warning(f"[voice] ASR end_utterance timed out for {user_id} — resetting stream")
                try:
                    await asr.open()  # recreate stream so the next utterance decodes cleanly
                except Exception:
                    pass
            except Exception as e:
                logger.error(f"ASR finalize error: {e}")
        else:
            logger.debug(f"[voice] No streaming ASR for user {user_id} — Whisper-only path")

        if stream_text and len(stream_text.strip()) >= 2:
            logger.info(f"[voice] Stream transcript from {user_id}: {stream_text}")

        # ── Pass 2: Whisper re-score (ALWAYS run for better accuracy) ──────
        # The streaming zipformer is fast but less accurate. Whisper base.en
        # gives much better results. We always run Whisper and pick the better
        # result. The ~0.5s extra latency is worth the accuracy improvement.
        final_text = stream_text
        from_whisper = False
        utterance_pcm = self._user_utterance_pcm.pop(user_id, bytearray())

        # Always run Whisper if available and we have enough audio
        needs_whisper = (
            self._whisper is not None
            and len(utterance_pcm) > 32000  # > 0.33s of audio
        )

        if needs_whisper:
            try:
                # Convert 48k PCM to 16k float32 for Whisper
                audio_f32 = self._resample_48k_to_16k_float32(user_id, bytes(utterance_pcm))

                # Trim trailing silence — the endpoint wait (~0.5s) leaves dead
                # air in the buffer; cutting it shrinks the WAV upload and
                # Whisper's transcribe time
                audio_f32 = self._trim_tail_silence(audio_f32)

                # Apply noise suppression if available — off the event loop
                # (it blocks for several hundred ms on multi-second utterances)
                # and BOUNDED — a slow suppress on weak CPU must not starve the
                # ASR decode executor and stall the whole turn.
                if self._noise_suppressor is not None:
                    try:
                        audio_f32 = await asyncio.wait_for(
                            asyncio.get_running_loop().run_in_executor(
                                None, self._noise_suppressor.suppress, audio_f32
                            ),
                            timeout=8.0,
                        )
                    except asyncio.TimeoutError:
                        logger.warning("[voice] Noise suppression timed out — using raw audio")

                # Run Whisper
                whisper_text = await self._whisper.transcribe(audio_f32)

                if whisper_text and len(whisper_text.strip()) >= 1:
                    # Pick the better result:
                    # The streaming ASR always outputs UPPERCASE and can prepend garbage
                    # words (e.g. "SOCIETY AND SO CAN YOU HEAR ME" when the user just said
                    # "Can you hear me?"). The Groq Whisper API gives proper case +
                    # punctuation and is much more accurate.
                    #
                    # Strategy:
                    # 1. If streaming was empty/very short, use Whisper
                    # 2. If one side produced a repetition loop (ASR hallucination),
                    #    prefer the other — loops beat the length heuristic
                    # 3. If both have results, prefer Whisper UNLESS the stream is
                    #    significantly longer (2x+ AND more words) — the stream being
                    #    slightly longer is often just garbage words prepended
                    stream_len = len(stream_text.strip()) if stream_text else 0
                    whisper_len = len(whisper_text.strip())
                    stream_words = len(stream_text.strip().split()) if stream_text else 0
                    whisper_words = len(whisper_text.strip().split())
                    stream_loops = _has_repeated_phrase(stream_text)
                    whisper_loops = _has_repeated_phrase(whisper_text)

                    if not stream_text or stream_len < 2:
                        logger.info(f"[voice] Whisper result (stream was empty): '{whisper_text}'")
                        final_text = whisper_text
                        from_whisper = True
                    elif stream_loops and not whisper_loops:
                        logger.info(f"[voice] Stream repeated itself — Whisper result: '{whisper_text}' (stream: '{stream_text[:80]}')")
                        final_text = whisper_text
                        from_whisper = True
                    elif whisper_loops and not stream_loops:
                        logger.info(f"[voice] Whisper repeated itself — keeping stream: '{stream_text}' (whisper: '{whisper_text[:80]}')")
                        # final_text is already stream_text
                    elif stream_words >= whisper_words * 2 and stream_len > whisper_len * 2:
                        # Stream is 2x longer AND has 2x more words → likely more complete
                        logger.info(f"[voice] Keeping stream (2x longer): '{stream_text}' (whisper: '{whisper_text}')")
                        # final_text is already stream_text
                    else:
                        # Prefer Whisper — it has better accuracy, case, and punctuation
                        # The stream being slightly longer is usually garbage words
                        logger.info(f"[voice] Whisper result (better quality): '{whisper_text}' (stream: '{stream_text}')")
                        final_text = whisper_text
                        from_whisper = True
            except Exception as e:
                logger.error(f"[voice] Whisper two-pass failed: {e!r}")
        elif stream_text:
            logger.info(f"[voice] Using stream result: '{stream_text}'")

        # Clean up utterance PCM
        if user_id in self._user_utterance_pcm:
            del self._user_utterance_pcm[user_id]

        if not final_text or len(final_text.strip()) < 2:
            logger.info(f"[voice] Empty final transcript from user {user_id} (audio={len(utterance_pcm)/96000:.1f}s)")
            return

        # Check minimum utterance duration
        utterance_duration = len(utterance_pcm) / (48000 * 2)  # 48k, 16-bit
        if utterance_duration < _MIN_UTTERANCE_MS / 1000:
            logger.info(f"[voice] Utterance too short ({utterance_duration:.2f}s) from user {user_id} — transcript '{final_text[:40]}'")
            return

        # ── Filter stream ASR hallucinations on short audio ───────────────
        # The streaming ASR can produce fragments on noise. If the utterance
        # is very short (< 1s) and the transcript is a single short word,
        # it's likely a false positive — but let common interjections through
        # so a quick "yeah"/"hi" still gets a reply.
        if utterance_duration < 1.0 and len(final_text.strip()) <= 5:
            if final_text.strip().lower().strip("'.,!?") not in _ALLOWED_SHORT:
                logger.info(f"[voice] Short audio ({utterance_duration:.1f}s) with tiny transcript '{final_text}' — likely noise, skipping")
                return

        # Skip 1-word fragments — these are usually VAD false endpoints
        # that cut a sentence in half (e.g. "WORRY", "YOU MIGHT")
        word_count = len(final_text.strip().split())
        if word_count < _MIN_TRANSCRIPT_WORDS:
            logger.debug(f"[voice] Transcript too short ({word_count} word(s)): '{final_text}' — likely VAD false endpoint, skipping")
            return

        # Filter single very short words (< 3 chars) that aren't meaningful
        # These are usually ASR noise — but allow common short words
        if word_count == 1 and len(final_text.strip().strip("'.,!?")) < 3:
            if final_text.strip().lower().strip("'.,!?") not in _ALLOWED_SHORT:
                logger.info(f"[voice] Single very short word '{final_text}' — likely noise, skipping")
                return

        logger.info(f"[voice] Final transcript from {user_id}: {final_text}")

        # ── Transcript deduplication ───────────────────────────────────────
        # If the same transcript (or very similar) was processed recently,
        # skip it. This prevents the bot from responding to repeated Whisper
        # hallucinations (e.g. "Thank you." appearing 5 times in a row).
        now = time.time()
        normalized = final_text.strip().lower()
        last_t, last_ts = self._user_last_transcript.get(user_id, ("", 0.0))
        if normalized == last_t and (now - last_ts) < 15.0:
            logger.info(f"[voice] Duplicate transcript '{final_text}' within 15s — skipping")
            return
        self._user_last_transcript[user_id] = (normalized, now)

        # ── Response cooldown (per-user) ──────────────────────────────────
        # Don't respond to the SAME user more than once every 2.5s — prevents
        # response stacking on rapid short utterances. Per-user so one person's
        # reply doesn't suppress another user's turn in a multi-user call.
        last_response_time = self._user_last_response.get(user_id, 0)
        if (now - last_response_time) < 2.5:
            logger.info(f"[voice] Response cooldown ({now - last_response_time:.1f}s since last) — skipping '{final_text[:40]}'")
            return

        # ── Skip while bot is speaking ─────────────────────────────────────
        # If the bot is currently speaking, don't process new transcripts
        # unless it's a long utterance (likely real speech, not hallucination).
        # Short utterances while the bot is talking are usually just the bot's
        # own audio being picked up by the user's mic (echo).
        if self._is_speaking:
            if word_count < 3 and utterance_duration < 2.0:
                logger.info(f"[voice] Bot is speaking, skipping short transcript '{final_text[:40]}' (likely echo)")
                return
            # Long utterance while bot is speaking = barge-in, allow it
            logger.info(f"[voice] Bot is speaking but got long utterance ({word_count} words) — allowing barge-in")

        # ── AI auto-correction ─────────────────────────────────────────────
        # Fix obvious ASR errors (wrong words, phonetic confusions) using a
        # fast LLM call. Whisper output is already clean English — correcting
        # it costs a full extra LLM round trip for nothing, so skip it.
        if _HAS_CORRECTOR and not from_whisper and len(final_text.strip().split()) >= 2:
            try:
                corrected = await auto_correct_transcript(final_text)
                if corrected and corrected != final_text:
                    logger.info(f"[voice] Auto-corrected: '{final_text}' → '{corrected}'")
                    final_text = corrected
            except Exception as e:
                logger.debug(f"[voice] Auto-correction failed: {e}")

        # Track speech count for proactive engagement
        self._user_speech_count[user_id] = self._user_speech_count.get(user_id, 0) + 1
        self._user_last_speech_time[user_id] = time.time()
        self._user_last_response[user_id] = time.time()

        # ── VC-move vote resolution — if a "can I go?" ask is pending and
        # this utterance is a decisive yes/no, resolve it instead of replying
        if await self._resolve_vc_move_vote(user_id, final_text):
            return

        # ── Turn hints — algorithmic intent detection (farewell / abuse /
        # vc-invite). Sets the prompt directive + TTS prosody for this turn.
        hints = self._build_turn_hints(user_id, final_text)

        # Spoken "leave the vc" — she obeys: brief goodbye, then disconnect.
        # (A farewell like 'i'm leaving' doesn't reach here — the scorer only
        # fires on imperative forms directed at the bot.)
        if hints.get("leave_cmd") and self._leave_cb:
            import random as _r
            try:
                await self._speak(_r.choice([
                    "alright, i'm out — see you lot",
                    "okay okay, i'm gone — laters",
                    "fine, i'll leave you to it — see you",
                    "alright, dipping out — catch you",
                ]))
            except Exception:
                pass
            try:
                await self._leave_cb()
            except Exception:
                pass
            return

        # ── Song request — dedicated path: songwriter lyrics + singing TTS ──
        if hints.get("sing"):
            try:
                if await self._handle_sing(user_id, final_text):
                    return
            except Exception as e:
                logger.error(f"[voice] Sing handler error: {e!r}")
            logger.info("[voice] Sing failed — falling back to normal reply")

        # ── Outsider action worker — spoken action requests ("ping john in
        # general", "react to that") run through the isolated engine. When
        # something was done we steer the reply to acknowledge it aloud —
        # the user hears "done, pinged him" not the engine's "done" text.
        # leave_cmd already returned above (native path keeps it).
        if getattr(self, "_action_cb", None):
            try:
                _act = await self._action_cb(user_id, final_text)
                if _act:
                    kind, note = _act
                    if kind == "info" and note:
                        hints["directive"] = (
                            (hints.get("directive") or "")
                            + f" You looked it up for them: {note} — relay it naturally."
                        ).strip()
                    elif kind == "exec":
                        hints["directive"] = (
                            (hints.get("directive") or "")
                            + " You're about to do what they asked — tell them "
                              "you're on it, brief and natural."
                        ).strip()
            except Exception as e:
                logger.debug(f"[voice] action worker error: {e}")

        # ── TTS pre-warm ─────────────────────────────────────────────────
        # Open the Fish Audio WebSocket NOW, in parallel with the LLM call —
        # the handshake (~0.1-0.6s) overlaps generation so the first sentence
        # can start playing almost as soon as it's ready.
        tts_cfg = self._random_tts_cfg(hints)
        tts = FishAudioTTS(tts_cfg)
        tts_open = asyncio.ensure_future(tts.open())

        async def _close_prewarm():
            try:
                tts_open.cancel()
            except Exception:
                pass
            try:
                await tts.close()
            except Exception:
                pass

        # ── Stale-reply check ────────────────────────────────────────────
        # A newer utterance may have endpointed while the prep above ran —
        # its finalize bumped this user's reply generation. If so, this
        # reply is already outdated: drop it (the deferred finalize will
        # answer their latest words instead).
        gen = self._user_reply_gen.get(user_id, 0)

        async def _drop_if_stale() -> bool:
            if self._user_reply_gen.get(user_id, 0) != gen:
                logger.info(f"[voice] Reply for {user_id} superseded by newer speech — dropping")
                await _close_prewarm()
                return True
            return False

        if await _drop_if_stale():
            return

        # ── Streaming path: LLM deltas → sentences → TTS as they arrive ──
        if self._on_transcript_stream is not None:
            try:
                spoken = await self._speak_stream(
                    self._on_transcript_stream(user_id, final_text, hints),
                    tts, tts_open,
                )
            except Exception as e:
                logger.error(f"[voice] Streaming reply error: {e!r}")
                spoken = False
            if spoken:
                return
            logger.info("[voice] Stream produced nothing — falling back to full reply")
            # The stream path consumed/closed the pre-warmed TTS — open a
            # fresh one for the fallback reply
            try:
                await tts.close()
            except Exception:
                pass
            tts_cfg = self._random_tts_cfg(hints)
            tts = FishAudioTTS(tts_cfg)
            tts_open = asyncio.ensure_future(tts.open())

        # ── Full-response path (fallback, or no stream configured) ────────
        try:
            response = await asyncio.wait_for(
                self._on_transcript(user_id, final_text, hints),
                timeout=30.0,
            )
        except asyncio.TimeoutError:
            logger.warning(f"[voice] LLM callback timed out for user {user_id}")
            await _close_prewarm()
            return
        except Exception as e:
            logger.error(f"LLM callback error: {e!r}")
            await _close_prewarm()
            return

        if not response or not response.strip():
            logger.warning(f"[voice] Empty LLM response for transcript: '{final_text[:50]}'")
            await _close_prewarm()
            return

        # Re-check staleness — the LLM call took seconds; they may have kept
        # talking and their newest utterance should win, not this old reply.
        if await _drop_if_stale():
            return

        logger.info(f"[voice] LLM response: {response[:100]}")
        await self._speak(response, tts=tts, tts_cfg=tts_cfg, tts_open=tts_open)

    def _random_tts_cfg(self, hints: Optional[dict] = None) -> TTSConfig:
        """Build a TTSConfig with per-turn loudness/speed variation.

        Vary the TTS volume slightly for each response to avoid the bot
        sounding exactly the same every time. Sometimes slightly softer
        (feeble), sometimes normal, occasionally a bit louder.
        This makes the voice feel more human and less mechanical.

        hints (from _build_turn_hints) can push the prosody further — an
        angry turn gets louder and faster, like a real person's voice."""
        import random as _rng
        from dataclasses import replace as _dc_replace
        volume_roll = _rng.random()
        if volume_roll < 0.20:
            volume_db, voice_mood = -2.0, "soft"
        elif volume_roll < 0.85:
            volume_db, voice_mood = 0.0, "normal"
        else:
            volume_db, voice_mood = 1.5, "lively"
        # Also vary speed slightly (0.95-1.05x) for natural variation
        speed = _rng.uniform(0.95, 1.05)
        if hints:
            volume_db = max(-20.0, min(20.0, volume_db + hints.get("vol_db", 0.0)))
            speed = max(0.5, min(2.0, speed * hints.get("speed_mult", 1.0)))
            if hints.get("vol_db") or hints.get("speed_mult", 1.0) != 1.0:
                voice_mood = "heated"
        logger.info(f"[voice] Voice variation: {voice_mood} (vol={volume_db}dB, speed={speed:.2f}x)")
        return _dc_replace(self._tts_config, volume_db=volume_db, speed=speed)

    async def _speak(self, text: str, tts=None, tts_cfg=None, tts_open=None) -> None:
        """Convert text to speech and play it in the voice channel.

        tts / tts_cfg / tts_open: optional pre-warmed TTS stream — opened
        concurrently with the LLM call so the handshake is already done."""
        if not self._voice_client or not self._loop:
            logger.warning("[voice] No voice client set, can't speak")
            if tts is not None:
                try:
                    await tts.close()
                except Exception:
                    pass
            return

        # Acquire speak lock to prevent concurrent _speak calls
        async with self._speak_lock:
            self._speak_active = True
            # Add breath effects for more natural speech
            text = _add_breath_effects(text)
            logger.info(f"[voice] TTS with breath effects: '{text[:80]}'")

            if tts_cfg is None:
                tts_cfg = self._random_tts_cfg()

            # Stop comfort noise and any existing playback
            self.stop_comfort_noise()
            if self._voice_client.is_playing():
                logger.info("[voice] Stopping existing playback before speaking")
                self._voice_client.stop()
                await asyncio.sleep(0.2)  # Give player time to clean up

            # Bail if the voice client dropped — don't waste a TTS stream
            if not self._voice_client.is_connected():
                logger.warning("[voice] Voice client disconnected — skipping speak")
                self._speak_active = False
                if tts is not None:
                    try:
                        await tts.close()
                    except Exception:
                        pass
                return

            self._is_speaking = True
            self._speak_deadline = time.time() + 35.0  # Safety deadline

            # Create playback queue
            play_queue: _queue.Queue = _queue.Queue()
            source = OpusAudioSource(play_queue)

            # TTS stream — reuse the pre-warmed one if provided, else open fresh
            if tts is None:
                tts = FishAudioTTS(tts_cfg)
            opened = False
            if tts_open is not None:
                try:
                    await tts_open  # usually already done — opened during the LLM call
                    opened = True
                except Exception as e:
                    logger.warning(f"TTS pre-warm failed, retrying live: {e}")
            if not opened:
                for attempt in range(3):
                    try:
                        await tts.open()
                        opened = True
                        break
                    except Exception as e:
                        logger.warning(f"TTS open failed (attempt {attempt+1}/3): {e}")
                        if attempt < 2:
                            await asyncio.sleep(2.0 * (attempt + 1))  # 2s, 4s backoff
            if not opened:
                logger.error(f"TTS open failed after 3 attempts — giving up")
                self._is_speaking = False
                self._speak_active = False
                return

            # Track packet count for debugging
            packet_count = [0]
            # Event signaled when playback completes or errors
            play_done = asyncio.Event()
            # Capture player thread errors (invisible without `after` callback)
            player_error = [None]

            def _on_playback_done(error):
                if error:
                    player_error[0] = error
                    logger.error(f"[voice] Player thread error: {error!r}")
                play_done.set()

            # Feed text in chunks and collect audio packets
            async def _feed_and_collect():
                demux = OggDemuxer()
                ogg_chunk_count = 0
                try:
                    chunks = chunk_text(text)
                    logger.info(f"[voice] TTS feeding {len(chunks)} chunks for: '{text[:60]}'")
                    for chunk in chunks:
                        if speakable(chunk):
                            await tts.push_text(chunk)
                            await tts.flush()
                    await tts.end_turn()

                    # Collect OGG chunks from TTS, demux into raw Opus frames
                    async for ogg_chunk in tts.packets():
                        if ogg_chunk is not None:
                            ogg_chunk_count += 1
                            # Feed OGG container to demuxer → extract raw Opus packets
                            demux.feed(ogg_chunk)
                            for opus_pkt in demux.packets():
                                packet_count[0] += 1
                                play_queue.put(opus_pkt)

                    # Flush any remaining packets from the demuxer
                    for opus_pkt in demux.flush():
                        packet_count[0] += 1
                        play_queue.put(opus_pkt)

                    logger.info(f"[voice] TTS collected {packet_count[0]} Opus frames from {ogg_chunk_count} OGG chunks")
                    # Signal end
                    play_queue.put(None)
                except Exception as e:
                    logger.error(f"TTS feed/collect error: {e!r}")
                    play_queue.put(None)
                finally:
                    await tts.close()

            # Start feeding TTS in background
            feed_task = asyncio.create_task(_feed_and_collect())

            # Play audio in Discord with `after` callback to surface player errors
            import time as _time
            play_start = _time.time()
            MAX_PLAY_DURATION = 45.0  # 45s max — room for a sung verse+hook
            try:
                logger.info(f"[voice] Calling play() (is_connected={self._voice_client.is_connected()})")
                self._voice_client.play(source, after=_on_playback_done)
                logger.info("[voice] play() succeeded, waiting for playback...")
                # Poll for completion with timeout (don't block forever)
                while self._voice_client.is_playing() and not play_done.is_set():
                    await asyncio.sleep(0.25)
                    elapsed = _time.time() - play_start
                    if elapsed > MAX_PLAY_DURATION:
                        logger.warning(f"[voice] Playback timed out after {elapsed:.1f}s (collected {packet_count[0]} packets), forcing stop")
                        self._voice_client.stop()
                        break
                logger.info(f"[voice] Playback finished (collected {packet_count[0]} packets, elapsed={_time.time()-play_start:.1f}s)")
            except discord.ClientException as e:
                logger.error(f"[voice] Play() client error: {e}")
            except Exception as e:
                logger.error(f"[voice] Voice playback error: {e!r}")
            finally:
                # Always stop player and clean up
                if self._voice_client and self._voice_client.is_playing():
                    self._voice_client.stop()
                self._is_speaking = False
                self._speak_active = False
                # Don't block forever on feed_task — cancel if it's stuck
                try:
                    await asyncio.wait_for(feed_task, timeout=5.0)
                except asyncio.TimeoutError:
                    logger.warning("[voice] Feed task didn't complete in 5s, cancelling")
                    feed_task.cancel()
                except Exception:
                    pass
                # Restart comfort noise to keep voice indicator active
                await asyncio.sleep(0.3)
                self.start_comfort_noise()
                self._last_bot_speech_time = time.time()

    async def _speak_stream(self, text_stream, tts, tts_open) -> bool:
        """Speak a reply while the LLM is still generating it.

        Drains an async iterator of text deltas, cuts complete sentences, and
        pushes each to the (pre-warmed) TTS stream immediately — the first
        sentence starts playing while later ones are still generating.

        Returns True if audio was actually produced; False means the stream
        yielded nothing usable and the caller should take the fallback path.
        """
        if not self._voice_client or not self._loop:
            return False

        async with self._speak_lock:
            self._speak_active = True
            try:
                # Stop comfort noise / existing playback upfront so the first
                # sentence can start the moment it's generated
                self.stop_comfort_noise()
                if self._voice_client.is_playing():
                    self._voice_client.stop()
                    await asyncio.sleep(0.2)

                if not self._voice_client.is_connected():
                    logger.warning("[voice] Voice client disconnected — skipping speak")
                    try:
                        await tts.close()
                    except Exception:
                        pass
                    return False

                self._is_speaking = True
                self._speak_deadline = time.time() + 35.0

                # Wait for the pre-warmed TTS stream (opened during LLM prep)
                opened = False
                try:
                    await tts_open
                    opened = True
                except Exception as e:
                    logger.warning(f"[voice] TTS pre-warm failed, opening live: {e}")
                if not opened:
                    try:
                        await tts.open()
                        opened = True
                    except Exception as e:
                        logger.warning(f"[voice] TTS open failed (stream path): {e}")
                        return False

                play_queue: _queue.Queue = _queue.Queue()
                source = OpusAudioSource(play_queue)
                play_done = asyncio.Event()
                packet_count = [0]
                full_text = [""]
                produced = [False]
                player_error = [None]

                def _on_playback_done(error):
                    if error:
                        player_error[0] = error
                        logger.error(f"[voice] Player thread error: {error!r}")
                    play_done.set()

                import re as _re
                from ..ai.reply import clean_for_speech
                boundary = _re.compile(r'(?<=[.!?;])\s+|\n+')

                async def _push(text: str):
                    text = clean_for_speech(text).strip()
                    if not speakable(text):
                        return
                    await tts.push_text(_add_breath_effects(text))
                    await tts.flush()
                    produced[0] = True
                    full_text[0] += text + " "

                async def _feed():
                    demux = OggDemuxer()
                    buf = ""
                    try:
                        aiter = text_stream.__aiter__()
                        while True:
                            try:
                                piece = await asyncio.wait_for(aiter.__anext__(), timeout=20.0)
                            except StopAsyncIteration:
                                break
                            except asyncio.TimeoutError:
                                logger.warning("[voice] LLM stream stalled 20s — cutting off")
                                break
                            buf += piece
                            # Cut and push every complete sentence boundary
                            while True:
                                m = boundary.search(buf)
                                if not m:
                                    break
                                sent, buf = buf[:m.start()], buf[m.end():]
                                await _push(sent)
                        # Tail: leftover fragment without a boundary — keep it
                        # if it's substantial (mirrors the dangling-fragment rule)
                        tail = buf.strip()
                        if tail and len(tail.split()) >= 2:
                            await _push(tail)
                        if produced[0]:
                            await tts.end_turn()
                            # Collect OGG chunks → demux → raw Opus frames
                            async for ogg_chunk in tts.packets():
                                if ogg_chunk is not None:
                                    demux.feed(ogg_chunk)
                                    for opus_pkt in demux.packets():
                                        packet_count[0] += 1
                                        play_queue.put(opus_pkt)
                            for opus_pkt in demux.flush():
                                packet_count[0] += 1
                                play_queue.put(opus_pkt)
                            logger.info(
                                f"[voice] Streamed reply '{full_text[0].strip()[:80]}' "
                                f"({packet_count[0]} frames)"
                            )
                    except Exception as e:
                        logger.error(f"[voice] Stream feed error: {e!r}")
                    finally:
                        play_queue.put(None)
                        try:
                            await tts.close()
                        except Exception:
                            pass

                feed_task = asyncio.create_task(_feed())

                import time as _time
                play_start = _time.time()
                MAX_PLAY_DURATION = 45.0
                try:
                    self._voice_client.play(source, after=_on_playback_done)
                    while self._voice_client.is_playing() and not play_done.is_set():
                        await asyncio.sleep(0.25)
                        if _time.time() - play_start > MAX_PLAY_DURATION:
                            logger.warning(f"[voice] Playback timed out, forcing stop")
                            self._voice_client.stop()
                            break
                except discord.ClientException as e:
                    logger.error(f"[voice] Play() client error: {e}")
                except Exception as e:
                    logger.error(f"[voice] Voice playback error: {e!r}")
                finally:
                    if self._voice_client and self._voice_client.is_playing():
                        self._voice_client.stop()
                    self._is_speaking = False
                    try:
                        await asyncio.wait_for(feed_task, timeout=5.0)
                    except asyncio.TimeoutError:
                        feed_task.cancel()
                    except Exception:
                        pass
                    await asyncio.sleep(0.3)
                    self.start_comfort_noise()
                    self._last_bot_speech_time = time.time()

                return produced[0]
            finally:
                self._speak_active = False

    def stop_speaking(self) -> None:
        """Interrupt current speech (barge-in)."""
        if self._voice_client and self._voice_client.is_playing():
            self._voice_client.stop()
        self._is_speaking = False
        # Restart comfort noise after a brief pause (called from barge-in)
        if self._loop and self._loop.is_running():
            self._loop.call_later(0.3, self.start_comfort_noise)

    def get_active_users(self) -> list:
        """Get list of (user_id, display_name, last_speech_time, speech_count)
        for users who have spoken in this VC session. Used by the proactive
        engagement system to pick users to ask questions to."""
        result = []
        for uid, count in self._user_speech_count.items():
            if count == 0:
                continue
            name = self._user_display_names.get(uid, str(uid))
            last_speech = self._user_last_speech_time.get(uid, 0)
            result.append((uid, name, last_speech, count))
        return result

    def get_quietest_user(self, min_silence_s: float = 30) -> Optional[tuple]:
        """Get the user who has been quiet the longest (hasn't spoken recently).
        Returns (user_id, display_name) or None if no users have spoken yet."""
        now = time.time()
        candidates = []
        for uid in self._user_speech_count:
            last = self._user_last_speech_time.get(uid, 0)
            silence = now - last if last > 0 else float('inf')
            if silence >= min_silence_s:
                name = self._user_display_names.get(uid, str(uid))
                candidates.append((silence, uid, name))
        if not candidates:
            return None
        # Return the one who's been quiet the longest
        candidates.sort(key=lambda x: x[0], reverse=True)
        return (candidates[0][1], candidates[0][2])

    def get_last_speech_time(self) -> float:
        """Get the timestamp of the most recent speech (by anyone or the bot)."""
        user_times = [t for t in self._user_last_speech_time.values() if t > 0]
        bot_time = self._last_bot_speech_time
        all_times = user_times + [bot_time]
        if not all_times:
            return 0
        return max(all_times)

    def set_user_display_name(self, user_id: int, name: str) -> None:
        """Set the display name for a user (called by the manager)."""
        self._user_display_names[user_id] = name

    @property
    def is_speaking(self) -> bool:
        return self._is_speaking

    async def cleanup_user(self, user_id: int) -> None:
        """Clean up resources for a user who left the channel."""
        self._user_vads.pop(user_id, None)
        asr = self._user_asrs.pop(user_id, None)
        if asr:
            await asr.close()
        self._user_speaking.pop(user_id, None)
        self._user_prerolls.pop(user_id, None)
        self._user_in_speech.pop(user_id, None)
        self._user_last_voice_time.pop(user_id, None)
        self._user_utterance_pcm.pop(user_id, None)
        self._user_decoders.pop(user_id, None)
        self._user_resamplers.pop(user_id, None)
        self._user_agc_gain.pop(user_id, None)
        self._user_rms_ema.pop(user_id, None)
        self._user_last_response.pop(user_id, None)
        self._user_utt_start.pop(user_id, None)
        self._user_farewell_at.pop(user_id, None)
        # Without this their stale timestamp keeps the VC "active" forever —
        # the silence-leave would never fire after they left for real.
        self._user_last_speech_time.pop(user_id, None)
        self._user_last_transcript.pop(user_id, None)
        self._user_ack_at.pop(user_id, None)
        self._user_reply_gen.pop(user_id, None)
        self._user_finalize_queued.discard(user_id)
        self._irritation.clear(user_id)
        self._finalizing.discard(user_id)
        self._user_pkt_queues.pop(user_id, None)
        pkt_task = self._user_pkt_workers.pop(user_id, None)
        if pkt_task and not pkt_task.done():
            pkt_task.cancel()
        task = self._user_watchdog_tasks.pop(user_id, None)
        if task is not None:
            task.cancel()

    async def cleanup(self) -> None:
        """Clean up all resources."""
        self.stop_comfort_noise()
        # Stop hmm sound player
        if self._hmm_player is not None:
            self._hmm_player.stop()
            self._hmm_player = None
        for uid in list(self._user_asrs.keys()):
            await self.cleanup_user(uid)
        self._user_vads.clear()
        self._user_speaking.clear()
        self._user_prerolls.clear()
        self._user_in_speech.clear()
        self._user_last_voice_time.clear()
        self._user_utterance_pcm.clear()
        self._user_decoders.clear()
        self._user_resamplers.clear()
        self._user_agc_gain.clear()
        self._user_rms_ema.clear()
        self._user_last_response.clear()
        self._user_utt_start.clear()
        self._user_farewell_at.clear()
        self._user_last_transcript.clear()
        self._user_ack_at.clear()
        self._user_reply_gen.clear()
        self._user_finalize_queued.clear()
        self._irritation.clear_all()
        self._pending_vc_move = None
        self._finalizing.clear()
        self._user_pkt_queues.clear()
        for task in self._user_pkt_workers.values():
            if not task.done():
                task.cancel()
        self._user_pkt_workers.clear()
        for task in self._user_watchdog_tasks.values():
            task.cancel()
        self._user_watchdog_tasks.clear()
        if self._dsp_pool is not None:
            self._dsp_pool.shutdown(wait=False)
            self._dsp_pool = None
        self._is_speaking = False
        self._speak_active = False
