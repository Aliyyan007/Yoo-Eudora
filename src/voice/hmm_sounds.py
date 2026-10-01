"""Algorithmic whispering/hmm sounds — makes the bot feel alive during silence.

Instead of just static comfort noise, the bot occasionally produces soft
"thinking" sounds like "hm", "hmm", "mhm", "huh" — like a real person
listening and acknowledging.

The sounds are ALGORITHMIC — not played on every silence. The algorithm:
1. Random intervals between sounds (15-50s of silence before next sound)
2. Random selection from multiple sound variations
3. Volume varies slightly each time (feeble/soft, not loud)
4. Probability-based: not every silence period gets a sound (60% chance)
5. Won't play if the bot just spoke (30s cooldown after bot speech)
6. Won't play if someone is currently talking

This creates a natural, unpredictable pattern that feels human — not
mechanical like a fixed interval would.
"""
from __future__ import annotations

import asyncio
import random
import time
from typing import Optional, TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from .pipeline import VoicePipeline


# ── Algorithmic hmm sound settings ──────────────────────────────────────────
_MIN_SILENCE_BEFORE_HMM_S = 15    # Need at least 15s of silence before first hmm
_MAX_SILENCE_BEFORE_HMM_S = 50    # But no more than 50s (random in this range)
_HMM_PROBABILITY = 0.60           # 60% chance to play hmm when silence threshold hit
_COOLDOWN_AFTER_BOT_SPEECH_S = 30 # Don't hmm for 30s after bot spoke
_COOLDOWN_AFTER_HMM_S = 20        # Min 20s between hmm sounds
_CHECK_INTERVAL_S = 5             # Check every 5s

# The soft "thinking" sounds the bot can make
# These are short, soft utterances that sound like someone listening
_HMM_SOUNDS = [
    "hm",
    "hmm",
    "mhm",
    "huh",
    "mm",
    "mmm",
    "hm hm",
    "yeah",
    "mhm mhm",
    "hmm...",
    "mm-hmm",
    "huh, interesting",
    "hm, right",
    "oh, okay",
    "mm, yeah",
    "hmm, I see",
    "right, right",
    "yeah, yeah",
    "mhm, go on",
    "hmm, makes sense",
]

# Cache of pre-generated TTS audio for hmm sounds
# Key: sound text, Value: list of Opus packets
_hmm_cache: dict[str, list[bytes]] = {}
_hmm_cache_loaded = False


async def _preload_hmm_sounds(pipeline: "VoicePipeline"):
    """Pre-generate TTS audio for all hmm sounds and cache them.
    This avoids latency when playing them during silence."""
    global _hmm_cache_loaded

    if _hmm_cache_loaded:
        return

    from .tts import FishAudioTTS, TTSConfig
    from .ogg_demux import OggDemuxer

    tts_config = pipeline._tts_config
    if not tts_config:
        logger.warning("[hmm] No TTS config, can't preload hmm sounds")
        return

    logger.info(f"[hmm] Pre-generating {len(_HMM_SOUNDS)} hmm sounds...")

    for sound_text in _HMM_SOUNDS:
        try:
            tts = FishAudioTTS(tts_config)
            await tts.open()

            # Feed the sound text
            await tts.push_text(sound_text)
            await tts.flush()
            await tts.end_turn()

            # Collect OGG chunks and demux into Opus packets
            demux = OggDemuxer()
            opus_packets = []
            async for ogg_chunk in tts.packets():
                if ogg_chunk is not None:
                    demux.feed(ogg_chunk)
                    for opus_pkt in demux.packets():
                        opus_packets.append(opus_pkt)

            # Flush remaining
            for opus_pkt in demux.flush():
                opus_packets.append(opus_pkt)

            await tts.close()

            if opus_packets:
                _hmm_cache[sound_text] = opus_packets

        except Exception as e:
            logger.debug(f"[hmm] Failed to preload '{sound_text}': {e}")

    _hmm_cache_loaded = True
    logger.info(f"[hmm] Pre-generated {len(_hmm_cache)}/{len(_HMM_SOUNDS)} hmm sounds")


class HmmSoundPlayer:
    """Plays algorithmic 'hmm' thinking sounds during silence.

    Runs as a background task, checking for silence periods and playing
    soft thinking sounds at random intervals to make the bot feel alive.
    """

    def __init__(self, pipeline: "VoicePipeline"):
        self._pipeline = pipeline
        self._task: Optional[asyncio.Task] = None
        self._last_hmm_time: float = 0.0
        self._next_hmm_threshold: float = _MIN_SILENCE_BEFORE_HMM_S
        self._running = False

    def start(self):
        """Start the hmm sound player background loop."""
        if self._task and not self._task.done():
            return
        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info("[hmm] Hmm sound player started")

    def stop(self):
        """Stop the hmm sound player."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
        self._task = None
        logger.info("[hmm] Hmm sound player stopped")

    async def _loop(self):
        """Main loop — check for silence and play hmm sounds."""
        # Preload hmm sounds on first run
        try:
            await _preload_hmm_sounds(self._pipeline)
        except Exception as e:
            logger.warning(f"[hmm] Preload failed: {e}")

        while self._running:
            try:
                await asyncio.sleep(_CHECK_INTERVAL_S)

                if not self._running:
                    break

                # Don't play if bot is currently speaking
                if self._pipeline.is_speaking:
                    continue

                # Don't play if comfort noise isn't active (bot not in VC)
                if not self._pipeline._comfort_noise_active:
                    continue

                now = time.time()

                # Cooldown after bot speech
                if now - self._pipeline._last_bot_speech_time < _COOLDOWN_AFTER_BOT_SPEECH_S:
                    continue

                # Cooldown after last hmm
                if now - self._last_hmm_time < _COOLDOWN_AFTER_HMM_S:
                    continue

                # Check silence duration
                last_speech = self._pipeline.get_last_speech_time()
                if last_speech == 0:
                    continue

                silence_duration = now - last_speech
                if silence_duration < self._next_hmm_threshold:
                    continue

                # Check if anyone's mic is active
                for uid, active in self._pipeline._user_mic_active.items():
                    if active:
                        continue

                # Probability check — not every silence gets a hmm
                if random.random() > _HMM_PROBABILITY:
                    # Skip this time, but set a new threshold for next check
                    self._next_hmm_threshold = silence_duration + random.uniform(
                        _MIN_SILENCE_BEFORE_HMM_S, _MAX_SILENCE_BEFORE_HMM_S
                    )
                    continue

                # Play a random hmm sound
                await self._play_random_hmm()

                # Set next threshold (random interval for algorithmic feel)
                self._last_hmm_time = time.time()
                self._next_hmm_threshold = random.uniform(
                    _MIN_SILENCE_BEFORE_HMM_S, _MAX_SILENCE_BEFORE_HMM_S
                )

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"[hmm] Loop error: {e}")

    async def _play_random_hmm(self):
        """Play a random hmm sound from the cache."""
        if not _hmm_cache:
            return

        # Pick a random sound
        sound_text = random.choice(list(_hmm_cache.keys()))
        opus_packets = _hmm_cache.get(sound_text, [])

        if not opus_packets:
            return

        logger.info(f"[hmm] Playing thinking sound: '{sound_text}' ({len(opus_packets)} packets)")

        # A TTS speak is in-flight (incl. TTS-open window) — don't steal the player
        if self._pipeline._is_speaking or self._pipeline._speak_active:
            return

        # Stop comfort noise briefly
        self._pipeline.stop_comfort_noise()

        try:
            # Play the hmm sound using the voice client
            vc = self._pipeline._voice_client
            loop = self._pipeline._loop
            if not vc or not loop:
                return

            if vc.is_playing():
                vc.stop()
                await asyncio.sleep(0.1)

            # Create a source from the cached Opus packets
            from .audio_source import CachedOpusSource
            source = CachedOpusSource(opus_packets, volume=0.5)  # 50% volume — soft/feeble

            # Play it
            play_done = asyncio.Event()

            def _on_done(error):
                if error:
                    logger.debug(f"[hmm] Playback error: {error}")
                play_done.set()

            vc.play(source, after=_on_done)

            # Wait for playback to finish (max 5s)
            try:
                await asyncio.wait_for(play_done.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                if vc.is_playing():
                    vc.stop()

        except Exception as e:
            logger.debug(f"[hmm] Playback failed: {e}")
        finally:
            # Restart comfort noise
            await asyncio.sleep(0.3)
            self._pipeline.start_comfort_noise()
