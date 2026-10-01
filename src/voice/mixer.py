"""Ambient background mixer for Discord voice channels.

Mixes a subtle looping ambient bed with TTS speech, ducking the ambient
when speech plays. Makes the bot sound like it's in a real room rather
than a silent void — giving a more human-like voice experience.

Based on the Hermes-agent VoiceMixer design:
- VoiceMixer is a discord.AudioSource that runs continuously
- Sums ambient loop + speech PCM frames (numpy int16 add + clip)
- Ducks ambient gain when speech is active, ramps back up when done
- synth_ambient_pcm() generates a subtle pad without needing audio files
"""
from __future__ import annotations

import threading
from typing import List, Optional

import discord
import numpy as np
from loguru import logger

# Discord-native frame geometry (48kHz, stereo, 20ms)
SAMPLE_RATE = 48000
CHANNELS = 2
SAMPLE_WIDTH = 2  # bytes per sample (s16)
FRAME_LENGTH_MS = 20
SAMPLES_PER_FRAME = SAMPLE_RATE * FRAME_LENGTH_MS // 1000  # 960
FRAME_SIZE = SAMPLES_PER_FRAME * CHANNELS * SAMPLE_WIDTH  # 3840 bytes
SILENCE_FRAME = b"\x00" * FRAME_SIZE


class MixerChild:
    """A single audio stream feeding into VoiceMixer."""

    __slots__ = (
        "name", "_pcm", "_pos", "loop", "gain",
        "is_speech", "fade_frames", "_fade_done", "_finished",
    )

    def __init__(
        self,
        name: str,
        pcm: bytes,
        *,
        loop: bool = False,
        gain: float = 1.0,
        is_speech: bool = False,
        fade_in_ms: int = 0,
    ):
        # Pad to whole frame so looping is seamless
        remainder = len(pcm) % FRAME_SIZE
        if remainder:
            pcm = pcm + b"\x00" * (FRAME_SIZE - remainder)
        self.name = name
        self._pcm = pcm
        self._pos = 0
        self.loop = loop
        self.gain = float(gain)
        self.is_speech = is_speech
        self.fade_frames = max(0, fade_in_ms // FRAME_LENGTH_MS)
        self._fade_done = 0
        self._finished = False

    @property
    def finished(self) -> bool:
        return self._finished

    def read_frame(self) -> Optional[np.ndarray]:
        """Return next 20ms frame as int16 ndarray, or None if done."""
        if self._finished:
            return None
        if self._pos >= len(self._pcm):
            if self.loop and self._pcm:
                self._pos = 0
            else:
                self._finished = True
                return None

        chunk = self._pcm[self._pos:self._pos + FRAME_SIZE]
        self._pos += FRAME_SIZE
        if len(chunk) < FRAME_SIZE:
            chunk = chunk + b"\x00" * (FRAME_SIZE - len(chunk))

        samples = np.frombuffer(chunk, dtype=np.int16).astype(np.float32)

        gain = self.gain
        if self.fade_frames and self._fade_done < self.fade_frames:
            self._fade_done += 1
            gain *= self._fade_done / self.fade_frames

        if gain != 1.0:
            samples = samples * gain
        return samples


class VoiceMixer(discord.AudioSource):
    """Continuous PCM mixer — ambient bed + speech with automatic ducking.

    Install once per guild on VC join via vc.play(mixer).
    Use set_ambient() for the idle loop, play_speech() for TTS replies.
    """

    def is_opus(self) -> bool:
        return False  # PCM mode — discord.py will encode

    def __init__(
        self,
        *,
        ambient_gain: float = 0.15,
        duck_gain: float = 0.05,
        speech_gain: float = 1.0,
        duck_release_ms: int = 400,
    ):
        self._lock = threading.Lock()
        self._ambient: Optional[MixerChild] = None
        self._speech: List[MixerChild] = []
        self._ambient_gain = float(ambient_gain)
        self._duck_gain = float(duck_gain)
        self._speech_gain = float(speech_gain)
        self._duck_release_frames = max(1, duck_release_ms // FRAME_LENGTH_MS)
        self._duck_release_left = 0
        self._closed = False
        self._speech_active = False

    def set_ambient(self, pcm: Optional[bytes], *, gain: Optional[float] = None) -> None:
        """Install or clear the looping ambient bed."""
        with self._lock:
            if gain is not None:
                self._ambient_gain = float(gain)
            if not pcm:
                self._ambient = None
                return
            self._ambient = MixerChild(
                "ambient", pcm, loop=True,
                gain=self._effective_ambient_gain(), fade_in_ms=300,
            )

    def _effective_ambient_gain(self) -> float:
        return self._duck_gain if self._speech_active else self._ambient_gain

    def play_speech(self, pcm: bytes, *, gain: Optional[float] = None,
                    fade_in_ms: int = 40) -> None:
        """Layer a one-shot speech clip over the ambient bed."""
        if not pcm:
            return
        with self._lock:
            child = MixerChild(
                "speech", pcm, loop=False,
                gain=self._speech_gain if gain is None else float(gain),
                is_speech=True, fade_in_ms=fade_in_ms,
            )
            self._speech.append(child)
            self._speech_active = True
            self._duck_release_left = 0
            if self._ambient is not None:
                self._ambient.gain = self._duck_gain

    @property
    def speech_active(self) -> bool:
        with self._lock:
            return self._speech_active

    def stop_speech(self) -> None:
        """Drop in-flight speech and release duck."""
        with self._lock:
            self._speech.clear()
            self._begin_duck_release_locked()

    def _begin_duck_release_locked(self) -> None:
        self._speech_active = False
        self._duck_release_left = self._duck_release_frames

    def read(self) -> bytes:
        """Return one 20ms mixed PCM frame (always FRAME_SIZE bytes)."""
        with self._lock:
            if self._closed:
                return SILENCE_FRAME

            acc: Optional[np.ndarray] = None

            # Speech children
            if self._speech:
                still_live: List[MixerChild] = []
                for child in self._speech:
                    frame = child.read_frame()
                    if frame is None:
                        continue
                    acc = frame if acc is None else acc + frame
                    still_live.append(child)
                self._speech = still_live
                if not self._speech and self._speech_active:
                    self._begin_duck_release_locked()

            # Ambient bed with duck-release ramp
            if self._ambient is not None:
                if self._duck_release_left > 0 and not self._speech_active:
                    self._duck_release_left -= 1
                    frac = 1.0 - (self._duck_release_left / self._duck_release_frames)
                    self._ambient.gain = (
                        self._duck_gain
                        + (self._ambient_gain - self._duck_gain) * frac
                    )
                elif not self._speech_active and self._duck_release_left == 0:
                    self._ambient.gain = self._ambient_gain
                amb = self._ambient.read_frame()
                if amb is not None:
                    acc = amb if acc is None else acc + amb

            if acc is None:
                return SILENCE_FRAME

            np.clip(acc, -32768, 32767, out=acc)
            return acc.astype(np.int16).tobytes()

    def cleanup(self) -> None:
        with self._lock:
            self._closed = True
            self._ambient = None
            self._speech.clear()


def synth_ambient_pcm(seconds: float = 4.0) -> bytes:
    """Synthesise a subtle looping ambient bed (no asset file required).

    A soft, slowly-pulsing low pad: two detuned sine partials with a gentle
    tremolo, plus a touch of filtered noise. Designed to loop seamlessly
    (whole number of cycles, zero-crossing endpoints) and sit quietly under
    speech. Mono content duplicated to stereo.
    """
    n = int(SAMPLE_RATE * seconds)
    t = np.arange(n, dtype=np.float64) / SAMPLE_RATE

    def _whole_cycle_freq(target: float) -> float:
        cycles = max(1, round(target * seconds))
        return cycles / seconds

    f1 = _whole_cycle_freq(110.0)
    f2 = _whole_cycle_freq(110.5)
    trem = _whole_cycle_freq(0.5)

    pad = (
        0.55 * np.sin(2 * np.pi * f1 * t)
        + 0.45 * np.sin(2 * np.pi * f2 * t)
    )
    tremolo = 0.6 + 0.4 * (0.5 * (1 + np.sin(2 * np.pi * trem * t)))
    signal = pad * tremolo

    # Smooth filtered noise for air
    rng = np.random.default_rng(7)
    noise = rng.standard_normal(n)
    kernel = np.ones(64) / 64.0
    noise = np.convolve(noise, kernel, mode="same")
    signal = signal + 0.08 * noise

    # Normalise to modest peak
    peak = float(np.max(np.abs(signal))) or 1.0
    signal = (signal / peak) * 0.5

    mono16 = (signal * 32767.0).astype(np.int16)
    stereo16 = np.repeat(mono16[:, None], CHANNELS, axis=1).reshape(-1)
    logger.info(f"[mixer] Synthesized {seconds:.1f}s ambient bed ({len(stereo16.tobytes())} bytes)")
    return stereo16.tobytes()
