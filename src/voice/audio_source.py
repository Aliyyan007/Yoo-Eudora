"""Bridge: TTS Opus packet queue → discord.py AudioSource for voice send.

With is_opus()=True, discord.py sends each read() return value as a
complete Opus packet directly into RTP — no PCM↔Opus conversion.

Also provides a ComfortNoiseSource that sends low-level background noise
to keep the Discord voice indicator active (the green ring) — making the
bot look like a real human in a call, not a silent bot.
"""
from __future__ import annotations

import queue as _queue
import random
import struct
from typing import Optional

import discord
from loguru import logger

# 3-byte Opus silence frame (CELT 20ms mono)
SILENCE_OPUS: bytes = b"\xf8\xff\xfe"
_READ_BUDGET_S = 0.015  # Must be < 20ms to keep player loop in cadence


class OpusAudioSource(discord.AudioSource):
    """Feed Opus packets from a sync queue to discord.py's player thread."""

    def __init__(self, frame_queue: "_queue.Queue[Optional[bytes]]") -> None:
        self._queue = frame_queue
        self._eof = False
        self._read_count = 0
        self._silence_count = 0
        self._packet_count = 0

    def is_opus(self) -> bool:
        return True

    def read(self) -> bytes:
        self._read_count += 1
        if self._eof:
            return b""
        try:
            item = self._queue.get(timeout=_READ_BUDGET_S)
        except _queue.Empty:
            self._silence_count += 1
            # Log if we've been sending silence for a while
            if self._silence_count == 50:
                logger.warning(f"[audio_source] 50 silence frames sent (no TTS audio yet)")
            elif self._silence_count == 200:
                logger.warning(f"[audio_source] 200 silence frames (1s) — TTS may be stalled")
            elif self._silence_count == 1000:
                logger.error(f"[audio_source] 1000 silence frames (5s) — TTS definitely stalled")
            return SILENCE_OPUS

        if item is None:
            self._eof = True
            logger.info(f"[audio_source] EOF after {self._read_count} reads, {self._packet_count} packets, {self._silence_count} silence")
            return b""
        self._packet_count += 1
        if self._packet_count <= 3:
            logger.info(f"[audio_source] Sending packet #{self._packet_count}: {len(item)} bytes")
        return item

    def cleanup(self) -> None:
        self._eof = True


class ComfortNoiseSource(discord.AudioSource):
    """Sends low-level comfort noise to keep the Discord voice indicator active.

    Discord shows a green ring around a user when they're "speaking" — which
    means sending audio packets. A real human in a call always has some background
    noise (keyboard, breathing, room ambience) that keeps the indicator flickering.

    This source generates very quiet PCM noise (48kHz stereo) and encodes it with
    the Opus encoder. The noise is so quiet it's barely audible but enough to
    keep the speaking indicator active — making the bot look like a real person
    sitting in the call, not a silent bot.

    Usage:
        vc.play(ComfortNoiseSource(), after=lambda e: ...)
    """

    def __init__(self, noise_level: int = 150) -> None:
        """
        Args:
            noise_level: Peak amplitude of the noise (0-32767).
                150 is very quiet — barely audible but keeps the indicator active.
                300-500 is more noticeable. 100 is almost silent.
        """
        self._noise_level = noise_level
        self._encoder = None
        self._read_count = 0
        self._stopped = False
        self._init_encoder()

    def _init_encoder(self) -> None:
        """Initialize the Opus encoder for comfort noise."""
        try:
            import opuslib
            self._encoder = opuslib.Encoder(48000, 2, "voip")
            # Low bitrate for noise — we don't need quality
            self._encoder.bitrate = 8000
        except ImportError:
            try:
                import opuslib_next
                self._encoder = opuslib_next.Encoder(48000, 2, "voip")
                self._encoder.bitrate = 8000
            except ImportError:
                logger.warning("[comfort_noise] No Opus encoder available — using silence frames")
                self._encoder = None

        # Pre-encode a pool of noise frames once — generating noise + opus
        # encoding per 20ms read() call is the biggest idle CPU cost on weak
        # hosts. A pool of random frames sounds identical.
        self._frame_pool: list[bytes] = []
        if self._encoder is not None:
            for _ in range(120):
                try:
                    n_samples = 960  # 20ms at 48kHz
                    samples = [
                        random.randint(-self._noise_level, self._noise_level)
                        for _ in range(n_samples * 2)
                    ]
                    pcm = struct.pack(f'<{len(samples)}h', *samples)
                    self._frame_pool.append(self._encoder.encode(pcm, n_samples))
                except Exception:
                    break
            if self._frame_pool:
                logger.info(f"[comfort_noise] Pre-encoded {len(self._frame_pool)} noise frames")

    def is_opus(self) -> bool:
        return True

    def read(self) -> bytes:
        if self._stopped:
            return b""

        self._read_count += 1

        if not self._frame_pool:
            # Fallback: send silence frames (won't keep indicator active
            # but keeps the connection alive)
            return SILENCE_OPUS

        return random.choice(self._frame_pool)

    def stop(self) -> None:
        self._stopped = True

    def cleanup(self) -> None:
        self._stopped = True


class CachedOpusSource(discord.AudioSource):
    """Plays pre-generated Opus packets from a list (e.g., cached hmm sounds).

    Unlike OpusAudioSource (which reads from a queue), this reads from a
    fixed list of Opus packets. Supports volume adjustment for softer
    playback (e.g., feeble/whispered hmm sounds).
    """

    def __init__(self, opus_packets: list[bytes], volume: float = 1.0) -> None:
        """
        Args:
            opus_packets: List of Opus-encoded audio packets
            volume: Volume multiplier (0.0-1.0). 1.0 = original, 0.5 = half volume.
                Note: volume adjustment requires decoding/re-encoding, which
                adds slight CPU overhead. For Opus packets, we decode to PCM,
                scale, and re-encode.
        """
        self._packets = list(opus_packets)
        self._index = 0
        self._volume = volume
        self._decoder = None
        self._encoder = None
        self._read_count = 0

        if volume < 1.0:
            self._init_codec()

    def _init_codec(self) -> None:
        """Initialize Opus decoder and encoder for volume adjustment."""
        try:
            import opuslib_next
            self._decoder = opuslib_next.Decoder(48000, 2)
            self._encoder = opuslib_next.Encoder(48000, 2, "voip")
            self._encoder.bitrate = 24000  # decent quality for speech
        except ImportError:
            try:
                import opuslib
                self._decoder = opuslib.Decoder(48000, 2)
                self._encoder = opuslib.Encoder(48000, 2, "voip")
                self._encoder.bitrate = 24000
            except ImportError:
                logger.warning("[cached_source] No Opus codec — can't adjust volume")
                self._decoder = None
                self._encoder = None

    def is_opus(self) -> bool:
        return True

    def read(self) -> bytes:
        if self._index >= len(self._packets):
            return b""  # EOF

        pkt = self._packets[self._index]
        self._index += 1
        self._read_count += 1

        if self._volume >= 1.0 or self._decoder is None:
            # No volume adjustment needed — return original packet
            return pkt

        try:
            # Decode Opus → PCM, scale volume, re-encode → Opus
            pcm = self._decoder.decode(pkt, 960)
            # Scale PCM samples by volume
            import struct
            n_samples = len(pcm) // 2
            samples = struct.unpack(f'<{n_samples}h', pcm)
            scaled = [int(s * self._volume) for s in samples]
            scaled_pcm = struct.pack(f'<{n_samples}h', *scaled)
            # Re-encode
            return self._encoder.encode(scaled_pcm, 960)
        except Exception as e:
            logger.debug(f"[cached_source] Volume adjust failed: {e}")
            return pkt

    def cleanup(self) -> None:
        self._packets.clear()
        self._index = 0
