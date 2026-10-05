"""Simple 48kHz → 16kHz resampler using soxr if available, else decimation."""
from __future__ import annotations

import struct
from typing import Optional


class Resampler:
    """Resample 48kHz mono PCM to 16kHz mono PCM (or vice versa)."""

    def __init__(self, src_rate: int = 48000, dst_rate: int = 16000):
        self._src = src_rate
        self._dst = dst_rate
        self._ratio = src_rate / dst_rate
        self._buf = bytearray()

        # Try to use soxr for quality
        try:
            import soxr
            self._soxr = soxr
            self._resampler = soxr.ResampleStream(
                src_rate, dst_rate, 1, dtype="int16"
            )
        except ImportError:
            self._soxr = None
            self._resampler = None

    def feed(self, pcm: bytes) -> bytes:
        """Feed input PCM, get resampled output."""
        if not pcm:
            return b""

        if self._resampler is not None:
            # Use soxr for high-quality resampling
            try:
                return self._resampler.resample_chunk(pcm)
            except Exception:
                pass

        # Fallback: simple decimation/interpolation
        if self._src == 48000 and self._dst == 16000:
            # Decimate by 3 (48k/16k = 3)
            self._buf.extend(pcm)
            output = bytearray()
            samples = len(self._buf) // 2
            # Take every 3rd sample
            for i in range(0, samples - 2, 3):
                offset = i * 2
                output.extend(self._buf[offset:offset + 2])
            # Keep remaining bytes
            consumed = (samples // 3) * 6  # 3 samples * 2 bytes
            del self._buf[:consumed]
            return bytes(output)
        elif self._src == 16000 and self._dst == 48000:
            # Upsample by 3 (repeat each sample 3x)
            self._buf.extend(pcm)
            output = bytearray()
            samples = len(self._buf) // 2
            for i in range(samples):
                offset = i * 2
                sample = self._buf[offset:offset + 2]
                output.extend(sample * 3)
            self._buf.clear()
            return bytes(output)
        else:
            # No resampling needed
            return pcm
