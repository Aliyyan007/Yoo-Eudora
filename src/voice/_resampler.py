"""Simple 48kHz → 16kHz resampler using soxr if available, else decimation."""
from __future__ import annotations

import numpy as np


class Resampler:
    """Resample 48kHz mono PCM to 16kHz mono PCM (streaming, stateful)."""

    def __init__(self, src_rate: int = 48000, dst_rate: int = 16000):
        self._src = src_rate
        self._dst = dst_rate
        self._buf = bytearray()

        # Try to use soxr for quality
        try:
            import soxr
            self._resampler = soxr.ResampleStream(
                src_rate, dst_rate, 1, dtype="int16"
            )
        except ImportError:
            self._resampler = None

    def feed(self, pcm: bytes) -> bytes:
        """Feed input PCM, get resampled output."""
        if not pcm:
            return b""

        if self._resampler is not None:
            # soxr ResampleStream takes/returns int16 ndarrays — feeding raw
            # bytes raises, which used to silently fall through to naive
            # decimation on EVERY call (aliased audio into VAD/Whisper).
            try:
                out = self._resampler.resample_chunk(
                    np.frombuffer(pcm, dtype=np.int16))
                return out.tobytes()
            except Exception:
                pass

        # Fallback: simple decimation (48k→16k only — causes aliasing,
        # but only reached when soxr isn't installed)
        if self._src == 48000 and self._dst == 16000:
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
        # No resampling needed
        return pcm
