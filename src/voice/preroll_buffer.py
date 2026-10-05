"""Per-user ring buffer of recent PCM frames; drained on VAD speech_start.

VAD has inherent latency — without buffering the first 200-300ms of each
utterance, ASR drops the leading syllable. This ring buffer holds the last
N frames of 48kHz mono PCM and drains them into ASR when speech starts.

Based on EchoTwin's PrerollRingBuffer (preroll_buffer.py).
"""
from __future__ import annotations

from collections import deque


class PrerollRingBuffer:
    """Keep up to N most recent PCM frame chunks; drain returns concatenated bytes.

    Each Discord voice packet is ~20ms of 48kHz mono PCM (1920 bytes).
    Default 15 frames = 300ms of audio — enough to capture the first syllable
    that VAD's latency would otherwise miss.
    """

    def __init__(self, max_frames: int = 15):
        self._dq: deque[bytes] = deque(maxlen=max(0, max_frames))

    def push(self, pcm_frame: bytes) -> None:
        """Push a PCM frame into the ring buffer. Oldest frames are dropped."""
        if self._dq.maxlen == 0:
            return
        self._dq.append(pcm_frame)

    def drain(self) -> bytes:
        """Return all buffered frames concatenated, then clear the buffer."""
        if not self._dq:
            return b""
        out = b"".join(self._dq)
        self._dq.clear()
        return out

    def clear(self) -> None:
        """Discard buffered frames. Call at utterance end so the previous
        utterance's tail doesn't get prepended to the next one."""
        self._dq.clear()

    def __len__(self) -> int:
        return len(self._dq)
