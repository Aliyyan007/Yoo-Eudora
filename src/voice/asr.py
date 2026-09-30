"""sherpa-onnx streaming ASR — speech-to-text as audio arrives.

Uses streaming zipformer model for English recognition.
Downloads model automatically from HuggingFace on first use.
"""
from __future__ import annotations

import asyncio
from typing import Optional

import numpy as np
from loguru import logger

# 20M streaming zipformer — ~4x smaller/faster than the big EN model while
# keeping the exact same filename layout (drop-in). Groq Whisper decides the
# final transcript anyway, so the stream only needs to be decent.
DEFAULT_REPO = "csukuangfj/sherpa-onnx-streaming-zipformer-en-20M-2023-02-17"
_MODEL_FILES = [
    "encoder-epoch-99-avg-1.int8.onnx",  # int8 — faster download, good accuracy
    "decoder-epoch-99-avg-1.onnx",
    "joiner-epoch-99-avg-1.int8.onnx",
    "tokens.txt",
]

_DECODE_MIN_SAMPLES = 6400   # 400ms @16k — larger batches amortize Python↔C++
                             # + infer-lock churn; partials are never displayed
                             # so coarser granularity costs nothing
_FLUSH_SILENCE_S = 0.4       # silence to flush decoder look-ahead


class StreamingASR:
    """Streaming speech-to-text using sherpa-onnx zipformer."""

    _recognizer_cache: dict = {}
    _cache_lock = asyncio.Lock()
    _infer_locks: dict = {}

    def __init__(
        self,
        repo: str = DEFAULT_REPO,
        num_threads: int = 2,
    ):
        self._repo = repo
        self._num_threads = num_threads
        self._recognizer = None
        self._stream = None
        self._reset_buffers()

    def _infer_lock(self) -> asyncio.Lock:
        lock = StreamingASR._infer_locks.get(self._repo)
        if lock is None:
            lock = StreamingASR._infer_locks[self._repo] = asyncio.Lock()
        return lock

    def _reset_buffers(self) -> None:
        """Reset per-utterance buffers. Resampler is recreated."""
        try:
            from ._resampler import Resampler
            self._resampler = Resampler(48000, 16000)
        except ImportError:
            self._resampler = None  # Will use manual resampling
        self._partial = ""
        self._pending = np.zeros(0, dtype=np.float32)
        self._inflight: Optional[asyncio.Task] = None

    async def preload(self) -> None:
        """Load the ASR model (shared across instances)."""
        if self._recognizer is not None:
            return
        async with StreamingASR._cache_lock:
            cached = StreamingASR._recognizer_cache.get(self._repo)
            if cached is not None:
                self._recognizer = cached
                return
            # Check for a previous failed load — don't retry forever
            if self._repo in StreamingASR._recognizer_cache:
                # None sentinel means load failed previously
                if StreamingASR._recognizer_cache[self._repo] is None:
                    raise RuntimeError(f"ASR model '{self._repo}' failed to load previously — not retrying")
            logger.info(f"Loading sherpa-onnx streaming ASR ({self._repo})...")
            loop = asyncio.get_event_loop()

            def _load():
                from pathlib import Path
                from huggingface_hub import snapshot_download
                import sherpa_onnx

                d = Path(snapshot_download(self._repo, allow_patterns=_MODEL_FILES))

                # Use modified_beam_search for better accuracy.
                # The streaming zipformer model uses BPE subword tokens,
                # so hotwords (whole words) often can't be encoded as single
                # tokens and get rejected. modified_beam_search alone still
                # improves accuracy over greedy_search by considering multiple
                # hypotheses during decoding.
                kwargs = dict(
                    tokens=str(d / "tokens.txt"),
                    encoder=str(d / _MODEL_FILES[0]),
                    decoder=str(d / _MODEL_FILES[1]),
                    joiner=str(d / _MODEL_FILES[2]),
                    num_threads=self._num_threads,
                    sample_rate=16000,
                    feature_dim=80,
                )

                # Try modified_beam_search first (better accuracy)
                try:
                    kwargs["decoding_method"] = "modified_beam_search"
                    kwargs["max_active_paths"] = 4
                    recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(**kwargs)
                    logger.info("sherpa-onnx ASR using modified_beam_search (max_active_paths=4)")
                except Exception as e_beam:
                    logger.warning(f"modified_beam_search failed ({e_beam}), falling back to greedy_search")
                    kwargs.pop("decoding_method", None)
                    kwargs.pop("max_active_paths", None)
                    recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(**kwargs)

                return recognizer

            try:
                recognizer = await loop.run_in_executor(None, _load)
                StreamingASR._recognizer_cache[self._repo] = recognizer
                self._recognizer = recognizer
                logger.info("sherpa-onnx streaming ASR loaded (shared)")
            except Exception as e:
                # Mark as failed so we don't retry in a tight loop
                StreamingASR._recognizer_cache[self._repo] = None
                logger.error(f"sherpa-onnx ASR load FAILED: {e!r}")
                raise

    async def open(self) -> None:
        """Start a new recognition stream."""
        await self.preload()
        self._reset_buffers()
        self._stream = self._recognizer.create_stream()

    async def feed_audio(self, pcm_48k_mono: bytes) -> None:
        """Feed 48kHz mono PCM audio. Resamples to 16k internally."""
        if self._resampler:
            pcm16 = self._resampler.feed(pcm_48k_mono)
        else:
            # Manual resample 48k → 16k (simple decimation by 3)
            import struct
            samples = struct.unpack(f'<{len(pcm_48k_mono)//2}h', pcm_48k_mono)
            pcm16 = b''.join(struct.pack('<h', samples[i]) for i in range(0, len(samples), 3))

        if not pcm16:
            return

        f32 = np.frombuffer(pcm16, dtype=np.int16).astype(np.float32) / 32768.0
        self._pending = np.concatenate([self._pending, f32])

        if len(self._pending) >= _DECODE_MIN_SAMPLES and (
            self._inflight is None or self._inflight.done()
        ):
            samples, self._pending = self._pending, np.zeros(0, dtype=np.float32)
            self._inflight = asyncio.create_task(self._decode(samples))

    async def _decode(self, samples: np.ndarray) -> None:
        stream, recognizer = self._stream, self._recognizer
        if stream is None:
            return
        loop = asyncio.get_event_loop()
        while True:
            async with self._infer_lock():
                def _run(buf=samples):
                    stream.accept_waveform(16000, buf)
                    while recognizer.is_ready(stream):
                        recognizer.decode_stream(stream)
                    return recognizer.get_result(stream)

                try:
                    self._partial = await loop.run_in_executor(None, _run)
                except Exception as e:
                    logger.error(f"ASR decode failed: {e}")
                    return
            if len(self._pending) >= _DECODE_MIN_SAMPLES and self._stream is stream:
                samples, self._pending = self._pending, np.zeros(0, dtype=np.float32)
                continue
            return

    def partial_text(self) -> str:
        return self._partial

    async def end_utterance(self) -> Optional[str]:
        """Finalize the current utterance and return the full text."""
        if self._inflight is not None and not self._inflight.done():
            try:
                await self._inflight
            except Exception:
                pass

        stream, recognizer = self._stream, self._recognizer
        if stream is None:
            return None

        pending = self._pending
        async with self._infer_lock():
            loop = asyncio.get_event_loop()

            def _finalize():
                if len(pending):
                    stream.accept_waveform(16000, pending)
                stream.accept_waveform(
                    16000, np.zeros(int(_FLUSH_SILENCE_S * 16000), dtype=np.float32)
                )
                stream.input_finished()
                while recognizer.is_ready(stream):
                    recognizer.decode_stream(stream)
                return recognizer.get_result(stream)

            try:
                text = await loop.run_in_executor(None, _finalize)
            except Exception as e:
                logger.error(f"ASR final decode failed: {e}")
                text = self._partial

        self._reset_buffers()
        self._stream = recognizer.create_stream()
        text = (text or "").strip()
        return text if text else None

    async def close(self) -> None:
        if self._inflight is not None and not self._inflight.done():
            self._inflight.cancel()
        self._reset_buffers()
        self._stream = None
