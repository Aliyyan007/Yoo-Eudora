"""Faster-Whisper two-pass ASR — high-accuracy final transcription.

After the streaming zipformer detects utterance end, this module re-scores
the full utterance with faster-whisper (CTranslate2 backend) for much better
accuracy. The streaming ASR is still used for live partials; this module
only runs on the finalized audio.

Model choices (CPU):
  - tiny.en    ~75MB  fastest, OK accuracy
  - base.en    ~145MB good balance for real-time CPU use
  - small.en   ~480MB better accuracy, ~2-3s on CPU for short utterances
  - medium.en  ~1.5GB high accuracy, ~5-10s on CPU

For a Discord bot on CPU, base.en or small.en is the sweet spot.
"""
from __future__ import annotations

import asyncio
import time
from typing import Optional

import numpy as np
from loguru import logger


class WhisperTwoPass:
    """Faster-Whisper batch ASR for final high-accuracy transcription.

    The model is loaded lazily on first use and shared across all instances
    (class-level cache). Inference is serialized via a class-level lock since
    CTranslate2 models are not documented as thread-safe.
    """

    _model_cache: dict = {}
    _cache_lock = asyncio.Lock()
    _infer_lock = asyncio.Lock()

    def __init__(
        self,
        model_size: str = "small.en",
        device: str = "cpu",
        compute_type: str = "int8",
        language: str = "en",
    ):
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._language = language
        self._model = None

    async def preload(self) -> None:
        """Load the Whisper model (shared across instances)."""
        if self._model is not None:
            return
        async with WhisperTwoPass._cache_lock:
            key = (self._model_size, self._device, self._compute_type)
            cached = WhisperTwoPass._model_cache.get(key)
            if cached is not None:
                self._model = cached
                return
            logger.info(f"[whisper] Loading faster-whisper ({self._model_size}, {self._device}/{self._compute_type})...")
            loop = asyncio.get_event_loop()

            def _load():
                from faster_whisper import WhisperModel
                return WhisperModel(
                    self._model_size,
                    device=self._device,
                    compute_type=self._compute_type,
                )

            try:
                model = await loop.run_in_executor(None, _load)
                WhisperTwoPass._model_cache[key] = model
                self._model = model
                logger.info(f"[whisper] faster-whisper loaded ({self._model_size})")
            except Exception as e:
                logger.error(f"[whisper] Failed to load faster-whisper: {e!r}")
                raise

    async def transcribe(self, pcm_16k_float32: np.ndarray,
                         prompt: Optional[str] = None) -> Optional[str]:
        """Transcribe 16kHz float32 mono audio. Returns text or None.

        This is the second pass — run after the streaming ASR has detected
        utterance end. The full utterance audio is re-scored for better accuracy.
        `prompt` maps to faster-whisper's initial_prompt — decode context.
        """
        if self._model is None:
            await self.preload()
        if self._model is None:
            return None

        # Skip very short audio (< 0.3s)
        if pcm_16k_float32.size < 4800:
            return None

        async with WhisperTwoPass._infer_lock:
            loop = asyncio.get_event_loop()
            t0 = time.time()

            def _transcribe():
                segments, _info = self._model.transcribe(
                    pcm_16k_float32,
                    language=self._language,
                    beam_size=1,         # fastest
                    best_of=1,           # fastest
                    vad_filter=True,     # trim leading/trailing silence
                    vad_parameters=dict(
                        min_silence_duration_ms=300,
                        speech_pad_ms=200,  # pad speech to avoid clipping
                    ),
                    without_timestamps=True,
                    initial_prompt=(prompt or None),
                )
                # segments is a generator — consume it
                text = " ".join(seg.text.strip() for seg in segments).strip()
                return text

            try:
                text = await loop.run_in_executor(None, _transcribe)
                elapsed = time.time() - t0
                logger.info(f"[whisper] Transcribed in {elapsed:.2f}s: '{text[:80]}'")
                return text if text else None
            except Exception as e:
                logger.error(f"[whisper] Transcription failed: {e!r}")
                return None

    async def close(self) -> None:
        """Nothing to close — model is shared and cached."""
        pass
