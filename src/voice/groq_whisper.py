"""Groq Whisper API two-pass ASR — fast, accurate cloud transcription.

Instead of running Whisper locally on CPU (2-6s for base.en), we use Groq's
whisper-large-v3-turbo API which:
- Processes 1 minute of audio in ~0.1s (100x faster than local CPU)
- Uses large-v3 model (much more accurate than base.en)
- Doesn't block the event loop or CPU
- No local model download needed
- Uses the same Groq API keys already configured

The audio is sent as a WAV file to the Groq API. The response is the
transcribed text.
"""
from __future__ import annotations

import asyncio
import io
import time
import wave
from typing import Optional

import numpy as np
from loguru import logger


class GroqWhisperTwoPass:
    """Cloud-based Whisper two-pass using Groq's whisper-large-v3-turbo API.

    Much faster and more accurate than local faster-whisper on CPU.
    Uses the same Groq API keys from src.ai.llm.
    """

    def __init__(self, model: str = "whisper-large-v3-turbo", language: str = "en"):
        self._model = model
        self._language = language
        self._groq_clients = None
        self._groq_cooldowns = None
        self._current_key_idx = 0

    def _init_clients(self):
        """Lazily import Groq clients from llm module (avoids circular import)."""
        if self._groq_clients is not None:
            return
        from ..ai import llm
        self._groq_clients = llm._groq_clients
        self._groq_cooldowns = llm._groq_cooldowns

    async def preload(self) -> None:
        """Nothing to preload — API-based, no local model."""
        self._init_clients()
        logger.info(f"[groq-whisper] Ready (model={self._model}, keys={len(self._groq_clients)})")

    async def transcribe(self, pcm_16k_float32: np.ndarray) -> Optional[str]:
        """Transcribe 16kHz float32 mono audio via Groq Whisper API.

        Args:
            pcm_16k_float32: 16kHz mono audio as float32 numpy array

        Returns:
            Transcribed text or None
        """
        self._init_clients()
        if not self._groq_clients:
            logger.warning("[groq-whisper] No Groq keys configured")
            return None

        # Skip very short audio (< 0.3s)
        if pcm_16k_float32.size < 4800:
            return None

        # Convert float32 to int16 WAV bytes
        audio_int16 = (pcm_16k_float32 * 32768.0).clip(-32768, 32767).astype(np.int16)
        wav_bytes = self._pcm_to_wav(audio_int16, 16000)

        # Call Groq API in thread pool with a 10s timeout (one call took 74s before!)
        loop = asyncio.get_event_loop()
        t0 = time.time()

        try:
            text = await asyncio.wait_for(
                loop.run_in_executor(None, self._call_groq_api, wav_bytes),
                timeout=10.0,
            )
            elapsed = time.time() - t0
            if text:
                # Filter out Whisper hallucinations on silence
                text = self._filter_hallucination(text, pcm_16k_float32)
                if text:
                    logger.info(f"[groq-whisper] Transcribed in {elapsed:.2f}s: '{text[:80]}'")
            return text
        except asyncio.TimeoutError:
            logger.warning(f"[groq-whisper] API call timed out after 10s, skipping two-pass")
            return None
        except Exception as e:
            logger.error(f"[groq-whisper] API call failed: {e!r}")
            return None

    # Known Whisper hallucinations on silence/noise
    # These are phrases Whisper outputs when given silence or background noise
    _HALLUCINATIONS = {
        "thank you", "thank you.", "thanks", "thanks.",
        "okay", "okay.", "ok", "ok.",
        "bye", "bye.", "goodbye", "goodbye.",
        "see you", "see you.", "see you all", "see you all.",
        "see you later", "see you later.",
        "please", "please.",
        "you", "you.",
        "yeah", "yeah.", "yes", "yes.",
        "no", "no.",
        "uh", "uh.", "um", "um.",
        "so", "so.", "and", "and.",
        "the", "the.", "this", "this.",
        "i", "i.", "a", "a.",
        "all", "all.", "very", "very.",
        "working", "working.",
        "i'll", "i'll.",
        "thank you very much", "thank you very much.",
        "thank you for watching", "thank you for watching.",
        "thanks for watching", "thanks for watching.",
        "please subscribe", "please subscribe.",
        "see you next time", "see you next time.",
    }

    def _filter_hallucination(self, text: str, audio: np.ndarray) -> Optional[str]:
        """Filter out known Whisper hallucinations on short/silent audio.

        Whisper is notorious for hallucinating phrases like "Thank you." and
        "Bye." on silence. If the audio is short (< 1.5s) and the transcript
        matches a known hallucination, return None.
        """
        audio_duration = audio.size / 16000.0  # 16kHz
        normalized = text.strip().lower()

        # Only filter on short audio — longer audio with these words is likely real
        if audio_duration < 1.5 and normalized in self._HALLUCINATIONS:
            logger.debug(f"[groq-whisper] Filtered hallucination '{text}' on {audio_duration:.1f}s audio")
            return None

        return text

    def _pcm_to_wav(self, pcm_int16: np.ndarray, sample_rate: int) -> bytes:
        """Convert int16 PCM numpy array to WAV bytes."""
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)  # 16-bit
            wav.setframerate(sample_rate)
            wav.writeframes(pcm_int16.tobytes())
        return buf.getvalue()

    def _call_groq_api(self, wav_bytes: bytes) -> Optional[str]:
        """Call Groq Whisper API with WAV audio. Synchronous (called in executor).

        Rotates through Groq keys on rate limits, same as the LLM caller.
        """
        n = len(self._groq_clients)
        if n == 0:
            return None

        import time as _time

        for attempt in range(n):
            idx = self._current_key_idx % n
            self._current_key_idx = (idx + 1) % n

            client = self._groq_clients[idx]

            # Check cooldown
            cooldown_until = self._groq_cooldowns.get(client.api_key, 0)
            if _time.time() < cooldown_until:
                continue

            try:
                # Create a file-like object for the API
                audio_file = ("audio.wav", wav_bytes, "audio/wav")
                response = client.audio.transcriptions.create(
                    model=self._model,
                    file=audio_file,
                    language=self._language,
                    response_format="text",
                )
                text = response.strip() if isinstance(response, str) else str(response).strip()
                return text if text else None

            except Exception as e:
                err = str(e)
                if "429" in err or "rate limit" in err.lower():
                    # Rate limited — cool this key and try next
                    is_tpd = "TPD" in err or "tokens per day" in err.lower()
                    wait_time = 3600 if is_tpd else 60
                    self._groq_cooldowns[client.api_key] = _time.time() + wait_time
                    logger.warning(f"[groq-whisper] Key {idx+1} rate limited, cooling {wait_time}s")
                    continue
                elif "404" in err or "model_not_found" in err:
                    logger.error(f"[groq-whisper] Model not found: {self._model}")
                    return None
                else:
                    logger.warning(f"[groq-whisper] Key {idx+1} failed: {e}")
                    continue

        logger.warning("[groq-whisper] All keys exhausted")
        return None

    async def close(self) -> None:
        """Nothing to close."""
        pass
