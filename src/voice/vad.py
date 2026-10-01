"""Silero VAD — voice activity detection on 16kHz mono PCM.

Detects when someone starts and stops speaking.
Uses double-threshold hysteresis + frame-based silence endpointing.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import onnxruntime


@dataclass
class VADResult:
    is_voice: bool
    utterance_ended: bool
    speech_started: bool


class WebRTCVAD:
    """webrtcvad (Google's C VAD) — ~50x cheaper per frame than silero-onnx.

    Accepts 48kHz frames directly — no resampling needed for VAD. Binary
    voice/no-voice per 20ms frame; hysteresis is done via the same
    frame-window + silence-chunk logic as SileroVAD.
    """

    input_rate = 48000

    def __init__(
        self,
        aggressiveness: int = 1,   # 0-3; 1 = permissive (good with AGC + downstream filters)
        min_silence_duration_ms: int = 600,
        frame_window: int = 2,
    ):
        import webrtcvad
        self._vad = webrtcvad.Vad(aggressiveness)
        self._frame_window = frame_window
        self._silence_ms = min_silence_duration_ms
        self.reset()

    def reset(self) -> None:
        self._voice_window: deque[bool] = deque(maxlen=self._frame_window)
        self._have_voice = False
        self._silence_chunks = 0
        # webrtcvad eats 20ms frames (1920 bytes @48kHz)
        self._silence_chunks_threshold = max(1, int(round(self._silence_ms / 20)))

    def feed(self, pcm_48k_16bit: bytes) -> VADResult:
        """Feed 48kHz 16-bit mono PCM. Returns VADResult for this batch."""
        utterance_ended = False
        is_voice_any = False
        speech_started = False
        # webrtcvad requires exact 10/20/30ms frames — 20ms = 1920 bytes
        for i in range(0, len(pcm_48k_16bit) - (len(pcm_48k_16bit) % 1920), 1920):
            try:
                cur_voice = self._vad.is_speech(pcm_48k_16bit[i:i + 1920], 48000)
            except Exception:
                cur_voice = False
            self._voice_window.append(cur_voice)
            chunk_have_voice = sum(self._voice_window) >= self._frame_window
            if chunk_have_voice:
                if not self._have_voice:
                    speech_started = True
                self._have_voice = True
                self._silence_chunks = 0
            else:
                if self._have_voice:
                    self._silence_chunks += 1
                    if self._silence_chunks >= self._silence_chunks_threshold:
                        utterance_ended = True
                        self._have_voice = False
                        self._silence_chunks = 0
            is_voice_any = is_voice_any or chunk_have_voice

        return VADResult(
            is_voice=is_voice_any,
            utterance_ended=utterance_ended,
            speech_started=speech_started,
        )


class SileroVAD:
    """Silero VAD with hysteresis + silence-duration endpoint detection."""

    input_rate = 16000

    def __init__(
        self,
        model_dir: str = "models/silero_vad",
        threshold: float = 0.5,
        threshold_low: float = 0.3,
        min_silence_duration_ms: int = 600,
        frame_window: int = 2,
    ):
        model_path = Path(model_dir) / "src" / "silero_vad" / "data" / "silero_vad.onnx"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Silero VAD model not found at {model_path}. "
                f"Download from https://github.com/snakers4/silero-vad"
            )
        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self._session = onnxruntime.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
            sess_options=opts,
        )
        self._th = threshold
        self._th_low = threshold_low
        self._silence_ms = min_silence_duration_ms
        self._frame_window = frame_window
        self.reset()

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, 64), dtype=np.float32)
        self._buf = bytearray()
        self._last_is_voice = False
        self._voice_window: deque[bool] = deque(maxlen=self._frame_window)
        self._have_voice = False
        self._silence_chunks = 0
        self._silence_chunks_threshold = max(1, int(round(self._silence_ms / 32)))

    def feed(self, pcm_16k_16bit: bytes) -> VADResult:
        """Feed 16kHz 16-bit mono PCM. Returns VADResult for this batch."""
        self._buf.extend(pcm_16k_16bit)
        utterance_ended = False
        is_voice_any = False
        speech_started = False

        # Silero requires 512-sample chunks (32ms at 16kHz)
        while len(self._buf) >= 512 * 2:
            chunk = bytes(self._buf[: 512 * 2])
            del self._buf[: 512 * 2]

            audio_int16 = np.frombuffer(chunk, dtype=np.int16)
            audio_f32 = audio_int16.astype(np.float32) / 32768.0
            audio_input = np.concatenate(
                [self._context, audio_f32.reshape(1, -1)], axis=1
            ).astype(np.float32)

            ort_inputs = {
                "input": audio_input,
                "state": self._state,
                "sr": np.array(16000, dtype=np.int64),
            }
            out, state = self._session.run(None, ort_inputs)
            self._state = state
            self._context = audio_input[:, -64:]
            prob = float(out.item())

            # Double-threshold hysteresis
            if prob >= self._th:
                cur_voice = True
            elif prob <= self._th_low:
                cur_voice = False
            else:
                cur_voice = self._last_is_voice
            self._last_is_voice = cur_voice

            self._voice_window.append(cur_voice)
            chunk_have_voice = sum(self._voice_window) >= self._frame_window

            if chunk_have_voice:
                if not self._have_voice:
                    speech_started = True
                self._have_voice = True
                self._silence_chunks = 0
            else:
                if self._have_voice:
                    self._silence_chunks += 1
                    if self._silence_chunks >= self._silence_chunks_threshold:
                        utterance_ended = True
                        self._have_voice = False
                        self._silence_chunks = 0
            is_voice_any = is_voice_any or chunk_have_voice

        return VADResult(
            is_voice=is_voice_any,
            utterance_ended=utterance_ended,
            speech_started=speech_started,
        )
