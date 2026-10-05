"""Noise suppression for Discord voice audio.

Uses noisereduce (spectral gating) to clean up background noise before
feeding to ASR. This improves ASR accuracy significantly in noisy
environments.

For real-time use, we apply noise reduction on a per-utterance basis
(after VAD endpoint, before ASR final) rather than per-frame, since
noisereduce works best on longer audio segments.
"""
from __future__ import annotations

import numpy as np
from loguru import logger


class NoiseSuppressor:
    """Spectral-gating noise suppression using noisereduce.

    Applied to the full utterance audio after VAD endpoint, before the
    final ASR pass. This is cheaper and more effective than per-frame
    noise reduction.
    """

    def __init__(self, enabled: bool = True):
        self._enabled = enabled
        self._noise_profile: Optional[np.ndarray] = None

    def suppress(self, pcm_16k_float32: np.ndarray) -> np.ndarray:
        """Apply noise suppression to 16kHz float32 mono audio.

        Returns cleaned audio of the same shape.
        If noisereduce is not available or disabled, returns the input unchanged.
        """
        if not self._enabled or pcm_16k_float32.size < 1600:
            return pcm_16k_float32

        try:
            import noisereduce as nr
            # Use stationary noise reduction (fast, good for constant background noise)
            # noise_clip is auto-estimated from the signal
            reduced = nr.reduce_noise(
                y=pcm_16k_float32,
                sr=16000,
                stationary=True,
                prop_decrease=0.8,  # reduce 80% of noise
            )
            return reduced.astype(np.float32)
        except ImportError:
            return pcm_16k_float32
        except Exception as e:
            logger.debug(f"[noise] Noise suppression failed: {e}")
            return pcm_16k_float32
