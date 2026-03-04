"""Voice Activity Detection — Silero VAD wrapper.

Detects speech boundaries in audio streams. Uses the ONNX variant of
Silero VAD for zero-VRAM, CPU-only inference (~1ms per chunk).
All configuration comes from AraSettings.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
from silero_vad import load_silero_vad

from ara.utils.errors import VoiceError
from ara.utils.logging import get_logger

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ara.config.settings import VADSettings

log = get_logger("vad")


class VADResult:
    """Result of processing a single audio chunk through VAD."""

    __slots__ = ("confidence", "is_speech", "latency_ms")

    def __init__(self, *, is_speech: bool, confidence: float, latency_ms: float) -> None:
        self.is_speech = is_speech
        self.confidence = confidence
        self.latency_ms = latency_ms


class SpeechSegmentTracker:
    """Tracks speech/silence durations to determine segment boundaries.

    Applies minimum speech/silence duration thresholds to prevent
    false triggers and premature cutoffs.
    """

    def __init__(self, settings: VADSettings, sample_rate: int) -> None:
        self._min_speech_samples = int(settings.min_speech_ms * sample_rate / 1000)
        self._min_silence_samples = int(settings.min_silence_ms * sample_rate / 1000)
        self._speech_samples = 0
        self._silence_samples = 0
        self._is_speaking = False

    @property
    def is_speaking(self) -> bool:
        """Whether the user is currently speaking (after threshold filtering)."""
        return self._is_speaking

    def update(self, is_speech: bool, chunk_samples: int) -> bool:
        """Update tracker with a new VAD result.

        Args:
            is_speech: Raw VAD output for this chunk.
            chunk_samples: Number of audio samples in the chunk.

        Returns:
            True if speech state changed (started or stopped speaking).
        """
        previous_state = self._is_speaking

        if is_speech:
            self._speech_samples += chunk_samples
            self._silence_samples = 0

            if not self._is_speaking and self._speech_samples >= self._min_speech_samples:
                self._is_speaking = True
                log.debug(
                    "speech_start",
                    accumulated_ms=round(self._speech_samples / 16),  # approx at 16kHz
                )
        else:
            self._silence_samples += chunk_samples

            if self._is_speaking and self._silence_samples >= self._min_silence_samples:
                self._is_speaking = False
                self._speech_samples = 0
                log.debug(
                    "speech_end",
                    silence_ms=round(self._silence_samples / 16),
                )

        return self._is_speaking != previous_state

    def reset(self) -> None:
        """Reset the tracker state."""
        self._speech_samples = 0
        self._silence_samples = 0
        self._is_speaking = False


class VADProcessor:
    """Silero VAD wrapper for speech detection.

    Processes audio chunks and returns speech probability. The model is
    stateful (maintains hidden states across chunks) so chunks must be
    fed in order.

    Usage:
        vad = VADProcessor(settings)
        vad.load()
        result = vad.process(audio_chunk)
        if result.is_speech:
            # handle speech
    """

    def __init__(self, settings: VADSettings) -> None:
        self._settings = settings
        self._model: object | None = None
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        """Whether the VAD model is loaded and ready."""
        return self._loaded

    def load(self) -> None:
        """Load the Silero VAD ONNX model.

        Raises:
            VoiceError: If model loading fails.
        """
        try:
            start = time.monotonic()
            self._model = load_silero_vad(onnx=True)
            elapsed_ms = (time.monotonic() - start) * 1000
            self._loaded = True

            log.info(
                "vad_loaded",
                backend="onnx",
                load_time_ms=round(elapsed_ms, 1),
                threshold=self._settings.threshold,
            )
        except Exception as exc:
            raise VoiceError(
                f"Failed to load Silero VAD: {exc}",
                operation="vad_load",
            ) from exc

    def process(self, audio: NDArray[np.float32], sample_rate: int = 16000) -> VADResult:
        """Process an audio chunk and return speech detection result.

        Args:
            audio: Audio samples as float32 array. Should be 512 samples
                   at 16kHz (32ms) for optimal performance.
            sample_rate: Sample rate of the audio (must be 16000).

        Returns:
            VADResult with speech detection confidence and timing.

        Raises:
            VoiceError: If processing fails or model not loaded.
        """
        if not self._loaded or self._model is None:
            raise VoiceError(
                "VAD model not loaded — call load() first",
                operation="vad_process",
            )

        start = time.monotonic()

        try:
            import torch

            tensor = torch.from_numpy(audio)
            confidence = self._model(tensor, sample_rate).item()
        except Exception as exc:
            raise VoiceError(
                f"VAD processing failed: {exc}",
                operation="vad_process",
                details={"audio_len": len(audio), "sample_rate": sample_rate},
            ) from exc

        elapsed_ms = (time.monotonic() - start) * 1000

        return VADResult(
            is_speech=confidence >= self._settings.threshold,
            confidence=round(confidence, 4),
            latency_ms=round(elapsed_ms, 2),
        )

    def reset(self) -> None:
        """Reset the model's internal hidden states.

        Call this between separate audio sessions to prevent state leakage.
        """
        if self._model is not None:
            self._model.reset_states()
            log.debug("vad_states_reset")

    def unload(self) -> None:
        """Release the model from memory."""
        self._model = None
        self._loaded = False
        log.info("vad_unloaded")
