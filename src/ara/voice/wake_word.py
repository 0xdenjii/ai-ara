"""Wake Word Detection — openWakeWord wrapper.

Uses 'hey jarvis' as the trigger phrase. Custom 'hey ara' model requires
training a custom openWakeWord model. Runs on CPU with ONNX backend.
All configuration comes from AraSettings.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np

from ara.utils.errors import VoiceError
from ara.utils.logging import get_logger

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ara.config.settings import WakeWordSettings

log = get_logger("wake_word")


class WakeWordResult:
    """Result of processing an audio frame through wake word detection."""

    __slots__ = ("confidence", "detected", "latency_ms", "model_name")

    def __init__(
        self,
        *,
        detected: bool,
        model_name: str,
        confidence: float,
        latency_ms: float,
    ) -> None:
        self.detected = detected
        self.model_name = model_name
        self.confidence = confidence
        self.latency_ms = latency_ms


class WakeWordDetector:
    """openWakeWord wake phrase detector.

    Processes audio frames and detects when the user says the wake phrase.
    Uses ONNX backend on Windows for cross-platform compatibility.

    Usage:
        detector = WakeWordDetector(settings)
        detector.load()
        result = detector.process(audio_frame)
        if result.detected:
            # wake word detected, start listening
    """

    def __init__(self, settings: WakeWordSettings) -> None:
        self._settings = settings
        self._model: object | None = None
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        """Whether the wake word model is loaded and ready."""
        return self._loaded

    @property
    def is_enabled(self) -> bool:
        """Whether wake word detection is enabled in settings."""
        return self._settings.enabled

    def load(self) -> None:
        """Load the openWakeWord model(s).

        Downloads models automatically if not already present.

        Raises:
            VoiceError: If model loading fails.
        """
        if not self._settings.enabled:
            log.info("wake_word_disabled")
            return

        try:
            import openwakeword
            from openwakeword.model import Model

            # Ensure models are downloaded
            openwakeword.utils.download_models()

            start = time.monotonic()

            self._model = Model(
                wakeword_models=list(self._settings.model_names),
                inference_framework=self._settings.inference_framework,
            )

            elapsed_ms = (time.monotonic() - start) * 1000
            self._loaded = True

            log.info(
                "wake_word_loaded",
                models=self._settings.model_names,
                threshold=self._settings.threshold,
                inference_framework=self._settings.inference_framework,
                load_time_ms=round(elapsed_ms, 1),
            )
        except ImportError as exc:
            raise VoiceError(
                "openwakeword not installed. Run: pip install openwakeword",
                operation="wake_word_load",
            ) from exc
        except Exception as exc:
            raise VoiceError(
                f"Failed to load openWakeWord: {exc}",
                operation="wake_word_load",
                details={"models": self._settings.model_names},
            ) from exc

    def process(self, audio: NDArray[np.float32]) -> WakeWordResult:
        """Process an audio frame for wake word detection.

        Args:
            audio: Audio samples as float32 array. openWakeWord expects
                   frames of 80ms+ at 16kHz (1280+ samples).

        Returns:
            WakeWordResult with detection status and confidence.

        Raises:
            VoiceError: If processing fails or model not loaded.
        """
        if not self._settings.enabled:
            return WakeWordResult(detected=False, model_name="", confidence=0.0, latency_ms=0.0)

        if not self._loaded or self._model is None:
            raise VoiceError(
                "Wake word model not loaded — call load() first",
                operation="wake_word_process",
            )

        start = time.monotonic()

        try:
            # openWakeWord expects int16 audio
            audio_int16 = (audio * 32767).astype(np.int16)
            predictions = self._model.predict(audio_int16)
        except Exception as exc:
            raise VoiceError(
                f"Wake word processing failed: {exc}",
                operation="wake_word_process",
                details={"audio_len": len(audio)},
            ) from exc

        elapsed_ms = (time.monotonic() - start) * 1000

        # Check each model for detection
        best_model = ""
        best_confidence = 0.0

        for model_name, confidence in predictions.items():
            if confidence > best_confidence:
                best_confidence = confidence
                best_model = model_name

        detected = best_confidence >= self._settings.threshold

        if detected:
            log.info(
                "wake_word_detected",
                model=best_model,
                confidence=round(best_confidence, 4),
                latency_ms=round(elapsed_ms, 2),
            )
            # Reset to prevent repeated triggers
            self._model.reset()

        return WakeWordResult(
            detected=detected,
            model_name=best_model,
            confidence=round(best_confidence, 4),
            latency_ms=round(elapsed_ms, 2),
        )

    def reset(self) -> None:
        """Reset the model's internal state."""
        if self._model is not None:
            self._model.reset()
            log.debug("wake_word_reset")

    def unload(self) -> None:
        """Release the model from memory."""
        self._model = None
        self._loaded = False
        log.info("wake_word_unloaded")
