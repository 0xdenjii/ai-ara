"""Text-to-Speech — Kokoro-82M ONNX wrapper.

Synthesizes speech from text using Kokoro's ONNX model. Runs entirely
on CPU, consuming ~200MB RAM and zero VRAM. Supports multiple voices
and sentence-aware chunked synthesis for low-latency streaming.
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

    from ara.config.settings import TTSSettings

log = get_logger("tts")


class TTSResult:
    """Result of a TTS synthesis operation."""

    __slots__ = ("duration_s", "latency_ms", "sample_rate", "samples")

    def __init__(
        self,
        *,
        samples: NDArray[np.float32],
        sample_rate: int,
        duration_s: float,
        latency_ms: float,
    ) -> None:
        self.samples = samples
        self.sample_rate = sample_rate
        self.duration_s = duration_s
        self.latency_ms = latency_ms


class TTSProcessor:
    """Kokoro-82M TTS wrapper for speech synthesis.

    Synthesizes text to audio using the Kokoro ONNX model. Supports
    voice selection, speed control, and sentence-aware chunking for
    streaming output.

    Usage:
        tts = TTSProcessor(settings)
        tts.load()
        result = tts.synthesize("Hello, how are you?")
        # result.samples is a numpy float32 array
        # result.sample_rate is 24000
    """

    def __init__(self, settings: TTSSettings) -> None:
        self._settings = settings
        self._kokoro: object | None = None
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        """Whether the TTS model is loaded and ready."""
        return self._loaded

    @property
    def voice(self) -> str:
        """Current voice identifier."""
        return self._settings.voice

    def load(self) -> None:
        """Load the Kokoro ONNX model and voice data.

        Raises:
            VoiceError: If model files are missing or loading fails.
        """
        try:
            from kokoro_onnx import Kokoro

            model_path = str(self._settings.model_path)
            voices_path = str(self._settings.voices_path)

            start = time.monotonic()

            self._kokoro = Kokoro(model_path, voices_path)

            elapsed_ms = (time.monotonic() - start) * 1000
            self._loaded = True

            log.info(
                "tts_loaded",
                model="kokoro-82m",
                model_path=model_path,
                voice=self._settings.voice,
                speed=self._settings.speed,
                load_time_ms=round(elapsed_ms, 1),
            )
        except ImportError as exc:
            raise VoiceError(
                "kokoro-onnx not installed. Run: pip install kokoro-onnx",
                operation="tts_load",
            ) from exc
        except FileNotFoundError as exc:
            raise VoiceError(
                f"Kokoro model files not found: {exc}. "
                "Download from: https://github.com/thewh1teagle/kokoro-onnx/releases",
                operation="tts_load",
                details={
                    "model_path": str(self._settings.model_path),
                    "voices_path": str(self._settings.voices_path),
                },
            ) from exc
        except Exception as exc:
            raise VoiceError(
                f"Failed to load Kokoro TTS: {exc}",
                operation="tts_load",
                details={
                    "model_path": str(self._settings.model_path),
                    "voices_path": str(self._settings.voices_path),
                },
            ) from exc

    def synthesize(
        self, text: str, *, voice: str | None = None, speed: float | None = None
    ) -> TTSResult:
        """Synthesize speech from text.

        Args:
            text: The text to synthesize.
            voice: Override the default voice (e.g., "af_sarah", "af_bella").
            speed: Override the default speech speed (1.0 = normal).

        Returns:
            TTSResult with audio samples, sample rate, and timing.

        Raises:
            VoiceError: If synthesis fails or model not loaded.
        """
        if not self._loaded or self._kokoro is None:
            raise VoiceError(
                "TTS model not loaded — call load() first",
                operation="tts_synthesize",
            )

        if not text.strip():
            return TTSResult(
                samples=np.array([], dtype=np.float32),
                sample_rate=self._settings.sample_rate,
                duration_s=0.0,
                latency_ms=0.0,
            )

        target_voice = voice or self._settings.voice
        target_speed = speed or self._settings.speed

        start = time.monotonic()

        try:
            samples, sample_rate = self._kokoro.create(
                text,
                voice=target_voice,
                speed=target_speed,
                lang=self._settings.lang,
            )
        except Exception as exc:
            raise VoiceError(
                f"TTS synthesis failed: {exc}",
                operation="tts_synthesize",
                details={
                    "text_len": len(text),
                    "voice": target_voice,
                    "speed": target_speed,
                },
            ) from exc

        elapsed_ms = (time.monotonic() - start) * 1000
        duration_s = len(samples) / sample_rate

        log.info(
            "tts_synthesized",
            text_len=len(text),
            voice=target_voice,
            duration_s=round(duration_s, 2),
            latency_ms=round(elapsed_ms, 1),
            rtf=round(elapsed_ms / (duration_s * 1000), 3) if duration_s > 0 else 0,
        )

        return TTSResult(
            samples=np.asarray(samples, dtype=np.float32),
            sample_rate=sample_rate,
            duration_s=round(duration_s, 3),
            latency_ms=round(elapsed_ms, 1),
        )

    def synthesize_sentences(
        self,
        sentences: list[str],
        *,
        voice: str | None = None,
        speed: float | None = None,
    ) -> list[TTSResult]:
        """Synthesize multiple sentences independently.

        This enables streaming TTS — synthesize the first sentence
        immediately while the LLM generates more text.

        Args:
            sentences: List of sentences to synthesize.
            voice: Override the default voice.
            speed: Override the default speed.

        Returns:
            List of TTSResult objects, one per sentence.
        """
        return [
            self.synthesize(sentence, voice=voice, speed=speed)
            for sentence in sentences
            if sentence.strip()
        ]

    def set_voice(self, voice: str) -> None:
        """Change the active voice.

        Args:
            voice: Voice identifier (e.g., "af_bella", "af_sarah", "af_nicole").
        """
        log.info("tts_voice_changed", from_voice=self._settings.voice, to_voice=voice)
        # Update the settings object's voice field
        object.__setattr__(self._settings, "voice", voice)

    def unload(self) -> None:
        """Release the model from memory."""
        self._kokoro = None
        self._loaded = False
        log.info("tts_unloaded")
