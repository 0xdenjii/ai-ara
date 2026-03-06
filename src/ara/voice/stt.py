"""Speech-to-Text — Moonshine v2 wrapper.

Provides streaming transcription using moonshine-voice (v2). Runs entirely
on CPU with ONNX Runtime, consuming ~300MB RAM and zero VRAM.
All configuration comes from AraSettings.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from ara.utils.errors import VoiceError
from ara.utils.logging import get_logger

if TYPE_CHECKING:
    from ara.config.settings import STTSettings

log = get_logger("stt")


class TranscriptionEvent:
    """A transcription update from the STT engine."""

    __slots__ = ("is_final", "latency_ms", "text", "timestamp")

    def __init__(self, *, text: str, is_final: bool, latency_ms: float, timestamp: float) -> None:
        self.text = text
        self.is_final = is_final
        self.latency_ms = latency_ms
        self.timestamp = timestamp


class STTProcessor:
    """Moonshine v2 streaming speech-to-text processor.

    Wraps the moonshine-voice Transcriber for streaming transcription.
    Audio is fed in chunks; transcription events are emitted as text
    becomes available.

    Usage:
        stt = STTProcessor(settings)
        stt.load()
        # Feed audio chunks from the mic...
        stt.feed_audio(audio_chunk)
        # Get accumulated transcription
        text = stt.get_transcript()
        stt.reset()  # between utterances
    """

    def __init__(self, settings: STTSettings) -> None:
        self._settings = settings
        self._transcriber: object | None = None
        self._loaded = False
        self._current_text = ""
        self._events: list[TranscriptionEvent] = []

    @property
    def is_loaded(self) -> bool:
        """Whether the STT model is loaded and ready."""
        return self._loaded

    @property
    def current_text(self) -> str:
        """The current accumulated transcription text."""
        return self._current_text

    def load(self) -> None:
        """Load the Moonshine v2 model.

        Auto-discovers the model path from moonshine cache if not
        explicitly configured.

        Raises:
            VoiceError: If model loading fails.
        """
        try:
            from moonshine_voice.moonshine_api import ModelArch
            from moonshine_voice.transcriber import Transcriber

            start = time.monotonic()

            model_path = self._settings.model_path
            model_arch_int = self._settings.model_arch

            if not model_path:
                model_path = self._discover_model_path()

            # Convert int setting to the ModelArch enum that Transcriber expects
            model_arch = ModelArch(model_arch_int)

            self._transcriber = Transcriber(
                model_path=model_path,
                model_arch=model_arch,
            )

            elapsed_ms = (time.monotonic() - start) * 1000
            self._loaded = True

            log.info(
                "stt_loaded",
                model="moonshine-v2-medium",
                model_path=str(model_path),
                model_arch=model_arch.name,
                load_time_ms=round(elapsed_ms, 1),
            )
        except ImportError as exc:
            raise VoiceError(
                "moonshine-voice not installed. Run: pip install moonshine-voice",
                operation="stt_load",
            ) from exc
        except Exception as exc:
            raise VoiceError(
                f"Failed to load Moonshine v2: {exc}",
                operation="stt_load",
                details={
                    "model_path": str(self._settings.model_path),
                    "model_arch": self._settings.model_arch,
                },
            ) from exc

    def _discover_model_path(self) -> str:
        """Auto-discover the Moonshine model path from its cache directory.

        Moonshine downloads to:
        {LOCALAPPDATA}/moonshine_voice/moonshine_voice/Cache/download.moonshine.ai/model/{variant}

        Returns:
            The path to the model directory.

        Raises:
            VoiceError: If the model cache directory cannot be found.
        """
        import os
        from pathlib import Path

        # moonshine-voice stores models in its local app data cache
        local_app_data = os.environ.get("LOCALAPPDATA", "")
        if local_app_data:
            cache_base = Path(local_app_data) / "moonshine_voice" / "moonshine_voice" / "Cache"
        else:
            cache_base = Path.home() / ".cache" / "moonshine_voice"

        # Look for the medium-streaming-en quantized model
        model_dir = (
            cache_base / "download.moonshine.ai" / "model" / "medium-streaming-en" / "quantized"
        )

        if model_dir.exists():
            log.info("stt_model_discovered", path=str(model_dir))
            return str(model_dir)

        # Try to find any model directory
        if cache_base.exists():
            for candidate in cache_base.rglob("encoder.ort"):
                discovered = str(candidate.parent)
                log.info("stt_model_discovered", path=discovered)
                return discovered

        raise VoiceError(
            "Moonshine v2 model not found. Run: python -m moonshine_voice.download --language en",
            operation="stt_discover_model",
            details={"searched": str(cache_base)},
        )

    def start_utterance(self) -> None:
        """Begin a new utterance transcription session.

        Call this when speech is detected by VAD to start accumulating
        transcription for a new user utterance.
        """
        self._current_text = ""
        self._events.clear()
        if self._transcriber is not None:
            try:
                self._transcriber.stop()
            except Exception as exc:
                log.debug("transcriber_stop_before_restart_failed", error=str(exc))
            self._transcriber.start()
        log.debug("stt_utterance_started")

    def feed_audio(
        self, audio_data: bytes | list[float], sample_rate: int = 16000
    ) -> TranscriptionEvent | None:
        """Feed audio data to the transcriber.

        Accepts either raw PCM bytes (int16) or a list of float32 samples.
        The moonshine-voice Transcriber expects List[float] and a sample_rate.

        Args:
            audio_data: Audio as int16 bytes or float32 sample list.
            sample_rate: Audio sample rate (must be 16000).

        Returns:
            A TranscriptionEvent if new text is available, else None.

        Raises:
            VoiceError: If the model is not loaded or processing fails.
        """
        if not self._loaded or self._transcriber is None:
            raise VoiceError(
                "STT model not loaded — call load() first",
                operation="stt_feed_audio",
            )

        start = time.monotonic()

        try:
            # Convert bytes to list of floats if needed
            if isinstance(audio_data, bytes):
                import numpy as np

                samples_i16 = np.frombuffer(audio_data, dtype=np.int16)
                float_list = (samples_i16.astype(np.float32) / 32768.0).tolist()
            else:
                float_list = audio_data

            self._transcriber.add_audio(float_list, sample_rate)

            # Get updated transcription
            transcript = self._transcriber.update_transcription()
            new_text = " ".join(line.text for line in transcript.lines if line.text.strip())
        except Exception as exc:
            raise VoiceError(
                f"STT processing failed: {exc}",
                operation="stt_feed_audio",
                details={"sample_rate": sample_rate},
            ) from exc

        elapsed_ms = (time.monotonic() - start) * 1000

        if new_text and new_text != self._current_text:
            self._current_text = new_text
            event = TranscriptionEvent(
                text=new_text,
                is_final=False,
                latency_ms=round(elapsed_ms, 2),
                timestamp=time.monotonic(),
            )
            self._events.append(event)
            return event

        return None

    def end_utterance(self) -> TranscriptionEvent | None:
        """Finalize the current utterance and get the final transcription.

        Returns:
            A final TranscriptionEvent with the complete transcription.
        """
        if not self._loaded or self._transcriber is None:
            return None

        start = time.monotonic()

        try:
            # Final update before stopping
            transcript = self._transcriber.update_transcription()
            final_text = " ".join(
                line.text for line in transcript.lines if line.text.strip()
            ).strip()
            self._transcriber.stop()

            # Use the latest text if update returned something new
            if not final_text:
                final_text = self._current_text.strip()
            else:
                self._current_text = final_text
        except Exception as exc:
            log.warning("stt_end_utterance_error", error=str(exc))
            final_text = self._current_text.strip()

        elapsed_ms = (time.monotonic() - start) * 1000

        if final_text:
            event = TranscriptionEvent(
                text=final_text,
                is_final=True,
                latency_ms=round(elapsed_ms, 2),
                timestamp=time.monotonic(),
            )
            log.info(
                "stt_utterance_complete",
                text_length=len(final_text),
                total_events=len(self._events),
            )
            return event

        return None

    def get_transcript(self) -> str:
        """Get the current accumulated transcription text."""
        return self._current_text.strip()

    def reset(self) -> None:
        """Reset transcription state between utterances."""
        self._current_text = ""
        self._events.clear()
        log.debug("stt_reset")

    def unload(self) -> None:
        """Release the model from memory."""
        self._transcriber = None
        self._loaded = False
        self._current_text = ""
        self._events.clear()
        log.info("stt_unloaded")
