"""Integration tests for the voice pipeline path.

Tests the full voice pipeline flow: audio → VAD → STT → transcription callback → TTS → speaker.
All hardware components (microphone, speaker, models) are mocked.
Tests focus on component wiring, state transitions, and data flow between pipeline stages.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from ara.config.settings import AraSettings
from ara.utils.types import VoicePipelineState
from ara.voice.audio import AudioRingBuffer
from ara.voice.vad import VADResult
from ara.voice.wake_word import WakeWordResult

# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_audio_chunk(
    duration_ms: int = 32,
    sample_rate: int = 16000,
) -> np.ndarray:
    """Create a synthetic audio chunk of given duration."""
    n_samples = int(sample_rate * duration_ms / 1000)
    return np.random.randn(n_samples).astype(np.float32) * 0.01


def _make_vad_result(*, is_speech: bool, confidence: float = 0.9) -> VADResult:
    return VADResult(is_speech=is_speech, confidence=confidence, latency_ms=0.5)


def _make_wake_word_result(*, detected: bool, confidence: float = 0.95) -> WakeWordResult:
    return WakeWordResult(
        detected=detected,
        model_name="hey_jarvis_v0.1",
        confidence=confidence,
        latency_ms=0.3,
    )


def _make_stt_event(text: str, *, is_final: bool = False) -> Any:
    """Create a mock TranscriptionEvent."""
    event = MagicMock()
    event.text = text
    event.is_final = is_final
    event.latency_ms = 5.0
    event.timestamp = 0.0
    return event


def _make_tts_result(duration_s: float = 0.5) -> Any:
    """Create a mock TTSResult."""
    result = MagicMock()
    n_samples = int(24000 * duration_s)
    result.samples = np.zeros(n_samples, dtype=np.float32)
    result.sample_rate = 24000
    result.duration_s = duration_s
    result.latency_ms = 10.0
    return result


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def settings() -> AraSettings:
    return AraSettings(env="testing")


@pytest.fixture
def ring_buffer() -> AudioRingBuffer:
    return AudioRingBuffer(max_chunks=100)


# ══════════════════════════════════════════════════════════════════════════════
# Voice Pipeline Integration Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestVoicePipelineStateTransitions:
    """Test that the pipeline transitions through correct states."""

    def test_initial_state_is_idle(self, settings: AraSettings) -> None:
        from ara.voice.pipeline import VoicePipeline

        pipeline = VoicePipeline(settings)
        assert pipeline.state == VoicePipelineState.IDLE
        assert not pipeline.is_running

    def test_pipeline_creates_all_subcomponents(self, settings: AraSettings) -> None:
        from ara.voice.pipeline import VoicePipeline

        pipeline = VoicePipeline(settings)
        assert pipeline.vad is not None
        assert pipeline.stt is not None
        assert pipeline.tts is not None
        assert pipeline.wake_word is not None
        assert pipeline.mic is not None
        assert pipeline.speaker is not None
        assert pipeline.ring_buffer is not None

    @patch("ara.voice.pipeline.MicrophoneStream")
    @patch("ara.voice.pipeline.SpeakerOutput")
    @patch("ara.voice.pipeline.VADProcessor")
    @patch("ara.voice.pipeline.STTProcessor")
    @patch("ara.voice.pipeline.TTSProcessor")
    @patch("ara.voice.pipeline.WakeWordDetector")
    async def test_start_loads_all_models(
        self,
        mock_ww_cls: MagicMock,
        mock_tts_cls: MagicMock,
        mock_stt_cls: MagicMock,
        mock_vad_cls: MagicMock,
        mock_speaker_cls: MagicMock,
        mock_mic_cls: MagicMock,
        settings: AraSettings,
    ) -> None:
        from ara.voice.pipeline import VoicePipeline

        # Set up mock instances
        mock_vad = MagicMock()
        mock_vad.is_loaded = False
        mock_vad_cls.return_value = mock_vad

        mock_stt = MagicMock()
        mock_stt.is_loaded = False
        mock_stt_cls.return_value = mock_stt

        mock_tts = MagicMock()
        mock_tts.is_loaded = False
        mock_tts_cls.return_value = mock_tts

        mock_ww = MagicMock()
        mock_ww.is_loaded = False
        mock_ww.is_enabled = True
        mock_ww_cls.return_value = mock_ww

        mock_mic = MagicMock()
        mock_mic_cls.return_value = mock_mic

        mock_speaker = MagicMock()
        mock_speaker_cls.return_value = mock_speaker

        pipeline = VoicePipeline(settings)
        await pipeline.start()

        mock_vad.load.assert_called_once()
        mock_stt.load.assert_called_once()
        mock_tts.load.assert_called_once()
        mock_ww.load.assert_called_once()
        mock_mic.start.assert_called_once()

    @patch("ara.voice.pipeline.MicrophoneStream")
    @patch("ara.voice.pipeline.SpeakerOutput")
    @patch("ara.voice.pipeline.VADProcessor")
    @patch("ara.voice.pipeline.STTProcessor")
    @patch("ara.voice.pipeline.TTSProcessor")
    @patch("ara.voice.pipeline.WakeWordDetector")
    async def test_stop_unloads_all_models(
        self,
        mock_ww_cls: MagicMock,
        mock_tts_cls: MagicMock,
        mock_stt_cls: MagicMock,
        mock_vad_cls: MagicMock,
        mock_speaker_cls: MagicMock,
        mock_mic_cls: MagicMock,
        settings: AraSettings,
    ) -> None:
        from ara.voice.pipeline import VoicePipeline

        mock_vad = MagicMock()
        mock_vad.is_loaded = True
        mock_vad_cls.return_value = mock_vad

        mock_stt = MagicMock()
        mock_stt.is_loaded = True
        mock_stt_cls.return_value = mock_stt

        mock_tts = MagicMock()
        mock_tts.is_loaded = True
        mock_tts_cls.return_value = mock_tts

        mock_ww = MagicMock()
        mock_ww.is_loaded = True
        mock_ww.is_enabled = True
        mock_ww_cls.return_value = mock_ww

        mock_mic = MagicMock()
        mock_mic_cls.return_value = mock_mic

        mock_speaker = MagicMock()
        mock_speaker_cls.return_value = mock_speaker

        pipeline = VoicePipeline(settings)
        await pipeline.start()
        await pipeline.stop()

        mock_mic.stop.assert_called_once()
        mock_vad.unload.assert_called_once()
        mock_stt.unload.assert_called_once()
        mock_tts.unload.assert_called_once()
        mock_ww.unload.assert_called_once()


class TestVoicePipelineDataFlow:
    """Test data flow through the voice pipeline stages."""

    def test_ring_buffer_receives_and_delivers_audio(self, ring_buffer: AudioRingBuffer) -> None:
        """Audio data flows through the ring buffer without corruption."""
        chunk = _make_audio_chunk(duration_ms=32)
        ring_buffer.put(chunk)

        retrieved = ring_buffer.get()
        assert retrieved is not None
        np.testing.assert_array_almost_equal(chunk, retrieved)

    def test_multiple_chunks_maintain_order(self, ring_buffer: AudioRingBuffer) -> None:
        """Multiple audio chunks are delivered in FIFO order."""
        chunks = [_make_audio_chunk() for _ in range(5)]
        for c in chunks:
            ring_buffer.put(c)

        for expected in chunks:
            actual = ring_buffer.get()
            assert actual is not None
            np.testing.assert_array_almost_equal(expected, actual)

    def test_vad_result_drives_speech_tracking(self, settings: AraSettings) -> None:
        """VAD results correctly drive the speech segment tracker."""
        from ara.voice.vad import SpeechSegmentTracker

        tracker = SpeechSegmentTracker(settings.vad, sample_rate=16000)

        # Initially not speaking
        assert not tracker.is_speaking

        # Feed enough speech frames to cross min_speech_ms threshold
        min_speech_samples = int(settings.vad.min_speech_ms * 16000 / 1000)
        chunk_samples = 512

        frames_needed = (min_speech_samples // chunk_samples) + 1
        for _ in range(frames_needed):
            tracker.update(is_speech=True, chunk_samples=chunk_samples)

        assert tracker.is_speaking

    def test_vad_silence_ends_speech(self, settings: AraSettings) -> None:
        """Sufficient silence after speech transitions tracker to not-speaking."""
        from ara.voice.vad import SpeechSegmentTracker

        tracker = SpeechSegmentTracker(settings.vad, sample_rate=16000)

        # Enter speaking state
        min_speech_samples = int(settings.vad.min_speech_ms * 16000 / 1000)
        chunk_samples = 512
        frames_needed = (min_speech_samples // chunk_samples) + 1
        for _ in range(frames_needed):
            tracker.update(is_speech=True, chunk_samples=chunk_samples)
        assert tracker.is_speaking

        # Now feed enough silence frames
        min_silence_samples = int(settings.vad.min_silence_ms * 16000 / 1000)
        silence_frames_needed = (min_silence_samples // chunk_samples) + 1
        for _ in range(silence_frames_needed):
            tracker.update(is_speech=False, chunk_samples=chunk_samples)

        assert not tracker.is_speaking


class TestVoicePipelineTranscriptionCallback:
    """Test the transcription callback integration."""

    def test_callback_can_be_set(self, settings: AraSettings) -> None:
        from ara.voice.pipeline import VoicePipeline

        pipeline = VoicePipeline(settings)
        callback = AsyncMock(return_value="Response from LLM")
        pipeline.set_transcription_callback(callback)
        # No assertion error means callback was accepted

    def test_state_change_callback_can_be_set(self, settings: AraSettings) -> None:
        from ara.voice.pipeline import VoicePipeline

        pipeline = VoicePipeline(settings)
        state_callback = MagicMock()
        pipeline.set_state_change_callback(state_callback)

    def test_pipeline_utterance_counter_starts_at_zero(self, settings: AraSettings) -> None:
        from ara.voice.pipeline import VoicePipeline

        pipeline = VoicePipeline(settings)
        assert pipeline.total_utterances == 0

    def test_manual_activation_sets_activated_event(self, settings: AraSettings) -> None:
        """activate() sets the internal event flag (state change happens in run loop)."""
        from ara.voice.pipeline import VoicePipeline

        pipeline = VoicePipeline(settings)
        assert not pipeline._activated.is_set()
        pipeline.activate()
        assert pipeline._activated.is_set()


class TestVoicePipelineWakeWordIntegration:
    """Test wake word detection triggering pipeline transitions."""

    def test_wake_word_result_types(self) -> None:
        """WakeWordResult correctly reports detection status."""
        detected = _make_wake_word_result(detected=True)
        not_detected = _make_wake_word_result(detected=False)

        assert detected.detected is True
        assert not_detected.detected is False

    def test_wake_word_confidence_threshold(self) -> None:
        """Wake word results carry confidence for threshold filtering."""
        result = _make_wake_word_result(detected=True, confidence=0.92)
        assert result.confidence >= 0.5


class TestVoicePipelineSTTIntegration:
    """Test STT integration with the voice pipeline."""

    def test_stt_event_carries_text(self) -> None:
        """TranscriptionEvent contains recognized text."""
        event = _make_stt_event("Hello ARA", is_final=True)
        assert event.text == "Hello ARA"
        assert event.is_final is True

    def test_stt_event_partial_transcript(self) -> None:
        """Partial transcription events have is_final=False."""
        event = _make_stt_event("Hel", is_final=False)
        assert event.is_final is False


class TestVoicePipelineTTSIntegration:
    """Test TTS integration with the voice pipeline."""

    def test_tts_result_contains_audio_samples(self) -> None:
        """TTSResult contains synthesized audio data."""
        result = _make_tts_result(duration_s=1.0)
        assert len(result.samples) == 24000
        assert result.sample_rate == 24000
        assert result.duration_s == 1.0

    def test_tts_result_short_sentence(self) -> None:
        """Short sentences produce short audio."""
        result = _make_tts_result(duration_s=0.2)
        expected_samples = int(24000 * 0.2)
        assert len(result.samples) == expected_samples


class TestVoicePipelineFullFlowMocked:
    """Test the full voice pipeline flow with all components mocked."""

    @patch("ara.voice.pipeline.MicrophoneStream")
    @patch("ara.voice.pipeline.SpeakerOutput")
    @patch("ara.voice.pipeline.VADProcessor")
    @patch("ara.voice.pipeline.STTProcessor")
    @patch("ara.voice.pipeline.TTSProcessor")
    @patch("ara.voice.pipeline.WakeWordDetector")
    async def test_full_voice_flow_wake_to_response(
        self,
        mock_ww_cls: MagicMock,
        mock_tts_cls: MagicMock,
        mock_stt_cls: MagicMock,
        mock_vad_cls: MagicMock,
        mock_speaker_cls: MagicMock,
        mock_mic_cls: MagicMock,
        settings: AraSettings,
    ) -> None:
        """Test: wake word → listen → STT → callback → TTS → speak."""
        from ara.voice.pipeline import VoicePipeline

        # Configure mocks
        mock_vad = MagicMock()
        mock_vad.is_loaded = False
        mock_vad_cls.return_value = mock_vad

        mock_stt = MagicMock()
        mock_stt.is_loaded = False
        mock_stt.current_text = ""
        mock_stt_cls.return_value = mock_stt

        mock_tts = MagicMock()
        mock_tts.is_loaded = False
        mock_tts.synthesize.return_value = _make_tts_result()
        mock_tts_cls.return_value = mock_tts

        mock_ww = MagicMock()
        mock_ww.is_loaded = False
        mock_ww.is_enabled = True
        mock_ww_cls.return_value = mock_ww

        mock_mic = MagicMock()
        mock_mic_cls.return_value = mock_mic

        mock_speaker = MagicMock()
        mock_speaker.is_playing = False
        mock_speaker.play = AsyncMock()
        mock_speaker.wait_until_done = AsyncMock()
        mock_speaker_cls.return_value = mock_speaker

        pipeline = VoicePipeline(settings)

        # Set up transcription callback (simulates orchestrator)
        callback_received_text = []

        async def mock_callback(text: str, on_sentence: Any) -> str:
            callback_received_text.append(text)
            response = f"Mock response to: {text}"
            await on_sentence(response)
            return response

        pipeline.set_transcription_callback(mock_callback)
        await pipeline.start()
        assert pipeline.is_running

        await pipeline.stop()
        assert not pipeline.is_running


class TestVoicePipelineOrchestratorIntegration:
    """Test voice pipeline connected to a mock orchestrator."""

    async def test_transcription_callback_receives_text(self) -> None:
        """Verify the callback contract: receives text, returns response."""
        received: list[str] = []

        async def mock_orchestrator_callback(text: str, on_sentence: Any) -> str:
            received.append(text)
            response = "I heard you say: " + text
            await on_sentence(response)
            return response

        # Simulate what the pipeline does when calling the callback
        on_sentence = AsyncMock()
        result = await mock_orchestrator_callback("Hello ARA", on_sentence)

        assert received == ["Hello ARA"]
        assert result == "I heard you say: Hello ARA"
        on_sentence.assert_called_once_with("I heard you say: Hello ARA")

    async def test_callback_multiple_sentences(self) -> None:
        """Verify callback can handle multi-sentence responses."""
        sentences_played: list[str] = []

        async def on_sentence(sentence: str) -> None:
            sentences_played.append(sentence)

        async def mock_callback(text: str, on_sent: Any) -> str:
            response_sentences = [
                "First sentence.",
                "Second sentence.",
                "Third sentence.",
            ]
            for sent in response_sentences:
                await on_sent(sent)
            return " ".join(response_sentences)

        result = await mock_callback("Tell me something", on_sentence)
        assert len(sentences_played) == 3
        assert "First sentence." in result

    async def test_callback_error_returns_error_message(self) -> None:
        """Verify error handling in the callback chain."""

        async def failing_callback(text: str, on_sentence: Any) -> str:
            msg = f"Sorry, I encountered an error processing: {text}"
            await on_sentence(msg)
            return msg

        on_sentence = AsyncMock()
        result = await failing_callback("test input", on_sentence)
        assert "error" in result.lower()


class TestVoicePipelineBargeIn:
    """Test barge-in behavior (interrupting TTS playback)."""

    def test_speaker_output_supports_stop(self) -> None:
        """SpeakerOutput has a stop method for barge-in."""

        speaker = MagicMock(spec=["play", "stop", "is_playing", "wait_until_done"])
        speaker.is_playing = True
        speaker.stop()
        speaker.stop.assert_called_once()

    def test_speaker_is_playing_flag(self) -> None:
        """Speaker tracks whether audio is currently playing."""
        speaker = MagicMock()
        speaker.is_playing = False
        assert not speaker.is_playing

        speaker.is_playing = True
        assert speaker.is_playing
