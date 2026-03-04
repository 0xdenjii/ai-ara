"""Tests for voice pipeline components.

Tests focus on unit behavior with mocked audio devices and models.
Integration tests with real models are in tests/integration/.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from ara.config.settings import AudioSettings, TTSSettings, VADSettings, WakeWordSettings
from ara.utils.types import VoicePipelineState
from ara.voice.audio import AudioRingBuffer, MicrophoneStream, SpeakerOutput
from ara.voice.vad import SpeechSegmentTracker, VADProcessor, VADResult
from ara.voice.wake_word import WakeWordDetector, WakeWordResult

# ── AudioRingBuffer Tests ────────────────────────────────────────────────────


class TestAudioRingBuffer:
    def test_put_and_get(self) -> None:
        buf = AudioRingBuffer(max_chunks=10)
        chunk = np.zeros(480, dtype=np.float32)
        buf.put(chunk)
        assert buf.chunk_count == 1

        result = buf.get()
        assert result is not None
        assert len(result) == 480
        assert buf.chunk_count == 0

    def test_get_empty_returns_none(self) -> None:
        buf = AudioRingBuffer()
        assert buf.get() is None

    def test_max_chunks_overflow(self) -> None:
        buf = AudioRingBuffer(max_chunks=3)
        for i in range(5):
            chunk = np.full(100, fill_value=float(i), dtype=np.float32)
            buf.put(chunk)

        assert buf.chunk_count == 3
        # Oldest chunks (0, 1) should be dropped; remaining are 2, 3, 4
        first = buf.get()
        assert first is not None
        assert first[0] == 2.0

    def test_get_all_drains_buffer(self) -> None:
        buf = AudioRingBuffer()
        for _ in range(5):
            buf.put(np.zeros(100, dtype=np.float32))

        chunks = buf.get_all()
        assert len(chunks) == 5
        assert buf.chunk_count == 0

    def test_peek_does_not_remove(self) -> None:
        buf = AudioRingBuffer()
        buf.put(np.zeros(100, dtype=np.float32))
        buf.put(np.ones(100, dtype=np.float32))

        peeked = buf.peek(1)
        assert len(peeked) == 1
        assert buf.chunk_count == 2

    def test_clear(self) -> None:
        buf = AudioRingBuffer()
        for _ in range(10):
            buf.put(np.zeros(100, dtype=np.float32))
        buf.clear()
        assert buf.chunk_count == 0

    def test_put_copies_data(self) -> None:
        """Verify that put() copies data, not references."""
        buf = AudioRingBuffer()
        chunk = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        buf.put(chunk)
        chunk[0] = 999.0  # modify original

        result = buf.get()
        assert result is not None
        assert result[0] == 1.0  # buffer has the original copy

    @pytest.mark.asyncio
    async def test_wait_for_data_timeout(self) -> None:
        buf = AudioRingBuffer()
        loop = asyncio.get_running_loop()
        buf.set_event_loop(loop)

        result = await buf.wait_for_data(timeout=0.05)
        assert result is False

    @pytest.mark.asyncio
    async def test_wait_for_data_signals(self) -> None:
        buf = AudioRingBuffer()
        loop = asyncio.get_running_loop()
        buf.set_event_loop(loop)

        # Schedule a put after a short delay
        async def delayed_put() -> None:
            await asyncio.sleep(0.02)
            buf.put(np.zeros(100, dtype=np.float32))

        task = asyncio.create_task(delayed_put())
        result = await buf.wait_for_data(timeout=1.0)
        await task  # ensure task completes
        assert result is True


# ── MicrophoneStream Tests ───────────────────────────────────────────────────


class TestMicrophoneStream:
    def test_initial_state(self) -> None:
        settings = AudioSettings()
        mic = MicrophoneStream(settings)
        assert mic.is_running is False

    @patch("ara.voice.audio.sd.InputStream")
    def test_start_and_stop(self, mock_stream_cls: MagicMock) -> None:
        mock_stream = MagicMock()
        mock_stream_cls.return_value = mock_stream

        settings = AudioSettings()
        mic = MicrophoneStream(settings)
        buf = AudioRingBuffer()

        mic.start(buf)
        assert mic.is_running is True
        mock_stream.start.assert_called_once()

        mic.stop()
        assert mic.is_running is False
        mock_stream.stop.assert_called_once()
        mock_stream.close.assert_called_once()

    @patch("ara.voice.audio.sd.InputStream")
    def test_double_start_no_op(self, mock_stream_cls: MagicMock) -> None:
        mock_stream = MagicMock()
        mock_stream_cls.return_value = mock_stream

        settings = AudioSettings()
        mic = MicrophoneStream(settings)
        buf = AudioRingBuffer()

        mic.start(buf)
        mic.start(buf)  # should be a no-op
        assert mock_stream_cls.call_count == 1


# ── SpeakerOutput Tests ─────────────────────────────────────────────────────


class TestSpeakerOutput:
    def test_initial_state(self) -> None:
        settings = TTSSettings()
        speaker = SpeakerOutput(settings)
        assert speaker.is_playing is False

    @patch("ara.voice.audio.sd.play")
    @patch("ara.voice.audio.sd.wait")
    @pytest.mark.asyncio
    async def test_play(self, mock_wait: MagicMock, mock_play: MagicMock) -> None:
        settings = TTSSettings()
        speaker = SpeakerOutput(settings)

        samples = np.zeros(48000, dtype=np.float32)
        await speaker.play(samples, 24000)

        mock_play.assert_called_once()
        assert speaker.is_playing is False

    @patch("ara.voice.audio.sd.stop")
    def test_stop_interrupt(self, mock_stop: MagicMock) -> None:
        settings = TTSSettings()
        speaker = SpeakerOutput(settings)
        speaker.stop()
        mock_stop.assert_called_once()


# ── VAD Tests ────────────────────────────────────────────────────────────────


class TestVADResult:
    def test_create(self) -> None:
        result = VADResult(is_speech=True, confidence=0.95, latency_ms=0.5)
        assert result.is_speech is True
        assert result.confidence == 0.95
        assert result.latency_ms == 0.5


class TestSpeechSegmentTracker:
    def test_initial_state(self) -> None:
        settings = VADSettings()
        tracker = SpeechSegmentTracker(settings, sample_rate=16000)
        assert tracker.is_speaking is False

    def test_speech_start_after_threshold(self) -> None:
        settings = VADSettings(min_speech_ms=100, min_silence_ms=500)
        tracker = SpeechSegmentTracker(settings, sample_rate=16000)

        # Feed speech chunks (100ms = 1600 samples at 16kHz)
        # First chunk: not enough accumulated speech
        changed = tracker.update(is_speech=True, chunk_samples=800)
        assert changed is False
        assert tracker.is_speaking is False

        # Second chunk: crosses 100ms threshold
        changed = tracker.update(is_speech=True, chunk_samples=800)
        assert changed is True
        assert tracker.is_speaking is True

    def test_silence_ends_speech(self) -> None:
        settings = VADSettings(min_speech_ms=100, min_silence_ms=200)
        tracker = SpeechSegmentTracker(settings, sample_rate=16000)

        # Start speaking
        tracker.update(is_speech=True, chunk_samples=1600)
        assert tracker.is_speaking is True

        # Short silence — not enough to end
        tracker.update(is_speech=False, chunk_samples=1600)
        assert tracker.is_speaking is True

        # More silence — crosses threshold
        changed = tracker.update(is_speech=False, chunk_samples=1600)
        assert changed is True
        assert tracker.is_speaking is False

    def test_reset(self) -> None:
        settings = VADSettings(min_speech_ms=100, min_silence_ms=200)
        tracker = SpeechSegmentTracker(settings, sample_rate=16000)
        tracker.update(is_speech=True, chunk_samples=1600)
        tracker.reset()
        assert tracker.is_speaking is False


class TestVADProcessor:
    def test_not_loaded(self) -> None:
        settings = VADSettings()
        vad = VADProcessor(settings)
        assert vad.is_loaded is False

    def test_process_without_load_raises(self) -> None:
        from ara.utils.errors import VoiceError

        settings = VADSettings()
        vad = VADProcessor(settings)

        with pytest.raises(VoiceError, match="not loaded"):
            vad.process(np.zeros(512, dtype=np.float32))


# ── WakeWord Tests ───────────────────────────────────────────────────────────


class TestWakeWordResult:
    def test_create(self) -> None:
        result = WakeWordResult(
            detected=True, model_name="hey_jarvis_v0.1", confidence=0.95, latency_ms=1.2
        )
        assert result.detected is True
        assert result.model_name == "hey_jarvis_v0.1"


class TestWakeWordDetector:
    def test_disabled_returns_no_detection(self) -> None:
        settings = WakeWordSettings(enabled=False)
        detector = WakeWordDetector(settings)

        result = detector.process(np.zeros(1280, dtype=np.float32))
        assert result.detected is False

    def test_not_loaded_raises(self) -> None:
        from ara.utils.errors import VoiceError

        settings = WakeWordSettings(enabled=True)
        detector = WakeWordDetector(settings)

        with pytest.raises(VoiceError, match="not loaded"):
            detector.process(np.zeros(1280, dtype=np.float32))

    def test_disabled_load_no_op(self) -> None:
        settings = WakeWordSettings(enabled=False)
        detector = WakeWordDetector(settings)
        detector.load()  # should not raise
        assert detector.is_loaded is False


# ── VoicePipeline Tests ─────────────────────────────────────────────────────


def _make_mock_settings() -> MagicMock:
    """Create a mock AraSettings with all voice sub-settings."""
    settings = MagicMock()
    settings.audio = AudioSettings()
    settings.tts = TTSSettings()
    settings.vad = VADSettings()
    settings.stt = MagicMock()
    settings.stt.model_path = ""
    settings.stt.model_arch = 5
    settings.wake_word = WakeWordSettings(enabled=False)
    return settings


class TestVoicePipelineInit:
    """Tests for VoicePipeline construction and initial state."""

    def test_initial_state(self) -> None:
        from ara.voice.pipeline import VoicePipeline

        settings = _make_mock_settings()
        pipeline = VoicePipeline(settings)

        assert pipeline.state == VoicePipelineState.IDLE
        assert pipeline.is_running is False
        assert pipeline.total_utterances == 0

    def test_components_created(self) -> None:
        from ara.voice.pipeline import VoicePipeline

        settings = _make_mock_settings()
        pipeline = VoicePipeline(settings)

        assert pipeline.vad is not None
        assert pipeline.stt is not None
        assert pipeline.tts is not None
        assert pipeline.wake_word is not None
        assert pipeline.mic is not None
        assert pipeline.speaker is not None
        assert pipeline.ring_buffer is not None

    def test_set_transcription_callback(self) -> None:
        from ara.voice.pipeline import VoicePipeline

        settings = _make_mock_settings()
        pipeline = VoicePipeline(settings)

        callback = AsyncMock(return_value="response")
        pipeline.set_transcription_callback(callback)
        # No assertion on private — just verify no error
        assert pipeline.state == VoicePipelineState.IDLE

    def test_set_state_change_callback(self) -> None:
        from ara.voice.pipeline import VoicePipeline

        settings = _make_mock_settings()
        pipeline = VoicePipeline(settings)

        callback = MagicMock()
        pipeline.set_state_change_callback(callback)
        assert pipeline.state == VoicePipelineState.IDLE


class TestVoicePipelineLifecycle:
    """Tests for start/stop lifecycle."""

    @pytest.mark.asyncio
    @patch("ara.voice.pipeline.MicrophoneStream")
    @patch("ara.voice.pipeline.WakeWordDetector")
    @patch("ara.voice.pipeline.TTSProcessor")
    @patch("ara.voice.pipeline.STTProcessor")
    @patch("ara.voice.pipeline.VADProcessor")
    async def test_start_loads_models_and_mic(
        self,
        mock_vad_cls: MagicMock,
        mock_stt_cls: MagicMock,
        mock_tts_cls: MagicMock,
        mock_ww_cls: MagicMock,
        mock_mic_cls: MagicMock,
    ) -> None:
        from ara.voice.pipeline import VoicePipeline

        mock_vad = MagicMock()
        mock_stt = MagicMock()
        mock_tts = MagicMock()
        mock_ww = MagicMock()
        mock_mic = MagicMock()

        mock_vad_cls.return_value = mock_vad
        mock_stt_cls.return_value = mock_stt
        mock_tts_cls.return_value = mock_tts
        mock_ww_cls.return_value = mock_ww
        mock_mic_cls.return_value = mock_mic

        settings = _make_mock_settings()
        pipeline = VoicePipeline(settings)

        await pipeline.start()

        assert pipeline.is_running is True
        mock_vad.load.assert_called_once()
        mock_stt.load.assert_called_once()
        mock_tts.load.assert_called_once()
        mock_ww.load.assert_called_once()
        mock_mic.start.assert_called_once()

        await pipeline.stop()
        assert pipeline.is_running is False

    @pytest.mark.asyncio
    @patch("ara.voice.pipeline.MicrophoneStream")
    @patch("ara.voice.pipeline.WakeWordDetector")
    @patch("ara.voice.pipeline.TTSProcessor")
    @patch("ara.voice.pipeline.STTProcessor")
    @patch("ara.voice.pipeline.VADProcessor")
    async def test_stop_unloads_all(
        self,
        mock_vad_cls: MagicMock,
        mock_stt_cls: MagicMock,
        mock_tts_cls: MagicMock,
        mock_ww_cls: MagicMock,
        mock_mic_cls: MagicMock,
    ) -> None:
        from ara.voice.pipeline import VoicePipeline

        mock_vad = MagicMock()
        mock_stt = MagicMock()
        mock_tts = MagicMock()
        mock_ww = MagicMock()
        mock_mic = MagicMock()

        mock_vad_cls.return_value = mock_vad
        mock_stt_cls.return_value = mock_stt
        mock_tts_cls.return_value = mock_tts
        mock_ww_cls.return_value = mock_ww
        mock_mic_cls.return_value = mock_mic

        settings = _make_mock_settings()
        pipeline = VoicePipeline(settings)
        await pipeline.start()
        await pipeline.stop()

        mock_vad.unload.assert_called_once()
        mock_stt.unload.assert_called_once()
        mock_tts.unload.assert_called_once()
        mock_ww.unload.assert_called_once()
        mock_mic.stop.assert_called_once()

    @pytest.mark.asyncio
    @patch("ara.voice.pipeline.MicrophoneStream")
    @patch("ara.voice.pipeline.WakeWordDetector")
    @patch("ara.voice.pipeline.TTSProcessor")
    @patch("ara.voice.pipeline.STTProcessor")
    @patch("ara.voice.pipeline.VADProcessor")
    async def test_double_start_no_op(
        self,
        mock_vad_cls: MagicMock,
        mock_stt_cls: MagicMock,
        mock_tts_cls: MagicMock,
        mock_ww_cls: MagicMock,
        mock_mic_cls: MagicMock,
    ) -> None:
        from ara.voice.pipeline import VoicePipeline

        settings = _make_mock_settings()
        pipeline = VoicePipeline(settings)
        await pipeline.start()
        await pipeline.start()  # second call should be a no-op

        # Only one load per component
        pipeline._vad.load.assert_called_once()
        await pipeline.stop()

    @pytest.mark.asyncio
    async def test_stop_when_not_running_is_no_op(self) -> None:
        from ara.voice.pipeline import VoicePipeline

        settings = _make_mock_settings()
        pipeline = VoicePipeline(settings)
        await pipeline.stop()  # should not raise
        assert pipeline.is_running is False


class TestVoicePipelineRun:
    """Tests for the main run loop and state transitions."""

    @pytest.mark.asyncio
    async def test_run_without_start_raises(self) -> None:
        from ara.utils.errors import VoiceError
        from ara.voice.pipeline import VoicePipeline

        settings = _make_mock_settings()
        pipeline = VoicePipeline(settings)

        with pytest.raises(VoiceError, match="not started"):
            await pipeline.run()

    @pytest.mark.asyncio
    @patch("ara.voice.pipeline.MicrophoneStream")
    @patch("ara.voice.pipeline.WakeWordDetector")
    @patch("ara.voice.pipeline.TTSProcessor")
    @patch("ara.voice.pipeline.STTProcessor")
    @patch("ara.voice.pipeline.VADProcessor")
    async def test_run_loop_exits_on_stop(
        self,
        mock_vad_cls: MagicMock,
        mock_stt_cls: MagicMock,
        mock_tts_cls: MagicMock,
        mock_ww_cls: MagicMock,
        mock_mic_cls: MagicMock,
    ) -> None:
        from ara.voice.pipeline import VoicePipeline

        settings = _make_mock_settings()
        pipeline = VoicePipeline(settings)
        await pipeline.start()

        # Stop the pipeline after a brief delay
        async def delayed_stop() -> None:
            await asyncio.sleep(0.05)
            await pipeline.stop()

        stop_task = asyncio.create_task(delayed_stop())
        await pipeline.run()  # should exit when stopped
        await stop_task

        assert pipeline.is_running is False


class TestVoicePipelineActivation:
    """Tests for manual activation (hotkey path)."""

    def test_activate_sets_event(self) -> None:
        from ara.voice.pipeline import VoicePipeline

        settings = _make_mock_settings()
        pipeline = VoicePipeline(settings)

        # Not idle, activate should be ignored
        pipeline._state = VoicePipelineState.LISTENING
        pipeline.activate()
        assert not pipeline._activated.is_set()

        # When idle, activate should set the event
        pipeline._state = VoicePipelineState.IDLE
        pipeline.activate()
        assert pipeline._activated.is_set()

    @pytest.mark.asyncio
    @patch("ara.voice.pipeline.MicrophoneStream")
    @patch("ara.voice.pipeline.WakeWordDetector")
    @patch("ara.voice.pipeline.TTSProcessor")
    @patch("ara.voice.pipeline.STTProcessor")
    @patch("ara.voice.pipeline.VADProcessor")
    async def test_manual_activation_transitions_to_listening(
        self,
        mock_vad_cls: MagicMock,
        mock_stt_cls: MagicMock,
        mock_tts_cls: MagicMock,
        mock_ww_cls: MagicMock,
        mock_mic_cls: MagicMock,
    ) -> None:
        from ara.voice.pipeline import VoicePipeline

        settings = _make_mock_settings()
        pipeline = VoicePipeline(settings)
        await pipeline.start()

        # Put audio in buffer so _handle_idle processes it
        pipeline.ring_buffer.set_event_loop(asyncio.get_running_loop())
        pipeline.ring_buffer.put(np.zeros(480, dtype=np.float32))

        # Set manual activation
        pipeline.activate()

        # Run one iteration of idle handler
        await pipeline._handle_idle()

        assert pipeline.state == VoicePipelineState.LISTENING
        await pipeline.stop()


class TestVoicePipelineStateChange:
    """Tests for state change callback mechanism."""

    def test_state_change_callback_invoked(self) -> None:
        from ara.voice.pipeline import VoicePipeline

        settings = _make_mock_settings()
        pipeline = VoicePipeline(settings)

        states_seen: list[VoicePipelineState] = []
        pipeline.set_state_change_callback(states_seen.append)

        pipeline._set_state(VoicePipelineState.LISTENING)
        pipeline._set_state(VoicePipelineState.PROCESSING)
        pipeline._set_state(VoicePipelineState.IDLE)

        assert states_seen == [
            VoicePipelineState.LISTENING,
            VoicePipelineState.PROCESSING,
            VoicePipelineState.IDLE,
        ]

    def test_duplicate_state_not_notified(self) -> None:
        from ara.voice.pipeline import VoicePipeline

        settings = _make_mock_settings()
        pipeline = VoicePipeline(settings)

        states_seen: list[VoicePipelineState] = []
        pipeline.set_state_change_callback(states_seen.append)

        pipeline._set_state(VoicePipelineState.LISTENING)
        pipeline._set_state(VoicePipelineState.LISTENING)  # duplicate — no callback

        assert len(states_seen) == 1

    def test_state_change_callback_error_does_not_crash(self) -> None:
        from ara.voice.pipeline import VoicePipeline

        settings = _make_mock_settings()
        pipeline = VoicePipeline(settings)

        def bad_callback(_state: VoicePipelineState) -> None:
            msg = "boom"
            raise RuntimeError(msg)

        pipeline.set_state_change_callback(bad_callback)

        # Should not raise
        pipeline._set_state(VoicePipelineState.LISTENING)
        assert pipeline.state == VoicePipelineState.LISTENING


class TestVoicePipelineProcessing:
    """Tests for the utterance processing flow."""

    @pytest.mark.asyncio
    async def test_process_utterance_no_callback(self) -> None:
        """Without a transcription callback, pipeline returns to IDLE."""
        from ara.voice.pipeline import VoicePipeline

        settings = _make_mock_settings()
        pipeline = VoicePipeline(settings)
        pipeline._state = VoicePipelineState.PROCESSING

        await pipeline._process_utterance("hello")

        assert pipeline.state == VoicePipelineState.IDLE
        assert pipeline.total_utterances == 1

    @pytest.mark.asyncio
    @patch("ara.voice.pipeline.SpeakerOutput")
    @patch("ara.voice.pipeline.TTSProcessor")
    async def test_process_utterance_with_callback(
        self,
        mock_tts_cls: MagicMock,
        mock_speaker_cls: MagicMock,
    ) -> None:
        """With a callback, pipeline processes text and returns to IDLE."""
        from ara.voice.pipeline import VoicePipeline

        mock_tts = MagicMock()
        mock_tts.synthesize.return_value = MagicMock(
            samples=np.zeros(24000, dtype=np.float32),
            sample_rate=24000,
        )
        mock_tts_cls.return_value = mock_tts

        mock_speaker = MagicMock()
        mock_speaker.play = AsyncMock()
        mock_speaker.is_playing = False
        mock_speaker_cls.return_value = mock_speaker

        settings = _make_mock_settings()
        pipeline = VoicePipeline(settings)
        pipeline._state = VoicePipelineState.PROCESSING
        pipeline._running = True

        sentences_received: list[str] = []

        async def mock_llm_handler(
            text: str,
            on_sentence: object,
        ) -> str:
            # Simulate LLM calling on_sentence for each sentence
            assert callable(on_sentence)
            await on_sentence("Hello there.")  # type: ignore[misc]
            sentences_received.append("Hello there.")
            return "Hello there."

        pipeline.set_transcription_callback(mock_llm_handler)
        await pipeline._process_utterance("hi")

        assert pipeline.total_utterances == 1
        assert len(sentences_received) == 1
        assert sentences_received[0] == "Hello there."

    @pytest.mark.asyncio
    async def test_process_utterance_callback_error_handled(self) -> None:
        """Errors in the transcription callback don't crash the pipeline."""
        from ara.voice.pipeline import VoicePipeline

        settings = _make_mock_settings()
        pipeline = VoicePipeline(settings)
        pipeline._state = VoicePipelineState.PROCESSING
        pipeline._running = True

        async def failing_callback(
            _text: str,
            _on_sentence: object,
        ) -> str:
            msg = "LLM exploded"
            raise RuntimeError(msg)

        pipeline.set_transcription_callback(failing_callback)
        # Should not raise
        await pipeline._process_utterance("hello")
        assert pipeline.total_utterances == 1
        assert pipeline.state == VoicePipelineState.IDLE


class TestVoicePipelineSpeaking:
    """Tests for the SPEAKING state handler."""

    @pytest.mark.asyncio
    async def test_speaking_returns_to_idle_when_done(self) -> None:
        from ara.voice.pipeline import VoicePipeline

        settings = _make_mock_settings()
        pipeline = VoicePipeline(settings)
        pipeline._state = VoicePipelineState.SPEAKING
        pipeline._running = True

        # Mock speaker as not playing
        pipeline._speaker = MagicMock()
        pipeline._speaker.is_playing = False

        await pipeline._handle_speaking()
        assert pipeline.state == VoicePipelineState.IDLE

    @pytest.mark.asyncio
    async def test_barge_in_during_speaking(self) -> None:
        from ara.voice.pipeline import VoicePipeline

        settings = _make_mock_settings()
        pipeline = VoicePipeline(settings)
        pipeline._state = VoicePipelineState.SPEAKING
        pipeline._running = True

        # Mock speaker as currently playing
        pipeline._speaker = MagicMock()
        pipeline._speaker.is_playing = True
        pipeline._speaker.stop = MagicMock()

        # Mock VAD detecting speech (barge-in)
        pipeline._vad = MagicMock()
        pipeline._vad.process.return_value = VADResult(
            is_speech=True,
            confidence=0.9,
            latency_ms=0.5,
        )

        # Put audio in buffer and set event loop
        loop = asyncio.get_running_loop()
        pipeline._ring_buffer.set_event_loop(loop)
        pipeline._ring_buffer.put(np.zeros(512, dtype=np.float32))

        await pipeline._handle_speaking()

        pipeline._speaker.stop.assert_called_once()
        assert pipeline.state == VoicePipelineState.LISTENING


# ── HotkeyListener Tests ────────────────────────────────────────────────────


class TestHotkeyListenerInit:
    """Tests for HotkeyListener construction and properties."""

    def test_initial_state(self) -> None:
        from ara.voice.hotkey import HotkeyListener

        listener = HotkeyListener(hotkey="ctrl+space")
        assert listener.hotkey == "ctrl+space"
        assert listener.is_registered is False

    def test_custom_hotkey(self) -> None:
        from ara.voice.hotkey import HotkeyListener

        listener = HotkeyListener(hotkey="ctrl+shift+a")
        assert listener.hotkey == "ctrl+shift+a"


class TestHotkeyListenerLifecycle:
    """Tests for start/stop lifecycle."""

    def test_start_without_callback_raises(self) -> None:
        from ara.utils.errors import VoiceError
        from ara.voice.hotkey import HotkeyListener

        listener = HotkeyListener(hotkey="ctrl+space")

        with pytest.raises(VoiceError, match="No callback set"):
            listener.start()

    @patch("ara.voice.hotkey.keyboard")
    def test_start_registers_hotkey(self, mock_kb: MagicMock) -> None:
        from ara.voice.hotkey import HotkeyListener

        listener = HotkeyListener(hotkey="ctrl+space")
        listener.set_callback(lambda: None)
        listener.start()

        assert listener.is_registered is True
        mock_kb.add_hotkey.assert_called_once()
        # Verify the hotkey string was passed
        call_args = mock_kb.add_hotkey.call_args
        assert call_args[0][0] == "ctrl+space"

    @patch("ara.voice.hotkey.keyboard")
    def test_double_start_no_op(self, mock_kb: MagicMock) -> None:
        from ara.voice.hotkey import HotkeyListener

        listener = HotkeyListener(hotkey="ctrl+space")
        listener.set_callback(lambda: None)
        listener.start()
        listener.start()  # second call should be no-op

        assert mock_kb.add_hotkey.call_count == 1

    @patch("ara.voice.hotkey.keyboard")
    def test_stop_unregisters_hotkey(self, mock_kb: MagicMock) -> None:
        from ara.voice.hotkey import HotkeyListener

        listener = HotkeyListener(hotkey="ctrl+space")
        listener.set_callback(lambda: None)
        listener.start()
        listener.stop()

        assert listener.is_registered is False
        mock_kb.remove_hotkey.assert_called_once_with("ctrl+space")

    def test_stop_when_not_registered_is_no_op(self) -> None:
        from ara.voice.hotkey import HotkeyListener

        listener = HotkeyListener(hotkey="ctrl+space")
        listener.stop()  # should not raise
        assert listener.is_registered is False


class TestHotkeyListenerCallback:
    """Tests for hotkey callback invocation."""

    @patch("ara.voice.hotkey.keyboard")
    def test_callback_invoked_on_press(self, mock_kb: MagicMock) -> None:
        from ara.voice.hotkey import HotkeyListener

        called = []
        listener = HotkeyListener(hotkey="ctrl+space")
        listener.set_callback(lambda: called.append(True))
        listener.start()

        # Simulate hotkey press by calling the internal handler
        listener._on_hotkey_pressed()
        assert len(called) == 1

    @patch("ara.voice.hotkey.keyboard")
    def test_callback_error_does_not_crash(self, mock_kb: MagicMock) -> None:
        from ara.voice.hotkey import HotkeyListener

        def bad_callback() -> None:
            msg = "boom"
            raise RuntimeError(msg)

        listener = HotkeyListener(hotkey="ctrl+space")
        listener.set_callback(bad_callback)
        listener.start()

        # Should not raise
        listener._on_hotkey_pressed()
        assert listener.is_registered is True

    @patch("ara.voice.hotkey.keyboard")
    def test_suppress_blocks_callback(self, mock_kb: MagicMock) -> None:
        from ara.voice.hotkey import HotkeyListener

        called = []
        listener = HotkeyListener(hotkey="ctrl+space")
        listener.set_callback(lambda: called.append(True))
        listener.start()

        listener.suppress()
        listener._on_hotkey_pressed()
        assert len(called) == 0  # suppressed — not called

        listener.unsuppress()
        listener._on_hotkey_pressed()
        assert len(called) == 1  # unsuppressed — called


class TestHotkeyListenerUpdate:
    """Tests for runtime hotkey changes."""

    @patch("ara.voice.hotkey.keyboard")
    def test_update_hotkey_while_registered(self, mock_kb: MagicMock) -> None:
        from ara.voice.hotkey import HotkeyListener

        listener = HotkeyListener(hotkey="ctrl+space")
        listener.set_callback(lambda: None)
        listener.start()

        listener.update_hotkey("ctrl+shift+space")

        assert listener.hotkey == "ctrl+shift+space"
        assert listener.is_registered is True
        # Should have been re-registered with new hotkey
        assert mock_kb.add_hotkey.call_count == 2

    def test_update_hotkey_while_not_registered(self) -> None:
        from ara.voice.hotkey import HotkeyListener

        listener = HotkeyListener(hotkey="ctrl+space")
        listener.update_hotkey("ctrl+shift+space")

        assert listener.hotkey == "ctrl+shift+space"
        assert listener.is_registered is False
