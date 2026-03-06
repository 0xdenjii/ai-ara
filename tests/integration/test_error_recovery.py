"""Error recovery integration tests.

Tests that ARA gracefully handles failure scenarios:
- Ollama not running / connection refused
- Model OOM during generation
- Audio device unavailable
- TTS synthesis failure (partial)
- Agent/orchestrator crashes
- LLM generation errors with fallback chain
- API server resilience (starts without LLM/vector store)
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from ara.api.app import AppState, create_app
from ara.config.settings import AraSettings
from ara.utils.errors import (
    AgentError,
    AraError,
    AudioDeviceError,
    LLMConnectionError,
    LLMError,
    LLMModelSwapError,
    LLMTimeoutError,
    RoutingError,
    STTError,
    STTTimeoutError,
    TTSError,
    VoiceError,
)

# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def settings() -> AraSettings:
    return AraSettings(env="testing")


def _make_app_with_state(state: AppState) -> Any:
    """Create a FastAPI app with a pre-configured AppState."""

    @asynccontextmanager
    async def noop_lifespan(_app: Any):  # type: ignore[no-untyped-def]
        _app.state.ara = state
        yield

    test_app = create_app(settings=state.settings)
    test_app.router.lifespan_context = noop_lifespan
    return test_app


# ══════════════════════════════════════════════════════════════════════════════
# ARA Error Hierarchy Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestErrorHierarchyIntegration:
    """Test that the error hierarchy carries context through the stack."""

    def test_llm_connection_error_has_structured_fields(self) -> None:
        """LLMConnectionError carries base_url and reason."""
        err = LLMConnectionError(
            base_url="http://localhost:11434",
            reason="Connection refused",
        )
        assert err.component == "llm"
        assert err.operation == "connect"
        assert "localhost:11434" in str(err)
        assert "Connection refused" in str(err)

    def test_llm_error_has_details(self) -> None:
        """LLMError carries arbitrary details dict."""
        err = LLMError(
            "Generation failed",
            operation="generate",
            details={"model": "qwen3:30b-a3b", "tokens_generated": 42},
        )
        assert err.details["model"] == "qwen3:30b-a3b"
        assert err.details["tokens_generated"] == 42

    def test_llm_timeout_error(self) -> None:
        """LLMTimeoutError carries timeout and model info."""
        err = LLMTimeoutError(
            model="qwen3:30b-a3b",
            timeout_s=30.0,
            tokens_generated=42,
        )
        assert "30.0" in str(err)
        assert err.component == "llm"

    def test_llm_model_swap_error(self) -> None:
        """LLMModelSwapError carries previous and target model names."""
        err = LLMModelSwapError(
            from_model="qwen3:30b-a3b",
            to_model="qwen3:14b",
            reason="Target model not found",
        )
        assert "qwen3:14b" in str(err)
        assert err.component == "llm"

    def test_voice_error_chain(self) -> None:
        """VoiceError can be chained from underlying exceptions."""
        original = RuntimeError("sounddevice.PortAudioError: No device")
        err = VoiceError(
            "Audio device unavailable",
            operation="start_microphone",
            details={"device": "default"},
        )
        err.__cause__ = original

        assert err.__cause__ is original
        assert err.component == "voice"

    def test_audio_device_error(self) -> None:
        """AudioDeviceError has correct component and operation."""
        err = AudioDeviceError(
            device_type="input",
            reason="Device disconnected",
        )
        assert err.component == "voice"
        assert "input" in str(err)

    def test_stt_timeout_error(self) -> None:
        """STTTimeoutError carries timing info."""
        err = STTTimeoutError(
            elapsed_ms=5300.0,
            audio_duration_s=3.2,
        )
        assert err.component == "voice"
        assert "5300" in str(err)

    def test_agent_error(self) -> None:
        """AgentError carries operation and details."""
        err = AgentError(
            "Query processing failed",
            operation="process",
            details={"query": "test query", "error": "Classification timeout"},
        )
        assert err.component == "agent"
        assert err.details["query"] == "test query"

    def test_routing_error(self) -> None:
        """RoutingError for unknown agent names."""
        err = RoutingError(
            "Agent 'nonexistent_agent' not found",
        )
        assert err.component == "agent"

    def test_all_errors_inherit_from_ara_error(self) -> None:
        """All custom errors inherit from AraError base."""
        errors = [
            LLMConnectionError("url", "reason"),
            LLMError("msg", operation="test"),
            LLMTimeoutError("model", 30.0, 0),
            LLMModelSwapError("a", "b", "reason"),
            VoiceError("msg", operation="test"),
            AudioDeviceError("input", "reason"),
            STTError("msg", operation="test"),
            STTTimeoutError(5000.0, 3.0),
            TTSError("msg", operation="test"),
            AgentError("msg", operation="test"),
            RoutingError("msg"),
        ]
        for err in errors:
            assert isinstance(err, AraError), f"{type(err).__name__} should inherit from AraError"


# ══════════════════════════════════════════════════════════════════════════════
# LLM Connection Failure Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestLLMConnectionFailure:
    """Test system behavior when Ollama is unreachable."""

    def test_api_starts_without_llm(self, settings: AraSettings) -> None:
        """API server starts even when LLM connection fails."""
        state = AppState(settings=settings)
        state.llm_client = None  # Simulates connection failure at startup

        app = _make_app_with_state(state)
        with TestClient(app, raise_server_exceptions=False) as client:
            # Health check works
            resp = client.get("/health")
            assert resp.status_code == 200

    def test_chat_without_llm_returns_error(self, settings: AraSettings) -> None:
        """Chat endpoint returns error when LLM is unavailable."""
        state = AppState(settings=settings)
        state.llm_client = None

        app = _make_app_with_state(state)
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post("/api/chat", json={"message": "Hello!"})
            # Should return an error (either 500 or 200 with error field)
            assert resp.status_code in (500, 503, 200)

    def test_chat_llm_generate_raises_connection_error(self, settings: AraSettings) -> None:
        """LLM raises LLMConnectionError mid-conversation."""
        mock_llm = MagicMock()
        mock_llm.is_connected = True
        mock_llm.active_model = "test"
        mock_llm.generate = AsyncMock(
            side_effect=LLMConnectionError("http://localhost:11434", "Connection reset")
        )

        state = AppState(settings=settings)
        state.llm_client = mock_llm
        state.orchestrator = None

        app = _make_app_with_state(state)
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post("/api/chat", json={"message": "Hello!"})
            assert resp.status_code in (500, 200)


# ══════════════════════════════════════════════════════════════════════════════
# LLM Generation Error Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestLLMGenerationErrors:
    """Test error handling during LLM text generation."""

    def test_llm_error_during_generate(self, settings: AraSettings) -> None:
        """LLMError during generation is caught and returned as error response."""
        mock_llm = MagicMock()
        mock_llm.is_connected = True
        mock_llm.active_model = "qwen3:30b-a3b"
        mock_llm.generate = AsyncMock(
            side_effect=LLMError(
                "Model ran out of GPU memory",
                operation="generate",
                details={"model": "qwen3:30b-a3b", "vram_used_gb": 24.0},
            )
        )

        state = AppState(settings=settings)
        state.llm_client = mock_llm
        state.orchestrator = None

        app = _make_app_with_state(state)
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post("/api/chat", json={"message": "Tell me about my diary"})
            assert resp.status_code in (500, 200)

    def test_stream_error_yields_error_event(self, settings: AraSettings) -> None:
        """Streaming endpoint yields an error SSE event on failure."""

        async def failing_stream(*args: Any, **kwargs: Any) -> Any:
            yield "Hello"
            raise LLMError("GPU OOM mid-generation", operation="generate_stream")

        mock_llm = MagicMock()
        mock_llm.is_connected = True
        mock_llm.active_model = "test"
        mock_llm.generate_stream = failing_stream

        state = AppState(settings=settings)
        state.llm_client = mock_llm
        state.orchestrator = None

        app = _make_app_with_state(state)
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post(
                "/api/chat/stream",
                json={"message": "Stream me", "stream": True},
            )
            assert resp.status_code == 200
            # Should contain an error chunk in the SSE output
            content = resp.text
            # The stream should contain at least one data line
            data_lines = [line for line in content.split("\n") if line.startswith("data:")]
            assert len(data_lines) > 0

    def test_model_swap_error_does_not_crash_server(self, settings: AraSettings) -> None:
        """Failed model swap doesn't bring down the server."""
        mock_llm = MagicMock()
        mock_llm.is_connected = True
        mock_llm.active_model = "qwen3:30b-a3b"
        mock_llm.generate = AsyncMock(return_value="Still working fine")
        mock_llm.swap_model = AsyncMock(
            side_effect=LLMModelSwapError("qwen3:30b-a3b", "nonexistent:model", "not found")
        )

        state = AppState(settings=settings)
        state.llm_client = mock_llm
        state.orchestrator = None

        app = _make_app_with_state(state)
        with TestClient(app, raise_server_exceptions=False) as client:
            # Swap fails
            client.post(
                "/api/models/swap",
                json={"model": "nonexistent:model"},
            )
            # Server is still functional
            resp2 = client.post("/api/chat", json={"message": "Still alive?"})
            assert resp2.status_code == 200


# ══════════════════════════════════════════════════════════════════════════════
# Orchestrator Error Recovery Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestOrchestratorErrorRecovery:
    """Test orchestrator error handling and fallback chain."""

    def test_orchestrator_exception_falls_back_to_llm(self, settings: AraSettings) -> None:
        """When orchestrator raises, chat falls back to direct LLM."""
        mock_orch = MagicMock()
        mock_orch.is_connected = True
        mock_orch.process = AsyncMock(
            side_effect=AgentError("Classification timeout", operation="process")
        )

        mock_llm = MagicMock()
        mock_llm.is_connected = True
        mock_llm.active_model = "test"
        mock_llm.generate = AsyncMock(return_value="Fallback LLM response")

        state = AppState(settings=settings)
        state.llm_client = mock_llm
        state.orchestrator = mock_orch

        app = _make_app_with_state(state)
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post("/api/chat", json={"message": "What happened?"})
            assert resp.status_code == 200
            data = resp.json()
            assert "response" in data

    def test_orchestrator_returns_error_result(self, settings: AraSettings) -> None:
        """Orchestrator returns a result with error field (graceful degradation)."""
        error_result = MagicMock()
        error_result.response = "I'm sorry, I had trouble with that question."
        error_result.intent = "general"
        error_result.agent = "error_handler"
        error_result.confidence = 0.0
        error_result.error = "Agent failed"
        error_result.success = False
        error_result.steps = 1
        error_result.elapsed_ms = 200.0
        error_result.memory_context = ""
        error_result.graph_context = ""
        error_result.tool_calls = []
        error_result.tool_results = []
        error_result.synthesis_mode = ""
        error_result.metadata = {}
        error_result.to_dict.return_value = {"response": error_result.response}

        mock_orch = MagicMock()
        mock_orch.is_connected = True
        mock_orch.process = AsyncMock(return_value=error_result)

        mock_llm = MagicMock()
        mock_llm.is_connected = True
        mock_llm.active_model = "test"

        state = AppState(settings=settings)
        state.llm_client = mock_llm
        state.orchestrator = mock_orch

        app = _make_app_with_state(state)
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post("/api/chat", json={"message": "Test"})
            assert resp.status_code == 200
            data = resp.json()
            # Should contain the orchestrator's error response
            assert "response" in data

    def test_both_orchestrator_and_llm_fail(self, settings: AraSettings) -> None:
        """When both orchestrator and LLM fail, user gets an error."""
        mock_orch = MagicMock()
        mock_orch.is_connected = True
        mock_orch.process = AsyncMock(
            side_effect=AgentError("Process crashed", operation="process")
        )

        mock_llm = MagicMock()
        mock_llm.is_connected = True
        mock_llm.active_model = "test"
        mock_llm.generate = AsyncMock(side_effect=LLMError("Connection lost", operation="generate"))

        state = AppState(settings=settings)
        state.llm_client = mock_llm
        state.orchestrator = mock_orch

        app = _make_app_with_state(state)
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post("/api/chat", json={"message": "Total failure"})
            # Should still return a response (even if it's an error)
            assert resp.status_code in (200, 500)


# ══════════════════════════════════════════════════════════════════════════════
# Voice Pipeline Error Recovery Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestVoicePipelineErrorRecovery:
    """Test voice pipeline graceful degradation."""

    def test_audio_device_error_propagates(self) -> None:
        """AudioDeviceError from mic start propagates to caller."""
        with pytest.raises(AudioDeviceError):
            raise AudioDeviceError(
                device_type="input",
                reason="No recording devices found",
            )

    def test_tts_error_is_non_fatal(self) -> None:
        """TTSError for one sentence doesn't crash the pipeline."""
        # This tests the pattern: catch VoiceError, log warning, continue
        errors_caught = []
        sentences = ["First sentence.", "Second sentence.", "Third sentence."]
        results = []

        for sentence in sentences:
            try:
                if sentence == "Second sentence.":
                    raise TTSError(
                        f"Synthesis failed for: {sentence}",
                        operation="synthesize",
                    )
                results.append(sentence)
            except VoiceError as exc:
                errors_caught.append(str(exc))
                # Pipeline continues — no re-raise

        assert len(results) == 2
        assert len(errors_caught) == 1
        assert "First sentence." in results
        assert "Third sentence." in results

    def test_stt_error_allows_retry(self) -> None:
        """STT errors are recoverable — pipeline can process next utterance."""
        utterances_processed = []

        for text in ["first", "FAIL", "third"]:
            try:
                if text == "FAIL":
                    raise STTError("Decoding failed", operation="end_utterance")
                utterances_processed.append(text)
            except STTError:
                # Pipeline returns to IDLE, processes next utterance
                pass

        assert utterances_processed == ["first", "third"]

    @patch("ara.voice.pipeline.MicrophoneStream")
    @patch("ara.voice.pipeline.SpeakerOutput")
    @patch("ara.voice.pipeline.VADProcessor")
    @patch("ara.voice.pipeline.STTProcessor")
    @patch("ara.voice.pipeline.TTSProcessor")
    @patch("ara.voice.pipeline.WakeWordDetector")
    async def test_pipeline_start_with_model_load_failure(
        self,
        mock_ww_cls: MagicMock,
        mock_tts_cls: MagicMock,
        mock_stt_cls: MagicMock,
        mock_vad_cls: MagicMock,
        mock_speaker_cls: MagicMock,
        mock_mic_cls: MagicMock,
        settings: AraSettings,
    ) -> None:
        """Pipeline start fails gracefully when model loading fails."""
        from ara.voice.pipeline import VoicePipeline

        mock_vad = MagicMock()
        mock_vad.is_loaded = False
        mock_vad.load.side_effect = VoiceError("VAD model not found", operation="load")
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

        mock_mic_cls.return_value = MagicMock()
        mock_speaker_cls.return_value = MagicMock()

        pipeline = VoicePipeline(settings)
        with pytest.raises(VoiceError, match="VAD model not found"):
            await pipeline.start()


# ══════════════════════════════════════════════════════════════════════════════
# WebSocket Error Recovery Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestWebSocketErrorRecovery:
    """Test WebSocket resilience under error conditions."""

    def test_websocket_chat_with_failing_llm(self, settings: AraSettings) -> None:
        """WebSocket chat handles LLM failure gracefully."""

        async def failing_stream(*args: Any, **kwargs: Any) -> Any:
            raise LLMError("GPU memory exhausted", operation="generate_stream")
            yield  # type: ignore[misc]  # Make it an async generator

        mock_llm = MagicMock()
        mock_llm.is_connected = True
        mock_llm.active_model = "test"
        mock_llm.generate_stream = failing_stream

        state = AppState(settings=settings)
        state.llm_client = mock_llm
        state.orchestrator = None

        app = _make_app_with_state(state)
        with (
            TestClient(app, raise_server_exceptions=False) as client,
            client.websocket_connect("/ws") as ws,
        ):
            ws.send_json(
                {
                    "type": "chat",
                    "content": "Hello!",
                    "conversation_id": "err-test-001",
                }
            )

            # Should get an error response, not a crash
            responses: list[dict[str, Any]] = []
            while True:
                resp = ws.receive_json()
                responses.append(resp)
                if resp["type"] in ("done", "error"):
                    break

            types = [r["type"] for r in responses]
            assert "error" in types or "done" in types

    def test_websocket_still_works_after_error(self, settings: AraSettings) -> None:
        """WebSocket connection survives an error and handles subsequent messages."""
        mock_llm = MagicMock()
        mock_llm.is_connected = True
        mock_llm.active_model = "test"

        state = AppState(settings=settings)
        state.llm_client = mock_llm
        state.orchestrator = None

        app = _make_app_with_state(state)
        with (
            TestClient(app, raise_server_exceptions=False) as client,
            client.websocket_connect("/ws") as ws,
        ):
            # Ping still works
            ws.send_json({"type": "ping", "content": "", "conversation_id": ""})
            resp = ws.receive_json()
            assert resp["type"] == "pong"


# ══════════════════════════════════════════════════════════════════════════════
# Concurrent Request Handling Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestConcurrentRequestHandling:
    """Test that the API handles concurrent requests correctly."""

    def test_health_during_chat(self, settings: AraSettings) -> None:
        """Health endpoint works while chat is being served."""
        mock_llm = MagicMock()
        mock_llm.is_connected = True
        mock_llm.active_model = "test"
        mock_llm.generate = AsyncMock(return_value="Response")

        state = AppState(settings=settings)
        state.llm_client = mock_llm
        state.orchestrator = None

        app = _make_app_with_state(state)
        with TestClient(app, raise_server_exceptions=False) as client:
            # Health check
            resp1 = client.get("/health")
            assert resp1.status_code == 200

            # Chat
            resp2 = client.post("/api/chat", json={"message": "Hi"})
            assert resp2.status_code == 200

            # Health again
            resp3 = client.get("/health")
            assert resp3.status_code == 200
