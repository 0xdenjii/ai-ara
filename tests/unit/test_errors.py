"""Tests for ARA custom exception hierarchy."""

from __future__ import annotations

from ara.utils.errors import (
    AgentError,
    APIError,
    AraError,
    AraMemoryError,
    AudioDeviceError,
    ConfigError,
    ConversationPersistenceError,
    DesktopError,
    EmbeddingError,
    GraphError,
    IngestionError,
    LLMConnectionError,
    LLMError,
    LLMModelSwapError,
    LLMTimeoutError,
    OfflineError,
    RetrievalError,
    RoutingError,
    STTError,
    STTTimeoutError,
    ToolExecutionError,
    ToolPermissionError,
    TTSError,
    VectorStoreError,
    VisionError,
    VoiceError,
    WakeWordError,
    WebBrowseError,
    WebError,
    WebSearchError,
)


class TestAraErrorBase:
    """Tests for the base exception class."""

    def test_base_error_format(self) -> None:
        err = AraError("test failure", component="test", operation="run")
        assert "[test:run]" in str(err)
        assert "test failure" in str(err)

    def test_base_error_with_details(self) -> None:
        err = AraError(
            "bad thing",
            component="x",
            operation="y",
            details={"key": "val", "count": 42},
        )
        msg = str(err)
        assert "key=val" in msg
        assert "count=42" in msg

    def test_base_error_attributes(self) -> None:
        err = AraError("msg", component="c", operation="o", details={"a": 1})
        assert err.component == "c"
        assert err.operation == "o"
        assert err.details == {"a": 1}


class TestLLMErrors:
    """Tests for LLM-specific errors."""

    def test_connection_error(self) -> None:
        err = LLMConnectionError("http://localhost:11434", "connection refused")
        assert isinstance(err, LLMError)
        assert isinstance(err, AraError)
        assert "connection refused" in str(err)
        assert err.component == "llm"

    def test_timeout_error(self) -> None:
        err = LLMTimeoutError("qwen3:30b-a3b", timeout_s=30.0, tokens_generated=150)
        assert "30.0s" in str(err)
        assert err.details["tokens_generated"] == 150

    def test_swap_error(self) -> None:
        err = LLMModelSwapError("qwen3:30b-a3b", "qwen3:14b", "VRAM exhausted")
        assert err.details["from"] == "qwen3:30b-a3b"
        assert err.details["to"] == "qwen3:14b"


class TestVoiceErrors:
    """Tests for voice pipeline errors."""

    def test_stt_timeout(self) -> None:
        err = STTTimeoutError(elapsed_ms=3000, audio_duration_s=5.2)
        assert isinstance(err, STTError)
        assert isinstance(err, VoiceError)
        assert "3000ms" in str(err)
        assert "5.2s" in str(err)

    def test_tts_error(self) -> None:
        err = TTSError("synthesis failed", details={"voice": "af_bella"})
        assert err.component == "voice"

    def test_audio_device_error(self) -> None:
        err = AudioDeviceError("microphone", "device not found")
        assert "microphone" in str(err)

    def test_wake_word_error(self) -> None:
        err = WakeWordError("model load failed")
        assert err.operation == "wake_word"


class TestMemoryErrors:
    """Tests for memory subsystem errors."""

    def test_ingestion_error(self) -> None:
        err = IngestionError("parse failed", details={"file": "diary.md"})
        assert isinstance(err, AraMemoryError)
        assert err.component == "memory"

    def test_embedding_error(self) -> None:
        err = EmbeddingError("model not loaded")
        assert err.operation == "embed"

    def test_vector_store_error(self) -> None:
        err = VectorStoreError("table not found", operation="upsert")
        assert err.operation == "upsert"

    def test_graph_error(self) -> None:
        err = GraphError("node not found", operation="traverse")
        assert err.operation == "traverse"

    def test_retrieval_error(self) -> None:
        err = RetrievalError("no results")
        assert err.operation == "retrieve"

    def test_conversation_persistence_error(self) -> None:
        err = ConversationPersistenceError("db locked", operation="store")
        assert err.operation == "store"


class TestAgentErrors:
    """Tests for agent orchestration errors."""

    def test_routing_error(self) -> None:
        err = RoutingError("ambiguous intent")
        assert isinstance(err, AgentError)
        assert err.operation == "route"

    def test_tool_execution_error(self) -> None:
        err = ToolExecutionError("desktop.click", "element not found")
        assert "desktop.click" in str(err)
        assert err.details["tool"] == "desktop.click"

    def test_tool_permission_error(self) -> None:
        err = ToolPermissionError("system.shell", "rm -rf")
        assert "Permission denied" in str(err)


class TestWebErrors:
    """Tests for internet connectivity errors."""

    def test_web_search_error(self) -> None:
        err = WebSearchError("API rate limited")
        assert isinstance(err, WebError)
        assert err.operation == "search"

    def test_web_browse_error(self) -> None:
        err = WebBrowseError("https://example.com", "timeout")
        assert "example.com" in str(err)

    def test_offline_error(self) -> None:
        err = OfflineError("search")
        assert "disabled" in str(err)
        assert "hint" in err.details


class TestOtherErrors:
    """Tests for remaining error types."""

    def test_config_error(self) -> None:
        err = ConfigError("missing .env")
        assert isinstance(err, AraError)
        assert err.component == "config"

    def test_vision_error(self) -> None:
        err = VisionError("image decode failed")
        assert err.component == "vision"

    def test_api_error_has_status_code(self) -> None:
        err = APIError("not found", status_code=404)
        assert err.status_code == 404
        assert err.details["status_code"] == 404

    def test_desktop_error(self) -> None:
        err = DesktopError("window not found", operation="focus")
        assert err.component == "desktop"


class TestInheritanceChain:
    """Verify the full inheritance hierarchy for except-clause matching."""

    def test_all_errors_inherit_from_ara_error(self) -> None:
        errors = [
            ConfigError("x"),
            LLMError("x"),
            LLMConnectionError("url", "reason"),
            LLMTimeoutError("m", 1.0, 0),
            LLMModelSwapError("a", "b", "r"),
            VoiceError("x", operation="test"),
            STTError("x"),
            STTTimeoutError(100, 1.0),
            TTSError("x"),
            AudioDeviceError("mic", "err"),
            WakeWordError("x"),
            AraMemoryError("x", operation="test"),
            IngestionError("x"),
            EmbeddingError("x"),
            VectorStoreError("x"),
            GraphError("x"),
            RetrievalError("x"),
            ConversationPersistenceError("x"),
            AgentError("x"),
            RoutingError("x"),
            ToolExecutionError("t", "r"),
            ToolPermissionError("t", "a"),
            VisionError("x"),
            WebError("x"),
            WebSearchError("x"),
            WebBrowseError("u", "r"),
            OfflineError("op"),
            APIError("x"),
            DesktopError("x"),
        ]
        for err in errors:
            assert isinstance(err, AraError), f"{type(err).__name__} does not inherit AraError"
