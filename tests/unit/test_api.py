"""Tests for Phase 8: FastAPI Backend.

Covers: schemas, app creation, health endpoint, chat route, memory route,
voice route, models route, settings route, WebSocket handler, error handling,
ThinkBlockFilter, filter_think_blocks.

All tests use httpx.AsyncClient with FastAPI's TestClient pattern.
No real LLM, no real Ollama, no real model loading.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from ara.api.app import AppState, create_app
from ara.api.schemas import (
    ChatRequest,
    ChatResponse,
    ErrorResponse,
    HealthResponse,
    MemorySearchRequest,
    MessageItem,
    ModelItem,
    PersonalitySettingsResponse,
    PersonalityUpdateRequest,
    SettingsResponse,
    StreamChunk,
    VoiceControlRequest,
    VoiceStatusResponse,
    WSMessage,
    WSResponse,
)
from ara.config.settings import AraSettings
from ara.core.streaming import ThinkBlockFilter, filter_think_blocks

# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def settings() -> AraSettings:
    return AraSettings(env="testing")


@pytest.fixture
def mock_app_state(settings: AraSettings) -> AppState:
    """AppState with mocked subsystems."""
    state = AppState(settings=settings)
    state.llm_client = MagicMock()
    state.llm_client.is_connected = True
    state.llm_client.active_model = "qwen3.5:35b"
    state.llm_client.generate = AsyncMock(return_value="Hello from ARA!")
    state.llm_client.generate_stream = AsyncMock()
    return state


@pytest.fixture
def app(mock_app_state: AppState) -> Any:
    """FastAPI app with mocked state (bypasses real lifespan).

    We build the app via create_app() but override the lifespan so
    TestClient doesn't attempt to connect to real Ollama/vector store.
    """
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def noop_lifespan(_app: Any):  # type: ignore[no-untyped-def]
        """No-op lifespan that just sets our mock state."""
        _app.state.ara = mock_app_state
        yield

    test_app = create_app(settings=mock_app_state.settings)
    # Replace the real lifespan with our no-op version.
    # Must set it on the router (FastAPI stores it there).
    test_app.router.lifespan_context = noop_lifespan

    return test_app


@pytest.fixture
def client(app: Any) -> Any:
    """Synchronous test client that runs the noop lifespan."""
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


# ══════════════════════════════════════════════════════════════════════════════
# Schema Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestSchemas:
    """Pydantic schema validation tests."""

    def test_chat_request_valid(self) -> None:
        req = ChatRequest(message="Hello ARA!")
        assert req.message == "Hello ARA!"
        assert req.stream is False
        assert req.history == []
        assert req.max_steps == 5

    def test_chat_request_with_history(self) -> None:
        req = ChatRequest(
            message="Follow up",
            history=[MessageItem(role="user", content="Hi")],
            conversation_id="conv-123",
        )
        assert len(req.history) == 1
        assert req.conversation_id == "conv-123"

    def test_chat_request_empty_message_rejected(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ChatRequest(message="")

    def test_chat_response_defaults(self) -> None:
        resp = ChatResponse(response="Hello!")
        assert resp.response == "Hello!"
        assert resp.intent == "general"
        assert resp.agent == "unknown"
        assert resp.error == ""

    def test_stream_chunk(self) -> None:
        chunk = StreamChunk(type="token", content="hello")
        assert chunk.type == "token"
        assert chunk.content == "hello"

    def test_error_response(self) -> None:
        err = ErrorResponse(error="VisionError", detail="Model not loaded")
        assert err.error == "VisionError"
        assert err.status_code == 500

    def test_health_response(self) -> None:
        h = HealthResponse(status="ok", uptime_s=42.5)
        assert h.status == "ok"
        assert h.uptime_s == 42.5

    def test_ws_message(self) -> None:
        msg = WSMessage(type="chat", content="Hello")
        assert msg.type == "chat"

    def test_ws_response(self) -> None:
        resp = WSResponse(type="token", content="Hi")
        assert resp.type == "token"

    def test_memory_search_request(self) -> None:
        req = MemorySearchRequest(query="gym routine")
        assert req.query == "gym routine"
        assert req.top_k == 10

    def test_model_item(self) -> None:
        item = ModelItem(role="primary_llm", name="Qwen3", location="gpu", vram_gb=16.5)
        assert item.vram_gb == 16.5

    def test_personality_settings(self) -> None:
        p = PersonalitySettingsResponse(humor_dial=9)
        assert p.humor_dial == 9

    def test_personality_update_partial(self) -> None:
        update = PersonalityUpdateRequest(humor_dial=3)
        assert update.humor_dial == 3
        assert update.auto_dial_down_on_distress is None

    def test_voice_status(self) -> None:
        v = VoiceStatusResponse(state="listening")
        assert v.state == "listening"

    def test_voice_control_request(self) -> None:
        req = VoiceControlRequest(action="start")
        assert req.action == "start"

    def test_settings_response(self) -> None:
        s = SettingsResponse()
        assert s.personality.humor_dial == 7
        assert s.desktop.enabled is True
        assert s.internet.internet_enabled is True


# ══════════════════════════════════════════════════════════════════════════════
# App Creation Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestAppCreation:
    """FastAPI app factory tests."""

    def test_create_app_returns_fastapi(self) -> None:
        app = create_app(settings=AraSettings(env="testing"))
        assert app.title == "ARA API"
        assert app.version == "0.1.0"

    def test_create_app_has_routes(self) -> None:
        app = create_app(settings=AraSettings(env="testing"))
        route_paths = [r.path for r in app.routes]
        assert "/health" in route_paths
        assert "/ws" in route_paths

    def test_create_app_has_cors(self) -> None:
        app = create_app(settings=AraSettings(env="testing"))
        # CORS middleware is in the middleware stack
        assert len(app.user_middleware) > 0


class TestAppState:
    """AppState tests."""

    def test_default_state(self) -> None:
        state = AppState()
        # Core
        assert state.llm_client is None
        assert state.orchestrator is None
        # Memory
        assert state.vector_store is None
        assert state.knowledge_graph is None
        assert state.conversation_store is None
        assert state.retrieval_pipeline is None
        assert state.ingestion_pipeline is None
        # Agents
        assert state.intent_router is None
        assert state.agent_registry is None
        assert state.context_manager is None
        # Tools
        assert state.tool_registry is None
        assert state.mcp_bridge is None
        # Pipelines
        assert state.vision_pipeline is None
        assert state.voice_pipeline is None
        # Proactive
        assert state.proactive_scheduler is None
        assert state.notification_queue is None
        assert state.mood_tracker is None
        # Obsidian
        assert state.obsidian_sync is None
        # Connectivity
        assert state.connectivity_manager is None

    def test_uptime(self) -> None:
        state = AppState()
        assert state.uptime_s >= 0.0


# ══════════════════════════════════════════════════════════════════════════════
# Health Endpoint
# ══════════════════════════════════════════════════════════════════════════════


class TestHealthEndpoint:
    """Health check endpoint tests."""

    def test_health_ok(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "uptime_s" in data

    def test_health_shows_loaded_models(self, client: TestClient) -> None:
        resp = client.get("/health")
        data = resp.json()
        assert "qwen3.5:35b" in data["models_loaded"]


# ══════════════════════════════════════════════════════════════════════════════
# Chat Endpoint
# ══════════════════════════════════════════════════════════════════════════════


class TestChatEndpoint:
    """Chat route tests."""

    def test_chat_basic(self, client: TestClient, mock_app_state: AppState) -> None:
        mock_app_state.llm_client.generate = AsyncMock(return_value="<think>\n\n</think>Hey there!")

        resp = client.post("/api/chat", json={"message": "Hello"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["response"] == "Hey there!"
        assert data["agent"] == "direct_llm"

    def test_chat_llm_disconnected(self, client: TestClient, mock_app_state: AppState) -> None:
        mock_app_state.llm_client.is_connected = False

        resp = client.post("/api/chat", json={"message": "Hello"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["error"] != ""

    def test_chat_with_history(self, client: TestClient, mock_app_state: AppState) -> None:
        mock_app_state.llm_client.generate = AsyncMock(return_value="Follow up response")

        resp = client.post(
            "/api/chat",
            json={
                "message": "And then?",
                "history": [
                    {"role": "user", "content": "Tell me a story"},
                    {"role": "assistant", "content": "Once upon a time..."},
                ],
            },
        )
        assert resp.status_code == 200

    def test_chat_validation_empty_message(self, client: TestClient) -> None:
        resp = client.post("/api/chat", json={"message": ""})
        assert resp.status_code == 422  # Validation error

    def test_chat_with_orchestrator(self, client: TestClient, mock_app_state: AppState) -> None:
        mock_orchestrator = AsyncMock()
        mock_result = MagicMock()
        mock_result.response = "Orchestrated response"
        mock_result.intent = "factual"
        mock_result.agent = "rag_agent"
        mock_result.confidence = 0.95
        mock_result.elapsed_ms = 150.0
        mock_result.tool_calls = []
        mock_result.tool_results = []
        mock_result.error = ""
        mock_orchestrator.process = AsyncMock(return_value=mock_result)
        mock_app_state.orchestrator = mock_orchestrator

        resp = client.post("/api/chat", json={"message": "What did I write about Mom?"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["response"] == "Orchestrated response"
        assert data["agent"] == "rag_agent"


class TestChatStreamEndpoint:
    """SSE streaming endpoint tests."""

    def test_stream_returns_event_stream(
        self, client: TestClient, mock_app_state: AppState
    ) -> None:
        async def fake_stream(*args: Any, **kwargs: Any) -> Any:
            for token in ["Hello", " ", "world", "!"]:
                yield token

        mock_app_state.llm_client.generate_stream = fake_stream

        resp = client.post("/api/chat/stream", json={"message": "Hi"})
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")

    def test_stream_llm_disconnected(self, client: TestClient, mock_app_state: AppState) -> None:
        mock_app_state.llm_client.is_connected = False

        resp = client.post("/api/chat/stream", json={"message": "Hi"})
        assert resp.status_code == 200
        # Should contain an error event
        assert "error" in resp.text


# ══════════════════════════════════════════════════════════════════════════════
# Memory Endpoint
# ══════════════════════════════════════════════════════════════════════════════


class TestMemoryEndpoint:
    """Memory route tests."""

    def test_memory_search_no_store(self, client: TestClient) -> None:
        resp = client.post("/api/memory/search", json={"query": "gym routine"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["results"] == []

    def test_memory_stats_no_store(self, client: TestClient) -> None:
        resp = client.get("/api/memory/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_chunks"] == 0

    def test_conversations_list(self, client: TestClient) -> None:
        resp = client.get("/api/memory/conversations")
        assert resp.status_code == 200
        data = resp.json()
        assert "conversations" in data

    def test_conversation_detail_not_found(self, client: TestClient) -> None:
        resp = client.get("/api/memory/conversations/nonexistent")
        assert resp.status_code == 200
        data = resp.json()
        assert data["conversation_id"] == "nonexistent"


# ══════════════════════════════════════════════════════════════════════════════
# Voice Endpoint
# ══════════════════════════════════════════════════════════════════════════════


class TestVoiceEndpoint:
    """Voice route tests."""

    def test_voice_status_no_pipeline(self, client: TestClient) -> None:
        resp = client.get("/api/voice/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["state"] == "unavailable"

    def test_voice_control_no_pipeline(self, client: TestClient) -> None:
        resp = client.post("/api/voice/control", json={"action": "start"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is False

    def test_voice_control_invalid_action(self, client: TestClient) -> None:
        resp = client.post("/api/voice/control", json={"action": "explode"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is False
        assert "Invalid action" in data["message"]


# ══════════════════════════════════════════════════════════════════════════════
# Models Endpoint
# ══════════════════════════════════════════════════════════════════════════════


class TestModelsEndpoint:
    """Models route tests."""

    def test_list_models(self, client: TestClient) -> None:
        resp = client.get("/api/models")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) > 0
        # Check primary LLM is in the list
        names = [m["name"] for m in data]
        assert "Qwen3.5-35B-A3B" in names

    def test_vram_usage(self, client: TestClient) -> None:
        resp = client.get("/api/models/vram")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_gb"] == 24.0
        assert "models" in data

    def test_swap_model_no_llm(self, client: TestClient, mock_app_state: AppState) -> None:
        mock_app_state.llm_client = None

        resp = client.post("/api/models/swap", json={"target_model": "qwen3:14b"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is False


# ══════════════════════════════════════════════════════════════════════════════
# Settings Endpoint
# ══════════════════════════════════════════════════════════════════════════════


class TestSettingsEndpoint:
    """Settings route tests."""

    def test_get_all_settings(self, client: TestClient) -> None:
        resp = client.get("/api/settings")
        assert resp.status_code == 200
        data = resp.json()
        assert "personality" in data
        assert "desktop" in data
        assert "internet" in data
        assert data["personality"]["humor_dial"] == 7

    def test_get_personality(self, client: TestClient) -> None:
        resp = client.get("/api/settings/personality")
        assert resp.status_code == 200
        data = resp.json()
        assert data["humor_dial"] == 7

    def test_update_personality(self, client: TestClient) -> None:
        resp = client.put("/api/settings/personality", json={"humor_dial": 3})
        assert resp.status_code == 200
        data = resp.json()
        assert data["humor_dial"] == 3

    def test_get_desktop(self, client: TestClient) -> None:
        resp = client.get("/api/settings/desktop")
        assert resp.status_code == 200
        data = resp.json()
        assert data["enabled"] is True

    def test_update_desktop(self, client: TestClient) -> None:
        resp = client.put("/api/settings/desktop", json={"enabled": False})
        assert resp.status_code == 200
        data = resp.json()
        assert data["enabled"] is False

    def test_get_internet(self, client: TestClient) -> None:
        resp = client.get("/api/settings/internet")
        assert resp.status_code == 200
        data = resp.json()
        assert data["internet_enabled"] is True

    def test_update_internet(self, client: TestClient) -> None:
        resp = client.put("/api/settings/internet", json={"internet_enabled": False})
        assert resp.status_code == 200
        data = resp.json()
        assert data["internet_enabled"] is False


# ══════════════════════════════════════════════════════════════════════════════
# WebSocket Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestWebSocket:
    """WebSocket handler tests."""

    def test_ws_ping_pong(self, client: TestClient) -> None:
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "ping"})
            data = ws.receive_json()
            assert data["type"] == "pong"

    def test_ws_status(self, client: TestClient) -> None:
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "status"})
            data = ws.receive_json()
            assert data["type"] == "status"
            assert "llm_connected" in data["metadata"]

    def test_ws_unknown_type(self, client: TestClient) -> None:
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "unknown_command"})
            data = ws.receive_json()
            assert data["type"] == "error"
            assert "Unknown" in data["content"]

    def test_ws_invalid_json(self, client: TestClient) -> None:
        with client.websocket_connect("/ws") as ws:
            ws.send_text("not-json{{{")
            data = ws.receive_json()
            assert data["type"] == "error"
            assert "Invalid JSON" in data["content"]

    def test_ws_empty_chat(self, client: TestClient) -> None:
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "chat", "content": ""})
            data = ws.receive_json()
            assert data["type"] == "error"
            assert "Empty" in data["content"]


# ══════════════════════════════════════════════════════════════════════════════
# ThinkBlockFilter Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestFilterThinkBlocks:
    """filter_think_blocks function tests."""

    def test_no_think_block(self) -> None:
        assert filter_think_blocks("Hello world") == "Hello world"

    def test_simple_think_block(self) -> None:
        text = "<think>reasoning here</think>Hello world"
        assert filter_think_blocks(text) == "Hello world"

    def test_multiline_think_block(self) -> None:
        text = "<think>\nline1\nline2\n</think>Result"
        assert filter_think_blocks(text) == "Result"

    def test_empty_think_block(self) -> None:
        text = "<think>\n\n</think>Output"
        assert filter_think_blocks(text) == "Output"

    def test_multiple_think_blocks(self) -> None:
        text = "<think>a</think>Hello <think>b</think>world"
        assert filter_think_blocks(text) == "Hello world"

    def test_strips_whitespace(self) -> None:
        text = "  <think>x</think>  Hello  "
        assert filter_think_blocks(text) == "Hello"

    def test_empty_string(self) -> None:
        assert filter_think_blocks("") == ""

    def test_only_think_block(self) -> None:
        assert filter_think_blocks("<think>hidden</think>") == ""


class TestThinkBlockFilter:
    """ThinkBlockFilter streaming filter tests."""

    def test_no_think_blocks(self) -> None:
        tbf = ThinkBlockFilter()
        result = ""
        for token in ["Hello", " ", "world"]:
            result += tbf.process(token)
        result += tbf.flush()
        assert result == "Hello world"

    def test_filter_simple_think_block(self) -> None:
        tbf = ThinkBlockFilter()
        tokens = ["<think>", "reasoning", "</think>", "Hello"]
        result = ""
        for token in tokens:
            result += tbf.process(token)
        result += tbf.flush()
        assert "reasoning" not in result
        assert "Hello" in result

    def test_filter_think_block_with_content_before(self) -> None:
        tbf = ThinkBlockFilter()
        tokens = ["Pre", "<think>", "hidden", "</think>", "Post"]
        result = ""
        for token in tokens:
            result += tbf.process(token)
        result += tbf.flush()
        assert "Pre" in result
        assert "Post" in result
        assert "hidden" not in result

    def test_empty_input(self) -> None:
        tbf = ThinkBlockFilter()
        assert tbf.process("") == ""
        assert tbf.flush() == ""

    def test_reset(self) -> None:
        tbf = ThinkBlockFilter()
        tbf.process("<think>")
        tbf.reset()
        assert tbf.process("visible") == "visible"

    def test_incomplete_think_block_dropped(self) -> None:
        tbf = ThinkBlockFilter()
        tbf.process("<think>")
        tbf.process("incomplete")
        # Flush without closing tag — should drop
        result = tbf.flush()
        assert result == ""

    def test_qwen3_style_empty_think(self) -> None:
        """Qwen3 with /no_think emits <think>\n\n</think>."""
        tbf = ThinkBlockFilter()
        tokens = ["<think>", "\n", "\n", "</think>", "Hello!"]
        result = ""
        for token in tokens:
            result += tbf.process(token)
        result += tbf.flush()
        assert result.strip() == "Hello!"


# ══════════════════════════════════════════════════════════════════════════════
# Error Handler Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestErrorHandling:
    """Error handler tests."""

    def test_ara_error_to_response(self) -> None:
        from ara.api.app import _ara_error_to_response
        from ara.utils.errors import VisionError

        exc = VisionError("Model not loaded", operation="analyze")
        resp = _ara_error_to_response(exc)
        assert resp.status_code == 500

    def test_api_error_with_status_code(self) -> None:
        from ara.api.app import _ara_error_to_response
        from ara.utils.errors import APIError

        exc = APIError("Not found", status_code=404)
        resp = _ara_error_to_response(exc)
        assert resp.status_code == 404
