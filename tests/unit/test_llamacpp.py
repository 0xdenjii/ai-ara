"""Tests for the llama.cpp backend client and LLMClient backend delegation.

Tests cover:
- LlamaCppClient connectivity (connect, close, health checks)
- Streaming generation via SSE (OpenAI-compatible format)
- Non-streaming generation
- Model listing
- Model swap rejection (llama.cpp doesn't support runtime swaps)
- Error handling (connection failures, HTTP errors, malformed SSE)
- LLMClient backend selection (ollama vs llamacpp delegation)
- VRAM budget computation
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from ara.config.settings import AraSettings
from ara.core.llamacpp import LlamaCppClient, build_server_args
from ara.core.llm import LLMClient
from ara.utils.errors import LLMConnectionError, LLMError, LLMModelSwapError
from ara.utils.types import ChatMessage, MessageRole

# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def settings() -> AraSettings:
    """Create settings with llamacpp backend configured."""
    return AraSettings(
        _env_file=None,
        llm={"backend_type": "llamacpp", "llamacpp_base_url": "http://localhost:8080"},
    )


@pytest.fixture
def ollama_settings() -> AraSettings:
    """Create settings with ollama backend (default)."""
    return AraSettings(
        _env_file=None,
        llm={"backend_type": "ollama", "ollama_base_url": "http://localhost:11434"},
    )


@pytest.fixture
def messages() -> list[ChatMessage]:
    """Sample conversation messages."""
    return [
        ChatMessage(role=MessageRole.SYSTEM, content="You are ARA."),
        ChatMessage(role=MessageRole.USER, content="Hello"),
    ]


def _make_sse_chunk(content: str, finish_reason: str | None = None) -> str:
    """Create an SSE data line for a chat completion chunk."""
    chunk = {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "model": "qwen3:30b-a3b",
        "choices": [
            {
                "index": 0,
                "delta": {"content": content} if content else {},
                "finish_reason": finish_reason,
            }
        ],
    }
    return f"data: {json.dumps(chunk)}"


def _make_sse_done() -> str:
    """Create the SSE [DONE] sentinel."""
    return "data: [DONE]"


# ── LlamaCppClient Tests ────────────────────────────────────────────────────


class TestLlamaCppConnect:
    """Tests for LlamaCppClient.connect()."""

    @pytest.mark.asyncio
    async def test_connect_success(self, settings: AraSettings) -> None:
        client = LlamaCppClient(settings)

        health_resp = httpx.Response(200, json={"status": "ok"})
        models_resp = httpx.Response(
            200,
            json={
                "data": [{"id": "qwen3-30b-a3b", "created": 1234567890}],
            },
        )

        with patch.object(
            client._client,
            "get",
            side_effect=[health_resp, models_resp],
        ):
            await client.connect()

        assert client.is_connected is True
        assert client._server_model == "qwen3-30b-a3b"

    @pytest.mark.asyncio
    async def test_connect_unhealthy_server(self, settings: AraSettings) -> None:
        client = LlamaCppClient(settings)

        health_resp = httpx.Response(503, json={"status": "loading model"})

        with (
            patch.object(client._client, "get", return_value=health_resp),
            pytest.raises(LLMConnectionError, match="unhealthy"),
        ):
            await client.connect()

        assert client.is_connected is False

    @pytest.mark.asyncio
    async def test_connect_unreachable(self, settings: AraSettings) -> None:
        client = LlamaCppClient(settings)

        with (
            patch.object(
                client._client,
                "get",
                side_effect=httpx.ConnectError("Connection refused"),
            ),
            pytest.raises(LLMConnectionError, match="Cannot connect"),
        ):
            await client.connect()

        assert client.is_connected is False

    @pytest.mark.asyncio
    async def test_connect_generic_error(self, settings: AraSettings) -> None:
        client = LlamaCppClient(settings)

        with (
            patch.object(
                client._client,
                "get",
                side_effect=RuntimeError("unexpected"),
            ),
            pytest.raises(LLMConnectionError, match="unexpected"),
        ):
            await client.connect()

        assert client.is_connected is False


class TestLlamaCppClose:
    """Tests for LlamaCppClient.close()."""

    @pytest.mark.asyncio
    async def test_close_resets_state(self, settings: AraSettings) -> None:
        client = LlamaCppClient(settings)
        client._connected = True

        with patch.object(client._client, "aclose", new_callable=AsyncMock):
            await client.close()

        assert client.is_connected is False


class TestLlamaCppGenerate:
    """Tests for LlamaCppClient.generate() (non-streaming)."""

    @pytest.mark.asyncio
    async def test_generate_returns_full_text(
        self, settings: AraSettings, messages: list[ChatMessage]
    ) -> None:
        client = LlamaCppClient(settings)

        sse_lines = [
            _make_sse_chunk("Hello"),
            _make_sse_chunk(" there"),
            _make_sse_chunk("!", finish_reason="stop"),
            _make_sse_done(),
        ]

        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.aiter_lines = _async_line_iter(sse_lines)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        with patch.object(client._client, "stream", return_value=mock_response):
            result = await client.generate(messages)

        assert result == "Hello there!"


class TestLlamaCppStream:
    """Tests for LlamaCppClient.generate_stream()."""

    @pytest.mark.asyncio
    async def test_stream_yields_tokens(
        self, settings: AraSettings, messages: list[ChatMessage]
    ) -> None:
        client = LlamaCppClient(settings)

        sse_lines = [
            _make_sse_chunk("Hello"),
            _make_sse_chunk(" world"),
            _make_sse_chunk("", finish_reason="stop"),
            _make_sse_done(),
        ]

        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.aiter_lines = _async_line_iter(sse_lines)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        with patch.object(client._client, "stream", return_value=mock_response):
            tokens: list[str] = []
            async for token in client.generate_stream(messages):
                tokens.append(token)  # noqa: PERF401

        assert tokens == ["Hello", " world"]

    @pytest.mark.asyncio
    async def test_stream_handles_empty_lines(
        self, settings: AraSettings, messages: list[ChatMessage]
    ) -> None:
        client = LlamaCppClient(settings)

        sse_lines = [
            "",  # empty line (SSE heartbeat)
            _make_sse_chunk("Hi"),
            "",
            _make_sse_chunk("!", finish_reason="stop"),
            _make_sse_done(),
        ]

        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.aiter_lines = _async_line_iter(sse_lines)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        with patch.object(client._client, "stream", return_value=mock_response):
            tokens: list[str] = []
            async for token in client.generate_stream(messages):
                tokens.append(token)  # noqa: PERF401

        assert tokens == ["Hi", "!"]

    @pytest.mark.asyncio
    async def test_stream_malformed_json_skipped(
        self, settings: AraSettings, messages: list[ChatMessage]
    ) -> None:
        client = LlamaCppClient(settings)

        sse_lines = [
            _make_sse_chunk("OK"),
            "data: {invalid json}",  # malformed — should be skipped
            _make_sse_chunk("!", finish_reason="stop"),
            _make_sse_done(),
        ]

        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.aiter_lines = _async_line_iter(sse_lines)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        with patch.object(client._client, "stream", return_value=mock_response):
            tokens: list[str] = []
            async for token in client.generate_stream(messages):
                tokens.append(token)  # noqa: PERF401

        assert tokens == ["OK", "!"]

    @pytest.mark.asyncio
    async def test_stream_http_error(
        self, settings: AraSettings, messages: list[ChatMessage]
    ) -> None:
        client = LlamaCppClient(settings)

        mock_response = AsyncMock()
        mock_response.status_code = 500
        mock_response.aread = AsyncMock(return_value=b"Internal Server Error")
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        with (
            patch.object(client._client, "stream", return_value=mock_response),
            pytest.raises(LLMError, match="HTTP 500"),
        ):
            async for _ in client.generate_stream(messages):
                pass

    @pytest.mark.asyncio
    async def test_stream_connection_lost(
        self, settings: AraSettings, messages: list[ChatMessage]
    ) -> None:
        client = LlamaCppClient(settings)

        with (
            patch.object(
                client._client,
                "stream",
                side_effect=httpx.ConnectError("Connection lost"),
            ),
            pytest.raises(LLMConnectionError, match="Lost connection"),
        ):
            async for _ in client.generate_stream(messages):
                pass

    @pytest.mark.asyncio
    async def test_stream_unexpected_error(
        self, settings: AraSettings, messages: list[ChatMessage]
    ) -> None:
        client = LlamaCppClient(settings)

        with (
            patch.object(
                client._client,
                "stream",
                side_effect=RuntimeError("something broke"),
            ),
            pytest.raises(LLMError, match="Unexpected error"),
        ):
            async for _ in client.generate_stream(messages):
                pass

    @pytest.mark.asyncio
    async def test_stream_uses_configured_temperature(
        self, settings: AraSettings, messages: list[ChatMessage]
    ) -> None:
        client = LlamaCppClient(settings)

        sse_lines = [
            _make_sse_chunk("Hi", finish_reason="stop"),
            _make_sse_done(),
        ]

        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.aiter_lines = _async_line_iter(sse_lines)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        with patch.object(client._client, "stream", return_value=mock_response) as mock_stream:
            async for _ in client.generate_stream(messages, temperature=0.5, max_tokens=100):
                pass

            # Verify the request body
            call_args = mock_stream.call_args
            request_body = call_args.kwargs.get("json", call_args[1].get("json", {}))
            assert request_body["temperature"] == 0.5
            assert request_body["max_tokens"] == 100
            assert request_body["stream"] is True

    @pytest.mark.asyncio
    async def test_stream_done_without_finish_reason(
        self, settings: AraSettings, messages: list[ChatMessage]
    ) -> None:
        """SSE stream ends with [DONE] but no finish_reason in last chunk."""
        client = LlamaCppClient(settings)

        sse_lines = [
            _make_sse_chunk("Hello"),
            _make_sse_done(),  # [DONE] without prior finish_reason
        ]

        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.aiter_lines = _async_line_iter(sse_lines)
        mock_response.__aenter__ = AsyncMock(return_value=mock_response)
        mock_response.__aexit__ = AsyncMock(return_value=False)

        with patch.object(client._client, "stream", return_value=mock_response):
            tokens: list[str] = []
            async for token in client.generate_stream(messages):
                tokens.append(token)  # noqa: PERF401

        assert tokens == ["Hello"]


class TestLlamaCppSwapModel:
    """Tests for LlamaCppClient.swap_model()."""

    @pytest.mark.asyncio
    async def test_swap_same_model_no_error(self, settings: AraSettings) -> None:
        client = LlamaCppClient(settings)
        # Swapping to the same model should be a no-op
        await client.swap_model(client.active_model)

    @pytest.mark.asyncio
    async def test_swap_different_model_raises(self, settings: AraSettings) -> None:
        client = LlamaCppClient(settings)
        with pytest.raises(LLMModelSwapError, match="does not support runtime model swaps"):
            await client.swap_model("qwen3:14b")

    @pytest.mark.asyncio
    async def test_swap_to_primary_raises(self, settings: AraSettings) -> None:
        client = LlamaCppClient(settings)
        # Primary model is already active, so this should be a no-op
        await client.swap_to_primary()

    @pytest.mark.asyncio
    async def test_swap_to_long_context_raises(self, settings: AraSettings) -> None:
        client = LlamaCppClient(settings)
        with pytest.raises(LLMModelSwapError):
            await client.swap_to_long_context()


class TestLlamaCppListModels:
    """Tests for LlamaCppClient.list_models()."""

    @pytest.mark.asyncio
    async def test_list_models_success(self, settings: AraSettings) -> None:
        client = LlamaCppClient(settings)

        resp = httpx.Response(
            200,
            json={
                "data": [{"id": "qwen3-30b", "created": 1234567890}],
            },
        )

        with patch.object(client._client, "get", return_value=resp):
            models = await client.list_models()

        assert len(models) == 1
        assert models[0]["name"] == "qwen3-30b"

    @pytest.mark.asyncio
    async def test_list_models_server_error(self, settings: AraSettings) -> None:
        client = LlamaCppClient(settings)

        resp = httpx.Response(500, json={"error": "internal"})

        with patch.object(client._client, "get", return_value=resp):
            models = await client.list_models()

        # Falls back to configured model name
        assert len(models) == 1
        assert models[0]["size_gb"] == 0.0

    @pytest.mark.asyncio
    async def test_list_models_connection_error(self, settings: AraSettings) -> None:
        client = LlamaCppClient(settings)

        with (
            patch.object(
                client._client,
                "get",
                side_effect=httpx.ConnectError("refused"),
            ),
            pytest.raises(LLMError, match="Failed to list models"),
        ):
            await client.list_models()


class TestLlamaCppRunningModels:
    """Tests for LlamaCppClient.get_running_models()."""

    @pytest.mark.asyncio
    async def test_running_models_healthy(self, settings: AraSettings) -> None:
        client = LlamaCppClient(settings)
        client._server_model = "qwen3-30b"

        resp = httpx.Response(200, json={"status": "ok"})

        with patch.object(client._client, "get", return_value=resp):
            running = await client.get_running_models()

        assert len(running) == 1
        assert running[0]["name"] == "qwen3-30b"
        assert running[0]["expires_at"] == "never"

    @pytest.mark.asyncio
    async def test_running_models_unhealthy(self, settings: AraSettings) -> None:
        client = LlamaCppClient(settings)

        resp = httpx.Response(503, json={"status": "loading"})

        with patch.object(client._client, "get", return_value=resp):
            running = await client.get_running_models()

        assert running == []

    @pytest.mark.asyncio
    async def test_running_models_error(self, settings: AraSettings) -> None:
        client = LlamaCppClient(settings)

        with (
            patch.object(
                client._client,
                "get",
                side_effect=RuntimeError("boom"),
            ),
            pytest.raises(LLMError, match="Failed to list running models"),
        ):
            await client.get_running_models()


class TestLlamaCppVramBudget:
    """Tests for LlamaCppClient.get_vram_budget()."""

    def test_vram_budget_returns_dict(self, settings: AraSettings) -> None:
        client = LlamaCppClient(settings)
        budget = client.get_vram_budget()
        assert isinstance(budget, dict)
        assert "total" in budget


class TestLlamaCppProperties:
    """Tests for LlamaCppClient properties."""

    def test_active_model(self, settings: AraSettings) -> None:
        client = LlamaCppClient(settings)
        assert client.active_model == "qwen3:30b-a3b"

    def test_is_connected_default_false(self, settings: AraSettings) -> None:
        client = LlamaCppClient(settings)
        assert client.is_connected is False


# ── LLMClient Backend Delegation Tests ──────────────────────────────────────


class TestLLMClientBackendSelection:
    """Tests for LLMClient selecting the correct backend."""

    def test_default_is_ollama(self, ollama_settings: AraSettings) -> None:
        with patch("ara.core.llm._OllamaBackend") as mock_cls:
            mock_cls.return_value = MagicMock()
            client = LLMClient(ollama_settings)
            assert client.backend_type == "ollama"

    def test_llamacpp_backend_selected(self, settings: AraSettings) -> None:
        client = LLMClient(settings)
        assert client.backend_type == "llamacpp"
        assert isinstance(client._delegate, LlamaCppClient)

    def test_backend_type_case_insensitive(self) -> None:
        s = AraSettings(
            _env_file=None,
            llm={"backend_type": "LlamaCpp"},
        )
        client = LLMClient(s)
        assert client.backend_type == "llamacpp"

    @pytest.mark.asyncio
    async def test_connect_delegates(self, settings: AraSettings) -> None:
        client = LLMClient(settings)
        client._delegate = MagicMock()
        client._delegate.connect = AsyncMock()
        await client.connect()
        client._delegate.connect.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_close_delegates(self, settings: AraSettings) -> None:
        client = LLMClient(settings)
        client._delegate = MagicMock()
        client._delegate.close = AsyncMock()
        await client.close()
        client._delegate.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_generate_delegates(
        self, settings: AraSettings, messages: list[ChatMessage]
    ) -> None:
        client = LLMClient(settings)
        client._delegate = MagicMock()
        client._delegate.generate = AsyncMock(return_value="response")
        result = await client.generate(messages, temperature=0.5)
        assert result == "response"
        client._delegate.generate.assert_awaited_once_with(
            messages,
            model=None,
            temperature=0.5,
            max_tokens=None,
        )

    @pytest.mark.asyncio
    async def test_swap_model_delegates(self, settings: AraSettings) -> None:
        client = LLMClient(settings)
        client._delegate = MagicMock()
        client._delegate.swap_model = AsyncMock()
        await client.swap_model("other-model")
        client._delegate.swap_model.assert_awaited_once_with("other-model")

    @pytest.mark.asyncio
    async def test_list_models_delegates(self, settings: AraSettings) -> None:
        client = LLMClient(settings)
        client._delegate = MagicMock()
        client._delegate.list_models = AsyncMock(return_value=[{"name": "test"}])
        result = await client.list_models()
        assert result == [{"name": "test"}]

    def test_active_model_delegates(self, settings: AraSettings) -> None:
        client = LLMClient(settings)
        assert client.active_model == "qwen3:30b-a3b"

    def test_is_connected_delegates(self, settings: AraSettings) -> None:
        client = LLMClient(settings)
        assert client.is_connected is False

    def test_vram_budget_delegates(self, settings: AraSettings) -> None:
        client = LLMClient(settings)
        budget = client.get_vram_budget()
        assert isinstance(budget, dict)


# ── Speculative Decoding / Server Args Tests ────────────────────────────────


class TestBuildServerArgs:
    """Tests for build_server_args() helper."""

    def test_basic_args(self) -> None:
        s = AraSettings(_env_file=None)
        args = build_server_args(s, "/models/qwen3-30b.gguf")

        assert "--model" in args
        assert "/models/qwen3-30b.gguf" in args
        assert "--host" in args
        assert "--port" in args
        assert "--ctx-size" in args
        assert "--batch-size" in args
        assert "--n-gpu-layers" in args

    def test_flash_attention_enabled(self) -> None:
        s = AraSettings(_env_file=None, llm={"llamacpp_flash_attn": True})
        args = build_server_args(s, "/models/model.gguf")
        assert "--flash-attn" in args

    def test_flash_attention_disabled(self) -> None:
        s = AraSettings(_env_file=None, llm={"llamacpp_flash_attn": False})
        args = build_server_args(s, "/models/model.gguf")
        assert "--flash-attn" not in args

    def test_custom_threads(self) -> None:
        s = AraSettings(_env_file=None, llm={"llamacpp_threads": 12})
        args = build_server_args(s, "/models/model.gguf")
        idx = args.index("--threads")
        assert args[idx + 1] == "12"

    def test_zero_threads_omitted(self) -> None:
        s = AraSettings(_env_file=None, llm={"llamacpp_threads": 0})
        args = build_server_args(s, "/models/model.gguf")
        assert "--threads" not in args

    def test_speculative_decoding_enabled(self) -> None:
        s = AraSettings(
            _env_file=None,
            llm={
                "speculative_enabled": True,
                "speculative_draft_max": 20,
                "speculative_draft_min": 3,
                "speculative_draft_p_min": 0.85,
                "speculative_n_gpu_layers_draft": 99,
            },
        )
        args = build_server_args(
            s,
            "/models/qwen3-30b.gguf",
            draft_model_path="/models/qwen3-1b.gguf",
        )

        assert "--model-draft" in args
        assert "/models/qwen3-1b.gguf" in args
        idx_max = args.index("--draft-max")
        assert args[idx_max + 1] == "20"
        idx_min = args.index("--draft-min")
        assert args[idx_min + 1] == "3"
        idx_p = args.index("--draft-p-min")
        assert args[idx_p + 1] == "0.85"
        idx_layers = args.index("--n-gpu-layers-draft")
        assert args[idx_layers + 1] == "99"

    def test_speculative_enabled_no_draft_path(self) -> None:
        """When speculative is enabled but no draft path, args omit draft flags."""
        s = AraSettings(_env_file=None, llm={"speculative_enabled": True})
        args = build_server_args(s, "/models/qwen3-30b.gguf")
        assert "--model-draft" not in args

    def test_speculative_disabled_ignores_draft(self) -> None:
        """When speculative is disabled, draft path is ignored."""
        s = AraSettings(_env_file=None, llm={"speculative_enabled": False})
        args = build_server_args(
            s,
            "/models/qwen3-30b.gguf",
            draft_model_path="/models/qwen3-1b.gguf",
        )
        assert "--model-draft" not in args

    def test_custom_host_and_port(self) -> None:
        s = AraSettings(
            _env_file=None,
            llm={
                "llamacpp_host": "0.0.0.0",
                "llamacpp_port": 9999,
            },
        )
        args = build_server_args(s, "/models/model.gguf")
        idx_host = args.index("--host")
        assert args[idx_host + 1] == "0.0.0.0"
        idx_port = args.index("--port")
        assert args[idx_port + 1] == "9999"

    def test_custom_context_and_batch(self) -> None:
        s = AraSettings(
            _env_file=None,
            llm={
                "llamacpp_context_size": 65536,
                "llamacpp_batch_size": 4096,
            },
        )
        args = build_server_args(s, "/models/model.gguf")
        idx_ctx = args.index("--ctx-size")
        assert args[idx_ctx + 1] == "65536"
        idx_batch = args.index("--batch-size")
        assert args[idx_batch + 1] == "4096"


class TestSpeculativeSettings:
    """Tests for speculative decoding settings in LLMSettings."""

    def test_speculative_defaults(self) -> None:
        s = AraSettings(_env_file=None)
        assert s.llm.speculative_enabled is False
        assert s.llm.speculative_draft_max == 16
        assert s.llm.speculative_draft_min == 5
        assert s.llm.speculative_draft_p_min == 0.9
        assert s.llm.speculative_n_gpu_layers_draft == -1

    def test_llamacpp_server_defaults(self) -> None:
        s = AraSettings(_env_file=None)
        assert s.llm.llamacpp_n_gpu_layers == -1
        assert s.llm.llamacpp_flash_attn is True
        assert s.llm.llamacpp_context_size == 32768
        assert s.llm.llamacpp_batch_size == 2048
        assert s.llm.llamacpp_threads == 0
        assert s.llm.llamacpp_host == "127.0.0.1"
        assert s.llm.llamacpp_port == 8080

    def test_speculative_custom_values(self) -> None:
        s = AraSettings(
            _env_file=None,
            llm={
                "speculative_enabled": True,
                "speculative_draft_max": 24,
                "speculative_draft_min": 8,
                "speculative_draft_p_min": 0.95,
            },
        )
        assert s.llm.speculative_enabled is True
        assert s.llm.speculative_draft_max == 24
        assert s.llm.speculative_draft_min == 8
        assert s.llm.speculative_draft_p_min == 0.95


# ── Helpers ──────────────────────────────────────────────────────────────────


def _async_line_iter(lines: list[str]):
    """Create a callable that returns an async iterator over lines."""

    async def _iter():
        for line in lines:
            yield line

    return _iter
