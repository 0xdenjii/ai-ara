"""llama.cpp server client — OpenAI-compatible API backend for ARA.

Provides the same async generation interface as the Ollama client but targets
a llama.cpp server (llama-server) running with --host and --port flags.
The llama.cpp server exposes an OpenAI-compatible /v1/chat/completions endpoint.

This backend is the intended production inference path, enabling:
- Speculative decoding (--draft-model flag on llama-server)
- Flash attention (--flash-attn flag)
- Direct GGUF model loading (no Ollama layer)
- Fine-grained KV cache and batch size control

Usage:
    client = LlamaCppClient(settings)
    await client.connect()
    async for token in client.generate_stream(messages):
        print(token, end="")
    await client.close()

Server launch:
    args = build_server_args(settings, model_path="/path/to/model.gguf")
    # Start llama-server with these args
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import TYPE_CHECKING

import httpx

from ara.utils.errors import LLMConnectionError, LLMError, LLMModelSwapError
from ara.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from ara.config.settings import AraSettings
    from ara.utils.types import ChatMessage

log = get_logger("llamacpp")


class LlamaCppClient:
    """Async client for llama.cpp server's OpenAI-compatible API.

    Implements the same public interface as the Ollama-based LLMClient
    so it can be used as a drop-in replacement. The server must be started
    externally with `llama-server --model <path> --host 0.0.0.0 --port 8080`.

    Key differences from Ollama backend:
    - Model is loaded at server startup, not via API calls
    - Model swap requires restarting the server with a different model
    - list_models() returns the single loaded model
    - Speculative decoding is configured at server startup (--draft-model)
    """

    def __init__(self, settings: AraSettings) -> None:
        self._settings = settings
        self._base_url = settings.llm.llamacpp_base_url.rstrip("/")
        self._active_model: str = settings.llm.primary_model
        self._connected: bool = False
        self._server_model: str = ""  # Model name reported by the server
        gen_timeout = settings.llm.llamacpp_generation_timeout_s
        health_timeout = settings.llm.llamacpp_health_timeout_s
        self._gen_timeout = gen_timeout
        self._health_timeout = health_timeout
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(
                connect=health_timeout,
                read=gen_timeout,
                write=health_timeout,
                pool=health_timeout,
            ),
        )

    @property
    def active_model(self) -> str:
        """Currently configured model name."""
        return self._active_model

    @property
    def is_connected(self) -> bool:
        """Whether the client has verified connectivity to the llama.cpp server."""
        return self._connected

    async def connect(self) -> None:
        """Verify connectivity to the llama.cpp server.

        Checks the /health endpoint (available in llama.cpp server) and
        retrieves the loaded model info from /v1/models.

        Raises:
            LLMConnectionError: If the server is unreachable or unhealthy.
        """
        base_url = self._base_url
        try:
            # Check server health
            health_response = await self._client.get(
                "/health",
                timeout=self._health_timeout,
            )
            if health_response.status_code != 200:
                health_data = health_response.json()
                status = health_data.get("status", "unknown")
                raise LLMConnectionError(
                    base_url,
                    f"Server unhealthy (status={status}). "
                    "Ensure model is fully loaded before connecting.",
                )

            # Retrieve loaded model info
            models_response = await self._client.get(
                "/v1/models",
                timeout=self._health_timeout,
            )
            if models_response.status_code == 200:
                models_data = models_response.json()
                model_list = models_data.get("data", [])
                if model_list:
                    self._server_model = model_list[0].get("id", "unknown")

            log.info(
                "llamacpp_connected",
                base_url=base_url,
                server_model=self._server_model,
                configured_model=self._active_model,
            )

            self._connected = True

        except LLMConnectionError:
            raise
        except httpx.ConnectError as exc:
            self._connected = False
            raise LLMConnectionError(
                base_url,
                f"Cannot connect to llama.cpp server: {exc}. Ensure llama-server is running.",
            ) from exc
        except Exception as exc:
            self._connected = False
            raise LLMConnectionError(base_url, str(exc)) from exc

    async def close(self) -> None:
        """Clean up HTTP client resources."""
        self._connected = False
        await self._client.aclose()
        log.info("client_closed", backend="llamacpp")

    async def generate(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Generate a complete response (non-streaming).

        Args:
            messages: Conversation history as ChatMessage objects.
            model: Ignored for llama.cpp (server loads one model at startup).
            temperature: Override default temperature.
            max_tokens: Override max response tokens.

        Returns:
            The complete generated text.

        Raises:
            LLMError: On generation failure.
        """
        result_parts: list[str] = []
        async for token in self.generate_stream(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        ):
            result_parts.append(token)  # noqa: PERF401
        return "".join(result_parts)

    async def generate_stream(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """Stream generated tokens one at a time via SSE.

        Uses the OpenAI-compatible /v1/chat/completions endpoint with
        stream=true. Each SSE event contains a delta with content tokens.

        Args:
            messages: Conversation history as ChatMessage objects.
            model: Model name to include in request (llama.cpp ignores this
                   but logs it; the loaded model is used regardless).
            temperature: Override default temperature.
            max_tokens: Override max response tokens.

        Yields:
            Individual tokens as they're generated.

        Raises:
            LLMError: On generation failure.
        """
        target_model = model or self._active_model
        temp = temperature if temperature is not None else self._settings.llm.temperature
        max_tok = max_tokens or self._settings.llm.max_response_tokens

        openai_messages = [{"role": msg.role.value, "content": msg.content} for msg in messages]

        request_body = {
            "model": target_model,
            "messages": openai_messages,
            "temperature": temp,
            "max_tokens": max_tok,
            "stream": True,
        }

        start_time = time.monotonic()
        tokens_generated = 0

        try:
            async with self._client.stream(
                "POST",
                "/v1/chat/completions",
                json=request_body,
                timeout=httpx.Timeout(
                    connect=self._health_timeout,
                    read=self._gen_timeout,
                    write=self._health_timeout,
                    pool=self._health_timeout,
                ),
            ) as response:
                if response.status_code != 200:
                    body = await response.aread()
                    raise LLMError(
                        f"llama.cpp server returned HTTP {response.status_code}: "
                        f"{body.decode(errors='replace')}",
                        operation="generate",
                        details={"model": target_model, "status": response.status_code},
                    )

                async for line in response.aiter_lines():
                    # SSE format: "data: {...}" or "data: [DONE]"
                    if not line.startswith("data: "):
                        continue

                    payload = line[6:]  # Strip "data: " prefix

                    if payload.strip() == "[DONE]":
                        break

                    try:
                        chunk = json.loads(payload)
                    except json.JSONDecodeError:
                        continue

                    choices = chunk.get("choices", [])
                    if not choices:
                        continue

                    delta = choices[0].get("delta", {})
                    content = delta.get("content", "")

                    if content:
                        tokens_generated += 1
                        yield content

                    # Check for finish_reason
                    finish_reason = choices[0].get("finish_reason")
                    if finish_reason is not None:
                        elapsed = time.monotonic() - start_time
                        log.info(
                            "generation_complete",
                            model=target_model,
                            backend="llamacpp",
                            tokens=tokens_generated,
                            elapsed_s=round(elapsed, 2),
                            tokens_per_sec=(
                                round(tokens_generated / elapsed, 1) if elapsed > 0 else 0
                            ),
                            finish_reason=finish_reason,
                        )

        except LLMError:
            raise
        except asyncio.CancelledError:
            log.info(
                "generation_cancelled",
                model=target_model,
                backend="llamacpp",
                tokens_generated=tokens_generated,
            )
            raise
        except httpx.ConnectError as exc:
            raise LLMConnectionError(
                self._base_url,
                f"Lost connection to llama.cpp server during generation: {exc}",
            ) from exc
        except Exception as exc:
            elapsed = time.monotonic() - start_time
            raise LLMError(
                f"Unexpected error during generation: {exc}",
                operation="generate",
                details={
                    "model": target_model,
                    "backend": "llamacpp",
                    "elapsed_s": round(elapsed, 2),
                    "tokens_generated": tokens_generated,
                },
            ) from exc

    async def swap_model(self, target_model: str, *, num_ctx: int | None = None) -> None:
        """Signal a model swap request.

        llama.cpp server loads a single model at startup. Swapping models
        requires restarting the server with a different --model flag.
        This method updates the internal tracking but cannot actually
        change the loaded model.

        For Ollama-managed model downloads and swaps, use the Ollama backend.

        Args:
            target_model: The model identifier to switch to.
            num_ctx: Ignored for llama.cpp (context is set at server start).

        Raises:
            LLMModelSwapError: Always, since llama.cpp doesn't support runtime swaps.
        """
        previous = self._active_model
        if previous == target_model:
            log.info("swap_skipped", reason="already_active", model=target_model)
            return

        # llama.cpp doesn't support runtime model swaps
        raise LLMModelSwapError(
            previous,
            target_model,
            "llama.cpp server does not support runtime model swaps. "
            "Restart the server with --model <path> to load a different model. "
            "Use the Ollama backend (ARA_BACKEND_TYPE=ollama) for dynamic model management.",
        )

    async def swap_to_primary(self) -> None:
        """Switch to the primary conversation model."""
        await self.swap_model(
            self._settings.llm.primary_model,
            num_ctx=self._settings.llm.primary_model_context_length,
        )

    async def swap_to_long_context(self) -> None:
        """Switch to the long-context model for diary analysis."""
        await self.swap_model(
            self._settings.llm.long_context_model,
            num_ctx=self._settings.llm.long_context_length,
        )

    async def list_models(self) -> list[dict[str, str | float]]:
        """List the model loaded in the llama.cpp server.

        Returns:
            List with a single dict containing the loaded model info.
        """
        try:
            response = await self._client.get(
                "/v1/models",
                timeout=self._health_timeout,
            )
            if response.status_code != 200:
                return [
                    {
                        "name": self._server_model or self._active_model,
                        "size_gb": 0.0,
                        "modified": "",
                    }
                ]

            data = response.json()
            models = data.get("data", [])
            return [
                {
                    "name": m.get("id", "unknown"),
                    "size_gb": 0.0,  # llama.cpp doesn't report size via API
                    "modified": str(m.get("created", "")),
                }
                for m in models
            ]
        except Exception as exc:
            raise LLMError(
                f"Failed to list models: {exc}",
                operation="list_models",
                details={"backend": "llamacpp"},
            ) from exc

    async def get_running_models(self) -> list[dict[str, str | float]]:
        """List the currently loaded model (always one for llama.cpp).

        Returns:
            List with a single dict for the loaded model.
        """
        try:
            # llama.cpp /slots endpoint gives info about loaded model slots
            response = await self._client.get(
                "/health",
                timeout=self._health_timeout,
            )
            model_name = self._server_model or self._active_model
            if response.status_code == 200:
                return [
                    {
                        "name": model_name,
                        "size_gb": 0.0,
                        "expires_at": "never",
                    }
                ]
            return []
        except Exception as exc:
            raise LLMError(
                f"Failed to list running models: {exc}",
                operation="ps",
                details={"backend": "llamacpp"},
            ) from exc

    def get_vram_budget(self) -> dict[str, float]:
        """Get the current VRAM budget.

        llama.cpp manages its own VRAM allocation via --n-gpu-layers.
        This returns a static estimate based on the configured model role.
        """
        from ara.config.models import ModelRole, calculate_vram_usage

        role_map = {
            self._settings.llm.primary_model: ModelRole.PRIMARY_LLM,
            self._settings.llm.long_context_model: ModelRole.LONG_CONTEXT_LLM,
        }
        active_role = role_map.get(self._active_model, ModelRole.PRIMARY_LLM)

        active_roles = [active_role]
        if active_role == ModelRole.PRIMARY_LLM:
            active_roles.append(ModelRole.VISION_ALWAYS_ON)

        return calculate_vram_usage(active_roles)


def build_server_args(
    settings: AraSettings,
    model_path: str,
    *,
    draft_model_path: str | None = None,
) -> list[str]:
    """Build command-line arguments for launching llama-server.

    Constructs the full argument list based on ARA's LLM settings,
    including optional speculative decoding and flash attention flags.

    Args:
        settings: ARA configuration.
        model_path: Path to the primary GGUF model file.
        draft_model_path: Path to the draft GGUF model file for speculative
            decoding. If None and speculative_enabled is True, speculative
            decoding is skipped with a warning.

    Returns:
        List of command-line arguments (without the executable name).
        Pass these to subprocess or os.execv after the llama-server path.

    Example:
        >>> args = build_server_args(settings, "/models/qwen3-30b-q4.gguf",
        ...                          draft_model_path="/models/qwen3-1b-q8.gguf")
        >>> # Start server: subprocess.run(["llama-server"] + args)
    """
    llm = settings.llm
    args: list[str] = [
        "--model",
        model_path,
        "--host",
        llm.llamacpp_host,
        "--port",
        str(llm.llamacpp_port),
        "--ctx-size",
        str(llm.llamacpp_context_size),
        "--batch-size",
        str(llm.llamacpp_batch_size),
        "--n-gpu-layers",
        str(llm.llamacpp_n_gpu_layers),
    ]

    if llm.llamacpp_threads > 0:
        args.extend(["--threads", str(llm.llamacpp_threads)])

    if llm.llamacpp_flash_attn:
        args.append("--flash-attn")

    # KV cache quantization (reduces VRAM usage for KV cache, q8_0 is ~lossless)
    if llm.llamacpp_kv_cache_type_k != "f16":
        args.extend(["--cache-type-k", llm.llamacpp_kv_cache_type_k])
    if llm.llamacpp_kv_cache_type_v != "f16":
        args.extend(["--cache-type-v", llm.llamacpp_kv_cache_type_v])

    # Speculative decoding
    if llm.speculative_enabled and draft_model_path:
        args.extend(
            [
                "--model-draft",
                draft_model_path,
                "--draft-max",
                str(llm.speculative_draft_max),
                "--draft-min",
                str(llm.speculative_draft_min),
                "--draft-p-min",
                str(llm.speculative_draft_p_min),
                "--n-gpu-layers-draft",
                str(llm.speculative_n_gpu_layers_draft),
            ]
        )
        log.info(
            "speculative_decoding_configured",
            draft_model=draft_model_path,
            draft_max=llm.speculative_draft_max,
            draft_min=llm.speculative_draft_min,
            draft_p_min=llm.speculative_draft_p_min,
        )
    elif llm.speculative_enabled and not draft_model_path:
        log.warning(
            "speculative_decoding_skipped",
            reason="draft_model_path not provided",
        )

    return args
