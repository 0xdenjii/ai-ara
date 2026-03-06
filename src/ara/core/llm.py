"""LLM client — unified async interface for ARA's inference needs.

Supports two backends:
- **Ollama** (default): Uses Ollama's native API for model management,
  downloading, and generation. Best for development and model iteration.
- **llama.cpp**: Uses llama.cpp server's OpenAI-compatible API for production.
  Enables speculative decoding, flash attention, and direct GGUF loading.

Backend selection is controlled by ARA_BACKEND_TYPE env var ("ollama" or "llamacpp").
All consumers import LLMClient — it delegates to the appropriate backend internally.
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

from ara.config.models import ModelRole, calculate_vram_usage
from ara.utils.errors import LLMConnectionError, LLMError, LLMModelSwapError
from ara.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from ara.config.settings import AraSettings
    from ara.utils.types import ChatMessage

log = get_logger("llm")


class LLMClient:
    """Async LLM client supporting Ollama and llama.cpp backends.

    Detects the configured backend from settings and delegates all operations
    to the appropriate implementation. The public interface is identical
    regardless of backend.

    Usage:
        client = LLMClient(settings)
        await client.connect()
        async for token in client.generate_stream(messages):
            print(token, end="")
        await client.close()
    """

    def __init__(self, settings: AraSettings) -> None:
        self._settings = settings
        self._backend_type = settings.llm.backend_type.lower()

        if self._backend_type == "llamacpp":
            from ara.core.llamacpp import LlamaCppClient

            self._delegate = LlamaCppClient(settings)
            log.info(
                "llm_backend_selected",
                backend="llamacpp",
                base_url=settings.llm.llamacpp_base_url,
            )
        else:
            self._delegate = _OllamaBackend(settings)
            log.info(
                "llm_backend_selected",
                backend="ollama",
                base_url=settings.llm.ollama_base_url,
            )

    @property
    def active_model(self) -> str:
        """Currently loaded model name."""
        return self._delegate.active_model

    @property
    def is_connected(self) -> bool:
        """Whether the client has verified connectivity to the backend."""
        return self._delegate.is_connected

    @property
    def backend_type(self) -> str:
        """The active backend type ("ollama" or "llamacpp")."""
        return self._backend_type

    async def connect(self) -> None:
        """Verify connectivity to the backend and check model availability.

        Raises:
            LLMConnectionError: If the backend is unreachable.
        """
        await self._delegate.connect()

    async def close(self) -> None:
        """Clean up client resources."""
        await self._delegate.close()

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
            model: Override the active model for this request.
            temperature: Override default temperature.
            max_tokens: Override max response tokens.

        Returns:
            The complete generated text.

        Raises:
            LLMError: On generation failure.
        """
        return await self._delegate.generate(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    async def generate_stream(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """Stream generated tokens one at a time.

        Args:
            messages: Conversation history as ChatMessage objects.
            model: Override the active model for this request.
            temperature: Override default temperature.
            max_tokens: Override max response tokens.

        Yields:
            Individual tokens as they're generated.

        Raises:
            LLMError: On generation failure.
        """
        async for token in self._delegate.generate_stream(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        ):
            yield token

    async def swap_model(self, target_model: str) -> None:
        """Hot-swap to a different model.

        For Ollama: loads the new model via API (previous model unloaded by Ollama).
        For llama.cpp: raises LLMModelSwapError (server must be restarted).

        Args:
            target_model: The model identifier to switch to.

        Raises:
            LLMModelSwapError: If the swap fails or is unsupported.
        """
        await self._delegate.swap_model(target_model)

    async def swap_to_primary(self) -> None:
        """Switch to the primary conversation model."""
        await self._delegate.swap_to_primary()

    async def swap_to_long_context(self) -> None:
        """Switch to the long-context model for diary analysis."""
        await self._delegate.swap_to_long_context()

    async def list_models(self) -> list[dict[str, str | float]]:
        """List models available in the backend.

        Returns:
            List of dicts with model name, size, and modification time.
        """
        return await self._delegate.list_models()

    async def get_running_models(self) -> list[dict[str, str | float]]:
        """List models currently loaded in VRAM.

        Returns:
            List of dicts with model name and VRAM usage.
        """
        return await self._delegate.get_running_models()

    def get_vram_budget(self) -> dict[str, float]:
        """Get the current VRAM budget for the active model configuration."""
        return self._delegate.get_vram_budget()


# ── Ollama Backend ───────────────────────────────────────────────────────────


class _OllamaBackend:
    """Internal Ollama backend implementation.

    Wraps Ollama's native Python SDK for model management and generation.
    This was the original LLMClient code, now encapsulated as one of two
    possible backends.
    """

    def __init__(self, settings: AraSettings) -> None:
        import ollama as ollama_lib

        self._settings = settings
        self._ollama_lib = ollama_lib
        self._client = ollama_lib.AsyncClient(host=settings.llm.ollama_base_url)
        self._active_model: str = settings.llm.primary_model
        self._connected: bool = False

    @property
    def active_model(self) -> str:
        return self._active_model

    @property
    def is_connected(self) -> bool:
        return self._connected

    async def connect(self) -> None:
        """Verify connectivity to Ollama and check model availability."""
        base_url = self._settings.llm.ollama_base_url
        try:
            models_response = await self._client.list()
            available = [m.model for m in models_response.models]
            log.info(
                "ollama_connected",
                base_url=base_url,
                models_available=len(available),
            )

            if self._active_model not in available:
                matching = [m for m in available if m.startswith(self._active_model.split(":")[0])]
                if matching:
                    log.warning(
                        "model_tag_mismatch",
                        requested=self._active_model,
                        available_matches=matching,
                    )
                else:
                    log.warning(
                        "model_not_found",
                        model=self._active_model,
                        available=available,
                        hint="Run: ollama pull " + self._active_model,
                    )

            self._connected = True

        except Exception as exc:
            self._connected = False
            raise LLMConnectionError(base_url, str(exc)) from exc

    async def close(self) -> None:
        self._connected = False
        log.info("client_closed", backend="ollama")

    async def generate(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
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
        target_model = model or self._active_model
        temp = temperature if temperature is not None else self._settings.llm.temperature
        max_tok = max_tokens or self._settings.llm.max_response_tokens

        ollama_messages = [{"role": msg.role.value, "content": msg.content} for msg in messages]

        start_time = time.monotonic()
        tokens_generated = 0

        try:
            stream = await self._client.chat(
                model=target_model,
                messages=ollama_messages,
                stream=True,
                options={
                    "temperature": temp,
                    "num_predict": max_tok,
                },
            )

            async for chunk in stream:
                token = chunk.message.content
                if token:
                    tokens_generated += 1
                    yield token

                if chunk.done:
                    elapsed = time.monotonic() - start_time
                    log.info(
                        "generation_complete",
                        model=target_model,
                        backend="ollama",
                        tokens=tokens_generated,
                        elapsed_s=round(elapsed, 2),
                        tokens_per_sec=round(tokens_generated / elapsed, 1) if elapsed > 0 else 0,
                    )

        except self._ollama_lib.ResponseError as exc:
            raise LLMError(
                f"Generation failed: {exc}",
                operation="generate",
                details={"model": target_model, "error": str(exc)},
            ) from exc
        except asyncio.CancelledError:
            log.info(
                "generation_cancelled",
                model=target_model,
                backend="ollama",
                tokens_generated=tokens_generated,
            )
            raise
        except Exception as exc:
            elapsed = time.monotonic() - start_time
            raise LLMError(
                f"Unexpected error during generation: {exc}",
                operation="generate",
                details={
                    "model": target_model,
                    "backend": "ollama",
                    "elapsed_s": round(elapsed, 2),
                    "tokens_generated": tokens_generated,
                },
            ) from exc

    async def swap_model(self, target_model: str) -> None:
        previous = self._active_model
        if previous == target_model:
            log.info("swap_skipped", reason="already_active", model=target_model)
            return

        log.info("model_swap_start", from_model=previous, to_model=target_model)

        try:
            models_response = await self._client.list()
            available = [m.model for m in models_response.models]

            if target_model not in available:
                raise LLMModelSwapError(
                    previous,
                    target_model,
                    f"Model '{target_model}' not found. Available: {available}",
                )

            # Pre-load the model by sending an empty generate request
            await self._client.chat(
                model=target_model,
                messages=[{"role": "user", "content": "hi"}],
                options={"num_predict": 1},
            )

            self._active_model = target_model
            log.info("model_swap_complete", from_model=previous, to_model=target_model)

        except LLMModelSwapError:
            raise
        except Exception as exc:
            raise LLMModelSwapError(previous, target_model, str(exc)) from exc

    async def swap_to_primary(self) -> None:
        await self.swap_model(self._settings.llm.primary_model)

    async def swap_to_long_context(self) -> None:
        await self.swap_model(self._settings.llm.long_context_model)

    async def list_models(self) -> list[dict[str, str | float]]:
        try:
            response = await self._client.list()
            return [
                {
                    "name": m.model,
                    "size_gb": round(m.size / (1024**3), 2),
                    "modified": str(m.modified_at),
                }
                for m in response.models
            ]
        except Exception as exc:
            raise LLMError(
                f"Failed to list models: {exc}",
                operation="list_models",
            ) from exc

    async def get_running_models(self) -> list[dict[str, str | float]]:
        try:
            response = await self._client.ps()
            return [
                {
                    "name": m.model,
                    "size_gb": round(m.size / (1024**3), 2),
                    "expires_at": str(m.expires_at),
                }
                for m in response.models
            ]
        except Exception as exc:
            raise LLMError(
                f"Failed to list running models: {exc}",
                operation="ps",
            ) from exc

    def get_vram_budget(self) -> dict[str, float]:
        role_map = {
            self._settings.llm.primary_model: ModelRole.PRIMARY_LLM,
            self._settings.llm.long_context_model: ModelRole.LONG_CONTEXT_LLM,
        }
        active_role = role_map.get(self._active_model, ModelRole.PRIMARY_LLM)

        active_roles = [active_role]
        if active_role == ModelRole.PRIMARY_LLM:
            active_roles.append(ModelRole.VISION_ALWAYS_ON)

        return calculate_vram_usage(active_roles)
