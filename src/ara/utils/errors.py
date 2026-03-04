"""Custom exception hierarchy for ARA.

Every exception carries actionable context: component, operation, and details.
No bare "something failed" — always describe what, where, and why.
"""

from __future__ import annotations

from typing import Any


class AraError(Exception):
    """Base exception for all ARA errors.

    Attributes:
        component: The subsystem that raised the error (e.g. "llm", "voice.stt", "memory.vector").
        operation: The operation that failed (e.g. "generate", "transcribe", "search").
        details: Structured context about the failure for logging and debugging.
    """

    def __init__(
        self,
        message: str,
        *,
        component: str,
        operation: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.component = component
        self.operation = operation
        self.details = details or {}
        full_message = f"[{component}:{operation}] {message}"
        if self.details:
            detail_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            full_message = f"{full_message} ({detail_str})"
        super().__init__(full_message)


# ── Configuration Errors ─────────────────────────────────────────────────────


class ConfigError(AraError):
    """Invalid or missing configuration."""

    def __init__(
        self,
        message: str,
        *,
        operation: str = "load",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, component="config", operation=operation, details=details)


class ModelNotFoundError(ConfigError):
    """A required model file or registry entry is missing."""

    def __init__(self, model_name: str, path: str) -> None:
        super().__init__(
            f"Model not found: {model_name}",
            operation="model_lookup",
            details={"model": model_name, "path": path},
        )


# ── LLM Errors ───────────────────────────────────────────────────────────────


class LLMError(AraError):
    """Base error for LLM inference failures."""

    def __init__(
        self,
        message: str,
        *,
        operation: str = "generate",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, component="llm", operation=operation, details=details)


class LLMConnectionError(LLMError):
    """Cannot connect to the LLM backend (Ollama / llama.cpp)."""

    def __init__(self, base_url: str, reason: str) -> None:
        super().__init__(
            f"Cannot connect to LLM backend: {reason}",
            operation="connect",
            details={"base_url": base_url, "reason": reason},
        )


class LLMTimeoutError(LLMError):
    """LLM inference exceeded the allowed time budget."""

    def __init__(self, model: str, timeout_s: float, tokens_generated: int) -> None:
        super().__init__(
            f"Inference timeout after {timeout_s}s",
            operation="generate",
            details={
                "model": model,
                "timeout_s": timeout_s,
                "tokens_generated": tokens_generated,
            },
        )


class LLMModelSwapError(LLMError):
    """Failed to swap between LLM models (e.g. primary ↔ long-context)."""

    def __init__(self, from_model: str, to_model: str, reason: str) -> None:
        super().__init__(
            f"Model swap failed: {reason}",
            operation="swap",
            details={"from": from_model, "to": to_model, "reason": reason},
        )


# ── Voice Pipeline Errors ────────────────────────────────────────────────────


class VoiceError(AraError):
    """Base error for the voice pipeline."""

    def __init__(
        self,
        message: str,
        *,
        operation: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, component="voice", operation=operation, details=details)


class STTError(VoiceError):
    """Speech-to-text transcription failure."""

    def __init__(
        self,
        message: str,
        *,
        operation: str = "transcribe",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, operation=operation, details=details)


class STTTimeoutError(STTError):
    """STT processing exceeded time budget."""

    def __init__(self, elapsed_ms: float, audio_duration_s: float) -> None:
        super().__init__(
            f"STT timeout (elapsed: {elapsed_ms:.0f}ms, audio: {audio_duration_s:.1f}s)",
            operation="transcribe",
            details={"elapsed_ms": elapsed_ms, "audio_duration_s": audio_duration_s},
        )


class TTSError(VoiceError):
    """Text-to-speech synthesis failure."""

    def __init__(
        self,
        message: str,
        *,
        operation: str = "synthesize",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, operation=operation, details=details)


class AudioDeviceError(VoiceError):
    """Audio input/output device error (missing mic, speaker, etc.)."""

    def __init__(self, device_type: str, reason: str) -> None:
        super().__init__(
            f"Audio device error ({device_type}): {reason}",
            operation="audio_io",
            details={"device_type": device_type, "reason": reason},
        )


class WakeWordError(VoiceError):
    """Wake word detection failure."""

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, operation="wake_word", details=details)


# ── Memory / RAG Errors ──────────────────────────────────────────────────────


class AraMemoryError(AraError):
    """Base error for memory subsystem."""

    def __init__(
        self,
        message: str,
        *,
        operation: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, component="memory", operation=operation, details=details)


class IngestionError(AraMemoryError):
    """Diary/document ingestion pipeline failure."""

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, operation="ingest", details=details)


class EmbeddingError(AraMemoryError):
    """Text embedding generation failure."""

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, operation="embed", details=details)


class VectorStoreError(AraMemoryError):
    """LanceDB vector store operation failure."""

    def __init__(
        self,
        message: str,
        *,
        operation: str = "search",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, operation=operation, details=details)


class GraphError(AraMemoryError):
    """Knowledge graph operation failure."""

    def __init__(
        self,
        message: str,
        *,
        operation: str = "query",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, operation=operation, details=details)


class RetrievalError(AraMemoryError):
    """Hybrid retrieval pipeline failure."""

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, operation="retrieve", details=details)


class ConversationPersistenceError(AraMemoryError):
    """Conversation storage or memory feedback loop failure."""

    def __init__(
        self,
        message: str,
        *,
        operation: str = "persist",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, operation=operation, details=details)


# ── Agent Errors ─────────────────────────────────────────────────────────────


class AgentError(AraError):
    """Base error for agent orchestration."""

    def __init__(
        self,
        message: str,
        *,
        operation: str = "execute",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, component="agent", operation=operation, details=details)


class RoutingError(AgentError):
    """Intent classification or agent routing failure."""

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, operation="route", details=details)


class ToolExecutionError(AgentError):
    """A tool call failed during agent execution."""

    def __init__(self, tool_name: str, reason: str) -> None:
        super().__init__(
            f"Tool '{tool_name}' failed: {reason}",
            operation="tool_call",
            details={"tool": tool_name, "reason": reason},
        )


class ToolPermissionError(AgentError):
    """A tool call was blocked by the permission system."""

    def __init__(self, tool_name: str, action: str) -> None:
        super().__init__(
            f"Permission denied for tool '{tool_name}' action '{action}'",
            operation="permission_check",
            details={"tool": tool_name, "action": action},
        )


# ── Vision Errors ────────────────────────────────────────────────────────────


class VisionError(AraError):
    """Base error for the vision pipeline."""

    def __init__(
        self,
        message: str,
        *,
        operation: str = "analyze",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, component="vision", operation=operation, details=details)


# ── Internet / Web Errors ────────────────────────────────────────────────────


class WebError(AraError):
    """Base error for the internet connectivity layer."""

    def __init__(
        self,
        message: str,
        *,
        operation: str = "fetch",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, component="web", operation=operation, details=details)


class WebSearchError(WebError):
    """Web search query failure."""

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message, operation="search", details=details)


class WebBrowseError(WebError):
    """Playwright browsing failure."""

    def __init__(self, url: str, reason: str) -> None:
        super().__init__(
            f"Failed to browse '{url}': {reason}",
            operation="browse",
            details={"url": url, "reason": reason},
        )


class OfflineError(WebError):
    """Attempted an internet operation while offline or internet is disabled."""

    def __init__(self, operation: str) -> None:
        super().__init__(
            "Internet is disabled or unavailable",
            operation=operation,
            details={"hint": "Enable internet via dashboard toggle or check network connection"},
        )


# ── API Errors ───────────────────────────────────────────────────────────────


class APIError(AraError):
    """Base error for FastAPI backend operations."""

    def __init__(
        self,
        message: str,
        *,
        operation: str = "request",
        status_code: int = 500,
        details: dict[str, Any] | None = None,
    ) -> None:
        combined_details = {"status_code": status_code, **(details or {})}
        super().__init__(message, component="api", operation=operation, details=combined_details)
        self.status_code = status_code


# ── Desktop Automation Errors ────────────────────────────────────────────────


class DesktopError(AraError):
    """Base error for desktop automation."""

    def __init__(
        self,
        message: str,
        *,
        operation: str = "execute",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, component="desktop", operation=operation, details=details)
