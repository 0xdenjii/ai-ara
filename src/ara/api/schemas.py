"""Pydantic request/response models for the ARA API.

All API data flows through these schemas for validation, serialization,
and OpenAPI documentation generation. Mirrors internal types from
ara.utils.types but adapted for HTTP transport.

Naming convention:
    - *Request: incoming request bodies
    - *Response: outgoing response bodies
    - *Item: nested objects within responses
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from ara import __version__

# ══════════════════════════════════════════════════════════════════════════════
# Common / Shared
# ══════════════════════════════════════════════════════════════════════════════


class ErrorResponse(BaseModel):
    """Standard error response body."""

    error: str
    detail: str = ""
    component: str = ""
    operation: str = ""
    status_code: int = 500


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "ok"
    version: str = __version__
    uptime_s: float = 0.0
    models_loaded: list[str] = Field(default_factory=list)


# ══════════════════════════════════════════════════════════════════════════════
# Chat
# ══════════════════════════════════════════════════════════════════════════════


class MessageItem(BaseModel):
    """A single message in conversation history."""

    role: str  # "user", "assistant", "system"
    content: str
    timestamp: datetime | None = None


class ChatRequest(BaseModel):
    """Request body for chat endpoint."""

    message: str = Field(..., min_length=1, max_length=10000)
    conversation_id: str = ""
    history: list[MessageItem] = Field(default_factory=list)
    stream: bool = False
    max_steps: int = Field(default=5, ge=1, le=20)


class ChatResponse(BaseModel):
    """Response body for non-streaming chat."""

    response: str
    conversation_id: str = ""
    intent: str = "general"
    agent: str = "unknown"
    confidence: float = 0.0
    elapsed_ms: float = 0.0
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    tool_results: list[dict[str, Any]] = Field(default_factory=list)
    error: str = ""


class StreamChunk(BaseModel):
    """A single chunk in a streaming response (SSE data payload)."""

    type: str  # "token", "done", "error", "metadata"
    content: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


# ══════════════════════════════════════════════════════════════════════════════
# WebSocket
# ══════════════════════════════════════════════════════════════════════════════


class WSMessage(BaseModel):
    """WebSocket message format (bidirectional)."""

    type: str  # "chat", "ping", "cancel", "status"
    content: str = ""
    conversation_id: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class WSResponse(BaseModel):
    """WebSocket response message."""

    type: str  # "token", "done", "error", "pong", "status"
    content: str = ""
    conversation_id: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


# ══════════════════════════════════════════════════════════════════════════════
# Memory
# ══════════════════════════════════════════════════════════════════════════════


class MemorySearchRequest(BaseModel):
    """Request body for memory search."""

    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=10, ge=1, le=100)
    source_filter: str = ""  # "diary", "conversation", etc.
    date_from: datetime | None = None
    date_to: datetime | None = None


class MemoryChunkItem(BaseModel):
    """A single memory chunk in search results."""

    chunk_id: str
    text: str
    source: str
    source_file: str = ""
    date: datetime | None = None
    section_title: str = ""
    score: float = 0.0
    retrieval_method: str = ""


class MemorySearchResponse(BaseModel):
    """Response body for memory search."""

    results: list[MemoryChunkItem] = Field(default_factory=list)
    query: str = ""
    total_candidates: int = 0
    retrieval_time_ms: float = 0.0


class MemoryStatsResponse(BaseModel):
    """Memory subsystem statistics."""

    total_chunks: int = 0
    sources: dict[str, int] = Field(default_factory=dict)
    vector_db_path: str = ""
    graph_entities: int = 0
    graph_relations: int = 0


# ══════════════════════════════════════════════════════════════════════════════
# Voice
# ══════════════════════════════════════════════════════════════════════════════


class VoiceStatusResponse(BaseModel):
    """Voice pipeline status."""

    state: str = "idle"  # idle, listening, processing, speaking
    wake_word_enabled: bool = False
    hotkey_enabled: bool = False
    stt_model: str = ""
    tts_model: str = ""
    tts_voice: str = ""


class VoiceControlRequest(BaseModel):
    """Voice pipeline control command."""

    action: str  # "start", "stop", "toggle_wake_word", "toggle_hotkey"


class VoiceControlResponse(BaseModel):
    """Result of a voice control action."""

    action: str
    success: bool
    state: str = "idle"
    message: str = ""


# ══════════════════════════════════════════════════════════════════════════════
# Models
# ══════════════════════════════════════════════════════════════════════════════


class ModelItem(BaseModel):
    """Information about a loaded or available model."""

    role: str  # "primary_llm", "vision_always_on", etc.
    name: str
    location: str  # "gpu" or "cpu"
    vram_gb: float = 0.0
    ram_mb: float = 0.0
    loaded: bool = False
    description: str = ""


class VRAMUsageResponse(BaseModel):
    """Current VRAM usage breakdown."""

    total_gb: float = 24.0
    used_gb: float = 0.0
    free_gb: float = 24.0
    models: list[ModelItem] = Field(default_factory=list)


class ModelSwapRequest(BaseModel):
    """Request to swap the active LLM model."""

    target_model: str  # Ollama model name, e.g. "qwen3:14b"
    reason: str = ""  # Optional context for why


class ModelSwapResponse(BaseModel):
    """Result of a model swap."""

    previous_model: str
    current_model: str
    success: bool
    elapsed_ms: float = 0.0
    message: str = ""


# ══════════════════════════════════════════════════════════════════════════════
# Settings
# ══════════════════════════════════════════════════════════════════════════════


class PersonalitySettingsResponse(BaseModel):
    """Current personality settings."""

    humor_dial: int = 7
    voice_identity: str = "female"
    auto_dial_down_on_distress: bool = True
    distress_dial_reduction: int = 2


class PersonalityUpdateRequest(BaseModel):
    """Update personality settings."""

    humor_dial: int | None = Field(default=None, ge=1, le=10)
    auto_dial_down_on_distress: bool | None = None
    distress_dial_reduction: int | None = Field(default=None, ge=0, le=5)


class DesktopSettingsResponse(BaseModel):
    """Current desktop automation settings."""

    enabled: bool = True
    require_confirmation: bool = True
    command_timeout_seconds: float = 30.0


class DesktopUpdateRequest(BaseModel):
    """Update desktop automation settings."""

    enabled: bool | None = None
    require_confirmation: bool | None = None


class InternetSettingsResponse(BaseModel):
    """Current internet connectivity settings."""

    internet_enabled: bool = True
    searxng_url: str = ""
    web_cache_ttl_s: int = 3600


class InternetUpdateRequest(BaseModel):
    """Update internet settings."""

    internet_enabled: bool | None = None


class SettingsResponse(BaseModel):
    """Full settings snapshot."""

    personality: PersonalitySettingsResponse = Field(
        default_factory=PersonalitySettingsResponse,
    )
    desktop: DesktopSettingsResponse = Field(
        default_factory=DesktopSettingsResponse,
    )
    internet: InternetSettingsResponse = Field(
        default_factory=InternetSettingsResponse,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Conversations
# ══════════════════════════════════════════════════════════════════════════════


class ConversationItem(BaseModel):
    """A conversation summary for listing."""

    conversation_id: str
    started_at: datetime | None = None
    ended_at: datetime | None = None
    turn_count: int = 0
    summary: str = ""
    is_active: bool = False


class ConversationDetailResponse(BaseModel):
    """Full conversation with turns."""

    conversation_id: str
    started_at: datetime | None = None
    ended_at: datetime | None = None
    turns: list[MessageItem] = Field(default_factory=list)
    summary: str = ""


class ConversationListResponse(BaseModel):
    """List of recent conversations."""

    conversations: list[ConversationItem] = Field(default_factory=list)
    total: int = 0
