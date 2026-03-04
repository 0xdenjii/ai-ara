"""Shared type definitions used across ARA subsystems.

Centralizes common data structures to avoid circular imports and duplication.
All types are immutable (frozen dataclasses or Pydantic models) unless mutation
is explicitly needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

# ── Enums ────────────────────────────────────────────────────────────────────


class MessageRole(StrEnum):
    """Role of a participant in a conversation turn."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class IntentType(StrEnum):
    """Classified intent of a user query for agent routing."""

    FACTUAL = "factual"
    EMOTIONAL = "emotional"
    PLANNING = "planning"
    REMINISCING = "reminiscing"
    CREATIVE = "creative"
    VISION = "vision"
    DESKTOP = "desktop"
    WEB_SEARCH = "web_search"
    RESEARCH = "research"
    GENERAL = "general"


class ConversationEndReason(StrEnum):
    """Why a conversation ended — drives memory feedback loop."""

    SILENCE_TIMEOUT = "silence_timeout"
    USER_FAREWELL = "user_farewell"
    MANUAL_TRIGGER = "manual_trigger"
    SESSION_CLOSE = "session_close"


class MemorySource(StrEnum):
    """Origin of a memory chunk in the vector store."""

    DIARY = "diary"
    CONVERSATION = "conversation"
    NOTE = "note"
    WEB_RESEARCH = "web_research"
    INGESTED_FILE = "ingested_file"


class VoicePipelineState(StrEnum):
    """Current state of the voice pipeline."""

    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"


# ── Data Structures ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ChatMessage:
    """A single message in a conversation."""

    role: MessageRole
    content: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ConversationTurn:
    """A user-assistant exchange pair, stored for persistence."""

    turn_id: str = field(default_factory=lambda: uuid4().hex[:12])
    conversation_id: str = ""
    user_message: str = ""
    assistant_message: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    intent: IntentType = IntentType.GENERAL
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MemoryChunk:
    """A chunk of text stored in the vector store with metadata."""

    chunk_id: str
    text: str
    source: MemorySource
    source_file: str = ""
    date: datetime | None = None
    section_title: str = ""
    embedding: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalResult:
    """A single result from hybrid retrieval (vector + BM25 + graph)."""

    chunk: MemoryChunk
    score: float
    retrieval_method: str  # "vector", "bm25", "graph", "hybrid"
    rerank_score: float | None = None


@dataclass(frozen=True)
class RetrievalContext:
    """Assembled context from retrieval pipeline, ready for LLM prompt injection."""

    results: list[RetrievalResult]
    query: str
    query_type: IntentType
    total_candidates: int = 0
    retrieval_time_ms: float = 0.0


@dataclass(frozen=True)
class GraphEntity:
    """An entity node in the knowledge graph."""

    entity_id: str
    name: str
    entity_type: str  # "person", "place", "event", "habit", "goal", "project"
    aliases: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GraphRelation:
    """A directed edge in the knowledge graph."""

    source_id: str
    target_id: str
    relation_type: str  # "discussed", "planned", "felt", "decided", "mentioned_with"
    weight: float = 1.0
    date: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolCall:
    """A structured tool invocation from the agent."""

    tool_name: str
    parameters: dict[str, Any]
    call_id: str = field(default_factory=lambda: uuid4().hex[:12])


@dataclass(frozen=True)
class ToolResult:
    """Result of a tool execution."""

    call_id: str
    tool_name: str
    success: bool
    output: str
    error: str | None = None
    execution_time_ms: float = 0.0


@dataclass(frozen=True)
class WebSearchResult:
    """A single web search result."""

    title: str
    url: str
    snippet: str
    source: str  # "searxng", "duckduckgo"
    relevance_score: float = 0.0


@dataclass(frozen=True)
class AudioSegment:
    """A segment of audio data with metadata."""

    samples: bytes
    sample_rate: int
    channels: int
    duration_s: float


@dataclass(frozen=True)
class TranscriptionResult:
    """Result from STT transcription."""

    text: str
    confidence: float = 0.0
    language: str = "en"
    duration_s: float = 0.0
    latency_ms: float = 0.0


@dataclass(frozen=True)
class VRAMSnapshot:
    """Current VRAM usage snapshot for monitoring."""

    total_gb: float
    used_gb: float
    free_gb: float
    models_loaded: list[str] = field(default_factory=list)
