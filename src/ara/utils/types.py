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
    OBSIDIAN = "obsidian"


class VoicePipelineState(StrEnum):
    """Current state of the voice pipeline."""

    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"


class EntityType(StrEnum):
    """Type of entity node in the knowledge graph."""

    PERSON = "person"
    PLACE = "place"
    EVENT = "event"
    PROJECT = "project"
    GOAL = "goal"
    HABIT = "habit"
    EMOTION_TAG = "emotion_tag"
    TOPIC = "topic"
    ORGANIZATION = "organization"
    DOCUMENT = "document"
    DECISION = "decision"
    MILESTONE = "milestone"
    HEALTH = "health"
    FINANCIAL = "financial"
    RELATIONSHIP = "relationship"
    MEDIA = "media"


class RelationType(StrEnum):
    """Type of relationship edge in the knowledge graph."""

    INTERACTED_WITH = "interacted_with"
    LOCATED_AT = "located_at"
    WORKED_ON = "worked_on"
    FELT = "felt"
    DISCUSSED = "discussed"
    PLANNED = "planned"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    MENTIONED_IN = "mentioned_in"
    RELATED_TO = "related_to"
    BELONGS_TO = "belongs_to"
    CAUSED = "caused"
    PRECEDED = "preceded"
    FOLLOWED = "followed"
    PART_OF = "part_of"
    CONSUMED = "consumed"
    CREATED = "created"
    EXPERIENCED = "experienced"


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
    """An entity node in the knowledge graph.

    Represents a person, place, event, project, goal, habit, emotion,
    topic, organization, document, decision, milestone, health state,
    financial event, relationship, or media item.
    """

    entity_id: str
    name: str
    entity_type: EntityType
    aliases: list[str] = field(default_factory=list)
    first_seen: datetime | None = None
    last_seen: datetime | None = None
    mention_count: int = 1
    confidence: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GraphRelation:
    """A directed edge in the knowledge graph with temporal annotations.

    Links two entities with a typed relationship, confidence score,
    and temporal context (when first/last observed, date range).
    """

    source_id: str
    target_id: str
    relation_type: RelationType
    weight: float = 1.0
    confidence: float = 1.0
    first_seen: datetime | None = None
    last_seen: datetime | None = None
    date: datetime | None = None
    source_chunk_id: str = ""
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


# ── Proactive Feature Types ─────────────────────────────────────────────────


class NotificationPriority(StrEnum):
    """Priority level for proactive notifications."""

    LOW = "low"  # Can wait indefinitely (fun facts, gentle reminders)
    MEDIUM = "medium"  # Should be delivered within an hour (daily recap)
    HIGH = "high"  # Deliver at next natural pause (habit nudge, mood check-in)
    URGENT = "urgent"  # Deliver immediately (critical pattern detected)


class SuggestionCategory(StrEnum):
    """Category of a proactive suggestion."""

    FOLLOW_UP = "follow_up"  # Follow up on a previous topic
    REMINDER = "reminder"  # Remind about mentioned plans/goals
    CHECK_IN = "check_in"  # Emotional check-in based on mood
    DISCOVERY = "discovery"  # Interesting pattern found in memories
    SOCIAL = "social"  # Haven't mentioned someone in a while
    HABIT = "habit"  # Habit-related nudge


@dataclass(frozen=True)
class ProactiveSuggestion:
    """A proactive suggestion generated from graph patterns.

    Represents a contextual suggestion ARA can offer when the user is idle,
    based on knowledge graph patterns, temporal triggers, or detected habits.
    """

    suggestion_id: str = field(default_factory=lambda: uuid4().hex[:12])
    category: SuggestionCategory = SuggestionCategory.FOLLOW_UP
    message: str = ""  # Human-readable suggestion text
    context: str = ""  # Why this suggestion was generated
    source_entities: list[str] = field(default_factory=list)  # entity IDs that triggered it
    confidence: float = 0.5
    created_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DailyRecap:
    """Summary of a day's interactions and key events.

    Generated at end of day from conversation history and new memory entries.
    """

    recap_id: str = field(default_factory=lambda: uuid4().hex[:12])
    date: datetime = field(default_factory=datetime.now)
    summary: str = ""  # LLM-generated narrative summary
    conversation_count: int = 0
    topics_discussed: list[str] = field(default_factory=list)
    entities_mentioned: list[str] = field(default_factory=list)
    mood_summary: str = ""  # Overall emotional tone
    highlights: list[str] = field(default_factory=list)  # Key moments
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HabitPattern:
    """A detected recurring behavior pattern from graph edges.

    Habits are identified by repeated entity interactions or temporal patterns
    in the knowledge graph (e.g., "exercises every morning", "calls mom weekly").
    """

    pattern_id: str = field(default_factory=lambda: uuid4().hex[:12])
    description: str = ""  # Human-readable habit description
    entity_id: str = ""  # Primary entity involved
    entity_name: str = ""  # Display name
    relation_type: str = ""  # Edge type that recurs
    occurrence_count: int = 0  # How many times detected
    first_seen: datetime | None = None
    last_seen: datetime | None = None
    avg_interval_days: float = 0.0  # Average days between occurrences
    confidence: float = 0.5
    is_positive: bool = True  # Positive habit (exercise) vs neutral/negative
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MoodEntry:
    """A single mood data point for sentiment trend tracking.

    Captured from conversation emotion analysis over time.
    """

    entry_id: str = field(default_factory=lambda: uuid4().hex[:12])
    timestamp: datetime = field(default_factory=datetime.now)
    valence: float = 0.0  # -1.0 (very negative) to 1.0 (very positive)
    intensity: float = 0.0  # 0.0 (neutral) to 1.0 (very intense)
    dominant_emotion: str = ""  # e.g., "joy", "sadness"
    emotions: dict[str, float] = field(default_factory=dict)  # emotion → confidence
    source_conversation_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MoodTrend:
    """Aggregated mood trend over a time window.

    Computed from a rolling window of MoodEntry data points.
    """

    window_days: int = 7
    avg_valence: float = 0.0
    avg_intensity: float = 0.0
    dominant_emotion: str = ""
    emotion_distribution: dict[str, float] = field(default_factory=dict)
    trend_direction: str = ""  # "improving", "declining", "stable"
    data_points: int = 0
    start_date: datetime | None = None
    end_date: datetime | None = None


@dataclass(frozen=True)
class MemoryReview:
    """A periodic memory review (weekly or monthly) summarizing a life chapter.

    Generated from knowledge graph communities, temporal entity clusters,
    and vector store time-range queries.
    """

    review_id: str = field(default_factory=lambda: uuid4().hex[:12])
    period_type: str = ""  # "weekly" or "monthly"
    start_date: datetime | None = None
    end_date: datetime | None = None
    summary: str = ""  # LLM-generated narrative
    key_entities: list[str] = field(default_factory=list)  # Most active entities
    key_topics: list[str] = field(default_factory=list)  # Dominant topics
    mood_summary: str = ""  # Emotional arc over the period
    milestones: list[str] = field(default_factory=list)  # Notable achievements
    connections_made: int = 0  # New graph edges
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Notification:
    """A queued notification awaiting delivery at a natural pause.

    Mutable (not frozen) because delivery status changes over time.
    """

    notification_id: str = field(default_factory=lambda: uuid4().hex[:12])
    priority: NotificationPriority = NotificationPriority.MEDIUM
    title: str = ""  # Short headline
    message: str = ""  # Full notification text
    source: str = ""  # Which proactive feature generated it
    created_at: datetime = field(default_factory=datetime.now)
    expires_at: datetime | None = None  # TTL-based expiration
    delivered: bool = False
    delivered_at: datetime | None = None
    dismissed: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


# ── Obsidian Integration Types ──────────────────────────────────────────────


class ObsidianTaskStatus(StrEnum):
    """Status of a task extracted from Obsidian checklist."""

    PENDING = "pending"
    DONE = "done"
    CANCELLED = "cancelled"
    IN_PROGRESS = "in_progress"


class ObsidianQuarter(StrEnum):
    """Time-of-day quarter for OPERATIONAL LOG placement."""

    Q1 = "q1"  # 00:00 - 06:00
    Q2 = "q2"  # 06:00 - 12:00
    Q3 = "q3"  # 12:00 - 18:00
    Q4 = "q4"  # 18:00 - 24:00


@dataclass(frozen=True)
class ObsidianTask:
    """A task extracted from an Obsidian daily note checklist."""

    text: str
    status: ObsidianTaskStatus = ObsidianTaskStatus.PENDING
    section: str = ""  # Which section the task was found in
    source_date: datetime | None = None


@dataclass(frozen=True)
class ObsidianFrontmatter:
    """YAML frontmatter parsed from an Obsidian daily note.

    Maps to the user's daily note template fields:
    weight_kg, sleep_hrs, calories, workout_focus, effort_score, checklist_score.
    """

    date: str = ""
    weight_kg: float = 0.0
    sleep_hrs: float = 0.0
    calories: int = 0
    workout_focus: str = ""
    effort_score: float = 0.0
    checklist_score: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DailyNoteData:
    """Fully parsed Obsidian daily note with all sections extracted."""

    date: str  # YYYY-MM-DD
    path: str  # Vault-relative path
    frontmatter: ObsidianFrontmatter = field(default_factory=ObsidianFrontmatter)
    brainstorm: str = ""  # BRAINSTORM section content
    operational_log: dict[str, str] = field(default_factory=dict)  # Q1-Q4 content
    protocol_checklist: list[ObsidianTask] = field(default_factory=list)  # THE PROTOCOL tasks
    inbox_items: list[str] = field(default_factory=list)  # INBOX ZERO items
    tasks: list[ObsidianTask] = field(default_factory=list)  # All tasks found anywhere
    raw_content: str = ""  # Full markdown content
    content_hash: str = ""  # SHA-256 for change detection


@dataclass(frozen=True)
class ObsidianSyncState:
    """State of the sync engine — tracks what has been synced and when."""

    last_sync_at: datetime | None = None
    notes_synced: int = 0
    notes_changed: int = 0
    notes_ingested: int = 0
    errors: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class MorningBriefing:
    """Generated morning briefing from yesterday's Obsidian data + ARA context."""

    briefing_id: str = field(default_factory=lambda: uuid4().hex[:12])
    date: str = ""  # YYYY-MM-DD (today)
    yesterday_summary: str = ""  # What happened yesterday
    pending_tasks: list[str] = field(default_factory=list)  # Incomplete tasks from yesterday
    today_suggestions: list[str] = field(default_factory=list)  # What ARA recommends for today
    habit_status: str = ""  # How habits are tracking
    mood_context: str = ""  # Emotional arc context
    protocol_score: float = 0.0  # Yesterday's protocol completion %
    generated_at: datetime = field(default_factory=datetime.now)


@dataclass(frozen=True)
class ProtocolScore:
    """Scoring result from analyzing THE PROTOCOL checklist."""

    date: str = ""
    total_items: int = 0
    completed_items: int = 0
    score_pct: float = 0.0  # 0.0 to 100.0
    missed_items: list[str] = field(default_factory=list)
    streak_days: int = 0  # Consecutive days with >80% score
