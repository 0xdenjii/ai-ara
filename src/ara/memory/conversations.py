"""Conversation persistence and memory feedback loop.

Stores every conversation turn in SQLite, detects conversation endings,
then processes completed conversations through the memory feedback loop:
    store → summarize → chunk → embed → graph update

Architecture:
    - SQLite for raw transcript storage (conversations + turns tables)
    - End-of-conversation triggers: silence timeout, farewell, manual, session close
    - Background processing: LLM summary, semantic chunking, embedding, graph update
    - Deduplication: conversations are only processed once for memory feedback

Usage:
    from ara.memory.conversations import ConversationStore

    store = ConversationStore(settings)
    store.connect()

    # During conversation
    conv_id = store.start_conversation()
    store.add_turn(conv_id, user_msg="hello", assistant_msg="hey!")

    # End conversation
    store.end_conversation(conv_id, reason=ConversationEndReason.USER_FAREWELL)

    # Memory feedback (async)
    await store.process_for_memory(conv_id, embedder, vector_store, graph)
"""

from __future__ import annotations

import sqlite3
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import structlog

from ara.utils.errors import ConversationPersistenceError
from ara.utils.types import (
    ConversationEndReason,
    ConversationTurn,
    IntentType,
    MemoryChunk,
    MemorySource,
)

if TYPE_CHECKING:
    from ara.config.settings import ConversationSettings
    from ara.memory.chunking import TextChunker
    from ara.memory.embeddings import EmbeddingModel
    from ara.memory.graph import KnowledgeGraph
    from ara.memory.vector_store import VectorStore

log = structlog.get_logger(component="conversations")


# ── Farewell Detection ───────────────────────────────────────────────────────

_FAREWELL_PHRASES = frozenset(
    {
        "bye",
        "goodbye",
        "good bye",
        "goodnight",
        "good night",
        "see you",
        "see ya",
        "later",
        "talk later",
        "i'm done",
        "im done",
        "that's all",
        "thats all",
        "gotta go",
        "signing off",
        "peace out",
        "night",
        "nighty night",
        "bye bye",
    }
)


# ── Data Types ───────────────────────────────────────────────────────────────


@dataclass
class Conversation:
    """A complete conversation with metadata.

    Attributes:
        conversation_id: Unique identifier.
        started_at: When the conversation began.
        ended_at: When the conversation ended (None if still active).
        end_reason: Why the conversation ended.
        turns: List of conversation turns.
        summary: LLM-generated summary (populated after processing).
        processed: Whether this conversation has been processed for memory.
        metadata: Additional metadata.
    """

    conversation_id: str = field(default_factory=lambda: uuid4().hex[:16])
    started_at: datetime = field(default_factory=datetime.now)
    ended_at: datetime | None = None
    end_reason: ConversationEndReason | None = None
    turns: list[ConversationTurn] = field(default_factory=list)
    summary: str = ""
    processed: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def turn_count(self) -> int:
        """Number of turns in this conversation."""
        return len(self.turns)

    @property
    def is_active(self) -> bool:
        """Whether the conversation is still ongoing."""
        return self.ended_at is None

    @property
    def duration_seconds(self) -> float:
        """Duration of the conversation in seconds."""
        end = self.ended_at or datetime.now()
        return (end - self.started_at).total_seconds()


# ── SQL Schema ───────────────────────────────────────────────────────────────

_CREATE_CONVERSATIONS_TABLE = """
CREATE TABLE IF NOT EXISTS conversations (
    conversation_id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    end_reason TEXT,
    summary TEXT DEFAULT '',
    processed INTEGER DEFAULT 0,
    metadata TEXT DEFAULT '{}'
)
"""

_CREATE_TURNS_TABLE = """
CREATE TABLE IF NOT EXISTS turns (
    turn_id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    user_message TEXT NOT NULL,
    assistant_message TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    intent TEXT DEFAULT 'general',
    metadata TEXT DEFAULT '{}',
    FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id)
)
"""

_CREATE_TURNS_INDEX = """
CREATE INDEX IF NOT EXISTS idx_turns_conversation
ON turns(conversation_id)
"""


# ── Conversation Store ───────────────────────────────────────────────────────


class ConversationStore:
    """SQLite-backed conversation storage with memory feedback loop.

    Stores raw conversation transcripts and provides methods for
    end-of-conversation detection and memory processing.

    Args:
        settings: Conversation settings (db path, timeouts).
    """

    def __init__(self, settings: ConversationSettings) -> None:
        self._db_path = settings.db_path
        self._silence_timeout_s = settings.silence_timeout_s
        self._min_turns = settings.min_turns_to_persist
        self._conn: sqlite3.Connection | None = None
        self._active_conversations: dict[str, datetime] = {}
        self._lock = threading.Lock()

    @property
    def is_connected(self) -> bool:
        """Whether the database connection is open."""
        return self._conn is not None

    def connect(self) -> None:
        """Open the SQLite database and create tables if needed.

        Raises:
            ConversationPersistenceError: If the database cannot be opened.
        """
        if self._conn is not None:
            return

        try:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(
                str(self._db_path),
                check_same_thread=False,
            )
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.execute(_CREATE_CONVERSATIONS_TABLE)
            self._conn.execute(_CREATE_TURNS_TABLE)
            self._conn.execute(_CREATE_TURNS_INDEX)
            self._conn.commit()

            log.info(
                "conversation_store_connected",
                path=str(self._db_path),
            )
        except Exception as exc:
            raise ConversationPersistenceError(
                f"Failed to open conversation database: {exc}",
                operation="connect",
                details={"path": str(self._db_path)},
            ) from exc

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
            log.info("conversation_store_closed")

    def _ensure_connected(self) -> sqlite3.Connection:
        """Get the database connection, raising if not connected."""
        if self._conn is None:
            raise ConversationPersistenceError(
                "Conversation store not connected — call connect() first",
                operation="query",
            )
        return self._conn

    # ── Conversation Lifecycle ───────────────────────────────────────────────

    def start_conversation(
        self,
        *,
        conversation_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Start a new conversation.

        Args:
            conversation_id: Optional custom ID. Generated if not provided.
            metadata: Optional metadata to attach.

        Returns:
            The conversation_id.
        """
        conn = self._ensure_connected()
        conv_id = conversation_id or uuid4().hex[:16]
        now = datetime.now()

        import json

        with self._lock:
            conn.execute(
                "INSERT INTO conversations (conversation_id, started_at, metadata) VALUES (?, ?, ?)",
                (conv_id, now.isoformat(), json.dumps(metadata or {})),
            )
            conn.commit()
            self._active_conversations[conv_id] = now

        log.info("conversation_started", conversation_id=conv_id)
        return conv_id

    def add_turn(
        self,
        conversation_id: str,
        *,
        user_message: str,
        assistant_message: str,
        intent: IntentType = IntentType.GENERAL,
        metadata: dict[str, Any] | None = None,
    ) -> ConversationTurn:
        """Add a turn (user + assistant exchange) to a conversation.

        Args:
            conversation_id: The conversation to add to.
            user_message: What the user said.
            assistant_message: What ARA replied.
            intent: Classified intent of the user message.
            metadata: Optional turn metadata.

        Returns:
            The created ConversationTurn.

        Raises:
            ConversationPersistenceError: If the conversation doesn't exist.
        """
        conn = self._ensure_connected()
        import json

        turn = ConversationTurn(
            conversation_id=conversation_id,
            user_message=user_message,
            assistant_message=assistant_message,
            intent=intent,
            metadata=metadata or {},
        )

        with self._lock:
            conn.execute(
                "INSERT INTO turns (turn_id, conversation_id, user_message, assistant_message, timestamp, intent, metadata) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    turn.turn_id,
                    conversation_id,
                    user_message,
                    assistant_message,
                    turn.timestamp.isoformat(),
                    intent.value,
                    json.dumps(metadata or {}),
                ),
            )
            conn.commit()
            # Update last activity time (inside lock to avoid TOCTOU races)
            self._active_conversations[conversation_id] = datetime.now()

        log.debug(
            "turn_added",
            conversation_id=conversation_id,
            turn_id=turn.turn_id,
            intent=intent.value,
        )
        return turn

    def end_conversation(
        self,
        conversation_id: str,
        *,
        reason: ConversationEndReason = ConversationEndReason.MANUAL_TRIGGER,
        summary: str = "",
    ) -> None:
        """Mark a conversation as ended.

        Args:
            conversation_id: The conversation to end.
            reason: Why the conversation ended.
            summary: Optional pre-generated summary.
        """
        conn = self._ensure_connected()
        now = datetime.now()

        with self._lock:
            conn.execute(
                "UPDATE conversations SET ended_at = ?, end_reason = ?, summary = ? WHERE conversation_id = ?",
                (now.isoformat(), reason.value, summary, conversation_id),
            )
            conn.commit()

        self._active_conversations.pop(conversation_id, None)

        log.info(
            "conversation_ended",
            conversation_id=conversation_id,
            reason=reason.value,
        )

    # ── End-of-Conversation Detection (P4D-02) ──────────────────────────────

    def detect_farewell(self, message: str) -> bool:
        """Check if a user message contains a farewell phrase.

        Args:
            message: The user's message text.

        Returns:
            True if the message appears to be a goodbye.
        """
        msg_lower = message.lower().strip().rstrip("!.,?")
        return msg_lower in _FAREWELL_PHRASES

    def check_silence_timeout(self, conversation_id: str) -> bool:
        """Check if a conversation has exceeded the silence timeout.

        Args:
            conversation_id: The conversation to check.

        Returns:
            True if the silence timeout has been exceeded.
        """
        last_activity = self._active_conversations.get(conversation_id)
        if last_activity is None:
            return False

        elapsed = (datetime.now() - last_activity).total_seconds()
        return elapsed >= self._silence_timeout_s

    def get_timed_out_conversations(self) -> list[str]:
        """Get all active conversations that have exceeded silence timeout.

        Returns:
            List of conversation_ids that should be ended.
        """
        timed_out: list[str] = []
        now = datetime.now()
        threshold = timedelta(seconds=self._silence_timeout_s)

        # Copy items to avoid RuntimeError if dict is modified concurrently
        for conv_id, last_activity in list(self._active_conversations.items()):
            if now - last_activity >= threshold:
                timed_out.append(conv_id)

        return timed_out

    # ── Queries ──────────────────────────────────────────────────────────────

    def get_conversation(self, conversation_id: str) -> Conversation | None:
        """Load a full conversation with all turns.

        Args:
            conversation_id: The conversation to load.

        Returns:
            Conversation object, or None if not found.
        """
        conn = self._ensure_connected()
        import json

        row = conn.execute(
            "SELECT conversation_id, started_at, ended_at, end_reason, summary, processed, metadata FROM conversations WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()

        if row is None:
            return None

        # Load turns
        turn_rows = conn.execute(
            "SELECT turn_id, conversation_id, user_message, assistant_message, timestamp, intent, metadata FROM turns WHERE conversation_id = ? ORDER BY timestamp",
            (conversation_id,),
        ).fetchall()

        turns: list[ConversationTurn] = [
            ConversationTurn(
                turn_id=tr[0],
                conversation_id=tr[1],
                user_message=tr[2],
                assistant_message=tr[3],
                timestamp=datetime.fromisoformat(tr[4]),
                intent=IntentType(tr[5]) if tr[5] else IntentType.GENERAL,
                metadata=json.loads(tr[6]) if tr[6] else {},
            )
            for tr in turn_rows
        ]

        end_reason: ConversationEndReason | None = None
        if row[3]:
            end_reason = ConversationEndReason(row[3])

        return Conversation(
            conversation_id=row[0],
            started_at=datetime.fromisoformat(row[1]),
            ended_at=datetime.fromisoformat(row[2]) if row[2] else None,
            end_reason=end_reason,
            turns=turns,
            summary=row[4] or "",
            processed=bool(row[5]),
            metadata=json.loads(row[6]) if row[6] else {},
        )

    def get_recent_conversations(
        self,
        *,
        limit: int = 10,
        include_active: bool = False,
    ) -> list[Conversation]:
        """Get recent conversations ordered by start time (newest first).

        Args:
            limit: Maximum number of conversations to return.
            include_active: Whether to include still-active conversations.

        Returns:
            List of Conversation objects.
        """
        conn = self._ensure_connected()

        if include_active:
            query = "SELECT conversation_id FROM conversations ORDER BY started_at DESC, rowid DESC LIMIT ?"
            rows = conn.execute(query, (limit,)).fetchall()
        else:
            query = "SELECT conversation_id FROM conversations WHERE ended_at IS NOT NULL ORDER BY started_at DESC, rowid DESC LIMIT ?"
            rows = conn.execute(query, (limit,)).fetchall()

        conversations: list[Conversation] = []
        for row in rows:
            conv = self.get_conversation(row[0])
            if conv is not None:
                conversations.append(conv)

        return conversations

    def get_unprocessed_conversations(self) -> list[Conversation]:
        """Get ended conversations that haven't been processed for memory.

        Returns:
            List of Conversation objects awaiting memory feedback processing.
        """
        conn = self._ensure_connected()

        rows = conn.execute(
            "SELECT conversation_id FROM conversations WHERE ended_at IS NOT NULL AND processed = 0 AND (SELECT COUNT(*) FROM turns WHERE turns.conversation_id = conversations.conversation_id) >= ?",
            (self._min_turns,),
        ).fetchall()

        conversations: list[Conversation] = []
        for row in rows:
            conv = self.get_conversation(row[0])
            if conv is not None:
                conversations.append(conv)

        return conversations

    def mark_processed(self, conversation_id: str, *, summary: str = "") -> None:
        """Mark a conversation as processed for memory feedback.

        Args:
            conversation_id: The conversation to mark.
            summary: The generated summary text.
        """
        conn = self._ensure_connected()
        with self._lock:
            conn.execute(
                "UPDATE conversations SET processed = 1, summary = ? WHERE conversation_id = ?",
                (summary, conversation_id),
            )
            conn.commit()

        log.info(
            "conversation_marked_processed",
            conversation_id=conversation_id,
            summary_len=len(summary),
        )

    def count_conversations(self) -> int:
        """Get total number of conversations."""
        conn = self._ensure_connected()
        row = conn.execute("SELECT COUNT(*) FROM conversations").fetchone()
        return row[0] if row else 0

    def count_turns(self, conversation_id: str | None = None) -> int:
        """Get total number of turns (optionally for a specific conversation)."""
        conn = self._ensure_connected()
        if conversation_id:
            row = conn.execute(
                "SELECT COUNT(*) FROM turns WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
        else:
            row = conn.execute("SELECT COUNT(*) FROM turns").fetchone()
        return row[0] if row else 0

    # ── Memory Feedback Loop (P4D-03 through P4D-05) ────────────────────────

    def build_transcript(self, conversation: Conversation) -> str:
        """Build a formatted transcript from conversation turns.

        Args:
            conversation: The conversation to transcribe.

        Returns:
            Formatted transcript text.
        """
        lines: list[str] = []
        for turn in conversation.turns:
            lines.append(f"User: {turn.user_message}")
            lines.append(f"ARA: {turn.assistant_message}")
        return "\n".join(lines)

    async def process_for_memory(
        self,
        conversation: Conversation,
        *,
        chunker: TextChunker,
        embedder: EmbeddingModel,
        vector_store: VectorStore,
        graph: KnowledgeGraph | None = None,
        summarizer_fn: Any | None = None,
    ) -> dict[str, Any]:
        """Process a completed conversation through the memory feedback loop.

        Steps:
            1. Build transcript text
            2. Generate summary (via summarizer_fn or simple extraction)
            3. Chunk the transcript semantically
            4. Embed chunks via nomic-embed-text
            5. Upsert into LanceDB (source=conversation)
            6. Extract entities/relationships → update graph (if available)

        Args:
            conversation: The completed conversation to process.
            chunker: Text chunker for semantic segmentation.
            embedder: Embedding model for vectorization.
            vector_store: LanceDB vector store for persistence.
            graph: Optional knowledge graph for entity/relationship updates.
            summarizer_fn: Optional async callable(text) -> str for summary.

        Returns:
            Dict with processing stats.

        Raises:
            ConversationPersistenceError: If processing fails.
        """
        start = time.perf_counter()

        if conversation.processed:
            log.warning(
                "conversation_already_processed",
                conversation_id=conversation.conversation_id,
            )
            return {"skipped": True, "reason": "already_processed"}

        transcript = self.build_transcript(conversation)
        if not transcript.strip():
            return {"skipped": True, "reason": "empty_transcript"}

        stats: dict[str, Any] = {
            "conversation_id": conversation.conversation_id,
            "turns": conversation.turn_count,
        }

        # Step 1: Generate summary
        summary = ""
        if summarizer_fn is not None:
            try:
                summary = await summarizer_fn(transcript)
            except Exception as exc:
                log.warning(
                    "summary_generation_failed",
                    conversation_id=conversation.conversation_id,
                    error=str(exc),
                )
                summary = _extractive_summary(transcript)
        else:
            summary = _extractive_summary(transcript)

        stats["summary_len"] = len(summary)

        # Step 2: Chunk the transcript
        source_file = f"conversation://{conversation.conversation_id}"
        chunks = chunker.chunk_text(
            transcript,
            source_file=source_file,
            date=conversation.started_at,
            section_title=f"Conversation {conversation.started_at.strftime('%Y-%m-%d %H:%M')}",
        )

        stats["chunks_created"] = len(chunks)

        # Step 3: Embed chunks
        if chunks:
            texts = [c.text for c in chunks]
            embeddings = embedder.embed_documents(texts)

            embedded_chunks: list[MemoryChunk] = []
            for chunk, embedding in zip(chunks, embeddings, strict=True):
                embedded_chunks.append(
                    MemoryChunk(
                        chunk_id=chunk.chunk_id,
                        text=chunk.text,
                        source=MemorySource.CONVERSATION,
                        source_file=source_file,
                        date=conversation.started_at,
                        section_title=chunk.section_title,
                        embedding=embedding,
                        metadata={
                            **chunk.metadata,
                            "conversation_id": conversation.conversation_id,
                            "turn_count": conversation.turn_count,
                        },
                    )
                )

            # Step 4: Upsert into vector store
            stored = vector_store.upsert(embedded_chunks)
            stats["chunks_stored"] = stored

        # Step 5: Update graph (if available)
        if graph is not None:
            graph_stats = self._update_graph(conversation, transcript, graph)
            stats["graph_updates"] = graph_stats

        # Mark as processed
        self.mark_processed(conversation.conversation_id, summary=summary)

        elapsed_ms = (time.perf_counter() - start) * 1000
        stats["elapsed_ms"] = round(elapsed_ms, 1)

        log.info(
            "conversation_processed_for_memory",
            **stats,
        )

        return stats

    def _update_graph(
        self,
        conversation: Conversation,
        transcript: str,
        graph: KnowledgeGraph,
    ) -> dict[str, int]:
        """Extract entities from conversation and update the knowledge graph.

        Uses basic heuristics to identify discussed topics, people mentioned,
        and decisions made. Full LLM-based extraction can be added via
        the relationship_extractor in a future enhancement.

        Args:
            conversation: The conversation metadata.
            transcript: The full transcript text.
            graph: The knowledge graph to update.

        Returns:
            Dict with counts of entities and relations added.
        """
        from ara.utils.types import EntityType, RelationType

        entities_added = 0
        relations_added = 0

        # Add a node for this conversation as an Event
        conv_entity_id = f"conv_{conversation.conversation_id}"
        conv_date = conversation.started_at
        title = f"Conversation on {conv_date.strftime('%Y-%m-%d %H:%M')}"

        graph.add_entity(
            title,
            EntityType.EVENT,
            entity_id=conv_entity_id,
            date=conv_date,
        )
        entities_added += 1

        # Extract topic mentions from user messages
        for turn in conversation.turns:
            # Check if any existing graph entities are mentioned
            words = turn.user_message.split()
            candidates: list[str] = list(words)
            candidates.extend(f"{words[i]} {words[i + 1]}" for i in range(len(words) - 1))

            for candidate in candidates:
                if len(candidate) < 3:
                    continue
                clean = candidate.strip("?!.,;:'\"")
                if not clean:
                    continue

                entity = graph.get_entity_by_name(clean)
                if entity is not None:
                    # Link conversation event to mentioned entity
                    graph.add_relation(
                        source_id=conv_entity_id,
                        target_id=entity.entity_id,
                        relation_type=RelationType.DISCUSSED,
                        confidence=0.7,
                        date=conv_date,
                    )
                    relations_added += 1

        return {
            "entities_added": entities_added,
            "relations_added": relations_added,
        }


# ── Helper Functions ─────────────────────────────────────────────────────────


def _extractive_summary(transcript: str, *, max_sentences: int = 5) -> str:
    """Generate a simple extractive summary from the transcript.

    Picks the first N user messages as a rough summary. Used as
    fallback when LLM summarization is unavailable.

    Args:
        transcript: The full transcript text.
        max_sentences: Maximum number of sentences to include.

    Returns:
        Brief extractive summary.
    """
    lines = transcript.strip().split("\n")
    user_lines = [
        line.removeprefix("User: ").strip() for line in lines if line.startswith("User: ")
    ]

    if not user_lines:
        return ""

    selected = user_lines[:max_sentences]
    return "Topics discussed: " + "; ".join(selected)
