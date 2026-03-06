"""Tests for conversation persistence and memory feedback loop.

Uses temporary SQLite databases (via tmp_path) and mocked external
dependencies (embedder, vector store, graph, chunker).
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest

from ara.memory.conversations import (
    Conversation,
    ConversationStore,
    _extractive_summary,
)
from ara.utils.errors import ConversationPersistenceError
from ara.utils.types import (
    ConversationEndReason,
    IntentType,
    MemoryChunk,
    MemorySource,
)

# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture()
def settings(tmp_path: object) -> MagicMock:
    """Mock ConversationSettings with a temp DB path."""
    from pathlib import Path

    s = MagicMock()
    s.db_path = Path(str(tmp_path)) / "test_conversations.sqlite"
    s.silence_timeout_s = 300
    s.min_turns_to_persist = 2
    return s


@pytest.fixture()
def store(settings: MagicMock) -> ConversationStore:
    """Connected ConversationStore for testing."""
    cs = ConversationStore(settings)
    cs.connect()
    yield cs  # type: ignore[misc]
    cs.close()


# ── Conversation Dataclass Tests ─────────────────────────────────────────────


class TestConversationDataclass:
    """Tests for the Conversation dataclass."""

    def test_defaults(self) -> None:
        c = Conversation()
        assert len(c.conversation_id) == 16
        assert c.is_active
        assert c.turn_count == 0
        assert c.summary == ""
        assert not c.processed
        assert c.duration_seconds >= 0

    def test_properties(self) -> None:
        c = Conversation(
            started_at=datetime(2024, 1, 1, 10, 0),
            ended_at=datetime(2024, 1, 1, 10, 30),
        )
        assert not c.is_active
        assert c.duration_seconds == pytest.approx(1800.0)

    def test_active_when_no_ended_at(self) -> None:
        c = Conversation()
        assert c.is_active


# ── Lifecycle Tests ──────────────────────────────────────────────────────────


class TestLifecycle:
    """Tests for connect/close lifecycle."""

    def test_connect(self, settings: MagicMock) -> None:
        cs = ConversationStore(settings)
        assert not cs.is_connected
        cs.connect()
        assert cs.is_connected
        cs.close()

    def test_double_connect_noop(self, store: ConversationStore) -> None:
        assert store.is_connected
        store.connect()
        assert store.is_connected

    def test_close(self, settings: MagicMock) -> None:
        cs = ConversationStore(settings)
        cs.connect()
        cs.close()
        assert not cs.is_connected

    def test_close_when_not_connected(self, settings: MagicMock) -> None:
        cs = ConversationStore(settings)
        cs.close()  # Should not raise

    def test_query_without_connect_raises(self, settings: MagicMock) -> None:
        cs = ConversationStore(settings)
        with pytest.raises(ConversationPersistenceError, match="not connected"):
            cs.start_conversation()


# ── Conversation CRUD Tests ──────────────────────────────────────────────────


class TestConversationCRUD:
    """Tests for starting, adding turns, and ending conversations."""

    def test_start_conversation(self, store: ConversationStore) -> None:
        conv_id = store.start_conversation()
        assert len(conv_id) == 16
        assert store.count_conversations() == 1

    def test_start_with_custom_id(self, store: ConversationStore) -> None:
        conv_id = store.start_conversation(conversation_id="custom123")
        assert conv_id == "custom123"

    def test_start_with_metadata(self, store: ConversationStore) -> None:
        conv_id = store.start_conversation(metadata={"source": "voice"})
        conv = store.get_conversation(conv_id)
        assert conv is not None
        assert conv.metadata.get("source") == "voice"

    def test_add_turn(self, store: ConversationStore) -> None:
        conv_id = store.start_conversation()
        turn = store.add_turn(
            conv_id,
            user_message="hello",
            assistant_message="hey there!",
        )
        assert turn.turn_id
        assert turn.user_message == "hello"
        assert turn.assistant_message == "hey there!"
        assert store.count_turns(conv_id) == 1

    def test_add_multiple_turns(self, store: ConversationStore) -> None:
        conv_id = store.start_conversation()
        store.add_turn(conv_id, user_message="hi", assistant_message="hello")
        store.add_turn(conv_id, user_message="how are you?", assistant_message="great!")
        assert store.count_turns(conv_id) == 2

    def test_add_turn_with_intent(self, store: ConversationStore) -> None:
        conv_id = store.start_conversation()
        turn = store.add_turn(
            conv_id,
            user_message="what did I do yesterday?",
            assistant_message="You went hiking.",
            intent=IntentType.REMINISCING,
        )
        assert turn.intent == IntentType.REMINISCING

    def test_end_conversation(self, store: ConversationStore) -> None:
        conv_id = store.start_conversation()
        store.add_turn(conv_id, user_message="hi", assistant_message="hey")
        store.end_conversation(conv_id, reason=ConversationEndReason.USER_FAREWELL)

        conv = store.get_conversation(conv_id)
        assert conv is not None
        assert not conv.is_active
        assert conv.end_reason == ConversationEndReason.USER_FAREWELL

    def test_get_nonexistent_conversation(self, store: ConversationStore) -> None:
        assert store.get_conversation("nonexistent") is None

    def test_get_conversation_loads_turns(self, store: ConversationStore) -> None:
        conv_id = store.start_conversation()
        store.add_turn(conv_id, user_message="hello", assistant_message="hi")
        store.add_turn(conv_id, user_message="bye", assistant_message="goodbye")

        conv = store.get_conversation(conv_id)
        assert conv is not None
        assert conv.turn_count == 2
        assert conv.turns[0].user_message == "hello"
        assert conv.turns[1].user_message == "bye"

    def test_count_all_turns(self, store: ConversationStore) -> None:
        c1 = store.start_conversation()
        c2 = store.start_conversation()
        store.add_turn(c1, user_message="a", assistant_message="b")
        store.add_turn(c2, user_message="c", assistant_message="d")
        store.add_turn(c2, user_message="e", assistant_message="f")
        assert store.count_turns() == 3


# ── End-of-Conversation Detection Tests ──────────────────────────────────────


class TestEndOfConversation:
    """Tests for farewell detection and silence timeout."""

    def test_detect_farewell_positive(self, store: ConversationStore) -> None:
        assert store.detect_farewell("bye")
        assert store.detect_farewell("Goodbye!")
        assert store.detect_farewell("good night")
        assert store.detect_farewell("see you")
        assert store.detect_farewell("I'm done")
        assert store.detect_farewell("that's all.")

    def test_detect_farewell_negative(self, store: ConversationStore) -> None:
        assert not store.detect_farewell("hello")
        assert not store.detect_farewell("what's up?")
        assert not store.detect_farewell("tell me more")
        assert not store.detect_farewell("bye the way")

    def test_silence_timeout_not_exceeded(self, store: ConversationStore) -> None:
        conv_id = store.start_conversation()
        assert not store.check_silence_timeout(conv_id)

    def test_silence_timeout_exceeded(self, store: ConversationStore) -> None:
        conv_id = store.start_conversation()
        # Manually set last activity to past
        store._active_conversations[conv_id] = datetime.now() - timedelta(seconds=400)
        assert store.check_silence_timeout(conv_id)

    def test_silence_timeout_unknown_conversation(self, store: ConversationStore) -> None:
        assert not store.check_silence_timeout("unknown_id")

    def test_get_timed_out_conversations(self, store: ConversationStore) -> None:
        c1 = store.start_conversation()
        c2 = store.start_conversation()
        # c1 is timed out, c2 is recent
        store._active_conversations[c1] = datetime.now() - timedelta(seconds=400)
        store._active_conversations[c2] = datetime.now()

        timed_out = store.get_timed_out_conversations()
        assert c1 in timed_out
        assert c2 not in timed_out


# ── Query Tests ──────────────────────────────────────────────────────────────


class TestQueries:
    """Tests for conversation queries."""

    def test_get_recent_conversations(self, store: ConversationStore) -> None:
        c1 = store.start_conversation()
        store.end_conversation(c1)
        c2 = store.start_conversation()
        store.end_conversation(c2)

        recent = store.get_recent_conversations(limit=10)
        assert len(recent) == 2
        # Newest first
        assert recent[0].conversation_id == c2

    def test_get_recent_excludes_active(self, store: ConversationStore) -> None:
        c1 = store.start_conversation()
        store.end_conversation(c1)
        store.start_conversation()  # Active, not ended

        recent = store.get_recent_conversations(include_active=False)
        assert len(recent) == 1

    def test_get_recent_includes_active(self, store: ConversationStore) -> None:
        c1 = store.start_conversation()
        store.end_conversation(c1)
        store.start_conversation()  # Active

        recent = store.get_recent_conversations(include_active=True)
        assert len(recent) == 2

    def test_get_unprocessed_conversations(self, store: ConversationStore) -> None:
        # c1: ended, enough turns, not processed → should appear
        c1 = store.start_conversation()
        store.add_turn(c1, user_message="a", assistant_message="b")
        store.add_turn(c1, user_message="c", assistant_message="d")
        store.end_conversation(c1)

        # c2: ended, not enough turns → should NOT appear (min_turns=2)
        c2 = store.start_conversation()
        store.add_turn(c2, user_message="x", assistant_message="y")
        store.end_conversation(c2)

        # c3: still active → should NOT appear
        c3 = store.start_conversation()
        store.add_turn(c3, user_message="hi", assistant_message="hey")
        store.add_turn(c3, user_message="more", assistant_message="stuff")

        unprocessed = store.get_unprocessed_conversations()
        ids = [c.conversation_id for c in unprocessed]
        assert c1 in ids
        assert c2 not in ids
        assert c3 not in ids

    def test_mark_processed(self, store: ConversationStore) -> None:
        conv_id = store.start_conversation()
        store.add_turn(conv_id, user_message="a", assistant_message="b")
        store.add_turn(conv_id, user_message="c", assistant_message="d")
        store.end_conversation(conv_id)

        store.mark_processed(conv_id, summary="We discussed topics A and B.")

        conv = store.get_conversation(conv_id)
        assert conv is not None
        assert conv.processed
        assert conv.summary == "We discussed topics A and B."

    def test_processed_excluded_from_unprocessed(self, store: ConversationStore) -> None:
        conv_id = store.start_conversation()
        store.add_turn(conv_id, user_message="a", assistant_message="b")
        store.add_turn(conv_id, user_message="c", assistant_message="d")
        store.end_conversation(conv_id)
        store.mark_processed(conv_id)

        assert len(store.get_unprocessed_conversations()) == 0


# ── Transcript Tests ─────────────────────────────────────────────────────────


class TestTranscript:
    """Tests for build_transcript."""

    def test_build_transcript(self, store: ConversationStore) -> None:
        conv = Conversation(
            turns=[],
        )
        # Manually set turns since ConversationTurn is frozen
        from ara.utils.types import ConversationTurn

        conv.turns = [
            ConversationTurn(
                user_message="hello",
                assistant_message="hi there!",
            ),
            ConversationTurn(
                user_message="how are you?",
                assistant_message="I'm great!",
            ),
        ]
        transcript = store.build_transcript(conv)
        assert "User: hello" in transcript
        assert "ARA: hi there!" in transcript
        assert "User: how are you?" in transcript
        assert "ARA: I'm great!" in transcript

    def test_empty_conversation_transcript(self, store: ConversationStore) -> None:
        conv = Conversation()
        transcript = store.build_transcript(conv)
        assert transcript == ""


# ── Extractive Summary Tests ─────────────────────────────────────────────────


class TestExtractiveSummary:
    """Tests for _extractive_summary."""

    def test_basic_summary(self) -> None:
        transcript = "User: hello\nARA: hi\nUser: how's the weather?\nARA: sunny"
        summary = _extractive_summary(transcript)
        assert "Topics discussed:" in summary
        assert "hello" in summary
        assert "weather" in summary

    def test_empty_transcript(self) -> None:
        assert _extractive_summary("") == ""

    def test_no_user_lines(self) -> None:
        assert _extractive_summary("ARA: hello") == ""

    def test_max_sentences_limit(self) -> None:
        lines = "\n".join([f"User: message {i}\nARA: reply {i}" for i in range(20)])
        summary = _extractive_summary(lines, max_sentences=3)
        # Should only have 3 user messages
        assert summary.count(";") <= 2  # N-1 separators for N items


# ── Memory Feedback Loop Tests ───────────────────────────────────────────────


class TestMemoryFeedback:
    """Tests for process_for_memory (the async memory feedback loop)."""

    def _make_conversation(self, n_turns: int = 3) -> Conversation:
        from ara.utils.types import ConversationTurn

        turns = [
            ConversationTurn(
                user_message=f"user message {i}",
                assistant_message=f"assistant reply {i}",
            )
            for i in range(n_turns)
        ]
        return Conversation(
            started_at=datetime(2024, 6, 15, 10, 0),
            ended_at=datetime(2024, 6, 15, 10, 30),
            end_reason=ConversationEndReason.USER_FAREWELL,
            turns=turns,
        )

    def _mock_deps(self) -> tuple[MagicMock, MagicMock, MagicMock, MagicMock]:
        chunker = MagicMock()
        chunker.chunk_text.return_value = [
            MemoryChunk(
                chunk_id="ch1",
                text="chunk text",
                source=MemorySource.CONVERSATION,
            )
        ]

        embedder = MagicMock()
        embedder.embed_documents.return_value = [[0.1] * 768]

        vector_store = MagicMock()
        vector_store.upsert.return_value = 1

        graph = MagicMock()
        graph.get_entity_by_name.return_value = None
        graph.add_entity.return_value = None
        graph.add_relation.return_value = None

        return chunker, embedder, vector_store, graph

    def test_process_for_memory(self, store: ConversationStore) -> None:
        conv = self._make_conversation()
        chunker, embedder, vector_store, graph = self._mock_deps()

        result = asyncio.get_event_loop().run_until_complete(
            store.process_for_memory(
                conv,
                chunker=chunker,
                embedder=embedder,
                vector_store=vector_store,
                graph=graph,
            )
        )

        assert result["turns"] == 3
        assert result["chunks_created"] == 1
        assert result["chunks_stored"] == 1
        assert "graph_updates" in result

    def test_process_skips_already_processed(self, store: ConversationStore) -> None:
        conv = self._make_conversation()
        conv.processed = True
        chunker, embedder, vector_store, _ = self._mock_deps()

        result = asyncio.get_event_loop().run_until_complete(
            store.process_for_memory(
                conv,
                chunker=chunker,
                embedder=embedder,
                vector_store=vector_store,
            )
        )

        assert result["skipped"]
        embedder.embed_documents.assert_not_called()

    def test_process_without_graph(self, store: ConversationStore) -> None:
        conv = self._make_conversation()
        chunker, embedder, vector_store, _ = self._mock_deps()

        result = asyncio.get_event_loop().run_until_complete(
            store.process_for_memory(
                conv,
                chunker=chunker,
                embedder=embedder,
                vector_store=vector_store,
                graph=None,
            )
        )

        assert "graph_updates" not in result
        assert result["chunks_created"] == 1

    def test_process_with_summarizer(self, store: ConversationStore) -> None:
        conv = self._make_conversation()
        chunker, embedder, vector_store, _ = self._mock_deps()

        async def mock_summarizer(text: str) -> str:
            return "Summary: discussed important topics"

        result = asyncio.get_event_loop().run_until_complete(
            store.process_for_memory(
                conv,
                chunker=chunker,
                embedder=embedder,
                vector_store=vector_store,
                summarizer_fn=mock_summarizer,
            )
        )

        assert result["summary_len"] > 0

    def test_process_empty_transcript(self, store: ConversationStore) -> None:
        conv = Conversation()  # No turns
        chunker, embedder, vector_store, _ = self._mock_deps()

        result = asyncio.get_event_loop().run_until_complete(
            store.process_for_memory(
                conv,
                chunker=chunker,
                embedder=embedder,
                vector_store=vector_store,
            )
        )

        assert result["skipped"]
        assert result["reason"] == "empty_transcript"

    def test_process_calls_vector_store_upsert(self, store: ConversationStore) -> None:
        conv = self._make_conversation()
        chunker, embedder, vector_store, _ = self._mock_deps()

        asyncio.get_event_loop().run_until_complete(
            store.process_for_memory(
                conv,
                chunker=chunker,
                embedder=embedder,
                vector_store=vector_store,
            )
        )

        vector_store.upsert.assert_called_once()
        chunks_arg = vector_store.upsert.call_args[0][0]
        assert len(chunks_arg) == 1
        assert chunks_arg[0].source == MemorySource.CONVERSATION

    def test_process_graph_adds_event_entity(self, store: ConversationStore) -> None:
        conv = self._make_conversation()
        chunker, embedder, vector_store, graph = self._mock_deps()

        asyncio.get_event_loop().run_until_complete(
            store.process_for_memory(
                conv,
                chunker=chunker,
                embedder=embedder,
                vector_store=vector_store,
                graph=graph,
            )
        )

        graph.add_entity.assert_called_once()
        call_kwargs = graph.add_entity.call_args[1]
        assert "conv_" in call_kwargs["entity_id"]
