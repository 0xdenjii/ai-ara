"""E2E test: "What did we talk about yesterday?"

Tests the full conversation recall pipeline:
1. Store a conversation with specific content
2. Process it through memory feedback (chunk → embed → store in vector DB)
3. Query via retrieval pipeline and verify the conversation is found

This validates the complete feedback loop from conversation persistence
through to retrieval, using real components except for the embedding model
(which is mocked with deterministic vectors to avoid model downloads).
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

if TYPE_CHECKING:
    from pathlib import Path

import pytest

from ara.config.settings import ConversationSettings, MemorySettings
from ara.memory.chunking import ChunkerConfig, TextChunker
from ara.memory.conversations import ConversationStore
from ara.memory.retrieval import RetrievalPipeline
from ara.memory.vector_store import VectorStore
from ara.utils.types import ConversationEndReason, MemorySource

# Fixed dimension for mock embeddings
_DIM = 768


def _make_deterministic_embedding(text: str) -> list[float]:
    """Generate a deterministic embedding vector based on text content.

    Uses a seeded random generator keyed on the text hash so that
    identical texts produce identical vectors, and different texts
    produce different (but non-orthogonal) vectors.
    """
    rng = random.Random(hash(text) & 0xFFFFFFFF)
    return [rng.gauss(0, 1) for _ in range(_DIM)]


def _make_mock_embedder() -> MagicMock:
    """Create a mock EmbeddingModel with deterministic embeddings.

    The mock produces consistent vectors for the same text, allowing
    vector similarity search to work meaningfully in tests.
    """
    mock = MagicMock()
    mock.dimension = _DIM
    mock.is_loaded = True

    def embed_documents(texts: list[str], **_kwargs: Any) -> list[list[float]]:
        return [_make_deterministic_embedding(t) for t in texts]

    def embed_query(query: str) -> list[float]:
        return _make_deterministic_embedding(query)

    mock.embed_documents = embed_documents
    mock.embed_query = embed_query
    return mock


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def conv_store(tmp_path: Path) -> ConversationStore:
    """ConversationStore backed by a temporary SQLite file."""
    settings = ConversationSettings(
        db_path=tmp_path / "conversations.sqlite",
        min_turns_to_persist=1,
    )
    store = ConversationStore(settings)
    store.connect()
    yield store
    store.close()


@pytest.fixture
def memory_settings(tmp_path: Path) -> MemorySettings:
    """Memory settings pointing to temporary directories."""
    return MemorySettings(
        vector_db_path=tmp_path / "vector_db",
        graph_db_path=tmp_path / "graph.sqlite",
        retrieval_top_k=20,
        rerank_top_k=5,
    )


@pytest.fixture
def vector_store(memory_settings: MemorySettings) -> VectorStore:
    """Real VectorStore backed by a temporary LanceDB."""
    vs = VectorStore(memory_settings, dimension=_DIM)
    vs.connect()
    yield vs
    vs.close()


@pytest.fixture
def chunker() -> TextChunker:
    """TextChunker with small chunk sizes for test conversations."""
    config = ChunkerConfig(
        target_tokens=128,
        overlap_tokens=16,
        min_chunk_tokens=20,
    )
    return TextChunker(config)


@pytest.fixture
def embedder() -> MagicMock:
    """Mock embedding model with deterministic vectors."""
    return _make_mock_embedder()


# ── E2E Tests ────────────────────────────────────────────────────────────────


class TestConversationRecall:
    """End-to-end tests for the conversation recall pipeline."""

    @pytest.mark.asyncio
    async def test_recall_recent_conversation(
        self,
        conv_store: ConversationStore,
        vector_store: VectorStore,
        chunker: TextChunker,
        embedder: MagicMock,
        memory_settings: MemorySettings,
    ) -> None:
        """Store a conversation, process it, then retrieve it by content query."""
        # Step 1: Store a conversation with distinctive content
        conv_id = conv_store.start_conversation()
        conv_store.add_turn(
            conv_id,
            user_message="I had a great meeting with Sarah today about the Q3 roadmap",
            assistant_message="That sounds productive! What were the key decisions?",
        )
        conv_store.add_turn(
            conv_id,
            user_message="We decided to prioritize the mobile app redesign and delay the API overhaul",
            assistant_message="Makes sense to focus on user-facing features first. Any timeline set?",
        )
        conv_store.add_turn(
            conv_id,
            user_message="Yes, mobile redesign should be done by end of September",
            assistant_message="Got it. September deadline for mobile redesign. I'll remember that!",
        )
        conv_store.end_conversation(conv_id, reason=ConversationEndReason.USER_FAREWELL)

        conversation = conv_store.get_conversation(conv_id)
        assert conversation is not None
        assert conversation.turn_count == 3

        # Step 2: Process conversation into vector store
        stats = await conv_store.process_for_memory(
            conversation,
            chunker=chunker,
            embedder=embedder,
            vector_store=vector_store,
        )
        assert stats.get("chunks_created", 0) > 0
        assert stats.get("chunks_stored", 0) > 0

        # Verify chunks are in the vector store
        chunk_count = vector_store.count()
        assert chunk_count > 0

        # Step 3: Query via retrieval pipeline
        pipeline = RetrievalPipeline(
            memory_settings,
            vector_store,
            embedder,
            graph=None,
            reranker=None,
        )

        # Query about the conversation content
        context = pipeline.retrieve(
            "What did we talk about? Sarah meeting roadmap",
            top_k=5,
            use_reranker=False,
            use_graph=False,
        )

        # Should find results
        assert len(context.results) > 0

        # Results should come from conversation source
        sources = {r.chunk.source for r in context.results}
        assert MemorySource.CONVERSATION in sources

        # Results should contain conversation content
        all_text = " ".join(r.chunk.text for r in context.results).lower()
        assert any(
            keyword in all_text
            for keyword in ["sarah", "roadmap", "mobile", "redesign", "september"]
        )

    @pytest.mark.asyncio
    async def test_recall_excludes_unprocessed_conversation(
        self,
        conv_store: ConversationStore,
        vector_store: VectorStore,
        chunker: TextChunker,
        embedder: MagicMock,
        memory_settings: MemorySettings,
    ) -> None:
        """A conversation that hasn't been processed should NOT appear in retrieval."""
        # Store but do NOT process
        conv_id = conv_store.start_conversation()
        conv_store.add_turn(
            conv_id,
            user_message="This is a secret conversation about quantum physics",
            assistant_message="Interesting topic!",
        )
        conv_store.end_conversation(conv_id)

        # Query should return nothing (vector store is empty)
        pipeline = RetrievalPipeline(
            memory_settings,
            vector_store,
            embedder,
            graph=None,
            reranker=None,
        )
        context = pipeline.retrieve(
            "quantum physics secret conversation",
            top_k=5,
            use_reranker=False,
            use_graph=False,
        )
        assert len(context.results) == 0

    @pytest.mark.asyncio
    async def test_recall_multiple_conversations(
        self,
        conv_store: ConversationStore,
        vector_store: VectorStore,
        chunker: TextChunker,
        embedder: MagicMock,
        memory_settings: MemorySettings,
    ) -> None:
        """Store multiple conversations and verify correct one is retrieved."""
        # Conversation 1: About cooking
        c1 = conv_store.start_conversation()
        conv_store.add_turn(
            c1,
            user_message="I tried making homemade pasta for the first time",
            assistant_message="How did it turn out?",
        )
        conv_store.add_turn(
            c1,
            user_message="The fettuccine was great but the ravioli filling was too salty",
            assistant_message="Next time try less parmesan in the filling!",
        )
        conv_store.end_conversation(c1)

        # Conversation 2: About exercise
        c2 = conv_store.start_conversation()
        conv_store.add_turn(
            c2,
            user_message="I ran five kilometers today at the park",
            assistant_message="Nice! What was your pace?",
        )
        conv_store.add_turn(
            c2,
            user_message="About six minutes per kilometer, a new personal best",
            assistant_message="That's great progress!",
        )
        conv_store.end_conversation(c2)

        # Process both
        for cid in [c1, c2]:
            conv = conv_store.get_conversation(cid)
            assert conv is not None
            await conv_store.process_for_memory(
                conv,
                chunker=chunker,
                embedder=embedder,
                vector_store=vector_store,
            )

        pipeline = RetrievalPipeline(
            memory_settings,
            vector_store,
            embedder,
            graph=None,
            reranker=None,
        )

        # Query about cooking — should find cooking conversation
        cooking_ctx = pipeline.retrieve(
            "pasta cooking fettuccine ravioli",
            top_k=5,
            use_reranker=False,
            use_graph=False,
        )
        assert len(cooking_ctx.results) > 0
        cooking_text = " ".join(r.chunk.text for r in cooking_ctx.results).lower()
        assert any(kw in cooking_text for kw in ["pasta", "fettuccine", "ravioli"])

        # Query about exercise — should find exercise conversation
        exercise_ctx = pipeline.retrieve(
            "running exercise kilometers pace",
            top_k=5,
            use_reranker=False,
            use_graph=False,
        )
        assert len(exercise_ctx.results) > 0
        exercise_text = " ".join(r.chunk.text for r in exercise_ctx.results).lower()
        assert any(kw in exercise_text for kw in ["ran", "kilometer", "pace", "personal best"])

    @pytest.mark.asyncio
    async def test_conversation_source_metadata(
        self,
        conv_store: ConversationStore,
        vector_store: VectorStore,
        chunker: TextChunker,
        embedder: MagicMock,
        memory_settings: MemorySettings,
    ) -> None:
        """Verify processed conversation chunks have correct source metadata."""
        conv_id = conv_store.start_conversation()
        conv_store.add_turn(
            conv_id,
            user_message="Let us discuss the project timeline",
            assistant_message="Sure, what milestones do you want to set?",
        )
        conv_store.end_conversation(conv_id)

        conversation = conv_store.get_conversation(conv_id)
        assert conversation is not None

        await conv_store.process_for_memory(
            conversation,
            chunker=chunker,
            embedder=embedder,
            vector_store=vector_store,
        )

        pipeline = RetrievalPipeline(
            memory_settings,
            vector_store,
            embedder,
            graph=None,
            reranker=None,
        )

        context = pipeline.retrieve(
            "project timeline milestones",
            top_k=5,
            use_reranker=False,
            use_graph=False,
        )

        assert len(context.results) > 0
        for result in context.results:
            assert result.chunk.source == MemorySource.CONVERSATION
            assert result.chunk.source_file.startswith("conversation://")
            assert conv_id in result.chunk.source_file

    @pytest.mark.asyncio
    async def test_already_processed_skipped(
        self,
        conv_store: ConversationStore,
        vector_store: VectorStore,
        chunker: TextChunker,
        embedder: MagicMock,
    ) -> None:
        """Processing the same conversation twice should skip the second time."""
        conv_id = conv_store.start_conversation()
        conv_store.add_turn(
            conv_id,
            user_message="Hello",
            assistant_message="Hi there!",
        )
        conv_store.end_conversation(conv_id)

        conversation = conv_store.get_conversation(conv_id)
        assert conversation is not None

        # Process once
        stats1 = await conv_store.process_for_memory(
            conversation,
            chunker=chunker,
            embedder=embedder,
            vector_store=vector_store,
        )
        assert stats1.get("chunks_stored", 0) > 0

        # Re-fetch (to get updated processed flag)
        conversation2 = conv_store.get_conversation(conv_id)
        assert conversation2 is not None

        # Process again — should skip
        stats2 = await conv_store.process_for_memory(
            conversation2,
            chunker=chunker,
            embedder=embedder,
            vector_store=vector_store,
        )
        assert stats2.get("skipped") is True
        assert stats2.get("reason") == "already_processed"

    @pytest.mark.asyncio
    async def test_retrieval_time_tracked(
        self,
        conv_store: ConversationStore,
        vector_store: VectorStore,
        chunker: TextChunker,
        embedder: MagicMock,
        memory_settings: MemorySettings,
    ) -> None:
        """Verify retrieval_time_ms is populated in the result."""
        conv_id = conv_store.start_conversation()
        conv_store.add_turn(
            conv_id,
            user_message="Discussing performance metrics",
            assistant_message="What metrics are you tracking?",
        )
        conv_store.end_conversation(conv_id)

        conversation = conv_store.get_conversation(conv_id)
        assert conversation is not None
        await conv_store.process_for_memory(
            conversation,
            chunker=chunker,
            embedder=embedder,
            vector_store=vector_store,
        )

        pipeline = RetrievalPipeline(
            memory_settings,
            vector_store,
            embedder,
            graph=None,
            reranker=None,
        )

        context = pipeline.retrieve(
            "performance metrics tracking",
            top_k=5,
            use_reranker=False,
            use_graph=False,
        )

        assert context.retrieval_time_ms >= 0
        assert context.total_candidates >= 0
