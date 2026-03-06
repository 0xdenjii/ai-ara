"""Tests for memory subsystem components (embeddings, vector store, etc.)."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import numpy as np
import pytest

from ara.config.settings import MemorySettings
from ara.memory.embeddings import EmbeddingModel
from ara.memory.vector_store import VectorStore
from ara.utils.errors import AraMemoryError
from ara.utils.types import MemoryChunk, MemorySource

if TYPE_CHECKING:
    from pathlib import Path

# ── EmbeddingModel Tests ────────────────────────────────────────────────────


class TestEmbeddingModelInit:
    def test_initial_state(self) -> None:
        settings = MemorySettings()
        model = EmbeddingModel(settings)
        assert model.is_loaded is False
        assert model.dimension == 0

    def test_unload(self) -> None:
        settings = MemorySettings()
        model = EmbeddingModel(settings)
        model.unload()  # should not raise
        assert model.is_loaded is False


class TestEmbeddingModelOperations:
    """Tests with mocked sentence-transformers model."""

    def _make_model_with_mock(self) -> tuple[EmbeddingModel, MagicMock]:
        """Create an EmbeddingModel with a mocked SentenceTransformer."""
        settings = MemorySettings()
        model = EmbeddingModel(settings)

        mock_st = MagicMock()
        # Simulate encode returning (N, 768) numpy array
        mock_st.encode.return_value = np.random.default_rng(42).random((1, 768)).astype(np.float32)

        model._model = mock_st
        model._loaded = True
        model._dimension = 768
        return model, mock_st

    def test_embed_documents(self) -> None:
        model, mock_st = self._make_model_with_mock()
        mock_st.encode.return_value = np.random.default_rng(42).random((2, 768)).astype(np.float32)

        result = model.embed_documents(["hello", "world"])
        assert len(result) == 2
        assert len(result[0]) == 768
        mock_st.encode.assert_called_once()

        # Check document prefix was applied
        call_args = mock_st.encode.call_args[0][0]
        assert call_args[0].startswith("search_document: ")

    def test_embed_query(self) -> None:
        model, mock_st = self._make_model_with_mock()

        result = model.embed_query("what happened yesterday?")
        assert len(result) == 768
        mock_st.encode.assert_called_once()

        # Check query prefix was applied
        call_args = mock_st.encode.call_args[0][0]
        assert call_args[0].startswith("search_query: ")

    def test_embed_empty_list_returns_empty(self) -> None:
        model, _ = self._make_model_with_mock()

        result = model.embed_documents([])
        assert result == []

    def test_embed_batch_custom_prefix(self) -> None:
        model, mock_st = self._make_model_with_mock()
        mock_st.encode.return_value = np.random.default_rng(42).random((2, 768)).astype(np.float32)

        result = model.embed_batch(["a", "b"], prefix="custom: ")
        assert len(result) == 2
        call_args = mock_st.encode.call_args[0][0]
        assert call_args[0].startswith("custom: ")

    def test_embed_without_load_raises(self) -> None:
        settings = MemorySettings()
        model = EmbeddingModel(settings)

        with pytest.raises(AraMemoryError, match="not loaded"):
            model.embed_documents(["test"])

        with pytest.raises(AraMemoryError, match="not loaded"):
            model.embed_query("test")

        with pytest.raises(AraMemoryError, match="not loaded"):
            model.embed_batch(["test"])


# ── VectorStore Tests ───────────────────────────────────────────────────────

_DIM = 8  # Small dimension for fast tests


def _make_settings(tmp_path: Path) -> MemorySettings:
    """Create MemorySettings pointing to a temp directory."""
    return MemorySettings(vector_db_path=str(tmp_path / "vector_db"))


def _random_vector(seed: int = 0) -> list[float]:
    """Generate a deterministic random vector of _DIM dimensions."""
    rng = np.random.default_rng(seed)
    vec = rng.random(_DIM).astype(np.float32)
    # Normalize for cosine similarity
    norm = float(np.linalg.norm(vec))
    return (vec / norm).tolist()


def _make_chunk(
    chunk_id: str,
    text: str,
    *,
    source: MemorySource = MemorySource.DIARY,
    source_file: str = "diary_2024.md",
    embedding: list[float] | None = None,
    date: str | None = None,
    section_title: str = "",
) -> MemoryChunk:
    """Create a MemoryChunk with a random embedding for testing."""
    from datetime import datetime

    parsed_date = datetime.fromisoformat(date) if date else None
    return MemoryChunk(
        chunk_id=chunk_id,
        text=text,
        source=source,
        source_file=source_file,
        date=parsed_date,
        section_title=section_title,
        embedding=embedding or _random_vector(hash(chunk_id) % 10000),
        metadata={
            "contextual_header": f"Source: {source_file}",
            "token_count": len(text.split()),
            "chunk_index": 0,
        },
    )


class TestVectorStoreInit:
    def test_initial_state(self, tmp_path: Path) -> None:
        settings = _make_settings(tmp_path)
        store = VectorStore(settings, dimension=_DIM)
        assert store.is_connected is False
        assert store.dimension == _DIM

    def test_connect_creates_directory(self, tmp_path: Path) -> None:
        db_path = tmp_path / "vector_db"
        settings = MemorySettings(vector_db_path=str(db_path))
        store = VectorStore(settings, dimension=_DIM)

        assert not db_path.exists()
        store.connect()
        assert db_path.exists()
        assert store.is_connected is True

    def test_double_connect_is_noop(self, tmp_path: Path) -> None:
        settings = _make_settings(tmp_path)
        store = VectorStore(settings, dimension=_DIM)
        store.connect()
        store.connect()  # Should not raise
        assert store.is_connected is True

    def test_close(self, tmp_path: Path) -> None:
        settings = _make_settings(tmp_path)
        store = VectorStore(settings, dimension=_DIM)
        store.connect()
        store.close()
        assert store.is_connected is False


class TestVectorStoreUpsert:
    def test_upsert_single_chunk(self, tmp_path: Path) -> None:
        settings = _make_settings(tmp_path)
        store = VectorStore(settings, dimension=_DIM)
        store.connect()

        chunk = _make_chunk("c1", "Today was a good day.")
        count = store.upsert([chunk])

        assert count == 1
        assert store.count() == 1

    def test_upsert_multiple_chunks(self, tmp_path: Path) -> None:
        settings = _make_settings(tmp_path)
        store = VectorStore(settings, dimension=_DIM)
        store.connect()

        chunks = [
            _make_chunk("c1", "Morning meditation was peaceful."),
            _make_chunk("c2", "Had lunch with Sarah at the cafe."),
            _make_chunk("c3", "Evening walk in the park was refreshing."),
        ]
        count = store.upsert(chunks)

        assert count == 3
        assert store.count() == 3

    def test_upsert_replaces_existing_chunk(self, tmp_path: Path) -> None:
        settings = _make_settings(tmp_path)
        store = VectorStore(settings, dimension=_DIM)
        store.connect()

        chunk_v1 = _make_chunk("c1", "Original text.")
        store.upsert([chunk_v1])
        assert store.count() == 1

        chunk_v2 = _make_chunk("c1", "Updated text.")
        store.upsert([chunk_v2])
        assert store.count() == 1  # Still 1, not 2

    def test_upsert_empty_list_returns_zero(self, tmp_path: Path) -> None:
        settings = _make_settings(tmp_path)
        store = VectorStore(settings, dimension=_DIM)
        store.connect()

        count = store.upsert([])
        assert count == 0

    def test_upsert_skips_chunks_without_embedding(self, tmp_path: Path) -> None:
        settings = _make_settings(tmp_path)
        store = VectorStore(settings, dimension=_DIM)
        store.connect()

        chunk_no_embed = MemoryChunk(
            chunk_id="c1",
            text="No embedding here.",
            source=MemorySource.DIARY,
            source_file="test.md",
            embedding=[],  # Empty embedding
            metadata={"contextual_header": "", "token_count": 3, "chunk_index": 0},
        )
        chunk_with_embed = _make_chunk("c2", "Has embedding.")
        count = store.upsert([chunk_no_embed, chunk_with_embed])

        assert count == 1  # Only the chunk with embedding
        assert store.count() == 1

    def test_upsert_without_connect_raises(self, tmp_path: Path) -> None:
        settings = _make_settings(tmp_path)
        store = VectorStore(settings, dimension=_DIM)

        chunk = _make_chunk("c1", "Test text.")
        with pytest.raises(AraMemoryError, match="not connected"):
            store.upsert([chunk])


class TestVectorStoreSearch:
    def _build_store(self, tmp_path: Path) -> VectorStore:
        """Build a store with 5 pre-loaded chunks."""
        settings = _make_settings(tmp_path)
        store = VectorStore(settings, dimension=_DIM)
        store.connect()

        chunks = [
            _make_chunk("c1", "Morning meditation was peaceful.", date="2024-06-15"),
            _make_chunk("c2", "Had lunch with Sarah at the cafe.", date="2024-06-15"),
            _make_chunk("c3", "Evening walk in the park.", date="2024-06-16"),
            _make_chunk(
                "c4",
                "Project meeting went well.",
                source=MemorySource.CONVERSATION,
                source_file="conv_001.json",
                date="2024-06-17",
            ),
            _make_chunk("c5", "Read a great book before bed.", date="2024-06-17"),
        ]
        store.upsert(chunks)
        return store

    def test_search_returns_results(self, tmp_path: Path) -> None:
        store = self._build_store(tmp_path)

        results = store.search(_random_vector(42), top_k=3)
        assert len(results) <= 3
        assert len(results) > 0
        # Each result should have chunk_id and text
        assert "chunk_id" in results[0]
        assert "text" in results[0]

    def test_search_respects_top_k(self, tmp_path: Path) -> None:
        store = self._build_store(tmp_path)

        results = store.search(_random_vector(42), top_k=2)
        assert len(results) <= 2

    def test_search_source_filter(self, tmp_path: Path) -> None:
        store = self._build_store(tmp_path)

        results = store.search(
            _random_vector(42),
            top_k=10,
            source_filter=MemorySource.CONVERSATION,
        )
        # Only c4 has CONVERSATION source
        for r in results:
            assert r["source"] == "conversation"

    def test_search_on_empty_store_raises(self, tmp_path: Path) -> None:
        settings = _make_settings(tmp_path)
        store = VectorStore(settings, dimension=_DIM)
        store.connect()
        # No data inserted, table is None

        with pytest.raises(AraMemoryError, match="not connected or empty"):
            store.search(_random_vector(42))

    def test_search_without_connect_raises(self, tmp_path: Path) -> None:
        settings = _make_settings(tmp_path)
        store = VectorStore(settings, dimension=_DIM)

        with pytest.raises(AraMemoryError, match="not connected"):
            store.search(_random_vector(42))


class TestVectorStoreFullTextSearch:
    def _build_store(self, tmp_path: Path) -> VectorStore:
        """Build a store with distinct text content for FTS testing."""
        settings = _make_settings(tmp_path)
        store = VectorStore(settings, dimension=_DIM)
        store.connect()

        chunks = [
            _make_chunk("c1", "Morning meditation was peaceful and calming."),
            _make_chunk("c2", "Had lunch with Sarah at the Italian cafe downtown."),
            _make_chunk("c3", "Evening walk in the park with the dog."),
            _make_chunk("c4", "Project deadline is next Monday for the meditation app."),
        ]
        store.upsert(chunks)
        return store

    def test_fts_returns_results(self, tmp_path: Path) -> None:
        store = self._build_store(tmp_path)

        results = store.full_text_search("meditation", top_k=5)
        assert len(results) > 0
        # Results should contain 'meditation' in text
        texts = [r["text"] for r in results]
        assert any("meditation" in t.lower() for t in texts)

    def test_fts_respects_top_k(self, tmp_path: Path) -> None:
        store = self._build_store(tmp_path)

        results = store.full_text_search("the", top_k=2)
        assert len(results) <= 2

    def test_fts_on_empty_store_raises(self, tmp_path: Path) -> None:
        settings = _make_settings(tmp_path)
        store = VectorStore(settings, dimension=_DIM)
        store.connect()

        with pytest.raises(AraMemoryError, match="not connected or empty"):
            store.full_text_search("meditation")


class TestVectorStoreDeleteBySource:
    def test_delete_by_source_file(self, tmp_path: Path) -> None:
        settings = _make_settings(tmp_path)
        store = VectorStore(settings, dimension=_DIM)
        store.connect()

        chunks = [
            _make_chunk("c1", "Chunk from file A.", source_file="file_a.md"),
            _make_chunk("c2", "Another chunk from file A.", source_file="file_a.md"),
            _make_chunk("c3", "Chunk from file B.", source_file="file_b.md"),
        ]
        store.upsert(chunks)
        assert store.count() == 3

        deleted = store.delete_by_source("file_a.md")
        assert deleted == 2
        assert store.count() == 1

    def test_delete_nonexistent_source_returns_zero(self, tmp_path: Path) -> None:
        settings = _make_settings(tmp_path)
        store = VectorStore(settings, dimension=_DIM)
        store.connect()

        chunk = _make_chunk("c1", "Some text.", source_file="file_a.md")
        store.upsert([chunk])

        deleted = store.delete_by_source("nonexistent.md")
        assert deleted == 0
        assert store.count() == 1

    def test_delete_on_disconnected_store_returns_zero(self, tmp_path: Path) -> None:
        settings = _make_settings(tmp_path)
        store = VectorStore(settings, dimension=_DIM)
        # Not connected
        deleted = store.delete_by_source("file_a.md")
        assert deleted == 0


class TestVectorStoreCount:
    def test_count_empty(self, tmp_path: Path) -> None:
        settings = _make_settings(tmp_path)
        store = VectorStore(settings, dimension=_DIM)
        store.connect()
        assert store.count() == 0

    def test_count_after_upsert(self, tmp_path: Path) -> None:
        settings = _make_settings(tmp_path)
        store = VectorStore(settings, dimension=_DIM)
        store.connect()

        chunks = [_make_chunk(f"c{i}", f"Chunk number {i}.") for i in range(5)]
        store.upsert(chunks)
        assert store.count() == 5

    def test_count_disconnected_returns_zero(self, tmp_path: Path) -> None:
        settings = _make_settings(tmp_path)
        store = VectorStore(settings, dimension=_DIM)
        assert store.count() == 0


class TestVectorStoreClear:
    def test_clear_removes_all_data(self, tmp_path: Path) -> None:
        settings = _make_settings(tmp_path)
        store = VectorStore(settings, dimension=_DIM)
        store.connect()

        chunks = [_make_chunk(f"c{i}", f"Text {i}.") for i in range(3)]
        store.upsert(chunks)
        assert store.count() == 3

        store.clear()
        assert store.count() == 0

    def test_clear_on_empty_store_is_noop(self, tmp_path: Path) -> None:
        settings = _make_settings(tmp_path)
        store = VectorStore(settings, dimension=_DIM)
        store.connect()
        store.clear()  # Should not raise
        assert store.count() == 0

    def test_clear_disconnected_is_noop(self, tmp_path: Path) -> None:
        settings = _make_settings(tmp_path)
        store = VectorStore(settings, dimension=_DIM)
        store.clear()  # Should not raise

    def test_upsert_after_clear_works(self, tmp_path: Path) -> None:
        settings = _make_settings(tmp_path)
        store = VectorStore(settings, dimension=_DIM)
        store.connect()

        chunks = [_make_chunk("c1", "Before clear.")]
        store.upsert(chunks)
        store.clear()

        chunks2 = [_make_chunk("c2", "After clear.")]
        store.upsert(chunks2)
        assert store.count() == 1


class TestVectorStoreReconnect:
    def test_data_persists_after_reconnect(self, tmp_path: Path) -> None:
        settings = _make_settings(tmp_path)

        # First connection: insert data
        store1 = VectorStore(settings, dimension=_DIM)
        store1.connect()
        chunks = [_make_chunk("c1", "Persistent data.")]
        store1.upsert(chunks)
        store1.close()

        # Second connection: data should still be there
        store2 = VectorStore(settings, dimension=_DIM)
        store2.connect()
        assert store2.count() == 1

        results = store2.search(_random_vector(42), top_k=1)
        assert len(results) == 1
        assert results[0]["chunk_id"] == "c1"
        store2.close()
