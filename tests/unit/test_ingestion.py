"""Tests for the ingestion pipeline (parse -> chunk -> embed -> store)."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import numpy as np
import pytest

from ara.config.settings import MemorySettings
from ara.memory.chunking import ChunkerConfig, TextChunker
from ara.memory.embeddings import EmbeddingModel
from ara.memory.ingestion import (
    BatchIngestionResult,
    IngestionManifest,
    IngestionPipeline,
    IngestionResult,
)
from ara.memory.vector_store import VectorStore
from ara.utils.errors import AraMemoryError

if TYPE_CHECKING:
    from pathlib import Path

_DIM = 8  # Small dimension for fast tests


def _mock_embedder() -> EmbeddingModel:
    """Create an EmbeddingModel with a mocked SentenceTransformer."""
    settings = MemorySettings()
    model = EmbeddingModel(settings)

    mock_st = MagicMock()
    model._model = mock_st
    model._loaded = True
    model._dimension = _DIM

    def _encode_side_effect(texts: list[str], **kwargs: object) -> np.ndarray:
        rng = np.random.default_rng(42)
        return rng.random((len(texts), _DIM)).astype(np.float32)

    mock_st.encode.side_effect = _encode_side_effect
    return model


def _build_pipeline(tmp_path: Path) -> IngestionPipeline:
    """Build a pipeline with real LanceDB and mocked embedder."""
    settings = MemorySettings(
        vector_db_path=str(tmp_path / "vector_db"),
        chunk_size=512,
        chunk_overlap=64,
    )
    chunker = TextChunker(
        ChunkerConfig(
            target_tokens=settings.chunk_size,
            overlap_tokens=settings.chunk_overlap,
        )
    )
    embedder = _mock_embedder()
    store = VectorStore(settings, dimension=_DIM)
    store.connect()

    manifest_path = tmp_path / "vector_db" / "manifest.json"

    return IngestionPipeline(
        chunker=chunker,
        embedder=embedder,
        store=store,
        manifest_path=manifest_path,
    )


def _write_diary(tmp_path: Path, name: str, content: str) -> Path:
    """Write a diary file and return its path."""
    file_path = tmp_path / name
    file_path.write_text(content, encoding="utf-8")
    return file_path


# ── IngestionResult / BatchIngestionResult ──────────────────────────────────


class TestIngestionResult:
    def test_create(self) -> None:
        result = IngestionResult(
            source_file="diary.md",
            chunks_created=10,
            chunks_embedded=10,
            chunks_stored=10,
            elapsed_ms=123.4,
        )
        assert result.source_file == "diary.md"
        assert result.chunks_stored == 10
        assert result.skipped is False

    def test_skipped_result(self) -> None:
        result = IngestionResult(
            source_file="diary.md",
            chunks_created=0,
            chunks_embedded=0,
            chunks_stored=0,
            elapsed_ms=1.0,
            skipped=True,
            skip_reason="unchanged",
        )
        assert result.skipped is True
        assert result.skip_reason == "unchanged"


class TestBatchIngestionResult:
    def test_create(self) -> None:
        result = BatchIngestionResult(
            files_processed=3,
            files_skipped=1,
            total_chunks=25,
            total_elapsed_ms=500.0,
        )
        assert result.files_processed == 3
        assert result.total_chunks == 25


# ── IngestionManifest ───────────────────────────────────────────────────────


class TestIngestionManifest:
    def test_empty_manifest(self, tmp_path: Path) -> None:
        manifest = IngestionManifest(tmp_path / "manifest.json")
        assert manifest.files == []
        assert manifest.has_file("anything") is False
        assert manifest.get_hash("anything") is None

    def test_set_and_get_hash(self, tmp_path: Path) -> None:
        manifest = IngestionManifest(tmp_path / "manifest.json")
        manifest.set_hash("diary.md", "abc123")
        assert manifest.has_file("diary.md") is True
        assert manifest.get_hash("diary.md") == "abc123"

    def test_save_and_reload(self, tmp_path: Path) -> None:
        path = tmp_path / "manifest.json"
        manifest1 = IngestionManifest(path)
        manifest1.set_hash("file_a.md", "hash_a")
        manifest1.set_hash("file_b.md", "hash_b")
        manifest1.save()

        # Reload from disk
        manifest2 = IngestionManifest(path)
        assert manifest2.has_file("file_a.md")
        assert manifest2.get_hash("file_a.md") == "hash_a"
        assert manifest2.has_file("file_b.md")

    def test_remove(self, tmp_path: Path) -> None:
        manifest = IngestionManifest(tmp_path / "manifest.json")
        manifest.set_hash("diary.md", "abc123")
        manifest.remove("diary.md")
        assert manifest.has_file("diary.md") is False

    def test_remove_nonexistent_is_noop(self, tmp_path: Path) -> None:
        manifest = IngestionManifest(tmp_path / "manifest.json")
        manifest.remove("nonexistent.md")  # Should not raise

    def test_corrupted_manifest_loads_empty(self, tmp_path: Path) -> None:
        path = tmp_path / "manifest.json"
        path.write_text("not valid json {{{", encoding="utf-8")
        manifest = IngestionManifest(path)
        assert manifest.files == []

    def test_files_list(self, tmp_path: Path) -> None:
        manifest = IngestionManifest(tmp_path / "manifest.json")
        manifest.set_hash("a.md", "h1")
        manifest.set_hash("b.md", "h2")
        manifest.set_hash("c.md", "h3")
        assert sorted(manifest.files) == ["a.md", "b.md", "c.md"]


# ── IngestionPipeline: ingest_file ─────────────────────────────────────────


class TestIngestionPipelineIngestFile:
    def test_ingest_simple_file(self, tmp_path: Path) -> None:
        pipeline = _build_pipeline(tmp_path)
        file_path = _write_diary(tmp_path, "diary.md", "Today was a good day.")

        result = pipeline.ingest_file(file_path)

        assert result.source_file == "diary.md"
        assert result.chunks_created >= 1
        assert result.chunks_stored >= 1
        assert result.skipped is False
        assert result.elapsed_ms > 0
        assert result.file_hash != ""

    def test_ingest_with_markdown_headers(self, tmp_path: Path) -> None:
        pipeline = _build_pipeline(tmp_path)
        content = (
            "# 2024-06-15 — A Great Day\n\n"
            "Morning was peaceful. Went for a walk.\n\n"
            "## Afternoon\n\n"
            "Had lunch with friends at the park.\n"
        )
        file_path = _write_diary(tmp_path, "diary.md", content)

        result = pipeline.ingest_file(file_path)

        assert result.chunks_created >= 1
        assert result.chunks_stored >= 1

    def test_ingest_skips_unchanged_file(self, tmp_path: Path) -> None:
        pipeline = _build_pipeline(tmp_path)
        file_path = _write_diary(tmp_path, "diary.md", "Some content here.")

        # First ingestion
        result1 = pipeline.ingest_file(file_path)
        assert result1.skipped is False

        # Second ingestion — same content
        result2 = pipeline.ingest_file(file_path)
        assert result2.skipped is True
        assert result2.skip_reason == "unchanged"

    def test_ingest_reingests_changed_file(self, tmp_path: Path) -> None:
        pipeline = _build_pipeline(tmp_path)
        file_path = _write_diary(tmp_path, "diary.md", "Original content.")

        result1 = pipeline.ingest_file(file_path)
        assert result1.skipped is False

        # Modify the file
        file_path.write_text("Updated content with changes.", encoding="utf-8")

        result2 = pipeline.ingest_file(file_path)
        assert result2.skipped is False
        assert result2.chunks_stored >= 1

    def test_ingest_force_reingests(self, tmp_path: Path) -> None:
        pipeline = _build_pipeline(tmp_path)
        file_path = _write_diary(tmp_path, "diary.md", "Some content.")

        pipeline.ingest_file(file_path)
        result = pipeline.ingest_file(file_path, force=True)

        assert result.skipped is False
        assert result.chunks_stored >= 1

    def test_ingest_empty_file_skips(self, tmp_path: Path) -> None:
        pipeline = _build_pipeline(tmp_path)
        file_path = _write_diary(tmp_path, "empty.md", "   \n  \n  ")

        result = pipeline.ingest_file(file_path)

        assert result.skipped is True
        assert result.skip_reason == "empty_file"

    def test_ingest_nonexistent_file_raises(self, tmp_path: Path) -> None:
        pipeline = _build_pipeline(tmp_path)

        with pytest.raises(AraMemoryError, match="File not found"):
            pipeline.ingest_file(tmp_path / "nonexistent.md")

    def test_ingest_directory_as_file_raises(self, tmp_path: Path) -> None:
        pipeline = _build_pipeline(tmp_path)

        with pytest.raises(AraMemoryError, match="not a file"):
            pipeline.ingest_file(tmp_path)

    def test_ingest_custom_source_name(self, tmp_path: Path) -> None:
        pipeline = _build_pipeline(tmp_path)
        file_path = _write_diary(tmp_path, "diary.md", "Content here.")

        result = pipeline.ingest_file(file_path, source_name="custom_name.md")

        assert result.source_file == "custom_name.md"
        assert pipeline.manifest.has_file("custom_name.md")

    def test_ingest_updates_manifest(self, tmp_path: Path) -> None:
        pipeline = _build_pipeline(tmp_path)
        file_path = _write_diary(tmp_path, "diary.md", "Test content.")

        result = pipeline.ingest_file(file_path)

        assert pipeline.manifest.has_file("diary.md")
        assert pipeline.manifest.get_hash("diary.md") == result.file_hash


# ── IngestionPipeline: ingest_text ─────────────────────────────────────────


class TestIngestionPipelineIngestText:
    def test_ingest_text(self, tmp_path: Path) -> None:
        pipeline = _build_pipeline(tmp_path)

        result = pipeline.ingest_text(
            "A conversation about machine learning and neural networks.",
            source_name="conv_001",
        )

        assert result.source_file == "conv_001"
        assert result.chunks_stored >= 1
        assert result.skipped is False

    def test_ingest_empty_text_skips(self, tmp_path: Path) -> None:
        pipeline = _build_pipeline(tmp_path)

        result = pipeline.ingest_text("   ", source_name="conv_001")

        assert result.skipped is True
        assert result.skip_reason == "empty_text"

    def test_ingest_text_skips_unchanged(self, tmp_path: Path) -> None:
        pipeline = _build_pipeline(tmp_path)
        text = "Some conversation content."

        pipeline.ingest_text(text, source_name="conv_001")
        result = pipeline.ingest_text(text, source_name="conv_001")

        assert result.skipped is True
        assert result.skip_reason == "unchanged"

    def test_ingest_text_force_reingests(self, tmp_path: Path) -> None:
        pipeline = _build_pipeline(tmp_path)
        text = "Some conversation content."

        pipeline.ingest_text(text, source_name="conv_001")
        result = pipeline.ingest_text(text, source_name="conv_001", force=True)

        assert result.skipped is False
        assert result.chunks_stored >= 1


# ── IngestionPipeline: ingest_directory ────────────────────────────────────


class TestIngestionPipelineIngestDirectory:
    def test_ingest_directory(self, tmp_path: Path) -> None:
        pipeline = _build_pipeline(tmp_path)
        diary_dir = tmp_path / "diaries"
        diary_dir.mkdir()
        _write_diary(diary_dir, "day1.md", "First day entry.")
        _write_diary(diary_dir, "day2.md", "Second day entry.")
        _write_diary(diary_dir, "day3.md", "Third day entry.")

        result = pipeline.ingest_directory(diary_dir)

        assert result.files_processed == 3
        assert result.files_skipped == 0
        assert result.total_chunks >= 3
        assert len(result.results) == 3
        assert len(result.errors) == 0

    def test_ingest_directory_skips_unchanged(self, tmp_path: Path) -> None:
        pipeline = _build_pipeline(tmp_path)
        diary_dir = tmp_path / "diaries"
        diary_dir.mkdir()
        _write_diary(diary_dir, "day1.md", "First day entry.")
        _write_diary(diary_dir, "day2.md", "Second day entry.")

        pipeline.ingest_directory(diary_dir)
        result = pipeline.ingest_directory(diary_dir)

        assert result.files_processed == 2
        assert result.files_skipped == 2

    def test_ingest_directory_with_pattern(self, tmp_path: Path) -> None:
        pipeline = _build_pipeline(tmp_path)
        diary_dir = tmp_path / "diaries"
        diary_dir.mkdir()
        _write_diary(diary_dir, "diary.md", "Markdown file.")
        _write_diary(diary_dir, "notes.txt", "Text file.")

        result = pipeline.ingest_directory(diary_dir, pattern="*.md")

        assert result.files_processed == 1

    def test_ingest_directory_recursive(self, tmp_path: Path) -> None:
        pipeline = _build_pipeline(tmp_path)
        diary_dir = tmp_path / "diaries"
        sub_dir = diary_dir / "2024"
        sub_dir.mkdir(parents=True)
        _write_diary(diary_dir, "main.md", "Main diary.")
        _write_diary(sub_dir, "jan.md", "January entry.")

        result = pipeline.ingest_directory(diary_dir, recursive=True)

        assert result.files_processed == 2

    def test_ingest_directory_nonrecursive(self, tmp_path: Path) -> None:
        pipeline = _build_pipeline(tmp_path)
        diary_dir = tmp_path / "diaries"
        sub_dir = diary_dir / "2024"
        sub_dir.mkdir(parents=True)
        _write_diary(diary_dir, "main.md", "Main diary.")
        _write_diary(sub_dir, "jan.md", "January entry.")

        result = pipeline.ingest_directory(diary_dir, recursive=False)

        assert result.files_processed == 1

    def test_ingest_nonexistent_directory_raises(self, tmp_path: Path) -> None:
        pipeline = _build_pipeline(tmp_path)

        with pytest.raises(AraMemoryError, match="Directory not found"):
            pipeline.ingest_directory(tmp_path / "nonexistent")

    def test_ingest_directory_force(self, tmp_path: Path) -> None:
        pipeline = _build_pipeline(tmp_path)
        diary_dir = tmp_path / "diaries"
        diary_dir.mkdir()
        _write_diary(diary_dir, "day1.md", "Entry content.")

        pipeline.ingest_directory(diary_dir)
        result = pipeline.ingest_directory(diary_dir, force=True)

        assert result.files_skipped == 0
        assert result.files_processed == 1


# ── IngestionPipeline: from_settings ───────────────────────────────────────


class TestIngestionPipelineFromSettings:
    def test_from_settings_creates_pipeline(self, tmp_path: Path) -> None:
        settings = MemorySettings(
            vector_db_path=str(tmp_path / "vector_db"),
        )
        pipeline = IngestionPipeline.from_settings(settings)

        assert pipeline is not None
        assert pipeline.manifest is not None

    def test_shutdown(self, tmp_path: Path) -> None:
        pipeline = _build_pipeline(tmp_path)
        pipeline.shutdown()
        # Should not raise, embedder unloaded and store closed


# ── End-to-end: vector store integration ───────────────────────────────────


class TestIngestionEndToEnd:
    def test_ingested_chunks_are_searchable(self, tmp_path: Path) -> None:
        """Verify that ingested chunks can be found via vector search."""
        pipeline = _build_pipeline(tmp_path)
        file_path = _write_diary(
            tmp_path,
            "diary.md",
            "Today I went to the park and played with the dog.",
        )

        pipeline.ingest_file(file_path)

        # Search with a random vector (dimension matches)
        rng = np.random.default_rng(99)
        query_vec = rng.random(_DIM).astype(np.float32).tolist()

        results = pipeline._store.search(query_vec, top_k=5)
        assert len(results) >= 1
        assert results[0]["source_file"] == "diary.md"

    def test_reingest_replaces_old_chunks(self, tmp_path: Path) -> None:
        """Verify that re-ingesting replaces old chunks."""
        pipeline = _build_pipeline(tmp_path)
        file_path = _write_diary(tmp_path, "diary.md", "Original content.")

        pipeline.ingest_file(file_path)
        count_after_first = pipeline._store.count()

        file_path.write_text("Updated content.", encoding="utf-8")
        pipeline.ingest_file(file_path)
        count_after_second = pipeline._store.count()

        # Should have same count (replaced, not accumulated)
        assert count_after_second == count_after_first

    def test_multiple_files_accumulate(self, tmp_path: Path) -> None:
        """Verify that ingesting multiple files accumulates chunks."""
        pipeline = _build_pipeline(tmp_path)
        _write_diary(tmp_path, "a.md", "File A content.")
        _write_diary(tmp_path, "b.md", "File B content.")

        pipeline.ingest_file(tmp_path / "a.md")
        count_a = pipeline._store.count()

        pipeline.ingest_file(tmp_path / "b.md")
        count_ab = pipeline._store.count()

        assert count_ab > count_a
