"""Full ingestion pipeline for diary files and text documents.

Orchestrates the complete flow: read file -> chunk -> embed -> store.
Handles re-ingestion (delete old chunks, re-process), file checksums
for change detection, and batch processing of directories.

All configuration comes from AraSettings.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from ara.memory.chunking import ChunkerConfig, TextChunker
from ara.memory.embeddings import EmbeddingModel
from ara.memory.vector_store import VectorStore
from ara.utils.errors import AraMemoryError
from ara.utils.logging import get_logger
from ara.utils.types import MemoryChunk

if TYPE_CHECKING:
    from ara.config.settings import MemorySettings

log = get_logger("ingestion")


@dataclass(frozen=True)
class IngestionResult:
    """Result of ingesting a single file."""

    source_file: str
    chunks_created: int
    chunks_embedded: int
    chunks_stored: int
    elapsed_ms: float
    skipped: bool = False
    skip_reason: str = ""
    file_hash: str = ""


@dataclass(frozen=True)
class BatchIngestionResult:
    """Result of ingesting multiple files."""

    files_processed: int
    files_skipped: int
    total_chunks: int
    total_elapsed_ms: float
    results: list[IngestionResult] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class IngestionManifest:
    """Tracks which files have been ingested and their checksums.

    Persisted as a JSON file alongside the vector DB so we can detect
    changed files and skip unchanged ones on re-ingestion.
    """

    def __init__(self, manifest_path: Path) -> None:
        self._path = manifest_path
        self._entries: dict[str, str] = {}  # source_file -> file_hash
        self._load()

    def _load(self) -> None:
        """Load manifest from disk if it exists."""
        if self._path.exists():
            try:
                data = json.loads(self._path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    self._entries = data
            except (json.JSONDecodeError, OSError) as exc:
                log.warning(
                    "manifest_load_failed",
                    path=str(self._path),
                    error=str(exc),
                )
                self._entries = {}

    def save(self) -> None:
        """Persist manifest to disk."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(self._entries, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def get_hash(self, source_file: str) -> str | None:
        """Get the stored hash for a source file, or None if not ingested."""
        return self._entries.get(source_file)

    def set_hash(self, source_file: str, file_hash: str) -> None:
        """Record the hash for an ingested file."""
        self._entries[source_file] = file_hash

    def remove(self, source_file: str) -> None:
        """Remove a file from the manifest."""
        self._entries.pop(source_file, None)

    def has_file(self, source_file: str) -> bool:
        """Check if a file has been ingested."""
        return source_file in self._entries

    @property
    def files(self) -> list[str]:
        """List all ingested file names."""
        return list(self._entries.keys())


class IngestionPipeline:
    """Orchestrates file ingestion: read -> chunk -> embed -> store.

    Components are provided at construction time and must be initialized
    (loaded/connected) before calling ingest methods. This allows the
    caller to control lifecycle and share components across pipelines.

    Usage:
        chunker = TextChunker(ChunkerConfig(target_tokens=512))
        embedder = EmbeddingModel(settings)
        embedder.load()
        store = VectorStore(settings, dimension=embedder.dimension)
        store.connect()

        pipeline = IngestionPipeline(
            chunker=chunker,
            embedder=embedder,
            store=store,
            manifest_path=Path("data/vector_db/manifest.json"),
        )
        result = pipeline.ingest_file(Path("data/diaries/diary_2024.md"))
    """

    def __init__(
        self,
        *,
        chunker: TextChunker,
        embedder: EmbeddingModel,
        store: VectorStore,
        manifest_path: Path,
        embedding_batch_size: int = 32,
    ) -> None:
        self._chunker = chunker
        self._embedder = embedder
        self._store = store
        self._manifest = IngestionManifest(manifest_path)
        self._embedding_batch_size = embedding_batch_size

    @classmethod
    def from_settings(cls, settings: MemorySettings) -> IngestionPipeline:
        """Create a pipeline with default components from settings.

        Note: The embedder and store are NOT loaded/connected yet.
        Call initialize() after construction.
        """
        chunker = TextChunker(
            ChunkerConfig(
                target_tokens=settings.chunk_size,
                overlap_tokens=settings.chunk_overlap,
            )
        )
        embedder = EmbeddingModel(settings)
        store = VectorStore(settings)
        manifest_path = Path(settings.vector_db_path) / "manifest.json"

        return cls(
            chunker=chunker,
            embedder=embedder,
            store=store,
            manifest_path=manifest_path,
        )

    def initialize(self) -> None:
        """Load embedding model and connect to vector store.

        Call this before any ingest operations when using from_settings().
        """
        if not self._embedder.is_loaded:
            self._embedder.load()
        if not self._store.is_connected:
            # Update store dimension to match embedder
            self._store._dimension = self._embedder.dimension
            self._store.connect()

    def shutdown(self) -> None:
        """Release resources."""
        self._store.close()
        self._embedder.unload()

    @property
    def manifest(self) -> IngestionManifest:
        """Access the ingestion manifest."""
        return self._manifest

    def ingest_file(
        self,
        file_path: Path,
        *,
        force: bool = False,
        source_name: str | None = None,
    ) -> IngestionResult:
        """Ingest a single file into the vector store.

        Args:
            file_path: Path to the file to ingest.
            force: If True, re-ingest even if file hasn't changed.
            source_name: Override the source file name in metadata.
                         Defaults to the file name.

        Returns:
            IngestionResult with details about what was done.

        Raises:
            AraMemoryError: If any pipeline step fails.
        """
        start = time.monotonic()
        source = source_name or file_path.name

        # Validate file exists
        if not file_path.exists():
            raise AraMemoryError(
                f"File not found: {file_path}",
                operation="ingest_file",
                details={"path": str(file_path)},
            )

        if not file_path.is_file():
            raise AraMemoryError(
                f"Path is not a file: {file_path}",
                operation="ingest_file",
                details={"path": str(file_path)},
            )

        # Compute file hash for change detection
        file_hash = self._hash_file(file_path)

        # Skip if unchanged (unless forced)
        if not force and self._manifest.has_file(source):
            stored_hash = self._manifest.get_hash(source)
            if stored_hash == file_hash:
                elapsed_ms = (time.monotonic() - start) * 1000
                log.info(
                    "file_skipped_unchanged",
                    source=source,
                    hash=file_hash[:12],
                )
                return IngestionResult(
                    source_file=source,
                    chunks_created=0,
                    chunks_embedded=0,
                    chunks_stored=0,
                    elapsed_ms=round(elapsed_ms, 1),
                    skipped=True,
                    skip_reason="unchanged",
                    file_hash=file_hash,
                )

        # Read file content
        try:
            text = file_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise AraMemoryError(
                f"Failed to read file: {exc}",
                operation="ingest_file",
                details={"path": str(file_path)},
            ) from exc

        if not text.strip():
            elapsed_ms = (time.monotonic() - start) * 1000
            return IngestionResult(
                source_file=source,
                chunks_created=0,
                chunks_embedded=0,
                chunks_stored=0,
                elapsed_ms=round(elapsed_ms, 1),
                skipped=True,
                skip_reason="empty_file",
                file_hash=file_hash,
            )

        # Delete old chunks for this source (for re-ingestion)
        if self._manifest.has_file(source):
            self._store.delete_by_source(source)
            log.info("old_chunks_deleted", source=source)

        # Chunk
        chunks = self._chunk_file(text, source)

        # Embed
        embedded_chunks = self._embed_chunks(chunks)

        # Store
        stored_count = self._store.upsert(embedded_chunks)

        # Update manifest
        self._manifest.set_hash(source, file_hash)
        self._manifest.save()

        elapsed_ms = (time.monotonic() - start) * 1000
        log.info(
            "file_ingested",
            source=source,
            chunks_created=len(chunks),
            chunks_embedded=len(embedded_chunks),
            chunks_stored=stored_count,
            elapsed_ms=round(elapsed_ms, 1),
        )

        return IngestionResult(
            source_file=source,
            chunks_created=len(chunks),
            chunks_embedded=len(embedded_chunks),
            chunks_stored=stored_count,
            elapsed_ms=round(elapsed_ms, 1),
            file_hash=file_hash,
        )

    def ingest_directory(
        self,
        dir_path: Path,
        *,
        pattern: str = "*.md",
        force: bool = False,
        recursive: bool = True,
    ) -> BatchIngestionResult:
        """Ingest all matching files in a directory.

        Args:
            dir_path: Path to the directory.
            pattern: Glob pattern for files to ingest.
            force: If True, re-ingest all files regardless of changes.
            recursive: If True, search subdirectories.

        Returns:
            BatchIngestionResult with per-file details.

        Raises:
            AraMemoryError: If the directory doesn't exist.
        """
        if not dir_path.exists() or not dir_path.is_dir():
            raise AraMemoryError(
                f"Directory not found: {dir_path}",
                operation="ingest_directory",
                details={"path": str(dir_path)},
            )

        start = time.monotonic()
        glob_method = dir_path.rglob if recursive else dir_path.glob
        files = sorted(glob_method(pattern))

        results: list[IngestionResult] = []
        errors: list[str] = []
        files_skipped = 0

        for file_path in files:
            if not file_path.is_file():
                continue
            try:
                result = self.ingest_file(file_path, force=force)
                results.append(result)
                if result.skipped:
                    files_skipped += 1
            except AraMemoryError as exc:
                error_msg = f"{file_path.name}: {exc}"
                errors.append(error_msg)
                log.error(
                    "file_ingestion_failed",
                    file=str(file_path),
                    error=str(exc),
                )

        total_chunks = sum(r.chunks_stored for r in results)
        total_elapsed_ms = (time.monotonic() - start) * 1000

        log.info(
            "directory_ingested",
            dir=str(dir_path),
            files_processed=len(results),
            files_skipped=files_skipped,
            total_chunks=total_chunks,
            errors=len(errors),
            elapsed_ms=round(total_elapsed_ms, 1),
        )

        return BatchIngestionResult(
            files_processed=len(results),
            files_skipped=files_skipped,
            total_chunks=total_chunks,
            total_elapsed_ms=round(total_elapsed_ms, 1),
            results=results,
            errors=errors,
        )

    def ingest_text(
        self,
        text: str,
        *,
        source_name: str,
        force: bool = False,
    ) -> IngestionResult:
        """Ingest raw text content (e.g., conversation transcript).

        Args:
            text: The text content to ingest.
            source_name: Identifier for this content in the store.
            force: If True, re-ingest even if content hasn't changed.

        Returns:
            IngestionResult with details.
        """
        start = time.monotonic()

        if not text.strip():
            elapsed_ms = (time.monotonic() - start) * 1000
            return IngestionResult(
                source_file=source_name,
                chunks_created=0,
                chunks_embedded=0,
                chunks_stored=0,
                elapsed_ms=round(elapsed_ms, 1),
                skipped=True,
                skip_reason="empty_text",
            )

        # Hash the text content for change detection
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

        if not force and self._manifest.has_file(source_name):
            stored_hash = self._manifest.get_hash(source_name)
            if stored_hash == content_hash:
                elapsed_ms = (time.monotonic() - start) * 1000
                return IngestionResult(
                    source_file=source_name,
                    chunks_created=0,
                    chunks_embedded=0,
                    chunks_stored=0,
                    elapsed_ms=round(elapsed_ms, 1),
                    skipped=True,
                    skip_reason="unchanged",
                    file_hash=content_hash,
                )

        # Delete old chunks for this source
        if self._manifest.has_file(source_name):
            self._store.delete_by_source(source_name)

        # Chunk (plain text mode, no markdown parsing)
        chunks = self._chunker.chunk_text(
            text,
            source_file=source_name,
        )

        # Embed
        embedded_chunks = self._embed_chunks(chunks)

        # Store
        stored_count = self._store.upsert(embedded_chunks)

        # Update manifest
        self._manifest.set_hash(source_name, content_hash)
        self._manifest.save()

        elapsed_ms = (time.monotonic() - start) * 1000
        log.info(
            "text_ingested",
            source=source_name,
            chunks_stored=stored_count,
            elapsed_ms=round(elapsed_ms, 1),
        )

        return IngestionResult(
            source_file=source_name,
            chunks_created=len(chunks),
            chunks_embedded=len(embedded_chunks),
            chunks_stored=stored_count,
            elapsed_ms=round(elapsed_ms, 1),
            file_hash=content_hash,
        )

    def _chunk_file(self, text: str, source_file: str) -> list[MemoryChunk]:
        """Chunk a file's text content based on extension/format."""
        # Use markdown-aware chunking (works for plain text too)
        return self._chunker.chunk_document(text, source_file=source_file)

    def _embed_chunks(self, chunks: list[MemoryChunk]) -> list[MemoryChunk]:
        """Embed all chunks and return new MemoryChunk objects with embeddings."""
        if not chunks:
            return []

        # Build texts for embedding — use contextual header + text for better retrieval
        texts: list[str] = []
        for chunk in chunks:
            header = chunk.metadata.get("contextual_header", "")
            if header:
                texts.append(f"{header}\n\n{chunk.text}")
            else:
                texts.append(chunk.text)

        # Embed in batches
        vectors = self._embedder.embed_documents(
            texts,
            batch_size=self._embedding_batch_size,
        )

        # Create new chunks with embeddings
        embedded: list[MemoryChunk] = []
        for chunk, vector in zip(chunks, vectors, strict=True):
            embedded.append(
                MemoryChunk(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    source=chunk.source,
                    source_file=chunk.source_file,
                    date=chunk.date,
                    section_title=chunk.section_title,
                    embedding=vector,
                    metadata=chunk.metadata,
                )
            )

        return embedded

    @staticmethod
    def _hash_file(file_path: Path) -> str:
        """Compute SHA-256 hash of a file's contents."""
        hasher = hashlib.sha256()
        with file_path.open("rb") as f:
            while True:
                data = f.read(8192)
                if not data:
                    break
                hasher.update(data)
        return hasher.hexdigest()
