"""LanceDB vector store for ARA's memory retrieval.

Provides CRUD operations for storing and searching embedded memory chunks.
LanceDB runs embedded (in-process), with no network overhead. Supports
native hybrid search: vector similarity + BM25 full-text + SQL predicates
in a single query.

All data is stored in the local filesystem under data/vector_db/.
All configuration comes from AraSettings.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import lancedb
import pyarrow as pa

from ara.utils.errors import AraMemoryError
from ara.utils.logging import get_logger
from ara.utils.types import MemoryChunk, MemorySource

if TYPE_CHECKING:
    from datetime import datetime

    from ara.config.settings import MemorySettings

log = get_logger("vector_store")

# LanceDB table name for memory chunks
_TABLE_NAME = "memory_chunks"

# PyArrow schema for the memory chunks table
_SCHEMA = pa.schema(
    [
        pa.field("chunk_id", pa.string()),
        pa.field("text", pa.string()),
        pa.field("source", pa.string()),
        pa.field("source_file", pa.string()),
        pa.field("date", pa.string()),  # ISO format string (nullable)
        pa.field("section_title", pa.string()),
        pa.field("contextual_header", pa.string()),
        pa.field("token_count", pa.int32()),
        pa.field("chunk_index", pa.int32()),
        pa.field("vector", pa.list_(pa.float32())),  # dimension set at runtime
    ]
)


class VectorStore:
    """LanceDB-backed vector store for memory chunks.

    Stores embedded text chunks with metadata and supports:
    - Vector similarity search (cosine)
    - Full-text BM25 search
    - Metadata filtering (date ranges, source types)
    - Hybrid search combining all three

    Usage:
        store = VectorStore(settings, dimension=768)
        store.connect()
        store.upsert(chunks_with_embeddings)
        results = store.search(query_vector, top_k=10)
        store.close()
    """

    def __init__(self, settings: MemorySettings, *, dimension: int = 768) -> None:
        self._settings = settings
        self._dimension = dimension
        self._db: lancedb.DBConnection | None = None
        self._table: lancedb.table.Table | None = None
        self._connected = False
        self._fts_index_created: bool = False

    @property
    def is_connected(self) -> bool:
        """Whether the store is connected and ready."""
        return self._connected

    @property
    def dimension(self) -> int:
        """Expected embedding vector dimension."""
        return self._dimension

    def connect(self) -> None:
        """Connect to the LanceDB database.

        Creates the database directory and table if they don't exist.

        Raises:
            AraMemoryError: If connection fails.
        """
        if self._connected:
            return

        db_path = Path(self._settings.vector_db_path)
        db_path.mkdir(parents=True, exist_ok=True)

        try:
            self._db = lancedb.connect(str(db_path))

            # Open existing table or create new one
            table_names = self._db.table_names()
            if _TABLE_NAME in table_names:
                self._table = self._db.open_table(_TABLE_NAME)
                log.info(
                    "vector_store_connected",
                    path=str(db_path),
                    table=_TABLE_NAME,
                    rows=self._table.count_rows(),
                )
            else:
                self._table = None  # Will be created on first upsert
                log.info(
                    "vector_store_connected",
                    path=str(db_path),
                    table=_TABLE_NAME,
                    rows=0,
                    status="empty",
                )

            self._connected = True
        except Exception as exc:
            raise AraMemoryError(
                f"Failed to connect to vector store: {exc}",
                operation="vector_store_connect",
                details={"path": str(db_path)},
            ) from exc

    def close(self) -> None:
        """Close the database connection."""
        self._table = None
        self._db = None
        self._connected = False
        log.info("vector_store_closed")

    def upsert(self, chunks: list[MemoryChunk]) -> int:
        """Insert or update memory chunks with their embeddings.

        Chunks are identified by chunk_id. If a chunk with the same ID
        exists, it is replaced.

        Args:
            chunks: List of MemoryChunk objects with populated embeddings.

        Returns:
            Number of chunks upserted.

        Raises:
            AraMemoryError: If the store is not connected or upsert fails.
        """
        if not self._connected or self._db is None:
            raise AraMemoryError(
                "Vector store not connected — call connect() first",
                operation="vector_store_upsert",
            )

        if not chunks:
            return 0

        start = time.monotonic()

        # Convert chunks to records
        records = []
        for chunk in chunks:
            if not chunk.embedding:
                log.warning(
                    "chunk_missing_embedding",
                    chunk_id=chunk.chunk_id,
                )
                continue

            records.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "source": chunk.source.value
                    if isinstance(chunk.source, MemorySource)
                    else chunk.source,
                    "source_file": chunk.source_file,
                    "date": chunk.date.isoformat() if chunk.date else "",
                    "section_title": chunk.section_title,
                    "contextual_header": chunk.metadata.get("contextual_header", ""),
                    "token_count": chunk.metadata.get("token_count", 0),
                    "chunk_index": chunk.metadata.get("chunk_index", 0),
                    "vector": chunk.embedding,
                }
            )

        if not records:
            return 0

        try:
            if self._table is None:
                # Create table with first batch
                self._table = self._db.create_table(_TABLE_NAME, data=records)
            else:
                # Delete existing chunks with same IDs, then add new ones
                chunk_ids = [r["chunk_id"] for r in records]
                id_list = ", ".join(
                    f"'{cid.replace(chr(39), chr(39) + chr(39))}'" for cid in chunk_ids
                )
                try:
                    self._table.delete(f"chunk_id IN ({id_list})")
                except Exception as exc:
                    log.debug("chunk_delete_before_upsert_failed", error=str(exc))
                self._table.add(records)

            self._fts_index_created = False  # Invalidate FTS index after data changes

            elapsed_ms = (time.monotonic() - start) * 1000
            log.info(
                "chunks_upserted",
                count=len(records),
                elapsed_ms=round(elapsed_ms, 1),
            )
            return len(records)

        except Exception as exc:
            raise AraMemoryError(
                f"Failed to upsert chunks: {exc}",
                operation="vector_store_upsert",
                details={"chunk_count": len(records)},
            ) from exc

    def search(
        self,
        query_vector: list[float],
        *,
        top_k: int = 10,
        source_filter: MemorySource | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Search for similar chunks using vector similarity.

        Args:
            query_vector: The query embedding vector.
            top_k: Number of results to return.
            source_filter: Optional filter by source type.
            date_from: Optional start date filter.
            date_to: Optional end date filter.

        Returns:
            List of result dicts with chunk data and distance score.

        Raises:
            AraMemoryError: If search fails.
        """
        if not self._connected or self._table is None:
            raise AraMemoryError(
                "Vector store not connected or empty",
                operation="vector_store_search",
            )

        start = time.monotonic()

        try:
            query = self._table.search(query_vector).limit(top_k)

            # Apply filters
            filters: list[str] = []
            if source_filter is not None:
                safe_source = source_filter.value.replace("'", "''")
                filters.append(f"source = '{safe_source}'")
            if date_from is not None:
                safe_date_from = date_from.isoformat().replace("'", "''")
                filters.append(f"date >= '{safe_date_from}'")
            if date_to is not None:
                safe_date_to = date_to.isoformat().replace("'", "''")
                filters.append(f"date <= '{safe_date_to}'")

            if filters:
                where_clause = " AND ".join(filters)
                query = query.where(where_clause)

            results = query.to_list()

            elapsed_ms = (time.monotonic() - start) * 1000
            log.debug(
                "vector_search_complete",
                top_k=top_k,
                results=len(results),
                elapsed_ms=round(elapsed_ms, 1),
            )
            return results

        except Exception as exc:
            raise AraMemoryError(
                f"Vector search failed: {exc}",
                operation="vector_store_search",
            ) from exc

    def full_text_search(
        self,
        query_text: str,
        *,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Search using BM25 full-text search on chunk text.

        Args:
            query_text: The search query string.
            top_k: Number of results to return.

        Returns:
            List of result dicts with chunk data and BM25 score.

        Raises:
            AraMemoryError: If search fails.
        """
        if not self._connected or self._table is None:
            raise AraMemoryError(
                "Vector store not connected or empty",
                operation="vector_store_fts",
            )

        start = time.monotonic()

        try:
            # Create FTS index if not already created
            if not self._fts_index_created:
                try:
                    self._table.create_fts_index("text", replace=True)
                    self._fts_index_created = True
                except Exception as exc:
                    log.debug("fts_index_creation_failed", error=str(exc))

            results = self._table.search(query_text, query_type="fts").limit(top_k).to_list()

            elapsed_ms = (time.monotonic() - start) * 1000
            log.debug(
                "fts_search_complete",
                query_len=len(query_text),
                results=len(results),
                elapsed_ms=round(elapsed_ms, 1),
            )
            return results

        except Exception as exc:
            raise AraMemoryError(
                f"Full-text search failed: {exc}",
                operation="vector_store_fts",
            ) from exc

    def delete_by_source(self, source_file: str) -> int:
        """Delete all chunks from a specific source file.

        Useful for re-ingesting an updated diary file.

        Args:
            source_file: The source file identifier.

        Returns:
            Approximate number of rows before deletion (exact count not available).
        """
        if not self._connected or self._table is None:
            return 0

        count_before = self._table.count_rows()
        try:
            safe_source_file = source_file.replace("'", "''")
            self._table.delete(f"source_file = '{safe_source_file}'")
            count_after = self._table.count_rows()
            deleted = count_before - count_after
            log.info(
                "chunks_deleted_by_source",
                source_file=source_file,
                deleted=deleted,
            )
            return deleted
        except Exception as exc:
            raise AraMemoryError(
                f"Failed to delete chunks: {exc}",
                operation="vector_store_delete",
                details={"source_file": source_file},
            ) from exc

    def count(self) -> int:
        """Return the total number of chunks in the store."""
        if not self._connected or self._table is None:
            return 0
        return self._table.count_rows()

    def clear(self) -> None:
        """Delete all data from the store.

        Drops and recreates the table.
        """
        if not self._connected or self._db is None:
            return

        try:
            if _TABLE_NAME in self._db.table_names():
                self._db.drop_table(_TABLE_NAME)
            self._table = None
            log.info("vector_store_cleared")
        except Exception as exc:
            raise AraMemoryError(
                f"Failed to clear vector store: {exc}",
                operation="vector_store_clear",
            ) from exc
