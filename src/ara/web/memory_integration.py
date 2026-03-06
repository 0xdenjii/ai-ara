"""Web content to memory integration — store research findings in LanceDB.

Optionally persists web research results (searches, page reads, deep research)
into the vector store so ARA can recall them in future conversations.

Architecture:
    - Takes ResearchResult or raw web content and converts to MemoryChunks
    - Uses the existing chunking + embedding pipeline
    - Stored with source=WEB_RESEARCH for filtering
    - Deduplication by URL to avoid storing the same page content twice

Usage:
    integrator = WebMemoryIntegrator(settings, vector_store, embedder)
    chunks_stored = await integrator.store_research(research_result)
"""

from __future__ import annotations

import hashlib
import time
from datetime import datetime
from typing import TYPE_CHECKING

from ara.utils.logging import get_logger
from ara.utils.types import MemoryChunk, MemorySource

if TYPE_CHECKING:
    from ara.memory.embeddings import EmbeddingModel
    from ara.memory.vector_store import VectorStore
    from ara.web.research import ResearchResult

log = get_logger("web_memory")


class WebMemoryIntegrator:
    """Stores web research results in the vector store for future recall.

    Args:
        vector_store: LanceDB vector store instance.
        embedder: Embedding model for text vectorization.
        max_chunk_chars: Maximum characters per chunk (longer content is split).
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedder: EmbeddingModel,
        *,
        max_chunk_chars: int = 2000,
    ) -> None:
        self._vector_store = vector_store
        self._embedder = embedder
        self._max_chunk_chars = max_chunk_chars
        self._stored_urls: set[str] = set()

    async def store_research(self, result: ResearchResult) -> int:
        """Store a research result in the vector store.

        Creates memory chunks from the research answer and used sources.
        Skips if the research query has already been stored.

        Args:
            result: The completed research result.

        Returns:
            Number of chunks stored.
        """
        if not result.answer or not result.answer.strip():
            return 0

        chunks: list[MemoryChunk] = []
        now = datetime.now()

        # Create a chunk for the synthesized answer
        answer_chunk_id = _content_hash(f"research:{result.query}")
        chunks.append(
            MemoryChunk(
                chunk_id=answer_chunk_id,
                text=f"Research: {result.query}\n\n{result.answer}",
                source=MemorySource.WEB_RESEARCH,
                source_file=f"research:{result.query[:80]}",
                date=now,
                section_title=f"Web Research: {result.query[:100]}",
                metadata={
                    "type": "research_answer",
                    "query": result.query,
                    "sources_count": len(result.sources),
                    "pages_read": result.pages_read,
                    "sub_queries": result.sub_queries,
                },
            )
        )

        # Create chunks for sources that were used in synthesis
        for source in result.sources:
            if not source.used_in_synthesis or not source.snippet:
                continue

            if source.url in self._stored_urls:
                continue

            source_chunk_id = _content_hash(f"source:{source.url}")
            chunks.append(
                MemoryChunk(
                    chunk_id=source_chunk_id,
                    text=f"{source.title}\n\nSource: {source.url}\n\n{source.snippet}",
                    source=MemorySource.WEB_RESEARCH,
                    source_file=source.url,
                    date=now,
                    section_title=source.title,
                    metadata={
                        "type": "web_source",
                        "url": source.url,
                        "relevance": source.relevance,
                        "research_query": result.query,
                    },
                )
            )
            self._stored_urls.add(source.url)

        if not chunks:
            return 0

        # Embed and store
        return await self._embed_and_store(chunks)

    async def store_page_content(
        self,
        url: str,
        title: str,
        content: str,
        *,
        context: str = "",
    ) -> int:
        """Store extracted page content in the vector store.

        Args:
            url: The page URL.
            title: Page title.
            content: Extracted page text.
            context: Optional context about why this page was read.

        Returns:
            Number of chunks stored.
        """
        if url in self._stored_urls:
            log.debug("page_already_stored", url=url[:100])
            return 0

        if not content.strip():
            return 0

        chunks: list[MemoryChunk] = []
        now = datetime.now()

        # Split long content into multiple chunks
        text_parts = _split_text(content, self._max_chunk_chars)

        for i, part in enumerate(text_parts):
            chunk_id = _content_hash(f"page:{url}:{i}")
            prefix = f"{title}\nSource: {url}"
            if context:
                prefix += f"\nContext: {context}"

            chunks.append(
                MemoryChunk(
                    chunk_id=chunk_id,
                    text=f"{prefix}\n\n{part}",
                    source=MemorySource.WEB_RESEARCH,
                    source_file=url,
                    date=now,
                    section_title=title,
                    metadata={
                        "type": "web_page",
                        "url": url,
                        "chunk_index": i,
                        "total_chunks": len(text_parts),
                    },
                )
            )

        self._stored_urls.add(url)
        return await self._embed_and_store(chunks)

    async def _embed_and_store(self, chunks: list[MemoryChunk]) -> int:
        """Embed chunks and upsert into the vector store.

        Args:
            chunks: List of memory chunks to embed and store.

        Returns:
            Number of chunks stored.
        """
        start = time.perf_counter()

        # Embed all chunks
        texts = [c.text for c in chunks]
        embeddings = self._embedder.embed_documents(texts)

        # Create chunks with embeddings
        embedded_chunks = [
            MemoryChunk(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                source=chunk.source,
                source_file=chunk.source_file,
                date=chunk.date,
                section_title=chunk.section_title,
                embedding=list(emb),
                metadata=chunk.metadata,
            )
            for chunk, emb in zip(chunks, embeddings, strict=True)
        ]

        # Store in vector DB
        self._vector_store.upsert(embedded_chunks)

        elapsed_ms = (time.perf_counter() - start) * 1000
        log.info(
            "web_content_stored",
            chunks=len(embedded_chunks),
            elapsed_ms=round(elapsed_ms, 1),
        )

        return len(embedded_chunks)

    def get_stored_url_count(self) -> int:
        """Get the number of unique URLs stored in this session.

        Returns:
            Count of stored URLs.
        """
        return len(self._stored_urls)

    def clear_url_cache(self) -> None:
        """Clear the stored URL deduplication cache."""
        self._stored_urls.clear()


def _content_hash(content: str) -> str:
    """Generate a deterministic chunk ID from content.

    Args:
        content: String to hash.

    Returns:
        Hex digest string suitable as a chunk ID.
    """
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def _split_text(text: str, max_chars: int) -> list[str]:
    """Split text into chunks of approximately max_chars.

    Splits on paragraph boundaries when possible.

    Args:
        text: Text to split.
        max_chars: Maximum characters per chunk.

    Returns:
        List of text chunks.
    """
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    paragraphs = text.split("\n\n")
    current: list[str] = []
    current_len = 0

    for para in paragraphs:
        if current_len + len(para) + 2 > max_chars and current:
            chunks.append("\n\n".join(current))
            current = []
            current_len = 0

        current.append(para)
        current_len += len(para) + 2

    if current:
        chunks.append("\n\n".join(current))

    return chunks
