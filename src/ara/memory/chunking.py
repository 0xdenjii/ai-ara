"""Markdown-aware hierarchical text chunker for diary ingestion.

Splits diary entries into semantically meaningful chunks that preserve
document structure. The hierarchy is:

    Document → Date sections → Header sections → Paragraphs → Sentences

Each chunk carries metadata about its position in the hierarchy:
source file, date, section title, chunk index.

Design decisions:
- Target ~512 tokens per chunk with 64-token overlap between consecutive chunks
- Uses tiktoken (cl100k_base) for accurate token counting
- Splits at paragraph boundaries first, then sentence boundaries for overflow
- Never splits mid-sentence unless a single sentence exceeds the max chunk size
- Short entries (< target size) are kept as single chunks

All configuration comes from AraSettings.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import tiktoken

from ara.utils.logging import get_logger
from ara.utils.types import MemoryChunk, MemorySource

if TYPE_CHECKING:
    from datetime import datetime

log = get_logger("chunking")

# Default tiktoken encoding — cl100k_base covers GPT-4, Qwen, and most modern models
_ENCODING = tiktoken.get_encoding("cl100k_base")

# Regex patterns for diary structure
_DATE_HEADER_PATTERN = re.compile(
    r"^#{1,2}\s+"  # h1 or h2
    r"(\d{4}[-/]\d{1,2}[-/]\d{1,2})"  # date in YYYY-MM-DD or YYYY/MM/DD
    r"(?:\s*[-\u2013\u2014]\s*(.+))?$",  # optional title after dash/en-dash/em-dash
    re.MULTILINE,
)

_SECTION_HEADER_PATTERN = re.compile(
    r"^(#{2,4})\s+(.+)$",
    re.MULTILINE,
)

_SENTENCE_BOUNDARY = re.compile(
    r"(?<=[.!?…])\s+(?=[A-Z\"\'])"  # sentence end followed by capital letter or quote
)


@dataclass(frozen=True)
class ChunkMetadata:
    """Metadata extracted from document structure for a single chunk."""

    source_file: str
    date: datetime | None = None
    section_title: str = ""
    chunk_index: int = 0
    total_chunks: int = 0
    is_overlap: bool = False
    contextual_header: str = ""


@dataclass
class ChunkerConfig:
    """Configuration for the text chunker."""

    target_tokens: int = 512
    overlap_tokens: int = 64
    min_chunk_tokens: int = 50
    max_chunk_tokens: int = 1024
    encoding_name: str = "cl100k_base"


class TextChunker:
    """Markdown-aware hierarchical text chunker.

    Processes markdown documents (diary entries) into semantically meaningful
    chunks with metadata about document structure.

    Usage:
        chunker = TextChunker()
        chunks = chunker.chunk_document(text, source_file="diary_2024.md")
    """

    def __init__(self, config: ChunkerConfig | None = None) -> None:
        self._config = config or ChunkerConfig()
        self._encoding = tiktoken.get_encoding(self._config.encoding_name)

    def count_tokens(self, text: str) -> int:
        """Count tokens in a text string."""
        return len(self._encoding.encode(text))

    def chunk_document(
        self,
        text: str,
        *,
        source_file: str = "",
        default_date: datetime | None = None,
        source: MemorySource = MemorySource.DIARY,
    ) -> list[MemoryChunk]:
        """Split a markdown document into chunks with metadata.

        Args:
            text: The full markdown document text.
            source_file: Name/path of the source file.
            default_date: Fallback date if none found in headers.
            source: The MemorySource type for all chunks (default: DIARY).

        Returns:
            List of MemoryChunk objects with populated metadata.
        """
        if not text.strip():
            return []

        # Parse document into sections
        sections = self._parse_sections(text, default_date=default_date)

        all_chunks: list[MemoryChunk] = []
        chunk_index = 0

        for section in sections:
            section_chunks = self._chunk_section(
                section["text"],
                date=section.get("date"),
                section_title=section.get("title", ""),
                source_file=source_file,
                start_index=chunk_index,
                source=source,
            )
            all_chunks.extend(section_chunks)
            chunk_index += len(section_chunks)

        # Update total_chunks in metadata
        total = len(all_chunks)
        final_chunks: list[MemoryChunk] = []
        for chunk in all_chunks:
            updated_meta = {**chunk.metadata, "total_chunks": total}
            final_chunks.append(
                MemoryChunk(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    source=chunk.source,
                    source_file=chunk.source_file,
                    date=chunk.date,
                    section_title=chunk.section_title,
                    metadata=updated_meta,
                )
            )

        log.info(
            "document_chunked",
            source_file=source_file,
            total_chunks=total,
            input_tokens=self.count_tokens(text),
        )

        return final_chunks

    def chunk_text(
        self,
        text: str,
        *,
        source_file: str = "",
        date: datetime | None = None,
        section_title: str = "",
        source: MemorySource = MemorySource.DIARY,
    ) -> list[MemoryChunk]:
        """Split plain text into chunks (no markdown parsing).

        Use this for non-markdown content like conversation transcripts.

        Args:
            text: Plain text to chunk.
            source_file: Source identifier.
            date: Associated date.
            section_title: Section context.
            source: The MemorySource type for all chunks (default: DIARY).

        Returns:
            List of MemoryChunk objects.
        """
        if not text.strip():
            return []

        return self._chunk_section(
            text,
            date=date,
            section_title=section_title,
            source_file=source_file,
            start_index=0,
            source=source,
        )

    def _parse_sections(
        self,
        text: str,
        *,
        default_date: datetime | None = None,
    ) -> list[dict[str, object]]:
        """Parse markdown into date/section blocks.

        Returns a list of dicts with keys: text, date, title.
        """
        from datetime import datetime as dt

        sections: list[dict[str, object]] = []
        current_date: datetime | None = default_date
        current_title = ""
        current_lines: list[str] = []

        for line in text.split("\n"):
            # Check for date headers
            date_match = _DATE_HEADER_PATTERN.match(line)
            if date_match:
                # Flush previous section
                if current_lines:
                    sections.append(
                        {
                            "text": "\n".join(current_lines).strip(),
                            "date": current_date,
                            "title": current_title,
                        }
                    )
                    current_lines = []

                date_str = date_match.group(1).replace("/", "-")
                try:
                    current_date = dt.strptime(date_str, "%Y-%m-%d")
                except ValueError:
                    current_date = default_date

                current_title = (date_match.group(2) or "").strip()
                continue

            # Check for section headers
            section_match = _SECTION_HEADER_PATTERN.match(line)
            if section_match:
                # Flush previous section
                if current_lines:
                    sections.append(
                        {
                            "text": "\n".join(current_lines).strip(),
                            "date": current_date,
                            "title": current_title,
                        }
                    )
                    current_lines = []

                current_title = section_match.group(2).strip()
                continue

            current_lines.append(line)

        # Flush final section
        if current_lines:
            section_text = "\n".join(current_lines).strip()
            if section_text:
                sections.append(
                    {
                        "text": section_text,
                        "date": current_date,
                        "title": current_title,
                    }
                )

        # If no sections found, treat entire text as one section
        if not sections:
            sections.append(
                {
                    "text": text.strip(),
                    "date": default_date,
                    "title": "",
                }
            )

        return sections

    def _chunk_section(
        self,
        text: str,
        *,
        date: datetime | None = None,
        section_title: str = "",
        source_file: str = "",
        start_index: int = 0,
        source: MemorySource = MemorySource.DIARY,
    ) -> list[MemoryChunk]:
        """Split a section of text into token-bounded chunks.

        Strategy:
        1. Split into paragraphs
        2. Greedily accumulate paragraphs until target_tokens is reached
        3. If a single paragraph exceeds max_chunk_tokens, split at sentences
        4. Add overlap_tokens of context from the previous chunk
        """
        if not text.strip():
            return []

        # Check if entire section fits in one chunk
        total_tokens = self.count_tokens(text)
        if total_tokens <= self._config.target_tokens:
            chunk = self._make_chunk(
                text=text,
                date=date,
                section_title=section_title,
                source_file=source_file,
                chunk_index=start_index,
                source=source,
            )
            return [chunk]

        # Split into paragraphs
        paragraphs = self._split_paragraphs(text)

        chunks: list[MemoryChunk] = []
        current_parts: list[str] = []
        current_tokens = 0
        overlap_text = ""

        for para in paragraphs:
            para_tokens = self.count_tokens(para)

            # If single paragraph exceeds max, split at sentences
            if para_tokens > self._config.max_chunk_tokens:
                # Flush current accumulation first
                if current_parts:
                    chunk_text = "\n\n".join(current_parts)
                    if overlap_text:
                        chunk_text = overlap_text + "\n\n" + chunk_text
                    chunk = self._make_chunk(
                        text=chunk_text.strip(),
                        date=date,
                        section_title=section_title,
                        source_file=source_file,
                        chunk_index=start_index + len(chunks),
                        source=source,
                    )
                    chunks.append(chunk)
                    overlap_text = self._get_overlap_tail("\n\n".join(current_parts))
                    current_parts = []
                    current_tokens = 0

                # Split the long paragraph at sentence boundaries
                sentence_chunks = self._split_long_paragraph(para)
                for sc in sentence_chunks:
                    sc_with_overlap = overlap_text + "\n\n" + sc if overlap_text else sc
                    chunk = self._make_chunk(
                        text=sc_with_overlap.strip(),
                        date=date,
                        section_title=section_title,
                        source_file=source_file,
                        chunk_index=start_index + len(chunks),
                        source=source,
                    )
                    chunks.append(chunk)
                    overlap_text = self._get_overlap_tail(sc)
                continue

            # Would adding this paragraph exceed target?
            would_be = current_tokens + para_tokens
            if would_be > self._config.target_tokens and current_parts:
                # Flush current accumulation
                chunk_text = "\n\n".join(current_parts)
                if overlap_text:
                    chunk_text = overlap_text + "\n\n" + chunk_text
                chunk = self._make_chunk(
                    text=chunk_text.strip(),
                    date=date,
                    section_title=section_title,
                    source_file=source_file,
                    chunk_index=start_index + len(chunks),
                    source=source,
                )
                chunks.append(chunk)
                overlap_text = self._get_overlap_tail("\n\n".join(current_parts))
                current_parts = []
                current_tokens = 0

            current_parts.append(para)
            current_tokens += para_tokens

        # Flush remaining
        if current_parts:
            chunk_text = "\n\n".join(current_parts)
            if overlap_text:
                chunk_text = overlap_text + "\n\n" + chunk_text

            # Skip if below minimum size (merge with previous)
            remaining_tokens = self.count_tokens(chunk_text)
            if remaining_tokens < self._config.min_chunk_tokens and chunks:
                # Merge with previous chunk
                prev = chunks.pop()
                merged_text = prev.text + "\n\n" + "\n\n".join(current_parts)
                chunk = self._make_chunk(
                    text=merged_text.strip(),
                    date=date,
                    section_title=section_title,
                    source_file=source_file,
                    chunk_index=prev.metadata.get("chunk_index", start_index + len(chunks)),
                    source=source,
                )
                chunks.append(chunk)
            else:
                chunk = self._make_chunk(
                    text=chunk_text.strip(),
                    date=date,
                    section_title=section_title,
                    source_file=source_file,
                    chunk_index=start_index + len(chunks),
                    source=source,
                )
                chunks.append(chunk)

        return chunks

    def _split_paragraphs(self, text: str) -> list[str]:
        """Split text into paragraphs (double newline separated)."""
        paragraphs = re.split(r"\n\s*\n", text.strip())
        return [p.strip() for p in paragraphs if p.strip()]

    def _split_long_paragraph(self, paragraph: str) -> list[str]:
        """Split a long paragraph at sentence boundaries into target-sized chunks."""
        sentences = _SENTENCE_BOUNDARY.split(paragraph)
        if len(sentences) <= 1:
            # Can't split further — return as-is even if oversized
            return [paragraph]

        chunks: list[str] = []
        current_parts: list[str] = []
        current_tokens = 0

        for sentence in sentences:
            sent_tokens = self.count_tokens(sentence)
            if current_tokens + sent_tokens > self._config.target_tokens and current_parts:
                chunks.append(" ".join(current_parts))
                current_parts = []
                current_tokens = 0
            current_parts.append(sentence)
            current_tokens += sent_tokens

        if current_parts:
            chunks.append(" ".join(current_parts))

        return chunks

    def _get_overlap_tail(self, text: str) -> str:
        """Get the last N tokens of text as overlap for the next chunk."""
        tokens = self._encoding.encode(text)
        if len(tokens) <= self._config.overlap_tokens:
            return text
        overlap_tokens = tokens[-self._config.overlap_tokens :]
        return self._encoding.decode(overlap_tokens)

    def _make_chunk(
        self,
        *,
        text: str,
        date: datetime | None,
        section_title: str,
        source_file: str,
        chunk_index: int,
        source: MemorySource = MemorySource.DIARY,
    ) -> MemoryChunk:
        """Create a MemoryChunk with a deterministic ID."""
        from hashlib import sha256

        # Deterministic chunk ID based on content + position
        id_input = f"{source_file}:{chunk_index}:{text[:100]}"
        chunk_id = sha256(id_input.encode()).hexdigest()[:16]

        # Build contextual header (prepended during embedding for better retrieval)
        header_parts: list[str] = []
        if source_file:
            header_parts.append(f"Source: {source_file}")
        if date is not None:
            header_parts.append(f"Date: {date.strftime('%Y-%m-%d')}")
        if section_title:
            header_parts.append(f"Section: {section_title}")
        contextual_header = " | ".join(header_parts)

        return MemoryChunk(
            chunk_id=chunk_id,
            text=text,
            source=source,
            source_file=source_file,
            date=date,
            section_title=section_title,
            metadata={
                "chunk_index": chunk_index,
                "total_chunks": 0,  # filled in by chunk_document
                "token_count": self.count_tokens(text),
                "contextual_header": contextual_header,
            },
        )
