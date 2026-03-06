"""Tests for the markdown-aware text chunker."""

from __future__ import annotations

from datetime import datetime

from ara.memory.chunking import ChunkerConfig, TextChunker
from ara.utils.types import MemorySource


class TestTextChunkerBasic:
    """Basic chunking behavior."""

    def test_empty_text_returns_empty(self) -> None:
        chunker = TextChunker()
        assert chunker.chunk_document("") == []
        assert chunker.chunk_document("   ") == []

    def test_short_text_single_chunk(self) -> None:
        chunker = TextChunker()
        text = "This is a short diary entry. Not much happened today."
        chunks = chunker.chunk_document(text, source_file="diary.md")

        assert len(chunks) == 1
        assert chunks[0].text == text
        assert chunks[0].source == MemorySource.DIARY
        assert chunks[0].source_file == "diary.md"

    def test_chunk_id_is_deterministic(self) -> None:
        chunker = TextChunker()
        text = "Same text produces same ID."
        chunks1 = chunker.chunk_document(text, source_file="test.md")
        chunks2 = chunker.chunk_document(text, source_file="test.md")
        assert chunks1[0].chunk_id == chunks2[0].chunk_id

    def test_different_text_different_id(self) -> None:
        chunker = TextChunker()
        chunks1 = chunker.chunk_document("Text A", source_file="test.md")
        chunks2 = chunker.chunk_document("Text B", source_file="test.md")
        assert chunks1[0].chunk_id != chunks2[0].chunk_id


class TestTextChunkerTokenCounting:
    """Token counting functionality."""

    def test_count_tokens(self) -> None:
        chunker = TextChunker()
        count = chunker.count_tokens("Hello world")
        assert count > 0
        assert isinstance(count, int)

    def test_empty_string_zero_tokens(self) -> None:
        chunker = TextChunker()
        assert chunker.count_tokens("") == 0


class TestTextChunkerSplitting:
    """Multi-chunk splitting behavior."""

    def test_long_text_splits_into_multiple_chunks(self) -> None:
        config = ChunkerConfig(target_tokens=50, overlap_tokens=10, min_chunk_tokens=10)
        chunker = TextChunker(config)

        # Generate text that exceeds target tokens
        paragraphs = [f"Paragraph {i}. " * 20 for i in range(10)]
        text = "\n\n".join(paragraphs)

        chunks = chunker.chunk_document(text, source_file="test.md")
        assert len(chunks) > 1

    def test_chunk_indices_are_sequential(self) -> None:
        config = ChunkerConfig(target_tokens=50, overlap_tokens=10, min_chunk_tokens=10)
        chunker = TextChunker(config)

        paragraphs = [f"Paragraph {i}. " * 20 for i in range(10)]
        text = "\n\n".join(paragraphs)

        chunks = chunker.chunk_document(text, source_file="test.md")
        indices = [c.metadata["chunk_index"] for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_total_chunks_in_metadata(self) -> None:
        config = ChunkerConfig(target_tokens=50, overlap_tokens=10, min_chunk_tokens=10)
        chunker = TextChunker(config)

        paragraphs = [f"Paragraph {i}. " * 20 for i in range(10)]
        text = "\n\n".join(paragraphs)

        chunks = chunker.chunk_document(text, source_file="test.md")
        for chunk in chunks:
            assert chunk.metadata["total_chunks"] == len(chunks)

    def test_all_chunks_have_token_count(self) -> None:
        config = ChunkerConfig(target_tokens=50, overlap_tokens=10, min_chunk_tokens=10)
        chunker = TextChunker(config)

        paragraphs = [f"Paragraph {i}. " * 20 for i in range(5)]
        text = "\n\n".join(paragraphs)

        chunks = chunker.chunk_document(text, source_file="test.md")
        for chunk in chunks:
            assert chunk.metadata["token_count"] > 0


class TestTextChunkerMarkdownParsing:
    """Markdown structure parsing."""

    def test_date_header_extraction(self) -> None:
        chunker = TextChunker()
        text = """# 2024-03-15

Today was a great day. I went to the park.

# 2024-03-16

Another beautiful day."""

        chunks = chunker.chunk_document(text, source_file="diary.md")
        dates = [c.date for c in chunks if c.date is not None]
        assert len(dates) >= 1
        assert any(d.year == 2024 and d.month == 3 and d.day == 15 for d in dates)

    def test_date_header_with_title(self) -> None:
        chunker = TextChunker()
        text = """# 2024-03-15 - A Great Day

Something happened."""

        chunks = chunker.chunk_document(text, source_file="diary.md")
        assert len(chunks) == 1
        assert chunks[0].date is not None
        assert chunks[0].date.day == 15

    def test_section_header_extraction(self) -> None:
        chunker = TextChunker()
        text = """## Morning

Had coffee and went for a run.

## Evening

Watched a movie with friends."""

        chunks = chunker.chunk_document(text, source_file="diary.md")
        titles = [c.section_title for c in chunks]
        assert "Morning" in titles or "Evening" in titles

    def test_default_date_used_when_no_header(self) -> None:
        chunker = TextChunker()
        default = datetime(2024, 1, 1)
        text = "No date header here. Just plain text."
        chunks = chunker.chunk_document(text, source_file="diary.md", default_date=default)
        assert chunks[0].date == default


class TestTextChunkerContextualHeaders:
    """Contextual header generation for improved retrieval."""

    def test_contextual_header_includes_source(self) -> None:
        chunker = TextChunker()
        text = "Some content here."
        chunks = chunker.chunk_document(text, source_file="diary_2024.md")
        header = chunks[0].metadata["contextual_header"]
        assert "diary_2024.md" in header

    def test_contextual_header_includes_date(self) -> None:
        chunker = TextChunker()
        text = """# 2024-06-15

Content for this date."""

        chunks = chunker.chunk_document(text, source_file="diary.md")
        header = chunks[0].metadata["contextual_header"]
        assert "2024-06-15" in header

    def test_contextual_header_includes_section(self) -> None:
        chunker = TextChunker()
        text = """## Reflections

Deep thoughts about life."""

        chunks = chunker.chunk_document(text, source_file="diary.md")
        header = chunks[0].metadata["contextual_header"]
        assert "Reflections" in header


class TestTextChunkerPlainText:
    """chunk_text method for non-markdown content."""

    def test_chunk_text_basic(self) -> None:
        chunker = TextChunker()
        text = "This is a conversation transcript."
        chunks = chunker.chunk_text(
            text,
            source_file="conv_001.txt",
            section_title="Chat",
        )
        assert len(chunks) == 1
        assert chunks[0].section_title == "Chat"

    def test_chunk_text_empty(self) -> None:
        chunker = TextChunker()
        assert chunker.chunk_text("") == []
