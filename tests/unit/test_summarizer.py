"""Tests for the summarizer module."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from ara.memory.summarizer import Summarizer, SummaryResult
from ara.utils.errors import AraMemoryError
from ara.utils.types import MemoryChunk, MemorySource

# ── SummaryResult ───────────────────────────────────────────────────────────


class TestSummaryResult:
    def test_create(self) -> None:
        result = SummaryResult(
            source_text="Some text here.",
            summary="Brief summary.",
            method="extractive",
            elapsed_ms=1.5,
        )
        assert result.summary == "Brief summary."
        assert result.method == "extractive"


# ── Extractive Summarization ───────────────────────────────────────────────


class TestSummarizerExtractive:
    def test_empty_text(self) -> None:
        s = Summarizer()
        result = s.summarize_extractive("")
        assert result.summary == ""
        assert result.method == "extractive"

    def test_whitespace_only(self) -> None:
        s = Summarizer()
        result = s.summarize_extractive("   \n  ")
        assert result.summary == ""

    def test_short_text_returns_opening(self) -> None:
        s = Summarizer()
        result = s.summarize_extractive("Today was a good day.")
        assert result.summary != ""
        assert result.method == "extractive"

    def test_extracts_first_person_statements(self) -> None:
        s = Summarizer()
        text = (
            "The weather was nice. "
            "I felt really happy about the promotion. "
            "The office was quiet. "
            "I decided to celebrate with friends."
        )
        result = s.summarize_extractive(text)
        # Should prefer first-person emotional/action statements
        assert "felt" in result.summary.lower() or "decided" in result.summary.lower()

    def test_fallback_to_opening_when_no_first_person(self) -> None:
        s = Summarizer()
        text = "The conference was held in Berlin. Many speakers presented their research. The keynote was excellent."
        result = s.summarize_extractive(text)
        assert "conference" in result.summary.lower() or "Berlin" in result.summary

    def test_max_sentences_respected(self) -> None:
        s = Summarizer(max_summary_sentences=1)
        text = "I felt happy. I decided to go. I met friends. I learned something new."
        result = s.summarize_extractive(text)
        # With max 1 sentence, should be concise
        assert result.summary.count(".") <= 2  # At most 1-2 periods

    def test_summary_ends_with_punctuation(self) -> None:
        s = Summarizer()
        result = s.summarize_extractive("I went to the store today")
        assert result.summary[-1] in ".!?"

    def test_elapsed_ms_is_positive(self) -> None:
        s = Summarizer()
        result = s.summarize_extractive("I felt great today. Everything went well.")
        assert result.elapsed_ms >= 0

    def test_multiline_diary_entry(self) -> None:
        s = Summarizer()
        text = (
            "# Morning\n\n"
            "Woke up early, couldn't sleep.\n\n"
            "I felt anxious about the interview.\n\n"
            "# Afternoon\n\n"
            "The interview went surprisingly well.\n"
            "I realized I had been overthinking it.\n\n"
            "Met Sarah for coffee after."
        )
        result = s.summarize_extractive(text)
        assert result.summary != ""
        assert len(result.summary) < len(text)


# ── LLM Summarization ─────────────────────────────────────────────────────


class TestSummarizerLLM:
    def test_no_llm_raises(self) -> None:
        s = Summarizer()
        with pytest.raises(AraMemoryError, match="No LLM client"):
            import asyncio

            asyncio.get_event_loop().run_until_complete(s.summarize_llm("Some text."))

    def test_has_llm_property(self) -> None:
        s = Summarizer()
        assert s.has_llm is False

        mock_llm = MagicMock()
        s2 = Summarizer(llm_client=mock_llm)
        assert s2.has_llm is True

    @pytest.mark.asyncio
    async def test_llm_empty_text(self) -> None:
        mock_llm = MagicMock()
        s = Summarizer(llm_client=mock_llm)
        result = await s.summarize_llm("")
        assert result.summary == ""
        assert result.method == "llm"

    @pytest.mark.asyncio
    async def test_llm_generates_summary(self) -> None:
        mock_llm = MagicMock()
        mock_llm.generate = AsyncMock(return_value="The author felt happy about a promotion.")
        s = Summarizer(llm_client=mock_llm)

        result = await s.summarize_llm("I got promoted today! So excited!")
        assert result.summary != ""
        assert result.method == "llm"
        mock_llm.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_summarize_prefers_llm_when_available(self) -> None:
        mock_llm = MagicMock()
        mock_llm.generate = AsyncMock(return_value="LLM-generated summary.")
        s = Summarizer(llm_client=mock_llm)

        result = await s.summarize("Some diary text.", prefer_llm=True)
        assert result.method == "llm"

    @pytest.mark.asyncio
    async def test_summarize_falls_back_to_extractive(self) -> None:
        s = Summarizer()  # No LLM
        result = await s.summarize("I felt happy today.")
        assert result.method == "extractive"

    @pytest.mark.asyncio
    async def test_summarize_falls_back_on_llm_error(self) -> None:
        mock_llm = MagicMock()
        mock_llm.generate = AsyncMock(side_effect=Exception("LLM down"))
        s = Summarizer(llm_client=mock_llm)

        result = await s.summarize("I felt happy today.", prefer_llm=True)
        assert result.method == "extractive"  # Fallback


# ── Chunk-to-Summary ──────────────────────────────────────────────────────


class TestSummarizerChunkToMemory:
    def test_creates_summary_chunk(self) -> None:
        s = Summarizer()
        original = MemoryChunk(
            chunk_id="abc123",
            text="Long diary entry here...",
            source=MemorySource.DIARY,
            source_file="diary.md",
            metadata={"chunk_index": 0, "token_count": 50},
        )
        summary_chunk = s.summarize_chunk_to_memory(original, "Short summary.")

        assert summary_chunk.text == "Short summary."
        assert summary_chunk.chunk_id != original.chunk_id
        assert summary_chunk.source == MemorySource.DIARY
        assert summary_chunk.source_file == "diary.md"
        assert summary_chunk.metadata["is_summary"] is True
        assert summary_chunk.metadata["original_chunk_id"] == "abc123"

    def test_summary_chunk_id_is_deterministic(self) -> None:
        s = Summarizer()
        original = MemoryChunk(
            chunk_id="abc123",
            text="Some text.",
            source=MemorySource.DIARY,
            metadata={},
        )
        chunk1 = s.summarize_chunk_to_memory(original, "Summary A.")
        chunk2 = s.summarize_chunk_to_memory(original, "Summary A.")
        assert chunk1.chunk_id == chunk2.chunk_id
