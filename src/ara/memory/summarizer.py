"""Progressive summarization for diary entries and text chunks.

Generates concise 1-2 sentence summaries of text content, either via
rule-based extraction (fast, offline) or LLM-based summarization
(higher quality, requires Ollama running).

Summaries are stored as searchable chunks in the vector store, enabling
retrieval of high-level overviews alongside detailed content.

All configuration comes from AraSettings.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ara.utils.errors import AraMemoryError
from ara.utils.logging import get_logger
from ara.utils.types import MemoryChunk, MessageRole

if TYPE_CHECKING:
    from ara.core.llm import LLMClient
    from ara.utils.types import ChatMessage

log = get_logger("summarizer")

# Max input length for rule-based summarization (characters)
_MAX_EXTRACTIVE_INPUT = 10_000

# Patterns for extracting key sentences
_FIRST_PERSON_PATTERNS = re.compile(
    r"(?:^|\. )"
    r"("
    r"I (?:felt|feel|realized|decided|learned|discovered|noticed|thought|believe|remember|"
    r"went|met|talked|called|started|finished|completed|broke|made|got|had|was|am|"
    r"need|want|hope|wish|love|hate|miss|fear|worry|plan|tried)"
    r"[^.!?]*[.!?]"
    r")",
    re.IGNORECASE | re.MULTILINE,
)

# LLM prompt for summarization
_SUMMARIZE_PROMPT = (
    "You are a personal diary summarizer. Given the following diary entry, "
    "write a concise 1-2 sentence summary capturing the key events, emotions, "
    "and decisions. Focus on what happened, how the person felt, and any "
    "important people mentioned. Write in third person past tense.\n\n"
    "Diary entry:\n{text}\n\n"
    "Summary:"
)


@dataclass(frozen=True)
class SummaryResult:
    """Result of summarizing a piece of text."""

    source_text: str
    summary: str
    method: str  # "extractive" or "llm"
    elapsed_ms: float


class Summarizer:
    """Text summarizer for diary entries and memory chunks.

    Supports two modes:
    - **Extractive** (default): Rule-based extraction of key sentences.
      Fast, works offline, no GPU needed. Good for bulk processing.
    - **LLM-based**: Uses Qwen3 via Ollama for high-quality abstractive
      summaries. Requires Ollama running. Better for rich diary entries.

    Usage:
        # Extractive (no LLM needed)
        summarizer = Summarizer()
        result = summarizer.summarize_extractive(text)

        # LLM-based (requires LLM client)
        summarizer = Summarizer(llm_client=client)
        result = await summarizer.summarize_llm(text)
    """

    def __init__(
        self,
        *,
        llm_client: LLMClient | None = None,
        max_summary_sentences: int = 2,
    ) -> None:
        self._llm = llm_client
        self._max_sentences = max_summary_sentences

    @property
    def has_llm(self) -> bool:
        """Whether an LLM client is available for abstractive summarization."""
        return self._llm is not None

    def summarize_extractive(self, text: str) -> SummaryResult:
        """Generate a summary using rule-based sentence extraction.

        Extracts the most informative sentences based on:
        1. First-person statements about feelings, actions, decisions
        2. Opening sentences of the text
        3. Sentences mentioning people or places

        Args:
            text: The text to summarize.

        Returns:
            SummaryResult with the extracted summary.
        """
        start = time.monotonic()

        if not text.strip():
            return SummaryResult(
                source_text=text,
                summary="",
                method="extractive",
                elapsed_ms=0.0,
            )

        # Truncate very long inputs
        working_text = text[:_MAX_EXTRACTIVE_INPUT] if len(text) > _MAX_EXTRACTIVE_INPUT else text

        # Strategy: find first-person key sentences
        key_sentences = _FIRST_PERSON_PATTERNS.findall(working_text)

        if key_sentences:
            # Take the most informative ones (longest = usually most detail)
            key_sentences.sort(key=len, reverse=True)
            selected = key_sentences[: self._max_sentences]
            summary = " ".join(s.strip() for s in selected)
        else:
            # Fallback: take the first 1-2 sentences of the text
            summary = self._extract_opening(working_text)

        # Clean up
        summary = self._clean_summary(summary)

        elapsed_ms = (time.monotonic() - start) * 1000
        log.debug(
            "extractive_summary",
            input_len=len(text),
            summary_len=len(summary),
            elapsed_ms=round(elapsed_ms, 1),
        )

        return SummaryResult(
            source_text=text,
            summary=summary,
            method="extractive",
            elapsed_ms=round(elapsed_ms, 1),
        )

    async def summarize_llm(
        self,
        text: str,
        *,
        model: str | None = None,
        max_input_tokens: int = 2048,
    ) -> SummaryResult:
        """Generate a summary using the LLM.

        Args:
            text: The text to summarize.
            model: Override model for this request.
            max_input_tokens: Max tokens to send to LLM (truncates if longer).

        Returns:
            SummaryResult with the LLM-generated summary.

        Raises:
            AraMemoryError: If no LLM client is configured or generation fails.
        """
        if self._llm is None:
            raise AraMemoryError(
                "No LLM client configured for summarization",
                operation="summarize_llm",
            )

        if not text.strip():
            return SummaryResult(
                source_text=text,
                summary="",
                method="llm",
                elapsed_ms=0.0,
            )

        start = time.monotonic()

        # Truncate input if too long
        from ara.utils.types import ChatMessage

        truncated_text = text[: max_input_tokens * 4]  # Rough char-to-token ratio
        prompt = _SUMMARIZE_PROMPT.format(text=truncated_text)

        # Note: Qwen3.5 controls thinking via API parameters, not inline /no_think.
        # We keep the prompt clean and rely on low temperature (0.3) to suppress
        # extended reasoning. The ThinkBlockFilter in streaming.py strips any
        # <think> blocks that may still appear.
        messages: list[ChatMessage] = [
            ChatMessage(role=MessageRole.USER, content=prompt),
        ]

        try:
            response = await self._llm.generate(
                messages,
                model=model,
                max_tokens=200,
                temperature=0.3,
            )

            summary = self._clean_summary(response.strip())

            elapsed_ms = (time.monotonic() - start) * 1000
            log.info(
                "llm_summary",
                input_len=len(text),
                summary_len=len(summary),
                elapsed_ms=round(elapsed_ms, 1),
            )

            return SummaryResult(
                source_text=text,
                summary=summary,
                method="llm",
                elapsed_ms=round(elapsed_ms, 1),
            )

        except Exception as exc:
            raise AraMemoryError(
                f"LLM summarization failed: {exc}",
                operation="summarize_llm",
                details={"input_len": len(text)},
            ) from exc

    async def summarize(
        self,
        text: str,
        *,
        prefer_llm: bool = True,
    ) -> SummaryResult:
        """Summarize text using the best available method.

        If an LLM client is available and prefer_llm is True, uses LLM.
        Otherwise falls back to extractive summarization.

        Args:
            text: Text to summarize.
            prefer_llm: Whether to prefer LLM if available.

        Returns:
            SummaryResult with the summary.
        """
        if prefer_llm and self._llm is not None:
            try:
                return await self.summarize_llm(text)
            except AraMemoryError:
                log.warning("llm_summary_fallback", reason="LLM failed, using extractive")
                return self.summarize_extractive(text)
        return self.summarize_extractive(text)

    def summarize_chunk_to_memory(
        self,
        chunk: MemoryChunk,
        summary: str,
    ) -> MemoryChunk:
        """Create a summary MemoryChunk from an existing chunk.

        The summary chunk gets a distinct chunk_id and is marked as a
        summary in its metadata, so it can be distinguished from
        full-content chunks during retrieval.

        Args:
            chunk: The original chunk being summarized.
            summary: The summary text.

        Returns:
            A new MemoryChunk containing the summary.
        """
        from hashlib import sha256

        summary_id = sha256(f"summary:{chunk.chunk_id}:{summary[:50]}".encode()).hexdigest()[:16]

        return MemoryChunk(
            chunk_id=summary_id,
            text=summary,
            source=chunk.source,
            source_file=chunk.source_file,
            date=chunk.date,
            section_title=chunk.section_title,
            metadata={
                **chunk.metadata,
                "is_summary": True,
                "original_chunk_id": chunk.chunk_id,
                "summary_method": "extractive",
            },
        )

    def _extract_opening(self, text: str) -> str:
        """Extract the first 1-2 sentences of text."""
        # Split on sentence boundaries
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        if not sentences:
            return text[:200].strip()

        selected = sentences[: self._max_sentences]
        return " ".join(s.strip() for s in selected if s.strip())

    @staticmethod
    def _clean_summary(summary: str) -> str:
        """Clean up a summary string."""
        # Remove multiple spaces
        summary = re.sub(r"\s+", " ", summary).strip()
        # Remove leading/trailing quotes if present
        if summary.startswith('"') and summary.endswith('"'):
            summary = summary[1:-1].strip()
        # Ensure ends with punctuation
        if summary and summary[-1] not in ".!?":
            summary += "."
        return summary
