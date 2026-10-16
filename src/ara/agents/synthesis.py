"""LlamaIndex response synthesis wrapper — generates answers from retrieved context.

Bridges ARA's custom retrieval pipeline with LlamaIndex's ResponseSynthesizer,
providing structured answer generation from retrieved memory chunks.

Architecture:
    - Our RetrievalPipeline handles all retrieval (vector + BM25 + graph + reranking)
    - LlamaIndex's ResponseSynthesizer handles answer generation from retrieved chunks
    - Uses LlamaIndex's Ollama LLM adapter to share the same inference backend
    - Supports multiple synthesis modes: compact, refine, tree_summarize

Usage:
    synth = ResponseSynthesis(settings)
    await synth.connect()
    answer = await synth.synthesize(query, retrieval_context)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

from llama_index.core.response_synthesizers import ResponseMode, get_response_synthesizer
from llama_index.core.schema import NodeWithScore, TextNode

from ara.utils.logging import get_logger

if TYPE_CHECKING:
    from llama_index.core.llms import LLM

    from ara.config.settings import AraSettings
    from ara.utils.types import RetrievalContext, RetrievalResult

log = get_logger("synthesis")


# ── Types ────────────────────────────────────────────────────────────────────


class SynthesisMode(StrEnum):
    """Available synthesis strategies."""

    COMPACT = "compact"
    REFINE = "refine"
    TREE_SUMMARIZE = "tree_summarize"
    SIMPLE = "simple_summarize"


@dataclass(frozen=True)
class SynthesisResult:
    """Result of response synthesis.

    Attributes:
        answer: The generated answer text.
        mode: The synthesis mode used.
        source_count: Number of source chunks used.
        synthesis_time_ms: Time taken for synthesis.
        metadata: Additional metadata (e.g., intermediate results from refine mode).
    """

    answer: str
    mode: SynthesisMode
    source_count: int
    synthesis_time_ms: float = 0.0
    metadata: dict[str, str | float | int] = field(default_factory=dict)


# ── Mode Selection ───────────────────────────────────────────────────────────

# Map SynthesisMode to LlamaIndex ResponseMode
_MODE_MAP: dict[SynthesisMode, ResponseMode] = {
    SynthesisMode.COMPACT: ResponseMode.COMPACT,
    SynthesisMode.REFINE: ResponseMode.REFINE,
    SynthesisMode.TREE_SUMMARIZE: ResponseMode.TREE_SUMMARIZE,
    SynthesisMode.SIMPLE: ResponseMode.SIMPLE_SUMMARIZE,
}

# Thresholds for auto-selecting synthesis mode
_COMPACT_MAX_CHUNKS = 5
_REFINE_MAX_CHUNKS = 15


def _auto_select_mode(chunk_count: int) -> SynthesisMode:
    """Auto-select the best synthesis mode based on chunk count.

    - Few chunks (≤5): COMPACT — concatenates and answers in one pass
    - Medium chunks (6-15): REFINE — iteratively refines answer per chunk
    - Many chunks (>15): TREE_SUMMARIZE — hierarchical summarization tree
    """
    if chunk_count <= _COMPACT_MAX_CHUNKS:
        return SynthesisMode.COMPACT
    if chunk_count <= _REFINE_MAX_CHUNKS:
        return SynthesisMode.REFINE
    return SynthesisMode.TREE_SUMMARIZE


# ── Response Synthesis ───────────────────────────────────────────────────────


class ResponseSynthesis:
    """LlamaIndex-based response synthesizer for ARA.

    Wraps LlamaIndex's ResponseSynthesizer to generate answers from
    retrieved memory chunks. Uses the same Ollama backend as the main
    LLM client.

    Args:
        settings: Application settings for model and connection config.
    """

    def __init__(self, settings: AraSettings) -> None:
        self._settings = settings
        self._llm: LLM | None = None
        self._connected = False

    @property
    def is_connected(self) -> bool:
        """Whether the synthesizer is connected to the LLM backend."""
        return self._connected

    async def connect(self) -> None:
        """Initialize the LlamaIndex LLM adapter for the configured backend.

        Supports both ``ollama`` and ``llamacpp`` backends. For ollama it
        uses the native LlamaIndex Ollama adapter; for llamacpp it uses the
        OpenAI adapter pointed at the local llama.cpp server (which exposes
        an OpenAI-compatible API).
        """
        backend = self._settings.llm.backend_type.lower()
        timeout_s = float(self._settings.llm.ollama_generation_timeout_s)

        if backend == "llamacpp":
            from llama_index.llms.openai import OpenAI

            self._llm = OpenAI(
                model=self._settings.llm.primary_model,
                api_base=self._settings.llm.llamacpp_base_url,
                api_key="not-needed",
                temperature=self._settings.llm.temperature,
                timeout=timeout_s,
            )
            log.info(
                "synthesis_connected",
                backend="llamacpp",
                model=self._settings.llm.primary_model,
                base_url=self._settings.llm.llamacpp_base_url,
            )
        else:
            from llama_index.llms.ollama import Ollama

            self._llm = Ollama(
                model=self._settings.llm.primary_model,
                base_url=self._settings.llm.ollama_base_url,
                temperature=self._settings.llm.temperature,
                request_timeout=timeout_s,
            )
            log.info(
                "synthesis_connected",
                backend="ollama",
                model=self._settings.llm.primary_model,
                base_url=self._settings.llm.ollama_base_url,
            )

        self._connected = True

    async def close(self) -> None:
        """Clean up resources."""
        self._llm = None
        self._connected = False
        log.info("synthesis_closed")

    async def synthesize(
        self,
        query: str,
        context: RetrievalContext,
        *,
        mode: SynthesisMode | None = None,
        system_prompt: str = "",
    ) -> SynthesisResult:
        """Generate an answer from retrieved context.

        Args:
            query: The user's original query.
            context: Retrieved memory chunks from RetrievalPipeline.
            mode: Override synthesis mode (auto-selected if None).
            system_prompt: Optional system prompt to prepend (personality, etc.).

        Returns:
            SynthesisResult with the generated answer.

        Raises:
            RuntimeError: If not connected.
            ValueError: If context has no results.
        """
        if not self._connected or self._llm is None:
            msg = "ResponseSynthesis not connected. Call connect() first."
            raise RuntimeError(msg)

        start = time.perf_counter()

        if not context.results:
            return SynthesisResult(
                answer="",
                mode=mode or SynthesisMode.COMPACT,
                source_count=0,
                synthesis_time_ms=0.0,
                metadata={"reason": "no_results"},
            )

        # Select synthesis mode
        selected_mode = mode or _auto_select_mode(len(context.results))
        llama_mode = _MODE_MAP[selected_mode]

        # Convert our RetrievalResults to LlamaIndex nodes
        nodes = self._results_to_nodes(context.results)

        # Build the synthesizer with current settings
        synthesizer = get_response_synthesizer(
            llm=self._llm,
            response_mode=llama_mode,
            verbose=False,
        )

        # Prepend system prompt to query if provided
        full_query = query
        if system_prompt:
            full_query = f"[System: {system_prompt}]\n\nUser query: {query}"

        # Run synthesis (LlamaIndex handles the LLM calls internally)
        response = await synthesizer.asynthesize(
            query=full_query,
            nodes=nodes,
        )

        elapsed_ms = (time.perf_counter() - start) * 1000
        answer = str(response)

        log.info(
            "synthesis_complete",
            mode=selected_mode.value,
            source_count=len(nodes),
            answer_length=len(answer),
            elapsed_ms=round(elapsed_ms, 1),
        )

        return SynthesisResult(
            answer=answer,
            mode=selected_mode,
            source_count=len(nodes),
            synthesis_time_ms=elapsed_ms,
            metadata={
                "query_type": context.query_type.value,
                "total_candidates": context.total_candidates,
            },
        )

    async def synthesize_from_text(
        self,
        query: str,
        texts: list[str],
        *,
        mode: SynthesisMode | None = None,
    ) -> SynthesisResult:
        """Generate an answer from raw text chunks (no RetrievalContext).

        Convenience method for when you have plain text chunks rather than
        full retrieval results.

        Args:
            query: The user's query.
            texts: List of text chunks to synthesize from.
            mode: Override synthesis mode.

        Returns:
            SynthesisResult with the generated answer.
        """
        if not self._connected or self._llm is None:
            msg = "ResponseSynthesis not connected. Call connect() first."
            raise RuntimeError(msg)

        start = time.perf_counter()

        if not texts:
            return SynthesisResult(
                answer="",
                mode=mode or SynthesisMode.COMPACT,
                source_count=0,
                synthesis_time_ms=0.0,
            )

        selected_mode = mode or _auto_select_mode(len(texts))
        llama_mode = _MODE_MAP[selected_mode]

        nodes = [
            NodeWithScore(
                node=TextNode(text=text, id_=f"text_{i}"),
                score=1.0,
            )
            for i, text in enumerate(texts)
        ]

        synthesizer = get_response_synthesizer(
            llm=self._llm,
            response_mode=llama_mode,
            verbose=False,
        )

        response = await synthesizer.asynthesize(query=query, nodes=nodes)
        elapsed_ms = (time.perf_counter() - start) * 1000
        answer = str(response)

        return SynthesisResult(
            answer=answer,
            mode=selected_mode,
            source_count=len(texts),
            synthesis_time_ms=elapsed_ms,
        )

    # ── Private Helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _results_to_nodes(results: list[RetrievalResult]) -> list[NodeWithScore]:
        """Convert ARA RetrievalResults to LlamaIndex NodeWithScore objects."""
        nodes: list[NodeWithScore] = []
        for result in results:
            chunk = result.chunk
            metadata: dict[str, str] = {
                "source": chunk.source.value,
                "source_file": chunk.source_file,
                "retrieval_method": result.retrieval_method,
            }
            if chunk.section_title:
                metadata["section_title"] = chunk.section_title
            if chunk.date:
                metadata["date"] = chunk.date.isoformat()

            node = TextNode(
                text=chunk.text,
                id_=chunk.chunk_id,
                metadata=metadata,
            )
            score = result.rerank_score if result.rerank_score is not None else result.score

            nodes.append(NodeWithScore(node=node, score=score))
        return nodes
