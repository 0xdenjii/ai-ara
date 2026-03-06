"""Hybrid retrieval pipeline — vector + BM25 + graph + reranker.

Orchestrates the full retrieval flow:
    1. Query analysis — classify intent (temporal, entity, emotional, factual, global)
    2. Query rewriting — expand dates, resolve entities, expand pronouns
    3. Parallel retrieval — vector similarity, BM25 keyword, graph traversal
    4. Result fusion — reciprocal rank fusion (RRF) to merge ranked lists
    5. Reranking — cross-encoder scoring of top candidates
    6. Context assembly — deduplicate, format for LLM prompt injection

Architecture:
    - Designed for ARA's personal diary/memory use case
    - All search runs locally (LanceDB + NetworkX + CPU embeddings)
    - Reranker is optional (degrades gracefully if not loaded)
    - Graph traversal supplements vector/BM25 with entity context

Usage:
    from ara.memory.retrieval import RetrievalPipeline

    pipeline = RetrievalPipeline(settings, vector_store, embedder, graph, reranker)
    context = pipeline.retrieve("what did I talk about with Mom last week?")
"""

from __future__ import annotations

import contextlib
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any

import structlog

from ara.utils.errors import RetrievalError
from ara.utils.types import (
    IntentType,
    MemoryChunk,
    MemorySource,
    RetrievalContext,
    RetrievalResult,
)

if TYPE_CHECKING:
    from ara.config.settings import MemorySettings
    from ara.memory.embeddings import EmbeddingModel
    from ara.memory.graph import KnowledgeGraph
    from ara.memory.reranker import Reranker
    from ara.memory.vector_store import VectorStore

log = structlog.get_logger(component="retrieval")


# ── Constants ────────────────────────────────────────────────────────────────

# Reciprocal Rank Fusion constant (standard value from literature)
_RRF_K = 60

# Maximum characters for context assembly
_MAX_CONTEXT_CHARS = 12_000

# Query type classification keywords
_TEMPORAL_KEYWORDS = frozenset(
    {
        "when",
        "yesterday",
        "last week",
        "last month",
        "last year",
        "today",
        "this week",
        "this month",
        "ago",
        "recently",
        "before",
        "after",
        "during",
        "since",
        "until",
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
        "2020",
        "2021",
        "2022",
        "2023",
        "2024",
        "2025",
        "2026",
    }
)

_ENTITY_KEYWORDS = frozenset(
    {
        "who",
        "whom",
        "person",
        "people",
        "friend",
        "family",
        "mom",
        "dad",
        "mother",
        "father",
        "brother",
        "sister",
        "where",
        "place",
        "city",
        "country",
        "location",
    }
)

_EMOTIONAL_KEYWORDS = frozenset(
    {
        "feel",
        "felt",
        "feeling",
        "emotion",
        "mood",
        "happy",
        "sad",
        "angry",
        "anxious",
        "excited",
        "stressed",
        "depressed",
        "frustrated",
        "grateful",
        "lonely",
        "love",
        "hate",
        "fear",
        "joy",
        "worry",
    }
)

_PLANNING_KEYWORDS = frozenset(
    {
        "plan",
        "goal",
        "todo",
        "want to",
        "going to",
        "need to",
        "should",
        "will",
        "intend",
        "aim",
        "target",
        "deadline",
        "schedule",
        "priority",
        "next step",
        "future",
    }
)

# Relative date patterns for query rewriting
_RELATIVE_DATE_PATTERNS: list[tuple[re.Pattern[str], int]] = [
    (re.compile(r"\byesterday\b", re.IGNORECASE), 1),
    (re.compile(r"\b(\d+)\s+days?\s+ago\b", re.IGNORECASE), 0),  # group(1) is days
    (re.compile(r"\blast\s+week\b", re.IGNORECASE), 7),
    (re.compile(r"\blast\s+month\b", re.IGNORECASE), 30),
    (re.compile(r"\blast\s+year\b", re.IGNORECASE), 365),
    (re.compile(r"\bthis\s+week\b", re.IGNORECASE), 7),
    (re.compile(r"\bthis\s+month\b", re.IGNORECASE), 30),
    (re.compile(r"\brecently\b", re.IGNORECASE), 14),
]


# ── Data Types ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class QueryAnalysis:
    """Result of analyzing a user query.

    Attributes:
        original_query: The raw query text.
        query_type: Classified intent type.
        rewritten_query: Query after expansion/rewriting.
        date_from: Extracted start date for temporal filtering.
        date_to: Extracted end date for temporal filtering.
        entity_names: Entity names extracted from the query.
        source_filter: Optional source type filter.
        confidence: Confidence in the classification.
    """

    original_query: str
    query_type: IntentType
    rewritten_query: str = ""
    date_from: datetime | None = None
    date_to: datetime | None = None
    entity_names: list[str] = field(default_factory=list)
    source_filter: MemorySource | None = None
    confidence: float = 0.8


# ── Retrieval Pipeline ───────────────────────────────────────────────────────


class RetrievalPipeline:
    """Hybrid retrieval orchestrator.

    Combines vector similarity search, BM25 keyword search, and
    knowledge graph traversal with cross-encoder reranking.

    Args:
        settings: Memory settings (top_k, rerank_top_k).
        vector_store: LanceDB vector store for similarity + BM25 search.
        embedding_model: Embedding model for query vectorization.
        graph: Knowledge graph for entity-based retrieval.
        reranker: Optional cross-encoder reranker.
    """

    def __init__(
        self,
        settings: MemorySettings,
        vector_store: VectorStore,
        embedding_model: EmbeddingModel,
        *,
        graph: KnowledgeGraph | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self._settings = settings
        self._vector_store = vector_store
        self._embedder = embedding_model
        self._graph = graph
        self._reranker = reranker
        self._retrieval_top_k = settings.retrieval_top_k
        self._rerank_top_k = settings.rerank_top_k

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        source_filter: MemorySource | None = None,
        use_reranker: bool = True,
        use_graph: bool = True,
    ) -> RetrievalContext:
        """Execute the full retrieval pipeline.

        Steps:
            1. Analyze and classify the query
            2. Rewrite the query (expand dates, entities)
            3. Run vector search + BM25 search (+ graph if enabled)
            4. Fuse results with reciprocal rank fusion
            5. Rerank with cross-encoder (if available and enabled)
            6. Assemble context for LLM prompt injection

        Args:
            query: The user's search query.
            top_k: Override for number of final results.
            source_filter: Optional filter to restrict by source type.
            use_reranker: Whether to apply cross-encoder reranking.
            use_graph: Whether to include graph traversal results.

        Returns:
            RetrievalContext ready for LLM prompt injection.

        Raises:
            RetrievalError: If retrieval fails critically.
        """
        start = time.perf_counter()
        final_top_k = top_k or self._rerank_top_k

        if not query.strip():
            return RetrievalContext(
                results=[],
                query=query,
                query_type=IntentType.GENERAL,
            )

        try:
            # Step 1+2: Analyze and rewrite query
            analysis = self.analyze_query(query)
            if source_filter is not None:
                analysis = QueryAnalysis(
                    original_query=analysis.original_query,
                    query_type=analysis.query_type,
                    rewritten_query=analysis.rewritten_query,
                    date_from=analysis.date_from,
                    date_to=analysis.date_to,
                    entity_names=analysis.entity_names,
                    source_filter=source_filter,
                    confidence=analysis.confidence,
                )

            search_query = analysis.rewritten_query or query

            # Step 3: Parallel retrieval from multiple sources
            vector_results = self._vector_search(search_query, analysis=analysis)
            bm25_results = self._bm25_search(search_query)

            graph_results: list[_ScoredChunk] = []
            if use_graph and self._graph is not None and analysis.entity_names:
                graph_results = self._graph_search(analysis)

            total_candidates = len(vector_results) + len(bm25_results) + len(graph_results)

            # Step 4: Fuse results with RRF
            fused = _reciprocal_rank_fusion(
                [vector_results, bm25_results, graph_results],
                k=_RRF_K,
            )

            # Step 5: Rerank (if enabled and available)
            if use_reranker and self._reranker is not None and self._reranker.is_loaded and fused:
                fused = self._apply_reranker(query, fused, top_k=final_top_k)
            else:
                fused = fused[:final_top_k]

            # Step 6: Build RetrievalResult objects
            results = _build_results(fused, reranked=use_reranker and self._reranker is not None)

            elapsed_ms = (time.perf_counter() - start) * 1000

            log.info(
                "retrieval_complete",
                query_len=len(query),
                query_type=analysis.query_type.value,
                vector_hits=len(vector_results),
                bm25_hits=len(bm25_results),
                graph_hits=len(graph_results),
                total_candidates=total_candidates,
                final_results=len(results),
                elapsed_ms=round(elapsed_ms, 1),
            )

            return RetrievalContext(
                results=results,
                query=query,
                query_type=analysis.query_type,
                total_candidates=total_candidates,
                retrieval_time_ms=elapsed_ms,
            )

        except RetrievalError:
            raise
        except Exception as exc:
            raise RetrievalError(
                f"Retrieval pipeline failed: {exc}",
                details={"query": query[:200]},
            ) from exc

    # ── Query Analysis (P4C-02) ──────────────────────────────────────────────

    def analyze_query(self, query: str) -> QueryAnalysis:
        """Classify the query type and extract structured parameters.

        Uses keyword matching for fast, offline classification. The LLM
        can later refine this in the agent layer if needed.

        Args:
            query: The user's raw query.

        Returns:
            QueryAnalysis with classified type, date range, entity names.
        """
        query_lower = query.lower()
        tokens = set(query_lower.split())

        # Score each category by keyword overlap
        scores: dict[IntentType, float] = {
            IntentType.FACTUAL: 0.0,
            IntentType.EMOTIONAL: 0.0,
            IntentType.PLANNING: 0.0,
            IntentType.REMINISCING: 0.0,
            IntentType.GENERAL: 0.1,  # Small baseline
        }

        # Check temporal keywords (contributes to REMINISCING)
        temporal_score = 0.0
        for kw in _TEMPORAL_KEYWORDS:
            if kw in query_lower:
                temporal_score += 1.0
        if temporal_score > 0:
            scores[IntentType.REMINISCING] += temporal_score

        # Check entity keywords
        entity_score = 0.0
        for kw in _ENTITY_KEYWORDS:
            if kw in query_lower:
                entity_score += 1.0
        if entity_score > 0:
            scores[IntentType.FACTUAL] += entity_score

        # Check emotional keywords
        for kw in _EMOTIONAL_KEYWORDS:
            if kw in query_lower:
                scores[IntentType.EMOTIONAL] += 1.0

        # Check planning keywords
        for kw in _PLANNING_KEYWORDS:
            if kw in query_lower:
                scores[IntentType.PLANNING] += 1.0

        # Question words boost factual
        question_words = {"what", "how", "why", "which", "tell"}
        if tokens & question_words:
            scores[IntentType.FACTUAL] += 0.5

        # Pick highest scoring type
        best_type = max(scores, key=lambda t: scores[t])
        best_score = scores[best_type]
        total_score = sum(scores.values())
        confidence = best_score / total_score if total_score > 0 else 0.5

        # Extract dates and entities
        date_from, date_to = _extract_date_range(query)
        entity_names = self._extract_query_entities(query)

        # Build rewritten query
        rewritten = _rewrite_query(query, date_from=date_from)

        return QueryAnalysis(
            original_query=query,
            query_type=best_type,
            rewritten_query=rewritten,
            date_from=date_from,
            date_to=date_to,
            entity_names=entity_names,
            confidence=min(1.0, confidence),
        )

    def _extract_query_entities(self, query: str) -> list[str]:
        """Extract entity names from the query using graph lookup.

        Checks each word and bigram against the knowledge graph to find
        known entities mentioned in the query.
        """
        if self._graph is None:
            return []

        found: list[str] = []
        words = query.split()

        # Check single words and bigrams
        candidates: list[str] = list(words)
        candidates.extend(f"{words[i]} {words[i + 1]}" for i in range(len(words) - 1))

        for candidate in candidates:
            # Skip very short words and common words
            if len(candidate) < 3:
                continue
            clean = candidate.strip("?!.,;:'\"")
            if not clean:
                continue
            entity = self._graph.get_entity_by_name(clean)
            if entity is not None and entity.name not in found:
                found.append(entity.name)

        return found

    # ── Vector Search (P4C-04) ───────────────────────────────────────────────

    def _vector_search(
        self,
        query: str,
        *,
        analysis: QueryAnalysis,
    ) -> list[_ScoredChunk]:
        """Run vector similarity search via LanceDB.

        Args:
            query: The (possibly rewritten) search query.
            analysis: Query analysis with optional date/source filters.

        Returns:
            List of scored chunks from vector search.
        """
        try:
            query_vector = self._embedder.embed_query(query)
        except Exception as exc:
            log.warning("vector_search_embed_failed", error=str(exc))
            return []

        try:
            raw_results = self._vector_store.search(
                query_vector,
                top_k=self._retrieval_top_k,
                source_filter=analysis.source_filter,
                date_from=analysis.date_from,
                date_to=analysis.date_to,
            )
        except Exception as exc:
            log.warning("vector_search_failed", error=str(exc))
            return []

        return [
            _ScoredChunk(
                chunk=_dict_to_chunk(r),
                score=1.0 - float(r.get("_distance", 0.5)),
                method="vector",
            )
            for r in raw_results
        ]

    # ── BM25 Search (P4C-04) ────────────────────────────────────────────────

    def _bm25_search(self, query: str) -> list[_ScoredChunk]:
        """Run BM25 full-text search via LanceDB FTS index.

        Args:
            query: The search query (keywords work best for BM25).

        Returns:
            List of scored chunks from BM25 search.
        """
        try:
            raw_results = self._vector_store.full_text_search(
                query,
                top_k=self._retrieval_top_k,
            )
        except Exception as exc:
            log.warning("bm25_search_failed", error=str(exc))
            return []

        return [
            _ScoredChunk(
                chunk=_dict_to_chunk(r),
                score=float(r.get("score", r.get("_score", 0.5))),
                method="bm25",
            )
            for r in raw_results
        ]

    # ── Graph Search (P4C-05) ────────────────────────────────────────────────

    def _graph_search(self, analysis: QueryAnalysis) -> list[_ScoredChunk]:
        """Search the knowledge graph for entity-related context.

        For each entity found in the query, retrieves the entity's
        context (relationships, neighbors) and creates synthetic chunks.

        Args:
            analysis: Query analysis with entity names.

        Returns:
            List of scored chunks from graph traversal.
        """
        if self._graph is None:
            return []

        results: list[_ScoredChunk] = []
        seen_entities: set[str] = set()

        for name in analysis.entity_names:
            entity = self._graph.get_entity_by_name(name)
            if entity is None or entity.entity_id in seen_entities:
                continue
            seen_entities.add(entity.entity_id)

            # Get entity context (natural language description of relationships)
            context_text = self._graph.get_entity_context(entity.entity_id)
            if not context_text.strip():
                continue

            # Create a synthetic chunk from graph context
            chunk = MemoryChunk(
                chunk_id=f"graph_{entity.entity_id}",
                text=context_text,
                source=MemorySource.DIARY,
                source_file=f"graph://{entity.name}",
                section_title=f"Knowledge graph: {entity.name}",
                metadata={
                    "entity_id": entity.entity_id,
                    "entity_type": entity.entity_type.value
                    if hasattr(entity.entity_type, "value")
                    else str(entity.entity_type),
                    "from_graph": True,
                },
            )

            results.append(
                _ScoredChunk(
                    chunk=chunk,
                    score=entity.confidence * 0.8,
                    method="graph",
                )
            )

            # Also get neighbors for multi-hop context
            neighbors = self._graph.get_neighbors(entity.entity_id, depth=1)
            for neighbor in neighbors[:5]:
                if neighbor.entity_id in seen_entities:
                    continue
                seen_entities.add(neighbor.entity_id)

                neighbor_context = self._graph.get_entity_context(
                    neighbor.entity_id, max_relations=10
                )
                if not neighbor_context.strip():
                    continue

                neighbor_chunk = MemoryChunk(
                    chunk_id=f"graph_{neighbor.entity_id}",
                    text=neighbor_context,
                    source=MemorySource.DIARY,
                    source_file=f"graph://{neighbor.name}",
                    section_title=f"Knowledge graph: {neighbor.name}",
                    metadata={
                        "entity_id": neighbor.entity_id,
                        "entity_type": neighbor.entity_type.value
                        if hasattr(neighbor.entity_type, "value")
                        else str(neighbor.entity_type),
                        "from_graph": True,
                        "hop_distance": 1,
                    },
                )

                results.append(
                    _ScoredChunk(
                        chunk=neighbor_chunk,
                        score=neighbor.confidence * 0.5,
                        method="graph",
                    )
                )

        return results

    # ── Reranking (P4C-06) ───────────────────────────────────────────────────

    def _apply_reranker(
        self,
        query: str,
        candidates: list[_ScoredChunk],
        *,
        top_k: int,
    ) -> list[_ScoredChunk]:
        """Apply cross-encoder reranking to fused candidates.

        Args:
            query: The original user query.
            candidates: Fused candidates from RRF.
            top_k: Number of results to return after reranking.

        Returns:
            Top-k candidates rescored by the cross-encoder.
        """
        if self._reranker is None or not self._reranker.is_loaded:
            return candidates[:top_k]

        from ara.memory.reranker import RerankCandidate

        rerank_candidates = [
            RerankCandidate(
                text=c.chunk.text,
                original_score=c.score,
                metadata={
                    "chunk_id": c.chunk.chunk_id,
                    "method": c.method,
                    "chunk": c.chunk,
                },
            )
            for c in candidates[: self._retrieval_top_k]
        ]

        try:
            reranked = self._reranker.rerank(query, rerank_candidates, top_k=top_k)
        except Exception as exc:
            log.warning("reranker_failed_fallback_to_rrf", error=str(exc))
            return candidates[:top_k]

        # Convert back to _ScoredChunk with rerank scores
        result: list[_ScoredChunk] = []
        for r in reranked:
            meta = r.metadata or {}
            chunk = meta.get("chunk")
            if chunk is None:
                continue
            result.append(
                _ScoredChunk(
                    chunk=chunk,
                    score=r.rerank_score,
                    method=meta.get("method", "reranked"),
                    rerank_score=r.rerank_score,
                )
            )

        return result

    # ── Context Assembly (P4C-07) ────────────────────────────────────────────

    @staticmethod
    def assemble_context(
        results: list[RetrievalResult],
        *,
        max_chars: int = _MAX_CONTEXT_CHARS,
        include_metadata: bool = True,
    ) -> str:
        """Assemble retrieval results into a text block for LLM prompt.

        Formats results with source attribution and deduplicates by
        chunk_id. Truncates to max_chars to fit context window.

        Args:
            results: Retrieval results to assemble.
            max_chars: Maximum characters in the assembled context.
            include_metadata: Whether to include source/date headers.

        Returns:
            Formatted context string ready for LLM prompt injection.
        """
        if not results:
            return ""

        # Deduplicate by chunk_id
        seen_ids: set[str] = set()
        unique_results: list[RetrievalResult] = []
        for r in results:
            if r.chunk.chunk_id not in seen_ids:
                seen_ids.add(r.chunk.chunk_id)
                unique_results.append(r)

        parts: list[str] = []
        total_chars = 0

        for idx, result in enumerate(unique_results, 1):
            chunk = result.chunk

            # Build header
            header_parts: list[str] = []
            if include_metadata:
                if chunk.source_file:
                    header_parts.append(f"Source: {chunk.source_file}")
                if chunk.date:
                    header_parts.append(f"Date: {chunk.date.strftime('%Y-%m-%d')}")
                if chunk.section_title:
                    header_parts.append(f"Section: {chunk.section_title}")

            header = f"[Memory {idx}]"
            if header_parts:
                header += f" ({', '.join(header_parts)})"

            entry = f"{header}\n{chunk.text.strip()}\n"

            if total_chars + len(entry) > max_chars:
                # Add truncated entry if there's space for at least part of it
                remaining = max_chars - total_chars
                if remaining > 100:
                    parts.append(entry[:remaining] + "...")
                break

            parts.append(entry)
            total_chars += len(entry)

        return "\n".join(parts)


# ── Internal Types ───────────────────────────────────────────────────────────


@dataclass
class _ScoredChunk:
    """Internal scored chunk used during retrieval pipeline processing."""

    chunk: MemoryChunk
    score: float
    method: str  # "vector", "bm25", "graph", "reranked"
    rerank_score: float | None = None


# ── Helper Functions ─────────────────────────────────────────────────────────


def _extract_date_range(
    query: str,
) -> tuple[datetime | None, datetime | None]:
    """Extract date range from temporal references in the query.

    Supports:
        - "yesterday" → (yesterday, today)
        - "last week" → (7 days ago, now)
        - "last month" → (30 days ago, now)
        - "N days ago" → (N days ago, now)
        - "recently" → (14 days ago, now)

    Returns:
        Tuple of (date_from, date_to). Either may be None.
    """
    now = datetime.now()
    date_from: datetime | None = None
    date_to: datetime | None = None

    for pattern, days in _RELATIVE_DATE_PATTERNS:
        match = pattern.search(query)
        if match:
            if days == 0 and match.lastindex and match.lastindex >= 1:
                # "N days ago" pattern — extract N from the match
                try:
                    days = int(match.group(1))
                except (ValueError, IndexError):
                    continue
            date_from = now - timedelta(days=days)
            date_to = now
            break

    return date_from, date_to


def _rewrite_query(
    query: str,
    *,
    date_from: datetime | None = None,
) -> str:
    """Rewrite the query for better retrieval.

    Expands temporal references with explicit dates and cleans up
    the query for embedding.

    Args:
        query: The original query.
        date_from: Extracted start date (for adding date context).

    Returns:
        Rewritten query string.
    """
    rewritten = query

    # Add date context if we extracted a date range
    if date_from is not None:
        date_str = date_from.strftime("%Y-%m-%d")
        # Don't duplicate if date is already in the query
        if date_str not in rewritten:
            rewritten = f"{rewritten} (around {date_str})"

    return rewritten


def _reciprocal_rank_fusion(
    ranked_lists: list[list[_ScoredChunk]],
    *,
    k: int = 60,
) -> list[_ScoredChunk]:
    """Merge multiple ranked lists using Reciprocal Rank Fusion (RRF).

    RRF score = sum(1 / (k + rank_i)) across all lists where the
    document appears. This is robust to score scale differences.

    Args:
        ranked_lists: Multiple ranked lists of scored chunks.
        k: RRF smoothing constant (standard: 60).

    Returns:
        Single merged list sorted by RRF score (descending).
    """
    # Accumulate RRF scores by chunk_id
    rrf_scores: dict[str, float] = {}
    chunk_lookup: dict[str, _ScoredChunk] = {}
    methods: dict[str, set[str]] = {}

    for ranked_list in ranked_lists:
        for rank, scored in enumerate(ranked_list):
            cid = scored.chunk.chunk_id
            rrf_score = 1.0 / (k + rank + 1)  # rank is 0-indexed
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + rrf_score

            # Keep the chunk with the highest original score
            if cid not in chunk_lookup or scored.score > chunk_lookup[cid].score:
                chunk_lookup[cid] = scored

            if cid not in methods:
                methods[cid] = set()
            methods[cid].add(scored.method)

    # Build merged list
    merged: list[_ScoredChunk] = []
    for cid, rrf_score in rrf_scores.items():
        original = chunk_lookup[cid]
        method = "hybrid" if len(methods.get(cid, set())) > 1 else original.method
        merged.append(
            _ScoredChunk(
                chunk=original.chunk,
                score=rrf_score,
                method=method,
            )
        )

    # Sort by RRF score descending
    merged.sort(key=lambda s: s.score, reverse=True)
    return merged


def _dict_to_chunk(raw: dict[str, Any]) -> MemoryChunk:
    """Convert a raw dict from VectorStore search to a MemoryChunk.

    Handles the field mapping between LanceDB row format and our
    MemoryChunk dataclass.
    """
    date_str = raw.get("date", "")
    date: datetime | None = None
    if date_str:
        with contextlib.suppress(ValueError, TypeError):
            date = datetime.fromisoformat(date_str)

    source_str = raw.get("source", "ingested_file")
    try:
        source = MemorySource(source_str)
    except ValueError:
        source = MemorySource.INGESTED_FILE

    return MemoryChunk(
        chunk_id=str(raw.get("chunk_id", "")),
        text=str(raw.get("text", "")),
        source=source,
        source_file=str(raw.get("source_file", "")),
        date=date,
        section_title=str(raw.get("section_title", "")),
        metadata={
            "contextual_header": raw.get("contextual_header", ""),
            "token_count": raw.get("token_count", 0),
            "chunk_index": raw.get("chunk_index", 0),
        },
    )


def _build_results(
    scored_chunks: list[_ScoredChunk],
    *,
    reranked: bool = False,
) -> list[RetrievalResult]:
    """Convert internal scored chunks to public RetrievalResult objects."""
    return [
        RetrievalResult(
            chunk=sc.chunk,
            score=sc.score,
            retrieval_method=sc.method,
            rerank_score=sc.rerank_score if reranked else None,
        )
        for sc in scored_chunks
    ]
