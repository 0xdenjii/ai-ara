"""Cross-encoder reranker using bge-reranker-v2-m3.

Takes a query and a list of candidate text passages, scores each
query-passage pair using a cross-encoder model, and returns the
top-k results sorted by relevance.

Architecture:
    - Uses sentence-transformers CrossEncoder (BAAI/bge-reranker-v2-m3)
    - Runs entirely on CPU (~1.1GB RAM, 0 VRAM)
    - Cross-encoders are more accurate than bi-encoders for reranking
      because they jointly attend to both query and passage
    - Expected improvement: +8-11% retrieval accuracy over vector-only

Usage:
    from ara.memory.reranker import Reranker

    reranker = Reranker(settings)
    reranker.load()
    scored = reranker.rerank("what did I do last week?", candidates, top_k=5)
    reranker.unload()
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import structlog

from ara.utils.errors import RetrievalError

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder as CrossEncoderType

    from ara.config.settings import MemorySettings

log = structlog.get_logger(component="reranker")


# ── Data Types ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RerankCandidate:
    """A candidate passage to be reranked.

    Attributes:
        text: The passage text to score against the query.
        original_score: The score from the initial retrieval stage.
        metadata: Arbitrary metadata to carry through (chunk_id, source, etc.).
    """

    text: str
    original_score: float = 0.0
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class RerankResult:
    """A reranked passage with cross-encoder score.

    Attributes:
        text: The passage text.
        rerank_score: Cross-encoder relevance score (higher = more relevant).
        original_score: Score from the initial retrieval stage.
        original_rank: Position in the original candidate list (0-indexed).
        metadata: Carried-through metadata from the candidate.
    """

    text: str
    rerank_score: float
    original_score: float = 0.0
    original_rank: int = 0
    metadata: dict[str, Any] | None = None


# ── Reranker ─────────────────────────────────────────────────────────────────


class Reranker:
    """Cross-encoder reranker using bge-reranker-v2-m3.

    Loads the model on CPU, scores query-passage pairs, and returns
    the top-k most relevant passages.

    Args:
        settings: Memory settings containing reranker model name.
        device: Device to load model on. Defaults to "cpu".
        batch_size: Batch size for scoring. Larger = faster but more RAM.
    """

    def __init__(
        self,
        settings: MemorySettings,
        *,
        device: str = "cpu",
        batch_size: int = 32,
    ) -> None:
        self._model_name = settings.reranker_model
        self._device = device
        self._batch_size = batch_size
        self._model: CrossEncoderType | None = None
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        """Whether the reranker model is loaded and ready."""
        return self._loaded and self._model is not None

    @property
    def model_name(self) -> str:
        """The model name/path."""
        return self._model_name

    def load(self) -> None:
        """Load the cross-encoder model into memory.

        Raises:
            RetrievalError: If the model cannot be loaded.
        """
        if self._loaded:
            return

        start = time.perf_counter()

        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RetrievalError(
                "sentence-transformers not installed. Run: pip install sentence-transformers",
                details={"package": "sentence-transformers"},
            ) from exc

        try:
            self._model = CrossEncoder(
                self._model_name,
                device=self._device,
                trust_remote_code=True,
            )
            self._loaded = True

            elapsed_ms = (time.perf_counter() - start) * 1000
            log.info(
                "reranker_loaded",
                model=self._model_name,
                device=self._device,
                elapsed_ms=round(elapsed_ms, 1),
            )
        except Exception as exc:
            raise RetrievalError(
                f"Failed to load reranker model '{self._model_name}': {exc}",
                details={"model": self._model_name, "device": self._device},
            ) from exc

    def unload(self) -> None:
        """Unload the model to free memory."""
        if not self._loaded:
            return

        self._model = None
        self._loaded = False
        log.info("reranker_unloaded", model=self._model_name)

    def rerank(
        self,
        query: str,
        candidates: list[RerankCandidate],
        *,
        top_k: int = 5,
    ) -> list[RerankResult]:
        """Rerank candidate passages by relevance to the query.

        Scores each (query, passage) pair with the cross-encoder and
        returns the top-k results sorted by rerank score (descending).

        Args:
            query: The search query.
            candidates: List of candidate passages to rerank.
            top_k: Number of top results to return.

        Returns:
            List of RerankResult sorted by rerank_score (highest first).

        Raises:
            RetrievalError: If the model is not loaded.
        """
        if not self._loaded or self._model is None:
            raise RetrievalError(
                "Reranker not loaded — call load() first",
                details={"model": self._model_name},
            )

        if not candidates:
            return []

        if not query.strip():
            log.warning("rerank_empty_query")
            return []

        start = time.perf_counter()

        # Build query-passage pairs for the cross-encoder
        pairs: list[list[str]] = [[query, c.text] for c in candidates]

        # Score all pairs
        try:
            scores = self._model.predict(
                pairs,
                batch_size=self._batch_size,
                show_progress_bar=False,
            )
        except Exception as exc:
            raise RetrievalError(
                f"Reranking failed: {exc}",
                details={
                    "query_len": len(query),
                    "candidate_count": len(candidates),
                },
            ) from exc

        # Build scored results
        scored_results: list[RerankResult] = []
        for idx, (candidate, score) in enumerate(zip(candidates, scores, strict=True)):
            scored_results.append(
                RerankResult(
                    text=candidate.text,
                    rerank_score=float(score),
                    original_score=candidate.original_score,
                    original_rank=idx,
                    metadata=candidate.metadata,
                )
            )

        # Sort by rerank score (highest first) and take top-k
        scored_results.sort(key=lambda r: r.rerank_score, reverse=True)
        top_results = scored_results[:top_k]

        elapsed_ms = (time.perf_counter() - start) * 1000
        log.info(
            "rerank_complete",
            query_len=len(query),
            candidates=len(candidates),
            top_k=top_k,
            returned=len(top_results),
            top_score=round(top_results[0].rerank_score, 4) if top_results else 0.0,
            elapsed_ms=round(elapsed_ms, 1),
        )

        return top_results

    def score_pair(self, query: str, passage: str) -> float:
        """Score a single query-passage pair.

        Useful for ad-hoc relevance scoring.

        Args:
            query: The search query.
            passage: The passage to score.

        Returns:
            Cross-encoder relevance score.

        Raises:
            RetrievalError: If the model is not loaded.
        """
        if not self._loaded or self._model is None:
            raise RetrievalError(
                "Reranker not loaded — call load() first",
                details={"model": self._model_name},
            )

        try:
            score = self._model.predict(
                [[query, passage]],
                show_progress_bar=False,
            )
            return float(score[0])
        except Exception as exc:
            raise RetrievalError(
                f"Pair scoring failed: {exc}",
                details={"query_len": len(query), "passage_len": len(passage)},
            ) from exc
