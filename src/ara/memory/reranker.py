"""Reranker supporting FlashRank (fast) and CrossEncoder (legacy) backends.

Takes a query and a list of candidate text passages, scores each
query-passage pair, and returns the top-k results sorted by relevance.

Architecture:
    - **FlashRank** (default): Uses ``flashrank.Ranker`` with lightweight ONNX
      models (~34MB). ~100ms/query on CPU. No PyTorch dependency for scoring.
    - **CrossEncoder** (legacy): Uses ``sentence_transformers.CrossEncoder``
      (e.g. BAAI/bge-reranker-v2-m3). ~16s/query on CPU. More accurate but
      impractically slow for interactive use.

    Backend selection is controlled by ``MemorySettings.reranker_backend``:
        - ``"flashrank"`` → FlashRank (recommended, ~100x faster)
        - ``"cross-encoder"`` → sentence-transformers CrossEncoder

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
from typing import TYPE_CHECKING, Any, Protocol

import structlog

from ara.utils.errors import RetrievalError

if TYPE_CHECKING:
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
    """A reranked passage with relevance score.

    Attributes:
        text: The passage text.
        rerank_score: Relevance score (higher = more relevant).
        original_score: Score from the initial retrieval stage.
        original_rank: Position in the original candidate list (0-indexed).
        metadata: Carried-through metadata from the candidate.
    """

    text: str
    rerank_score: float
    original_score: float = 0.0
    original_rank: int = 0
    metadata: dict[str, Any] | None = None


# ── Backend Protocol ─────────────────────────────────────────────────────────


class _RerankerBackend(Protocol):
    """Internal protocol for reranker backend implementations."""

    def load(self, model_name: str) -> None: ...
    def unload(self) -> None: ...
    def predict(self, query: str, passages: list[str]) -> list[float]: ...
    def predict_pair(self, query: str, passage: str) -> float: ...


# ── FlashRank Backend ────────────────────────────────────────────────────────


class _FlashRankBackend:
    """FlashRank ONNX reranker — ~100ms/query, ~34MB model.

    Uses ``flashrank.Ranker`` which downloads and caches the model
    on first use. Scores are normalized relevance scores.
    """

    def __init__(self) -> None:
        self._ranker: Any = None

    def load(self, model_name: str) -> None:
        try:
            from flashrank import Ranker
        except ImportError as exc:
            raise RetrievalError(
                "flashrank not installed. Run: pip install flashrank",
                details={"package": "flashrank"},
            ) from exc

        self._ranker = Ranker(model_name=model_name)

    def unload(self) -> None:
        self._ranker = None

    def predict(self, query: str, passages: list[str]) -> list[float]:
        if self._ranker is None:
            msg = "FlashRank ranker not loaded"
            raise RetrievalError(msg)

        from flashrank import RerankRequest

        request = RerankRequest(query=query, passages=[{"text": p} for p in passages])
        results = self._ranker.rerank(request)

        # FlashRank returns results sorted by score. We need scores in the
        # ORIGINAL passage order so the caller can map them back correctly.
        # Each result has {"text": ..., "score": ..., "id": original_index}
        score_by_index: dict[int, float] = {}
        for r in results:
            # FlashRank includes "id" as the original position index
            score_by_index[r["id"]] = float(r["score"])

        return [score_by_index.get(i, 0.0) for i in range(len(passages))]

    def predict_pair(self, query: str, passage: str) -> float:
        scores = self.predict(query, [passage])
        return scores[0]


# ── CrossEncoder Backend ─────────────────────────────────────────────────────


class _CrossEncoderBackend:
    """sentence-transformers CrossEncoder — accurate but slow (~16s on CPU).

    Legacy backend kept for comparison. Uses BAAI/bge-reranker-v2-m3 or
    similar cross-encoder models.
    """

    def __init__(self, *, device: str = "cpu", batch_size: int = 32) -> None:
        self._device = device
        self._batch_size = batch_size
        self._model: Any = None

    def load(self, model_name: str) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RetrievalError(
                "sentence-transformers not installed. Run: pip install sentence-transformers",
                details={"package": "sentence-transformers"},
            ) from exc

        self._model = CrossEncoder(
            model_name,
            device=self._device,
            trust_remote_code=True,
        )

    def unload(self) -> None:
        self._model = None

    def predict(self, query: str, passages: list[str]) -> list[float]:
        if self._model is None:
            msg = "CrossEncoder model not loaded"
            raise RetrievalError(msg)

        pairs: list[list[str]] = [[query, p] for p in passages]
        scores = self._model.predict(
            pairs,
            batch_size=self._batch_size,
            show_progress_bar=False,
        )
        return [float(s) for s in scores]

    def predict_pair(self, query: str, passage: str) -> float:
        if self._model is None:
            msg = "CrossEncoder model not loaded"
            raise RetrievalError(msg)

        scores = self._model.predict(
            [[query, passage]],
            show_progress_bar=False,
        )
        return float(scores[0])


# ── Reranker ─────────────────────────────────────────────────────────────────


class Reranker:
    """Reranker with pluggable backend (FlashRank or CrossEncoder).

    Loads the configured backend, scores query-passage pairs, and returns
    the top-k most relevant passages.

    Args:
        settings: Memory settings containing reranker model name and backend type.
        device: Device for CrossEncoder backend. Defaults to "cpu".
        batch_size: Batch size for CrossEncoder backend.
    """

    def __init__(
        self,
        settings: MemorySettings,
        *,
        device: str = "cpu",
        batch_size: int = 32,
    ) -> None:
        self._model_name = settings.reranker_model
        self._backend_type = getattr(settings, "reranker_backend", "flashrank")
        self._device = device
        self._batch_size = batch_size
        self._backend: _FlashRankBackend | _CrossEncoderBackend | None = None
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        """Whether the reranker model is loaded and ready."""
        return self._loaded and self._backend is not None

    @property
    def model_name(self) -> str:
        """The model name/path."""
        return self._model_name

    @property
    def backend_type(self) -> str:
        """The active backend type ('flashrank' or 'cross-encoder')."""
        return self._backend_type

    def load(self) -> None:
        """Load the reranker model into memory.

        Creates the appropriate backend based on ``reranker_backend`` setting
        and loads the model.

        Raises:
            RetrievalError: If the model cannot be loaded.
        """
        if self._loaded:
            return

        start = time.perf_counter()

        try:
            if self._backend_type == "flashrank":
                backend = _FlashRankBackend()
            elif self._backend_type == "cross-encoder":
                backend = _CrossEncoderBackend(
                    device=self._device,
                    batch_size=self._batch_size,
                )
            else:
                raise RetrievalError(
                    f"Unknown reranker backend: '{self._backend_type}'. "
                    "Supported: 'flashrank', 'cross-encoder'.",
                    details={"backend": self._backend_type},
                )

            backend.load(self._model_name)
            self._backend = backend
            self._loaded = True

            elapsed_ms = (time.perf_counter() - start) * 1000
            log.info(
                "reranker_loaded",
                model=self._model_name,
                backend=self._backend_type,
                elapsed_ms=round(elapsed_ms, 1),
            )
        except RetrievalError:
            raise
        except Exception as exc:
            raise RetrievalError(
                f"Failed to load reranker '{self._model_name}' "
                f"(backend={self._backend_type}): {exc}",
                details={
                    "model": self._model_name,
                    "backend": self._backend_type,
                },
            ) from exc

    def unload(self) -> None:
        """Unload the model to free memory."""
        if not self._loaded:
            return

        if self._backend is not None:
            self._backend.unload()
        self._backend = None
        self._loaded = False
        log.info("reranker_unloaded", model=self._model_name, backend=self._backend_type)

    def rerank(
        self,
        query: str,
        candidates: list[RerankCandidate],
        *,
        top_k: int = 5,
    ) -> list[RerankResult]:
        """Rerank candidate passages by relevance to the query.

        Scores each (query, passage) pair with the loaded backend and
        returns the top-k results sorted by score (descending).

        Args:
            query: The search query.
            candidates: List of candidate passages to rerank.
            top_k: Number of top results to return.

        Returns:
            List of RerankResult sorted by rerank_score (highest first).

        Raises:
            RetrievalError: If the model is not loaded.
        """
        if not self._loaded or self._backend is None:
            raise RetrievalError(
                "Reranker not loaded — call load() first",
                details={"model": self._model_name, "backend": self._backend_type},
            )

        if not candidates:
            return []

        if not query.strip():
            log.warning("rerank_empty_query")
            return []

        start = time.perf_counter()

        # Score all passages
        passages = [c.text for c in candidates]
        try:
            scores = self._backend.predict(query, passages)
        except RetrievalError:
            raise
        except Exception as exc:
            raise RetrievalError(
                f"Reranking failed: {exc}",
                details={
                    "query_len": len(query),
                    "candidate_count": len(candidates),
                    "backend": self._backend_type,
                },
            ) from exc

        # Build scored results
        scored_results: list[RerankResult] = []
        for idx, (candidate, score) in enumerate(zip(candidates, scores, strict=True)):
            scored_results.append(
                RerankResult(
                    text=candidate.text,
                    rerank_score=score,
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
            backend=self._backend_type,
        )

        return top_results

    def score_pair(self, query: str, passage: str) -> float:
        """Score a single query-passage pair.

        Useful for ad-hoc relevance scoring.

        Args:
            query: The search query.
            passage: The passage to score.

        Returns:
            Relevance score.

        Raises:
            RetrievalError: If the model is not loaded.
        """
        if not self._loaded or self._backend is None:
            raise RetrievalError(
                "Reranker not loaded — call load() first",
                details={"model": self._model_name, "backend": self._backend_type},
            )

        try:
            return self._backend.predict_pair(query, passage)
        except RetrievalError:
            raise
        except Exception as exc:
            raise RetrievalError(
                f"Pair scoring failed: {exc}",
                details={"query_len": len(query), "passage_len": len(passage)},
            ) from exc
