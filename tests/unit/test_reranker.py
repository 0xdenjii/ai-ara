"""Tests for the reranker with pluggable backends (FlashRank + CrossEncoder).

Uses mocked backends to avoid loading real models in unit tests.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ara.memory.reranker import (
    RerankCandidate,
    Reranker,
    RerankResult,
    _CrossEncoderBackend,
    _FlashRankBackend,
)
from ara.utils.errors import RetrievalError

# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture()
def flashrank_settings() -> MagicMock:
    """Mock MemorySettings for FlashRank backend."""
    s = MagicMock()
    s.reranker_backend = "flashrank"
    s.reranker_model = "ms-marco-MiniLM-L-12-v2"
    return s


@pytest.fixture()
def crossencoder_settings() -> MagicMock:
    """Mock MemorySettings for CrossEncoder backend."""
    s = MagicMock()
    s.reranker_backend = "cross-encoder"
    s.reranker_model = "BAAI/bge-reranker-v2-m3"
    return s


@pytest.fixture()
def mock_flashrank_backend() -> _FlashRankBackend:
    """A FlashRank backend with a mocked ranker."""
    backend = _FlashRankBackend()
    mock_ranker = MagicMock()
    # FlashRank returns results as list of dicts with id, text, score
    mock_ranker.rerank = MagicMock(
        side_effect=lambda req: [
            {"id": i, "text": p["text"], "score": 1.0 - i * 0.1} for i, p in enumerate(req.passages)
        ]
    )
    backend._ranker = mock_ranker
    return backend


@pytest.fixture()
def mock_crossencoder_backend() -> _CrossEncoderBackend:
    """A CrossEncoder backend with a mocked model."""
    backend = _CrossEncoderBackend()
    mock_model = MagicMock()
    mock_model.predict = MagicMock(
        side_effect=lambda pairs, **kw: np.array([1.0 - i * 0.1 for i in range(len(pairs))])
    )
    backend._model = mock_model
    return backend


@pytest.fixture()
def loaded_flashrank_reranker(
    flashrank_settings: MagicMock, mock_flashrank_backend: _FlashRankBackend
) -> Reranker:
    """A reranker with FlashRank backend mocked and loaded."""
    reranker = Reranker(flashrank_settings)
    reranker._backend = mock_flashrank_backend
    reranker._loaded = True
    return reranker


@pytest.fixture()
def loaded_crossencoder_reranker(
    crossencoder_settings: MagicMock, mock_crossencoder_backend: _CrossEncoderBackend
) -> Reranker:
    """A reranker with CrossEncoder backend mocked and loaded."""
    reranker = Reranker(crossencoder_settings)
    reranker._backend = mock_crossencoder_backend
    reranker._loaded = True
    return reranker


def _make_candidates(n: int) -> list[RerankCandidate]:
    """Create n test candidates."""
    return [
        RerankCandidate(
            text=f"passage {i}",
            original_score=1.0 - i * 0.05,
            metadata={"chunk_id": f"chunk_{i}"},
        )
        for i in range(n)
    ]


# ── Data Type Tests ──────────────────────────────────────────────────────────


class TestDataTypes:
    """Tests for RerankCandidate and RerankResult frozen dataclasses."""

    def test_candidate_defaults(self) -> None:
        c = RerankCandidate(text="hello")
        assert c.text == "hello"
        assert c.original_score == 0.0
        assert c.metadata is None

    def test_candidate_frozen(self) -> None:
        c = RerankCandidate(text="x")
        with pytest.raises(AttributeError):
            c.text = "y"  # type: ignore[misc]

    def test_result_defaults(self) -> None:
        r = RerankResult(text="hello", rerank_score=0.95)
        assert r.text == "hello"
        assert r.rerank_score == 0.95
        assert r.original_score == 0.0
        assert r.original_rank == 0
        assert r.metadata is None

    def test_result_frozen(self) -> None:
        r = RerankResult(text="x", rerank_score=0.5)
        with pytest.raises(AttributeError):
            r.rerank_score = 0.9  # type: ignore[misc]


# ── Lifecycle Tests ──────────────────────────────────────────────────────────


class TestLifecycle:
    """Tests for load/unload lifecycle."""

    def test_not_loaded_initially(self, flashrank_settings: MagicMock) -> None:
        r = Reranker(flashrank_settings)
        assert not r.is_loaded

    def test_model_name(self, flashrank_settings: MagicMock) -> None:
        r = Reranker(flashrank_settings)
        assert r.model_name == "ms-marco-MiniLM-L-12-v2"

    def test_backend_type_flashrank(self, flashrank_settings: MagicMock) -> None:
        r = Reranker(flashrank_settings)
        assert r.backend_type == "flashrank"

    def test_backend_type_crossencoder(self, crossencoder_settings: MagicMock) -> None:
        r = Reranker(crossencoder_settings)
        assert r.backend_type == "cross-encoder"

    @patch("ara.memory.reranker.Ranker", create=True)
    def test_load_flashrank_success(
        self, mock_ranker_cls: MagicMock, flashrank_settings: MagicMock
    ) -> None:
        mock_ranker_cls.return_value = MagicMock()
        with patch.dict(
            "sys.modules",
            {"flashrank": MagicMock(Ranker=mock_ranker_cls, RerankRequest=MagicMock())},
        ):
            r = Reranker(flashrank_settings)
            r.load()
            assert r.is_loaded

    @patch("ara.memory.reranker.CrossEncoder", create=True)
    def test_load_crossencoder_success(
        self, mock_ce_cls: MagicMock, crossencoder_settings: MagicMock
    ) -> None:
        mock_ce_cls.return_value = MagicMock()
        with patch.dict(
            "sys.modules", {"sentence_transformers": MagicMock(CrossEncoder=mock_ce_cls)}
        ):
            r = Reranker(crossencoder_settings)
            r.load()
            assert r.is_loaded

    def test_double_load_noop(self, loaded_flashrank_reranker: Reranker) -> None:
        assert loaded_flashrank_reranker.is_loaded
        loaded_flashrank_reranker.load()  # Should not raise
        assert loaded_flashrank_reranker.is_loaded

    def test_unload_flashrank(self, loaded_flashrank_reranker: Reranker) -> None:
        loaded_flashrank_reranker.unload()
        assert not loaded_flashrank_reranker.is_loaded

    def test_unload_crossencoder(self, loaded_crossencoder_reranker: Reranker) -> None:
        loaded_crossencoder_reranker.unload()
        assert not loaded_crossencoder_reranker.is_loaded

    def test_unload_when_not_loaded(self, flashrank_settings: MagicMock) -> None:
        r = Reranker(flashrank_settings)
        r.unload()  # Should not raise
        assert not r.is_loaded

    def test_unknown_backend_raises(self) -> None:
        s = MagicMock()
        s.reranker_backend = "nonexistent"
        s.reranker_model = "some-model"
        r = Reranker(s)
        with pytest.raises(RetrievalError, match="Unknown reranker backend"):
            r.load()


# ── FlashRank Rerank Tests ───────────────────────────────────────────────────


class TestFlashRankRerank:
    """Tests for reranking with FlashRank backend."""

    def test_rerank_not_loaded_raises(self, flashrank_settings: MagicMock) -> None:
        r = Reranker(flashrank_settings)
        with pytest.raises(RetrievalError, match="not loaded"):
            r.rerank("query", _make_candidates(3))

    def test_empty_candidates_returns_empty(self, loaded_flashrank_reranker: Reranker) -> None:
        result = loaded_flashrank_reranker.rerank("query", [])
        assert result == []

    def test_empty_query_returns_empty(self, loaded_flashrank_reranker: Reranker) -> None:
        result = loaded_flashrank_reranker.rerank("  ", _make_candidates(3))
        assert result == []

    def test_rerank_returns_top_k(self, loaded_flashrank_reranker: Reranker) -> None:
        candidates = _make_candidates(10)
        results = loaded_flashrank_reranker.rerank("test query", candidates, top_k=3)
        assert len(results) == 3

    def test_rerank_sorted_by_score(self, loaded_flashrank_reranker: Reranker) -> None:
        candidates = _make_candidates(5)
        results = loaded_flashrank_reranker.rerank("test query", candidates, top_k=5)
        for i in range(len(results) - 1):
            assert results[i].rerank_score >= results[i + 1].rerank_score

    def test_rerank_preserves_metadata(self, loaded_flashrank_reranker: Reranker) -> None:
        candidates = [
            RerankCandidate(
                text="passage A",
                original_score=0.9,
                metadata={"chunk_id": "abc", "key": "value"},
            ),
        ]
        results = loaded_flashrank_reranker.rerank("query", candidates, top_k=1)
        assert len(results) == 1
        assert results[0].metadata is not None
        assert results[0].metadata["chunk_id"] == "abc"
        assert results[0].metadata["key"] == "value"

    def test_rerank_preserves_original_score(self, loaded_flashrank_reranker: Reranker) -> None:
        candidates = [
            RerankCandidate(text="hello", original_score=0.77),
        ]
        results = loaded_flashrank_reranker.rerank("query", candidates, top_k=1)
        assert results[0].original_score == 0.77

    def test_rerank_sets_original_rank(self, loaded_flashrank_reranker: Reranker) -> None:
        candidates = _make_candidates(3)
        results = loaded_flashrank_reranker.rerank("query", candidates, top_k=3)
        original_ranks = {r.original_rank for r in results}
        assert original_ranks == {0, 1, 2}

    def test_rerank_top_k_larger_than_candidates(self, loaded_flashrank_reranker: Reranker) -> None:
        candidates = _make_candidates(2)
        results = loaded_flashrank_reranker.rerank("query", candidates, top_k=10)
        assert len(results) == 2

    def test_rerank_predict_failure_raises(self, loaded_flashrank_reranker: Reranker) -> None:
        loaded_flashrank_reranker._backend._ranker.rerank.side_effect = RuntimeError("ONNX error")  # type: ignore[union-attr]
        with pytest.raises(RetrievalError, match="Reranking failed"):
            loaded_flashrank_reranker.rerank("query", _make_candidates(3))


# ── CrossEncoder Rerank Tests ────────────────────────────────────────────────


class TestCrossEncoderRerank:
    """Tests for reranking with CrossEncoder backend."""

    def test_rerank_not_loaded_raises(self, crossencoder_settings: MagicMock) -> None:
        r = Reranker(crossencoder_settings)
        with pytest.raises(RetrievalError, match="not loaded"):
            r.rerank("query", _make_candidates(3))

    def test_empty_candidates_returns_empty(self, loaded_crossencoder_reranker: Reranker) -> None:
        result = loaded_crossencoder_reranker.rerank("query", [])
        assert result == []

    def test_rerank_returns_top_k(self, loaded_crossencoder_reranker: Reranker) -> None:
        candidates = _make_candidates(10)
        results = loaded_crossencoder_reranker.rerank("test query", candidates, top_k=3)
        assert len(results) == 3

    def test_rerank_sorted_by_score(self, loaded_crossencoder_reranker: Reranker) -> None:
        candidates = _make_candidates(5)
        results = loaded_crossencoder_reranker.rerank("test query", candidates, top_k=5)
        for i in range(len(results) - 1):
            assert results[i].rerank_score >= results[i + 1].rerank_score

    def test_rerank_preserves_metadata(self, loaded_crossencoder_reranker: Reranker) -> None:
        candidates = [
            RerankCandidate(
                text="passage A",
                original_score=0.9,
                metadata={"chunk_id": "abc", "key": "value"},
            ),
        ]
        results = loaded_crossencoder_reranker.rerank("query", candidates, top_k=1)
        assert len(results) == 1
        assert results[0].metadata is not None
        assert results[0].metadata["chunk_id"] == "abc"

    def test_rerank_predict_called_with_pairs(
        self,
        loaded_crossencoder_reranker: Reranker,
        mock_crossencoder_backend: _CrossEncoderBackend,
    ) -> None:
        candidates = _make_candidates(2)
        loaded_crossencoder_reranker.rerank("test query", candidates, top_k=2)
        call_args = mock_crossencoder_backend._model.predict.call_args
        pairs = call_args[0][0]
        assert len(pairs) == 2
        assert pairs[0] == ["test query", "passage 0"]
        assert pairs[1] == ["test query", "passage 1"]

    def test_rerank_predict_failure_raises(
        self,
        loaded_crossencoder_reranker: Reranker,
        mock_crossencoder_backend: _CrossEncoderBackend,
    ) -> None:
        mock_crossencoder_backend._model.predict.side_effect = RuntimeError("GPU OOM")
        with pytest.raises(RetrievalError, match="Reranking failed"):
            loaded_crossencoder_reranker.rerank("query", _make_candidates(3))


# ── Score Pair Tests ─────────────────────────────────────────────────────────


class TestScorePair:
    """Tests for the score_pair method."""

    def test_score_pair_not_loaded_raises(self, flashrank_settings: MagicMock) -> None:
        r = Reranker(flashrank_settings)
        with pytest.raises(RetrievalError, match="not loaded"):
            r.score_pair("query", "passage")

    def test_score_pair_flashrank_returns_float(self, loaded_flashrank_reranker: Reranker) -> None:
        score = loaded_flashrank_reranker.score_pair("test query", "test passage")
        assert isinstance(score, float)

    def test_score_pair_crossencoder_returns_float(
        self,
        loaded_crossencoder_reranker: Reranker,
        mock_crossencoder_backend: _CrossEncoderBackend,
    ) -> None:
        mock_crossencoder_backend._model.predict.side_effect = None
        mock_crossencoder_backend._model.predict.return_value = np.array([0.85])
        score = loaded_crossencoder_reranker.score_pair("test query", "test passage")
        assert isinstance(score, float)
        assert score == pytest.approx(0.85)

    def test_score_pair_crossencoder_passes_correct_format(
        self,
        loaded_crossencoder_reranker: Reranker,
        mock_crossencoder_backend: _CrossEncoderBackend,
    ) -> None:
        mock_crossencoder_backend._model.predict.side_effect = None
        mock_crossencoder_backend._model.predict.return_value = np.array([0.5])
        loaded_crossencoder_reranker.score_pair("q", "p")
        call_args = mock_crossencoder_backend._model.predict.call_args
        assert call_args[0][0] == [["q", "p"]]
