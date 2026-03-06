"""Tests for the hybrid retrieval pipeline.

All external dependencies (VectorStore, EmbeddingModel, KnowledgeGraph,
Reranker) are mocked. Tests cover query analysis, query rewriting,
RRF fusion, context assembly, and the full retrieval orchestrator.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from ara.memory.retrieval import (
    QueryAnalysis,
    RetrievalPipeline,
    _build_results,
    _dict_to_chunk,
    _extract_date_range,
    _reciprocal_rank_fusion,
    _rewrite_query,
    _ScoredChunk,
)
from ara.utils.types import (
    IntentType,
    MemoryChunk,
    MemorySource,
    RetrievalContext,
    RetrievalResult,
)

# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_chunk(
    chunk_id: str = "c1",
    text: str = "some text",
    source: MemorySource = MemorySource.DIARY,
    source_file: str = "diary.md",
    date: datetime | None = None,
    section_title: str = "",
) -> MemoryChunk:
    return MemoryChunk(
        chunk_id=chunk_id,
        text=text,
        source=source,
        source_file=source_file,
        date=date,
        section_title=section_title,
    )


def _make_scored(
    chunk_id: str = "c1",
    text: str = "some text",
    score: float = 0.9,
    method: str = "vector",
) -> _ScoredChunk:
    return _ScoredChunk(
        chunk=_make_chunk(chunk_id=chunk_id, text=text),
        score=score,
        method=method,
    )


def _mock_settings() -> MagicMock:
    s = MagicMock()
    s.retrieval_top_k = 20
    s.rerank_top_k = 5
    s.reranker_model = "BAAI/bge-reranker-v2-m3"
    return s


def _mock_vector_store(
    search_results: list[dict] | None = None,
    fts_results: list[dict] | None = None,
) -> MagicMock:
    vs = MagicMock()
    vs.search.return_value = search_results or []
    vs.full_text_search.return_value = fts_results or []
    return vs


def _mock_embedder(vector: list[float] | None = None) -> MagicMock:
    m = MagicMock()
    m.embed_query.return_value = vector or [0.1] * 768
    return m


def _mock_graph(
    entity: MagicMock | None = None,
    neighbors: list[MagicMock] | None = None,
    context: str = "",
) -> MagicMock:
    g = MagicMock()
    g.get_entity_by_name.return_value = entity
    g.get_neighbors.return_value = neighbors or []
    g.get_entity_context.return_value = context
    return g


def _make_entity_mock(
    name: str = "Mom",
    entity_id: str = "e1",
    confidence: float = 0.9,
) -> MagicMock:
    e = MagicMock()
    e.name = name
    e.entity_id = entity_id
    e.confidence = confidence
    e.entity_type = MagicMock()
    e.entity_type.value = "person"
    return e


def _raw_search_result(
    chunk_id: str = "c1",
    text: str = "result text",
    distance: float = 0.2,
    source: str = "diary",
    source_file: str = "diary.md",
    date: str = "",
) -> dict:
    return {
        "chunk_id": chunk_id,
        "text": text,
        "_distance": distance,
        "source": source,
        "source_file": source_file,
        "date": date,
        "section_title": "",
        "contextual_header": "",
        "token_count": 50,
        "chunk_index": 0,
    }


# ── Query Analysis Tests ─────────────────────────────────────────────────────


class TestQueryAnalysis:
    """Tests for query type classification."""

    def _make_pipeline(self, graph: MagicMock | None = None) -> RetrievalPipeline:
        return RetrievalPipeline(
            _mock_settings(),
            _mock_vector_store(),
            _mock_embedder(),
            graph=graph,
        )

    def test_temporal_query(self) -> None:
        p = self._make_pipeline()
        a = p.analyze_query("what did I do last week?")
        assert a.query_type == IntentType.REMINISCING

    def test_emotional_query(self) -> None:
        p = self._make_pipeline()
        a = p.analyze_query("when did I feel happy and excited?")
        assert a.query_type in (IntentType.EMOTIONAL, IntentType.REMINISCING)

    def test_planning_query(self) -> None:
        p = self._make_pipeline()
        a = p.analyze_query("what are my goals and plans for the future?")
        assert a.query_type == IntentType.PLANNING

    def test_factual_query(self) -> None:
        p = self._make_pipeline()
        a = p.analyze_query("who is the person I met at the conference?")
        assert a.query_type == IntentType.FACTUAL

    def test_general_query(self) -> None:
        p = self._make_pipeline()
        a = p.analyze_query("hello there")
        assert a.query_type == IntentType.GENERAL

    def test_original_query_preserved(self) -> None:
        p = self._make_pipeline()
        a = p.analyze_query("test query 123")
        assert a.original_query == "test query 123"

    def test_confidence_between_0_and_1(self) -> None:
        p = self._make_pipeline()
        a = p.analyze_query("when did I feel sad last month?")
        assert 0.0 <= a.confidence <= 1.0

    def test_entity_extraction_with_graph(self) -> None:
        entity = _make_entity_mock(name="Mom")
        graph = _mock_graph(entity=entity)
        p = self._make_pipeline(graph=graph)
        a = p.analyze_query("what did Mom say about cooking?")
        assert "Mom" in a.entity_names

    def test_entity_extraction_without_graph(self) -> None:
        p = self._make_pipeline(graph=None)
        a = p.analyze_query("what did Mom say?")
        assert a.entity_names == []


# ── Date Extraction Tests ────────────────────────────────────────────────────


class TestDateExtraction:
    """Tests for _extract_date_range."""

    def test_yesterday(self) -> None:
        d_from, d_to = _extract_date_range("what happened yesterday?")
        assert d_from is not None
        assert d_to is not None
        # yesterday should be about 1 day ago
        diff = d_to - d_from
        assert diff.days <= 1

    def test_last_week(self) -> None:
        d_from, d_to = _extract_date_range("what did I do last week?")
        assert d_from is not None
        assert d_to is not None
        diff = d_to - d_from
        assert diff.days == 7

    def test_last_month(self) -> None:
        d_from, d_to = _extract_date_range("what happened last month?")
        assert d_from is not None
        diff = (d_to or datetime.now()) - d_from
        assert diff.days == 30

    def test_n_days_ago(self) -> None:
        d_from, d_to = _extract_date_range("what was I doing 5 days ago?")
        assert d_from is not None
        diff = (d_to or datetime.now()) - d_from
        assert diff.days == 5

    def test_recently(self) -> None:
        d_from, d_to = _extract_date_range("what have I been doing recently?")
        assert d_from is not None
        diff = (d_to or datetime.now()) - d_from
        assert diff.days == 14

    def test_no_temporal_reference(self) -> None:
        d_from, d_to = _extract_date_range("what is the meaning of life?")
        assert d_from is None
        assert d_to is None

    def test_this_week(self) -> None:
        d_from, _d_to = _extract_date_range("what happened this week?")
        assert d_from is not None

    def test_last_year(self) -> None:
        d_from, d_to = _extract_date_range("what did I do last year?")
        assert d_from is not None
        diff = (d_to or datetime.now()) - d_from
        assert diff.days == 365


# ── Query Rewriting Tests ────────────────────────────────────────────────────


class TestQueryRewriting:
    """Tests for _rewrite_query."""

    def test_no_date_no_change(self) -> None:
        result = _rewrite_query("hello world")
        assert result == "hello world"

    def test_date_appended(self) -> None:
        d = datetime(2025, 6, 15)
        result = _rewrite_query("what happened?", date_from=d)
        assert "2025-06-15" in result
        assert "around" in result

    def test_date_not_duplicated(self) -> None:
        d = datetime(2025, 6, 15)
        result = _rewrite_query("what happened on 2025-06-15?", date_from=d)
        assert result.count("2025-06-15") == 1


# ── RRF Fusion Tests ────────────────────────────────────────────────────────


class TestReciprocalRankFusion:
    """Tests for _reciprocal_rank_fusion."""

    def test_empty_lists(self) -> None:
        result = _reciprocal_rank_fusion([[], [], []])
        assert result == []

    def test_single_list(self) -> None:
        chunks = [
            _make_scored("c1", score=0.9),
            _make_scored("c2", score=0.8),
        ]
        result = _reciprocal_rank_fusion([chunks])
        assert len(result) == 2

    def test_duplicate_chunks_merged(self) -> None:
        """Same chunk in multiple lists should be merged with boosted score."""
        list_a = [_make_scored("c1", score=0.9, method="vector")]
        list_b = [_make_scored("c1", score=0.8, method="bm25")]
        result = _reciprocal_rank_fusion([list_a, list_b])
        assert len(result) == 1
        assert result[0].method == "hybrid"  # Appeared in multiple lists

    def test_rrf_scores_sum(self) -> None:
        """Chunk appearing in multiple lists should have higher RRF score."""
        shared = _make_scored("shared", score=0.9)
        unique = _make_scored("unique", score=0.95)
        list_a = [shared, unique]
        list_b = [_make_scored("shared", score=0.8)]  # Same chunk_id
        result = _reciprocal_rank_fusion([list_a, list_b])
        # "shared" should rank higher because it appears in both lists
        shared_result = next(r for r in result if r.chunk.chunk_id == "shared")
        unique_result = next(r for r in result if r.chunk.chunk_id == "unique")
        assert shared_result.score > unique_result.score

    def test_rrf_sorted_descending(self) -> None:
        chunks = [
            _make_scored("c1", score=0.5),
            _make_scored("c2", score=0.9),
            _make_scored("c3", score=0.7),
        ]
        result = _reciprocal_rank_fusion([chunks])
        for i in range(len(result) - 1):
            assert result[i].score >= result[i + 1].score

    def test_rrf_three_lists(self) -> None:
        list_a = [_make_scored("c1"), _make_scored("c2")]
        list_b = [_make_scored("c3"), _make_scored("c1")]
        list_c = [_make_scored("c1"), _make_scored("c4")]
        result = _reciprocal_rank_fusion([list_a, list_b, list_c])
        # c1 appears in all three, should be ranked first
        assert result[0].chunk.chunk_id == "c1"


# ── Dict to Chunk Tests ─────────────────────────────────────────────────────


class TestDictToChunk:
    """Tests for _dict_to_chunk conversion."""

    def test_basic_conversion(self) -> None:
        raw = _raw_search_result()
        chunk = _dict_to_chunk(raw)
        assert chunk.chunk_id == "c1"
        assert chunk.text == "result text"
        assert chunk.source == MemorySource.DIARY

    def test_date_parsing(self) -> None:
        raw = _raw_search_result(date="2024-06-15T10:30:00")
        chunk = _dict_to_chunk(raw)
        assert chunk.date is not None
        assert chunk.date.year == 2024

    def test_invalid_date_ignored(self) -> None:
        raw = _raw_search_result(date="not-a-date")
        chunk = _dict_to_chunk(raw)
        assert chunk.date is None

    def test_empty_date_is_none(self) -> None:
        raw = _raw_search_result(date="")
        chunk = _dict_to_chunk(raw)
        assert chunk.date is None

    def test_unknown_source_fallback(self) -> None:
        raw = _raw_search_result(source="unknown_source")
        chunk = _dict_to_chunk(raw)
        assert chunk.source == MemorySource.INGESTED_FILE

    def test_metadata_populated(self) -> None:
        raw = _raw_search_result()
        chunk = _dict_to_chunk(raw)
        assert "token_count" in chunk.metadata
        assert chunk.metadata["token_count"] == 50


# ── Build Results Tests ──────────────────────────────────────────────────────


class TestBuildResults:
    """Tests for _build_results conversion."""

    def test_basic_conversion(self) -> None:
        scored = [_make_scored("c1", score=0.9, method="vector")]
        results = _build_results(scored)
        assert len(results) == 1
        assert results[0].score == 0.9
        assert results[0].retrieval_method == "vector"
        assert results[0].rerank_score is None

    def test_reranked_flag(self) -> None:
        scored = [
            _ScoredChunk(
                chunk=_make_chunk(),
                score=0.95,
                method="reranked",
                rerank_score=0.95,
            )
        ]
        results = _build_results(scored, reranked=True)
        assert results[0].rerank_score == 0.95

    def test_empty_list(self) -> None:
        assert _build_results([]) == []


# ── Context Assembly Tests ───────────────────────────────────────────────────


class TestContextAssembly:
    """Tests for RetrievalPipeline.assemble_context."""

    def test_empty_results(self) -> None:
        assert RetrievalPipeline.assemble_context([]) == ""

    def test_single_result(self) -> None:
        results = [
            RetrievalResult(
                chunk=_make_chunk(text="Hello world", source_file="diary.md"),
                score=0.9,
                retrieval_method="vector",
            ),
        ]
        ctx = RetrievalPipeline.assemble_context(results)
        assert "Hello world" in ctx
        assert "[Memory 1]" in ctx
        assert "diary.md" in ctx

    def test_deduplication_by_chunk_id(self) -> None:
        chunk = _make_chunk(chunk_id="same", text="duplicate")
        results = [
            RetrievalResult(chunk=chunk, score=0.9, retrieval_method="vector"),
            RetrievalResult(chunk=chunk, score=0.8, retrieval_method="bm25"),
        ]
        ctx = RetrievalPipeline.assemble_context(results)
        assert ctx.count("duplicate") == 1

    def test_max_chars_truncation(self) -> None:
        results = [
            RetrievalResult(
                chunk=_make_chunk(chunk_id=f"c{i}", text="x" * 500),
                score=0.9 - i * 0.1,
                retrieval_method="vector",
            )
            for i in range(10)
        ]
        ctx = RetrievalPipeline.assemble_context(results, max_chars=1000)
        assert len(ctx) <= 1100  # Some tolerance for headers

    def test_no_metadata(self) -> None:
        results = [
            RetrievalResult(
                chunk=_make_chunk(text="test"),
                score=0.9,
                retrieval_method="vector",
            ),
        ]
        ctx = RetrievalPipeline.assemble_context(results, include_metadata=False)
        assert "[Memory 1]" in ctx
        assert "Source:" not in ctx

    def test_date_in_metadata(self) -> None:
        results = [
            RetrievalResult(
                chunk=_make_chunk(text="entry", date=datetime(2024, 3, 15)),
                score=0.9,
                retrieval_method="vector",
            ),
        ]
        ctx = RetrievalPipeline.assemble_context(results)
        assert "2024-03-15" in ctx

    def test_multiple_results_numbered(self) -> None:
        results = [
            RetrievalResult(
                chunk=_make_chunk(chunk_id=f"c{i}", text=f"text {i}"),
                score=0.9,
                retrieval_method="vector",
            )
            for i in range(3)
        ]
        ctx = RetrievalPipeline.assemble_context(results)
        assert "[Memory 1]" in ctx
        assert "[Memory 2]" in ctx
        assert "[Memory 3]" in ctx


# ── Pipeline Orchestrator Tests ──────────────────────────────────────────────


class TestRetrievalPipeline:
    """Tests for the full retrieval pipeline."""

    def test_empty_query_returns_empty_context(self) -> None:
        p = RetrievalPipeline(_mock_settings(), _mock_vector_store(), _mock_embedder())
        ctx = p.retrieve("")
        assert ctx.results == []
        assert ctx.query_type == IntentType.GENERAL

    def test_whitespace_query_returns_empty(self) -> None:
        p = RetrievalPipeline(_mock_settings(), _mock_vector_store(), _mock_embedder())
        ctx = p.retrieve("   ")
        assert ctx.results == []

    def test_basic_vector_search(self) -> None:
        vs = _mock_vector_store(search_results=[_raw_search_result("c1", "found by vector", 0.1)])
        p = RetrievalPipeline(_mock_settings(), vs, _mock_embedder())
        ctx = p.retrieve("test query")
        assert len(ctx.results) >= 1
        assert ctx.query == "test query"
        assert ctx.retrieval_time_ms > 0

    def test_bm25_results_included(self) -> None:
        vs = _mock_vector_store(
            fts_results=[
                {
                    "chunk_id": "c2",
                    "text": "found by bm25",
                    "score": 5.0,
                    "source": "diary",
                    "source_file": "notes.md",
                    "date": "",
                    "section_title": "",
                    "contextual_header": "",
                    "token_count": 30,
                    "chunk_index": 0,
                }
            ]
        )
        p = RetrievalPipeline(_mock_settings(), vs, _mock_embedder())
        ctx = p.retrieve("test query")
        assert ctx.total_candidates >= 1

    def test_graph_results_included(self) -> None:
        entity = _make_entity_mock(name="Mom")
        graph = _mock_graph(entity=entity, context="Mom is my mother, very caring.")
        p = RetrievalPipeline(_mock_settings(), _mock_vector_store(), _mock_embedder(), graph=graph)
        ctx = p.retrieve("what did Mom say?")
        # Graph should contribute results since "Mom" is a known entity
        assert ctx.total_candidates >= 1

    def test_graph_disabled(self) -> None:
        entity = _make_entity_mock(name="Mom")
        graph = _mock_graph(entity=entity, context="Mom context")
        p = RetrievalPipeline(_mock_settings(), _mock_vector_store(), _mock_embedder(), graph=graph)
        p.retrieve("what did Mom say?", use_graph=False)
        # Entity extraction in analyze_query still uses graph for name lookup,
        # but _graph_search (get_entity_context) should NOT be called
        graph.get_entity_context.assert_not_called()

    def test_source_filter_applied(self) -> None:
        vs = _mock_vector_store(search_results=[_raw_search_result()])
        p = RetrievalPipeline(_mock_settings(), vs, _mock_embedder())
        p.retrieve("test", source_filter=MemorySource.DIARY)
        # Verify source_filter was passed to vector search
        call_kwargs = vs.search.call_args[1]
        assert call_kwargs["source_filter"] == MemorySource.DIARY

    def test_reranker_applied_when_available(self) -> None:
        vs = _mock_vector_store(
            search_results=[
                _raw_search_result("c1", "passage 1", 0.1),
                _raw_search_result("c2", "passage 2", 0.2),
            ]
        )
        reranker = MagicMock()
        reranker.is_loaded = True
        reranker.rerank.return_value = [
            MagicMock(
                text="passage 2",
                rerank_score=0.95,
                original_score=0.8,
                metadata={
                    "chunk_id": "c2",
                    "method": "vector",
                    "chunk": _make_chunk("c2", "passage 2"),
                },
            )
        ]
        p = RetrievalPipeline(_mock_settings(), vs, _mock_embedder(), reranker=reranker)
        p.retrieve("test query")
        reranker.rerank.assert_called_once()

    def test_reranker_disabled(self) -> None:
        reranker = MagicMock()
        reranker.is_loaded = True
        p = RetrievalPipeline(
            _mock_settings(), _mock_vector_store(), _mock_embedder(), reranker=reranker
        )
        p.retrieve("test query", use_reranker=False)
        reranker.rerank.assert_not_called()

    def test_reranker_fallback_on_error(self) -> None:
        vs = _mock_vector_store(search_results=[_raw_search_result("c1", "text", 0.1)])
        reranker = MagicMock()
        reranker.is_loaded = True
        reranker.rerank.side_effect = RuntimeError("model error")
        p = RetrievalPipeline(_mock_settings(), vs, _mock_embedder(), reranker=reranker)
        # Should not raise — falls back to RRF results
        ctx = p.retrieve("test query")
        assert isinstance(ctx, RetrievalContext)

    def test_vector_search_error_handled(self) -> None:
        vs = _mock_vector_store()
        vs.search.side_effect = RuntimeError("DB error")
        p = RetrievalPipeline(_mock_settings(), vs, _mock_embedder())
        # Should not raise — vector search failure is non-fatal
        ctx = p.retrieve("test query")
        assert isinstance(ctx, RetrievalContext)

    def test_embed_error_handled(self) -> None:
        embedder = _mock_embedder()
        embedder.embed_query.side_effect = RuntimeError("model error")
        p = RetrievalPipeline(_mock_settings(), _mock_vector_store(), embedder)
        ctx = p.retrieve("test query")
        assert isinstance(ctx, RetrievalContext)

    def test_top_k_override(self) -> None:
        vs = _mock_vector_store(
            search_results=[_raw_search_result(f"c{i}", f"text {i}", 0.1 * i) for i in range(10)]
        )
        p = RetrievalPipeline(_mock_settings(), vs, _mock_embedder())
        ctx = p.retrieve("test query", top_k=3)
        assert len(ctx.results) <= 3

    def test_retrieval_context_fields(self) -> None:
        vs = _mock_vector_store(search_results=[_raw_search_result()])
        p = RetrievalPipeline(_mock_settings(), vs, _mock_embedder())
        ctx = p.retrieve("test query")
        assert ctx.query == "test query"
        assert isinstance(ctx.query_type, IntentType)
        assert ctx.retrieval_time_ms >= 0.0
        assert ctx.total_candidates >= 0


# ── QueryAnalysis Type Tests ─────────────────────────────────────────────────


class TestQueryAnalysisType:
    """Tests for the QueryAnalysis dataclass."""

    def test_defaults(self) -> None:
        qa = QueryAnalysis(original_query="test", query_type=IntentType.GENERAL)
        assert qa.original_query == "test"
        assert qa.query_type == IntentType.GENERAL
        assert qa.rewritten_query == ""
        assert qa.date_from is None
        assert qa.date_to is None
        assert qa.entity_names == []
        assert qa.source_filter is None
        assert qa.confidence == 0.8

    def test_frozen(self) -> None:
        qa = QueryAnalysis(original_query="x", query_type=IntentType.FACTUAL)
        with pytest.raises(AttributeError):
            qa.query_type = IntentType.GENERAL  # type: ignore[misc]
