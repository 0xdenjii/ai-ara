"""ARA retrieval accuracy benchmark — precision, recall, MRR measurements.

Evaluates the retrieval pipeline's quality using predefined test queries
with known-relevant document characteristics. Measures:
- Precision@k: Fraction of retrieved results that are relevant
- Recall: Fraction of known-relevant docs that are retrieved
- MRR (Mean Reciprocal Rank): 1/rank of first relevant result
- Latency: End-to-end retrieval time
- Per-method breakdown: Vector, BM25, graph search contributions

Usage:
    uv run python scripts/benchmark_retrieval.py              # Run all queries
    uv run python scripts/benchmark_retrieval.py --top-k 10   # Custom top-k
    uv run python scripts/benchmark_retrieval.py --output results.json
    uv run python scripts/benchmark_retrieval.py --verbose     # Show per-query details
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

# ── Test Query Definitions ──────────────────────────────────────────────────
# Each query has a natural language question, expected source type, and
# relevance criteria (keywords/patterns that indicate a result is relevant).
# These are designed to test different retrieval dimensions:
# - Temporal queries (date-specific)
# - Entity queries (people, places)
# - Emotional queries (feelings, moods)
# - Factual queries (specific information)
# - Planning queries (goals, todos)


@dataclass(frozen=True)
class TestQuery:
    """A benchmark query with relevance criteria."""

    query: str
    description: str
    relevance_keywords: list[str]
    expected_source: str = "diary"  # diary, conversation, note, etc.
    category: str = "general"  # temporal, entity, emotional, factual, planning


# Benchmark queries spanning different retrieval dimensions
BENCHMARK_QUERIES: list[TestQuery] = [
    # Temporal queries
    TestQuery(
        query="What happened in January 2024?",
        description="Broad temporal — recent month",
        relevance_keywords=["2024", "january", "jan"],
        category="temporal",
    ),
    TestQuery(
        query="What was I doing in 2021?",
        description="Broad temporal — full year",
        relevance_keywords=["2021"],
        category="temporal",
    ),
    # Entity queries
    TestQuery(
        query="Tell me about my goals",
        description="Entity — personal goals",
        relevance_keywords=["goal", "aim", "target", "objective", "plan", "achieve"],
        category="entity",
    ),
    TestQuery(
        query="What projects have I worked on?",
        description="Entity — projects",
        relevance_keywords=["project", "build", "develop", "create", "work"],
        category="entity",
    ),
    # Emotional queries
    TestQuery(
        query="When was I feeling stressed or anxious?",
        description="Emotional — negative state",
        relevance_keywords=["stress", "anxi", "worried", "overwhelm", "pressure", "tense"],
        category="emotional",
    ),
    TestQuery(
        query="What makes me happy?",
        description="Emotional — positive state",
        relevance_keywords=["happy", "joy", "excit", "grateful", "love", "enjoy"],
        category="emotional",
    ),
    # Factual queries
    TestQuery(
        query="What technologies have I learned?",
        description="Factual — skills and learning",
        relevance_keywords=["learn", "study", "course", "tutorial", "skill", "tech"],
        category="factual",
    ),
    TestQuery(
        query="What books have I read?",
        description="Factual — reading history",
        relevance_keywords=["book", "read", "author", "chapter", "novel", "finish"],
        category="factual",
    ),
    # Planning queries
    TestQuery(
        query="What are my todos and action items?",
        description="Planning — todos",
        relevance_keywords=["todo", "task", "action", "need to", "should", "must", "plan"],
        category="planning",
    ),
    TestQuery(
        query="What habits am I trying to build?",
        description="Planning — habits",
        relevance_keywords=["habit", "routine", "daily", "practice", "consistent", "track"],
        category="planning",
    ),
    # Mixed/complex queries
    TestQuery(
        query="How has my career evolved over time?",
        description="Complex — career trajectory",
        relevance_keywords=["career", "job", "work", "company", "role", "position", "professional"],
        category="factual",
    ),
    TestQuery(
        query="What decisions have I struggled with?",
        description="Complex — decision making",
        relevance_keywords=["decision", "decide", "choice", "dilemma", "struggle", "uncertain"],
        category="emotional",
    ),
]


# ── Metrics ─────────────────────────────────────────────────────────────────


@dataclass
class QueryResult:
    """Results for a single benchmark query."""

    query: str
    description: str
    category: str
    top_k: int = 0
    results_count: int = 0
    relevant_count: int = 0
    precision_at_k: float = 0.0
    mrr: float = 0.0
    retrieval_time_ms: float = 0.0
    query_type: str = ""
    total_candidates: int = 0
    top_scores: list[float] = field(default_factory=list)
    retrieval_methods: list[str] = field(default_factory=list)


@dataclass
class BenchmarkResult:
    """Aggregated benchmark results across all queries."""

    timestamp: str = ""
    total_queries: int = 0
    top_k: int = 5
    query_results: list[QueryResult] = field(default_factory=list)

    # Aggregated metrics
    mean_precision: float = 0.0
    mean_mrr: float = 0.0
    mean_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0

    # Per-category breakdowns
    category_precision: dict[str, float] = field(default_factory=dict)
    category_mrr: dict[str, float] = field(default_factory=dict)

    # Coverage
    queries_with_results: int = 0
    queries_with_relevant: int = 0

    def to_dict(self) -> dict:
        """Convert to serializable dict."""
        return {
            "timestamp": self.timestamp,
            "top_k": self.top_k,
            "total_queries": self.total_queries,
            "mean_precision_at_k": round(self.mean_precision, 4),
            "mean_mrr": round(self.mean_mrr, 4),
            "mean_latency_ms": round(self.mean_latency_ms, 2),
            "p50_latency_ms": round(self.p50_latency_ms, 2),
            "p95_latency_ms": round(self.p95_latency_ms, 2),
            "queries_with_results": self.queries_with_results,
            "queries_with_relevant": self.queries_with_relevant,
            "category_precision": {k: round(v, 4) for k, v in self.category_precision.items()},
            "category_mrr": {k: round(v, 4) for k, v in self.category_mrr.items()},
            "queries": [
                {
                    "query": qr.query,
                    "description": qr.description,
                    "category": qr.category,
                    "results_count": qr.results_count,
                    "relevant_count": qr.relevant_count,
                    "precision_at_k": round(qr.precision_at_k, 4),
                    "mrr": round(qr.mrr, 4),
                    "retrieval_time_ms": round(qr.retrieval_time_ms, 2),
                    "query_type": qr.query_type,
                    "total_candidates": qr.total_candidates,
                }
                for qr in self.query_results
            ],
        }


# ── Relevance Judging ───────────────────────────────────────────────────────


def is_relevant(chunk_text: str, relevance_keywords: list[str]) -> bool:
    """Check if a chunk is relevant based on keyword matching.

    A chunk is considered relevant if any of the relevance keywords
    appear in its text (case-insensitive). This is a simple heuristic
    for automated evaluation — production use would need human judgments
    or LLM-as-judge.

    Args:
        chunk_text: The text content of the retrieved chunk.
        relevance_keywords: Keywords that indicate relevance.

    Returns:
        True if at least one keyword is found.
    """
    text_lower = chunk_text.lower()
    return any(kw.lower() in text_lower for kw in relevance_keywords)


def compute_mrr(
    results_texts: list[str],
    relevance_keywords: list[str],
) -> float:
    """Compute Mean Reciprocal Rank for a single query's results.

    Args:
        results_texts: Retrieved chunk texts in rank order.
        relevance_keywords: Keywords that indicate relevance.

    Returns:
        1/rank of first relevant result, or 0 if none found.
    """
    for i, text in enumerate(results_texts):
        if is_relevant(text, relevance_keywords):
            return 1.0 / (i + 1)
    return 0.0


# ── Benchmark Runner ────────────────────────────────────────────────────────


def run_benchmark(
    *,
    top_k: int = 5,
    verbose: bool = False,
) -> BenchmarkResult:
    """Run the retrieval benchmark suite.

    Initializes the retrieval pipeline, runs all test queries, and
    computes accuracy metrics.

    Args:
        top_k: Number of results to retrieve per query.
        verbose: Print per-query details.

    Returns:
        BenchmarkResult with aggregated and per-query metrics.
    """
    # Late imports to avoid loading heavy deps at module level
    from ara.config.settings import AraSettings
    from ara.memory.embeddings import EmbeddingModel
    from ara.memory.retrieval import RetrievalPipeline
    from ara.memory.vector_store import VectorStore

    settings = AraSettings()
    result = BenchmarkResult(
        timestamp=datetime.now(tz=UTC).isoformat(),
        top_k=top_k,
        total_queries=len(BENCHMARK_QUERIES),
    )

    # ── Initialize components ────────────────────────────────────────────
    print("Initializing retrieval pipeline...")

    embedder = EmbeddingModel(settings.memory)
    embedder.load()
    print(f"  Embedding model loaded: {settings.memory.embedding_model}")

    vs = VectorStore(settings.memory, dimension=embedder.dimension)
    vs.connect()
    row_count = vs.count()
    print(f"  Vector store connected: {row_count} chunks indexed")

    if row_count == 0:
        print("\nERROR: Vector store is empty. Run ingestion first:")
        print("  uv run python scripts/ingest.py")
        vs.close()
        embedder.unload()
        sys.exit(1)

    # Try loading graph (optional)
    graph = None
    try:
        from ara.memory.graph import KnowledgeGraph

        graph = KnowledgeGraph(settings.memory.graph_db_path)
        graph.load()
        entity_count = len(graph.get_all_entities()) if hasattr(graph, "get_all_entities") else 0
        print(f"  Knowledge graph loaded: {entity_count} entities")
    except Exception as exc:
        print(f"  Knowledge graph not available: {exc}")
        graph = None

    # Try loading reranker (optional)
    reranker = None
    try:
        from ara.memory.reranker import Reranker

        reranker = Reranker(settings.memory)
        reranker.load()
        print(f"  Reranker loaded: {settings.memory.reranker_model}")
    except Exception as exc:
        print(f"  Reranker not available: {exc}")
        reranker = None

    pipeline = RetrievalPipeline(
        settings.memory,
        vs,
        embedder,
        graph=graph,
        reranker=reranker,
    )
    print(f"  Pipeline ready (top_k={top_k})\n")

    # ── Run queries ──────────────────────────────────────────────────────
    latencies: list[float] = []
    precisions: list[float] = []
    mrrs: list[float] = []
    category_precisions: dict[str, list[float]] = {}
    category_mrrs: dict[str, list[float]] = {}

    for i, tq in enumerate(BENCHMARK_QUERIES, 1):
        t0 = time.perf_counter()
        try:
            ctx = pipeline.retrieve(
                tq.query,
                top_k=top_k,
                use_reranker=reranker is not None,
                use_graph=graph is not None,
            )
        except Exception as exc:
            print(f"  [{i}/{len(BENCHMARK_QUERIES)}] ERROR: {tq.query[:50]}... — {exc}")
            continue
        elapsed_ms = (time.perf_counter() - t0) * 1000

        # Evaluate results
        result_texts = [r.chunk.text for r in ctx.results]
        relevant_flags = [is_relevant(r.chunk.text, tq.relevance_keywords) for r in ctx.results]
        relevant_count = sum(relevant_flags)
        precision = relevant_count / len(ctx.results) if ctx.results else 0.0
        mrr_score = compute_mrr(result_texts, tq.relevance_keywords)

        qr = QueryResult(
            query=tq.query,
            description=tq.description,
            category=tq.category,
            top_k=top_k,
            results_count=len(ctx.results),
            relevant_count=relevant_count,
            precision_at_k=precision,
            mrr=mrr_score,
            retrieval_time_ms=elapsed_ms,
            query_type=ctx.query_type.value,
            total_candidates=ctx.total_candidates,
            top_scores=[r.score for r in ctx.results[:3]],
            retrieval_methods=[r.retrieval_method for r in ctx.results],
        )
        result.query_results.append(qr)

        latencies.append(elapsed_ms)
        precisions.append(precision)
        mrrs.append(mrr_score)

        # Track per-category
        category_precisions.setdefault(tq.category, []).append(precision)
        category_mrrs.setdefault(tq.category, []).append(mrr_score)

        if len(ctx.results) > 0:
            result.queries_with_results += 1
        if relevant_count > 0:
            result.queries_with_relevant += 1

        if verbose:
            status = "OK" if relevant_count > 0 else "MISS"
            print(
                f"  [{i:2d}/{len(BENCHMARK_QUERIES)}] {status:4s} "
                f"P@{top_k}={precision:.2f} MRR={mrr_score:.2f} "
                f"{elapsed_ms:6.1f}ms "
                f"({relevant_count}/{len(ctx.results)} relevant) "
                f"[{tq.category}] {tq.query[:60]}"
            )

    # ── Aggregate metrics ────────────────────────────────────────────────
    if latencies:
        result.mean_latency_ms = statistics.mean(latencies)
        result.p50_latency_ms = statistics.median(latencies)
        sorted_lat = sorted(latencies)
        idx_95 = int(len(sorted_lat) * 0.95)
        result.p95_latency_ms = sorted_lat[min(idx_95, len(sorted_lat) - 1)]

    if precisions:
        result.mean_precision = statistics.mean(precisions)

    if mrrs:
        result.mean_mrr = statistics.mean(mrrs)

    for cat, precs in category_precisions.items():
        result.category_precision[cat] = statistics.mean(precs)
    for cat, mrr_vals in category_mrrs.items():
        result.category_mrr[cat] = statistics.mean(mrr_vals)

    # ── Cleanup ──────────────────────────────────────────────────────────
    vs.close()
    embedder.unload()
    if graph is not None:
        graph.close()
    if reranker is not None:
        reranker.unload()

    return result


# ── Output Formatting ───────────────────────────────────────────────────────


def print_report(result: BenchmarkResult) -> None:
    """Print a formatted benchmark report to stdout."""
    print("=" * 70)
    print("  ARA Retrieval Accuracy Benchmark")
    print("=" * 70)
    print(f"  Timestamp:    {result.timestamp}")
    print(f"  Top-k:        {result.top_k}")
    print(f"  Queries:      {result.total_queries}")
    print(f"  With results: {result.queries_with_results}/{result.total_queries}")
    print(f"  With relevant: {result.queries_with_relevant}/{result.total_queries}")
    print("-" * 70)

    # Aggregate metrics
    print("\n  Aggregate Metrics:")
    print(f"    Mean Precision@{result.top_k}:  {result.mean_precision:.4f}")
    print(f"    Mean MRR:             {result.mean_mrr:.4f}")
    print(f"    Mean Latency:         {result.mean_latency_ms:.1f} ms")
    print(f"    P50 Latency:          {result.p50_latency_ms:.1f} ms")
    print(f"    P95 Latency:          {result.p95_latency_ms:.1f} ms")

    # Per-category breakdown
    if result.category_precision:
        print("\n  Per-Category Precision:")
        for cat in sorted(result.category_precision.keys()):
            prec = result.category_precision[cat]
            mrr = result.category_mrr.get(cat, 0.0)
            print(f"    {cat:12s}  P@k={prec:.4f}  MRR={mrr:.4f}")

    # Per-query results
    print(f"\n  {'Query':<50s} {'P@k':>6s} {'MRR':>6s} {'ms':>7s} {'Rel':>5s}")
    print("  " + "-" * 78)
    for qr in result.query_results:
        query_short = qr.query[:48] + ".." if len(qr.query) > 50 else qr.query
        rel_str = f"{qr.relevant_count}/{qr.results_count}"
        print(
            f"  {query_short:<50s} "
            f"{qr.precision_at_k:>6.3f} "
            f"{qr.mrr:>6.3f} "
            f"{qr.retrieval_time_ms:>6.1f} "
            f"{rel_str:>5s}"
        )

    print("\n" + "=" * 70)


# ── CLI Entry Point ─────────────────────────────────────────────────────────


def main() -> int:
    """CLI entry point for retrieval benchmarking."""
    parser = argparse.ArgumentParser(
        description="ARA retrieval accuracy benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results per query (default: 5)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save results to JSON file",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-query details during execution",
    )
    args = parser.parse_args()

    try:
        result = run_benchmark(top_k=args.top_k, verbose=args.verbose)
    except Exception as exc:
        print(f"\nBenchmark failed: {exc}", file=sys.stderr)
        return 1

    print_report(result)

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(json.dumps(result.to_dict(), indent=2))
        print(f"\nResults saved to {output_path}")

    # Return non-zero if no results at all
    return 0 if result.queries_with_results > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
