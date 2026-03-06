"""ARA combined benchmark suite — latency and quality measurements.

Benchmarks:
    - LLM: TTFT, tokens/sec, generation latency
    - Embedding: Throughput (docs/sec), latency per batch
    - Vector DB: Search latency, index size
    - API: Chat endpoint latency, streaming first-token time
    - Memory: Retrieval pipeline latency

Usage:
    uv run python scripts/benchmark.py              # Run all benchmarks
    uv run python scripts/benchmark.py --suite llm   # Run only LLM benchmarks
    uv run python scripts/benchmark.py --suite api    # Run only API benchmarks
    uv run python scripts/benchmark.py --output results.json  # Save results
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from datetime import UTC

# ── Data Types ───────────────────────────────────────────────────────────────


@dataclass
class BenchmarkMetric:
    """A single metric measurement."""

    name: str
    value: float
    unit: str
    samples: list[float] = field(default_factory=list)

    @property
    def mean(self) -> float:
        return statistics.mean(self.samples) if self.samples else self.value

    @property
    def stdev(self) -> float:
        return statistics.stdev(self.samples) if len(self.samples) >= 2 else 0.0

    @property
    def p50(self) -> float:
        return statistics.median(self.samples) if self.samples else self.value

    @property
    def p95(self) -> float:
        if not self.samples:
            return self.value
        sorted_s = sorted(self.samples)
        idx = int(len(sorted_s) * 0.95)
        return sorted_s[min(idx, len(sorted_s) - 1)]


@dataclass
class SuiteResult:
    """Results from a benchmark suite."""

    suite_name: str
    metrics: list[BenchmarkMetric] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    elapsed_s: float = 0.0


@dataclass
class BenchmarkReport:
    """Full benchmark report across all suites."""

    timestamp: str = ""
    suites: list[SuiteResult] = field(default_factory=list)
    total_elapsed_s: float = 0.0

    def to_dict(self) -> dict:
        """Convert to serializable dict."""
        return {
            "timestamp": self.timestamp,
            "total_elapsed_s": self.total_elapsed_s,
            "suites": [
                {
                    "name": s.suite_name,
                    "elapsed_s": s.elapsed_s,
                    "errors": s.errors,
                    "metrics": [
                        {
                            "name": m.name,
                            "value": m.mean,
                            "unit": m.unit,
                            "stdev": m.stdev,
                            "p50": m.p50,
                            "p95": m.p95,
                            "samples": len(m.samples),
                        }
                        for m in s.metrics
                    ],
                }
                for s in self.suites
            ],
        }


# ── Benchmark Suites ────────────────────────────────────────────────────────


async def benchmark_llm(runs: int = 3) -> SuiteResult:
    """Benchmark LLM inference via Ollama."""
    result = SuiteResult(suite_name="llm")
    t0 = time.perf_counter()

    try:
        from ollama import AsyncClient

        client = AsyncClient()

        # Verify connection
        try:
            await client.list()
        except Exception as exc:
            result.errors.append(f"Ollama not reachable: {exc}")
            return result

        prompts = [
            ("short", "Reply in one sentence: What is the speed of light?"),
            ("medium", "Explain quantum entanglement in 3 paragraphs."),
            ("code", "Write a Python function that implements binary search. Include docstring."),
        ]

        ttft_samples: list[float] = []
        tps_samples: list[float] = []
        latency_samples: list[float] = []

        for run_idx in range(runs):
            for name, prompt in prompts:
                gen_start = time.perf_counter()
                first_token_time: float | None = None
                token_count = 0

                try:
                    async for chunk in await client.chat(
                        model="qwen3:30b-a3b",
                        messages=[{"role": "user", "content": prompt}],
                        stream=True,
                    ):
                        if first_token_time is None:
                            first_token_time = time.perf_counter()
                        if chunk.get("message", {}).get("content"):
                            token_count += 1

                    gen_end = time.perf_counter()
                    total_s = gen_end - gen_start

                    if first_token_time is not None:
                        ttft_ms = (first_token_time - gen_start) * 1000
                        ttft_samples.append(ttft_ms)

                    if total_s > 0 and token_count > 0:
                        tps = token_count / total_s
                        tps_samples.append(tps)

                    latency_samples.append(total_s * 1000)

                except Exception as exc:
                    result.errors.append(f"LLM generation failed ({name}, run {run_idx}): {exc}")

        if ttft_samples:
            m = BenchmarkMetric(
                name="llm_ttft",
                value=statistics.mean(ttft_samples),
                unit="ms",
                samples=ttft_samples,
            )
            result.metrics.append(m)

        if tps_samples:
            m = BenchmarkMetric(
                name="llm_tokens_per_sec",
                value=statistics.mean(tps_samples),
                unit="tok/s",
                samples=tps_samples,
            )
            result.metrics.append(m)

        if latency_samples:
            m = BenchmarkMetric(
                name="llm_total_latency",
                value=statistics.mean(latency_samples),
                unit="ms",
                samples=latency_samples,
            )
            result.metrics.append(m)

    except ImportError:
        result.errors.append("ollama package not installed")
    except Exception as exc:
        result.errors.append(f"LLM benchmark error: {exc}")

    result.elapsed_s = time.perf_counter() - t0
    return result


async def benchmark_embedding(runs: int = 3) -> SuiteResult:
    """Benchmark embedding model throughput."""
    result = SuiteResult(suite_name="embedding")
    t0 = time.perf_counter()

    try:
        from ara.config.settings import AraSettings
        from ara.memory.embeddings import EmbeddingModel

        settings = AraSettings()
        model = EmbeddingModel(settings.memory)

        # Test documents of varying lengths
        test_docs = [
            "Hello world, this is a short test document.",
            "Machine learning models require significant computational resources. " * 10,
            "The history of artificial intelligence spans decades of research. " * 25,
        ]

        latency_samples: list[float] = []
        throughput_samples: list[float] = []

        for _ in range(runs):
            # Single doc latency
            for doc in test_docs:
                start = time.perf_counter()
                model.embed_documents([doc])
                elapsed_ms = (time.perf_counter() - start) * 1000
                latency_samples.append(elapsed_ms)

            # Batch throughput
            batch = test_docs * 5  # 15 docs
            start = time.perf_counter()
            model.embed_documents(batch)
            elapsed_s = time.perf_counter() - start
            if elapsed_s > 0:
                throughput_samples.append(len(batch) / elapsed_s)

        if latency_samples:
            result.metrics.append(
                BenchmarkMetric(
                    name="embed_latency",
                    value=statistics.mean(latency_samples),
                    unit="ms",
                    samples=latency_samples,
                )
            )

        if throughput_samples:
            result.metrics.append(
                BenchmarkMetric(
                    name="embed_throughput",
                    value=statistics.mean(throughput_samples),
                    unit="docs/s",
                    samples=throughput_samples,
                )
            )

    except ImportError as exc:
        result.errors.append(f"Embedding model not available: {exc}")
    except Exception as exc:
        result.errors.append(f"Embedding benchmark error: {exc}")

    result.elapsed_s = time.perf_counter() - t0
    return result


async def benchmark_vector_search(runs: int = 3) -> SuiteResult:
    """Benchmark vector store search latency."""
    result = SuiteResult(suite_name="vector_search")
    t0 = time.perf_counter()

    try:
        from ara.config.settings import AraSettings
        from ara.memory.embeddings import EmbeddingModel
        from ara.memory.vector_store import VectorStore

        settings = AraSettings()
        vs = VectorStore(settings.memory)
        vs.connect()

        embed = EmbeddingModel(settings.memory)

        queries = [
            "What did I write about my startup project?",
            "Tell me about my feelings last month",
            "programming projects and technical work",
        ]

        search_latency: list[float] = []

        for _ in range(runs):
            for query in queries:
                start = time.perf_counter()
                query_vec = embed.embed_query(query)
                vs.search(query_vec, limit=10)
                elapsed_ms = (time.perf_counter() - start) * 1000
                search_latency.append(elapsed_ms)

        if search_latency:
            result.metrics.append(
                BenchmarkMetric(
                    name="vector_search_latency",
                    value=statistics.mean(search_latency),
                    unit="ms",
                    samples=search_latency,
                )
            )

        # Index size
        try:
            count = vs.count()
            result.metrics.append(
                BenchmarkMetric(name="vector_index_size", value=float(count), unit="chunks")
            )
        except Exception:
            pass

        vs.close()

    except Exception as exc:
        result.errors.append(f"Vector search benchmark error: {exc}")

    result.elapsed_s = time.perf_counter() - t0
    return result


async def benchmark_api(runs: int = 5) -> SuiteResult:
    """Benchmark API endpoint latency (requires running server)."""
    result = SuiteResult(suite_name="api")
    t0 = time.perf_counter()

    try:
        import httpx

        base_url = "http://127.0.0.1:8000"

        async with httpx.AsyncClient(base_url=base_url, timeout=30.0) as client:
            # Health check latency
            health_latency: list[float] = []
            for _ in range(runs):
                start = time.perf_counter()
                resp = await client.get("/health")
                elapsed_ms = (time.perf_counter() - start) * 1000
                if resp.status_code == 200:
                    health_latency.append(elapsed_ms)

            if health_latency:
                result.metrics.append(
                    BenchmarkMetric(
                        name="api_health_latency",
                        value=statistics.mean(health_latency),
                        unit="ms",
                        samples=health_latency,
                    )
                )

            # Chat endpoint latency
            chat_latency: list[float] = []
            for _ in range(min(runs, 3)):
                start = time.perf_counter()
                resp = await client.post(
                    "/api/chat",
                    json={"message": "Hello, benchmark test"},
                )
                elapsed_ms = (time.perf_counter() - start) * 1000
                if resp.status_code == 200:
                    chat_latency.append(elapsed_ms)

            if chat_latency:
                result.metrics.append(
                    BenchmarkMetric(
                        name="api_chat_latency",
                        value=statistics.mean(chat_latency),
                        unit="ms",
                        samples=chat_latency,
                    )
                )

            # Streaming first-token latency
            stream_ttft: list[float] = []
            for _ in range(min(runs, 3)):
                start = time.perf_counter()
                first_token_time: float | None = None
                try:
                    async with client.stream(
                        "POST",
                        "/api/chat/stream",
                        json={"message": "Say hello in one word", "stream": True},
                    ) as resp:
                        async for line in resp.aiter_lines():
                            if line.startswith("data:") and first_token_time is None:
                                data_str = line.removeprefix("data:").strip()
                                if data_str:
                                    chunk = json.loads(data_str)
                                    if chunk.get("type") == "token":
                                        first_token_time = time.perf_counter()
                                        break
                except Exception:
                    continue

                if first_token_time is not None:
                    ttft_ms = (first_token_time - start) * 1000
                    stream_ttft.append(ttft_ms)

            if stream_ttft:
                result.metrics.append(
                    BenchmarkMetric(
                        name="api_stream_ttft",
                        value=statistics.mean(stream_ttft),
                        unit="ms",
                        samples=stream_ttft,
                    )
                )

    except ImportError:
        result.errors.append("httpx not installed")
    except httpx.ConnectError:
        result.errors.append("API server not running at http://127.0.0.1:8000")
    except Exception as exc:
        result.errors.append(f"API benchmark error: {exc}")

    result.elapsed_s = time.perf_counter() - t0
    return result


# ── Report Printing ──────────────────────────────────────────────────────────


def print_report(report: BenchmarkReport) -> None:
    """Print a formatted benchmark report."""
    print(f"\n{'=' * 70}")
    print("  ARA Benchmark Report")
    print(f"  {report.timestamp}")
    print(f"  Total time: {report.total_elapsed_s:.1f}s")
    print(f"{'=' * 70}\n")

    for suite in report.suites:
        header = f"  [{suite.suite_name.upper()}]"
        if suite.errors:
            header += f" ({len(suite.errors)} error(s))"
        print(header)
        print(f"  {'─' * 60}")

        if suite.errors:
            for err in suite.errors:
                print(f"    ERROR: {err}")
            print()
            continue

        if not suite.metrics:
            print("    No metrics collected")
            print()
            continue

        for m in suite.metrics:
            line = f"    {m.name:<30} {m.mean:>10.1f} {m.unit}"
            if m.stdev > 0:
                line += f"  (std={m.stdev:.1f}, p50={m.p50:.1f}, p95={m.p95:.1f})"
            elif m.samples:
                line += f"  (n={len(m.samples)})"
            print(line)

        print(f"    {'─' * 50}")
        print(f"    Suite time: {suite.elapsed_s:.1f}s")
        print()


# ── Main ─────────────────────────────────────────────────────────────────────


SUITE_MAP = {
    "llm": benchmark_llm,
    "embedding": benchmark_embedding,
    "vector_search": benchmark_vector_search,
    "api": benchmark_api,
}


async def run_benchmarks(
    suites: list[str] | None = None,
    runs: int = 3,
) -> BenchmarkReport:
    """Run selected benchmark suites."""
    from datetime import datetime

    report = BenchmarkReport(timestamp=datetime.now(tz=UTC).isoformat())
    t0 = time.perf_counter()

    targets = suites or list(SUITE_MAP.keys())

    for name in targets:
        if name not in SUITE_MAP:
            print(f"  Unknown suite: {name}")
            continue

        print(f"  Running {name} benchmarks...")
        suite_result = await SUITE_MAP[name](runs=runs)
        report.suites.append(suite_result)

    report.total_elapsed_s = time.perf_counter() - t0
    return report


def main() -> int:
    """Run ARA benchmarks."""
    parser = argparse.ArgumentParser(
        description="ARA combined benchmark suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--suite",
        nargs="+",
        choices=list(SUITE_MAP.keys()),
        help="Run specific benchmark suites (default: all)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of runs per benchmark (default: 3)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Save results to JSON file",
    )
    args = parser.parse_args()

    report = asyncio.run(run_benchmarks(suites=args.suite, runs=args.runs))

    print_report(report)

    if args.output:
        args.output.write_text(json.dumps(report.to_dict(), indent=2))
        print(f"  Results saved to {args.output}")

    # Return non-zero if any suite had errors
    has_errors = any(s.errors for s in report.suites)
    return 1 if has_errors else 0


if __name__ == "__main__":
    sys.exit(main())
