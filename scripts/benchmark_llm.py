"""Benchmark LLM inference: TTFT, generation speed, and throughput.

Runs a series of prompts against the active Ollama model and reports
Time-To-First-Token (TTFT), tokens/sec, and total latency.

Usage:
    uv run python scripts/benchmark_llm.py
    uv run python scripts/benchmark_llm.py --model qwen3:30b-a3b --runs 3
    uv run python scripts/benchmark_llm.py --no-think  # Disable thinking mode
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
import time
from dataclasses import dataclass, field

from ollama import AsyncClient


@dataclass(frozen=True)
class BenchmarkPrompt:
    """A benchmark test case."""

    name: str
    system: str
    user: str
    expected_min_tokens: int


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""

    prompt_name: str
    ttft_ms: float
    total_time_s: float
    prompt_tokens: int
    generated_tokens: int
    tokens_per_sec: float
    prompt_eval_rate: float


@dataclass
class AggregateResult:
    """Aggregated results across multiple runs."""

    prompt_name: str
    runs: list[BenchmarkResult] = field(default_factory=list)

    @property
    def avg_ttft_ms(self) -> float:
        return statistics.mean(r.ttft_ms for r in self.runs)

    @property
    def min_ttft_ms(self) -> float:
        return min(r.ttft_ms for r in self.runs)

    @property
    def avg_tokens_per_sec(self) -> float:
        return statistics.mean(r.tokens_per_sec for r in self.runs)

    @property
    def max_tokens_per_sec(self) -> float:
        return max(r.tokens_per_sec for r in self.runs)

    @property
    def avg_total_time_s(self) -> float:
        return statistics.mean(r.total_time_s for r in self.runs)

    @property
    def avg_generated_tokens(self) -> float:
        return statistics.mean(r.generated_tokens for r in self.runs)


# Benchmark prompts of varying complexity
PROMPTS: list[BenchmarkPrompt] = [
    BenchmarkPrompt(
        name="short_greeting",
        system="You are a helpful assistant. Respond concisely.",
        user="Hello, how are you?",
        expected_min_tokens=10,
    ),
    BenchmarkPrompt(
        name="medium_explanation",
        system="You are a helpful assistant. Respond concisely in 2-3 sentences.",
        user="Explain what a Mixture of Experts model is and why it's efficient.",
        expected_min_tokens=30,
    ),
    BenchmarkPrompt(
        name="longer_creative",
        system="You are a witty assistant. Write exactly 4 sentences.",
        user="Write a short paragraph about why AI assistants should have personality.",
        expected_min_tokens=40,
    ),
    BenchmarkPrompt(
        name="coding_task",
        system="You are a senior Python developer. Write clean, typed code.",
        user="Write a Python function that finds the longest palindromic substring in a string. Include type hints.",
        expected_min_tokens=50,
    ),
]

NO_THINK_SUFFIX = " /no_think"


async def run_single_benchmark(
    client: AsyncClient,
    model: str,
    prompt: BenchmarkPrompt,
    *,
    disable_thinking: bool = False,
    max_tokens: int = 512,
) -> BenchmarkResult:
    """Run a single benchmark and measure timings."""
    user_content = prompt.user
    if disable_thinking:
        user_content += NO_THINK_SUFFIX

    messages = [
        {"role": "system", "content": prompt.system},
        {"role": "user", "content": user_content},
    ]

    start_time = time.perf_counter()
    first_token_time: float | None = None
    generated_tokens = 0
    prompt_tokens = 0
    prompt_eval_rate = 0.0

    stream = await client.chat(
        model=model,
        messages=messages,
        stream=True,
        options={
            "temperature": 0.8,
            "num_predict": max_tokens,
        },
    )

    async for chunk in stream:
        token = chunk.message.content
        if token and first_token_time is None:
            first_token_time = time.perf_counter()

        if token:
            generated_tokens += 1

        if chunk.done:
            # Ollama provides eval stats in the final chunk
            prompt_tokens = getattr(chunk, "prompt_eval_count", 0) or 0
            prompt_eval_dur_ns = getattr(chunk, "prompt_eval_duration", 0) or 0
            if prompt_eval_dur_ns > 0 and prompt_tokens > 0:
                prompt_eval_rate = prompt_tokens / (prompt_eval_dur_ns / 1e9)

    end_time = time.perf_counter()

    if first_token_time is None:
        first_token_time = end_time

    ttft_ms = (first_token_time - start_time) * 1000
    total_time_s = end_time - start_time
    gen_time_s = end_time - first_token_time

    tokens_per_sec = generated_tokens / gen_time_s if gen_time_s > 0 else 0.0

    return BenchmarkResult(
        prompt_name=prompt.name,
        ttft_ms=round(ttft_ms, 1),
        total_time_s=round(total_time_s, 2),
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        tokens_per_sec=round(tokens_per_sec, 1),
        prompt_eval_rate=round(prompt_eval_rate, 1),
    )


async def warmup(client: AsyncClient, model: str) -> None:
    """Ensure model is loaded in VRAM before benchmarking."""
    print(f"  Warming up model '{model}'...")
    stream = await client.chat(
        model=model,
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        options={"num_predict": 1},
    )
    async for chunk in stream:
        if chunk.done:
            break
    print("  Warmup complete.\n")


def print_results(
    aggregates: list[AggregateResult],
    *,
    model: str,
    thinking_disabled: bool,
) -> None:
    """Print benchmark results in a formatted table."""
    think_status = "DISABLED (/no_think)" if thinking_disabled else "ENABLED (default)"

    print("\n" + "=" * 80)
    print(f"  BENCHMARK RESULTS — {model}")
    print(f"  Thinking mode: {think_status}")
    print("=" * 80)
    print(
        f"  {'Prompt':<22} {'Avg TTFT':>10} {'Min TTFT':>10} {'Avg t/s':>10} "
        f"{'Max t/s':>10} {'Avg Tokens':>12} {'Avg Total':>10}"
    )
    print("-" * 80)

    for agg in aggregates:
        print(
            f"  {agg.prompt_name:<22} {agg.avg_ttft_ms:>8.1f}ms {agg.min_ttft_ms:>8.1f}ms "
            f"{agg.avg_tokens_per_sec:>10.1f} {agg.max_tokens_per_sec:>10.1f} "
            f"{agg.avg_generated_tokens:>12.0f} {agg.avg_total_time_s:>8.2f}s"
        )

    print("-" * 80)

    # Overall summary
    all_ttfts = [r.ttft_ms for agg in aggregates for r in agg.runs]
    all_tps = [r.tokens_per_sec for agg in aggregates for r in agg.runs]
    print(
        f"  {'OVERALL':<22} {statistics.mean(all_ttfts):>8.1f}ms {min(all_ttfts):>8.1f}ms "
        f"{statistics.mean(all_tps):>10.1f} {max(all_tps):>10.1f}"
    )
    print("=" * 80)


async def main() -> None:
    """Run the benchmark suite."""
    parser = argparse.ArgumentParser(description="Benchmark LLM inference performance")
    parser.add_argument("--model", default="qwen3:30b-a3b", help="Ollama model name")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs per prompt")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max tokens per generation")
    parser.add_argument("--no-think", action="store_true", help="Disable Qwen3 thinking mode")
    parser.add_argument("--host", default="http://localhost:11434", help="Ollama API base URL")
    args = parser.parse_args()

    client = AsyncClient(host=args.host)

    print(f"\nLLM Benchmark — {args.model}")
    print(f"  Runs per prompt: {args.runs}")
    print(f"  Max tokens: {args.max_tokens}")
    print(f"  Thinking: {'disabled' if args.no_think else 'enabled'}")
    print()

    # Warmup
    await warmup(client, args.model)

    aggregates: list[AggregateResult] = []

    for prompt in PROMPTS:
        agg = AggregateResult(prompt_name=prompt.name)

        for run_idx in range(args.runs):
            print(f"  [{prompt.name}] Run {run_idx + 1}/{args.runs}...", end=" ", flush=True)

            result = await run_single_benchmark(
                client,
                args.model,
                prompt,
                disable_thinking=args.no_think,
                max_tokens=args.max_tokens,
            )

            agg.runs.append(result)
            print(
                f"TTFT={result.ttft_ms:.0f}ms, "
                f"{result.tokens_per_sec:.1f} t/s, "
                f"{result.generated_tokens} tokens, "
                f"{result.total_time_s:.2f}s total"
            )

        aggregates.append(agg)
        print()

    print_results(
        aggregates,
        model=args.model,
        thinking_disabled=args.no_think,
    )


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
