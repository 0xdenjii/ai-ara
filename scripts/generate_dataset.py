"""CLI script to generate fine-tuning training data for ARA.

Usage:
    # Scan diary files and show stats (with category breakdown)
    uv run python scripts/generate_dataset.py scan --diary-dir dairy/

    # Generate training data using local Ollama (requires qwen3.5:35b loaded)
    uv run python scripts/generate_dataset.py generate --diary-dir dairy/ --target 500

    # Generate using OpenAI-compatible API (GPT-4o-mini, ~$0.50 for 500 samples)
    uv run python scripts/generate_dataset.py generate --diary-dir dairy/ --target 500 \\
        --backend openai --api-key sk-... --model gpt-4o-mini

    # Generate using any OpenAI-compatible endpoint (e.g. local vLLM, Together, etc.)
    uv run python scripts/generate_dataset.py generate --diary-dir dairy/ --target 500 \\
        --backend openai --api-base http://localhost:8000/v1 --model my-model

    # Resume interrupted generation (auto-detects checkpoint)
    uv run python scripts/generate_dataset.py generate --diary-dir dairy/ --target 500

    # Score existing dataset with LLM-as-judge
    uv run python scripts/generate_dataset.py score --input data/fine_tuning/all.json

    # Split dataset into train/val/test
    uv run python scripts/generate_dataset.py split --input data/fine_tuning/scored.json

    # Show dataset stats
    uv run python scripts/generate_dataset.py stats --input data/fine_tuning/train.jsonl

    # Show checkpoint status
    uv run python scripts/generate_dataset.py status
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

# Fix Windows console encoding
sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

# ── Checkpoint management ────────────────────────────────────────────────────

CHECKPOINT_FILENAME = "checkpoint.json"
DEFAULT_OUTPUT_DIR = "data/fine_tuning"


def _checkpoint_path(output_dir: Path) -> Path:
    return output_dir / CHECKPOINT_FILENAME


def _load_checkpoint(output_dir: Path) -> dict[str, Any]:
    """Load checkpoint from disk. Returns empty dict if no checkpoint exists."""
    cp_path = _checkpoint_path(output_dir)
    if cp_path.exists():
        with cp_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_checkpoint(
    output_dir: Path,
    samples: list[dict[str, Any]],
    *,
    phase: str,
    phase_index: int,
    target: int,
    concurrency: int,
    failed: int,
    completed_phases: list[str],
) -> None:
    """Save checkpoint atomically (write to .tmp then rename)."""
    cp_path = _checkpoint_path(output_dir)
    tmp_path = cp_path.with_suffix(".tmp")
    data = {
        "phase": phase,
        "phase_index": phase_index,
        "target": target,
        "concurrency": concurrency,
        "failed": failed,
        "completed_phases": completed_phases,
        "total_samples": len(samples),
        "samples": samples,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    tmp_path.replace(cp_path)


# ── LLM client management ───────────────────────────────────────────────────

_LLM_CLIENT = None
_OPENAI_CLIENT: Any = None
_OPENAI_MODEL: str = ""
_SEMAPHORE: asyncio.Semaphore | None = None
_BACKEND: str = "ollama"  # "ollama" or "openai"


async def _get_llm_client():
    """Get or create a reusable Ollama LLM client."""
    global _LLM_CLIENT
    if _LLM_CLIENT is None:
        from ara.config.settings import AraSettings
        from ara.core.llm import LLMClient

        settings = AraSettings()
        _LLM_CLIENT = LLMClient(settings)
        await _LLM_CLIENT.connect()
    return _LLM_CLIENT


def _init_openai_client(
    api_key: str,
    api_base: str = "https://api.openai.com/v1",
    model: str = "gpt-4o-mini",
) -> None:
    """Initialize the OpenAI-compatible HTTP client.

    Uses httpx directly for minimal dependencies — no openai SDK required.
    Works with any OpenAI-compatible API (OpenAI, Together, vLLM, etc.).
    """
    global _OPENAI_CLIENT, _OPENAI_MODEL
    import httpx

    _OPENAI_CLIENT = httpx.AsyncClient(
        base_url=api_base,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        timeout=httpx.Timeout(120.0, connect=10.0),
    )
    _OPENAI_MODEL = model


def _extract_json_from_response(raw: str) -> str:
    """Extract JSON from an LLM response, handling think blocks and noise.

    The model may wrap JSON in <think>...</think> blocks, markdown fences,
    or prefix it with explanation text. This extracts the first valid
    JSON array or object found anywhere in the response.
    """
    import re

    # First: try to find JSON inside think blocks (model sometimes puts answer there)
    think_pattern = re.compile(r"<think>(.*?)</think>", flags=re.DOTALL)
    think_matches = think_pattern.findall(raw)

    # Also get content outside think blocks
    outside_think = think_pattern.sub("", raw).strip()

    # Candidates: outside-think content first (preferred), then inside-think content
    candidates = [outside_think, *think_matches]

    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate:
            continue

        # Strip markdown code fences
        if candidate.startswith("```"):
            first_nl = candidate.find("\n")
            if first_nl != -1:
                candidate = candidate[first_nl + 1 :]
            if candidate.endswith("```"):
                candidate = candidate[:-3]
            candidate = candidate.strip()

        # Try to find a JSON array [...] (our expected format)
        bracket_start = candidate.find("[")
        bracket_end = candidate.rfind("]")
        if bracket_start != -1 and bracket_end > bracket_start:
            return candidate[bracket_start : bracket_end + 1]

        # Fall back to JSON object {...}
        brace_start = candidate.find("{")
        brace_end = candidate.rfind("}")
        if brace_start != -1 and brace_end > brace_start:
            return candidate[brace_start : brace_end + 1]

    # Last resort: return everything with think blocks stripped
    return outside_think


async def _generate_fn_ollama(prompt: str) -> str:
    """Generate a response using the local Ollama API.

    Uses /no_think to suppress Qwen3.5 reasoning, caps tokens at 1024
    (JSON conversations are short), and robustly extracts JSON from
    any wrapper the model adds.
    """
    from ara.utils.types import ChatMessage, MessageRole

    assert _SEMAPHORE is not None, "Semaphore must be initialized before calling _generate_fn"

    async with _SEMAPHORE:
        client = await _get_llm_client()
        messages = [
            ChatMessage(role=MessageRole.USER, content=prompt + " /no_think"),
        ]
        response = await client.generate(messages, max_tokens=1024, temperature=0.8)
        return _extract_json_from_response(response)


async def _generate_fn_openai(prompt: str) -> str:
    """Generate a response using an OpenAI-compatible API.

    Works with OpenAI, Together, vLLM, or any provider that implements
    the /v1/chat/completions endpoint. Uses httpx directly.
    """
    assert _SEMAPHORE is not None, "Semaphore must be initialized before calling _generate_fn"
    assert _OPENAI_CLIENT is not None, "OpenAI client must be initialized"

    async with _SEMAPHORE:
        payload = {
            "model": _OPENAI_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            "max_tokens": 1024,
            "temperature": 0.8,
        }

        response = await _OPENAI_CLIENT.post("/chat/completions", json=payload)
        response.raise_for_status()
        data = response.json()
        raw = data["choices"][0]["message"]["content"]
        return _extract_json_from_response(raw)


async def _generate_fn(prompt: str) -> str:
    """Route to the active backend (Ollama or OpenAI)."""
    if _BACKEND == "openai":
        return await _generate_fn_openai(prompt)
    return await _generate_fn_ollama(prompt)


# ── Progress display ─────────────────────────────────────────────────────────


class ProgressTracker:
    """Track and display generation progress with ETA."""

    def __init__(self, phase: str, total: int, start_index: int = 0) -> None:
        self.phase = phase
        self.total = total
        self.current = start_index
        self.successes = 0
        self.failures = 0
        self.start_time = time.monotonic()

    def tick(self, *, success: bool) -> None:
        self.current += 1
        if success:
            self.successes += 1
        else:
            self.failures += 1

    def display(self) -> None:
        elapsed = time.monotonic() - self.start_time
        done = self.successes + self.failures
        rate = done / elapsed if elapsed > 0 else 0
        remaining = self.total - self.current
        eta_secs = remaining / rate if rate > 0 else 0
        eta_str = _format_duration(eta_secs)
        pct = (self.current / max(self.total, 1)) * 100

        print(
            f"\r  [{self.phase}] {self.current}/{self.total} "
            f"({pct:.0f}%) | ok={self.successes} fail={self.failures} | "
            f"{rate:.1f} samples/s | ETA: {eta_str}    ",
            end="",
            flush=True,
        )


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


# ── Generation with checkpointing ───────────────────────────────────────────


async def _generate_batch(
    tasks: list[tuple[str, Any]],
    concurrency: int,
) -> list[Any | None]:
    """Run a batch of generation tasks with bounded concurrency.

    Args:
        tasks: List of (task_type, coroutine) tuples.
        concurrency: Max concurrent tasks.

    Returns:
        List of results (TrainingSample or None).
    """
    semaphore = asyncio.Semaphore(concurrency)

    async def _run(coro: Any) -> Any:
        async with semaphore:
            return await coro

    coros = [_run(coro) for _, coro in tasks]
    return await asyncio.gather(*coros, return_exceptions=True)


async def cmd_generate(args: argparse.Namespace) -> None:
    """Generate training data with checkpointing and resume support."""
    from ara.fine_tuning.dataset import (
        GENERAL_TOPICS,
        ConversationType,
        build_weighted_segment_pool,
        generate_general_conversation,
        generate_knowledge_qa,
        generate_personality_conversation,
        get_humor_level_for_index,
        get_segment_category_stats,
        get_target_counts,
        save_dataset,
        scan_diary_directory,
    )

    global _SEMAPHORE, _BACKEND

    diary_dir = Path(args.diary_dir)
    target = args.target
    output_dir = Path(args.output_dir)
    concurrency = args.concurrency
    output_dir.mkdir(parents=True, exist_ok=True)

    _SEMAPHORE = asyncio.Semaphore(concurrency)

    # Configure backend
    _BACKEND = getattr(args, "backend", "ollama")
    if _BACKEND == "openai":
        api_key = getattr(args, "api_key", "") or ""
        api_base = getattr(args, "api_base", "https://api.openai.com/v1")
        model = getattr(args, "model", "gpt-4o-mini")

        if not api_key:
            # Try environment variable
            import os

            api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            print("ERROR: --api-key or OPENAI_API_KEY env var required for openai backend")
            return

        _init_openai_client(api_key, api_base, model)
        print(f"Backend: OpenAI-compatible ({api_base}, model={model})")
        print(f"Concurrency: {concurrency} (higher values recommended for cloud APIs)")
    else:
        print(f"Backend: Ollama (local, concurrency={concurrency})")

    # Scan diary
    print(f"\nScanning diary directory: {diary_dir}")
    segments = scan_diary_directory(diary_dir)
    print(f"Found {len(segments)} segments")

    if not segments:
        print("No segments found. Nothing to generate.")
        return

    # Show category distribution
    cat_stats = get_segment_category_stats(segments)
    print("\nCategory distribution:")
    for cat, count in sorted(cat_stats.items()):
        print(f"  {cat}: {count}")

    counts = get_target_counts(target)
    diary_target = counts[ConversationType.PERSONALITY] + counts[ConversationType.KNOWLEDGE_QA]

    # Build weighted pool for diary-based phases (personality + knowledge_qa)
    weighted_pool = build_weighted_segment_pool(segments, diary_target)
    print(
        f"\nWeighted pool: {len(weighted_pool)} selections for {diary_target} diary-based samples"
    )

    print(f"\nTarget: {target} samples")
    print("Distribution:")
    for ct, count in counts.items():
        print(f"  {ct.value}: {count}")

    # Check for existing checkpoint
    checkpoint = _load_checkpoint(output_dir)
    all_samples: list[dict[str, Any]] = []
    completed_phases: list[str] = []
    failed_total = 0
    resume_phase = ""
    resume_index = 0

    if checkpoint and checkpoint.get("target") == target:
        all_samples = checkpoint.get("samples", [])
        completed_phases = checkpoint.get("completed_phases", [])
        failed_total = checkpoint.get("failed", 0)
        resume_phase = checkpoint.get("phase", "")
        resume_index = checkpoint.get("phase_index", 0)
        print(
            f"\n** Resuming from checkpoint: {len(all_samples)} samples, "
            f"phase={resume_phase}, index={resume_index}, failed={failed_total}"
        )
    elif checkpoint:
        print(
            f"\n** Checkpoint exists but target mismatch "
            f"(checkpoint={checkpoint.get('target')}, requested={target}). Starting fresh."
        )

    # ── Phase 1: Personality conversations ────────────────────────────────
    personality_target = counts[ConversationType.PERSONALITY]
    phase_name = "personality"

    if phase_name not in completed_phases:
        start_idx = resume_index if resume_phase == phase_name else 0
        print(
            f"\nPhase 1: Generating {personality_target} personality conversations "
            f"(starting at {start_idx})..."
        )
        tracker = ProgressTracker(phase_name, personality_target, start_idx)

        for i in range(start_idx, personality_target):
            seg = weighted_pool[i % len(weighted_pool)]
            humor = get_humor_level_for_index(i, personality_target)

            try:
                sample = await generate_personality_conversation(seg, humor, _generate_fn)
                if sample:
                    all_samples.append(sample.to_full_dict())
                    tracker.tick(success=True)
                else:
                    failed_total += 1
                    tracker.tick(success=False)
            except Exception as exc:
                print(f"\n  [ERROR] personality[{i}]: {exc}")
                failed_total += 1
                tracker.tick(success=False)

            tracker.display()

            # Checkpoint every 10 samples
            if (i + 1) % 10 == 0 or i == personality_target - 1:
                _save_checkpoint(
                    output_dir,
                    all_samples,
                    phase=phase_name,
                    phase_index=i + 1,
                    target=target,
                    concurrency=concurrency,
                    failed=failed_total,
                    completed_phases=completed_phases,
                )

        completed_phases.append(phase_name)
        _save_checkpoint(
            output_dir,
            all_samples,
            phase=phase_name,
            phase_index=personality_target,
            target=target,
            concurrency=concurrency,
            failed=failed_total,
            completed_phases=completed_phases,
        )
        print(f"\n  Phase 1 complete: {tracker.successes} ok, {tracker.failures} failed")
    else:
        print("\nPhase 1: personality -- SKIPPED (checkpoint)")

    # ── Phase 2: Knowledge Q&A ───────────────────────────────────────────
    qa_target = counts[ConversationType.KNOWLEDGE_QA]
    phase_name = "knowledge_qa"

    if phase_name not in completed_phases:
        start_idx = resume_index if resume_phase == phase_name else 0
        print(
            f"\nPhase 2: Generating {qa_target} knowledge Q&A conversations "
            f"(starting at {start_idx})..."
        )
        tracker = ProgressTracker(phase_name, qa_target, start_idx)

        for i in range(start_idx, qa_target):
            # Offset into weighted pool so knowledge QA uses different segments than personality
            pool_idx = (personality_target + i) % len(weighted_pool)
            seg = weighted_pool[pool_idx]
            humor = get_humor_level_for_index(i, qa_target)

            try:
                sample = await generate_knowledge_qa(seg, humor, _generate_fn)
                if sample:
                    all_samples.append(sample.to_full_dict())
                    tracker.tick(success=True)
                else:
                    failed_total += 1
                    tracker.tick(success=False)
            except Exception as exc:
                print(f"\n  [ERROR] knowledge_qa[{i}]: {exc}")
                failed_total += 1
                tracker.tick(success=False)

            tracker.display()

            if (i + 1) % 10 == 0 or i == qa_target - 1:
                _save_checkpoint(
                    output_dir,
                    all_samples,
                    phase=phase_name,
                    phase_index=i + 1,
                    target=target,
                    concurrency=concurrency,
                    failed=failed_total,
                    completed_phases=completed_phases,
                )

        completed_phases.append(phase_name)
        _save_checkpoint(
            output_dir,
            all_samples,
            phase=phase_name,
            phase_index=qa_target,
            target=target,
            concurrency=concurrency,
            failed=failed_total,
            completed_phases=completed_phases,
        )
        print(f"\n  Phase 2 complete: {tracker.successes} ok, {tracker.failures} failed")
    else:
        print("\nPhase 2: knowledge_qa -- SKIPPED (checkpoint)")

    # ── Phase 3: General conversations ───────────────────────────────────
    general_target = counts[ConversationType.GENERAL]
    phase_name = "general"

    if phase_name not in completed_phases:
        start_idx = resume_index if resume_phase == phase_name else 0
        print(
            f"\nPhase 3: Generating {general_target} general conversations "
            f"(starting at {start_idx})..."
        )
        tracker = ProgressTracker(phase_name, general_target, start_idx)

        for i in range(start_idx, general_target):
            topic = GENERAL_TOPICS[i % len(GENERAL_TOPICS)]
            humor = get_humor_level_for_index(i, general_target)

            try:
                sample = await generate_general_conversation(topic, humor, _generate_fn)
                if sample:
                    all_samples.append(sample.to_full_dict())
                    tracker.tick(success=True)
                else:
                    failed_total += 1
                    tracker.tick(success=False)
            except Exception as exc:
                print(f"\n  [ERROR] general[{i}]: {exc}")
                failed_total += 1
                tracker.tick(success=False)

            tracker.display()

            if (i + 1) % 10 == 0 or i == general_target - 1:
                _save_checkpoint(
                    output_dir,
                    all_samples,
                    phase=phase_name,
                    phase_index=i + 1,
                    target=target,
                    concurrency=concurrency,
                    failed=failed_total,
                    completed_phases=completed_phases,
                )

        completed_phases.append(phase_name)
        _save_checkpoint(
            output_dir,
            all_samples,
            phase=phase_name,
            phase_index=general_target,
            target=target,
            concurrency=concurrency,
            failed=failed_total,
            completed_phases=completed_phases,
        )
        print(f"\n  Phase 3 complete: {tracker.successes} ok, {tracker.failures} failed")
    else:
        print("\nPhase 3: general -- SKIPPED (checkpoint)")

    # ── Save final dataset ───────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("Generation complete!")
    print(f"  Total samples: {len(all_samples)}")
    print(f"  Failed: {failed_total}")
    print(f"  Success rate: {len(all_samples) / (len(all_samples) + failed_total) * 100:.1f}%")

    # Convert checkpoint dicts back to TrainingSample objects for save_dataset
    from ara.fine_tuning.dataset import (
        ChatMLMessage,
        ConversationType,
        HumorLevel,
        TrainingSample,
        count_conversation_tokens,
    )

    training_samples: list[TrainingSample] = []
    for d in all_samples:
        messages = [ChatMLMessage(role=m["role"], content=m["content"]) for m in d["messages"]]
        training_samples.append(
            TrainingSample(
                sample_id=d["sample_id"],
                conversation_type=ConversationType(d["conversation_type"]),
                humor_level=HumorLevel(d["humor_level"]),
                messages=messages,
                source_segments=d.get("source_segments", []),
                quality_score=float(d.get("quality_score", 0.0)),
                token_count=d.get("token_count", count_conversation_tokens(messages)),
                metadata=d.get("metadata", {}),
            )
        )

    output_path = output_dir / "all.json"
    save_dataset(training_samples, output_path, format="json")
    print(f"  Saved to: {output_path}")

    # Cleanup checkpoint on successful completion
    cp_path = _checkpoint_path(output_dir)
    if cp_path.exists():
        completed_cp = cp_path.with_name("checkpoint_completed.json")
        cp_path.rename(completed_cp)
        print(f"  Checkpoint archived: {completed_cp.name}")

    # Close OpenAI client if used
    if _OPENAI_CLIENT is not None:
        await _OPENAI_CLIENT.aclose()


async def cmd_score(args: argparse.Namespace) -> None:
    """Score existing dataset with LLM-as-judge."""
    from ara.fine_tuning.dataset import (
        compute_stats,
        filter_by_quality,
        load_dataset,
        save_dataset,
        score_sample,
    )

    global _SEMAPHORE, _BACKEND
    _SEMAPHORE = asyncio.Semaphore(args.concurrency if hasattr(args, "concurrency") else 1)

    # Configure backend
    _BACKEND = getattr(args, "backend", "ollama")
    if _BACKEND == "openai":
        import os

        api_key = getattr(args, "api_key", "") or os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            print("ERROR: --api-key or OPENAI_API_KEY required for openai backend")
            return
        _init_openai_client(
            api_key,
            getattr(args, "api_base", "https://api.openai.com/v1"),
            getattr(args, "model", "gpt-4o-mini"),
        )

    input_path = Path(args.input)
    samples = load_dataset(input_path)
    print(f"Loaded {len(samples)} samples from {input_path}")

    unscored = [s for s in samples if s.quality_score <= 0]
    print(f"Unscored: {len(unscored)}, Already scored: {len(samples) - len(unscored)}")

    if unscored:
        print("Scoring with LLM-as-judge...")
        for i, sample in enumerate(unscored):
            score = await score_sample(sample, _generate_fn)
            sample.quality_score = score
            if (i + 1) % 20 == 0:
                print(f"  Scored {i + 1}/{len(unscored)}")

    # Filter
    min_score = args.min_score
    kept, filtered_count = filter_by_quality(samples, min_score=min_score)
    print("\nScoring complete!")
    print(f"  Kept: {len(kept)} (score >= {min_score})")
    print(f"  Filtered: {filtered_count}")

    stats = compute_stats(kept)
    print(f"  Avg quality: {stats.avg_quality_score:.2f}")

    output_path = Path(args.output) if args.output else input_path.parent / "scored.json"
    save_dataset(kept, output_path, format="json")
    print(f"  Saved to: {output_path}")


def cmd_scan(args: argparse.Namespace) -> None:
    """Scan diary directory and show segment stats with category breakdown."""
    from ara.fine_tuning.dataset import (
        CATEGORY_WEIGHTS,
        get_segment_category_stats,
        scan_diary_directory,
    )

    diary_dir = Path(args.diary_dir)
    if not diary_dir.exists():
        print(f"Directory not found: {diary_dir}")
        sys.exit(1)

    segments = scan_diary_directory(diary_dir)
    print(f"\nScanned: {diary_dir}")
    print(f"Total segments: {len(segments)}")

    if segments:
        total_tokens = sum(s.token_count for s in segments)
        avg_tokens = total_tokens / len(segments)
        dated = sum(1 for s in segments if s.date)
        sources = {s.source_file for s in segments}

        print(f"Total tokens: {total_tokens:,}")
        print(f"Avg tokens/segment: {avg_tokens:.0f}")
        print(f"Dated segments: {dated}")
        print(f"Unique source files: {len(sources)}")

        # Category breakdown with weights
        cat_stats = get_segment_category_stats(segments)
        print("\nCategory breakdown (for weighted sampling):")
        total_weighted = 0
        for cat_name in sorted(cat_stats):
            count = cat_stats[cat_name]
            from ara.fine_tuning.dataset import SegmentCategory

            weight = CATEGORY_WEIGHTS.get(SegmentCategory(cat_name), 1)
            weighted = count * weight
            total_weighted += weighted
            print(f"  {cat_name}: {count:>6} segments x {weight}x = {weighted:>6} weighted")
        print(f"  Total weighted pool: {total_weighted:,}")

        print("\nSource files (top 20):")
        src_counts = sorted(
            ((src, sum(1 for s in segments if s.source_file == src)) for src in sources),
            key=lambda x: -x[1],
        )
        for src, count in src_counts[:20]:
            print(f"  {count:>5} | {src}")


def cmd_split(args: argparse.Namespace) -> None:
    """Split dataset into train/val/test."""
    from ara.fine_tuning.dataset import compute_stats, load_dataset, save_dataset, split_dataset

    input_path = Path(args.input)
    samples = load_dataset(input_path)
    print(f"Loaded {len(samples)} samples")

    train, val, test = split_dataset(
        samples,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    output_dir = Path(args.output_dir)
    save_dataset(train, output_dir / "train.jsonl", format="jsonl")
    save_dataset(val, output_dir / "val.jsonl", format="jsonl")
    save_dataset(test, output_dir / "test.jsonl", format="jsonl")

    print("\nSplit complete:")
    print(f"  Train: {len(train)} samples -> {output_dir / 'train.jsonl'}")
    print(f"  Val:   {len(val)} samples -> {output_dir / 'val.jsonl'}")
    print(f"  Test:  {len(test)} samples -> {output_dir / 'test.jsonl'}")

    stats = compute_stats(train)
    print("\nTrain set stats:")
    print(f"  Personality: {stats.personality_count}")
    print(f"  Knowledge QA: {stats.knowledge_qa_count}")
    print(f"  General: {stats.general_count}")
    print(f"  Avg quality: {stats.avg_quality_score:.2f}")
    print(f"  Total tokens: {stats.total_tokens:,}")


def cmd_stats(args: argparse.Namespace) -> None:
    """Show dataset statistics."""
    from ara.fine_tuning.dataset import compute_stats, load_dataset

    input_path = Path(args.input)
    samples = load_dataset(input_path)
    stats = compute_stats(samples)

    print(f"\nDataset: {input_path}")
    print(f"  Total samples: {stats.total_samples}")
    print(f"  Personality: {stats.personality_count}")
    print(f"  Knowledge QA: {stats.knowledge_qa_count}")
    print(f"  General: {stats.general_count}")
    print(f"  Scored: {stats.scored_count}")
    print(f"  Avg quality: {stats.avg_quality_score:.2f}")
    print(f"  Avg tokens/sample: {stats.avg_token_count:.0f}")
    print(f"  Total tokens: {stats.total_tokens:,}")
    print("\n  By humor level:")
    for level, count in sorted(stats.by_humor_level.items()):
        print(f"    {level}: {count}")


def cmd_status(args: argparse.Namespace) -> None:
    """Show checkpoint status."""
    output_dir = Path(args.output_dir)
    checkpoint = _load_checkpoint(output_dir)

    if not checkpoint:
        print("No checkpoint found.")
        return

    print(f"Checkpoint: {_checkpoint_path(output_dir)}")
    print(f"  Phase: {checkpoint.get('phase', 'unknown')}")
    print(f"  Phase index: {checkpoint.get('phase_index', 0)}")
    print(f"  Target: {checkpoint.get('target', 0)}")
    print(f"  Total samples: {checkpoint.get('total_samples', 0)}")
    print(f"  Failed: {checkpoint.get('failed', 0)}")
    print(f"  Completed phases: {checkpoint.get('completed_phases', [])}")
    print(f"  Timestamp: {checkpoint.get('timestamp', 'unknown')}")
    print(f"  Concurrency: {checkpoint.get('concurrency', 1)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="ARA fine-tuning dataset generator")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # scan
    scan_parser = subparsers.add_parser("scan", help="Scan diary directory")
    scan_parser.add_argument("--diary-dir", required=True, help="Path to diary directory")

    # generate
    gen_parser = subparsers.add_parser("generate", help="Generate training data")
    gen_parser.add_argument("--diary-dir", required=True, help="Path to diary directory")
    gen_parser.add_argument("--target", type=int, default=500, help="Target sample count")
    gen_parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory")
    gen_parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Max concurrent LLM calls (1 for local Ollama, 5-10 for cloud APIs)",
    )
    gen_parser.add_argument(
        "--backend",
        choices=["ollama", "openai"],
        default="ollama",
        help="LLM backend: 'ollama' (local) or 'openai' (any OpenAI-compatible API)",
    )
    gen_parser.add_argument(
        "--api-key",
        default="",
        help="API key for OpenAI backend (or set OPENAI_API_KEY env var)",
    )
    gen_parser.add_argument(
        "--api-base",
        default="https://api.openai.com/v1",
        help="API base URL for OpenAI-compatible backend",
    )
    gen_parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="Model name for OpenAI backend (default: gpt-4o-mini)",
    )

    # score
    score_parser = subparsers.add_parser("score", help="Score dataset with LLM-as-judge")
    score_parser.add_argument("--input", required=True, help="Input dataset file")
    score_parser.add_argument("--min-score", type=float, default=4.0, help="Minimum quality score")
    score_parser.add_argument("--output", default="", help="Output path (default: scored.json)")
    score_parser.add_argument("--concurrency", type=int, default=1, help="Max concurrent LLM calls")
    score_parser.add_argument(
        "--backend", choices=["ollama", "openai"], default="ollama", help="LLM backend"
    )
    score_parser.add_argument("--api-key", default="", help="API key for OpenAI backend")
    score_parser.add_argument(
        "--api-base", default="https://api.openai.com/v1", help="API base URL"
    )
    score_parser.add_argument("--model", default="gpt-4o-mini", help="Model for OpenAI backend")

    # split
    split_parser = subparsers.add_parser("split", help="Split dataset into train/val/test")
    split_parser.add_argument("--input", required=True, help="Input dataset file")
    split_parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory")
    split_parser.add_argument("--train-ratio", type=float, default=0.8)
    split_parser.add_argument("--val-ratio", type=float, default=0.1)
    split_parser.add_argument("--test-ratio", type=float, default=0.1)

    # stats
    stats_parser = subparsers.add_parser("stats", help="Show dataset stats")
    stats_parser.add_argument("--input", required=True, help="Input dataset file")

    # status
    status_parser = subparsers.add_parser("status", help="Show checkpoint status")
    status_parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory")

    args = parser.parse_args()

    if args.command == "scan":
        cmd_scan(args)
    elif args.command == "generate":
        asyncio.run(cmd_generate(args))
    elif args.command == "score":
        asyncio.run(cmd_score(args))
    elif args.command == "split":
        cmd_split(args)
    elif args.command == "stats":
        cmd_stats(args)
    elif args.command == "status":
        cmd_status(args)


if __name__ == "__main__":
    main()
