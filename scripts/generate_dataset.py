"""CLI script to generate fine-tuning training data for ARA.

Usage:
    # Scan diary files and show stats
    uv run python scripts/generate_dataset.py scan --diary-dir dairy/

    # Generate training data (requires Ollama with qwen3:30b-a3b loaded)
    uv run python scripts/generate_dataset.py generate --diary-dir dairy/ --target 3000

    # Score existing dataset with LLM-as-judge
    uv run python scripts/generate_dataset.py score --input data/fine_tuning/all.json

    # Split dataset into train/val/test
    uv run python scripts/generate_dataset.py split --input data/fine_tuning/scored.json

    # Show dataset stats
    uv run python scripts/generate_dataset.py stats --input data/fine_tuning/train.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Fix Windows console encoding
sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]


async def _generate_fn(prompt: str) -> str:
    """Generate a response using the Ollama API."""
    from ara.core.llm import LLMClient
    from ara.config.settings import LLMSettings
    from ara.utils.types import ChatMessage, MessageRole

    settings = LLMSettings()
    client = LLMClient(settings)

    messages = [
        ChatMessage(role=MessageRole.USER, content=prompt),
    ]
    response = await client.generate(messages, max_tokens=4096, temperature=0.9)
    return response


def cmd_scan(args: argparse.Namespace) -> None:
    """Scan diary directory and show segment stats."""
    from ara.fine_tuning.dataset import scan_diary_directory

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
        print("\nSource files:")
        for src in sorted(sources):
            count = sum(1 for s in segments if s.source_file == src)
            print(f"  {src}: {count} segments")


async def cmd_generate(args: argparse.Namespace) -> None:
    """Generate training data from diary segments."""
    from ara.fine_tuning.dataset import (
        ConversationType,
        GENERAL_TOPICS,
        generate_general_conversation,
        generate_knowledge_qa,
        generate_personality_conversation,
        get_humor_level_for_index,
        get_target_counts,
        save_dataset,
        scan_diary_directory,
    )

    diary_dir = Path(args.diary_dir)
    target = args.target
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning diary directory: {diary_dir}")
    segments = scan_diary_directory(diary_dir)
    print(f"Found {len(segments)} segments")

    if not segments:
        print("No segments found. Nothing to generate.")
        return

    counts = get_target_counts(target)
    print("\nTarget distribution:")
    for ct, count in counts.items():
        print(f"  {ct.value}: {count}")

    all_samples = []
    failed = 0

    # Generate personality conversations
    personality_target = counts[ConversationType.PERSONALITY]
    print(f"\nGenerating {personality_target} personality conversations...")
    for i in range(min(personality_target, len(segments))):
        seg = segments[i % len(segments)]
        humor = get_humor_level_for_index(i, personality_target)
        sample = await generate_personality_conversation(seg, humor, _generate_fn)
        if sample:
            all_samples.append(sample)
            if len(all_samples) % 50 == 0:
                print(f"  Progress: {len(all_samples)} samples generated")
        else:
            failed += 1

    # Generate knowledge Q&A
    qa_target = counts[ConversationType.KNOWLEDGE_QA]
    print(f"\nGenerating {qa_target} knowledge Q&A conversations...")
    for i in range(min(qa_target, len(segments))):
        seg = segments[i % len(segments)]
        humor = get_humor_level_for_index(i, qa_target)
        sample = await generate_knowledge_qa(seg, humor, _generate_fn)
        if sample:
            all_samples.append(sample)
        else:
            failed += 1

    # Generate general conversations
    general_target = counts[ConversationType.GENERAL]
    print(f"\nGenerating {general_target} general conversations...")
    for i in range(general_target):
        topic = GENERAL_TOPICS[i % len(GENERAL_TOPICS)]
        humor = get_humor_level_for_index(i, general_target)
        sample = await generate_general_conversation(topic, humor, _generate_fn)
        if sample:
            all_samples.append(sample)
        else:
            failed += 1

    print("\nGeneration complete!")
    print(f"  Successful: {len(all_samples)}")
    print(f"  Failed: {failed}")

    # Save all samples
    output_path = output_dir / "all.json"
    save_dataset(all_samples, output_path, format="json")
    print(f"  Saved to: {output_path}")


async def cmd_score(args: argparse.Namespace) -> None:
    """Score existing dataset with LLM-as-judge."""
    from ara.fine_tuning.dataset import (
        compute_stats,
        filter_by_quality,
        load_dataset,
        save_dataset,
        score_sample,
    )

    input_path = Path(args.input)
    samples = load_dataset(input_path)
    print(f"Loaded {len(samples)} samples from {input_path}")

    print("Scoring with LLM-as-judge...")
    for i, sample in enumerate(samples):
        if sample.quality_score > 0:
            continue  # Already scored
        score = await score_sample(sample, _generate_fn)
        sample.quality_score = score
        if (i + 1) % 50 == 0:
            print(f"  Scored {i + 1}/{len(samples)}")

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


def main() -> None:
    parser = argparse.ArgumentParser(description="ARA fine-tuning dataset generator")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # scan
    scan_parser = subparsers.add_parser("scan", help="Scan diary directory")
    scan_parser.add_argument("--diary-dir", required=True, help="Path to diary directory")

    # generate
    gen_parser = subparsers.add_parser("generate", help="Generate training data")
    gen_parser.add_argument("--diary-dir", required=True, help="Path to diary directory")
    gen_parser.add_argument("--target", type=int, default=3000, help="Target sample count")
    gen_parser.add_argument("--output-dir", default="data/fine_tuning", help="Output directory")

    # score
    score_parser = subparsers.add_parser("score", help="Score dataset with LLM-as-judge")
    score_parser.add_argument("--input", required=True, help="Input dataset file")
    score_parser.add_argument("--min-score", type=float, default=4.0, help="Minimum quality score")
    score_parser.add_argument("--output", default="", help="Output path (default: scored.json)")

    # split
    split_parser = subparsers.add_parser("split", help="Split dataset into train/val/test")
    split_parser.add_argument("--input", required=True, help="Input dataset file")
    split_parser.add_argument("--output-dir", default="data/fine_tuning", help="Output directory")
    split_parser.add_argument("--train-ratio", type=float, default=0.8)
    split_parser.add_argument("--val-ratio", type=float, default=0.1)
    split_parser.add_argument("--test-ratio", type=float, default=0.1)

    # stats
    stats_parser = subparsers.add_parser("stats", help="Show dataset stats")
    stats_parser.add_argument("--input", required=True, help="Input dataset file")

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


if __name__ == "__main__":
    main()
