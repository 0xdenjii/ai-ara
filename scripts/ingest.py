"""CLI tool for ingesting diary files and text documents into ARA's memory.

Usage:
    # Ingest a single file
    uv run python scripts/ingest.py file data/diaries/diary_2024.md

    # Ingest all markdown files in a directory
    uv run python scripts/ingest.py dir data/diaries/

    # Force re-ingestion (skip change detection)
    uv run python scripts/ingest.py dir data/diaries/ --force

    # Ingest with a specific pattern
    uv run python scripts/ingest.py dir data/diaries/ --pattern "*.txt"

    # Dry run (analyze without ingesting)
    uv run python scripts/ingest.py dir data/diaries/ --dry-run

    # Show manifest (what's been ingested)
    uv run python scripts/ingest.py status
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ara.config.settings import MemorySettings
from ara.memory.chunking import ChunkerConfig, TextChunker
from ara.memory.ingestion import IngestionManifest, IngestionPipeline


def _print_header(text: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {text}")
    print(f"{'=' * 60}\n")


def _print_result(result: object, verbose: bool = False) -> None:
    """Print an IngestionResult."""
    from ara.memory.ingestion import IngestionResult

    if not isinstance(result, IngestionResult):
        return

    status = "SKIPPED" if result.skipped else "OK"
    reason = f" ({result.skip_reason})" if result.skipped else ""

    print(f"  [{status}] {result.source_file}{reason}")
    if not result.skipped:
        print(f"         chunks: {result.chunks_stored}, time: {result.elapsed_ms:.0f}ms")
    if verbose and result.file_hash:
        print(f"         hash: {result.file_hash[:16]}...")


def cmd_ingest_file(args: argparse.Namespace) -> int:
    """Ingest a single file."""
    file_path = Path(args.path)

    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        return 1

    if args.dry_run:
        _print_header("DRY RUN — Analyzing file")
        _analyze_file(file_path)
        return 0

    _print_header(f"Ingesting: {file_path.name}")

    settings = MemorySettings()
    pipeline = IngestionPipeline.from_settings(settings)

    print("Loading embedding model...")
    start = time.monotonic()
    pipeline.initialize()
    print(f"Model loaded in {(time.monotonic() - start) * 1000:.0f}ms")

    try:
        result = pipeline.ingest_file(file_path, force=args.force)
        _print_result(result, verbose=args.verbose)

        if not result.skipped:
            print(f"\nTotal chunks stored: {result.chunks_stored}")
            print(f"Total time: {result.elapsed_ms:.0f}ms")
    finally:
        pipeline.shutdown()

    return 0


def cmd_ingest_dir(args: argparse.Namespace) -> int:
    """Ingest all matching files in a directory."""
    dir_path = Path(args.path)

    if not dir_path.exists() or not dir_path.is_dir():
        print(f"Error: Directory not found: {dir_path}")
        return 1

    pattern = args.pattern

    if args.dry_run:
        _print_header(f"DRY RUN — Analyzing directory: {dir_path}")
        _analyze_directory(dir_path, pattern, args.recursive)
        return 0

    _print_header(f"Ingesting directory: {dir_path}")
    print(f"  Pattern: {pattern}")
    print(f"  Recursive: {args.recursive}")
    print(f"  Force: {args.force}")

    settings = MemorySettings()
    pipeline = IngestionPipeline.from_settings(settings)

    print("\nLoading embedding model...")
    start = time.monotonic()
    pipeline.initialize()
    print(f"Model loaded in {(time.monotonic() - start) * 1000:.0f}ms\n")

    try:
        result = pipeline.ingest_directory(
            dir_path,
            pattern=pattern,
            force=args.force,
            recursive=args.recursive,
        )

        print("Results:")
        for file_result in result.results:
            _print_result(file_result, verbose=args.verbose)

        if result.errors:
            print(f"\nErrors ({len(result.errors)}):")
            for err in result.errors:
                print(f"  [ERROR] {err}")

        print(f"\n{'─' * 40}")
        print(f"  Files processed: {result.files_processed}")
        print(f"  Files skipped:   {result.files_skipped}")
        print(f"  Total chunks:    {result.total_chunks}")
        print(f"  Total time:      {result.total_elapsed_ms:.0f}ms")
        print(f"  Errors:          {len(result.errors)}")
    finally:
        pipeline.shutdown()

    return 0


def cmd_status(args: argparse.Namespace) -> int:
    """Show ingestion status (manifest contents)."""
    settings = MemorySettings()
    manifest_path = Path(settings.vector_db_path) / "manifest.json"

    _print_header("Ingestion Status")

    if not manifest_path.exists():
        print("  No manifest found. Nothing has been ingested yet.")
        return 0

    manifest = IngestionManifest(manifest_path)
    files = manifest.files

    if not files:
        print("  Manifest is empty. Nothing has been ingested yet.")
        return 0

    print(f"  Manifest: {manifest_path}")
    print(f"  Files ingested: {len(files)}\n")

    for source_file in sorted(files):
        file_hash = manifest.get_hash(source_file) or "unknown"
        print(f"  {source_file}")
        print(f"    hash: {file_hash[:16]}...")

    return 0


def _analyze_file(file_path: Path) -> None:
    """Analyze a file without ingesting it."""
    text = file_path.read_text(encoding="utf-8")
    chunker = TextChunker(ChunkerConfig())

    chunks = chunker.chunk_document(text, source_file=file_path.name)

    print(f"  File: {file_path.name}")
    print(f"  Size: {len(text):,} chars")
    print(f"  Lines: {len(text.splitlines()):,}")
    print(f"  Chunks: {len(chunks)}")
    if chunks:
        tokens = [c.metadata.get("token_count", 0) for c in chunks]
        print(
            f"  Tokens per chunk: min={min(tokens)}, max={max(tokens)}, avg={sum(tokens) // len(tokens)}"
        )
        dates = {c.date for c in chunks if c.date is not None}
        if dates:
            print(
                f"  Date range: {min(dates).strftime('%Y-%m-%d')} to {max(dates).strftime('%Y-%m-%d')}"
            )
            print(f"  Unique dates: {len(dates)}")


def _analyze_directory(dir_path: Path, pattern: str, recursive: bool) -> None:
    """Analyze all matching files in a directory."""
    glob_method = dir_path.rglob if recursive else dir_path.glob
    files = sorted(f for f in glob_method(pattern) if f.is_file())

    print(f"  Files found: {len(files)}\n")

    total_chars = 0
    total_chunks = 0
    chunker = TextChunker(ChunkerConfig())

    for file_path in files:
        text = file_path.read_text(encoding="utf-8")
        chunks = chunker.chunk_document(text, source_file=file_path.name)
        total_chars += len(text)
        total_chunks += len(chunks)
        print(f"  {file_path.name}: {len(text):,} chars, {len(chunks)} chunks")

    print(f"\n  {'─' * 30}")
    print(f"  Total files: {len(files)}")
    print(f"  Total size: {total_chars:,} chars")
    print(f"  Total chunks: {total_chunks}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="ARA Memory Ingestion CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # File command
    file_parser = subparsers.add_parser("file", help="Ingest a single file")
    file_parser.add_argument("path", help="Path to the file")
    file_parser.add_argument("--force", action="store_true", help="Force re-ingestion")
    file_parser.add_argument("--dry-run", action="store_true", help="Analyze without ingesting")
    file_parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")

    # Directory command
    dir_parser = subparsers.add_parser("dir", help="Ingest all files in a directory")
    dir_parser.add_argument("path", help="Path to the directory")
    dir_parser.add_argument("--pattern", default="*.md", help="File glob pattern (default: *.md)")
    dir_parser.add_argument("--force", action="store_true", help="Force re-ingestion")
    dir_parser.add_argument(
        "--no-recursive", dest="recursive", action="store_false", help="Don't search subdirectories"
    )
    dir_parser.add_argument("--dry-run", action="store_true", help="Analyze without ingesting")
    dir_parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")

    # Status command
    subparsers.add_parser("status", help="Show ingestion status")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    if args.command == "file":
        return cmd_ingest_file(args)
    if args.command == "dir":
        return cmd_ingest_dir(args)
    if args.command == "status":
        return cmd_status(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
