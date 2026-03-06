"""CLI tool for preprocessing diary data before ingestion.

Scans the dairy/ directory, classifies all files by content type and
priority tier, and produces a report. Can also run the full ingestion
pipeline on classified files.

Usage:
    # Scan and report (no ingestion)
    uv run python scripts/preprocess_diary.py scan dairy/

    # Scan with verbose per-file output
    uv run python scripts/preprocess_diary.py scan dairy/ --verbose

    # Export classification to JSON
    uv run python scripts/preprocess_diary.py scan dairy/ --export report.json

    # Run ingestion on a specific tier
    uv run python scripts/preprocess_diary.py ingest dairy/ --tier tier_1

    # Run ingestion on all tiers
    uv run python scripts/preprocess_diary.py ingest dairy/ --tier all

    # Dry run (classify + chunk analysis, no embedding/storage)
    uv run python scripts/preprocess_diary.py ingest dairy/ --dry-run
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

# Force UTF-8 stdout/stderr on Windows to avoid cp1252 encoding errors
# with Unicode characters in diary filenames and content
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ara.memory.preprocessor import (
    DiaryPreprocessor,
    FileClassification,
    PreprocessResult,
)


def _print_header(text: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {text}")
    print(f"{'=' * 60}\n")


def _print_tier(
    tier_name: str,
    files: list[FileClassification],
    verbose: bool = False,
) -> None:
    """Print summary for a tier."""
    if not files:
        print(f"  {tier_name}: 0 files")
        return

    total_lines = sum(f.line_count for f in files)
    total_chars = sum(f.char_count for f in files)

    # Group by content type
    by_type: dict[str, int] = {}
    for f in files:
        by_type[f.content_type] = by_type.get(f.content_type, 0) + 1

    # Group by source export
    by_export: dict[str, int] = {}
    for f in files:
        by_export[f.source_export] = by_export.get(f.source_export, 0) + 1

    print(f"  {tier_name}: {len(files)} files ({total_lines:,} lines, {total_chars:,} chars)")
    print(f"    By type:   {', '.join(f'{t}={c}' for t, c in sorted(by_type.items()))}")
    print(f"    By export: {', '.join(f'{e}={c}' for e, c in sorted(by_export.items()))}")

    if verbose:
        print()
        for f in sorted(files, key=lambda x: (x.source_export, x.clean_name)):
            tags_str = ", ".join(sorted(f.tags - {f"export:{f.source_export}"}))
            print(f"    [{f.content_type:20s}] {f.source_export}/{f.clean_name}")
            if tags_str:
                print(f"    {'':20s}   tags: {tags_str}")
            print(f"    {'':20s}   {f.line_count} lines, {f.char_count} chars")
        print()


def cmd_scan(args: argparse.Namespace) -> int:
    """Scan and classify all diary files."""
    dairy_root = Path(args.path)

    if not dairy_root.exists():
        print(f"Error: Path not found: {dairy_root}")
        return 1

    _print_header(f"Scanning: {dairy_root}")

    preprocessor = DiaryPreprocessor()
    start = time.monotonic()
    result = preprocessor.scan(dairy_root)
    elapsed = time.monotonic() - start

    print(f"  Scan completed in {elapsed * 1000:.0f}ms")
    print(f"  Total files found: {result.total_files}")
    print()

    _print_tier("TIER 1 (Personal diary, life docs)", result.tier_1_files, args.verbose)
    print()
    _print_tier("TIER 2 (Todos, study notes, research)", result.tier_2_files, args.verbose)
    print()
    _print_tier("TIER 3 (Startup workspace)", result.tier_3_files, args.verbose)
    print()
    _print_tier("EXCLUDED (binary, keys, empty)", result.excluded_files, args.verbose)

    if result.errors:
        print(f"\n  Errors ({len(result.errors)}):")
        for err in result.errors:
            print(f"    [ERROR] {err}")

    # Summary
    print(f"\n{'-' * 60}")
    summary = result.summary()
    print(f"  Tier 1:    {summary['tier_1']:>5} files (ingest first)")
    print(f"  Tier 2:    {summary['tier_2']:>5} files (ingest with tags)")
    print(f"  Tier 3:    {summary['tier_3']:>5} files (low priority)")
    print(f"  Excluded:  {summary['excluded']:>5} files (skip)")
    print(f"  Ingestible:{summary['ingestible']:>5} files total")
    print(f"  Errors:    {summary['errors']:>5}")

    # Export if requested
    if args.export:
        _export_report(result, Path(args.export))

    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    """Ingest classified diary files."""
    dairy_root = Path(args.path)

    if not dairy_root.exists():
        print(f"Error: Path not found: {dairy_root}")
        return 1

    _print_header(f"Preprocessing + Ingesting: {dairy_root}")

    preprocessor = DiaryPreprocessor()
    print("Scanning files...")
    result = preprocessor.scan(dairy_root)
    print(f"  Found {result.ingestible_count} ingestible files\n")

    # Select files based on tier
    tier = args.tier
    files_to_ingest: list[FileClassification] = []
    if tier in ("tier_1", "all"):
        files_to_ingest.extend(result.tier_1_files)
    if tier in ("tier_2", "all"):
        files_to_ingest.extend(result.tier_2_files)
    if tier in ("tier_3", "all"):
        files_to_ingest.extend(result.tier_3_files)

    print(f"  Selected {len(files_to_ingest)} files for tier={tier}")

    if args.dry_run:
        print("\n  DRY RUN — analyzing chunk counts only:\n")
        from ara.memory.chunking import ChunkerConfig, TextChunker

        chunker = TextChunker(ChunkerConfig())
        total_chunks = 0
        for fc in files_to_ingest:
            try:
                cleaned = preprocessor.clean_file(fc.path)
                source_name = preprocessor.make_source_name(fc)
                chunks = chunker.chunk_document(cleaned, source_file=source_name)
                total_chunks += len(chunks)
                print(f"    {source_name}: {len(cleaned):,} chars -> {len(chunks)} chunks")
            except Exception as exc:
                print(f"    [ERROR] {fc.clean_name}: {exc}")

        print(f"\n  Total chunks: {total_chunks}")
        return 0

    # Real ingestion
    from ara.config.settings import MemorySettings
    from ara.memory.ingestion import IngestionPipeline

    settings = MemorySettings()
    pipeline = IngestionPipeline.from_settings(settings)

    print("\nLoading embedding model...")
    start = time.monotonic()
    pipeline.initialize()
    print(f"Model loaded in {(time.monotonic() - start) * 1000:.0f}ms\n")

    ingested = 0
    skipped = 0
    errors = 0

    try:
        for fc in files_to_ingest:
            try:
                cleaned = preprocessor.clean_file(fc.path)
                source_name = preprocessor.make_source_name(fc)

                # Add classification metadata
                ingest_result = pipeline.ingest_text(
                    cleaned,
                    source_name=source_name,
                    force=args.force,
                )

                if ingest_result.skipped:
                    skipped += 1
                    status = "SKIP"
                else:
                    ingested += 1
                    status = "OK"

                print(
                    f"  [{status:4s}] {source_name} "
                    f"({ingest_result.chunks_stored} chunks, "
                    f"{ingest_result.elapsed_ms:.0f}ms)"
                )
            except Exception as exc:
                errors += 1
                print(f"  [ERR ] {fc.clean_name}: {exc}")
    finally:
        pipeline.shutdown()

    print(f"\n{'-' * 60}")
    print(f"  Ingested: {ingested}")
    print(f"  Skipped:  {skipped}")
    print(f"  Errors:   {errors}")

    return 0


def _export_report(result: PreprocessResult, output_path: Path) -> None:
    """Export classification report to JSON."""
    data = {
        "summary": result.summary(),
        "tier_1": [_fc_to_dict(f) for f in result.tier_1_files],
        "tier_2": [_fc_to_dict(f) for f in result.tier_2_files],
        "tier_3": [_fc_to_dict(f) for f in result.tier_3_files],
        "excluded": [_fc_to_dict(f) for f in result.excluded_files],
        "errors": result.errors,
    }
    output_path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    print(f"\n  Report exported to: {output_path}")


def _fc_to_dict(fc: FileClassification) -> dict[str, object]:
    """Convert FileClassification to a JSON-serializable dict."""
    return {
        "path": str(fc.path),
        "content_type": fc.content_type,
        "tier": fc.tier,
        "source_export": fc.source_export,
        "clean_name": fc.clean_name,
        "tags": sorted(fc.tags),
        "line_count": fc.line_count,
        "char_count": fc.char_count,
        "has_urls": fc.has_urls,
        "has_todos": fc.has_todos,
        "has_narrative": fc.has_narrative,
        "has_tech_content": fc.has_tech_content,
        "has_startup_content": fc.has_startup_content,
        "skip_reason": fc.skip_reason,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="ARA Diary Preprocessor CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Scan command
    scan_parser = subparsers.add_parser("scan", help="Scan and classify diary files")
    scan_parser.add_argument("path", help="Path to the dairy/ directory")
    scan_parser.add_argument("--verbose", "-v", action="store_true", help="Show per-file details")
    scan_parser.add_argument("--export", help="Export report to JSON file")

    # Ingest command
    ingest_parser = subparsers.add_parser("ingest", help="Ingest classified diary files")
    ingest_parser.add_argument("path", help="Path to the dairy/ directory")
    ingest_parser.add_argument(
        "--tier",
        choices=["tier_1", "tier_2", "tier_3", "all"],
        default="all",
        help="Which tier(s) to ingest (default: all)",
    )
    ingest_parser.add_argument("--force", action="store_true", help="Force re-ingestion")
    ingest_parser.add_argument("--dry-run", action="store_true", help="Analyze without ingesting")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    if args.command == "scan":
        return cmd_scan(args)
    if args.command == "ingest":
        return cmd_ingest(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
