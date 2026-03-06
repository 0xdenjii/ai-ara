"""CLI tool for syncing Obsidian daily notes into ARA's RAG memory.

Connects to the Obsidian Local REST API, reads daily notes, parses them,
and ingests the content into LanceDB via the ingestion pipeline.

Usage:
    # Boot sync — ingest recent daily notes (default: last 7 days)
    uv run python scripts/obsidian_sync.py boot

    # Boot sync with extended lookback (e.g., all 21 existing notes since Feb 9)
    uv run python scripts/obsidian_sync.py boot --lookback 30

    # Sync a specific date range
    uv run python scripts/obsidian_sync.py range --start 2026-02-09 --end 2026-03-06

    # Sync a single date
    uv run python scripts/obsidian_sync.py date 2026-03-06

    # Incremental sync (today + yesterday only)
    uv run python scripts/obsidian_sync.py incremental

    # List available daily notes in the vault
    uv run python scripts/obsidian_sync.py list

    # Check connection to Obsidian REST API
    uv run python scripts/obsidian_sync.py check

Prerequisites:
    - Obsidian desktop app must be running
    - Local REST API plugin must be installed and enabled
    - ARA_OBSIDIAN_API_KEY must be set in .env
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from ara.config.settings import AraSettings
from ara.memory.ingestion import IngestionPipeline
from ara.obsidian.client import ObsidianClient
from ara.obsidian.parser import DailyNoteParser
from ara.obsidian.sync import ObsidianSync


def _print_header(title: str) -> None:
    """Print a formatted section header."""
    width = 60
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}\n")


def _print_sync_state(state: object) -> None:
    """Print sync state results."""
    print(f"  Notes synced:    {state.notes_synced}")  # type: ignore[attr-defined]
    print(f"  Notes changed:   {state.notes_changed}")  # type: ignore[attr-defined]
    print(f"  Notes ingested:  {state.notes_ingested}")  # type: ignore[attr-defined]
    if state.errors:  # type: ignore[attr-defined]
        print(f"  Errors:          {len(state.errors)}")  # type: ignore[attr-defined]
        for err in state.errors:  # type: ignore[attr-defined]
            print(f"    - {err}")
    else:
        print("  Errors:          0")
    print(f"  Sync timestamp:  {state.last_sync_at}")  # type: ignore[attr-defined]


async def _create_pipeline(settings: AraSettings) -> IngestionPipeline:
    """Create and initialize the ingestion pipeline."""
    pipeline = IngestionPipeline.from_settings(settings.memory)
    pipeline.initialize()
    return pipeline


async def _create_client(settings: AraSettings) -> ObsidianClient:
    """Create and connect the Obsidian client."""
    client = ObsidianClient(settings.obsidian)
    await client.connect()
    return client


async def cmd_check(settings: AraSettings) -> int:
    """Check connection to Obsidian REST API."""
    _print_header("Obsidian REST API Connection Check")

    if not settings.obsidian.api_key:
        print("  ERROR: ARA_OBSIDIAN_API_KEY is not set in .env")
        print("  Get the API key from Obsidian > Settings > Local REST API")
        return 1

    print(f"  API URL: {settings.obsidian.api_url}")
    print(f"  API key: {settings.obsidian.api_key[:8]}...{settings.obsidian.api_key[-4:]}")

    try:
        client = await _create_client(settings)
        print("  Status:  CONNECTED")
        await client.close()
        return 0
    except Exception as exc:
        print(f"  Status:  FAILED — {exc}")
        print("\n  Troubleshooting:")
        print("  1. Is Obsidian desktop app running?")
        print("  2. Is the Local REST API plugin enabled?")
        print("  3. Is the API key correct?")
        return 1


async def _list_files_recursive(
    client: ObsidianClient,
    folder: str,
) -> list[str]:
    """Recursively list all .md files under a folder."""
    all_files: list[str] = []
    try:
        entries = await client.list_files(folder)
    except Exception:
        return all_files

    for entry in entries:
        full_path = f"{folder.rstrip('/')}/{entry}"
        if entry.endswith("/"):
            # It's a subdirectory — recurse
            subfolder = full_path.rstrip("/")
            sub_files = await _list_files_recursive(client, subfolder)
            all_files.extend(sub_files)
        elif entry.endswith(".md"):
            all_files.append(full_path)

    return all_files


async def cmd_list(settings: AraSettings) -> int:
    """List daily notes available in the vault."""
    _print_header("Available Daily Notes")

    try:
        client = await _create_client(settings)
    except Exception as exc:
        print(f"  Connection failed: {exc}")
        return 1

    try:
        folder = settings.obsidian.daily_note_folder
        print(f"  Scanning: {folder}/ (recursive)\n")

        md_files = sorted(await _list_files_recursive(client, folder))

        if not md_files:
            print("  No daily notes found.")
            return 0

        print(f"  Found {len(md_files)} daily notes:\n")
        for f in md_files:
            print(f"    {f}")

        return 0
    finally:
        await client.close()


async def cmd_boot(settings: AraSettings, lookback_days: int) -> int:
    """Run boot sync with configurable lookback."""
    _print_header(f"Boot Sync (lookback: {lookback_days} days)")

    if not settings.obsidian.api_key:
        print("  ERROR: ARA_OBSIDIAN_API_KEY is not set in .env")
        return 1

    # Override lookback if specified
    settings.obsidian.sync_lookback_days = lookback_days

    try:
        client = await _create_client(settings)
    except Exception as exc:
        print(f"  Connection failed: {exc}")
        return 1

    try:
        pipeline = await _create_pipeline(settings)
    except Exception as exc:
        print(f"  Pipeline initialization failed: {exc}")
        await client.close()
        return 1

    try:
        parser = DailyNoteParser()
        sync = ObsidianSync(
            settings=settings.obsidian,
            client=client,
            parser=parser,
            ingestion=pipeline,
        )

        print(f"  Connected to Obsidian at {settings.obsidian.api_url}")
        print(f"  Checking daily notes from {lookback_days} days ago to today...\n")

        state = await sync.boot_sync()
        _print_sync_state(state)

        return 0 if not state.errors else 1
    finally:
        await client.close()
        pipeline.shutdown()


async def cmd_incremental(settings: AraSettings) -> int:
    """Run incremental sync (today + yesterday)."""
    _print_header("Incremental Sync")

    if not settings.obsidian.api_key:
        print("  ERROR: ARA_OBSIDIAN_API_KEY is not set in .env")
        return 1

    try:
        client = await _create_client(settings)
    except Exception as exc:
        print(f"  Connection failed: {exc}")
        return 1

    try:
        pipeline = await _create_pipeline(settings)
    except Exception as exc:
        print(f"  Pipeline initialization failed: {exc}")
        await client.close()
        return 1

    try:
        parser = DailyNoteParser()
        sync = ObsidianSync(
            settings=settings.obsidian,
            client=client,
            parser=parser,
            ingestion=pipeline,
        )

        print("  Checking today and yesterday for changes...\n")

        state = await sync.incremental_sync()
        _print_sync_state(state)

        return 0 if not state.errors else 1
    finally:
        await client.close()
        pipeline.shutdown()


async def cmd_date(settings: AraSettings, date_str: str) -> int:
    """Sync a single specific date."""
    _print_header(f"Sync Single Date: {date_str}")

    if not settings.obsidian.api_key:
        print("  ERROR: ARA_OBSIDIAN_API_KEY is not set in .env")
        return 1

    # Validate date format
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        print(f"  ERROR: Invalid date format '{date_str}'. Use YYYY-MM-DD.")
        return 1

    try:
        client = await _create_client(settings)
    except Exception as exc:
        print(f"  Connection failed: {exc}")
        return 1

    try:
        pipeline = await _create_pipeline(settings)
    except Exception as exc:
        print(f"  Pipeline initialization failed: {exc}")
        await client.close()
        return 1

    try:
        parser = DailyNoteParser()
        sync = ObsidianSync(
            settings=settings.obsidian,
            client=client,
            parser=parser,
            ingestion=pipeline,
        )

        note = await sync.sync_specific_date(date_str)
        if note is None:
            print(f"  No daily note found for {date_str}")
            return 1

        print(f"  Synced: {date_str}")
        print(f"  Brainstorm:     {'present' if note.brainstorm else 'empty'}")
        print(
            f"  Ops Log:        {sum(1 for q in [note.q1, note.q2, note.q3, note.q4] if q)} quarters"
        )
        print(f"  Protocol tasks: {len(note.protocol_tasks)}")
        print(f"  Total tasks:    {len(note.all_tasks)}")

        # Force ingest this note
        if pipeline is not None:
            path = client.build_daily_note_path(date_str)
            content = await client.read_note(path)
            result = pipeline.ingest_text(
                text=content,
                source_name=f"obsidian:{path}",
                force=True,
            )
            print(f"\n  Ingested: {result.chunks_created} chunks, {result.chunks_stored} stored")

        return 0
    finally:
        await client.close()
        pipeline.shutdown()


async def cmd_range(settings: AraSettings, start_date: str, end_date: str) -> int:
    """Sync a date range."""
    _print_header(f"Sync Date Range: {start_date} to {end_date}")

    if not settings.obsidian.api_key:
        print("  ERROR: ARA_OBSIDIAN_API_KEY is not set in .env")
        return 1

    # Validate dates
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
        end_dt = datetime.strptime(end_date, "%Y-%m-%d").date()
    except ValueError as exc:
        print(f"  ERROR: Invalid date format. Use YYYY-MM-DD. ({exc})")
        return 1

    if start_dt > end_dt:
        print("  ERROR: Start date must be before or equal to end date.")
        return 1

    try:
        client = await _create_client(settings)
    except Exception as exc:
        print(f"  Connection failed: {exc}")
        return 1

    try:
        pipeline = await _create_pipeline(settings)
    except Exception as exc:
        print(f"  Pipeline initialization failed: {exc}")
        await client.close()
        return 1

    try:
        parser = DailyNoteParser()
        sync = ObsidianSync(
            settings=settings.obsidian,
            client=client,
            parser=parser,
            ingestion=pipeline,
        )

        # Use scan_date_range if available, otherwise iterate
        notes = await sync.scan_date_range(start_date, end_date)
        print(f"  Found {len(notes)} daily notes in range\n")

        # Force ingest each
        ingested_count = 0
        for note in notes:
            path = client.build_daily_note_path(note.date)
            content = await client.read_note(path)
            result = pipeline.ingest_text(
                text=content,
                source_name=f"obsidian:{path}",
                force=True,
            )
            status = f"{result.chunks_created} chunks" if result.chunks_stored > 0 else "skipped"
            print(f"    {note.date}: {status}")
            if result.chunks_stored > 0:
                ingested_count += 1

        print(f"\n  Total ingested: {ingested_count}/{len(notes)} notes")
        return 0
    finally:
        await client.close()
        pipeline.shutdown()


def main() -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="Sync Obsidian daily notes into ARA's RAG memory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # check
    subparsers.add_parser("check", help="Check Obsidian REST API connection")

    # list
    subparsers.add_parser("list", help="List available daily notes")

    # boot
    boot_parser = subparsers.add_parser("boot", help="Boot sync (scan recent notes)")
    boot_parser.add_argument(
        "--lookback",
        type=int,
        default=7,
        help="Days to look back (default: 7, use 30 for full history)",
    )

    # incremental
    subparsers.add_parser("incremental", help="Incremental sync (today + yesterday)")

    # date
    date_parser = subparsers.add_parser("date", help="Sync a single date")
    date_parser.add_argument("date", help="Date in YYYY-MM-DD format")

    # range
    range_parser = subparsers.add_parser("range", help="Sync a date range")
    range_parser.add_argument("--start", required=True, help="Start date (YYYY-MM-DD)")
    range_parser.add_argument("--end", required=True, help="End date (YYYY-MM-DD)")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    settings = AraSettings()

    if args.command == "check":
        return asyncio.run(cmd_check(settings))
    if args.command == "list":
        return asyncio.run(cmd_list(settings))
    if args.command == "boot":
        return asyncio.run(cmd_boot(settings, lookback_days=args.lookback))
    if args.command == "incremental":
        return asyncio.run(cmd_incremental(settings))
    if args.command == "date":
        return asyncio.run(cmd_date(settings, date_str=args.date))
    if args.command == "range":
        return asyncio.run(cmd_range(settings, start_date=args.start, end_date=args.end))

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
