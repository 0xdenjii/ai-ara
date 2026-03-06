"""Obsidian vault sync engine.

Manages synchronization between Obsidian vault and ARA's knowledge base:
- Boot sync: On startup, scan recent daily notes and ingest changed ones
- Incremental sync: Periodically check for new/changed notes
- Change detection: SHA-256 content hashing via IngestionManifest

Architecture:
    ObsidianSync depends on ObsidianClient (for vault I/O), DailyNoteParser
    (for content parsing), and optionally IngestionPipeline (for RAG ingestion).
    It maintains a hash manifest to detect changes without re-reading unchanged files.

Usage:
    from ara.obsidian.sync import ObsidianSync

    sync = ObsidianSync(settings.obsidian, client, parser)
    state = await sync.boot_sync()          # Full sync on startup
    state = await sync.incremental_sync()   # Check for changes
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import structlog

from ara.obsidian.parser import DailyNoteParser
from ara.utils.errors import ObsidianReadError
from ara.utils.types import DailyNoteData, ObsidianSyncState

if TYPE_CHECKING:
    from ara.config.settings import ObsidianSettings
    from ara.memory.ingestion import IngestionPipeline
    from ara.obsidian.client import ObsidianClient

log = structlog.get_logger(component="obsidian.sync")

# Pattern for daily note filenames
_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})\.md$")


class ObsidianSync:
    """Syncs Obsidian vault daily notes with ARA's knowledge base.

    Manages content hashing for change detection, batch ingestion into
    the vector store, and parsed note caching for downstream consumers
    (writer, briefing generator, habit tracker).

    Attributes:
        settings: ObsidianSettings configuration.
        notes_cache: Dict mapping date strings to parsed DailyNoteData.
    """

    def __init__(
        self,
        settings: ObsidianSettings,
        client: ObsidianClient,
        parser: DailyNoteParser | None = None,
        ingestion: IngestionPipeline | None = None,
    ) -> None:
        self._settings = settings
        self._client = client
        self._parser = parser or DailyNoteParser()
        self._ingestion = ingestion
        self._hash_cache: dict[str, str] = {}  # path -> content_hash
        self.notes_cache: dict[str, DailyNoteData] = {}  # date -> parsed data
        self._last_sync: datetime | None = None

    @property
    def last_sync(self) -> datetime | None:
        """Timestamp of the last successful sync."""
        return self._last_sync

    async def boot_sync(self) -> ObsidianSyncState:
        """Full sync on startup — scan recent daily notes and ingest changes.

        Looks back `sync_lookback_days` from today, reads all daily notes,
        detects changes via content hashing, and ingests changed notes.

        Returns:
            ObsidianSyncState with sync results.

        Raises:
            ObsidianSyncError: If the sync encounters a critical failure.
        """
        log.info("obsidian_boot_sync_start", lookback_days=self._settings.sync_lookback_days)

        today = datetime.now().date()
        errors: list[str] = []
        notes_synced = 0
        notes_changed = 0
        notes_ingested = 0

        # Generate date range to check
        dates_to_check = []
        for days_back in range(self._settings.sync_lookback_days + 1):
            dt = today - timedelta(days=days_back)
            dates_to_check.append(dt.isoformat())

        for date_str in dates_to_check:
            try:
                result = await self._sync_daily_note(date_str)
                if result is not None:
                    notes_synced += 1
                    changed, ingested = result
                    if changed:
                        notes_changed += 1
                    if ingested:
                        notes_ingested += 1
            except ObsidianReadError:
                # Note doesn't exist for this date — that's normal
                continue
            except Exception as exc:
                error_msg = f"Failed to sync {date_str}: {exc}"
                log.warning("obsidian_sync_note_error", date=date_str, error=str(exc))
                errors.append(error_msg)

        self._last_sync = datetime.now()

        state = ObsidianSyncState(
            last_sync_at=self._last_sync,
            notes_synced=notes_synced,
            notes_changed=notes_changed,
            notes_ingested=notes_ingested,
            errors=errors,
        )

        log.info(
            "obsidian_boot_sync_complete",
            synced=notes_synced,
            changed=notes_changed,
            ingested=notes_ingested,
            errors=len(errors),
        )

        return state

    async def incremental_sync(self) -> ObsidianSyncState:
        """Check for changes since last sync.

        Shorter window — only checks today and yesterday for efficiency.

        Returns:
            ObsidianSyncState with sync results.
        """
        log.debug("obsidian_incremental_sync_start")

        today = datetime.now().date()
        errors: list[str] = []
        notes_synced = 0
        notes_changed = 0
        notes_ingested = 0

        # Check today and yesterday
        for days_back in range(2):
            dt = today - timedelta(days=days_back)
            date_str = dt.isoformat()
            try:
                result = await self._sync_daily_note(date_str)
                if result is not None:
                    notes_synced += 1
                    changed, ingested = result
                    if changed:
                        notes_changed += 1
                    if ingested:
                        notes_ingested += 1
            except ObsidianReadError:
                continue
            except Exception as exc:
                error_msg = f"Failed to sync {date_str}: {exc}"
                log.warning("obsidian_sync_note_error", date=date_str, error=str(exc))
                errors.append(error_msg)

        self._last_sync = datetime.now()

        state = ObsidianSyncState(
            last_sync_at=self._last_sync,
            notes_synced=notes_synced,
            notes_changed=notes_changed,
            notes_ingested=notes_ingested,
            errors=errors,
        )

        if notes_changed > 0:
            log.info(
                "obsidian_incremental_sync_changes",
                changed=notes_changed,
                ingested=notes_ingested,
            )
        else:
            log.debug("obsidian_incremental_sync_no_changes")

        return state

    async def sync_specific_date(self, date_str: str) -> DailyNoteData | None:
        """Sync a specific daily note by date.

        Args:
            date_str: Date in YYYY-MM-DD format.

        Returns:
            Parsed DailyNoteData if the note exists, None otherwise.
        """
        try:
            result = await self._sync_daily_note(date_str)
            if result is not None:
                return self.notes_cache.get(date_str)
        except ObsidianReadError:
            return None
        return None

    async def get_daily_note(self, date_str: str) -> DailyNoteData | None:
        """Get a parsed daily note, using cache if available.

        Falls back to reading from vault if not in cache.

        Args:
            date_str: Date in YYYY-MM-DD format.

        Returns:
            Parsed DailyNoteData or None if note doesn't exist.
        """
        if date_str in self.notes_cache:
            return self.notes_cache[date_str]
        return await self.sync_specific_date(date_str)

    async def scan_date_range(self, start_date: str, end_date: str) -> list[DailyNoteData]:
        """Scan and parse all daily notes in a date range.

        Args:
            start_date: Start date (YYYY-MM-DD, inclusive).
            end_date: End date (YYYY-MM-DD, inclusive).

        Returns:
            List of parsed daily notes found in the range.
        """
        start = datetime.strptime(start_date, "%Y-%m-%d").date()
        end = datetime.strptime(end_date, "%Y-%m-%d").date()

        notes: list[DailyNoteData] = []
        current = start
        while current <= end:
            date_str = current.isoformat()
            note = await self.get_daily_note(date_str)
            if note is not None:
                notes.append(note)
            current += timedelta(days=1)

        return notes

    async def _sync_daily_note(self, date_str: str) -> tuple[bool, bool] | None:
        """Sync a single daily note.

        Args:
            date_str: Date in YYYY-MM-DD format.

        Returns:
            Tuple of (content_changed, was_ingested) if note exists, None if not found.

        Raises:
            ObsidianReadError: Propagated from client.
        """
        path = self._client.build_daily_note_path(date_str)

        # Read note with hash
        content, content_hash = await self._client.read_note_with_hash(path)

        # Check if content changed
        previous_hash = self._hash_cache.get(path)
        content_changed = previous_hash != content_hash

        if not content_changed and date_str in self.notes_cache:
            return (False, False)

        # Parse the note
        parsed = self._parser.parse(date_str, content, path)
        self.notes_cache[date_str] = parsed
        self._hash_cache[path] = content_hash

        # Ingest into RAG if changed and ingestion is available
        was_ingested = False
        if content_changed and self._settings.ingest_on_sync and self._ingestion is not None:
            try:
                result = self._ingestion.ingest_text(
                    text=content,
                    source_name=f"obsidian:{path}",
                    force=True,
                )
                was_ingested = result.chunks_stored > 0
                log.debug(
                    "obsidian_note_ingested",
                    path=path,
                    chunks=result.chunks_stored,
                )
            except Exception as exc:
                log.warning(
                    "obsidian_ingest_error",
                    path=path,
                    error=str(exc),
                )

        return (content_changed, was_ingested)

    async def ingest_all_cached(self) -> int:
        """Force ingest all cached notes into the vector store.

        Processes notes in batches of ``settings.ingest_batch_size`` to avoid
        memory pressure with large vaults.

        Returns:
            Number of notes successfully ingested.
        """
        if self._ingestion is None:
            log.warning("obsidian_ingest_no_pipeline")
            return 0

        batch_size = self._settings.ingest_batch_size
        sorted_notes = sorted(self.notes_cache.items())
        ingested = 0

        for batch_start in range(0, len(sorted_notes), batch_size):
            batch = sorted_notes[batch_start : batch_start + batch_size]
            for _date_str, note in batch:
                try:
                    result = self._ingestion.ingest_text(
                        text=note.raw_content,
                        source_name=f"obsidian:{note.path}",
                        force=True,
                    )
                    if result.chunks_stored > 0:
                        ingested += 1
                except Exception as exc:
                    log.warning(
                        "obsidian_ingest_error",
                        path=note.path,
                        error=str(exc),
                    )

            log.debug(
                "obsidian_ingest_batch_complete",
                batch_start=batch_start,
                batch_size=len(batch),
                ingested_so_far=ingested,
            )

        log.info("obsidian_ingest_all_complete", ingested=ingested, total=len(self.notes_cache))
        return ingested

    def get_cached_dates(self) -> list[str]:
        """Return all cached date strings, sorted chronologically."""
        return sorted(self.notes_cache.keys())

    def clear_cache(self) -> None:
        """Clear all caches (note data + hashes)."""
        self.notes_cache.clear()
        self._hash_cache.clear()
        log.debug("obsidian_cache_cleared")
