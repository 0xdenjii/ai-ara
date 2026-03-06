"""Comprehensive tests for the Obsidian vault integration.

Tests cover:
- ObsidianClient: connection, read, write, append, delete, search, helpers
- DailyNoteParser: frontmatter, sections, tasks, edge cases
- ObsidianSync: boot sync, incremental sync, change detection, caching
- DiaryWriter: entry formatting, emotional logs, conversation summaries, ops log
- BriefingGenerator: morning briefing, end-of-day recap, template fallbacks
- ProtocolTracker: scoring, streaks, trends, missed items
- Settings: ObsidianSettings configuration
- Error classes: all Obsidian error variants
- Types: new Obsidian-related data types
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from ara.config.settings import AraSettings, ObsidianSettings
from ara.obsidian.client import ObsidianClient
from ara.obsidian.daily_briefing import BriefingGenerator
from ara.obsidian.habits import ProtocolTracker
from ara.obsidian.parser import DailyNoteParser
from ara.obsidian.sync import ObsidianSync
from ara.obsidian.writer import DiaryWriter, _get_current_quarter
from ara.utils.errors import (
    ObsidianConnectionError,
    ObsidianError,
    ObsidianParseError,
    ObsidianReadError,
    ObsidianSyncError,
    ObsidianWriteError,
)
from ara.utils.types import (
    DailyNoteData,
    MemorySource,
    MorningBriefing,
    ObsidianFrontmatter,
    ObsidianQuarter,
    ObsidianSyncState,
    ObsidianTask,
    ObsidianTaskStatus,
    ProtocolScore,
)

# ── Test Fixtures ────────────────────────────────────────────────────────────

SAMPLE_DAILY_NOTE = """---
date: 2026-03-06
weight_kg: 72.5
sleep_hrs: 7.5
calories: 2200
workout_focus: push
effort_score: 8.0
checklist_score: 85.0
---

# 2026-03-06

## BRAINSTORM

- Think about ARA's Obsidian integration
- Research MCP protocols for vault access
- Maybe add habit tracking to daily notes

## MISSION CONTROL

```dataviewjs
// scoring logic here
```

## OPERATIONAL LOG

> [!info] Q1 (00:00 - 06:00)
> Slept well, woke at 5:30 AM

> [!info] Q2 (06:00 - 12:00)
> Morning workout — push day
> Worked on ARA codebase

> [!info] Q3 (12:00 - 18:00)
> Lunch meeting with team
> Deep work on Obsidian integration

> [!info] Q4 (18:00 - 24:00)
> Evening review and planning

## THE PROTOCOL

- [x] Wake up before 6 AM
- [x] Workout
- [ ] Read for 30 minutes
- [x] Meditate 10 minutes
- [ ] Journal before bed
- [x] No social media before noon

## INBOX ZERO

- Follow up on PR review
- Buy groceries
"""

MINIMAL_DAILY_NOTE = """---
date: 2026-03-05
---

# 2026-03-05

Not much today.
"""


@pytest.fixture
def settings() -> ObsidianSettings:
    """Create test ObsidianSettings."""
    return ObsidianSettings(
        enabled=True,
        api_url="https://127.0.0.1:27124",
        api_key="test_api_key_12345",
        verify_ssl=False,
        daily_note_folder="10_Journal/Daily",
        daily_note_pattern="{year}/{mm}-{month}/{date}.md",
        inbox_path="00_Inbox/RAM.md",
        sync_interval_s=300,
        sync_on_boot=True,
        sync_lookback_days=7,
        write_to_ara_log=True,
        write_to_ops_log=True,
        voice="first_person",
        morning_briefing_enabled=True,
        morning_briefing_hour=7,
        end_of_day_recap_enabled=True,
        end_of_day_recap_hour=23,
        protocol_tracking_enabled=True,
        protocol_streak_threshold_pct=80.0,
        ingest_on_sync=True,
    )


@pytest.fixture
def mock_client(settings: ObsidianSettings) -> ObsidianClient:
    """Create an ObsidianClient with mocked HTTP client."""
    client = ObsidianClient(settings)
    client._connected = True
    client._client = AsyncMock(spec=httpx.AsyncClient)
    return client


# ── Settings Tests ───────────────────────────────────────────────────────────


class TestObsidianSettings:
    """Test ObsidianSettings configuration."""

    def test_default_settings(self) -> None:
        s = ObsidianSettings()
        assert s.enabled is True
        assert s.api_url == "https://127.0.0.1:27124"
        assert s.verify_ssl is False
        assert s.voice == "first_person"
        assert s.sync_interval_s == 300
        assert s.morning_briefing_hour == 7

    def test_custom_settings(self, settings: ObsidianSettings) -> None:
        assert settings.api_key == "test_api_key_12345"
        assert settings.daily_note_folder == "10_Journal/Daily"
        assert settings.protocol_streak_threshold_pct == 80.0

    def test_settings_in_ara_settings(self) -> None:
        ara = AraSettings()
        assert hasattr(ara, "obsidian")
        assert isinstance(ara.obsidian, ObsidianSettings)

    def test_voice_options(self) -> None:
        for voice in ("first_person", "third_person", "factual"):
            s = ObsidianSettings(voice=voice)
            assert s.voice == voice


# ── Error Tests ──────────────────────────────────────────────────────────────


class TestObsidianErrors:
    """Test Obsidian error hierarchy."""

    def test_base_error(self) -> None:
        err = ObsidianError("test", operation="test_op")
        assert "obsidian" in str(err)
        assert "test_op" in str(err)
        assert err.component == "obsidian"

    def test_connection_error(self) -> None:
        err = ObsidianConnectionError("https://localhost:27124", "refused")
        assert "refused" in str(err)
        assert err.details["base_url"] == "https://localhost:27124"

    def test_read_error(self) -> None:
        err = ObsidianReadError("path/to/note.md", "not found")
        assert "path/to/note.md" in str(err)
        assert err.details["path"] == "path/to/note.md"

    def test_write_error(self) -> None:
        err = ObsidianWriteError("path/to/note.md", "permission denied")
        assert "permission denied" in str(err)

    def test_sync_error(self) -> None:
        err = ObsidianSyncError("sync failed", details={"notes": 5})
        assert "sync" in str(err)
        assert err.details["notes"] == 5

    def test_parse_error(self) -> None:
        err = ObsidianParseError("note.md", "invalid frontmatter")
        assert "note.md" in str(err)

    def test_error_inheritance(self) -> None:
        from ara.utils.errors import AraError

        assert issubclass(ObsidianError, AraError)
        assert issubclass(ObsidianConnectionError, ObsidianError)
        assert issubclass(ObsidianReadError, ObsidianError)
        assert issubclass(ObsidianWriteError, ObsidianError)
        assert issubclass(ObsidianSyncError, ObsidianError)
        assert issubclass(ObsidianParseError, ObsidianError)


# ── Type Tests ───────────────────────────────────────────────────────────────


class TestObsidianTypes:
    """Test new Obsidian-related types."""

    def test_memory_source_obsidian(self) -> None:
        assert MemorySource.OBSIDIAN == "obsidian"
        assert MemorySource.OBSIDIAN in list(MemorySource)

    def test_task_status_enum(self) -> None:
        assert ObsidianTaskStatus.PENDING == "pending"
        assert ObsidianTaskStatus.DONE == "done"
        assert ObsidianTaskStatus.CANCELLED == "cancelled"
        assert ObsidianTaskStatus.IN_PROGRESS == "in_progress"

    def test_quarter_enum(self) -> None:
        assert ObsidianQuarter.Q1 == "q1"
        assert ObsidianQuarter.Q4 == "q4"

    def test_obsidian_task(self) -> None:
        task = ObsidianTask(text="Wake up early", status=ObsidianTaskStatus.DONE)
        assert task.text == "Wake up early"
        assert task.status == ObsidianTaskStatus.DONE

    def test_frontmatter(self) -> None:
        fm = ObsidianFrontmatter(
            date="2026-03-06",
            weight_kg=72.5,
            sleep_hrs=7.5,
            calories=2200,
        )
        assert fm.weight_kg == 72.5
        assert fm.raw == {}

    def test_daily_note_data(self) -> None:
        note = DailyNoteData(date="2026-03-06", path="test/path.md")
        assert note.date == "2026-03-06"
        assert note.brainstorm == ""
        assert note.operational_log == {}

    def test_sync_state(self) -> None:
        state = ObsidianSyncState(notes_synced=5, notes_changed=2)
        assert state.notes_synced == 5
        assert state.errors == []

    def test_morning_briefing(self) -> None:
        briefing = MorningBriefing(date="2026-03-06")
        assert briefing.date == "2026-03-06"
        assert briefing.pending_tasks == []
        assert len(briefing.briefing_id) == 12

    def test_protocol_score(self) -> None:
        score = ProtocolScore(
            date="2026-03-06",
            total_items=6,
            completed_items=4,
            score_pct=66.67,
            missed_items=["Read", "Journal"],
        )
        assert score.total_items == 6
        assert len(score.missed_items) == 2

    def test_frozen_types(self) -> None:
        task = ObsidianTask(text="test")
        with pytest.raises(AttributeError):
            task.text = "changed"  # type: ignore[misc]

        fm = ObsidianFrontmatter()
        with pytest.raises(AttributeError):
            fm.date = "changed"  # type: ignore[misc]


# ── Client Tests ─────────────────────────────────────────────────────────────


class TestObsidianClient:
    """Test ObsidianClient REST API wrapper."""

    def test_init(self, settings: ObsidianSettings) -> None:
        client = ObsidianClient(settings)
        assert not client.connected
        assert client.base_url == "https://127.0.0.1:27124"

    @pytest.mark.asyncio
    async def test_connect_success(self, settings: ObsidianSettings) -> None:
        client = ObsidianClient(settings)
        mock_response = MagicMock()
        mock_response.status_code = 200

        with patch("ara.obsidian.client.httpx.AsyncClient") as mock_cls:
            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(return_value=mock_response)
            mock_cls.return_value = mock_instance

            await client.connect()
            assert client.connected

    @pytest.mark.asyncio
    async def test_connect_auth_failure(self, settings: ObsidianSettings) -> None:
        client = ObsidianClient(settings)
        mock_response = MagicMock()
        mock_response.status_code = 401

        with patch("ara.obsidian.client.httpx.AsyncClient") as mock_cls:
            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(return_value=mock_response)
            mock_cls.return_value = mock_instance

            with pytest.raises(ObsidianConnectionError, match="Authentication"):
                await client.connect()

    @pytest.mark.asyncio
    async def test_connect_refused(self, settings: ObsidianSettings) -> None:
        client = ObsidianClient(settings)

        with patch("ara.obsidian.client.httpx.AsyncClient") as mock_cls:
            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(side_effect=httpx.ConnectError("refused"))
            mock_cls.return_value = mock_instance

            with pytest.raises(ObsidianConnectionError, match="Connection refused"):
                await client.connect()

    @pytest.mark.asyncio
    async def test_connect_timeout(self, settings: ObsidianSettings) -> None:
        client = ObsidianClient(settings)

        with patch("ara.obsidian.client.httpx.AsyncClient") as mock_cls:
            mock_instance = AsyncMock()
            mock_instance.get = AsyncMock(side_effect=httpx.TimeoutException("timed out"))
            mock_cls.return_value = mock_instance

            with pytest.raises(ObsidianConnectionError, match="timed out"):
                await client.connect()

    @pytest.mark.asyncio
    async def test_close(self, mock_client: ObsidianClient) -> None:
        await mock_client.close()
        assert not mock_client.connected

    @pytest.mark.asyncio
    async def test_read_note_success(self, mock_client: ObsidianClient) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "# Hello"
        mock_client._client.get = AsyncMock(return_value=mock_response)

        content = await mock_client.read_note("test.md")
        assert content == "# Hello"

    @pytest.mark.asyncio
    async def test_read_note_not_found(self, mock_client: ObsidianClient) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_client._client.get = AsyncMock(return_value=mock_response)

        with pytest.raises(ObsidianReadError, match="not found"):
            await mock_client.read_note("missing.md")

    @pytest.mark.asyncio
    async def test_read_note_not_connected(self, settings: ObsidianSettings) -> None:
        client = ObsidianClient(settings)
        with pytest.raises(ObsidianConnectionError, match="Not connected"):
            await client.read_note("test.md")

    @pytest.mark.asyncio
    async def test_list_files_dict_format(self, mock_client: ObsidianClient) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"files": ["a.md", "c.md", "b.md"]}
        mock_client._client.get = AsyncMock(return_value=mock_response)

        files = await mock_client.list_files("test/")
        assert files == ["a.md", "b.md", "c.md"]

    @pytest.mark.asyncio
    async def test_list_files_list_format(self, mock_client: ObsidianClient) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = ["a.md", "b.md"]
        mock_client._client.get = AsyncMock(return_value=mock_response)

        files = await mock_client.list_files("/")
        assert files == ["a.md", "b.md"]

    @pytest.mark.asyncio
    async def test_search(self, mock_client: ObsidianClient) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"filename": "note1.md", "matches": "found text"},
        ]
        mock_client._client.post = AsyncMock(return_value=mock_response)

        results = await mock_client.search("test query")
        assert len(results) == 1
        assert results[0]["filename"] == "note1.md"

    @pytest.mark.asyncio
    async def test_write_note(self, mock_client: ObsidianClient) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 204
        mock_client._client.put = AsyncMock(return_value=mock_response)

        await mock_client.write_note("test.md", "# Content")
        mock_client._client.put.assert_called_once()

    @pytest.mark.asyncio
    async def test_append_note(self, mock_client: ObsidianClient) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 204
        mock_client._client.post = AsyncMock(return_value=mock_response)

        await mock_client.append_note("test.md", "\nAppended")
        mock_client._client.post.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_note(self, mock_client: ObsidianClient) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 204
        mock_client._client.delete = AsyncMock(return_value=mock_response)

        await mock_client.delete_note("test.md")
        mock_client._client.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_note_idempotent(self, mock_client: ObsidianClient) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_client._client.delete = AsyncMock(return_value=mock_response)

        # Should not raise — idempotent
        await mock_client.delete_note("already_gone.md")

    @pytest.mark.asyncio
    async def test_note_exists_true(self, mock_client: ObsidianClient) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "content"
        mock_client._client.get = AsyncMock(return_value=mock_response)

        assert await mock_client.note_exists("exists.md") is True

    @pytest.mark.asyncio
    async def test_note_exists_false(self, mock_client: ObsidianClient) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_client._client.get = AsyncMock(return_value=mock_response)

        assert await mock_client.note_exists("missing.md") is False

    @pytest.mark.asyncio
    async def test_read_note_with_hash(self, mock_client: ObsidianClient) -> None:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "hello world"
        mock_client._client.get = AsyncMock(return_value=mock_response)

        content, content_hash = await mock_client.read_note_with_hash("test.md")
        assert content == "hello world"
        expected_hash = hashlib.sha256(b"hello world").hexdigest()
        assert content_hash == expected_hash

    def test_build_daily_note_path(self, mock_client: ObsidianClient) -> None:
        path = mock_client.build_daily_note_path("2026-03-06")
        assert path == "10_Journal/Daily/2026/03-March/2026-03-06.md"

    def test_build_daily_note_path_january(self, mock_client: ObsidianClient) -> None:
        path = mock_client.build_daily_note_path("2026-01-15")
        assert path == "10_Journal/Daily/2026/01-January/2026-01-15.md"

    @pytest.mark.asyncio
    async def test_ensure_daily_note_exists(self, mock_client: ObsidianClient) -> None:
        # Note already exists
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "existing content"
        mock_client._client.get = AsyncMock(return_value=mock_response)

        path = await mock_client.ensure_daily_note("2026-03-06")
        assert "2026-03-06" in path

    @pytest.mark.asyncio
    async def test_ensure_daily_note_creates(self, mock_client: ObsidianClient) -> None:
        # First call: not found. Second call: created.
        not_found = MagicMock()
        not_found.status_code = 404

        created = MagicMock()
        created.status_code = 204

        mock_client._client.get = AsyncMock(return_value=not_found)
        mock_client._client.put = AsyncMock(return_value=created)

        path = await mock_client.ensure_daily_note("2026-03-06")
        assert "2026-03-06" in path
        mock_client._client.put.assert_called_once()

    @pytest.mark.asyncio
    async def test_write_note_connection_error(self, mock_client: ObsidianClient) -> None:
        mock_client._client.put = AsyncMock(side_effect=httpx.ConnectError("lost"))
        with pytest.raises(ObsidianWriteError, match="Connection lost"):
            await mock_client.write_note("test.md", "content")

    @pytest.mark.asyncio
    async def test_read_note_timeout(self, mock_client: ObsidianClient) -> None:
        mock_client._client.get = AsyncMock(side_effect=httpx.TimeoutException("timeout"))
        with pytest.raises(ObsidianReadError, match="timed out"):
            await mock_client.read_note("test.md")


# ── Parser Tests ─────────────────────────────────────────────────────────────


class TestDailyNoteParser:
    """Test DailyNoteParser for daily note extraction."""

    def setup_method(self) -> None:
        self.parser = DailyNoteParser()

    def test_parse_full_note(self) -> None:
        note = self.parser.parse("2026-03-06", SAMPLE_DAILY_NOTE, "test.md")
        assert note.date == "2026-03-06"
        assert note.path == "test.md"
        assert note.content_hash != ""

    def test_parse_frontmatter(self) -> None:
        note = self.parser.parse("2026-03-06", SAMPLE_DAILY_NOTE, "test.md")
        fm = note.frontmatter
        assert fm.date == "2026-03-06"
        assert fm.weight_kg == 72.5
        assert fm.sleep_hrs == 7.5
        assert fm.calories == 2200
        assert fm.workout_focus == "push"
        assert fm.effort_score == 8.0
        assert fm.checklist_score == 85.0

    def test_parse_brainstorm(self) -> None:
        note = self.parser.parse("2026-03-06", SAMPLE_DAILY_NOTE, "test.md")
        assert "ARA's Obsidian integration" in note.brainstorm
        assert "MCP protocols" in note.brainstorm

    def test_parse_operational_log(self) -> None:
        note = self.parser.parse("2026-03-06", SAMPLE_DAILY_NOTE, "test.md")
        assert "Q1" in note.operational_log
        assert "Q2" in note.operational_log
        assert "Q3" in note.operational_log
        assert "Q4" in note.operational_log
        assert "Slept well" in note.operational_log["Q1"]
        assert "Morning workout" in note.operational_log["Q2"]

    def test_parse_protocol_tasks(self) -> None:
        note = self.parser.parse("2026-03-06", SAMPLE_DAILY_NOTE, "test.md")
        protocol = note.protocol_checklist
        assert len(protocol) == 6
        done = [t for t in protocol if t.status == ObsidianTaskStatus.DONE]
        pending = [t for t in protocol if t.status == ObsidianTaskStatus.PENDING]
        assert len(done) == 4  # Wake, Workout, Meditate, No social media
        assert len(pending) == 2  # Read, Journal

    def test_parse_inbox(self) -> None:
        note = self.parser.parse("2026-03-06", SAMPLE_DAILY_NOTE, "test.md")
        assert "Follow up on PR review" in note.inbox_items
        assert "Buy groceries" in note.inbox_items

    def test_parse_all_tasks(self) -> None:
        note = self.parser.parse("2026-03-06", SAMPLE_DAILY_NOTE, "test.md")
        assert len(note.tasks) >= 6  # At least protocol tasks

    def test_parse_minimal_note(self) -> None:
        note = self.parser.parse("2026-03-05", MINIMAL_DAILY_NOTE, "test.md")
        assert note.date == "2026-03-05"
        assert note.brainstorm == ""
        assert note.operational_log == {}
        assert note.protocol_checklist == []

    def test_parse_empty_raises(self) -> None:
        with pytest.raises(ObsidianParseError, match="Empty"):
            self.parser.parse("2026-03-06", "", "empty.md")

    def test_parse_whitespace_only_raises(self) -> None:
        with pytest.raises(ObsidianParseError, match="Empty"):
            self.parser.parse("2026-03-06", "   \n  \n  ", "blank.md")

    def test_parse_no_frontmatter(self) -> None:
        content = "# 2026-03-06\n\nJust a heading and text."
        note = self.parser.parse("2026-03-06", content, "test.md")
        assert note.frontmatter.date == ""  # No frontmatter parsed
        assert note.frontmatter.weight_kg == 0.0

    def test_content_hash_deterministic(self) -> None:
        note1 = self.parser.parse("2026-03-06", SAMPLE_DAILY_NOTE, "test.md")
        note2 = self.parser.parse("2026-03-06", SAMPLE_DAILY_NOTE, "test.md")
        assert note1.content_hash == note2.content_hash

    def test_content_hash_changes(self) -> None:
        note1 = self.parser.parse("2026-03-06", SAMPLE_DAILY_NOTE, "test.md")
        note2 = self.parser.parse("2026-03-06", SAMPLE_DAILY_NOTE + "\nextra", "test.md")
        assert note1.content_hash != note2.content_hash

    def test_parse_task_statuses(self) -> None:
        content = """---
date: 2026-03-06
---

## Tasks
- [x] Done task
- [X] Also done
- [ ] Pending task
- [-] Cancelled task
- [/] In progress task
"""
        note = self.parser.parse("2026-03-06", content, "test.md")
        statuses = {t.text: t.status for t in note.tasks}
        assert statuses["Done task"] == ObsidianTaskStatus.DONE
        assert statuses["Also done"] == ObsidianTaskStatus.DONE
        assert statuses["Pending task"] == ObsidianTaskStatus.PENDING
        assert statuses["Cancelled task"] == ObsidianTaskStatus.CANCELLED
        assert statuses["In progress task"] == ObsidianTaskStatus.IN_PROGRESS

    def test_extract_ara_log(self) -> None:
        content = """# 2026-03-06

## BRAINSTORM

Some notes

## ARA's LOG

- **14:32** — Denji told me about the deployment stress.
- **15:10** — We discussed workout plans.

## INBOX ZERO
"""
        result = self.parser.extract_ara_log(content)
        assert "deployment stress" in result
        assert "workout plans" in result

    def test_extract_ara_log_missing(self) -> None:
        result = self.parser.extract_ara_log(SAMPLE_DAILY_NOTE)
        assert result == ""

    def test_safe_float_edge_cases(self) -> None:
        assert DailyNoteParser._safe_float("") == 0.0
        assert DailyNoteParser._safe_float("abc") == 0.0
        assert DailyNoteParser._safe_float("3.14") == 3.14
        assert DailyNoteParser._safe_float("7") == 7.0

    def test_safe_int_edge_cases(self) -> None:
        assert DailyNoteParser._safe_int("") == 0
        assert DailyNoteParser._safe_int("abc") == 0
        assert DailyNoteParser._safe_int("2200") == 2200
        assert DailyNoteParser._safe_int("2200.5") == 2200

    def test_frontmatter_with_quotes(self) -> None:
        content = """---
date: "2026-03-06"
workout_focus: 'push'
---

# Test
"""
        note = self.parser.parse("2026-03-06", content, "test.md")
        assert note.frontmatter.date == "2026-03-06"
        assert note.frontmatter.workout_focus == "push"


# ── Sync Tests ───────────────────────────────────────────────────────────────


class TestObsidianSync:
    """Test ObsidianSync vault sync engine."""

    @pytest.fixture
    def mock_sync(self, settings: ObsidianSettings) -> ObsidianSync:
        client = AsyncMock()
        client.build_daily_note_path = MagicMock(side_effect=lambda d: f"10_Journal/Daily/{d}.md")
        client.read_note_with_hash = AsyncMock(return_value=(SAMPLE_DAILY_NOTE, "hash123"))
        return ObsidianSync(settings, client)

    @pytest.mark.asyncio
    async def test_boot_sync(self, mock_sync: ObsidianSync) -> None:
        state = await mock_sync.boot_sync()
        assert isinstance(state, ObsidianSyncState)
        assert state.notes_synced > 0
        assert state.last_sync_at is not None

    @pytest.mark.asyncio
    async def test_boot_sync_caches_notes(self, mock_sync: ObsidianSync) -> None:
        await mock_sync.boot_sync()
        assert len(mock_sync.notes_cache) > 0
        dates = mock_sync.get_cached_dates()
        assert len(dates) > 0

    @pytest.mark.asyncio
    async def test_incremental_sync(self, mock_sync: ObsidianSync) -> None:
        state = await mock_sync.incremental_sync()
        assert isinstance(state, ObsidianSyncState)
        # Should check today and yesterday
        assert state.notes_synced <= 2

    @pytest.mark.asyncio
    async def test_change_detection(self, mock_sync: ObsidianSync) -> None:
        # First sync — content is new
        await mock_sync.boot_sync()
        first_state_changed = any(True for d in mock_sync.get_cached_dates())
        assert first_state_changed

        # Second sync — same hash, no change
        state2 = await mock_sync.incremental_sync()
        assert state2.notes_changed == 0

    @pytest.mark.asyncio
    async def test_change_detection_content_changed(self, mock_sync: ObsidianSync) -> None:
        # First sync
        await mock_sync.boot_sync()

        # Change content hash for next read
        mock_sync._client.read_note_with_hash = AsyncMock(
            return_value=(SAMPLE_DAILY_NOTE + "\nnew content", "new_hash_456")
        )

        state = await mock_sync.incremental_sync()
        assert state.notes_changed > 0

    @pytest.mark.asyncio
    async def test_sync_handles_missing_notes(self, settings: ObsidianSettings) -> None:
        client = AsyncMock()
        client.build_daily_note_path = MagicMock(side_effect=lambda d: f"path/{d}.md")
        client.read_note_with_hash = AsyncMock(side_effect=ObsidianReadError("path", "not found"))

        sync = ObsidianSync(settings, client)
        state = await sync.boot_sync()
        # Should handle gracefully — all dates "not found"
        assert state.notes_synced == 0
        assert len(state.errors) == 0

    @pytest.mark.asyncio
    async def test_get_daily_note_from_cache(self, mock_sync: ObsidianSync) -> None:
        await mock_sync.boot_sync()
        dates = mock_sync.get_cached_dates()
        if dates:
            note = await mock_sync.get_daily_note(dates[0])
            assert note is not None
            assert note.date == dates[0]

    @pytest.mark.asyncio
    async def test_get_daily_note_fallback_to_vault(self, mock_sync: ObsidianSync) -> None:
        # Not in cache — should read from vault
        note = await mock_sync.get_daily_note("2026-01-01")
        assert note is not None

    @pytest.mark.asyncio
    async def test_scan_date_range(self, mock_sync: ObsidianSync) -> None:
        notes = await mock_sync.scan_date_range("2026-03-01", "2026-03-03")
        assert len(notes) == 3  # All 3 dates synced

    def test_clear_cache(self, mock_sync: ObsidianSync) -> None:
        mock_sync.notes_cache["2026-03-06"] = DailyNoteData(date="2026-03-06", path="test.md")
        mock_sync.clear_cache()
        assert len(mock_sync.notes_cache) == 0

    @pytest.mark.asyncio
    async def test_sync_with_ingestion(self, settings: ObsidianSettings) -> None:
        client = AsyncMock()
        client.build_daily_note_path = MagicMock(side_effect=lambda d: f"path/{d}.md")
        client.read_note_with_hash = AsyncMock(return_value=(SAMPLE_DAILY_NOTE, "hash123"))

        ingestion = MagicMock()
        ingestion.ingest_text = MagicMock(return_value=MagicMock(chunks_stored=3))

        sync = ObsidianSync(settings, client, ingestion=ingestion)
        state = await sync.boot_sync()
        assert state.notes_ingested > 0

    @pytest.mark.asyncio
    async def test_ingest_all_cached(self, mock_sync: ObsidianSync) -> None:
        # No ingestion pipeline — returns 0
        result = await mock_sync.ingest_all_cached()
        assert result == 0


# ── Writer Tests ─────────────────────────────────────────────────────────────


class TestDiaryWriter:
    """Test DiaryWriter for Obsidian daily note entries."""

    @pytest.fixture
    def writer(self, settings: ObsidianSettings) -> DiaryWriter:
        client = AsyncMock()
        client.ensure_daily_note = AsyncMock(return_value="test/2026-03-06.md")
        client.read_note = AsyncMock(return_value=SAMPLE_DAILY_NOTE)
        client.write_note = AsyncMock()
        return DiaryWriter(settings, client)

    @pytest.mark.asyncio
    async def test_write_entry(self, writer: DiaryWriter) -> None:
        now = datetime(2026, 3, 6, 14, 30)
        await writer.write_entry("Denji is working on the Obsidian integration.", now=now)
        writer._client.write_note.assert_called()

    @pytest.mark.asyncio
    async def test_write_entry_key_highlight(self, writer: DiaryWriter) -> None:
        now = datetime(2026, 3, 6, 14, 30)
        await writer.write_entry("Important thing happened.", is_key_highlight=True, now=now)
        # Should write to both ARA LOG and OPS LOG
        assert writer._client.write_note.call_count >= 1

    @pytest.mark.asyncio
    async def test_write_emotional_log_first_person(self, writer: DiaryWriter) -> None:
        now = datetime(2026, 3, 6, 15, 0)
        await writer.write_emotional_log("stressed", 0.7, "deployment pressure", now=now)
        call_args = writer._client.write_note.call_args_list
        assert len(call_args) >= 1
        # Check that first-person voice was used
        written_content = (
            call_args[0][0][1] if call_args[0][0] else call_args[0][1].get("content", "")
        )
        assert "Denji is feeling" in written_content or "ARA" in str(call_args)

    @pytest.mark.asyncio
    async def test_write_emotional_log_third_person(self, settings: ObsidianSettings) -> None:
        settings_3p = ObsidianSettings(**{**settings.model_dump(), "voice": "third_person"})
        client = AsyncMock()
        client.ensure_daily_note = AsyncMock(return_value="test/2026-03-06.md")
        client.read_note = AsyncMock(return_value=SAMPLE_DAILY_NOTE)
        client.write_note = AsyncMock()

        writer = DiaryWriter(settings_3p, client)
        await writer.write_emotional_log("happy", 0.5, "good news", now=datetime(2026, 3, 6, 10, 0))
        client.write_note.assert_called()

    @pytest.mark.asyncio
    async def test_write_conversation_summary(self, writer: DiaryWriter) -> None:
        now = datetime(2026, 3, 6, 16, 0)
        await writer.write_conversation_summary(
            "We discussed workout plans and sleep optimization.",
            topics=["fitness", "sleep"],
            now=now,
        )
        writer._client.write_note.assert_called()

    @pytest.mark.asyncio
    async def test_write_task_update(self, writer: DiaryWriter) -> None:
        now = datetime(2026, 3, 6, 17, 0)
        await writer.write_task_update("Fix CI pipeline", "completed", now=now)
        writer._client.write_note.assert_called()

    @pytest.mark.asyncio
    async def test_write_custom(self, writer: DiaryWriter) -> None:
        now = datetime(2026, 3, 6, 8, 0)
        await writer.write_custom("System boot completed.", to_ara_log=True, now=now)
        writer._client.write_note.assert_called()

    def test_get_current_quarter(self) -> None:
        assert _get_current_quarter(datetime(2026, 3, 6, 3, 0)) == ObsidianQuarter.Q1
        assert _get_current_quarter(datetime(2026, 3, 6, 9, 0)) == ObsidianQuarter.Q2
        assert _get_current_quarter(datetime(2026, 3, 6, 14, 0)) == ObsidianQuarter.Q3
        assert _get_current_quarter(datetime(2026, 3, 6, 20, 0)) == ObsidianQuarter.Q4

    def test_intensity_word(self) -> None:
        assert DiaryWriter._intensity_word(0.9) == "very"
        assert DiaryWriter._intensity_word(0.6) == "quite"
        assert DiaryWriter._intensity_word(0.4) == "somewhat"
        assert DiaryWriter._intensity_word(0.1) == "mildly"

    def test_summarize_truncation(self, writer: DiaryWriter) -> None:
        short = "Short text"
        assert writer._summarize(short) == short

        long = "A" * 200
        summary = writer._summarize(long, max_len=50)
        assert len(summary) == 50
        assert summary.endswith("...")

    @pytest.mark.asyncio
    async def test_append_to_ara_log_creates_section(self, writer: DiaryWriter) -> None:
        # Content without ARA's LOG section
        writer._client.read_note = AsyncMock(
            return_value="# 2026-03-06\n\n## BRAINSTORM\nSome notes"
        )
        await writer._append_to_ara_log("test.md", "- **14:30** — Test entry")
        call_args = writer._client.write_note.call_args[0]
        assert "## ARA's LOG" in call_args[1]
        assert "Test entry" in call_args[1]


# ── Briefing Tests ───────────────────────────────────────────────────────────


class TestBriefingGenerator:
    """Test BriefingGenerator for morning briefings and EOD recaps."""

    @pytest.fixture
    def mock_gen(self, settings: ObsidianSettings) -> BriefingGenerator:
        sync = AsyncMock(spec=ObsidianSync)
        sync.get_daily_note = AsyncMock(return_value=None)
        writer = AsyncMock(spec=DiaryWriter)
        return BriefingGenerator(settings, sync, writer=writer)

    @pytest.mark.asyncio
    async def test_morning_briefing_no_data(self, mock_gen: BriefingGenerator) -> None:
        now = datetime(2026, 3, 6, 7, 0)
        briefing = await mock_gen.generate_morning_briefing(now=now)
        assert isinstance(briefing, MorningBriefing)
        assert briefing.date == "2026-03-06"
        assert (
            "yesterday" in briefing.yesterday_summary.lower()
            or "no daily note" in briefing.yesterday_summary.lower()
        )

    @pytest.mark.asyncio
    async def test_morning_briefing_with_data(self, settings: ObsidianSettings) -> None:
        note = DailyNoteData(
            date="2026-03-05",
            path="test.md",
            frontmatter=ObsidianFrontmatter(sleep_hrs=7.5, workout_focus="push"),
            brainstorm="Work on ARA",
            tasks=[
                ObsidianTask(text="Fix bug", status=ObsidianTaskStatus.DONE),
                ObsidianTask(text="Write tests", status=ObsidianTaskStatus.PENDING),
            ],
            protocol_checklist=[
                ObsidianTask(text="Workout", status=ObsidianTaskStatus.DONE),
                ObsidianTask(text="Read", status=ObsidianTaskStatus.PENDING),
            ],
        )

        sync = AsyncMock(spec=ObsidianSync)
        sync.get_daily_note = AsyncMock(return_value=note)
        writer = AsyncMock(spec=DiaryWriter)

        gen = BriefingGenerator(settings, sync, writer=writer)
        now = datetime(2026, 3, 6, 7, 0)
        briefing = await gen.generate_morning_briefing(now=now)

        assert "Write tests" in briefing.pending_tasks
        assert briefing.protocol_score == 50.0

    @pytest.mark.asyncio
    async def test_eod_recap_template(self, mock_gen: BriefingGenerator) -> None:
        now = datetime(2026, 3, 6, 23, 0)
        recap = await mock_gen.generate_end_of_day_recap(now=now)
        assert isinstance(recap, str)
        assert len(recap) > 0

    def test_extract_pending_tasks(self, mock_gen: BriefingGenerator) -> None:
        note = DailyNoteData(
            date="2026-03-06",
            path="test.md",
            tasks=[
                ObsidianTask(text="Done", status=ObsidianTaskStatus.DONE),
                ObsidianTask(text="Pending", status=ObsidianTaskStatus.PENDING),
                ObsidianTask(text="InProgress", status=ObsidianTaskStatus.IN_PROGRESS),
            ],
        )
        pending = mock_gen._extract_pending_tasks(note)
        assert "Pending" in pending
        assert "InProgress" in pending
        assert "Done" not in pending

    def test_extract_pending_tasks_none(self, mock_gen: BriefingGenerator) -> None:
        assert mock_gen._extract_pending_tasks(None) == []

    def test_generate_today_suggestions(self, mock_gen: BriefingGenerator) -> None:
        suggestions = mock_gen._generate_today_suggestions(
            pending_tasks=["Fix bug"],
            protocol_score=40.0,
            habit_status="Streak: 5 days",
            mood_context="stable",
        )
        assert len(suggestions) > 0
        assert any("Fix bug" in s for s in suggestions)

    def test_generate_template_briefing(self, mock_gen: BriefingGenerator) -> None:
        text = mock_gen._generate_template_briefing(
            yesterday_summary="Worked on ARA",
            pending_tasks=["Task 1", "Task 2"],
            protocol_score=75.0,
            habit_status="Streak: 3 days",
        )
        assert "Worked on ARA" in text
        assert "75%" in text

    def test_generate_template_eod_recap(self, mock_gen: BriefingGenerator) -> None:
        text = mock_gen._generate_template_eod_recap(
            conversation_count=3,
            completed_tasks=["Fix bug"],
            pending_tasks=["Write tests"],
            protocol_score=80.0,
        )
        assert "3 conversations" in text
        assert "Fix bug" in text


# ── Habit Tracker Tests ──────────────────────────────────────────────────────


class TestProtocolTracker:
    """Test ProtocolTracker for protocol scoring and streaks."""

    @pytest.fixture
    def tracker(self, settings: ObsidianSettings) -> ProtocolTracker:
        sync = AsyncMock(spec=ObsidianSync)
        sync.get_daily_note = AsyncMock(return_value=None)
        return ProtocolTracker(settings, sync)

    @pytest.mark.asyncio
    async def test_score_empty_date(self, tracker: ProtocolTracker) -> None:
        score = await tracker.score_date("2026-03-06")
        assert score.date == "2026-03-06"
        assert score.total_items == 0
        assert score.score_pct == 0.0

    @pytest.mark.asyncio
    async def test_score_with_data(self, settings: ObsidianSettings) -> None:
        note = DailyNoteData(
            date="2026-03-06",
            path="test.md",
            protocol_checklist=[
                ObsidianTask(text="Workout", status=ObsidianTaskStatus.DONE),
                ObsidianTask(text="Read", status=ObsidianTaskStatus.DONE),
                ObsidianTask(text="Meditate", status=ObsidianTaskStatus.DONE),
                ObsidianTask(text="Journal", status=ObsidianTaskStatus.PENDING),
            ],
        )
        sync = AsyncMock(spec=ObsidianSync)
        sync.get_daily_note = AsyncMock(return_value=note)

        tracker = ProtocolTracker(settings, sync)
        score = await tracker.score_date("2026-03-06")
        assert score.total_items == 4
        assert score.completed_items == 3
        assert score.score_pct == 75.0
        assert "Journal" in score.missed_items

    @pytest.mark.asyncio
    async def test_score_cached(self, tracker: ProtocolTracker) -> None:
        # Score once
        await tracker.score_date("2026-03-06")
        # Score again — should use cache
        score = await tracker.score_date("2026-03-06")
        assert score.date == "2026-03-06"
        # sync.get_daily_note should only be called once
        assert tracker._sync.get_daily_note.call_count == 1

    def test_streak_empty(self, tracker: ProtocolTracker) -> None:
        assert tracker.get_current_streak() == 0

    def test_streak_calculation(self, tracker: ProtocolTracker) -> None:
        # Manually populate scores — 3 consecutive days above 80%
        tracker.scores["2026-03-06"] = ProtocolScore(
            date="2026-03-06", total_items=5, completed_items=5, score_pct=100.0
        )
        tracker.scores["2026-03-05"] = ProtocolScore(
            date="2026-03-05", total_items=5, completed_items=4, score_pct=80.0
        )
        tracker.scores["2026-03-04"] = ProtocolScore(
            date="2026-03-04", total_items=5, completed_items=5, score_pct=100.0
        )
        tracker.scores["2026-03-03"] = ProtocolScore(
            date="2026-03-03", total_items=5, completed_items=2, score_pct=40.0
        )

        # Streak should be 3 (2026-03-04, 2026-03-05, 2026-03-06)
        assert tracker.get_current_streak() == 3

    def test_streak_broken(self, tracker: ProtocolTracker) -> None:
        tracker.scores["2026-03-06"] = ProtocolScore(
            date="2026-03-06", total_items=5, completed_items=2, score_pct=40.0
        )
        assert tracker.get_current_streak() == 0

    @pytest.mark.asyncio
    async def test_weekly_trend(self, tracker: ProtocolTracker) -> None:
        trend = await tracker.get_weekly_trend("2026-03-06")
        assert len(trend) == 7  # 7 days

    @pytest.mark.asyncio
    async def test_weekly_average(self, tracker: ProtocolTracker) -> None:
        avg = await tracker.get_weekly_average("2026-03-06")
        assert avg == 0.0  # No scored days (all empty)

    @pytest.mark.asyncio
    async def test_weekly_average_with_data(self, settings: ObsidianSettings) -> None:
        sync = AsyncMock(spec=ObsidianSync)
        sync.get_daily_note = AsyncMock(return_value=None)
        tracker = ProtocolTracker(settings, sync)

        # Manually add scores
        tracker.scores["2026-03-06"] = ProtocolScore(
            date="2026-03-06", total_items=5, completed_items=5, score_pct=100.0
        )
        tracker.scores["2026-03-05"] = ProtocolScore(
            date="2026-03-05", total_items=5, completed_items=3, score_pct=60.0
        )

        avg = await tracker.get_weekly_average("2026-03-06")
        assert avg == 80.0  # (100 + 60) / 2

    @pytest.mark.asyncio
    async def test_most_missed_items(self, settings: ObsidianSettings) -> None:
        from datetime import datetime, timedelta

        sync = AsyncMock(spec=ObsidianSync)
        sync.get_daily_note = AsyncMock(return_value=None)
        tracker = ProtocolTracker(settings, sync)

        # Use relative dates so test doesn't break as calendar advances
        today = datetime.now().date().isoformat()
        yesterday = (datetime.now().date() - timedelta(days=1)).isoformat()

        # Add scores with missed items
        tracker.scores[today] = ProtocolScore(
            date=today,
            total_items=3,
            completed_items=1,
            score_pct=33.3,
            missed_items=["Read", "Journal"],
        )
        tracker.scores[yesterday] = ProtocolScore(
            date=yesterday,
            total_items=3,
            completed_items=2,
            score_pct=66.7,
            missed_items=["Read"],
        )

        missed = await tracker.get_most_missed_items(days=2)
        assert missed[0][0] == "Read"
        assert missed[0][1] == 2

    def test_get_stats_empty(self, tracker: ProtocolTracker) -> None:
        stats = tracker.get_stats()
        assert stats["avg_score"] == 0.0
        assert stats["total_days_scored"] == 0

    def test_get_stats_with_data(self, tracker: ProtocolTracker) -> None:
        tracker.scores["2026-03-06"] = ProtocolScore(
            date="2026-03-06", total_items=5, completed_items=5, score_pct=100.0
        )
        tracker.scores["2026-03-05"] = ProtocolScore(
            date="2026-03-05", total_items=5, completed_items=3, score_pct=60.0
        )
        stats = tracker.get_stats()
        assert stats["avg_score"] == 80.0
        assert stats["total_days_scored"] == 2
        assert stats["best_score"] == 100.0
        assert stats["worst_score"] == 60.0

    def test_clear_cache(self, tracker: ProtocolTracker) -> None:
        tracker.scores["2026-03-06"] = ProtocolScore(date="2026-03-06")
        tracker.clear_cache()
        assert len(tracker.scores) == 0

    @pytest.mark.asyncio
    async def test_score_range(self, tracker: ProtocolTracker) -> None:
        scores = await tracker.score_range("2026-03-04", "2026-03-06")
        assert len(scores) == 3


# ── Scheduler Integration Tests ──────────────────────────────────────────────


class TestSchedulerObsidianIntegration:
    """Test that ObsidianSync is properly integrated into ProactiveScheduler."""

    def test_scheduler_accepts_obsidian_params(self) -> None:
        from ara.config.settings import ProactiveSettings
        from ara.proactive.notifications import NotificationQueue
        from ara.proactive.scheduler import ProactiveScheduler

        queue = NotificationQueue(ProactiveSettings())
        scheduler = ProactiveScheduler(
            ProactiveSettings(),
            notification_queue=queue,
            obsidian_sync=None,
            obsidian_settings=None,
            briefing_generator=None,
        )
        assert scheduler._obsidian_sync is None
        assert scheduler._obsidian_settings is None
        assert scheduler._briefing is None

    def test_scheduler_stats_include_obsidian(self) -> None:
        from ara.config.settings import ProactiveSettings
        from ara.proactive.notifications import NotificationQueue
        from ara.proactive.scheduler import ProactiveScheduler

        queue = NotificationQueue(ProactiveSettings())
        scheduler = ProactiveScheduler(
            ProactiveSettings(),
            notification_queue=queue,
        )
        stats = scheduler.get_stats()
        assert "last_obsidian_sync" in stats
        assert "last_morning_briefing" in stats
        assert "last_eod_recap" in stats

    @pytest.mark.asyncio
    async def test_check_obsidian_sync_no_sync_engine(self) -> None:
        from ara.config.settings import ProactiveSettings
        from ara.proactive.notifications import NotificationQueue
        from ara.proactive.scheduler import ProactiveScheduler

        queue = NotificationQueue(ProactiveSettings())
        scheduler = ProactiveScheduler(
            ProactiveSettings(),
            notification_queue=queue,
        )
        # Should be a no-op when obsidian_sync is None
        await scheduler._check_obsidian_sync(datetime.now())

    @pytest.mark.asyncio
    async def test_check_morning_briefing_disabled(self) -> None:
        from ara.config.settings import ProactiveSettings
        from ara.proactive.notifications import NotificationQueue
        from ara.proactive.scheduler import ProactiveScheduler

        queue = NotificationQueue(ProactiveSettings())
        scheduler = ProactiveScheduler(
            ProactiveSettings(),
            notification_queue=queue,
        )
        # Should be a no-op when briefing_generator is None
        await scheduler._check_morning_briefing(datetime.now())


# ── Package Import Tests ─────────────────────────────────────────────────────


class TestObsidianPackage:
    """Test that all Obsidian package exports work."""

    def test_imports(self) -> None:
        from ara.obsidian import (
            BriefingGenerator,
            DailyNoteParser,
            DiaryWriter,
            ObsidianClient,
            ObsidianSync,
            ProtocolTracker,
        )

        assert BriefingGenerator is not None
        assert DailyNoteParser is not None
        assert DiaryWriter is not None
        assert ObsidianClient is not None
        assert ObsidianSync is not None
        assert ProtocolTracker is not None
