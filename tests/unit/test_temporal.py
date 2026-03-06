"""Tests for the temporal metadata extraction module."""

from __future__ import annotations

from datetime import datetime

from ara.memory.temporal import TemporalExtractor

# ── Basic Extraction ───────────────────────────────────────────────────────


class TestTemporalExtractorBasic:
    def test_empty_text(self) -> None:
        ext = TemporalExtractor()
        meta = ext.extract("")
        assert meta.has_temporal_content is False
        assert meta.references == []

    def test_no_temporal_content(self) -> None:
        ext = TemporalExtractor()
        meta = ext.extract("The cat sat on the mat.")
        assert meta.has_temporal_content is False

    def test_reference_date_default(self) -> None:
        ext = TemporalExtractor()
        assert ext.reference_date is not None

    def test_reference_date_custom(self) -> None:
        ref = datetime(2024, 6, 15)
        ext = TemporalExtractor(reference_date=ref)
        assert ext.reference_date == ref


# ── Absolute Date Extraction ──────────────────────────────────────────────


class TestAbsoluteDates:
    def test_iso_date(self) -> None:
        ext = TemporalExtractor()
        meta = ext.extract("Meeting on 2024-06-15 was productive.")
        assert meta.has_temporal_content is True
        assert len(meta.absolute_dates) == 1
        assert meta.absolute_dates[0] == datetime(2024, 6, 15)

    def test_slash_date(self) -> None:
        ext = TemporalExtractor()
        meta = ext.extract("Event on 2024/03/20 went well.")
        assert len(meta.absolute_dates) == 1
        assert meta.absolute_dates[0] == datetime(2024, 3, 20)

    def test_named_month_date(self) -> None:
        ext = TemporalExtractor()
        meta = ext.extract("I started on June 15, 2024.")
        assert len(meta.absolute_dates) == 1
        assert meta.absolute_dates[0] == datetime(2024, 6, 15)

    def test_abbreviated_month(self) -> None:
        ext = TemporalExtractor()
        meta = ext.extract("Due by Jan 5, 2025.")
        assert len(meta.absolute_dates) == 1
        assert meta.absolute_dates[0] == datetime(2025, 1, 5)

    def test_multiple_dates(self) -> None:
        ext = TemporalExtractor()
        meta = ext.extract("From 2024-01-01 to 2024-12-31.")
        assert len(meta.absolute_dates) == 2

    def test_invalid_date_ignored(self) -> None:
        ext = TemporalExtractor()
        meta = ext.extract("Version 2024-13-45 released.")
        # Month 13 and day 45 are invalid — should be skipped
        assert len(meta.absolute_dates) == 0


# ── Relative Date Extraction ─────────────────────────────────────────────


class TestRelativeDates:
    def test_yesterday(self) -> None:
        ref = datetime(2024, 6, 15)
        ext = TemporalExtractor(reference_date=ref)
        meta = ext.extract("Yesterday was tough.")
        assert "yesterday" in meta.relative_references
        # Should resolve to June 14
        rel_refs = [r for r in meta.references if r.label == "yesterday"]
        assert rel_refs[0].resolved_date == datetime(2024, 6, 14)

    def test_today(self) -> None:
        ref = datetime(2024, 6, 15)
        ext = TemporalExtractor(reference_date=ref)
        meta = ext.extract("Today was great!")
        assert "today" in meta.relative_references

    def test_tomorrow(self) -> None:
        ref = datetime(2024, 6, 15)
        ext = TemporalExtractor(reference_date=ref)
        meta = ext.extract("I have a meeting tomorrow.")
        assert "tomorrow" in meta.relative_references

    def test_last_week(self) -> None:
        ext = TemporalExtractor()
        meta = ext.extract("Last week I went hiking.")
        assert "last_week" in meta.relative_references

    def test_this_morning(self) -> None:
        ext = TemporalExtractor()
        meta = ext.extract("This morning I meditated.")
        assert "this_morning" in meta.relative_references

    def test_n_days_ago(self) -> None:
        ref = datetime(2024, 6, 15)
        ext = TemporalExtractor(reference_date=ref)
        meta = ext.extract("3 days ago I started the project.")
        refs = [r for r in meta.references if "ago" in r.label]
        assert len(refs) == 1
        assert refs[0].resolved_date == datetime(2024, 6, 12)

    def test_n_weeks_ago(self) -> None:
        ref = datetime(2024, 6, 15)
        ext = TemporalExtractor(reference_date=ref)
        meta = ext.extract("2 weeks ago we had a party.")
        refs = [r for r in meta.references if "ago" in r.label]
        assert len(refs) == 1
        assert refs[0].resolved_date == datetime(2024, 6, 1)

    def test_in_n_days(self) -> None:
        ref = datetime(2024, 6, 15)
        ext = TemporalExtractor(reference_date=ref)
        meta = ext.extract("Deadline is in 5 days.")
        refs = [r for r in meta.references if r.label.startswith("in_")]
        assert len(refs) == 1
        assert refs[0].resolved_date == datetime(2024, 6, 20)


# ── Time of Day ───────────────────────────────────────────────────────────


class TestTimeOfDay:
    def test_morning(self) -> None:
        ext = TemporalExtractor()
        meta = ext.extract("I went for a walk in the morning.")
        assert "morning" in meta.time_of_day

    def test_evening(self) -> None:
        ext = TemporalExtractor()
        meta = ext.extract("Had dinner in the evening.")
        assert "evening" in meta.time_of_day

    def test_multiple_times_of_day(self) -> None:
        ext = TemporalExtractor()
        meta = ext.extract("Morning was busy, afternoon was relaxing, night was quiet.")
        assert "morning" in meta.time_of_day
        assert "afternoon" in meta.time_of_day
        assert "night" in meta.time_of_day

    def test_deduplicates(self) -> None:
        ext = TemporalExtractor()
        meta = ext.extract("Morning meeting. Another morning task.")
        # Should only have one "morning" entry
        assert meta.time_of_day.count("morning") == 1


# ── Days of Week ──────────────────────────────────────────────────────────


class TestDaysOfWeek:
    def test_single_day(self) -> None:
        ext = TemporalExtractor()
        meta = ext.extract("See you on Monday.")
        assert "monday" in meta.days_of_week

    def test_multiple_days(self) -> None:
        ext = TemporalExtractor()
        meta = ext.extract("Monday was hectic, but Friday was great.")
        assert "monday" in meta.days_of_week
        assert "friday" in meta.days_of_week

    def test_case_insensitive(self) -> None:
        ext = TemporalExtractor()
        meta = ext.extract("TUESDAY was interesting.")
        assert "tuesday" in meta.days_of_week


# ── extract_as_dict ──────────────────────────────────────────────────────


class TestExtractAsDict:
    def test_empty_text(self) -> None:
        ext = TemporalExtractor()
        result = ext.extract_as_dict("")
        assert result["has_temporal_content"] is False

    def test_with_content(self) -> None:
        ext = TemporalExtractor()
        result = ext.extract_as_dict("Yesterday morning I went to the park on 2024-06-15.")
        assert result["has_temporal_content"] is True
        assert "relative_references" in result
        assert "time_of_day" in result
        assert "absolute_dates" in result

    def test_dict_format_for_metadata(self) -> None:
        ext = TemporalExtractor()
        result = ext.extract_as_dict("Today was great!")
        # Should be a plain dict suitable for chunk metadata
        assert isinstance(result, dict)
        assert isinstance(result["has_temporal_content"], bool)
