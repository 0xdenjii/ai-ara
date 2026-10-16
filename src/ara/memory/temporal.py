"""Temporal metadata extraction from diary entries and text content.

Extracts dates, time references, relative dates, and temporal markers
from text using regex-based pattern matching. No LLM or external
dependencies required — runs entirely on CPU with zero overhead.

Extracted temporal data is attached as metadata on MemoryChunks for
time-aware retrieval (e.g., "what happened last Tuesday?").

All patterns are designed for English diary-style writing.
"""

from __future__ import annotations

import calendar
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from ara.utils.logging import get_logger

log = get_logger("temporal")

# ── Date Patterns ───────────────────────────────────────────────────────────

# ISO/standard date formats: 2024-06-15, 2024/06/15, 15/06/2024, June 15 2024
_ISO_DATE = re.compile(r"\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b")

_SLASH_DATE_DMY = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b")

_MONTH_NAMES = (
    "january|february|march|april|may|june|"
    "july|august|september|october|november|december|"
    "jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec"
)

_NAMED_DATE = re.compile(
    rf"\b({_MONTH_NAMES})\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s*(\d{{4}})?\b",
    re.IGNORECASE,
)

# ── Time Patterns ───────────────────────────────────────────────────────────

_TIME_PATTERN = re.compile(
    r"\b(\d{1,2}):(\d{2})\s*(?:([ap]m))?\b"
    r"|\b(\d{1,2})\s*([ap]m)\b",
    re.IGNORECASE,
)

# ── Relative Date Patterns ─────────────────────────────────────────────────

_RELATIVE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\btoday\b", re.IGNORECASE), "today"),
    (re.compile(r"\byesterday\b", re.IGNORECASE), "yesterday"),
    (re.compile(r"\btomorrow\b", re.IGNORECASE), "tomorrow"),
    (re.compile(r"\blast\s+night\b", re.IGNORECASE), "last_night"),
    (re.compile(r"\bthis\s+morning\b", re.IGNORECASE), "this_morning"),
    (re.compile(r"\bthis\s+evening\b", re.IGNORECASE), "this_evening"),
    (re.compile(r"\bthis\s+afternoon\b", re.IGNORECASE), "this_afternoon"),
    (re.compile(r"\blast\s+week\b", re.IGNORECASE), "last_week"),
    (re.compile(r"\bthis\s+week\b", re.IGNORECASE), "this_week"),
    (re.compile(r"\bnext\s+week\b", re.IGNORECASE), "next_week"),
    (re.compile(r"\blast\s+month\b", re.IGNORECASE), "last_month"),
    (re.compile(r"\bthis\s+month\b", re.IGNORECASE), "this_month"),
    (re.compile(r"\bnext\s+month\b", re.IGNORECASE), "next_month"),
    (re.compile(r"\blast\s+year\b", re.IGNORECASE), "last_year"),
    (re.compile(r"\bthis\s+year\b", re.IGNORECASE), "this_year"),
    (re.compile(r"\bnext\s+year\b", re.IGNORECASE), "next_year"),
]

# "N days/weeks/months ago", "in N days/weeks/months"
_N_AGO_PATTERN = re.compile(
    r"\b(\d+)\s+(days?|weeks?|months?|years?)\s+ago\b",
    re.IGNORECASE,
)

_IN_N_PATTERN = re.compile(
    r"\bin\s+(\d+)\s+(days?|weeks?|months?|years?)\b",
    re.IGNORECASE,
)

# Day of week references: "last Monday", "next Friday", "on Tuesday"
_DAY_OF_WEEK = re.compile(
    r"\b(?:last|next|on|this)?\s*"
    r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    re.IGNORECASE,
)

# ── Time-of-day markers ────────────────────────────────────────────────────

_TIME_OF_DAY_MARKERS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b(?:in the\s+)?morning\b", re.IGNORECASE), "morning"),
    (re.compile(r"\b(?:in the\s+)?afternoon\b", re.IGNORECASE), "afternoon"),
    (re.compile(r"\b(?:in the\s+)?evening\b", re.IGNORECASE), "evening"),
    (re.compile(r"\b(?:at\s+)?night\b", re.IGNORECASE), "night"),
    (re.compile(r"\b(?:at\s+)?dawn\b", re.IGNORECASE), "dawn"),
    (re.compile(r"\b(?:at\s+)?dusk\b", re.IGNORECASE), "dusk"),
    (re.compile(r"\b(?:at\s+)?midnight\b", re.IGNORECASE), "midnight"),
    (re.compile(r"\b(?:at\s+)?noon\b", re.IGNORECASE), "noon"),
]

# ── Month name to number mapping ───────────────────────────────────────────

_MONTH_MAP: dict[str, int] = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sep": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}


@dataclass(frozen=True)
class TemporalReference:
    """A single temporal reference found in text."""

    text: str  # The matched text span
    ref_type: (
        str  # "absolute_date", "relative_date", "time", "time_of_day", "day_of_week", "duration"
    )
    resolved_date: datetime | None = None  # Resolved to absolute date if possible
    label: str = ""  # Normalized label (e.g., "yesterday", "last_week", "morning")


@dataclass(frozen=True)
class TemporalMetadata:
    """All temporal information extracted from a piece of text."""

    references: list[TemporalReference] = field(default_factory=list)
    absolute_dates: list[datetime] = field(default_factory=list)
    relative_references: list[str] = field(default_factory=list)
    time_of_day: list[str] = field(default_factory=list)
    days_of_week: list[str] = field(default_factory=list)
    has_temporal_content: bool = False


class TemporalExtractor:
    """Extract temporal metadata from text using regex patterns.

    All extraction is rule-based (no ML/LLM), runs on CPU in <1ms
    for typical diary entries.

    Usage:
        extractor = TemporalExtractor()
        metadata = extractor.extract("Yesterday I went to the park at 3pm.")
        # metadata.relative_references == ["yesterday"]
        # metadata.time_of_day == ["afternoon"]
    """

    def __init__(self, *, reference_date: datetime | None = None) -> None:
        """Initialize with an optional reference date for resolving relative dates.

        Args:
            reference_date: Base date for resolving "yesterday", "last week", etc.
                           Defaults to current date/time.
        """
        self._reference_date = reference_date or datetime.now()

    @property
    def reference_date(self) -> datetime:
        """The reference date used for resolving relative dates."""
        return self._reference_date

    @reference_date.setter
    def reference_date(self, value: datetime) -> None:
        """Update the reference date."""
        self._reference_date = value

    def extract(self, text: str) -> TemporalMetadata:
        """Extract all temporal metadata from text.

        Args:
            text: The text to analyze.

        Returns:
            TemporalMetadata with all found references.
        """
        if not text.strip():
            return TemporalMetadata()

        references: list[TemporalReference] = []

        # Extract absolute dates
        abs_dates = self._extract_absolute_dates(text)
        references.extend(abs_dates)

        # Extract relative date references
        rel_refs = self._extract_relative_dates(text)
        references.extend(rel_refs)

        # Extract time references
        time_refs = self._extract_times(text)
        references.extend(time_refs)

        # Extract time-of-day markers
        tod_refs = self._extract_time_of_day(text)
        references.extend(tod_refs)

        # Extract day-of-week references
        dow_refs = self._extract_days_of_week(text)
        references.extend(dow_refs)

        # Extract duration references
        dur_refs = self._extract_durations(text)
        references.extend(dur_refs)

        # Build metadata
        absolute_dates = [
            r.resolved_date
            for r in references
            if r.resolved_date is not None and r.ref_type == "absolute_date"
        ]
        relative_references = [r.label for r in references if r.ref_type == "relative_date"]
        time_of_day = [r.label for r in references if r.ref_type == "time_of_day"]
        days_of_week = [r.label for r in references if r.ref_type == "day_of_week"]

        return TemporalMetadata(
            references=references,
            absolute_dates=absolute_dates,
            relative_references=relative_references,
            time_of_day=time_of_day,
            days_of_week=days_of_week,
            has_temporal_content=len(references) > 0,
        )

    def extract_as_dict(self, text: str) -> dict[str, object]:
        """Extract temporal metadata as a flat dictionary for chunk metadata.

        Returns a dict suitable for merging into MemoryChunk.metadata.
        """
        meta = self.extract(text)
        result: dict[str, object] = {
            "has_temporal_content": meta.has_temporal_content,
        }

        if meta.absolute_dates:
            result["absolute_dates"] = [d.isoformat() for d in meta.absolute_dates]

        if meta.relative_references:
            result["relative_references"] = meta.relative_references

        if meta.time_of_day:
            result["time_of_day"] = meta.time_of_day

        if meta.days_of_week:
            result["days_of_week"] = meta.days_of_week

        return result

    def _extract_absolute_dates(self, text: str) -> list[TemporalReference]:
        """Extract absolute date references (ISO, named months, etc.)."""
        results: list[TemporalReference] = []

        # ISO dates: 2024-06-15
        for match in _ISO_DATE.finditer(text):
            try:
                dt = datetime(
                    year=int(match.group(1)),
                    month=int(match.group(2)),
                    day=int(match.group(3)),
                )
                results.append(
                    TemporalReference(
                        text=match.group(0),
                        ref_type="absolute_date",
                        resolved_date=dt,
                        label=dt.strftime("%Y-%m-%d"),
                    )
                )
            except ValueError:
                continue

        # DD/MM/YYYY dates
        for match in _SLASH_DATE_DMY.finditer(text):
            try:
                dt = datetime(
                    year=int(match.group(3)),
                    month=int(match.group(2)),
                    day=int(match.group(1)),
                )
                results.append(
                    TemporalReference(
                        text=match.group(0),
                        ref_type="absolute_date",
                        resolved_date=dt,
                        label=dt.strftime("%Y-%m-%d"),
                    )
                )
            except ValueError:
                continue

        # Named dates: "June 15, 2024" or "Jun 15"
        for match in _NAMED_DATE.finditer(text):
            month_str = match.group(1).lower()
            month_num = _MONTH_MAP.get(month_str)
            if month_num is None:
                continue

            day = int(match.group(2))
            year_str = match.group(3)
            year = int(year_str) if year_str else self._reference_date.year

            try:
                dt = datetime(year=year, month=month_num, day=day)
                results.append(
                    TemporalReference(
                        text=match.group(0),
                        ref_type="absolute_date",
                        resolved_date=dt,
                        label=dt.strftime("%Y-%m-%d"),
                    )
                )
            except ValueError:
                continue

        return results

    def _extract_relative_dates(self, text: str) -> list[TemporalReference]:
        """Extract relative date references (yesterday, last week, etc.)."""
        results: list[TemporalReference] = []

        for pattern, label in _RELATIVE_PATTERNS:
            for match in pattern.finditer(text):
                resolved = self._resolve_relative(label)
                results.append(
                    TemporalReference(
                        text=match.group(0),
                        ref_type="relative_date",
                        resolved_date=resolved,
                        label=label,
                    )
                )

        # "N days/weeks ago" pattern
        for match in _N_AGO_PATTERN.finditer(text):
            amount = int(match.group(1))
            unit = match.group(2).lower().rstrip("s")
            resolved = self._resolve_n_ago(amount, unit)
            results.append(
                TemporalReference(
                    text=match.group(0),
                    ref_type="relative_date",
                    resolved_date=resolved,
                    label=f"{amount}_{unit}_ago",
                )
            )

        # "in N days/weeks" pattern
        for match in _IN_N_PATTERN.finditer(text):
            amount = int(match.group(1))
            unit = match.group(2).lower().rstrip("s")
            resolved = self._resolve_n_ahead(amount, unit)
            results.append(
                TemporalReference(
                    text=match.group(0),
                    ref_type="relative_date",
                    resolved_date=resolved,
                    label=f"in_{amount}_{unit}",
                )
            )

        return results

    def _extract_times(self, text: str) -> list[TemporalReference]:
        """Extract time references (3pm, 14:30, etc.)."""
        return [
            TemporalReference(
                text=match.group(0),
                ref_type="time",
                label=match.group(0).strip().lower(),
            )
            for match in _TIME_PATTERN.finditer(text)
        ]

    def _extract_time_of_day(self, text: str) -> list[TemporalReference]:
        """Extract time-of-day markers (morning, evening, etc.)."""
        results: list[TemporalReference] = []
        seen_labels: set[str] = set()

        for pattern, label in _TIME_OF_DAY_MARKERS:
            for match in pattern.finditer(text):
                if label not in seen_labels:
                    results.append(
                        TemporalReference(
                            text=match.group(0),
                            ref_type="time_of_day",
                            label=label,
                        )
                    )
                    seen_labels.add(label)

        return results

    def _extract_days_of_week(self, text: str) -> list[TemporalReference]:
        """Extract day-of-week references."""
        results: list[TemporalReference] = []
        seen: set[str] = set()

        for match in _DAY_OF_WEEK.finditer(text):
            day = match.group(1).lower()
            if day not in seen:
                results.append(
                    TemporalReference(
                        text=match.group(0),
                        ref_type="day_of_week",
                        label=day,
                    )
                )
                seen.add(day)

        return results

    def _extract_durations(self, text: str) -> list[TemporalReference]:
        """Extract duration references (for N ago and in N ahead patterns).

        These are already captured in _extract_relative_dates, so this
        method captures standalone duration mentions like 'for 3 hours'.
        """
        duration_pattern = re.compile(
            r"\bfor\s+(\d+)\s+(hours?|minutes?|days?|weeks?|months?)\b",
            re.IGNORECASE,
        )

        return [
            TemporalReference(
                text=match.group(0),
                ref_type="duration",
                label=f"{match.group(1)}_{match.group(2).lower().rstrip('s')}",
            )
            for match in duration_pattern.finditer(text)
        ]

    def _resolve_relative(self, label: str) -> datetime | None:
        """Resolve a relative date label to an absolute datetime."""
        ref = self._reference_date
        mapping: dict[str, datetime] = {
            "today": ref,
            "yesterday": ref - timedelta(days=1),
            "tomorrow": ref + timedelta(days=1),
            "last_night": ref - timedelta(days=1),
            "this_morning": ref.replace(hour=8, minute=0, second=0, microsecond=0),
            "this_afternoon": ref.replace(hour=14, minute=0, second=0, microsecond=0),
            "this_evening": ref.replace(hour=18, minute=0, second=0, microsecond=0),
            "last_week": ref - timedelta(weeks=1),
            "this_week": ref,
            "next_week": ref + timedelta(weeks=1),
            "last_month": self._safe_month_replace(
                ref,
                ref.month - 1 if ref.month > 1 else 12,
                ref.year if ref.month > 1 else ref.year - 1,
            ),
            "this_month": ref,
            "next_month": self._safe_month_replace(
                ref,
                ref.month + 1 if ref.month < 12 else 1,
                ref.year if ref.month < 12 else ref.year + 1,
            ),
            "last_year": ref.replace(year=ref.year - 1),
            "this_year": ref,
            "next_year": ref.replace(year=ref.year + 1),
        }
        return mapping.get(label)

    def _resolve_n_ago(self, amount: int, unit: str) -> datetime | None:
        """Resolve 'N units ago' to an absolute datetime."""
        ref = self._reference_date
        if unit == "day":
            return ref - timedelta(days=amount)
        if unit == "week":
            return ref - timedelta(weeks=amount)
        if unit == "month":
            new_month = ref.month - amount
            new_year = ref.year
            while new_month < 1:
                new_month += 12
                new_year -= 1
            return self._safe_month_replace(ref, new_month, new_year)
        if unit == "year":
            return ref.replace(year=ref.year - amount)
        return None

    def _resolve_n_ahead(self, amount: int, unit: str) -> datetime | None:
        """Resolve 'in N units' to an absolute datetime."""
        ref = self._reference_date
        if unit == "day":
            return ref + timedelta(days=amount)
        if unit == "week":
            return ref + timedelta(weeks=amount)
        if unit == "month":
            new_month = ref.month + amount
            new_year = ref.year
            while new_month > 12:
                new_month -= 12
                new_year += 1
            return self._safe_month_replace(ref, new_month, new_year)
        if unit == "year":
            return ref.replace(year=ref.year + amount)
        return None

    @staticmethod
    def _safe_month_replace(ref: datetime, month: int, year: int) -> datetime:
        """Replace month/year on a datetime, clamping the day to the month's max.

        Prevents crashes like March 31 → February 31 (which doesn't exist).
        """
        max_day = calendar.monthrange(year, month)[1]
        return ref.replace(year=year, month=month, day=min(ref.day, max_day))
