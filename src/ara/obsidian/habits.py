"""Protocol scoring and habit tracking from Obsidian daily notes.

Analyzes THE PROTOCOL checklist in daily notes to track completion rates,
streaks, and trends. Integrates with ARA's existing HabitDetector for
a unified habit tracking system.

THE PROTOCOL is the user's daily checklist of habits/routines that they
want to maintain. Each daily note has a "THE PROTOCOL" section with
checkbox items. This module scores them and tracks streaks.

Usage:
    from ara.obsidian.habits import ProtocolTracker

    tracker = ProtocolTracker(settings, sync)
    score = await tracker.score_date("2026-03-06")
    streak = tracker.get_current_streak()
    trends = await tracker.get_weekly_trend()
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import structlog

from ara.utils.types import (
    DailyNoteData,
    ObsidianTaskStatus,
    ProtocolScore,
)

if TYPE_CHECKING:
    from ara.config.settings import ObsidianSettings
    from ara.obsidian.sync import ObsidianSync

log = structlog.get_logger(component="obsidian.habits")


class ProtocolTracker:
    """Tracks THE PROTOCOL checklist scores across daily notes.

    Computes daily scores, maintains streak tracking, and generates
    trend analysis for habit monitoring.

    Attributes:
        settings: ObsidianSettings configuration.
        scores: Dict mapping date strings to ProtocolScore.
    """

    def __init__(self, settings: ObsidianSettings, sync: ObsidianSync) -> None:
        self._settings = settings
        self._sync = sync
        self.scores: dict[str, ProtocolScore] = {}

    async def score_date(self, date_str: str) -> ProtocolScore:
        """Score THE PROTOCOL checklist for a specific date.

        Args:
            date_str: Date in YYYY-MM-DD format.

        Returns:
            ProtocolScore with completion metrics.
        """
        # Check cache first
        if date_str in self.scores:
            return self.scores[date_str]

        note = await self._sync.get_daily_note(date_str)
        score = self._compute_score(date_str, note)
        self.scores[date_str] = score

        log.debug(
            "protocol_score_computed",
            date=date_str,
            score=score.score_pct,
            items=score.total_items,
        )

        return score

    async def score_range(self, start_date: str, end_date: str) -> list[ProtocolScore]:
        """Score all daily notes in a date range.

        Args:
            start_date: Start date (YYYY-MM-DD, inclusive).
            end_date: End date (YYYY-MM-DD, inclusive).

        Returns:
            List of ProtocolScore objects for each date.
        """
        start = datetime.strptime(start_date, "%Y-%m-%d").date()
        end = datetime.strptime(end_date, "%Y-%m-%d").date()

        scores: list[ProtocolScore] = []
        current = start
        while current <= end:
            date_str = current.isoformat()
            score = await self.score_date(date_str)
            scores.append(score)
            current += timedelta(days=1)

        return scores

    def get_current_streak(self) -> int:
        """Calculate the current consecutive-day streak above threshold.

        Counts backwards from the most recent scored date, checking if
        each day's protocol score exceeds the streak threshold.

        Returns:
            Number of consecutive days at or above threshold.
        """
        if not self.scores:
            return 0

        threshold = self._settings.protocol_streak_threshold_pct
        sorted_dates = sorted(self.scores.keys(), reverse=True)

        streak = 0
        for date_str in sorted_dates:
            score = self.scores[date_str]
            if score.total_items == 0:
                # No protocol data for this date — don't break streak, skip
                continue
            if score.score_pct >= threshold:
                streak += 1
            else:
                break

        return streak

    async def get_weekly_trend(self, end_date: str | None = None) -> dict[str, float]:
        """Get protocol scores for the last 7 days.

        Args:
            end_date: End date (YYYY-MM-DD). Defaults to today.

        Returns:
            Dict mapping date strings to score percentages.
        """
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        end = datetime.strptime(end_date, "%Y-%m-%d").date()
        start = end - timedelta(days=6)

        trend: dict[str, float] = {}
        current = start
        while current <= end:
            date_str = current.isoformat()
            score = await self.score_date(date_str)
            trend[date_str] = score.score_pct
            current += timedelta(days=1)

        return trend

    async def get_weekly_average(self, end_date: str | None = None) -> float:
        """Get the average protocol score for the last 7 days.

        Args:
            end_date: End date (YYYY-MM-DD). Defaults to today.

        Returns:
            Average score percentage (0-100).
        """
        trend = await self.get_weekly_trend(end_date)
        scored_days = [v for v in trend.values() if v > 0]
        if not scored_days:
            return 0.0
        return sum(scored_days) / len(scored_days)

    async def get_most_missed_items(self, days: int = 14) -> list[tuple[str, int]]:
        """Find the most frequently missed protocol items.

        Args:
            days: Number of days to look back.

        Returns:
            List of (item_text, miss_count) sorted by frequency.
        """
        end = datetime.now().date()
        start = end - timedelta(days=days - 1)

        miss_counts: dict[str, int] = {}
        current = start
        while current <= end:
            date_str = current.isoformat()
            score = await self.score_date(date_str)
            for item in score.missed_items:
                miss_counts[item] = miss_counts.get(item, 0) + 1
            current += timedelta(days=1)

        return sorted(miss_counts.items(), key=lambda x: x[1], reverse=True)

    def get_stats(self) -> dict[str, float | int]:
        """Get aggregate statistics across all scored dates.

        Returns:
            Dict with avg_score, total_days, streak, best_score, worst_score.
        """
        if not self.scores:
            return {
                "avg_score": 0.0,
                "total_days_scored": 0,
                "current_streak": 0,
                "best_score": 0.0,
                "worst_score": 0.0,
            }

        scored = [s for s in self.scores.values() if s.total_items > 0]
        if not scored:
            return {
                "avg_score": 0.0,
                "total_days_scored": 0,
                "current_streak": 0,
                "best_score": 0.0,
                "worst_score": 0.0,
            }

        pcts = [s.score_pct for s in scored]
        return {
            "avg_score": sum(pcts) / len(pcts),
            "total_days_scored": len(scored),
            "current_streak": self.get_current_streak(),
            "best_score": max(pcts),
            "worst_score": min(pcts),
        }

    # ── Internal ─────────────────────────────────────────────────────────

    def _compute_score(self, date_str: str, note: DailyNoteData | None) -> ProtocolScore:
        """Compute protocol score from a parsed daily note."""
        if note is None or not note.protocol_checklist:
            return ProtocolScore(date=date_str)

        total = len(note.protocol_checklist)
        completed = sum(1 for t in note.protocol_checklist if t.status == ObsidianTaskStatus.DONE)
        missed = [t.text for t in note.protocol_checklist if t.status != ObsidianTaskStatus.DONE]
        pct = (completed / total) * 100.0 if total > 0 else 0.0

        return ProtocolScore(
            date=date_str,
            total_items=total,
            completed_items=completed,
            score_pct=pct,
            missed_items=missed,
            streak_days=0,  # Computed separately via get_current_streak()
        )

    def clear_cache(self) -> None:
        """Clear all cached scores."""
        self.scores.clear()
        log.debug("protocol_scores_cleared")
