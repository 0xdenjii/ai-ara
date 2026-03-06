"""Mood tracking over time — sentiment analysis of conversations with trend detection.

Captures emotional valence and intensity from each conversation, stores
MoodEntry data points, and computes rolling trends. Integrates with the
existing EmotionTagger for keyword-based sentiment analysis.

Architecture:
    - Each completed conversation is scored for mood (valence + intensity)
    - MoodEntry points are stored in SQLite for historical analysis
    - Rolling windows compute MoodTrend (improving / declining / stable)
    - Notifications are queued when significant mood shifts are detected

Usage:
    from ara.proactive.mood import MoodTracker

    tracker = MoodTracker(settings.proactive, db_path=Path("data/mood.sqlite"))
    tracker.connect()

    # Record mood from a conversation
    entry = tracker.record_from_text("I had a great day, feeling really happy!", conv_id="abc123")

    # Get current trend
    trend = tracker.get_trend(window_days=7)
    # trend.trend_direction == "stable" or "improving" or "declining"

    # Get mood history
    entries = tracker.get_entries(days=30)
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from ara.memory.emotion import EmotionTagger
from ara.utils.errors import MoodTrackingError
from ara.utils.types import MoodEntry, MoodTrend

if TYPE_CHECKING:
    from ara.config.settings import ProactiveSettings

log = structlog.get_logger(component="mood")

# ── SQL Schema ───────────────────────────────────────────────────────────────

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS mood_entries (
    entry_id                TEXT PRIMARY KEY,
    timestamp               TEXT NOT NULL,
    valence                 REAL NOT NULL DEFAULT 0.0,
    intensity               REAL NOT NULL DEFAULT 0.0,
    dominant_emotion        TEXT NOT NULL DEFAULT '',
    emotions_json           TEXT NOT NULL DEFAULT '{}',
    source_conversation_id  TEXT NOT NULL DEFAULT '',
    metadata_json           TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_mood_timestamp ON mood_entries(timestamp);
CREATE INDEX IF NOT EXISTS idx_mood_conversation ON mood_entries(source_conversation_id);
"""

# Valence shift threshold to consider "significant" mood change
_SIGNIFICANT_VALENCE_SHIFT = 0.3

# Minimum data points needed to compute a meaningful trend
_MIN_TREND_POINTS = 3


class MoodTracker:
    """Tracks emotional state over time from conversation content.

    Records MoodEntry data points from each conversation's emotional
    content, persists them to SQLite, and computes rolling trends
    (improving, declining, stable) for ARA to reference in check-ins.

    Thread-safe via a reentrant lock on all database operations.
    """

    def __init__(
        self,
        settings: ProactiveSettings,
        *,
        db_path: Path | None = None,
    ) -> None:
        self._window_days = settings.mood_window_days
        self._enabled = settings.mood_tracking_enabled
        self._db_path = db_path or Path("./data/mood.sqlite")
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.RLock()
        self._tagger = EmotionTagger()

    def connect(self) -> None:
        """Open the SQLite database and initialize schema."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.executescript(_SCHEMA_SQL)
        self._conn.commit()
        log.info("mood_tracker_connected", db_path=str(self._db_path))

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
            log.info("mood_tracker_closed")

    @property
    def is_connected(self) -> bool:
        """Whether the database is connected."""
        return self._conn is not None

    # ── Recording ─────────────────────────────────────────────────────────

    def record_from_text(
        self,
        text: str,
        *,
        conversation_id: str = "",
        timestamp: datetime | None = None,
    ) -> MoodEntry | None:
        """Analyze text for emotion and record a mood data point.

        Uses EmotionTagger to extract valence, intensity, and emotions
        from the conversation text. Only records if emotional content
        is detected.

        Args:
            text: Conversation text to analyze.
            conversation_id: ID of the source conversation.
            timestamp: When the mood was recorded (defaults to now).

        Returns:
            MoodEntry if emotional content detected, None otherwise.
        """
        if not self._enabled:
            return None

        emotion_data = self._tagger.tag(text)
        if not emotion_data.has_emotional_content:
            return None

        now = timestamp or datetime.now()

        emotions_dict: dict[str, float] = {}
        for tag in emotion_data.tags:
            emotions_dict[tag.emotion] = tag.confidence

        entry = MoodEntry(
            timestamp=now,
            valence=emotion_data.emotional_valence,
            intensity=emotion_data.emotional_intensity,
            dominant_emotion=emotion_data.dominant_emotion,
            emotions=emotions_dict,
            source_conversation_id=conversation_id,
        )

        self._persist_entry(entry)

        log.info(
            "mood_recorded",
            valence=entry.valence,
            intensity=entry.intensity,
            dominant=entry.dominant_emotion,
            conversation_id=conversation_id,
        )

        return entry

    def record_entry(self, entry: MoodEntry) -> None:
        """Directly record a pre-built MoodEntry.

        Useful when mood data comes from external analysis rather
        than the built-in EmotionTagger.

        Args:
            entry: The MoodEntry to persist.
        """
        self._persist_entry(entry)

    # ── Querying ──────────────────────────────────────────────────────────

    def get_entries(
        self,
        *,
        days: int | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int = 100,
    ) -> list[MoodEntry]:
        """Retrieve mood entries from the database.

        Args:
            days: Lookback window in days from now.
            start: Start of date range (inclusive).
            end: End of date range (inclusive).
            limit: Maximum number of entries to return.

        Returns:
            List of MoodEntry objects, newest first.
        """
        conn = self._ensure_connected()
        conditions: list[str] = []
        params: list[object] = []

        if days is not None:
            cutoff = (datetime.now() - timedelta(days=days)).isoformat()
            conditions.append("timestamp >= ?")
            params.append(cutoff)

        if start is not None:
            conditions.append("timestamp >= ?")
            params.append(start.isoformat())

        if end is not None:
            conditions.append("timestamp <= ?")
            params.append(end.isoformat())

        where = ""
        if conditions:
            where = "WHERE " + " AND ".join(conditions)

        query = f"SELECT * FROM mood_entries {where} ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        with self._lock:
            rows = conn.execute(query, params).fetchall()

        return [self._row_to_entry(row) for row in rows]

    def get_trend(self, *, window_days: int | None = None) -> MoodTrend:
        """Compute the rolling mood trend over a time window.

        Analyzes recent MoodEntry data points to determine whether
        mood is improving, declining, or stable.

        Args:
            window_days: Number of days to analyze (defaults to settings).

        Returns:
            MoodTrend with direction, averages, and emotion distribution.
        """
        days = window_days or self._window_days
        entries = self.get_entries(days=days, limit=1000)

        if not entries:
            return MoodTrend(window_days=days)

        # Compute averages
        total_valence = sum(e.valence for e in entries)
        total_intensity = sum(e.intensity for e in entries)
        n = len(entries)
        avg_valence = total_valence / n
        avg_intensity = total_intensity / n

        # Emotion distribution (frequency-weighted)
        emotion_counts: dict[str, float] = {}
        for entry in entries:
            for emotion, confidence in entry.emotions.items():
                emotion_counts[emotion] = emotion_counts.get(emotion, 0.0) + confidence

        # Normalize distribution to sum to 1.0
        total_emotion = sum(emotion_counts.values())
        emotion_dist: dict[str, float] = {}
        if total_emotion > 0:
            emotion_dist = {k: v / total_emotion for k, v in emotion_counts.items()}

        # Find dominant emotion
        dominant = max(emotion_dist, key=emotion_dist.get, default="") if emotion_dist else ""

        # Determine trend direction by comparing first half vs second half
        trend_direction = "stable"
        if n >= _MIN_TREND_POINTS:
            mid = n // 2
            # entries are newest-first, so first half = recent, second half = older
            recent_avg = sum(e.valence for e in entries[:mid]) / mid
            older_avg = sum(e.valence for e in entries[mid:]) / (n - mid)
            delta = recent_avg - older_avg

            if delta > _SIGNIFICANT_VALENCE_SHIFT:
                trend_direction = "improving"
            elif delta < -_SIGNIFICANT_VALENCE_SHIFT:
                trend_direction = "declining"

        # Date range
        start_date = entries[-1].timestamp if entries else None
        end_date = entries[0].timestamp if entries else None

        return MoodTrend(
            window_days=days,
            avg_valence=round(avg_valence, 3),
            avg_intensity=round(avg_intensity, 3),
            dominant_emotion=dominant,
            emotion_distribution=emotion_dist,
            trend_direction=trend_direction,
            data_points=n,
            start_date=start_date,
            end_date=end_date,
        )

    def detect_significant_shift(self) -> tuple[bool, str]:
        """Check if there's been a significant mood shift recently.

        Compares the last 24h average valence against the rolling window
        average. If the delta exceeds the threshold, a shift is detected.

        Returns:
            Tuple of (shift_detected, description).
        """
        recent = self.get_entries(days=1, limit=100)
        if len(recent) < 2:
            return False, ""

        recent_avg = sum(e.valence for e in recent) / len(recent)

        window_trend = self.get_trend()
        if window_trend.data_points < _MIN_TREND_POINTS:
            return False, ""

        delta = recent_avg - window_trend.avg_valence

        if abs(delta) >= _SIGNIFICANT_VALENCE_SHIFT:
            direction = "more positive" if delta > 0 else "more negative"
            description = (
                f"Mood has shifted {direction} in the last 24h "
                f"(recent: {recent_avg:.2f}, baseline: {window_trend.avg_valence:.2f})"
            )
            return True, description

        return False, ""

    def get_stats(self) -> dict[str, object]:
        """Get mood tracking statistics.

        Returns:
            Dict with total entries, date range, current trend.
        """
        conn = self._ensure_connected()
        with self._lock:
            row = conn.execute(
                "SELECT COUNT(*), MIN(timestamp), MAX(timestamp) FROM mood_entries"
            ).fetchone()

        total = row[0] if row else 0
        trend = self.get_trend()

        return {
            "enabled": self._enabled,
            "total_entries": total,
            "earliest_entry": row[1] if row else None,
            "latest_entry": row[2] if row else None,
            "current_trend": trend.trend_direction,
            "avg_valence": trend.avg_valence,
            "avg_intensity": trend.avg_intensity,
            "dominant_emotion": trend.dominant_emotion,
        }

    # ── Internal Helpers ──────────────────────────────────────────────────

    def _persist_entry(self, entry: MoodEntry) -> None:
        """Write a MoodEntry to SQLite."""
        conn = self._ensure_connected()
        import json

        emotions_json = json.dumps(entry.emotions)
        metadata_json = json.dumps(entry.metadata)

        with self._lock:
            conn.execute(
                """INSERT OR REPLACE INTO mood_entries
                (entry_id, timestamp, valence, intensity, dominant_emotion,
                 emotions_json, source_conversation_id, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    entry.entry_id,
                    entry.timestamp.isoformat(),
                    entry.valence,
                    entry.intensity,
                    entry.dominant_emotion,
                    emotions_json,
                    entry.source_conversation_id,
                    metadata_json,
                ),
            )
            conn.commit()

    def _row_to_entry(self, row: tuple[object, ...]) -> MoodEntry:
        """Convert a SQLite row to a MoodEntry."""
        import json

        return MoodEntry(
            entry_id=str(row[0]),
            timestamp=datetime.fromisoformat(str(row[1])),
            valence=float(row[2]),  # type: ignore[arg-type]
            intensity=float(row[3]),  # type: ignore[arg-type]
            dominant_emotion=str(row[4]),
            emotions=json.loads(str(row[5])),
            source_conversation_id=str(row[6]),
            metadata=json.loads(str(row[7])),
        )

    def _ensure_connected(self) -> sqlite3.Connection:
        """Return the database connection, raising if not connected."""
        if self._conn is None:
            raise MoodTrackingError(
                "Mood tracker not connected — call connect() first",
                details={"db_path": str(self._db_path)},
            )
        return self._conn
