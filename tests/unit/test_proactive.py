"""Tests for the proactive features module (Phase 12).

Covers:
- Notification queue (P12-06): priority ordering, TTL, cooldown, delivery, eviction
- Mood tracking (P12-04): recording, trends, significant shifts, persistence
- Habit detection (P12-03): pattern detection, classification, regularity
- Suggestion engine (P12-01): strategy execution, deduplication, rate limiting
- Daily recap (P12-02): conversation gathering, summary generation
- Memory review (P12-05): weekly/monthly reviews, date checks
- Scheduler (P12 integration): lifecycle, feature checks
- Proactive types and error hierarchy
"""

from __future__ import annotations

import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from ara.config.settings import ProactiveSettings
from ara.utils.errors import (
    MoodTrackingError,
    NotificationError,
    ProactiveError,
    RecapError,
    ReviewError,
    SuggestionError,
)
from ara.utils.types import (
    DailyRecap,
    EntityType,
    GraphEntity,
    GraphRelation,
    HabitPattern,
    MemoryReview,
    MoodEntry,
    MoodTrend,
    Notification,
    NotificationPriority,
    ProactiveSuggestion,
    RelationType,
    SuggestionCategory,
    VoicePipelineState,
)

# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture()
def proactive_settings() -> ProactiveSettings:
    """Default proactive settings for testing."""
    return ProactiveSettings(
        enabled=True,
        idle_check_interval_s=60,
        idle_threshold_s=120,
        max_suggestions_per_hour=5,
        daily_recap_enabled=True,
        daily_recap_hour=22,
        daily_recap_minute=0,
        min_interactions_for_recap=1,
        habit_detection_enabled=True,
        habit_min_occurrences=2,
        habit_lookback_days=30,
        mood_tracking_enabled=True,
        mood_window_days=7,
        weekly_review_enabled=True,
        weekly_review_day="sunday",
        weekly_review_hour=20,
        monthly_review_enabled=True,
        monthly_review_day=1,
        notification_queue_max=10,
        notification_ttl_hours=24,
        notification_cooldown_s=30,
    )


@pytest.fixture()
def notification_queue(proactive_settings: ProactiveSettings):
    """Create a NotificationQueue for testing."""
    from ara.proactive.notifications import NotificationQueue

    return NotificationQueue(proactive_settings)


@pytest.fixture()
def mood_tracker(proactive_settings: ProactiveSettings):
    """Create a MoodTracker with a temp database."""
    from ara.proactive.mood import MoodTracker

    with tempfile.TemporaryDirectory() as tmp:
        tracker = MoodTracker(proactive_settings, db_path=Path(tmp) / "mood.sqlite")
        tracker.connect()
        yield tracker
        tracker.close()


@pytest.fixture()
def mock_graph() -> MagicMock:
    """Create a mock KnowledgeGraph."""
    graph = MagicMock()
    graph.get_entities_by_type.return_value = []
    graph.get_relations.return_value = []
    graph.get_neighbors.return_value = []
    graph.detect_communities.return_value = []
    return graph


@pytest.fixture()
def mock_conversations() -> MagicMock:
    """Create a mock ConversationStore."""
    store = MagicMock()
    store.get_recent_conversations.return_value = []
    return store


# ══════════════════════════════════════════════════════════════════════════════
# PROACTIVE TYPES
# ══════════════════════════════════════════════════════════════════════════════


class TestProactiveTypes:
    """Tests for proactive type definitions."""

    def test_notification_priority_values(self):
        assert NotificationPriority.LOW == "low"
        assert NotificationPriority.MEDIUM == "medium"
        assert NotificationPriority.HIGH == "high"
        assert NotificationPriority.URGENT == "urgent"

    def test_suggestion_category_values(self):
        assert SuggestionCategory.FOLLOW_UP == "follow_up"
        assert SuggestionCategory.REMINDER == "reminder"
        assert SuggestionCategory.CHECK_IN == "check_in"
        assert SuggestionCategory.DISCOVERY == "discovery"
        assert SuggestionCategory.SOCIAL == "social"
        assert SuggestionCategory.HABIT == "habit"

    def test_proactive_suggestion_defaults(self):
        s = ProactiveSuggestion()
        assert s.suggestion_id  # auto-generated
        assert s.category == SuggestionCategory.FOLLOW_UP
        assert s.message == ""
        assert s.confidence == 0.5
        assert s.source_entities == []

    def test_daily_recap_defaults(self):
        r = DailyRecap()
        assert r.recap_id
        assert r.summary == ""
        assert r.conversation_count == 0
        assert r.topics_discussed == []
        assert r.highlights == []

    def test_habit_pattern_defaults(self):
        h = HabitPattern(description="exercising", entity_name="exercise")
        assert h.pattern_id
        assert h.description == "exercising"
        assert h.occurrence_count == 0
        assert h.is_positive is True
        assert h.avg_interval_days == 0.0

    def test_mood_entry_defaults(self):
        m = MoodEntry()
        assert m.entry_id
        assert m.valence == 0.0
        assert m.intensity == 0.0
        assert m.emotions == {}

    def test_mood_trend_defaults(self):
        t = MoodTrend()
        assert t.window_days == 7
        assert t.trend_direction == ""
        assert t.data_points == 0

    def test_memory_review_defaults(self):
        r = MemoryReview(period_type="weekly")
        assert r.review_id
        assert r.period_type == "weekly"
        assert r.key_entities == []
        assert r.milestones == []

    def test_notification_is_mutable(self):
        n = Notification(title="test")
        assert n.delivered is False
        n.delivered = True
        assert n.delivered is True

    def test_notification_has_auto_id(self):
        n1 = Notification()
        n2 = Notification()
        assert n1.notification_id != n2.notification_id


# ══════════════════════════════════════════════════════════════════════════════
# ERROR HIERARCHY
# ══════════════════════════════════════════════════════════════════════════════


class TestProactiveErrors:
    """Tests for proactive error hierarchy."""

    def test_proactive_error_is_ara_error(self):
        err = ProactiveError("test")
        assert err.component == "proactive"
        assert err.operation == "generate"

    def test_suggestion_error(self):
        err = SuggestionError("failed", details={"reason": "timeout"})
        assert err.component == "proactive"
        assert err.operation == "suggest"
        assert err.details["reason"] == "timeout"

    def test_recap_error(self):
        err = RecapError("failed")
        assert err.operation == "recap"

    def test_mood_tracking_error(self):
        err = MoodTrackingError("failed")
        assert err.operation == "mood_track"

    def test_review_error(self):
        err = ReviewError("failed")
        assert err.operation == "review"

    def test_notification_error(self):
        err = NotificationError("queue full")
        assert err.operation == "notify"

    def test_error_inheritance_chain(self):
        from ara.utils.errors import AraError

        assert issubclass(ProactiveError, AraError)
        assert issubclass(SuggestionError, ProactiveError)
        assert issubclass(RecapError, ProactiveError)
        assert issubclass(MoodTrackingError, ProactiveError)
        assert issubclass(ReviewError, ProactiveError)
        assert issubclass(NotificationError, ProactiveError)


# ══════════════════════════════════════════════════════════════════════════════
# NOTIFICATION QUEUE (P12-06)
# ══════════════════════════════════════════════════════════════════════════════


class TestNotificationQueue:
    """Tests for the notification queue."""

    def test_enqueue_creates_notification(self, notification_queue):
        n = notification_queue.enqueue(title="Test", message="Hello")
        assert n.title == "Test"
        assert n.message == "Hello"
        assert n.notification_id
        assert n.priority == NotificationPriority.MEDIUM
        assert notification_queue.size == 1

    def test_priority_ordering(self, notification_queue):
        notification_queue.enqueue(title="Low", message="a", priority=NotificationPriority.LOW)
        notification_queue.enqueue(
            title="Urgent", message="b", priority=NotificationPriority.URGENT
        )
        notification_queue.enqueue(title="High", message="c", priority=NotificationPriority.HIGH)

        pending = notification_queue.get_pending()
        assert pending[0].title == "Urgent"
        assert pending[1].title == "High"
        assert pending[2].title == "Low"

    def test_deliver_removes_from_queue(self, notification_queue):
        notification_queue.enqueue(title="Test", message="Hello")
        n = notification_queue.try_deliver()
        assert n is not None
        assert n.title == "Test"
        assert n.delivered is True
        assert notification_queue.size == 0

    def test_deliver_respects_voice_state(self, notification_queue):
        notification_queue.enqueue(title="Test", message="Hello")
        result = notification_queue.try_deliver(voice_state=VoicePipelineState.SPEAKING)
        assert result is None
        assert notification_queue.size == 1

    def test_urgent_bypasses_voice_state(self, notification_queue):
        notification_queue.enqueue(
            title="Urgent", message="Now!", priority=NotificationPriority.URGENT
        )
        result = notification_queue.try_deliver(voice_state=VoicePipelineState.SPEAKING)
        assert result is not None
        assert result.title == "Urgent"

    def test_cooldown_enforcement(self, notification_queue):
        notification_queue.enqueue(title="First", message="a")
        notification_queue.try_deliver()

        notification_queue.enqueue(title="Second", message="b")
        # Should be blocked by cooldown (30s configured)
        result = notification_queue.try_deliver()
        assert result is None

    def test_cooldown_bypassed_for_urgent(self, notification_queue):
        notification_queue.enqueue(title="First", message="a")
        notification_queue.try_deliver()

        notification_queue.enqueue(
            title="Urgent", message="b", priority=NotificationPriority.URGENT
        )
        result = notification_queue.try_deliver()
        assert result is not None

    def test_ttl_expiration(self, proactive_settings):
        from ara.proactive.notifications import NotificationQueue

        settings = ProactiveSettings(notification_ttl_hours=0)  # 0h TTL = immediate expire
        queue = NotificationQueue(settings)
        queue.enqueue(title="Will Expire", message="gone")

        # Manually expire it
        queue._queue[0].expires_at = datetime.now() - timedelta(hours=1)

        result = queue.try_deliver()
        assert result is None
        assert queue.size == 0

    def test_eviction_when_full(self, proactive_settings):
        from ara.proactive.notifications import NotificationQueue

        settings = ProactiveSettings(notification_queue_max=2)
        queue = NotificationQueue(settings)
        queue.enqueue(title="A", message="1", priority=NotificationPriority.HIGH)
        queue.enqueue(title="B", message="2", priority=NotificationPriority.LOW)

        # Third should evict lowest priority
        queue.enqueue(title="C", message="3", priority=NotificationPriority.MEDIUM)
        assert queue.size == 2
        pending = queue.get_pending()
        titles = {n.title for n in pending}
        assert "A" in titles  # HIGH stays
        assert "C" in titles  # MEDIUM stays
        # "B" (LOW) was evicted

    def test_dismiss(self, notification_queue):
        n = notification_queue.enqueue(title="Test", message="Hello")
        result = notification_queue.dismiss(n.notification_id)
        assert result is True
        assert notification_queue.size == 0

    def test_dismiss_nonexistent(self, notification_queue):
        result = notification_queue.dismiss("nonexistent")
        assert result is False

    def test_peek_without_removing(self, notification_queue):
        notification_queue.enqueue(title="Test", message="Hello")
        peeked = notification_queue.peek()
        assert peeked is not None
        assert notification_queue.size == 1

    def test_clear(self, notification_queue):
        notification_queue.enqueue(title="A", message="1")
        notification_queue.enqueue(title="B", message="2")
        count = notification_queue.clear()
        assert count == 2
        assert notification_queue.is_empty

    def test_callback_on_delivery(self, notification_queue):
        delivered: list[Notification] = []
        notification_queue.register_callback(lambda n: delivered.append(n))

        notification_queue.enqueue(title="Test", message="Hello")
        notification_queue.try_deliver()
        assert len(delivered) == 1
        assert delivered[0].title == "Test"

    def test_stats(self, notification_queue):
        notification_queue.enqueue(title="A", message="1")
        notification_queue.enqueue(title="B", message="2", priority=NotificationPriority.HIGH)

        stats = notification_queue.get_stats()
        assert stats["queue_size"] == 2
        assert stats["total_enqueued"] == 2

    def test_source_metadata_preserved(self, notification_queue):
        n = notification_queue.enqueue(
            title="Test",
            message="Hello",
            source="recap",
            metadata={"key": "value"},
        )
        assert n.source == "recap"
        assert n.metadata["key"] == "value"

    def test_empty_queue_deliver_returns_none(self, notification_queue):
        assert notification_queue.try_deliver() is None

    def test_empty_queue_peek_returns_none(self, notification_queue):
        assert notification_queue.peek() is None


# ══════════════════════════════════════════════════════════════════════════════
# MOOD TRACKING (P12-04)
# ══════════════════════════════════════════════════════════════════════════════


class TestMoodTracker:
    """Tests for mood tracking and sentiment analysis."""

    def test_record_from_emotional_text(self, mood_tracker):
        entry = mood_tracker.record_from_text("I feel so happy and excited today!")
        assert entry is not None
        assert entry.valence > 0
        assert entry.dominant_emotion != ""

    def test_record_from_neutral_text(self, mood_tracker):
        entry = mood_tracker.record_from_text("The weather is 72 degrees.")
        # Neutral text should not create an entry
        assert entry is None

    def test_record_negative_mood(self, mood_tracker):
        entry = mood_tracker.record_from_text("I feel really sad and lonely today.")
        assert entry is not None
        assert entry.valence < 0

    def test_record_with_conversation_id(self, mood_tracker):
        entry = mood_tracker.record_from_text(
            "I'm thrilled about the promotion!",
            conversation_id="conv123",
        )
        assert entry is not None
        assert entry.source_conversation_id == "conv123"

    def test_get_entries_after_recording(self, mood_tracker):
        mood_tracker.record_from_text("I'm happy today!")
        mood_tracker.record_from_text("Feeling great!")

        entries = mood_tracker.get_entries(days=1)
        assert len(entries) >= 1  # At least one (some texts may not trigger)

    def test_get_entries_with_date_filter(self, mood_tracker):
        now = datetime.now()
        mood_tracker.record_from_text("Happy!", timestamp=now - timedelta(days=10))
        mood_tracker.record_from_text("Excited!", timestamp=now)

        recent = mood_tracker.get_entries(days=5)
        old = mood_tracker.get_entries(days=15)
        assert len(recent) <= len(old)

    def test_get_trend_empty_db(self, mood_tracker):
        trend = mood_tracker.get_trend()
        assert trend.data_points == 0
        assert trend.trend_direction == ""

    def test_get_trend_with_data(self, mood_tracker):
        # Record a few mood entries
        now = datetime.now()
        mood_tracker.record_from_text("I'm very happy!", timestamp=now)
        mood_tracker.record_from_text(
            "Feeling amazing and excited!", timestamp=now - timedelta(hours=1)
        )
        mood_tracker.record_from_text("Thrilled about life!", timestamp=now - timedelta(hours=2))

        trend = mood_tracker.get_trend(window_days=1)
        assert trend.data_points >= 1

    def test_detect_no_shift_with_insufficient_data(self, mood_tracker):
        shifted, description = mood_tracker.detect_significant_shift()
        assert shifted is False
        assert description == ""

    def test_record_entry_directly(self, mood_tracker):
        entry = MoodEntry(
            valence=0.8,
            intensity=0.6,
            dominant_emotion="joy",
            emotions={"joy": 0.8},
        )
        mood_tracker.record_entry(entry)
        entries = mood_tracker.get_entries(days=1)
        assert len(entries) == 1
        assert entries[0].valence == 0.8

    def test_disabled_tracker_returns_none(self, proactive_settings):
        from ara.proactive.mood import MoodTracker

        proactive_settings.mood_tracking_enabled = False
        with tempfile.TemporaryDirectory() as tmp:
            tracker = MoodTracker(proactive_settings, db_path=Path(tmp) / "mood.sqlite")
            tracker.connect()
            result = tracker.record_from_text("I'm so happy!")
            assert result is None
            tracker.close()

    def test_not_connected_raises(self, proactive_settings):
        from ara.proactive.mood import MoodTracker

        tracker = MoodTracker(proactive_settings, db_path=Path("dummy.sqlite"))
        with pytest.raises(MoodTrackingError, match="not connected"):
            tracker.get_entries()

    def test_get_stats(self, mood_tracker):
        mood_tracker.record_from_text("I'm happy!")
        stats = mood_tracker.get_stats()
        assert "total_entries" in stats
        assert "current_trend" in stats
        assert stats["enabled"] is True

    def test_connect_close_lifecycle(self, proactive_settings):
        from ara.proactive.mood import MoodTracker

        with tempfile.TemporaryDirectory() as tmp:
            tracker = MoodTracker(proactive_settings, db_path=Path(tmp) / "mood.sqlite")
            assert not tracker.is_connected
            tracker.connect()
            assert tracker.is_connected
            tracker.close()
            assert not tracker.is_connected


# ══════════════════════════════════════════════════════════════════════════════
# HABIT DETECTION (P12-03)
# ══════════════════════════════════════════════════════════════════════════════


class TestHabitDetector:
    """Tests for habit pattern detection."""

    def _make_detector(self, settings: ProactiveSettings, graph: MagicMock):
        from ara.proactive.habits import HabitDetector

        return HabitDetector(settings, graph=graph)

    def test_detect_empty_graph(self, proactive_settings, mock_graph):
        detector = self._make_detector(proactive_settings, mock_graph)
        patterns = detector.detect_patterns()
        assert patterns == []

    def test_detect_with_recurring_entity(self, proactive_settings, mock_graph):
        now = datetime.now()
        entity = GraphEntity(
            entity_id="e1",
            name="Exercise",
            entity_type=EntityType.HABIT,
            mention_count=5,
            first_seen=now - timedelta(days=20),
            last_seen=now - timedelta(days=1),
        )
        mock_graph.get_entities_by_type.return_value = [entity]

        # Return relations with timestamps
        relations = [
            GraphRelation(
                source_id="e1",
                target_id="e2",
                relation_type=RelationType.EXPERIENCED,
                last_seen=now - timedelta(days=i * 3),
            )
            for i in range(5)
        ]
        mock_graph.get_relations.return_value = relations

        detector = self._make_detector(proactive_settings, mock_graph)
        patterns = detector.detect_patterns()
        assert len(patterns) > 0
        assert patterns[0].entity_name == "Exercise"

    def test_classify_active_pattern(self, proactive_settings, mock_graph):
        detector = self._make_detector(proactive_settings, mock_graph)
        pattern = HabitPattern(
            description="exercising",
            entity_name="Exercise",
            last_seen=datetime.now() - timedelta(days=2),
            avg_interval_days=3.0,
        )
        assert detector.classify_pattern(pattern) == "active"

    def test_classify_fading_pattern(self, proactive_settings, mock_graph):
        detector = self._make_detector(proactive_settings, mock_graph)
        pattern = HabitPattern(
            description="exercising",
            entity_name="Exercise",
            last_seen=datetime.now() - timedelta(days=6),
            avg_interval_days=3.0,
        )
        assert detector.classify_pattern(pattern) == "fading"

    def test_classify_broken_pattern(self, proactive_settings, mock_graph):
        detector = self._make_detector(proactive_settings, mock_graph)
        pattern = HabitPattern(
            description="exercising",
            entity_name="Exercise",
            last_seen=datetime.now() - timedelta(days=15),
            avg_interval_days=3.0,
        )
        assert detector.classify_pattern(pattern) == "broken"

    def test_disabled_returns_empty(self, proactive_settings, mock_graph):
        proactive_settings.habit_detection_enabled = False
        detector = self._make_detector(proactive_settings, mock_graph)
        assert detector.detect_patterns() == []

    def test_regularity_computation(self, proactive_settings, mock_graph):
        detector = self._make_detector(proactive_settings, mock_graph)

        now = datetime.now()
        # Perfect regularity: every 3 days
        regular_timestamps = [now - timedelta(days=i * 3) for i in range(5)]
        regular_timestamps.sort()
        score = detector._compute_regularity(regular_timestamps)
        assert score > 0.8  # Should be near 1.0

    def test_irregularity_gives_low_score(self, proactive_settings, mock_graph):
        detector = self._make_detector(proactive_settings, mock_graph)

        now = datetime.now()
        # Irregular: random intervals
        timestamps = sorted([now - timedelta(days=d) for d in [1, 3, 15, 16, 45]])
        score = detector._compute_regularity(timestamps)
        assert score < 0.8

    def test_get_active_habits(self, proactive_settings, mock_graph):
        detector = self._make_detector(proactive_settings, mock_graph)
        # With empty graph, should return empty
        active = detector.get_active_habits()
        assert active == []

    def test_get_fading_habits(self, proactive_settings, mock_graph):
        detector = self._make_detector(proactive_settings, mock_graph)
        fading = detector.get_fading_habits()
        assert fading == []

    @pytest.mark.asyncio()
    async def test_generate_nudges_empty(self, proactive_settings, mock_graph):
        detector = self._make_detector(proactive_settings, mock_graph)
        llm = AsyncMock()
        nudges = await detector.generate_nudges([], llm_client=llm)
        assert nudges == []

    @pytest.mark.asyncio()
    async def test_generate_nudges_for_fading(self, proactive_settings, mock_graph):
        detector = self._make_detector(proactive_settings, mock_graph)
        llm = AsyncMock()
        llm.generate = AsyncMock(return_value="Hey, time to exercise!")

        pattern = HabitPattern(
            description="exercising",
            entity_name="Exercise",
            last_seen=datetime.now() - timedelta(days=6),
            avg_interval_days=3.0,
        )
        nudges = await detector.generate_nudges([pattern], llm_client=llm)
        assert len(nudges) == 1
        assert nudges[0]["nudge_text"] == "Hey, time to exercise!"

    @pytest.mark.asyncio()
    async def test_generate_nudges_llm_fallback(self, proactive_settings, mock_graph):
        detector = self._make_detector(proactive_settings, mock_graph)
        llm = AsyncMock()
        llm.generate = AsyncMock(side_effect=Exception("LLM down"))

        pattern = HabitPattern(
            description="exercising",
            entity_name="Exercise",
            last_seen=datetime.now() - timedelta(days=10),
            avg_interval_days=3.0,
        )
        nudges = await detector.generate_nudges([pattern], llm_client=llm)
        assert len(nudges) == 1
        assert "gentle reminder" in nudges[0]["nudge_text"]


# ══════════════════════════════════════════════════════════════════════════════
# SUGGESTION ENGINE (P12-01)
# ══════════════════════════════════════════════════════════════════════════════


class TestSuggestionEngine:
    """Tests for the proactive suggestion engine."""

    def _make_engine(
        self,
        settings: ProactiveSettings,
        graph: MagicMock,
        mood: MagicMock | None = None,
        habits: MagicMock | None = None,
    ):
        from ara.proactive.suggestions import SuggestionEngine

        return SuggestionEngine(
            settings,
            graph=graph,
            mood_tracker=mood,
            habit_detector=habits,
        )

    def test_generate_empty_graph(self, proactive_settings, mock_graph):
        engine = self._make_engine(proactive_settings, mock_graph)
        suggestions = engine.generate()
        assert suggestions == []

    def test_social_strategy_finds_dormant_people(self, proactive_settings, mock_graph):
        dormant_person = GraphEntity(
            entity_id="p1",
            name="Mom",
            entity_type=EntityType.PERSON,
            mention_count=10,
            last_seen=datetime.now() - timedelta(days=30),
        )
        mock_graph.get_entities_by_type.return_value = [dormant_person]

        engine = self._make_engine(proactive_settings, mock_graph)
        suggestions = engine.generate()

        social_suggestions = [s for s in suggestions if s.category == SuggestionCategory.SOCIAL]
        assert len(social_suggestions) >= 1
        assert "Mom" in social_suggestions[0].message

    def test_goal_strategy_finds_stale_goals(self, proactive_settings, mock_graph):
        stale_goal = GraphEntity(
            entity_id="g1",
            name="Learn Rust",
            entity_type=EntityType.GOAL,
            mention_count=3,
            last_seen=datetime.now() - timedelta(days=14),
        )

        def side_effect(entity_type):
            if entity_type == EntityType.GOAL:
                return [stale_goal]
            return []

        mock_graph.get_entities_by_type.side_effect = side_effect

        engine = self._make_engine(proactive_settings, mock_graph)
        suggestions = engine.generate()

        reminders = [s for s in suggestions if s.category == SuggestionCategory.REMINDER]
        assert len(reminders) >= 1
        assert "Learn Rust" in reminders[0].message

    def test_disabled_returns_empty(self, proactive_settings, mock_graph):
        proactive_settings.enabled = False
        engine = self._make_engine(proactive_settings, mock_graph)
        assert engine.generate() == []

    def test_rate_limiting(self, proactive_settings, mock_graph):
        proactive_settings.max_suggestions_per_hour = 1
        engine = self._make_engine(proactive_settings, mock_graph)

        engine.generate()
        # Second call should be rate limited
        result = engine.generate()
        assert result == []

    def test_deduplication_by_entity(self, proactive_settings, mock_graph):
        engine = self._make_engine(proactive_settings, mock_graph)

        candidates = [
            ProactiveSuggestion(
                category=SuggestionCategory.SOCIAL,
                message="A",
                source_entities=["e1"],
                confidence=0.5,
            ),
            ProactiveSuggestion(
                category=SuggestionCategory.FOLLOW_UP,
                message="B",
                source_entities=["e1"],
                confidence=0.8,
            ),
        ]
        deduplicated = engine._deduplicate(candidates)
        assert len(deduplicated) == 1
        assert deduplicated[0].confidence == 0.8  # Higher confidence wins

    def test_max_suggestions_cap(self, proactive_settings, mock_graph):
        # Create many dormant people
        people = [
            GraphEntity(
                entity_id=f"p{i}",
                name=f"Person {i}",
                entity_type=EntityType.PERSON,
                mention_count=5,
                last_seen=datetime.now() - timedelta(days=30),
            )
            for i in range(10)
        ]
        mock_graph.get_entities_by_type.return_value = people

        engine = self._make_engine(proactive_settings, mock_graph)
        suggestions = engine.generate(max_suggestions=3)
        assert len(suggestions) <= 3

    def test_enqueue_suggestions(self, proactive_settings, mock_graph, notification_queue):
        engine = self._make_engine(proactive_settings, mock_graph)
        suggestions = [
            ProactiveSuggestion(
                category=SuggestionCategory.SOCIAL,
                message="Check on Mom",
            )
        ]
        enqueued = engine.enqueue_suggestions(suggestions, notification_queue=notification_queue)
        assert enqueued == 1
        assert notification_queue.size == 1

    def test_check_in_strategy_with_mood_shift(self, proactive_settings, mock_graph):
        mock_mood = MagicMock()
        mock_mood.is_connected = True
        mock_mood.detect_significant_shift.return_value = (True, "Mood shifted negative")
        mock_mood.get_trend.return_value = MoodTrend(avg_valence=-0.5)

        engine = self._make_engine(proactive_settings, mock_graph, mood=mock_mood)
        suggestions = engine.generate()

        check_ins = [s for s in suggestions if s.category == SuggestionCategory.CHECK_IN]
        assert len(check_ins) >= 1


# ══════════════════════════════════════════════════════════════════════════════
# DAILY RECAP (P12-02)
# ══════════════════════════════════════════════════════════════════════════════


class TestRecapGenerator:
    """Tests for daily recap generation."""

    def _make_recap(
        self,
        settings: ProactiveSettings,
        conversations: MagicMock,
        graph: MagicMock,
        mood: MagicMock | None = None,
    ):
        from ara.proactive.recap import RecapGenerator

        return RecapGenerator(
            settings,
            conversations=conversations,
            graph=graph,
            mood_tracker=mood,
        )

    @pytest.mark.asyncio()
    async def test_generate_with_no_conversations(
        self, proactive_settings, mock_conversations, mock_graph
    ):
        proactive_settings.min_interactions_for_recap = 3
        generator = self._make_recap(proactive_settings, mock_conversations, mock_graph)
        recap = await generator.generate_daily_recap()
        assert recap is None  # Not enough conversations

    @pytest.mark.asyncio()
    async def test_generate_fallback_summary(
        self, proactive_settings, mock_conversations, mock_graph
    ):

        now = datetime.now()
        conv = MagicMock()
        conv.started_at = now
        conv.turns = [MagicMock(user_message="Hello", assistant_message="Hi")]
        conv.turn_count = 1
        conv.summary = "Greeted the user"
        conv.conversation_id = "c1"
        mock_conversations.get_recent_conversations.return_value = [conv]

        generator = self._make_recap(proactive_settings, mock_conversations, mock_graph)
        recap = await generator.generate_daily_recap()
        assert recap is not None
        assert recap.conversation_count == 1
        assert "1 conversation" in recap.summary

    @pytest.mark.asyncio()
    async def test_generate_with_llm(self, proactive_settings, mock_conversations, mock_graph):
        now = datetime.now()
        conv = MagicMock()
        conv.started_at = now
        conv.turns = [MagicMock(user_message="Hello", assistant_message="Hi")]
        conv.turn_count = 1
        conv.summary = "Discussed work"
        conv.conversation_id = "c1"
        mock_conversations.get_recent_conversations.return_value = [conv]

        llm = AsyncMock()
        llm.generate = AsyncMock(return_value="Today was productive! You discussed work.")

        generator = self._make_recap(proactive_settings, mock_conversations, mock_graph)
        recap = await generator.generate_daily_recap(llm_client=llm)
        assert recap is not None
        assert "productive" in recap.summary

    @pytest.mark.asyncio()
    async def test_disabled_returns_none(self, proactive_settings, mock_conversations, mock_graph):
        proactive_settings.daily_recap_enabled = False
        generator = self._make_recap(proactive_settings, mock_conversations, mock_graph)
        recap = await generator.generate_daily_recap()
        assert recap is None

    @pytest.mark.asyncio()
    async def test_llm_failure_falls_back(self, proactive_settings, mock_conversations, mock_graph):
        now = datetime.now()
        conv = MagicMock()
        conv.started_at = now
        conv.turns = [MagicMock(user_message="Hello", assistant_message="Hi")]
        conv.turn_count = 1
        conv.summary = "Chat"
        conv.conversation_id = "c1"
        mock_conversations.get_recent_conversations.return_value = [conv]

        llm = AsyncMock()
        llm.generate = AsyncMock(side_effect=Exception("LLM offline"))

        generator = self._make_recap(proactive_settings, mock_conversations, mock_graph)
        recap = await generator.generate_daily_recap(llm_client=llm)
        assert recap is not None
        # Should have fallback summary
        assert "conversation" in recap.summary.lower()


# ══════════════════════════════════════════════════════════════════════════════
# MEMORY REVIEW (P12-05)
# ══════════════════════════════════════════════════════════════════════════════


class TestReviewGenerator:
    """Tests for weekly/monthly memory reviews."""

    def _make_review(
        self,
        settings: ProactiveSettings,
        graph: MagicMock,
        conversations: MagicMock,
        mood: MagicMock | None = None,
    ):
        from ara.proactive.review import ReviewGenerator

        return ReviewGenerator(
            settings,
            graph=graph,
            conversations=conversations,
            mood_tracker=mood,
        )

    @pytest.mark.asyncio()
    async def test_generate_weekly_review(self, proactive_settings, mock_graph, mock_conversations):
        generator = self._make_review(proactive_settings, mock_graph, mock_conversations)
        review = await generator.generate_weekly_review()
        assert review is not None
        assert review.period_type == "weekly"

    @pytest.mark.asyncio()
    async def test_generate_monthly_review(
        self, proactive_settings, mock_graph, mock_conversations
    ):
        generator = self._make_review(proactive_settings, mock_graph, mock_conversations)
        review = await generator.generate_monthly_review()
        assert review is not None
        assert review.period_type == "monthly"

    @pytest.mark.asyncio()
    async def test_disabled_weekly_returns_none(
        self, proactive_settings, mock_graph, mock_conversations
    ):
        proactive_settings.weekly_review_enabled = False
        generator = self._make_review(proactive_settings, mock_graph, mock_conversations)
        review = await generator.generate_weekly_review()
        assert review is None

    @pytest.mark.asyncio()
    async def test_disabled_monthly_returns_none(
        self, proactive_settings, mock_graph, mock_conversations
    ):
        proactive_settings.monthly_review_enabled = False
        generator = self._make_review(proactive_settings, mock_graph, mock_conversations)
        review = await generator.generate_monthly_review()
        assert review is None

    @pytest.mark.asyncio()
    async def test_review_with_llm(self, proactive_settings, mock_graph, mock_conversations):
        llm = AsyncMock()
        llm.generate = AsyncMock(return_value="Great week! You accomplished a lot.")

        generator = self._make_review(proactive_settings, mock_graph, mock_conversations)
        review = await generator.generate_weekly_review(llm_client=llm)
        assert review is not None
        assert "accomplished" in review.summary

    @pytest.mark.asyncio()
    async def test_review_with_active_entities(
        self, proactive_settings, mock_graph, mock_conversations
    ):
        now = datetime.now()
        entity = GraphEntity(
            entity_id="e1",
            name="Work",
            entity_type=EntityType.PROJECT,
            mention_count=10,
            last_seen=now - timedelta(days=2),
            first_seen=now - timedelta(days=30),
        )
        mock_graph.get_entities_by_type.return_value = [entity]

        generator = self._make_review(proactive_settings, mock_graph, mock_conversations)
        review = await generator.generate_weekly_review()
        assert review is not None
        assert "Work" in review.key_entities

    def test_is_weekly_review_day(self, proactive_settings, mock_graph, mock_conversations):
        generator = self._make_review(proactive_settings, mock_graph, mock_conversations)
        # This will be True or False depending on the day — just check it doesn't crash
        result = generator.is_weekly_review_day()
        assert isinstance(result, bool)

    def test_is_monthly_review_day(self, proactive_settings, mock_graph, mock_conversations):
        generator = self._make_review(proactive_settings, mock_graph, mock_conversations)
        result = generator.is_monthly_review_day()
        assert isinstance(result, bool)

    @pytest.mark.asyncio()
    async def test_review_llm_failure_fallback(
        self, proactive_settings, mock_graph, mock_conversations
    ):
        llm = AsyncMock()
        llm.generate = AsyncMock(side_effect=Exception("LLM offline"))

        generator = self._make_review(proactive_settings, mock_graph, mock_conversations)
        review = await generator.generate_weekly_review(llm_client=llm)
        assert review is not None
        # Should use fallback template
        assert "conversation" in review.summary.lower() or "weekly" in review.summary.lower()


# ══════════════════════════════════════════════════════════════════════════════
# SCHEDULER (P12 integration)
# ══════════════════════════════════════════════════════════════════════════════


class TestProactiveScheduler:
    """Tests for the proactive scheduler."""

    def _make_scheduler(
        self,
        settings: ProactiveSettings,
        notification_queue,
        **kwargs,
    ):
        from ara.proactive.scheduler import ProactiveScheduler

        return ProactiveScheduler(
            settings,
            notification_queue=notification_queue,
            **kwargs,
        )

    @pytest.mark.asyncio()
    async def test_start_stop_lifecycle(self, proactive_settings, notification_queue):
        scheduler = self._make_scheduler(proactive_settings, notification_queue)
        assert not scheduler.is_running

        await scheduler.start()
        assert scheduler.is_running

        await scheduler.stop()
        assert not scheduler.is_running

    @pytest.mark.asyncio()
    async def test_disabled_scheduler_does_not_start(self, proactive_settings, notification_queue):
        proactive_settings.enabled = False
        scheduler = self._make_scheduler(proactive_settings, notification_queue)

        await scheduler.start()
        assert not scheduler.is_running  # Should not start when disabled

    def test_update_voice_state(self, proactive_settings, notification_queue):
        scheduler = self._make_scheduler(proactive_settings, notification_queue)
        scheduler.update_voice_state(VoicePipelineState.SPEAKING)
        assert scheduler._voice_state == VoicePipelineState.SPEAKING

    def test_record_user_activity(self, proactive_settings, notification_queue):
        scheduler = self._make_scheduler(proactive_settings, notification_queue)
        before = scheduler._last_user_activity
        import time

        time.sleep(0.01)
        scheduler.record_user_activity()
        assert scheduler._last_user_activity >= before

    def test_get_stats(self, proactive_settings, notification_queue):
        scheduler = self._make_scheduler(proactive_settings, notification_queue)
        stats = scheduler.get_stats()
        assert "running" in stats
        assert "enabled" in stats
        assert "voice_state" in stats
        assert "notification_queue" in stats

    def test_idle_detection(self, proactive_settings, notification_queue):
        scheduler = self._make_scheduler(proactive_settings, notification_queue)
        # Just recorded activity — should not be idle
        scheduler.record_user_activity()
        assert not scheduler._is_user_idle(datetime.now())

        # Simulate being idle for longer than threshold
        scheduler._last_user_activity = datetime.now() - timedelta(seconds=300)
        assert scheduler._is_user_idle(datetime.now())

    def test_notification_delivery_check(self, proactive_settings, notification_queue):
        scheduler = self._make_scheduler(proactive_settings, notification_queue)
        notification_queue.enqueue(title="Test", message="Hello")
        scheduler._check_notification_delivery()
        # Should have delivered the notification
        assert notification_queue.size == 0

    @pytest.mark.asyncio()
    async def test_double_start_warning(self, proactive_settings, notification_queue):
        scheduler = self._make_scheduler(proactive_settings, notification_queue)
        await scheduler.start()
        await scheduler.start()  # Should log warning but not crash
        assert scheduler.is_running
        await scheduler.stop()


# ══════════════════════════════════════════════════════════════════════════════
# PROACTIVE SETTINGS
# ══════════════════════════════════════════════════════════════════════════════


class TestProactiveSettings:
    """Tests for ProactiveSettings configuration."""

    def test_default_values(self):
        settings = ProactiveSettings()
        assert settings.enabled is True
        assert settings.idle_check_interval_s == 300
        assert settings.daily_recap_hour == 22
        assert settings.habit_min_occurrences == 3
        assert settings.mood_window_days == 7
        assert settings.weekly_review_day == "sunday"
        assert settings.notification_queue_max == 20

    def test_custom_values(self):
        settings = ProactiveSettings(
            enabled=False,
            idle_check_interval_s=60,
            daily_recap_hour=20,
            mood_window_days=14,
        )
        assert settings.enabled is False
        assert settings.idle_check_interval_s == 60
        assert settings.daily_recap_hour == 20
        assert settings.mood_window_days == 14

    def test_hour_validation(self):
        # Valid
        settings = ProactiveSettings(daily_recap_hour=23)
        assert settings.daily_recap_hour == 23

    def test_settings_in_ara_settings(self):
        from ara.config.settings import AraSettings

        settings = AraSettings()
        assert hasattr(settings, "proactive")
        assert settings.proactive.enabled is True
