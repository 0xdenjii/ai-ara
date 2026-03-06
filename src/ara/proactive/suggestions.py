"""Idle-state proactive suggestions engine.

Queries the knowledge graph for recurring patterns, recent entities,
and temporal triggers to generate contextual suggestions when the user
is idle. Suggestions are categorized and routed through the notification
queue for delivery at natural pauses.

Architecture:
    - Multiple suggestion strategies run in parallel
    - Each strategy produces ProactiveSuggestion candidates
    - Candidates are deduplicated, scored, and ranked
    - Top suggestions are enqueued to the notification system
    - Rate-limited to avoid overwhelming the user

Strategies:
    1. Follow-up: Recent topics that had open threads
    2. Social: People not mentioned in a while
    3. Goals: Goals/projects that haven't been worked on recently
    4. Check-in: Mood-based emotional check-ins
    5. Discovery: Interesting graph patterns (clusters, bridges)
    6. Habit: Fading habits that need a nudge

Usage:
    from ara.proactive.suggestions import SuggestionEngine

    engine = SuggestionEngine(settings.proactive, graph=graph, mood_tracker=mood)
    suggestions = engine.generate()

    for s in suggestions:
        queue.enqueue(title=s.category.value, message=s.message, ...)
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import structlog

from ara.utils.errors import SuggestionError
from ara.utils.types import (
    EntityType,
    NotificationPriority,
    ProactiveSuggestion,
    SuggestionCategory,
)

if TYPE_CHECKING:
    from ara.config.settings import ProactiveSettings
    from ara.memory.graph import KnowledgeGraph
    from ara.proactive.habits import HabitDetector
    from ara.proactive.mood import MoodTracker
    from ara.proactive.notifications import NotificationQueue

log = structlog.get_logger(component="suggestions")

# Thresholds for each strategy
_SOCIAL_DORMANT_DAYS = 14  # Haven't mentioned someone in 2 weeks
_GOAL_STALE_DAYS = 7  # Goal/project untouched for a week
_RECENT_TOPIC_HOURS = 48  # Topics from the last 2 days for follow-ups

# Priority mapping for suggestion categories
_CATEGORY_PRIORITY: dict[SuggestionCategory, NotificationPriority] = {
    SuggestionCategory.CHECK_IN: NotificationPriority.HIGH,
    SuggestionCategory.HABIT: NotificationPriority.HIGH,
    SuggestionCategory.FOLLOW_UP: NotificationPriority.MEDIUM,
    SuggestionCategory.SOCIAL: NotificationPriority.MEDIUM,
    SuggestionCategory.REMINDER: NotificationPriority.MEDIUM,
    SuggestionCategory.DISCOVERY: NotificationPriority.LOW,
}


class SuggestionEngine:
    """Generates contextual proactive suggestions from knowledge graph patterns.

    Runs multiple suggestion strategies in sequence, each producing
    candidate ProactiveSuggestion objects. Candidates are deduplicated
    by entity ID, scored, and the top N are returned.

    Thread-safe — reads from graph and mood tracker under their locks.
    """

    def __init__(
        self,
        settings: ProactiveSettings,
        *,
        graph: KnowledgeGraph,
        mood_tracker: MoodTracker | None = None,
        habit_detector: HabitDetector | None = None,
    ) -> None:
        self._max_per_hour = settings.max_suggestions_per_hour
        self._enabled = settings.enabled
        self._graph = graph
        self._mood_tracker = mood_tracker
        self._habit_detector = habit_detector
        self._last_generation_time: datetime | None = None
        self._generation_count_this_hour = 0
        self._hour_start: datetime | None = None

    def generate(self, *, max_suggestions: int = 3) -> list[ProactiveSuggestion]:
        """Generate proactive suggestions from all strategies.

        Runs each strategy, collects candidates, deduplicates by entity,
        and returns the top N sorted by confidence.

        Args:
            max_suggestions: Maximum number of suggestions to return.

        Returns:
            List of ProactiveSuggestion objects.

        Raises:
            SuggestionError: If suggestion generation fails.
        """
        if not self._enabled:
            return []

        # Rate limit check
        if not self._check_rate_limit():
            log.debug("suggestions_rate_limited")
            return []

        try:
            candidates: list[ProactiveSuggestion] = []

            # Run each strategy
            candidates.extend(self._strategy_social())
            candidates.extend(self._strategy_goals())
            candidates.extend(self._strategy_follow_up())
            candidates.extend(self._strategy_check_in())
            candidates.extend(self._strategy_discovery())
            candidates.extend(self._strategy_habits())

            # Deduplicate by entity ID (keep highest confidence)
            deduplicated = self._deduplicate(candidates)

            # Sort by confidence (highest first)
            deduplicated.sort(key=lambda s: s.confidence, reverse=True)

            result = deduplicated[:max_suggestions]

            self._last_generation_time = datetime.now()
            self._generation_count_this_hour += 1

            log.info(
                "suggestions_generated",
                total_candidates=len(candidates),
                deduplicated=len(deduplicated),
                returned=len(result),
            )

            return result

        except Exception as exc:
            raise SuggestionError(
                f"Failed to generate suggestions: {exc}",
                details={"error": str(exc)},
            ) from exc

    def enqueue_suggestions(
        self,
        suggestions: list[ProactiveSuggestion],
        *,
        notification_queue: NotificationQueue,
    ) -> int:
        """Enqueue generated suggestions to the notification system.

        Args:
            suggestions: Suggestions to enqueue.
            notification_queue: The notification queue to enqueue to.

        Returns:
            Number of suggestions successfully enqueued.
        """
        enqueued = 0
        for suggestion in suggestions:
            priority = _CATEGORY_PRIORITY.get(suggestion.category, NotificationPriority.MEDIUM)
            try:
                notification_queue.enqueue(
                    title=f"{suggestion.category.value.replace('_', ' ').title()}",
                    message=suggestion.message,
                    priority=priority,
                    source="suggestions",
                    metadata={
                        "suggestion_id": suggestion.suggestion_id,
                        "category": suggestion.category.value,
                        "confidence": suggestion.confidence,
                        "source_entities": suggestion.source_entities,
                    },
                )
                enqueued += 1
            except Exception:
                log.warning(
                    "suggestion_enqueue_failed",
                    suggestion_id=suggestion.suggestion_id,
                    exc_info=True,
                )

        return enqueued

    # ── Strategies ────────────────────────────────────────────────────────

    def _strategy_social(self) -> list[ProactiveSuggestion]:
        """Find people the user hasn't mentioned in a while.

        Scans PERSON entities and checks last_seen against the dormancy
        threshold. Only includes people with enough mentions to be
        considered meaningful relationships.
        """
        suggestions: list[ProactiveSuggestion] = []
        cutoff = datetime.now() - timedelta(days=_SOCIAL_DORMANT_DAYS)

        people = self._graph.get_entities_by_type(EntityType.PERSON)
        for person in people:
            # Only suggest for people mentioned at least 3 times (real relationships)
            if person.mention_count < 3:
                continue

            if person.last_seen is not None and person.last_seen < cutoff:
                days_ago = (datetime.now() - person.last_seen).days
                suggestions.append(
                    ProactiveSuggestion(
                        category=SuggestionCategory.SOCIAL,
                        message=(
                            f"You haven't mentioned {person.name} in {days_ago} days. "
                            f"Everything good with them?"
                        ),
                        context=f"Last seen {days_ago} days ago, mentioned {person.mention_count} times total",
                        source_entities=[person.entity_id],
                        confidence=min(1.0, person.mention_count / 10.0),
                    )
                )

        return suggestions

    def _strategy_goals(self) -> list[ProactiveSuggestion]:
        """Find goals and projects that haven't been worked on recently.

        Scans GOAL and PROJECT entities for staleness based on last_seen.
        """
        suggestions: list[ProactiveSuggestion] = []
        cutoff = datetime.now() - timedelta(days=_GOAL_STALE_DAYS)

        for entity_type in (EntityType.GOAL, EntityType.PROJECT):
            entities = self._graph.get_entities_by_type(entity_type)
            for entity in entities:
                if entity.last_seen is not None and entity.last_seen < cutoff:
                    days_stale = (datetime.now() - entity.last_seen).days
                    type_label = entity_type.value

                    suggestions.append(
                        ProactiveSuggestion(
                            category=SuggestionCategory.REMINDER,
                            message=(
                                f"Your {type_label} '{entity.name}' hasn't seen action "
                                f"in {days_stale} days. Want to revisit it?"
                            ),
                            context=f"{type_label.title()} stale for {days_stale} days",
                            source_entities=[entity.entity_id],
                            confidence=min(1.0, 0.3 + days_stale / 30.0),
                        )
                    )

        return suggestions

    def _strategy_follow_up(self) -> list[ProactiveSuggestion]:
        """Suggest follow-ups on recently discussed topics.

        Finds TOPIC and EVENT entities mentioned in the last 48 hours
        that might benefit from a follow-up conversation.
        """
        recent_cutoff = datetime.now() - timedelta(hours=_RECENT_TOPIC_HOURS)

        suggestions: list[ProactiveSuggestion] = []
        for entity_type in (EntityType.TOPIC, EntityType.EVENT):
            entities = self._graph.get_entities_by_type(entity_type)
            suggestions.extend(
                ProactiveSuggestion(
                    category=SuggestionCategory.FOLLOW_UP,
                    message=(
                        f"You were talking about '{entity.name}' recently. Any updates on that?"
                    ),
                    context=f"Discussed {entity.mention_count} times, last seen recently",
                    source_entities=[entity.entity_id],
                    confidence=min(1.0, entity.mention_count / 5.0),
                )
                for entity in entities
                if entity.last_seen is not None
                and entity.last_seen >= recent_cutoff
                and entity.mention_count >= 2
            )

        return suggestions

    def _strategy_check_in(self) -> list[ProactiveSuggestion]:
        """Generate mood-based check-in suggestions.

        Uses the mood tracker to detect significant shifts and suggest
        emotional check-ins.
        """
        if self._mood_tracker is None or not self._mood_tracker.is_connected:
            return []

        suggestions: list[ProactiveSuggestion] = []

        shift_detected, description = self._mood_tracker.detect_significant_shift()
        if shift_detected:
            trend = self._mood_tracker.get_trend()
            if trend.avg_valence < -0.3:
                message = (
                    "I've noticed things have been a bit rough lately. "
                    "Want to talk about it? No pressure."
                )
            elif trend.avg_valence > 0.3:
                message = "You've been in a really good mood recently! Anything exciting going on?"
            else:
                message = "How are you feeling today? Just checking in."

            suggestions.append(
                ProactiveSuggestion(
                    category=SuggestionCategory.CHECK_IN,
                    message=message,
                    context=description,
                    confidence=0.8,
                )
            )

        return suggestions

    def _strategy_discovery(self) -> list[ProactiveSuggestion]:
        """Find interesting graph patterns — bridge entities, growing clusters.

        Looks for entities that connect otherwise separate communities
        (bridge nodes) or clusters that have been growing.
        """
        suggestions: list[ProactiveSuggestion] = []

        # Find entities with high degree centrality (connected to many things)
        try:
            communities = self._graph.detect_communities()
        except Exception as exc:
            log.warning("community_detection_failed", error=str(exc))
            return suggestions

        # If we have multiple communities, look for entities that bridge them
        if len(communities) >= 2:
            # Find entities in multiple communities' neighborhood
            large_communities = [c for c in communities if len(c) >= 3]
            if len(large_communities) >= 2:
                community_b = {e.entity_id for e in large_communities[1]}

                # Look for bridge entities (connected to both communities)
                for entity in large_communities[0]:
                    neighbors = self._graph.get_neighbors(entity.entity_id, depth=1)
                    neighbor_ids = {n.entity_id for n in neighbors}
                    if neighbor_ids & community_b:
                        suggestions.append(
                            ProactiveSuggestion(
                                category=SuggestionCategory.DISCOVERY,
                                message=(
                                    f"Interesting — '{entity.name}' connects two different "
                                    f"areas of your life. That's a pretty unique position."
                                ),
                                context="Bridge entity between communities",
                                source_entities=[entity.entity_id],
                                confidence=0.5,
                            )
                        )
                        break  # Only one discovery per run

        return suggestions

    def _strategy_habits(self) -> list[ProactiveSuggestion]:
        """Generate suggestions from fading habits.

        Uses the HabitDetector to find habits that are fading and
        converts them to suggestions.
        """
        if self._habit_detector is None:
            return []

        fading = self._habit_detector.get_fading_habits()

        return [
            ProactiveSuggestion(
                category=SuggestionCategory.HABIT,
                message=(
                    f"Your habit of {pattern.description} seems to be fading. "
                    f"Last time was "
                    f"{(datetime.now() - pattern.last_seen).days if pattern.last_seen else '?'}"
                    f" days ago."
                ),
                context=f"Habit fading: {pattern.occurrence_count} occurrences, "
                f"avg interval: {pattern.avg_interval_days:.0f} days",
                source_entities=[pattern.entity_id],
                confidence=pattern.confidence,
            )
            for pattern in fading[:2]  # Max 2 habit suggestions per round
        ]

    # ── Internal Helpers ──────────────────────────────────────────────────

    def _deduplicate(self, candidates: list[ProactiveSuggestion]) -> list[ProactiveSuggestion]:
        """Deduplicate suggestions by entity ID, keeping highest confidence.

        Args:
            candidates: All suggestion candidates.

        Returns:
            Deduplicated list.
        """
        seen_entities: dict[str, ProactiveSuggestion] = {}
        no_entity: list[ProactiveSuggestion] = []

        for s in candidates:
            if not s.source_entities:
                no_entity.append(s)
                continue

            key = s.source_entities[0]
            existing = seen_entities.get(key)
            if existing is None or s.confidence > existing.confidence:
                seen_entities[key] = s

        return list(seen_entities.values()) + no_entity

    def _check_rate_limit(self) -> bool:
        """Check if we're within the hourly generation limit.

        Returns:
            True if we can generate more suggestions this hour.
        """
        now = datetime.now()

        # Reset counter at the start of each hour
        if self._hour_start is None or (now - self._hour_start).total_seconds() >= 3600:
            self._hour_start = now
            self._generation_count_this_hour = 0

        return self._generation_count_this_hour < self._max_per_hour
