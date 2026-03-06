"""Habit pattern detection from knowledge graph edge frequencies.

Analyzes recurring entity interactions in the knowledge graph to detect
behavioral patterns — e.g., "exercises every morning", "calls mom weekly",
"works on Work project most days". Generates gentle nudges when habits
are broken or maintained.

Architecture:
    - Scans graph edges for recurring entity mentions (by relation type)
    - Computes frequency, average interval, and recency for each pattern
    - Classifies patterns as active, fading, or broken
    - Generates context-aware nudges via LLM for broken/fading habits

Usage:
    from ara.proactive.habits import HabitDetector

    detector = HabitDetector(settings.proactive, graph=graph)
    patterns = detector.detect_patterns()
    nudges = await detector.generate_nudges(patterns, llm_client=llm)
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import structlog

from ara.utils.types import (
    EntityType,
    GraphEntity,
    GraphRelation,
    HabitPattern,
    RelationType,
)

if TYPE_CHECKING:
    from ara.config.settings import ProactiveSettings
    from ara.core.llm import LLMClient
    from ara.memory.graph import KnowledgeGraph

log = structlog.get_logger(component="habits")

# Entity types most likely to form habits
_HABIT_ENTITY_TYPES: frozenset[EntityType] = frozenset(
    {
        EntityType.PERSON,
        EntityType.HABIT,
        EntityType.GOAL,
        EntityType.PROJECT,
        EntityType.HEALTH,
        EntityType.TOPIC,
        EntityType.PLACE,
    }
)

# Relation types that indicate habitual interaction
_HABIT_RELATION_TYPES: frozenset[RelationType] = frozenset(
    {
        RelationType.INTERACTED_WITH,
        RelationType.WORKED_ON,
        RelationType.DISCUSSED,
        RelationType.EXPERIENCED,
        RelationType.CONSUMED,
        RelationType.LOCATED_AT,
    }
)

# Prompt for generating habit nudges
_NUDGE_PROMPT = """You are ARA, a sarcastic but warm AI best friend. Generate a brief,
friendly nudge about a habit pattern. Keep it under 2 sentences. Be warm, not naggy.

Habit: {description}
Last seen: {last_seen}
Days since last: {days_since}
Status: {status}

Generate a natural, conversational nudge. Don't be preachy or lecture-like.
If the habit is positive and maintained, acknowledge it warmly.
If it's fading, gently remind without guilt-tripping."""


class HabitDetector:
    """Detects recurring behavioral patterns from knowledge graph edges.

    Scans the graph for entity interactions that recur with regularity,
    computing frequency and interval statistics. Patterns are classified as:
    - active: habit maintained within expected interval
    - fading: interval exceeded by 1.5x (gentle nudge territory)
    - broken: interval exceeded by 3x (stronger nudge)

    Thread-safe — reads from graph under its internal lock.
    """

    def __init__(
        self,
        settings: ProactiveSettings,
        *,
        graph: KnowledgeGraph,
    ) -> None:
        self._min_occurrences = settings.habit_min_occurrences
        self._lookback_days = settings.habit_lookback_days
        self._enabled = settings.habit_detection_enabled
        self._graph = graph

    def detect_patterns(self) -> list[HabitPattern]:
        """Scan the knowledge graph for recurring behavior patterns.

        Finds entities with multiple time-stamped relation edges,
        computes frequency and interval statistics, and returns
        HabitPattern objects sorted by confidence.

        Returns:
            List of detected HabitPattern objects.
        """
        if not self._enabled:
            return []

        cutoff = datetime.now() - timedelta(days=self._lookback_days)
        patterns: list[HabitPattern] = []

        # Scan entities of habit-forming types
        for entity_type in _HABIT_ENTITY_TYPES:
            entities = self._graph.get_entities_by_type(entity_type)
            for entity in entities:
                entity_patterns = self._analyze_entity(entity, cutoff)
                patterns.extend(entity_patterns)

        # Sort by confidence (highest first)
        patterns.sort(key=lambda p: p.confidence, reverse=True)

        log.info("habit_patterns_detected", count=len(patterns))
        return patterns

    def classify_pattern(self, pattern: HabitPattern) -> str:
        """Classify a habit pattern's current status.

        Args:
            pattern: The HabitPattern to classify.

        Returns:
            One of "active", "fading", or "broken".
        """
        if pattern.last_seen is None or pattern.avg_interval_days <= 0:
            return "active"

        days_since = (datetime.now() - pattern.last_seen).total_seconds() / 86400

        if days_since <= pattern.avg_interval_days * 1.5:
            return "active"
        if days_since <= pattern.avg_interval_days * 3.0:
            return "fading"
        return "broken"

    async def generate_nudges(
        self,
        patterns: list[HabitPattern],
        *,
        llm_client: LLMClient,
    ) -> list[dict[str, str]]:
        """Generate LLM-powered nudges for fading/broken habits.

        Only generates nudges for patterns that are fading or broken —
        active habits are acknowledged without a nudge.

        Args:
            patterns: Detected HabitPattern list.
            llm_client: LLM client for generating natural nudge text.

        Returns:
            List of dicts with keys: pattern_id, status, nudge_text.
        """
        from ara.utils.types import ChatMessage, MessageRole

        nudges: list[dict[str, str]] = []

        for pattern in patterns:
            status = self.classify_pattern(pattern)
            if status == "active":
                continue

            days_since = 0.0
            if pattern.last_seen:
                days_since = (datetime.now() - pattern.last_seen).total_seconds() / 86400

            prompt = _NUDGE_PROMPT.format(
                description=pattern.description,
                last_seen=(
                    pattern.last_seen.strftime("%Y-%m-%d") if pattern.last_seen else "unknown"
                ),
                days_since=f"{days_since:.0f}",
                status=status,
            )

            try:
                messages = [
                    ChatMessage(role=MessageRole.SYSTEM, content="You are ARA."),
                    ChatMessage(role=MessageRole.USER, content=prompt),
                ]
                response = await llm_client.generate(messages)
                nudge_text = response.strip() if response else ""

                nudges.append(
                    {
                        "pattern_id": pattern.pattern_id,
                        "status": status,
                        "nudge_text": nudge_text,
                        "entity_name": pattern.entity_name,
                    }
                )
            except Exception:
                log.warning(
                    "habit_nudge_generation_failed",
                    pattern_id=pattern.pattern_id,
                    exc_info=True,
                )
                # Fallback: simple nudge without LLM
                nudges.append(
                    {
                        "pattern_id": pattern.pattern_id,
                        "status": status,
                        "nudge_text": (
                            f"Hey, it's been {days_since:.0f} days since "
                            f"you last {pattern.description.lower()}. "
                            "Just a gentle reminder."
                        ),
                        "entity_name": pattern.entity_name,
                    }
                )

        return nudges

    def get_active_habits(self) -> list[HabitPattern]:
        """Get only currently active (maintained) habits.

        Returns:
            List of HabitPattern objects classified as "active".
        """
        patterns = self.detect_patterns()
        return [p for p in patterns if self.classify_pattern(p) == "active"]

    def get_fading_habits(self) -> list[HabitPattern]:
        """Get habits that are starting to fade.

        Returns:
            List of HabitPattern objects classified as "fading".
        """
        patterns = self.detect_patterns()
        return [p for p in patterns if self.classify_pattern(p) == "fading"]

    # ── Internal Analysis ─────────────────────────────────────────────────

    def _analyze_entity(
        self,
        entity: GraphEntity,
        cutoff: datetime,
    ) -> list[HabitPattern]:
        """Analyze a single entity for habitual interaction patterns.

        Looks at all relations involving this entity, groups by relation type,
        and identifies patterns with sufficient recurrence.

        Args:
            entity: The entity to analyze.
            cutoff: Only consider relations after this datetime.

        Returns:
            List of HabitPattern objects for this entity.
        """
        relations = self._graph.get_relations(entity.entity_id, direction="both")

        # Group relations by type
        by_type: dict[RelationType, list[GraphRelation]] = {}
        for rel in relations:
            if rel.relation_type not in _HABIT_RELATION_TYPES:
                continue
            by_type.setdefault(rel.relation_type, []).append(rel)

        patterns: list[HabitPattern] = []

        for rel_type, rels in by_type.items():
            # Filter to relations within lookback window
            recent_rels = [r for r in rels if r.last_seen is not None and r.last_seen >= cutoff]

            # Also count by mention_count on the entity itself as a proxy
            occurrence_count = len(recent_rels)
            if occurrence_count < self._min_occurrences:
                # Fall back to entity mention_count if we have few timestamped relations
                if entity.mention_count >= self._min_occurrences:
                    occurrence_count = entity.mention_count
                else:
                    continue

            # Compute temporal stats
            timestamps = sorted([r.last_seen for r in recent_rels if r.last_seen is not None])

            avg_interval = 0.0
            if len(timestamps) >= 2:
                intervals = [
                    (timestamps[i + 1] - timestamps[i]).total_seconds() / 86400
                    for i in range(len(timestamps) - 1)
                ]
                avg_interval = sum(intervals) / len(intervals) if intervals else 0.0

            first_seen = timestamps[0] if timestamps else entity.first_seen
            last_seen = timestamps[-1] if timestamps else entity.last_seen

            # Build description
            description = self._build_description(entity, rel_type)

            # Confidence based on occurrence count and regularity
            regularity_score = self._compute_regularity(timestamps)
            confidence = min(1.0, (occurrence_count / 10.0) * 0.5 + regularity_score * 0.5)

            # Positive habit detection (health, goals are positive; generic are neutral)
            is_positive = entity.entity_type in (
                EntityType.HEALTH,
                EntityType.GOAL,
                EntityType.HABIT,
            )

            pattern = HabitPattern(
                description=description,
                entity_id=entity.entity_id,
                entity_name=entity.name,
                relation_type=rel_type.value,
                occurrence_count=occurrence_count,
                first_seen=first_seen,
                last_seen=last_seen,
                avg_interval_days=round(avg_interval, 1),
                confidence=round(confidence, 3),
                is_positive=is_positive,
            )
            patterns.append(pattern)

        return patterns

    def _build_description(self, entity: GraphEntity, rel_type: RelationType) -> str:
        """Build a human-readable description of the habit pattern.

        Args:
            entity: The entity involved.
            rel_type: The relation type.

        Returns:
            Natural language description.
        """
        verb_map: dict[RelationType, str] = {
            RelationType.INTERACTED_WITH: "interacting with",
            RelationType.WORKED_ON: "working on",
            RelationType.DISCUSSED: "discussing",
            RelationType.EXPERIENCED: "experiencing",
            RelationType.CONSUMED: "engaging with",
            RelationType.LOCATED_AT: "being at",
        }

        verb = verb_map.get(rel_type, "engaging with")
        return f"{verb} {entity.name}"

    def _compute_regularity(self, timestamps: list[datetime]) -> float:
        """Compute how regular a pattern is (0.0 = random, 1.0 = clockwork).

        Uses coefficient of variation of intervals — lower CV = more regular.

        Args:
            timestamps: Sorted list of occurrence timestamps.

        Returns:
            Regularity score between 0.0 and 1.0.
        """
        if len(timestamps) < 3:
            return 0.5  # Not enough data to judge

        intervals = [
            (timestamps[i + 1] - timestamps[i]).total_seconds() / 86400
            for i in range(len(timestamps) - 1)
        ]

        mean_interval = sum(intervals) / len(intervals)
        if mean_interval == 0:
            return 0.5

        variance = sum((x - mean_interval) ** 2 for x in intervals) / len(intervals)
        std_dev = variance**0.5
        cv = std_dev / mean_interval  # coefficient of variation

        # Convert CV to regularity score (lower CV = higher regularity)
        # CV of 0 = perfect regularity (1.0), CV of 2+ = very irregular (0.0)
        regularity = max(0.0, 1.0 - cv / 2.0)
        return round(regularity, 3)
