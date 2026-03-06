"""Weekly and monthly memory review generation.

Synthesizes life chapter summaries from the knowledge graph, conversation
history, mood data, and vector store. Reviews capture the emotional arc,
key milestones, dominant topics, and entity activity over a period.

Architecture:
    - Time-bounded graph scans for entities active in the period
    - Community detection for identifying thematic clusters
    - Mood trend analysis for the emotional arc
    - Conversation count and topic extraction
    - LLM synthesis for narrative summary
    - Stored as MemoryReview and enqueued as notification

Usage:
    from ara.proactive.review import ReviewGenerator

    generator = ReviewGenerator(settings.proactive, graph=graph, conversations=store)
    review = await generator.generate_weekly_review(llm_client=llm)
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import structlog

from ara.utils.errors import ReviewError
from ara.utils.types import ChatMessage, EntityType, MemoryReview, MessageRole

if TYPE_CHECKING:
    from ara.config.settings import ProactiveSettings
    from ara.core.llm import LLMClient
    from ara.memory.conversations import ConversationStore
    from ara.memory.graph import KnowledgeGraph
    from ara.proactive.mood import MoodTracker

log = structlog.get_logger(component="review")

# Day name to weekday number mapping
_DAY_TO_WEEKDAY: dict[str, int] = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}

# Review generation prompt
_REVIEW_PROMPT = """You are ARA, a sarcastic but warm AI best friend. Generate a {period_type}
review summary for your human. Keep it personal and reflective (3-5 sentences).
Don't be generic — reference specific entities, achievements, and emotional trends.

Period: {start_date} to {end_date}

Key entities active this period: {entities}
Dominant topics: {topics}
People mentioned: {people}
Conversation count: {conv_count}
Mood arc: {mood_arc}
Milestones: {milestones}

Generate a warm, reflective {period_type} summary. Acknowledge both achievements
and challenges. Make it feel like a friend recapping the period."""

# Fallback template
_FALLBACK_TEMPLATE = (
    "This {period_type} ({start_date} to {end_date}): "
    "{conv_count} conversation{plural}. "
    "Main topics: {topics_str}. "
    "{mood_line}"
    "Key entities: {entities_str}."
)


class ReviewGenerator:
    """Generates weekly and monthly memory reviews.

    Scans the knowledge graph and conversation history for a time period,
    identifies key entities, topics, mood trends, and milestones, then
    synthesizes a narrative summary.

    Thread-safe — reads from graph, conversations, and mood tracker
    under their respective locks.
    """

    def __init__(
        self,
        settings: ProactiveSettings,
        *,
        graph: KnowledgeGraph,
        conversations: ConversationStore,
        mood_tracker: MoodTracker | None = None,
    ) -> None:
        self._weekly_enabled = settings.weekly_review_enabled
        self._monthly_enabled = settings.monthly_review_enabled
        self._weekly_review_day = settings.weekly_review_day
        self._monthly_review_day = settings.monthly_review_day
        self._graph = graph
        self._conversations = conversations
        self._mood_tracker = mood_tracker

    async def generate_weekly_review(
        self,
        *,
        llm_client: LLMClient | None = None,
        end_date: datetime | None = None,
    ) -> MemoryReview | None:
        """Generate a weekly memory review.

        Args:
            llm_client: LLM client for narrative generation.
            end_date: End of the review period (defaults to now).

        Returns:
            MemoryReview object, or None if disabled.
        """
        if not self._weekly_enabled:
            return None

        end = end_date or datetime.now()
        start = end - timedelta(days=7)

        return await self._generate_review(
            period_type="weekly",
            start_date=start,
            end_date=end,
            llm_client=llm_client,
        )

    async def generate_monthly_review(
        self,
        *,
        llm_client: LLMClient | None = None,
        end_date: datetime | None = None,
    ) -> MemoryReview | None:
        """Generate a monthly memory review.

        Args:
            llm_client: LLM client for narrative generation.
            end_date: End of the review period (defaults to now).

        Returns:
            MemoryReview object, or None if disabled.
        """
        if not self._monthly_enabled:
            return None

        end = end_date or datetime.now()
        start = end - timedelta(days=30)

        return await self._generate_review(
            period_type="monthly",
            start_date=start,
            end_date=end,
            llm_client=llm_client,
        )

    def is_weekly_review_day(self) -> bool:
        """Check if today is the configured weekly review day.

        Returns:
            True if today matches the weekly review day setting.
        """
        target_weekday = _DAY_TO_WEEKDAY.get(self._weekly_review_day, 6)
        return datetime.now().weekday() == target_weekday

    def is_monthly_review_day(self) -> bool:
        """Check if today is the configured monthly review day.

        Returns:
            True if today's date matches the monthly review day setting.
        """
        return datetime.now().day == self._monthly_review_day

    # ── Core Review Generation ────────────────────────────────────────────

    async def _generate_review(
        self,
        *,
        period_type: str,
        start_date: datetime,
        end_date: datetime,
        llm_client: LLMClient | None = None,
    ) -> MemoryReview:
        """Generate a review for the specified period.

        Args:
            period_type: "weekly" or "monthly".
            start_date: Start of the period.
            end_date: End of the period.
            llm_client: Optional LLM client.

        Returns:
            MemoryReview object.

        Raises:
            ReviewError: If review generation fails.
        """
        try:
            # Gather data for the period
            key_entities = self._get_active_entities(start_date, end_date)
            key_topics = self._get_active_topics(start_date, end_date)
            people = self._get_active_people(start_date, end_date)
            conv_count = self._count_conversations(start_date, end_date)
            milestones = self._get_milestones(start_date, end_date)
            mood_arc = self._get_mood_arc(start_date, end_date)
            connections_made = self._count_new_connections(start_date, end_date)

            # Generate narrative
            entity_names = [e.name for e in key_entities[:10]]
            topic_names = [e.name for e in key_topics[:10]]
            people_names = [e.name for e in people[:10]]
            milestone_names = [e.name for e in milestones[:5]]

            if llm_client is not None:
                summary = await self._generate_llm_review(
                    llm_client=llm_client,
                    period_type=period_type,
                    start_date=start_date,
                    end_date=end_date,
                    entities=entity_names,
                    topics=topic_names,
                    people=people_names,
                    conv_count=conv_count,
                    mood_arc=mood_arc,
                    milestones=milestone_names,
                )
            else:
                summary = self._generate_fallback_review(
                    period_type=period_type,
                    start_date=start_date,
                    end_date=end_date,
                    entities=entity_names,
                    topics=topic_names,
                    conv_count=conv_count,
                    mood_arc=mood_arc,
                )

            review = MemoryReview(
                period_type=period_type,
                start_date=start_date,
                end_date=end_date,
                summary=summary,
                key_entities=entity_names,
                key_topics=topic_names,
                mood_summary=mood_arc,
                milestones=milestone_names,
                connections_made=connections_made,
            )

            log.info(
                "review_generated",
                period_type=period_type,
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                entities=len(key_entities),
                conversations=conv_count,
            )

            return review

        except ReviewError:
            raise
        except Exception as exc:
            raise ReviewError(
                f"Failed to generate {period_type} review: {exc}",
                details={
                    "period_type": period_type,
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat(),
                },
            ) from exc

    # ── Data Gathering ────────────────────────────────────────────────────

    def _get_active_entities(self, start: datetime, end: datetime) -> list[object]:
        """Get entities that were active during the period.

        Returns entities sorted by mention count (most active first).
        """
        from ara.utils.types import GraphEntity

        active: list[GraphEntity] = []
        for entity_type in EntityType:
            entities = self._graph.get_entities_by_type(entity_type)
            active.extend(
                entity
                for entity in entities
                if entity.last_seen is not None and start <= entity.last_seen <= end
            )

        active.sort(key=lambda e: e.mention_count, reverse=True)
        return active[:20]

    def _get_active_topics(self, start: datetime, end: datetime) -> list[object]:
        """Get topic entities active during the period."""
        from ara.utils.types import GraphEntity

        topics = self._graph.get_entities_by_type(EntityType.TOPIC)
        active: list[GraphEntity] = [
            t for t in topics if t.last_seen is not None and start <= t.last_seen <= end
        ]
        active.sort(key=lambda e: e.mention_count, reverse=True)
        return active[:10]

    def _get_active_people(self, start: datetime, end: datetime) -> list[object]:
        """Get person entities active during the period."""
        from ara.utils.types import GraphEntity

        people = self._graph.get_entities_by_type(EntityType.PERSON)
        active: list[GraphEntity] = [
            p for p in people if p.last_seen is not None and start <= p.last_seen <= end
        ]
        active.sort(key=lambda e: e.mention_count, reverse=True)
        return active[:10]

    def _get_milestones(self, start: datetime, end: datetime) -> list[object]:
        """Get milestone entities from the period."""
        from ara.utils.types import GraphEntity

        milestones = self._graph.get_entities_by_type(EntityType.MILESTONE)
        active: list[GraphEntity] = [
            m for m in milestones if m.last_seen is not None and start <= m.last_seen <= end
        ]
        return active

    def _count_conversations(self, start: datetime, end: datetime) -> int:
        """Count conversations in the period."""
        recent = self._conversations.get_recent_conversations(limit=200, include_active=False)
        return sum(1 for c in recent if start <= c.started_at <= end)

    def _count_new_connections(self, start: datetime, end: datetime) -> int:
        """Estimate new graph edges created during the period.

        Uses entity first_seen timestamps as a proxy for new connections.
        """
        count = 0
        for entity_type in EntityType:
            entities = self._graph.get_entities_by_type(entity_type)
            for entity in entities:
                if entity.first_seen is not None and start <= entity.first_seen <= end:
                    count += 1
        return count

    def _get_mood_arc(self, start: datetime, end: datetime) -> str:
        """Get a textual description of the mood arc over the period."""
        if self._mood_tracker is None or not self._mood_tracker.is_connected:
            return "no mood data available"

        entries = self._mood_tracker.get_entries(start=start, end=end, limit=500)
        if not entries:
            return "no mood data for this period"

        avg_valence = sum(e.valence for e in entries) / len(entries)

        # Split into halves for arc
        mid = len(entries) // 2
        if mid > 0:
            recent_avg = sum(e.valence for e in entries[:mid]) / mid
            older_avg = sum(e.valence for e in entries[mid:]) / (len(entries) - mid)

            if recent_avg > older_avg + 0.2:
                arc = "started lower but ended on an upswing"
            elif recent_avg < older_avg - 0.2:
                arc = "started well but dipped toward the end"
            else:
                arc = "relatively stable throughout"
        else:
            arc = "limited data"

        if avg_valence > 0.3:
            tone = "generally positive"
        elif avg_valence < -0.3:
            tone = "somewhat challenging"
        else:
            tone = "mixed"

        return f"{tone} period, {arc}"

    # ── Summary Generation ────────────────────────────────────────────────

    async def _generate_llm_review(
        self,
        *,
        llm_client: LLMClient,
        period_type: str,
        start_date: datetime,
        end_date: datetime,
        entities: list[str],
        topics: list[str],
        people: list[str],
        conv_count: int,
        mood_arc: str,
        milestones: list[str],
    ) -> str:
        """Generate narrative review using the LLM."""
        prompt = _REVIEW_PROMPT.format(
            period_type=period_type,
            start_date=start_date.strftime("%B %d"),
            end_date=end_date.strftime("%B %d, %Y"),
            entities=", ".join(entities) if entities else "quiet period",
            topics=", ".join(topics) if topics else "no dominant topics",
            people=", ".join(people) if people else "no one in particular",
            conv_count=conv_count,
            mood_arc=mood_arc,
            milestones=", ".join(milestones) if milestones else "none noted",
        )

        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content="You are ARA."),
            ChatMessage(role=MessageRole.USER, content=prompt),
        ]

        try:
            response = await llm_client.generate(messages)
            return (
                response.strip()
                if response
                else self._generate_fallback_review(
                    period_type=period_type,
                    start_date=start_date,
                    end_date=end_date,
                    entities=entities,
                    topics=topics,
                    conv_count=conv_count,
                    mood_arc=mood_arc,
                )
            )
        except Exception:
            log.warning("review_llm_generation_failed", exc_info=True)
            return self._generate_fallback_review(
                period_type=period_type,
                start_date=start_date,
                end_date=end_date,
                entities=entities,
                topics=topics,
                conv_count=conv_count,
                mood_arc=mood_arc,
            )

    def _generate_fallback_review(
        self,
        *,
        period_type: str,
        start_date: datetime,
        end_date: datetime,
        entities: list[str],
        topics: list[str],
        conv_count: int,
        mood_arc: str,
    ) -> str:
        """Generate template-based review when LLM is unavailable."""
        plural = "s" if conv_count != 1 else ""
        topics_str = ", ".join(topics[:5]) if topics else "various things"
        entities_str = ", ".join(entities[:5]) if entities else "various entities"
        mood_line = f"Mood: {mood_arc}. " if mood_arc else ""

        return _FALLBACK_TEMPLATE.format(
            period_type=period_type,
            start_date=start_date.strftime("%B %d"),
            end_date=end_date.strftime("%B %d"),
            conv_count=conv_count,
            plural=plural,
            topics_str=topics_str,
            mood_line=mood_line,
            entities_str=entities_str,
        )
