"""Daily recap generation — end-of-day summary of interactions and events.

Gathers today's conversations from the ConversationStore, extracts topics
and entities from the knowledge graph, computes mood summary, and generates
a narrative recap via the LLM.

Architecture:
    - Conversation data from ConversationStore.get_recent_conversations()
    - Entity/topic extraction from KnowledgeGraph (entities seen today)
    - Mood summary from MoodTracker (today's valence/dominant emotion)
    - LLM synthesis for natural-language narrative
    - Result stored as DailyRecap and enqueued as notification

Usage:
    from ara.proactive.recap import RecapGenerator

    generator = RecapGenerator(settings.proactive, conversations=store, graph=graph)
    recap = await generator.generate_daily_recap(llm_client=llm)
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import structlog

from ara.utils.errors import RecapError
from ara.utils.types import ChatMessage, DailyRecap, MessageRole

if TYPE_CHECKING:
    from ara.config.settings import ProactiveSettings
    from ara.core.llm import LLMClient
    from ara.memory.conversations import ConversationStore
    from ara.memory.graph import KnowledgeGraph
    from ara.proactive.mood import MoodTracker

log = structlog.get_logger(component="recap")

# Recap generation prompt template
_RECAP_PROMPT = """You are ARA, a sarcastic but warm AI best friend. Generate a brief daily
recap for your human. Keep it conversational and personal (2-4 sentences).
Don't be generic — reference specific topics and people mentioned.

Today's conversations:
{conversation_summary}

Topics discussed: {topics}
People mentioned: {people}
Mood: {mood_summary}
Number of conversations: {conv_count}

Generate a warm, personal end-of-day summary. Be specific, not generic."""

# Fallback recap when LLM is unavailable
_FALLBACK_TEMPLATE = (
    "Today you had {conv_count} conversation{plural}. Topics included: {topics_str}. {mood_line}"
)


class RecapGenerator:
    """Generates daily recap summaries from conversation and memory data.

    Collects today's conversations, extracts entities and topics from
    the knowledge graph, and synthesizes a narrative summary using the
    LLM. Falls back to a template-based summary if the LLM is unavailable.

    Thread-safe — reads from ConversationStore and KnowledgeGraph
    under their respective locks.
    """

    def __init__(
        self,
        settings: ProactiveSettings,
        *,
        conversations: ConversationStore,
        graph: KnowledgeGraph,
        mood_tracker: MoodTracker | None = None,
    ) -> None:
        self._enabled = settings.daily_recap_enabled
        self._min_interactions = settings.min_interactions_for_recap
        self._conversations = conversations
        self._graph = graph
        self._mood_tracker = mood_tracker

    async def generate_daily_recap(
        self,
        *,
        llm_client: LLMClient | None = None,
        target_date: datetime | None = None,
    ) -> DailyRecap | None:
        """Generate a daily recap for the specified date (defaults to today).

        Args:
            llm_client: LLM client for narrative generation. If None,
                uses template-based fallback.
            target_date: Date to generate recap for (defaults to today).

        Returns:
            DailyRecap object, or None if not enough data.

        Raises:
            RecapError: If recap generation fails catastrophically.
        """
        if not self._enabled:
            return None

        date = target_date or datetime.now()

        try:
            # Gather today's conversations
            today_convs = self._get_todays_conversations(date)

            if len(today_convs) < self._min_interactions:
                log.debug(
                    "recap_skipped_insufficient_data",
                    conversations=len(today_convs),
                    minimum=self._min_interactions,
                )
                return None

            # Extract data
            topics = self._extract_topics(today_convs)
            people = self._extract_people(date)
            entities_mentioned = self._extract_entities_today(date)
            mood_summary = self._get_mood_summary()
            conversation_summary = self._build_conversation_summary(today_convs)

            # Generate narrative
            if llm_client is not None:
                summary = await self._generate_llm_summary(
                    llm_client=llm_client,
                    conversation_summary=conversation_summary,
                    topics=topics,
                    people=people,
                    mood_summary=mood_summary,
                    conv_count=len(today_convs),
                )
            else:
                summary = self._generate_fallback_summary(
                    topics=topics,
                    mood_summary=mood_summary,
                    conv_count=len(today_convs),
                )

            # Extract highlights (turns with strong emotions or key decisions)
            highlights = self._extract_highlights(today_convs)

            recap = DailyRecap(
                date=date,
                summary=summary,
                conversation_count=len(today_convs),
                topics_discussed=topics,
                entities_mentioned=entities_mentioned,
                mood_summary=mood_summary,
                highlights=highlights,
            )

            log.info(
                "daily_recap_generated",
                date=date.strftime("%Y-%m-%d"),
                conversations=len(today_convs),
                topics=len(topics),
            )

            return recap

        except RecapError:
            raise
        except Exception as exc:
            raise RecapError(
                f"Failed to generate daily recap: {exc}",
                details={"date": date.isoformat(), "error": str(exc)},
            ) from exc

    # ── Data Gathering ────────────────────────────────────────────────────

    def _get_todays_conversations(self, date: datetime) -> list[dict[str, object]]:
        """Get conversations from the target date.

        Args:
            date: The target date.

        Returns:
            List of conversation dicts with key fields.
        """
        # Get recent conversations (enough to cover today)
        recent = self._conversations.get_recent_conversations(limit=50, include_active=False)

        # Filter to today
        start_of_day = date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_day = start_of_day + timedelta(days=1)

        today_convs: list[dict[str, object]] = []
        for conv in recent:
            if start_of_day <= conv.started_at < end_of_day:
                turns_text: list[str] = []
                for turn in conv.turns:
                    turns_text.append(f"User: {turn.user_message}")
                    turns_text.append(f"ARA: {turn.assistant_message}")

                today_convs.append(
                    {
                        "id": conv.conversation_id,
                        "started_at": conv.started_at.strftime("%H:%M"),
                        "turn_count": conv.turn_count,
                        "summary": conv.summary,
                        "turns_text": turns_text,
                    }
                )

        return today_convs

    def _extract_topics(self, conversations: list[dict[str, object]]) -> list[str]:
        """Extract topic names from conversations via their summaries.

        Uses conversation summaries as topic proxies. Also checks
        graph for TOPIC entities mentioned today.

        Args:
            conversations: Today's conversations.

        Returns:
            List of topic strings.
        """
        topics: list[str] = []
        for conv in conversations:
            summary = str(conv.get("summary", ""))
            if summary:
                # Use first sentence of summary as topic
                first_sentence = summary.split(".")[0].strip()
                if first_sentence and first_sentence not in topics:
                    topics.append(first_sentence)

        # Also get TOPIC entities from graph that were recently seen
        from ara.utils.types import EntityType

        topic_entities = self._graph.get_entities_by_type(EntityType.TOPIC)
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        for entity in topic_entities:
            if (
                entity.last_seen is not None
                and entity.last_seen >= today
                and entity.name not in topics
            ):
                topics.append(entity.name)

        return topics[:10]  # Cap at 10 topics

    def _extract_people(self, date: datetime) -> list[str]:
        """Get people mentioned today from the knowledge graph.

        Args:
            date: The target date.

        Returns:
            List of person names.
        """
        from ara.utils.types import EntityType

        people = self._graph.get_entities_by_type(EntityType.PERSON)
        today = date.replace(hour=0, minute=0, second=0, microsecond=0)

        mentioned_today = [
            p.name for p in people if p.last_seen is not None and p.last_seen >= today
        ]
        return mentioned_today

    def _extract_entities_today(self, date: datetime) -> list[str]:
        """Get all entity names mentioned today.

        Args:
            date: The target date.

        Returns:
            List of entity names.
        """
        today = date.replace(hour=0, minute=0, second=0, microsecond=0)
        entities: list[str] = []

        from ara.utils.types import EntityType

        for entity_type in EntityType:
            type_entities = self._graph.get_entities_by_type(entity_type)
            entities.extend(
                entity.name
                for entity in type_entities
                if entity.last_seen is not None and entity.last_seen >= today
            )

        return entities[:20]  # Cap

    def _get_mood_summary(self) -> str:
        """Get today's mood summary from the mood tracker.

        Returns:
            Human-readable mood description, or empty string.
        """
        if self._mood_tracker is None or not self._mood_tracker.is_connected:
            return ""

        trend = self._mood_tracker.get_trend(window_days=1)
        if trend.data_points == 0:
            return ""

        valence_desc = "neutral"
        if trend.avg_valence > 0.6:
            valence_desc = "very positive"
        elif trend.avg_valence > 0.3:
            valence_desc = "positive"
        elif trend.avg_valence < -0.6:
            valence_desc = "very low"
        elif trend.avg_valence < -0.3:
            valence_desc = "low"

        dominant = trend.dominant_emotion
        if dominant:
            return f"Overall mood: {valence_desc} (dominant: {dominant})"
        return f"Overall mood: {valence_desc}"

    def _build_conversation_summary(self, conversations: list[dict[str, object]]) -> str:
        """Build a text summary of today's conversations for the LLM prompt.

        Args:
            conversations: Today's conversations.

        Returns:
            Formatted text summary.
        """
        parts: list[str] = []
        for i, conv in enumerate(conversations, 1):
            started = conv.get("started_at", "")
            turn_count = conv.get("turn_count", 0)
            summary = conv.get("summary", "")

            if summary:
                parts.append(f"{i}. [{started}] ({turn_count} turns): {summary}")
            else:
                parts.append(f"{i}. [{started}] ({turn_count} turns)")

        return "\n".join(parts) if parts else "No conversations recorded."

    def _extract_highlights(self, conversations: list[dict[str, object]]) -> list[str]:
        """Extract notable moments from today's conversations.

        Uses conversation summaries as highlights. In the future, this
        could use emotion analysis to find emotionally significant moments.

        Args:
            conversations: Today's conversations.

        Returns:
            List of highlight strings.
        """
        highlights: list[str] = []
        for conv in conversations:
            summary = str(conv.get("summary", ""))
            if summary and len(summary) > 20:
                highlights.append(summary)

        return highlights[:5]  # Cap at 5 highlights

    # ── Summary Generation ────────────────────────────────────────────────

    async def _generate_llm_summary(
        self,
        *,
        llm_client: LLMClient,
        conversation_summary: str,
        topics: list[str],
        people: list[str],
        mood_summary: str,
        conv_count: int,
    ) -> str:
        """Generate a narrative recap using the LLM.

        Args:
            llm_client: The LLM client.
            conversation_summary: Formatted conversation text.
            topics: List of topic strings.
            people: List of people mentioned.
            mood_summary: Mood description.
            conv_count: Number of conversations.

        Returns:
            LLM-generated narrative summary.
        """
        prompt = _RECAP_PROMPT.format(
            conversation_summary=conversation_summary,
            topics=", ".join(topics) if topics else "nothing specific",
            people=", ".join(people) if people else "no one in particular",
            mood_summary=mood_summary or "no mood data",
            conv_count=conv_count,
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
                else self._generate_fallback_summary(
                    topics=topics, mood_summary=mood_summary, conv_count=conv_count
                )
            )
        except Exception:
            log.warning("recap_llm_generation_failed", exc_info=True)
            return self._generate_fallback_summary(
                topics=topics, mood_summary=mood_summary, conv_count=conv_count
            )

    def _generate_fallback_summary(
        self,
        *,
        topics: list[str],
        mood_summary: str,
        conv_count: int,
    ) -> str:
        """Generate a template-based recap when LLM is unavailable.

        Args:
            topics: List of topic strings.
            mood_summary: Mood description.
            conv_count: Number of conversations.

        Returns:
            Template-based summary string.
        """
        plural = "s" if conv_count != 1 else ""
        topics_str = ", ".join(topics[:5]) if topics else "various things"
        mood_line = f" {mood_summary}." if mood_summary else ""

        return _FALLBACK_TEMPLATE.format(
            conv_count=conv_count,
            plural=plural,
            topics_str=topics_str,
            mood_line=mood_line,
        )
