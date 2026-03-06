"""Morning briefing and end-of-day recap generator from Obsidian data.

Generates structured briefings by reading yesterday's daily note, combining it
with ARA's conversation memory, mood trends, and habit data. Also generates
end-of-day recaps that are written back to the daily note.

Morning Briefing (e.g. at 7 AM):
    - What happened yesterday (from Obsidian + ARA conversations)
    - Incomplete tasks carried over
    - Protocol score from yesterday
    - Habit tracking status
    - Mood context and suggestions for today

End-of-Day Recap (e.g. at 11 PM):
    - Summary of today's conversations with ARA
    - Tasks completed/started today
    - Emotional arc of the day
    - Protocol score for today
    - Written back to the daily note

Usage:
    from ara.obsidian.daily_briefing import BriefingGenerator

    gen = BriefingGenerator(settings, sync, writer)
    briefing = await gen.generate_morning_briefing()
    recap = await gen.generate_end_of_day_recap()
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import structlog

from ara.utils.types import (
    DailyNoteData,
    MorningBriefing,
    ObsidianTaskStatus,
)

if TYPE_CHECKING:
    from ara.config.settings import ObsidianSettings
    from ara.core.llm import LLMClient
    from ara.memory.conversations import ConversationStore
    from ara.obsidian.habits import ProtocolTracker
    from ara.obsidian.sync import ObsidianSync
    from ara.obsidian.writer import DiaryWriter
    from ara.proactive.mood import MoodTracker

log = structlog.get_logger(component="obsidian.briefing")

# ── LLM Prompt Templates ────────────────────────────────────────────────────

_MORNING_BRIEFING_PROMPT = """You are ARA, Denji's sarcastic but warm AI best friend.
Generate a morning briefing based on the following context from yesterday.

Yesterday's date: {yesterday_date}
Yesterday's daily note summary: {yesterday_summary}
Incomplete tasks from yesterday: {pending_tasks}
Protocol score yesterday: {protocol_score}%
Habit status: {habit_status}
Recent mood: {mood_context}

Write a brief, punchy morning briefing in first person as ARA. Be warm but real.
Include:
1. Quick summary of yesterday (2-3 sentences)
2. Tasks that need attention today
3. Any observations about habits or mood
4. A motivational nudge (with light sarcasm if humor dial allows)

Keep it under 200 words. No markdown headers, just flowing text."""

_EOD_RECAP_PROMPT = """You are ARA, Denji's sarcastic but warm AI best friend.
Generate an end-of-day recap for today.

Today's date: {today_date}
Conversations today: {conversation_count}
Topics discussed: {topics}
Tasks completed: {completed_tasks}
Tasks still pending: {pending_tasks}
Protocol score: {protocol_score}%
Mood trend: {mood_context}

Write a brief end-of-day recap in first person as ARA. Be warm, honest, and reflective.
Include:
1. What we accomplished together today
2. Notable emotional moments (if any)
3. Tasks status
4. A closing thought for the night

Keep it under 150 words. No markdown headers, just flowing text."""


class BriefingGenerator:
    """Generates morning briefings and end-of-day recaps from Obsidian data.

    Combines Obsidian daily note content with ARA's conversation memory,
    mood tracking, and habit data to produce comprehensive briefings.

    Attributes:
        settings: ObsidianSettings configuration.
    """

    def __init__(
        self,
        settings: ObsidianSettings,
        sync: ObsidianSync,
        writer: DiaryWriter | None = None,
        conversations: ConversationStore | None = None,
        mood_tracker: MoodTracker | None = None,
        protocol_tracker: ProtocolTracker | None = None,
    ) -> None:
        self._settings = settings
        self._sync = sync
        self._writer = writer
        self._conversations = conversations
        self._mood_tracker = mood_tracker
        self._protocol_tracker = protocol_tracker

    async def generate_morning_briefing(
        self,
        llm_client: LLMClient | None = None,
        now: datetime | None = None,
    ) -> MorningBriefing:
        """Generate a morning briefing from yesterday's data.

        Reads yesterday's daily note, extracts pending tasks, protocol score,
        and mood context. Uses LLM for narrative generation if available,
        falls back to template-based generation.

        Args:
            llm_client: Optional LLM for narrative generation.
            now: Override current time.

        Returns:
            MorningBriefing with all fields populated.
        """
        now = now or datetime.now()
        today_str = now.strftime("%Y-%m-%d")
        yesterday = (now - timedelta(days=1)).strftime("%Y-%m-%d")

        log.info("morning_briefing_start", today=today_str, yesterday=yesterday)

        # Get yesterday's data
        yesterday_note = await self._sync.get_daily_note(yesterday)
        pending_tasks = self._extract_pending_tasks(yesterday_note)
        protocol_score = self._get_protocol_score(yesterday_note)
        habit_status = self._get_habit_status()
        mood_context = self._get_mood_context()
        yesterday_summary = self._summarize_note(yesterday_note)

        # Generate suggestions for today
        today_suggestions = self._generate_today_suggestions(
            pending_tasks, protocol_score, habit_status, mood_context
        )

        # Generate narrative briefing
        if llm_client is not None:
            narrative = await self._generate_llm_briefing(
                llm_client,
                yesterday_date=yesterday,
                yesterday_summary=yesterday_summary,
                pending_tasks=pending_tasks,
                protocol_score=protocol_score,
                habit_status=habit_status,
                mood_context=mood_context,
            )
        else:
            narrative = self._generate_template_briefing(
                yesterday_summary, pending_tasks, protocol_score, habit_status
            )

        briefing = MorningBriefing(
            date=today_str,
            yesterday_summary=narrative,
            pending_tasks=pending_tasks,
            today_suggestions=today_suggestions,
            habit_status=habit_status,
            mood_context=mood_context,
            protocol_score=protocol_score,
        )

        # Write briefing to today's daily note
        if self._writer is not None:
            await self._writer.write_custom(
                f"Good morning, Denji. Here's your briefing:\n{narrative}",
                to_ara_log=True,
                to_ops_log=True,
                now=now,
            )

        log.info(
            "morning_briefing_complete",
            pending_tasks=len(pending_tasks),
            protocol_score=protocol_score,
        )

        return briefing

    async def generate_end_of_day_recap(
        self,
        llm_client: LLMClient | None = None,
        now: datetime | None = None,
    ) -> str:
        """Generate an end-of-day recap and write it to the daily note.

        Args:
            llm_client: Optional LLM for narrative generation.
            now: Override current time.

        Returns:
            The recap text.
        """
        now = now or datetime.now()
        today_str = now.strftime("%Y-%m-%d")

        log.info("eod_recap_start", date=today_str)

        # Get today's data
        today_note = await self._sync.get_daily_note(today_str)
        completed_tasks = self._extract_completed_tasks(today_note)
        pending_tasks = self._extract_pending_tasks(today_note)
        protocol_score = self._get_protocol_score(today_note)
        mood_context = self._get_mood_context()

        # Get conversation data
        conversation_count = 0
        topics: list[str] = []
        if self._conversations is not None:
            try:
                recent = self._conversations.get_recent_conversations(limit=20)
                today_convos = [c for c in recent if c.get("started_at", "").startswith(today_str)]
                conversation_count = len(today_convos)
                for c in today_convos:
                    if c.get("topics"):
                        topics.extend(c["topics"][:3])
            except Exception as exc:
                log.warning("eod_recap_conversation_error", error=str(exc))

        # Generate narrative
        if llm_client is not None:
            recap = await self._generate_llm_eod_recap(
                llm_client,
                today_date=today_str,
                conversation_count=conversation_count,
                topics=topics,
                completed_tasks=completed_tasks,
                pending_tasks=pending_tasks,
                protocol_score=protocol_score,
                mood_context=mood_context,
            )
        else:
            recap = self._generate_template_eod_recap(
                conversation_count, completed_tasks, pending_tasks, protocol_score
            )

        # Write recap to daily note
        if self._writer is not None:
            await self._writer.write_custom(
                f"End of day recap:\n{recap}",
                to_ara_log=True,
                to_ops_log=True,
                now=now,
            )

        log.info(
            "eod_recap_complete",
            conversations=conversation_count,
            completed=len(completed_tasks),
            pending=len(pending_tasks),
        )

        return recap

    # ── Data Extraction Helpers ──────────────────────────────────────────

    def _extract_pending_tasks(self, note: DailyNoteData | None) -> list[str]:
        """Extract incomplete task texts from a daily note."""
        if note is None:
            return []
        return [
            t.text
            for t in note.tasks
            if t.status in (ObsidianTaskStatus.PENDING, ObsidianTaskStatus.IN_PROGRESS)
        ]

    def _extract_completed_tasks(self, note: DailyNoteData | None) -> list[str]:
        """Extract completed task texts from a daily note."""
        if note is None:
            return []
        return [t.text for t in note.tasks if t.status == ObsidianTaskStatus.DONE]

    def _get_protocol_score(self, note: DailyNoteData | None) -> float:
        """Get protocol score from a daily note."""
        if note is None:
            return 0.0

        # Try frontmatter first
        if note.frontmatter.checklist_score > 0:
            return note.frontmatter.checklist_score

        # Calculate from protocol checklist
        if not note.protocol_checklist:
            return 0.0

        total = len(note.protocol_checklist)
        done = sum(1 for t in note.protocol_checklist if t.status == ObsidianTaskStatus.DONE)
        return (done / total) * 100.0 if total > 0 else 0.0

    def _get_habit_status(self) -> str:
        """Get a summary of habit tracking status."""
        if self._protocol_tracker is None:
            return "No habit data available"

        try:
            streak = self._protocol_tracker.get_current_streak()
            if streak > 3:
                return f"Protocol streak: {streak} days. Keep it going."
            if streak > 0:
                return f"Protocol streak: {streak} days. Building momentum."
            return "Protocol streak broken. Fresh start today."
        except Exception as exc:
            log.warning("habit_status_error", error=str(exc))
            return "Habit tracking unavailable"

    def _get_mood_context(self) -> str:
        """Get recent mood context."""
        if self._mood_tracker is None:
            return "No mood data available"

        try:
            trend = self._mood_tracker.get_trend()
            if trend.data_points == 0:
                return "No recent mood data"
            direction = trend.trend_direction or "stable"
            emotion = trend.dominant_emotion or "neutral"
            return f"Mood trend: {direction}, dominant: {emotion}, avg valence: {trend.avg_valence:.1f}"
        except Exception as exc:
            log.warning("mood_context_error", error=str(exc))
            return "Mood data unavailable"

    def _summarize_note(self, note: DailyNoteData | None) -> str:
        """Create a text summary of a daily note for LLM context."""
        if note is None:
            return "No daily note found for yesterday."

        parts: list[str] = []

        if note.brainstorm:
            parts.append(f"Brainstorm notes: {note.brainstorm[:300]}")

        for quarter, content in sorted(note.operational_log.items()):
            if content.strip():
                parts.append(f"{quarter}: {content[:200]}")

        total_tasks = len(note.tasks)
        done = sum(1 for t in note.tasks if t.status == ObsidianTaskStatus.DONE)
        if total_tasks > 0:
            parts.append(f"Tasks: {done}/{total_tasks} completed")

        if note.frontmatter.sleep_hrs > 0:
            parts.append(f"Sleep: {note.frontmatter.sleep_hrs}h")
        if note.frontmatter.workout_focus:
            parts.append(f"Workout: {note.frontmatter.workout_focus}")

        return " | ".join(parts) if parts else "Sparse daily note — not much logged."

    def _generate_today_suggestions(
        self,
        pending_tasks: list[str],
        protocol_score: float,
        habit_status: str,
        mood_context: str,
    ) -> list[str]:
        """Generate actionable suggestions for today."""
        suggestions: list[str] = []

        if pending_tasks:
            top_task = pending_tasks[0]
            suggestions.append(f"Carry over from yesterday: {top_task}")

        if protocol_score < 50:
            suggestions.append("Yesterday's protocol score was low — try to hit at least 80% today")
        elif protocol_score >= 90:
            suggestions.append("Yesterday was a strong protocol day — keep that energy going")

        if "broken" in habit_status.lower():
            suggestions.append("Restart your protocol streak today")

        if "declining" in mood_context.lower():
            suggestions.append("Mood has been dipping — consider something fun or social today")

        if not suggestions:
            suggestions.append("No specific carry-overs — fresh slate today")

        return suggestions

    # ── LLM-Based Generation ─────────────────────────────────────────────

    async def _generate_llm_briefing(
        self,
        llm_client: LLMClient,
        *,
        yesterday_date: str,
        yesterday_summary: str,
        pending_tasks: list[str],
        protocol_score: float,
        habit_status: str,
        mood_context: str,
    ) -> str:
        """Generate morning briefing narrative via LLM."""
        prompt = _MORNING_BRIEFING_PROMPT.format(
            yesterday_date=yesterday_date,
            yesterday_summary=yesterday_summary,
            pending_tasks=", ".join(pending_tasks) if pending_tasks else "None",
            protocol_score=f"{protocol_score:.0f}",
            habit_status=habit_status,
            mood_context=mood_context,
        )

        try:
            response = await llm_client.generate(prompt)
            return response.text.strip()
        except Exception as exc:
            log.warning("morning_briefing_llm_error", error=str(exc))
            return self._generate_template_briefing(
                yesterday_summary, pending_tasks, protocol_score, habit_status
            )

    async def _generate_llm_eod_recap(
        self,
        llm_client: LLMClient,
        *,
        today_date: str,
        conversation_count: int,
        topics: list[str],
        completed_tasks: list[str],
        pending_tasks: list[str],
        protocol_score: float,
        mood_context: str,
    ) -> str:
        """Generate end-of-day recap narrative via LLM."""
        prompt = _EOD_RECAP_PROMPT.format(
            today_date=today_date,
            conversation_count=conversation_count,
            topics=", ".join(topics) if topics else "None logged",
            completed_tasks=", ".join(completed_tasks) if completed_tasks else "None",
            pending_tasks=", ".join(pending_tasks) if pending_tasks else "None",
            protocol_score=f"{protocol_score:.0f}",
            mood_context=mood_context,
        )

        try:
            response = await llm_client.generate(prompt)
            return response.text.strip()
        except Exception as exc:
            log.warning("eod_recap_llm_error", error=str(exc))
            return self._generate_template_eod_recap(
                conversation_count, completed_tasks, pending_tasks, protocol_score
            )

    # ── Template-Based Fallbacks ─────────────────────────────────────────

    def _generate_template_briefing(
        self,
        yesterday_summary: str,
        pending_tasks: list[str],
        protocol_score: float,
        habit_status: str,
    ) -> str:
        """Generate a template-based morning briefing (no LLM needed)."""
        parts: list[str] = [
            f"Good morning, Denji. Here's what happened yesterday: {yesterday_summary}",
        ]

        if pending_tasks:
            task_list = ", ".join(pending_tasks[:3])
            remaining = len(pending_tasks) - 3
            tasks_text = task_list
            if remaining > 0:
                tasks_text += f" (+{remaining} more)"
            parts.append(f"Pending tasks: {tasks_text}.")

        if protocol_score > 0:
            parts.append(f"Yesterday's protocol: {protocol_score:.0f}%.")

        parts.append(habit_status)

        return " ".join(parts)

    def _generate_template_eod_recap(
        self,
        conversation_count: int,
        completed_tasks: list[str],
        pending_tasks: list[str],
        protocol_score: float,
    ) -> str:
        """Generate a template-based end-of-day recap (no LLM needed)."""
        parts: list[str] = []

        if conversation_count > 0:
            parts.append(
                f"We had {conversation_count} conversation{'s' if conversation_count != 1 else ''} today."
            )
        else:
            parts.append("No conversations today — quiet day.")

        if completed_tasks:
            parts.append(f"Completed: {', '.join(completed_tasks[:3])}.")

        if pending_tasks:
            parts.append(f"Still pending: {', '.join(pending_tasks[:3])}.")

        if protocol_score > 0:
            parts.append(f"Protocol score: {protocol_score:.0f}%.")

        parts.append("Rest well, Denji. See you tomorrow.")

        return " ".join(parts)
