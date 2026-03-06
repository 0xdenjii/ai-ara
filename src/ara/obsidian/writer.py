"""Obsidian diary writer — ARA's first-person voice in the user's vault.

Handles writing ARA's observations, emotional logs, conversation summaries,
and daily entries to the user's Obsidian daily notes. Writes to both a
dedicated "ARA's LOG" section and key highlights into the OPERATIONAL LOG
Q1-Q4 callouts based on time of day.

Voice: First-person ARA perspective (configurable).
    "Denji told me he's stressed about the deployment. I reminded him he's handled worse."

Architecture:
    DiaryWriter depends on ObsidianClient for I/O and reads voice/location
    settings from ObsidianSettings. Entries are composed with timestamps
    and formatted for Obsidian markdown.

Usage:
    from ara.obsidian.writer import DiaryWriter

    writer = DiaryWriter(settings.obsidian, client)
    await writer.write_entry("Denji shared that he's feeling overwhelmed with work.")
    await writer.write_emotional_log("stressed", 0.7, "deployment pressure")
    await writer.write_conversation_summary("We discussed his workout routine and sleep schedule.")
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

import structlog

from ara.utils.errors import ObsidianWriteError
from ara.utils.types import ObsidianQuarter

if TYPE_CHECKING:
    from ara.config.settings import ObsidianSettings
    from ara.obsidian.client import ObsidianClient

log = structlog.get_logger(component="obsidian.writer")

# Quarter boundaries (hour ranges)
_QUARTER_RANGES: dict[ObsidianQuarter, tuple[int, int]] = {
    ObsidianQuarter.Q1: (0, 6),
    ObsidianQuarter.Q2: (6, 12),
    ObsidianQuarter.Q3: (12, 18),
    ObsidianQuarter.Q4: (18, 24),
}


def _get_current_quarter(now: datetime | None = None) -> ObsidianQuarter:
    """Determine which quarter of the day we're in."""
    hour = (now or datetime.now()).hour
    for quarter, (start, end) in _QUARTER_RANGES.items():
        if start <= hour < end:
            return quarter
    return ObsidianQuarter.Q4  # Fallback


def _format_time(now: datetime | None = None) -> str:
    """Format current time as HH:MM for log entries."""
    return (now or datetime.now()).strftime("%H:%M")


class DiaryWriter:
    """Writes ARA's diary entries to Obsidian daily notes.

    Supports writing to:
    - Dedicated ARA's LOG section (detailed entries)
    - OPERATIONAL LOG Q callouts (key highlights by time of day)
    - Both simultaneously (default configuration)

    The voice is configurable: first_person (default), third_person, or factual.

    Attributes:
        settings: ObsidianSettings configuration.
    """

    def __init__(self, settings: ObsidianSettings, client: ObsidianClient) -> None:
        self._settings = settings
        self._client = client

    async def write_entry(
        self,
        content: str,
        *,
        is_key_highlight: bool = False,
        now: datetime | None = None,
    ) -> None:
        """Write a general entry to today's daily note.

        Args:
            content: The entry text (will be formatted with ARA voice).
            is_key_highlight: If True, also write to OPERATIONAL LOG Q callout.
            now: Override current time (for testing).

        Raises:
            ObsidianWriteError: If writing fails.
        """
        now = now or datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        path = await self._client.ensure_daily_note(date_str)

        time_str = _format_time(now)
        formatted = self._format_entry(content, time_str)

        # Write to ARA's LOG section
        if self._settings.write_to_ara_log:
            await self._append_to_ara_log(path, formatted)

        # Write key highlights to OPERATIONAL LOG Q callout
        if is_key_highlight and self._settings.write_to_ops_log:
            quarter = _get_current_quarter(now)
            ops_entry = f"> **[ARA {time_str}]** {self._summarize(content)}"
            await self._append_to_ops_log(path, quarter, ops_entry)

        log.debug("obsidian_entry_written", date=date_str, highlight=is_key_highlight)

    async def write_emotional_log(
        self,
        emotion: str,
        intensity: float,
        context: str,
        *,
        now: datetime | None = None,
    ) -> None:
        """Write an emotional observation to the daily note.

        This is triggered when ARA detects significant emotional content
        in a conversation. Always written as a key highlight.

        Args:
            emotion: Dominant emotion (e.g. "stressed", "excited", "grateful").
            intensity: Emotion intensity 0.0-1.0.
            context: What triggered the emotion.
            now: Override current time.
        """
        now = now or datetime.now()
        intensity_word = self._intensity_word(intensity)

        if self._settings.voice == "first_person":
            content = (
                f"Denji is feeling {intensity_word} {emotion} right now. "
                f"We were talking about {context}. "
                f"I{'ll keep an eye on this' if intensity > 0.6 else ' noted it'}."
            )
        elif self._settings.voice == "third_person":
            content = (
                f"Denji expressed {intensity_word} {emotion} during a conversation about {context}."
            )
        else:  # factual
            content = (
                f"Emotional state: {emotion} (intensity: {intensity:.1f}). Context: {context}."
            )

        await self.write_entry(content, is_key_highlight=True, now=now)

    async def write_conversation_summary(
        self,
        summary: str,
        topics: list[str] | None = None,
        *,
        now: datetime | None = None,
    ) -> None:
        """Write a conversation summary to the daily note.

        Called when a conversation ends, capturing key topics and takeaways.

        Args:
            summary: LLM-generated or extractive summary.
            topics: Key topics discussed.
            now: Override current time.
        """
        now = now or datetime.now()
        topics_str = ", ".join(topics) if topics else "various things"

        if self._settings.voice == "first_person":
            content = f"Denji and I just had a conversation about {topics_str}. {summary}"
        elif self._settings.voice == "third_person":
            content = f"Conversation completed. Topics: {topics_str}. {summary}"
        else:  # factual
            content = f"Conversation logged. Topics: [{topics_str}]. Summary: {summary}"

        await self.write_entry(content, is_key_highlight=True, now=now)

    async def write_task_update(
        self,
        task_text: str,
        status: str,
        *,
        now: datetime | None = None,
    ) -> None:
        """Write a task status update to the daily note.

        Args:
            task_text: Description of the task.
            status: New status (completed, started, etc.).
            now: Override current time.
        """
        now = now or datetime.now()

        if self._settings.voice == "first_person":
            if status == "completed":
                content = f"Denji just finished: {task_text}. Nice work."
            elif status == "started":
                content = f"Denji started working on: {task_text}."
            else:
                content = f"Task update — {task_text}: {status}."
        elif self._settings.voice == "third_person":
            content = f"Task '{task_text}' marked as {status}."
        else:  # factual
            content = f"Task update: [{task_text}] -> {status}."

        await self.write_entry(content, is_key_highlight=False, now=now)

    async def write_custom(
        self,
        text: str,
        *,
        to_ara_log: bool = True,
        to_ops_log: bool = False,
        now: datetime | None = None,
    ) -> None:
        """Write raw text without voice formatting.

        Used for internal system notes, sync status, etc.

        Args:
            text: Raw text to write.
            to_ara_log: Write to ARA's LOG section.
            to_ops_log: Write to OPERATIONAL LOG.
            now: Override current time.
        """
        now = now or datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        path = await self._client.ensure_daily_note(date_str)
        time_str = _format_time(now)
        stamped = f"- **{time_str}** — {text}"

        if to_ara_log:
            await self._append_to_ara_log(path, stamped)

        if to_ops_log:
            quarter = _get_current_quarter(now)
            await self._append_to_ops_log(path, quarter, f"> **[ARA {time_str}]** {text}")

    # ── Internal methods ─────────────────────────────────────────────────

    def _format_entry(self, content: str, time_str: str) -> str:
        """Format an entry with timestamp as a bullet point."""
        return f"- **{time_str}** — {content}"

    def _summarize(self, content: str, max_len: int = 120) -> str:
        """Create a short summary for OPERATIONAL LOG highlights.

        Truncates content to fit in a callout line.
        """
        if len(content) <= max_len:
            return content
        return content[: max_len - 3] + "..."

    @staticmethod
    def _intensity_word(intensity: float) -> str:
        """Convert intensity float to a human-readable word."""
        if intensity >= 0.8:
            return "very"
        if intensity >= 0.5:
            return "quite"
        if intensity >= 0.3:
            return "somewhat"
        return "mildly"

    async def _append_to_ara_log(self, path: str, entry: str) -> None:
        """Append an entry to the ARA's LOG section of a daily note.

        If the section doesn't exist, creates it at the end of the note.
        """
        try:
            content = await self._client.read_note(path)
        except Exception as exc:
            raise ObsidianWriteError(path, f"Cannot read note for ARA LOG append: {exc}") from exc

        heading = self._settings.ara_log_heading

        if heading in content:
            # Section exists — append after it
            parts = content.split(heading, 1)
            before = parts[0]
            after = parts[1]

            # Find where the section ends (next heading or end of file)
            lines = after.split("\n")
            insert_idx = len(lines)
            for i, line in enumerate(lines):
                if i == 0:
                    continue  # Skip the first line (rest of heading line)
                if line.startswith(("## ", "# ")):
                    insert_idx = i
                    break

            lines.insert(insert_idx, entry)
            new_content = before + heading + "\n".join(lines)
        else:
            # Section doesn't exist — create it at the end
            new_content = content.rstrip() + f"\n\n{heading}\n\n{entry}\n"

        await self._client.write_note(path, new_content)

    async def _append_to_ops_log(self, path: str, quarter: ObsidianQuarter, entry: str) -> None:
        """Append an entry to the appropriate OPERATIONAL LOG Q callout.

        Finds the Q callout block and appends within it.
        """
        try:
            content = await self._client.read_note(path)
        except Exception as exc:
            raise ObsidianWriteError(path, f"Cannot read note for OPS LOG append: {exc}") from exc

        quarter_label = quarter.value.upper()  # "Q1", "Q2", etc.

        # Find the quarter callout
        lines = content.split("\n")
        insert_idx: int | None = None

        for i, line in enumerate(lines):
            if f"] {quarter_label}" in line and line.strip().startswith(">"):
                # Found the quarter header — find end of callout block
                for j in range(i + 1, len(lines)):
                    # Callout lines start with ">"
                    if not lines[j].startswith(">") and lines[j].strip():
                        insert_idx = j
                        break
                else:
                    # Callout extends to end of file or empty lines
                    insert_idx = len(lines)
                break

        if insert_idx is not None:
            lines.insert(insert_idx, entry)
            new_content = "\n".join(lines)
            await self._client.write_note(path, new_content)
        else:
            log.debug(
                "obsidian_ops_quarter_not_found",
                path=path,
                quarter=quarter_label,
            )
