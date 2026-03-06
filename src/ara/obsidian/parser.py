"""Parser for Obsidian daily notes.

Extracts structured data from the user's daily note template format:
- YAML frontmatter (date, weight_kg, sleep_hrs, calories, workout_focus, effort_score, checklist_score)
- BRAINSTORM section
- MISSION CONTROL (dataviewjs scoring — skipped for parsing)
- OPERATIONAL LOG (Q1-Q4 time-of-day callouts)
- THE PROTOCOL (checklist items)
- INBOX ZERO (items)
- ARA's LOG (ARA-authored entries)

This parser is tolerant of missing sections — not every daily note has all sections.

Usage:
    from ara.obsidian.parser import DailyNoteParser

    parser = DailyNoteParser()
    note = parser.parse("2026-03-06", content, "10_Journal/Daily/2026/03-March/2026-03-06.md")
"""

from __future__ import annotations

import hashlib
import re

import structlog

from ara.utils.errors import ObsidianParseError
from ara.utils.types import (
    DailyNoteData,
    ObsidianFrontmatter,
    ObsidianTask,
    ObsidianTaskStatus,
)

log = structlog.get_logger(component="obsidian.parser")

# ── Regex patterns ───────────────────────────────────────────────────────────

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_SECTION_RE = re.compile(r"^(#{1,3})\s+(.+)$", re.MULTILINE)
_TASK_RE = re.compile(r"^\s*-\s+\[([xX \-/])\]\s+(.+)$", re.MULTILINE)
_CALLOUT_HEADER_RE = re.compile(r"^>\s*\[![\w-]+\]\s*(.+)?$", re.MULTILINE)
# Match Q1/Q2/Q3/Q4 callout blocks (e.g. "> [!info] Q1 (00:00 - 06:00)")
_QUARTER_RE = re.compile(
    r">\s*\[![\w-]+\]\s*Q([1-4])\s*\(.*?\)",
    re.MULTILINE,
)

# Frontmatter field extractors (line-by-line YAML — no need for full YAML parser)
_FM_FIELD_RE = re.compile(r"^(\w[\w_]*)\s*:\s*(.*)$", re.MULTILINE)


class DailyNoteParser:
    """Parses Obsidian daily notes into structured DailyNoteData.

    Handles the user's specific daily note template with frontmatter,
    brainstorm, operational log, protocol checklist, and inbox sections.
    Tolerant of missing sections and partial content.
    """

    def parse(self, date_str: str, content: str, path: str) -> DailyNoteData:
        """Parse a daily note's markdown content into structured data.

        Args:
            date_str: The date of the note in YYYY-MM-DD format.
            content: Raw markdown content of the note.
            path: Vault-relative path for error context.

        Returns:
            Fully parsed DailyNoteData.

        Raises:
            ObsidianParseError: If content is fundamentally unparseable.
        """
        if not content or not content.strip():
            raise ObsidianParseError(path, "Empty note content")

        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

        try:
            frontmatter = self._parse_frontmatter(content, path)
            brainstorm = self._extract_section(content, "BRAINSTORM")
            operational_log = self._extract_operational_log(content)
            protocol_tasks = self._extract_protocol(content)
            inbox_items = self._extract_inbox(content)
            all_tasks = self._extract_all_tasks(content)

            return DailyNoteData(
                date=date_str,
                path=path,
                frontmatter=frontmatter,
                brainstorm=brainstorm,
                operational_log=operational_log,
                protocol_checklist=protocol_tasks,
                inbox_items=inbox_items,
                tasks=all_tasks,
                raw_content=content,
                content_hash=content_hash,
            )
        except ObsidianParseError:
            raise
        except Exception as exc:
            raise ObsidianParseError(path, f"Unexpected parse error: {exc}") from exc

    def _parse_frontmatter(self, content: str, path: str) -> ObsidianFrontmatter:
        """Extract YAML frontmatter from note content.

        Uses simple line-by-line parsing (not a full YAML parser) since
        the frontmatter schema is known and simple.
        """
        match = _FRONTMATTER_RE.match(content)
        if not match:
            return ObsidianFrontmatter()

        fm_text = match.group(1)
        raw: dict[str, str] = {}

        for field_match in _FM_FIELD_RE.finditer(fm_text):
            key = field_match.group(1).strip()
            value = field_match.group(2).strip()
            # Strip surrounding quotes
            if (value.startswith('"') and value.endswith('"')) or (
                value.startswith("'") and value.endswith("'")
            ):
                value = value[1:-1]
            raw[key] = value

        return ObsidianFrontmatter(
            date=raw.get("date", ""),
            weight_kg=self._safe_float(raw.get("weight_kg", "")),
            sleep_hrs=self._safe_float(raw.get("sleep_hrs", "")),
            calories=self._safe_int(raw.get("calories", "")),
            workout_focus=raw.get("workout_focus", ""),
            effort_score=self._safe_float(raw.get("effort_score", "")),
            checklist_score=self._safe_float(raw.get("checklist_score", "")),
            raw=raw,
        )

    def _extract_section(self, content: str, heading: str) -> str:
        """Extract content under a specific heading (case-insensitive).

        Returns content between the heading and the next heading of same or higher level.
        """
        lines = content.split("\n")
        capturing = False
        captured: list[str] = []
        heading_level = 0

        for line in lines:
            section_match = _SECTION_RE.match(line)
            if section_match:
                level = len(section_match.group(1))
                title = section_match.group(2).strip()

                if title.upper() == heading.upper():
                    capturing = True
                    heading_level = level
                    continue

                if capturing and level <= heading_level:
                    break

            if capturing:
                captured.append(line)

        return "\n".join(captured).strip()

    def _extract_operational_log(self, content: str) -> dict[str, str]:
        """Extract OPERATIONAL LOG quarter callout blocks (Q1-Q4).

        Each quarter is a callout block like:
            > [!info] Q1 (00:00 - 06:00)
            > Content here...
        """
        result: dict[str, str] = {}

        # First find the OPERATIONAL LOG section
        log_section = self._extract_section(content, "OPERATIONAL LOG")
        if not log_section:
            return result

        # Split by quarter markers
        lines = log_section.split("\n")
        current_quarter: str = ""
        current_lines: list[str] = []

        for line in lines:
            quarter_match = _QUARTER_RE.match(line)
            if quarter_match:
                # Save previous quarter
                if current_quarter:
                    result[current_quarter] = "\n".join(current_lines).strip()
                current_quarter = f"Q{quarter_match.group(1)}"
                current_lines = []
                continue

            if current_quarter:
                # Strip callout prefix "> " for cleaner content
                stripped = line
                if stripped.startswith("> "):
                    stripped = stripped[2:]
                elif stripped == ">":
                    stripped = ""
                current_lines.append(stripped)

        # Save last quarter
        if current_quarter:
            result[current_quarter] = "\n".join(current_lines).strip()

        return result

    def _extract_protocol(self, content: str) -> list[ObsidianTask]:
        """Extract THE PROTOCOL checklist items."""
        protocol_section = self._extract_section(content, "THE PROTOCOL")
        if not protocol_section:
            return []

        return self._parse_tasks(protocol_section)

    def _extract_inbox(self, content: str) -> list[str]:
        """Extract INBOX ZERO items (may be tasks or plain text)."""
        inbox_section = self._extract_section(content, "INBOX ZERO")
        if not inbox_section:
            return []

        items: list[str] = []
        for line in inbox_section.split("\n"):
            line = line.strip()
            if line.startswith("- "):
                # Remove task checkbox if present
                task_match = _TASK_RE.match(f"  {line}")
                if task_match:
                    items.append(task_match.group(2).strip())
                else:
                    items.append(line[2:].strip())
            elif line:
                items.append(line)

        return items

    def _extract_all_tasks(self, content: str) -> list[ObsidianTask]:
        """Extract all task checkboxes from the entire note."""
        return self._parse_tasks(content)

    def _parse_tasks(self, text: str) -> list[ObsidianTask]:
        """Parse markdown task checkboxes into ObsidianTask objects.

        Supports:
            - [ ] pending
            - [x] or [X] done
            - [-] cancelled
            - [/] in progress
        """
        tasks: list[ObsidianTask] = []
        for match in _TASK_RE.finditer(text):
            marker = match.group(1)
            task_text = match.group(2).strip()

            if marker in ("x", "X"):
                status = ObsidianTaskStatus.DONE
            elif marker == "-":
                status = ObsidianTaskStatus.CANCELLED
            elif marker == "/":
                status = ObsidianTaskStatus.IN_PROGRESS
            else:
                status = ObsidianTaskStatus.PENDING

            tasks.append(ObsidianTask(text=task_text, status=status))

        return tasks

    @staticmethod
    def _safe_float(value: str) -> float:
        """Parse a float from a string, returning 0.0 on failure."""
        if not value:
            return 0.0
        try:
            return float(value)
        except (ValueError, TypeError):
            return 0.0

    @staticmethod
    def _safe_int(value: str) -> int:
        """Parse an int from a string, returning 0 on failure."""
        if not value:
            return 0
        try:
            return int(float(value))
        except (ValueError, TypeError):
            return 0

    def extract_ara_log(self, content: str, heading: str = "## ARA's LOG") -> str:
        """Extract the ARA's LOG section content from a daily note.

        Args:
            content: Raw markdown content.
            heading: The heading to look for (default: "## ARA's LOG").

        Returns:
            Content under ARA's LOG heading, or empty string if not found.
        """
        # Strip the leading ## for section extraction
        clean_heading = heading.lstrip("#").strip()
        return self._extract_section(content, clean_heading)
