"""Obsidian vault integration for ARA.

Provides bidirectional sync between ARA and the user's Obsidian vault
via the Local REST API plugin. ARA reads daily notes for context and
writes diary entries, conversation summaries, and emotional logs back.

Modules:
    client: Async HTTP client for Obsidian Local REST API
    parser: Daily note parser (frontmatter, sections, tasks)
    sync: Vault sync engine (boot sync, incremental, change detection)
    writer: Diary writer (ARA's first-person entries, emotional logs)
    daily_briefing: Morning briefing and end-of-day recap generator
    habits: Protocol scoring and habit tracking from checklist data
"""

from ara.obsidian.client import ObsidianClient
from ara.obsidian.daily_briefing import BriefingGenerator
from ara.obsidian.habits import ProtocolTracker
from ara.obsidian.parser import DailyNoteParser
from ara.obsidian.sync import ObsidianSync
from ara.obsidian.writer import DiaryWriter

__all__ = [
    "BriefingGenerator",
    "DailyNoteParser",
    "DiaryWriter",
    "ObsidianClient",
    "ObsidianSync",
    "ProtocolTracker",
]
