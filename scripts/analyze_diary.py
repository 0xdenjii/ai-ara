"""Diary data analysis tool for pre-ingestion exploration.

Scans diary files and produces statistics to help decide cleaning
strategy before ingesting into ARA's memory. Classifies entries as
todo-only, narrative, or mixed.

Usage:
    uv run python scripts/analyze_diary.py data/diaries/

    # With detailed per-file output
    uv run python scripts/analyze_diary.py data/diaries/ --verbose

    # Show sample entries from each category
    uv run python scripts/analyze_diary.py data/diaries/ --samples 3

    # Export analysis report to JSON
    uv run python scripts/analyze_diary.py data/diaries/ --export report.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ara.memory.chunking import ChunkerConfig, TextChunker
from ara.memory.emotion import EmotionTagger


# ── Entry Classification ───────────────────────────────────────────────────

# Patterns that indicate todo/task content
_TODO_PATTERNS = [
    re.compile(r"^[-*]\s*\[[ x]\]", re.MULTILINE),  # - [ ] or - [x]
    re.compile(r"^[-*]\s+\w", re.MULTILINE),  # - bullet items
    re.compile(r"^\d+\.\s+\w", re.MULTILINE),  # 1. numbered items
    re.compile(r"\b(?:TODO|FIXME|DONE|PENDING)\b", re.IGNORECASE),
]

# Patterns that indicate narrative/personal content
_NARRATIVE_PATTERNS = [
    re.compile(
        r"\bI\s+(?:felt|feel|realized|decided|learned|went|met|talked|was)\b", re.IGNORECASE
    ),
    re.compile(r"\btoday\s+(?:was|I)\b", re.IGNORECASE),
    re.compile(r"\b(?:happy|sad|angry|anxious|excited|worried|grateful)\b", re.IGNORECASE),
    re.compile(r"\b(?:talked|called|met|visited|hung out)\s+(?:with|to)\b", re.IGNORECASE),
    re.compile(r"\b(?:morning|afternoon|evening|night)\b", re.IGNORECASE),
]

# URL pattern
_URL_PATTERN = re.compile(
    r"https?://[^\s<>\"')\]]+",
    re.IGNORECASE,
)


@dataclass
class EntryAnalysis:
    """Analysis of a single diary entry/section."""

    text: str
    date: str
    word_count: int
    line_count: int
    classification: str  # "todo_only", "narrative", "mixed", "short", "link_dump"
    todo_line_count: int = 0
    narrative_line_count: int = 0
    url_count: int = 0
    urls: list[str] = field(default_factory=list)
    emotions: list[str] = field(default_factory=list)


@dataclass
class FileAnalysis:
    """Analysis of a single diary file."""

    file_name: str
    file_size: int
    total_entries: int
    total_words: int
    entries: list[EntryAnalysis] = field(default_factory=list)
    classification_counts: dict[str, int] = field(default_factory=dict)
    date_range: tuple[str, str] = ("", "")
    url_count: int = 0


@dataclass
class DiaryAnalysis:
    """Full analysis of all diary files."""

    total_files: int = 0
    total_entries: int = 0
    total_words: int = 0
    total_chars: int = 0
    total_urls: int = 0
    files: list[FileAnalysis] = field(default_factory=list)
    classification_summary: dict[str, int] = field(default_factory=dict)
    all_urls: list[str] = field(default_factory=list)
    date_coverage: list[str] = field(default_factory=list)
    emotion_summary: dict[str, int] = field(default_factory=dict)


def classify_entry(text: str) -> tuple[str, int, int]:
    """Classify a text entry as todo_only, narrative, mixed, short, or link_dump.

    Returns (classification, todo_lines, narrative_lines).
    """
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    if not lines:
        return "short", 0, 0

    word_count = len(text.split())
    if word_count < 10:
        return "short", 0, 0

    # Count todo-like lines
    todo_lines = 0
    for line in lines:
        for pattern in _TODO_PATTERNS:
            if pattern.search(line):
                todo_lines += 1
                break

    # Count narrative lines
    narrative_lines = 0
    for line in lines:
        for pattern in _NARRATIVE_PATTERNS:
            if pattern.search(line):
                narrative_lines += 1
                break

    # Count URLs
    url_count = len(_URL_PATTERN.findall(text))

    # Classification logic
    total_lines = len(lines)
    todo_ratio = todo_lines / total_lines if total_lines > 0 else 0
    narrative_ratio = narrative_lines / total_lines if total_lines > 0 else 0
    url_ratio = url_count / total_lines if total_lines > 0 else 0

    if url_ratio > 0.5 and narrative_ratio < 0.2:
        return "link_dump", todo_lines, narrative_lines

    if todo_ratio > 0.6 and narrative_ratio < 0.15:
        return "todo_only", todo_lines, narrative_lines

    if narrative_ratio > 0.3 or (word_count > 50 and todo_ratio < 0.3):
        return "narrative", todo_lines, narrative_lines

    return "mixed", todo_lines, narrative_lines


def analyze_file(file_path: Path, *, emotion_tagger: EmotionTagger | None = None) -> FileAnalysis:
    """Analyze a single diary file."""
    text = file_path.read_text(encoding="utf-8")

    chunker = TextChunker(ChunkerConfig())
    chunks = chunker.chunk_document(text, source_file=file_path.name)

    entries: list[EntryAnalysis] = []
    all_urls: list[str] = []
    classification_counts: Counter[str] = Counter()
    dates: list[str] = []

    # Use chunks as entries (each chunk roughly corresponds to a section)
    for chunk in chunks:
        classification, todo_lines, narrative_lines = classify_entry(chunk.text)
        urls = _URL_PATTERN.findall(chunk.text)
        all_urls.extend(urls)

        emotions: list[str] = []
        if emotion_tagger:
            emotion_meta = emotion_tagger.tag(chunk.text)
            emotions = [t.emotion for t in emotion_meta.tags]

        date_str = chunk.date.strftime("%Y-%m-%d") if chunk.date else ""
        if date_str:
            dates.append(date_str)

        entry = EntryAnalysis(
            text=chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text,
            date=date_str,
            word_count=len(chunk.text.split()),
            line_count=len(chunk.text.splitlines()),
            classification=classification,
            todo_line_count=todo_lines,
            narrative_line_count=narrative_lines,
            url_count=len(urls),
            urls=urls,
            emotions=emotions,
        )
        entries.append(entry)
        classification_counts[classification] += 1

    date_range = (min(dates), max(dates)) if dates else ("", "")

    return FileAnalysis(
        file_name=file_path.name,
        file_size=len(text),
        total_entries=len(entries),
        total_words=len(text.split()),
        entries=entries,
        classification_counts=dict(classification_counts),
        date_range=date_range,
        url_count=len(all_urls),
    )


def analyze_directory(
    dir_path: Path,
    *,
    pattern: str = "*.md",
    recursive: bool = True,
) -> DiaryAnalysis:
    """Analyze all diary files in a directory."""
    glob_method = dir_path.rglob if recursive else dir_path.glob
    files = sorted(f for f in glob_method(pattern) if f.is_file())

    emotion_tagger = EmotionTagger()
    analysis = DiaryAnalysis()
    all_dates: list[str] = []
    all_urls: list[str] = []
    emotion_counter: Counter[str] = Counter()

    for file_path in files:
        try:
            file_analysis = analyze_file(file_path, emotion_tagger=emotion_tagger)
            analysis.files.append(file_analysis)
            analysis.total_files += 1
            analysis.total_entries += file_analysis.total_entries
            analysis.total_words += file_analysis.total_words
            analysis.total_chars += file_analysis.file_size

            for cls, count in file_analysis.classification_counts.items():
                analysis.classification_summary[cls] = (
                    analysis.classification_summary.get(cls, 0) + count
                )

            if file_analysis.date_range[0]:
                all_dates.extend(file_analysis.date_range)

            # Collect URLs
            for entry in file_analysis.entries:
                all_urls.extend(entry.urls)
                for emo in entry.emotions:
                    emotion_counter[emo] += 1

        except Exception as exc:
            print(f"  [ERROR] {file_path.name}: {exc}")

    analysis.total_urls = len(all_urls)
    analysis.all_urls = list(set(all_urls))  # Deduplicate
    analysis.date_coverage = sorted(set(all_dates))
    analysis.emotion_summary = dict(emotion_counter.most_common(15))

    return analysis


def print_report(analysis: DiaryAnalysis, *, verbose: bool = False, samples: int = 0) -> None:
    """Print the analysis report."""
    print(f"\n{'=' * 60}")
    print("  ARA Diary Analysis Report")
    print(f"{'=' * 60}\n")

    print(f"  Total files:     {analysis.total_files}")
    print(f"  Total entries:   {analysis.total_entries}")
    print(f"  Total words:     {analysis.total_words:,}")
    print(f"  Total chars:     {analysis.total_chars:,}")
    print(f"  Total URLs:      {analysis.total_urls}")
    print(f"  Unique URLs:     {len(analysis.all_urls)}")

    if analysis.date_coverage:
        print(f"  Date range:      {analysis.date_coverage[0]} to {analysis.date_coverage[-1]}")
        print(f"  Unique dates:    {len(analysis.date_coverage)}")

    # Classification breakdown
    print(f"\n{'─' * 40}")
    print("  Entry Classification:")
    total = sum(analysis.classification_summary.values())
    for cls in ["narrative", "mixed", "todo_only", "short", "link_dump"]:
        count = analysis.classification_summary.get(cls, 0)
        pct = (count / total * 100) if total > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"    {cls:12s}: {count:4d} ({pct:5.1f}%) {bar}")

    # Emotion summary
    if analysis.emotion_summary:
        print(f"\n{'─' * 40}")
        print("  Top Emotions Detected:")
        for emo, count in list(analysis.emotion_summary.items())[:10]:
            print(f"    {emo:15s}: {count}")

    # Per-file breakdown
    if verbose:
        print(f"\n{'─' * 40}")
        print("  Per-File Breakdown:\n")
        for fa in analysis.files:
            print(f"  {fa.file_name}")
            print(
                f"    Size: {fa.file_size:,} chars, {fa.total_words:,} words, {fa.total_entries} entries"
            )
            if fa.date_range[0]:
                print(f"    Dates: {fa.date_range[0]} to {fa.date_range[1]}")
            for cls, count in fa.classification_counts.items():
                print(f"    {cls}: {count}")
            if fa.url_count:
                print(f"    URLs: {fa.url_count}")
            print()

    # Sample entries
    if samples > 0:
        for cls in ["narrative", "todo_only", "mixed", "link_dump"]:
            entries = [e for f in analysis.files for e in f.entries if e.classification == cls]
            if entries:
                print(f"\n{'─' * 40}")
                print(f"  Sample '{cls}' entries (showing {min(samples, len(entries))}):\n")
                for entry in entries[:samples]:
                    date_prefix = f"[{entry.date}] " if entry.date else ""
                    preview = entry.text.replace("\n", " ")[:150]
                    print(f"    {date_prefix}{preview}")
                    if entry.emotions:
                        print(f"    → emotions: {', '.join(entry.emotions)}")
                    print()

    # Recommendations
    print(f"\n{'=' * 60}")
    print("  Recommendations:")
    print(f"{'=' * 60}\n")

    todo_count = analysis.classification_summary.get("todo_only", 0)
    short_count = analysis.classification_summary.get("short", 0)
    narrative_count = analysis.classification_summary.get("narrative", 0)
    link_count = analysis.classification_summary.get("link_dump", 0)

    garbage_count = todo_count + short_count + link_count
    if total > 0:
        garbage_pct = garbage_count / total * 100
        signal_pct = (
            (narrative_count + analysis.classification_summary.get("mixed", 0)) / total * 100
        )

        print(f"  Signal entries:  {signal_pct:.0f}% (narrative + mixed)")
        print(f"  Noise entries:   {garbage_pct:.0f}% (todo_only + short + link_dump)")
        print()

        if garbage_pct > 40:
            print("  ⚠ High noise ratio. Recommend LLM-assisted cleaning before ingestion.")
            print("    Rule-based filter can auto-remove todo_only and short entries.")
            print("    Mixed entries should be reviewed by LLM for hidden signal.")
        else:
            print("  ✓ Acceptable signal-to-noise ratio. Rule-based filter should suffice.")

    if analysis.total_urls > 50:
        print(f"\n  ⚠ {analysis.total_urls} URLs found. Extract as metadata, don't embed URL text.")

    print()


def export_report(analysis: DiaryAnalysis, output_path: Path) -> None:
    """Export analysis as JSON."""
    data = {
        "total_files": analysis.total_files,
        "total_entries": analysis.total_entries,
        "total_words": analysis.total_words,
        "total_chars": analysis.total_chars,
        "total_urls": analysis.total_urls,
        "unique_urls": len(analysis.all_urls),
        "date_range": {
            "start": analysis.date_coverage[0] if analysis.date_coverage else "",
            "end": analysis.date_coverage[-1] if analysis.date_coverage else "",
            "unique_dates": len(analysis.date_coverage),
        },
        "classification_summary": analysis.classification_summary,
        "emotion_summary": analysis.emotion_summary,
        "files": [
            {
                "name": f.file_name,
                "size": f.file_size,
                "words": f.total_words,
                "entries": f.total_entries,
                "classifications": f.classification_counts,
                "urls": f.url_count,
            }
            for f in analysis.files
        ],
    }

    output_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Report exported to: {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze diary data before ingestion into ARA's memory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("path", help="Path to diary directory")
    parser.add_argument("--pattern", default="*.md", help="File glob pattern (default: *.md)")
    parser.add_argument("--no-recursive", dest="recursive", action="store_false")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show per-file breakdown")
    parser.add_argument("--samples", type=int, default=0, help="Show N sample entries per category")
    parser.add_argument("--export", type=str, default="", help="Export report to JSON file")

    args = parser.parse_args()

    dir_path = Path(args.path)
    if not dir_path.exists() or not dir_path.is_dir():
        print(f"Error: Directory not found: {dir_path}")
        return 1

    print(f"Scanning {dir_path}...")
    analysis = analyze_directory(
        dir_path,
        pattern=args.pattern,
        recursive=args.recursive,
    )

    print_report(analysis, verbose=args.verbose, samples=args.samples)

    if args.export:
        export_report(analysis, Path(args.export))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
