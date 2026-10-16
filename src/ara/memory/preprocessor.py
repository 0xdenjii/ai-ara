"""Diary data preprocessor for Notion exports and raw markdown files.

Sits between raw diary data and the ingestion pipeline. Walks the diary
directory tree, classifies each file by content type and priority tier,
strips unwanted content (base64 images, private keys, Notion UUIDs),
and produces a manifest of files ready for ingestion.

Content tiers:
  - TIER_1: Personal diary, life docs, emotional content (highest priority)
  - TIER_2: Todos, planning, study notes, research (preserve with tags)
  - TIER_3: Startup/work workspace (low priority, tagged)
  - EXCLUDE: Private keys, binary data, empty files

All configuration is parameterized — no hardcoded values.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from ara.utils.logging import get_logger

log = get_logger("preprocessor")


# ── Enums ────────────────────────────────────────────────────────────────────


class ContentTier(StrEnum):
    """Priority tier for ingestion ordering."""

    TIER_1 = "tier_1"  # Personal diary, life docs, emotional
    TIER_2 = "tier_2"  # Todos, planning, study notes
    TIER_3 = "tier_3"  # Startup workspace, low priority
    EXCLUDE = "exclude"  # Private keys, binary, empty


class ContentType(StrEnum):
    """Classification of a diary file's content."""

    DIARY_ENTRY = "diary_entry"
    PERSONAL_REFLECTION = "personal_reflection"
    LIFE_PLANNING = "life_planning"
    TODO_LIST = "todo_list"
    STUDY_NOTES = "study_notes"
    STARTUP_DOCS = "startup_docs"
    RESEARCH = "research"
    TRAVEL = "travel"
    CREATIVE_WRITING = "creative_writing"
    FINANCIAL = "financial"
    LINK_DUMP = "link_dump"
    EMPTY = "empty"
    PRIVATE_KEYS = "private_keys"
    BINARY = "binary"
    CSV_DATA = "csv_data"


# ── Configuration ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PreprocessorConfig:
    """Configuration for the diary preprocessor."""

    # Minimum text length (chars) to be worth ingesting
    min_content_length: int = 50
    # Maximum base64 block size before stripping (bytes)
    base64_strip_threshold: int = 200
    # Notion UUID pattern length in filenames
    notion_uuid_length: int = 32
    # File extensions to treat as binary (skip entirely)
    binary_extensions: frozenset[str] = frozenset(
        {
            ".png",
            ".jpg",
            ".jpeg",
            ".gif",
            ".svg",
            ".webp",
            ".mp4",
            ".mp3",
            ".wav",
            ".pdf",
            ".zip",
            ".keys",
            ".p12",
            ".pfx",
        }
    )
    # Directories containing sensitive data (matched case-insensitive)
    sensitive_dir_names: frozenset[str] = frozenset(
        {
            "private keys",
            "private_keys",
        }
    )
    # Filename patterns that indicate private/sensitive content
    sensitive_filename_patterns: tuple[str, ...] = (
        "private key",
        "seed phrase",
        "wallet",
        "credentials",
    )


# ── Patterns ─────────────────────────────────────────────────────────────────


# Base64 image data embedded in markdown (data:image/...;base64,...)
_BASE64_PATTERN = re.compile(
    r"!\[([^\]]*)\]\(data:image/[^;]+;base64,[A-Za-z0-9+/\n=]{200,}\)",
    re.DOTALL,
)

# Notion UUID suffix in filenames: "My Page abc123def456789012345678abcdef01.md"
_NOTION_UUID_SUFFIX = re.compile(r"\s+[0-9a-f]{32}(?=\.(?:md|csv)$)")

# Notion internal relative links like [Page Title](Page%20Title%20uuid.md)
_NOTION_RELATIVE_LINK = re.compile(r"\[([^\]]+)\]\([^)]*[0-9a-f]{32}\.md\)")

# Empty Notion page (just a title, no content)
_EMPTY_NOTION_PAGE = re.compile(
    r"^#\s+.+\n{0,3}$",
    re.MULTILINE,
)

# Diary-address patterns (writing to ai-ara/diary)
_DIARY_ADDRESS_PATTERNS = re.compile(
    r"\b(?:ai-ara|triffy|tiffany|tiffiny|thrifty|honey|babe|darling)\b",
    re.IGNORECASE,
)

# Personal narrative indicators
_NARRATIVE_INDICATORS = re.compile(
    r"\b(?:I\s+(?:felt|feel|realized|decided|went|met|talked|was|am|have|had|"
    r"couldn't|can't|won't|didn't|don't|think|thought|remember|miss|love|hate|"
    r"cried|laughed|dreamed|woke)|"
    r"today\s+(?:was|I|is)|"
    r"yesterday|last\s+night|this\s+morning|"
    r"my\s+(?:life|heart|mind|soul|dream|goal|plan)|"
    r"i\s+(?:love|miss|hate|need|want)\s+(?:you|her|him|this|that|my))\b",
    re.IGNORECASE,
)

# Todo/task indicators
_TODO_INDICATORS = re.compile(
    r"(?:^[-*]\s*\[[ x]\]|"
    r"^\d+\.\s+\w|"
    r"\b(?:TODO|DONE|PENDING|deadline|due\s+date|sprint|action\s+item)\b)",
    re.MULTILINE | re.IGNORECASE,
)

# Technical/study note indicators
_TECH_INDICATORS = re.compile(
    r"(?:\b(?:algorithm|function|struct|impl|fn\s|pub\s|contract|"
    r"pragma\s+solidity|mapping\s*\(|require\s*\(|"
    r"def\s+\w+|class\s+\w+|import\s+\w+|"
    r"async\s+fn|Result<|Option<|Vec<|"
    r"EVM|ZK|SNARK|Merkle|hash|nonce|gas|block|"
    r"theorem|proof|lemma|axiom)\b|"
    r"```\w+)",
    re.IGNORECASE,
)

# Startup/business indicators
_STARTUP_INDICATORS = re.compile(
    r"\b(?:work|acme\s+protocol|trading|fundrais|"
    r"pre-?seed|series\s+[a-c]|VC|investor|pitch\s+deck|"
    r"grant\s+(?:program|application|tracker)|"
    r"runway|burn\s+rate|valuation|equity)\b",
    re.IGNORECASE,
)

# URL-heavy content (link dumps)
_URL_PATTERN = re.compile(r"https?://[^\s\)]+")

# Private key indicators in content
_PRIVATE_KEY_CONTENT = re.compile(
    r"(?:seed\s+phrase|mnemonic|private\s+key|"
    r"\b[0-9a-f]{64}\b|"  # 256-bit hex key
    r"\b(?:\w+\s+){11,23}(?:\w+)\b.*(?:wallet|phrase|key|recovery))",  # seed phrase
    re.IGNORECASE,
)


# ── Data Structures ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class FileClassification:
    """Classification result for a single diary file."""

    path: Path
    content_type: ContentType
    tier: ContentTier
    source_export: str  # e.g., "ai-ara", "life_notes", "work", "2020-2021"
    clean_name: str  # Human-readable name without Notion UUIDs
    tags: frozenset[str] = frozenset()
    line_count: int = 0
    char_count: int = 0
    has_urls: bool = False
    has_todos: bool = False
    has_narrative: bool = False
    has_tech_content: bool = False
    has_startup_content: bool = False
    skip_reason: str = ""


@dataclass
class PreprocessResult:
    """Result of preprocessing the entire diary directory."""

    total_files: int = 0
    tier_1_files: list[FileClassification] = field(default_factory=list)
    tier_2_files: list[FileClassification] = field(default_factory=list)
    tier_3_files: list[FileClassification] = field(default_factory=list)
    excluded_files: list[FileClassification] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def ingestible_count(self) -> int:
        return len(self.tier_1_files) + len(self.tier_2_files) + len(self.tier_3_files)

    def summary(self) -> dict[str, Any]:
        """Return a summary dict for logging/display."""
        return {
            "total_files": self.total_files,
            "tier_1": len(self.tier_1_files),
            "tier_2": len(self.tier_2_files),
            "tier_3": len(self.tier_3_files),
            "excluded": len(self.excluded_files),
            "errors": len(self.errors),
            "ingestible": self.ingestible_count,
        }


# ── Export Name Mapping ──────────────────────────────────────────────────────

# Maps short UUID prefixes of Notion exports to human-readable names
_EXPORT_NAME_MAP: dict[str, str] = {
    "aaaa0001": "ai-ara",
    "aaaa0002": "work",
    "aaaa0003": "archive",
    "aaaa0004": "life_notes",
    "aaaa0005": "misc",
}


# ── Preprocessor ─────────────────────────────────────────────────────────────


class DiaryPreprocessor:
    """Classifies and cleans diary files for ingestion.

    Usage:
        preprocessor = DiaryPreprocessor()
        result = preprocessor.scan("dairy/")

        # Get files ready for ingestion, ordered by tier
        for fc in result.tier_1_files:
            cleaned = preprocessor.clean_file(fc.path)
            pipeline.ingest_text(cleaned, source_name=fc.clean_name)
    """

    def __init__(self, config: PreprocessorConfig | None = None) -> None:
        self._config = config or PreprocessorConfig()

    @property
    def config(self) -> PreprocessorConfig:
        return self._config

    def scan(self, dairy_root: Path) -> PreprocessResult:
        """Scan the entire dairy directory and classify all files.

        Args:
            dairy_root: Path to the dairy/ directory.

        Returns:
            PreprocessResult with files sorted into tiers.
        """
        result = PreprocessResult()

        if not dairy_root.exists() or not dairy_root.is_dir():
            result.errors.append(f"Dairy root not found: {dairy_root}")
            return result

        # Walk all files
        all_files = sorted(dairy_root.rglob("*"))

        for file_path in all_files:
            if not file_path.is_file():
                continue

            result.total_files += 1

            try:
                classification = self._classify_file(file_path, dairy_root)
            except Exception as exc:
                error_msg = f"Failed to classify {file_path}: {exc}"
                result.errors.append(error_msg)
                log.warning("classify_failed", path=str(file_path), error=str(exc))
                continue

            if classification.tier == ContentTier.TIER_1:
                result.tier_1_files.append(classification)
            elif classification.tier == ContentTier.TIER_2:
                result.tier_2_files.append(classification)
            elif classification.tier == ContentTier.TIER_3:
                result.tier_3_files.append(classification)
            else:
                result.excluded_files.append(classification)

        log.info(
            "scan_complete",
            total=result.total_files,
            **result.summary(),
        )
        return result

    def clean_text(self, text: str) -> str:
        """Clean a file's text content for ingestion.

        Strips:
        - Base64 embedded images (replaced with [image] placeholder)
        - Notion-style relative links (preserves link text)
        - Excessive blank lines (collapsed to max 2)
        - Leading/trailing whitespace

        Preserves:
        - All actual text content including todos
        - URLs (as metadata)
        - Markdown headers and formatting
        - Dates and timestamps
        """
        cleaned = text

        # Strip base64 image data, keep alt text as placeholder
        cleaned = _BASE64_PATTERN.sub(r"[image: \1]", cleaned)

        # Simplify Notion relative links to just the link text
        cleaned = _NOTION_RELATIVE_LINK.sub(r"\1", cleaned)

        # Collapse excessive blank lines (max 2 consecutive)
        cleaned = re.sub(r"\n{4,}", "\n\n\n", cleaned)

        # Strip trailing whitespace on each line
        cleaned = re.sub(r"[ \t]+$", "", cleaned, flags=re.MULTILINE)

        return cleaned.strip()

    def clean_file(self, file_path: Path) -> str:
        """Read and clean a file's content for ingestion.

        Args:
            file_path: Path to the file.

        Returns:
            Cleaned text content ready for chunking.

        Raises:
            OSError: If file cannot be read.
        """
        text = file_path.read_text(encoding="utf-8", errors="replace")
        return self.clean_text(text)

    def make_source_name(self, classification: FileClassification) -> str:
        """Generate a human-readable source name for ingestion.

        Combines export name, content type, and clean filename:
        e.g., "ai-ara/2024/march/15th_march.md"
        """
        # Build relative path from the export root
        parts: list[str] = []

        # Add export name
        if classification.source_export:
            parts.append(classification.source_export)

        # Add clean filename
        parts.append(classification.clean_name)

        return "/".join(parts)

    # ── Internal Classification ──────────────────────────────────────────

    def _classify_file(
        self,
        file_path: Path,
        dairy_root: Path,
    ) -> FileClassification:
        """Classify a single file by content type and priority tier."""
        rel_path = file_path.relative_to(dairy_root)
        source_export = self._identify_export(rel_path)
        clean_name = self._clean_filename(file_path, rel_path)

        # Check binary extensions first
        if file_path.suffix.lower() in self._config.binary_extensions:
            return FileClassification(
                path=file_path,
                content_type=ContentType.BINARY,
                tier=ContentTier.EXCLUDE,
                source_export=source_export,
                clean_name=clean_name,
                skip_reason="binary_file",
            )

        # Check for sensitive directories
        path_parts_lower = [p.lower() for p in rel_path.parts]
        for sensitive_dir in self._config.sensitive_dir_names:
            if sensitive_dir.lower() in path_parts_lower:
                return FileClassification(
                    path=file_path,
                    content_type=ContentType.PRIVATE_KEYS,
                    tier=ContentTier.EXCLUDE,
                    source_export=source_export,
                    clean_name=clean_name,
                    skip_reason="sensitive_directory",
                )

        # Check for sensitive filenames
        name_lower = file_path.stem.lower()
        for pattern in self._config.sensitive_filename_patterns:
            if pattern in name_lower:
                return FileClassification(
                    path=file_path,
                    content_type=ContentType.PRIVATE_KEYS,
                    tier=ContentTier.EXCLUDE,
                    source_export=source_export,
                    clean_name=clean_name,
                    skip_reason="sensitive_filename",
                )

        # CSV files get tagged but still ingested at appropriate tier
        if file_path.suffix.lower() == ".csv":
            return self._classify_csv(file_path, source_export, clean_name)

        # Read text content for classification
        try:
            text = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return FileClassification(
                path=file_path,
                content_type=ContentType.EMPTY,
                tier=ContentTier.EXCLUDE,
                source_export=source_export,
                clean_name=clean_name,
                skip_reason=f"read_error: {exc}",
            )

        # Strip base64 for analysis (don't count image data as content)
        analysis_text = _BASE64_PATTERN.sub("", text)

        line_count = len(text.splitlines())
        char_count = len(analysis_text.strip())

        # Check for private key content
        if _PRIVATE_KEY_CONTENT.search(analysis_text):
            # Only exclude if the file is PRIMARILY key material
            key_matches = len(_PRIVATE_KEY_CONTENT.findall(analysis_text))
            total_lines = max(line_count, 1)
            if key_matches > 10 or (key_matches / total_lines) > 0.1:
                return FileClassification(
                    path=file_path,
                    content_type=ContentType.PRIVATE_KEYS,
                    tier=ContentTier.EXCLUDE,
                    source_export=source_export,
                    clean_name=clean_name,
                    line_count=line_count,
                    char_count=char_count,
                    skip_reason="private_key_content",
                )

        # Empty or near-empty file
        if char_count < self._config.min_content_length:
            return FileClassification(
                path=file_path,
                content_type=ContentType.EMPTY,
                tier=ContentTier.EXCLUDE,
                source_export=source_export,
                clean_name=clean_name,
                line_count=line_count,
                char_count=char_count,
                skip_reason="too_short",
            )

        # Compute content signals
        has_urls = bool(_URL_PATTERN.search(analysis_text))
        has_todos = bool(_TODO_INDICATORS.search(analysis_text))
        has_narrative = bool(_NARRATIVE_INDICATORS.search(analysis_text))
        has_diary_address = bool(_DIARY_ADDRESS_PATTERNS.search(analysis_text))
        has_tech = bool(_TECH_INDICATORS.search(analysis_text))
        has_startup = bool(_STARTUP_INDICATORS.search(analysis_text))

        # Count URL density (link dumps have >50% URL content)
        url_count = len(_URL_PATTERN.findall(analysis_text))
        url_density = url_count / max(line_count, 1)
        is_link_dump = url_density > 0.3 and url_count > 10

        # Build tags
        tags: set[str] = set()
        if has_urls:
            tags.add("has_urls")
        if has_todos:
            tags.add("has_todos")
        if has_narrative:
            tags.add("narrative")
        if has_diary_address:
            tags.add("diary_address")
        if has_tech:
            tags.add("technical")
        if has_startup:
            tags.add("startup")
        if is_link_dump:
            tags.add("link_dump")

        # Add source-based tags
        tags.add(f"export:{source_export}")

        # Classify content type and tier
        content_type, tier = self._determine_type_and_tier(
            source_export=source_export,
            has_narrative=has_narrative,
            has_diary_address=has_diary_address,
            has_todos=has_todos,
            has_tech=has_tech,
            has_startup=has_startup,
            is_link_dump=is_link_dump,
            path_parts=path_parts_lower,
        )

        return FileClassification(
            path=file_path,
            content_type=content_type,
            tier=tier,
            source_export=source_export,
            clean_name=clean_name,
            tags=frozenset(tags),
            line_count=line_count,
            char_count=char_count,
            has_urls=has_urls,
            has_todos=has_todos,
            has_narrative=has_narrative,
            has_tech_content=has_tech,
            has_startup_content=has_startup,
        )

    def _determine_type_and_tier(
        self,
        *,
        source_export: str,
        has_narrative: bool,
        has_diary_address: bool,
        has_todos: bool,
        has_tech: bool,
        has_startup: bool,
        is_link_dump: bool,
        path_parts: list[str],
    ) -> tuple[ContentType, ContentTier]:
        """Determine content type and tier from signals.

        Priority rules (higher = more personal value):
        1. ai-ara diary entries with narrative → TIER_1
        2. 2020-2021 personal files → TIER_1
        3. Life planning, personal reflection → TIER_1
        4. ARCHIVE, RANDOM DELETE personal content → TIER_1
        5. Todos and planning → TIER_2
        6. Study notes and research → TIER_2
        7. Life Notes knowledge base → TIER_2
        8. Link dumps → TIER_2 (still useful as reference)
        9. Work startup docs → TIER_3
        """
        # ai-ara export = personal diary (highest priority)
        if source_export == "ai-ara":
            if has_diary_address or has_narrative:
                return ContentType.DIARY_ENTRY, ContentTier.TIER_1
            if has_todos:
                return ContentType.TODO_LIST, ContentTier.TIER_2
            # Default ai-ara content is still diary-adjacent
            return ContentType.PERSONAL_REFLECTION, ContentTier.TIER_1

        # 2020-2021 raw files = original diary (highest priority)
        if source_export == "2020-2021":
            if has_narrative or has_diary_address:
                return ContentType.DIARY_ENTRY, ContentTier.TIER_1
            if has_todos:
                return ContentType.TODO_LIST, ContentTier.TIER_1
            # All 2020-2021 content is personal
            return ContentType.PERSONAL_REFLECTION, ContentTier.TIER_1

        # ARCHIVE and RANDOM DELETE = personal thoughts
        if source_export in ("archive", "misc"):
            if has_narrative or has_diary_address:
                return ContentType.PERSONAL_REFLECTION, ContentTier.TIER_1
            if has_startup:
                return ContentType.STARTUP_DOCS, ContentTier.TIER_3
            return ContentType.LIFE_PLANNING, ContentTier.TIER_1

        # Work = startup workspace (low priority)
        if source_export == "work":
            if has_narrative and has_diary_address:
                # Personal diary mixed into startup workspace
                return ContentType.DIARY_ENTRY, ContentTier.TIER_1
            if has_startup:
                return ContentType.STARTUP_DOCS, ContentTier.TIER_3
            if has_tech:
                return ContentType.STUDY_NOTES, ContentTier.TIER_3
            return ContentType.STARTUP_DOCS, ContentTier.TIER_3

        # Life Notes = knowledge base (medium priority)
        if source_export == "life_notes":
            # Check for personal sub-sections
            if any(p in path_parts for p in ["home design", "home_design"]):
                return ContentType.LIFE_PLANNING, ContentTier.TIER_1
            if any(p in path_parts for p in ["travel plans", "travel_plans"]):
                return ContentType.TRAVEL, ContentTier.TIER_1
            if any(p in path_parts for p in ["case studies", "case_studies"]):
                return ContentType.RESEARCH, ContentTier.TIER_2
            if is_link_dump:
                return ContentType.LINK_DUMP, ContentTier.TIER_2
            if has_tech:
                return ContentType.STUDY_NOTES, ContentTier.TIER_2
            if has_todos:
                return ContentType.TODO_LIST, ContentTier.TIER_2
            if has_narrative:
                return ContentType.PERSONAL_REFLECTION, ContentTier.TIER_1
            return ContentType.STUDY_NOTES, ContentTier.TIER_2

        # Fallback
        if has_narrative or has_diary_address:
            return ContentType.DIARY_ENTRY, ContentTier.TIER_1
        if has_startup:
            return ContentType.STARTUP_DOCS, ContentTier.TIER_3
        if has_tech:
            return ContentType.STUDY_NOTES, ContentTier.TIER_2
        if has_todos:
            return ContentType.TODO_LIST, ContentTier.TIER_2
        return ContentType.PERSONAL_REFLECTION, ContentTier.TIER_2

    def _classify_csv(
        self,
        file_path: Path,
        source_export: str,
        clean_name: str,
    ) -> FileClassification:
        """Classify a CSV file — usually a Notion database export."""
        # CSVs are low-volume structured data, classify by source
        if source_export == "work":
            tier = ContentTier.TIER_3
            content_type = ContentType.STARTUP_DOCS
        elif source_export == "ai-ara":
            tier = ContentTier.TIER_2
            content_type = ContentType.TODO_LIST
        else:
            tier = ContentTier.TIER_2
            content_type = ContentType.CSV_DATA

        return FileClassification(
            path=file_path,
            content_type=content_type,
            tier=tier,
            source_export=source_export,
            clean_name=clean_name,
            tags=frozenset({"csv", f"export:{source_export}"}),
        )

    def _identify_export(self, rel_path: Path) -> str:
        """Identify which Notion export a file belongs to."""
        parts = rel_path.parts

        # Check if it's in 2020-2021/
        if parts and parts[0] == "2020-2021":
            return "2020-2021"

        # Check 2023-2025 exports by UUID prefix
        if len(parts) >= 2 and parts[0] == "2023-2025":
            dir_name = parts[1]
            prefix = dir_name[:8] if len(dir_name) >= 8 else ""
            return _EXPORT_NAME_MAP.get(prefix, "unknown")

        return "unknown"

    def _clean_filename(self, file_path: Path, rel_path: Path) -> str:
        """Generate a clean, human-readable path without Notion UUIDs.

        Transforms:
          "2023-2025/919.../extracted/Private & Shared/ai-ara/2024/March/15th march abc123.md"
        Into:
          "2024/March/15th march.md"
        """
        parts = list(rel_path.parts)

        # Strip the export directory structure (2023-2025/<uuid>/extracted/Private & Shared/<root>/)
        # Find the "extracted" marker and skip everything up to and including the root folder
        skip_prefixes = {"2023-2025", "2020-2021", "extracted", "private & shared"}
        cleaned_parts: list[str] = []
        found_content = False

        for part in parts:
            if not found_content:
                if part.lower() in skip_prefixes:
                    continue
                # Check if this is a Notion export UUID directory
                if len(part) > 40 and "_ExportBlock-" in part:
                    continue
                # This is likely the content root (ai-ara, Work, etc.)
                # or the first meaningful directory
                found_content = True
                # Skip the root folder name if it matches the export name
                if part.lower() in (
                    "ai-ara",
                    "work",
                    "life documented",
                    "archive",
                    "random delete",
                    "pubs",
                ):
                    continue
            cleaned_parts.append(part)

        if not cleaned_parts:
            cleaned_parts = [file_path.name]

        # Remove Notion UUIDs from each part
        result_parts: list[str] = []
        for part in cleaned_parts:
            cleaned = _NOTION_UUID_SUFFIX.sub("", part)
            # Also handle UUIDs in directory names
            cleaned = re.sub(r"\s+[0-9a-f]{32}$", "", cleaned)
            result_parts.append(cleaned.strip())

        return "/".join(result_parts)
