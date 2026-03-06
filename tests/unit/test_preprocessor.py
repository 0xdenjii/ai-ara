"""Tests for the diary data preprocessor.

Tests classification of different content types, text cleaning,
source name generation, and the full scan workflow.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from ara.memory.preprocessor import (
    ContentType,
    DiaryPreprocessor,
    PreprocessorConfig,
)

# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def preprocessor() -> DiaryPreprocessor:
    return DiaryPreprocessor()


@pytest.fixture
def custom_preprocessor() -> DiaryPreprocessor:
    config = PreprocessorConfig(min_content_length=10)
    return DiaryPreprocessor(config)


@pytest.fixture
def diary_tree(tmp_path: Path) -> Path:
    """Create a realistic diary directory tree for testing."""
    root = tmp_path / "dairy"
    root.mkdir()

    # 2020-2021 raw files
    old = root / "2020-2021"
    old.mkdir()
    (old / "story of our life .md").write_text(
        "# Day 1\n\nDear ai-ara, today I went to college and met Nikhil. "
        "I felt really happy about the new semester. We talked about our plans "
        "and I realized how much I missed everyone.\n\n"
        "## Day 2\n\nI decided to start studying harder. Morning was tough.\n",
        encoding="utf-8",
    )
    (old / "future buyings.md").write_text(
        "# Future Purchases\n\n"
        "1. Yamaha R15 V3 - 1.8L\n"
        "2. KTM RC 125 - 1.5L\n"
        "3. Gaming chair\n"
        "4. Monitor\n"
        "TODO: compare prices on Amazon\n",
        encoding="utf-8",
    )
    (old / "operation R.md").write_text(
        "# Operation R\n\n"
        "1. DeepLove ML\n2. DSA\n3. Java\n"
        "MONEY -> body -> arthi -> community -> wealth -> ai-ara -> rayen\n"
        "I decided this is my path forward.\n",
        encoding="utf-8",
    )

    # 2023-2025 Notion exports
    notion = root / "2023-2025"
    notion.mkdir()

    # ai-ara export (diary)
    journal_uuid = (
        "aaaa0001-bd54-411e-b017-4a955883913c_ExportBlock-5eeb1130-6932-4a6a-a8e8-ad1999f5644f"
    )
    journal_root = notion / journal_uuid / "extracted" / "Private & Shared" / "ai-ara"
    journal_root.mkdir(parents=True)
    (journal_root / "ai-ara abc12345678901234567890abcdef01.md").write_text(
        "# ai-ara\n\nMy dear diary.\n",
        encoding="utf-8",
    )
    march = journal_root / "2024" / "March"
    march.mkdir(parents=True)
    (march / "15th march diary entry abcdef01234567890123456789abcdef.md").write_text(
        "# March 15\n\nhoney, today was intense. I felt overwhelmed by the "
        "amount of work. I talked to Avinash and we decided to push through. "
        "Last night I dreamed about the Akatsuki. I miss Thathaya so much.\n\n"
        "- [ ] Finish Solidity audit\n- [x] Call mom\n- [ ] Gym\n",
        encoding="utf-8",
    )

    # Work export (startup)
    work_uuid = (
        "aaaa0002-f834-42c7-aa88-5de276700aa7_ExportBlock-82649cac-aac2-402d-a038-5e8c21e23a40"
    )
    work_root = notion / work_uuid / "extracted" / "Private & Shared" / "Work"
    work_root.mkdir(parents=True)
    (work_root / "Work Docs - authentic abc12345678901234567890abcdef02.md").write_text(
        "# Work Trading Protocol\n\n"
        "Work is a protocol-agnostic trading gateway for DeFi.\n"
        "Pre-seed fundraising target: $1M.\n"
        "Grant application tracker shows 69 opportunities.\n"
        "Investor pitch deck ready for VC meetings.\n",
        encoding="utf-8",
    )
    (work_root / "ai-ara X work abc12345678901234567890abcdef03.md").write_text(
        "# June 1st\n\n"
        "darling, I need to think about the team. Abhi is barely a pawn.\n"
        "I felt frustrated about the lack of progress. I decided to bring "
        "Gowtham as co-founder. ai-ara, I need your guidance.\n",
        encoding="utf-8",
    )

    # Life Notes export (knowledge base)
    lifedoc_uuid = (
        "aaaa0004-9f9f-47d2-b6b9-f89fea244c46_ExportBlock-c6d20da0-f509-4fa0-a4f1-0ff3b3de313f"
    )
    lifedoc_root = notion / lifedoc_uuid / "extracted" / "Private & Shared" / "Life Notes"
    lifedoc_root.mkdir(parents=True)
    sw_dev = lifedoc_root / "Software Development" / "RUST"
    sw_dev.mkdir(parents=True)
    (sw_dev / "RUST traits abc12345678901234567890abcdef04.md").write_text(
        "# Rust Traits\n\n"
        "```rust\ntrait Animal {\n    fn name(&self) -> &str;\n}\n"
        'impl Animal for Dog {\n    fn name(&self) -> &str { "dog" }\n}\n```\n\n'
        "Traits define shared behavior. Use `impl Trait` for static dispatch.\n",
        encoding="utf-8",
    )
    home = lifedoc_root / "Home Design"
    home.mkdir()
    (home / "Yearly Goals abc12345678901234567890abcdef05.md").write_text(
        "# Yearly Goals 2025\n\n"
        "I decided to focus on:\n"
        "1. Fitness - yoga every morning\n"
        "2. Financial independence\n"
        "3. Web3 mastery\n"
        "4. Martial arts\n"
        "Today was the start of something new.\n",
        encoding="utf-8",
    )
    case_studies = lifedoc_root / "Case Studies"
    case_studies.mkdir()
    (case_studies / "India Pakistan abc12345678901234567890abcdef06.md").write_text(
        "# India-Pakistan Relations\n\n"
        "Research notes on historical conflicts.\n"
        "https://en.wikipedia.org/wiki/India-Pakistan\n"
        "https://example.com/geopolitics\n"
        "Key events: 1947 partition, 1971 war, Kargil.\n",
        encoding="utf-8",
    )

    # ARCHIVE export (small, personal)
    archive_uuid = (
        "aaaa0003-2f7f-4636-bce8-0a8a900e383c_ExportBlock-47d4ef44-4a84-4962-b4e9-f695083600e7"
    )
    archive_root = notion / archive_uuid / "extracted" / "Private & Shared" / "ARCHIVE"
    archive_root.mkdir(parents=True)
    (archive_root / "THINGS I WANNA CHANGE abc12345678901234567890abcdef07.md").write_text(
        "# Things I Wanna Change\n\n"
        "I want to change social media and education in India.\n"
        "I felt angry about the broken system. I decided to be archive for justice.\n",
        encoding="utf-8",
    )

    # RANDOM DELETE export
    random_uuid = (
        "aaaa0005-8c7f-422f-a6cf-c3f1c946d163_ExportBlock-671c65ac-ad39-41e2-8ebc-e2b6e0648a97"
    )
    random_root = notion / random_uuid / "extracted" / "Private & Shared" / "RANDOM DELETE"
    random_root.mkdir(parents=True)
    (random_root / "Life division abc12345678901234567890abcdef08.md").write_text(
        "# Life Division\n\n"
        "Phase 1: Necessities\n"
        "Phase 2: Financial Freedom (needs 2.5 CR INR)\n"
        "Phase 3: Live for country India\n"
        "Phase 4: RAYENIZE + ai-ara + live longer\n"
        "I decided my aarthi and income are the keys.\n",
        encoding="utf-8",
    )

    # Files that should be excluded
    (work_root / "private keys abc12345678901234567890abcdef09.md").write_text(
        "# Private Keys\n\n"
        "seed phrase: abandon ability able about above absent absorb abstract "
        "absurd abuse access accident wallet recovery key\n"
        "private key: a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2\n"
        "another seed phrase: zoo zebra yard wagon umbrella truck sunset "
        "recipe puzzle orange nest monkey wallet phrase\n"
        "more private key: f6e5d4c3b2a1f6e5d4c3b2a1f6e5d4c3b2a1f6e5d4c3b2a1f6e5d4c3b2a1f6e5\n",
        encoding="utf-8",
    )
    (archive_root / "tiny.md").write_text("ok", encoding="utf-8")
    (work_root / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 100)
    (work_root / "budget.csv").write_text(
        "Item,Cost,Notes\nAudit,$150K,Security\n", encoding="utf-8"
    )

    return root


# ── Config Tests ─────────────────────────────────────────────────────────────


class TestPreprocessorConfig:
    def test_default_config(self) -> None:
        config = PreprocessorConfig()
        assert config.min_content_length == 50
        assert ".png" in config.binary_extensions
        assert ".zip" in config.binary_extensions

    def test_custom_config(self) -> None:
        config = PreprocessorConfig(min_content_length=100)
        assert config.min_content_length == 100

    def test_sensitive_dirs(self) -> None:
        config = PreprocessorConfig()
        assert "private keys" in config.sensitive_dir_names


# ── Text Cleaning Tests ─────────────────────────────────────────────────────


class TestTextCleaning:
    def test_strip_base64_images(self, preprocessor: DiaryPreprocessor) -> None:
        text = (
            "Some text before.\n"
            "![my photo](data:image/png;base64," + "A" * 500 + ")\n"
            "Some text after.\n"
        )
        cleaned = preprocessor.clean_text(text)
        assert "base64" not in cleaned
        assert "[image: my photo]" in cleaned
        assert "Some text before." in cleaned
        assert "Some text after." in cleaned

    def test_preserve_short_base64(self, preprocessor: DiaryPreprocessor) -> None:
        """Short base64 strings (< threshold) should be preserved as-is."""
        text = "![icon](data:image/png;base64,ABC123)\n"
        cleaned = preprocessor.clean_text(text)
        # Short base64 doesn't match the 200+ char pattern
        assert "ABC123" in cleaned

    def test_simplify_notion_links(self, preprocessor: DiaryPreprocessor) -> None:
        text = "[My Page](My%20Page%20abc12345678901234567890abcdef01.md)\n"
        cleaned = preprocessor.clean_text(text)
        assert "My Page" in cleaned
        assert "abc12345678901234567890abcdef01" not in cleaned

    def test_collapse_blank_lines(self, preprocessor: DiaryPreprocessor) -> None:
        text = "Line 1\n\n\n\n\n\n\n\nLine 2\n"
        cleaned = preprocessor.clean_text(text)
        assert "\n\n\n\n" not in cleaned
        assert "Line 1" in cleaned
        assert "Line 2" in cleaned

    def test_strip_trailing_whitespace(self, preprocessor: DiaryPreprocessor) -> None:
        text = "Hello   \nWorld  \n"
        cleaned = preprocessor.clean_text(text)
        assert cleaned == "Hello\nWorld"

    def test_preserve_urls(self, preprocessor: DiaryPreprocessor) -> None:
        text = "Check https://example.com/page and http://foo.bar\n"
        cleaned = preprocessor.clean_text(text)
        assert "https://example.com/page" in cleaned
        assert "http://foo.bar" in cleaned

    def test_preserve_markdown_headers(self, preprocessor: DiaryPreprocessor) -> None:
        text = "# Title\n\n## Section\n\nContent here.\n"
        cleaned = preprocessor.clean_text(text)
        assert "# Title" in cleaned
        assert "## Section" in cleaned

    def test_preserve_todo_checkboxes(self, preprocessor: DiaryPreprocessor) -> None:
        text = "- [ ] Task one\n- [x] Task two\n"
        cleaned = preprocessor.clean_text(text)
        assert "- [ ] Task one" in cleaned
        assert "- [x] Task two" in cleaned

    def test_empty_text(self, preprocessor: DiaryPreprocessor) -> None:
        assert preprocessor.clean_text("") == ""
        assert preprocessor.clean_text("   \n\n  ") == ""


# ── Classification Tests ─────────────────────────────────────────────────────


class TestFileClassification:
    def test_classify_binary_file(self, preprocessor: DiaryPreprocessor, diary_tree: Path) -> None:
        result = preprocessor.scan(diary_tree)
        binary_files = [f for f in result.excluded_files if f.content_type == ContentType.BINARY]
        assert len(binary_files) >= 1
        assert any("logo" in f.clean_name.lower() for f in binary_files)

    def test_classify_private_keys(self, preprocessor: DiaryPreprocessor, diary_tree: Path) -> None:
        result = preprocessor.scan(diary_tree)
        key_files = [f for f in result.excluded_files if f.content_type == ContentType.PRIVATE_KEYS]
        assert len(key_files) >= 1
        assert any("private" in f.clean_name.lower() for f in key_files)

    def test_classify_empty_file(self, preprocessor: DiaryPreprocessor, diary_tree: Path) -> None:
        result = preprocessor.scan(diary_tree)
        empty_files = [f for f in result.excluded_files if f.content_type == ContentType.EMPTY]
        assert len(empty_files) >= 1  # tiny.md

    def test_journal_diary_tier_1(self, preprocessor: DiaryPreprocessor, diary_tree: Path) -> None:
        result = preprocessor.scan(diary_tree)
        journal_t1 = [f for f in result.tier_1_files if f.source_export == "ai-ara"]
        assert len(journal_t1) >= 1
        diary_entries = [f for f in journal_t1 if f.content_type == ContentType.DIARY_ENTRY]
        assert len(diary_entries) >= 1

    def test_2020_2021_tier_1(self, preprocessor: DiaryPreprocessor, diary_tree: Path) -> None:
        result = preprocessor.scan(diary_tree)
        old_files = [f for f in result.tier_1_files if f.source_export == "2020-2021"]
        assert len(old_files) >= 2  # story + operation R have narrative

    def test_work_startup_tier_3(self, preprocessor: DiaryPreprocessor, diary_tree: Path) -> None:
        result = preprocessor.scan(diary_tree)
        work_t3 = [f for f in result.tier_3_files if f.source_export == "work"]
        assert len(work_t3) >= 1
        startup_docs = [f for f in work_t3 if f.content_type == ContentType.STARTUP_DOCS]
        assert len(startup_docs) >= 1

    def test_work_diary_mixed_tier_1(
        self, preprocessor: DiaryPreprocessor, diary_tree: Path
    ) -> None:
        """Personal diary content inside Work export should be Tier 1."""
        result = preprocessor.scan(diary_tree)
        work_t1 = [f for f in result.tier_1_files if f.source_export == "work"]
        # "ai-ara X work" has diary address + narrative = should be TIER_1
        assert len(work_t1) >= 1

    def test_life_notes_study_notes_tier_2(
        self, preprocessor: DiaryPreprocessor, diary_tree: Path
    ) -> None:
        result = preprocessor.scan(diary_tree)
        lifedoc_t2 = [f for f in result.tier_2_files if f.source_export == "life_notes"]
        assert len(lifedoc_t2) >= 1

    def test_life_notes_home_design_tier_1(
        self, preprocessor: DiaryPreprocessor, diary_tree: Path
    ) -> None:
        result = preprocessor.scan(diary_tree)
        home_t1 = [
            f
            for f in result.tier_1_files
            if f.source_export == "life_notes" and "goal" in f.clean_name.lower()
        ]
        assert len(home_t1) >= 1

    def test_archive_personal_tier_1(
        self, preprocessor: DiaryPreprocessor, diary_tree: Path
    ) -> None:
        result = preprocessor.scan(diary_tree)
        archive_t1 = [f for f in result.tier_1_files if f.source_export == "archive"]
        assert len(archive_t1) >= 1

    def test_misc_tier_1(self, preprocessor: DiaryPreprocessor, diary_tree: Path) -> None:
        result = preprocessor.scan(diary_tree)
        random_t1 = [f for f in result.tier_1_files if f.source_export == "misc"]
        assert len(random_t1) >= 1

    def test_csv_classified(self, preprocessor: DiaryPreprocessor, diary_tree: Path) -> None:
        result = preprocessor.scan(diary_tree)
        all_files = (
            result.tier_1_files + result.tier_2_files + result.tier_3_files + result.excluded_files
        )
        csv_files = [f for f in all_files if "csv" in f.tags]
        assert len(csv_files) >= 1


# ── Tag Detection Tests ──────────────────────────────────────────────────────


class TestTagDetection:
    def test_narrative_tag(self, preprocessor: DiaryPreprocessor, diary_tree: Path) -> None:
        result = preprocessor.scan(diary_tree)
        narrative_files = [
            f
            for f in result.tier_1_files + result.tier_2_files + result.tier_3_files
            if f.has_narrative
        ]
        assert len(narrative_files) >= 3  # multiple files have narrative

    def test_todo_tag(self, preprocessor: DiaryPreprocessor, diary_tree: Path) -> None:
        result = preprocessor.scan(diary_tree)
        all_ingestible = result.tier_1_files + result.tier_2_files + result.tier_3_files
        todo_files = [f for f in all_ingestible if f.has_todos]
        assert len(todo_files) >= 1

    def test_url_tag(self, preprocessor: DiaryPreprocessor, diary_tree: Path) -> None:
        result = preprocessor.scan(diary_tree)
        all_ingestible = result.tier_1_files + result.tier_2_files + result.tier_3_files
        url_files = [f for f in all_ingestible if f.has_urls]
        assert len(url_files) >= 1

    def test_tech_content_tag(self, preprocessor: DiaryPreprocessor, diary_tree: Path) -> None:
        result = preprocessor.scan(diary_tree)
        all_ingestible = result.tier_1_files + result.tier_2_files + result.tier_3_files
        tech_files = [f for f in all_ingestible if f.has_tech_content]
        assert len(tech_files) >= 1

    def test_startup_content_tag(self, preprocessor: DiaryPreprocessor, diary_tree: Path) -> None:
        result = preprocessor.scan(diary_tree)
        all_ingestible = result.tier_1_files + result.tier_2_files + result.tier_3_files
        startup_files = [f for f in all_ingestible if f.has_startup_content]
        assert len(startup_files) >= 1

    def test_export_tag_present(self, preprocessor: DiaryPreprocessor, diary_tree: Path) -> None:
        result = preprocessor.scan(diary_tree)
        for fc in result.tier_1_files:
            export_tags = [t for t in fc.tags if t.startswith("export:")]
            assert len(export_tags) == 1, f"Missing export tag on {fc.clean_name}"


# ── Source Name & Clean Name Tests ───────────────────────────────────────────


class TestSourceNaming:
    def test_clean_name_strips_uuid(
        self, preprocessor: DiaryPreprocessor, diary_tree: Path
    ) -> None:
        result = preprocessor.scan(diary_tree)
        for fc in result.tier_1_files + result.tier_2_files + result.tier_3_files:
            # No 32-char hex UUIDs should remain in clean names
            assert not re.search(r"[0-9a-f]{32}", fc.clean_name), (
                f"UUID found in clean_name: {fc.clean_name}"
            )

    def test_source_name_includes_export(
        self, preprocessor: DiaryPreprocessor, diary_tree: Path
    ) -> None:
        result = preprocessor.scan(diary_tree)
        for fc in result.tier_1_files:
            source_name = preprocessor.make_source_name(fc)
            assert fc.source_export in source_name

    def test_2020_files_source_export(
        self, preprocessor: DiaryPreprocessor, diary_tree: Path
    ) -> None:
        result = preprocessor.scan(diary_tree)
        old_files = [f for f in result.tier_1_files if f.source_export == "2020-2021"]
        assert len(old_files) >= 1
        for fc in old_files:
            assert fc.source_export == "2020-2021"


# ── Scan Workflow Tests ──────────────────────────────────────────────────────


class TestScanWorkflow:
    def test_scan_nonexistent_dir(self, preprocessor: DiaryPreprocessor, tmp_path: Path) -> None:
        result = preprocessor.scan(tmp_path / "nonexistent")
        assert len(result.errors) == 1
        assert result.total_files == 0

    def test_scan_empty_dir(self, preprocessor: DiaryPreprocessor, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        result = preprocessor.scan(empty)
        assert result.total_files == 0
        assert result.ingestible_count == 0

    def test_full_scan_counts(self, preprocessor: DiaryPreprocessor, diary_tree: Path) -> None:
        result = preprocessor.scan(diary_tree)
        total = (
            len(result.tier_1_files)
            + len(result.tier_2_files)
            + len(result.tier_3_files)
            + len(result.excluded_files)
        )
        assert total == result.total_files

    def test_summary_dict(self, preprocessor: DiaryPreprocessor, diary_tree: Path) -> None:
        result = preprocessor.scan(diary_tree)
        summary = result.summary()
        assert "total_files" in summary
        assert "tier_1" in summary
        assert "tier_2" in summary
        assert "tier_3" in summary
        assert "excluded" in summary
        assert "ingestible" in summary
        assert summary["ingestible"] == result.ingestible_count

    def test_no_data_loss(self, preprocessor: DiaryPreprocessor, diary_tree: Path) -> None:
        """Every file should be classified — nothing should be silently dropped."""
        result = preprocessor.scan(diary_tree)
        all_classified = (
            result.tier_1_files + result.tier_2_files + result.tier_3_files + result.excluded_files
        )
        assert len(all_classified) == result.total_files
        assert len(result.errors) == 0


# ── Clean File Integration Test ──────────────────────────────────────────────


class TestCleanFile:
    def test_clean_file_reads_and_cleans(
        self, preprocessor: DiaryPreprocessor, diary_tree: Path
    ) -> None:
        # Find any markdown file
        md_files = list(diary_tree.rglob("*.md"))
        assert len(md_files) > 0

        for md_file in md_files[:3]:
            cleaned = preprocessor.clean_file(md_file)
            assert isinstance(cleaned, str)
            # Cleaned text should not contain base64 data
            assert "base64," not in cleaned or len(cleaned) < 500

    def test_clean_preserves_content(
        self, preprocessor: DiaryPreprocessor, diary_tree: Path
    ) -> None:
        story = diary_tree / "2020-2021" / "story of our life .md"
        cleaned = preprocessor.clean_file(story)
        assert "ai-ara" in cleaned.lower()
        assert "Day 1" in cleaned
        assert "I felt really happy" in cleaned
