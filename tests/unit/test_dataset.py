"""Tests for the fine-tuning dataset preparation module."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from ara.fine_tuning.dataset import (
    GENERAL_TOPICS,
    HUMOR_DISTRIBUTION,
    ChatMLMessage,
    ConversationType,
    DiarySegment,
    HumorLevel,
    TrainingSample,
    build_generation_prompt,
    build_system_message_for_training,
    compute_stats,
    count_conversation_tokens,
    count_tokens,
    create_training_sample,
    filter_by_quality,
    generate_general_conversation,
    generate_knowledge_qa,
    generate_personality_conversation,
    get_humor_level_for_index,
    get_target_counts,
    load_dataset,
    parse_diary_file,
    parse_llm_json_response,
    save_dataset,
    scan_diary_directory,
    score_sample,
    split_dataset,
    validate_conversation_messages,
)

# ── Token Counting ──────────────────────────────────────────────────────────


class TestCountTokens:
    def test_empty_string(self) -> None:
        assert count_tokens("") == 0

    def test_single_word(self) -> None:
        result = count_tokens("hello")
        assert result > 0

    def test_longer_text(self) -> None:
        result = count_tokens("The quick brown fox jumps over the lazy dog.")
        assert result > 5

    def test_consistency(self) -> None:
        text = "Hello world"
        assert count_tokens(text) == count_tokens(text)


class TestCountConversationTokens:
    def test_empty_conversation(self) -> None:
        assert count_conversation_tokens([]) == 0

    def test_single_message(self) -> None:
        msgs = [ChatMLMessage(role="user", content="Hello")]
        result = count_conversation_tokens(msgs)
        # Content tokens + ~4 overhead
        assert result > count_tokens("Hello")

    def test_multi_message(self) -> None:
        msgs = [
            ChatMLMessage(role="user", content="Hello"),
            ChatMLMessage(role="assistant", content="Hi there!"),
        ]
        result = count_conversation_tokens(msgs)
        assert result > count_tokens("Hello") + count_tokens("Hi there!")


# ── DiarySegment ────────────────────────────────────────────────────────────


class TestDiarySegment:
    def test_compute_id_deterministic(self) -> None:
        id1 = DiarySegment.compute_id("hello", "file.md")
        id2 = DiarySegment.compute_id("hello", "file.md")
        assert id1 == id2

    def test_compute_id_different_inputs(self) -> None:
        id1 = DiarySegment.compute_id("hello", "file1.md")
        id2 = DiarySegment.compute_id("hello", "file2.md")
        assert id1 != id2

    def test_compute_id_length(self) -> None:
        seg_id = DiarySegment.compute_id("test", "test.md")
        assert len(seg_id) == 16


# ── ChatML Types ────────────────────────────────────────────────────────────


class TestChatMLMessage:
    def test_creation(self) -> None:
        msg = ChatMLMessage(role="user", content="hello")
        assert msg.role == "user"
        assert msg.content == "hello"


class TestTrainingSample:
    def _make_sample(self) -> TrainingSample:
        return TrainingSample(
            sample_id="test_123",
            conversation_type=ConversationType.PERSONALITY,
            humor_level=HumorLevel.WITTY_IRREVERENT,
            messages=[
                ChatMLMessage(role="system", content="You are ARA"),
                ChatMLMessage(role="user", content="Hey"),
                ChatMLMessage(role="assistant", content="Sup"),
            ],
            source_segments=["seg1"],
            quality_score=4.5,
            token_count=100,
        )

    def test_to_chatml_dict(self) -> None:
        sample = self._make_sample()
        result = sample.to_chatml_dict()
        assert "messages" in result
        assert len(result["messages"]) == 3
        assert result["messages"][0]["role"] == "system"

    def test_to_full_dict(self) -> None:
        sample = self._make_sample()
        result = sample.to_full_dict()
        assert result["sample_id"] == "test_123"
        assert result["conversation_type"] == "personality"
        assert result["humor_level"] == "witty_irreverent"
        assert result["quality_score"] == 4.5
        assert len(result["messages"]) == 3
        assert result["source_segments"] == ["seg1"]


# ── Diary Parsing ───────────────────────────────────────────────────────────


class TestParseDiaryFile:
    def test_parse_with_date_headers(self, tmp_path: Path) -> None:
        content = textwrap.dedent("""\
            ## 2024-01-15
            Today was a good day. I went for a run and felt amazing afterwards.
            The weather was perfect for it. Clear skies, cool breeze.
            I should do this more often instead of just talking about it.
            The park was beautiful and I ran into an old friend which was nice.
            We chatted for a while and made plans to meet up again next week.

            ## 2024-01-16
            Not as motivated today. Stayed in bed too long scrolling through
            my phone. Need to break this habit. Maybe I should charge my phone
            in another room. That might actually work. Also I need to get back
            on track with my reading goal. I was supposed to finish that book
            by end of month but I have barely started the second chapter.
        """)
        diary_file = tmp_path / "test_diary.md"
        diary_file.write_text(content, encoding="utf-8")

        segments = parse_diary_file(diary_file)
        assert len(segments) == 2
        assert segments[0].date == "2024-01-15"
        assert segments[1].date == "2024-01-16"

    def test_parse_without_date_headers(self, tmp_path: Path) -> None:
        content = (
            "Just a regular note with enough content to pass the minimum token threshold. " * 10
        )
        diary_file = tmp_path / "note.md"
        diary_file.write_text(content, encoding="utf-8")

        segments = parse_diary_file(diary_file)
        assert len(segments) == 1
        assert segments[0].date == ""
        assert segments[0].section_title == "note"

    def test_parse_empty_file(self, tmp_path: Path) -> None:
        diary_file = tmp_path / "empty.md"
        diary_file.write_text("", encoding="utf-8")

        segments = parse_diary_file(diary_file)
        assert segments == []

    def test_parse_short_file_filtered(self, tmp_path: Path) -> None:
        diary_file = tmp_path / "short.md"
        diary_file.write_text("Too short.", encoding="utf-8")

        segments = parse_diary_file(diary_file)
        assert segments == []

    def test_parse_c_style_dates(self, tmp_path: Path) -> None:
        content = textwrap.dedent("""\
            #include<23/3/2020>
            Today was interesting. We explored the research lab and found some
            fascinating results. The experiment went well and we documented
            everything properly for future reference. I need to follow up
            on those results next week. The team was excited and we discussed
            the implications of our findings for the project. Everyone agreed
            that we should publish a paper about this. I also met with the
            professor to discuss my thesis progress and timeline ahead.
        """)
        diary_file = tmp_path / "code_diary.md"
        diary_file.write_text(content, encoding="utf-8")

        segments = parse_diary_file(diary_file)
        assert len(segments) >= 1
        assert segments[0].date == "23/3/2020"

    def test_segments_have_valid_ids(self, tmp_path: Path) -> None:
        content = textwrap.dedent("""\
            ## 2024-01-15
            A reasonably long diary entry that should pass the minimum token
            threshold for being included in the training dataset. This entry
            talks about various things that happened during the day.
        """)
        diary_file = tmp_path / "diary.md"
        diary_file.write_text(content, encoding="utf-8")

        segments = parse_diary_file(diary_file)
        for seg in segments:
            assert len(seg.segment_id) == 16
            assert seg.token_count > 0


class TestScanDiaryDirectory:
    def test_scan_finds_all_md_files(self, tmp_path: Path) -> None:
        # Create test diary files
        (tmp_path / "diary1.md").write_text(
            "## 2024-01-01\n" + "This is a diary entry. " * 20,
            encoding="utf-8",
        )
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "diary2.md").write_text(
            "## 2024-02-01\n" + "Another diary entry. " * 20,
            encoding="utf-8",
        )

        segments = scan_diary_directory(tmp_path)
        assert len(segments) >= 2

    def test_scan_skips_readme(self, tmp_path: Path) -> None:
        (tmp_path / "README.md").write_text(
            "# Project README\n" + "Documentation content. " * 20,
            encoding="utf-8",
        )
        segments = scan_diary_directory(tmp_path)
        assert segments == []

    def test_scan_skips_work(self, tmp_path: Path) -> None:
        work_dir = tmp_path / "work"
        work_dir.mkdir()
        (work_dir / "notes.md").write_text(
            "## 2024-01-01\n" + "Work project notes. " * 20,
            encoding="utf-8",
        )
        segments = scan_diary_directory(tmp_path)
        assert segments == []

    def test_scan_sorts_by_date(self, tmp_path: Path) -> None:
        (tmp_path / "a.md").write_text(
            "## 2024-03-01\n" + "March entry content. " * 20,
            encoding="utf-8",
        )
        (tmp_path / "b.md").write_text(
            "## 2024-01-01\n" + "January entry content. " * 20,
            encoding="utf-8",
        )

        segments = scan_diary_directory(tmp_path)
        if len(segments) >= 2:
            assert segments[0].date <= segments[1].date


# ── JSON Parsing ────────────────────────────────────────────────────────────


class TestParseLlmJsonResponse:
    def test_parse_clean_json_array(self) -> None:
        response = '[{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hey"}]'
        result = parse_llm_json_response(response)
        assert len(result) == 2
        assert result[0]["role"] == "user"

    def test_parse_json_object(self) -> None:
        response = '{"score": 4, "reason": "good"}'
        result = parse_llm_json_response(response)
        assert result["score"] == 4

    def test_parse_with_markdown_code_block(self) -> None:
        response = '```json\n[{"role": "user", "content": "hi"}]\n```'
        result = parse_llm_json_response(response)
        assert len(result) == 1

    def test_parse_with_surrounding_text(self) -> None:
        response = 'Here is the conversation:\n[{"role": "user", "content": "hi"}]\nDone!'
        result = parse_llm_json_response(response)
        assert len(result) == 1

    def test_parse_invalid_raises(self) -> None:
        with pytest.raises(ValueError, match="Failed to parse"):
            parse_llm_json_response("not json at all")


# ── Message Validation ──────────────────────────────────────────────────────


class TestValidateConversationMessages:
    def test_valid_conversation(self) -> None:
        raw = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hey there!"},
        ]
        result = validate_conversation_messages(raw)
        assert result is not None
        assert len(result) == 2

    def test_empty_list(self) -> None:
        assert validate_conversation_messages([]) is None

    def test_single_message(self) -> None:
        assert validate_conversation_messages([{"role": "user", "content": "hi"}]) is None

    def test_must_start_with_user(self) -> None:
        raw = [
            {"role": "assistant", "content": "Hello"},
            {"role": "user", "content": "Hey"},
        ]
        assert validate_conversation_messages(raw) is None

    def test_no_consecutive_assistant(self) -> None:
        raw = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hey"},
            {"role": "assistant", "content": "What's up"},
        ]
        assert validate_conversation_messages(raw) is None

    def test_allows_consecutive_user(self) -> None:
        raw = [
            {"role": "user", "content": "Hi"},
            {"role": "user", "content": "Are you there?"},
            {"role": "assistant", "content": "Yeah, I'm here"},
        ]
        result = validate_conversation_messages(raw)
        assert result is not None
        assert len(result) == 3

    def test_invalid_role(self) -> None:
        raw = [
            {"role": "user", "content": "Hi"},
            {"role": "system", "content": "nope"},
        ]
        assert validate_conversation_messages(raw) is None

    def test_empty_content(self) -> None:
        raw = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": ""},
        ]
        assert validate_conversation_messages(raw) is None


# ── Training Sample Creation ────────────────────────────────────────────────


class TestCreateTrainingSample:
    def test_creates_with_system_message(self) -> None:
        messages = [
            ChatMLMessage(role="user", content="Hey"),
            ChatMLMessage(role="assistant", content="Sup"),
        ]
        sample = create_training_sample(
            messages,
            ConversationType.PERSONALITY,
            HumorLevel.WITTY_IRREVERENT,
            ["seg1"],
        )
        assert sample.messages[0].role == "system"
        assert len(sample.messages) == 3
        assert sample.conversation_type == ConversationType.PERSONALITY
        assert sample.token_count > 0

    def test_creates_without_system_message(self) -> None:
        messages = [
            ChatMLMessage(role="user", content="Hey"),
            ChatMLMessage(role="assistant", content="Sup"),
        ]
        sample = create_training_sample(
            messages,
            ConversationType.GENERAL,
            HumorLevel.SUPPORTIVE,
            [],
            include_system=False,
        )
        assert sample.messages[0].role == "user"
        assert len(sample.messages) == 2

    def test_deterministic_id(self) -> None:
        messages = [
            ChatMLMessage(role="user", content="Hey"),
            ChatMLMessage(role="assistant", content="Sup"),
        ]
        s1 = create_training_sample(
            messages, ConversationType.PERSONALITY, HumorLevel.WITTY_IRREVERENT, []
        )
        s2 = create_training_sample(
            messages, ConversationType.PERSONALITY, HumorLevel.WITTY_IRREVERENT, []
        )
        assert s1.sample_id == s2.sample_id


# ── Humor Level Mapping ────────────────────────────────────────────────────


class TestHumorLevelMapping:
    def test_get_humor_level_for_index_distribution(self) -> None:
        total = 100
        levels = [get_humor_level_for_index(i, total) for i in range(total)]
        # Should have all humor levels represented
        unique = set(levels)
        assert len(unique) >= 3  # At least 3 different levels

    def test_get_humor_level_single_item(self) -> None:
        level = get_humor_level_for_index(0, 1)
        assert isinstance(level, HumorLevel)

    def test_target_counts(self) -> None:
        counts = get_target_counts(1000)
        assert counts[ConversationType.PERSONALITY] == 600
        assert counts[ConversationType.KNOWLEDGE_QA] == 200
        assert counts[ConversationType.GENERAL] == 200


# ── Prompt Building ─────────────────────────────────────────────────────────


class TestBuildGenerationPrompt:
    def test_personality_prompt_includes_diary(self) -> None:
        from ara.fine_tuning.dataset import _PERSONALITY_GENERATION_PROMPT

        prompt = build_generation_prompt(
            _PERSONALITY_GENERATION_PROMPT,
            HumorLevel.WITTY_IRREVERENT,
            diary_text="Today was great!",
        )
        assert "Today was great!" in prompt
        assert "WITTY IRREVERENT" in prompt

    def test_general_prompt_includes_topic(self) -> None:
        from ara.fine_tuning.dataset import _GENERAL_PROMPT

        prompt = build_generation_prompt(
            _GENERAL_PROMPT,
            HumorLevel.SUPPORTIVE,
            topic="workout plan",
        )
        assert "workout plan" in prompt
        assert "SUPPORTIVE" in prompt


class TestBuildSystemMessageForTraining:
    def test_includes_personality(self) -> None:
        result = build_system_message_for_training(HumorLevel.WITTY_IRREVERENT)
        assert "ARA" in result
        assert "WITTY IRREVERENT" in result

    def test_different_humor_levels(self) -> None:
        supportive = build_system_message_for_training(HumorLevel.SUPPORTIVE)
        grok = build_system_message_for_training(HumorLevel.FULL_GROK)
        assert supportive != grok
        assert "SUPPORTIVE" in supportive
        assert "FULL GROK" in grok


# ── Async Generation (with mocked LLM) ─────────────────────────────────────


class TestGeneratePersonalityConversation:
    @pytest.fixture()
    def segment(self) -> DiarySegment:
        return DiarySegment(
            text="Today I went to the gym for the first time in months. Felt great but also slightly dead.",
            source_file="diary.md",
            date="2024-01-15",
            section_title="",
            token_count=30,
            segment_id="abc123",
        )

    @pytest.fixture()
    def mock_llm(self) -> AsyncMock:
        response = json.dumps(
            [
                {"role": "user", "content": "I finally went to the gym today"},
                {
                    "role": "assistant",
                    "content": "Wait, you actually went? I need to mark this on the calendar.",
                },
            ]
        )
        return AsyncMock(return_value=response)

    @pytest.mark.asyncio
    async def test_generates_sample(self, segment: DiarySegment, mock_llm: AsyncMock) -> None:
        result = await generate_personality_conversation(
            segment, HumorLevel.WITTY_IRREVERENT, mock_llm
        )
        assert result is not None
        assert result.conversation_type == ConversationType.PERSONALITY
        assert result.humor_level == HumorLevel.WITTY_IRREVERENT
        assert len(result.messages) >= 3  # system + 2 turns

    @pytest.mark.asyncio
    async def test_returns_none_on_invalid_response(self, segment: DiarySegment) -> None:
        mock_llm = AsyncMock(return_value="not json")
        result = await generate_personality_conversation(
            segment, HumorLevel.WITTY_IRREVERENT, mock_llm
        )
        assert result is None


class TestGenerateKnowledgeQA:
    @pytest.fixture()
    def segment(self) -> DiarySegment:
        return DiarySegment(
            text="Started learning Rust this week. It's hard but the compiler errors are actually helpful.",
            source_file="diary.md",
            date="2024-02-01",
            section_title="",
            token_count=25,
            segment_id="def456",
        )

    @pytest.mark.asyncio
    async def test_generates_qa(self, segment: DiarySegment) -> None:
        response = json.dumps(
            [
                {"role": "user", "content": "When did I start learning Rust?"},
                {
                    "role": "assistant",
                    "content": "Early February. You were cursing the borrow checker but secretly loving it.",
                },
            ]
        )
        mock_llm = AsyncMock(return_value=response)
        result = await generate_knowledge_qa(segment, HumorLevel.WARM_SARCASTIC, mock_llm)
        assert result is not None
        assert result.conversation_type == ConversationType.KNOWLEDGE_QA


class TestGenerateGeneralConversation:
    @pytest.mark.asyncio
    async def test_generates_general(self) -> None:
        response = json.dumps(
            [
                {"role": "user", "content": "I need a workout plan"},
                {"role": "assistant", "content": "Bold of you to assume I'd let you skip leg day."},
            ]
        )
        mock_llm = AsyncMock(return_value=response)
        result = await generate_general_conversation("workout plan", HumorLevel.FULL_GROK, mock_llm)
        assert result is not None
        assert result.conversation_type == ConversationType.GENERAL


# ── Scoring ─────────────────────────────────────────────────────────────────


class TestScoreSample:
    @pytest.mark.asyncio
    async def test_scores_sample(self) -> None:
        sample = TrainingSample(
            sample_id="test",
            conversation_type=ConversationType.PERSONALITY,
            humor_level=HumorLevel.WITTY_IRREVERENT,
            messages=[
                ChatMLMessage(role="user", content="Hey"),
                ChatMLMessage(role="assistant", content="Sup"),
            ],
            source_segments=[],
            quality_score=0.0,
            token_count=20,
        )
        mock_llm = AsyncMock(return_value='{"score": 4, "reason": "good personality"}')
        score = await score_sample(sample, mock_llm)
        assert score == 4.0

    @pytest.mark.asyncio
    async def test_clamps_score(self) -> None:
        sample = TrainingSample(
            sample_id="test",
            conversation_type=ConversationType.PERSONALITY,
            humor_level=HumorLevel.WITTY_IRREVERENT,
            messages=[
                ChatMLMessage(role="user", content="Hey"),
                ChatMLMessage(role="assistant", content="Sup"),
            ],
            source_segments=[],
            quality_score=0.0,
            token_count=20,
        )
        mock_llm = AsyncMock(return_value='{"score": 10, "reason": "amazing"}')
        score = await score_sample(sample, mock_llm)
        assert score == 5.0

    @pytest.mark.asyncio
    async def test_returns_zero_on_failure(self) -> None:
        sample = TrainingSample(
            sample_id="test",
            conversation_type=ConversationType.PERSONALITY,
            humor_level=HumorLevel.WITTY_IRREVERENT,
            messages=[
                ChatMLMessage(role="user", content="Hey"),
                ChatMLMessage(role="assistant", content="Sup"),
            ],
            source_segments=[],
            quality_score=0.0,
            token_count=20,
        )
        mock_llm = AsyncMock(return_value="not json")
        score = await score_sample(sample, mock_llm)
        assert score == 0.0


# ── Dataset Stats ───────────────────────────────────────────────────────────


class TestComputeStats:
    def test_empty_samples(self) -> None:
        stats = compute_stats([])
        assert stats.total_samples == 0
        assert stats.avg_quality_score == 0.0

    def test_computes_correctly(self) -> None:
        samples = [
            TrainingSample(
                sample_id="a",
                conversation_type=ConversationType.PERSONALITY,
                humor_level=HumorLevel.WITTY_IRREVERENT,
                messages=[],
                source_segments=[],
                quality_score=4.0,
                token_count=100,
            ),
            TrainingSample(
                sample_id="b",
                conversation_type=ConversationType.KNOWLEDGE_QA,
                humor_level=HumorLevel.SUPPORTIVE,
                messages=[],
                source_segments=[],
                quality_score=5.0,
                token_count=200,
            ),
            TrainingSample(
                sample_id="c",
                conversation_type=ConversationType.GENERAL,
                humor_level=HumorLevel.WITTY_IRREVERENT,
                messages=[],
                source_segments=[],
                quality_score=0.0,
                token_count=150,
            ),
        ]
        stats = compute_stats(samples)
        assert stats.total_samples == 3
        assert stats.personality_count == 1
        assert stats.knowledge_qa_count == 1
        assert stats.general_count == 1
        assert stats.scored_count == 2
        assert stats.avg_quality_score == 4.5
        assert stats.total_tokens == 450


# ── Quality Filtering ───────────────────────────────────────────────────────


class TestFilterByQuality:
    def test_keeps_unscored(self) -> None:
        samples = [
            TrainingSample(
                sample_id="a",
                conversation_type=ConversationType.PERSONALITY,
                humor_level=HumorLevel.WITTY_IRREVERENT,
                messages=[],
                source_segments=[],
                quality_score=0.0,
                token_count=100,
            ),
        ]
        kept, filtered = filter_by_quality(samples, min_score=4.0)
        assert len(kept) == 1
        assert filtered == 0

    def test_filters_low_scores(self) -> None:
        samples = [
            TrainingSample(
                sample_id="a",
                conversation_type=ConversationType.PERSONALITY,
                humor_level=HumorLevel.WITTY_IRREVERENT,
                messages=[],
                source_segments=[],
                quality_score=4.5,
                token_count=100,
            ),
            TrainingSample(
                sample_id="b",
                conversation_type=ConversationType.PERSONALITY,
                humor_level=HumorLevel.WITTY_IRREVERENT,
                messages=[],
                source_segments=[],
                quality_score=2.0,
                token_count=100,
            ),
            TrainingSample(
                sample_id="c",
                conversation_type=ConversationType.PERSONALITY,
                humor_level=HumorLevel.WITTY_IRREVERENT,
                messages=[],
                source_segments=[],
                quality_score=3.5,
                token_count=100,
            ),
        ]
        kept, filtered = filter_by_quality(samples, min_score=4.0)
        assert len(kept) == 1
        assert filtered == 2


# ── Dataset Splitting ───────────────────────────────────────────────────────


class TestSplitDataset:
    def _make_samples(self, n: int) -> list[TrainingSample]:
        return [
            TrainingSample(
                sample_id=f"s{i}",
                conversation_type=ConversationType.PERSONALITY,
                humor_level=HumorLevel.WITTY_IRREVERENT,
                messages=[],
                source_segments=[],
                quality_score=4.0,
                token_count=100,
            )
            for i in range(n)
        ]

    def test_split_ratios(self) -> None:
        samples = self._make_samples(100)
        train, val, test = split_dataset(samples)
        assert len(train) == 80
        assert len(val) == 10
        assert len(test) == 10

    def test_custom_ratios(self) -> None:
        samples = self._make_samples(100)
        train, val, test = split_dataset(samples, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2)
        assert len(train) == 60
        assert len(val) == 20
        assert len(test) == 20

    def test_invalid_ratios_raises(self) -> None:
        with pytest.raises(ValueError, match=r"must sum to 1\.0"):
            split_dataset([], train_ratio=0.5, val_ratio=0.1, test_ratio=0.1)

    def test_deterministic(self) -> None:
        samples = self._make_samples(50)
        train1, _, _ = split_dataset(samples, seed=42)
        train2, _, _ = split_dataset(samples, seed=42)
        assert [s.sample_id for s in train1] == [s.sample_id for s in train2]


# ── Save / Load ─────────────────────────────────────────────────────────────


class TestSaveLoadDataset:
    def _make_sample(self) -> TrainingSample:
        return TrainingSample(
            sample_id="test_001",
            conversation_type=ConversationType.PERSONALITY,
            humor_level=HumorLevel.WITTY_IRREVERENT,
            messages=[
                ChatMLMessage(role="system", content="You are ARA"),
                ChatMLMessage(role="user", content="Hey"),
                ChatMLMessage(role="assistant", content="Sup"),
            ],
            source_segments=["seg1"],
            quality_score=4.5,
            token_count=100,
        )

    def test_save_jsonl(self, tmp_path: Path) -> None:
        samples = [self._make_sample()]
        path = tmp_path / "train.jsonl"
        count = save_dataset(samples, path, format="jsonl")
        assert count == 1
        assert path.exists()
        lines = path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert len(data["messages"]) == 3

    def test_save_json(self, tmp_path: Path) -> None:
        samples = [self._make_sample()]
        path = tmp_path / "train.json"
        count = save_dataset(samples, path, format="json")
        assert count == 1
        data = json.loads(path.read_text(encoding="utf-8"))
        assert len(data) == 1
        assert data[0]["sample_id"] == "test_001"

    def test_save_invalid_format(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="Unsupported format"):
            save_dataset([], tmp_path / "out.txt", format="csv")

    def test_load_jsonl(self, tmp_path: Path) -> None:
        samples = [self._make_sample()]
        path = tmp_path / "data.jsonl"
        save_dataset(samples, path, format="jsonl")
        loaded = load_dataset(path)
        assert len(loaded) == 1
        assert len(loaded[0].messages) == 3

    def test_load_json(self, tmp_path: Path) -> None:
        samples = [self._make_sample()]
        path = tmp_path / "data.json"
        save_dataset(samples, path, format="json")
        loaded = load_dataset(path)
        assert len(loaded) == 1
        assert loaded[0].sample_id == "test_001"
        assert loaded[0].quality_score == 4.5

    def test_load_missing_file(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_dataset(Path("/nonexistent/file.jsonl"))

    def test_load_invalid_extension(self, tmp_path: Path) -> None:
        path = tmp_path / "data.csv"
        path.write_text("hello", encoding="utf-8")
        with pytest.raises(ValueError, match="Unsupported file extension"):
            load_dataset(path)

    def test_roundtrip_json(self, tmp_path: Path) -> None:
        original = self._make_sample()
        path = tmp_path / "roundtrip.json"
        save_dataset([original], path, format="json")
        loaded = load_dataset(path)
        assert loaded[0].sample_id == original.sample_id
        assert loaded[0].conversation_type == original.conversation_type
        assert loaded[0].humor_level == original.humor_level
        assert loaded[0].quality_score == original.quality_score


# ── Constants ───────────────────────────────────────────────────────────────


class TestConstants:
    def test_general_topics_not_empty(self) -> None:
        assert len(GENERAL_TOPICS) >= 20

    def test_humor_distribution_sums_to_one(self) -> None:
        total = sum(HUMOR_DISTRIBUTION.values())
        assert abs(total - 1.0) < 0.01

    def test_all_humor_levels_in_distribution(self) -> None:
        for level in HumorLevel:
            assert level in HUMOR_DISTRIBUTION

    def test_conversation_types(self) -> None:
        assert len(ConversationType) == 3

    def test_humor_levels(self) -> None:
        assert len(HumorLevel) == 5
