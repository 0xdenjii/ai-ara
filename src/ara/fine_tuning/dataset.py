"""Fine-tuning dataset preparation for ARA personality injection.

Converts diary entries, personal knowledge, and synthetic conversations into
ChatML multi-turn format suitable for QLoRA training via Unsloth/TRL.

Pipeline:
1. Parse diary files into dated segments (reuses existing chunking infrastructure)
2. Convert segments into 3 types of ChatML conversations:
   a. Personality conversations — user talks, ARA responds in-character with diary context
   b. Knowledge Q&A — user asks about their life, ARA answers from diary memory
   c. General capability — user makes general requests, ARA responds with persona voice
3. LLM-as-judge quality scoring — keep only 4+/5 rated samples
4. Humor-dial calibration — generate variants at different humor levels
5. Dataset split — 60% personality, 20% knowledge, 20% general
6. Export to HuggingFace datasets format for training

All generation uses the local Ollama LLM. No cloud APIs.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import tiktoken

from ara.config.settings import PersonalitySettings
from ara.core.prompts import _get_humor_block, build_system_prompt
from ara.utils.logging import get_logger

if TYPE_CHECKING:
    from pathlib import Path

log = get_logger("dataset")

# Token counter — same encoding used across the project
_ENCODING = tiktoken.get_encoding("cl100k_base")

# ── Constants ────────────────────────────────────────────────────────────────

# Minimum tokens for a diary segment to be useful for training
MIN_SEGMENT_TOKENS = 50
# Maximum tokens for a single training conversation (system + turns)
MAX_CONVERSATION_TOKENS = 4096
# Maximum tokens per individual turn
MAX_TURN_TOKENS = 1024


class ConversationType(StrEnum):
    """Categories of training conversations."""

    PERSONALITY = "personality"  # Casual conversation showing ARA's voice
    KNOWLEDGE_QA = "knowledge_qa"  # Questions about user's life from diary
    GENERAL = "general"  # General requests with ARA persona


class HumorLevel(StrEnum):
    """Humor dial ranges for training data distribution."""

    SUPPORTIVE = "supportive"  # Dial 1-3
    WARM_SARCASTIC = "warm_sarcastic"  # Dial 4-5
    WITTY_IRREVERENT = "witty_irreverent"  # Dial 6-7 (default)
    FULL_GROK = "full_grok"  # Dial 8-9
    UNFILTERED = "unfiltered"  # Dial 10


@dataclass(frozen=True)
class DiarySegment:
    """A parsed segment from a diary file."""

    text: str
    source_file: str
    date: str  # YYYY-MM-DD or empty
    section_title: str
    token_count: int
    segment_id: str  # Deterministic hash

    @staticmethod
    def compute_id(text: str, source_file: str) -> str:
        """Generate deterministic ID from content + source."""
        content = f"{source_file}:{text[:200]}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]


@dataclass
class ChatMLMessage:
    """A single message in ChatML format."""

    role: str  # "system", "user", "assistant"
    content: str


@dataclass
class TrainingSample:
    """A complete training conversation in ChatML format."""

    sample_id: str
    conversation_type: ConversationType
    humor_level: HumorLevel
    messages: list[ChatMLMessage]
    source_segments: list[str]  # Segment IDs used to generate this
    quality_score: float  # 0.0-5.0, set by LLM-as-judge
    token_count: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_chatml_dict(self) -> dict[str, Any]:
        """Convert to the dict format expected by TRL's SFTTrainer."""
        return {
            "messages": [{"role": m.role, "content": m.content} for m in self.messages],
        }

    def to_full_dict(self) -> dict[str, Any]:
        """Convert to full dict with metadata for storage/analysis."""
        return {
            "sample_id": self.sample_id,
            "conversation_type": self.conversation_type.value,
            "humor_level": self.humor_level.value,
            "messages": [{"role": m.role, "content": m.content} for m in self.messages],
            "source_segments": self.source_segments,
            "quality_score": self.quality_score,
            "token_count": self.token_count,
            "metadata": self.metadata,
        }


def count_tokens(text: str) -> int:
    """Count tokens using tiktoken cl100k_base encoding."""
    return len(_ENCODING.encode(text))


def count_conversation_tokens(messages: list[ChatMLMessage]) -> int:
    """Count total tokens across all messages in a conversation."""
    total = 0
    for msg in messages:
        # ChatML overhead per message: ~4 tokens for role markers
        total += count_tokens(msg.content) + 4
    return total


# ── Diary Parsing ────────────────────────────────────────────────────────────

# Date header patterns (multiple formats from diary files)
_DATE_PATTERNS = [
    # ## 2024-01-15 or ## 2024/01/15
    re.compile(
        r"^#{1,3}\s+(\d{4}[-/]\d{1,2}[-/]\d{1,2})(?:\s*[-\u2013\u2014]\s*(.+))?$", re.MULTILINE
    ),
    # ## January 15, 2024
    re.compile(
        r"^#{1,3}\s+((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4})(?:\s*[-\u2013\u2014]\s*(.+))?$",
        re.MULTILINE,
    ),
    # ## 15/01/2024 (DD/MM/YYYY)
    re.compile(r"^#{1,3}\s+(\d{1,2}/\d{1,2}/\d{4})(?:\s*[-\u2013\u2014]\s*(.+))?$", re.MULTILINE),
    # C-style diary format: #include<23/3/2020>
    re.compile(r"^#include<(\d{1,2}/\d{1,2}/\d{4})>", re.MULTILINE),
]

_SECTION_PATTERN = re.compile(r"^(#{2,4})\s+(.+)$", re.MULTILINE)


def parse_diary_file(file_path: Path) -> list[DiarySegment]:
    """Parse a diary markdown file into dated segments.

    Splits on date headers, then on section headers within date blocks.
    Segments smaller than MIN_SEGMENT_TOKENS are merged with adjacent segments.

    Args:
        file_path: Path to the markdown diary file.

    Returns:
        List of DiarySegment objects.
    """
    text = file_path.read_text(encoding="utf-8", errors="replace")
    source = file_path.name

    # Find all date boundaries
    date_positions: list[tuple[int, str, str]] = []  # (position, date, title)
    for pattern in _DATE_PATTERNS:
        for match in pattern.finditer(text):
            date_str = match.group(1)
            title = match.group(2) if match.lastindex and match.lastindex >= 2 else ""
            date_positions.append((match.start(), date_str, title or ""))

    # Sort by position
    date_positions.sort(key=lambda x: x[0])

    if not date_positions:
        # No date headers — treat entire file as one segment
        tokens = count_tokens(text)
        if tokens >= MIN_SEGMENT_TOKENS:
            seg_id = DiarySegment.compute_id(text, source)
            return [
                DiarySegment(
                    text=text.strip(),
                    source_file=source,
                    date="",
                    section_title=file_path.stem,
                    token_count=tokens,
                    segment_id=seg_id,
                )
            ]
        return []

    segments: list[DiarySegment] = []

    for i, (pos, date_str, title) in enumerate(date_positions):
        # Extract text from this date header to the next
        end_pos = date_positions[i + 1][0] if i + 1 < len(date_positions) else len(text)
        block = text[pos:end_pos].strip()

        if not block:
            continue

        # Split block into sub-sections if it's large
        sub_segments = _split_block_into_segments(block, source, date_str, title)
        segments.extend(sub_segments)

    # Filter out tiny segments
    segments = [s for s in segments if s.token_count >= MIN_SEGMENT_TOKENS]

    log.info("parsed_diary_file", file=source, segments=len(segments))
    return segments


def _split_block_into_segments(
    block: str,
    source: str,
    date_str: str,
    title: str,
) -> list[DiarySegment]:
    """Split a date-bounded block into segments respecting token limits.

    If the block is small enough, returns it as a single segment.
    Otherwise, splits at paragraph boundaries.
    """
    tokens = count_tokens(block)

    if tokens <= MAX_TURN_TOKENS:
        seg_id = DiarySegment.compute_id(block, source)
        return [
            DiarySegment(
                text=block,
                source_file=source,
                date=date_str,
                section_title=title,
                token_count=tokens,
                segment_id=seg_id,
            )
        ]

    # Split at double newlines (paragraph boundaries)
    paragraphs = re.split(r"\n\n+", block)
    segments: list[DiarySegment] = []
    current_parts: list[str] = []
    current_tokens = 0

    for para in paragraphs:
        para_tokens = count_tokens(para)

        if current_tokens + para_tokens > MAX_TURN_TOKENS and current_parts:
            # Flush accumulated parts
            combined = "\n\n".join(current_parts)
            seg_id = DiarySegment.compute_id(combined, source)
            segments.append(
                DiarySegment(
                    text=combined,
                    source_file=source,
                    date=date_str,
                    section_title=title,
                    token_count=count_tokens(combined),
                    segment_id=seg_id,
                )
            )
            current_parts = []
            current_tokens = 0

        current_parts.append(para)
        current_tokens += para_tokens

    # Flush remaining
    if current_parts:
        combined = "\n\n".join(current_parts)
        seg_id = DiarySegment.compute_id(combined, source)
        segments.append(
            DiarySegment(
                text=combined,
                source_file=source,
                date=date_str,
                section_title=title,
                token_count=count_tokens(combined),
                segment_id=seg_id,
            )
        )

    return segments


def scan_diary_directory(diary_dir: Path) -> list[DiarySegment]:
    """Scan all markdown files in a diary directory recursively.

    Args:
        diary_dir: Root directory containing diary files.

    Returns:
        All parsed segments sorted by date (empty dates last).
    """
    all_segments: list[DiarySegment] = []
    md_files = sorted(diary_dir.rglob("*.md"))

    for md_file in md_files:
        # Skip READMEs, index files, and non-diary content
        name_lower = md_file.name.lower()
        if name_lower in ("readme.md", "index.md", "_index.md"):
            continue
        # Skip files that are likely project docs (Work gets low priority)
        if "work" in str(md_file).lower():
            continue

        segments = parse_diary_file(md_file)
        all_segments.extend(segments)

    # Sort: segments with dates first (chronological), dateless last
    all_segments.sort(key=lambda s: (s.date == "", s.date))

    log.info("scanned_diary_directory", dir=str(diary_dir), total_segments=len(all_segments))
    return all_segments


# ── Prompt Templates for Synthetic Data Generation ────────────────────────────

_PERSONALITY_GENERATION_PROMPT = """You are generating training data for a personal AI assistant called ARA.
ARA is a sarcastic, warm, deeply loyal best friend who knows the user's life from diary entries.
ARA uses she/her pronouns. She never says "As an AI..." or breaks the friend illusion.

Here is a diary entry from the user:
---
{diary_text}
---

Generate a natural multi-turn conversation (2-4 turns) where the user casually brings up something
related to this diary content, and ARA responds in-character.

{humor_instruction}

Rules:
- ARA references the diary content naturally, like a friend who remembers
- ARA uses contractions, slang, sentence fragments — real person energy
- ARA may tease, joke, encourage, or call out patterns — depending on humor level
- The user's messages should feel natural and casual (not like test prompts)
- Do NOT include system messages. Only user and assistant turns.
- Each message should be 1-3 sentences. Keep it punchy.
- Do NOT use emojis unless they feel natural for the context

Output format — output ONLY valid JSON, no markdown:
[
  {{"role": "user", "content": "..."}},
  {{"role": "assistant", "content": "..."}},
  ...
]"""


_KNOWLEDGE_QA_PROMPT = """You are generating training data for a personal AI assistant called ARA.
ARA knows the user's entire life from years of diary entries. She's like a best friend with perfect memory.

Here is diary context about the user:
---
{diary_text}
---

Generate a multi-turn Q&A conversation (2-4 turns) where the user asks about their own life/past,
and ARA answers from memory — warmly, specifically, like a friend recalling shared history.

{humor_instruction}

Rules:
- User asks things like "what did I say about...", "when was the last time I...", "remember when..."
- ARA answers with specific details from the diary, using "remember when..." not "according to my records..."
- ARA may add her own commentary, teasing, or encouragement based on the humor level
- Keep each message 1-3 sentences
- Output ONLY valid JSON, no markdown:
[
  {{"role": "user", "content": "..."}},
  {{"role": "assistant", "content": "..."}},
  ...
]"""


_GENERAL_PROMPT = """You are generating training data for a personal AI assistant called ARA.
ARA is a sarcastic, warm best friend. She's witty, pop-culture fluent, and never robotic.

Generate a multi-turn conversation (2-4 turns) where the user makes a general request or has a casual
conversation, and ARA responds with her distinctive personality.

Topic: {topic}

{humor_instruction}

Rules:
- ARA is helpful but NEVER corporate or robotic
- She uses contractions, slang, real-person energy
- She may joke, tease, or be irreverent depending on humor level
- Keep each message 1-3 sentences
- Output ONLY valid JSON, no markdown:
[
  {{"role": "user", "content": "..."}},
  {{"role": "assistant", "content": "..."}},
  ...
]"""


_JUDGE_PROMPT = """Rate this training conversation for a personal AI assistant called ARA.
ARA should be sarcastic, warm, witty, and feel like a real best friend — never robotic.

Conversation:
{conversation}

Rate on a scale of 1-5:
1 = Terrible (robotic, generic, breaks character)
2 = Poor (mostly bland, occasional personality)
3 = Decent (has personality but inconsistent or forced)
4 = Good (natural personality, feels like a real friend)
5 = Excellent (perfect ARA voice — sharp, warm, memorable)

Consider:
- Does ARA sound like a real person (contractions, slang, fragments)?
- Is the humor level appropriate and consistent?
- Does ARA avoid "As an AI..." or corporate speak?
- Are the responses memorable and specific (not generic)?
- Does the conversation flow naturally?

Output ONLY a JSON object:
{{"score": <1-5>, "reason": "<1 sentence explanation>"}}"""


# General topics for capability training data
GENERAL_TOPICS = [
    "helping with a workout plan",
    "discussing a movie recommendation",
    "morning routine check-in",
    "helping write an email",
    "cooking recipe suggestion",
    "late night existential conversation",
    "planning a weekend trip",
    "debugging a coding problem",
    "relationship advice",
    "music recommendation",
    "complaining about weather",
    "celebrating a small win",
    "dealing with procrastination",
    "discussing a book",
    "helping with a presentation",
    "3am random thoughts",
    "dealing with a bad day",
    "financial planning question",
    "health and fitness check-in",
    "career advice discussion",
    "reminiscing about the past",
    "planning goals for the month",
    "tech news discussion",
    "dealing with impostor syndrome",
    "creative writing help",
    "gaming discussion",
    "travel stories and plans",
    "self-improvement habits",
    "discussing crypto/web3",
    "startup idea brainstorming",
]


def _humor_level_to_dial(level: HumorLevel) -> int:
    """Convert a HumorLevel enum to its representative dial value."""
    mapping = {
        HumorLevel.SUPPORTIVE: 2,
        HumorLevel.WARM_SARCASTIC: 5,
        HumorLevel.WITTY_IRREVERENT: 7,
        HumorLevel.FULL_GROK: 9,
        HumorLevel.UNFILTERED: 10,
    }
    return mapping[level]


def _dial_to_humor_level(dial: int) -> HumorLevel:
    """Convert a humor dial value (1-10) to a HumorLevel enum."""
    if dial <= 3:
        return HumorLevel.SUPPORTIVE
    if dial <= 5:
        return HumorLevel.WARM_SARCASTIC
    if dial <= 7:
        return HumorLevel.WITTY_IRREVERENT
    if dial <= 9:
        return HumorLevel.FULL_GROK
    return HumorLevel.UNFILTERED


def build_generation_prompt(
    prompt_template: str,
    humor_level: HumorLevel,
    *,
    diary_text: str = "",
    topic: str = "",
) -> str:
    """Build a prompt for synthetic conversation generation.

    Args:
        prompt_template: One of the _*_PROMPT templates.
        humor_level: Target humor calibration for the generated conversation.
        diary_text: Diary segment text (for personality/knowledge prompts).
        topic: Topic string (for general prompts).

    Returns:
        Formatted prompt string.
    """
    dial = _humor_level_to_dial(humor_level)
    humor_block = _get_humor_block(dial)
    humor_instruction = f"Target humor calibration:\n{humor_block}"

    return prompt_template.format(
        diary_text=diary_text,
        humor_instruction=humor_instruction,
        topic=topic,
    )


def build_system_message_for_training(
    humor_level: HumorLevel,
    *,
    memory_context: str = "",
) -> str:
    """Build a system prompt for inclusion in training samples.

    Uses the same prompt assembly as production but with a fixed personality config.
    """
    dial = _humor_level_to_dial(humor_level)
    personality = PersonalitySettings(humor_dial=dial)
    return build_system_prompt(personality, memory_context=memory_context)


def parse_llm_json_response(response: str) -> Any:
    """Parse a JSON response from the LLM, handling common formatting issues.

    LLMs sometimes wrap JSON in markdown code blocks or add extra text.
    This function strips those artifacts and parses the JSON.

    Args:
        response: Raw LLM response string.

    Returns:
        Parsed JSON object.

    Raises:
        ValueError: If the response cannot be parsed as valid JSON.
    """
    cleaned = response.strip()

    # Strip markdown code blocks
    if cleaned.startswith("```"):
        # Remove opening fence (possibly with language tag)
        first_newline = cleaned.index("\n")
        cleaned = cleaned[first_newline + 1 :]
        # Remove closing fence
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

    # Try parsing directly
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Try extracting JSON array or object from surrounding text
    for start_char, end_char in [("[", "]"), ("{", "}")]:
        start_idx = cleaned.find(start_char)
        end_idx = cleaned.rfind(end_char)
        if start_idx != -1 and end_idx > start_idx:
            candidate = cleaned[start_idx : end_idx + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue

    msg = f"Failed to parse LLM response as JSON: {cleaned[:200]}..."
    raise ValueError(msg)


def validate_conversation_messages(
    raw_messages: list[dict[str, str]],
) -> list[ChatMLMessage] | None:
    """Validate and convert raw message dicts from LLM output.

    Args:
        raw_messages: List of {"role": "...", "content": "..."} dicts.

    Returns:
        List of ChatMLMessage objects if valid, None if invalid.
    """
    if not raw_messages or len(raw_messages) < 2:
        return None

    valid_roles = {"user", "assistant"}
    messages: list[ChatMLMessage] = []

    for msg in raw_messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role not in valid_roles:
            return None
        if not content or not content.strip():
            return None
        if count_tokens(content) > MAX_TURN_TOKENS:
            return None

        messages.append(ChatMLMessage(role=role, content=content.strip()))

    # Must start with user and alternate
    if messages[0].role != "user":
        return None

    # Check alternation — consecutive assistant messages are invalid
    for i in range(1, len(messages)):
        if messages[i].role == messages[i - 1].role and messages[i].role == "assistant":
            return None

    return messages


def create_training_sample(
    messages: list[ChatMLMessage],
    conversation_type: ConversationType,
    humor_level: HumorLevel,
    source_segment_ids: list[str],
    *,
    include_system: bool = True,
    memory_context: str = "",
    quality_score: float = 0.0,
) -> TrainingSample:
    """Create a complete TrainingSample with system message prepended.

    Args:
        messages: User/assistant conversation turns.
        conversation_type: Type of training data.
        humor_level: Humor calibration level.
        source_segment_ids: IDs of diary segments used.
        include_system: Whether to prepend the system prompt.
        memory_context: Optional memory context for system prompt.
        quality_score: Quality rating (0-5).

    Returns:
        Complete TrainingSample ready for training.
    """
    all_messages: list[ChatMLMessage] = []

    if include_system:
        system_content = build_system_message_for_training(
            humor_level,
            memory_context=memory_context,
        )
        all_messages.append(ChatMLMessage(role="system", content=system_content))

    all_messages.extend(messages)

    total_tokens = count_conversation_tokens(all_messages)

    # Generate deterministic sample ID
    content_hash = hashlib.sha256(
        json.dumps([m.content for m in all_messages]).encode()
    ).hexdigest()[:16]
    sample_id = f"{conversation_type.value}_{humor_level.value}_{content_hash}"

    return TrainingSample(
        sample_id=sample_id,
        conversation_type=conversation_type,
        humor_level=humor_level,
        messages=all_messages,
        source_segments=source_segment_ids,
        quality_score=quality_score,
        token_count=total_tokens,
    )


# ── Async Generation Functions (use LLM) ────────────────────────────────────


async def generate_personality_conversation(
    segment: DiarySegment,
    humor_level: HumorLevel,
    llm_generate_fn: Any,
) -> TrainingSample | None:
    """Generate a personality training conversation from a diary segment.

    Uses the LLM to create a natural multi-turn conversation where ARA
    references the diary content in-character.

    Args:
        segment: Diary segment to base the conversation on.
        humor_level: Target humor level for ARA's responses.
        llm_generate_fn: Async callable that takes a prompt string and returns a response string.

    Returns:
        TrainingSample if generation succeeded, None if it failed.
    """
    prompt = build_generation_prompt(
        _PERSONALITY_GENERATION_PROMPT,
        humor_level,
        diary_text=segment.text,
    )

    try:
        response = await llm_generate_fn(prompt)
        raw_messages = parse_llm_json_response(response)
        messages = validate_conversation_messages(raw_messages)
        if messages is None:
            log.warning("invalid_personality_conversation", segment_id=segment.segment_id)
            return None

        return create_training_sample(
            messages,
            ConversationType.PERSONALITY,
            humor_level,
            [segment.segment_id],
            memory_context=segment.text[:500],
        )
    except (ValueError, KeyError, TypeError) as exc:
        log.warning("personality_generation_failed", error=str(exc), segment_id=segment.segment_id)
        return None


async def generate_knowledge_qa(
    segment: DiarySegment,
    humor_level: HumorLevel,
    llm_generate_fn: Any,
) -> TrainingSample | None:
    """Generate a knowledge Q&A training conversation from a diary segment.

    Creates a conversation where the user asks about their own life
    and ARA answers from diary memory.

    Args:
        segment: Diary segment to base the Q&A on.
        humor_level: Target humor level.
        llm_generate_fn: Async callable(prompt) -> response.

    Returns:
        TrainingSample if generation succeeded, None if failed.
    """
    prompt = build_generation_prompt(
        _KNOWLEDGE_QA_PROMPT,
        humor_level,
        diary_text=segment.text,
    )

    try:
        response = await llm_generate_fn(prompt)
        raw_messages = parse_llm_json_response(response)
        messages = validate_conversation_messages(raw_messages)
        if messages is None:
            log.warning("invalid_knowledge_qa", segment_id=segment.segment_id)
            return None

        return create_training_sample(
            messages,
            ConversationType.KNOWLEDGE_QA,
            humor_level,
            [segment.segment_id],
            memory_context=segment.text[:500],
        )
    except (ValueError, KeyError, TypeError) as exc:
        log.warning("knowledge_qa_generation_failed", error=str(exc), segment_id=segment.segment_id)
        return None


async def generate_general_conversation(
    topic: str,
    humor_level: HumorLevel,
    llm_generate_fn: Any,
) -> TrainingSample | None:
    """Generate a general capability training conversation.

    Creates a conversation on a given topic with ARA's personality.

    Args:
        topic: Conversation topic/scenario.
        humor_level: Target humor level.
        llm_generate_fn: Async callable(prompt) -> response.

    Returns:
        TrainingSample if generation succeeded, None if failed.
    """
    prompt = build_generation_prompt(
        _GENERAL_PROMPT,
        humor_level,
        topic=topic,
    )

    try:
        response = await llm_generate_fn(prompt)
        raw_messages = parse_llm_json_response(response)
        messages = validate_conversation_messages(raw_messages)
        if messages is None:
            log.warning("invalid_general_conversation", topic=topic)
            return None

        return create_training_sample(
            messages,
            ConversationType.GENERAL,
            humor_level,
            [],
        )
    except (ValueError, KeyError, TypeError) as exc:
        log.warning("general_generation_failed", error=str(exc), topic=topic)
        return None


async def score_sample(
    sample: TrainingSample,
    llm_generate_fn: Any,
) -> float:
    """Score a training sample using LLM-as-judge.

    Args:
        sample: Training sample to score.
        llm_generate_fn: Async callable(prompt) -> response.

    Returns:
        Quality score (1.0-5.0). Returns 0.0 on failure.
    """
    # Format conversation for the judge
    conv_text = "\n".join(f"{m.role}: {m.content}" for m in sample.messages if m.role != "system")

    prompt = _JUDGE_PROMPT.format(conversation=conv_text)

    try:
        response = await llm_generate_fn(prompt)
        result = parse_llm_json_response(response)
        score = float(result.get("score", 0))
        reason = result.get("reason", "")
        log.debug("sample_scored", sample_id=sample.sample_id, score=score, reason=reason)
        return max(1.0, min(5.0, score))
    except (ValueError, KeyError, TypeError) as exc:
        log.warning("scoring_failed", error=str(exc), sample_id=sample.sample_id)
        return 0.0


# ── Dataset Management ───────────────────────────────────────────────────────


@dataclass
class DatasetStats:
    """Statistics about a generated dataset."""

    total_samples: int = 0
    personality_count: int = 0
    knowledge_qa_count: int = 0
    general_count: int = 0
    by_humor_level: dict[str, int] = field(default_factory=dict)
    avg_quality_score: float = 0.0
    avg_token_count: float = 0.0
    total_tokens: int = 0
    scored_count: int = 0
    filtered_count: int = 0  # Samples removed by quality filter


def compute_stats(samples: list[TrainingSample]) -> DatasetStats:
    """Compute aggregate statistics for a list of training samples."""
    stats = DatasetStats()
    stats.total_samples = len(samples)

    if not samples:
        return stats

    quality_sum = 0.0
    scored = 0

    for s in samples:
        if s.conversation_type == ConversationType.PERSONALITY:
            stats.personality_count += 1
        elif s.conversation_type == ConversationType.KNOWLEDGE_QA:
            stats.knowledge_qa_count += 1
        else:
            stats.general_count += 1

        level_key = s.humor_level.value
        stats.by_humor_level[level_key] = stats.by_humor_level.get(level_key, 0) + 1
        stats.total_tokens += s.token_count

        if s.quality_score > 0:
            quality_sum += s.quality_score
            scored += 1

    stats.scored_count = scored
    stats.avg_quality_score = quality_sum / scored if scored > 0 else 0.0
    stats.avg_token_count = stats.total_tokens / stats.total_samples
    return stats


def filter_by_quality(
    samples: list[TrainingSample],
    min_score: float = 4.0,
) -> tuple[list[TrainingSample], int]:
    """Filter samples by quality score.

    Samples with score 0.0 (unscored) are kept.
    Samples with score >= min_score are kept.
    Everything else is filtered out.

    Args:
        samples: Training samples to filter.
        min_score: Minimum quality score to keep.

    Returns:
        Tuple of (kept samples, number filtered out).
    """
    kept: list[TrainingSample] = []
    filtered = 0
    for s in samples:
        if s.quality_score == 0.0 or s.quality_score >= min_score:
            kept.append(s)
        else:
            filtered += 1
    return kept, filtered


def split_dataset(
    samples: list[TrainingSample],
    *,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[TrainingSample], list[TrainingSample], list[TrainingSample]]:
    """Split samples into train/validation/test sets.

    Uses deterministic shuffling based on sample_id hash for reproducibility.

    Args:
        samples: All training samples.
        train_ratio: Fraction for training (default 0.8).
        val_ratio: Fraction for validation (default 0.1).
        test_ratio: Fraction for test (default 0.1).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train, validation, test) sample lists.
    """
    import random

    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.01:
        msg = f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"
        raise ValueError(msg)

    # Deterministic shuffle
    rng = random.Random(seed)
    shuffled = list(samples)
    rng.shuffle(shuffled)

    n = len(shuffled)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train = shuffled[:train_end]
    val = shuffled[train_end:val_end]
    test = shuffled[val_end:]

    return train, val, test


def save_dataset(
    samples: list[TrainingSample],
    output_path: Path,
    *,
    format: str = "jsonl",
) -> int:
    """Save training samples to disk.

    Args:
        samples: Training samples to save.
        output_path: File path to write to.
        format: Output format — "jsonl" (one JSON per line) or "json" (single array).

    Returns:
        Number of samples written.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == "jsonl":
        with output_path.open("w", encoding="utf-8") as f:
            for sample in samples:
                line = json.dumps(sample.to_chatml_dict(), ensure_ascii=False)
                f.write(line + "\n")
    elif format == "json":
        data = [sample.to_full_dict() for sample in samples]
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    else:
        msg = f"Unsupported format: {format}. Use 'jsonl' or 'json'."
        raise ValueError(msg)

    log.info("dataset_saved", path=str(output_path), samples=len(samples), format=format)
    return len(samples)


def load_dataset(input_path: Path) -> list[TrainingSample]:
    """Load training samples from a JSON or JSONL file.

    Args:
        input_path: Path to the dataset file.

    Returns:
        List of TrainingSample objects.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If the file format is invalid.
    """
    if not input_path.exists():
        msg = f"Dataset file not found: {input_path}"
        raise FileNotFoundError(msg)

    samples: list[TrainingSample] = []

    if input_path.suffix == ".jsonl":
        with input_path.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    sample = _dict_to_sample(data, f"line_{line_num}")
                    samples.append(sample)
                except (json.JSONDecodeError, KeyError) as exc:
                    log.warning("invalid_sample_line", line=line_num, error=str(exc))
    elif input_path.suffix == ".json":
        with input_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            for i, item in enumerate(data):
                try:
                    sample = _dict_to_sample(item, f"item_{i}")
                    samples.append(sample)
                except KeyError as exc:
                    log.warning("invalid_sample_item", index=i, error=str(exc))
    else:
        msg = f"Unsupported file extension: {input_path.suffix}. Use .jsonl or .json."
        raise ValueError(msg)

    log.info("dataset_loaded", path=str(input_path), samples=len(samples))
    return samples


def _dict_to_sample(data: dict[str, Any], fallback_id: str) -> TrainingSample:
    """Convert a dict to a TrainingSample.

    Handles both the full format (from .json) and the minimal ChatML format (from .jsonl).
    """
    messages_data = data.get("messages", [])
    messages = [ChatMLMessage(role=m["role"], content=m["content"]) for m in messages_data]

    return TrainingSample(
        sample_id=data.get("sample_id", fallback_id),
        conversation_type=ConversationType(data.get("conversation_type", "personality")),
        humor_level=HumorLevel(data.get("humor_level", "witty_irreverent")),
        messages=messages,
        source_segments=data.get("source_segments", []),
        quality_score=float(data.get("quality_score", 0.0)),
        token_count=data.get("token_count", count_conversation_tokens(messages)),
        metadata=data.get("metadata", {}),
    )


# ── Humor Distribution ──────────────────────────────────────────────────────

# Target distribution for humor levels in training data (per projectmanagement.md)
# 500 samples at supportive (3-4), 1000 at witty (6-7), 500 at grok/unfiltered (8-10)
# Plus warm_sarcastic (4-5) fills the gap
HUMOR_DISTRIBUTION: dict[HumorLevel, float] = {
    HumorLevel.SUPPORTIVE: 0.15,  # ~15% — low humor
    HumorLevel.WARM_SARCASTIC: 0.20,  # ~20% — medium-low
    HumorLevel.WITTY_IRREVERENT: 0.35,  # ~35% — default, most common
    HumorLevel.FULL_GROK: 0.20,  # ~20% — high
    HumorLevel.UNFILTERED: 0.10,  # ~10% — extreme
}


def get_humor_level_for_index(index: int, total: int) -> HumorLevel:
    """Deterministically assign a humor level based on position in the dataset.

    Distributes samples according to HUMOR_DISTRIBUTION ratios.

    Args:
        index: Sample index (0-based).
        total: Total number of samples being generated.

    Returns:
        HumorLevel for this sample.
    """
    position = index / max(total, 1)
    cumulative = 0.0
    for level, ratio in HUMOR_DISTRIBUTION.items():
        cumulative += ratio
        if position < cumulative:
            return level
    return HumorLevel.WITTY_IRREVERENT  # Fallback


def get_target_counts(
    total_samples: int,
) -> dict[ConversationType, int]:
    """Calculate target counts per conversation type.

    Split: 60% personality, 20% knowledge Q&A, 20% general.

    Args:
        total_samples: Total number of training samples desired.

    Returns:
        Dict mapping ConversationType to target count.
    """
    return {
        ConversationType.PERSONALITY: int(total_samples * 0.60),
        ConversationType.KNOWLEDGE_QA: int(total_samples * 0.20),
        ConversationType.GENERAL: int(total_samples * 0.20),
    }
