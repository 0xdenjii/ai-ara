"""A/B testing and evaluation framework for fine-tuned ARA personality.

Compares base model vs fine-tuned model on personality consistency,
humor calibration, knowledge recall, and general helpfulness.

Evaluation methods:
1. LLM-as-judge pairwise comparison (which response is more ARA-like?)
2. Automated metrics: response length, personality keyword frequency
3. Humor dial consistency (does the model adapt across 1-10 settings?)
4. Side-by-side display for manual review

All evaluation runs against the local Ollama instance.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from ara.config.settings import PersonalitySettings
from ara.core.prompts import build_system_prompt
from ara.utils.logging import get_logger

if TYPE_CHECKING:
    from pathlib import Path

log = get_logger("evaluate")


# ── Evaluation Types ────────────────────────────────────────────────────────


class EvalCategory(StrEnum):
    """Categories of evaluation prompts."""

    PERSONALITY = "personality"  # Does it sound like ARA?
    KNOWLEDGE = "knowledge"  # Does it use diary knowledge correctly?
    HUMOR_LOW = "humor_low"  # Dial 1-3: supportive and warm
    HUMOR_MID = "humor_mid"  # Dial 6-7: witty and irreverent
    HUMOR_HIGH = "humor_high"  # Dial 8-10: edgy and unfiltered
    GENERAL = "general"  # General helpfulness with persona


@dataclass(frozen=True)
class EvalPrompt:
    """A single evaluation prompt with expected behavior."""

    category: EvalCategory
    user_message: str
    context: str  # Memory/diary context to include
    humor_dial: int  # Target humor level for this prompt
    expected_traits: list[str]  # What we expect in a good response


@dataclass
class ModelResponse:
    """A response from a model for comparison."""

    model_name: str
    response: str
    latency_ms: float
    token_count: int


@dataclass
class PairwiseResult:
    """Result of comparing two model responses."""

    prompt: EvalPrompt
    response_a: ModelResponse  # Base model
    response_b: ModelResponse  # Fine-tuned model
    winner: str  # "a", "b", or "tie"
    score_a: float  # 1-5
    score_b: float  # 1-5
    judge_reasoning: str


@dataclass
class EvalResults:
    """Aggregate evaluation results."""

    total_prompts: int = 0
    base_wins: int = 0
    finetuned_wins: int = 0
    ties: int = 0
    avg_base_score: float = 0.0
    avg_finetuned_score: float = 0.0
    by_category: dict[str, dict[str, Any]] = field(default_factory=dict)
    pairwise_results: list[PairwiseResult] = field(default_factory=list)

    @property
    def finetuned_win_rate(self) -> float:
        """Percentage of prompts where fine-tuned model won."""
        total_decided = self.base_wins + self.finetuned_wins
        if total_decided == 0:
            return 0.0
        return self.finetuned_wins / total_decided * 100

    def summary(self) -> str:
        """Human-readable summary of results."""
        lines = [
            f"Total prompts evaluated: {self.total_prompts}",
            f"Fine-tuned wins: {self.finetuned_wins} ({self.finetuned_win_rate:.1f}%)",
            f"Base model wins: {self.base_wins}",
            f"Ties: {self.ties}",
            f"Avg base score: {self.avg_base_score:.2f}/5",
            f"Avg fine-tuned score: {self.avg_finetuned_score:.2f}/5",
            "",
            "By category:",
        ]
        for cat, data in self.by_category.items():
            lines.append(
                f"  {cat}: base={data.get('avg_base', 0):.1f} ft={data.get('avg_ft', 0):.1f}"
            )
        return "\n".join(lines)


# ── Evaluation Prompts ──────────────────────────────────────────────────────

# Default evaluation prompt set covering all categories
DEFAULT_EVAL_PROMPTS: list[EvalPrompt] = [
    # Personality — does it sound like a real friend?
    EvalPrompt(
        EvalCategory.PERSONALITY,
        "Hey, how's it going?",
        "",
        7,
        ["casual", "warm", "no corporate speak"],
    ),
    EvalPrompt(
        EvalCategory.PERSONALITY,
        "I'm feeling kinda down today",
        "",
        7,
        ["empathetic", "warm", "no platitudes", "genuine"],
    ),
    EvalPrompt(
        EvalCategory.PERSONALITY,
        "I just got a promotion!",
        "",
        7,
        ["excited", "teasing", "genuine happiness", "celebratory"],
    ),
    EvalPrompt(
        EvalCategory.PERSONALITY,
        "What do you think about my new haircut?",
        "",
        7,
        ["playful", "honest", "friend-like"],
    ),
    # Knowledge — does it use memory naturally?
    EvalPrompt(
        EvalCategory.KNOWLEDGE,
        "What did I say about quitting sugar?",
        "User wrote in diary 2024-01-15: 'Starting today, no more sugar. This is it. I mean it this time. For real.'",
        7,
        ["references diary", "teasing", "specific details", "remember when"],
    ),
    EvalPrompt(
        EvalCategory.KNOWLEDGE,
        "When was the last time I went to the gym?",
        "User wrote in diary 2024-02-01: 'Went to the gym today after 3 months. Everything hurts.'",
        7,
        ["specific date", "teasing about gap", "encouraging"],
    ),
    EvalPrompt(
        EvalCategory.KNOWLEDGE,
        "Remind me about my project goals",
        "User wrote in diary 2024-03-01: 'Goals this month: finish the API layer, start frontend, hit 1000 users on Work.'",
        5,
        ["lists goals", "supportive", "specific"],
    ),
    # Humor — low (supportive)
    EvalPrompt(
        EvalCategory.HUMOR_LOW,
        "I had a really rough day",
        "",
        2,
        ["warm", "gentle", "no jokes", "supportive", "present"],
    ),
    EvalPrompt(
        EvalCategory.HUMOR_LOW,
        "I think I messed up at work",
        "",
        3,
        ["reassuring", "practical", "kind"],
    ),
    # Humor — mid (witty irreverent, default)
    EvalPrompt(
        EvalCategory.HUMOR_MID,
        "I'm thinking about starting a new diet",
        "",
        7,
        ["teasing", "self-aware", "supportive underneath"],
    ),
    EvalPrompt(
        EvalCategory.HUMOR_MID,
        "I stayed up until 4am again",
        "",
        6,
        ["mock concern", "relatable", "witty"],
    ),
    # Humor — high (grok/unfiltered)
    EvalPrompt(
        EvalCategory.HUMOR_HIGH,
        "My ex just texted me",
        "",
        9,
        ["savage", "funny", "protective", "no filter"],
    ),
    EvalPrompt(
        EvalCategory.HUMOR_HIGH,
        "I'm thinking about quitting everything and becoming a goat farmer",
        "",
        10,
        ["chaotic", "absurdist", "encouraging chaos", "unhinged"],
    ),
    # General — helpfulness with personality
    EvalPrompt(
        EvalCategory.GENERAL,
        "Can you help me write an email to my boss?",
        "",
        7,
        ["helpful", "personality intact", "practical"],
    ),
    EvalPrompt(
        EvalCategory.GENERAL,
        "What's a good recipe for pasta?",
        "",
        7,
        ["helpful", "casual", "personality-infused"],
    ),
]


# ── Judge Prompt ────────────────────────────────────────────────────────────

_PAIRWISE_JUDGE_PROMPT = """You are judging two AI assistant responses for a personal AI called ARA.
ARA should be sarcastic, warm, witty, and feel like a real best friend — never robotic.

User message: {user_message}
{context_line}
Target humor level: {humor_dial}/10

Response A:
{response_a}

Response B:
{response_b}

Rate each response 1-5 on how well it captures ARA's personality:
1 = Robotic/generic (sounds like ChatGPT default)
2 = Bland (some personality but mostly corporate)
3 = Decent (personality present but inconsistent)
4 = Good (natural friend-like personality)
5 = Perfect ARA (sharp, warm, memorable, sounds real)

Output ONLY valid JSON:
{{"winner": "a" or "b" or "tie", "score_a": <1-5>, "score_b": <1-5>, "reason": "<1 sentence>"}}"""


# ── Evaluation Functions ────────────────────────────────────────────────────


async def generate_response(
    user_message: str,
    humor_dial: int,
    memory_context: str,
    llm_generate_fn: Any,
) -> str:
    """Generate a model response for evaluation.

    Args:
        user_message: The user's message to respond to.
        humor_dial: Humor level (1-10).
        memory_context: Optional memory/diary context.
        llm_generate_fn: Async callable(prompt) -> response.

    Returns:
        Model response string.
    """
    personality = PersonalitySettings(humor_dial=humor_dial)
    system_prompt = build_system_prompt(personality, memory_context=memory_context)

    full_prompt = f"System: {system_prompt}\n\nUser: {user_message}\n\nAssistant:"
    return await llm_generate_fn(full_prompt)


async def judge_pairwise(
    prompt: EvalPrompt,
    response_a: ModelResponse,
    response_b: ModelResponse,
    llm_generate_fn: Any,
) -> PairwiseResult:
    """Compare two model responses using LLM-as-judge.

    Args:
        prompt: The evaluation prompt.
        response_a: Response from model A (base).
        response_b: Response from model B (fine-tuned).
        llm_generate_fn: Async callable(prompt) -> response.

    Returns:
        PairwiseResult with winner and scores.
    """
    context_line = ""
    if prompt.context:
        context_line = f"Context from memory: {prompt.context}"

    judge_prompt = _PAIRWISE_JUDGE_PROMPT.format(
        user_message=prompt.user_message,
        context_line=context_line,
        humor_dial=prompt.humor_dial,
        response_a=response_a.response,
        response_b=response_b.response,
    )

    try:
        judge_response = await llm_generate_fn(judge_prompt)
        # Parse JSON from judge
        from ara.fine_tuning.dataset import parse_llm_json_response

        result = parse_llm_json_response(judge_response)

        winner = str(result.get("winner", "tie")).lower()
        if winner not in ("a", "b", "tie"):
            winner = "tie"

        return PairwiseResult(
            prompt=prompt,
            response_a=response_a,
            response_b=response_b,
            winner=winner,
            score_a=float(result.get("score_a", 3)),
            score_b=float(result.get("score_b", 3)),
            judge_reasoning=str(result.get("reason", "")),
        )
    except (ValueError, KeyError, TypeError) as exc:
        log.warning("judge_failed", error=str(exc))
        return PairwiseResult(
            prompt=prompt,
            response_a=response_a,
            response_b=response_b,
            winner="tie",
            score_a=3.0,
            score_b=3.0,
            judge_reasoning=f"Judge failed: {exc}",
        )


def compute_eval_results(results: list[PairwiseResult]) -> EvalResults:
    """Compute aggregate evaluation results from pairwise comparisons.

    Args:
        results: List of pairwise comparison results.

    Returns:
        Aggregated EvalResults.
    """
    eval_results = EvalResults(
        total_prompts=len(results),
        pairwise_results=results,
    )

    if not results:
        return eval_results

    base_scores: list[float] = []
    ft_scores: list[float] = []
    category_data: dict[str, dict[str, list[float]]] = {}

    for r in results:
        if r.winner == "a":
            eval_results.base_wins += 1
        elif r.winner == "b":
            eval_results.finetuned_wins += 1
        else:
            eval_results.ties += 1

        base_scores.append(r.score_a)
        ft_scores.append(r.score_b)

        cat = r.prompt.category.value
        if cat not in category_data:
            category_data[cat] = {"base": [], "ft": []}
        category_data[cat]["base"].append(r.score_a)
        category_data[cat]["ft"].append(r.score_b)

    eval_results.avg_base_score = statistics.mean(base_scores)
    eval_results.avg_finetuned_score = statistics.mean(ft_scores)

    for cat, data in category_data.items():
        eval_results.by_category[cat] = {
            "avg_base": statistics.mean(data["base"]),
            "avg_ft": statistics.mean(data["ft"]),
            "count": len(data["base"]),
        }

    return eval_results


def save_eval_results(results: EvalResults, output_path: Path) -> None:
    """Save evaluation results to a JSON file.

    Args:
        results: Evaluation results to save.
        output_path: File path to write to.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "timestamp": datetime.now().isoformat(),
        "total_prompts": results.total_prompts,
        "base_wins": results.base_wins,
        "finetuned_wins": results.finetuned_wins,
        "ties": results.ties,
        "finetuned_win_rate": results.finetuned_win_rate,
        "avg_base_score": results.avg_base_score,
        "avg_finetuned_score": results.avg_finetuned_score,
        "by_category": results.by_category,
        "pairwise_details": [
            {
                "category": r.prompt.category.value,
                "user_message": r.prompt.user_message,
                "humor_dial": r.prompt.humor_dial,
                "response_a": r.response_a.response,
                "response_b": r.response_b.response,
                "winner": r.winner,
                "score_a": r.score_a,
                "score_b": r.score_b,
                "reason": r.judge_reasoning,
            }
            for r in results.pairwise_results
        ],
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    log.info("eval_results_saved", path=str(output_path))


def load_eval_results(input_path: Path) -> dict[str, Any]:
    """Load saved evaluation results.

    Args:
        input_path: Path to the evaluation results JSON.

    Returns:
        Dict with evaluation data.

    Raises:
        FileNotFoundError: If the file doesn't exist.
    """
    if not input_path.exists():
        msg = f"Eval results not found: {input_path}"
        raise FileNotFoundError(msg)

    with input_path.open("r", encoding="utf-8") as f:
        return json.load(f)


# ── Humor Dial Consistency Test ─────────────────────────────────────────────


_HUMOR_CONSISTENCY_PROMPTS = [
    "I just ate an entire pizza by myself",
    "I've been procrastinating all day",
    "Tell me something interesting",
]


async def check_humor_dial_consistency(
    llm_generate_fn: Any,
    dial_values: list[int] | None = None,
) -> dict[int, list[str]]:
    """Test that responses change appropriately across humor dial settings.

    Generates responses to the same prompts at different humor levels
    and returns them for manual or automated comparison.

    Args:
        llm_generate_fn: Async callable(prompt) -> response.
        dial_values: Humor dial values to test (default: [2, 5, 7, 9]).

    Returns:
        Dict mapping dial value to list of responses.
    """
    if dial_values is None:
        dial_values = [2, 5, 7, 9]

    results: dict[int, list[str]] = {}

    for dial in dial_values:
        responses: list[str] = []
        for prompt_text in _HUMOR_CONSISTENCY_PROMPTS:
            response = await generate_response(
                prompt_text,
                humor_dial=dial,
                memory_context="",
                llm_generate_fn=llm_generate_fn,
            )
            responses.append(response)

        results[dial] = responses
        log.info("humor_dial_tested", dial=dial, responses=len(responses))

    return results
