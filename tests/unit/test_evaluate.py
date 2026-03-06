"""Tests for the fine-tuning evaluation module."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from ara.fine_tuning.evaluate import (
    DEFAULT_EVAL_PROMPTS,
    EvalCategory,
    EvalPrompt,
    EvalResults,
    ModelResponse,
    PairwiseResult,
    check_humor_dial_consistency,
    compute_eval_results,
    generate_response,
    judge_pairwise,
    load_eval_results,
    save_eval_results,
)

# ── EvalPrompt ──────────────────────────────────────────────────────────────


class TestEvalPrompt:
    def test_creation(self) -> None:
        prompt = EvalPrompt(
            EvalCategory.PERSONALITY,
            "Hey there",
            "",
            7,
            ["casual", "warm"],
        )
        assert prompt.category == EvalCategory.PERSONALITY
        assert prompt.user_message == "Hey there"
        assert prompt.humor_dial == 7

    def test_with_context(self) -> None:
        prompt = EvalPrompt(
            EvalCategory.KNOWLEDGE,
            "What did I do yesterday?",
            "User wrote: went to the gym",
            5,
            ["specific", "memory"],
        )
        assert prompt.context == "User wrote: went to the gym"


class TestDefaultEvalPrompts:
    def test_has_all_categories(self) -> None:
        categories = {p.category for p in DEFAULT_EVAL_PROMPTS}
        assert EvalCategory.PERSONALITY in categories
        assert EvalCategory.KNOWLEDGE in categories
        assert EvalCategory.HUMOR_LOW in categories
        assert EvalCategory.HUMOR_MID in categories
        assert EvalCategory.HUMOR_HIGH in categories
        assert EvalCategory.GENERAL in categories

    def test_minimum_prompts(self) -> None:
        assert len(DEFAULT_EVAL_PROMPTS) >= 10

    def test_all_have_traits(self) -> None:
        for p in DEFAULT_EVAL_PROMPTS:
            assert len(p.expected_traits) > 0


# ── ModelResponse ───────────────────────────────────────────────────────────


class TestModelResponse:
    def test_creation(self) -> None:
        resp = ModelResponse(
            model_name="qwen3:30b-a3b",
            response="Hey, what's up?",
            latency_ms=250.0,
            token_count=10,
        )
        assert resp.model_name == "qwen3:30b-a3b"
        assert resp.response == "Hey, what's up?"


# ── PairwiseResult ──────────────────────────────────────────────────────────


class TestPairwiseResult:
    def _make_result(
        self, winner: str = "b", score_a: float = 3.0, score_b: float = 4.5
    ) -> PairwiseResult:
        return PairwiseResult(
            prompt=EvalPrompt(EvalCategory.PERSONALITY, "Hi", "", 7, ["warm"]),
            response_a=ModelResponse("base", "Hello.", 200, 5),
            response_b=ModelResponse("finetuned", "Hey! What's up?", 180, 8),
            winner=winner,
            score_a=score_a,
            score_b=score_b,
            judge_reasoning="Fine-tuned is more natural",
        )

    def test_winner_is_b(self) -> None:
        result = self._make_result()
        assert result.winner == "b"
        assert result.score_b > result.score_a


# ── EvalResults ─────────────────────────────────────────────────────────────


class TestEvalResults:
    def test_empty_results(self) -> None:
        results = EvalResults()
        assert results.total_prompts == 0
        assert results.finetuned_win_rate == 0.0

    def test_win_rate_calculation(self) -> None:
        results = EvalResults(
            total_prompts=10,
            base_wins=3,
            finetuned_wins=7,
            ties=0,
        )
        assert results.finetuned_win_rate == 70.0

    def test_win_rate_with_ties(self) -> None:
        results = EvalResults(
            total_prompts=10,
            base_wins=2,
            finetuned_wins=6,
            ties=2,
        )
        assert results.finetuned_win_rate == 75.0

    def test_summary(self) -> None:
        results = EvalResults(
            total_prompts=5,
            base_wins=1,
            finetuned_wins=3,
            ties=1,
            avg_base_score=3.2,
            avg_finetuned_score=4.1,
        )
        summary = results.summary()
        assert "Total prompts evaluated: 5" in summary
        assert "Fine-tuned wins: 3" in summary
        assert "Avg fine-tuned score: 4.10/5" in summary


# ── Compute Results ─────────────────────────────────────────────────────────


class TestComputeEvalResults:
    def _make_result(
        self,
        category: EvalCategory,
        winner: str,
        score_a: float,
        score_b: float,
    ) -> PairwiseResult:
        return PairwiseResult(
            prompt=EvalPrompt(category, "test", "", 7, []),
            response_a=ModelResponse("base", "resp a", 200, 10),
            response_b=ModelResponse("ft", "resp b", 180, 10),
            winner=winner,
            score_a=score_a,
            score_b=score_b,
            judge_reasoning="test",
        )

    def test_empty_results(self) -> None:
        results = compute_eval_results([])
        assert results.total_prompts == 0

    def test_computes_wins(self) -> None:
        pairwise = [
            self._make_result(EvalCategory.PERSONALITY, "b", 3.0, 4.0),
            self._make_result(EvalCategory.PERSONALITY, "b", 2.5, 4.5),
            self._make_result(EvalCategory.PERSONALITY, "a", 4.0, 3.0),
            self._make_result(EvalCategory.HUMOR_MID, "tie", 3.5, 3.5),
        ]
        results = compute_eval_results(pairwise)
        assert results.total_prompts == 4
        assert results.finetuned_wins == 2
        assert results.base_wins == 1
        assert results.ties == 1

    def test_computes_averages(self) -> None:
        pairwise = [
            self._make_result(EvalCategory.PERSONALITY, "b", 3.0, 4.0),
            self._make_result(EvalCategory.PERSONALITY, "b", 2.0, 5.0),
        ]
        results = compute_eval_results(pairwise)
        assert results.avg_base_score == 2.5
        assert results.avg_finetuned_score == 4.5

    def test_by_category(self) -> None:
        pairwise = [
            self._make_result(EvalCategory.PERSONALITY, "b", 3.0, 4.0),
            self._make_result(EvalCategory.KNOWLEDGE, "b", 2.0, 5.0),
        ]
        results = compute_eval_results(pairwise)
        assert "personality" in results.by_category
        assert "knowledge" in results.by_category
        assert results.by_category["personality"]["avg_ft"] == 4.0
        assert results.by_category["knowledge"]["avg_ft"] == 5.0


# ── Save/Load Results ───────────────────────────────────────────────────────


class TestSaveLoadEvalResults:
    def test_save_results(self, tmp_path: Path) -> None:
        results = EvalResults(
            total_prompts=2,
            base_wins=0,
            finetuned_wins=2,
            ties=0,
            avg_base_score=2.5,
            avg_finetuned_score=4.5,
        )
        path = tmp_path / "eval.json"
        save_eval_results(results, path)
        assert path.exists()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["total_prompts"] == 2
        assert data["finetuned_win_rate"] == 100.0

    def test_load_results(self, tmp_path: Path) -> None:
        results = EvalResults(total_prompts=5, finetuned_wins=3, base_wins=2)
        path = tmp_path / "eval.json"
        save_eval_results(results, path)
        loaded = load_eval_results(path)
        assert loaded["total_prompts"] == 5

    def test_load_missing_file(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_eval_results(Path("/nonexistent/eval.json"))


# ── Async Functions (mocked LLM) ───────────────────────────────────────────


class TestGenerateResponse:
    @pytest.mark.asyncio
    async def test_generates_response(self) -> None:
        mock_llm = AsyncMock(return_value="Hey! What's going on?")
        response = await generate_response("Hi", 7, "", mock_llm)
        assert response == "Hey! What's going on?"
        mock_llm.assert_called_once()


class TestJudgePairwise:
    @pytest.mark.asyncio
    async def test_judges_correctly(self) -> None:
        prompt = EvalPrompt(EvalCategory.PERSONALITY, "Hi", "", 7, ["warm"])
        response_a = ModelResponse("base", "Hello, how can I help?", 200, 10)
        response_b = ModelResponse("ft", "Hey! What's up?", 180, 8)

        mock_llm = AsyncMock(
            return_value='{"winner": "b", "score_a": 2, "score_b": 5, "reason": "More natural"}'
        )
        result = await judge_pairwise(prompt, response_a, response_b, mock_llm)
        assert result.winner == "b"
        assert result.score_a == 2.0
        assert result.score_b == 5.0

    @pytest.mark.asyncio
    async def test_handles_judge_failure(self) -> None:
        prompt = EvalPrompt(EvalCategory.PERSONALITY, "Hi", "", 7, ["warm"])
        response_a = ModelResponse("base", "Hello", 200, 5)
        response_b = ModelResponse("ft", "Hey", 180, 4)

        mock_llm = AsyncMock(return_value="not json")
        result = await judge_pairwise(prompt, response_a, response_b, mock_llm)
        assert result.winner == "tie"
        assert result.score_a == 3.0
        assert result.score_b == 3.0


class TestHumorDialConsistency:
    @pytest.mark.asyncio
    async def test_returns_responses_per_dial(self) -> None:
        mock_llm = AsyncMock(return_value="Some response")
        results = await check_humor_dial_consistency(mock_llm, dial_values=[2, 7])
        assert 2 in results
        assert 7 in results
        assert len(results[2]) == 3  # 3 prompts per dial
        assert len(results[7]) == 3


# ── EvalCategory ────────────────────────────────────────────────────────────


class TestEvalCategory:
    def test_all_categories(self) -> None:
        assert len(EvalCategory) == 6
        assert EvalCategory.PERSONALITY.value == "personality"
        assert EvalCategory.HUMOR_HIGH.value == "humor_high"
