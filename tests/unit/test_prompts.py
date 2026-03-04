"""Tests for ARA system prompt management."""

from __future__ import annotations

from datetime import datetime

from ara.config.settings import PersonalitySettings
from ara.core.prompts import build_system_message, build_system_prompt
from ara.utils.types import MessageRole


class TestBuildSystemPrompt:
    """Tests for dynamic system prompt assembly."""

    def test_contains_base_personality(self) -> None:
        personality = PersonalitySettings()
        prompt = build_system_prompt(personality)
        assert "ARA" in prompt
        assert "she/her" in prompt
        assert "friend" in prompt.lower()

    def test_instructs_against_ai_disclaimer(self) -> None:
        personality = PersonalitySettings()
        prompt = build_system_prompt(personality)
        # The prompt should instruct ARA to never say "As an AI..."
        assert 'Never say "As an AI..."' in prompt

    def test_humor_dial_1_supportive(self) -> None:
        personality = PersonalitySettings(humor_dial=1)
        prompt = build_system_prompt(personality)
        assert "SUPPORTIVE" in prompt
        assert "GROK" not in prompt

    def test_humor_dial_5_warm_sarcastic(self) -> None:
        personality = PersonalitySettings(humor_dial=5)
        prompt = build_system_prompt(personality)
        assert "WARM SARCASTIC" in prompt

    def test_humor_dial_7_witty_default(self) -> None:
        personality = PersonalitySettings(humor_dial=7)
        prompt = build_system_prompt(personality)
        assert "WITTY IRREVERENT" in prompt

    def test_humor_dial_9_full_grok(self) -> None:
        personality = PersonalitySettings(humor_dial=9)
        prompt = build_system_prompt(personality)
        assert "GROK" in prompt

    def test_humor_dial_10_unfiltered(self) -> None:
        personality = PersonalitySettings(humor_dial=10)
        prompt = build_system_prompt(personality)
        assert "UNFILTERED" in prompt

    def test_distress_detection_enabled(self) -> None:
        personality = PersonalitySettings(
            auto_dial_down_on_distress=True,
            distress_dial_reduction=2,
        )
        prompt = build_system_prompt(personality)
        assert "distress" in prompt.lower()
        assert "2" in prompt

    def test_distress_detection_disabled(self) -> None:
        personality = PersonalitySettings(auto_dial_down_on_distress=False)
        prompt = build_system_prompt(personality)
        # Should not contain the distress instruction
        assert "automatically lower" not in prompt

    def test_memory_context_injected(self) -> None:
        personality = PersonalitySettings()
        prompt = build_system_prompt(
            personality,
            memory_context="User had a tough day at work yesterday. Mentioned feeling stressed.",
        )
        assert "tough day at work" in prompt
        assert "---" in prompt  # context delimiters

    def test_empty_memory_context_excluded(self) -> None:
        personality = PersonalitySettings()
        prompt = build_system_prompt(personality, memory_context="")
        assert "Relevant context from memory" not in prompt

    def test_tools_listed(self) -> None:
        personality = PersonalitySettings()
        prompt = build_system_prompt(
            personality,
            available_tools=["desktop.click", "web.search", "memory.recall"],
        )
        assert "desktop.click" in prompt
        assert "web.search" in prompt

    def test_no_tools_excluded(self) -> None:
        personality = PersonalitySettings()
        prompt = build_system_prompt(personality, available_tools=None)
        assert "tools:" not in prompt.lower() or "access to these tools" not in prompt

    def test_current_time_included(self) -> None:
        personality = PersonalitySettings()
        fixed_time = datetime(2026, 3, 4, 14, 30)
        prompt = build_system_prompt(personality, current_time=fixed_time)
        assert "March 04, 2026" in prompt
        assert "02:30 PM" in prompt

    def test_all_humor_levels_produce_output(self) -> None:
        for dial in range(1, 11):
            personality = PersonalitySettings(humor_dial=dial)
            prompt = build_system_prompt(personality)
            assert len(prompt) > 100, f"Dial {dial} produced suspiciously short prompt"


class TestBuildSystemMessage:
    """Tests for the ChatMessage wrapper."""

    def test_returns_system_message(self) -> None:
        personality = PersonalitySettings()
        msg = build_system_message(personality)
        assert msg.role == MessageRole.SYSTEM
        assert "ARA" in msg.content
        assert len(msg.content) > 100
