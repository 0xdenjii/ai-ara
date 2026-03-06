"""Conversation context manager — sliding window + summarization for context limits.

Manages the conversation history that flows through the agent pipeline,
ensuring it stays within the LLM's context window by summarizing older
turns while preserving recent context.

Architecture:
    - Sliding window: keeps the last N turns in full
    - Summarization: older turns are compressed into a running summary
    - Token estimation: rough char-based estimation (4 chars ≈ 1 token)
    - Context budget: reserves space for system prompt + retrieval context

Usage:
    manager = ContextManager(settings, llm_client)
    managed_history = await manager.prepare_context(
        conversation_history,
        system_prompt_tokens=500,
        retrieval_context_tokens=2000,
    )
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ara.utils.logging import get_logger
from ara.utils.types import ChatMessage, MessageRole

if TYPE_CHECKING:
    from ara.config.settings import AraSettings
    from ara.core.llm import LLMClient

log = get_logger("context_manager")


# ── Constants ────────────────────────────────────────────────────────────────

# Rough chars-per-token ratio for estimation (conservative)
_CHARS_PER_TOKEN = 4

# Minimum turns to keep in full (never summarize the last N turns)
_MIN_RECENT_TURNS = 4

# Summarization prompt
_SUMMARIZE_PROMPT = """Summarize this conversation history concisely. Preserve:
- Key topics discussed
- Important decisions made
- User's emotional state or mood
- Any action items or promises
- Names, dates, and specific facts mentioned

Write a 2-4 sentence summary. Do NOT include any preamble."""


class ContextManager:
    """Manages conversation context to stay within LLM token limits.

    Uses a sliding window approach: recent turns are kept verbatim,
    older turns are summarized into a running context summary.

    Args:
        settings: Application settings (for context length limits).
        llm_client: LLM for summarizing older turns.
    """

    def __init__(
        self,
        settings: AraSettings,
        llm_client: LLMClient,
    ) -> None:
        self._settings = settings
        self._llm = llm_client
        self._running_summary: str = ""

    @property
    def running_summary(self) -> str:
        """Current running summary of older conversation turns."""
        return self._running_summary

    def reset(self) -> None:
        """Reset the running summary (new conversation)."""
        self._running_summary = ""

    async def prepare_context(
        self,
        history: list[ChatMessage],
        *,
        system_prompt_tokens: int = 500,
        retrieval_context_tokens: int = 2000,
        max_context_tokens: int | None = None,
    ) -> list[ChatMessage]:
        """Prepare conversation history to fit within context window.

        Keeps recent turns verbatim, summarizes older turns if needed.

        Args:
            history: Full conversation history.
            system_prompt_tokens: Estimated tokens used by system prompt.
            retrieval_context_tokens: Estimated tokens used by retrieval context.
            max_context_tokens: Override for max context window. Defaults to
                settings.llm.primary_model_context_length.

        Returns:
            Trimmed history that fits within context budget, potentially
            with a summary message prepended for older context.
        """
        max_tokens = max_context_tokens or self._settings.llm.primary_model_context_length
        response_budget = self._settings.llm.max_response_tokens

        # Available tokens for conversation history
        available = max_tokens - system_prompt_tokens - retrieval_context_tokens - response_budget

        if available <= 0:
            log.warning(
                "context_budget_exhausted",
                max_tokens=max_tokens,
                system=system_prompt_tokens,
                retrieval=retrieval_context_tokens,
                response=response_budget,
            )
            # Return only the most recent turn
            return history[-2:] if len(history) >= 2 else history

        # Estimate tokens for each message
        message_tokens = [self._estimate_tokens(msg.content) for msg in history]
        total_history_tokens = sum(message_tokens)

        # If everything fits, return as-is
        if total_history_tokens <= available:
            if self._running_summary:
                return self._prepend_summary(history)
            return list(history)

        # Need to trim: keep recent turns, summarize older ones
        return await self._trim_with_summary(history, message_tokens, available)

    async def _trim_with_summary(
        self,
        history: list[ChatMessage],
        message_tokens: list[int],
        available_tokens: int,
    ) -> list[ChatMessage]:
        """Trim history by summarizing older turns.

        Args:
            history: Full conversation history.
            message_tokens: Token estimates per message.
            available_tokens: Maximum tokens for history.

        Returns:
            Trimmed history with summary prepended.
        """
        # Reserve some tokens for the summary message itself
        summary_budget = 200  # tokens for the summary
        remaining = available_tokens - summary_budget

        # Work backwards to find how many recent messages fit
        recent_messages: list[ChatMessage] = []
        recent_tokens = 0
        cutoff_index = len(history)

        for i in range(len(history) - 1, -1, -1):
            msg_tok = message_tokens[i]
            if recent_tokens + msg_tok > remaining and len(recent_messages) >= _MIN_RECENT_TURNS:
                cutoff_index = i + 1
                break
            recent_tokens += msg_tok
            recent_messages.insert(0, history[i])
        else:
            cutoff_index = 0

        # If there are older messages to summarize
        if cutoff_index > 0:
            older_messages = history[:cutoff_index]
            await self._update_summary(older_messages)

        return self._prepend_summary(recent_messages)

    async def _update_summary(self, older_messages: list[ChatMessage]) -> None:
        """Update the running summary with older messages.

        Args:
            older_messages: Messages to incorporate into the summary.
        """
        # Format older messages for summarization
        turns_text = []
        for msg in older_messages:
            role = "User" if msg.role == MessageRole.USER else "ARA"
            turns_text.append(f"{role}: {msg.content}")

        conversation_text = "\n".join(turns_text)

        # Include existing summary for continuity
        prompt_parts = []
        if self._running_summary:
            prompt_parts.append(f"Previous summary: {self._running_summary}")
        prompt_parts.append(f"New conversation to summarize:\n{conversation_text}")

        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=_SUMMARIZE_PROMPT),
            ChatMessage(role=MessageRole.USER, content="\n\n".join(prompt_parts)),
        ]

        try:
            self._running_summary = await self._llm.generate(
                messages, temperature=0.3, max_tokens=300
            )
            log.info(
                "context_summarized",
                older_turns=len(older_messages),
                summary_length=len(self._running_summary),
            )
        except Exception as exc:
            log.warning(
                "context_summarization_failed",
                error=str(exc),
            )
            # Keep existing summary on failure
            if not self._running_summary:
                # Extractive fallback: just keep the last few older messages
                self._running_summary = " | ".join(msg.content[:100] for msg in older_messages[-3:])

    def _prepend_summary(self, recent_messages: list[ChatMessage]) -> list[ChatMessage]:
        """Prepend the running summary as a system message.

        Args:
            recent_messages: Recent messages to keep.

        Returns:
            Messages with summary prepended.
        """
        if not self._running_summary:
            return list(recent_messages)

        summary_msg = ChatMessage(
            role=MessageRole.SYSTEM,
            content=(f"Earlier conversation summary: {self._running_summary}"),
        )
        return [summary_msg, *recent_messages]

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Rough token count estimation.

        Uses a simple chars/4 heuristic. Not precise but sufficient
        for context budget planning.

        Args:
            text: Text to estimate.

        Returns:
            Estimated token count.
        """
        return max(1, len(text) // _CHARS_PER_TOKEN)
