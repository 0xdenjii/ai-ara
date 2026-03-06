"""Token streaming utilities for ARA's LLM output.

Provides buffering, sentence detection, and callback-based streaming
for both voice (TTS needs complete sentences) and text (UI needs tokens).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ara.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable

log = get_logger("streaming")

# Sentence-ending pattern: period, exclamation, question mark, ellipsis
# followed by whitespace or end of string. Handles common edge cases.
_SENTENCE_END_PATTERN = re.compile(
    r"(?<=[.!?…])"  # lookbehind for sentence-ending punctuation
    r"(?:\s|$)"  # followed by whitespace or end
)

# Patterns that look like sentence endings but aren't
_FALSE_ENDINGS = re.compile(
    r"(?:"
    r"(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|etc|i\.e|e\.g)\.\s"  # abbreviations
    r"|(?:\d\.)\s"  # numbered lists like "1. "
    r")"
)


@dataclass
class StreamBuffer:
    """Accumulates streaming tokens and emits complete sentences.

    Used by the voice pipeline to synthesize speech on sentence boundaries
    rather than waiting for the entire response.

    Usage:
        buffer = StreamBuffer()
        for token in stream:
            sentences = buffer.add(token)
            for sentence in sentences:
                await tts.synthesize(sentence)
        # Don't forget the remainder
        final = buffer.flush()
        if final:
            await tts.synthesize(final)
    """

    _buffer: str = field(default="", init=False)
    _total_tokens: int = field(default=0, init=False)

    @property
    def pending(self) -> str:
        """Text accumulated but not yet emitted as a complete sentence."""
        return self._buffer

    @property
    def total_tokens(self) -> int:
        """Total number of tokens processed."""
        return self._total_tokens

    def add(self, token: str) -> list[str]:
        """Add a token and return any complete sentences.

        Args:
            token: A single token from the LLM stream.

        Returns:
            List of complete sentences (may be empty if no sentence boundary found).
        """
        self._total_tokens += 1
        self._buffer += token

        sentences: list[str] = []
        while True:
            match = _SENTENCE_END_PATTERN.search(self._buffer)
            if match is None:
                break

            split_pos = match.end()
            candidate = self._buffer[:split_pos].strip()

            # Check for false positives (abbreviations, numbered lists)
            if _FALSE_ENDINGS.search(candidate):
                # Check if this is the ONLY match — if so, skip it
                remaining = self._buffer[split_pos:]
                next_match = _SENTENCE_END_PATTERN.search(remaining)
                if next_match is None:
                    break

            if candidate:
                sentences.append(candidate)
            self._buffer = self._buffer[split_pos:]

        return sentences

    def flush(self) -> str:
        """Return any remaining text in the buffer and reset.

        Call this when the stream ends to get the final partial sentence.
        """
        remaining = self._buffer.strip()
        self._buffer = ""
        return remaining

    def reset(self) -> None:
        """Clear the buffer and token count."""
        self._buffer = ""
        self._total_tokens = 0


async def stream_with_callbacks(
    token_stream: AsyncIterator[str],
    *,
    on_token: Callable[[str], Any] | None = None,
    on_sentence: Callable[[str], Any] | None = None,
    on_complete: Callable[[str], Any] | None = None,
) -> str:
    """Process a token stream with optional callbacks at different granularities.

    This is the main entry point for consuming LLM output. It:
    1. Calls on_token for every token (for UI streaming)
    2. Calls on_sentence for each complete sentence (for TTS)
    3. Calls on_complete with the full text when done

    Args:
        token_stream: Async iterator of tokens from LLMClient.generate_stream().
        on_token: Called with each individual token.
        on_sentence: Called with each complete sentence.
        on_complete: Called with the full assembled response.

    Returns:
        The complete generated text.
    """
    buffer = StreamBuffer()
    full_text_parts: list[str] = []

    async for token in token_stream:
        full_text_parts.append(token)

        if on_token is not None:
            result = on_token(token)
            # Support async callbacks
            if hasattr(result, "__await__"):
                await result

        if on_sentence is not None:
            sentences = buffer.add(token)
            for sentence in sentences:
                result = on_sentence(sentence)
                if hasattr(result, "__await__"):
                    await result

    # Handle any remaining text in the buffer
    if on_sentence is not None:
        remaining = buffer.flush()
        if remaining:
            result = on_sentence(remaining)
            if hasattr(result, "__await__"):
                await result

    full_text = "".join(full_text_parts)

    if on_complete is not None:
        result = on_complete(full_text)
        if hasattr(result, "__await__"):
            await result

    return full_text


async def collect_stream(token_stream: AsyncIterator[str]) -> str:
    """Collect all tokens from a stream into a single string.

    Simple utility when you don't need callbacks.
    """
    parts: list[str] = []
    async for token in token_stream:
        parts.append(token)  # noqa: PERF401
    return "".join(parts)


# ── Think Block Filtering ────────────────────────────────────────────────────
# Qwen3 emits <think>...</think> reasoning blocks by default.
# These must be filtered from user-visible output.

# Pattern to match complete think blocks (including multi-line)
_THINK_BLOCK_PATTERN = re.compile(r"<think>.*?</think>", flags=re.DOTALL)


def filter_think_blocks(text: str) -> str:
    """Remove <think>...</think> blocks from completed text.

    Args:
        text: Full text that may contain think blocks.

    Returns:
        Text with all think blocks removed and leading/trailing whitespace stripped.
    """
    return _THINK_BLOCK_PATTERN.sub("", text).strip()


@dataclass
class ThinkBlockFilter:
    """Streaming filter that strips <think>...</think> blocks token-by-token.

    Tracks whether we're inside a think block and buffers tokens that
    might be part of opening/closing tags. Only emits tokens that are
    outside think blocks.

    Usage:
        tbf = ThinkBlockFilter()
        async for token in llm_stream:
            filtered = tbf.process(token)
            if filtered:
                print(filtered, end="")
    """

    _accumulated: str = field(default="", init=False)
    _in_think: bool = field(default=False, init=False)
    _emitted_up_to: int = field(default=0, init=False)

    def process(self, token: str) -> str:
        """Process a single token and return visible text (if any).

        Args:
            token: A single token from the LLM stream.

        Returns:
            Text to display (empty string if inside a think block or buffering).
        """
        self._accumulated += token

        # If inside a think block, check for closing tag
        if self._in_think:
            if "</think>" in self._accumulated:
                # Remove the complete think block
                self._accumulated = _THINK_BLOCK_PATTERN.sub("", self._accumulated)
                self._in_think = False
                self._emitted_up_to = 0
                # Return any text that follows the closing tag
                if self._accumulated:
                    result = self._accumulated
                    self._accumulated = ""
                    return result
            return ""

        # Check if we're entering a think block
        if "<think>" in self._accumulated:
            # Emit everything before the <think> tag
            idx = self._accumulated.index("<think>")
            before = self._accumulated[:idx]
            self._accumulated = self._accumulated[idx:]
            self._in_think = True
            self._emitted_up_to = 0
            return before

        # Check for partial tag at the end (e.g., "<thi" could become "<think>")
        # Buffer the last few characters that could be a partial tag
        partial_tag_len = 0
        for i in range(1, min(len("<think>"), len(self._accumulated) + 1)):
            if "<think>"[:i] == self._accumulated[-i:]:
                partial_tag_len = i
                break

        if partial_tag_len > 0:
            safe = self._accumulated[self._emitted_up_to : -partial_tag_len]
            self._emitted_up_to = len(self._accumulated) - partial_tag_len
            return safe

        # No think block concerns — emit everything new
        result = self._accumulated[self._emitted_up_to :]
        self._emitted_up_to = len(self._accumulated)
        return result

    def flush(self) -> str:
        """Return any remaining buffered text.

        Call this when the stream ends to get any partial content
        that was being held back for potential tag matching.
        """
        if self._in_think:
            # Incomplete think block — drop it
            return ""
        remaining = self._accumulated[self._emitted_up_to :]
        self._accumulated = ""
        self._emitted_up_to = 0
        return remaining

    def reset(self) -> None:
        """Reset the filter state."""
        self._accumulated = ""
        self._in_think = False
        self._emitted_up_to = 0
