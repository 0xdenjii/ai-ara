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
