"""Tests for token streaming utilities."""

from __future__ import annotations

import asyncio

import pytest

from ara.core.streaming import StreamBuffer, collect_stream, stream_with_callbacks


class TestStreamBuffer:
    """Tests for the sentence-aware stream buffer."""

    def test_empty_buffer(self) -> None:
        buf = StreamBuffer()
        assert buf.pending == ""
        assert buf.total_tokens == 0

    def test_single_sentence(self) -> None:
        buf = StreamBuffer()
        sentences = buf.add("Hello world. ")
        assert sentences == ["Hello world."]

    def test_accumulates_partial(self) -> None:
        buf = StreamBuffer()
        assert buf.add("Hello ") == []
        assert buf.add("world") == []
        assert buf.pending == "Hello world"

    def test_sentence_boundary_on_period(self) -> None:
        buf = StreamBuffer()
        buf.add("Hello")
        buf.add(" world")
        sentences = buf.add(". ")
        assert sentences == ["Hello world."]

    def test_multiple_sentences(self) -> None:
        buf = StreamBuffer()
        sentences = buf.add("First sentence. Second sentence! Third? ")
        assert len(sentences) == 3
        assert sentences[0] == "First sentence."
        assert sentences[1] == "Second sentence!"
        assert sentences[2] == "Third?"

    def test_question_mark(self) -> None:
        buf = StreamBuffer()
        sentences = buf.add("How are you? ")
        assert sentences == ["How are you?"]

    def test_exclamation(self) -> None:
        buf = StreamBuffer()
        sentences = buf.add("Wow! ")
        assert sentences == ["Wow!"]

    def test_flush_returns_remainder(self) -> None:
        buf = StreamBuffer()
        buf.add("incomplete sentence")
        remainder = buf.flush()
        assert remainder == "incomplete sentence"
        assert buf.pending == ""

    def test_flush_empty(self) -> None:
        buf = StreamBuffer()
        assert buf.flush() == ""

    def test_token_count(self) -> None:
        buf = StreamBuffer()
        buf.add("a")
        buf.add("b")
        buf.add("c")
        assert buf.total_tokens == 3

    def test_reset(self) -> None:
        buf = StreamBuffer()
        buf.add("some text")
        buf.reset()
        assert buf.pending == ""
        assert buf.total_tokens == 0

    def test_sentence_at_end_of_stream(self) -> None:
        buf = StreamBuffer()
        # Token-by-token simulation
        buf.add("I")
        buf.add(" am")
        buf.add(" fine")
        sentences = buf.add(".")
        # Period without trailing space won't match (needs space or EOL)
        # The buffer still holds it
        remainder = buf.flush()
        total_text = "".join(s for s in sentences) + remainder
        assert "I am fine." in total_text


class TestStreamWithCallbacks:
    """Tests for the callback-based stream processor."""

    @pytest.mark.asyncio
    async def test_on_token_callback(self) -> None:
        tokens_received: list[str] = []

        async def mock_stream():
            for t in ["Hello", " ", "world", "!"]:
                yield t

        await stream_with_callbacks(
            mock_stream(),
            on_token=lambda t: tokens_received.append(t),
        )
        assert tokens_received == ["Hello", " ", "world", "!"]

    @pytest.mark.asyncio
    async def test_on_sentence_callback(self) -> None:
        sentences_received: list[str] = []

        async def mock_stream():
            for t in ["Hello", " ", "world", ".", " ", "How", " ", "are", " ", "you", "?"]:
                yield t

        await stream_with_callbacks(
            mock_stream(),
            on_sentence=lambda s: sentences_received.append(s),
        )
        assert len(sentences_received) >= 1
        # At minimum the full text should be covered
        combined = " ".join(sentences_received)
        assert "Hello world" in combined

    @pytest.mark.asyncio
    async def test_on_complete_callback(self) -> None:
        complete_text: list[str] = []

        async def mock_stream():
            for t in ["Hello", " ", "world"]:
                yield t

        result = await stream_with_callbacks(
            mock_stream(),
            on_complete=lambda text: complete_text.append(text),
        )
        assert result == "Hello world"
        assert complete_text == ["Hello world"]

    @pytest.mark.asyncio
    async def test_returns_full_text(self) -> None:
        async def mock_stream():
            for t in ["a", "b", "c"]:
                yield t

        result = await stream_with_callbacks(mock_stream())
        assert result == "abc"

    @pytest.mark.asyncio
    async def test_async_callbacks(self) -> None:
        tokens: list[str] = []

        async def async_on_token(t: str) -> None:
            await asyncio.sleep(0)  # simulate async work
            tokens.append(t)

        async def mock_stream():
            for t in ["x", "y"]:
                yield t

        await stream_with_callbacks(mock_stream(), on_token=async_on_token)
        assert tokens == ["x", "y"]


class TestCollectStream:
    """Tests for the simple stream collector."""

    @pytest.mark.asyncio
    async def test_collect(self) -> None:
        async def mock_stream():
            for t in ["Hello", " ", "world"]:
                yield t

        result = await collect_stream(mock_stream())
        assert result == "Hello world"

    @pytest.mark.asyncio
    async def test_collect_empty(self) -> None:
        async def mock_stream():
            return
            yield

        result = await collect_stream(mock_stream())
        assert result == ""
