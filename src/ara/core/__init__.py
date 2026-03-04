"""Core LLM interaction — client, streaming, prompt management."""

from ara.core.llm import LLMClient
from ara.core.prompts import build_system_message, build_system_prompt
from ara.core.streaming import StreamBuffer, collect_stream, stream_with_callbacks

__all__ = [
    "LLMClient",
    "StreamBuffer",
    "build_system_message",
    "build_system_prompt",
    "collect_stream",
    "stream_with_callbacks",
]
