"""Chat route — POST for single response, SSE streaming for real-time tokens.

Endpoints:
    POST /api/chat — Non-streaming chat (returns full response)
    POST /api/chat/stream — Streaming chat via Server-Sent Events (SSE)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from ara.api.app import AppState, get_app_state
from ara.api.schemas import ChatRequest, ChatResponse, StreamChunk
from ara.utils.errors import LLMError
from ara.utils.logging import get_logger
from ara.utils.types import ChatMessage, MessageRole

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

log = get_logger("api.chat")

router = APIRouter(prefix="/api/chat", tags=["chat"])


def _build_messages(request: ChatRequest) -> list[ChatMessage]:
    """Convert API message history to internal ChatMessage format."""
    messages: list[ChatMessage] = []
    for msg in request.history:
        role = (
            MessageRole(msg.role)
            if msg.role in MessageRole.__members__.values()
            else MessageRole.USER
        )
        messages.append(ChatMessage(role=role, content=msg.content))
    messages.append(ChatMessage(role=MessageRole.USER, content=request.message))
    return messages


@router.post("", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    state: AppState = Depends(get_app_state),
) -> ChatResponse:
    """Process a chat message and return the full response.

    If an orchestrator is available, routes through the full agent pipeline.
    Otherwise falls back to direct LLM generation.
    """
    if not state.llm_client or not state.llm_client.is_connected:
        return ChatResponse(
            response="",
            error="LLM is not connected. Check Ollama status.",
        )

    messages = _build_messages(request)

    # Try orchestrator first
    if state.orchestrator:
        try:
            history = [
                ChatMessage(role=MessageRole(m.role), content=m.content) for m in request.history
            ]
            result = await state.orchestrator.process(
                request.message,
                conversation_id=request.conversation_id,
                conversation_history=history,
                max_steps=request.max_steps,
            )
            return ChatResponse(
                response=result.response,
                conversation_id=request.conversation_id,
                intent=result.intent,
                agent=result.agent,
                confidence=result.confidence,
                elapsed_ms=result.elapsed_ms,
                tool_calls=result.tool_calls,
                tool_results=result.tool_results,
                error=result.error,
            )
        except Exception as exc:
            log.warning("orchestrator_failed_fallback_to_llm", error=str(exc))

    # Direct LLM fallback
    try:
        from ara.core.prompts import build_system_message

        system_msg = build_system_message(state.settings.personality)
        all_messages = [system_msg, *messages]
        response_text = await state.llm_client.generate(all_messages)

        # Filter think blocks
        from ara.core.streaming import filter_think_blocks

        response_text = filter_think_blocks(response_text)

        return ChatResponse(
            response=response_text,
            conversation_id=request.conversation_id,
            intent="general",
            agent="direct_llm",
        )
    except LLMError as exc:
        return ChatResponse(response="", error=str(exc))


async def _stream_tokens(
    state: AppState,
    request: ChatRequest,
) -> AsyncIterator[str]:
    """Generate SSE events for streaming chat.

    Yields Server-Sent Events in the format:
        data: {"type": "token", "content": "..."}
        data: {"type": "done", "metadata": {...}}
        data: {"type": "error", "content": "..."}
    """
    if not state.llm_client or not state.llm_client.is_connected:
        chunk = StreamChunk(type="error", content="LLM is not connected")
        yield f"data: {chunk.model_dump_json()}\n\n"
        return

    messages = _build_messages(request)

    try:
        from ara.core.prompts import build_system_message
        from ara.core.streaming import ThinkBlockFilter

        system_msg = build_system_message(state.settings.personality)
        all_messages = [system_msg, *messages]

        think_filter = ThinkBlockFilter()
        token_count = 0

        async for token in state.llm_client.generate_stream(all_messages):
            filtered = think_filter.process(token)
            if filtered:
                token_count += 1
                chunk = StreamChunk(type="token", content=filtered)
                yield f"data: {chunk.model_dump_json()}\n\n"

        # Done event
        done_chunk = StreamChunk(
            type="done",
            metadata={"token_count": token_count},
        )
        yield f"data: {done_chunk.model_dump_json()}\n\n"

    except Exception as exc:
        log.error("stream_error", error=str(exc))
        error_chunk = StreamChunk(type="error", content=str(exc))
        yield f"data: {error_chunk.model_dump_json()}\n\n"


@router.post("/stream")
async def chat_stream(
    request: ChatRequest,
    state: AppState = Depends(get_app_state),
) -> StreamingResponse:
    """Stream chat response tokens via Server-Sent Events (SSE).

    Returns a text/event-stream response with JSON-encoded chunks.
    Each chunk has type "token", "done", or "error".
    """
    return StreamingResponse(
        _stream_tokens(state, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
