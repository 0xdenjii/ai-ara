"""WebSocket handler — real-time bidirectional chat streaming.

Provides:
    - /ws endpoint for persistent WebSocket connections
    - Bidirectional message flow: client sends queries, server streams tokens
    - Supports ping/pong keepalive and cancel signals
    - JSON message format matching WSMessage/WSResponse schemas

Architecture:
    The WebSocket handler maintains a persistent connection per client.
    Each incoming "chat" message triggers a streaming LLM response.
    The client can send "cancel" to abort an in-progress generation.
    "ping" messages get "pong" responses for keepalive.

Usage (client-side):
    ws = new WebSocket("ws://localhost:8000/ws")
    ws.send(JSON.stringify({type: "chat", content: "Hello ARA!"}))
    ws.onmessage = (event) => {
        const msg = JSON.parse(event.data)
        if (msg.type === "token") console.log(msg.content)
        if (msg.type === "done") console.log("Complete!")
    }
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING

from fastapi import WebSocket, WebSocketDisconnect

from ara.api.app import AppState
from ara.api.schemas import WSResponse
from ara.utils.logging import get_logger
from ara.utils.types import ChatMessage, MessageRole

if TYPE_CHECKING:
    from fastapi import FastAPI

log = get_logger("api.ws")


async def _load_conversation_history(
    state: AppState,
    conversation_id: str,
    max_turns: int = 10,
) -> list[ChatMessage]:
    """Load recent conversation history from the ConversationStore.

    Args:
        state: Application state with conversation_store.
        conversation_id: The active conversation to load history for.
        max_turns: Maximum number of recent turns to include.

    Returns:
        List of ChatMessage objects representing the conversation history.
    """
    if (
        not conversation_id
        or not state.conversation_store
        or not state.conversation_store.is_connected
    ):
        return []

    try:
        loop = asyncio.get_running_loop()
        conversation = await loop.run_in_executor(
            None,
            state.conversation_store.get_conversation,
            conversation_id,
        )
        if not conversation or not conversation.turns:
            return []

        history: list[ChatMessage] = []
        # Take the most recent turns
        recent_turns = conversation.turns[-max_turns:]
        for turn in recent_turns:
            if turn.user_message:
                history.append(ChatMessage(role=MessageRole.USER, content=turn.user_message))
            if turn.assistant_message:
                history.append(
                    ChatMessage(role=MessageRole.ASSISTANT, content=turn.assistant_message)
                )
        return history
    except Exception as exc:
        log.warning("ws_history_load_failed", error=str(exc))
        return []


async def _handle_chat(
    ws: WebSocket,
    state: AppState,
    content: str,
    conversation_id: str,
    cancel_event: asyncio.Event,
) -> None:
    """Handle a chat message over WebSocket — stream tokens back.

    Routes through the orchestrator if available, otherwise falls back
    to direct LLM streaming.
    """
    if not state.llm_client or not state.llm_client.is_connected:
        resp = WSResponse(type="error", content="LLM is not connected")
        await ws.send_text(resp.model_dump_json())
        return

    # Load conversation history from store for context continuity
    conversation_history = await _load_conversation_history(state, conversation_id)

    # Try orchestrator first (non-streaming, sends full response)
    if state.orchestrator:
        try:
            result = await state.orchestrator.process(
                content,
                conversation_id=conversation_id,
                conversation_history=conversation_history,
            )
            # Persist the turn to conversation store
            if (
                result.response
                and state.conversation_store
                and state.conversation_store.is_connected
            ):
                try:
                    loop = asyncio.get_running_loop()
                    if not conversation_id:
                        conversation_id = await loop.run_in_executor(
                            None,
                            state.conversation_store.start_conversation,
                        )
                    await loop.run_in_executor(
                        None,
                        lambda: state.conversation_store.add_turn(
                            conversation_id,
                            user_message=content,
                            assistant_message=result.response,
                        ),
                    )
                except Exception as exc:
                    log.warning("ws_conversation_tracking_failed", error=str(exc))

            # Send orchestrator response as a single token + done
            if result.response:
                resp = WSResponse(
                    type="token",
                    content=result.response,
                    conversation_id=conversation_id,
                )
                await ws.send_text(resp.model_dump_json())
            done_resp = WSResponse(
                type="done",
                conversation_id=conversation_id,
                metadata={
                    "intent": result.intent,
                    "agent": result.agent,
                    "elapsed_ms": result.elapsed_ms,
                },
            )
            await ws.send_text(done_resp.model_dump_json())
            return
        except Exception as exc:
            log.warning("ws_orchestrator_failed_fallback_to_llm", error=str(exc))

    # Fallback: direct LLM streaming
    try:
        from ara.core.prompts import build_system_message
        from ara.core.streaming import ThinkBlockFilter

        system_msg = build_system_message(state.settings.personality)
        messages: list[ChatMessage] = [system_msg]
        # Include conversation history in fallback path too
        if conversation_history:
            messages.extend(conversation_history[-6:])
        messages.append(ChatMessage(role=MessageRole.USER, content=content))

        think_filter = ThinkBlockFilter()
        token_count = 0

        async for token in state.llm_client.generate_stream(messages):
            if cancel_event.is_set():
                cancel_resp = WSResponse(type="done", content="", metadata={"cancelled": True})
                await ws.send_text(cancel_resp.model_dump_json())
                return

            filtered = think_filter.process(token)
            if filtered:
                token_count += 1
                resp = WSResponse(
                    type="token",
                    content=filtered,
                    conversation_id=conversation_id,
                )
                await ws.send_text(resp.model_dump_json())

        done_resp = WSResponse(
            type="done",
            conversation_id=conversation_id,
            metadata={"token_count": token_count},
        )
        await ws.send_text(done_resp.model_dump_json())

    except Exception as exc:
        log.error("ws_chat_error", error=str(exc))
        # Sanitize error message to prevent leaking internal details to client
        error_resp = WSResponse(
            type="error", content="An internal error occurred. Please try again."
        )
        await ws.send_text(error_resp.model_dump_json())


async def websocket_endpoint(ws: WebSocket) -> None:
    """Main WebSocket handler for bidirectional chat.

    Message types:
        Incoming (client → server):
            - chat: Start a new chat query
            - ping: Keepalive check
            - cancel: Cancel current generation

        Outgoing (server → client):
            - token: Streaming token
            - done: Generation complete
            - error: Error occurred
            - pong: Keepalive response
            - status: Server status update
    """
    await ws.accept()
    state: AppState = ws.app.state.ara

    cancel_event = asyncio.Event()
    active_task: asyncio.Task[None] | None = None

    log.info("ws_connected", client=str(ws.client))

    try:
        while True:
            raw = await ws.receive_text()

            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                error_resp = WSResponse(type="error", content="Invalid JSON")
                await ws.send_text(error_resp.model_dump_json())
                continue

            msg_type = data.get("type", "")
            content = data.get("content", "")
            conversation_id = data.get("conversation_id", "")

            if msg_type == "ping":
                pong = WSResponse(type="pong")
                await ws.send_text(pong.model_dump_json())

            elif msg_type == "cancel":
                cancel_event.set()
                if active_task and not active_task.done():
                    active_task.cancel()

            elif msg_type == "chat":
                if not content.strip():
                    error_resp = WSResponse(type="error", content="Empty message")
                    await ws.send_text(error_resp.model_dump_json())
                    continue

                # Cancel any in-flight generation
                cancel_event.clear()
                if active_task and not active_task.done():
                    active_task.cancel()

                active_task = asyncio.create_task(
                    _handle_chat(ws, state, content, conversation_id, cancel_event),
                )

            elif msg_type == "status":
                status_resp = WSResponse(
                    type="status",
                    metadata={
                        "llm_connected": bool(
                            state.llm_client and state.llm_client.is_connected,
                        ),
                        "uptime_s": round(state.uptime_s, 1),
                    },
                )
                await ws.send_text(status_resp.model_dump_json())

            else:
                unknown_resp = WSResponse(
                    type="error",
                    content=f"Unknown message type: {msg_type}",
                )
                await ws.send_text(unknown_resp.model_dump_json())

    except WebSocketDisconnect:
        log.info("ws_disconnected", client=str(ws.client))
    except Exception as exc:
        log.error("ws_error", error=str(exc))
    finally:
        if active_task and not active_task.done():
            active_task.cancel()


def register_websocket(app: FastAPI) -> None:
    """Register the WebSocket endpoint on the app."""
    app.websocket("/ws")(websocket_endpoint)
