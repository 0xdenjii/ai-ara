"""Agent state definitions for LangGraph orchestration.

Defines the shared state that flows through the LangGraph state graph.
All agents read from and write to this state. The state is a TypedDict
that LangGraph uses for automatic state management.

Architecture:
    - AgentState: The main state flowing through the orchestrator graph
    - SubAgentState: Subset of state for individual agent subgraphs
    - State is immutable per-step — LangGraph handles merging via reducers
"""

from __future__ import annotations

import operator
from datetime import datetime
from typing import Annotated, Any, TypedDict

from ara.utils.types import ChatMessage, ToolCall, ToolResult

# ── Reducers ─────────────────────────────────────────────────────────────────
# LangGraph uses reducers to merge state updates. We use `operator.add`
# for list fields (appends) and simple replacement for scalar fields.


def _replace(existing: Any, new: Any) -> Any:
    """Reducer that replaces the existing value with the new value."""
    return new


# ── Agent State (Main Orchestrator) ──────────────────────────────────────────


class AgentState(TypedDict, total=False):
    """State flowing through the main LangGraph orchestrator.

    LangGraph manages this state automatically. Each node receives the
    current state and returns partial updates. Lists use `add` reducer
    (append), scalars use replacement.

    Fields:
        query: The user's original query text.
        conversation_id: Active conversation ID for persistence.
        conversation_history: Recent conversation turns for context.
        intent: Classified intent type from the router.
        agent: Target agent name from routing.
        confidence: Router's classification confidence.

        memory_context: Retrieved context from RAG/memory (formatted text).
        retrieval_results: Raw retrieval results for synthesis.
        graph_context: Knowledge graph context (entity summaries).

        tool_calls: Accumulated tool invocations.
        tool_results: Accumulated tool execution results.

        response: Final generated response text.
        synthesis_mode: Which synthesis strategy was used.

        error: Error message if something went wrong.
        metadata: Arbitrary metadata for debugging/logging.

        steps: Number of reasoning steps taken (for ReAct loop control).
        max_steps: Maximum allowed reasoning steps.
        should_continue: Whether the agent loop should continue.
    """

    # Input
    query: str
    conversation_id: str
    conversation_history: Annotated[list[ChatMessage], operator.add]
    timestamp: str  # ISO format

    # Routing
    intent: str  # IntentType.value
    agent: str
    confidence: float

    # Memory / Retrieval
    memory_context: str
    retrieval_results: list[dict[str, Any]]
    graph_context: str

    # Tool calls
    tool_calls: Annotated[list[dict[str, Any]], operator.add]
    tool_results: Annotated[list[dict[str, Any]], operator.add]

    # Output
    response: str
    synthesis_mode: str

    # Control flow
    error: str
    metadata: dict[str, Any]
    steps: int
    max_steps: int
    should_continue: bool


# ── Helper Functions ─────────────────────────────────────────────────────────


def create_initial_state(
    query: str,
    *,
    conversation_id: str = "",
    conversation_history: list[ChatMessage] | None = None,
    max_steps: int = 5,
) -> AgentState:
    """Create the initial state for a new query through the orchestrator.

    Args:
        query: The user's query text.
        conversation_id: Active conversation ID.
        conversation_history: Recent conversation turns.
        max_steps: Maximum reasoning steps for ReAct loops.

    Returns:
        Initial AgentState dict ready for LangGraph invocation.
    """
    return AgentState(
        query=query,
        conversation_id=conversation_id,
        conversation_history=conversation_history or [],
        timestamp=datetime.now().isoformat(),
        intent="",
        agent="",
        confidence=0.0,
        memory_context="",
        retrieval_results=[],
        graph_context="",
        tool_calls=[],
        tool_results=[],
        response="",
        synthesis_mode="",
        error="",
        metadata={},
        steps=0,
        max_steps=max_steps,
        should_continue=True,
    )


def tool_call_to_dict(tc: ToolCall) -> dict[str, Any]:
    """Convert a ToolCall dataclass to a serializable dict for state."""
    return {
        "tool_name": tc.tool_name,
        "parameters": tc.parameters,
        "call_id": tc.call_id,
    }


def tool_result_to_dict(tr: ToolResult) -> dict[str, Any]:
    """Convert a ToolResult dataclass to a serializable dict for state."""
    return {
        "call_id": tr.call_id,
        "tool_name": tr.tool_name,
        "success": tr.success,
        "output": tr.output,
        "error": tr.error,
        "execution_time_ms": tr.execution_time_ms,
    }


def dict_to_tool_call(d: dict[str, Any]) -> ToolCall:
    """Reconstruct a ToolCall from a state dict."""
    return ToolCall(
        tool_name=d["tool_name"],
        parameters=d["parameters"],
        call_id=d.get("call_id", ""),
    )


def dict_to_tool_result(d: dict[str, Any]) -> ToolResult:
    """Reconstruct a ToolResult from a state dict."""
    return ToolResult(
        call_id=d["call_id"],
        tool_name=d["tool_name"],
        success=d["success"],
        output=d["output"],
        error=d.get("error"),
        execution_time_ms=d.get("execution_time_ms", 0.0),
    )
