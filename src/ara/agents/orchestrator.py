"""LangGraph orchestrator — main agent state graph with nested subgraphs.

The central nervous system of ARA's agent architecture. Routes queries
through intent classification → agent selection → execution → response.

Architecture:
    - LangGraph StateGraph with conditional routing
    - Each agent runs as a node in the graph
    - Router classifies intent and selects the target agent
    - Context manager handles conversation history trimming
    - Supports ReAct-style reasoning loops for complex queries

Graph structure:
    START → classify_intent → route_to_agent → [agent_node] → format_response → END
                                                     ↓
                                              (ReAct loop if needed)
                                              should_continue? → [agent_node]

Usage:
    orchestrator = Orchestrator(settings, llm, router, registry, ...)
    await orchestrator.connect()
    response = await orchestrator.process("What did I write about Mom last week?")
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from langgraph.graph import END, START, StateGraph

from ara.agents.state import AgentState, create_initial_state
from ara.utils.errors import AgentError, RoutingError
from ara.utils.logging import get_logger
from ara.utils.types import ChatMessage, IntentType

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

    from ara.agents.base import AgentRegistry
    from ara.agents.context_manager import ContextManager
    from ara.agents.router import IntentRouter
    from ara.config.settings import AraSettings
    from ara.core.llm import LLMClient

log = get_logger("orchestrator")


# ── Orchestrator ─────────────────────────────────────────────────────────────


class Orchestrator:
    """Main LangGraph orchestrator for ARA's agent system.

    Builds and compiles a LangGraph StateGraph that routes user queries
    through intent classification → agent execution → response formatting.

    Args:
        settings: Application settings.
        llm_client: LLM client for direct generation.
        router: Intent classification router.
        registry: Agent registry with all registered agents.
        context_manager: Conversation context manager.
    """

    def __init__(
        self,
        settings: AraSettings,
        llm_client: LLMClient,
        router: IntentRouter,
        registry: AgentRegistry,
        context_manager: ContextManager | None = None,
    ) -> None:
        self._settings = settings
        self._llm = llm_client
        self._router = router
        self._registry = registry
        self._context_manager = context_manager
        self._graph: CompiledStateGraph | None = None
        self._connected = False

    @property
    def is_connected(self) -> bool:
        """Whether the orchestrator has been initialized."""
        return self._connected

    async def connect(self) -> None:
        """Initialize the orchestrator and compile the LangGraph.

        Connects all registered agents and builds the state graph.
        """
        # Connect all agents
        await self._registry.connect_all()

        # Build and compile the graph
        self._graph = self._build_graph()
        self._connected = True

        log.info(
            "orchestrator_connected",
            agents=self._registry.agent_names,
            agent_count=self._registry.agent_count,
        )

    async def close(self) -> None:
        """Shut down the orchestrator and all agents."""
        await self._registry.close_all()
        self._graph = None
        self._connected = False
        log.info("orchestrator_closed")

    async def process(
        self,
        query: str,
        *,
        conversation_id: str = "",
        conversation_history: list[ChatMessage] | None = None,
        max_steps: int = 5,
    ) -> OrchestratorResult:
        """Process a user query through the full agent pipeline.

        This is the main entry point for the entire agent system.

        Args:
            query: The user's query text.
            conversation_id: Active conversation ID.
            conversation_history: Recent conversation turns.
            max_steps: Maximum reasoning steps for ReAct loops.

        Returns:
            OrchestratorResult with the response and metadata.

        Raises:
            AgentError: If processing fails critically.
        """
        if not self._connected or self._graph is None:
            msg = "Orchestrator not connected. Call connect() first."
            raise AgentError(msg, operation="process")

        start = time.perf_counter()

        # Manage context window
        managed_history = conversation_history or []
        if self._context_manager and managed_history:
            managed_history = await self._context_manager.prepare_context(managed_history)

        # Create initial state
        state = create_initial_state(
            query,
            conversation_id=conversation_id,
            conversation_history=managed_history,
            max_steps=max_steps,
        )

        try:
            # Run the LangGraph
            final_state = await self._graph.ainvoke(state)

            elapsed_ms = (time.perf_counter() - start) * 1000

            result = OrchestratorResult(
                response=final_state.get("response", ""),
                intent=final_state.get("intent", "general"),
                agent=final_state.get("agent", "unknown"),
                confidence=final_state.get("confidence", 0.0),
                memory_context=final_state.get("memory_context", ""),
                graph_context=final_state.get("graph_context", ""),
                tool_calls=final_state.get("tool_calls", []),
                tool_results=final_state.get("tool_results", []),
                synthesis_mode=final_state.get("synthesis_mode", ""),
                error=final_state.get("error", ""),
                metadata=final_state.get("metadata", {}),
                steps=final_state.get("steps", 0),
                elapsed_ms=elapsed_ms,
            )

            log.info(
                "query_processed",
                query=query[:80],
                intent=result.intent,
                agent=result.agent,
                has_response=bool(result.response),
                elapsed_ms=round(elapsed_ms, 1),
            )

            return result

        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000
            log.error(
                "orchestrator_process_failed",
                query=query[:80],
                error=str(exc),
                elapsed_ms=round(elapsed_ms, 1),
            )
            raise AgentError(
                f"Query processing failed: {exc}",
                operation="process",
                details={"query": query[:100], "error": str(exc)},
            ) from exc

    # ── Graph Construction ───────────────────────────────────────────────────

    def _build_graph(self) -> CompiledStateGraph:
        """Build and compile the LangGraph state graph.

        Graph structure:
            START → classify → route → [dynamic agent nodes] → END

        Returns:
            Compiled StateGraph ready for invocation.
        """
        builder = StateGraph(AgentState)

        # Add nodes
        builder.add_node("classify", self._classify_node)
        builder.add_node("execute_agent", self._execute_agent_node)
        builder.add_node("handle_error", self._handle_error_node)

        # Add edges
        builder.add_edge(START, "classify")
        builder.add_conditional_edges(
            "classify",
            self._route_after_classify,
            {
                "execute": "execute_agent",
                "error": "handle_error",
            },
        )
        builder.add_conditional_edges(
            "execute_agent",
            self._route_after_execute,
            {
                "done": END,
                "continue": "execute_agent",
                "error": "handle_error",
            },
        )
        builder.add_edge("handle_error", END)

        return builder.compile()

    # ── Graph Nodes ──────────────────────────────────────────────────────────

    async def _classify_node(self, state: AgentState) -> dict[str, Any]:
        """Node: classify intent and determine target agent.

        Args:
            state: Current graph state.

        Returns:
            State update with intent, agent, and confidence.
        """
        query = state.get("query", "")
        history = state.get("conversation_history", [])

        try:
            result = await self._router.classify(
                query,
                conversation_history=history if history else None,
            )

            return {
                "intent": result.intent.value,
                "agent": result.agent,
                "confidence": result.confidence,
                "metadata": {
                    "classification_reasoning": result.reasoning,
                    "classification_time_ms": result.classification_time_ms,
                },
            }
        except Exception as exc:
            log.error("classification_failed", error=str(exc))
            return {
                "intent": IntentType.GENERAL.value,
                "agent": "rag_agent",
                "confidence": 0.0,
                "error": f"Classification failed: {exc}",
            }

    async def _execute_agent_node(self, state: AgentState) -> dict[str, Any]:
        """Node: execute the selected agent.

        Args:
            state: Current graph state with agent selection.

        Returns:
            State update from agent execution.
        """
        agent_name = state.get("agent", "")
        current_steps = state.get("steps", 0)

        try:
            agent = self._registry.get(agent_name)
            result = await agent.execute(state)

            # Increment step counter
            new_steps = current_steps + 1
            result_dict = dict(result) if isinstance(result, dict) else {}
            result_dict["steps"] = new_steps

            return result_dict

        except RoutingError as exc:
            log.error("agent_not_found", agent=agent_name, error=str(exc))
            return {
                "error": f"Agent '{agent_name}' not found: {exc}",
                "should_continue": False,
            }
        except Exception as exc:
            log.error(
                "agent_execution_failed",
                agent=agent_name,
                error=str(exc),
            )
            return {
                "error": f"Agent '{agent_name}' failed: {exc}",
                "should_continue": False,
            }

    async def _handle_error_node(self, state: AgentState) -> dict[str, Any]:
        """Node: handle errors with a graceful fallback response.

        Args:
            state: Current graph state with error.

        Returns:
            State update with fallback response.
        """
        error = state.get("error", "Unknown error")
        query = state.get("query", "")

        log.warning(
            "error_fallback",
            error=error,
            query=query[:80],
        )

        # Generate a graceful fallback response
        try:
            messages = [
                ChatMessage(
                    role="system",  # type: ignore[arg-type]
                    content=(
                        "You are ARA. Something went wrong while processing the user's "
                        "request. Apologize briefly and naturally (not robotic), explain "
                        "you had trouble, and ask if they'd like to try again or rephrase."
                    ),
                ),
                ChatMessage(role="user", content=query),  # type: ignore[arg-type]
            ]
            response = await self._llm.generate(messages, max_tokens=200)
        except Exception as exc:
            log.warning("error_recovery_llm_failed", error=str(exc))
            response = (
                "Sorry, I ran into an issue processing that. "
                "Could you try rephrasing your question?"
            )

        return {
            "response": response,
            "should_continue": False,
        }

    # ── Routing Functions ────────────────────────────────────────────────────

    @staticmethod
    def _route_after_classify(state: AgentState) -> str:
        """Determine next node after classification.

        Args:
            state: Current state.

        Returns:
            "execute" to run agent, or "error" if classification failed.
        """
        if state.get("error") and not state.get("agent"):
            return "error"
        return "execute"

    @staticmethod
    def _route_after_execute(state: AgentState) -> str:
        """Determine next node after agent execution.

        Supports ReAct-style loops: if the agent sets should_continue=True
        and we haven't exceeded max_steps, loop back to the agent.

        Args:
            state: Current state.

        Returns:
            "done", "continue", or "error".
        """
        if state.get("error") and not state.get("response"):
            return "error"

        should_continue = state.get("should_continue", False)
        steps = state.get("steps", 0)
        max_steps = state.get("max_steps", 5)

        if should_continue and steps < max_steps:
            return "continue"

        return "done"


# ── Result Type ──────────────────────────────────────────────────────────────


class OrchestratorResult:
    """Result from the orchestrator pipeline.

    Contains the response, routing info, retrieval context, tool results,
    and performance metrics.
    """

    __slots__ = (
        "agent",
        "confidence",
        "elapsed_ms",
        "error",
        "graph_context",
        "intent",
        "memory_context",
        "metadata",
        "response",
        "steps",
        "synthesis_mode",
        "tool_calls",
        "tool_results",
    )

    def __init__(
        self,
        response: str,
        intent: str,
        agent: str,
        confidence: float,
        memory_context: str = "",
        graph_context: str = "",
        tool_calls: list[dict[str, Any]] | None = None,
        tool_results: list[dict[str, Any]] | None = None,
        synthesis_mode: str = "",
        error: str = "",
        metadata: dict[str, Any] | None = None,
        steps: int = 0,
        elapsed_ms: float = 0.0,
    ) -> None:
        self.response = response
        self.intent = intent
        self.agent = agent
        self.confidence = confidence
        self.memory_context = memory_context
        self.graph_context = graph_context
        self.tool_calls = tool_calls or []
        self.tool_results = tool_results or []
        self.synthesis_mode = synthesis_mode
        self.error = error
        self.metadata = metadata or {}
        self.steps = steps
        self.elapsed_ms = elapsed_ms

    @property
    def success(self) -> bool:
        """Whether processing completed successfully."""
        return bool(self.response) and not self.error

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary (for API responses)."""
        return {
            "response": self.response,
            "intent": self.intent,
            "agent": self.agent,
            "confidence": self.confidence,
            "synthesis_mode": self.synthesis_mode,
            "error": self.error,
            "steps": self.steps,
            "elapsed_ms": round(self.elapsed_ms, 1),
            "tool_calls": self.tool_calls,
            "tool_results": self.tool_results,
            "metadata": self.metadata,
        }
