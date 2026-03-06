"""Base agent class and plugin interface for ARA's agent system.

Defines the abstract contract that all agents must implement, plus the
plugin registration system for hot-loadable agents. Each agent declares
its capabilities, required permissions, and supported intent types.

Architecture:
    - BaseAgent: Abstract class with lifecycle hooks (connect, close, execute)
    - AgentCapability: Declares what an agent can do (for routing + UI)
    - AgentRegistry: Manages registered agents, handles hot-loading
    - Each agent runs as a LangGraph subgraph node

Design principles:
    - Every agent is independently testable
    - Agents declare permissions (what system resources they need)
    - Agents are stateless between queries (state flows through LangGraph)
    - Hot-loading: new agents can be added/removed at runtime
"""

from __future__ import annotations

import importlib
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from ara.utils.errors import AgentError, RoutingError
from ara.utils.logging import get_logger
from ara.utils.types import IntentType  # noqa: TC001 — used at runtime in dataclass fields

if TYPE_CHECKING:
    from ara.agents.state import AgentState
    from ara.config.settings import AraSettings

log = get_logger("agents")


# ── Permission System ────────────────────────────────────────────────────────


class AgentPermission(StrEnum):
    """Permissions that agents can request.

    Each agent declares which permissions it needs. The orchestrator
    checks these before allowing execution.
    """

    MEMORY_READ = "memory_read"  # Read from vector store / knowledge graph
    MEMORY_WRITE = "memory_write"  # Write to vector store / knowledge graph
    LLM_GENERATE = "llm_generate"  # Generate text via LLM
    WEB_SEARCH = "web_search"  # Search the internet
    WEB_BROWSE = "web_browse"  # Browse web pages
    FILE_READ = "file_read"  # Read files on the filesystem
    FILE_WRITE = "file_write"  # Write files on the filesystem
    DESKTOP_CONTROL = "desktop_control"  # Mouse, keyboard, window control
    SYSTEM_EXECUTE = "system_execute"  # Run system commands / PowerShell
    MEDIA_CONTROL = "media_control"  # Control media playback


# ── Agent Capability ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class AgentCapability:
    """Declares what an agent can do.

    Used by the registry for routing decisions and by the UI
    to show available capabilities.

    Attributes:
        name: Unique agent identifier (e.g., "rag_agent").
        display_name: Human-readable name (e.g., "Knowledge & Memory").
        description: What this agent does (shown in UI/help).
        supported_intents: Which intent types this agent handles.
        required_permissions: What system access this agent needs.
        priority: Higher priority agents are preferred when multiple
            agents can handle the same intent (default 100).
        enabled: Whether this agent is currently active.
    """

    name: str
    display_name: str
    description: str
    supported_intents: frozenset[IntentType]
    required_permissions: frozenset[AgentPermission]
    priority: int = 100
    enabled: bool = True


# ── Base Agent ───────────────────────────────────────────────────────────────


class BaseAgent(ABC):
    """Abstract base class for all ARA agents.

    Subclasses must implement:
        - capability: Property returning AgentCapability
        - _execute: The core agent logic
        - connect / close: Lifecycle hooks

    The `execute` method handles timing, logging, error handling,
    and delegates to `_execute` for the actual work.
    """

    def __init__(self, settings: AraSettings) -> None:
        self._settings = settings
        self._connected = False

    @property
    @abstractmethod
    def capability(self) -> AgentCapability:
        """Return this agent's capability declaration."""

    @property
    def name(self) -> str:
        """Agent's unique name."""
        return self.capability.name

    @property
    def is_connected(self) -> bool:
        """Whether the agent has been initialized."""
        return self._connected

    async def connect(self) -> None:
        """Initialize agent resources (models, connections, etc.).

        Override this in subclasses that need setup. The base implementation
        just marks the agent as connected.
        """
        self._connected = True
        log.info("agent_connected", agent=self.name)

    async def close(self) -> None:
        """Release agent resources.

        Override this in subclasses that need cleanup. The base implementation
        just marks the agent as disconnected.
        """
        self._connected = False
        log.info("agent_closed", agent=self.name)

    async def execute(self, state: AgentState) -> AgentState:
        """Execute the agent on the given state.

        Handles timing, logging, and error handling. Delegates to
        `_execute` for the actual agent logic.

        Args:
            state: Current LangGraph state.

        Returns:
            Updated state with agent's output.

        Raises:
            AgentError: If execution fails.
        """
        start = time.perf_counter()
        agent_name = self.name

        if not self._connected:
            raise AgentError(
                f"Agent '{agent_name}' not connected. Call connect() first.",
                operation="execute",
                details={"agent": agent_name},
            )

        log.info(
            "agent_execute_start",
            agent=agent_name,
            query=state.get("query", "")[:80],
        )

        try:
            result_state = await self._execute(state)
            elapsed_ms = (time.perf_counter() - start) * 1000

            log.info(
                "agent_execute_complete",
                agent=agent_name,
                elapsed_ms=round(elapsed_ms, 1),
                has_response=bool(result_state.get("response")),
                has_error=bool(result_state.get("error")),
            )
            return result_state

        except AgentError:
            raise
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000
            log.error(
                "agent_execute_failed",
                agent=agent_name,
                error=str(exc),
                elapsed_ms=round(elapsed_ms, 1),
            )
            raise AgentError(
                f"Agent '{agent_name}' execution failed: {exc}",
                operation="execute",
                details={"agent": agent_name, "error": str(exc)},
            ) from exc

    @abstractmethod
    async def _execute(self, state: AgentState) -> AgentState:
        """Core agent logic — implement in subclasses.

        Args:
            state: Current state from the orchestrator graph.

        Returns:
            Partial state update (only fields that changed).
        """


# ── Agent Registry ───────────────────────────────────────────────────────────


class AgentRegistry:
    """Registry for managing agent instances.

    Handles agent registration, lookup by name or intent, hot-loading
    of new agents, and lifecycle management (connect/close all).

    Usage:
        registry = AgentRegistry(settings)
        registry.register(rag_agent)
        registry.register(memory_agent)
        await registry.connect_all()

        agent = registry.get("rag_agent")
        agents = registry.get_for_intent(IntentType.REMINISCING)

        await registry.close_all()
    """

    def __init__(self, settings: AraSettings) -> None:
        self._settings = settings
        self._agents: dict[str, BaseAgent] = {}

    @property
    def agent_names(self) -> list[str]:
        """List of registered agent names."""
        return list(self._agents.keys())

    @property
    def agent_count(self) -> int:
        """Number of registered agents."""
        return len(self._agents)

    def register(self, agent: BaseAgent) -> None:
        """Register an agent instance.

        Args:
            agent: The agent to register.

        Raises:
            AgentError: If an agent with the same name is already registered.
        """
        name = agent.name
        if name in self._agents:
            raise AgentError(
                f"Agent '{name}' is already registered",
                operation="register",
                details={"agent": name, "existing": list(self._agents)},
            )
        self._agents[name] = agent
        log.info(
            "agent_registered",
            agent=name,
            intents=[i.value for i in agent.capability.supported_intents],
            permissions=[p.value for p in agent.capability.required_permissions],
        )

    def unregister(self, name: str) -> BaseAgent | None:
        """Unregister and return an agent by name.

        Args:
            name: The agent name to unregister.

        Returns:
            The removed agent, or None if not found.
        """
        agent = self._agents.pop(name, None)
        if agent:
            log.info("agent_unregistered", agent=name)
        return agent

    def get(self, name: str) -> BaseAgent:
        """Get a registered agent by name.

        Args:
            name: The agent name to look up.

        Returns:
            The registered agent.

        Raises:
            RoutingError: If no agent with that name is registered.
        """
        agent = self._agents.get(name)
        if agent is None:
            raise RoutingError(
                f"No agent registered with name '{name}'",
                details={"name": name, "available": list(self._agents)},
            )
        return agent

    def get_for_intent(self, intent: IntentType) -> list[BaseAgent]:
        """Get all agents that support a given intent type.

        Returns agents sorted by priority (highest first).

        Args:
            intent: The intent type to match.

        Returns:
            List of agents that handle this intent, sorted by priority.
        """
        matching = [
            agent
            for agent in self._agents.values()
            if intent in agent.capability.supported_intents and agent.capability.enabled
        ]
        return sorted(matching, key=lambda a: a.capability.priority, reverse=True)

    def get_capabilities(self) -> list[AgentCapability]:
        """Get capability declarations for all registered agents."""
        return [agent.capability for agent in self._agents.values()]

    def get_tool_names(self) -> list[str]:
        """Get agent names for system prompt tool availability."""
        return [
            agent.capability.display_name
            for agent in self._agents.values()
            if agent.capability.enabled
        ]

    async def connect_all(self) -> None:
        """Connect all registered agents.

        Raises:
            AgentError: If any agent fails to connect (with details).
        """
        for name, agent in self._agents.items():
            try:
                await agent.connect()
            except Exception as exc:
                raise AgentError(
                    f"Failed to connect agent '{name}': {exc}",
                    operation="connect",
                    details={"agent": name, "error": str(exc)},
                ) from exc

    async def close_all(self) -> None:
        """Close all registered agents. Logs errors but doesn't raise."""
        for name, agent in self._agents.items():
            try:
                await agent.close()
            except Exception as exc:
                log.error(
                    "agent_close_failed",
                    agent=name,
                    error=str(exc),
                )

    def load_plugin(self, module_path: str, class_name: str) -> BaseAgent:
        """Hot-load an agent from a Python module path.

        Imports the module, instantiates the agent class, and registers it.

        Args:
            module_path: Python module path (e.g., "ara.agents.custom_agent").
            class_name: The agent class name to instantiate.

        Returns:
            The newly loaded and registered agent.

        Raises:
            AgentError: If loading fails.
        """
        try:
            module = importlib.import_module(module_path)
            agent_cls = getattr(module, class_name)

            if not issubclass(agent_cls, BaseAgent):
                raise AgentError(
                    f"Class '{class_name}' is not a BaseAgent subclass",
                    operation="load_plugin",
                    details={"module": module_path, "class": class_name},
                )

            agent = agent_cls(self._settings)
            self.register(agent)

            log.info(
                "plugin_loaded",
                module=module_path,
                class_name=class_name,
                agent=agent.name,
            )
            return agent

        except AgentError:
            raise
        except Exception as exc:
            raise AgentError(
                f"Failed to load plugin '{module_path}:{class_name}': {exc}",
                operation="load_plugin",
                details={"module": module_path, "class": class_name, "error": str(exc)},
            ) from exc
