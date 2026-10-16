"""Tests for ARA agent orchestration — router, agents, registry, state, context manager.

Covers:
    - IntentRouter: LLM classification parsing, fallbacks, overrides
    - BaseAgent / AgentRegistry: registration, lifecycle, intent lookup, plugins
    - AgentState: creation, serialization helpers
    - ContextManager: sliding window, summarization
    - RAGAgent: retrieval → synthesis flow with mocked dependencies
    - MemoryAgent: graph context building
    - ToolAgent: tool call parsing and execution
    - WebSearchAgent: search and synthesis
    - WebBrowseAgent: URL extraction and page fetching
    - Orchestrator: full graph compilation and routing
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from ara.agents.base import AgentCapability, AgentPermission, AgentRegistry, BaseAgent
from ara.agents.context_manager import ContextManager
from ara.agents.router import IntentRouter, RouteOverride
from ara.agents.state import (
    AgentState,
    create_initial_state,
    dict_to_tool_call,
    dict_to_tool_result,
    tool_call_to_dict,
    tool_result_to_dict,
)
from ara.utils.errors import AgentError, RoutingError
from ara.utils.types import (
    ChatMessage,
    IntentType,
    MessageRole,
    ToolCall,
    ToolResult,
)

# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture()
def mock_settings():
    """Create mock settings for agent tests."""
    settings = MagicMock()
    settings.llm.primary_model = "qwen3:30b-a3b"
    settings.llm.ollama_base_url = "http://localhost:11434"
    settings.llm.temperature = 0.8
    settings.llm.max_response_tokens = 2048
    settings.llm.primary_model_context_length = 32768
    settings.personality.humor_dial = 7
    settings.personality.voice_identity = "female"
    settings.personality.auto_dial_down_on_distress = True
    settings.personality.distress_dial_reduction = 2
    settings.internet.internet_enabled = True
    settings.internet.max_research_steps = 5
    settings.desktop.require_confirmation = True
    settings.desktop.enabled = True
    settings.memory.retrieval_top_k = 20
    settings.memory.rerank_top_k = 5
    return settings


@pytest.fixture()
def mock_llm():
    """Create mock LLM client."""
    llm = AsyncMock()
    llm.generate = AsyncMock(return_value="test response")
    llm.generate_stream = AsyncMock()
    return llm


@pytest.fixture()
def mock_retrieval():
    """Create mock retrieval pipeline."""
    retrieval = MagicMock()
    retrieval.retrieve = MagicMock(
        return_value=MagicMock(
            results=[],
            query="test",
            query_type=IntentType.GENERAL,
            total_candidates=0,
        )
    )
    retrieval.assemble_context = MagicMock(return_value="")
    return retrieval


class ConcreteAgent(BaseAgent):
    """Concrete test agent for testing BaseAgent."""

    def __init__(self, settings: Any, name: str = "test_agent") -> None:
        super().__init__(settings)
        self._name = name

    @property
    def capability(self) -> AgentCapability:
        return AgentCapability(
            name=self._name,
            display_name="Test Agent",
            description="A test agent",
            supported_intents=frozenset({IntentType.GENERAL}),
            required_permissions=frozenset({AgentPermission.LLM_GENERATE}),
        )

    async def _execute(self, state: AgentState) -> AgentState:
        return {"response": f"handled by {self._name}"}  # type: ignore[return-value]


# ═══════════════════════════════════════════════════════════════════════════
# AgentState Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestCreateInitialState:
    """Tests for create_initial_state()."""

    def test_basic_creation(self):
        state = create_initial_state("hello")
        assert state["query"] == "hello"
        assert state["conversation_id"] == ""
        assert state["conversation_history"] == []
        assert state["intent"] == ""
        assert state["agent"] == ""
        assert state["confidence"] == 0.0
        assert state["response"] == ""
        assert state["error"] == ""
        assert state["steps"] == 0
        assert state["max_steps"] == 5
        assert state["should_continue"] is False

    def test_with_conversation_id(self):
        state = create_initial_state("hello", conversation_id="conv123")
        assert state["conversation_id"] == "conv123"

    def test_with_history(self):
        history = [ChatMessage(role=MessageRole.USER, content="hi")]
        state = create_initial_state("hello", conversation_history=history)
        assert len(state["conversation_history"]) == 1

    def test_with_custom_max_steps(self):
        state = create_initial_state("hello", max_steps=10)
        assert state["max_steps"] == 10

    def test_timestamp_is_set(self):
        state = create_initial_state("hello")
        assert state["timestamp"]  # ISO format string


class TestToolCallSerialization:
    """Tests for tool call/result serialization helpers."""

    def test_tool_call_to_dict(self):
        tc = ToolCall(tool_name="open_app", parameters={"name": "chrome"}, call_id="abc123")
        d = tool_call_to_dict(tc)
        assert d["tool_name"] == "open_app"
        assert d["parameters"] == {"name": "chrome"}
        assert d["call_id"] == "abc123"

    def test_tool_result_to_dict(self):
        tr = ToolResult(
            call_id="abc123",
            tool_name="open_app",
            success=True,
            output="opened",
            error=None,
            execution_time_ms=50.0,
        )
        d = tool_result_to_dict(tr)
        assert d["success"] is True
        assert d["output"] == "opened"
        assert d["execution_time_ms"] == 50.0

    def test_dict_to_tool_call(self):
        d = {"tool_name": "run_cmd", "parameters": {"cmd": "ls"}, "call_id": "xyz"}
        tc = dict_to_tool_call(d)
        assert tc.tool_name == "run_cmd"
        assert tc.parameters == {"cmd": "ls"}

    def test_dict_to_tool_result(self):
        d = {
            "call_id": "xyz",
            "tool_name": "run_cmd",
            "success": False,
            "output": "",
            "error": "permission denied",
        }
        tr = dict_to_tool_result(d)
        assert tr.success is False
        assert tr.error == "permission denied"

    def test_roundtrip_tool_call(self):
        tc = ToolCall(tool_name="screenshot", parameters={})
        d = tool_call_to_dict(tc)
        tc2 = dict_to_tool_call(d)
        assert tc2.tool_name == tc.tool_name
        assert tc2.parameters == tc.parameters


# ═══════════════════════════════════════════════════════════════════════════
# Router Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestIntentRouterParsing:
    """Tests for IntentRouter._parse_classification()."""

    def _make_router(self, mock_llm, mock_settings):
        return IntentRouter(mock_llm, mock_settings)

    def test_parse_valid_json(self, mock_llm, mock_settings):
        router = self._make_router(mock_llm, mock_settings)
        result = router._parse_classification(
            '{"intent": "factual", "confidence": 0.95, "reasoning": "asking a fact"}',
            "test query",
        )
        assert result.intent == IntentType.FACTUAL
        assert result.confidence == 0.95
        assert result.agent == "rag_agent"

    def test_parse_json_in_markdown_fences(self, mock_llm, mock_settings):
        router = self._make_router(mock_llm, mock_settings)
        result = router._parse_classification(
            '```json\n{"intent": "emotional", "confidence": 0.8, "reasoning": "expressing feelings"}\n```',
            "I feel sad",
        )
        assert result.intent == IntentType.EMOTIONAL

    def test_parse_json_with_surrounding_text(self, mock_llm, mock_settings):
        router = self._make_router(mock_llm, mock_settings)
        result = router._parse_classification(
            'Here is my classification:\n{"intent": "planning", "confidence": 0.7, "reasoning": "making plans"}',
            "plan my week",
        )
        assert result.intent == IntentType.PLANNING

    def test_parse_unknown_intent_falls_back(self, mock_llm, mock_settings):
        router = self._make_router(mock_llm, mock_settings)
        result = router._parse_classification(
            '{"intent": "unknown_type", "confidence": 0.5, "reasoning": "unknown"}',
            "test",
        )
        assert result.intent == IntentType.GENERAL
        assert result.confidence <= 0.3

    def test_parse_invalid_json_falls_back(self, mock_llm, mock_settings):
        router = self._make_router(mock_llm, mock_settings)
        result = router._parse_classification(
            "this is not json at all",
            "test",
        )
        assert result.intent == IntentType.GENERAL
        assert result.confidence == 0.0

    def test_parse_low_confidence_downgrades(self, mock_llm, mock_settings):
        router = self._make_router(mock_llm, mock_settings)
        result = router._parse_classification(
            '{"intent": "web_search", "confidence": 0.1, "reasoning": "maybe?"}',
            "test",
        )
        assert result.intent == IntentType.GENERAL  # Downgraded

    def test_parse_all_intent_types(self, mock_llm, mock_settings):
        router = self._make_router(mock_llm, mock_settings)
        for intent_type in IntentType:
            result = router._parse_classification(
                f'{{"intent": "{intent_type.value}", "confidence": 0.9, "reasoning": "test"}}',
                "test",
            )
            assert result.intent == intent_type

    def test_confidence_clamped_to_range(self, mock_llm, mock_settings):
        router = self._make_router(mock_llm, mock_settings)
        result = router._parse_classification(
            '{"intent": "factual", "confidence": 5.0, "reasoning": "test"}',
            "test",
        )
        assert result.confidence == 1.0

    def test_missing_fields_use_defaults(self, mock_llm, mock_settings):
        router = self._make_router(mock_llm, mock_settings)
        result = router._parse_classification('{"intent": "factual"}', "test")
        assert result.confidence == 0.5
        assert "No reasoning" in result.reasoning


class TestIntentRouterClassify:
    """Tests for IntentRouter.classify()."""

    @pytest.mark.asyncio
    async def test_classify_with_override(self, mock_llm, mock_settings):
        router = IntentRouter(mock_llm, mock_settings)
        override = RouteOverride(
            intent=IntentType.DESKTOP,
            agent="tool_agent",
            reason="user said open app",
        )
        result = await router.classify("open chrome", override=override)
        assert result.intent == IntentType.DESKTOP
        assert result.agent == "tool_agent"
        assert result.confidence == 1.0

    @pytest.mark.asyncio
    async def test_classify_calls_llm(self, mock_llm, mock_settings):
        mock_llm.generate.return_value = (
            '{"intent": "factual", "confidence": 0.9, "reasoning": "question"}'
        )
        router = IntentRouter(mock_llm, mock_settings)
        result = await router.classify("what is python?")
        assert result.intent == IntentType.FACTUAL
        mock_llm.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_classify_fallback_on_llm_error(self, mock_llm, mock_settings):
        mock_llm.generate.side_effect = RuntimeError("connection lost")
        router = IntentRouter(mock_llm, mock_settings)
        result = await router.classify("test query")
        assert result.intent == IntentType.GENERAL
        assert result.confidence == 0.0

    @pytest.mark.asyncio
    async def test_classify_with_history(self, mock_llm, mock_settings):
        mock_llm.generate.return_value = (
            '{"intent": "reminiscing", "confidence": 0.85, "reasoning": "follow-up"}'
        )
        router = IntentRouter(mock_llm, mock_settings)
        history = [
            ChatMessage(role=MessageRole.USER, content="tell me about Mom"),
            ChatMessage(role=MessageRole.ASSISTANT, content="Your mom is..."),
        ]
        result = await router.classify("what else?", conversation_history=history)
        assert result.intent == IntentType.REMINISCING

    @pytest.mark.asyncio
    async def test_classify_batch(self, mock_llm, mock_settings):
        mock_llm.generate.return_value = (
            '{"intent": "general", "confidence": 0.8, "reasoning": "chat"}'
        )
        router = IntentRouter(mock_llm, mock_settings)
        results = await router.classify_batch(["hi", "hello", "yo"])
        assert len(results) == 3

    def test_route_table_defaults(self, mock_llm, mock_settings):
        router = IntentRouter(mock_llm, mock_settings)
        table = router.route_table
        assert table[IntentType.FACTUAL] == "rag_agent"
        assert table[IntentType.REMINISCING] == "memory_agent"
        assert table[IntentType.DESKTOP] == "tool_agent"
        assert table[IntentType.WEB_SEARCH] == "web_search_agent"
        assert table[IntentType.RESEARCH] == "research_agent"

    def test_update_route(self, mock_llm, mock_settings):
        router = IntentRouter(mock_llm, mock_settings)
        router.update_route(IntentType.FACTUAL, "custom_agent")
        assert router.route_table[IntentType.FACTUAL] == "custom_agent"

    def test_get_agent_for_intent(self, mock_llm, mock_settings):
        router = IntentRouter(mock_llm, mock_settings)
        assert router.get_agent_for_intent(IntentType.FACTUAL) == "rag_agent"

    def test_get_agent_for_missing_intent_raises(self, mock_llm, mock_settings):
        router = IntentRouter(mock_llm, mock_settings)
        # Manually clear the route table to test missing intent
        router._route_table.clear()
        with pytest.raises(RoutingError):
            router.get_agent_for_intent(IntentType.FACTUAL)


# ═══════════════════════════════════════════════════════════════════════════
# BaseAgent / AgentRegistry Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestBaseAgent:
    """Tests for BaseAgent abstract class."""

    @pytest.mark.asyncio
    async def test_connect_marks_connected(self, mock_settings):
        agent = ConcreteAgent(mock_settings)
        assert not agent.is_connected
        await agent.connect()
        assert agent.is_connected

    @pytest.mark.asyncio
    async def test_close_marks_disconnected(self, mock_settings):
        agent = ConcreteAgent(mock_settings)
        await agent.connect()
        await agent.close()
        assert not agent.is_connected

    @pytest.mark.asyncio
    async def test_execute_requires_connection(self, mock_settings):
        agent = ConcreteAgent(mock_settings)
        state = create_initial_state("test")
        with pytest.raises(AgentError, match="not connected"):
            await agent.execute(state)

    @pytest.mark.asyncio
    async def test_execute_returns_updated_state(self, mock_settings):
        agent = ConcreteAgent(mock_settings)
        await agent.connect()
        state = create_initial_state("test")
        result = await agent.execute(state)
        assert result["response"] == "handled by test_agent"

    def test_agent_name(self, mock_settings):
        agent = ConcreteAgent(mock_settings, name="my_agent")
        assert agent.name == "my_agent"

    def test_capability_properties(self, mock_settings):
        agent = ConcreteAgent(mock_settings)
        cap = agent.capability
        assert cap.name == "test_agent"
        assert cap.display_name == "Test Agent"
        assert IntentType.GENERAL in cap.supported_intents


class TestAgentRegistry:
    """Tests for AgentRegistry."""

    def test_register_agent(self, mock_settings):
        registry = AgentRegistry(mock_settings)
        agent = ConcreteAgent(mock_settings)
        registry.register(agent)
        assert "test_agent" in registry.agent_names
        assert registry.agent_count == 1

    def test_register_duplicate_raises(self, mock_settings):
        registry = AgentRegistry(mock_settings)
        agent1 = ConcreteAgent(mock_settings)
        agent2 = ConcreteAgent(mock_settings)
        registry.register(agent1)
        with pytest.raises(AgentError, match="already registered"):
            registry.register(agent2)

    def test_unregister_agent(self, mock_settings):
        registry = AgentRegistry(mock_settings)
        agent = ConcreteAgent(mock_settings)
        registry.register(agent)
        removed = registry.unregister("test_agent")
        assert removed is agent
        assert registry.agent_count == 0

    def test_unregister_nonexistent_returns_none(self, mock_settings):
        registry = AgentRegistry(mock_settings)
        assert registry.unregister("nonexistent") is None

    def test_get_agent(self, mock_settings):
        registry = AgentRegistry(mock_settings)
        agent = ConcreteAgent(mock_settings)
        registry.register(agent)
        assert registry.get("test_agent") is agent

    def test_get_nonexistent_raises(self, mock_settings):
        registry = AgentRegistry(mock_settings)
        with pytest.raises(RoutingError, match="No agent registered"):
            registry.get("nonexistent")

    def test_get_for_intent(self, mock_settings):
        registry = AgentRegistry(mock_settings)
        agent = ConcreteAgent(mock_settings)
        registry.register(agent)
        agents = registry.get_for_intent(IntentType.GENERAL)
        assert len(agents) == 1
        assert agents[0] is agent

    def test_get_for_intent_no_match(self, mock_settings):
        registry = AgentRegistry(mock_settings)
        agent = ConcreteAgent(mock_settings)
        registry.register(agent)
        agents = registry.get_for_intent(IntentType.VISION)
        assert len(agents) == 0

    def test_get_capabilities(self, mock_settings):
        registry = AgentRegistry(mock_settings)
        registry.register(ConcreteAgent(mock_settings, "a1"))
        registry.register(ConcreteAgent(mock_settings, "a2"))
        caps = registry.get_capabilities()
        assert len(caps) == 2

    def test_get_tool_names(self, mock_settings):
        registry = AgentRegistry(mock_settings)
        registry.register(ConcreteAgent(mock_settings))
        names = registry.get_tool_names()
        assert "Test Agent" in names

    @pytest.mark.asyncio
    async def test_connect_all(self, mock_settings):
        registry = AgentRegistry(mock_settings)
        a1 = ConcreteAgent(mock_settings, "a1")
        a2 = ConcreteAgent(mock_settings, "a2")
        registry.register(a1)
        registry.register(a2)
        await registry.connect_all()
        assert a1.is_connected
        assert a2.is_connected

    @pytest.mark.asyncio
    async def test_close_all(self, mock_settings):
        registry = AgentRegistry(mock_settings)
        a1 = ConcreteAgent(mock_settings, "a1")
        registry.register(a1)
        await registry.connect_all()
        await registry.close_all()
        assert not a1.is_connected

    def test_load_plugin_invalid_module(self, mock_settings):
        registry = AgentRegistry(mock_settings)
        with pytest.raises(AgentError, match="Failed to load"):
            registry.load_plugin("nonexistent.module", "SomeAgent")

    def test_priority_sorting(self, mock_settings):
        registry = AgentRegistry(mock_settings)

        class LowPriority(ConcreteAgent):
            @property
            def capability(self):
                return AgentCapability(
                    name="low",
                    display_name="Low",
                    description="Low priority",
                    supported_intents=frozenset({IntentType.GENERAL}),
                    required_permissions=frozenset(),
                    priority=50,
                )

        class HighPriority(ConcreteAgent):
            @property
            def capability(self):
                return AgentCapability(
                    name="high",
                    display_name="High",
                    description="High priority",
                    supported_intents=frozenset({IntentType.GENERAL}),
                    required_permissions=frozenset(),
                    priority=200,
                )

        registry.register(LowPriority(mock_settings, "low"))
        registry.register(HighPriority(mock_settings, "high"))

        agents = registry.get_for_intent(IntentType.GENERAL)
        assert agents[0].name == "high"
        assert agents[1].name == "low"


# ═══════════════════════════════════════════════════════════════════════════
# ContextManager Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestContextManager:
    """Tests for ContextManager sliding window + summarization."""

    @pytest.mark.asyncio
    async def test_short_history_passes_through(self, mock_llm, mock_settings):
        manager = ContextManager(mock_settings, mock_llm)
        history = [
            ChatMessage(role=MessageRole.USER, content="hi"),
            ChatMessage(role=MessageRole.ASSISTANT, content="hello"),
        ]
        result = await manager.prepare_context(history)
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_long_history_gets_trimmed(self, mock_llm, mock_settings):
        mock_settings.llm.primary_model_context_length = 100  # Very small
        mock_settings.llm.max_response_tokens = 20
        mock_llm.generate.return_value = "Summary of conversation"

        manager = ContextManager(mock_settings, mock_llm)
        # Create a long history that won't fit
        history = [ChatMessage(role=MessageRole.USER, content="x" * 50) for _ in range(20)]
        result = await manager.prepare_context(
            history,
            system_prompt_tokens=10,
            retrieval_context_tokens=10,
        )
        # Should have fewer messages than original
        assert len(result) < len(history)

    def test_estimate_tokens(self):
        assert ContextManager._estimate_tokens("hello world") >= 1
        assert ContextManager._estimate_tokens("a" * 100) == 25
        assert ContextManager._estimate_tokens("") == 1

    def test_reset_clears_summary(self, mock_llm, mock_settings):
        manager = ContextManager(mock_settings, mock_llm)
        manager._running_summary = "some old summary"
        manager.reset()
        assert manager.running_summary == ""

    @pytest.mark.asyncio
    async def test_summary_prepended(self, mock_llm, mock_settings):
        manager = ContextManager(mock_settings, mock_llm)
        manager._running_summary = "previously discussed weather"
        history = [ChatMessage(role=MessageRole.USER, content="hi")]
        result = await manager.prepare_context(history)
        assert result[0].role == MessageRole.SYSTEM
        assert "previously discussed weather" in result[0].content


# ═══════════════════════════════════════════════════════════════════════════
# RAGAgent Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestRAGAgent:
    """Tests for RAGAgent."""

    @pytest.mark.asyncio
    async def test_empty_query_returns_error(self, mock_llm, mock_settings, mock_retrieval):
        from ara.agents.rag_agent import RAGAgent

        agent = RAGAgent(mock_settings, mock_llm, mock_retrieval)
        await agent.connect()
        state = create_initial_state("")
        result = await agent.execute(state)
        assert result.get("error")

    @pytest.mark.asyncio
    async def test_direct_generation_when_no_context(self, mock_llm, mock_settings, mock_retrieval):
        from ara.agents.rag_agent import RAGAgent

        mock_retrieval.retrieve.return_value = MagicMock(
            results=[], query="test", query_type=IntentType.GENERAL
        )
        mock_llm.generate.return_value = "I'm here to help!"

        agent = RAGAgent(mock_settings, mock_llm, mock_retrieval)
        await agent.connect()
        state = create_initial_state("hello")
        result = await agent.execute(state)
        assert result["response"] == "I'm here to help!"
        assert result["synthesis_mode"] == "direct"

    @pytest.mark.asyncio
    async def test_retrieval_failure_falls_back(self, mock_llm, mock_settings, mock_retrieval):
        from ara.agents.rag_agent import RAGAgent

        mock_retrieval.retrieve.side_effect = RuntimeError("db error")
        mock_llm.generate.return_value = "fallback response"

        agent = RAGAgent(mock_settings, mock_llm, mock_retrieval)
        await agent.connect()
        state = create_initial_state("test query")
        result = await agent.execute(state)
        assert result["response"] == "fallback response"

    def test_capability(self, mock_settings, mock_llm, mock_retrieval):
        from ara.agents.rag_agent import RAGAgent

        agent = RAGAgent(mock_settings, mock_llm, mock_retrieval)
        cap = agent.capability
        assert cap.name == "rag_agent"
        assert IntentType.FACTUAL in cap.supported_intents
        assert IntentType.EMOTIONAL in cap.supported_intents
        assert IntentType.GENERAL in cap.supported_intents


# ═══════════════════════════════════════════════════════════════════════════
# MemoryAgent Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestMemoryAgent:
    """Tests for MemoryAgent."""

    @pytest.mark.asyncio
    async def test_empty_query(self, mock_llm, mock_settings, mock_retrieval):
        from ara.agents.memory_agent import MemoryAgent

        agent = MemoryAgent(mock_settings, mock_llm, mock_retrieval)
        await agent.connect()
        result = await agent.execute(create_initial_state(""))
        assert result.get("error")

    @pytest.mark.asyncio
    async def test_generates_response_without_graph(self, mock_llm, mock_settings, mock_retrieval):
        from ara.agents.memory_agent import MemoryAgent

        mock_llm.generate.return_value = "I remember that..."

        agent = MemoryAgent(mock_settings, mock_llm, mock_retrieval, graph=None)
        await agent.connect()
        result = await agent.execute(create_initial_state("what did I write about Mom?"))
        assert result["response"] == "I remember that..."

    @pytest.mark.asyncio
    async def test_builds_graph_context(self, mock_llm, mock_settings, mock_retrieval):
        from ara.agents.memory_agent import MemoryAgent

        mock_entity = MagicMock()
        mock_entity.entity_id = "e1"
        mock_entity.name = "Mom"
        mock_entity.entity_type = "person"

        mock_graph = MagicMock()
        mock_graph.search_entities.return_value = [mock_entity]
        mock_graph.get_entity_context.return_value = "Mom is connected to Dad via interacted_with"
        mock_llm.generate.return_value = "memory response"

        agent = MemoryAgent(mock_settings, mock_llm, mock_retrieval, graph=mock_graph)
        await agent.connect()
        result = await agent.execute(create_initial_state("tell me about Mom"))
        assert result["graph_context"]  # Should have some graph context

    def test_capability(self, mock_settings, mock_llm, mock_retrieval):
        from ara.agents.memory_agent import MemoryAgent

        agent = MemoryAgent(mock_settings, mock_llm, mock_retrieval)
        assert agent.capability.name == "memory_agent"
        assert IntentType.REMINISCING in agent.capability.supported_intents


# ═══════════════════════════════════════════════════════════════════════════
# ToolAgent Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestToolAgent:
    """Tests for ToolAgent."""

    @pytest.mark.asyncio
    async def test_parse_valid_tool_calls(self, mock_llm, mock_settings):
        from ara.agents.tool_agent import ToolAgent

        agent = ToolAgent(mock_settings, mock_llm)
        calls = agent._parse_tool_calls(
            '[{"tool_name": "open_application", "parameters": {"name": "chrome"}}]'
        )
        assert len(calls) == 1
        assert calls[0].tool_name == "open_application"

    @pytest.mark.asyncio
    async def test_parse_tool_calls_with_fences(self, mock_llm, mock_settings):
        from ara.agents.tool_agent import ToolAgent

        agent = ToolAgent(mock_settings, mock_llm)
        calls = agent._parse_tool_calls(
            '```json\n[{"tool_name": "screenshot", "parameters": {}}]\n```'
        )
        assert len(calls) == 1

    @pytest.mark.asyncio
    async def test_parse_invalid_json_returns_empty(self, mock_llm, mock_settings):
        from ara.agents.tool_agent import ToolAgent

        agent = ToolAgent(mock_settings, mock_llm)
        calls = agent._parse_tool_calls("not json")
        assert len(calls) == 0

    @pytest.mark.asyncio
    async def test_clarify_response(self, mock_llm, mock_settings):
        from ara.agents.tool_agent import ToolAgent

        mock_llm.generate.return_value = (
            '[{"tool_name": "clarify", "parameters": {"message": "What app?"}}]'
        )
        agent = ToolAgent(mock_settings, mock_llm)
        await agent.connect()
        result = await agent.execute(create_initial_state("open the thing"))
        assert "What app?" in result["response"]

    @pytest.mark.asyncio
    async def test_confirmation_required(self, mock_llm, mock_settings):
        from ara.agents.tool_agent import ToolAgent

        mock_llm.generate.side_effect = [
            '[{"tool_name": "run_command", "parameters": {"command": "rm -rf"}}]',
            "I deleted everything",
        ]
        agent = ToolAgent(mock_settings, mock_llm)
        await agent.connect()
        result = await agent.execute(create_initial_state("delete everything"))
        assert "Shall I go ahead" in result["response"]
        assert result.get("metadata", {}).get("awaiting_confirmation")

    def test_capability(self, mock_llm, mock_settings):
        from ara.agents.tool_agent import ToolAgent

        agent = ToolAgent(mock_settings, mock_llm)
        assert agent.capability.name == "tool_agent"
        assert IntentType.DESKTOP in agent.capability.supported_intents


# ═══════════════════════════════════════════════════════════════════════════
# WebSearchAgent Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestWebSearchAgent:
    """Tests for WebSearchAgent."""

    @pytest.mark.asyncio
    async def test_internet_disabled_returns_message(self, mock_llm, mock_settings):
        from ara.agents.web_search_agent import WebSearchAgent

        mock_settings.internet.internet_enabled = False
        agent = WebSearchAgent(mock_settings, mock_llm)
        await agent.connect()
        result = await agent.execute(create_initial_state("latest news"))
        assert "disabled" in result["response"]

    @pytest.mark.asyncio
    async def test_search_with_custom_fn(self, mock_llm, mock_settings):
        from ara.agents.web_search_agent import WebSearchAgent
        from ara.utils.types import WebSearchResult

        async def mock_search(query, max_results):
            return [
                WebSearchResult(
                    title="Test Result",
                    url="https://example.com",
                    snippet="This is a test result",
                    source="mock",
                    relevance_score=0.9,
                ),
            ]

        mock_llm.generate.return_value = "Based on recent information..."

        agent = WebSearchAgent(mock_settings, mock_llm, search_fn=mock_search)
        await agent.connect()
        result = await agent.execute(create_initial_state("what is the weather?"))
        assert result["response"]
        assert result.get("metadata", {}).get("results_count") == 1

    @pytest.mark.asyncio
    async def test_search_failure(self, mock_llm, mock_settings):
        from ara.agents.web_search_agent import WebSearchAgent

        async def failing_search(query, max_results):
            raise ConnectionError("No internet")

        agent = WebSearchAgent(mock_settings, mock_llm, search_fn=failing_search)
        await agent.connect()
        result = await agent.execute(create_initial_state("news"))
        assert "couldn't complete" in result["response"]

    def test_capability(self, mock_llm, mock_settings):
        from ara.agents.web_search_agent import WebSearchAgent

        agent = WebSearchAgent(mock_settings, mock_llm)
        assert agent.capability.name == "web_search_agent"
        assert IntentType.WEB_SEARCH in agent.capability.supported_intents


# ═══════════════════════════════════════════════════════════════════════════
# WebBrowseAgent Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestWebBrowseAgent:
    """Tests for WebBrowseAgent."""

    @pytest.mark.asyncio
    async def test_internet_disabled(self, mock_llm, mock_settings):
        from ara.agents.web_browse_agent import WebBrowseAgent

        mock_settings.internet.internet_enabled = False
        agent = WebBrowseAgent(mock_settings, mock_llm)
        await agent.connect()
        result = await agent.execute(create_initial_state("read https://example.com"))
        assert "disabled" in result["response"]

    @pytest.mark.asyncio
    async def test_url_extraction_regex(self, mock_llm, mock_settings):
        from ara.agents.web_browse_agent import WebBrowseAgent

        async def mock_browser(url):
            return "Page content here"

        mock_llm.generate.return_value = "The page says..."

        agent = WebBrowseAgent(mock_settings, mock_llm, browser_fn=mock_browser)
        await agent.connect()
        result = await agent.execute(create_initial_state("read https://example.com/page"))
        assert result["response"]

    @pytest.mark.asyncio
    async def test_no_url_found(self, mock_llm, mock_settings):
        from ara.agents.web_browse_agent import WebBrowseAgent

        mock_llm.generate.return_value = '{"urls": [], "action": "none"}'
        agent = WebBrowseAgent(mock_settings, mock_llm)
        await agent.connect()
        result = await agent.execute(create_initial_state("read that article"))
        assert "URL" in result["response"] or "url" in result["response"].lower()

    def test_clear_cache(self, mock_llm, mock_settings):
        from ara.agents.web_browse_agent import WebBrowseAgent

        agent = WebBrowseAgent(mock_settings, mock_llm)
        agent._content_cache["https://example.com"] = "cached content"
        agent.clear_cache()
        assert len(agent._content_cache) == 0


# ═══════════════════════════════════════════════════════════════════════════
# ResearchAgent Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestResearchAgent:
    """Tests for ResearchAgent."""

    @pytest.mark.asyncio
    async def test_internet_disabled(self, mock_llm, mock_settings):
        from ara.agents.research_agent import ResearchAgent

        mock_settings.internet.internet_enabled = False
        agent = ResearchAgent(mock_settings, mock_llm)
        await agent.connect()
        result = await agent.execute(create_initial_state("research quantum computing"))
        assert "disabled" in result["response"]

    @pytest.mark.asyncio
    async def test_decompose_query(self, mock_llm, mock_settings):
        from ara.agents.research_agent import ResearchAgent

        mock_llm.generate.return_value = '{"sub_questions": ["what is X?", "how does X work?"], "search_queries": ["X definition", "X mechanism"]}'

        agent = ResearchAgent(mock_settings, mock_llm)
        plan = await agent._decompose_query("explain X in detail")
        assert len(plan.sub_questions) == 2
        assert len(plan.search_queries) == 2

    @pytest.mark.asyncio
    async def test_decompose_fallback_on_bad_json(self, mock_llm, mock_settings):
        from ara.agents.research_agent import ResearchAgent

        mock_llm.generate.return_value = "not json"

        agent = ResearchAgent(mock_settings, mock_llm)
        plan = await agent._decompose_query("explain X")
        assert len(plan.sub_questions) == 1
        assert plan.sub_questions[0] == "explain X"

    def test_capability(self, mock_llm, mock_settings):
        from ara.agents.research_agent import ResearchAgent

        agent = ResearchAgent(mock_settings, mock_llm)
        assert agent.capability.name == "research_agent"
        assert IntentType.RESEARCH in agent.capability.supported_intents


# ═══════════════════════════════════════════════════════════════════════════
# Orchestrator Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestOrchestrator:
    """Tests for the LangGraph Orchestrator."""

    @pytest.mark.asyncio
    async def test_process_requires_connection(self, mock_llm, mock_settings):
        from ara.agents.orchestrator import Orchestrator

        router = IntentRouter(mock_llm, mock_settings)
        registry = AgentRegistry(mock_settings)
        orchestrator = Orchestrator(mock_settings, mock_llm, router, registry)

        with pytest.raises(AgentError, match="not connected"):
            await orchestrator.process("hello")

    @pytest.mark.asyncio
    async def test_full_pipeline(self, mock_llm, mock_settings, mock_retrieval):
        from ara.agents.orchestrator import Orchestrator
        from ara.agents.rag_agent import RAGAgent

        # Setup
        mock_llm.generate.side_effect = [
            # Router classification
            '{"intent": "general", "confidence": 0.9, "reasoning": "casual chat"}',
            # RAG agent response
            "Hey there! How can I help?",
        ]

        router = IntentRouter(mock_llm, mock_settings)
        registry = AgentRegistry(mock_settings)
        rag_agent = RAGAgent(mock_settings, mock_llm, mock_retrieval)
        registry.register(rag_agent)

        orchestrator = Orchestrator(mock_settings, mock_llm, router, registry)
        await orchestrator.connect()

        result = await orchestrator.process("hello")
        assert result.response
        assert result.intent == "general"
        assert result.agent == "rag_agent"
        assert result.elapsed_ms > 0

        await orchestrator.close()

    @pytest.mark.asyncio
    async def test_orchestrator_result_to_dict(self):
        from ara.agents.orchestrator import OrchestratorResult

        result = OrchestratorResult(
            response="test",
            intent="factual",
            agent="rag_agent",
            confidence=0.9,
            elapsed_ms=100.0,
        )
        d = result.to_dict()
        assert d["response"] == "test"
        assert d["intent"] == "factual"
        assert d["elapsed_ms"] == 100.0

    @pytest.mark.asyncio
    async def test_orchestrator_result_success(self):
        from ara.agents.orchestrator import OrchestratorResult

        good = OrchestratorResult(response="yes", intent="general", agent="rag", confidence=0.9)
        assert good.success

        bad = OrchestratorResult(
            response="", intent="general", agent="rag", confidence=0.0, error="failed"
        )
        assert not bad.success


# ═══════════════════════════════════════════════════════════════════════════
# Synthesis Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestSynthesisMode:
    """Tests for synthesis mode auto-selection."""

    def test_auto_select_mode_few_chunks(self):
        from ara.agents.synthesis import SynthesisMode, _auto_select_mode

        assert _auto_select_mode(3) == SynthesisMode.COMPACT

    def test_auto_select_mode_medium_chunks(self):
        from ara.agents.synthesis import SynthesisMode, _auto_select_mode

        assert _auto_select_mode(10) == SynthesisMode.REFINE

    def test_auto_select_mode_many_chunks(self):
        from ara.agents.synthesis import SynthesisMode, _auto_select_mode

        assert _auto_select_mode(20) == SynthesisMode.TREE_SUMMARIZE


class TestResponseSynthesis:
    """Tests for ResponseSynthesis."""

    def test_not_connected_raises(self, mock_settings):
        from ara.agents.synthesis import ResponseSynthesis

        synth = ResponseSynthesis(mock_settings)
        assert not synth.is_connected

    @pytest.mark.asyncio
    async def test_synthesize_empty_results(self, mock_settings):
        from ara.agents.synthesis import ResponseSynthesis

        synth = ResponseSynthesis(mock_settings)
        await synth.connect()

        empty_context = MagicMock()
        empty_context.results = []

        result = await synth.synthesize("test query", empty_context)
        assert result.answer == ""
        assert result.source_count == 0

    @pytest.mark.asyncio
    async def test_results_to_nodes(self):
        from ara.agents.synthesis import ResponseSynthesis
        from ara.utils.types import MemoryChunk, MemorySource, RetrievalResult

        results = [
            RetrievalResult(
                chunk=MemoryChunk(
                    chunk_id="c1",
                    text="test text",
                    source=MemorySource.DIARY,
                    source_file="diary.md",
                ),
                score=0.8,
                retrieval_method="vector",
                rerank_score=0.9,
            )
        ]
        nodes = ResponseSynthesis._results_to_nodes(results)
        assert len(nodes) == 1
        assert nodes[0].score == 0.9  # Uses rerank_score


# ═══════════════════════════════════════════════════════════════════════════
# AgentPermission Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestAgentPermission:
    """Tests for AgentPermission enum."""

    def test_all_permissions_exist(self):
        perms = list(AgentPermission)
        assert len(perms) == 10
        assert AgentPermission.MEMORY_READ in perms
        assert AgentPermission.DESKTOP_CONTROL in perms
        assert AgentPermission.WEB_SEARCH in perms


class TestAgentCapability:
    """Tests for AgentCapability dataclass."""

    def test_frozen(self):
        cap = AgentCapability(
            name="test",
            display_name="Test",
            description="desc",
            supported_intents=frozenset({IntentType.GENERAL}),
            required_permissions=frozenset({AgentPermission.LLM_GENERATE}),
        )
        with pytest.raises(AttributeError):
            cap.name = "changed"  # type: ignore[misc]

    def test_defaults(self):
        cap = AgentCapability(
            name="test",
            display_name="Test",
            description="desc",
            supported_intents=frozenset(),
            required_permissions=frozenset(),
        )
        assert cap.priority == 100
        assert cap.enabled is True
