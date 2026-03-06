"""Agent orchestration — routing, RAG agent, memory agent, tool agent, web agents.

Provides LangGraph-based agent orchestration with nested subgraphs,
LLM-based intent classification, and a plugin system for hot-loadable agents.

Main components:
    - Orchestrator: Main LangGraph state graph (entry point for all queries)
    - IntentRouter: LLM-based intent classification
    - AgentRegistry: Manages agent registration and lifecycle
    - BaseAgent: Abstract base class for all agents
    - ContextManager: Sliding window + summarization for context limits
    - ResponseSynthesis: LlamaIndex-based answer generation

Agents:
    - RAGAgent: Knowledge retrieval + response synthesis (factual, emotional, planning, creative)
    - MemoryAgent: Personal memory search + graph traversal (reminiscing)
    - ToolAgent: Desktop automation via structured tool calls
    - WebSearchAgent: Internet search with result synthesis
    - WebBrowseAgent: Headless browser for page content extraction
    - ResearchAgent: Multi-step deep research with citations
"""

from ara.agents.base import AgentCapability, AgentPermission, AgentRegistry, BaseAgent
from ara.agents.context_manager import ContextManager
from ara.agents.memory_agent import MemoryAgent
from ara.agents.orchestrator import Orchestrator, OrchestratorResult
from ara.agents.rag_agent import RAGAgent
from ara.agents.research_agent import ResearchAgent
from ara.agents.router import ClassificationResult, IntentRouter, RouteOverride
from ara.agents.state import AgentState, create_initial_state
from ara.agents.synthesis import ResponseSynthesis, SynthesisMode, SynthesisResult
from ara.agents.tool_agent import ToolAgent
from ara.agents.web_browse_agent import WebBrowseAgent
from ara.agents.web_search_agent import WebSearchAgent

__all__ = [
    "AgentCapability",
    "AgentPermission",
    "AgentRegistry",
    "AgentState",
    "BaseAgent",
    "ClassificationResult",
    "ContextManager",
    "IntentRouter",
    "MemoryAgent",
    "Orchestrator",
    "OrchestratorResult",
    "RAGAgent",
    "ResearchAgent",
    "ResponseSynthesis",
    "RouteOverride",
    "SynthesisMode",
    "SynthesisResult",
    "ToolAgent",
    "WebBrowseAgent",
    "WebSearchAgent",
    "create_initial_state",
]
