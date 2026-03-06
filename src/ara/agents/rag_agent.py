"""RAG agent — retrieves relevant context and generates responses.

The primary agent for most user queries. Handles factual questions,
emotional support, planning help, creative tasks, and general conversation
by retrieving relevant memories and synthesizing personalized responses.

Architecture:
    - Uses ARA's custom RetrievalPipeline for hybrid search
    - Uses LlamaIndex ResponseSynthesizer for answer generation
    - Falls back to direct LLM generation if no relevant context is found
    - Builds personality-infused system prompts with memory context

Flow:
    1. Retrieve relevant context (vector + BM25 + graph + rerank)
    2. Assemble context string with source attribution
    3. Build system prompt (personality + memory context + available tools)
    4. Generate response via LlamaIndex synthesis or direct LLM
    5. Return updated state with response

Usage:
    agent = RAGAgent(settings, llm_client, retrieval_pipeline, synthesizer)
    await agent.connect()
    result_state = await agent.execute(state)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ara.agents.base import AgentCapability, AgentPermission, BaseAgent
from ara.core.prompts import build_system_prompt
from ara.utils.logging import get_logger
from ara.utils.types import ChatMessage, IntentType, MessageRole

if TYPE_CHECKING:
    from ara.agents.state import AgentState
    from ara.agents.synthesis import ResponseSynthesis
    from ara.config.settings import AraSettings
    from ara.core.llm import LLMClient
    from ara.memory.retrieval import RetrievalPipeline

log = get_logger("rag_agent")

# ── Capability Declaration ───────────────────────────────────────────────────

_RAG_CAPABILITY = AgentCapability(
    name="rag_agent",
    display_name="Knowledge & Memory",
    description=(
        "Retrieves relevant memories, diary entries, and knowledge to answer questions. "
        "Handles factual queries, emotional support, planning, creative tasks, and general chat."
    ),
    supported_intents=frozenset(
        {
            IntentType.FACTUAL,
            IntentType.EMOTIONAL,
            IntentType.PLANNING,
            IntentType.CREATIVE,
            IntentType.VISION,
            IntentType.GENERAL,
        }
    ),
    required_permissions=frozenset(
        {
            AgentPermission.MEMORY_READ,
            AgentPermission.LLM_GENERATE,
        }
    ),
    priority=100,
)

# Minimum retrieval score to consider context relevant
_MIN_RELEVANCE_SCORE = 0.15

# If no retrieval results above this score, fall back to direct LLM
_DIRECT_LLM_THRESHOLD = 3


# ── RAG Agent ────────────────────────────────────────────────────────────────


class RAGAgent(BaseAgent):
    """RAG agent — retrieves context and generates personalized responses.

    This is the workhorse agent that handles most user queries by:
    1. Searching ARA's memory for relevant context
    2. Synthesizing a response using LlamaIndex (with personality)
    3. Falling back to direct LLM generation when no relevant context exists

    Args:
        settings: Application settings.
        llm_client: LLM client for direct generation.
        retrieval_pipeline: Hybrid retrieval pipeline.
        synthesizer: LlamaIndex response synthesizer.
    """

    def __init__(
        self,
        settings: AraSettings,
        llm_client: LLMClient,
        retrieval_pipeline: RetrievalPipeline,
        synthesizer: ResponseSynthesis | None = None,
    ) -> None:
        super().__init__(settings)
        self._llm = llm_client
        self._retrieval = retrieval_pipeline
        self._synthesizer = synthesizer

    @property
    def capability(self) -> AgentCapability:
        return _RAG_CAPABILITY

    async def _execute(self, state: AgentState) -> AgentState:
        """Execute RAG pipeline: retrieve → synthesize → respond.

        Args:
            state: Current orchestrator state with query and context.

        Returns:
            Updated state with response and memory context.
        """
        query = state.get("query", "")

        if not query.strip():
            return {"response": "", "error": "Empty query"}  # type: ignore[return-value]

        # Step 1: Retrieve relevant context
        try:
            retrieval_context = self._retrieval.retrieve(query)
        except Exception as exc:
            log.warning(
                "retrieval_failed",
                query=query[:80],
                error=str(exc),
            )
            # Fall through to direct LLM generation
            retrieval_context = None

        # Step 2: Decide whether to use retrieved context or go direct
        has_relevant_context = (
            retrieval_context is not None
            and len(retrieval_context.results) >= _DIRECT_LLM_THRESHOLD
        )

        memory_context = ""
        if has_relevant_context and retrieval_context is not None:
            memory_context = self._retrieval.assemble_context(retrieval_context.results)

        # Step 3: Build system prompt with personality + context
        system_prompt = build_system_prompt(
            self._settings.personality,
            memory_context=memory_context,
            available_tools=None,
        )

        # Step 4: Generate response
        if has_relevant_context and self._synthesizer is not None and retrieval_context is not None:
            # Use LlamaIndex synthesis for context-aware responses
            try:
                synthesis_result = await self._synthesizer.synthesize(
                    query,
                    retrieval_context,
                    system_prompt=system_prompt,
                )
                response = synthesis_result.answer
                synthesis_mode = synthesis_result.mode.value
            except Exception as exc:
                log.warning(
                    "synthesis_failed_fallback_to_direct",
                    error=str(exc),
                )
                response = await self._direct_generate(query, system_prompt, state)
                synthesis_mode = "direct_fallback"
        else:
            # Direct LLM generation (no relevant context or no synthesizer)
            response = await self._direct_generate(query, system_prompt, state)
            synthesis_mode = "direct"

        # Step 5: Build retrieval results for state
        retrieval_dicts: list[dict[str, object]] = []
        if retrieval_context is not None:
            retrieval_dicts.extend(
                {
                    "chunk_id": r.chunk.chunk_id,
                    "score": r.score,
                    "rerank_score": r.rerank_score,
                    "method": r.retrieval_method,
                    "source": r.chunk.source.value,
                    "text_preview": r.chunk.text[:100],
                }
                for r in retrieval_context.results
            )

        return {  # type: ignore[return-value]
            "response": response,
            "memory_context": memory_context,
            "retrieval_results": retrieval_dicts,
            "synthesis_mode": synthesis_mode,
        }

    async def _direct_generate(
        self,
        query: str,
        system_prompt: str,
        state: AgentState,
    ) -> str:
        """Generate a response directly from the LLM (no retrieval context).

        Used when:
        - No relevant context was found in memory
        - Synthesis failed and we need a fallback
        - The query is general conversation that doesn't need memory

        Args:
            query: The user's query.
            system_prompt: Assembled system prompt with personality.
            state: Current state (for conversation history).

        Returns:
            Generated response text.
        """
        messages: list[ChatMessage] = [
            ChatMessage(role=MessageRole.SYSTEM, content=system_prompt),
        ]

        # Add conversation history for context
        history = state.get("conversation_history", [])
        if history:
            messages.extend(history[-6:])  # Last 3 turns

        # Add the current query
        messages.append(ChatMessage(role=MessageRole.USER, content=query))

        response = await self._llm.generate(messages)
        return response
