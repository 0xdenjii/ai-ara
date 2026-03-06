"""Memory agent — deep personal memory search and knowledge graph traversal.

Specializes in queries about the user's past: diary entries, personal events,
relationships, patterns, and temporal queries. Uses the knowledge graph for
entity-centric context alongside vector/BM25 retrieval.

Architecture:
    - Graph-first: prioritizes entity lookup and graph traversal
    - Temporal-aware: handles "last week", "in 2023", "when did I..." queries
    - Entity-centric: builds rich context from entity neighbors and relationships
    - Falls back to standard RAG when graph has no relevant entities

Flow:
    1. Extract entities from query
    2. Search knowledge graph for matching entities
    3. Build entity context (neighbors, relationships, timelines)
    4. Retrieve memory chunks (vector + BM25 with entity filter)
    5. Merge graph context with retrieved chunks
    6. Generate response with full memory context

Usage:
    agent = MemoryAgent(settings, llm_client, retrieval_pipeline, graph)
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
    from ara.config.settings import AraSettings
    from ara.core.llm import LLMClient
    from ara.memory.graph import KnowledgeGraph
    from ara.memory.retrieval import RetrievalPipeline

log = get_logger("memory_agent")


# ── Capability Declaration ───────────────────────────────────────────────────

_MEMORY_CAPABILITY = AgentCapability(
    name="memory_agent",
    display_name="Personal Memory",
    description=(
        "Deep search through personal memories, diary entries, and the knowledge graph. "
        "Handles queries about past events, relationships, patterns, and temporal lookups."
    ),
    supported_intents=frozenset(
        {
            IntentType.REMINISCING,
        }
    ),
    required_permissions=frozenset(
        {
            AgentPermission.MEMORY_READ,
            AgentPermission.LLM_GENERATE,
        }
    ),
    priority=150,  # Higher priority than RAG for reminiscing queries
)

# Maximum entities to process from graph
_MAX_ENTITY_RESULTS = 10

# Maximum graph context characters
_MAX_GRAPH_CONTEXT_CHARS = 4000

# System prompt addition for memory-focused queries
_MEMORY_SYSTEM_ADDITION = """You are answering a personal memory question. You have access to the user's diary entries, past conversations, and knowledge graph.

Guidelines for memory responses:
- Reference specific dates, events, and details when available
- Use phrases like "You wrote about..." or "Remember when..." — not "According to my records..."
- If you find conflicting memories, mention both and let the user clarify
- If the memory is emotional, be sensitive and warm
- If you can't find a specific memory, say so honestly — don't fabricate
- Connect related memories when relevant ("This reminds me of what you wrote about...")"""


class MemoryAgent(BaseAgent):
    """Memory agent — deep personal memory search and graph traversal.

    Args:
        settings: Application settings.
        llm_client: LLM client for response generation.
        retrieval_pipeline: Hybrid retrieval pipeline.
        graph: Knowledge graph for entity-centric context.
    """

    def __init__(
        self,
        settings: AraSettings,
        llm_client: LLMClient,
        retrieval_pipeline: RetrievalPipeline,
        graph: KnowledgeGraph | None = None,
    ) -> None:
        super().__init__(settings)
        self._llm = llm_client
        self._retrieval = retrieval_pipeline
        self._graph = graph

    @property
    def capability(self) -> AgentCapability:
        return _MEMORY_CAPABILITY

    async def _execute(self, state: AgentState) -> AgentState:
        """Execute memory search: graph lookup → retrieval → generate.

        Args:
            state: Current orchestrator state.

        Returns:
            Updated state with response and memory/graph context.
        """
        query = state.get("query", "")
        if not query.strip():
            return {"response": "", "error": "Empty query"}  # type: ignore[return-value]

        # Step 1: Build graph context (entity-centric)
        graph_context = ""
        if self._graph is not None:
            graph_context = self._build_graph_context(query)

        # Step 2: Retrieve memory chunks (with graph enhancement)
        try:
            retrieval_context = self._retrieval.retrieve(
                query,
                use_graph=self._graph is not None,
            )
        except Exception as exc:
            log.warning("retrieval_failed", error=str(exc))
            retrieval_context = None

        # Step 3: Assemble combined context
        memory_context_parts: list[str] = []

        if graph_context:
            memory_context_parts.append("Knowledge Graph Context:\n" + graph_context)

        if retrieval_context is not None and retrieval_context.results:
            chunk_context = self._retrieval.assemble_context(retrieval_context.results)
            if chunk_context:
                memory_context_parts.append("Retrieved Memory Chunks:\n" + chunk_context)

        combined_context = "\n\n".join(memory_context_parts)

        # Step 4: Build system prompt with memory focus
        base_system = build_system_prompt(
            self._settings.personality,
            memory_context=combined_context,
        )
        system_prompt = base_system + "\n\n" + _MEMORY_SYSTEM_ADDITION

        # Step 5: Generate response
        messages: list[ChatMessage] = [
            ChatMessage(role=MessageRole.SYSTEM, content=system_prompt),
        ]

        # Add conversation history
        history = state.get("conversation_history", [])
        if history:
            messages.extend(history[-6:])

        messages.append(ChatMessage(role=MessageRole.USER, content=query))

        response = await self._llm.generate(messages)

        # Build retrieval info for state
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
            "memory_context": combined_context,
            "graph_context": graph_context,
            "retrieval_results": retrieval_dicts,
            "synthesis_mode": "memory_direct",
        }

    def _build_graph_context(self, query: str) -> str:
        """Build rich context from the knowledge graph based on query entities.

        Searches for entities mentioned in the query, retrieves their
        graph context (neighbors, relationships, timelines), and formats
        it for LLM consumption.

        Args:
            query: The user's query to extract entities from.

        Returns:
            Formatted graph context string.
        """
        if self._graph is None:
            return ""

        # Search for entities in the query
        words = query.split()
        entity_results = self._graph.search_entities(query, limit=_MAX_ENTITY_RESULTS)

        if not entity_results:
            # Try individual significant words (>3 chars, not common words)
            common = frozenset(
                {
                    "what",
                    "when",
                    "where",
                    "about",
                    "with",
                    "from",
                    "that",
                    "this",
                    "have",
                    "been",
                    "were",
                    "they",
                    "their",
                    "there",
                    "would",
                    "could",
                    "should",
                    "will",
                    "does",
                    "didn",
                    "remember",
                    "tell",
                    "talk",
                    "last",
                    "time",
                }
            )
            for word in words:
                if len(word) > 3 and word.lower() not in common:
                    results = self._graph.search_entities(word, limit=3)
                    entity_results.extend(results)

            # Deduplicate by entity_id
            seen_ids: set[str] = set()
            unique: list[dict[str, object]] = []
            for ent in entity_results:
                eid = str(ent.get("entity_id", ""))
                if eid not in seen_ids:
                    seen_ids.add(eid)
                    unique.append(ent)
            entity_results = unique[:_MAX_ENTITY_RESULTS]

        if not entity_results:
            return ""

        # Build context for each found entity
        context_parts: list[str] = []
        total_chars = 0

        for entity in entity_results:
            entity_id = str(entity.get("entity_id", ""))
            entity_name = str(entity.get("name", ""))
            entity_type = str(entity.get("entity_type", ""))

            if not entity_id:
                continue

            # Get full entity context from graph
            entity_context = self._graph.get_entity_context(entity_id)
            if not entity_context:
                continue

            # Format entity info
            part = f"Entity: {entity_name} ({entity_type})\n"

            # Add neighbors/relationships
            neighbors = entity_context.get("neighbors", [])
            if neighbors:
                rel_lines: list[str] = []
                for neighbor in neighbors[:8]:  # Limit relationships per entity
                    rel_type = neighbor.get("relation_type", "related_to")
                    target_name = neighbor.get("name", "?")
                    rel_lines.append(f"  - {rel_type} → {target_name}")
                part += "Relationships:\n" + "\n".join(rel_lines) + "\n"

            # Check total context size
            if total_chars + len(part) > _MAX_GRAPH_CONTEXT_CHARS:
                break

            context_parts.append(part)
            total_chars += len(part)

        return "\n".join(context_parts)
