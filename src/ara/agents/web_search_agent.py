"""Web search agent — internet search with result ranking and answer synthesis.

Handles queries requiring live internet data: current events, latest news,
real-time information, product lookups, etc. Searches via DuckDuckGo (default)
or SearXNG (self-hosted), ranks results, and synthesizes a response.

Architecture:
    - Uses duckduckgo-search library for internet search (no API key required)
    - SearXNG self-hosted instance as preferred backend (if available)
    - Extracts top-N results, ranks by relevance
    - Synthesizes answer with source citations
    - Respects offline mode (returns graceful error when internet disabled)

Flow:
    1. Check connectivity status
    2. Search via DuckDuckGo/SearXNG
    3. Rank and filter results
    4. Synthesize response with citations
    5. Optionally cache results for future recall

Usage:
    agent = WebSearchAgent(settings, llm_client)
    await agent.connect()
    result_state = await agent.execute(state)
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from ara.agents.base import AgentCapability, AgentPermission, BaseAgent
from ara.utils.errors import OfflineError
from ara.utils.logging import get_logger
from ara.utils.types import (
    ChatMessage,
    IntentType,
    MessageRole,
    WebSearchResult,
)

if TYPE_CHECKING:
    from ara.agents.state import AgentState
    from ara.config.settings import AraSettings
    from ara.core.llm import LLMClient

log = get_logger("web_search_agent")


# ── Capability Declaration ───────────────────────────────────────────────────

_WEB_SEARCH_CAPABILITY = AgentCapability(
    name="web_search_agent",
    display_name="Web Search",
    description=(
        "Searches the internet for current information, news, real-time data, "
        "and answers questions that require live web access."
    ),
    supported_intents=frozenset(
        {
            IntentType.WEB_SEARCH,
        }
    ),
    required_permissions=frozenset(
        {
            AgentPermission.WEB_SEARCH,
            AgentPermission.LLM_GENERATE,
        }
    ),
    priority=100,
)

# Maximum search results to fetch
_MAX_SEARCH_RESULTS = 8

# Maximum results to include in synthesis context
_MAX_SYNTHESIS_RESULTS = 5

# System prompt for search result synthesis
_SEARCH_SYNTHESIS_PROMPT = """You are ARA, answering a question using web search results. Use the search results below to provide an accurate, up-to-date answer.

Rules:
- Cite your sources naturally: "According to [source]..." or "Based on recent reports..."
- If results are conflicting, mention the different perspectives
- If results don't fully answer the question, say what you found and what's missing
- Be concise but thorough
- Include relevant dates when the information is time-sensitive
- If you're not confident in the accuracy, say so"""


class WebSearchAgent(BaseAgent):
    """Web search agent — searches the internet and synthesizes answers.

    Args:
        settings: Application settings (includes internet config).
        llm_client: LLM for synthesizing search results.
        search_fn: Optional custom search function (for testing).
            Signature: async (query, max_results) -> list[WebSearchResult]
    """

    def __init__(
        self,
        settings: AraSettings,
        llm_client: LLMClient,
        *,
        search_fn: Any | None = None,
    ) -> None:
        super().__init__(settings)
        self._llm = llm_client
        self._search_fn = search_fn

    @property
    def capability(self) -> AgentCapability:
        return _WEB_SEARCH_CAPABILITY

    async def _execute(self, state: AgentState) -> AgentState:
        """Execute web search: check connectivity → search → synthesize.

        Args:
            state: Current orchestrator state.

        Returns:
            Updated state with response and search metadata.
        """
        query = state.get("query", "")
        if not query.strip():
            return {"response": "", "error": "Empty query"}  # type: ignore[return-value]

        # Step 1: Check if internet is enabled
        if not self._settings.internet.internet_enabled:
            return {  # type: ignore[return-value]
                "response": (
                    "I can't search the web right now — internet access is disabled. "
                    "You can enable it in the dashboard settings."
                ),
                "error": "internet_disabled",
            }

        # Step 2: Perform web search
        start = time.perf_counter()
        try:
            results = await self._search(query)
        except Exception as exc:
            log.error("web_search_failed", query=query[:80], error=str(exc))
            return {  # type: ignore[return-value]
                "response": f"I couldn't complete the web search: {exc}. Want me to try again?",
                "error": f"search_failed: {exc}",
            }

        search_time_ms = (time.perf_counter() - start) * 1000

        if not results:
            return {  # type: ignore[return-value]
                "response": (
                    "I searched the web but didn't find relevant results for that query. "
                    "Could you rephrase or be more specific?"
                ),
                "metadata": {"search_time_ms": search_time_ms},
            }

        log.info(
            "web_search_complete",
            query=query[:80],
            results_count=len(results),
            elapsed_ms=round(search_time_ms, 1),
        )

        # Step 3: Synthesize response from search results (include conversation history)
        history = state.get("conversation_history", [])
        response = await self._synthesize_response(
            query,
            results[:_MAX_SYNTHESIS_RESULTS],
            history=history,
        )

        # Step 4: Build metadata
        result_dicts = [
            {
                "title": r.title,
                "url": r.url,
                "snippet": r.snippet[:200],
                "source": r.source,
                "relevance": r.relevance_score,
            }
            for r in results
        ]

        return {  # type: ignore[return-value]
            "response": response,
            "metadata": {
                "search_results": result_dicts,
                "search_time_ms": search_time_ms,
                "results_count": len(results),
            },
        }

    async def _search(self, query: str) -> list[WebSearchResult]:
        """Perform web search via configured backend.

        Args:
            query: Search query string.

        Returns:
            List of search results ranked by relevance.
        """
        # Use custom search function if provided (for testing)
        if self._search_fn is not None:
            return await self._search_fn(query, _MAX_SEARCH_RESULTS)

        # Default: use duckduckgo-search
        return await self._duckduckgo_search(query)

    async def _duckduckgo_search(self, query: str) -> list[WebSearchResult]:
        """Search using the duckduckgo-search library.

        Args:
            query: Search query.

        Returns:
            List of WebSearchResult objects.
        """
        try:
            from ddgs import DDGS
        except ImportError:
            try:
                from duckduckgo_search import DDGS
            except ImportError as exc:
                raise OfflineError("search") from exc

        results: list[WebSearchResult] = []

        # duckduckgo-search is sync, run in executor to avoid blocking
        import asyncio

        loop = asyncio.get_running_loop()

        def _sync_search() -> list[dict[str, str]]:
            # Note: Do NOT use `with DDGS() as ddgs:` — the context manager
            # silently breaks when called from asyncio.run_in_executor(),
            # returning 0 results. Direct instantiation works correctly.
            ddgs = DDGS()
            return list(ddgs.text(query, max_results=_MAX_SEARCH_RESULTS))

        raw_results = await loop.run_in_executor(None, _sync_search)

        for i, item in enumerate(raw_results):
            score = 1.0 - (i / max(len(raw_results), 1)) * 0.5  # Position-based scoring
            results.append(
                WebSearchResult(
                    title=item.get("title", ""),
                    url=item.get("href", item.get("link", "")),
                    snippet=item.get("body", item.get("snippet", "")),
                    source="duckduckgo",
                    relevance_score=round(score, 3),
                )
            )

        return results

    async def _synthesize_response(
        self,
        query: str,
        results: list[WebSearchResult],
        *,
        history: list[ChatMessage] | None = None,
    ) -> str:
        """Synthesize a natural language response from search results.

        Args:
            query: Original user query.
            results: Search results to synthesize from.
            history: Optional conversation history for follow-up context.

        Returns:
            Synthesized response text with citations.
        """
        # Format search results as context
        context_parts: list[str] = []
        for i, r in enumerate(results, 1):
            context_parts.append(f"[{i}] {r.title}\n    URL: {r.url}\n    {r.snippet}")
        context = "\n\n".join(context_parts)

        messages: list[ChatMessage] = [
            ChatMessage(role=MessageRole.SYSTEM, content=_SEARCH_SYNTHESIS_PROMPT),
        ]
        # Include recent conversation history for follow-up context
        if history:
            messages.extend(history[-4:])
        messages.append(
            ChatMessage(
                role=MessageRole.USER,
                content=f"Search results:\n\n{context}\n\nUser question: {query}",
            ),
        )

        return await self._llm.generate(messages, max_tokens=800)
