"""Deep research pipeline — multi-step web research with citations.

Provides a structured research pipeline that:
1. Decomposes a research question into sub-queries
2. Searches each sub-query
3. Reads promising pages in depth
4. Synthesizes findings with source citations

Architecture:
    - Uses WebSearchService for search and WebBrowser for page reading
    - LLM decomposes queries and synthesizes results
    - Iterative: search → read → evaluate → follow links → synthesize
    - Respects configurable max steps to prevent infinite loops
    - All sources are tracked for citation

Usage:
    pipeline = ResearchPipeline(settings, llm_client, search_service, browser)
    result = await pipeline.research("What are the latest advances in quantum computing?")
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ara.utils.errors import OfflineError
from ara.utils.logging import get_logger

if TYPE_CHECKING:
    from ara.config.settings import InternetSettings
    from ara.core.llm import LLMClient
    from ara.utils.types import ChatMessage
    from ara.web.browser import WebBrowser
    from ara.web.search import WebSearchService

log = get_logger("research")


@dataclass
class ResearchSource:
    """A source used during research.

    Attributes:
        url: Source URL.
        title: Page title.
        snippet: Brief excerpt or snippet.
        relevance: How relevant this source was (0-1).
        used_in_synthesis: Whether this source contributed to the final answer.
    """

    url: str
    title: str
    snippet: str
    relevance: float = 0.0
    used_in_synthesis: bool = False


@dataclass
class ResearchResult:
    """Result of a research pipeline execution.

    Attributes:
        query: The original research query.
        answer: Synthesized answer with citations.
        sources: List of sources used.
        sub_queries: Sub-queries that were investigated.
        steps_taken: Number of research steps executed.
        total_time_ms: Total research time.
        pages_read: Number of pages read in depth.
    """

    query: str
    answer: str = ""
    sources: list[ResearchSource] = field(default_factory=list)
    sub_queries: list[str] = field(default_factory=list)
    steps_taken: int = 0
    total_time_ms: float = 0.0
    pages_read: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "query": self.query,
            "answer": self.answer,
            "sources": [
                {"url": s.url, "title": s.title, "relevance": s.relevance}
                for s in self.sources
                if s.used_in_synthesis
            ],
            "sub_queries": self.sub_queries,
            "steps_taken": self.steps_taken,
            "total_time_ms": round(self.total_time_ms, 1),
            "pages_read": self.pages_read,
        }


# Maximum sub-queries per research session
_MAX_SUB_QUERIES = 4

# Maximum pages to read in depth per research session
_MAX_PAGES_TO_READ = 5

# Maximum content from each page to include in synthesis
_MAX_PAGE_EXCERPT_CHARS = 4000

# Prompt for decomposing research queries
_DECOMPOSE_PROMPT = """You are a research assistant. Break down the following research question into 2-4 specific, searchable sub-questions.

Rules:
- Each sub-question should target a different aspect of the main question
- Sub-questions should be specific enough for a web search
- Return ONLY a JSON array of strings: ["sub-question 1", "sub-question 2", ...]
- Do NOT include the original question as a sub-question
- Keep each sub-question under 100 characters"""

# Prompt for synthesizing research findings
_SYNTHESIS_PROMPT = """You are ARA, synthesizing research findings. Create a comprehensive answer using the information gathered from multiple sources.

Rules:
- Cite sources naturally: "According to [Source Title]..." or "[1]"
- Cover all relevant aspects found across sources
- Note any conflicting information between sources
- If information is incomplete, explicitly state what's missing
- Be thorough but concise
- Use clear formatting (paragraphs, bullet points where appropriate)
- Include a brief "Sources" section at the end listing the key sources used"""


class ResearchPipeline:
    """Multi-step deep research with search, read, and synthesis.

    Args:
        settings: Internet configuration.
        llm_client: LLM for query decomposition and synthesis.
        search_service: Web search service.
        browser: Web browser for page reading.
    """

    def __init__(
        self,
        settings: InternetSettings,
        llm_client: LLMClient,
        search_service: WebSearchService,
        browser: WebBrowser,
    ) -> None:
        self._settings = settings
        self._llm = llm_client
        self._search = search_service
        self._browser = browser

    async def research(
        self,
        query: str,
        *,
        max_steps: int | None = None,
    ) -> ResearchResult:
        """Execute a deep research pipeline for the given query.

        Steps:
        1. Decompose query into sub-queries
        2. Search each sub-query
        3. Read top-ranked pages
        4. Synthesize findings with citations

        Args:
            query: The research question.
            max_steps: Maximum research steps (overrides settings).

        Returns:
            ResearchResult with synthesized answer and sources.

        Raises:
            OfflineError: If internet is disabled.
        """
        if not self._settings.internet_enabled:
            raise OfflineError("research")

        effective_max_steps = max_steps or self._settings.max_research_steps
        start = time.perf_counter()

        result = ResearchResult(query=query)

        # Step 1: Decompose query into sub-queries
        sub_queries = await self._decompose_query(query)
        result.sub_queries = sub_queries
        result.steps_taken += 1

        log.info("research_decomposed", query=query[:80], sub_queries=len(sub_queries))

        # Step 2: Search each sub-query
        all_search_results: list[ResearchSource] = []
        seen_urls: set[str] = set()

        for sq in sub_queries[:_MAX_SUB_QUERIES]:
            if result.steps_taken >= effective_max_steps:
                break

            try:
                search_results = await self._search.search(sq, max_results=5)
                for sr in search_results:
                    if sr.url not in seen_urls:
                        seen_urls.add(sr.url)
                        all_search_results.append(
                            ResearchSource(
                                url=sr.url,
                                title=sr.title,
                                snippet=sr.snippet,
                                relevance=sr.relevance_score,
                            )
                        )
                result.steps_taken += 1
            except Exception as exc:
                log.warning("sub_query_search_failed", query=sq[:80], error=str(exc))

        # Sort by relevance and take top results for reading
        all_search_results.sort(key=lambda s: s.relevance, reverse=True)
        result.sources = all_search_results

        # Step 3: Read top-ranked pages in depth
        page_contents: list[dict[str, str]] = []
        pages_to_read = min(_MAX_PAGES_TO_READ, len(all_search_results))

        for source in all_search_results[:pages_to_read]:
            if result.steps_taken >= effective_max_steps:
                break

            try:
                page = await self._browser.get_page_content(source.url)
                excerpt = page.content[:_MAX_PAGE_EXCERPT_CHARS]
                page_contents.append(
                    {
                        "url": source.url,
                        "title": source.title or page.title,
                        "content": excerpt,
                    }
                )
                result.pages_read += 1
                source.used_in_synthesis = True
                result.steps_taken += 1
            except Exception as exc:
                log.warning("page_read_failed", url=source.url[:100], error=str(exc))
                # Use snippet as fallback
                if source.snippet:
                    page_contents.append(
                        {
                            "url": source.url,
                            "title": source.title,
                            "content": source.snippet,
                        }
                    )
                    source.used_in_synthesis = True

        # Step 4: Synthesize findings
        if page_contents:
            answer = await self._synthesize(query, page_contents)
        elif all_search_results:
            # Fallback: synthesize from snippets only
            snippet_contents = [
                {
                    "url": s.url,
                    "title": s.title,
                    "content": s.snippet,
                }
                for s in all_search_results[:_MAX_PAGES_TO_READ]
            ]
            for s in all_search_results[:_MAX_PAGES_TO_READ]:
                s.used_in_synthesis = True
            answer = await self._synthesize(query, snippet_contents)
        else:
            answer = (
                "I couldn't find enough information to answer that question. "
                "The search didn't return relevant results. Could you try rephrasing?"
            )

        result.answer = answer
        result.total_time_ms = (time.perf_counter() - start) * 1000

        log.info(
            "research_complete",
            query=query[:80],
            sub_queries=len(sub_queries),
            pages_read=result.pages_read,
            sources_used=sum(1 for s in result.sources if s.used_in_synthesis),
            steps=result.steps_taken,
            elapsed_ms=round(result.total_time_ms, 1),
        )

        return result

    async def _decompose_query(self, query: str) -> list[str]:
        """Decompose a research question into sub-queries using the LLM.

        Args:
            query: The research question.

        Returns:
            List of sub-query strings.
        """
        import json

        from ara.utils.types import ChatMessage, MessageRole

        messages: list[ChatMessage] = [
            ChatMessage(role=MessageRole.SYSTEM, content=_DECOMPOSE_PROMPT),
            ChatMessage(role=MessageRole.USER, content=f"Research question: {query}"),
        ]

        raw = await self._llm.generate(messages, temperature=0.3, max_tokens=500)

        # Parse JSON array from response
        try:
            # Strip markdown fences if present
            import re

            cleaned = raw.strip()
            cleaned = re.sub(r"^```(?:json)?\s*\n?", "", cleaned)
            cleaned = re.sub(r"\n?```\s*$", "", cleaned)

            # Extract JSON array
            array_match = re.search(r"\[.*\]", cleaned, re.DOTALL)
            if array_match:
                cleaned = array_match.group(0)

            sub_queries = json.loads(cleaned)

            if isinstance(sub_queries, list):
                return [str(q) for q in sub_queries[:_MAX_SUB_QUERIES]]
        except (json.JSONDecodeError, AttributeError):
            pass

        # Fallback: use original query as single sub-query
        log.warning("query_decomposition_failed", query=query[:80])
        return [query]

    async def _synthesize(
        self,
        query: str,
        page_contents: list[dict[str, str]],
    ) -> str:
        """Synthesize research findings into a comprehensive answer.

        Args:
            query: Original research query.
            page_contents: List of {"url": ..., "title": ..., "content": ...} dicts.

        Returns:
            Synthesized answer with citations.
        """
        from ara.utils.types import ChatMessage, MessageRole

        # Format sources for LLM context
        context_parts: list[str] = []
        for i, pc in enumerate(page_contents, 1):
            context_parts.append(
                f"[Source {i}] {pc['title']}\nURL: {pc['url']}\nContent:\n{pc['content']}\n---"
            )

        context = "\n\n".join(context_parts)

        messages: list[ChatMessage] = [
            ChatMessage(role=MessageRole.SYSTEM, content=_SYNTHESIS_PROMPT),
            ChatMessage(
                role=MessageRole.USER,
                content=f"Research findings:\n\n{context}\n\nResearch question: {query}",
            ),
        ]

        return await self._llm.generate(messages, max_tokens=2000)
