"""Web search abstraction — DuckDuckGo and SearXNG backends.

Provides a unified search interface that agents can use. Supports multiple
search backends with automatic fallback:
1. SearXNG (self-hosted, preferred — no tracking, privacy-first)
2. DuckDuckGo (via duckduckgo-search library — no API key required)

Architecture:
    - SearchBackend enum selects the provider
    - WebSearchService wraps both backends behind a unified async API
    - Results are normalized to WebSearchResult dataclass
    - Rate limiting prevents abuse
    - Results can be cached via the caching module

Usage:
    service = WebSearchService(settings)
    results = await service.search("latest Python news", max_results=5)
"""

from __future__ import annotations

import asyncio
import time
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from ara.utils.errors import OfflineError, WebSearchError
from ara.utils.logging import get_logger
from ara.utils.types import WebSearchResult

if TYPE_CHECKING:
    from ara.config.settings import InternetSettings

log = get_logger("web_search")


class SearchBackend(StrEnum):
    """Available search backends."""

    DUCKDUCKGO = "duckduckgo"
    SEARXNG = "searxng"
    AUTO = "auto"  # Try SearXNG first, fall back to DuckDuckGo


# Default maximum results per search
_DEFAULT_MAX_RESULTS = 8

# Minimum interval between searches to avoid rate limiting (seconds)
_MIN_SEARCH_INTERVAL_S = 1.0


class WebSearchService:
    """Unified web search service with multiple backends.

    Args:
        settings: Internet configuration.
        backend: Which search backend to use.
        search_fn: Optional custom search function for testing.
            Signature: async (query, max_results) -> list[WebSearchResult]
    """

    def __init__(
        self,
        settings: InternetSettings,
        *,
        backend: SearchBackend = SearchBackend.AUTO,
        search_fn: Any | None = None,
    ) -> None:
        self._settings = settings
        self._backend = backend
        self._search_fn = search_fn
        self._last_search_time: float = 0.0

    @property
    def backend(self) -> SearchBackend:
        """Currently configured search backend."""
        return self._backend

    async def search(
        self,
        query: str,
        *,
        max_results: int = _DEFAULT_MAX_RESULTS,
        region: str = "wt-wt",
    ) -> list[WebSearchResult]:
        """Search the web and return normalized results.

        Args:
            query: Search query string.
            max_results: Maximum number of results to return.
            region: Search region code (default: worldwide).

        Returns:
            List of WebSearchResult objects sorted by relevance.

        Raises:
            OfflineError: If internet is disabled.
            WebSearchError: If search fails on all backends.
        """
        if not self._settings.internet_enabled:
            raise OfflineError("search")

        if not query.strip():
            return []

        # Rate limiting
        now = time.monotonic()
        elapsed = now - self._last_search_time
        if elapsed < _MIN_SEARCH_INTERVAL_S:
            await asyncio.sleep(_MIN_SEARCH_INTERVAL_S - elapsed)
        self._last_search_time = time.monotonic()

        # Custom search function (for testing)
        if self._search_fn is not None:
            return await self._search_fn(query, max_results)

        # Route to backend
        start = time.perf_counter()

        if self._backend == SearchBackend.DUCKDUCKGO:
            results = await self._duckduckgo_search(query, max_results, region)
        elif self._backend == SearchBackend.SEARXNG:
            results = await self._searxng_search(query, max_results)
        elif self._backend == SearchBackend.AUTO:
            results = await self._auto_search(query, max_results, region)
        else:
            raise WebSearchError(
                f"Unknown search backend: {self._backend}",
                details={"backend": self._backend},
            )

        elapsed_ms = (time.perf_counter() - start) * 1000
        log.info(
            "search_complete",
            query=query[:80],
            backend=self._backend.value,
            results=len(results),
            elapsed_ms=round(elapsed_ms, 1),
        )

        return results

    async def news_search(
        self,
        query: str,
        *,
        max_results: int = _DEFAULT_MAX_RESULTS,
        region: str = "wt-wt",
    ) -> list[WebSearchResult]:
        """Search for recent news articles.

        Args:
            query: News search query.
            max_results: Maximum results.
            region: Search region.

        Returns:
            List of news results.

        Raises:
            OfflineError: If internet is disabled.
            WebSearchError: If search fails.
        """
        if not self._settings.internet_enabled:
            raise OfflineError("news_search")

        if not query.strip():
            return []

        try:
            return await self._duckduckgo_news(query, max_results, region)
        except Exception as exc:
            raise WebSearchError(
                f"News search failed: {exc}",
                details={"query": query[:80]},
            ) from exc

    async def _auto_search(
        self,
        query: str,
        max_results: int,
        region: str,
    ) -> list[WebSearchResult]:
        """Try SearXNG first, fall back to DuckDuckGo.

        Args:
            query: Search query.
            max_results: Maximum results.
            region: Search region.

        Returns:
            Search results from whichever backend succeeds.
        """
        # Try SearXNG first (self-hosted, preferred)
        try:
            results = await self._searxng_search(query, max_results)
            if results:
                return results
        except Exception as exc:
            log.debug("searxng_unavailable", error=str(exc))

        # Fall back to DuckDuckGo
        return await self._duckduckgo_search(query, max_results, region)

    async def _duckduckgo_search(
        self,
        query: str,
        max_results: int,
        region: str,
    ) -> list[WebSearchResult]:
        """Search using the duckduckgo-search library.

        Args:
            query: Search query.
            max_results: Maximum results.
            region: Region code.

        Returns:
            Normalized search results.

        Raises:
            WebSearchError: If DuckDuckGo search fails.
        """
        try:
            from ddgs import DDGS
        except ImportError:
            try:
                from duckduckgo_search import DDGS
            except ImportError as exc:
                raise WebSearchError(
                    "ddgs library not installed",
                    details={"hint": "pip install ddgs"},
                ) from exc

        loop = asyncio.get_running_loop()

        def _sync_search() -> list[dict[str, str]]:
            ddgs = DDGS()
            return list(ddgs.text(query, max_results=max_results, region=region))

        try:
            raw_results = await loop.run_in_executor(None, _sync_search)
        except Exception as exc:
            raise WebSearchError(
                f"DuckDuckGo search failed: {exc}",
                details={"query": query[:80]},
            ) from exc

        return _normalize_ddg_results(raw_results)

    async def _duckduckgo_news(
        self,
        query: str,
        max_results: int,
        region: str,
    ) -> list[WebSearchResult]:
        """Search DuckDuckGo news.

        Args:
            query: News query.
            max_results: Maximum results.
            region: Region code.

        Returns:
            Normalized news results.
        """
        try:
            from ddgs import DDGS
        except ImportError:
            try:
                from duckduckgo_search import DDGS
            except ImportError as exc:
                raise WebSearchError(
                    "ddgs library not installed",
                ) from exc

        loop = asyncio.get_running_loop()

        def _sync_news() -> list[dict[str, str]]:
            ddgs = DDGS()
            return list(ddgs.news(query, max_results=max_results, region=region))

        try:
            raw_results = await loop.run_in_executor(None, _sync_news)
        except Exception as exc:
            raise WebSearchError(f"DuckDuckGo news search failed: {exc}") from exc

        results: list[WebSearchResult] = []
        for i, item in enumerate(raw_results):
            score = 1.0 - (i / max(len(raw_results), 1)) * 0.5
            results.append(
                WebSearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", item.get("link", "")),
                    snippet=item.get("body", item.get("excerpt", "")),
                    source="duckduckgo_news",
                    relevance_score=round(score, 3),
                )
            )
        return results

    async def _searxng_search(
        self,
        query: str,
        max_results: int,
    ) -> list[WebSearchResult]:
        """Search using a self-hosted SearXNG instance.

        Args:
            query: Search query.
            max_results: Maximum results.

        Returns:
            Normalized search results.

        Raises:
            WebSearchError: If SearXNG is unreachable or returns errors.
        """
        import httpx

        searxng_url = self._settings.searxng_url.rstrip("/")
        search_url = f"{searxng_url}/search"

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(
                    search_url,
                    params={
                        "q": query,
                        "format": "json",
                        "pageno": 1,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
        except httpx.ConnectError as exc:
            raise WebSearchError(
                f"SearXNG unavailable at {searxng_url}: {exc}",
                details={"url": searxng_url},
            ) from exc
        except Exception as exc:
            raise WebSearchError(f"SearXNG search failed: {exc}") from exc

        raw_results = data.get("results", [])[:max_results]
        results: list[WebSearchResult] = []

        for i, item in enumerate(raw_results):
            score = 1.0 - (i / max(len(raw_results), 1)) * 0.5
            results.append(
                WebSearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    snippet=item.get("content", ""),
                    source="searxng",
                    relevance_score=round(score, 3),
                )
            )

        return results


def _normalize_ddg_results(raw_results: list[dict[str, str]]) -> list[WebSearchResult]:
    """Normalize DuckDuckGo search results to WebSearchResult format.

    Args:
        raw_results: Raw results from duckduckgo-search library.

    Returns:
        List of normalized WebSearchResult objects.
    """
    results: list[WebSearchResult] = []
    for i, item in enumerate(raw_results):
        score = 1.0 - (i / max(len(raw_results), 1)) * 0.5
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
