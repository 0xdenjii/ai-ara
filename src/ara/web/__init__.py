"""Internet connectivity layer — web search, browsing, deep research, online/offline state.

Modules:
    connectivity: Online/offline state management with network detection and user toggle.
    search: Web search abstraction (DuckDuckGo + SearXNG backends).
    browser: Playwright headless browsing + content extraction (with httpx fallback).
    research: Deep multi-step research pipeline with citations.
    cache: In-memory web content caching with TTL and LRU eviction.
    memory_integration: Store web research findings in LanceDB for future recall.
"""

from ara.web.browser import ContentFormat, PageContent, WebBrowser
from ara.web.cache import WebCache
from ara.web.connectivity import ConnectivityManager, ConnectivityState
from ara.web.memory_integration import WebMemoryIntegrator
from ara.web.research import ResearchPipeline, ResearchResult, ResearchSource
from ara.web.search import SearchBackend, WebSearchService

__all__ = [
    "ConnectivityManager",
    "ConnectivityState",
    "ContentFormat",
    "PageContent",
    "ResearchPipeline",
    "ResearchResult",
    "ResearchSource",
    "SearchBackend",
    "WebBrowser",
    "WebCache",
    "WebMemoryIntegrator",
    "WebSearchService",
]
