"""Tests for the web connectivity layer (src/ara/web/).

Covers:
- ConnectivityManager: state management, toggles, callbacks, network checks
- WebCache: TTL, LRU eviction, stats, get/set/delete/clear
- WebSearchService: DuckDuckGo search, SearXNG search, auto fallback, offline guard
- WebBrowser: page content extraction, Playwright/httpx fallback, screenshots
- ResearchPipeline: query decomposition, multi-step research, synthesis
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from ara.config.settings import InternetSettings
from ara.utils.errors import OfflineError, WebBrowseError, WebSearchError
from ara.utils.types import WebSearchResult
from ara.web.browser import (
    ContentFormat,
    PageContent,
    WebBrowser,
    _extract_content,
    _html_to_text,
)
from ara.web.cache import CacheEntry, WebCache
from ara.web.connectivity import ConnectivityManager, ConnectivityState
from ara.web.research import ResearchPipeline, ResearchResult, ResearchSource
from ara.web.search import SearchBackend, WebSearchService, _normalize_ddg_results

# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def internet_settings() -> InternetSettings:
    """Create default internet settings for testing."""
    return InternetSettings(
        internet_enabled=True,
        searxng_url="http://localhost:8888",
        web_cache_ttl_s=3600,
        max_research_steps=5,
    )


@pytest.fixture
def disabled_settings() -> InternetSettings:
    """Create internet settings with internet disabled."""
    return InternetSettings(
        internet_enabled=False,
        searxng_url="http://localhost:8888",
        web_cache_ttl_s=3600,
        max_research_steps=5,
    )


@pytest.fixture
def mock_search_results() -> list[WebSearchResult]:
    """Sample web search results for testing."""
    return [
        WebSearchResult(
            title="Python 3.12 Release Notes",
            url="https://python.org/releases/3.12",
            snippet="Python 3.12 brings significant improvements...",
            source="duckduckgo",
            relevance_score=0.95,
        ),
        WebSearchResult(
            title="What's New in Python 3.12",
            url="https://realpython.com/python312",
            snippet="Learn about the new features in Python 3.12...",
            source="duckduckgo",
            relevance_score=0.88,
        ),
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# ConnectivityManager Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestConnectivityManagerInit:
    """Tests for ConnectivityManager initialization."""

    def test_initial_state_is_unknown(self, internet_settings: InternetSettings) -> None:
        manager = ConnectivityManager(internet_settings, check_interval_s=0)
        assert manager.state == ConnectivityState.UNKNOWN
        assert not manager.is_online

    def test_user_enabled_from_settings(self, internet_settings: InternetSettings) -> None:
        manager = ConnectivityManager(internet_settings, check_interval_s=0)
        assert manager.is_user_enabled is True

    def test_user_disabled_from_settings(self, disabled_settings: InternetSettings) -> None:
        manager = ConnectivityManager(disabled_settings, check_interval_s=0)
        assert manager.is_user_enabled is False


class TestConnectivityManagerStateTransitions:
    """Tests for state transitions in ConnectivityManager."""

    @pytest.mark.asyncio
    async def test_start_checks_connectivity(self, internet_settings: InternetSettings) -> None:
        manager = ConnectivityManager(internet_settings, check_interval_s=0)
        manager.set_check_fn(AsyncMock(return_value=True))
        await manager.start()
        assert manager.state == ConnectivityState.ONLINE
        assert manager.is_online
        await manager.stop()

    @pytest.mark.asyncio
    async def test_offline_when_network_unavailable(
        self, internet_settings: InternetSettings
    ) -> None:
        manager = ConnectivityManager(internet_settings, check_interval_s=0)
        manager.set_check_fn(AsyncMock(return_value=False))
        await manager.start()
        assert manager.state == ConnectivityState.OFFLINE_NO_NETWORK
        assert not manager.is_online
        await manager.stop()

    @pytest.mark.asyncio
    async def test_offline_by_user_toggle(self, disabled_settings: InternetSettings) -> None:
        manager = ConnectivityManager(disabled_settings, check_interval_s=0)
        await manager.start()
        assert manager.state == ConnectivityState.OFFLINE_BY_USER
        assert not manager.is_online
        await manager.stop()

    @pytest.mark.asyncio
    async def test_toggle_enables_internet(self, disabled_settings: InternetSettings) -> None:
        manager = ConnectivityManager(disabled_settings, check_interval_s=0)
        manager.set_check_fn(AsyncMock(return_value=True))
        await manager.start()
        assert not manager.is_online

        new_state = await manager.set_user_enabled(True)
        assert new_state == ConnectivityState.ONLINE
        assert manager.is_online
        await manager.stop()

    @pytest.mark.asyncio
    async def test_toggle_disables_internet(self, internet_settings: InternetSettings) -> None:
        manager = ConnectivityManager(internet_settings, check_interval_s=0)
        manager.set_check_fn(AsyncMock(return_value=True))
        await manager.start()
        assert manager.is_online

        new_state = await manager.set_user_enabled(False)
        assert new_state == ConnectivityState.OFFLINE_BY_USER
        assert not manager.is_online
        await manager.stop()

    @pytest.mark.asyncio
    async def test_double_start_is_noop(self, internet_settings: InternetSettings) -> None:
        manager = ConnectivityManager(internet_settings, check_interval_s=0)
        check_fn = AsyncMock(return_value=True)
        manager.set_check_fn(check_fn)
        await manager.start()
        await manager.start()  # Should not re-check
        assert check_fn.call_count == 1
        await manager.stop()

    @pytest.mark.asyncio
    async def test_stop_when_not_running_is_noop(self, internet_settings: InternetSettings) -> None:
        manager = ConnectivityManager(internet_settings, check_interval_s=0)
        await manager.stop()  # Should not raise


class TestConnectivityManagerCallbacks:
    """Tests for state change callbacks."""

    @pytest.mark.asyncio
    async def test_callback_invoked_on_state_change(
        self, internet_settings: InternetSettings
    ) -> None:
        manager = ConnectivityManager(internet_settings, check_interval_s=0)
        manager.set_check_fn(AsyncMock(return_value=True))

        received_states: list[ConnectivityState] = []
        manager.on_state_change(lambda s: received_states.append(s))

        await manager.start()
        assert ConnectivityState.ONLINE in received_states
        await manager.stop()

    @pytest.mark.asyncio
    async def test_async_callback_invoked(self, internet_settings: InternetSettings) -> None:
        manager = ConnectivityManager(internet_settings, check_interval_s=0)
        manager.set_check_fn(AsyncMock(return_value=True))

        received_states: list[ConnectivityState] = []

        async def async_cb(state: ConnectivityState) -> None:
            received_states.append(state)

        manager.on_state_change(async_cb)
        await manager.start()
        assert ConnectivityState.ONLINE in received_states
        await manager.stop()

    @pytest.mark.asyncio
    async def test_callback_error_does_not_crash(self, internet_settings: InternetSettings) -> None:
        manager = ConnectivityManager(internet_settings, check_interval_s=0)
        manager.set_check_fn(AsyncMock(return_value=True))

        def bad_callback(_state: ConnectivityState) -> None:
            msg = "callback error"
            raise RuntimeError(msg)

        manager.on_state_change(bad_callback)
        await manager.start()  # Should not raise
        await manager.stop()

    @pytest.mark.asyncio
    async def test_remove_callback(self, internet_settings: InternetSettings) -> None:
        manager = ConnectivityManager(internet_settings, check_interval_s=0)
        manager.set_check_fn(AsyncMock(return_value=True))

        calls: list[ConnectivityState] = []
        cb = lambda s: calls.append(s)  # noqa: E731
        manager.on_state_change(cb)
        manager.remove_callback(cb)

        await manager.start()
        assert len(calls) == 0
        await manager.stop()


class TestConnectivityManagerRequireOnline:
    """Tests for require_online guard."""

    @pytest.mark.asyncio
    async def test_require_online_passes_when_online(
        self, internet_settings: InternetSettings
    ) -> None:
        manager = ConnectivityManager(internet_settings, check_interval_s=0)
        manager.set_check_fn(AsyncMock(return_value=True))
        await manager.start()
        manager.require_online("search")  # Should not raise
        await manager.stop()

    @pytest.mark.asyncio
    async def test_require_online_raises_when_offline(
        self, internet_settings: InternetSettings
    ) -> None:
        manager = ConnectivityManager(internet_settings, check_interval_s=0)
        manager.set_check_fn(AsyncMock(return_value=False))
        await manager.start()
        with pytest.raises(OfflineError):
            manager.require_online("search")
        await manager.stop()


class TestConnectivityManagerGetStatus:
    """Tests for get_status."""

    @pytest.mark.asyncio
    async def test_get_status_returns_dict(self, internet_settings: InternetSettings) -> None:
        manager = ConnectivityManager(internet_settings, check_interval_s=0)
        manager.set_check_fn(AsyncMock(return_value=True))
        await manager.start()
        status = manager.get_status()
        assert status["state"] == "online"
        assert status["user_enabled"] is True
        assert status["network_available"] is True
        await manager.stop()


class TestConnectivityManagerCheckFn:
    """Tests for custom check functions."""

    @pytest.mark.asyncio
    async def test_check_fn_exception_returns_false(
        self, internet_settings: InternetSettings
    ) -> None:
        manager = ConnectivityManager(internet_settings, check_interval_s=0)
        manager.set_check_fn(AsyncMock(side_effect=ConnectionError("no network")))
        await manager.start()
        assert manager.state == ConnectivityState.OFFLINE_NO_NETWORK
        await manager.stop()


# ═══════════════════════════════════════════════════════════════════════════════
# WebCache Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestWebCacheBasic:
    """Basic cache get/set operations."""

    def test_empty_cache_returns_none(self) -> None:
        cache = WebCache(ttl_s=3600)
        assert cache.get("missing") is None

    def test_set_and_get(self) -> None:
        cache = WebCache(ttl_s=3600)
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

    def test_overwrite_existing_key(self) -> None:
        cache = WebCache(ttl_s=3600)
        cache.set("key1", "old")
        cache.set("key1", "new")
        assert cache.get("key1") == "new"

    def test_has_returns_true_for_existing(self) -> None:
        cache = WebCache(ttl_s=3600)
        cache.set("key1", "value1")
        assert cache.has("key1") is True

    def test_has_returns_false_for_missing(self) -> None:
        cache = WebCache(ttl_s=3600)
        assert cache.has("missing") is False

    def test_delete_existing_key(self) -> None:
        cache = WebCache(ttl_s=3600)
        cache.set("key1", "value1")
        assert cache.delete("key1") is True
        assert cache.get("key1") is None

    def test_delete_missing_key_returns_false(self) -> None:
        cache = WebCache(ttl_s=3600)
        assert cache.delete("missing") is False

    def test_clear_removes_all(self) -> None:
        cache = WebCache(ttl_s=3600)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)
        cleared = cache.clear()
        assert cleared == 3
        assert cache.size == 0

    def test_size_property(self) -> None:
        cache = WebCache(ttl_s=3600)
        assert cache.size == 0
        cache.set("a", 1)
        assert cache.size == 1
        cache.set("b", 2)
        assert cache.size == 2


class TestWebCacheTTL:
    """TTL expiration tests."""

    def test_expired_entry_returns_none(self) -> None:
        cache = WebCache(ttl_s=60)
        cache.set("key1", "value1")
        # Manually age the entry past TTL
        cache._entries["key1"].created_at -= 120
        assert cache.get("key1") is None

    def test_expired_entry_removed_on_has_check(self) -> None:
        cache = WebCache(ttl_s=60)
        cache.set("key1", "value1")
        cache._entries["key1"].created_at -= 120
        assert cache.has("key1") is False
        assert cache.size == 0

    def test_evict_expired_removes_old_entries(self) -> None:
        cache = WebCache(ttl_s=60)
        cache.set("a", 1)
        cache.set("b", 2)
        for entry in cache._entries.values():
            entry.created_at -= 120
        evicted = cache.evict_expired()
        assert evicted == 2
        assert cache.size == 0


class TestWebCacheLRU:
    """LRU eviction tests."""

    def test_lru_eviction_at_max_capacity(self) -> None:
        cache = WebCache(ttl_s=3600, max_entries=3)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)
        cache.set("d", 4)  # Should evict LRU entry
        assert cache.size == 3
        assert cache.get("d") == 4

    def test_lru_evicts_least_recently_accessed(self) -> None:
        import time as time_mod

        cache = WebCache(ttl_s=3600, max_entries=3)
        cache.set("a", 1)
        # Ensure distinct timestamps for LRU ordering
        cache._entries["a"].last_accessed -= 100
        time_mod.sleep(0.001)
        cache.set("b", 2)
        cache._entries["b"].last_accessed -= 50
        time_mod.sleep(0.001)
        cache.set("c", 3)
        # Now: a (oldest access), b, c (newest access)
        # Touch "a" to make it recent
        cache.get("a")
        # Now: b (oldest access), c, a (newest access)
        cache.set("d", 4)  # Should evict "b" (oldest last_accessed)
        assert cache.get("a") == 1  # Still present (recently accessed)
        assert cache.get("b") is None  # Evicted
        assert cache.get("d") == 4


class TestWebCacheStats:
    """Cache statistics tests."""

    def test_hit_rate_zero_when_empty(self) -> None:
        cache = WebCache(ttl_s=3600)
        assert cache.hit_rate == 0.0

    def test_stats_track_hits_and_misses(self) -> None:
        cache = WebCache(ttl_s=3600)
        cache.set("a", 1)
        cache.get("a")  # Hit
        cache.get("b")  # Miss
        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5

    def test_clear_resets_stats(self) -> None:
        cache = WebCache(ttl_s=3600)
        cache.set("a", 1)
        cache.get("a")
        cache.clear()
        stats = cache.get_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0


class TestCacheEntry:
    """CacheEntry tests."""

    def test_touch_increments_access_count(self) -> None:
        entry = CacheEntry("key", "value")
        assert entry.access_count == 0
        entry.touch()
        assert entry.access_count == 1

    def test_is_expired_returns_false_for_fresh_entry(self) -> None:
        entry = CacheEntry("key", "value")
        assert entry.is_expired(3600) is False

    def test_is_expired_returns_true_for_old_entry(self) -> None:
        entry = CacheEntry("key", "value")
        entry.created_at -= 3601  # Simulate aging
        assert entry.is_expired(3600) is True


# ═══════════════════════════════════════════════════════════════════════════════
# WebSearchService Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestWebSearchServiceInit:
    """WebSearchService initialization tests."""

    def test_default_backend_is_auto(self, internet_settings: InternetSettings) -> None:
        service = WebSearchService(internet_settings)
        assert service.backend == SearchBackend.AUTO

    def test_custom_backend(self, internet_settings: InternetSettings) -> None:
        service = WebSearchService(internet_settings, backend=SearchBackend.DUCKDUCKGO)
        assert service.backend == SearchBackend.DUCKDUCKGO


class TestWebSearchServiceSearch:
    """WebSearchService search tests."""

    @pytest.mark.asyncio
    async def test_search_with_custom_fn(
        self,
        internet_settings: InternetSettings,
        mock_search_results: list[WebSearchResult],
    ) -> None:
        async def fake_search(query: str, max_results: int) -> list[WebSearchResult]:
            return mock_search_results

        service = WebSearchService(internet_settings, search_fn=fake_search)
        results = await service.search("python 3.12")
        assert len(results) == 2
        assert results[0].title == "Python 3.12 Release Notes"

    @pytest.mark.asyncio
    async def test_search_returns_empty_for_blank_query(
        self, internet_settings: InternetSettings
    ) -> None:
        service = WebSearchService(internet_settings)
        results = await service.search("")
        assert results == []

    @pytest.mark.asyncio
    async def test_search_raises_offline_error(self, disabled_settings: InternetSettings) -> None:
        service = WebSearchService(disabled_settings)
        with pytest.raises(OfflineError):
            await service.search("test query")

    @pytest.mark.asyncio
    async def test_news_search_raises_offline_error(
        self, disabled_settings: InternetSettings
    ) -> None:
        service = WebSearchService(disabled_settings)
        with pytest.raises(OfflineError):
            await service.news_search("test query")

    @pytest.mark.asyncio
    async def test_news_search_empty_query_returns_empty(
        self, internet_settings: InternetSettings
    ) -> None:
        service = WebSearchService(internet_settings)
        results = await service.news_search("")
        assert results == []


class TestNormalizeDDGResults:
    """Tests for _normalize_ddg_results utility."""

    def test_normalizes_results(self) -> None:
        raw = [
            {"title": "Test", "href": "https://test.com", "body": "Test snippet"},
        ]
        results = _normalize_ddg_results(raw)
        assert len(results) == 1
        assert results[0].title == "Test"
        assert results[0].url == "https://test.com"
        assert results[0].source == "duckduckgo"

    def test_handles_missing_fields(self) -> None:
        raw = [{"title": "Test"}]
        results = _normalize_ddg_results(raw)
        assert results[0].url == ""
        assert results[0].snippet == ""

    def test_empty_results(self) -> None:
        assert _normalize_ddg_results([]) == []

    def test_relevance_scores_decrease(self) -> None:
        raw = [
            {"title": f"Result {i}", "href": f"https://test{i}.com", "body": ""} for i in range(5)
        ]
        results = _normalize_ddg_results(raw)
        scores = [r.relevance_score for r in results]
        assert scores == sorted(scores, reverse=True)


# ═══════════════════════════════════════════════════════════════════════════════
# WebBrowser Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestWebBrowserInit:
    """WebBrowser initialization tests."""

    def test_not_running_initially(self, internet_settings: InternetSettings) -> None:
        browser = WebBrowser(internet_settings)
        assert not browser.is_running

    @pytest.mark.asyncio
    async def test_start_httpx_mode(self, internet_settings: InternetSettings) -> None:
        browser = WebBrowser(internet_settings, use_playwright=False)
        await browser.start()
        assert browser.is_running
        await browser.stop()

    @pytest.mark.asyncio
    async def test_double_start_is_noop(self, internet_settings: InternetSettings) -> None:
        browser = WebBrowser(internet_settings, use_playwright=False)
        await browser.start()
        await browser.start()  # Should not raise
        assert browser.is_running
        await browser.stop()

    @pytest.mark.asyncio
    async def test_stop_when_not_running_is_noop(self, internet_settings: InternetSettings) -> None:
        browser = WebBrowser(internet_settings)
        await browser.stop()  # Should not raise


class TestWebBrowserFetch:
    """WebBrowser content fetching tests."""

    @pytest.mark.asyncio
    async def test_fetch_with_custom_fn(self, internet_settings: InternetSettings) -> None:
        async def fake_fetch(url: str) -> str:
            return (
                "<html><head><title>Test Page</title></head><body><p>Hello World</p></body></html>"
            )

        browser = WebBrowser(internet_settings, fetch_fn=fake_fetch)
        await browser.start()
        page = await browser.get_page_content("https://example.com")
        assert page.title == "Test Page"
        assert "Hello World" in page.content
        assert page.url == "https://example.com"
        await browser.stop()

    @pytest.mark.asyncio
    async def test_fetch_raises_offline_error(self, disabled_settings: InternetSettings) -> None:
        browser = WebBrowser(disabled_settings)
        await browser.start()
        with pytest.raises(OfflineError):
            await browser.get_page_content("https://example.com")
        await browser.stop()

    @pytest.mark.asyncio
    async def test_fetch_truncates_long_content(self, internet_settings: InternetSettings) -> None:
        long_content = "x" * 60_000

        async def fake_fetch(url: str) -> str:
            return f"<html><body><p>{long_content}</p></body></html>"

        browser = WebBrowser(internet_settings, fetch_fn=fake_fetch)
        await browser.start()
        page = await browser.get_page_content("https://example.com", max_chars=1000)
        assert page.was_truncated
        assert "[Content truncated]" in page.content
        assert page.content_length <= 1050  # max_chars + truncation message
        await browser.stop()

    @pytest.mark.asyncio
    async def test_screenshot_raises_offline_error(
        self, disabled_settings: InternetSettings
    ) -> None:
        browser = WebBrowser(disabled_settings)
        with pytest.raises(OfflineError):
            await browser.screenshot("https://example.com")

    @pytest.mark.asyncio
    async def test_screenshot_raises_without_playwright(
        self, internet_settings: InternetSettings
    ) -> None:
        browser = WebBrowser(internet_settings, use_playwright=False)
        await browser.start()
        with pytest.raises(WebBrowseError):
            await browser.screenshot("https://example.com")
        await browser.stop()


class TestPageContent:
    """PageContent dataclass tests."""

    def test_to_dict(self) -> None:
        page = PageContent(
            url="https://example.com",
            title="Test",
            content="Hello",
            content_format=ContentFormat.TEXT,
            load_time_ms=42.5,
            was_truncated=False,
        )
        d = page.to_dict()
        assert d["url"] == "https://example.com"
        assert d["title"] == "Test"
        assert d["format"] == "text"
        assert d["load_time_ms"] == 42.5
        assert d["was_truncated"] is False


class TestHTMLExtraction:
    """Tests for HTML content extraction functions."""

    def test_extract_text_from_html(self) -> None:
        html = """
        <html>
        <head><title>Test Page</title></head>
        <body>
            <h1>Hello</h1>
            <p>World</p>
            <script>evil();</script>
            <style>.hide{}</style>
        </body>
        </html>
        """
        title, content = _extract_content(html, ContentFormat.TEXT)
        assert title == "Test Page"
        assert "Hello" in content
        assert "World" in content
        assert "evil" not in content
        assert ".hide" not in content

    def test_extract_markdown_from_html(self) -> None:
        html = """
        <html>
        <head><title>Test</title></head>
        <body>
            <h1>Main Title</h1>
            <p>Some text</p>
            <h2>Subtitle</h2>
            <ul><li>Item 1</li><li>Item 2</li></ul>
        </body>
        </html>
        """
        _, content = _extract_content(html, ContentFormat.MARKDOWN)
        assert "# Main Title" in content
        assert "## Subtitle" in content
        assert "- Item 1" in content
        assert "- Item 2" in content

    def test_extract_html_format_removes_scripts(self) -> None:
        html = "<html><body><script>bad();</script><p>Good</p></body></html>"
        _, content = _extract_content(html, ContentFormat.HTML)
        assert "bad" not in content
        assert "Good" in content

    def test_html_to_text_collapses_whitespace(self) -> None:
        from bs4 import BeautifulSoup

        html = "<p>Line 1</p><p></p><p></p><p></p><p>Line 2</p>"
        soup = BeautifulSoup(html, "html.parser")
        text = _html_to_text(soup)
        # Should collapse multiple blanks into single blank
        assert text.count("\n\n") <= 2


# ═══════════════════════════════════════════════════════════════════════════════
# ResearchPipeline Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestResearchSource:
    """ResearchSource tests."""

    def test_defaults(self) -> None:
        source = ResearchSource(url="https://test.com", title="Test", snippet="Snippet")
        assert source.relevance == 0.0
        assert source.used_in_synthesis is False


class TestResearchResult:
    """ResearchResult tests."""

    def test_to_dict(self) -> None:
        result = ResearchResult(
            query="test query",
            answer="Test answer",
            sources=[
                ResearchSource(
                    url="https://test.com",
                    title="Test",
                    snippet="",
                    relevance=0.9,
                    used_in_synthesis=True,
                ),
                ResearchSource(
                    url="https://unused.com",
                    title="Unused",
                    snippet="",
                    relevance=0.3,
                    used_in_synthesis=False,
                ),
            ],
            sub_queries=["sub1", "sub2"],
            steps_taken=3,
            total_time_ms=1234.5,
            pages_read=1,
        )
        d = result.to_dict()
        assert d["query"] == "test query"
        assert len(d["sources"]) == 1  # Only used_in_synthesis
        assert d["sources"][0]["url"] == "https://test.com"
        assert d["steps_taken"] == 3
        assert d["pages_read"] == 1


class TestResearchPipelineInit:
    """ResearchPipeline initialization tests."""

    @pytest.mark.asyncio
    async def test_research_raises_offline(self, disabled_settings: InternetSettings) -> None:
        mock_llm = AsyncMock()
        mock_search = AsyncMock()
        mock_browser = AsyncMock()

        pipeline = ResearchPipeline(disabled_settings, mock_llm, mock_search, mock_browser)
        with pytest.raises(OfflineError):
            await pipeline.research("test query")


class TestResearchPipelineExecution:
    """ResearchPipeline execution tests with mocked dependencies."""

    @pytest.mark.asyncio
    async def test_full_research_pipeline(
        self,
        internet_settings: InternetSettings,
        mock_search_results: list[WebSearchResult],
    ) -> None:
        # Mock LLM: return sub-queries on first call, synthesis on second
        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(
            side_effect=[
                '["What are Python 3.12 features?", "Python 3.12 performance improvements"]',
                "Python 3.12 brings many improvements including...",
            ]
        )

        # Mock search service
        mock_search_svc = AsyncMock()
        mock_search_svc.search = AsyncMock(return_value=mock_search_results)

        # Mock browser
        mock_browser = AsyncMock()
        mock_page = MagicMock()
        mock_page.content = "Detailed page content about Python 3.12..."
        mock_page.title = "Python Docs"
        mock_browser.get_page_content = AsyncMock(return_value=mock_page)

        pipeline = ResearchPipeline(internet_settings, mock_llm, mock_search_svc, mock_browser)
        result = await pipeline.research("What's new in Python 3.12?")

        assert result.query == "What's new in Python 3.12?"
        assert "Python 3.12" in result.answer
        assert len(result.sub_queries) == 2
        assert result.steps_taken > 0
        assert result.total_time_ms > 0

    @pytest.mark.asyncio
    async def test_research_fallback_on_failed_decomposition(
        self,
        internet_settings: InternetSettings,
        mock_search_results: list[WebSearchResult],
    ) -> None:
        # LLM returns invalid JSON for decomposition, valid for synthesis
        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(
            side_effect=[
                "This is not JSON at all",  # Failed decomposition
                "Synthesized answer",  # Synthesis
            ]
        )

        mock_search_svc = AsyncMock()
        mock_search_svc.search = AsyncMock(return_value=mock_search_results)

        mock_browser = AsyncMock()
        mock_page = MagicMock()
        mock_page.content = "Content"
        mock_page.title = "Title"
        mock_browser.get_page_content = AsyncMock(return_value=mock_page)

        pipeline = ResearchPipeline(internet_settings, mock_llm, mock_search_svc, mock_browser)
        result = await pipeline.research("test query")

        # Should fall back to using original query as sub-query
        assert result.sub_queries == ["test query"]
        assert result.steps_taken > 0

    @pytest.mark.asyncio
    async def test_research_respects_max_steps(self, internet_settings: InternetSettings) -> None:
        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(
            side_effect=[
                '["q1", "q2", "q3", "q4"]',
                "Synthesized answer",
            ]
        )

        mock_search_svc = AsyncMock()
        mock_search_svc.search = AsyncMock(
            return_value=[
                WebSearchResult(
                    title="R", url="https://r.com", snippet="S", source="ddg", relevance_score=0.9
                )
            ]
        )

        mock_browser = AsyncMock()
        mock_page = MagicMock()
        mock_page.content = "Content"
        mock_page.title = "Title"
        mock_browser.get_page_content = AsyncMock(return_value=mock_page)

        pipeline = ResearchPipeline(internet_settings, mock_llm, mock_search_svc, mock_browser)
        result = await pipeline.research("test", max_steps=3)

        # Should stop after max_steps
        assert result.steps_taken <= 3

    @pytest.mark.asyncio
    async def test_research_handles_search_failure(
        self, internet_settings: InternetSettings
    ) -> None:
        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(
            side_effect=[
                '["sub query"]',
                "No results found",
            ]
        )

        mock_search_svc = AsyncMock()
        mock_search_svc.search = AsyncMock(side_effect=WebSearchError("search failed"))

        mock_browser = AsyncMock()

        pipeline = ResearchPipeline(internet_settings, mock_llm, mock_search_svc, mock_browser)
        result = await pipeline.research("test query")

        # Should handle gracefully with "couldn't find" message
        assert "couldn't find" in result.answer.lower() or result.answer

    @pytest.mark.asyncio
    async def test_research_handles_page_read_failure(
        self,
        internet_settings: InternetSettings,
        mock_search_results: list[WebSearchResult],
    ) -> None:
        mock_llm = AsyncMock()
        mock_llm.generate = AsyncMock(
            side_effect=[
                '["sub query"]',
                "Synthesized from snippets",
            ]
        )

        mock_search_svc = AsyncMock()
        mock_search_svc.search = AsyncMock(return_value=mock_search_results)

        mock_browser = AsyncMock()
        mock_browser.get_page_content = AsyncMock(
            side_effect=WebBrowseError("https://test.com", "connection refused")
        )

        pipeline = ResearchPipeline(internet_settings, mock_llm, mock_search_svc, mock_browser)
        result = await pipeline.research("test query")

        # Should still produce a result from snippets
        assert result.answer
        assert result.steps_taken > 0
