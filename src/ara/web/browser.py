"""Playwright headless browser — page content extraction, JS rendering, screenshots.

Provides a managed browser pool for loading web pages, rendering JavaScript,
extracting clean text/markdown content, and capturing screenshots.

Architecture:
    - Uses Playwright async API with headless Chromium
    - Falls back to httpx + BeautifulSoup for simple static pages
    - HTML → clean text extraction with tag removal and whitespace normalization
    - Content caching (per-session, configurable TTL via CacheManager)
    - Screenshot capture for vision pipeline integration

Usage:
    browser = WebBrowser(settings)
    await browser.start()
    content = await browser.get_page_content("https://example.com")
    screenshot = await browser.screenshot("https://example.com")
    await browser.stop()
"""

from __future__ import annotations

import asyncio
import time
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from ara.utils.errors import OfflineError, WebBrowseError
from ara.utils.logging import get_logger

if TYPE_CHECKING:
    from ara.config.settings import InternetSettings

log = get_logger("web_browser")


class ContentFormat(StrEnum):
    """Output format for extracted page content."""

    TEXT = "text"  # Plain text (default)
    MARKDOWN = "markdown"  # Markdown (preserves headers, links, lists)
    HTML = "html"  # Raw HTML (cleaned of scripts/styles)


# Maximum content length before truncation
_MAX_CONTENT_CHARS = 50_000

# Default page load timeout (milliseconds)
_DEFAULT_TIMEOUT_MS = 30_000

# User agent for requests
_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

# Tags to remove from HTML before text extraction
_REMOVE_TAGS = ["script", "style", "nav", "footer", "header", "aside", "noscript", "iframe"]


class PageContent:
    """Extracted page content with metadata.

    Attributes:
        url: The page URL.
        title: Page title from <title> tag.
        content: Extracted text content.
        format: Content format (text, markdown, html).
        content_length: Length of extracted content.
        load_time_ms: Time to load and extract the page.
        was_truncated: Whether content was truncated.
    """

    def __init__(
        self,
        *,
        url: str,
        title: str,
        content: str,
        content_format: ContentFormat = ContentFormat.TEXT,
        load_time_ms: float = 0.0,
        was_truncated: bool = False,
    ) -> None:
        self.url = url
        self.title = title
        self.content = content
        self.format = content_format
        self.content_length = len(content)
        self.load_time_ms = load_time_ms
        self.was_truncated = was_truncated

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "url": self.url,
            "title": self.title,
            "content_length": self.content_length,
            "format": self.format.value,
            "load_time_ms": round(self.load_time_ms, 1),
            "was_truncated": self.was_truncated,
        }


class WebBrowser:
    """Managed headless browser for web content extraction.

    Supports two modes:
    1. **Playwright mode** (default): Full JS rendering via headless Chromium.
       Used for dynamic pages, SPAs, and JS-heavy sites.
    2. **httpx mode** (fallback): Lightweight HTTP + BeautifulSoup.
       Used when Playwright is unavailable or for simple static pages.

    Args:
        settings: Internet configuration.
        use_playwright: Whether to use Playwright (True) or httpx fallback (False).
        fetch_fn: Optional custom fetch function for testing.
            Signature: async (url) -> str (returns page HTML)
    """

    def __init__(
        self,
        settings: InternetSettings,
        *,
        use_playwright: bool = True,
        fetch_fn: Any | None = None,
    ) -> None:
        self._settings = settings
        self._use_playwright = use_playwright
        self._fetch_fn = fetch_fn
        self._playwright: Any | None = None
        self._browser: Any | None = None
        self._running = False

    @property
    def is_running(self) -> bool:
        """Whether the browser is currently running."""
        return self._running

    async def start(self) -> None:
        """Start the browser. Launches Playwright Chromium in headless mode.

        If Playwright is unavailable, falls back to httpx mode.

        Does nothing if already running.
        """
        if self._running:
            return

        if self._use_playwright:
            try:
                from playwright.async_api import async_playwright

                self._playwright = await async_playwright().start()
                self._browser = await self._playwright.chromium.launch(
                    headless=True,
                    args=[
                        "--disable-gpu",
                        "--no-sandbox",
                        "--disable-dev-shm-usage",
                    ],
                )
                self._running = True
                log.info("playwright_browser_started")
            except Exception as exc:
                log.warning(
                    "playwright_unavailable",
                    error=str(exc),
                    fallback="httpx",
                )
                self._use_playwright = False
                self._running = True
        else:
            self._running = True
            log.info("httpx_browser_mode_started")

    async def stop(self) -> None:
        """Stop the browser and clean up resources.

        Does nothing if not running.
        """
        if not self._running:
            return

        if self._browser is not None:
            try:
                await self._browser.close()
            except Exception as exc:
                log.warning("browser_close_error", error=str(exc))
            self._browser = None

        if self._playwright is not None:
            try:
                await self._playwright.stop()
            except Exception as exc:
                log.warning("playwright_stop_error", error=str(exc))
            self._playwright = None

        self._running = False
        log.info("browser_stopped")

    async def get_page_content(
        self,
        url: str,
        *,
        content_format: ContentFormat = ContentFormat.TEXT,
        timeout_ms: int = _DEFAULT_TIMEOUT_MS,
        max_chars: int = _MAX_CONTENT_CHARS,
    ) -> PageContent:
        """Load a web page and extract its content.

        Args:
            url: The URL to load.
            content_format: Desired output format.
            timeout_ms: Page load timeout in milliseconds.
            max_chars: Maximum content characters before truncation.

        Returns:
            PageContent object with extracted text and metadata.

        Raises:
            OfflineError: If internet is disabled.
            WebBrowseError: If page loading or extraction fails.
        """
        if not self._settings.internet_enabled:
            raise OfflineError("browse")

        start = time.perf_counter()

        # Custom fetch function (for testing)
        if self._fetch_fn is not None:
            html = await self._fetch_fn(url)
            title, content = _extract_content(html, content_format)
            was_truncated = len(content) > max_chars
            if was_truncated:
                content = content[:max_chars] + "\n\n[Content truncated]"
            load_time_ms = (time.perf_counter() - start) * 1000
            return PageContent(
                url=url,
                title=title,
                content=content,
                content_format=content_format,
                load_time_ms=load_time_ms,
                was_truncated=was_truncated,
            )

        # Fetch via Playwright or httpx
        try:
            if self._use_playwright and self._browser is not None:
                html, page_title = await self._playwright_fetch(url, timeout_ms)
            else:
                html, page_title = await self._httpx_fetch(url, timeout_ms)
        except OfflineError:
            raise
        except Exception as exc:
            raise WebBrowseError(url, str(exc)) from exc

        # Extract content
        title, content = _extract_content(html, content_format)
        if page_title and not title:
            title = page_title

        was_truncated = len(content) > max_chars
        if was_truncated:
            content = content[:max_chars] + "\n\n[Content truncated]"

        load_time_ms = (time.perf_counter() - start) * 1000

        log.info(
            "page_loaded",
            url=url[:100],
            chars=len(content),
            truncated=was_truncated,
            elapsed_ms=round(load_time_ms, 1),
        )

        return PageContent(
            url=url,
            title=title,
            content=content,
            content_format=content_format,
            load_time_ms=load_time_ms,
            was_truncated=was_truncated,
        )

    async def screenshot(
        self,
        url: str,
        *,
        timeout_ms: int = _DEFAULT_TIMEOUT_MS,
        full_page: bool = False,
    ) -> bytes:
        """Capture a screenshot of a web page.

        Requires Playwright mode. Falls back to error if Playwright is unavailable.

        Args:
            url: The URL to screenshot.
            timeout_ms: Page load timeout.
            full_page: Whether to capture the full scrollable page.

        Returns:
            Screenshot as PNG bytes.

        Raises:
            OfflineError: If internet is disabled.
            WebBrowseError: If screenshot capture fails.
        """
        if not self._settings.internet_enabled:
            raise OfflineError("screenshot")

        if not self._use_playwright or self._browser is None:
            raise WebBrowseError(url, "Playwright required for screenshots but unavailable")

        try:
            page = await self._browser.new_page(user_agent=_USER_AGENT)
            try:
                await page.goto(url, timeout=timeout_ms, wait_until="networkidle")
                screenshot_bytes: bytes = await page.screenshot(full_page=full_page)
                return screenshot_bytes
            finally:
                await page.close()
        except Exception as exc:
            raise WebBrowseError(url, f"Screenshot failed: {exc}") from exc

    async def _playwright_fetch(
        self,
        url: str,
        timeout_ms: int,
    ) -> tuple[str, str]:
        """Fetch page HTML using Playwright.

        Args:
            url: URL to load.
            timeout_ms: Load timeout.

        Returns:
            Tuple of (html_content, page_title).
        """
        page = await self._browser.new_page(user_agent=_USER_AGENT)
        try:
            await page.goto(url, timeout=timeout_ms, wait_until="domcontentloaded")
            # Wait a bit for dynamic content to load
            await asyncio.sleep(1.0)
            html = await page.content()
            title = await page.title()
            return html, title
        finally:
            await page.close()

    async def _httpx_fetch(
        self,
        url: str,
        timeout_ms: int,
    ) -> tuple[str, str]:
        """Fetch page HTML using httpx (lightweight fallback).

        Args:
            url: URL to fetch.
            timeout_ms: Request timeout.

        Returns:
            Tuple of (html_content, page_title).
        """
        import httpx

        timeout_s = timeout_ms / 1000.0

        async with httpx.AsyncClient(
            timeout=timeout_s,
            follow_redirects=True,
            headers={"User-Agent": _USER_AGENT},
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            html = resp.text

        # Extract title from HTML
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        title_tag = soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else ""

        return html, title


def _extract_content(html: str, content_format: ContentFormat) -> tuple[str, str]:
    """Extract clean content from raw HTML.

    Args:
        html: Raw HTML string.
        content_format: Desired output format.

    Returns:
        Tuple of (title, content).
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")

    # Extract title
    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else ""

    if content_format == ContentFormat.HTML:
        # Return cleaned HTML (remove scripts/styles but keep structure)
        for tag in soup(_REMOVE_TAGS):
            tag.decompose()
        return title, str(soup)

    if content_format == ContentFormat.MARKDOWN:
        return title, _html_to_markdown(soup)

    # Default: plain text
    return title, _html_to_text(soup)


def _html_to_text(soup: Any) -> str:
    """Convert BeautifulSoup parsed HTML to clean plain text.

    Args:
        soup: Parsed BeautifulSoup object.

    Returns:
        Clean plain text.
    """
    # Remove unwanted tags
    for tag in soup(_REMOVE_TAGS):
        tag.decompose()

    # Get text
    text = soup.get_text(separator="\n", strip=True)

    # Clean up whitespace — collapse multiple blank lines
    lines = []
    prev_blank = False
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            if not prev_blank:
                lines.append("")
            prev_blank = True
        else:
            lines.append(stripped)
            prev_blank = False

    return "\n".join(lines)


def _html_to_markdown(soup: Any) -> str:
    """Convert BeautifulSoup parsed HTML to basic markdown.

    Preserves headers, links, lists, and basic formatting.

    Args:
        soup: Parsed BeautifulSoup object.

    Returns:
        Markdown-formatted text.
    """
    # Remove unwanted tags
    for tag in soup(_REMOVE_TAGS):
        tag.decompose()

    lines: list[str] = []

    for element in soup.find_all(
        ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "blockquote", "pre", "code"]
    ):
        tag_name = element.name
        text = element.get_text(strip=True)

        if not text:
            continue

        if tag_name.startswith("h"):
            level = int(tag_name[1])
            lines.append(f"{'#' * level} {text}")
            lines.append("")
        elif tag_name == "p":
            lines.append(text)
            lines.append("")
        elif tag_name == "li":
            lines.append(f"- {text}")
        elif tag_name == "blockquote":
            lines.append(f"> {text}")
            lines.append("")
        elif tag_name in ("pre", "code"):
            lines.append(f"```\n{text}\n```")
            lines.append("")
        else:
            lines.append(text)

    return "\n".join(lines)
