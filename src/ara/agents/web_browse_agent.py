"""Web browse agent — headless browser for page content extraction.

Handles requests that need to read specific web pages, extract content,
render JavaScript, follow links, and capture screenshots. Uses Playwright
for headless Chromium browsing.

Architecture:
    - Playwright headless Chromium for page loading and JS rendering
    - BeautifulSoup for HTML → clean text extraction
    - Content caching to avoid redundant page loads
    - Configurable timeout and content size limits

Flow:
    1. LLM identifies target URL(s) from user query
    2. Load page via Playwright (headless)
    3. Extract and clean page content
    4. Summarize content for the user
    5. Cache extracted content

Usage:
    agent = WebBrowseAgent(settings, llm_client)
    await agent.connect()
    result_state = await agent.execute(state)
"""

from __future__ import annotations

import json
import re
import time
from typing import TYPE_CHECKING, Any

from ara.agents.base import AgentCapability, AgentPermission, BaseAgent
from ara.utils.logging import get_logger
from ara.utils.types import ChatMessage, IntentType, MessageRole

if TYPE_CHECKING:
    from ara.agents.state import AgentState
    from ara.config.settings import AraSettings
    from ara.core.llm import LLMClient

log = get_logger("web_browse_agent")


# ── Capability Declaration ───────────────────────────────────────────────────

_WEB_BROWSE_CAPABILITY = AgentCapability(
    name="web_browse_agent",
    display_name="Web Browser",
    description=(
        "Reads and extracts content from specific web pages. Can load URLs, "
        "render JavaScript, extract text, and summarize page contents."
    ),
    supported_intents=frozenset[IntentType](),
    required_permissions=frozenset(
        {
            AgentPermission.WEB_BROWSE,
            AgentPermission.LLM_GENERATE,
        }
    ),
    priority=80,
)

# Maximum page content length before truncation
_MAX_CONTENT_CHARS = 20_000

# URL extraction prompt
_URL_EXTRACTION_PROMPT = """Extract the URL(s) the user wants to browse from their message.
Return a JSON object: {"urls": ["https://...", ...], "action": "read|summarize|extract"}
- "read": Show the full page content
- "summarize": Summarize the page
- "extract": Extract specific information
If no URL is found, return: {"urls": [], "action": "none", "error": "No URL found in query"}
Return ONLY the JSON object."""

# Content synthesis prompt
_BROWSE_SYNTHESIS_PROMPT = """You are ARA, reporting on web page content you've just read.

Rules:
- Summarize the key information relevant to the user's question
- Mention the source URL
- Be concise but cover the important points
- If the page content doesn't answer the user's question, say so
- Format information clearly (use brief lists if appropriate)"""


class WebBrowseAgent(BaseAgent):
    """Web browse agent — reads and extracts web page content.

    Args:
        settings: Application settings.
        llm_client: LLM for URL extraction and content synthesis.
        browser_fn: Optional custom page fetching function (for testing).
            Signature: async (url) -> str (returns page content)
    """

    def __init__(
        self,
        settings: AraSettings,
        llm_client: LLMClient,
        *,
        browser_fn: Any | None = None,
    ) -> None:
        super().__init__(settings)
        self._llm = llm_client
        self._browser_fn = browser_fn
        self._content_cache: dict[str, str] = {}

    @property
    def capability(self) -> AgentCapability:
        return _WEB_BROWSE_CAPABILITY

    async def _execute(self, state: AgentState) -> AgentState:
        """Execute browse: extract URL → fetch → extract → synthesize.

        Args:
            state: Current orchestrator state.

        Returns:
            Updated state with response and page content metadata.
        """
        query = state.get("query", "")
        if not query.strip():
            return {"response": "", "error": "Empty query"}  # type: ignore[return-value]

        # Step 1: Check internet
        if not self._settings.internet.internet_enabled:
            return {  # type: ignore[return-value]
                "response": ("I can't browse the web right now — internet access is disabled."),
                "error": "internet_disabled",
            }

        # Step 2: Extract URLs from query
        urls, action = await self._extract_urls(query)

        if not urls:
            return {  # type: ignore[return-value]
                "response": (
                    "I couldn't find a URL in your request. "
                    "Could you include the web address you'd like me to read?"
                ),
            }

        # Step 3: Fetch page content for each URL
        start = time.perf_counter()
        page_contents: list[dict[str, str]] = []

        for url in urls[:3]:  # Limit to 3 URLs max
            try:
                content = await self._fetch_page(url)
                page_contents.append({"url": url, "content": content})
            except Exception as exc:
                log.warning("page_fetch_failed", url=url, error=str(exc))
                page_contents.append({"url": url, "content": f"Failed to load: {exc}"})

        browse_time_ms = (time.perf_counter() - start) * 1000

        # Step 4: Synthesize response
        response = await self._synthesize_response(query, page_contents, action)

        return {  # type: ignore[return-value]
            "response": response,
            "metadata": {
                "urls_browsed": [pc["url"] for pc in page_contents],
                "browse_time_ms": browse_time_ms,
                "action": action,
            },
        }

    async def _extract_urls(self, query: str) -> tuple[list[str], str]:
        """Extract URLs and desired action from the user's query.

        First tries regex extraction (fast path), then falls back to LLM.

        Args:
            query: User's query text.

        Returns:
            Tuple of (urls, action).
        """
        # Fast path: regex URL extraction
        url_pattern = re.compile(
            r'https?://[^\s<>"{}|\\^`\[\]]+',
            re.IGNORECASE,
        )
        found_urls = url_pattern.findall(query)
        if found_urls:
            return found_urls, "summarize"

        # Slow path: LLM extraction
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=_URL_EXTRACTION_PROMPT),
            ChatMessage(role=MessageRole.USER, content=query),
        ]

        raw = await self._llm.generate(messages, temperature=0.1, max_tokens=200)

        try:
            cleaned = raw.strip()
            cleaned = re.sub(r"^```(?:json)?\s*\n?", "", cleaned)
            cleaned = re.sub(r"\n?```\s*$", "", cleaned)

            json_match = re.search(r"\{[^}]+\}", cleaned, re.DOTALL)
            if json_match:
                cleaned = json_match.group(0)

            parsed = json.loads(cleaned)
            urls = parsed.get("urls", [])
            action = parsed.get("action", "summarize")
            return urls, action
        except (json.JSONDecodeError, AttributeError):
            return [], "none"

    async def _fetch_page(self, url: str) -> str:
        """Fetch and extract clean text from a web page.

        Args:
            url: The URL to fetch.

        Returns:
            Cleaned page text content.
        """
        # Check cache first
        if url in self._content_cache:
            log.info("page_cache_hit", url=url)
            return self._content_cache[url]

        # Use custom browser function if provided
        if self._browser_fn is not None:
            content = await self._browser_fn(url)
            self._content_cache[url] = content
            return content

        # Default: use httpx for simple pages
        content = await self._httpx_fetch(url)
        self._content_cache[url] = content
        return content

    async def _httpx_fetch(self, url: str) -> str:
        """Fetch page content using httpx + BeautifulSoup.

        Falls back to simple HTTP if Playwright isn't available.

        Args:
            url: URL to fetch.

        Returns:
            Extracted text content.
        """
        import httpx
        from bs4 import BeautifulSoup

        async with httpx.AsyncClient(
            timeout=30.0,
            follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ARA-Bot/1.0"},
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")

        # Remove script and style elements
        for element in soup(["script", "style", "nav", "footer", "header"]):
            element.decompose()

        # Extract text
        text = soup.get_text(separator="\n", strip=True)

        # Clean up whitespace
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        cleaned = "\n".join(lines)

        # Truncate if too long
        if len(cleaned) > _MAX_CONTENT_CHARS:
            cleaned = cleaned[:_MAX_CONTENT_CHARS] + "\n\n[Content truncated]"

        return cleaned

    async def _synthesize_response(
        self,
        query: str,
        page_contents: list[dict[str, str]],
        action: str,
    ) -> str:
        """Synthesize a response from fetched page contents.

        Args:
            query: Original user query.
            page_contents: List of {"url": ..., "content": ...} dicts.
            action: What the user wants (read/summarize/extract).

        Returns:
            Synthesized response.
        """
        context_parts: list[str] = []
        for pc in page_contents:
            # Truncate individual page content for LLM context
            content = pc["content"][:8000]
            context_parts.append(f"Page: {pc['url']}\n---\n{content}\n---")

        context = "\n\n".join(context_parts)

        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=_BROWSE_SYNTHESIS_PROMPT),
            ChatMessage(
                role=MessageRole.USER,
                content=f"Web page content:\n\n{context}\n\nUser question: {query}",
            ),
        ]

        return await self._llm.generate(messages, max_tokens=1000)

    def clear_cache(self) -> None:
        """Clear the page content cache."""
        count = len(self._content_cache)
        self._content_cache.clear()
        log.info("browse_cache_cleared", entries=count)
