"""Research agent — deep multi-step investigation with citations.

Handles complex queries requiring multiple search → read → follow → synthesize
cycles. Produces comprehensive, cited research reports.

Architecture:
    - Multi-step: iteratively searches, reads, and follows promising leads
    - Citation tracking: every claim maps back to a source URL
    - Configurable depth: max_steps controls how deep the research goes
    - Combines web search results with local memory for personal context

Flow:
    1. Decompose query into sub-questions
    2. For each sub-question:
       a. Search the web
       b. Read top results
       c. Extract key findings
    3. Synthesize all findings into a structured report
    4. Add citations

Usage:
    agent = ResearchAgent(settings, llm_client, web_search_agent, web_browse_agent)
    await agent.connect()
    result_state = await agent.execute(state)
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ara.agents.base import AgentCapability, AgentPermission, BaseAgent
from ara.utils.logging import get_logger
from ara.utils.types import ChatMessage, IntentType, MessageRole

if TYPE_CHECKING:
    from ara.agents.state import AgentState
    from ara.agents.web_browse_agent import WebBrowseAgent
    from ara.agents.web_search_agent import WebSearchAgent
    from ara.config.settings import AraSettings
    from ara.core.llm import LLMClient

log = get_logger("research_agent")


# ── Capability Declaration ───────────────────────────────────────────────────

_RESEARCH_CAPABILITY = AgentCapability(
    name="research_agent",
    display_name="Deep Research",
    description=(
        "Conducts in-depth, multi-step research on complex topics. Searches "
        "multiple sources, reads pages, follows leads, and synthesizes "
        "a comprehensive report with citations."
    ),
    supported_intents=frozenset(
        {
            IntentType.RESEARCH,
        }
    ),
    required_permissions=frozenset(
        {
            AgentPermission.WEB_SEARCH,
            AgentPermission.WEB_BROWSE,
            AgentPermission.LLM_GENERATE,
        }
    ),
    priority=100,
)


# ── Research Data Structures ─────────────────────────────────────────────────


@dataclass
class ResearchFinding:
    """A single finding from the research process."""

    source_url: str
    source_title: str
    key_points: list[str]
    relevance: str  # "high", "medium", "low"
    sub_question: str


@dataclass
class ResearchPlan:
    """Decomposed research plan with sub-questions."""

    original_query: str
    sub_questions: list[str]
    search_queries: list[str]


# ── Prompts ──────────────────────────────────────────────────────────────────

_DECOMPOSE_PROMPT = """You are a research planner. Decompose the user's research question into 2-4 specific sub-questions that together would fully answer the main question.

Return a JSON object:
{
  "sub_questions": ["question 1", "question 2", ...],
  "search_queries": ["optimized search query 1", "optimized search query 2", ...]
}

Rules:
- Sub-questions should be specific and searchable
- Search queries should be optimized for web search (no question marks, key terms only)
- 2-4 sub-questions maximum
- Return ONLY the JSON object"""

_EXTRACT_FINDINGS_PROMPT = """Extract key findings from this web page content that are relevant to the research question.

Return a JSON object:
{
  "key_points": ["finding 1", "finding 2", ...],
  "relevance": "high|medium|low",
  "follow_up_urls": []
}

Rules:
- Extract only factual, relevant information
- Each key point should be a complete, standalone statement
- Mark relevance based on how well the content answers the research question
- Return ONLY the JSON object"""

_SYNTHESIS_PROMPT = """You are ARA, presenting research findings to the user. Synthesize the collected findings into a clear, comprehensive report.

Rules:
- Organize by theme or sub-topic, not by source
- Cite sources naturally: "According to [title]..." or "[title] reports that..."
- Include a brief summary at the top
- Mention any conflicting information between sources
- If the research is incomplete, say what questions remain unanswered
- Be thorough but readable
- End with a brief conclusion or recommendation if appropriate"""


class ResearchAgent(BaseAgent):
    """Research agent — multi-step deep research with citations.

    Args:
        settings: Application settings.
        llm_client: LLM for planning, extraction, and synthesis.
        search_agent: Web search agent for finding sources.
        browse_agent: Web browse agent for reading pages.
    """

    def __init__(
        self,
        settings: AraSettings,
        llm_client: LLMClient,
        search_agent: WebSearchAgent | None = None,
        browse_agent: WebBrowseAgent | None = None,
    ) -> None:
        super().__init__(settings)
        self._llm = llm_client
        self._search = search_agent
        self._browse = browse_agent

    @property
    def capability(self) -> AgentCapability:
        return _RESEARCH_CAPABILITY

    async def _execute(self, state: AgentState) -> AgentState:
        """Execute multi-step research: plan → search → read → synthesize.

        Args:
            state: Current orchestrator state.

        Returns:
            Updated state with comprehensive research response.
        """
        query = state.get("query", "")
        max_steps = self._settings.internet.max_research_steps

        if not query.strip():
            return {"response": "", "error": "Empty query"}  # type: ignore[return-value]

        if not self._settings.internet.internet_enabled:
            return {  # type: ignore[return-value]
                "response": ("I can't conduct research right now — internet access is disabled."),
                "error": "internet_disabled",
            }

        start = time.perf_counter()

        # Step 1: Decompose query into sub-questions
        plan = await self._decompose_query(query)

        log.info(
            "research_plan_created",
            sub_questions=len(plan.sub_questions),
            search_queries=len(plan.search_queries),
        )

        # Step 2: Research each sub-question
        all_findings: list[ResearchFinding] = []
        steps_taken = 0

        for i, (sub_q, search_q) in enumerate(
            zip(plan.sub_questions, plan.search_queries, strict=False)
        ):
            if steps_taken >= max_steps:
                log.info("research_max_steps_reached", steps=steps_taken)
                break

            log.info(
                "research_step",
                step=i + 1,
                sub_question=sub_q[:80],
            )

            # Search for this sub-question
            findings = await self._research_sub_question(sub_q, search_q)
            all_findings.extend(findings)
            steps_taken += 1

        # Step 3: Synthesize all findings
        response = await self._synthesize_findings(query, plan, all_findings)

        elapsed_ms = (time.perf_counter() - start) * 1000

        log.info(
            "research_complete",
            findings=len(all_findings),
            steps=steps_taken,
            elapsed_ms=round(elapsed_ms, 1),
        )

        return {  # type: ignore[return-value]
            "response": response,
            "metadata": {
                "research_steps": steps_taken,
                "findings_count": len(all_findings),
                "sub_questions": plan.sub_questions,
                "elapsed_ms": elapsed_ms,
            },
            "steps": steps_taken,
        }

    async def _decompose_query(self, query: str) -> ResearchPlan:
        """Decompose a complex query into sub-questions and search queries.

        Args:
            query: The user's research question.

        Returns:
            ResearchPlan with sub-questions and search queries.
        """
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=_DECOMPOSE_PROMPT),
            ChatMessage(role=MessageRole.USER, content=query),
        ]

        raw = await self._llm.generate(messages, temperature=0.3, max_tokens=500)

        try:
            cleaned = raw.strip()
            cleaned = re.sub(r"^```(?:json)?\s*\n?", "", cleaned)
            cleaned = re.sub(r"\n?```\s*$", "", cleaned)

            json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if json_match:
                cleaned = json_match.group(0)

            parsed = json.loads(cleaned)
            sub_questions = parsed.get("sub_questions", [query])
            search_queries = parsed.get("search_queries", sub_questions)

            return ResearchPlan(
                original_query=query,
                sub_questions=sub_questions[:4],  # Cap at 4
                search_queries=search_queries[:4],
            )
        except (json.JSONDecodeError, AttributeError):
            # Fallback: use the original query as the only sub-question
            return ResearchPlan(
                original_query=query,
                sub_questions=[query],
                search_queries=[query],
            )

    async def _research_sub_question(
        self,
        sub_question: str,
        search_query: str,
    ) -> list[ResearchFinding]:
        """Research a single sub-question: search → read → extract.

        Args:
            sub_question: The sub-question to research.
            search_query: Optimized search query.

        Returns:
            List of findings from this sub-question.
        """
        findings: list[ResearchFinding] = []

        # Search the web
        if self._search is not None:
            from ara.agents.state import create_initial_state

            search_state = create_initial_state(search_query)
            search_result = await self._search.execute(search_state)

            metadata = search_result.get("metadata", {})
            search_results = metadata.get("search_results", [])

            # Read top results
            for result in search_results[:3]:  # Top 3 results per sub-question
                url = result.get("url", "")
                title = result.get("title", "")
                snippet = result.get("snippet", "")

                if not url:
                    continue

                # Try to read the full page
                page_content = ""
                if self._browse is not None:
                    try:
                        page_content = await self._browse.fetch_page(url)
                    except Exception as exc:
                        log.warning("research_page_fetch_failed", url=url, error=str(exc))
                        page_content = snippet  # Fallback to snippet

                if not page_content:
                    page_content = snippet

                # Extract findings from page content
                extracted = await self._extract_findings(
                    sub_question, title, url, page_content[:6000]
                )
                if extracted:
                    findings.append(extracted)
        else:
            # No search agent available — create a stub finding
            findings.append(
                ResearchFinding(
                    source_url="",
                    source_title="No search available",
                    key_points=["Web search agent not configured"],
                    relevance="low",
                    sub_question=sub_question,
                )
            )

        return findings

    async def _extract_findings(
        self,
        sub_question: str,
        title: str,
        url: str,
        content: str,
    ) -> ResearchFinding | None:
        """Extract key findings from page content.

        Args:
            sub_question: The sub-question this content relates to.
            title: Page title.
            url: Page URL.
            content: Page text content.

        Returns:
            ResearchFinding or None if extraction fails.
        """
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=_EXTRACT_FINDINGS_PROMPT),
            ChatMessage(
                role=MessageRole.USER,
                content=(
                    f"Research question: {sub_question}\n\n"
                    f"Page title: {title}\n"
                    f"Page content:\n{content}"
                ),
            ),
        ]

        raw = await self._llm.generate(messages, temperature=0.2, max_tokens=500)

        try:
            cleaned = raw.strip()
            cleaned = re.sub(r"^```(?:json)?\s*\n?", "", cleaned)
            cleaned = re.sub(r"\n?```\s*$", "", cleaned)

            json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if json_match:
                cleaned = json_match.group(0)

            parsed = json.loads(cleaned)
            key_points = parsed.get("key_points", [])
            relevance = parsed.get("relevance", "medium")

            if not key_points:
                return None

            return ResearchFinding(
                source_url=url,
                source_title=title,
                key_points=key_points,
                relevance=relevance,
                sub_question=sub_question,
            )
        except (json.JSONDecodeError, AttributeError):
            return None

    async def _synthesize_findings(
        self,
        query: str,
        plan: ResearchPlan,
        findings: list[ResearchFinding],
    ) -> str:
        """Synthesize all findings into a comprehensive research report.

        Args:
            query: Original user query.
            plan: The research plan used.
            findings: All collected findings.

        Returns:
            Synthesized research report.
        """
        if not findings:
            return (
                "I wasn't able to find relevant information for your research question. "
                "Could you try rephrasing it or breaking it into more specific questions?"
            )

        # Format findings for synthesis
        findings_text_parts: list[str] = []
        for i, finding in enumerate(findings, 1):
            points = "\n".join(f"  - {p}" for p in finding.key_points)
            findings_text_parts.append(
                f"Source [{i}]: {finding.source_title}\n"
                f"  URL: {finding.source_url}\n"
                f"  Relevance: {finding.relevance}\n"
                f"  Sub-question: {finding.sub_question}\n"
                f"  Findings:\n{points}"
            )

        findings_text = "\n\n".join(findings_text_parts)

        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=_SYNTHESIS_PROMPT),
            ChatMessage(
                role=MessageRole.USER,
                content=(
                    f"Research question: {query}\n\n"
                    f"Sub-questions investigated:\n"
                    + "\n".join(f"- {sq}" for sq in plan.sub_questions)
                    + f"\n\nCollected findings:\n\n{findings_text}"
                ),
            ),
        ]

        return await self._llm.generate(messages, max_tokens=2000)
