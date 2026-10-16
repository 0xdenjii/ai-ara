"""Intent classification router — LLM-based query routing for agent selection.

Classifies user queries into one of 10 IntentTypes using the primary LLM,
then maps each intent to the appropriate agent subgraph. This is the entry
point for every user query in the orchestration pipeline.

Architecture:
    - Single code path: always uses LLM for classification (no keyword fallback)
    - Structured JSON output: LLM returns {"intent": "...", "confidence": 0.X, "reasoning": "..."}
    - Graceful fallback: if LLM fails or returns invalid JSON, defaults to GENERAL
    - Configurable routing table: maps IntentType → agent name

Usage:
    router = IntentRouter(llm_client, settings)
    result = await router.classify("what did I write about Mom last week?")
    # result.intent == IntentType.REMINISCING
    # result.agent == "memory_agent"
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ara.utils.errors import RoutingError
from ara.utils.logging import get_logger
from ara.utils.types import ChatMessage, IntentType, MessageRole

if TYPE_CHECKING:
    from ara.config.settings import AraSettings
    from ara.core.llm import LLMClient

log = get_logger("router")


# ── Classification Prompt ────────────────────────────────────────────────────

_CLASSIFICATION_PROMPT = """You are a query intent classifier. Your ONLY job is to classify the user's query into exactly one intent category.

Categories:
- factual: Questions about facts, knowledge, how things work, explanations, definitions, math, coding
- emotional: Expressing feelings, venting, seeking emotional support, discussing moods, relationships
- planning: Making plans, setting goals, scheduling, organizing, task management, to-do lists
- reminiscing: Recalling past events, "remember when", diary references, personal history, nostalgia
- creative: Writing, brainstorming, storytelling, poetry, jokes, roleplay, creative tasks
- vision: Describing or asking about images, screenshots, photos, visual content (requires an image)
- desktop: Computer control, opening apps, file operations, system commands, automation tasks
- web_search: Current events, latest news, real-time data, stock prices, weather, anything requiring live internet
- research: Deep multi-step investigation, comparing products, analyzing topics from multiple sources, literature review
- general: Casual chat, greetings, small talk, unclear intent, or doesn't fit any above category

Rules:
1. Return ONLY a JSON object with exactly these fields: {"intent": "<category>", "confidence": <0.0-1.0>, "reasoning": "<1 sentence>"}
2. Do NOT include any text before or after the JSON object
3. If the query mentions past events, diary entries, or personal memories → reminiscing
4. If the query asks about current/real-time information → web_search
5. If the query asks for in-depth analysis from multiple sources → research
6. If ambiguous between two categories, pick the one most likely to help the user
7. Confidence should reflect how certain you are (0.5 = guessing, 0.9+ = very certain)"""


# ── Route Table ──────────────────────────────────────────────────────────────

_INTENT_TO_AGENT: dict[IntentType, str] = {
    IntentType.FACTUAL: "rag_agent",
    IntentType.EMOTIONAL: "rag_agent",
    IntentType.PLANNING: "rag_agent",
    IntentType.REMINISCING: "memory_agent",
    IntentType.CREATIVE: "rag_agent",
    IntentType.VISION: "rag_agent",
    IntentType.DESKTOP: "tool_agent",
    IntentType.WEB_SEARCH: "web_search_agent",
    IntentType.RESEARCH: "research_agent",
    IntentType.GENERAL: "rag_agent",
}

# Minimum confidence to accept the LLM's classification
_MIN_CONFIDENCE = 0.3


# ── Data Types ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ClassificationResult:
    """Result of intent classification.

    Attributes:
        intent: The classified intent type.
        agent: The agent name that should handle this query.
        confidence: LLM's self-reported confidence (0.0 to 1.0).
        reasoning: LLM's explanation for the classification.
        classification_time_ms: Wall-clock time for classification.
    """

    intent: IntentType
    agent: str
    confidence: float
    reasoning: str
    classification_time_ms: float = 0.0


@dataclass
class RouteOverride:
    """Manual routing override — bypasses LLM classification.

    Use this when the caller already knows which agent should handle the query
    (e.g., user explicitly asked to search the web, or a follow-up in an
    active agent session).
    """

    intent: IntentType
    agent: str
    reason: str = "manual_override"


# ── Router ───────────────────────────────────────────────────────────────────


class IntentRouter:
    """LLM-based intent classifier and agent router.

    Sends the user's query to the LLM with a classification prompt,
    parses the structured JSON response, and maps the intent to an
    agent name from the routing table.

    Args:
        llm_client: The LLM client for generating classifications.
        settings: Application settings for model config.
        route_table: Optional custom intent→agent mapping (overrides default).
    """

    def __init__(
        self,
        llm_client: LLMClient,
        settings: AraSettings,
        *,
        route_table: dict[IntentType, str] | None = None,
    ) -> None:
        self._llm = llm_client
        self._settings = settings
        self._route_table = route_table or dict(_INTENT_TO_AGENT)
        self._classification_temperature = 0.1  # Low temp for deterministic classification

    @property
    def route_table(self) -> dict[IntentType, str]:
        """Current intent-to-agent routing table (read-only copy)."""
        return dict(self._route_table)

    def update_route(self, intent: IntentType, agent_name: str) -> None:
        """Update the routing table for a specific intent.

        Args:
            intent: The intent type to reroute.
            agent_name: The agent name to route to.
        """
        self._route_table[intent] = agent_name
        log.info("route_updated", intent=intent.value, agent=agent_name)

    async def classify(
        self,
        query: str,
        *,
        conversation_history: list[ChatMessage] | None = None,
        override: RouteOverride | None = None,
    ) -> ClassificationResult:
        """Classify a user query and determine the target agent.

        Args:
            query: The user's raw query text.
            conversation_history: Optional recent conversation context for
                better classification (e.g., follow-up questions).
            override: If provided, bypasses LLM classification entirely.

        Returns:
            ClassificationResult with intent, target agent, and confidence.

        Raises:
            RoutingError: If classification fails and no fallback is possible.
        """
        start = time.perf_counter()

        # Fast path: manual override
        if override is not None:
            elapsed_ms = (time.perf_counter() - start) * 1000
            log.info(
                "route_override",
                intent=override.intent.value,
                agent=override.agent,
                reason=override.reason,
            )
            return ClassificationResult(
                intent=override.intent,
                agent=override.agent,
                confidence=1.0,
                reasoning=f"Manual override: {override.reason}",
                classification_time_ms=elapsed_ms,
            )

        # Build classification messages
        messages = self._build_classification_messages(query, conversation_history)

        try:
            # Call LLM for classification
            raw_response = await self._llm.generate(
                messages,
                temperature=self._classification_temperature,
                max_tokens=150,  # Classification needs very few tokens
            )

            # Parse the structured response
            result = self._parse_classification(raw_response, query)
            elapsed_ms = (time.perf_counter() - start) * 1000
            result_with_time = ClassificationResult(
                intent=result.intent,
                agent=result.agent,
                confidence=result.confidence,
                reasoning=result.reasoning,
                classification_time_ms=elapsed_ms,
            )

            log.info(
                "query_classified",
                query=query[:80],
                intent=result_with_time.intent.value,
                agent=result_with_time.agent,
                confidence=result_with_time.confidence,
                elapsed_ms=round(elapsed_ms, 1),
            )
            return result_with_time

        except RoutingError:
            raise
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000
            log.warning(
                "classification_failed_fallback",
                query=query[:80],
                error=str(exc),
                fallback_intent="general",
            )
            # Graceful fallback to GENERAL
            return ClassificationResult(
                intent=IntentType.GENERAL,
                agent=self._route_table[IntentType.GENERAL],
                confidence=0.0,
                reasoning=f"Classification failed ({exc!s}), defaulting to general",
                classification_time_ms=elapsed_ms,
            )

    async def classify_batch(
        self,
        queries: list[str],
    ) -> list[ClassificationResult]:
        """Classify multiple queries sequentially.

        Args:
            queries: List of user queries to classify.

        Returns:
            List of ClassificationResult in the same order as input.
        """
        results: list[ClassificationResult] = []
        for query in queries:
            result = await self.classify(query)
            results.append(result)
        return results

    def get_agent_for_intent(self, intent: IntentType) -> str:
        """Look up the agent name for a given intent type.

        Args:
            intent: The intent type to look up.

        Returns:
            The agent name registered for this intent.

        Raises:
            RoutingError: If no agent is registered for the intent.
        """
        agent = self._route_table.get(intent)
        if agent is None:
            raise RoutingError(
                f"No agent registered for intent '{intent.value}'",
                details={"intent": intent.value, "registered_intents": list(self._route_table)},
            )
        return agent

    # ── Private Methods ──────────────────────────────────────────────────────

    def _build_classification_messages(
        self,
        query: str,
        conversation_history: list[ChatMessage] | None,
    ) -> list[ChatMessage]:
        """Build the message list for the classification LLM call."""
        messages: list[ChatMessage] = [
            ChatMessage(role=MessageRole.SYSTEM, content=_CLASSIFICATION_PROMPT),
        ]

        # Include recent conversation context (last 3 turns max) for follow-up understanding
        if conversation_history:
            context_turns = conversation_history[-6:]  # last 3 user+assistant pairs
            messages.extend(
                ChatMessage(role=msg.role, content=msg.content) for msg in context_turns
            )

        # The actual query to classify
        messages.append(ChatMessage(role=MessageRole.USER, content=query))
        return messages

    def _parse_classification(
        self,
        raw_response: str,
        original_query: str,
    ) -> ClassificationResult:
        """Parse the LLM's JSON classification response.

        Handles:
            - Clean JSON
            - JSON wrapped in markdown code fences
            - Missing fields (uses defaults)
            - Invalid intent values (falls back to GENERAL)
            - Malformed JSON (falls back to GENERAL)

        Args:
            raw_response: Raw text from the LLM.
            original_query: The original query (for error context).

        Returns:
            Parsed ClassificationResult.
        """
        cleaned = raw_response.strip()

        # Strip think blocks (Qwen3.5 may emit <think>...</think> reasoning)
        cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL)
        # Also strip unclosed think blocks
        cleaned = re.sub(r"<think>.*", "", cleaned, flags=re.DOTALL)

        # Strip markdown code fences if present
        cleaned = re.sub(r"^```(?:json)?\s*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```\s*$", "", cleaned)
        cleaned = cleaned.strip()

        # Try to extract JSON object if there's surrounding text
        json_match = re.search(r"\{[^}]+\}", cleaned, re.DOTALL)
        if json_match:
            cleaned = json_match.group(0)

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            log.warning(
                "classification_json_parse_failed",
                raw_response=raw_response[:200],
                query=original_query[:80],
            )
            return self._fallback_result(original_query, "JSON parse failed")

        if not isinstance(parsed, dict):
            return self._fallback_result(original_query, "Response is not a JSON object")

        # Extract fields with defaults
        intent_str = parsed.get("intent", "general")
        confidence = parsed.get("confidence", 0.5)
        reasoning = parsed.get("reasoning", "No reasoning provided")

        # Validate intent string
        try:
            intent = IntentType(intent_str.lower().strip())
        except ValueError:
            log.warning(
                "unknown_intent_value",
                intent_raw=intent_str,
                query=original_query[:80],
            )
            intent = IntentType.GENERAL
            confidence = min(confidence, 0.3)  # Reduce confidence for unknown intents

        # Validate confidence range
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.5
        confidence = max(0.0, min(1.0, confidence))

        # If confidence is below threshold, downgrade to GENERAL
        if confidence < _MIN_CONFIDENCE and intent != IntentType.GENERAL:
            log.info(
                "low_confidence_downgrade",
                original_intent=intent.value,
                confidence=confidence,
                threshold=_MIN_CONFIDENCE,
            )
            intent = IntentType.GENERAL

        # Look up the target agent
        agent = self._route_table.get(intent, self._route_table[IntentType.GENERAL])

        return ClassificationResult(
            intent=intent,
            agent=agent,
            confidence=confidence,
            reasoning=str(reasoning),
        )

    def _fallback_result(self, query: str, reason: str) -> ClassificationResult:
        """Create a fallback classification when parsing fails."""
        return ClassificationResult(
            intent=IntentType.GENERAL,
            agent=self._route_table[IntentType.GENERAL],
            confidence=0.0,
            reasoning=f"Fallback: {reason}",
        )
