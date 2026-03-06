"""LLM-based relationship and nuanced entity extraction.

Supplements the spaCy NER extractor (entity_extractor.py) with deeper
extraction capabilities using the LLM. While spaCy catches surface-level
entities (people, places, orgs), this module extracts:

- Relationships between entities (who did what with whom)
- Goals, decisions, and commitments
- Habits and recurring patterns
- Emotional states tied to events/people
- Milestones and achievements
- Health observations
- Financial events

The extractor sends text to the LLM with a structured prompt and parses
the JSON response into GraphEntity/GraphRelation objects.

Architecture:
    - Async (uses LLMClient from ara.core.llm)
    - Prompt engineering for JSON-structured output
    - Fallback parsing for malformed LLM responses
    - Designed to work offline with local Qwen3-30B-A3B

Usage:
    from ara.memory.relationship_extractor import RelationshipExtractor

    extractor = RelationshipExtractor(llm_client)
    result = await extractor.extract(text, existing_entities=spacy_entities)
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import structlog

from ara.utils.types import (
    ChatMessage,
    EntityType,
    MessageRole,
    RelationType,
)

if TYPE_CHECKING:
    from ara.core.llm import LLMClient
    from ara.memory.entity_extractor import ExtractedEntity

log = structlog.get_logger(component="relationship_extractor")

# ── Prompt Templates ─────────────────────────────────────────────────────────

_EXTRACTION_SYSTEM_PROMPT = """\
You are an entity and relationship extraction system. Given a text passage \
(usually from a personal diary or notes), extract structured information.

Return ONLY valid JSON with this exact schema:
{
  "entities": [
    {
      "name": "string",
      "type": "person|place|event|project|goal|habit|emotion_tag|topic|organization|document|decision|milestone|health|financial|relationship|media",
      "aliases": ["string"],
      "confidence": 0.0-1.0
    }
  ],
  "relations": [
    {
      "source": "entity name",
      "target": "entity name",
      "type": "interacted_with|located_at|worked_on|felt|discussed|planned|completed|cancelled|mentioned_in|related_to|belongs_to|caused|preceded|followed|part_of|consumed|created|experienced",
      "confidence": 0.0-1.0,
      "context": "brief description of the relationship"
    }
  ]
}

Rules:
- Extract ALL entities, not just people and places. Include goals, decisions, habits, emotions, health states, milestones.
- For diary text, "I" or "me" refers to the author — do NOT create an entity for "I".
- Relations should connect entity names exactly as they appear in the entities list.
- Focus on meaningful relationships, not trivial mentions.
- confidence: 1.0 = explicit in text, 0.7 = strongly implied, 0.5 = inferred.
- Return ONLY the JSON object, no markdown, no explanation.\
"""

_EXTRACTION_USER_TEMPLATE = """\
Extract entities and relationships from this text:

---
{text}
---

{context_hint}\
"""


# ── Data Types ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ExtractedRelationship:
    """A relationship extracted by the LLM.

    Attributes:
        source_name: Name of the source entity.
        target_name: Name of the target entity.
        relation_type: Mapped RelationType.
        confidence: LLM's confidence estimate.
        context: Brief description of the relationship.
    """

    source_name: str
    target_name: str
    relation_type: RelationType
    confidence: float = 0.8
    context: str = ""


@dataclass
class LLMExtractionResult:
    """Result of LLM-based extraction.

    Attributes:
        entities: Entities found by the LLM (may overlap with spaCy entities).
        relationships: Relationships between entities.
        raw_response: The raw LLM output for debugging.
        elapsed_ms: Processing time.
        parse_success: Whether the JSON was parsed successfully.
    """

    entities: list[ExtractedEntity] = field(default_factory=list)
    relationships: list[ExtractedRelationship] = field(default_factory=list)
    raw_response: str = ""
    elapsed_ms: float = 0.0
    parse_success: bool = True


# ── Valid type mappings ──────────────────────────────────────────────────────

_ENTITY_TYPE_MAP: dict[str, EntityType] = {t.value: t for t in EntityType}
_RELATION_TYPE_MAP: dict[str, RelationType] = {t.value: t for t in RelationType}


# ── Relationship Extractor ───────────────────────────────────────────────────


class RelationshipExtractor:
    """LLM-based relationship and nuanced entity extractor.

    Sends text to the LLM with a structured extraction prompt and parses
    the response. Works asynchronously with the LLMClient.

    Args:
        llm_client: The async LLM client to use for inference.
        model: Optional model override (defaults to LLM client's active model).
        temperature: Temperature for extraction (lower = more consistent).
        max_tokens: Maximum tokens for LLM response.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        *,
        model: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> None:
        self._llm = llm_client
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    async def extract(
        self,
        text: str,
        *,
        existing_entities: list[ExtractedEntity] | None = None,
        source_hint: str = "",
    ) -> LLMExtractionResult:
        """Extract entities and relationships from text using the LLM.

        Args:
            text: The text to analyze.
            existing_entities: Entities already found by spaCy (provided as
                context so the LLM can focus on what spaCy missed).
            source_hint: Optional hint about the source (e.g. "diary entry
                from 2024-01-15") to help the LLM understand context.

        Returns:
            LLMExtractionResult with entities and relationships.

        Raises:
            GraphError: If the LLM call fails entirely.
        """
        start = time.perf_counter()

        # Build context hint
        context_parts: list[str] = []
        if existing_entities:
            known = ", ".join(f"{e.name} ({e.entity_type.value})" for e in existing_entities[:20])
            context_parts.append(
                f"Already identified entities: {known}. "
                "Focus on relationships between these AND any entities you find "
                "that were missed (especially goals, decisions, habits, emotions)."
            )
        if source_hint:
            context_parts.append(f"Source: {source_hint}")

        context_hint = "\n".join(context_parts) if context_parts else ""

        user_content = _EXTRACTION_USER_TEMPLATE.format(
            text=text[:8000],  # Limit to ~8K chars to fit context window
            context_hint=context_hint,
        )

        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=_EXTRACTION_SYSTEM_PROMPT),
            ChatMessage(role=MessageRole.USER, content=user_content),
        ]

        try:
            raw_response = await self._llm.generate(
                messages,
                model=self._model,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000
            log.warning(
                "llm_extraction_failed",
                error=str(exc),
                elapsed_ms=round(elapsed_ms, 1),
            )
            return LLMExtractionResult(
                raw_response=str(exc),
                elapsed_ms=elapsed_ms,
                parse_success=False,
            )

        elapsed_ms = (time.perf_counter() - start) * 1000

        # Parse the JSON response
        result = _parse_llm_response(raw_response)
        result.raw_response = raw_response
        result.elapsed_ms = elapsed_ms

        log.debug(
            "llm_extraction_complete",
            entities=len(result.entities),
            relationships=len(result.relationships),
            parse_success=result.parse_success,
            elapsed_ms=round(elapsed_ms, 1),
        )

        return result

    async def extract_batch(
        self,
        texts: list[str],
        *,
        source_hints: list[str] | None = None,
    ) -> list[LLMExtractionResult]:
        """Extract from multiple texts sequentially.

        LLM calls are sequential (not parallel) to avoid overloading
        the local inference server.

        Args:
            texts: List of texts to process.
            source_hints: Optional list of source hints, one per text.

        Returns:
            List of LLMExtractionResult, one per input text.
        """
        results: list[LLMExtractionResult] = []
        hints = source_hints or [""] * len(texts)

        for text, hint in zip(texts, hints, strict=False):
            result = await self.extract(text, source_hint=hint)
            results.append(result)

        return results


# ── Response Parsing ─────────────────────────────────────────────────────────


def _parse_llm_response(raw: str) -> LLMExtractionResult:
    """Parse the LLM's JSON response into structured types.

    Handles common LLM output quirks:
    - Markdown code fences around JSON
    - Trailing commas
    - Missing fields
    - Invalid enum values
    """
    from ara.memory.entity_extractor import ExtractedEntity

    # Strip markdown code fences
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*\n?", "", cleaned)
    cleaned = re.sub(r"\n?```\s*$", "", cleaned)
    cleaned = cleaned.strip()

    # Try to find JSON object in the response
    json_match = re.search(r"\{[\s\S]*\}", cleaned)
    if not json_match:
        log.warning("llm_response_no_json", raw_length=len(raw))
        return LLMExtractionResult(parse_success=False)

    json_str = json_match.group(0)

    # Remove trailing commas before closing braces/brackets (common LLM mistake)
    json_str = re.sub(r",\s*([}\]])", r"\1", json_str)

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as exc:
        log.warning("llm_response_json_parse_failed", error=str(exc))
        return LLMExtractionResult(parse_success=False)

    # Parse entities
    entities: list[ExtractedEntity] = []
    for raw_entity in data.get("entities", []):
        name = str(raw_entity.get("name", "")).strip()
        if not name:
            continue

        type_str = str(raw_entity.get("type", "")).lower().strip()
        entity_type = _ENTITY_TYPE_MAP.get(type_str)
        if entity_type is None:
            continue

        confidence = _clamp_float(raw_entity.get("confidence", 0.8))

        entities.append(
            ExtractedEntity(
                name=name,
                entity_type=entity_type,
                confidence=confidence,
            )
        )

    # Parse relationships
    relationships: list[ExtractedRelationship] = []
    for raw_rel in data.get("relations", []):
        source = str(raw_rel.get("source", "")).strip()
        target = str(raw_rel.get("target", "")).strip()
        if not source or not target:
            continue

        type_str = str(raw_rel.get("type", "")).lower().strip()
        rel_type = _RELATION_TYPE_MAP.get(type_str)
        if rel_type is None:
            # Try common alternative names
            rel_type = _fuzzy_match_relation_type(type_str)
            if rel_type is None:
                continue

        confidence = _clamp_float(raw_rel.get("confidence", 0.8))
        context = str(raw_rel.get("context", "")).strip()

        relationships.append(
            ExtractedRelationship(
                source_name=source,
                target_name=target,
                relation_type=rel_type,
                confidence=confidence,
                context=context,
            )
        )

    return LLMExtractionResult(
        entities=entities,
        relationships=relationships,
        parse_success=True,
    )


def _clamp_float(value: object, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Clamp a value to [min_val, max_val], handling non-numeric input."""
    try:
        f = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.8
    return max(min_val, min(max_val, f))


def _fuzzy_match_relation_type(type_str: str) -> RelationType | None:
    """Try to match common alternative relation type names."""
    alternatives: dict[str, RelationType] = {
        "knows": RelationType.INTERACTED_WITH,
        "met": RelationType.INTERACTED_WITH,
        "talked_to": RelationType.INTERACTED_WITH,
        "visited": RelationType.LOCATED_AT,
        "lives_in": RelationType.LOCATED_AT,
        "works_at": RelationType.BELONGS_TO,
        "member_of": RelationType.BELONGS_TO,
        "studies": RelationType.DISCUSSED,
        "about": RelationType.RELATED_TO,
        "leads_to": RelationType.CAUSED,
        "resulted_in": RelationType.CAUSED,
        "before": RelationType.PRECEDED,
        "after": RelationType.FOLLOWED,
        "includes": RelationType.PART_OF,
        "contains": RelationType.PART_OF,
        "read": RelationType.CONSUMED,
        "watched": RelationType.CONSUMED,
        "listened_to": RelationType.CONSUMED,
        "built": RelationType.CREATED,
        "made": RelationType.CREATED,
        "wrote": RelationType.CREATED,
        "went_through": RelationType.EXPERIENCED,
        "suffered": RelationType.EXPERIENCED,
        "enjoyed": RelationType.EXPERIENCED,
        "achieved": RelationType.COMPLETED,
        "finished": RelationType.COMPLETED,
        "gave_up": RelationType.CANCELLED,
        "abandoned": RelationType.CANCELLED,
        "wants": RelationType.PLANNED,
        "intends": RelationType.PLANNED,
        "associated_with": RelationType.RELATED_TO,
        "connected_to": RelationType.RELATED_TO,
    }
    return alternatives.get(type_str)
