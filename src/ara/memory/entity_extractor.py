"""Entity extraction from text using spaCy NER.

Extracts named entities (people, places, organizations, dates, events)
from text chunks and maps them to ARA's knowledge graph schema. This is
the fast first-pass extractor — a supplementary LLM-based extractor
handles nuanced relationships and diary-specific entities (goals, habits,
emotions, decisions, etc.) in a separate module.

spaCy NER labels mapped to ARA EntityType:
    PERSON       -> EntityType.PERSON
    GPE          -> EntityType.PLACE  (geopolitical entities: countries, cities)
    LOC          -> EntityType.PLACE  (non-GPE locations: mountains, lakes)
    FAC          -> EntityType.PLACE  (facilities: buildings, airports)
    ORG          -> EntityType.ORGANIZATION
    PRODUCT      -> EntityType.PROJECT (or TOPIC, context-dependent)
    EVENT        -> EntityType.EVENT
    WORK_OF_ART  -> EntityType.MEDIA
    DATE         -> (extracted as temporal metadata, not a node)
    MONEY        -> EntityType.FINANCIAL
    NORP         -> EntityType.TOPIC  (nationalities, religious groups)
    LAW          -> EntityType.DOCUMENT

Usage:
    from ara.memory.entity_extractor import EntityExtractor

    extractor = EntityExtractor()
    extractor.load()

    results = extractor.extract("I met Mom in Springfield yesterday.")
    for entity_name, entity_type, confidence in results:
        print(f"{entity_name} ({entity_type.value})")

    extractor.unload()
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import structlog

from ara.utils.errors import GraphError
from ara.utils.types import EntityType

if TYPE_CHECKING:
    from spacy.language import Language
    from spacy.tokens import Doc

log = structlog.get_logger(component="entity_extractor")

# ── spaCy NER label → ARA EntityType mapping ────────────────────────────────

_SPACY_TO_ENTITY_TYPE: dict[str, EntityType] = {
    "PERSON": EntityType.PERSON,
    "GPE": EntityType.PLACE,
    "LOC": EntityType.PLACE,
    "FAC": EntityType.PLACE,
    "ORG": EntityType.ORGANIZATION,
    "PRODUCT": EntityType.TOPIC,
    "EVENT": EntityType.EVENT,
    "WORK_OF_ART": EntityType.MEDIA,
    "MONEY": EntityType.FINANCIAL,
    "NORP": EntityType.TOPIC,
    "LAW": EntityType.DOCUMENT,
}

# Labels to skip (not useful as graph entities)
_SKIP_LABELS: frozenset[str] = frozenset(
    {
        "DATE",
        "TIME",
        "PERCENT",
        "QUANTITY",
        "ORDINAL",
        "CARDINAL",
        "LANGUAGE",
    }
)

# Common noise entities that spaCy frequently misidentifies in diary text
_NOISE_ENTITIES: frozenset[str] = frozenset(
    {
        "today",
        "tomorrow",
        "yesterday",
        "tonight",
        "morning",
        "evening",
        "afternoon",
        "night",
        "the",
        "a",
        "an",
        "i",
        "me",
        "my",
        "we",
        "one",
        "two",
        "three",
        "first",
        "second",
        "third",
        "last",
        "next",
        "this",
        "that",
        "some",
        "few",
        "many",
        "much",
    }
)

# Minimum entity name length (single characters are noise)
_MIN_ENTITY_LENGTH = 2


# ── Data Types ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ExtractedEntity:
    """An entity extracted from text by NER.

    Attributes:
        name: The entity text as found in the source.
        entity_type: Mapped ARA EntityType.
        confidence: Extraction confidence (spaCy doesn't give per-entity
            confidence easily, so this is based on heuristics).
        spacy_label: The original spaCy NER label.
        start_char: Start character offset in the source text.
        end_char: End character offset in the source text.
    """

    name: str
    entity_type: EntityType
    confidence: float = 1.0
    spacy_label: str = ""
    start_char: int = 0
    end_char: int = 0


@dataclass
class ExtractionResult:
    """Result of entity extraction from a text chunk.

    Attributes:
        entities: Deduplicated list of extracted entities.
        raw_count: Total entities found before deduplication.
        elapsed_ms: Processing time in milliseconds.
        text_length: Length of input text in characters.
    """

    entities: list[ExtractedEntity] = field(default_factory=list)
    raw_count: int = 0
    elapsed_ms: float = 0.0
    text_length: int = 0


# ── Entity Extractor ─────────────────────────────────────────────────────────


class EntityExtractor:
    """Extract named entities from text using spaCy NER.

    Uses the transformer-based model (en_core_web_trf) for higher accuracy
    on informal diary text. Falls back to the small model if the transformer
    model is not available.

    The extractor is stateful — call ``load()`` before use and ``unload()``
    when done to free memory.

    Args:
        model_name: spaCy model to load. Defaults to "en_core_web_trf".
        confidence_threshold: Minimum confidence to include an entity.
        max_text_length: Maximum text length to process (longer texts are
            truncated to avoid memory issues with transformer models).
    """

    def __init__(
        self,
        *,
        model_name: str = "en_core_web_trf",
        confidence_threshold: float = 0.5,
        max_text_length: int = 50_000,
    ) -> None:
        self._model_name = model_name
        self._confidence_threshold = confidence_threshold
        self._max_text_length = max_text_length
        self._nlp: Language | None = None
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        """Whether the spaCy model is loaded."""
        return self._loaded

    def load(self) -> None:
        """Load the spaCy NER model into memory.

        Raises:
            GraphError: If the model cannot be loaded.
        """
        if self._loaded:
            return

        import spacy as _spacy

        start = time.perf_counter()
        try:
            self._nlp = _spacy.load(
                self._model_name,
                disable=["tagger", "parser", "lemmatizer", "attribute_ruler"],
            )
        except OSError as exc:
            # Try falling back to small model
            log.warning(
                "spacy_model_not_found_falling_back",
                model=self._model_name,
                error=str(exc),
            )
            try:
                self._nlp = _spacy.load(
                    "en_core_web_sm",
                    disable=["tagger", "parser", "lemmatizer", "attribute_ruler"],
                )
                self._model_name = "en_core_web_sm"
            except OSError:
                raise GraphError(
                    "No spaCy model available. Install with: "
                    "uv pip install en_core_web_trf@https://github.com/explosion/"
                    "spacy-models/releases/download/en_core_web_trf-3.8.0/"
                    "en_core_web_trf-3.8.0.tar.gz",
                    operation="load_ner_model",
                    details={"attempted_models": [self._model_name, "en_core_web_sm"]},
                ) from exc

        elapsed_ms = (time.perf_counter() - start) * 1000
        self._loaded = True
        log.info(
            "ner_model_loaded",
            model=self._model_name,
            load_time_ms=round(elapsed_ms, 1),
        )

    def unload(self) -> None:
        """Unload the spaCy model to free memory."""
        self._nlp = None
        self._loaded = False
        log.info("ner_model_unloaded")

    def extract(self, text: str) -> ExtractionResult:
        """Extract named entities from text.

        Args:
            text: The text to process.

        Returns:
            ExtractionResult with deduplicated entities.

        Raises:
            GraphError: If the model is not loaded.
        """
        if not self._loaded or self._nlp is None:
            raise GraphError(
                "NER model not loaded — call load() first",
                operation="extract_entities",
            )

        start = time.perf_counter()

        # Truncate if too long (transformer models have memory limits)
        processed_text = (
            text[: self._max_text_length] if len(text) > self._max_text_length else text
        )

        doc: Doc = self._nlp(processed_text)

        raw_entities: list[ExtractedEntity] = []
        for ent in doc.ents:
            # Skip labels we don't map to graph entities
            if ent.label_ in _SKIP_LABELS:
                continue

            entity_type = _SPACY_TO_ENTITY_TYPE.get(ent.label_)
            if entity_type is None:
                continue

            # Clean the entity name
            cleaned = _clean_entity_name(ent.text)
            if not cleaned:
                continue

            # Filter noise
            if cleaned.lower() in _NOISE_ENTITIES:
                continue

            if len(cleaned) < _MIN_ENTITY_LENGTH:
                continue

            confidence = _estimate_confidence(cleaned, ent.label_, entity_type)
            if confidence < self._confidence_threshold:
                continue

            raw_entities.append(
                ExtractedEntity(
                    name=cleaned,
                    entity_type=entity_type,
                    confidence=confidence,
                    spacy_label=ent.label_,
                    start_char=ent.start_char,
                    end_char=ent.end_char,
                )
            )

        # Deduplicate (same name + same type → keep highest confidence)
        deduped = _deduplicate_entities(raw_entities)

        elapsed_ms = (time.perf_counter() - start) * 1000
        log.debug(
            "entities_extracted",
            raw_count=len(raw_entities),
            deduped_count=len(deduped),
            elapsed_ms=round(elapsed_ms, 1),
            text_length=len(text),
        )

        return ExtractionResult(
            entities=deduped,
            raw_count=len(raw_entities),
            elapsed_ms=elapsed_ms,
            text_length=len(text),
        )

    def extract_batch(self, texts: list[str]) -> list[ExtractionResult]:
        """Extract entities from multiple texts using spaCy's pipe().

        More efficient than calling extract() in a loop because spaCy
        batches the transformer inference.

        Args:
            texts: List of texts to process.

        Returns:
            List of ExtractionResult, one per input text.
        """
        if not self._loaded or self._nlp is None:
            raise GraphError(
                "NER model not loaded — call load() first",
                operation="extract_entities_batch",
            )

        start = time.perf_counter()
        truncated = [
            t[: self._max_text_length] if len(t) > self._max_text_length else t for t in texts
        ]

        results: list[ExtractionResult] = []
        for doc in self._nlp.pipe(truncated, batch_size=8):
            raw_entities: list[ExtractedEntity] = []
            for ent in doc.ents:
                if ent.label_ in _SKIP_LABELS:
                    continue
                entity_type = _SPACY_TO_ENTITY_TYPE.get(ent.label_)
                if entity_type is None:
                    continue
                cleaned = _clean_entity_name(ent.text)
                if not cleaned or cleaned.lower() in _NOISE_ENTITIES:
                    continue
                if len(cleaned) < _MIN_ENTITY_LENGTH:
                    continue
                confidence = _estimate_confidence(cleaned, ent.label_, entity_type)
                if confidence < self._confidence_threshold:
                    continue
                raw_entities.append(
                    ExtractedEntity(
                        name=cleaned,
                        entity_type=entity_type,
                        confidence=confidence,
                        spacy_label=ent.label_,
                        start_char=ent.start_char,
                        end_char=ent.end_char,
                    )
                )

            deduped = _deduplicate_entities(raw_entities)
            results.append(
                ExtractionResult(
                    entities=deduped,
                    raw_count=len(raw_entities),
                    elapsed_ms=0.0,
                    text_length=len(doc.text),
                )
            )

        total_ms = (time.perf_counter() - start) * 1000
        # Distribute time evenly for logging
        per_doc_ms = total_ms / len(results) if results else 0.0
        for r in results:
            r.elapsed_ms = per_doc_ms  # type: ignore[misc]

        log.debug(
            "batch_entities_extracted",
            batch_size=len(texts),
            total_entities=sum(len(r.entities) for r in results),
            elapsed_ms=round(total_ms, 1),
        )

        return results


# ── Helper Functions ─────────────────────────────────────────────────────────


def _clean_entity_name(name: str) -> str:
    """Clean an entity name extracted by spaCy.

    Removes leading/trailing whitespace, newlines, possessives,
    and normalizes internal whitespace.
    """
    # Strip whitespace and newlines
    cleaned = name.strip().replace("\n", " ")

    # Remove possessive suffix
    if cleaned.endswith("'s"):
        cleaned = cleaned[:-2].strip()

    # Normalize internal whitespace
    cleaned = re.sub(r"\s+", " ", cleaned)

    # Remove leading/trailing punctuation (but keep internal hyphens, etc.)
    cleaned = cleaned.strip(".,;:!?()[]{}\"'`~@#$%^&*_+=|\\/<>")

    return cleaned.strip()


def _estimate_confidence(name: str, spacy_label: str, entity_type: EntityType) -> float:
    """Estimate extraction confidence based on heuristics.

    spaCy transformer model doesn't expose per-entity confidence directly,
    so we use heuristics:
    - Capitalized names get higher confidence (proper nouns)
    - Very short names get lower confidence (more likely noise)
    - PERSON entities with single word get slightly lower confidence
    """
    conf = 0.85  # Base confidence for transformer model

    # Capitalization boost
    if name[0].isupper():
        conf += 0.05

    # Length penalty
    if len(name) < 3:
        conf -= 0.15
    elif len(name) < 5:
        conf -= 0.05

    # Single-word PERSON entities are sometimes noise
    if entity_type == EntityType.PERSON and " " not in name:
        conf -= 0.05

    # All-caps might be an acronym — slightly lower
    if name.isupper() and len(name) <= 5:
        conf -= 0.10

    return max(0.0, min(1.0, conf))


def _deduplicate_entities(entities: list[ExtractedEntity]) -> list[ExtractedEntity]:
    """Deduplicate entities by (lowercase name, entity_type).

    Keeps the instance with the highest confidence for each unique pair.
    """
    seen: dict[tuple[str, EntityType], ExtractedEntity] = {}
    for ent in entities:
        key = (ent.name.lower(), ent.entity_type)
        existing = seen.get(key)
        if existing is None or ent.confidence > existing.confidence:
            seen[key] = ent
    return list(seen.values())
