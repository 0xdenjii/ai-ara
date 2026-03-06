"""Tests for entity extraction via spaCy NER.

These tests require the spaCy model to be installed. They load the model
once per session for efficiency (transformer model load takes ~5-10s).
"""

from __future__ import annotations

import pytest

from ara.memory.entity_extractor import (
    EntityExtractor,
    ExtractedEntity,
    ExtractionResult,
    _clean_entity_name,
    _deduplicate_entities,
    _estimate_confidence,
)
from ara.utils.errors import GraphError
from ara.utils.types import EntityType

# ── Fixtures ─────────────────────────────────────────────────────────────────

# Use en_core_web_sm for tests (fast, no transformer overhead)
# Real production uses en_core_web_trf
_TEST_MODEL = "en_core_web_sm"


@pytest.fixture(scope="module")
def extractor() -> EntityExtractor:
    """Module-scoped extractor to avoid reloading model per test."""
    ext = EntityExtractor(model_name=_TEST_MODEL, confidence_threshold=0.3)
    try:
        ext.load()
    except GraphError:
        pytest.skip(f"spaCy model '{_TEST_MODEL}' not installed")
    yield ext
    ext.unload()


# ── Lifecycle Tests ──────────────────────────────────────────────────────────


class TestLifecycle:
    """Tests for extractor load/unload lifecycle."""

    def test_load_and_unload(self) -> None:
        ext = EntityExtractor(model_name=_TEST_MODEL)
        try:
            ext.load()
        except GraphError:
            pytest.skip(f"spaCy model '{_TEST_MODEL}' not installed")
        assert ext.is_loaded
        ext.unload()
        assert not ext.is_loaded

    def test_double_load_is_noop(self) -> None:
        ext = EntityExtractor(model_name=_TEST_MODEL)
        try:
            ext.load()
        except GraphError:
            pytest.skip(f"spaCy model '{_TEST_MODEL}' not installed")
        ext.load()
        assert ext.is_loaded
        ext.unload()

    def test_extract_before_load_raises(self) -> None:
        ext = EntityExtractor()
        with pytest.raises(GraphError, match="not loaded"):
            ext.extract("Hello world")

    def test_batch_before_load_raises(self) -> None:
        ext = EntityExtractor()
        with pytest.raises(GraphError, match="not loaded"):
            ext.extract_batch(["Hello"])


# ── Extraction Tests ─────────────────────────────────────────────────────────


class TestExtraction:
    """Tests for entity extraction from text."""

    def test_extract_person(self, extractor: EntityExtractor) -> None:
        result = extractor.extract("John Smith went to the store.")
        names = {e.name for e in result.entities}
        types = {e.entity_type for e in result.entities}
        assert "John Smith" in names
        assert EntityType.PERSON in types

    def test_extract_place(self, extractor: EntityExtractor) -> None:
        result = extractor.extract("I traveled to New York City last week.")
        place_entities = [e for e in result.entities if e.entity_type == EntityType.PLACE]
        place_names = {e.name for e in place_entities}
        assert any("New York" in n for n in place_names)

    def test_extract_organization(self, extractor: EntityExtractor) -> None:
        result = extractor.extract("She works at Google as an engineer.")
        org_entities = [e for e in result.entities if e.entity_type == EntityType.ORGANIZATION]
        org_names = {e.name for e in org_entities}
        assert "Google" in org_names

    def test_extract_multiple_entities(self, extractor: EntityExtractor) -> None:
        result = extractor.extract("Alice and Bob met in London at Microsoft headquarters.")
        assert len(result.entities) >= 2

    def test_extract_returns_result_metadata(self, extractor: EntityExtractor) -> None:
        text = "Barack Obama visited Tokyo in 2015."
        result = extractor.extract(text)
        assert result.text_length == len(text)
        assert result.elapsed_ms > 0
        assert result.raw_count >= 0

    def test_extract_empty_text(self, extractor: EntityExtractor) -> None:
        result = extractor.extract("")
        assert len(result.entities) == 0
        assert result.raw_count == 0

    def test_extract_no_entities(self, extractor: EntityExtractor) -> None:
        result = extractor.extract("the cat sat on the mat")
        # May or may not find entities — just shouldn't crash
        assert isinstance(result, ExtractionResult)

    def test_extract_deduplicates(self, extractor: EntityExtractor) -> None:
        result = extractor.extract("Paris is beautiful. I love Paris. Paris in spring.")
        paris_count = sum(1 for e in result.entities if e.name.lower() == "paris")
        assert paris_count <= 1  # Should be deduplicated

    def test_extract_filters_noise(self, extractor: EntityExtractor) -> None:
        result = extractor.extract("Today I went to the store yesterday.")
        names_lower = {e.name.lower() for e in result.entities}
        assert "today" not in names_lower
        assert "yesterday" not in names_lower

    def test_extract_long_text_truncated(self) -> None:
        ext = EntityExtractor(model_name=_TEST_MODEL, max_text_length=100)
        try:
            ext.load()
        except GraphError:
            pytest.skip(f"spaCy model '{_TEST_MODEL}' not installed")
        long_text = "John Smith is great. " * 100
        result = ext.extract(long_text)
        # Should not crash, just truncate
        assert isinstance(result, ExtractionResult)
        ext.unload()

    def test_extract_diary_style_text(self, extractor: EntityExtractor) -> None:
        """Test with diary-style informal text."""
        text = (
            "Talked to Mom today about Work. She thinks I should "
            "focus more on health. Dad called from Springfield. "
            "Feeling stressed about the pitch to Pantera Capital."
        )
        result = extractor.extract(text)
        # We expect at least some entities to be found
        assert len(result.entities) >= 1


# ── Batch Extraction Tests ───────────────────────────────────────────────────


class TestBatchExtraction:
    """Tests for batch entity extraction."""

    def test_batch_extract(self, extractor: EntityExtractor) -> None:
        texts = [
            "Alice went to London.",
            "Bob works at Microsoft.",
            "Charlie visited Tokyo.",
        ]
        results = extractor.extract_batch(texts)
        assert len(results) == 3
        for result in results:
            assert isinstance(result, ExtractionResult)

    def test_batch_empty_list(self, extractor: EntityExtractor) -> None:
        results = extractor.extract_batch([])
        assert results == []


# ── Helper Function Tests ────────────────────────────────────────────────────


class TestCleanEntityName:
    """Tests for entity name cleaning."""

    def test_strip_whitespace(self) -> None:
        assert _clean_entity_name("  John  ") == "John"

    def test_remove_possessive(self) -> None:
        assert _clean_entity_name("John's") == "John"

    def test_normalize_whitespace(self) -> None:
        assert _clean_entity_name("New   York   City") == "New York City"

    def test_remove_newlines(self) -> None:
        assert _clean_entity_name("New\nYork") == "New York"

    def test_strip_punctuation(self) -> None:
        assert _clean_entity_name("(Google)") == "Google"
        assert _clean_entity_name('"Microsoft"') == "Microsoft"

    def test_empty_string(self) -> None:
        assert _clean_entity_name("") == ""

    def test_preserves_hyphens(self) -> None:
        assert _clean_entity_name("Jean-Pierre") == "Jean-Pierre"


class TestEstimateConfidence:
    """Tests for confidence estimation heuristics."""

    def test_capitalized_gets_boost(self) -> None:
        conf_cap = _estimate_confidence("John", "PERSON", EntityType.PERSON)
        conf_low = _estimate_confidence("john", "PERSON", EntityType.PERSON)
        assert conf_cap > conf_low

    def test_short_name_penalty(self) -> None:
        conf_long = _estimate_confidence("Alexander", "PERSON", EntityType.PERSON)
        conf_short = _estimate_confidence("Al", "PERSON", EntityType.PERSON)
        assert conf_long > conf_short

    def test_single_word_person_penalty(self) -> None:
        conf_full = _estimate_confidence("John Smith", "PERSON", EntityType.PERSON)
        conf_single = _estimate_confidence("John", "PERSON", EntityType.PERSON)
        assert conf_full > conf_single

    def test_all_caps_acronym_penalty(self) -> None:
        conf_normal = _estimate_confidence("Google", "ORG", EntityType.ORGANIZATION)
        conf_caps = _estimate_confidence("IBM", "ORG", EntityType.ORGANIZATION)
        assert conf_normal > conf_caps

    def test_confidence_bounded(self) -> None:
        conf = _estimate_confidence("X", "PERSON", EntityType.PERSON)
        assert 0.0 <= conf <= 1.0


class TestDeduplicateEntities:
    """Tests for entity deduplication."""

    def test_dedup_same_name_same_type(self) -> None:
        entities = [
            ExtractedEntity(name="Paris", entity_type=EntityType.PLACE, confidence=0.8),
            ExtractedEntity(name="Paris", entity_type=EntityType.PLACE, confidence=0.9),
        ]
        deduped = _deduplicate_entities(entities)
        assert len(deduped) == 1
        assert deduped[0].confidence == 0.9  # Keeps highest

    def test_dedup_case_insensitive(self) -> None:
        entities = [
            ExtractedEntity(name="paris", entity_type=EntityType.PLACE, confidence=0.7),
            ExtractedEntity(name="Paris", entity_type=EntityType.PLACE, confidence=0.9),
        ]
        deduped = _deduplicate_entities(entities)
        assert len(deduped) == 1

    def test_dedup_different_types_kept(self) -> None:
        entities = [
            ExtractedEntity(name="Phoenix", entity_type=EntityType.PERSON, confidence=0.8),
            ExtractedEntity(name="Phoenix", entity_type=EntityType.PLACE, confidence=0.8),
        ]
        deduped = _deduplicate_entities(entities)
        assert len(deduped) == 2

    def test_dedup_empty(self) -> None:
        assert _deduplicate_entities([]) == []
