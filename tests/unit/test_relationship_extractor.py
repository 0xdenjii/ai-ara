"""Tests for LLM-based relationship extraction.

Tests focus on JSON parsing logic since the LLM is mocked.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from ara.memory.relationship_extractor import (
    RelationshipExtractor,
    _clamp_float,
    _fuzzy_match_relation_type,
    _parse_llm_response,
)
from ara.utils.types import EntityType, RelationType

# ── Parse Tests ──────────────────────────────────────────────────────────────


class TestParseLLMResponse:
    """Tests for parsing LLM JSON output."""

    def test_parse_valid_json(self) -> None:
        raw = """{
            "entities": [
                {"name": "Mom", "type": "person", "confidence": 0.95},
                {"name": "Springfield", "type": "place", "confidence": 0.9}
            ],
            "relations": [
                {
                    "source": "Mom",
                    "target": "Springfield",
                    "type": "located_at",
                    "confidence": 0.85,
                    "context": "Mom lives in Springfield"
                }
            ]
        }"""
        result = _parse_llm_response(raw)
        assert result.parse_success
        assert len(result.entities) == 2
        assert len(result.relationships) == 1
        assert result.entities[0].name == "Mom"
        assert result.entities[0].entity_type == EntityType.PERSON
        assert result.relationships[0].source_name == "Mom"
        assert result.relationships[0].relation_type == RelationType.LOCATED_AT

    def test_parse_with_markdown_fences(self) -> None:
        raw = """```json
        {
            "entities": [{"name": "Alice", "type": "person", "confidence": 0.9}],
            "relations": []
        }
        ```"""
        result = _parse_llm_response(raw)
        assert result.parse_success
        assert len(result.entities) == 1
        assert result.entities[0].name == "Alice"

    def test_parse_with_trailing_commas(self) -> None:
        raw = """{
            "entities": [
                {"name": "Bob", "type": "person", "confidence": 0.9,},
            ],
            "relations": [],
        }"""
        result = _parse_llm_response(raw)
        assert result.parse_success
        assert len(result.entities) == 1

    def test_parse_empty_entities(self) -> None:
        raw = '{"entities": [], "relations": []}'
        result = _parse_llm_response(raw)
        assert result.parse_success
        assert len(result.entities) == 0
        assert len(result.relationships) == 0

    def test_parse_invalid_json(self) -> None:
        raw = "This is not JSON at all"
        result = _parse_llm_response(raw)
        assert not result.parse_success

    def test_parse_partial_json(self) -> None:
        raw = '{"entities": [{"name": "Alice", "type": "person"'
        result = _parse_llm_response(raw)
        assert not result.parse_success

    def test_parse_skips_invalid_entity_type(self) -> None:
        raw = """{
            "entities": [
                {"name": "X", "type": "invalid_type", "confidence": 0.9},
                {"name": "Y", "type": "person", "confidence": 0.8}
            ],
            "relations": []
        }"""
        result = _parse_llm_response(raw)
        assert result.parse_success
        assert len(result.entities) == 1
        assert result.entities[0].name == "Y"

    def test_parse_skips_empty_names(self) -> None:
        raw = """{
            "entities": [
                {"name": "", "type": "person", "confidence": 0.9},
                {"name": "Valid", "type": "person", "confidence": 0.8}
            ],
            "relations": []
        }"""
        result = _parse_llm_response(raw)
        assert len(result.entities) == 1

    def test_parse_skips_invalid_relation_type(self) -> None:
        raw = """{
            "entities": [
                {"name": "A", "type": "person"},
                {"name": "B", "type": "place"}
            ],
            "relations": [
                {"source": "A", "target": "B", "type": "completely_unknown"}
            ]
        }"""
        result = _parse_llm_response(raw)
        assert result.parse_success
        assert len(result.relationships) == 0

    def test_parse_fuzzy_relation_type(self) -> None:
        raw = """{
            "entities": [
                {"name": "A", "type": "person"},
                {"name": "B", "type": "person"}
            ],
            "relations": [
                {"source": "A", "target": "B", "type": "knows", "confidence": 0.7}
            ]
        }"""
        result = _parse_llm_response(raw)
        assert result.parse_success
        assert len(result.relationships) == 1
        assert result.relationships[0].relation_type == RelationType.INTERACTED_WITH

    def test_parse_all_entity_types(self) -> None:
        entities = [
            {"name": f"test_{t.value}", "type": t.value, "confidence": 0.9} for t in EntityType
        ]
        raw = json.dumps({"entities": entities, "relations": []})
        result = _parse_llm_response(raw)
        assert result.parse_success
        assert len(result.entities) == 16

    def test_parse_all_relation_types(self) -> None:
        relations = [
            {"source": "A", "target": "B", "type": t.value, "confidence": 0.8} for t in RelationType
        ]
        entities = [
            {"name": "A", "type": "person"},
            {"name": "B", "type": "person"},
        ]
        raw = json.dumps({"entities": entities, "relations": relations})
        result = _parse_llm_response(raw)
        assert result.parse_success
        assert len(result.relationships) == 18

    def test_parse_json_embedded_in_text(self) -> None:
        raw = """Here is the extraction result:
        {"entities": [{"name": "Paris", "type": "place"}], "relations": []}
        Hope this helps!"""
        result = _parse_llm_response(raw)
        assert result.parse_success
        assert len(result.entities) == 1

    def test_parse_confidence_clamped(self) -> None:
        raw = """{
            "entities": [
                {"name": "A", "type": "person", "confidence": 1.5},
                {"name": "B", "type": "person", "confidence": -0.5}
            ],
            "relations": []
        }"""
        result = _parse_llm_response(raw)
        assert result.entities[0].confidence == 1.0
        assert result.entities[1].confidence == 0.0

    def test_parse_missing_confidence_defaults(self) -> None:
        raw = '{"entities": [{"name": "A", "type": "person"}], "relations": []}'
        result = _parse_llm_response(raw)
        assert result.entities[0].confidence == 0.8  # Default

    def test_parse_diary_style_output(self) -> None:
        """Test realistic diary extraction output."""
        raw = """{
            "entities": [
                {"name": "Mom", "type": "person", "confidence": 0.95},
                {"name": "Work", "type": "project", "confidence": 0.9},
                {"name": "feeling overwhelmed", "type": "emotion_tag", "confidence": 0.8},
                {"name": "launch MVP by March", "type": "goal", "confidence": 0.85},
                {"name": "daily meditation", "type": "habit", "confidence": 0.7}
            ],
            "relations": [
                {"source": "Mom", "target": "Work", "type": "discussed", "confidence": 0.8, "context": "Mom asked about Work progress"},
                {"source": "feeling overwhelmed", "target": "launch MVP by March", "type": "caused", "confidence": 0.7, "context": "Pressure from deadline causing stress"}
            ]
        }"""
        result = _parse_llm_response(raw)
        assert result.parse_success
        assert len(result.entities) == 5
        assert len(result.relationships) == 2

        types = {e.entity_type for e in result.entities}
        assert EntityType.PERSON in types
        assert EntityType.PROJECT in types
        assert EntityType.EMOTION_TAG in types
        assert EntityType.GOAL in types
        assert EntityType.HABIT in types


# ── Helper Function Tests ────────────────────────────────────────────────────


class TestClampFloat:
    """Tests for float clamping."""

    def test_normal_value(self) -> None:
        assert _clamp_float(0.5) == 0.5

    def test_above_max(self) -> None:
        assert _clamp_float(1.5) == 1.0

    def test_below_min(self) -> None:
        assert _clamp_float(-0.5) == 0.0

    def test_non_numeric(self) -> None:
        assert _clamp_float("invalid") == 0.8

    def test_none(self) -> None:
        assert _clamp_float(None) == 0.8


class TestFuzzyMatchRelationType:
    """Tests for fuzzy relation type matching."""

    def test_knows_maps_to_interacted_with(self) -> None:
        assert _fuzzy_match_relation_type("knows") == RelationType.INTERACTED_WITH

    def test_visited_maps_to_located_at(self) -> None:
        assert _fuzzy_match_relation_type("visited") == RelationType.LOCATED_AT

    def test_built_maps_to_created(self) -> None:
        assert _fuzzy_match_relation_type("built") == RelationType.CREATED

    def test_achieved_maps_to_completed(self) -> None:
        assert _fuzzy_match_relation_type("achieved") == RelationType.COMPLETED

    def test_unknown_returns_none(self) -> None:
        assert _fuzzy_match_relation_type("totally_unknown") is None


# ── Extractor Tests (with mocked LLM) ───────────────────────────────────────


class TestRelationshipExtractor:
    """Tests with mocked LLM client."""

    @pytest.fixture()
    def mock_llm(self) -> MagicMock:
        """Create a mock LLM client."""
        llm = MagicMock()
        llm.generate = AsyncMock()
        return llm

    @pytest.mark.asyncio()
    async def test_extract_success(self, mock_llm: MagicMock) -> None:
        mock_llm.generate.return_value = """{
            "entities": [{"name": "Alice", "type": "person", "confidence": 0.9}],
            "relations": []
        }"""
        extractor = RelationshipExtractor(mock_llm)
        result = await extractor.extract("Alice went home.")
        assert result.parse_success
        assert len(result.entities) == 1
        assert result.elapsed_ms > 0

    @pytest.mark.asyncio()
    async def test_extract_with_existing_entities(self, mock_llm: MagicMock) -> None:
        from ara.memory.entity_extractor import ExtractedEntity

        mock_llm.generate.return_value = """{
            "entities": [{"name": "feeling happy", "type": "emotion_tag"}],
            "relations": [
                {"source": "Alice", "target": "feeling happy", "type": "felt", "confidence": 0.8}
            ]
        }"""

        existing = [
            ExtractedEntity(name="Alice", entity_type=EntityType.PERSON),
        ]

        extractor = RelationshipExtractor(mock_llm)
        result = await extractor.extract("Alice was very happy.", existing_entities=existing)
        assert result.parse_success
        assert len(result.relationships) == 1

    @pytest.mark.asyncio()
    async def test_extract_llm_failure(self, mock_llm: MagicMock) -> None:
        mock_llm.generate.side_effect = RuntimeError("Ollama not running")
        extractor = RelationshipExtractor(mock_llm)
        result = await extractor.extract("Some text")
        assert not result.parse_success
        assert result.elapsed_ms > 0

    @pytest.mark.asyncio()
    async def test_extract_batch(self, mock_llm: MagicMock) -> None:
        mock_llm.generate.return_value = '{"entities": [], "relations": []}'
        extractor = RelationshipExtractor(mock_llm)
        results = await extractor.extract_batch(["Text 1", "Text 2"])
        assert len(results) == 2
        assert mock_llm.generate.call_count == 2

    @pytest.mark.asyncio()
    async def test_extract_with_source_hint(self, mock_llm: MagicMock) -> None:
        mock_llm.generate.return_value = '{"entities": [], "relations": []}'
        extractor = RelationshipExtractor(mock_llm)
        result = await extractor.extract(
            "Some diary text",
            source_hint="diary entry from 2024-01-15",
        )
        assert result.parse_success
        # Verify source hint was included in the prompt
        call_args = mock_llm.generate.call_args
        messages = call_args[0][0]
        user_msg = messages[-1].content
        assert "2024-01-15" in user_msg


# Need json import for test_parse_all_entity_types
import json  # noqa: E402
