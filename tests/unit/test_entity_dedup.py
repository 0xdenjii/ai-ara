"""Tests for entity deduplication via embedding similarity.

Uses a mock EmbeddingModel to avoid loading real models in unit tests.
Tests cover cosine similarity, match description, deduplicator lifecycle,
filtering, thresholds, and edge cases.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from ara.memory.entity_dedup import (
    DeduplicationResult,
    DuplicateCandidate,
    EntityDeduplicator,
    _cosine_similarity,
    _describe_match,
)
from ara.utils.types import EntityType, GraphEntity

# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_entity(
    name: str,
    entity_type: EntityType = EntityType.PERSON,
    aliases: list[str] | None = None,
    entity_id: str = "",
) -> GraphEntity:
    """Create a GraphEntity for testing."""
    return GraphEntity(
        entity_id=entity_id or name.lower().replace(" ", "_"),
        name=name,
        entity_type=entity_type,
        aliases=aliases or [],
    )


def _mock_embedder(embeddings_map: dict[str, list[float]] | None = None) -> MagicMock:
    """Create a mock EmbeddingModel that returns deterministic embeddings.

    Args:
        embeddings_map: Maps text to embedding vector. If a text isn't in the
            map, returns a zero vector.
    """
    mock = MagicMock()
    lookup = embeddings_map or {}

    def _embed_batch(texts: list[str], **_kwargs: object) -> list[list[float]]:
        result: list[list[float]] = []
        for text in texts:
            if text in lookup:
                result.append(lookup[text])
            else:
                result.append([0.0] * 4)
        return result

    mock.embed_batch = MagicMock(side_effect=_embed_batch)
    return mock


# ── Cosine Similarity Tests ──────────────────────────────────────────────────


class TestCosineSimilarity:
    """Tests for the _cosine_similarity helper function."""

    def test_identical_vectors(self) -> None:
        vec = [1.0, 2.0, 3.0]
        assert _cosine_similarity(vec, vec) == pytest.approx(1.0)

    def test_orthogonal_vectors(self) -> None:
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert _cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self) -> None:
        a = [1.0, 2.0, 3.0]
        b = [-1.0, -2.0, -3.0]
        assert _cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_empty_vectors(self) -> None:
        assert _cosine_similarity([], []) == 0.0

    def test_mismatched_lengths(self) -> None:
        assert _cosine_similarity([1.0, 2.0], [1.0]) == 0.0

    def test_zero_vector(self) -> None:
        assert _cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0

    def test_both_zero_vectors(self) -> None:
        assert _cosine_similarity([0.0, 0.0], [0.0, 0.0]) == 0.0

    def test_one_dimensional(self) -> None:
        assert _cosine_similarity([3.0], [5.0]) == pytest.approx(1.0)

    def test_negative_similarity_possible(self) -> None:
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert _cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_known_similarity(self) -> None:
        # cos([1,1], [1,0]) = 1/sqrt(2) ≈ 0.7071
        a = [1.0, 1.0]
        b = [1.0, 0.0]
        assert _cosine_similarity(a, b) == pytest.approx(1.0 / (2**0.5), abs=1e-6)


# ── Describe Match Tests ─────────────────────────────────────────────────────


class TestDescribeMatch:
    """Tests for the _describe_match helper function."""

    def test_exact_name_match(self) -> None:
        a = _make_entity("John")
        b = _make_entity("john")
        result = _describe_match(a, b)
        assert "exact name match" in result

    def test_name_is_alias_of_other(self) -> None:
        a = _make_entity("Mom", aliases=["mother", "mama"])
        b = _make_entity("mother")
        result = _describe_match(a, b)
        assert "alias" in result.lower()

    def test_reverse_alias_match(self) -> None:
        a = _make_entity("John")
        b = _make_entity("Jack", aliases=["john"])
        result = _describe_match(a, b)
        assert "alias" in result.lower()

    def test_shared_alias(self) -> None:
        a = _make_entity("Mom", aliases=["mother"])
        b = _make_entity("Mama", aliases=["mother"])
        result = _describe_match(a, b)
        assert "shared alias" in result

    def test_substring_match(self) -> None:
        a = _make_entity("React")
        b = _make_entity("React Native")
        result = _describe_match(a, b)
        assert "substring" in result

    def test_embedding_similarity_fallback(self) -> None:
        a = _make_entity("Alice")
        b = _make_entity("Bob")
        result = _describe_match(a, b)
        assert "embedding similarity" in result

    def test_exact_match_takes_priority(self) -> None:
        """Exact name match should take priority over alias or substring."""
        a = _make_entity("test", aliases=["test"])
        b = _make_entity("Test")
        result = _describe_match(a, b)
        assert "exact name match" in result


# ── Data Types Tests ─────────────────────────────────────────────────────────


class TestDataTypes:
    """Tests for DuplicateCandidate and DeduplicationResult."""

    def test_duplicate_candidate_frozen(self) -> None:
        a = _make_entity("Alice")
        b = _make_entity("Bob")
        c = DuplicateCandidate(entity_a=a, entity_b=b, similarity=0.95)
        assert c.entity_a is a
        assert c.entity_b is b
        assert c.similarity == 0.95
        assert c.matched_on == ""
        with pytest.raises(AttributeError):
            c.similarity = 0.5  # type: ignore[misc]

    def test_duplicate_candidate_with_matched_on(self) -> None:
        a = _make_entity("Alice")
        b = _make_entity("Bob")
        c = DuplicateCandidate(
            entity_a=a, entity_b=b, similarity=0.9, matched_on="shared alias: 'al'"
        )
        assert c.matched_on == "shared alias: 'al'"

    def test_dedup_result_defaults(self) -> None:
        r = DeduplicationResult()
        assert r.candidates == []
        assert r.auto_merge == []
        assert r.review_needed == []
        assert r.elapsed_ms == 0.0

    def test_dedup_result_mutable(self) -> None:
        r = DeduplicationResult()
        a = _make_entity("X")
        b = _make_entity("Y")
        c = DuplicateCandidate(entity_a=a, entity_b=b, similarity=0.95)
        r.candidates.append(c)
        assert len(r.candidates) == 1


# ── EntityDeduplicator Tests ─────────────────────────────────────────────────


class TestEntityDeduplicator:
    """Tests for the EntityDeduplicator class."""

    def test_init_defaults(self) -> None:
        mock = _mock_embedder()
        dedup = EntityDeduplicator(mock)
        assert dedup._auto_threshold == 0.92
        assert dedup._review_threshold == 0.75

    def test_init_custom_thresholds(self) -> None:
        mock = _mock_embedder()
        dedup = EntityDeduplicator(mock, auto_merge_threshold=0.85, review_threshold=0.60)
        assert dedup._auto_threshold == 0.85
        assert dedup._review_threshold == 0.60

    def test_empty_entity_list(self) -> None:
        mock = _mock_embedder()
        dedup = EntityDeduplicator(mock)
        result = dedup.find_duplicates([])
        assert result.candidates == []
        assert result.auto_merge == []
        assert result.review_needed == []

    def test_single_entity(self) -> None:
        mock = _mock_embedder()
        dedup = EntityDeduplicator(mock)
        result = dedup.find_duplicates([_make_entity("Alice")])
        assert result.candidates == []

    def test_identical_embeddings_auto_merge(self) -> None:
        """Two entities with identical embeddings should auto-merge."""
        vec = [1.0, 0.0, 0.0, 0.0]
        embeddings = {
            "Alice": vec,
            "alice": vec,
        }
        mock = _mock_embedder(embeddings)
        dedup = EntityDeduplicator(mock, auto_merge_threshold=0.92, review_threshold=0.75)
        entities = [
            _make_entity("Alice", entity_id="a1"),
            _make_entity("alice", entity_id="a2"),
        ]
        result = dedup.find_duplicates(entities)
        assert len(result.candidates) == 1
        assert result.candidates[0].similarity == pytest.approx(1.0)
        assert len(result.auto_merge) == 1
        assert len(result.review_needed) == 0

    def test_medium_similarity_review_needed(self) -> None:
        """Similarity between review and auto-merge thresholds goes to review."""
        # cos([1,0,0,0], [0.85,0.53,0,0]) ≈ 0.85
        embeddings = {
            "Alice": [1.0, 0.0, 0.0, 0.0],
            "Alicia": [0.85, 0.53, 0.0, 0.0],
        }
        mock = _mock_embedder(embeddings)
        dedup = EntityDeduplicator(mock, auto_merge_threshold=0.92, review_threshold=0.75)
        entities = [_make_entity("Alice"), _make_entity("Alicia")]
        result = dedup.find_duplicates(entities)
        assert len(result.candidates) == 1
        assert len(result.auto_merge) == 0
        assert len(result.review_needed) == 1

    def test_low_similarity_no_candidates(self) -> None:
        """Orthogonal embeddings should produce no candidates."""
        embeddings = {
            "Alice": [1.0, 0.0, 0.0, 0.0],
            "Blockchain": [0.0, 1.0, 0.0, 0.0],
        }
        mock = _mock_embedder(embeddings)
        dedup = EntityDeduplicator(mock, review_threshold=0.75)
        entities = [_make_entity("Alice"), _make_entity("Blockchain", entity_type=EntityType.TOPIC)]
        result = dedup.find_duplicates(entities)
        # Different types → different groups → no comparison
        assert len(result.candidates) == 0

    def test_filter_by_entity_type(self) -> None:
        """Filtering by type should only compare entities of that type."""
        vec = [1.0, 0.0, 0.0, 0.0]
        embeddings = {
            "Alice": vec,
            "Bob": vec,
            "New York": vec,
        }
        mock = _mock_embedder(embeddings)
        dedup = EntityDeduplicator(mock)
        entities = [
            _make_entity("Alice", entity_type=EntityType.PERSON),
            _make_entity("Bob", entity_type=EntityType.PERSON),
            _make_entity("New York", entity_type=EntityType.PLACE),
        ]
        result = dedup.find_duplicates(entities, entity_type=EntityType.PERSON)
        # Only Person-type entities compared
        assert len(result.candidates) == 1
        names = {result.candidates[0].entity_a.name, result.candidates[0].entity_b.name}
        assert names == {"Alice", "Bob"}

    def test_groups_by_type_when_no_filter(self) -> None:
        """Without type filter, entities are grouped by type — no cross-type comparison."""
        vec = [1.0, 0.0, 0.0, 0.0]
        embeddings = {
            "Alice": vec,
            "New York": vec,
        }
        mock = _mock_embedder(embeddings)
        dedup = EntityDeduplicator(mock)
        entities = [
            _make_entity("Alice", entity_type=EntityType.PERSON),
            _make_entity("New York", entity_type=EntityType.PLACE),
        ]
        result = dedup.find_duplicates(entities)
        # One entity per type → no within-group pairs
        assert len(result.candidates) == 0

    def test_multiple_duplicates_sorted_by_similarity(self) -> None:
        """Results should be sorted by similarity (highest first)."""
        embeddings = {
            "Mom": [1.0, 0.0, 0.0, 0.0],
            "Mother": [0.99, 0.14, 0.0, 0.0],
            "Mama": [0.90, 0.44, 0.0, 0.0],
        }
        mock = _mock_embedder(embeddings)
        dedup = EntityDeduplicator(mock, review_threshold=0.70)
        entities = [
            _make_entity("Mom"),
            _make_entity("Mother"),
            _make_entity("Mama"),
        ]
        result = dedup.find_duplicates(entities)
        assert len(result.candidates) >= 2
        # Check descending order
        for i in range(len(result.candidates) - 1):
            assert result.candidates[i].similarity >= result.candidates[i + 1].similarity

    def test_elapsed_ms_tracked(self) -> None:
        mock = _mock_embedder()
        dedup = EntityDeduplicator(mock)
        result = dedup.find_duplicates([])
        assert result.elapsed_ms >= 0.0

    def test_embed_batch_called_with_correct_texts(self) -> None:
        """Verify embed_batch receives name + aliases joined."""
        mock = _mock_embedder()
        dedup = EntityDeduplicator(mock)
        entities = [
            _make_entity("Mom", aliases=["mother", "mama"]),
            _make_entity("Dad", aliases=["father"]),
        ]
        dedup.find_duplicates(entities)
        mock.embed_batch.assert_called_once()
        call_args = mock.embed_batch.call_args[0][0]
        assert call_args[0] == "Mom | mother | mama"
        assert call_args[1] == "Dad | father"

    def test_compute_similarity_two_entities(self) -> None:
        """compute_similarity should return cosine sim for a pair."""
        vec = [1.0, 0.0, 0.0, 0.0]
        embeddings = {
            "Alice": vec,
            "Bob": [0.0, 1.0, 0.0, 0.0],
        }
        mock = _mock_embedder(embeddings)
        dedup = EntityDeduplicator(mock)
        sim = dedup.compute_similarity(_make_entity("Alice"), _make_entity("Bob"))
        assert sim == pytest.approx(0.0)

    def test_compute_similarity_identical(self) -> None:
        vec = [1.0, 0.5, 0.0, 0.0]
        embeddings = {
            "Alice": vec,
            "alice": vec,
        }
        mock = _mock_embedder(embeddings)
        dedup = EntityDeduplicator(mock)
        sim = dedup.compute_similarity(_make_entity("Alice"), _make_entity("alice"))
        assert sim == pytest.approx(1.0)


# ── Edge Cases ───────────────────────────────────────────────────────────────


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_all_entities_same_name_different_ids(self) -> None:
        """Multiple entities with identical names should all be flagged."""
        vec = [1.0, 0.0, 0.0, 0.0]
        embeddings = {"Alice": vec}
        mock = _mock_embedder(embeddings)
        dedup = EntityDeduplicator(mock)
        entities = [
            _make_entity("Alice", entity_id="a1"),
            _make_entity("Alice", entity_id="a2"),
            _make_entity("Alice", entity_id="a3"),
        ]
        result = dedup.find_duplicates(entities)
        # 3 entities → 3 pairs: (a1,a2), (a1,a3), (a2,a3)
        assert len(result.candidates) == 3

    def test_entities_with_only_aliases(self) -> None:
        """Entities with aliases but no name overlap should use embedding sim."""
        embeddings = {
            "X | foo": [1.0, 0.0, 0.0, 0.0],
            "Y | bar": [0.0, 1.0, 0.0, 0.0],
        }
        mock = _mock_embedder(embeddings)
        dedup = EntityDeduplicator(mock)
        entities = [
            _make_entity("X", aliases=["foo"]),
            _make_entity("Y", aliases=["bar"]),
        ]
        result = dedup.find_duplicates(entities)
        assert len(result.candidates) == 0

    def test_single_entity_per_type_no_comparison(self) -> None:
        """Each type group with only one entity should produce no candidates."""
        mock = _mock_embedder()
        dedup = EntityDeduplicator(mock)
        entities = [
            _make_entity("Alice", entity_type=EntityType.PERSON),
            _make_entity("NYC", entity_type=EntityType.PLACE),
            _make_entity("ai-ara", entity_type=EntityType.PROJECT),
            _make_entity("workout", entity_type=EntityType.HABIT),
        ]
        result = dedup.find_duplicates(entities)
        assert len(result.candidates) == 0
        # embed_batch should not be called (no groups with 2+ entities)
        mock.embed_batch.assert_not_called()

    def test_threshold_boundary_exact(self) -> None:
        """Similarity exactly at threshold boundaries."""
        # Construct vectors such that cos(a,b) = exactly 0.92
        # cos([1,0], [0.92, sqrt(1-0.92^2)]) = 0.92
        import math

        embeddings = {
            "A": [1.0, 0.0, 0.0, 0.0],
            "B": [0.92, math.sqrt(1 - 0.92**2), 0.0, 0.0],
        }
        mock = _mock_embedder(embeddings)
        dedup = EntityDeduplicator(mock, auto_merge_threshold=0.92, review_threshold=0.75)
        entities = [_make_entity("A"), _make_entity("B")]
        result = dedup.find_duplicates(entities)
        assert len(result.candidates) == 1
        # Exactly 0.92 should be auto_merge (>= threshold)
        assert len(result.auto_merge) == 1
        assert len(result.review_needed) == 0

    def test_matched_on_populated(self) -> None:
        """Candidates should have meaningful matched_on descriptions."""
        vec = [1.0, 0.0, 0.0, 0.0]
        embeddings = {
            "Mom | mother": vec,
            "Mother": vec,
        }
        mock = _mock_embedder(embeddings)
        dedup = EntityDeduplicator(mock)
        entities = [
            _make_entity("Mom", aliases=["mother"]),
            _make_entity("Mother"),
        ]
        result = dedup.find_duplicates(entities)
        assert len(result.candidates) == 1
        assert result.candidates[0].matched_on != ""
        assert "alias" in result.candidates[0].matched_on.lower()

    def test_many_entities_pairwise_count(self) -> None:
        """N entities should produce at most N*(N-1)/2 comparisons."""
        n = 5
        vec = [1.0, 0.0, 0.0, 0.0]
        embeddings = {f"E{i}": vec for i in range(n)}
        mock = _mock_embedder(embeddings)
        dedup = EntityDeduplicator(mock)
        entities = [_make_entity(f"E{i}", entity_id=f"e{i}") for i in range(n)]
        result = dedup.find_duplicates(entities)
        max_pairs = n * (n - 1) // 2
        assert len(result.candidates) == max_pairs
