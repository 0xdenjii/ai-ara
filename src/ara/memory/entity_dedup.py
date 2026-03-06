"""Entity deduplication for the knowledge graph.

Detects duplicate entities that refer to the same real-world entity
(e.g., "Mom" = "my mother" = "mama") using embedding similarity.

Strategy:
    1. Group entities by EntityType (only compare within same type)
    2. Compute embedding similarity between entity names + aliases
    3. If similarity > threshold, flag as potential duplicate
    4. Merge confirmed duplicates via KnowledgeGraph.merge_entities()

The deduplication is conservative — it suggests merges but doesn't
auto-merge unless the similarity is very high (>0.92). Medium-confidence
candidates (0.75-0.92) can be confirmed by the LLM or user.

Usage:
    from ara.memory.entity_dedup import EntityDeduplicator

    dedup = EntityDeduplicator(embedding_model)
    candidates = dedup.find_duplicates(graph.get_entities_by_type(EntityType.PERSON))

    for group in candidates:
        # Each group is a list of entities that likely refer to the same thing
        keep = group[0]  # Highest mention count
        for entity in group[1:]:
            graph.merge_entities(keep.entity_id, entity.entity_id)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from ara.memory.embeddings import EmbeddingModel
    from ara.utils.types import EntityType, GraphEntity

log = structlog.get_logger(component="entity_dedup")


# ── Data Types ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class DuplicateCandidate:
    """A pair of entities that may be duplicates.

    Attributes:
        entity_a: First entity.
        entity_b: Second entity.
        similarity: Cosine similarity between their embeddings.
        matched_on: What text was compared (name, alias, or combination).
    """

    entity_a: GraphEntity
    entity_b: GraphEntity
    similarity: float
    matched_on: str = ""


@dataclass
class DeduplicationResult:
    """Result of deduplication analysis.

    Attributes:
        candidates: List of duplicate candidate pairs, sorted by similarity.
        auto_merge: Pairs with similarity above auto-merge threshold.
        review_needed: Pairs that need human/LLM review.
        elapsed_ms: Processing time.
    """

    candidates: list[DuplicateCandidate] = field(default_factory=list)
    auto_merge: list[DuplicateCandidate] = field(default_factory=list)
    review_needed: list[DuplicateCandidate] = field(default_factory=list)
    elapsed_ms: float = 0.0


# ── Entity Deduplicator ─────────────────────────────────────────────────────


class EntityDeduplicator:
    """Detect duplicate entities using embedding similarity.

    Compares entity names and aliases within the same EntityType group
    to find potential duplicates.

    Args:
        embedding_model: The embedding model to compute similarity.
        auto_merge_threshold: Similarity above this triggers automatic merge.
        review_threshold: Similarity above this flags for review.
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        *,
        auto_merge_threshold: float = 0.92,
        review_threshold: float = 0.75,
    ) -> None:
        self._embedder = embedding_model
        self._auto_threshold = auto_merge_threshold
        self._review_threshold = review_threshold

    def find_duplicates(
        self,
        entities: list[GraphEntity],
        *,
        entity_type: EntityType | None = None,
    ) -> DeduplicationResult:
        """Find potential duplicate entities.

        Args:
            entities: List of entities to check for duplicates.
            entity_type: If provided, only check entities of this type.
                If None, groups entities by type and checks within each group.

        Returns:
            DeduplicationResult with candidates sorted by similarity.
        """
        start = time.perf_counter()

        if entity_type:
            filtered = [e for e in entities if e.entity_type == entity_type]
            candidates = self._find_duplicates_in_group(filtered)
        else:
            # Group by type and find duplicates within each group
            type_groups: dict[EntityType, list[GraphEntity]] = {}
            for e in entities:
                type_groups.setdefault(e.entity_type, []).append(e)

            candidates: list[DuplicateCandidate] = []
            for group in type_groups.values():
                if len(group) < 2:
                    continue
                candidates.extend(self._find_duplicates_in_group(group))

        # Sort by similarity (highest first)
        candidates.sort(key=lambda c: c.similarity, reverse=True)

        # Split into auto-merge and review-needed
        auto_merge = [c for c in candidates if c.similarity >= self._auto_threshold]
        review_needed = [
            c for c in candidates if self._review_threshold <= c.similarity < self._auto_threshold
        ]

        elapsed_ms = (time.perf_counter() - start) * 1000

        log.info(
            "dedup_complete",
            total_entities=len(entities),
            candidates=len(candidates),
            auto_merge=len(auto_merge),
            review_needed=len(review_needed),
            elapsed_ms=round(elapsed_ms, 1),
        )

        return DeduplicationResult(
            candidates=candidates,
            auto_merge=auto_merge,
            review_needed=review_needed,
            elapsed_ms=elapsed_ms,
        )

    def _find_duplicates_in_group(
        self,
        entities: list[GraphEntity],
    ) -> list[DuplicateCandidate]:
        """Find duplicates within a single type group.

        For each entity, builds a representative text from its name and
        aliases, embeds them all, then computes pairwise similarities.
        """
        if len(entities) < 2:
            return []

        # Build representative texts for each entity
        texts: list[str] = []
        for entity in entities:
            # Combine name + aliases for a richer embedding
            parts = [entity.name, *entity.aliases]
            texts.append(" | ".join(parts))

        # Compute embeddings for all texts
        embeddings = self._embedder.embed_batch(texts)

        # Pairwise similarity check
        candidates: list[DuplicateCandidate] = []
        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                sim = _cosine_similarity(embeddings[i], embeddings[j])
                if sim >= self._review_threshold:
                    # Determine what matched
                    matched_on = _describe_match(entities[i], entities[j])
                    candidates.append(
                        DuplicateCandidate(
                            entity_a=entities[i],
                            entity_b=entities[j],
                            similarity=sim,
                            matched_on=matched_on,
                        )
                    )

        return candidates

    def compute_similarity(
        self,
        entity_a: GraphEntity,
        entity_b: GraphEntity,
    ) -> float:
        """Compute embedding similarity between two entities.

        Useful for ad-hoc dedup checks.
        """
        text_a = " | ".join([entity_a.name, *entity_a.aliases])
        text_b = " | ".join([entity_b.name, *entity_b.aliases])
        embeddings = self._embedder.embed_batch([text_a, text_b])
        return _cosine_similarity(embeddings[0], embeddings[1])


# ── Helper Functions ─────────────────────────────────────────────────────────


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    Returns a value between -1.0 and 1.0.
    """
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0

    dot = sum(a * b for a, b in zip(vec_a, vec_b, strict=True))
    norm_a = sum(a * a for a in vec_a) ** 0.5
    norm_b = sum(b * b for b in vec_b) ** 0.5

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot / (norm_a * norm_b)


def _describe_match(entity_a: GraphEntity, entity_b: GraphEntity) -> str:
    """Describe what caused a match between two entities."""
    name_a_lower = entity_a.name.lower()
    name_b_lower = entity_b.name.lower()

    # Exact name match (case-insensitive)
    if name_a_lower == name_b_lower:
        return f"exact name match: '{entity_a.name}'"

    # Name is alias of other
    aliases_a = {a.lower() for a in entity_a.aliases}
    aliases_b = {a.lower() for a in entity_b.aliases}

    if name_b_lower in aliases_a:
        return f"'{entity_b.name}' is alias of '{entity_a.name}'"
    if name_a_lower in aliases_b:
        return f"'{entity_a.name}' is alias of '{entity_b.name}'"

    # Shared alias
    shared = aliases_a & aliases_b
    if shared:
        return f"shared alias: '{next(iter(shared))}'"

    # Substring match
    if name_a_lower in name_b_lower or name_b_lower in name_a_lower:
        return f"substring: '{entity_a.name}' / '{entity_b.name}'"

    return f"embedding similarity: '{entity_a.name}' / '{entity_b.name}'"
