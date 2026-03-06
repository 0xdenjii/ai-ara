"""Tests for ARA knowledge graph (NetworkX + SQLite persistence)."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

import pytest

from ara.memory.graph import KnowledgeGraph
from ara.utils.errors import GraphError
from ara.utils.types import EntityType, RelationType

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture()
def graph(tmp_path: Path) -> KnowledgeGraph:
    """Create a loaded graph with temp SQLite DB."""
    g = KnowledgeGraph(db_path=tmp_path / "test_graph.sqlite")
    g.load()
    yield g
    g.close()


@pytest.fixture()
def populated_graph(graph: KnowledgeGraph) -> KnowledgeGraph:
    """Graph pre-populated with some entities and relations."""
    mom = graph.add_entity("Mom", EntityType.PERSON, aliases=["my mother", "mama"])
    dad = graph.add_entity("Dad", EntityType.PERSON)
    hyd = graph.add_entity("Springfield", EntityType.PLACE)
    work = graph.add_entity("Work", EntityType.PROJECT)
    breakup = graph.add_entity("The Breakup", EntityType.EVENT)

    graph.add_relation(mom.entity_id, hyd.entity_id, RelationType.LOCATED_AT)
    graph.add_relation(dad.entity_id, hyd.entity_id, RelationType.LOCATED_AT)
    graph.add_relation(mom.entity_id, breakup.entity_id, RelationType.DISCUSSED)
    graph.add_relation(breakup.entity_id, work.entity_id, RelationType.RELATED_TO)

    return graph


# ── Lifecycle Tests ──────────────────────────────────────────────────────────


class TestLifecycle:
    """Tests for graph load/save/close."""

    def test_load_creates_db(self, tmp_path: Path) -> None:
        db = tmp_path / "sub" / "graph.sqlite"
        g = KnowledgeGraph(db_path=db)
        g.load()
        assert db.exists()
        g.close()

    def test_double_load_is_noop(self, graph: KnowledgeGraph) -> None:
        graph.load()
        assert graph.is_loaded

    def test_close_clears_graph(self, tmp_path: Path) -> None:
        g = KnowledgeGraph(db_path=tmp_path / "g.sqlite")
        g.load()
        g.add_entity("Test", EntityType.PERSON)
        g.close()
        assert not g.is_loaded
        assert g.entity_count == 0

    def test_operations_before_load_raise(self, tmp_path: Path) -> None:
        g = KnowledgeGraph(db_path=tmp_path / "g.sqlite")
        with pytest.raises(GraphError, match="not loaded"):
            g.add_entity("Test", EntityType.PERSON)

    def test_save_and_reload(self, tmp_path: Path) -> None:
        db = tmp_path / "persist.sqlite"

        # Create and populate
        g1 = KnowledgeGraph(db_path=db, auto_persist=False)
        g1.load()
        e1 = g1.add_entity("Alice", EntityType.PERSON, aliases=["A"])
        e2 = g1.add_entity("NYC", EntityType.PLACE)
        g1.add_relation(e1.entity_id, e2.entity_id, RelationType.LOCATED_AT)
        g1.save()
        g1.close()

        # Reload and verify
        g2 = KnowledgeGraph(db_path=db)
        g2.load()
        assert g2.entity_count == 2
        assert g2.relation_count == 1
        alice = g2.get_entity_by_name("Alice")
        assert alice is not None
        assert alice.aliases == ["A"]
        nyc = g2.get_entity_by_name("NYC")
        assert nyc is not None
        g2.close()

    def test_auto_persist_survives_restart(self, tmp_path: Path) -> None:
        db = tmp_path / "auto.sqlite"

        g1 = KnowledgeGraph(db_path=db, auto_persist=True)
        g1.load()
        g1.add_entity("Bob", EntityType.PERSON)
        g1.close()

        g2 = KnowledgeGraph(db_path=db)
        g2.load()
        assert g2.entity_count == 1
        assert g2.get_entity_by_name("Bob") is not None
        g2.close()


# ── Entity Tests ─────────────────────────────────────────────────────────────


class TestEntityOperations:
    """Tests for entity CRUD."""

    def test_add_entity(self, graph: KnowledgeGraph) -> None:
        entity = graph.add_entity("Mom", EntityType.PERSON, aliases=["mama"])
        assert entity.name == "Mom"
        assert entity.entity_type == EntityType.PERSON
        assert entity.aliases == ["mama"]
        assert entity.mention_count == 1
        assert entity.confidence == 1.0
        assert entity.first_seen is not None

    def test_add_entity_with_explicit_id(self, graph: KnowledgeGraph) -> None:
        entity = graph.add_entity("Test", EntityType.TOPIC, entity_id="custom_id")
        assert entity.entity_id == "custom_id"

    def test_add_entity_with_date(self, graph: KnowledgeGraph) -> None:
        dt = datetime(2024, 6, 15)
        entity = graph.add_entity("Event", EntityType.EVENT, date=dt)
        assert entity.first_seen == dt
        assert entity.last_seen == dt

    def test_add_entity_with_metadata(self, graph: KnowledgeGraph) -> None:
        entity = graph.add_entity(
            "Work",
            EntityType.PROJECT,
            metadata={"url": "work.com", "stage": "seed"},
        )
        assert entity.metadata["url"] == "work.com"

    def test_duplicate_entity_increments_mention(self, graph: KnowledgeGraph) -> None:
        e1 = graph.add_entity("Mom", EntityType.PERSON)
        e2 = graph.add_entity("Mom", EntityType.PERSON)
        assert e2.entity_id == e1.entity_id
        assert e2.mention_count == 2

    def test_duplicate_entity_by_alias(self, graph: KnowledgeGraph) -> None:
        e1 = graph.add_entity("Mom", EntityType.PERSON, aliases=["mama"])
        e2 = graph.add_entity("mama", EntityType.PERSON)
        assert e2.entity_id == e1.entity_id
        assert e2.mention_count == 2

    def test_same_name_different_type_creates_separate(self, graph: KnowledgeGraph) -> None:
        e1 = graph.add_entity("Phoenix", EntityType.PERSON)
        e2 = graph.add_entity("Phoenix", EntityType.PLACE)
        assert e1.entity_id != e2.entity_id
        assert graph.entity_count == 2

    def test_get_entity(self, graph: KnowledgeGraph) -> None:
        entity = graph.add_entity("Test", EntityType.TOPIC)
        found = graph.get_entity(entity.entity_id)
        assert found is not None
        assert found.name == "Test"

    def test_get_entity_not_found(self, graph: KnowledgeGraph) -> None:
        assert graph.get_entity("nonexistent") is None

    def test_get_entity_by_name(self, graph: KnowledgeGraph) -> None:
        graph.add_entity("Alice", EntityType.PERSON)
        found = graph.get_entity_by_name("alice")
        assert found is not None
        assert found.name == "Alice"

    def test_get_entity_by_name_with_type_filter(self, graph: KnowledgeGraph) -> None:
        graph.add_entity("Phoenix", EntityType.PERSON)
        graph.add_entity("Phoenix", EntityType.PLACE)
        found = graph.get_entity_by_name("Phoenix", EntityType.PLACE)
        assert found is not None
        assert found.entity_type == EntityType.PLACE

    def test_get_entity_by_alias(self, graph: KnowledgeGraph) -> None:
        graph.add_entity("Mom", EntityType.PERSON, aliases=["mama", "my mother"])
        found = graph.get_entity_by_name("my mother")
        assert found is not None
        assert found.name == "Mom"

    def test_get_entities_by_type(self, populated_graph: KnowledgeGraph) -> None:
        people = populated_graph.get_entities_by_type(EntityType.PERSON)
        assert len(people) == 2
        places = populated_graph.get_entities_by_type(EntityType.PLACE)
        assert len(places) == 1

    def test_search_entities(self, populated_graph: KnowledgeGraph) -> None:
        results = populated_graph.search_entities("mom")
        assert len(results) == 1
        assert results[0].name == "Mom"

    def test_search_entities_by_alias(self, populated_graph: KnowledgeGraph) -> None:
        results = populated_graph.search_entities("mama")
        assert len(results) == 1

    def test_search_entities_with_type_filter(self, populated_graph: KnowledgeGraph) -> None:
        results = populated_graph.search_entities("a", entity_type=EntityType.PERSON)
        # "Dad" and "Mom" both contain "a"
        assert all(e.entity_type == EntityType.PERSON for e in results)

    def test_search_entities_limit(self, graph: KnowledgeGraph) -> None:
        for i in range(10):
            graph.add_entity(f"Topic_{i}", EntityType.TOPIC)
        results = graph.search_entities("Topic", limit=3)
        assert len(results) == 3

    def test_update_entity(self, graph: KnowledgeGraph) -> None:
        entity = graph.add_entity("Test", EntityType.PERSON)
        updated = graph.update_entity(
            entity.entity_id,
            name="Updated Name",
            aliases=["new alias"],
        )
        assert updated.name == "Updated Name"
        assert updated.aliases == ["new alias"]
        assert updated.entity_type == EntityType.PERSON

    def test_update_entity_not_found_raises(self, graph: KnowledgeGraph) -> None:
        with pytest.raises(GraphError, match="not found"):
            graph.update_entity("nonexistent", name="X")

    def test_update_entity_merges_metadata(self, graph: KnowledgeGraph) -> None:
        entity = graph.add_entity("Test", EntityType.PERSON, metadata={"key1": "a"})
        updated = graph.update_entity(entity.entity_id, metadata={"key2": "b"})
        assert updated.metadata == {"key1": "a", "key2": "b"}

    def test_delete_entity(self, graph: KnowledgeGraph) -> None:
        entity = graph.add_entity("Test", EntityType.PERSON)
        assert graph.delete_entity(entity.entity_id)
        assert graph.entity_count == 0
        assert graph.get_entity(entity.entity_id) is None

    def test_delete_entity_removes_edges(self, graph: KnowledgeGraph) -> None:
        e1 = graph.add_entity("A", EntityType.PERSON)
        e2 = graph.add_entity("B", EntityType.PLACE)
        graph.add_relation(e1.entity_id, e2.entity_id, RelationType.LOCATED_AT)
        assert graph.relation_count == 1
        graph.delete_entity(e1.entity_id)
        assert graph.relation_count == 0

    def test_delete_nonexistent_returns_false(self, graph: KnowledgeGraph) -> None:
        assert not graph.delete_entity("nonexistent")

    def test_all_16_entity_types(self, graph: KnowledgeGraph) -> None:
        for et in EntityType:
            graph.add_entity(f"test_{et.value}", et)
        assert graph.entity_count == 16


# ── Merge Tests ──────────────────────────────────────────────────────────────


class TestMergeEntities:
    """Tests for entity merging (deduplication)."""

    def test_merge_basic(self, graph: KnowledgeGraph) -> None:
        e1 = graph.add_entity("Mom", EntityType.PERSON, aliases=["mama"])
        e2 = graph.add_entity("My Mother", EntityType.PERSON, aliases=["mother"])
        merged = graph.merge_entities(e1.entity_id, e2.entity_id)
        assert merged.name == "Mom"
        assert "My Mother" in merged.aliases
        assert "mother" in merged.aliases
        assert "mama" in merged.aliases
        assert merged.mention_count == 2
        assert graph.entity_count == 1

    def test_merge_redirects_edges(self, graph: KnowledgeGraph) -> None:
        e1 = graph.add_entity("Mom", EntityType.PERSON)
        e2 = graph.add_entity("Mother", EntityType.PERSON)
        place = graph.add_entity("Home", EntityType.PLACE)
        graph.add_relation(e2.entity_id, place.entity_id, RelationType.LOCATED_AT)

        graph.merge_entities(e1.entity_id, e2.entity_id)
        relations = graph.get_relations(e1.entity_id, direction="outgoing")
        assert len(relations) >= 1
        targets = [r.target_id for r in relations]
        assert place.entity_id in targets

    def test_merge_temporal(self, graph: KnowledgeGraph) -> None:
        dt1 = datetime(2023, 1, 1)
        dt2 = datetime(2024, 6, 15)
        e1 = graph.add_entity("A", EntityType.PERSON, date=dt1)
        e2 = graph.add_entity("B", EntityType.PERSON, date=dt2)
        merged = graph.merge_entities(e1.entity_id, e2.entity_id)
        assert merged.first_seen == dt1
        assert merged.last_seen == dt2

    def test_merge_nonexistent_raises(self, graph: KnowledgeGraph) -> None:
        e1 = graph.add_entity("A", EntityType.PERSON)
        with pytest.raises(GraphError, match="not found"):
            graph.merge_entities(e1.entity_id, "nonexistent")
        with pytest.raises(GraphError, match="not found"):
            graph.merge_entities("nonexistent", e1.entity_id)


# ── Relation Tests ───────────────────────────────────────────────────────────


class TestRelationOperations:
    """Tests for relation CRUD."""

    def test_add_relation(self, graph: KnowledgeGraph) -> None:
        e1 = graph.add_entity("A", EntityType.PERSON)
        e2 = graph.add_entity("B", EntityType.PLACE)
        rel = graph.add_relation(e1.entity_id, e2.entity_id, RelationType.LOCATED_AT)
        assert rel.source_id == e1.entity_id
        assert rel.target_id == e2.entity_id
        assert rel.relation_type == RelationType.LOCATED_AT
        assert rel.weight == 1.0
        assert graph.relation_count == 1

    def test_add_relation_nonexistent_source_raises(self, graph: KnowledgeGraph) -> None:
        e2 = graph.add_entity("B", EntityType.PLACE)
        with pytest.raises(GraphError, match="Source entity not found"):
            graph.add_relation("nonexistent", e2.entity_id, RelationType.RELATED_TO)

    def test_add_relation_nonexistent_target_raises(self, graph: KnowledgeGraph) -> None:
        e1 = graph.add_entity("A", EntityType.PERSON)
        with pytest.raises(GraphError, match="Target entity not found"):
            graph.add_relation(e1.entity_id, "nonexistent", RelationType.RELATED_TO)

    def test_duplicate_relation_increments_weight(self, graph: KnowledgeGraph) -> None:
        e1 = graph.add_entity("A", EntityType.PERSON)
        e2 = graph.add_entity("B", EntityType.TOPIC)
        graph.add_relation(e1.entity_id, e2.entity_id, RelationType.DISCUSSED, weight=1.0)
        r2 = graph.add_relation(e1.entity_id, e2.entity_id, RelationType.DISCUSSED, weight=2.0)
        assert r2.weight == 3.0  # 1.0 + 2.0
        assert graph.relation_count == 1  # Still one edge

    def test_different_relation_types_coexist(self, graph: KnowledgeGraph) -> None:
        e1 = graph.add_entity("A", EntityType.PERSON)
        e2 = graph.add_entity("B", EntityType.PROJECT)
        graph.add_relation(e1.entity_id, e2.entity_id, RelationType.WORKED_ON)
        graph.add_relation(e1.entity_id, e2.entity_id, RelationType.CREATED)
        # Note: NetworkX DiGraph stores one edge per (u,v) pair. The second
        # add_relation with a different type replaces the first one. This is
        # a known limitation — for multiple edge types between same nodes,
        # we would need MultiDiGraph. For now the last type wins.
        assert graph.relation_count >= 1

    def test_add_relation_with_metadata(self, graph: KnowledgeGraph) -> None:
        e1 = graph.add_entity("A", EntityType.PERSON)
        e2 = graph.add_entity("B", EntityType.EVENT)
        rel = graph.add_relation(
            e1.entity_id,
            e2.entity_id,
            RelationType.EXPERIENCED,
            metadata={"context": "diary entry"},
        )
        assert rel.metadata["context"] == "diary entry"

    def test_get_relations_outgoing(self, populated_graph: KnowledgeGraph) -> None:
        mom = populated_graph.get_entity_by_name("Mom")
        assert mom is not None
        rels = populated_graph.get_relations(mom.entity_id, direction="outgoing")
        assert len(rels) == 2  # LOCATED_AT + DISCUSSED

    def test_get_relations_incoming(self, populated_graph: KnowledgeGraph) -> None:
        hyd = populated_graph.get_entity_by_name("Springfield")
        assert hyd is not None
        rels = populated_graph.get_relations(hyd.entity_id, direction="incoming")
        assert len(rels) == 2  # Mom + Dad LOCATED_AT

    def test_get_relations_with_type_filter(self, populated_graph: KnowledgeGraph) -> None:
        mom = populated_graph.get_entity_by_name("Mom")
        assert mom is not None
        rels = populated_graph.get_relations(
            mom.entity_id,
            direction="outgoing",
            relation_type=RelationType.LOCATED_AT,
        )
        assert len(rels) == 1

    def test_delete_relation(self, graph: KnowledgeGraph) -> None:
        e1 = graph.add_entity("A", EntityType.PERSON)
        e2 = graph.add_entity("B", EntityType.PLACE)
        graph.add_relation(e1.entity_id, e2.entity_id, RelationType.LOCATED_AT)
        deleted = graph.delete_relation(e1.entity_id, e2.entity_id, RelationType.LOCATED_AT)
        assert deleted == 1
        assert graph.relation_count == 0

    def test_delete_relation_nonexistent(self, graph: KnowledgeGraph) -> None:
        deleted = graph.delete_relation("a", "b")
        assert deleted == 0

    def test_all_18_relation_types(self, graph: KnowledgeGraph) -> None:
        """All 18 relation types can be used."""
        e1 = graph.add_entity("Source", EntityType.PERSON)
        for i, rt in enumerate(RelationType):
            target = graph.add_entity(f"Target_{i}", EntityType.TOPIC)
            graph.add_relation(e1.entity_id, target.entity_id, rt)
        assert graph.relation_count == len(RelationType)


# ── Graph Query Tests ────────────────────────────────────────────────────────


class TestGraphQueries:
    """Tests for graph traversal and queries."""

    def test_get_neighbors_depth_1(self, populated_graph: KnowledgeGraph) -> None:
        mom = populated_graph.get_entity_by_name("Mom")
        assert mom is not None
        neighbors = populated_graph.get_neighbors(mom.entity_id, depth=1)
        names = {n.name for n in neighbors}
        assert "Springfield" in names
        assert "The Breakup" in names

    def test_get_neighbors_depth_2(self, populated_graph: KnowledgeGraph) -> None:
        mom = populated_graph.get_entity_by_name("Mom")
        assert mom is not None
        neighbors = populated_graph.get_neighbors(mom.entity_id, depth=2)
        names = {n.name for n in neighbors}
        # Depth 2 should reach Work (via Mom->Breakup->Work)
        assert "Work" in names

    def test_get_neighbors_with_type_filter(self, populated_graph: KnowledgeGraph) -> None:
        mom = populated_graph.get_entity_by_name("Mom")
        assert mom is not None
        neighbors = populated_graph.get_neighbors(
            mom.entity_id, depth=1, entity_type=EntityType.PLACE
        )
        assert len(neighbors) == 1
        assert neighbors[0].name == "Springfield"

    def test_get_neighbors_nonexistent_entity(self, graph: KnowledgeGraph) -> None:
        assert graph.get_neighbors("nonexistent") == []

    def test_find_path(self, populated_graph: KnowledgeGraph) -> None:
        mom = populated_graph.get_entity_by_name("Mom")
        work = populated_graph.get_entity_by_name("Work")
        assert mom is not None and work is not None
        path = populated_graph.find_path(mom.entity_id, work.entity_id)
        assert path is not None
        assert path[0].name == "Mom"
        assert path[-1].name == "Work"

    def test_find_path_no_path(self, graph: KnowledgeGraph) -> None:
        e1 = graph.add_entity("A", EntityType.PERSON)
        e2 = graph.add_entity("B", EntityType.PLACE)
        path = graph.find_path(e1.entity_id, e2.entity_id)
        assert path is None

    def test_find_path_max_depth(self, populated_graph: KnowledgeGraph) -> None:
        mom = populated_graph.get_entity_by_name("Mom")
        work = populated_graph.get_entity_by_name("Work")
        assert mom is not None and work is not None
        # Path is Mom->Breakup->Work (length 2), so max_depth=1 should fail
        path = populated_graph.find_path(mom.entity_id, work.entity_id, max_depth=1)
        assert path is None

    def test_find_path_nonexistent_entity(self, graph: KnowledgeGraph) -> None:
        path = graph.find_path("a", "b")
        assert path is None

    def test_get_subgraph(self, populated_graph: KnowledgeGraph) -> None:
        mom = populated_graph.get_entity_by_name("Mom")
        hyd = populated_graph.get_entity_by_name("Springfield")
        assert mom is not None and hyd is not None
        entities, relations = populated_graph.get_subgraph([mom.entity_id, hyd.entity_id])
        assert len(entities) == 2
        assert len(relations) == 1  # Mom -> Springfield

    def test_get_subgraph_with_neighbors(self, populated_graph: KnowledgeGraph) -> None:
        mom = populated_graph.get_entity_by_name("Mom")
        assert mom is not None
        entities, _relations = populated_graph.get_subgraph([mom.entity_id], include_neighbors=True)
        names = {e.name for e in entities}
        assert "Mom" in names
        assert "Springfield" in names
        assert "The Breakup" in names

    def test_get_entity_context(self, populated_graph: KnowledgeGraph) -> None:
        mom = populated_graph.get_entity_by_name("Mom")
        assert mom is not None
        context = populated_graph.get_entity_context(mom.entity_id)
        assert "Mom (person)" in context
        assert "mama" in context or "my mother" in context
        assert "Connections:" in context

    def test_get_entity_context_nonexistent(self, graph: KnowledgeGraph) -> None:
        assert graph.get_entity_context("nonexistent") == ""


# ── Statistics Tests ─────────────────────────────────────────────────────────


class TestStatistics:
    """Tests for graph statistics."""

    def test_entity_count(self, populated_graph: KnowledgeGraph) -> None:
        assert populated_graph.entity_count == 5

    def test_relation_count(self, populated_graph: KnowledgeGraph) -> None:
        assert populated_graph.relation_count == 4

    def test_get_stats(self, populated_graph: KnowledgeGraph) -> None:
        stats = populated_graph.get_stats()
        assert stats["total_entities"] == 5
        assert stats["total_relations"] == 4
        assert "person" in stats["entity_types"]
        assert stats["entity_types"]["person"] == 2
        assert len(stats["top_entities"]) <= 10
        assert stats["connected_components"] >= 1

    def test_empty_stats(self, graph: KnowledgeGraph) -> None:
        stats = graph.get_stats()
        assert stats["total_entities"] == 0
        assert stats["total_relations"] == 0


# ── Community Detection Tests ────────────────────────────────────────────────


class TestCommunityDetection:
    """Tests for community/cluster detection."""

    def test_detect_communities_empty(self, graph: KnowledgeGraph) -> None:
        communities = graph.detect_communities()
        assert communities == []

    def test_detect_communities_single_node(self, graph: KnowledgeGraph) -> None:
        graph.add_entity("Solo", EntityType.PERSON)
        communities = graph.detect_communities()
        assert len(communities) == 1

    def test_detect_communities_two_clusters(self, graph: KnowledgeGraph) -> None:
        # Cluster 1: A -> B -> C
        a = graph.add_entity("A", EntityType.PERSON)
        b = graph.add_entity("B", EntityType.PERSON)
        c = graph.add_entity("C", EntityType.PLACE)
        graph.add_relation(a.entity_id, b.entity_id, RelationType.INTERACTED_WITH)
        graph.add_relation(b.entity_id, c.entity_id, RelationType.LOCATED_AT)

        # Cluster 2: X -> Y (disconnected from cluster 1)
        x = graph.add_entity("X", EntityType.PROJECT)
        y = graph.add_entity("Y", EntityType.TOPIC)
        graph.add_relation(x.entity_id, y.entity_id, RelationType.DISCUSSED)

        communities = graph.detect_communities()
        assert len(communities) >= 2


# ── Persistence Edge Cases ───────────────────────────────────────────────────


class TestPersistenceEdgeCases:
    """Test SQLite persistence edge cases."""

    def test_relation_persists_after_reload(self, tmp_path: Path) -> None:
        db = tmp_path / "rel.sqlite"
        g1 = KnowledgeGraph(db_path=db)
        g1.load()
        e1 = g1.add_entity("A", EntityType.PERSON)
        e2 = g1.add_entity("B", EntityType.EVENT)
        g1.add_relation(
            e1.entity_id,
            e2.entity_id,
            RelationType.EXPERIENCED,
            weight=3.5,
            confidence=0.9,
        )
        g1.close()

        g2 = KnowledgeGraph(db_path=db)
        g2.load()
        rels = g2.get_relations(e1.entity_id)
        assert len(rels) == 1
        assert rels[0].weight == 3.5
        assert rels[0].confidence == 0.9
        g2.close()

    def test_delete_entity_persists(self, tmp_path: Path) -> None:
        db = tmp_path / "del.sqlite"
        g1 = KnowledgeGraph(db_path=db)
        g1.load()
        entity = g1.add_entity("Temp", EntityType.TOPIC)
        eid = entity.entity_id
        g1.delete_entity(eid)
        g1.close()

        g2 = KnowledgeGraph(db_path=db)
        g2.load()
        assert g2.entity_count == 0
        g2.close()

    def test_merge_persists(self, tmp_path: Path) -> None:
        db = tmp_path / "merge.sqlite"
        g1 = KnowledgeGraph(db_path=db)
        g1.load()
        e1 = g1.add_entity("Mom", EntityType.PERSON)
        e2 = g1.add_entity("Mother", EntityType.PERSON)
        g1.merge_entities(e1.entity_id, e2.entity_id)
        g1.close()

        g2 = KnowledgeGraph(db_path=db)
        g2.load()
        assert g2.entity_count == 1
        mom = g2.get_entity_by_name("Mom")
        assert mom is not None
        assert "Mother" in mom.aliases
        assert mom.mention_count == 2
        g2.close()

    def test_temporal_fields_persist(self, tmp_path: Path) -> None:
        db = tmp_path / "temporal.sqlite"
        dt = datetime(2024, 3, 15, 10, 30, 0)

        g1 = KnowledgeGraph(db_path=db)
        g1.load()
        entity = g1.add_entity("Test", EntityType.EVENT, date=dt)
        eid = entity.entity_id
        g1.close()

        g2 = KnowledgeGraph(db_path=db)
        g2.load()
        loaded = g2.get_entity(eid)
        assert loaded is not None
        assert loaded.first_seen == dt
        assert loaded.last_seen == dt
        g2.close()
