"""Knowledge graph for ARA's memory system.

In-memory NetworkX directed graph persisted to SQLite. Tracks entities
(people, places, events, projects, goals, habits, emotions, topics,
organizations, documents, decisions, milestones, health, financial,
relationships, media) and typed relationships between them.

The graph complements the vector store: vector search finds relevant text
chunks, while the graph enables multi-hop reasoning ("Who was I with when
I decided to quit?", "What happened after the breakup?").

Architecture:
    - NetworkX DiGraph held in memory for fast traversal
    - SQLite file on disk for persistence (load on startup, save on mutation)
    - All mutations go through this module — never modify the graph directly
    - Thread-safe via a reentrant lock on all write operations

Usage:
    from ara.memory.graph import KnowledgeGraph

    graph = KnowledgeGraph(db_path=Path("data/graph_db/graph.sqlite"))
    graph.load()

    # Add entities
    person = graph.add_entity("Mom", EntityType.PERSON, aliases=["my mother", "mama"])
    place = graph.add_entity("Springfield", EntityType.PLACE)

    # Add relationship
    graph.add_relation(person.entity_id, place.entity_id, RelationType.LOCATED_AT)

    # Query
    neighbors = graph.get_neighbors(person.entity_id)
    path = graph.find_path(person.entity_id, place.entity_id)

    # Persist
    graph.save()
    graph.close()
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import uuid4

import networkx as nx
import structlog

from ara.utils.errors import GraphError
from ara.utils.types import (
    EntityType,
    GraphEntity,
    GraphRelation,
    RelationType,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

log = structlog.get_logger(component="graph")

# ── SQL Schema ───────────────────────────────────────────────────────────────

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS entities (
    entity_id   TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    aliases     TEXT NOT NULL DEFAULT '[]',
    first_seen  TEXT,
    last_seen   TEXT,
    mention_count INTEGER NOT NULL DEFAULT 1,
    confidence  REAL NOT NULL DEFAULT 1.0,
    metadata    TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS relations (
    rowid         INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id     TEXT NOT NULL,
    target_id     TEXT NOT NULL,
    relation_type TEXT NOT NULL,
    weight        REAL NOT NULL DEFAULT 1.0,
    confidence    REAL NOT NULL DEFAULT 1.0,
    first_seen    TEXT,
    last_seen     TEXT,
    date          TEXT,
    source_chunk_id TEXT NOT NULL DEFAULT '',
    metadata      TEXT NOT NULL DEFAULT '{}',
    FOREIGN KEY (source_id) REFERENCES entities(entity_id) ON DELETE CASCADE,
    FOREIGN KEY (target_id) REFERENCES entities(entity_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type);
CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name);
CREATE INDEX IF NOT EXISTS idx_relations_source ON relations(source_id);
CREATE INDEX IF NOT EXISTS idx_relations_target ON relations(target_id);
CREATE INDEX IF NOT EXISTS idx_relations_type ON relations(relation_type);
"""


# ── Helper Functions ─────────────────────────────────────────────────────────


def _dt_to_iso(dt: datetime | None) -> str | None:
    """Convert datetime to ISO string, or None."""
    return dt.isoformat() if dt else None


def _iso_to_dt(iso: str | None) -> datetime | None:
    """Convert ISO string back to datetime, or None."""
    if not iso:
        return None
    return datetime.fromisoformat(iso)


def _entity_from_row(row: sqlite3.Row) -> GraphEntity:
    """Reconstruct a GraphEntity from a SQLite row."""
    return GraphEntity(
        entity_id=row["entity_id"],
        name=row["name"],
        entity_type=EntityType(row["entity_type"]),
        aliases=json.loads(row["aliases"]),
        first_seen=_iso_to_dt(row["first_seen"]),
        last_seen=_iso_to_dt(row["last_seen"]),
        mention_count=row["mention_count"],
        confidence=row["confidence"],
        metadata=json.loads(row["metadata"]),
    )


def _relation_from_row(row: sqlite3.Row) -> GraphRelation:
    """Reconstruct a GraphRelation from a SQLite row."""
    return GraphRelation(
        source_id=row["source_id"],
        target_id=row["target_id"],
        relation_type=RelationType(row["relation_type"]),
        weight=row["weight"],
        confidence=row["confidence"],
        first_seen=_iso_to_dt(row["first_seen"]),
        last_seen=_iso_to_dt(row["last_seen"]),
        date=_iso_to_dt(row["date"]),
        source_chunk_id=row["source_chunk_id"],
        metadata=json.loads(row["metadata"]),
    )


# ── Knowledge Graph ──────────────────────────────────────────────────────────


class KnowledgeGraph:
    """In-memory NetworkX knowledge graph with SQLite persistence.

    Thread-safe for concurrent reads and serialized writes. The graph is
    loaded entirely into memory on ``load()`` and persisted on ``save()``.
    Individual mutations (add/update/delete) are auto-persisted to SQLite
    by default (``auto_persist=True``).

    Args:
        db_path: Path to the SQLite database file. Parent dirs are created.
        auto_persist: If True, every mutation writes to SQLite immediately.
            Set to False for batch operations, then call ``save()`` manually.
    """

    def __init__(self, db_path: Path, *, auto_persist: bool = True) -> None:
        self._db_path = Path(db_path)
        self._auto_persist = auto_persist
        self._graph: nx.DiGraph = nx.DiGraph()
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.RLock()
        self._loaded = False

    # ── Lifecycle ────────────────────────────────────────────────────────

    def load(self) -> None:
        """Load graph from SQLite into memory. Creates DB if it doesn't exist."""
        with self._lock:
            if self._loaded:
                return

            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(
                str(self._db_path),
                check_same_thread=False,
            )
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.executescript(_SCHEMA_SQL)
            self._conn.commit()

            # Load entities into graph nodes
            entity_count = 0
            for row in self._conn.execute("SELECT * FROM entities"):
                entity = _entity_from_row(row)
                self._graph.add_node(
                    entity.entity_id,
                    entity=entity,
                )
                entity_count += 1

            # Load relations into graph edges
            relation_count = 0
            for row in self._conn.execute("SELECT * FROM relations"):
                relation = _relation_from_row(row)
                self._graph.add_edge(
                    relation.source_id,
                    relation.target_id,
                    key=row["rowid"],
                    relation=relation,
                )
                relation_count += 1

            self._loaded = True
            log.info(
                "graph_loaded",
                db_path=str(self._db_path),
                entities=entity_count,
                relations=relation_count,
            )

    def save(self) -> None:
        """Persist the full in-memory graph to SQLite (full rewrite).

        This is a bulk operation — drops all existing data and re-inserts
        from the in-memory graph. Use for batch operations where
        ``auto_persist=False``.
        """
        with self._lock:
            self._ensure_loaded()
            conn = self._get_conn()

            start = time.perf_counter()
            conn.execute("BEGIN")
            try:
                conn.execute("DELETE FROM relations")
                conn.execute("DELETE FROM entities")

                for node_id in self._graph.nodes:
                    entity: GraphEntity = self._graph.nodes[node_id]["entity"]
                    conn.execute(
                        """INSERT INTO entities
                           (entity_id, name, entity_type, aliases, first_seen,
                            last_seen, mention_count, confidence, metadata)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            entity.entity_id,
                            entity.name,
                            entity.entity_type.value,
                            json.dumps(entity.aliases),
                            _dt_to_iso(entity.first_seen),
                            _dt_to_iso(entity.last_seen),
                            entity.mention_count,
                            entity.confidence,
                            json.dumps(entity.metadata),
                        ),
                    )

                for _u, _v, data in self._graph.edges(data=True):
                    relation: GraphRelation = data["relation"]
                    conn.execute(
                        """INSERT INTO relations
                           (source_id, target_id, relation_type, weight, confidence,
                            first_seen, last_seen, date, source_chunk_id, metadata)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            relation.source_id,
                            relation.target_id,
                            relation.relation_type.value,
                            relation.weight,
                            relation.confidence,
                            _dt_to_iso(relation.first_seen),
                            _dt_to_iso(relation.last_seen),
                            _dt_to_iso(relation.date),
                            relation.source_chunk_id,
                            json.dumps(relation.metadata),
                        ),
                    )

                conn.commit()
            except Exception:
                conn.rollback()
                raise

            elapsed_ms = (time.perf_counter() - start) * 1000
            log.info(
                "graph_saved",
                entities=self._graph.number_of_nodes(),
                relations=self._graph.number_of_edges(),
                elapsed_ms=round(elapsed_ms, 1),
            )

    def close(self) -> None:
        """Close the SQLite connection and clear the in-memory graph."""
        with self._lock:
            if self._conn:
                self._conn.close()
                self._conn = None
            self._graph.clear()
            self._loaded = False
            log.info("graph_closed")

    @property
    def is_loaded(self) -> bool:
        """Whether the graph has been loaded from disk."""
        return self._loaded

    # ── Entity Operations ────────────────────────────────────────────────

    def add_entity(
        self,
        name: str,
        entity_type: EntityType,
        *,
        aliases: Sequence[str] = (),
        confidence: float = 1.0,
        date: datetime | None = None,
        metadata: dict[str, object] | None = None,
        entity_id: str | None = None,
    ) -> GraphEntity:
        """Add a new entity node to the graph.

        If an entity with the same name and type already exists, its
        mention count is incremented and temporal fields are updated
        instead of creating a duplicate.

        Args:
            name: Primary display name of the entity.
            entity_type: One of the 16 EntityType values.
            aliases: Alternative names for deduplication.
            confidence: Extraction confidence score (0.0 to 1.0).
            date: When this entity was observed (for temporal tracking).
            metadata: Arbitrary key-value pairs.
            entity_id: Optional explicit ID. Auto-generated if not provided.

        Returns:
            The created or updated GraphEntity.
        """
        with self._lock:
            self._ensure_loaded()

            # Check for existing entity with same name + type
            existing = self._find_entity_by_name_and_type(name, entity_type)
            if existing is not None:
                return self._update_entity_mention(existing, date=date, aliases=list(aliases))

            eid = entity_id or uuid4().hex[:16]
            now = date or datetime.now()
            entity = GraphEntity(
                entity_id=eid,
                name=name,
                entity_type=entity_type,
                aliases=list(aliases),
                first_seen=now,
                last_seen=now,
                mention_count=1,
                confidence=confidence,
                metadata=metadata or {},
            )
            self._graph.add_node(eid, entity=entity)

            if self._auto_persist:
                self._persist_entity(entity)

            log.debug(
                "entity_added",
                entity_id=eid,
                name=name,
                entity_type=entity_type.value,
            )
            return entity

    def get_entity(self, entity_id: str) -> GraphEntity | None:
        """Get an entity by its ID, or None if not found."""
        if entity_id not in self._graph.nodes:
            return None
        return self._graph.nodes[entity_id].get("entity")

    def get_entity_by_name(
        self,
        name: str,
        entity_type: EntityType | None = None,
    ) -> GraphEntity | None:
        """Find an entity by name (case-insensitive) and optionally type.

        Also searches aliases.
        """
        name_lower = name.lower()
        for node_id in self._graph.nodes:
            entity: GraphEntity = self._graph.nodes[node_id]["entity"]
            if entity_type and entity.entity_type != entity_type:
                continue
            if entity.name.lower() == name_lower:
                return entity
            if any(a.lower() == name_lower for a in entity.aliases):
                return entity
        return None

    def get_entities_by_type(self, entity_type: EntityType) -> list[GraphEntity]:
        """Get all entities of a specific type."""
        results: list[GraphEntity] = []
        for node_id in self._graph.nodes:
            entity: GraphEntity = self._graph.nodes[node_id]["entity"]
            if entity.entity_type == entity_type:
                results.append(entity)
        return results

    def search_entities(
        self,
        query: str,
        *,
        entity_type: EntityType | None = None,
        limit: int = 20,
    ) -> list[GraphEntity]:
        """Search entities by substring match on name and aliases.

        Args:
            query: Substring to search for (case-insensitive).
            entity_type: Optional filter by entity type.
            limit: Maximum number of results.

        Returns:
            Matching entities sorted by mention count (most mentioned first).
        """
        query_lower = query.lower()
        results: list[GraphEntity] = []
        for node_id in self._graph.nodes:
            entity: GraphEntity = self._graph.nodes[node_id]["entity"]
            if entity_type and entity.entity_type != entity_type:
                continue
            name_match = query_lower in entity.name.lower()
            alias_match = any(query_lower in a.lower() for a in entity.aliases)
            if name_match or alias_match:
                results.append(entity)

        results.sort(key=lambda e: e.mention_count, reverse=True)
        return results[:limit]

    def update_entity(
        self,
        entity_id: str,
        *,
        name: str | None = None,
        aliases: list[str] | None = None,
        confidence: float | None = None,
        metadata: dict[str, object] | None = None,
    ) -> GraphEntity:
        """Update an existing entity's fields.

        Only specified (non-None) fields are updated. Returns the updated entity.

        Raises:
            GraphError: If entity_id does not exist.
        """
        with self._lock:
            self._ensure_loaded()
            old = self.get_entity(entity_id)
            if old is None:
                raise GraphError(
                    f"Entity not found: {entity_id}",
                    operation="update_entity",
                    details={"entity_id": entity_id},
                )

            updated = GraphEntity(
                entity_id=old.entity_id,
                name=name if name is not None else old.name,
                entity_type=old.entity_type,
                aliases=aliases if aliases is not None else old.aliases,
                first_seen=old.first_seen,
                last_seen=old.last_seen,
                mention_count=old.mention_count,
                confidence=confidence if confidence is not None else old.confidence,
                metadata={**old.metadata, **(metadata or {})},
            )
            self._graph.nodes[entity_id]["entity"] = updated

            if self._auto_persist:
                self._persist_entity(updated)

            return updated

    def delete_entity(self, entity_id: str) -> bool:
        """Remove an entity and all its edges from the graph.

        Returns True if the entity existed and was removed, False otherwise.
        """
        with self._lock:
            self._ensure_loaded()
            if entity_id not in self._graph.nodes:
                return False

            self._graph.remove_node(entity_id)

            if self._auto_persist:
                conn = self._get_conn()
                conn.execute(
                    "DELETE FROM relations WHERE source_id = ? OR target_id = ?",
                    (entity_id, entity_id),
                )
                conn.execute("DELETE FROM entities WHERE entity_id = ?", (entity_id,))
                conn.commit()

            log.debug("entity_deleted", entity_id=entity_id)
            return True

    def merge_entities(
        self,
        keep_id: str,
        merge_id: str,
    ) -> GraphEntity:
        """Merge two entities: keep one, absorb the other's data and edges.

        The ``merge_id`` entity is deleted. Its aliases, mention count,
        and temporal range are merged into ``keep_id``. All edges from/to
        ``merge_id`` are redirected to ``keep_id``.

        Args:
            keep_id: Entity to keep (the canonical one).
            merge_id: Entity to absorb and delete.

        Returns:
            The updated kept entity.

        Raises:
            GraphError: If either entity doesn't exist.
        """
        with self._lock:
            self._ensure_loaded()
            keep = self.get_entity(keep_id)
            merge = self.get_entity(merge_id)

            if keep is None:
                raise GraphError(
                    f"Keep entity not found: {keep_id}",
                    operation="merge_entities",
                    details={"keep_id": keep_id},
                )
            if merge is None:
                raise GraphError(
                    f"Merge entity not found: {merge_id}",
                    operation="merge_entities",
                    details={"merge_id": merge_id},
                )

            # Merge aliases (include merge entity's name as alias)
            merged_aliases = list(set(keep.aliases + merge.aliases + [merge.name]))
            # Remove keep entity's own name from aliases
            if keep.name in merged_aliases:
                merged_aliases.remove(keep.name)

            # Merge temporal range
            first_seen = keep.first_seen
            if merge.first_seen and (not first_seen or merge.first_seen < first_seen):
                first_seen = merge.first_seen
            last_seen = keep.last_seen
            if merge.last_seen and (not last_seen or merge.last_seen > last_seen):
                last_seen = merge.last_seen

            # Merge metadata
            merged_meta = {**merge.metadata, **keep.metadata}

            updated = GraphEntity(
                entity_id=keep.entity_id,
                name=keep.name,
                entity_type=keep.entity_type,
                aliases=merged_aliases,
                first_seen=first_seen,
                last_seen=last_seen,
                mention_count=keep.mention_count + merge.mention_count,
                confidence=max(keep.confidence, merge.confidence),
                metadata=merged_meta,
            )
            self._graph.nodes[keep_id]["entity"] = updated

            # Redirect edges from merge_id to keep_id
            # Incoming edges
            for pred in list(self._graph.predecessors(merge_id)):
                if pred == keep_id:
                    continue
                for _, _, data in self._graph.edges(pred, data=True):
                    if data.get("relation") and data["relation"].target_id == merge_id:
                        old_rel: GraphRelation = data["relation"]
                        new_rel = GraphRelation(
                            source_id=pred,
                            target_id=keep_id,
                            relation_type=old_rel.relation_type,
                            weight=old_rel.weight,
                            confidence=old_rel.confidence,
                            first_seen=old_rel.first_seen,
                            last_seen=old_rel.last_seen,
                            date=old_rel.date,
                            source_chunk_id=old_rel.source_chunk_id,
                            metadata=old_rel.metadata,
                        )
                        self._graph.add_edge(pred, keep_id, relation=new_rel)

            # Outgoing edges
            for succ in list(self._graph.successors(merge_id)):
                if succ == keep_id:
                    continue
                for _, _, data in self._graph.edges(merge_id, data=True):
                    if data.get("relation") and data["relation"].target_id == succ:
                        old_rel = data["relation"]
                        new_rel = GraphRelation(
                            source_id=keep_id,
                            target_id=succ,
                            relation_type=old_rel.relation_type,
                            weight=old_rel.weight,
                            confidence=old_rel.confidence,
                            first_seen=old_rel.first_seen,
                            last_seen=old_rel.last_seen,
                            date=old_rel.date,
                            source_chunk_id=old_rel.source_chunk_id,
                            metadata=old_rel.metadata,
                        )
                        self._graph.add_edge(keep_id, succ, relation=new_rel)

            # Remove merged entity (and its edges)
            self._graph.remove_node(merge_id)

            if self._auto_persist:
                self.save()

            log.info(
                "entities_merged",
                keep_id=keep_id,
                merge_id=merge_id,
                aliases=merged_aliases,
                mention_count=updated.mention_count,
            )
            return updated

    # ── Relation Operations ──────────────────────────────────────────────

    def add_relation(
        self,
        source_id: str,
        target_id: str,
        relation_type: RelationType,
        *,
        weight: float = 1.0,
        confidence: float = 1.0,
        date: datetime | None = None,
        source_chunk_id: str = "",
        metadata: dict[str, object] | None = None,
    ) -> GraphRelation:
        """Add a directed relationship between two entities.

        If the exact same edge (source, target, type) already exists,
        its weight is incremented and temporal fields are updated.

        Args:
            source_id: Source entity ID.
            target_id: Target entity ID.
            relation_type: One of the 18 RelationType values.
            weight: Edge weight (higher = stronger relationship).
            confidence: Extraction confidence (0.0 to 1.0).
            date: When this relationship was observed.
            source_chunk_id: The chunk ID this was extracted from.
            metadata: Arbitrary key-value pairs.

        Returns:
            The created or updated GraphRelation.

        Raises:
            GraphError: If either entity doesn't exist.
        """
        with self._lock:
            self._ensure_loaded()

            if source_id not in self._graph.nodes:
                raise GraphError(
                    f"Source entity not found: {source_id}",
                    operation="add_relation",
                    details={"source_id": source_id, "target_id": target_id},
                )
            if target_id not in self._graph.nodes:
                raise GraphError(
                    f"Target entity not found: {target_id}",
                    operation="add_relation",
                    details={"source_id": source_id, "target_id": target_id},
                )

            # Check for existing edge with same type
            existing = self._find_relation(source_id, target_id, relation_type)
            if existing is not None:
                return self._update_relation_weight(
                    source_id,
                    target_id,
                    existing,
                    weight_delta=weight,
                    date=date,
                )

            now = date or datetime.now()
            relation = GraphRelation(
                source_id=source_id,
                target_id=target_id,
                relation_type=relation_type,
                weight=weight,
                confidence=confidence,
                first_seen=now,
                last_seen=now,
                date=date,
                source_chunk_id=source_chunk_id,
                metadata=metadata or {},
            )
            self._graph.add_edge(source_id, target_id, relation=relation)

            if self._auto_persist:
                self._persist_relation(relation)

            log.debug(
                "relation_added",
                source_id=source_id,
                target_id=target_id,
                relation_type=relation_type.value,
            )
            return relation

    def get_relations(
        self,
        entity_id: str,
        *,
        direction: str = "both",
        relation_type: RelationType | None = None,
    ) -> list[GraphRelation]:
        """Get all relations for an entity.

        Args:
            entity_id: The entity to query.
            direction: "outgoing", "incoming", or "both".
            relation_type: Optional filter by relation type.

        Returns:
            List of GraphRelation objects.
        """
        results: list[GraphRelation] = []

        if direction in ("outgoing", "both"):
            for _, _, data in self._graph.out_edges(entity_id, data=True):
                rel: GraphRelation = data["relation"]
                if relation_type and rel.relation_type != relation_type:
                    continue
                results.append(rel)

        if direction in ("incoming", "both"):
            for _, _, data in self._graph.in_edges(entity_id, data=True):
                rel = data["relation"]
                if relation_type and rel.relation_type != relation_type:
                    continue
                results.append(rel)

        return results

    def delete_relation(
        self,
        source_id: str,
        target_id: str,
        relation_type: RelationType | None = None,
    ) -> int:
        """Delete relation(s) between two entities.

        Args:
            source_id: Source entity ID.
            target_id: Target entity ID.
            relation_type: If specified, only delete edges of this type.
                If None, delete ALL edges between source and target.

        Returns:
            Number of edges deleted.
        """
        with self._lock:
            self._ensure_loaded()
            deleted = 0

            if self._graph.has_edge(source_id, target_id):
                edge_data = self._graph.get_edge_data(source_id, target_id)
                if edge_data is None:
                    return 0

                rel: GraphRelation = edge_data.get("relation")  # type: ignore[assignment]
                if rel is None:
                    return 0

                if relation_type is None or rel.relation_type == relation_type:
                    self._graph.remove_edge(source_id, target_id)
                    deleted = 1

            if deleted > 0 and self._auto_persist:
                conn = self._get_conn()
                if relation_type:
                    conn.execute(
                        "DELETE FROM relations WHERE source_id = ? AND target_id = ? "
                        "AND relation_type = ?",
                        (source_id, target_id, relation_type.value),
                    )
                else:
                    conn.execute(
                        "DELETE FROM relations WHERE source_id = ? AND target_id = ?",
                        (source_id, target_id),
                    )
                conn.commit()

            return deleted

    # ── Graph Queries ────────────────────────────────────────────────────

    def get_neighbors(
        self,
        entity_id: str,
        *,
        depth: int = 1,
        entity_type: EntityType | None = None,
        relation_type: RelationType | None = None,
    ) -> list[GraphEntity]:
        """Get neighboring entities within N hops.

        Args:
            entity_id: Starting entity.
            depth: Maximum number of hops (1 = direct neighbors).
            entity_type: Only return neighbors of this type.
            relation_type: Only traverse edges of this type.

        Returns:
            List of neighboring entities (excluding the starting entity).
        """
        if entity_id not in self._graph.nodes:
            return []

        visited: set[str] = {entity_id}
        frontier: set[str] = {entity_id}

        for _ in range(depth):
            next_frontier: set[str] = set()
            for nid in frontier:
                # Successors
                for succ in self._graph.successors(nid):
                    if succ in visited:
                        continue
                    if relation_type:
                        edge_data = self._graph.get_edge_data(nid, succ)
                        if edge_data and edge_data.get("relation"):
                            rel: GraphRelation = edge_data["relation"]
                            if rel.relation_type != relation_type:
                                continue
                    next_frontier.add(succ)
                # Predecessors
                for pred in self._graph.predecessors(nid):
                    if pred in visited:
                        continue
                    if relation_type:
                        edge_data = self._graph.get_edge_data(pred, nid)
                        if edge_data and edge_data.get("relation"):
                            rel = edge_data["relation"]
                            if rel.relation_type != relation_type:
                                continue
                    next_frontier.add(pred)

            visited.update(next_frontier)
            frontier = next_frontier

        visited.discard(entity_id)
        results: list[GraphEntity] = []
        for nid in visited:
            entity: GraphEntity = self._graph.nodes[nid]["entity"]
            if entity_type and entity.entity_type != entity_type:
                continue
            results.append(entity)

        return results

    def find_path(
        self,
        source_id: str,
        target_id: str,
        *,
        max_depth: int = 5,
    ) -> list[GraphEntity] | None:
        """Find shortest path between two entities.

        Uses undirected BFS (ignores edge direction) to find the shortest
        path. Returns None if no path exists within max_depth.

        Args:
            source_id: Starting entity.
            target_id: Ending entity.
            max_depth: Maximum path length to search.

        Returns:
            List of entities along the path (inclusive), or None.
        """
        if source_id not in self._graph.nodes or target_id not in self._graph.nodes:
            return None

        undirected = self._graph.to_undirected()
        try:
            path = nx.shortest_path(undirected, source_id, target_id)
        except nx.NetworkXNoPath:
            return None

        if len(path) - 1 > max_depth:
            return None

        return [self._graph.nodes[nid]["entity"] for nid in path]

    def get_subgraph(
        self,
        entity_ids: Sequence[str],
        *,
        include_neighbors: bool = False,
    ) -> tuple[list[GraphEntity], list[GraphRelation]]:
        """Extract a subgraph containing the specified entities.

        Args:
            entity_ids: Entity IDs to include.
            include_neighbors: If True, also include direct neighbors.

        Returns:
            Tuple of (entities, relations) in the subgraph.
        """
        node_ids = set(entity_ids)

        if include_neighbors:
            for eid in list(entity_ids):
                if eid in self._graph.nodes:
                    node_ids.update(self._graph.successors(eid))
                    node_ids.update(self._graph.predecessors(eid))

        # Filter to existing nodes only
        node_ids = {nid for nid in node_ids if nid in self._graph.nodes}

        entities = [self._graph.nodes[nid]["entity"] for nid in node_ids]
        relations: list[GraphRelation] = []
        for u, v, data in self._graph.edges(data=True):
            if u in node_ids and v in node_ids:
                relations.append(data["relation"])

        return entities, relations

    def get_entity_context(
        self,
        entity_id: str,
        *,
        max_relations: int = 50,
    ) -> str:
        """Generate a natural language summary of an entity's graph context.

        Useful for injecting into LLM prompts as structured context.

        Args:
            entity_id: The entity to describe.
            max_relations: Maximum number of relations to include.

        Returns:
            A formatted string describing the entity and its connections.
        """
        entity = self.get_entity(entity_id)
        if entity is None:
            return ""

        lines: list[str] = []
        lines.append(f"{entity.name} ({entity.entity_type.value})")
        if entity.aliases:
            lines.append(f"  Also known as: {', '.join(entity.aliases)}")
        if entity.first_seen:
            lines.append(f"  First mentioned: {entity.first_seen.strftime('%Y-%m-%d')}")
        if entity.last_seen:
            lines.append(f"  Last mentioned: {entity.last_seen.strftime('%Y-%m-%d')}")
        lines.append(f"  Mentioned {entity.mention_count} times")

        relations = self.get_relations(entity_id)[:max_relations]
        if relations:
            lines.append("  Connections:")
            for rel in relations:
                other_id = rel.target_id if rel.source_id == entity_id else rel.source_id
                other = self.get_entity(other_id)
                other_name = other.name if other else other_id
                direction = "->" if rel.source_id == entity_id else "<-"
                lines.append(f"    {direction} {rel.relation_type.value} {other_name}")

        return "\n".join(lines)

    # ── Statistics ───────────────────────────────────────────────────────

    @property
    def entity_count(self) -> int:
        """Total number of entities in the graph."""
        return self._graph.number_of_nodes()

    @property
    def relation_count(self) -> int:
        """Total number of relations in the graph."""
        return self._graph.number_of_edges()

    def get_stats(self) -> dict[str, object]:
        """Get comprehensive graph statistics.

        Returns:
            Dictionary with entity/relation counts, type distributions,
            most connected entities, etc.
        """
        type_counts: dict[str, int] = {}
        for node_id in self._graph.nodes:
            entity: GraphEntity = self._graph.nodes[node_id]["entity"]
            t = entity.entity_type.value
            type_counts[t] = type_counts.get(t, 0) + 1

        relation_type_counts: dict[str, int] = {}
        for _, _, data in self._graph.edges(data=True):
            rel: GraphRelation = data["relation"]
            t = rel.relation_type.value
            relation_type_counts[t] = relation_type_counts.get(t, 0) + 1

        # Top entities by degree
        degree_list = sorted(
            self._graph.degree(),
            key=lambda x: x[1],
            reverse=True,
        )[:10]
        top_entities = []
        for node_id, degree in degree_list:
            entity = self._graph.nodes[node_id]["entity"]
            top_entities.append(
                {
                    "name": entity.name,
                    "type": entity.entity_type.value,
                    "degree": degree,
                    "mentions": entity.mention_count,
                }
            )

        connected = nx.number_connected_components(self._graph.to_undirected())

        return {
            "total_entities": self._graph.number_of_nodes(),
            "total_relations": self._graph.number_of_edges(),
            "entity_types": type_counts,
            "relation_types": relation_type_counts,
            "top_entities": top_entities,
            "connected_components": connected,
            "db_path": str(self._db_path),
        }

    # ── Community Detection ──────────────────────────────────────────────

    def detect_communities(self) -> list[list[GraphEntity]]:
        """Detect communities (clusters) in the graph.

        Uses the Louvain method on the undirected version of the graph.
        Returns a list of communities, each being a list of entities
        sorted by mention count.

        This is useful for detecting "life chapters" — clusters of
        entities that co-occur frequently (e.g., college friends cluster,
        work colleagues cluster, etc.).
        """
        if self._graph.number_of_nodes() == 0:
            return []

        undirected = self._graph.to_undirected()
        try:
            communities = nx.community.louvain_communities(
                undirected,
                seed=42,
            )
        except Exception:
            # Fallback to connected components if Louvain fails
            communities = list(nx.connected_components(undirected))

        result: list[list[GraphEntity]] = []
        for community in communities:
            entities = [
                self._graph.nodes[nid]["entity"] for nid in community if nid in self._graph.nodes
            ]
            entities.sort(key=lambda e: e.mention_count, reverse=True)
            if entities:
                result.append(entities)

        # Sort communities by size (largest first)
        result.sort(key=len, reverse=True)
        return result

    # ── Internal Helpers ─────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        """Raise if the graph has not been loaded."""
        if not self._loaded:
            raise GraphError(
                "Graph not loaded — call load() first",
                operation="ensure_loaded",
            )

    def _get_conn(self) -> sqlite3.Connection:
        """Get the SQLite connection, raising if not available."""
        if self._conn is None:
            raise GraphError(
                "No database connection — call load() first",
                operation="get_connection",
            )
        return self._conn

    def _find_entity_by_name_and_type(
        self,
        name: str,
        entity_type: EntityType,
    ) -> GraphEntity | None:
        """Find an existing entity with matching name and type."""
        name_lower = name.lower()
        for node_id in self._graph.nodes:
            entity: GraphEntity = self._graph.nodes[node_id]["entity"]
            if entity.entity_type != entity_type:
                continue
            if entity.name.lower() == name_lower:
                return entity
            if any(a.lower() == name_lower for a in entity.aliases):
                return entity
        return None

    def _update_entity_mention(
        self,
        entity: GraphEntity,
        *,
        date: datetime | None = None,
        aliases: list[str] | None = None,
    ) -> GraphEntity:
        """Increment mention count and update temporal fields."""
        now = date or datetime.now()
        last_seen = now
        if entity.last_seen and entity.last_seen > now:
            last_seen = entity.last_seen

        merged_aliases = list(set(entity.aliases + (aliases or [])))
        # Remove the entity's own name from aliases
        if entity.name in merged_aliases:
            merged_aliases.remove(entity.name)

        updated = GraphEntity(
            entity_id=entity.entity_id,
            name=entity.name,
            entity_type=entity.entity_type,
            aliases=merged_aliases,
            first_seen=entity.first_seen,
            last_seen=last_seen,
            mention_count=entity.mention_count + 1,
            confidence=entity.confidence,
            metadata=entity.metadata,
        )
        self._graph.nodes[entity.entity_id]["entity"] = updated

        if self._auto_persist:
            self._persist_entity(updated)

        return updated

    def _find_relation(
        self,
        source_id: str,
        target_id: str,
        relation_type: RelationType,
    ) -> GraphRelation | None:
        """Find an existing edge with matching source, target, and type."""
        if not self._graph.has_edge(source_id, target_id):
            return None
        edge_data = self._graph.get_edge_data(source_id, target_id)
        if edge_data is None:
            return None
        rel: GraphRelation | None = edge_data.get("relation")
        if rel and rel.relation_type == relation_type:
            return rel
        return None

    def _update_relation_weight(
        self,
        source_id: str,
        target_id: str,
        relation: GraphRelation,
        *,
        weight_delta: float,
        date: datetime | None = None,
    ) -> GraphRelation:
        """Increment relation weight and update temporal fields."""
        now = date or datetime.now()
        last_seen = now
        if relation.last_seen and relation.last_seen > now:
            last_seen = relation.last_seen

        updated = GraphRelation(
            source_id=relation.source_id,
            target_id=relation.target_id,
            relation_type=relation.relation_type,
            weight=relation.weight + weight_delta,
            confidence=relation.confidence,
            first_seen=relation.first_seen,
            last_seen=last_seen,
            date=date or relation.date,
            source_chunk_id=relation.source_chunk_id,
            metadata=relation.metadata,
        )
        self._graph.edges[source_id, target_id]["relation"] = updated

        if self._auto_persist:
            self._persist_relation_update(updated)

        return updated

    def _persist_entity(self, entity: GraphEntity) -> None:
        """Write a single entity to SQLite (upsert)."""
        conn = self._get_conn()
        conn.execute(
            """INSERT OR REPLACE INTO entities
               (entity_id, name, entity_type, aliases, first_seen,
                last_seen, mention_count, confidence, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                entity.entity_id,
                entity.name,
                entity.entity_type.value,
                json.dumps(entity.aliases),
                _dt_to_iso(entity.first_seen),
                _dt_to_iso(entity.last_seen),
                entity.mention_count,
                entity.confidence,
                json.dumps(entity.metadata),
            ),
        )
        conn.commit()

    def _persist_relation(self, relation: GraphRelation) -> None:
        """Write a single relation to SQLite (insert)."""
        conn = self._get_conn()
        conn.execute(
            """INSERT INTO relations
               (source_id, target_id, relation_type, weight, confidence,
                first_seen, last_seen, date, source_chunk_id, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                relation.source_id,
                relation.target_id,
                relation.relation_type.value,
                relation.weight,
                relation.confidence,
                _dt_to_iso(relation.first_seen),
                _dt_to_iso(relation.last_seen),
                _dt_to_iso(relation.date),
                relation.source_chunk_id,
                json.dumps(relation.metadata),
            ),
        )
        conn.commit()

    def _persist_relation_update(self, relation: GraphRelation) -> None:
        """Update an existing relation in SQLite by source/target/type."""
        conn = self._get_conn()
        conn.execute(
            """UPDATE relations SET weight = ?, confidence = ?,
               first_seen = ?, last_seen = ?, date = ?,
               source_chunk_id = ?, metadata = ?
               WHERE source_id = ? AND target_id = ? AND relation_type = ?""",
            (
                relation.weight,
                relation.confidence,
                _dt_to_iso(relation.first_seen),
                _dt_to_iso(relation.last_seen),
                _dt_to_iso(relation.date),
                relation.source_chunk_id,
                json.dumps(relation.metadata),
                relation.source_id,
                relation.target_id,
                relation.relation_type.value,
            ),
        )
        conn.commit()
