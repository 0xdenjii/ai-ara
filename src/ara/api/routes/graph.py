"""Knowledge graph route — entity search, neighbors, stats.

Endpoints:
    GET  /api/graph/stats — Graph statistics (entity/relation counts, type distributions)
    GET  /api/graph/search — Search entities by name substring
    GET  /api/graph/entity/{entity_id} — Get entity details with context
    GET  /api/graph/entity/{entity_id}/neighbors — Get entity neighbors (multi-hop)
    GET  /api/graph/entity/{entity_id}/relations — Get entity relations
    GET  /api/graph/communities — Detect and list graph communities
    GET  /api/graph/subgraph — Extract a subgraph for a set of entity IDs
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from ara.api.app import AppState, get_app_state
from ara.utils.logging import get_logger

log = get_logger("api.graph")

router = APIRouter(prefix="/api/graph", tags=["graph"])


# ── Response models (inline to avoid bloating schemas.py) ────────────────────


class GraphEntityItem(BaseModel):
    """A graph entity for API responses."""

    entity_id: str
    name: str
    entity_type: str
    aliases: list[str] = Field(default_factory=list)
    first_seen: str | None = None
    last_seen: str | None = None
    mention_count: int = 0
    confidence: float = 1.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class GraphRelationItem(BaseModel):
    """A graph relation for API responses."""

    source_id: str
    target_id: str
    relation_type: str
    weight: float = 1.0
    confidence: float = 1.0
    first_seen: str | None = None
    last_seen: str | None = None
    source_chunk_id: str = ""


class GraphStatsResponse(BaseModel):
    """Graph statistics."""

    entity_count: int = 0
    relation_count: int = 0
    type_distribution: dict[str, int] = Field(default_factory=dict)
    top_entities: list[dict[str, Any]] = Field(default_factory=list)
    connected_components: int = 0


class GraphSearchResponse(BaseModel):
    """Search results for graph entities."""

    entities: list[GraphEntityItem] = Field(default_factory=list)
    query: str = ""
    total: int = 0


class GraphNeighborResponse(BaseModel):
    """Neighbors of a graph entity."""

    entity_id: str
    neighbors: list[GraphEntityItem] = Field(default_factory=list)
    depth: int = 1


class GraphRelationsResponse(BaseModel):
    """Relations of a graph entity."""

    entity_id: str
    relations: list[GraphRelationItem] = Field(default_factory=list)


class GraphSubgraphResponse(BaseModel):
    """A subgraph extraction result."""

    entities: list[GraphEntityItem] = Field(default_factory=list)
    relations: list[GraphRelationItem] = Field(default_factory=list)


class GraphCommunityResponse(BaseModel):
    """Community detection results."""

    communities: list[list[GraphEntityItem]] = Field(default_factory=list)
    total_communities: int = 0


class GraphEntityDetailResponse(BaseModel):
    """Detailed entity info with context string."""

    entity: GraphEntityItem | None = None
    context: str = ""
    found: bool = False


# ── Helpers ──────────────────────────────────────────────────────────────────


def _entity_to_item(entity: Any) -> GraphEntityItem:
    """Convert a GraphEntity dataclass to an API response item."""
    return GraphEntityItem(
        entity_id=entity.entity_id,
        name=entity.name,
        entity_type=str(entity.entity_type),
        aliases=list(entity.aliases) if entity.aliases else [],
        first_seen=entity.first_seen.isoformat() if entity.first_seen else None,
        last_seen=entity.last_seen.isoformat() if entity.last_seen else None,
        mention_count=entity.mention_count,
        confidence=entity.confidence,
        metadata=dict(entity.metadata) if entity.metadata else {},
    )


def _relation_to_item(rel: Any) -> GraphRelationItem:
    """Convert a GraphRelation dataclass to an API response item."""
    return GraphRelationItem(
        source_id=rel.source_id,
        target_id=rel.target_id,
        relation_type=str(rel.relation_type),
        weight=rel.weight,
        confidence=rel.confidence,
        first_seen=rel.first_seen.isoformat() if rel.first_seen else None,
        last_seen=rel.last_seen.isoformat() if rel.last_seen else None,
        source_chunk_id=rel.source_chunk_id,
    )


def _graph_unavailable_msg() -> str:
    return "Knowledge graph is not initialized"


# ── Endpoints ────────────────────────────────────────────────────────────────


@router.get("/stats", response_model=GraphStatsResponse)
async def graph_stats(
    state: AppState = Depends(get_app_state),
) -> GraphStatsResponse:
    """Get knowledge graph statistics."""
    if not state.knowledge_graph:
        return GraphStatsResponse()

    try:
        stats = state.knowledge_graph.get_stats()
        return GraphStatsResponse(
            entity_count=int(stats.get("total_entities", 0)),  # type: ignore[arg-type]
            relation_count=int(stats.get("total_relations", 0)),  # type: ignore[arg-type]
            type_distribution=stats.get("entity_types", {}),  # type: ignore[arg-type]
            top_entities=stats.get("top_entities", []),  # type: ignore[arg-type]
            connected_components=int(stats.get("connected_components", 0)),  # type: ignore[arg-type]
        )
    except Exception as exc:
        log.error("graph_stats_failed", error=str(exc))
        return GraphStatsResponse()


@router.get("/search", response_model=GraphSearchResponse)
async def graph_search(
    query: str = Query(default="", max_length=200),
    entity_type: str | None = Query(default=None),
    limit: int = Query(default=20, ge=1, le=100),
    state: AppState = Depends(get_app_state),
) -> GraphSearchResponse:
    """Search entities by name substring."""
    if not state.knowledge_graph:
        return GraphSearchResponse(query=query)

    try:
        from ara.utils.types import EntityType

        et = EntityType(entity_type) if entity_type else None
        entities = state.knowledge_graph.search_entities(query, entity_type=et, limit=limit)
        items = [_entity_to_item(e) for e in entities]
        return GraphSearchResponse(entities=items, query=query, total=len(items))
    except Exception as exc:
        log.error("graph_search_failed", error=str(exc))
        return GraphSearchResponse(query=query)


@router.get("/entity/{entity_id}", response_model=GraphEntityDetailResponse)
async def graph_entity_detail(
    entity_id: str,
    state: AppState = Depends(get_app_state),
) -> GraphEntityDetailResponse:
    """Get detailed information about an entity including context."""
    if not state.knowledge_graph:
        return GraphEntityDetailResponse()

    try:
        entity = state.knowledge_graph.get_entity(entity_id)
        if not entity:
            return GraphEntityDetailResponse(found=False)

        context = state.knowledge_graph.get_entity_context(entity_id)
        return GraphEntityDetailResponse(
            entity=_entity_to_item(entity),
            context=context,
            found=True,
        )
    except Exception as exc:
        log.error("graph_entity_detail_failed", error=str(exc))
        return GraphEntityDetailResponse()


@router.get("/entity/{entity_id}/neighbors", response_model=GraphNeighborResponse)
async def graph_entity_neighbors(
    entity_id: str,
    depth: int = Query(default=1, ge=1, le=3),
    entity_type: str | None = Query(default=None),
    relation_type: str | None = Query(default=None),
    state: AppState = Depends(get_app_state),
) -> GraphNeighborResponse:
    """Get neighbors of an entity (multi-hop)."""
    if not state.knowledge_graph:
        return GraphNeighborResponse(entity_id=entity_id)

    try:
        from ara.utils.types import EntityType, RelationType

        et = EntityType(entity_type) if entity_type else None
        rt = RelationType(relation_type) if relation_type else None
        neighbors = state.knowledge_graph.get_neighbors(
            entity_id, depth=depth, entity_type=et, relation_type=rt
        )
        items = [_entity_to_item(e) for e in neighbors]
        return GraphNeighborResponse(entity_id=entity_id, neighbors=items, depth=depth)
    except Exception as exc:
        log.error("graph_neighbors_failed", error=str(exc))
        return GraphNeighborResponse(entity_id=entity_id)


@router.get("/entity/{entity_id}/relations", response_model=GraphRelationsResponse)
async def graph_entity_relations(
    entity_id: str,
    direction: str = Query(default="both", pattern="^(both|outgoing|incoming)$"),
    relation_type: str | None = Query(default=None),
    state: AppState = Depends(get_app_state),
) -> GraphRelationsResponse:
    """Get relations of an entity."""
    if not state.knowledge_graph:
        return GraphRelationsResponse(entity_id=entity_id)

    try:
        from ara.utils.types import RelationType

        rt = RelationType(relation_type) if relation_type else None
        relations = state.knowledge_graph.get_relations(
            entity_id, direction=direction, relation_type=rt
        )
        items = [_relation_to_item(r) for r in relations]
        return GraphRelationsResponse(entity_id=entity_id, relations=items)
    except Exception as exc:
        log.error("graph_relations_failed", error=str(exc))
        return GraphRelationsResponse(entity_id=entity_id)


@router.get("/communities", response_model=GraphCommunityResponse)
async def graph_communities(
    state: AppState = Depends(get_app_state),
) -> GraphCommunityResponse:
    """Detect and list graph communities (clusters)."""
    if not state.knowledge_graph:
        return GraphCommunityResponse()

    try:
        communities = state.knowledge_graph.detect_communities()
        community_items = [[_entity_to_item(e) for e in community] for community in communities]
        return GraphCommunityResponse(
            communities=community_items,
            total_communities=len(community_items),
        )
    except Exception as exc:
        log.error("graph_communities_failed", error=str(exc))
        return GraphCommunityResponse()


@router.get("/subgraph", response_model=GraphSubgraphResponse)
async def graph_subgraph(
    entity_ids: str = Query(
        ...,
        description="Comma-separated entity IDs",
        min_length=1,
    ),
    include_neighbors: bool = Query(default=False),
    state: AppState = Depends(get_app_state),
) -> GraphSubgraphResponse:
    """Extract a subgraph for the given entity IDs."""
    if not state.knowledge_graph:
        return GraphSubgraphResponse()

    try:
        ids = [eid.strip() for eid in entity_ids.split(",") if eid.strip()]
        entities, relations = state.knowledge_graph.get_subgraph(
            ids, include_neighbors=include_neighbors
        )
        entity_items = [_entity_to_item(e) for e in entities]
        relation_items = [_relation_to_item(r) for r in relations]
        return GraphSubgraphResponse(entities=entity_items, relations=relation_items)
    except Exception as exc:
        log.error("graph_subgraph_failed", error=str(exc))
        return GraphSubgraphResponse()
