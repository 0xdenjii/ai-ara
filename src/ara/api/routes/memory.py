"""Memory route — browsing, search, filter by date/type/emotion.

Endpoints:
    POST /api/memory/search — Hybrid retrieval search
    GET  /api/memory/stats — Memory subsystem statistics
    GET  /api/memory/conversations — List recent conversations
    GET  /api/memory/conversations/{id} — Get conversation detail
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from ara.api.app import AppState, get_app_state
from ara.api.schemas import (
    ConversationDetailResponse,
    ConversationItem,
    ConversationListResponse,
    MemoryChunkItem,
    MemorySearchRequest,
    MemorySearchResponse,
    MemoryStatsResponse,
    MessageItem,
)
from ara.utils.logging import get_logger

log = get_logger("api.memory")

router = APIRouter(prefix="/api/memory", tags=["memory"])


@router.post("/search", response_model=MemorySearchResponse)
async def memory_search(
    request: MemorySearchRequest,
    state: AppState = Depends(get_app_state),
) -> MemorySearchResponse:
    """Search memory using hybrid retrieval (vector + BM25).

    Returns matching memory chunks ranked by relevance.
    """
    if not state.vector_store:
        return MemorySearchResponse(query=request.query)

    try:
        from ara.memory.embeddings import EmbeddingModel

        embedder = EmbeddingModel(state.settings)
        embedder.load()

        query_embedding = embedder.embed_query(request.query)

        results = state.vector_store.search(
            query_embedding,
            top_k=request.top_k,
            source_filter=request.source_filter or None,
        )

        items: list[MemoryChunkItem] = [
            MemoryChunkItem(
                chunk_id=r.get("chunk_id", ""),
                text=r.get("text", ""),
                source=r.get("source", ""),
                source_file=r.get("source_file", ""),
                score=r.get("_distance", 0.0),
                retrieval_method="vector",
            )
            for r in results
        ]

        embedder.unload()

        return MemorySearchResponse(
            results=items,
            query=request.query,
            total_candidates=len(items),
        )

    except Exception as exc:
        log.error("memory_search_failed", error=str(exc))
        return MemorySearchResponse(query=request.query)


@router.get("/stats", response_model=MemoryStatsResponse)
async def memory_stats(
    state: AppState = Depends(get_app_state),
) -> MemoryStatsResponse:
    """Get memory subsystem statistics."""
    stats = MemoryStatsResponse()

    if state.vector_store:
        try:
            stats.total_chunks = state.vector_store.count()
            stats.vector_db_path = str(state.settings.data_dir / "vector_db")
        except Exception as exc:
            log.warning("memory_stats_error", error=str(exc))

    return stats


@router.get("/conversations", response_model=ConversationListResponse)
async def list_conversations(
    limit: int = Query(default=20, ge=1, le=100),
    include_active: bool = Query(default=False),
    state: AppState = Depends(get_app_state),
) -> ConversationListResponse:
    """List recent conversations."""
    try:
        from ara.memory.conversations import ConversationStore

        store = ConversationStore(state.settings.conversation)
        store.connect()

        convos = store.get_recent_conversations(
            limit=limit,
            include_active=include_active,
        )
        store.close()

        items: list[ConversationItem] = [
            ConversationItem(
                conversation_id=c.conversation_id,
                started_at=c.started_at,
                ended_at=c.ended_at,
                turn_count=c.turn_count,
                summary=c.summary,
                is_active=c.is_active,
            )
            for c in convos
        ]

        return ConversationListResponse(conversations=items, total=len(items))

    except Exception as exc:
        log.error("list_conversations_failed", error=str(exc))
        return ConversationListResponse()


@router.get("/conversations/{conversation_id}", response_model=ConversationDetailResponse)
async def get_conversation(
    conversation_id: str,
    state: AppState = Depends(get_app_state),
) -> ConversationDetailResponse:
    """Get a specific conversation with all turns."""
    try:
        from ara.memory.conversations import ConversationStore

        store = ConversationStore(state.settings.conversation)
        store.connect()

        convo = store.get_conversation(conversation_id)
        store.close()

        if not convo:
            return ConversationDetailResponse(conversation_id=conversation_id)

        turns: list[MessageItem] = []
        for t in convo.turns:
            if t.user_message:
                turns.append(
                    MessageItem(role="user", content=t.user_message, timestamp=t.timestamp)
                )
            if t.assistant_message:
                turns.append(
                    MessageItem(
                        role="assistant", content=t.assistant_message, timestamp=t.timestamp
                    )
                )

        return ConversationDetailResponse(
            conversation_id=convo.conversation_id,
            started_at=convo.started_at,
            ended_at=convo.ended_at,
            turns=turns,
            summary=convo.summary,
        )

    except Exception as exc:
        log.error("get_conversation_failed", error=str(exc))
        return ConversationDetailResponse(conversation_id=conversation_id)
