"""Memory route — browsing, search, filter by date/type/emotion, ingestion.

Endpoints:
    POST /api/memory/search — Hybrid retrieval search
    GET  /api/memory/stats — Memory subsystem statistics
    GET  /api/memory/conversations — List recent conversations
    GET  /api/memory/conversations/{id} — Get conversation detail
    POST /api/memory/ingest — Ingest a text file into the knowledge base
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile
from pydantic import BaseModel

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


# ── Ingestion response model ────────────────────────────────────────────────


class IngestResponse(BaseModel):
    """Response from file ingestion."""

    success: bool = False
    chunks_created: int = 0
    source_file: str = ""
    elapsed_ms: float = 0.0
    error: str = ""


# Allowed ingestion file types
_ALLOWED_INGEST_EXTENSIONS = frozenset(
    {
        ".md",
        ".txt",
        ".json",
        ".csv",
        ".yaml",
        ".yml",
        ".log",
    }
)

_MAX_INGEST_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


# ── Search ───────────────────────────────────────────────────────────────────


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

        embedder = EmbeddingModel(state.settings.memory)
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


# ── Stats ────────────────────────────────────────────────────────────────────


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

    if state.knowledge_graph:
        try:
            stats.graph_entities = state.knowledge_graph.entity_count
            stats.graph_relations = state.knowledge_graph.relation_count
        except Exception as exc:
            log.warning("graph_stats_error", error=str(exc))

    return stats


# ── Conversations ────────────────────────────────────────────────────────────


@router.get("/conversations", response_model=ConversationListResponse)
async def list_conversations(
    limit: int = Query(default=20, ge=1, le=100),
    include_active: bool = Query(default=False),
    state: AppState = Depends(get_app_state),
) -> ConversationListResponse:
    """List recent conversations."""
    store = state.conversation_store
    if not store:
        # Fallback: create a temporary store (backwards compatibility)
        try:
            from ara.memory.conversations import ConversationStore

            store = ConversationStore(state.settings.conversation)
            store.connect()
        except Exception as exc:
            log.error("list_conversations_failed", error=str(exc))
            return ConversationListResponse()

    try:
        convos = store.get_recent_conversations(
            limit=limit,
            include_active=include_active,
        )

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
    store = state.conversation_store
    if not store:
        try:
            from ara.memory.conversations import ConversationStore

            store = ConversationStore(state.settings.conversation)
            store.connect()
        except Exception as exc:
            log.error("get_conversation_failed", error=str(exc))
            return ConversationDetailResponse(conversation_id=conversation_id)

    try:
        convo = store.get_conversation(conversation_id)

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


# ── Ingestion ────────────────────────────────────────────────────────────────


@router.post("/ingest", response_model=IngestResponse)
async def ingest_file(
    file: UploadFile = File(...),
    source_name: str = Form(default=""),
    state: AppState = Depends(get_app_state),
) -> IngestResponse:
    """Ingest a text file into the knowledge base.

    Supported formats: .md, .txt, .json, .csv, .yaml, .yml, .log
    Maximum size: 10 MB

    The file is chunked, embedded, and stored in the vector database.
    """
    if not state.ingestion_pipeline:
        return IngestResponse(error="Ingestion pipeline is not initialized")

    # Validate file extension
    filename = file.filename or "unknown.txt"
    ext = Path(filename).suffix.lower()
    if ext not in _ALLOWED_INGEST_EXTENSIONS:
        return IngestResponse(
            error=f"Unsupported file type: {ext}. Allowed: {', '.join(sorted(_ALLOWED_INGEST_EXTENSIONS))}",
            source_file=filename,
        )

    # Read file content
    try:
        content = await file.read()
    except Exception as exc:
        return IngestResponse(
            error=f"Failed to read uploaded file: {exc}",
            source_file=filename,
        )

    if len(content) > _MAX_INGEST_FILE_SIZE:
        return IngestResponse(
            error=f"File too large ({len(content)} bytes). Maximum: {_MAX_INGEST_FILE_SIZE} bytes",
            source_file=filename,
        )

    if not content.strip():
        return IngestResponse(error="Uploaded file is empty", source_file=filename)

    # Save to temp file and ingest
    tmp_path: Path | None = None
    try:
        import time

        start = time.monotonic()

        with tempfile.NamedTemporaryFile(
            delete=False, suffix=ext, dir=str(state.settings.data_dir), mode="wb"
        ) as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)

        # Rename to use original filename for better source tracking
        target_name = source_name or filename
        result = state.ingestion_pipeline.ingest_file(tmp_path, force=True)

        elapsed = (time.monotonic() - start) * 1000

        chunks_created = result.chunks_stored if hasattr(result, "chunks_stored") else 0

        return IngestResponse(
            success=True,
            chunks_created=chunks_created,
            source_file=target_name,
            elapsed_ms=round(elapsed, 1),
        )

    except Exception as exc:
        log.error("ingest_failed", error=str(exc), filename=filename)
        return IngestResponse(
            error=f"Ingestion failed: {exc}",
            source_file=filename,
        )
    finally:
        if tmp_path and tmp_path.exists():
            import contextlib

            with contextlib.suppress(OSError):
                tmp_path.unlink()
