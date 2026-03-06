"""FastAPI application — main entry point for the ARA API.

Provides:
    - CORS-configured FastAPI application
    - Lifespan events (startup/shutdown) for service initialization
    - Dependency injection via AppState for all routes
    - Health check endpoint
    - Error handlers mapping ARA exceptions to HTTP responses

Usage:
    from ara.api.app import create_app
    app = create_app()
    # Run with: uvicorn ara.api.app:app --host 127.0.0.1 --port 8000

    Or for programmatic use:
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ara.api.schemas import ErrorResponse, HealthResponse
from ara.config.settings import AraSettings, get_settings
from ara.utils.errors import AraError, VisionError
from ara.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from ara.agents.orchestrator import Orchestrator
    from ara.core.llm import LLMClient
    from ara.memory.vector_store import VectorStore
    from ara.vision.pipeline import VisionPipeline
    from ara.voice.pipeline import VoicePipeline

log = get_logger("api")


# ── Application State ────────────────────────────────────────────────────────


@dataclass
class AppState:
    """Shared application state injected into all routes.

    Holds references to all ARA subsystems. Populated during lifespan
    startup and cleaned up during shutdown.

    This is NOT a Pydantic model — it's a mutable container that lives
    on app.state for the lifetime of the server process.
    """

    settings: AraSettings = field(default_factory=get_settings)
    llm_client: LLMClient | None = None
    orchestrator: Orchestrator | None = None
    vector_store: VectorStore | None = None
    vision_pipeline: VisionPipeline | None = None
    voice_pipeline: VoicePipeline | None = None
    start_time: float = field(default_factory=time.monotonic)

    @property
    def uptime_s(self) -> float:
        """Seconds since the server started."""
        return time.monotonic() - self.start_time


def get_app_state(request: Request) -> AppState:
    """Extract AppState from the request (dependency injection).

    Usage in routes:
        @router.get("/endpoint")
        async def endpoint(state: AppState = Depends(get_app_state)):
            ...
    """
    return request.app.state.ara  # type: ignore[no-any-return]


# ── Lifespan ─────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage application lifecycle — initialize and shut down services.

    This is called once at startup and once at shutdown. Heavy model
    loading should be done here, not in route handlers.
    """
    settings = get_settings()
    app_state = AppState(settings=settings)

    log.info("api_starting", host=settings.api.host, port=settings.api.port)

    # ── Startup ──────────────────────────────────────────────────────────
    try:
        # LLM client
        from ara.core.llm import LLMClient

        llm = LLMClient(settings)
        try:
            await llm.connect()
            app_state.llm_client = llm
            log.info("llm_connected", model=llm.active_model)
        except Exception as exc:
            log.warning("llm_connect_failed", error=str(exc), hint="Chat will be unavailable")

        # Vector store
        try:
            from ara.memory.vector_store import VectorStore

            vs = VectorStore(settings)
            vs.connect()
            app_state.vector_store = vs
            log.info("vector_store_connected", count=vs.count())
        except Exception as exc:
            log.warning("vector_store_connect_failed", error=str(exc))

    except Exception as exc:
        log.error("startup_failed", error=str(exc))

    app.state.ara = app_state
    log.info("api_started")

    yield

    # ── Shutdown ─────────────────────────────────────────────────────────
    log.info("api_shutting_down")

    if app_state.llm_client:
        await app_state.llm_client.close()
    if app_state.vector_store:
        app_state.vector_store.close()
    if app_state.vision_pipeline:
        await app_state.vision_pipeline.stop()

    log.info("api_stopped")


# ── Error Handlers ───────────────────────────────────────────────────────────


def _ara_error_to_response(exc: AraError) -> JSONResponse:
    """Convert an ARA error to a structured JSON response."""
    from ara.utils.errors import APIError

    status_code = getattr(exc, "status_code", 500)
    if isinstance(exc, APIError):
        status_code = exc.status_code

    error_body = ErrorResponse(
        error=type(exc).__name__,
        detail=str(exc),
        component=exc.component,
        operation=exc.operation,
        status_code=status_code,
    )
    return JSONResponse(status_code=status_code, content=error_body.model_dump())


# ── App Factory ──────────────────────────────────────────────────────────────


def create_app(*, settings: AraSettings | None = None) -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        settings: Override settings (useful for testing).

    Returns:
        Configured FastAPI instance.
    """
    _settings = settings or get_settings()

    app = FastAPI(
        title="ARA API",
        description="Privacy-first personal AI assistant API",
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # ── CORS ─────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_settings.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Error handlers ───────────────────────────────────────────────────
    @app.exception_handler(AraError)
    async def ara_error_handler(_request: Request, exc: AraError) -> JSONResponse:
        log.warning("api_error", error=type(exc).__name__, detail=str(exc))
        return _ara_error_to_response(exc)

    @app.exception_handler(VisionError)
    async def vision_error_handler(_request: Request, exc: VisionError) -> JSONResponse:
        log.warning("vision_api_error", detail=str(exc))
        return _ara_error_to_response(exc)

    # ── Health ───────────────────────────────────────────────────────────
    @app.get("/health", response_model=HealthResponse, tags=["system"])
    async def health(request: Request) -> HealthResponse:
        state = get_app_state(request)
        models: list[str] = []
        if state.llm_client and state.llm_client.is_connected:
            models.append(state.llm_client.active_model)
        return HealthResponse(
            status="ok",
            uptime_s=round(state.uptime_s, 1),
            models_loaded=models,
        )

    # ── Register routes ──────────────────────────────────────────────────
    from ara.api.routes.chat import router as chat_router
    from ara.api.routes.memory import router as memory_router
    from ara.api.routes.models import router as models_router
    from ara.api.routes.settings import router as settings_router
    from ara.api.routes.voice import router as voice_router

    app.include_router(chat_router)
    app.include_router(memory_router)
    app.include_router(models_router)
    app.include_router(voice_router)
    app.include_router(settings_router)

    # Register WebSocket
    from ara.api.websocket import register_websocket

    register_websocket(app)

    return app


# Module-level app instance for `uvicorn ara.api.app:app`
app = create_app()
