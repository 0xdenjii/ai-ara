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

from ara import __version__
from ara.api.schemas import ErrorResponse, HealthResponse
from ara.config.settings import AraSettings, get_settings
from ara.utils.errors import AraError, VisionError
from ara.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from ara.agents.base import AgentRegistry
    from ara.agents.context_manager import ContextManager
    from ara.agents.orchestrator import Orchestrator
    from ara.agents.router import IntentRouter
    from ara.core.llm import LLMClient
    from ara.memory.conversations import ConversationStore
    from ara.memory.graph import KnowledgeGraph
    from ara.memory.ingestion import IngestionPipeline
    from ara.memory.retrieval import RetrievalPipeline
    from ara.memory.vector_store import VectorStore
    from ara.obsidian.sync import ObsidianSync
    from ara.proactive.mood import MoodTracker
    from ara.proactive.notifications import NotificationQueue
    from ara.proactive.scheduler import ProactiveScheduler
    from ara.tools.mcp import MCPBridge
    from ara.tools.registry import ToolRegistry
    from ara.vision.pipeline import VisionPipeline
    from ara.voice.pipeline import VoicePipeline
    from ara.web.connectivity import ConnectivityManager

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

    # Core
    llm_client: LLMClient | None = None
    orchestrator: Orchestrator | None = None

    # Memory
    vector_store: VectorStore | None = None
    knowledge_graph: KnowledgeGraph | None = None
    conversation_store: ConversationStore | None = None
    retrieval_pipeline: RetrievalPipeline | None = None
    ingestion_pipeline: IngestionPipeline | None = None

    # Agents
    intent_router: IntentRouter | None = None
    agent_registry: AgentRegistry | None = None
    context_manager: ContextManager | None = None

    # Tools
    tool_registry: ToolRegistry | None = None
    mcp_bridge: MCPBridge | None = None

    # Pipelines
    vision_pipeline: VisionPipeline | None = None
    voice_pipeline: VoicePipeline | None = None

    # Proactive
    proactive_scheduler: ProactiveScheduler | None = None
    notification_queue: NotificationQueue | None = None
    mood_tracker: MoodTracker | None = None

    # Obsidian
    obsidian_sync: ObsidianSync | None = None

    # Connectivity
    connectivity_manager: ConnectivityManager | None = None

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


async def _init_memory_subsystems(settings: AraSettings, app_state: AppState) -> None:
    """Initialize memory subsystems: VectorStore, KnowledgeGraph, ConversationStore."""
    # Vector store
    try:
        from ara.memory.vector_store import VectorStore

        vs = VectorStore(settings.memory)
        vs.connect()
        app_state.vector_store = vs
        log.info("vector_store_connected", count=vs.count())
    except Exception as exc:
        log.warning("vector_store_connect_failed", error=str(exc))

    # Knowledge graph
    try:
        from ara.memory.graph import KnowledgeGraph

        graph_dir = settings.data_dir / "graph_db"
        graph_dir.mkdir(parents=True, exist_ok=True)
        kg = KnowledgeGraph(graph_dir / "graph.sqlite")
        kg.load()
        app_state.knowledge_graph = kg
        log.info(
            "knowledge_graph_loaded",
            nodes=kg.node_count,
            edges=kg.edge_count,
        )
    except Exception as exc:
        log.warning("knowledge_graph_load_failed", error=str(exc))

    # Conversation store
    try:
        from ara.memory.conversations import ConversationStore

        cs = ConversationStore(settings.conversation)
        cs.connect()
        app_state.conversation_store = cs
        log.info("conversation_store_connected")
    except Exception as exc:
        log.warning("conversation_store_connect_failed", error=str(exc))


async def _init_retrieval(settings: AraSettings, app_state: AppState) -> None:
    """Initialize embedding model, reranker, retrieval pipeline, and ingestion."""
    if not app_state.vector_store:
        log.warning("retrieval_skipped", reason="vector_store not available")
        return

    try:
        from ara.memory.embeddings import EmbeddingModel
        from ara.memory.reranker import Reranker
        from ara.memory.retrieval import RetrievalPipeline

        embedder = EmbeddingModel(settings.memory)
        embedder.load()

        reranker = Reranker(settings.memory)
        reranker.load()

        rp = RetrievalPipeline(
            settings.memory,
            app_state.vector_store,
            embedder,
            graph=app_state.knowledge_graph,
            reranker=reranker,
        )
        app_state.retrieval_pipeline = rp
        log.info("retrieval_pipeline_ready")
    except Exception as exc:
        log.warning("retrieval_init_failed", error=str(exc))

    # Ingestion pipeline
    try:
        from ara.memory.ingestion import IngestionPipeline

        ip = IngestionPipeline.from_settings(settings.memory)
        ip.initialize()
        app_state.ingestion_pipeline = ip
        log.info("ingestion_pipeline_ready")
    except Exception as exc:
        log.warning("ingestion_init_failed", error=str(exc))


async def _init_tools(settings: AraSettings, app_state: AppState) -> None:
    """Initialize tool registry, register built-in tools, connect MCP."""
    try:
        from ara.tools.registry import ToolRegistry

        registry = ToolRegistry(settings)
        app_state.tool_registry = registry

        # Register built-in tool sets
        from ara.tools.desktop import register_desktop_tools
        from ara.tools.media import register_media_tools
        from ara.tools.system import register_system_tools

        desktop_names = register_desktop_tools(registry, settings)
        system_names = register_system_tools(registry, settings)
        media_names = register_media_tools(registry, settings)
        log.info(
            "tools_registered",
            desktop=len(desktop_names),
            system=len(system_names),
            media=len(media_names),
        )

        # MCP bridge (external tool servers)
        if settings.mcp.enabled:
            try:
                from ara.tools.mcp import MCPBridge, discover_mcp_servers

                bridge = MCPBridge(settings)
                configs = discover_mcp_servers()
                if configs:
                    proxies = await bridge.connect_all(configs)
                    mcp_names = bridge.register_tools(registry, proxies)
                    app_state.mcp_bridge = bridge
                    log.info(
                        "mcp_connected",
                        servers=len(configs),
                        tools=len(mcp_names),
                    )
                else:
                    log.info("mcp_no_servers_found")
            except Exception as exc:
                log.warning("mcp_connect_failed", error=str(exc))

        # Initialize all registered tools
        await registry.connect_all()
        log.info("tool_registry_ready", total=len(registry))
    except Exception as exc:
        log.warning("tools_init_failed", error=str(exc))


async def _init_agents(settings: AraSettings, app_state: AppState) -> None:
    """Initialize agents, router, orchestrator."""
    if not app_state.llm_client:
        log.warning("agents_skipped", reason="llm_client not available")
        return

    llm = app_state.llm_client

    try:
        from ara.agents.base import AgentRegistry
        from ara.agents.context_manager import ContextManager
        from ara.agents.orchestrator import Orchestrator
        from ara.agents.rag_agent import RAGAgent
        from ara.agents.router import IntentRouter

        # Intent router
        router = IntentRouter(llm, settings)
        app_state.intent_router = router

        # Agent registry
        agent_reg = AgentRegistry(settings)

        # RAG agent (primary fallback for most intents)
        if app_state.retrieval_pipeline:
            try:
                from ara.agents.synthesis import ResponseSynthesis

                synth = ResponseSynthesis(settings)
                await synth.connect()

                rag_agent = RAGAgent(settings, llm, app_state.retrieval_pipeline, synth)
                agent_reg.register(rag_agent)
            except Exception as exc:
                log.warning("rag_agent_init_failed", error=str(exc))

        # Memory agent
        if app_state.retrieval_pipeline:
            try:
                from ara.agents.memory_agent import MemoryAgent

                mem_agent = MemoryAgent(
                    settings,
                    llm,
                    app_state.retrieval_pipeline,
                    graph=app_state.knowledge_graph,
                )
                agent_reg.register(mem_agent)
            except Exception as exc:
                log.warning("memory_agent_init_failed", error=str(exc))

        # Tool agent
        try:
            from ara.agents.tool_agent import ToolAgent

            tool_agent = ToolAgent(settings, llm, tool_registry=app_state.tool_registry)
            agent_reg.register(tool_agent)
        except Exception as exc:
            log.warning("tool_agent_init_failed", error=str(exc))

        # Web agents
        try:
            from ara.agents.research_agent import ResearchAgent
            from ara.agents.web_browse_agent import WebBrowseAgent
            from ara.agents.web_search_agent import WebSearchAgent

            search_agent = WebSearchAgent(settings, llm)
            browse_agent = WebBrowseAgent(settings, llm)
            research_agent = ResearchAgent(
                settings,
                llm,
                search_agent=search_agent,
                browse_agent=browse_agent,
            )
            agent_reg.register(search_agent)
            agent_reg.register(browse_agent)
            agent_reg.register(research_agent)
        except Exception as exc:
            log.warning("web_agents_init_failed", error=str(exc))

        # Connect all agents
        await agent_reg.connect_all()
        app_state.agent_registry = agent_reg

        # Context manager
        ctx_mgr = ContextManager(settings, llm)
        app_state.context_manager = ctx_mgr

        # Orchestrator
        orch = Orchestrator(settings, llm, router, agent_reg, ctx_mgr)
        await orch.connect()
        app_state.orchestrator = orch
        log.info("orchestrator_ready", agents=len(agent_reg))
    except Exception as exc:
        log.warning("agents_init_failed", error=str(exc))


async def _init_connectivity(settings: AraSettings, app_state: AppState) -> None:
    """Initialize internet connectivity manager."""
    try:
        from ara.web.connectivity import ConnectivityManager

        cm = ConnectivityManager(settings.internet)
        await cm.start()
        app_state.connectivity_manager = cm
        log.info("connectivity_manager_started", online=cm.is_online)
    except Exception as exc:
        log.warning("connectivity_init_failed", error=str(exc))


async def _init_vision(settings: AraSettings, app_state: AppState) -> None:
    """Initialize vision pipeline."""
    try:
        from ara.vision.pipeline import VisionPipeline

        vp = VisionPipeline(settings, llm_client=app_state.llm_client)
        await vp.start()
        app_state.vision_pipeline = vp
        log.info("vision_pipeline_started")
    except Exception as exc:
        log.warning("vision_init_failed", error=str(exc))


async def _init_voice(settings: AraSettings, app_state: AppState) -> None:
    """Initialize voice pipeline (loads models but does NOT start the main loop)."""
    try:
        from ara.voice.pipeline import VoicePipeline

        voice = VoicePipeline(settings)
        await voice.start()
        app_state.voice_pipeline = voice
        log.info("voice_pipeline_started")
    except Exception as exc:
        log.warning("voice_init_failed", error=str(exc))


async def _init_obsidian(settings: AraSettings, app_state: AppState) -> None:
    """Initialize Obsidian sync + briefing generator."""
    if not settings.obsidian.enabled:
        log.info("obsidian_disabled")
        return

    try:
        from ara.obsidian.client import ObsidianClient
        from ara.obsidian.parser import DailyNoteParser
        from ara.obsidian.sync import ObsidianSync

        client = ObsidianClient(settings.obsidian)
        await client.connect()

        parser = DailyNoteParser()
        sync = ObsidianSync(
            settings.obsidian,
            client,
            parser,
            ingestion=app_state.ingestion_pipeline,
        )

        if settings.obsidian.sync_on_boot:
            try:
                sync_state = await sync.boot_sync()
                log.info(
                    "obsidian_boot_sync_done",
                    synced=sync_state.synced_count,
                    errors=sync_state.error_count,
                )
            except Exception as exc:
                log.warning("obsidian_boot_sync_failed", error=str(exc))

        app_state.obsidian_sync = sync
        log.info("obsidian_sync_ready")
    except Exception as exc:
        log.warning("obsidian_init_failed", error=str(exc))


async def _init_proactive(settings: AraSettings, app_state: AppState) -> None:
    """Initialize proactive scheduler and supporting subsystems."""
    if not settings.proactive.enabled:
        log.info("proactive_disabled")
        return

    try:
        from ara.proactive.habits import HabitDetector
        from ara.proactive.mood import MoodTracker
        from ara.proactive.notifications import NotificationQueue
        from ara.proactive.recap import RecapGenerator
        from ara.proactive.review import ReviewGenerator
        from ara.proactive.scheduler import ProactiveScheduler
        from ara.proactive.suggestions import SuggestionEngine

        # Notification queue
        nq = NotificationQueue(settings.proactive)
        app_state.notification_queue = nq

        # Mood tracker
        mood_db = settings.data_dir / "mood.sqlite"
        mood_db.parent.mkdir(parents=True, exist_ok=True)
        mt = MoodTracker(settings.proactive, db_path=mood_db)
        mt.connect()
        app_state.mood_tracker = mt

        # Habit detector (needs graph)
        habit_detector: HabitDetector | None = None
        if app_state.knowledge_graph:
            habit_detector = HabitDetector(settings.proactive, graph=app_state.knowledge_graph)

        # Suggestion engine (needs graph)
        suggestion_engine: SuggestionEngine | None = None
        if app_state.knowledge_graph:
            suggestion_engine = SuggestionEngine(
                settings.proactive,
                graph=app_state.knowledge_graph,
                mood_tracker=mt,
                habit_detector=habit_detector,
            )

        # Recap generator (needs conversations + graph)
        recap_gen: RecapGenerator | None = None
        if app_state.conversation_store and app_state.knowledge_graph:
            recap_gen = RecapGenerator(
                settings.proactive,
                conversations=app_state.conversation_store,
                graph=app_state.knowledge_graph,
                mood_tracker=mt,
            )

        # Review generator (needs graph + conversations)
        review_gen: ReviewGenerator | None = None
        if app_state.knowledge_graph and app_state.conversation_store:
            review_gen = ReviewGenerator(
                settings.proactive,
                graph=app_state.knowledge_graph,
                conversations=app_state.conversation_store,
                mood_tracker=mt,
            )

        # Briefing generator (needs obsidian sync)
        briefing_gen = None
        if app_state.obsidian_sync:
            try:
                from ara.obsidian.daily_briefing import BriefingGenerator
                from ara.obsidian.habits import ProtocolTracker
                from ara.obsidian.writer import DiaryWriter

                writer = DiaryWriter(settings.obsidian, app_state.obsidian_sync._client)
                protocol_tracker = ProtocolTracker(settings.obsidian, app_state.obsidian_sync)
                briefing_gen = BriefingGenerator(
                    settings.obsidian,
                    app_state.obsidian_sync,
                    writer=writer,
                    conversations=app_state.conversation_store,
                    mood_tracker=mt,
                    protocol_tracker=protocol_tracker,
                )
            except Exception as exc:
                log.warning("briefing_generator_init_failed", error=str(exc))

        # Scheduler
        scheduler = ProactiveScheduler(
            settings.proactive,
            notification_queue=nq,
            suggestion_engine=suggestion_engine,
            mood_tracker=mt,
            habit_detector=habit_detector,
            recap_generator=recap_gen,
            review_generator=review_gen,
            llm_client=app_state.llm_client,
            obsidian_sync=app_state.obsidian_sync,
            obsidian_settings=settings.obsidian if settings.obsidian.enabled else None,
            briefing_generator=briefing_gen,
        )
        await scheduler.start()
        app_state.proactive_scheduler = scheduler
        log.info("proactive_scheduler_started")
    except Exception as exc:
        log.warning("proactive_init_failed", error=str(exc))


async def _shutdown(app_state: AppState) -> None:
    """Shut down all subsystems in reverse initialization order."""
    # Proactive
    if app_state.proactive_scheduler:
        try:
            await app_state.proactive_scheduler.stop()
        except Exception as exc:
            log.warning("proactive_shutdown_error", error=str(exc))

    if app_state.mood_tracker:
        try:
            app_state.mood_tracker.close()
        except Exception as exc:
            log.warning("mood_tracker_close_error", error=str(exc))

    # Voice
    if app_state.voice_pipeline:
        try:
            await app_state.voice_pipeline.stop()
        except Exception as exc:
            log.warning("voice_shutdown_error", error=str(exc))

    # Vision
    if app_state.vision_pipeline:
        try:
            await app_state.vision_pipeline.stop()
        except Exception as exc:
            log.warning("vision_shutdown_error", error=str(exc))

    # Orchestrator
    if app_state.orchestrator:
        try:
            await app_state.orchestrator.close()
        except Exception as exc:
            log.warning("orchestrator_shutdown_error", error=str(exc))

    # Agent registry
    if app_state.agent_registry:
        try:
            await app_state.agent_registry.close_all()
        except Exception as exc:
            log.warning("agent_registry_shutdown_error", error=str(exc))

    # MCP bridge
    if app_state.mcp_bridge:
        try:
            await app_state.mcp_bridge.close_all()
        except Exception as exc:
            log.warning("mcp_shutdown_error", error=str(exc))

    # Tool registry
    if app_state.tool_registry:
        try:
            await app_state.tool_registry.close_all()
        except Exception as exc:
            log.warning("tool_registry_shutdown_error", error=str(exc))

    # Connectivity
    if app_state.connectivity_manager:
        try:
            await app_state.connectivity_manager.stop()
        except Exception as exc:
            log.warning("connectivity_shutdown_error", error=str(exc))

    # Obsidian (client close)
    if app_state.obsidian_sync:
        try:
            await app_state.obsidian_sync._client.close()
        except Exception as exc:
            log.warning("obsidian_shutdown_error", error=str(exc))

    # Ingestion
    if app_state.ingestion_pipeline:
        try:
            app_state.ingestion_pipeline.shutdown()
        except Exception as exc:
            log.warning("ingestion_shutdown_error", error=str(exc))

    # Conversation store
    if app_state.conversation_store:
        try:
            app_state.conversation_store.close()
        except Exception as exc:
            log.warning("conversation_store_close_error", error=str(exc))

    # Knowledge graph
    if app_state.knowledge_graph:
        try:
            app_state.knowledge_graph.save()
            app_state.knowledge_graph.close()
        except Exception as exc:
            log.warning("knowledge_graph_close_error", error=str(exc))

    # Vector store
    if app_state.vector_store:
        try:
            app_state.vector_store.close()
        except Exception as exc:
            log.warning("vector_store_close_error", error=str(exc))

    # LLM client
    if app_state.llm_client:
        try:
            await app_state.llm_client.close()
        except Exception as exc:
            log.warning("llm_shutdown_error", error=str(exc))


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage application lifecycle — initialize and shut down services.

    This is called once at startup and once at shutdown. Heavy model
    loading is done here, not in route handlers.

    Initialization order respects the dependency graph:
    1. LLM client (no deps)
    2. Memory subsystems (no deps)
    3. Retrieval pipeline (needs vector store, embedder)
    4. Tools (no deps)
    5. Connectivity (no deps)
    6. Vision (needs LLM optionally)
    7. Voice (no deps)
    8. Obsidian (needs ingestion optionally)
    9. Agents (needs LLM, retrieval, tools, graph)
    10. Proactive (needs everything)
    """
    settings = get_settings()
    app_state = AppState(settings=settings)

    log.info("api_starting", host=settings.api.host, port=settings.api.port)

    # ── Startup ──────────────────────────────────────────────────────────
    try:
        # 1. LLM client
        try:
            from ara.core.llm import LLMClient

            llm = LLMClient(settings)
            await llm.connect()
            app_state.llm_client = llm
            log.info("llm_connected", model=llm.active_model)
        except Exception as exc:
            log.warning(
                "llm_connect_failed",
                error=str(exc),
                hint="Chat will be unavailable until LLM is connected",
            )

        # 2. Memory subsystems
        await _init_memory_subsystems(settings, app_state)

        # 3. Retrieval pipeline
        await _init_retrieval(settings, app_state)

        # 4. Tools
        await _init_tools(settings, app_state)

        # 5. Connectivity
        await _init_connectivity(settings, app_state)

        # 6. Vision
        await _init_vision(settings, app_state)

        # 7. Voice
        await _init_voice(settings, app_state)

        # 8. Obsidian
        await _init_obsidian(settings, app_state)

        # 9. Agents + orchestrator
        await _init_agents(settings, app_state)

        # 10. Proactive
        await _init_proactive(settings, app_state)

    except Exception as exc:
        log.error("startup_failed", error=str(exc))

    app.state.ara = app_state
    log.info("api_started")

    yield

    # ── Shutdown ─────────────────────────────────────────────────────────
    log.info("api_shutting_down")
    await _shutdown(app_state)
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
        version=__version__,
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
    from ara.api.routes.graph import router as graph_router
    from ara.api.routes.memory import router as memory_router
    from ara.api.routes.models import router as models_router
    from ara.api.routes.settings import router as settings_router
    from ara.api.routes.vision import router as vision_router
    from ara.api.routes.voice import router as voice_router

    app.include_router(chat_router)
    app.include_router(memory_router)
    app.include_router(models_router)
    app.include_router(voice_router)
    app.include_router(settings_router)
    app.include_router(graph_router)
    app.include_router(vision_router)

    # Register WebSocket
    from ara.api.websocket import register_websocket

    register_websocket(app)

    return app


# Module-level app instance for `uvicorn ara.api.app:app`
app = create_app()
