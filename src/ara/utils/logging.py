"""Structured logging for ARA using structlog.

Provides trace-ID-aware, structured, JSON-capable logging.
All log entries include component, trace_id, and timestamp.
No private data (diary content, audio, API keys) should ever be logged.
"""

from __future__ import annotations

import sys
import uuid
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from structlog.types import EventDict, WrappedLogger

# ── Trace ID Context ────────────────────────────────────────────────────────
# Each request/conversation/voice-turn gets a unique trace ID for correlation.

_trace_id_var: ContextVar[str] = ContextVar("trace_id", default="no-trace")


def set_trace_id(trace_id: str | None = None) -> str:
    """Set the current trace ID. Generates a new one if not provided.

    Returns the trace ID that was set.
    """
    tid = trace_id or uuid.uuid4().hex[:12]
    _trace_id_var.set(tid)
    return tid


def get_trace_id() -> str:
    """Get the current trace ID from context."""
    return _trace_id_var.get()


def new_trace_id() -> str:
    """Generate and set a fresh trace ID. Returns the new ID."""
    return set_trace_id()


# ── Custom Processors ───────────────────────────────────────────────────────


def add_trace_id(_logger: WrappedLogger, _method: str, event_dict: EventDict) -> EventDict:
    """Inject the current trace ID into every log entry."""
    event_dict["trace_id"] = _trace_id_var.get()
    return event_dict


def add_component(_logger: WrappedLogger, _method: str, event_dict: EventDict) -> EventDict:
    """Ensure a component field exists (defaults to 'ara')."""
    event_dict.setdefault("component", "ara")
    return event_dict


# ── Setup ────────────────────────────────────────────────────────────────────


def setup_logging(
    *,
    log_level: str = "INFO",
    json_output: bool = False,
) -> None:
    """Configure structlog for the application.

    Args:
        log_level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        json_output: If True, render logs as JSON lines (for production).
                     If False, render as colored console output (for development).
    """
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        add_trace_id,
        add_component,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if json_output:
        renderer: structlog.types.Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(
            colors=sys.stderr.isatty(),
        )

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Configure the stdlib root logger to use structlog formatting
    import logging

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))


def get_logger(component: str, **initial_binds: Any) -> structlog.stdlib.BoundLogger:
    """Get a structured logger bound to a specific component.

    Args:
        component: The subsystem name (e.g. "llm", "voice.stt", "memory.vector").
        **initial_binds: Additional key-value pairs to bind to all log entries.

    Returns:
        A bound structlog logger.

    Usage:
        log = get_logger("voice.stt")
        log.info("transcription_complete", audio_duration_s=3.2, latency_ms=48)
    """
    return structlog.get_logger(component=component, **initial_binds)
