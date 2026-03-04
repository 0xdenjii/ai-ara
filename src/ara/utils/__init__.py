"""Shared utilities — logging, error handling, type definitions."""

from ara.utils.errors import AraError
from ara.utils.logging import get_logger, get_trace_id, new_trace_id, set_trace_id, setup_logging

__all__ = [
    "AraError",
    "get_logger",
    "get_trace_id",
    "new_trace_id",
    "set_trace_id",
    "setup_logging",
]
