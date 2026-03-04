"""Tests for ARA structured logging."""

from __future__ import annotations

from ara.utils.logging import get_logger, get_trace_id, new_trace_id, set_trace_id, setup_logging


class TestTraceId:
    """Tests for trace ID context management."""

    def test_default_trace_id(self) -> None:
        # Reset by setting a known value, then check default behavior
        assert isinstance(get_trace_id(), str)

    def test_set_and_get_trace_id(self) -> None:
        tid = set_trace_id("abc123")
        assert tid == "abc123"
        assert get_trace_id() == "abc123"

    def test_set_trace_id_generates_when_none(self) -> None:
        tid = set_trace_id()
        assert len(tid) == 12
        assert get_trace_id() == tid

    def test_new_trace_id(self) -> None:
        old_id = get_trace_id()
        new_id = new_trace_id()
        assert new_id != old_id or len(new_id) == 12
        assert get_trace_id() == new_id


class TestSetupLogging:
    """Tests for logging configuration."""

    def test_setup_dev_mode(self) -> None:
        # Should not raise
        setup_logging(log_level="DEBUG", json_output=False)

    def test_setup_json_mode(self) -> None:
        # Should not raise
        setup_logging(log_level="INFO", json_output=True)


class TestGetLogger:
    """Tests for component-bound loggers."""

    def test_get_logger_returns_bound_logger(self) -> None:
        setup_logging(log_level="DEBUG")
        log = get_logger("voice.stt")
        assert log is not None

    def test_logger_with_binds(self) -> None:
        setup_logging(log_level="DEBUG")
        log = get_logger("memory.vector", table="chunks")
        assert log is not None
