"""Tests for Phase 6: Desktop Automation & Tools.

Covers: registry, permissions, audit logging, desktop tools, system tools,
media tools, MCP bridge.  All tests are sandboxed — no real desktop
automation, no real process killing, no real MCP servers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ara.config.settings import AraSettings
from ara.tools.permissions import AuditLogger, PermissionChecker, PermissionPolicy
from ara.tools.registry import (
    BaseTool,
    ParameterType,
    ToolParameter,
    ToolRegistry,
    ToolRiskLevel,
    ToolSchema,
)
from ara.utils.errors import DesktopError, ToolExecutionError, ToolPermissionError
from ara.utils.types import ToolCall, ToolResult

if TYPE_CHECKING:
    from pathlib import Path


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def settings() -> AraSettings:
    return AraSettings(env="testing")


@pytest.fixture
def settings_disabled() -> AraSettings:
    s = AraSettings(env="testing")
    s.desktop.enabled = False
    return s


class DummyTool(BaseTool):
    """A test tool for registry tests."""

    def __init__(
        self,
        settings: AraSettings,
        name: str = "dummy",
        category: str = "test",
        risk: ToolRiskLevel = ToolRiskLevel.SAFE,
        output: str = "ok",
    ) -> None:
        super().__init__(settings)
        self._name = name
        self._category = category
        self._risk = risk
        self._output = output

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name=self._name,
            description=f"Test tool: {self._name}",
            parameters=(
                ToolParameter(name="msg", param_type=ParameterType.STRING, description="A message"),
            ),
            risk_level=self._risk,
            category=self._category,
        )

    async def execute(self, parameters: dict[str, Any]) -> str:
        return self._output


class FailingTool(BaseTool):
    """A tool that always raises DesktopError."""

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="failing",
            description="Always fails",
            parameters=(),
            risk_level=ToolRiskLevel.SAFE,
            category="test",
        )

    async def execute(self, parameters: dict[str, Any]) -> str:
        raise DesktopError("intentional failure", operation="test")


class CrashingTool(BaseTool):
    """A tool that raises an unexpected exception."""

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="crashing",
            description="Crashes",
            parameters=(),
            risk_level=ToolRiskLevel.SAFE,
            category="test",
        )

    async def execute(self, parameters: dict[str, Any]) -> str:
        msg = "boom"
        raise RuntimeError(msg)


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL SCHEMA TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestToolSchema:
    """Tests for ToolSchema and ToolParameter."""

    def test_schema_basic(self) -> None:
        schema = ToolSchema(name="test", description="A test tool")
        assert schema.name == "test"
        assert schema.description == "A test tool"
        assert schema.parameters == ()
        assert schema.risk_level == ToolRiskLevel.SAFE
        assert schema.category == "general"

    def test_schema_with_parameters(self) -> None:
        schema = ToolSchema(
            name="click",
            description="Click at position",
            parameters=(
                ToolParameter(name="x", param_type=ParameterType.INTEGER, description="X coord"),
                ToolParameter(name="y", param_type=ParameterType.INTEGER, description="Y coord"),
                ToolParameter(
                    name="button",
                    param_type=ParameterType.STRING,
                    description="Button",
                    required=False,
                    default="left",
                    enum=("left", "right"),
                ),
            ),
            risk_level=ToolRiskLevel.MODERATE,
            category="desktop",
        )
        assert len(schema.parameters) == 3
        assert schema.parameters[2].required is False
        assert schema.parameters[2].default == "left"

    def test_schema_to_prompt_text_no_params(self) -> None:
        schema = ToolSchema(name="screenshot", description="Takes a screenshot")
        text = schema.to_prompt_text()
        assert "screenshot" in text
        assert "Takes a screenshot" in text
        assert "Parameters: {}" in text

    def test_schema_to_prompt_text_with_params(self) -> None:
        schema = ToolSchema(
            name="click",
            description="Click at position",
            parameters=(
                ToolParameter(name="x", param_type=ParameterType.INTEGER, description="X"),
                ToolParameter(
                    name="button",
                    param_type=ParameterType.STRING,
                    description="btn",
                    required=False,
                    default="left",
                    enum=("left", "right"),
                ),
            ),
        )
        text = schema.to_prompt_text()
        assert '"x"' in text
        assert '"button"' in text
        assert "optional" in text
        assert "left|right" in text

    def test_validate_parameters_valid(self) -> None:
        schema = ToolSchema(
            name="test",
            description="Test",
            parameters=(
                ToolParameter(name="name", param_type=ParameterType.STRING, description="name"),
            ),
        )
        errors = schema.validate_parameters({"name": "hello"})
        assert errors == []

    def test_validate_parameters_missing_required(self) -> None:
        schema = ToolSchema(
            name="test",
            description="Test",
            parameters=(
                ToolParameter(name="name", param_type=ParameterType.STRING, description="name"),
            ),
        )
        errors = schema.validate_parameters({})
        assert len(errors) == 1
        assert "Missing required parameter" in errors[0]

    def test_validate_parameters_unknown(self) -> None:
        schema = ToolSchema(
            name="test",
            description="Test",
            parameters=(),
        )
        errors = schema.validate_parameters({"extra": "value"})
        assert len(errors) == 1
        assert "Unknown parameter" in errors[0]

    def test_validate_parameters_enum_violation(self) -> None:
        schema = ToolSchema(
            name="test",
            description="Test",
            parameters=(
                ToolParameter(
                    name="action",
                    param_type=ParameterType.STRING,
                    description="action",
                    enum=("play", "pause"),
                ),
            ),
        )
        errors = schema.validate_parameters({"action": "explode"})
        assert len(errors) == 1
        assert "must be one of" in errors[0]

    def test_validate_parameters_enum_valid(self) -> None:
        schema = ToolSchema(
            name="test",
            description="Test",
            parameters=(
                ToolParameter(
                    name="action",
                    param_type=ParameterType.STRING,
                    description="action",
                    enum=("play", "pause"),
                ),
            ),
        )
        errors = schema.validate_parameters({"action": "play"})
        assert errors == []

    def test_risk_level_values(self) -> None:
        assert ToolRiskLevel.SAFE == "safe"
        assert ToolRiskLevel.MODERATE == "moderate"
        assert ToolRiskLevel.DANGEROUS == "dangerous"

    def test_parameter_type_values(self) -> None:
        assert ParameterType.STRING == "string"
        assert ParameterType.INTEGER == "integer"
        assert ParameterType.FLOAT == "float"
        assert ParameterType.BOOLEAN == "boolean"


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL REGISTRY TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestToolRegistry:
    """Tests for ToolRegistry."""

    def test_register_tool(self, settings: AraSettings) -> None:
        registry = ToolRegistry(settings)
        tool = DummyTool(settings)
        registry.register(tool)
        assert registry.tool_count == 1
        assert "dummy" in registry.tool_names

    def test_register_replaces(self, settings: AraSettings) -> None:
        registry = ToolRegistry(settings)
        registry.register(DummyTool(settings, name="dup", output="first"))
        registry.register(DummyTool(settings, name="dup", output="second"))
        assert registry.tool_count == 1

    def test_unregister(self, settings: AraSettings) -> None:
        registry = ToolRegistry(settings)
        registry.register(DummyTool(settings))
        assert registry.unregister("dummy") is True
        assert registry.tool_count == 0
        assert registry.unregister("nonexistent") is False

    def test_get_tool(self, settings: AraSettings) -> None:
        registry = ToolRegistry(settings)
        tool = DummyTool(settings)
        registry.register(tool)
        assert registry.get_tool("dummy") is tool
        assert registry.get_tool("missing") is None

    def test_get_tools_by_category(self, settings: AraSettings) -> None:
        registry = ToolRegistry(settings)
        registry.register(DummyTool(settings, name="a", category="desktop"))
        registry.register(DummyTool(settings, name="b", category="desktop"))
        registry.register(DummyTool(settings, name="c", category="system"))
        assert len(registry.get_tools_by_category("desktop")) == 2
        assert len(registry.get_tools_by_category("system")) == 1
        assert len(registry.get_tools_by_category("media")) == 0

    @pytest.mark.asyncio
    async def test_execute_success(self, settings: AraSettings) -> None:
        registry = ToolRegistry(settings)
        registry.register(DummyTool(settings, output="hello world"))
        result = await registry.execute(ToolCall(tool_name="dummy", parameters={"msg": "hi"}))
        assert result.success is True
        assert result.output == "hello world"
        assert result.execution_time_ms >= 0.0

    @pytest.mark.asyncio
    async def test_execute_not_found(self, settings: AraSettings) -> None:
        registry = ToolRegistry(settings)
        with pytest.raises(ToolExecutionError, match="Tool not found"):
            await registry.execute(ToolCall(tool_name="missing", parameters={}))

    @pytest.mark.asyncio
    async def test_execute_validation_error(self, settings: AraSettings) -> None:
        registry = ToolRegistry(settings)
        registry.register(DummyTool(settings))
        with pytest.raises(ToolExecutionError, match="Missing required parameter"):
            await registry.execute(ToolCall(tool_name="dummy", parameters={}))

    @pytest.mark.asyncio
    async def test_execute_desktop_error(self, settings: AraSettings) -> None:
        registry = ToolRegistry(settings)
        registry.register(FailingTool(settings))
        result = await registry.execute(ToolCall(tool_name="failing", parameters={}))
        assert result.success is False
        assert "intentional failure" in (result.error or "")

    @pytest.mark.asyncio
    async def test_execute_unexpected_error(self, settings: AraSettings) -> None:
        registry = ToolRegistry(settings)
        registry.register(CrashingTool(settings))
        result = await registry.execute(ToolCall(tool_name="crashing", parameters={}))
        assert result.success is False
        assert "RuntimeError" in (result.error or "")

    @pytest.mark.asyncio
    async def test_execute_desktop_disabled(self, settings_disabled: AraSettings) -> None:
        registry = ToolRegistry(settings_disabled)
        registry.register(DummyTool(settings_disabled))
        with pytest.raises(ToolPermissionError):
            await registry.execute(ToolCall(tool_name="dummy", parameters={"msg": "test"}))

    @pytest.mark.asyncio
    async def test_execute_dangerous_blocked(self, settings: AraSettings) -> None:
        registry = ToolRegistry(settings)
        registry.register(DummyTool(settings, name="danger", risk=ToolRiskLevel.DANGEROUS))
        with pytest.raises(ToolPermissionError):
            await registry.execute(ToolCall(tool_name="danger", parameters={"msg": "test"}))

    @pytest.mark.asyncio
    async def test_execute_moderate_allowed(self, settings: AraSettings) -> None:
        registry = ToolRegistry(settings)
        registry.register(DummyTool(settings, name="mod", risk=ToolRiskLevel.MODERATE))
        result = await registry.execute(ToolCall(tool_name="mod", parameters={"msg": "test"}))
        assert result.success is True

    def test_generate_tool_prompt_empty(self, settings: AraSettings) -> None:
        registry = ToolRegistry(settings)
        assert registry.generate_tool_prompt() == "No tools available."

    def test_generate_tool_prompt_with_tools(self, settings: AraSettings) -> None:
        registry = ToolRegistry(settings)
        registry.register(DummyTool(settings, name="alpha", category="desktop"))
        registry.register(DummyTool(settings, name="beta", category="system"))
        prompt = registry.generate_tool_prompt()
        assert "Available tools:" in prompt
        assert "[DESKTOP]" in prompt
        assert "[SYSTEM]" in prompt
        assert "alpha" in prompt
        assert "beta" in prompt

    @pytest.mark.asyncio
    async def test_connect_and_close_all(self, settings: AraSettings) -> None:
        registry = ToolRegistry(settings)
        tool = DummyTool(settings)
        tool.connect = AsyncMock()  # type: ignore[method-assign]
        tool.close = AsyncMock()  # type: ignore[method-assign]
        registry.register(tool)
        await registry.connect_all()
        tool.connect.assert_called_once()
        await registry.close_all()
        tool.close.assert_called_once()

    def test_get_stats(self, settings: AraSettings) -> None:
        registry = ToolRegistry(settings)
        registry.register(DummyTool(settings))
        stats = registry.get_stats()
        assert stats["tool_count"] == 1
        assert stats["execution_count"] == 0
        assert "dummy" in stats["tools"]

    @pytest.mark.asyncio
    async def test_execution_count_tracked(self, settings: AraSettings) -> None:
        registry = ToolRegistry(settings)
        registry.register(DummyTool(settings))
        await registry.execute(ToolCall(tool_name="dummy", parameters={"msg": "a"}))
        await registry.execute(ToolCall(tool_name="dummy", parameters={"msg": "b"}))
        assert registry.get_stats()["execution_count"] == 2

    @pytest.mark.asyncio
    async def test_audit_logger_called(self, settings: AraSettings) -> None:
        audit_fn = MagicMock()
        registry = ToolRegistry(settings, audit_logger=audit_fn)
        registry.register(DummyTool(settings))
        await registry.execute(ToolCall(tool_name="dummy", parameters={"msg": "hi"}))
        audit_fn.assert_called_once()
        call_args = audit_fn.call_args
        assert isinstance(call_args[0][0], ToolCall)
        assert isinstance(call_args[0][1], ToolResult)

    def test_load_plugin_invalid_module(self, settings: AraSettings) -> None:
        registry = ToolRegistry(settings)
        with pytest.raises(ToolExecutionError, match="Could not import"):
            registry.load_plugin("nonexistent.module", "SomeTool")

    def test_load_plugin_missing_class(self, settings: AraSettings) -> None:
        registry = ToolRegistry(settings)
        with pytest.raises(ToolExecutionError, match="not found in module"):
            registry.load_plugin("ara.tools.registry", "NonexistentTool")

    def test_load_plugin_not_base_tool(self, settings: AraSettings) -> None:
        registry = ToolRegistry(settings)
        with pytest.raises(ToolExecutionError, match="not a subclass"):
            registry.load_plugin("ara.tools.registry", "ToolSchema")


# ═══════════════════════════════════════════════════════════════════════════════
# PERMISSION TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestPermissionPolicy:
    """Tests for PermissionPolicy."""

    def test_default_policy(self) -> None:
        policy = PermissionPolicy.default()
        assert policy.allow_dangerous is False
        assert policy.require_confirmation_for_moderate is True
        assert len(policy.allowed_tools) == 0
        assert len(policy.denied_tools) == 0

    def test_locked_down_policy(self) -> None:
        policy = PermissionPolicy.locked_down()
        assert "run_command" in policy.denied_tools
        assert "file_write" in policy.denied_tools
        assert policy.max_calls_per_minute == 10

    def test_custom_policy(self) -> None:
        policy = PermissionPolicy(
            allowed_tools={"screenshot", "list_windows"},
            denied_tools=set(),
            allow_dangerous=False,
        )
        assert len(policy.allowed_tools) == 2


class TestPermissionChecker:
    """Tests for PermissionChecker."""

    def test_allow_safe_tool(self, settings: AraSettings) -> None:
        checker = PermissionChecker(settings, PermissionPolicy.default())
        assert checker(tool_name="screenshot", risk_level=ToolRiskLevel.SAFE) is True

    def test_deny_dangerous_tool(self, settings: AraSettings) -> None:
        checker = PermissionChecker(settings, PermissionPolicy.default())
        assert checker(tool_name="run_command", risk_level=ToolRiskLevel.DANGEROUS) is False

    def test_allow_dangerous_with_policy(self, settings: AraSettings) -> None:
        policy = PermissionPolicy(allow_dangerous=True)
        checker = PermissionChecker(settings, policy)
        assert checker(tool_name="run_command", risk_level=ToolRiskLevel.DANGEROUS) is True

    def test_deny_by_denylist(self, settings: AraSettings) -> None:
        policy = PermissionPolicy(denied_tools={"screenshot"})
        checker = PermissionChecker(settings, policy)
        assert checker(tool_name="screenshot", risk_level=ToolRiskLevel.SAFE) is False

    def test_deny_by_whitelist(self, settings: AraSettings) -> None:
        policy = PermissionPolicy(allowed_tools={"screenshot"})
        checker = PermissionChecker(settings, policy)
        assert checker(tool_name="screenshot", risk_level=ToolRiskLevel.SAFE) is True
        assert checker(tool_name="other_tool", risk_level=ToolRiskLevel.SAFE) is False

    def test_deny_by_category(self, settings: AraSettings) -> None:
        policy = PermissionPolicy(denied_categories={"system"})
        checker = PermissionChecker(settings, policy)
        checker.set_tool_category("run_command", "system")
        assert checker(tool_name="run_command", risk_level=ToolRiskLevel.SAFE) is False

    def test_allow_by_category_whitelist(self, settings: AraSettings) -> None:
        policy = PermissionPolicy(allowed_categories={"desktop"})
        checker = PermissionChecker(settings, policy)
        checker.set_tool_category("screenshot", "desktop")
        checker.set_tool_category("run_command", "system")
        assert checker(tool_name="screenshot", risk_level=ToolRiskLevel.SAFE) is True
        assert checker(tool_name="run_command", risk_level=ToolRiskLevel.SAFE) is False

    def test_desktop_disabled(self, settings_disabled: AraSettings) -> None:
        checker = PermissionChecker(settings_disabled, PermissionPolicy.default())
        assert checker(tool_name="screenshot", risk_level=ToolRiskLevel.SAFE) is False

    def test_rate_limiting(self, settings: AraSettings) -> None:
        policy = PermissionPolicy(max_calls_per_minute=3)
        checker = PermissionChecker(settings, policy)
        assert checker("t1", ToolRiskLevel.SAFE) is True
        assert checker("t2", ToolRiskLevel.SAFE) is True
        assert checker("t3", ToolRiskLevel.SAFE) is True
        assert checker("t4", ToolRiskLevel.SAFE) is False  # Rate limited

    def test_update_policy(self, settings: AraSettings) -> None:
        checker = PermissionChecker(settings, PermissionPolicy.default())
        assert checker.policy.allow_dangerous is False
        new_policy = PermissionPolicy(allow_dangerous=True)
        checker.update_policy(new_policy)
        assert checker.policy.allow_dangerous is True


# ═══════════════════════════════════════════════════════════════════════════════
# AUDIT LOGGER TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestAuditLogger:
    """Tests for AuditLogger (SQLite-backed)."""

    @pytest.mark.asyncio
    async def test_connect_creates_db(self, tmp_path: Path) -> None:
        db_path = tmp_path / "audit.sqlite"
        logger = AuditLogger(db_path)
        await logger.connect()
        assert db_path.exists()
        await logger.close()

    @pytest.mark.asyncio
    async def test_log_entry(self, tmp_path: Path) -> None:
        db_path = tmp_path / "audit.sqlite"
        logger = AuditLogger(db_path)
        await logger.connect()

        tc = ToolCall(tool_name="screenshot", parameters={})
        tr = ToolResult(
            call_id=tc.call_id,
            tool_name="screenshot",
            success=True,
            output="ok",
            execution_time_ms=42.5,
        )
        logger.log(tc, tr)

        entries = logger.query_recent(limit=10)
        assert len(entries) == 1
        assert entries[0]["tool_name"] == "screenshot"
        assert entries[0]["success"] is True
        assert entries[0]["execution_time_ms"] == 42.5

        await logger.close()

    @pytest.mark.asyncio
    async def test_log_failure(self, tmp_path: Path) -> None:
        db_path = tmp_path / "audit.sqlite"
        logger = AuditLogger(db_path)
        await logger.connect()

        tc = ToolCall(tool_name="run_command", parameters={"command": "bad"})
        tr = ToolResult(
            call_id=tc.call_id,
            tool_name="run_command",
            success=False,
            output="",
            error="denied",
            execution_time_ms=1.0,
        )
        logger.log(tc, tr)

        entries = logger.query_recent()
        assert len(entries) == 1
        assert entries[0]["success"] is False
        assert entries[0]["error"] == "denied"

        await logger.close()

    @pytest.mark.asyncio
    async def test_query_by_tool(self, tmp_path: Path) -> None:
        db_path = tmp_path / "audit.sqlite"
        logger = AuditLogger(db_path)
        await logger.connect()

        for name in ["screenshot", "click", "screenshot", "click", "click"]:
            tc = ToolCall(tool_name=name, parameters={})
            tr = ToolResult(call_id=tc.call_id, tool_name=name, success=True, output="ok")
            logger.log(tc, tr)

        ss_entries = logger.query_by_tool("screenshot")
        assert len(ss_entries) == 2

        click_entries = logger.query_by_tool("click")
        assert len(click_entries) == 3

        await logger.close()

    @pytest.mark.asyncio
    async def test_get_stats(self, tmp_path: Path) -> None:
        db_path = tmp_path / "audit.sqlite"
        logger = AuditLogger(db_path)
        await logger.connect()

        # 2 successes, 1 failure
        for success in [True, True, False]:
            tc = ToolCall(tool_name="test", parameters={})
            tr = ToolResult(call_id=tc.call_id, tool_name="test", success=success, output="")
            logger.log(tc, tr)

        stats = logger.get_stats()
        assert stats["total_calls"] == 3
        assert stats["success_count"] == 2
        assert stats["error_count"] == 1
        assert "test" in stats["by_tool"]

        await logger.close()

    @pytest.mark.asyncio
    async def test_clear(self, tmp_path: Path) -> None:
        db_path = tmp_path / "audit.sqlite"
        logger = AuditLogger(db_path)
        await logger.connect()

        tc = ToolCall(tool_name="test", parameters={})
        tr = ToolResult(call_id=tc.call_id, tool_name="test", success=True, output="ok")
        logger.log(tc, tr)
        assert len(logger.query_recent()) == 1

        logger.clear()
        assert len(logger.query_recent()) == 0

        await logger.close()

    @pytest.mark.asyncio
    async def test_log_without_connect(self, tmp_path: Path) -> None:
        logger = AuditLogger(tmp_path / "audit.sqlite")
        # Should not raise, just warn
        tc = ToolCall(tool_name="test", parameters={})
        tr = ToolResult(call_id=tc.call_id, tool_name="test", success=True, output="ok")
        logger.log(tc, tr)

    @pytest.mark.asyncio
    async def test_query_without_connect(self, tmp_path: Path) -> None:
        logger = AuditLogger(tmp_path / "audit.sqlite")
        assert logger.query_recent() == []
        assert logger.query_by_tool("test") == []
        assert logger.get_stats()["total_calls"] == 0

    @pytest.mark.asyncio
    async def test_parameters_stored_as_json(self, tmp_path: Path) -> None:
        db_path = tmp_path / "audit.sqlite"
        logger = AuditLogger(db_path)
        await logger.connect()

        tc = ToolCall(tool_name="click", parameters={"x": 100, "y": 200, "button": "left"})
        tr = ToolResult(call_id=tc.call_id, tool_name="click", success=True, output="ok")
        logger.log(tc, tr)

        entries = logger.query_recent()
        assert entries[0]["parameters"]["x"] == 100
        assert entries[0]["parameters"]["button"] == "left"

        await logger.close()


# ═══════════════════════════════════════════════════════════════════════════════
# DESKTOP TOOL TESTS (Mocked)
# ═══════════════════════════════════════════════════════════════════════════════


class TestDesktopTools:
    """Tests for desktop tools — all actual operations mocked."""

    @pytest.mark.asyncio
    async def test_open_application_known(self, settings: AraSettings) -> None:
        from ara.tools.desktop import OpenApplicationTool

        tool = OpenApplicationTool(settings)
        assert tool.schema.name == "open_application"
        assert tool.schema.category == "desktop"

        with patch("ara.tools.desktop.subprocess.Popen") as mock_popen:
            result = await tool.execute({"name": "notepad"})
            mock_popen.assert_called_once()
            assert "notepad" in result.lower()

    @pytest.mark.asyncio
    async def test_open_application_uri(self, settings: AraSettings) -> None:
        from ara.tools.desktop import OpenApplicationTool

        tool = OpenApplicationTool(settings)

        with patch("ara.tools.desktop.os.startfile") as mock_startfile:
            result = await tool.execute({"name": "settings"})
            mock_startfile.assert_called_once_with("ms-settings:")
            assert "settings" in result.lower()

    @pytest.mark.asyncio
    async def test_open_application_fallback(self, settings: AraSettings) -> None:
        from ara.tools.desktop import OpenApplicationTool

        tool = OpenApplicationTool(settings)

        with patch("ara.tools.desktop.os.startfile") as mock_startfile:
            await tool.execute({"name": "some_random_app"})
            mock_startfile.assert_called_once_with("some_random_app")

    @pytest.mark.asyncio
    async def test_close_application(self, settings: AraSettings) -> None:
        from ara.tools.desktop import CloseApplicationTool

        tool = CloseApplicationTool(settings)
        assert tool.schema.risk_level == ToolRiskLevel.MODERATE

        mock_proc = MagicMock()
        mock_proc.info = {"name": "notepad.exe", "pid": 1234}

        with patch("ara.tools.desktop.psutil.process_iter", return_value=[mock_proc]):
            result = await tool.execute({"name": "notepad"})
            mock_proc.terminate.assert_called_once()
            assert "1" in result

    @pytest.mark.asyncio
    async def test_close_application_not_found(self, settings: AraSettings) -> None:
        from ara.tools.desktop import CloseApplicationTool

        tool = CloseApplicationTool(settings)

        with patch("ara.tools.desktop.psutil.process_iter", return_value=[]):
            result = await tool.execute({"name": "nonexistent"})
            assert "No running process" in result

    @pytest.mark.asyncio
    async def test_type_text(self, settings: AraSettings) -> None:
        from ara.tools.desktop import TypeTextTool

        tool = TypeTextTool(settings)

        with patch("ara.tools.desktop.pyautogui.typewrite") as mock_type:
            result = await tool.execute({"text": "hello"})
            mock_type.assert_called_once_with("hello", interval=0.0)
            assert "5 characters" in result

    @pytest.mark.asyncio
    async def test_type_text_too_long(self, settings: AraSettings) -> None:
        from ara.tools.desktop import TypeTextTool

        tool = TypeTextTool(settings)
        with pytest.raises(DesktopError, match="Text too long"):
            await tool.execute({"text": "x" * 20000})

    @pytest.mark.asyncio
    async def test_press_key(self, settings: AraSettings) -> None:
        from ara.tools.desktop import PressKeyTool

        tool = PressKeyTool(settings)

        with patch("ara.tools.desktop.pyautogui.hotkey") as mock_hotkey:
            result = await tool.execute({"keys": "ctrl+c"})
            mock_hotkey.assert_called_once_with("ctrl", "c")
            assert "ctrl+c" in result

    @pytest.mark.asyncio
    async def test_press_key_alias(self, settings: AraSettings) -> None:
        from ara.tools.desktop import PressKeyTool

        tool = PressKeyTool(settings)

        with patch("ara.tools.desktop.pyautogui.hotkey") as mock_hotkey:
            await tool.execute({"keys": "control+esc"})
            mock_hotkey.assert_called_once_with("ctrl", "escape")

    @pytest.mark.asyncio
    async def test_click(self, settings: AraSettings) -> None:
        from ara.tools.desktop import ClickTool

        tool = ClickTool(settings)

        with (
            patch("ara.tools.desktop.pyautogui.click") as mock_click,
            patch("ara.tools.desktop.pyautogui.size", return_value=(1920, 1080)),
        ):
            result = await tool.execute({"x": 100, "y": 200})
            mock_click.assert_called_once_with(x=100, y=200, button="left", clicks=1)
            assert "(100, 200)" in result

    @pytest.mark.asyncio
    async def test_click_out_of_bounds(self, settings: AraSettings) -> None:
        from ara.tools.desktop import ClickTool

        tool = ClickTool(settings)

        with (
            patch("ara.tools.desktop.pyautogui.size", return_value=(1920, 1080)),
            pytest.raises(DesktopError, match="out of screen bounds"),
        ):
            await tool.execute({"x": 3000, "y": 200})

    @pytest.mark.asyncio
    async def test_screenshot(self, settings: AraSettings, tmp_path: Path) -> None:
        from ara.tools.desktop import ScreenshotTool

        settings.data_dir = tmp_path
        tool = ScreenshotTool(settings)

        mock_img = MagicMock()
        mock_img.width = 1920
        mock_img.height = 1080

        with patch("ara.tools.desktop.pyautogui.screenshot", return_value=mock_img):
            result = await tool.execute({})
            mock_img.save.assert_called_once()
            assert "1920x1080" in result

    @pytest.mark.asyncio
    async def test_open_url(self, settings: AraSettings) -> None:
        from ara.tools.desktop import OpenURLTool

        tool = OpenURLTool(settings)

        with patch("webbrowser.open") as mock_open:
            result = await tool.execute({"url": "https://example.com"})
            mock_open.assert_called_once_with("https://example.com")
            assert "example.com" in result

    @pytest.mark.asyncio
    async def test_open_url_adds_https(self, settings: AraSettings) -> None:
        from ara.tools.desktop import OpenURLTool

        tool = OpenURLTool(settings)

        with patch("webbrowser.open") as mock_open:
            await tool.execute({"url": "example.com"})
            mock_open.assert_called_once_with("https://example.com")

    @pytest.mark.asyncio
    async def test_list_windows_fallback(self, settings: AraSettings) -> None:
        from ara.tools.desktop import ListWindowsTool

        tool = ListWindowsTool(settings)

        # Mock psutil fallback path (pywinauto import fails)
        mock_proc = MagicMock()
        mock_proc.info = {"name": "chrome.exe"}

        with (
            patch("ara.tools.desktop.psutil.process_iter", return_value=[mock_proc]),
            patch.dict("sys.modules", {"pywinauto": None}),
            patch("builtins.__import__", side_effect=ImportError),
        ):
            result = await tool.execute({})
            assert "chrome.exe" in result or "Running processes" in result

    @pytest.mark.asyncio
    async def test_list_processes(self, settings: AraSettings) -> None:
        from ara.tools.desktop import ListProcessesTool

        tool = ListProcessesTool(settings)

        mock_mem = MagicMock()
        mock_mem.rss = 100 * 1024 * 1024  # 100MB

        mock_proc = MagicMock()
        mock_proc.info = {
            "name": "chrome.exe",
            "pid": 1234,
            "cpu_percent": 5.0,
            "memory_info": mock_mem,
        }

        with patch("ara.tools.desktop.psutil.process_iter", return_value=[mock_proc]):
            result = await tool.execute({"sort_by": "memory", "limit": 5})
            assert "chrome.exe" in result
            assert "100.0MB" in result

    @pytest.mark.asyncio
    async def test_system_info(self, settings: AraSettings) -> None:
        from ara.tools.desktop import SystemInfoTool

        tool = SystemInfoTool(settings)

        with (
            patch("ara.tools.desktop.psutil.cpu_percent", return_value=25.0),
            patch("ara.tools.desktop.psutil.cpu_count", return_value=16),
            patch("ara.tools.desktop.psutil.virtual_memory") as mock_mem,
            patch("ara.tools.desktop.psutil.disk_usage") as mock_disk,
            patch("ara.tools.desktop.psutil.boot_time", return_value=0.0),
            patch("ara.tools.desktop.psutil.sensors_battery", return_value=None),
            patch("ara.tools.desktop.time.time", return_value=3600.0),
        ):
            mock_mem.return_value = MagicMock(percent=50.0, used=32 * 1024**3, total=64 * 1024**3)
            mock_disk.return_value = MagicMock(
                percent=60.0, used=500 * 1024**3, total=1000 * 1024**3
            )

            result = await tool.execute({})
            assert "25.0%" in result
            assert "16 cores" in result
            assert "50.0%" in result

    @pytest.mark.asyncio
    async def test_focus_window(self, settings: AraSettings) -> None:
        from ara.tools.desktop import FocusWindowTool

        tool = FocusWindowTool(settings)
        assert tool.schema.name == "focus_window"

    def test_register_desktop_tools(self, settings: AraSettings) -> None:
        from ara.tools.desktop import register_desktop_tools

        registry = ToolRegistry(settings)
        names = register_desktop_tools(registry, settings)
        assert len(names) == 11
        assert "open_application" in names
        assert "screenshot" in names
        assert "system_info" in names


# ═══════════════════════════════════════════════════════════════════════════════
# SYSTEM TOOL TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestSystemTools:
    """Tests for system tools — command execution, file I/O."""

    def test_command_whitelist_allowed(self) -> None:
        from ara.tools.system import _is_command_allowed

        assert _is_command_allowed("dir") is True
        assert _is_command_allowed("Get-Process") is True
        assert _is_command_allowed("ipconfig /all") is True
        assert _is_command_allowed("git status") is True
        assert _is_command_allowed("echo hello") is True

    def test_command_whitelist_blocked(self) -> None:
        from ara.tools.system import _is_command_allowed

        assert _is_command_allowed("rm -rf /") is False
        assert _is_command_allowed("format C:") is False
        assert _is_command_allowed("shutdown /s") is False
        assert _is_command_allowed("reg delete HKLM") is False
        assert _is_command_allowed("random_dangerous_thing") is False

    def test_command_blacklist_overrides_whitelist(self) -> None:
        from ara.tools.system import _is_command_allowed

        # "del /f" is in blacklist but starts with "del" which is not whitelisted anyway
        assert _is_command_allowed("del /f something") is False

    @pytest.mark.asyncio
    async def test_run_command_blocked(self, settings: AraSettings) -> None:
        from ara.tools.system import RunCommandTool

        tool = RunCommandTool(settings)
        with pytest.raises(DesktopError, match="not allowed"):
            await tool.execute({"command": "rm -rf /"})

    @pytest.mark.asyncio
    async def test_run_command_allowed(self, settings: AraSettings) -> None:
        from ara.tools.system import RunCommandTool

        tool = RunCommandTool(settings)

        mock_process = AsyncMock()
        mock_process.communicate = AsyncMock(return_value=(b"test output", b""))
        mock_process.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            result = await tool.execute({"command": "echo hello"})
            assert "test output" in result

    @pytest.mark.asyncio
    async def test_run_command_failure(self, settings: AraSettings) -> None:
        from ara.tools.system import RunCommandTool

        tool = RunCommandTool(settings)

        mock_process = AsyncMock()
        mock_process.communicate = AsyncMock(return_value=(b"", b"error msg"))
        mock_process.returncode = 1

        with patch("asyncio.create_subprocess_exec", return_value=mock_process):
            result = await tool.execute({"command": "dir nonexistent"})
            assert "failed" in result.lower()

    @pytest.mark.asyncio
    async def test_file_read(self, settings: AraSettings, tmp_path: Path) -> None:
        from ara.tools.system import FileReadTool

        tool = FileReadTool(settings)
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello\nworld\nfoo")

        result = await tool.execute({"path": str(test_file)})
        assert "hello" in result
        assert "world" in result

    @pytest.mark.asyncio
    async def test_file_read_with_line_limit(self, settings: AraSettings, tmp_path: Path) -> None:
        from ara.tools.system import FileReadTool

        tool = FileReadTool(settings)
        test_file = tmp_path / "test.txt"
        test_file.write_text("\n".join(f"line {i}" for i in range(100)))

        result = await tool.execute({"path": str(test_file), "lines": 5})
        assert "line 0" in result
        assert "line 4" in result
        assert "95 more lines" in result

    @pytest.mark.asyncio
    async def test_file_read_not_found(self, settings: AraSettings) -> None:
        from ara.tools.system import FileReadTool

        tool = FileReadTool(settings)
        with pytest.raises(DesktopError, match="File not found"):
            await tool.execute({"path": "C:/nonexistent/file.txt"})

    @pytest.mark.asyncio
    async def test_file_read_too_large(self, settings: AraSettings, tmp_path: Path) -> None:
        from ara.tools.system import FileReadTool

        tool = FileReadTool(settings)
        large_file = tmp_path / "large.txt"
        large_file.write_text("x" * 200_000)

        with pytest.raises(DesktopError, match="too large"):
            await tool.execute({"path": str(large_file)})

    @pytest.mark.asyncio
    async def test_file_write(self, settings: AraSettings, tmp_path: Path) -> None:
        from ara.tools.system import FileWriteTool

        tool = FileWriteTool(settings)
        target = tmp_path / "output.txt"

        result = await tool.execute({"path": str(target), "content": "hello world"})
        assert target.read_text() == "hello world"
        assert "11 chars" in result

    @pytest.mark.asyncio
    async def test_file_write_append(self, settings: AraSettings, tmp_path: Path) -> None:
        from ara.tools.system import FileWriteTool

        tool = FileWriteTool(settings)
        target = tmp_path / "output.txt"
        target.write_text("first ")

        result = await tool.execute({"path": str(target), "content": "second", "append": True})
        assert target.read_text() == "first second"
        assert "Appended" in result

    @pytest.mark.asyncio
    async def test_environment_info_specific(self, settings: AraSettings) -> None:
        from ara.tools.system import EnvironmentInfoTool

        tool = EnvironmentInfoTool(settings)

        with patch.dict("os.environ", {"MY_VAR": "test_value"}):
            result = await tool.execute({"variable": "MY_VAR"})
            assert "test_value" in result

    @pytest.mark.asyncio
    async def test_environment_info_not_set(self, settings: AraSettings) -> None:
        from ara.tools.system import EnvironmentInfoTool

        tool = EnvironmentInfoTool(settings)
        result = await tool.execute({"variable": "UNLIKELY_VARIABLE_12345"})
        assert "not set" in result

    @pytest.mark.asyncio
    async def test_environment_info_all(self, settings: AraSettings) -> None:
        from ara.tools.system import EnvironmentInfoTool

        tool = EnvironmentInfoTool(settings)
        result = await tool.execute({})
        assert "Key environment variables" in result

    def test_register_system_tools(self, settings: AraSettings) -> None:
        from ara.tools.system import register_system_tools

        registry = ToolRegistry(settings)
        names = register_system_tools(registry, settings)
        assert len(names) == 4
        assert "run_command" in names
        assert "file_read" in names
        assert "file_write" in names


# ═══════════════════════════════════════════════════════════════════════════════
# MEDIA TOOL TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestMediaTools:
    """Tests for media tools — all ctypes calls mocked."""

    @pytest.mark.asyncio
    async def test_volume_mute(self, settings: AraSettings) -> None:
        from ara.tools.media import VolumeControlTool

        tool = VolumeControlTool(settings)

        with patch("ara.tools.media._press_vk") as mock_vk:
            result = await tool.execute({"action": "mute"})
            mock_vk.assert_called_once()
            assert "muted" in result.lower()

    @pytest.mark.asyncio
    async def test_volume_up(self, settings: AraSettings) -> None:
        from ara.tools.media import VolumeControlTool

        tool = VolumeControlTool(settings)

        with patch("ara.tools.media._press_vk") as mock_vk:
            result = await tool.execute({"action": "up"})
            assert mock_vk.call_count == 3  # 3 presses
            assert "raised" in result.lower()

    @pytest.mark.asyncio
    async def test_volume_down(self, settings: AraSettings) -> None:
        from ara.tools.media import VolumeControlTool

        tool = VolumeControlTool(settings)

        with patch("ara.tools.media._press_vk") as mock_vk:
            result = await tool.execute({"action": "down"})
            assert mock_vk.call_count == 3
            assert "lowered" in result.lower()

    @pytest.mark.asyncio
    async def test_volume_set_with_pycaw(self, settings: AraSettings) -> None:
        from ara.tools.media import VolumeControlTool

        tool = VolumeControlTool(settings)

        mock_volume = MagicMock()
        mock_interface = MagicMock()
        mock_interface.QueryInterface.return_value = mock_volume
        mock_devices = MagicMock()
        mock_devices.Activate.return_value = mock_interface

        with (
            patch.dict(
                "sys.modules",
                {"comtypes": MagicMock(), "pycaw": MagicMock(), "pycaw.pycaw": MagicMock()},
            ),
            patch("ara.tools.media.VolumeControlTool._set_volume") as mock_set,
        ):
            mock_set.return_value = "Volume set to 75%."
            result = await tool.execute({"action": "set", "level": 75})
            assert "75%" in result

    @pytest.mark.asyncio
    async def test_volume_unknown_action(self, settings: AraSettings) -> None:
        from ara.tools.media import VolumeControlTool

        tool = VolumeControlTool(settings)
        with pytest.raises(DesktopError, match="Unknown volume action"):
            await tool.execute({"action": "explode"})

    @pytest.mark.asyncio
    async def test_media_play(self, settings: AraSettings) -> None:
        from ara.tools.media import MediaControlTool

        tool = MediaControlTool(settings)

        with patch("ara.tools.media._press_vk") as mock_vk:
            result = await tool.execute({"action": "play"})
            mock_vk.assert_called_once()
            assert "play/pause" in result.lower()

    @pytest.mark.asyncio
    async def test_media_next(self, settings: AraSettings) -> None:
        from ara.tools.media import MediaControlTool

        tool = MediaControlTool(settings)

        with patch("ara.tools.media._press_vk") as mock_vk:
            result = await tool.execute({"action": "next"})
            mock_vk.assert_called_once()
            assert "next" in result.lower()

    @pytest.mark.asyncio
    async def test_media_unknown_action(self, settings: AraSettings) -> None:
        from ara.tools.media import MediaControlTool

        tool = MediaControlTool(settings)
        with pytest.raises(DesktopError, match="Unknown media action"):
            await tool.execute({"action": "explode"})

    def test_register_media_tools(self, settings: AraSettings) -> None:
        from ara.tools.media import register_media_tools

        registry = ToolRegistry(settings)
        names = register_media_tools(registry, settings)
        assert len(names) == 2
        assert "volume_control" in names
        assert "media_control" in names


# ═══════════════════════════════════════════════════════════════════════════════
# MCP TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestMCPConfig:
    """Tests for MCP configuration types."""

    def test_mcp_server_config(self) -> None:
        from ara.tools.mcp import MCPServerConfig

        config = MCPServerConfig(
            name="test_server",
            transport="stdio",
            command="python",
            args=("-m", "test_server"),
        )
        assert config.name == "test_server"
        assert config.transport == "stdio"
        assert config.command == "python"

    def test_mcp_server_config_http(self) -> None:
        from ara.tools.mcp import MCPServerConfig

        config = MCPServerConfig(
            name="test_server",
            transport="http",
            url="http://localhost:3000",
        )
        assert config.transport == "http"
        assert config.url == "http://localhost:3000"


class TestMCPConnection:
    """Tests for MCPConnection (mocked)."""

    @pytest.mark.asyncio
    async def test_connection_init(self) -> None:
        from ara.tools.mcp import MCPConnection, MCPServerConfig

        config = MCPServerConfig(name="test", transport="stdio", command="echo")
        conn = MCPConnection(config)
        assert conn.connected is False
        assert conn.server_name == "test"
        assert conn.discovered_tools == []

    @pytest.mark.asyncio
    async def test_connection_invalid_transport(self) -> None:
        from ara.tools.mcp import MCPConnection, MCPServerConfig

        config = MCPServerConfig(name="test", transport="invalid")
        conn = MCPConnection(config)
        with pytest.raises(DesktopError, match="Unknown MCP transport"):
            await conn.connect()

    @pytest.mark.asyncio
    async def test_connection_no_command(self) -> None:
        from ara.tools.mcp import MCPConnection, MCPServerConfig

        config = MCPServerConfig(name="test", transport="stdio", command="")
        conn = MCPConnection(config)
        with pytest.raises(DesktopError, match="no command"):
            await conn.connect()

    @pytest.mark.asyncio
    async def test_connection_close(self) -> None:
        from ara.tools.mcp import MCPConnection, MCPServerConfig

        config = MCPServerConfig(name="test", transport="stdio", command="echo")
        conn = MCPConnection(config)
        conn._process = AsyncMock()
        conn._process.terminate = MagicMock()
        conn._process.wait = AsyncMock()
        conn._connected = True

        await conn.close()
        assert conn.connected is False


class TestMCPToolProxy:
    """Tests for MCPToolProxy."""

    def test_proxy_schema_generation(self, settings: AraSettings) -> None:
        from ara.tools.mcp import MCPConnection, MCPServerConfig, MCPToolProxy

        config = MCPServerConfig(name="test_srv", transport="stdio", command="echo")
        conn = MCPConnection(config)

        tool_def = {
            "name": "search_files",
            "description": "Search for files by pattern",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Glob pattern"},
                    "max_results": {"type": "integer", "description": "Max results"},
                },
                "required": ["pattern"],
            },
        }

        proxy = MCPToolProxy(settings, conn, tool_def, "test_srv")
        schema = proxy.schema
        assert schema.name == "mcp_test_srv_search_files"
        assert "[MCP:test_srv]" in schema.description
        assert len(schema.parameters) == 2
        assert schema.parameters[0].name == "pattern"
        assert schema.parameters[0].required is True
        assert schema.parameters[1].name == "max_results"
        assert schema.parameters[1].required is False

    def test_proxy_schema_with_enum(self, settings: AraSettings) -> None:
        from ara.tools.mcp import MCPConnection, MCPServerConfig, MCPToolProxy

        config = MCPServerConfig(name="srv", transport="stdio", command="echo")
        conn = MCPConnection(config)

        tool_def = {
            "name": "set_mode",
            "description": "Set mode",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "mode": {"type": "string", "enum": ["fast", "slow"], "description": "Mode"},
                },
                "required": ["mode"],
            },
        }

        proxy = MCPToolProxy(settings, conn, tool_def, "srv")
        assert proxy.schema.parameters[0].enum == ("fast", "slow")


class TestMCPBridge:
    """Tests for MCPBridge."""

    def test_bridge_init(self, settings: AraSettings) -> None:
        from ara.tools.mcp import MCPBridge

        bridge = MCPBridge(settings)
        assert bridge.connection_count == 0
        assert bridge.tool_count == 0

    @pytest.mark.asyncio
    async def test_bridge_close_all(self, settings: AraSettings) -> None:
        from ara.tools.mcp import MCPBridge

        bridge = MCPBridge(settings)
        await bridge.close_all()
        assert bridge.connection_count == 0

    def test_bridge_register_tools(self, settings: AraSettings) -> None:
        from ara.tools.mcp import MCPBridge

        bridge = MCPBridge(settings)
        registry = ToolRegistry(settings)
        names = bridge.register_tools(registry)
        assert names == []  # No tools discovered yet


# Integration tests: registry + permissions + audit pipeline


class TestRegistryIntegration:
    """Integration tests for registry + permissions + audit pipeline."""

    @pytest.mark.asyncio
    async def test_full_pipeline(self, settings: AraSettings, tmp_path: Path) -> None:
        """Full pipeline: permission check -> execute -> audit log."""
        policy = PermissionPolicy.default()
        checker = PermissionChecker(settings, policy)
        audit = AuditLogger(tmp_path / "audit.sqlite")
        await audit.connect()

        registry = ToolRegistry(settings, permission_checker=checker, audit_logger=audit.log)
        registry.register(DummyTool(settings, output="success!"))

        result = await registry.execute(ToolCall(tool_name="dummy", parameters={"msg": "test"}))
        assert result.success is True
        assert result.output == "success!"

        entries = audit.query_recent()
        assert len(entries) == 1
        assert entries[0]["tool_name"] == "dummy"
        assert entries[0]["success"] is True

        await audit.close()

    @pytest.mark.asyncio
    async def test_permission_denied_pipeline(self, settings: AraSettings, tmp_path: Path) -> None:
        """Permission denial is raised before execution."""
        policy = PermissionPolicy(denied_tools={"dummy"})
        checker = PermissionChecker(settings, policy)
        audit = AuditLogger(tmp_path / "audit.sqlite")
        await audit.connect()

        registry = ToolRegistry(settings, permission_checker=checker, audit_logger=audit.log)
        registry.register(DummyTool(settings))

        with pytest.raises(ToolPermissionError):
            await registry.execute(ToolCall(tool_name="dummy", parameters={"msg": "test"}))

        # No audit entry — permission denied before execution
        entries = audit.query_recent()
        assert len(entries) == 0

        await audit.close()

    @pytest.mark.asyncio
    async def test_dangerous_allowed_with_custom_checker(self, settings: AraSettings) -> None:
        """Custom permission checker can allow dangerous tools."""
        policy = PermissionPolicy(allow_dangerous=True)
        checker = PermissionChecker(settings, policy)

        registry = ToolRegistry(settings, permission_checker=checker)
        registry.register(
            DummyTool(settings, name="danger", risk=ToolRiskLevel.DANGEROUS, output="ran!")
        )

        result = await registry.execute(ToolCall(tool_name="danger", parameters={"msg": "test"}))
        assert result.success is True
        assert result.output == "ran!"

    @pytest.mark.asyncio
    async def test_all_desktop_tools_register(self, settings: AraSettings) -> None:
        """Verify all tool registration helpers work together."""
        from ara.tools.desktop import register_desktop_tools
        from ara.tools.media import register_media_tools
        from ara.tools.system import register_system_tools

        registry = ToolRegistry(settings)
        d = register_desktop_tools(registry, settings)
        s = register_system_tools(registry, settings)
        m = register_media_tools(registry, settings)

        total = len(d) + len(s) + len(m)
        assert total == registry.tool_count
        assert total == 17  # 11 desktop + 4 system + 2 media

    @pytest.mark.asyncio
    async def test_prompt_generation_with_all_tools(self, settings: AraSettings) -> None:
        """Verify prompt generation with all tools registered."""
        from ara.tools.desktop import register_desktop_tools
        from ara.tools.media import register_media_tools
        from ara.tools.system import register_system_tools

        registry = ToolRegistry(settings)
        register_desktop_tools(registry, settings)
        register_system_tools(registry, settings)
        register_media_tools(registry, settings)

        prompt = registry.generate_tool_prompt()
        assert "Available tools:" in prompt
        assert "[DESKTOP]" in prompt
        assert "[SYSTEM]" in prompt
        assert "[MEDIA]" in prompt
        # Verify all tools appear
        assert "open_application" in prompt
        assert "run_command" in prompt
        assert "volume_control" in prompt
