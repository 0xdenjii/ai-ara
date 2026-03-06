"""Tool registry — plugin system for desktop automation and tool execution.

Provides:
    - BaseTool: Abstract base class for all tools
    - ToolParameter / ToolSchema: Declarative tool metadata for LLM prompt generation
    - ToolRegistry: Central registry that discovers, holds, and executes tools
    - Integrates with ToolCall/ToolResult types and the permission system

Architecture:
    Each tool is a BaseTool subclass that declares a ToolSchema (name, description,
    parameters) and implements an async execute() method.  The ToolRegistry holds all
    registered tools and provides a single ``execute(ToolCall) -> ToolResult`` entry
    point that the ToolAgent calls.

    Tools can be registered statically (via ``register()``) or discovered at runtime
    via ``load_plugin()`` for hot-loadable extensibility.

Usage:
    registry = ToolRegistry(settings)
    registry.register(OpenApplicationTool(settings))
    result = await registry.execute(ToolCall(tool_name="open_application", parameters={...}))
"""

from __future__ import annotations

import importlib
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from ara.utils.errors import DesktopError, ToolExecutionError, ToolPermissionError
from ara.utils.logging import get_logger
from ara.utils.types import ToolCall, ToolResult

if TYPE_CHECKING:
    from ara.config.settings import AraSettings

log = get_logger("tool_registry")


# ── Parameter Types ──────────────────────────────────────────────────────────


class ParameterType(StrEnum):
    """Supported parameter types for tool schemas."""

    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    OBJECT = "object"
    ARRAY = "array"


# ── Tool Risk Level ──────────────────────────────────────────────────────────


class ToolRiskLevel(StrEnum):
    """Risk classification for tools.

    Determines whether user confirmation is needed before execution.
    """

    SAFE = "safe"  # No confirmation needed (e.g., screenshot, volume query)
    MODERATE = "moderate"  # Confirmation if settings.require_confirmation is True
    DANGEROUS = "dangerous"  # Always requires confirmation


# ── Tool Schema ──────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ToolParameter:
    """Schema for a single tool parameter.

    Used to generate LLM-facing tool descriptions and to validate inputs.

    Attributes:
        name: Parameter name (e.g. "path", "command").
        param_type: Data type.
        description: Human-readable description for the LLM.
        required: Whether the parameter is mandatory.
        default: Default value if not provided.
        enum: Allowed values (e.g. ["play", "pause", "stop"]).
    """

    name: str
    param_type: ParameterType
    description: str
    required: bool = True
    default: Any = None
    enum: tuple[str, ...] | None = None


@dataclass(frozen=True)
class ToolSchema:
    """Full schema for a tool — name, description, parameters, risk level.

    This is the metadata that gets injected into the LLM prompt so it knows
    what tools are available and how to call them.

    Attributes:
        name: Unique tool identifier (e.g. "open_application").
        description: What the tool does (1-2 sentences).
        parameters: Ordered tuple of ToolParameter definitions.
        risk_level: How dangerous this tool is.
        category: Grouping category (e.g. "desktop", "system", "media").
    """

    name: str
    description: str
    parameters: tuple[ToolParameter, ...] = ()
    risk_level: ToolRiskLevel = ToolRiskLevel.SAFE
    category: str = "general"

    def to_prompt_text(self) -> str:
        """Format this tool's schema as text for LLM prompts.

        Returns:
            A single line like: ``- open_application: Opens an app. Parameters: {"name": "app_name"}``
        """
        if not self.parameters:
            return f"- {self.name}: {self.description} Parameters: {{}}"

        param_parts: list[str] = []
        for p in self.parameters:
            type_hint = p.param_type.value
            if p.enum:
                type_hint = "|".join(p.enum)
            if p.required:
                param_parts.append(f'"{p.name}": "<{type_hint}>"')
            else:
                default_repr = repr(p.default) if p.default is not None else "null"
                param_parts.append(
                    f'"{p.name}": "<{type_hint}> (optional, default={default_repr})"'
                )

        params_str = "{" + ", ".join(param_parts) + "}"
        return f"- {self.name}: {self.description} Parameters: {params_str}"

    def validate_parameters(self, params: dict[str, Any]) -> list[str]:
        """Validate parameters against the schema.

        Returns:
            List of validation error strings (empty if valid).
        """
        errors: list[str] = []
        for p in self.parameters:
            if p.required and p.name not in params:
                errors.append(f"Missing required parameter: '{p.name}'")
            if p.enum and p.name in params:
                val = str(params[p.name])
                if val not in p.enum:
                    errors.append(
                        f"Parameter '{p.name}' must be one of {list(p.enum)}, got '{val}'"
                    )
        # Warn about unknown parameters
        known_names = {p.name for p in self.parameters}
        errors.extend(f"Unknown parameter: '{key}'" for key in params if key not in known_names)
        return errors


# ── Base Tool ────────────────────────────────────────────────────────────────


class BaseTool(ABC):
    """Abstract base class for all tools.

    Subclasses must implement:
        - ``schema`` property: returns the ToolSchema
        - ``execute(params)`` method: runs the tool and returns output string

    The registry calls ``execute()`` and wraps the result into a ToolResult.
    """

    def __init__(self, settings: AraSettings) -> None:
        self._settings = settings

    @property
    @abstractmethod
    def schema(self) -> ToolSchema:
        """Return the tool's schema definition."""

    async def connect(self) -> None:  # noqa: B027
        """Initialize any resources the tool needs.

        Override in subclasses that need setup (e.g., launch browser, connect to API).
        """

    async def close(self) -> None:  # noqa: B027
        """Release resources.

        Override in subclasses that hold resources.
        """

    @abstractmethod
    async def execute(self, parameters: dict[str, Any]) -> str:
        """Execute the tool with the given parameters.

        Args:
            parameters: Validated parameters dict.

        Returns:
            Human-readable output string describing what happened.

        Raises:
            DesktopError: If the tool fails.
        """


# ── Tool Registry ────────────────────────────────────────────────────────────


class ToolRegistry:
    """Central registry for tool discovery, validation, and execution.

    This is the ``tool_executor`` that the ToolAgent calls.  It:
        1. Holds all registered BaseTool instances
        2. Validates ToolCall parameters against schemas
        3. Checks permission/risk level before execution
        4. Measures execution time
        5. Returns structured ToolResult objects

    The registry also generates LLM-facing tool descriptions from schemas,
    so the LLM prompt is always in sync with actually-available tools.

    Args:
        settings: Application settings.
        permission_checker: Optional callable ``(tool_name, risk_level) -> bool``
            that returns True if execution is allowed.  If not provided,
            all SAFE tools execute freely, MODERATE/DANGEROUS tools check
            ``settings.desktop.require_confirmation``.
        audit_logger: Optional callable ``(ToolCall, ToolResult) -> None``
            for logging all tool executions.
    """

    def __init__(
        self,
        settings: AraSettings,
        *,
        permission_checker: Any | None = None,
        audit_logger: Any | None = None,
    ) -> None:
        self._settings = settings
        self._tools: dict[str, BaseTool] = {}
        self._permission_checker = permission_checker
        self._audit_logger = audit_logger
        self._execution_count: int = 0
        self._error_count: int = 0

    # ── Registration ─────────────────────────────────────────────────────

    def register(self, tool: BaseTool) -> None:
        """Register a tool.  Replaces any existing tool with the same name.

        Args:
            tool: A BaseTool instance.
        """
        name = tool.schema.name
        if name in self._tools:
            log.info("tool_replaced", name=name)
        self._tools[name] = tool
        log.debug("tool_registered", name=name, category=tool.schema.category)

    def unregister(self, name: str) -> bool:
        """Unregister a tool by name.

        Returns:
            True if the tool was found and removed, False otherwise.
        """
        if name in self._tools:
            del self._tools[name]
            log.debug("tool_unregistered", name=name)
            return True
        return False

    def get_tool(self, name: str) -> BaseTool | None:
        """Look up a tool by name.

        Returns:
            The BaseTool instance or None if not found.
        """
        return self._tools.get(name)

    @property
    def tool_names(self) -> list[str]:
        """All registered tool names."""
        return list(self._tools.keys())

    @property
    def tool_count(self) -> int:
        """Number of registered tools."""
        return len(self._tools)

    def get_tools_by_category(self, category: str) -> list[BaseTool]:
        """Get all tools in a category.

        Args:
            category: Category string (e.g. "desktop", "system", "media").

        Returns:
            List of matching tools.
        """
        return [t for t in self._tools.values() if t.schema.category == category]

    # ── LLM Prompt Generation ────────────────────────────────────────────

    def generate_tool_prompt(self) -> str:
        """Generate the tool availability section for LLM system prompts.

        Returns:
            Multi-line text listing all registered tools with their schemas.
        """
        if not self._tools:
            return "No tools available."

        lines: list[str] = ["Available tools:"]
        # Group by category
        categories: dict[str, list[BaseTool]] = {}
        for tool in self._tools.values():
            cat = tool.schema.category
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(tool)

        for cat in sorted(categories.keys()):
            tools = sorted(categories[cat], key=lambda t: t.schema.name)
            lines.append(f"\n[{cat.upper()}]")
            lines.extend(tool.schema.to_prompt_text() for tool in tools)

        return "\n".join(lines)

    # ── Execution ────────────────────────────────────────────────────────

    async def execute(self, tool_call: ToolCall) -> ToolResult:
        """Execute a tool call.  This is the ``tool_executor`` entry point.

        1. Look up the tool
        2. Validate parameters
        3. Check permissions
        4. Execute
        5. Log audit trail
        6. Return ToolResult

        Args:
            tool_call: The structured tool call to execute.

        Returns:
            ToolResult with success/failure status, output, and timing.

        Raises:
            ToolExecutionError: If the tool is not found or validation fails.
            ToolPermissionError: If the permission check blocks execution.
        """
        tool_name = tool_call.tool_name
        call_id = tool_call.call_id
        start_time = time.monotonic()

        # 1. Look up
        tool = self._tools.get(tool_name)
        if tool is None:
            self._error_count += 1
            raise ToolExecutionError(
                tool_name,
                f"Tool not found. Available tools: {', '.join(self.tool_names)}",
            )

        schema = tool.schema

        # 2. Validate parameters
        validation_errors = schema.validate_parameters(tool_call.parameters)
        if validation_errors:
            self._error_count += 1
            raise ToolExecutionError(
                tool_name,
                f"Parameter validation failed: {'; '.join(validation_errors)}",
            )

        # 3. Check permissions
        if not self._check_permission(tool_name, schema.risk_level):
            raise ToolPermissionError(tool_name, f"risk_level={schema.risk_level.value}")

        # 4. Execute
        try:
            output = await tool.execute(tool_call.parameters)
            elapsed = (time.monotonic() - start_time) * 1000
            self._execution_count += 1

            result = ToolResult(
                call_id=call_id,
                tool_name=tool_name,
                success=True,
                output=output,
                execution_time_ms=round(elapsed, 2),
            )
        except DesktopError as exc:
            elapsed = (time.monotonic() - start_time) * 1000
            self._error_count += 1
            result = ToolResult(
                call_id=call_id,
                tool_name=tool_name,
                success=False,
                output="",
                error=str(exc),
                execution_time_ms=round(elapsed, 2),
            )
        except Exception as exc:
            elapsed = (time.monotonic() - start_time) * 1000
            self._error_count += 1
            log.error(
                "tool_execution_unexpected_error",
                tool=tool_name,
                error=str(exc),
                elapsed_ms=round(elapsed, 2),
            )
            result = ToolResult(
                call_id=call_id,
                tool_name=tool_name,
                success=False,
                output="",
                error=f"Unexpected error: {type(exc).__name__}: {exc}",
                execution_time_ms=round(elapsed, 2),
            )

        # 5. Audit log
        self._audit(tool_call, result)

        return result

    def _check_permission(self, tool_name: str, risk_level: ToolRiskLevel) -> bool:
        """Check if a tool call is permitted.

        Args:
            tool_name: Name of the tool.
            risk_level: The tool's declared risk level.

        Returns:
            True if execution is allowed.
        """
        # External permission checker takes precedence
        if self._permission_checker is not None:
            return bool(self._permission_checker(tool_name, risk_level))

        # Desktop disabled globally
        if not self._settings.desktop.enabled:
            return False

        # SAFE tools always allowed
        if risk_level == ToolRiskLevel.SAFE:
            return True

        # DANGEROUS tools always blocked without explicit permission checker.
        # MODERATE tools are allowed because the ToolAgent handles the confirmation
        # flow before calling execute — if we get here, it's already confirmed.
        return risk_level != ToolRiskLevel.DANGEROUS

    def _audit(self, tool_call: ToolCall, result: ToolResult) -> None:
        """Log a tool execution to the audit trail.

        Args:
            tool_call: The call that was made.
            result: The result that was produced.
        """
        log.info(
            "tool_executed",
            tool=tool_call.tool_name,
            call_id=tool_call.call_id,
            success=result.success,
            elapsed_ms=result.execution_time_ms,
        )

        if self._audit_logger is not None:
            try:
                self._audit_logger(tool_call, result)
            except Exception as exc:
                log.warning("audit_logger_error", error=str(exc))

    # ── Lifecycle ────────────────────────────────────────────────────────

    async def connect_all(self) -> None:
        """Initialize all registered tools."""
        for name, tool in self._tools.items():
            try:
                await tool.connect()
                log.debug("tool_connected", name=name)
            except Exception as exc:
                log.error("tool_connect_failed", name=name, error=str(exc))

    async def close_all(self) -> None:
        """Release resources for all registered tools."""
        for name, tool in self._tools.items():
            try:
                await tool.close()
                log.debug("tool_closed", name=name)
            except Exception as exc:
                log.warning("tool_close_error", name=name, error=str(exc))

    # ── Plugin Loading ───────────────────────────────────────────────────

    def load_plugin(self, module_path: str, class_name: str) -> BaseTool:
        """Dynamically load a tool from an external module.

        Args:
            module_path: Dotted module path (e.g. "ara.tools.desktop").
            class_name: Name of the BaseTool subclass to instantiate.

        Returns:
            The instantiated and registered tool.

        Raises:
            ToolExecutionError: If loading fails.
        """
        try:
            module = importlib.import_module(module_path)
        except ModuleNotFoundError as exc:
            raise ToolExecutionError(
                class_name,
                f"Could not import module '{module_path}': {exc}",
            ) from exc

        cls = getattr(module, class_name, None)
        if cls is None:
            raise ToolExecutionError(
                class_name,
                f"Class '{class_name}' not found in module '{module_path}'",
            )

        if not isinstance(cls, type) or not issubclass(cls, BaseTool):
            raise ToolExecutionError(
                class_name,
                f"'{class_name}' is not a subclass of BaseTool",
            )

        instance = cls(self._settings)
        self.register(instance)
        log.info("plugin_loaded", module=module_path, tool=class_name)
        return instance

    # ── Stats ────────────────────────────────────────────────────────────

    def get_stats(self) -> dict[str, Any]:
        """Return registry statistics.

        Returns:
            Dict with tool_count, execution_count, error_count, and tool names.
        """
        return {
            "tool_count": self.tool_count,
            "execution_count": self._execution_count,
            "error_count": self._error_count,
            "tools": {
                name: {
                    "category": tool.schema.category,
                    "risk_level": tool.schema.risk_level.value,
                    "parameters": [p.name for p in tool.schema.parameters],
                }
                for name, tool in self._tools.items()
            },
        }
