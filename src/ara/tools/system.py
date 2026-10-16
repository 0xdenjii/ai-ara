"""System tools — subprocess execution, PowerShell, file operations, environment queries.

Provides BaseTool implementations for:
    - RunCommandTool: Execute whitelisted shell/PowerShell commands
    - FileReadTool: Read file contents (with size limits)
    - FileWriteTool: Write/append to files
    - EnvironmentInfoTool: Query environment variables and system paths

Security:
    All command execution goes through a command whitelist.  Commands not
    on the whitelist are rejected.  File operations are restricted to
    configured safe directories.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ara.tools.registry import (
    BaseTool,
    ParameterType,
    ToolParameter,
    ToolRiskLevel,
    ToolSchema,
)
from ara.utils.errors import DesktopError
from ara.utils.logging import get_logger

if TYPE_CHECKING:
    from ara.config.settings import AraSettings

log = get_logger("system_tools")

# ── Command Whitelist ────────────────────────────────────────────────────────
# Only commands starting with these prefixes are allowed.
# This prevents arbitrary code execution while still being useful.

_COMMAND_WHITELIST_PREFIXES: tuple[str, ...] = (
    # PowerShell informational commands
    "get-",
    "where",
    "select-",
    "sort-",
    "measure-",
    "format-",
    "out-",
    "convertto-",
    "convertfrom-",
    "test-",
    "resolve-",
    # Listing / info
    "dir",
    "ls",
    "pwd",
    "hostname",
    "whoami",
    "date",
    "time",
    "ver",
    "systeminfo",
    "ipconfig",
    "netstat",
    "ping",
    "nslookup",
    "tracert",
    "tasklist",
    "findstr",
    "git status",
    "git log",
    "git diff",
    "git branch",
    "git remote",
    "git show",
    "git tag",
    # Python (read-only commands only)
    "python --version",
    "python -m pip list",
    # Node
    "node --version",
    "npm --version",
    "npm list",
)

# These command patterns are always blocked, regardless of whitelist.
_COMMAND_BLACKLIST_PATTERNS: tuple[str, ...] = (
    "rm -rf",
    "del /f",
    "format ",
    "shutdown",
    "restart",
    "reg delete",
    "reg add",
    "net user",
    "net localgroup",
    "takeown",
    "icacls",
    "bcdedit",
    "diskpart",
    "cipher /w",
    "invoke-expression",
    "invoke-webrequest",
    "start-process",
    "new-object",
    "add-type",
    "set-content",
    "remove-item",
    "move-item",
    "copy-item",
    "out-file",
    "set-executionpolicy",
    "downloadstring",
    "downloadfile",
    "iex ",
    "iwr ",
)

# Shell operators that enable command chaining/injection.
# Commands containing any of these are rejected.
_SHELL_OPERATORS: tuple[str, ...] = (
    ";",  # statement separator
    "|",  # pipe
    "&&",  # conditional AND
    "||",  # conditional OR
    "$(",  # subexpression
    "& ",  # call operator
    "&{",  # script block call
    "`",  # backtick escape
    ">",  # output redirect (file write)
    "<",  # input redirect
    ">>",  # append redirect
    "2>",  # stderr redirect
    "\n",  # newline injection (second command line)
    "\r",  # carriage return injection
)

# Maximum output length from commands (characters)
_MAX_OUTPUT_LENGTH = 10000


def _is_path_allowed(path: Path, settings: AraSettings) -> bool:
    """Check if a resolved path is within allowed directories.

    Allowed directories are:
        - The user's home directory (and subdirectories)
        - The project data directory (and subdirectories)

    Args:
        path: Resolved (absolute) path to check.
        settings: Application settings for data_dir lookup.

    Returns:
        True if the path is within an allowed directory.
    """
    resolved = path.resolve()
    home_dir = Path.home().resolve()
    data_dir = Path(settings.data_dir).resolve()

    return (
        resolved in (home_dir, data_dir)
        or _is_subpath(resolved, home_dir)
        or _is_subpath(resolved, data_dir)
    )


def _is_subpath(child: Path, parent: Path) -> bool:
    """Check if child is a sub-path of parent."""
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def _is_command_allowed(command: str) -> bool:
    """Check if a command is allowed by the whitelist.

    Validates against blacklist, shell operators (injection prevention),
    and whitelist. All three checks must pass.

    Args:
        command: The command string to check.

    Returns:
        True if the command is allowed.
    """
    cmd_lower = command.strip().lower()

    # Check blacklist first
    for pattern in _COMMAND_BLACKLIST_PATTERNS:
        if pattern in cmd_lower:
            return False

    # Block shell operators that enable command injection
    for op in _SHELL_OPERATORS:
        if op in command:
            return False

    # Check whitelist
    return any(cmd_lower.startswith(prefix.lower()) for prefix in _COMMAND_WHITELIST_PREFIXES)


# ── Run Command ──────────────────────────────────────────────────────────────


class RunCommandTool(BaseTool):
    """Execute a command via PowerShell with whitelist enforcement."""

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="run_command",
            description=(
                "Runs a command via PowerShell. Only whitelisted commands are allowed "
                "(informational commands like Get-Process, dir, ipconfig, git status, etc.)."
            ),
            parameters=(
                ToolParameter(
                    name="command",
                    param_type=ParameterType.STRING,
                    description="The command to execute.",
                ),
                ToolParameter(
                    name="timeout",
                    param_type=ParameterType.FLOAT,
                    description=(
                        f"Timeout in seconds (max {self._settings.desktop.command_timeout_seconds})."
                    ),
                    required=False,
                    default=self._settings.desktop.command_timeout_seconds,
                ),
            ),
            risk_level=ToolRiskLevel.DANGEROUS,
            category="system",
        )

    async def execute(self, parameters: dict[str, Any]) -> str:
        command = parameters["command"].strip()
        max_timeout = self._settings.desktop.command_timeout_seconds
        timeout = min(float(parameters.get("timeout", max_timeout)), max_timeout)

        if not _is_command_allowed(command):
            raise DesktopError(
                f"Command not allowed: '{command}'. Only whitelisted informational commands are permitted.",
                operation="run_command",
                details={"command": command},
            )

        log.info("executing_command", command=command[:100], timeout=timeout)

        try:
            process = await asyncio.create_subprocess_exec(
                "powershell.exe",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout,
            )

            stdout = stdout_bytes.decode("utf-8", errors="replace").strip()
            stderr = stderr_bytes.decode("utf-8", errors="replace").strip()

            if len(stdout) > _MAX_OUTPUT_LENGTH:
                stdout = (
                    stdout[:_MAX_OUTPUT_LENGTH]
                    + f"\n... (truncated, total {len(stdout_bytes)} bytes)"
                )

            if process.returncode != 0:
                error_msg = stderr[:500] if stderr else f"Exit code {process.returncode}"
                return (
                    f"Command failed (exit {process.returncode}):\n{error_msg}\n\nOutput:\n{stdout}"
                )

            return stdout if stdout else "(no output)"

        except TimeoutError as exc:
            # Kill the timed-out process to prevent subprocess leaks
            try:
                process.kill()
                await process.wait()
            except (ProcessLookupError, OSError):
                pass  # Process already exited
            raise DesktopError(
                f"Command timed out after {timeout}s: '{command[:80]}'",
                operation="run_command",
                details={"command": command, "timeout": timeout},
            ) from exc
        except OSError as exc:
            raise DesktopError(
                f"Failed to execute command: {exc}",
                operation="run_command",
                details={"command": command},
            ) from exc


# ── File Read ────────────────────────────────────────────────────────────────


class FileReadTool(BaseTool):
    """Read contents of a file."""

    @property
    def schema(self) -> ToolSchema:
        max_kb = self._settings.desktop.max_file_read_bytes // 1024
        return ToolSchema(
            name="file_read",
            description=f"Reads the contents of a text file (up to {max_kb}KB).",
            parameters=(
                ToolParameter(
                    name="path",
                    param_type=ParameterType.STRING,
                    description="Absolute or relative file path.",
                ),
                ToolParameter(
                    name="lines",
                    param_type=ParameterType.INTEGER,
                    description="Maximum number of lines to return (0 = all).",
                    required=False,
                    default=0,
                ),
            ),
            risk_level=ToolRiskLevel.MODERATE,
            category="system",
        )

    async def execute(self, parameters: dict[str, Any]) -> str:
        path_str = parameters["path"]
        max_lines = int(parameters.get("lines", 0))
        path = Path(path_str).resolve()

        if not _is_path_allowed(path, self._settings):
            raise DesktopError(
                f"Access denied: '{path}' is outside allowed directories (home and data).",
                operation="file_read",
                details={"path": str(path)},
            )

        if not path.exists():
            raise DesktopError(
                f"File not found: {path}",
                operation="file_read",
                details={"path": str(path)},
            )

        if not path.is_file():
            raise DesktopError(
                f"Not a file: {path}",
                operation="file_read",
                details={"path": str(path)},
            )

        file_size = path.stat().st_size
        max_read = self._settings.desktop.max_file_read_bytes
        if file_size > max_read:
            raise DesktopError(
                f"File too large: {file_size} bytes (max {max_read})",
                operation="file_read",
                details={"path": str(path), "size": file_size, "max_bytes": max_read},
            )

        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise DesktopError(
                f"Failed to read file: {exc}",
                operation="file_read",
                details={"path": str(path)},
            ) from exc

        if max_lines > 0:
            content_lines = content.splitlines()
            if len(content_lines) > max_lines:
                content = "\n".join(content_lines[:max_lines])
                content += f"\n... ({len(content_lines) - max_lines} more lines)"

        return content


# ── File Write ───────────────────────────────────────────────────────────────


class FileWriteTool(BaseTool):
    """Write content to a file."""

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="file_write",
            description="Writes content to a file. Creates the file if it doesn't exist.",
            parameters=(
                ToolParameter(
                    name="path",
                    param_type=ParameterType.STRING,
                    description="Absolute or relative file path.",
                ),
                ToolParameter(
                    name="content",
                    param_type=ParameterType.STRING,
                    description="Content to write.",
                ),
                ToolParameter(
                    name="append",
                    param_type=ParameterType.BOOLEAN,
                    description="Whether to append instead of overwrite.",
                    required=False,
                    default=False,
                ),
            ),
            risk_level=ToolRiskLevel.DANGEROUS,
            category="system",
        )

    async def execute(self, parameters: dict[str, Any]) -> str:
        path_str = parameters["path"]
        content = parameters["content"]
        append = bool(parameters.get("append", False))
        path = Path(path_str).resolve()

        if not _is_path_allowed(path, self._settings):
            raise DesktopError(
                f"Access denied: '{path}' is outside allowed directories (home and data).",
                operation="file_write",
                details={"path": str(path)},
            )

        # Create parent directories if needed
        path.parent.mkdir(parents=True, exist_ok=True)

        mode = "a" if append else "w"
        try:
            path.write_text(content, encoding="utf-8") if not append else _append_file(
                path, content
            )
        except OSError as exc:
            raise DesktopError(
                f"Failed to write file: {exc}",
                operation="file_write",
                details={"path": str(path), "mode": mode},
            ) from exc

        action = "Appended to" if append else "Wrote"
        return f"{action} {path} ({len(content)} chars)."


def _append_file(path: Path, content: str) -> None:
    """Append content to a file."""
    with path.open("a", encoding="utf-8") as f:
        f.write(content)


# ── Environment Info ─────────────────────────────────────────────────────────


class EnvironmentInfoTool(BaseTool):
    """Query environment variables and system paths."""

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="environment_info",
            description="Returns environment information: specific variable, PATH entries, or all variables.",
            parameters=(
                ToolParameter(
                    name="variable",
                    param_type=ParameterType.STRING,
                    description="Specific environment variable to query (optional, shows all if omitted).",
                    required=False,
                ),
            ),
            risk_level=ToolRiskLevel.SAFE,
            category="system",
        )

    async def execute(self, parameters: dict[str, Any]) -> str:
        variable = parameters.get("variable")

        if variable:
            value = os.environ.get(variable.strip())
            if value is None:
                return f"Environment variable '{variable}' is not set."
            # Special formatting for PATH
            if variable.upper() == "PATH":
                entries = value.split(os.pathsep)
                return f"PATH ({len(entries)} entries):\n" + "\n".join(f"  {e}" for e in entries)
            return f"{variable}={value}"

        # Show summary of key variables
        key_vars = [
            "USERNAME",
            "COMPUTERNAME",
            "OS",
            "PROCESSOR_ARCHITECTURE",
            "HOMEPATH",
            "APPDATA",
            "TEMP",
            "PYTHONPATH",
        ]
        lines: list[str] = ["Key environment variables:"]
        for var in key_vars:
            val = os.environ.get(var, "(not set)")
            lines.append(f"  {var}={val}")
        lines.append(f"\nTotal environment variables: {len(os.environ)}")
        return "\n".join(lines)


# ── Registration Helper ──────────────────────────────────────────────────────


def register_system_tools(registry: Any, settings: AraSettings) -> list[str]:
    """Register all system tools with a ToolRegistry.

    Args:
        registry: ToolRegistry instance.
        settings: Application settings.

    Returns:
        List of registered tool names.
    """
    tools: list[BaseTool] = [
        RunCommandTool(settings),
        FileReadTool(settings),
        FileWriteTool(settings),
        EnvironmentInfoTool(settings),
    ]

    names: list[str] = []
    for tool in tools:
        registry.register(tool)
        names.append(tool.schema.name)

    return names
