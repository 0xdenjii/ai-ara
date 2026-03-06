"""MCP (Model Context Protocol) bridge — connects ARA to external MCP tool servers.

Provides:
    - MCPConnection: Manages a connection to a single MCP server
    - MCPBridge: Discovers MCP servers, creates tool wrappers, registers them in the registry
    - MCPToolProxy: A BaseTool wrapper that proxies calls to an MCP server

Architecture:
    MCP servers expose tools via a JSON-RPC-like protocol over stdio or HTTP.
    The MCPBridge connects to configured servers, discovers their available tools,
    and wraps each as a BaseTool that the ToolRegistry can execute.

    This enables ARA to use external tool ecosystems (e.g., file servers,
    database connectors, API bridges) without building custom integrations.

Configuration:
    MCP servers are declared in settings or a config file.  Each server
    specifies a transport (stdio command, or HTTP URL) and optional env vars.

Note:
    This is a simplified MCP client that handles the core protocol.
    Full MCP spec compliance (resources, prompts, sampling) can be added later.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ara import __version__
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
    from ara.tools.registry import ToolRegistry

log = get_logger("mcp_bridge")

# MCP protocol version — single source of truth
_MCP_PROTOCOL_VERSION = "2024-11-05"


# ── MCP Server Config ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class MCPServerConfig:
    """Configuration for an MCP server.

    Attributes:
        name: Human-readable server name.
        transport: "stdio" or "http".
        command: Command to launch the server (for stdio transport).
        args: Command arguments (for stdio transport).
        url: Server URL (for http transport).
        env: Additional environment variables to pass.
    """

    name: str
    transport: str  # "stdio" or "http"
    command: str = ""
    args: tuple[str, ...] = ()
    url: str = ""
    env: dict[str, str] = field(default_factory=dict)


# ── MCP Connection ───────────────────────────────────────────────────────────


def _check_jsonrpc_error(response: dict[str, Any], method: str, server_name: str) -> None:
    """Check a JSON-RPC response for an error and raise if present.

    Args:
        response: Parsed JSON-RPC response.
        method: The RPC method that was called (for error context).
        server_name: MCP server name (for error context).

    Raises:
        DesktopError: If the response contains a JSON-RPC error.
    """
    error = response.get("error")
    if error is not None:
        code = error.get("code", -1) if isinstance(error, dict) else -1
        message = error.get("message", str(error)) if isinstance(error, dict) else str(error)
        raise DesktopError(
            f"MCP server '{server_name}' returned JSON-RPC error on '{method}': {message}",
            operation="mcp_request",
            details={"server": server_name, "method": method, "code": code},
        )


class MCPConnection:
    """Manages a connection to a single MCP server.

    Handles stdio-based JSON-RPC communication for tool discovery and execution.

    Args:
        config: Server configuration.
        connect_timeout_s: Timeout for connection and request operations.
    """

    def __init__(self, config: MCPServerConfig, *, connect_timeout_s: float = 30.0) -> None:
        self._config = config
        self._connect_timeout_s = connect_timeout_s
        self._process: asyncio.subprocess.Process | None = None
        self._request_id: int = 0
        self._connected: bool = False
        self._server_info: dict[str, Any] = {}
        self._tools: list[dict[str, Any]] = []

    @property
    def connected(self) -> bool:
        """Whether the connection is active."""
        return self._connected

    @property
    def server_name(self) -> str:
        """Server name."""
        return self._config.name

    @property
    def discovered_tools(self) -> list[dict[str, Any]]:
        """Tools discovered from the server."""
        return list(self._tools)

    async def connect(self) -> None:
        """Connect to the MCP server and perform initialization handshake.

        For stdio transport, launches the server process and sends initialize.
        For http transport, sends an initialization request via HTTP.
        """
        if self._config.transport == "stdio":
            await self._connect_stdio()
        elif self._config.transport == "http":
            await self._connect_http()
        else:
            raise DesktopError(
                f"Unknown MCP transport: '{self._config.transport}'",
                operation="mcp_connect",
                details={"server": self._config.name},
            )

    async def _connect_stdio(self) -> None:
        """Connect via stdio transport."""
        if not self._config.command:
            raise DesktopError(
                f"MCP server '{self._config.name}' has no command configured",
                operation="mcp_connect",
            )

        import os

        env = {**os.environ, **self._config.env}

        try:
            self._process = await asyncio.create_subprocess_exec(
                self._config.command,
                *self._config.args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
        except FileNotFoundError as exc:
            raise DesktopError(
                f"MCP server command not found: '{self._config.command}'",
                operation="mcp_connect",
                details={"server": self._config.name, "command": self._config.command},
            ) from exc

        # Send initialize request
        response = await self._send_request(
            "initialize",
            {
                "protocolVersion": _MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "ara", "version": __version__},
            },
        )

        self._server_info = response.get("result", {})
        self._connected = True

        # Send initialized notification
        await self._send_notification("notifications/initialized")

        # Discover tools
        await self._discover_tools()

        log.info(
            "mcp_connected",
            server=self._config.name,
            tools=len(self._tools),
        )

    async def _connect_http(self) -> None:
        """Connect via HTTP transport (streamable HTTP or SSE)."""
        # HTTP transport implementation - uses httpx for requests
        try:
            import httpx
        except ImportError as exc:
            raise DesktopError(
                "httpx required for HTTP MCP transport",
                operation="mcp_connect",
            ) from exc

        url = self._config.url.rstrip("/")
        timeout = self._connect_timeout_s
        async with httpx.AsyncClient(timeout=timeout) as client:
            payload = {
                "jsonrpc": "2.0",
                "id": self._next_id(),
                "method": "initialize",
                "params": {
                    "protocolVersion": _MCP_PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": {"name": "ara", "version": __version__},
                },
            }
            resp = await client.post(f"{url}/mcp", json=payload)
            resp.raise_for_status()
            result = resp.json()
            _check_jsonrpc_error(result, "initialize", self._config.name)
            self._server_info = result.get("result", {})

            # Send initialized notification
            notif_payload = {
                "jsonrpc": "2.0",
                "method": "notifications/initialized",
            }
            await client.post(f"{url}/mcp", json=notif_payload)

        self._connected = True

        # Discover tools via HTTP
        async with httpx.AsyncClient(timeout=timeout) as client:
            tools_payload = {
                "jsonrpc": "2.0",
                "id": self._next_id(),
                "method": "tools/list",
                "params": {},
            }
            resp = await client.post(f"{url}/mcp", json=tools_payload)
            resp.raise_for_status()
            result = resp.json()
            _check_jsonrpc_error(result, "tools/list", self._config.name)
            self._tools = result.get("result", {}).get("tools", [])

        log.info(
            "mcp_connected_http",
            server=self._config.name,
            url=url,
            tools=len(self._tools),
        )

    async def _discover_tools(self) -> None:
        """Discover available tools from the connected server."""
        response = await self._send_request("tools/list", {})
        self._tools = response.get("result", {}).get("tools", [])

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Call a tool on the MCP server.

        Args:
            tool_name: Name of the tool to call.
            arguments: Tool arguments.

        Returns:
            Text content from the tool response.
        """
        if not self._connected:
            raise DesktopError(
                f"Not connected to MCP server '{self._config.name}'",
                operation="mcp_call_tool",
            )

        if self._config.transport == "http":
            return await self._call_tool_http(tool_name, arguments)

        response = await self._send_request(
            "tools/call",
            {
                "name": tool_name,
                "arguments": arguments,
            },
        )

        result = response.get("result", {})
        content_parts = result.get("content", [])

        # Extract text content
        text_parts = [
            str(part.get("text", ""))
            for part in content_parts
            if isinstance(part, dict) and part.get("type") == "text"
        ]

        return "\n".join(text_parts) if text_parts else str(result)

    async def _call_tool_http(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Call a tool via HTTP transport."""
        import httpx

        url = self._config.url.rstrip("/")
        async with httpx.AsyncClient(timeout=self._connect_timeout_s) as client:
            payload = {
                "jsonrpc": "2.0",
                "id": self._next_id(),
                "method": "tools/call",
                "params": {"name": tool_name, "arguments": arguments},
            }
            resp = await client.post(f"{url}/mcp", json=payload)
            resp.raise_for_status()
            result = resp.json()
            _check_jsonrpc_error(result, "tools/call", self._config.name)

        content_parts = result.get("result", {}).get("content", [])
        text_parts = [
            str(part.get("text", ""))
            for part in content_parts
            if isinstance(part, dict) and part.get("type") == "text"
        ]

        return "\n".join(text_parts) if text_parts else str(result.get("result", {}))

    async def close(self) -> None:
        """Close the MCP server connection."""
        if self._process is not None:
            try:
                self._process.terminate()
                await asyncio.wait_for(self._process.wait(), timeout=5.0)
            except (TimeoutError, ProcessLookupError):
                self._process.kill()
            self._process = None
        self._connected = False
        log.debug("mcp_disconnected", server=self._config.name)

    def _next_id(self) -> int:
        """Get the next request ID."""
        self._request_id += 1
        return self._request_id

    async def _send_request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """Send a JSON-RPC request and read the response.

        Args:
            method: RPC method name.
            params: Method parameters.

        Returns:
            Parsed JSON response.
        """
        if self._process is None or self._process.stdin is None or self._process.stdout is None:
            raise DesktopError(
                "MCP server process not running",
                operation="mcp_request",
                details={"server": self._config.name, "method": method},
            )

        request = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": method,
            "params": params,
        }

        message = json.dumps(request) + "\n"
        self._process.stdin.write(message.encode("utf-8"))
        await self._process.stdin.drain()

        # Read response line
        try:
            line = await asyncio.wait_for(
                self._process.stdout.readline(),
                timeout=self._connect_timeout_s,
            )
        except TimeoutError as exc:
            raise DesktopError(
                f"MCP server '{self._config.name}' timed out on '{method}' "
                f"(timeout={self._connect_timeout_s}s)",
                operation="mcp_request",
                details={"method": method, "timeout_s": self._connect_timeout_s},
            ) from exc

        if not line:
            raise DesktopError(
                f"MCP server '{self._config.name}' closed connection",
                operation="mcp_request",
                details={"method": method},
            )

        result: dict[str, Any] = json.loads(line.decode("utf-8"))
        _check_jsonrpc_error(result, method, self._config.name)
        return result

    async def _send_notification(self, method: str, params: dict[str, Any] | None = None) -> None:
        """Send a JSON-RPC notification (no response expected).

        Args:
            method: RPC method name.
            params: Optional parameters.
        """
        if self._process is None or self._process.stdin is None:
            return

        notification: dict[str, Any] = {
            "jsonrpc": "2.0",
            "method": method,
        }
        if params:
            notification["params"] = params

        message = json.dumps(notification) + "\n"
        self._process.stdin.write(message.encode("utf-8"))
        await self._process.stdin.drain()


# ── MCP Tool Proxy ───────────────────────────────────────────────────────────


class MCPToolProxy(BaseTool):
    """A BaseTool wrapper that proxies execution to an MCP server tool.

    Created automatically by MCPBridge for each tool discovered from MCP servers.

    Args:
        settings: Application settings.
        connection: The MCP connection to proxy through.
        tool_def: The tool definition from the MCP server's tools/list response.
        server_name: Name of the MCP server (for categorization).
    """

    def __init__(
        self,
        settings: AraSettings,
        connection: MCPConnection,
        tool_def: dict[str, Any],
        server_name: str,
    ) -> None:
        super().__init__(settings)
        self._connection = connection
        self._tool_def = tool_def
        self._server_name = server_name

    @property
    def schema(self) -> ToolSchema:
        name = str(self._tool_def.get("name", "unknown"))
        description = str(self._tool_def.get("description", "MCP tool"))

        # Convert MCP input schema to ToolParameters
        input_schema = self._tool_def.get("inputSchema", {})
        properties = input_schema.get("properties", {})
        required_params = set(input_schema.get("required", []))

        params: list[ToolParameter] = []
        for prop_name, prop_def in properties.items():
            json_type = str(prop_def.get("type", "string"))
            param_type_map = {
                "string": ParameterType.STRING,
                "integer": ParameterType.INTEGER,
                "number": ParameterType.FLOAT,
                "boolean": ParameterType.BOOLEAN,
                "object": ParameterType.OBJECT,
                "array": ParameterType.ARRAY,
            }
            param_type = param_type_map.get(json_type, ParameterType.STRING)
            prop_desc = str(prop_def.get("description", f"{prop_name} parameter"))

            enum_values = prop_def.get("enum")
            enum_tuple = tuple(str(v) for v in enum_values) if enum_values else None

            params.append(
                ToolParameter(
                    name=prop_name,
                    param_type=param_type,
                    description=prop_desc,
                    required=prop_name in required_params,
                    default=prop_def.get("default"),
                    enum=enum_tuple,
                )
            )

        return ToolSchema(
            name=f"mcp_{self._server_name}_{name}",
            description=f"[MCP:{self._server_name}] {description}",
            parameters=tuple(params),
            risk_level=ToolRiskLevel.MODERATE,
            category=f"mcp_{self._server_name}",
        )

    async def execute(self, parameters: dict[str, Any]) -> str:
        tool_name = str(self._tool_def.get("name", ""))
        return await self._connection.call_tool(tool_name, parameters)


# ── MCP Bridge ───────────────────────────────────────────────────────────────


class MCPBridge:
    """Manages multiple MCP server connections and registers their tools.

    Args:
        settings: Application settings.
    """

    def __init__(self, settings: AraSettings) -> None:
        self._settings = settings
        self._connections: dict[str, MCPConnection] = {}
        self._proxies: list[MCPToolProxy] = []

    @property
    def connection_count(self) -> int:
        """Number of active connections."""
        return sum(1 for c in self._connections.values() if c.connected)

    @property
    def tool_count(self) -> int:
        """Total number of MCP tools discovered."""
        return len(self._proxies)

    async def connect_server(self, config: MCPServerConfig) -> list[MCPToolProxy]:
        """Connect to an MCP server and create tool proxies.

        Args:
            config: Server configuration.

        Returns:
            List of MCPToolProxy instances for the server's tools.

        Raises:
            DesktopError: If MCP is disabled in settings.
        """
        if not self._settings.mcp.enabled:
            log.info("mcp_disabled", server=config.name)
            return []

        conn = MCPConnection(config, connect_timeout_s=self._settings.mcp.connect_timeout_s)
        await conn.connect()
        self._connections[config.name] = conn

        proxies: list[MCPToolProxy] = []
        for tool_def in conn.discovered_tools:
            proxy = MCPToolProxy(self._settings, conn, tool_def, config.name)
            proxies.append(proxy)

        self._proxies.extend(proxies)
        log.info(
            "mcp_bridge_server_connected",
            server=config.name,
            tools=len(proxies),
        )
        return proxies

    async def connect_all(self, configs: list[MCPServerConfig]) -> list[MCPToolProxy]:
        """Connect to multiple MCP servers with retry logic.

        Retries each server up to ``settings.mcp.max_retries`` times before
        giving up. Skips all servers if MCP is disabled globally.

        Args:
            configs: List of server configurations.

        Returns:
            All discovered tool proxies.
        """
        if not self._settings.mcp.enabled:
            log.info("mcp_all_disabled")
            return []

        max_retries = self._settings.mcp.max_retries
        all_proxies: list[MCPToolProxy] = []

        for config in configs:
            connected = False
            for attempt in range(1, max_retries + 2):  # 1 initial + max_retries
                try:
                    proxies = await self.connect_server(config)
                    all_proxies.extend(proxies)
                    connected = True
                    break
                except DesktopError as exc:
                    if attempt <= max_retries:
                        log.warning(
                            "mcp_bridge_connect_retry",
                            server=config.name,
                            attempt=attempt,
                            max_retries=max_retries,
                            error=str(exc),
                        )
                        await asyncio.sleep(min(attempt * 2.0, 10.0))
                    else:
                        log.error(
                            "mcp_bridge_connect_failed",
                            server=config.name,
                            attempts=attempt,
                            error=str(exc),
                        )

            if not connected:
                log.warning(
                    "mcp_bridge_server_skipped",
                    server=config.name,
                    reason="all_retries_exhausted",
                )

        return all_proxies

    def register_tools(
        self, registry: ToolRegistry, proxies: list[MCPToolProxy] | None = None
    ) -> list[str]:
        """Register MCP tool proxies with a ToolRegistry.

        Args:
            registry: ToolRegistry instance.
            proxies: Specific proxies to register (defaults to all discovered).

        Returns:
            List of registered tool names.
        """
        tools_to_register = proxies if proxies is not None else self._proxies
        names: list[str] = []
        for proxy in tools_to_register:
            registry.register(proxy)
            names.append(proxy.schema.name)
        return names

    async def close_all(self) -> None:
        """Close all MCP server connections."""
        for name, conn in self._connections.items():
            try:
                await conn.close()
            except Exception as exc:
                log.warning("mcp_bridge_close_error", server=name, error=str(exc))
        self._connections.clear()
        self._proxies.clear()


# ── MCP Auto-Discovery ───────────────────────────────────────────────────────


def discover_mcp_servers() -> list[MCPServerConfig]:
    """Auto-discover MCP server configurations from system AI agent configs.

    Scans known config locations for Claude Desktop, Antigravity, and other
    AI tools. Returns a deduplicated list of MCPServerConfig objects.

    Discovered sources:
        - Claude Desktop: %APPDATA%/Claude/claude_desktop_config.json
        - Antigravity: %APPDATA%/Antigravity/User/mcp.json

    Returns:
        List of MCPServerConfig for all discovered servers.
    """
    import os
    from pathlib import Path

    configs: list[MCPServerConfig] = []
    seen_names: set[str] = set()

    appdata = os.environ.get("APPDATA", "")

    # ── Claude Desktop ───────────────────────────────────────────────────
    claude_config_path = Path(appdata) / "Claude" / "claude_desktop_config.json"
    if claude_config_path.exists():
        try:
            raw = json.loads(claude_config_path.read_text(encoding="utf-8"))
            servers = raw.get("mcpServers", {})
            for name, server_def in servers.items():
                if name in seen_names:
                    continue
                # Skip obsidian — ARA has its own direct REST integration
                if name == "obsidian":
                    log.info("mcp_discovery_skip", server=name, reason="ara_has_native_integration")
                    continue
                config = _parse_claude_desktop_server(name, server_def)
                if config is not None:
                    configs.append(config)
                    seen_names.add(name)
            log.info(
                "mcp_discovery_claude_desktop",
                path=str(claude_config_path),
                servers_found=len(servers),
                servers_added=len([c for c in configs if c.name in seen_names]),
            )
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("mcp_discovery_claude_desktop_error", error=str(exc))

    # ── Antigravity ──────────────────────────────────────────────────────
    antigravity_config_path = Path(appdata) / "Antigravity" / "User" / "mcp.json"
    if antigravity_config_path.exists():
        try:
            raw = json.loads(antigravity_config_path.read_text(encoding="utf-8"))
            servers = raw.get("servers", {})
            for name, server_def in servers.items():
                # Antigravity uses opaque names like "my-mcp-server-58771d19"
                # Derive a human-readable name from the URL if possible
                readable_name = _derive_server_name(name, server_def)
                if readable_name in seen_names:
                    continue
                config = _parse_antigravity_server(readable_name, server_def)
                if config is not None:
                    configs.append(config)
                    seen_names.add(readable_name)
            log.info(
                "mcp_discovery_antigravity",
                path=str(antigravity_config_path),
                servers_found=len(servers),
            )
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("mcp_discovery_antigravity_error", error=str(exc))

    log.info(
        "mcp_discovery_complete",
        total_servers=len(configs),
        server_names=[c.name for c in configs],
    )
    return configs


def _parse_claude_desktop_server(name: str, server_def: dict[str, Any]) -> MCPServerConfig | None:
    """Parse a Claude Desktop MCP server definition into MCPServerConfig."""
    command = server_def.get("command", "")
    args = server_def.get("args", [])
    env = server_def.get("env", {})

    if not command:
        log.warning("mcp_discovery_skip_no_command", server=name)
        return None

    return MCPServerConfig(
        name=name,
        transport="stdio",
        command=command,
        args=tuple(str(a) for a in args),
        env={str(k): str(v) for k, v in env.items()},
    )


def _parse_antigravity_server(name: str, server_def: dict[str, Any]) -> MCPServerConfig | None:
    """Parse an Antigravity MCP server definition into MCPServerConfig."""
    transport_type = server_def.get("type", "http")
    url = server_def.get("url", "")
    headers = server_def.get("headers", {})

    if transport_type == "http" and url:
        return MCPServerConfig(
            name=name,
            transport="http",
            url=url,
            env={str(k): str(v) for k, v in headers.items()},
        )

    log.warning("mcp_discovery_unsupported_transport", server=name, transport=transport_type)
    return None


def _derive_server_name(opaque_name: str, server_def: dict[str, Any]) -> str:
    """Derive a human-readable name from a server definition."""
    url = server_def.get("url", "")
    # Extract service name from URL (e.g., "https://mcp.context7.com/mcp" -> "context7")
    if "context7" in url:
        return "context7"
    if "figma" in url:
        return "figma"
    # Fall back to the opaque name
    return opaque_name
