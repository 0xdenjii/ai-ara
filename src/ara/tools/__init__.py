"""Desktop automation and tools — registry, desktop control, system commands, media, MCP.

Modules:
    registry: BaseTool, ToolSchema, ToolRegistry — plugin system core
    permissions: PermissionPolicy, PermissionChecker, AuditLogger
    desktop: Mouse, keyboard, window, process tools (pyautogui/pywinauto/psutil)
    system: Command execution, file I/O, environment queries
    media: Volume control, media playback
    mcp: MCP server bridge for external tool ecosystems
"""

from ara.tools.desktop import (
    ClickTool,
    CloseApplicationTool,
    FocusWindowTool,
    ListProcessesTool,
    ListWindowsTool,
    OpenApplicationTool,
    OpenURLTool,
    PressKeyTool,
    ScreenshotTool,
    SystemInfoTool,
    TypeTextTool,
    register_desktop_tools,
)
from ara.tools.mcp import MCPBridge, MCPConnection, MCPServerConfig, MCPToolProxy
from ara.tools.media import (
    MediaControlTool,
    VolumeControlTool,
    register_media_tools,
)
from ara.tools.permissions import AuditLogger, PermissionChecker, PermissionPolicy
from ara.tools.registry import (
    BaseTool,
    ParameterType,
    ToolParameter,
    ToolRegistry,
    ToolRiskLevel,
    ToolSchema,
)
from ara.tools.system import (
    EnvironmentInfoTool,
    FileReadTool,
    FileWriteTool,
    RunCommandTool,
    register_system_tools,
)

__all__ = [
    "AuditLogger",
    "BaseTool",
    "ClickTool",
    "CloseApplicationTool",
    "EnvironmentInfoTool",
    "FileReadTool",
    "FileWriteTool",
    "FocusWindowTool",
    "ListProcessesTool",
    "ListWindowsTool",
    "MCPBridge",
    "MCPConnection",
    "MCPServerConfig",
    "MCPToolProxy",
    "MediaControlTool",
    "OpenApplicationTool",
    "OpenURLTool",
    "ParameterType",
    "PermissionChecker",
    "PermissionPolicy",
    "PressKeyTool",
    "RunCommandTool",
    "ScreenshotTool",
    "SystemInfoTool",
    "ToolParameter",
    "ToolRegistry",
    "ToolRiskLevel",
    "ToolSchema",
    "TypeTextTool",
    "VolumeControlTool",
    "register_desktop_tools",
    "register_media_tools",
    "register_system_tools",
]
