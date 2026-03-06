"""Desktop automation tools — mouse, keyboard, window management, process monitoring.

Provides BaseTool implementations for:
    - OpenApplicationTool: Launch apps by name using subprocess + pywinauto
    - CloseApplicationTool: Close running applications
    - TypeTextTool: Type text at cursor position
    - PressKeyTool: Press keyboard shortcuts
    - ClickTool: Click at screen coordinates
    - ScreenshotTool: Capture screen to file
    - ListWindowsTool: List open windows
    - ListProcessesTool: List running processes

All tools are Windows-native and use pyautogui, pywinauto, keyboard, and psutil.
"""

from __future__ import annotations

import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import psutil
import pyautogui

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

log = get_logger("desktop_tools")

# Disable pyautogui fail-safe (moves mouse to corner to abort)
# We handle safety through our permission system instead.
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.1  # 100ms pause between pyautogui calls


# ── Known Application Registry ──────────────────────────────────────────────
# Maps common app names to their executables on Windows.
# This avoids relying on PATH alone and handles common aliases.

_APP_REGISTRY: dict[str, str] = {
    "notepad": "notepad.exe",
    "calculator": "calc.exe",
    "calc": "calc.exe",
    "paint": "mspaint.exe",
    "explorer": "explorer.exe",
    "file explorer": "explorer.exe",
    "cmd": "cmd.exe",
    "command prompt": "cmd.exe",
    "terminal": "wt.exe",
    "windows terminal": "wt.exe",
    "powershell": "powershell.exe",
    "task manager": "taskmgr.exe",
    "taskmgr": "taskmgr.exe",
    "settings": "ms-settings:",
    "control panel": "control.exe",
    "snipping tool": "SnippingTool.exe",
    "wordpad": "wordpad.exe",
    "chrome": "chrome.exe",
    "google chrome": "chrome.exe",
    "firefox": "firefox.exe",
    "edge": "msedge.exe",
    "microsoft edge": "msedge.exe",
    "code": "code.exe",
    "vscode": "code.exe",
    "visual studio code": "code.exe",
    "spotify": "spotify.exe",
    "discord": "discord.exe",
    "slack": "slack.exe",
    "teams": "ms-teams.exe",
    "microsoft teams": "ms-teams.exe",
    "obs": "obs64.exe",
    "obs studio": "obs64.exe",
    "steam": "steam.exe",
    "brave": "brave.exe",
    "brave browser": "brave.exe",
}


# ── Open Application ────────────────────────────────────────────────────────


class OpenApplicationTool(BaseTool):
    """Launch an application by name."""

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="open_application",
            description="Opens/launches an application by name.",
            parameters=(
                ToolParameter(
                    name="name",
                    param_type=ParameterType.STRING,
                    description="Application name (e.g. 'chrome', 'notepad', 'vscode').",
                ),
            ),
            risk_level=ToolRiskLevel.SAFE,
            category="desktop",
        )

    async def execute(self, parameters: dict[str, Any]) -> str:
        name = parameters["name"].strip().lower()

        # Check known registry
        executable = _APP_REGISTRY.get(name)

        if executable:
            # URI-based launchers (e.g. ms-settings:)
            if executable.endswith(":"):
                try:
                    os.startfile(executable)
                    return f"Opened '{name}' via URI handler."
                except OSError as exc:
                    raise DesktopError(
                        f"Failed to open '{name}' via URI: {exc}",
                        operation="open_application",
                        details={"app": name, "uri": executable},
                    ) from exc
            else:
                try:
                    subprocess.Popen(
                        [executable],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        creationflags=subprocess.DETACHED_PROCESS,
                    )
                    return f"Launched '{name}' ({executable})."
                except FileNotFoundError:
                    # Fall through to search
                    pass
                except OSError as exc:
                    raise DesktopError(
                        f"Failed to launch '{executable}': {exc}",
                        operation="open_application",
                        details={"app": name, "executable": executable},
                    ) from exc

        # Try os.startfile as fallback (uses Windows file associations)
        try:
            os.startfile(name)
            return f"Opened '{name}' via Windows file association."
        except OSError as exc:
            raise DesktopError(
                f"Could not find or open application '{name}': {exc}",
                operation="open_application",
                details={"app": name},
            ) from exc


# ── Close Application ────────────────────────────────────────────────────────


class CloseApplicationTool(BaseTool):
    """Close a running application by name."""

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="close_application",
            description="Closes a running application by name. Sends a graceful termination signal.",
            parameters=(
                ToolParameter(
                    name="name",
                    param_type=ParameterType.STRING,
                    description="Application name or process name to close.",
                ),
            ),
            risk_level=ToolRiskLevel.MODERATE,
            category="desktop",
        )

    async def execute(self, parameters: dict[str, Any]) -> str:
        name = parameters["name"].strip().lower()
        closed_count = 0

        # Build list of process name patterns to search for
        patterns = [name]
        executable = _APP_REGISTRY.get(name)
        if executable and not executable.endswith(":"):
            patterns.append(executable.lower())

        for proc in psutil.process_iter(["name", "pid"]):
            try:
                proc_name = (proc.info["name"] or "").lower()
                if any(pat in proc_name for pat in patterns):
                    proc.terminate()
                    closed_count += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        if closed_count == 0:
            return f"No running process found matching '{name}'."
        return f"Closed {closed_count} process(es) matching '{name}'."


# ── Type Text ────────────────────────────────────────────────────────────────


class TypeTextTool(BaseTool):
    """Type text at the current cursor position."""

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="type_text",
            description="Types text at the current cursor position using keyboard simulation.",
            parameters=(
                ToolParameter(
                    name="text",
                    param_type=ParameterType.STRING,
                    description="The text to type.",
                ),
                ToolParameter(
                    name="interval",
                    param_type=ParameterType.FLOAT,
                    description="Seconds between each character (for realistic typing).",
                    required=False,
                    default=0.0,
                ),
            ),
            risk_level=ToolRiskLevel.MODERATE,
            category="desktop",
        )

    async def execute(self, parameters: dict[str, Any]) -> str:
        text = parameters["text"]
        interval = float(parameters.get("interval", 0.0))

        char_count = len(text)
        if char_count > 10000:
            raise DesktopError(
                f"Text too long ({char_count} chars, max 10000)",
                operation="type_text",
                details={"length": char_count},
            )

        pyautogui.typewrite(text, interval=interval) if text.isascii() else pyautogui.write(text)
        return f"Typed {char_count} characters."


# ── Press Key ────────────────────────────────────────────────────────────────


class PressKeyTool(BaseTool):
    """Press a keyboard shortcut or key combination."""

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="press_key",
            description="Presses a keyboard shortcut (e.g. 'ctrl+c', 'alt+tab', 'enter').",
            parameters=(
                ToolParameter(
                    name="keys",
                    param_type=ParameterType.STRING,
                    description="Key combination separated by '+' (e.g. 'ctrl+shift+s', 'alt+f4').",
                ),
            ),
            risk_level=ToolRiskLevel.MODERATE,
            category="desktop",
        )

    async def execute(self, parameters: dict[str, Any]) -> str:
        keys_str = parameters["keys"].strip().lower()
        keys = [k.strip() for k in keys_str.split("+")]

        # Map common aliases
        key_map = {
            "ctrl": "ctrl",
            "control": "ctrl",
            "alt": "alt",
            "shift": "shift",
            "win": "win",
            "windows": "win",
            "enter": "enter",
            "return": "enter",
            "esc": "escape",
            "escape": "escape",
            "tab": "tab",
            "space": "space",
            "backspace": "backspace",
            "delete": "delete",
            "del": "delete",
            "home": "home",
            "end": "end",
            "pageup": "pageup",
            "pagedown": "pagedown",
            "up": "up",
            "down": "down",
            "left": "left",
            "right": "right",
        }

        # Explicit lookup avoids type-checker str|None ambiguity from dict.get()
        mapped_keys: list[str] = [key_map[k] if k in key_map else k for k in keys]  # noqa: SIM401

        pyautogui.hotkey(*mapped_keys)
        return "Pressed key combination: " + "+".join(mapped_keys) + "."


# ── Click ────────────────────────────────────────────────────────────────────


class ClickTool(BaseTool):
    """Click at a screen position."""

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="click",
            description="Clicks at screen coordinates (x, y). Supports left/right/double click.",
            parameters=(
                ToolParameter(
                    name="x",
                    param_type=ParameterType.INTEGER,
                    description="X coordinate in pixels.",
                ),
                ToolParameter(
                    name="y",
                    param_type=ParameterType.INTEGER,
                    description="Y coordinate in pixels.",
                ),
                ToolParameter(
                    name="button",
                    param_type=ParameterType.STRING,
                    description="Mouse button: 'left', 'right', or 'middle'.",
                    required=False,
                    default="left",
                    enum=("left", "right", "middle"),
                ),
                ToolParameter(
                    name="clicks",
                    param_type=ParameterType.INTEGER,
                    description="Number of clicks (1 or 2 for double-click).",
                    required=False,
                    default=1,
                ),
            ),
            risk_level=ToolRiskLevel.SAFE,
            category="desktop",
        )

    async def execute(self, parameters: dict[str, Any]) -> str:
        x = int(parameters["x"])
        y = int(parameters["y"])
        button = str(parameters.get("button", "left"))
        clicks = int(parameters.get("clicks", 1))

        # Validate coordinates are on screen
        screen_w, screen_h = pyautogui.size()
        if not (0 <= x < screen_w and 0 <= y < screen_h):
            raise DesktopError(
                f"Click coordinates ({x}, {y}) out of screen bounds ({screen_w}x{screen_h})",
                operation="click",
                details={"x": x, "y": y, "screen": f"{screen_w}x{screen_h}"},
            )

        pyautogui.click(x=x, y=y, button=button, clicks=clicks)
        return f"Clicked at ({x}, {y}) with {button} button ({clicks}x)."


# ── Screenshot ───────────────────────────────────────────────────────────────


class ScreenshotTool(BaseTool):
    """Capture a screenshot of the screen."""

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="screenshot",
            description="Takes a screenshot and saves it to a file.",
            parameters=(
                ToolParameter(
                    name="path",
                    param_type=ParameterType.STRING,
                    description="File path to save the screenshot (optional, defaults to data/screenshots/).",
                    required=False,
                ),
                ToolParameter(
                    name="region",
                    param_type=ParameterType.STRING,
                    description="Region to capture as 'x,y,width,height' (optional, captures full screen).",
                    required=False,
                ),
            ),
            risk_level=ToolRiskLevel.SAFE,
            category="desktop",
        )

    async def execute(self, parameters: dict[str, Any]) -> str:
        path_str = parameters.get("path")

        if path_str:
            save_path = Path(path_str)
        else:
            screenshot_dir = self._settings.data_dir / self._settings.desktop.screenshot_dir
            screenshot_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = screenshot_dir / f"screenshot_{timestamp}.png"

        region: tuple[int, int, int, int] | None = None
        region_str = parameters.get("region")
        if region_str:
            parts = [int(p.strip()) for p in str(region_str).split(",")]
            if len(parts) == 4:
                region = (parts[0], parts[1], parts[2], parts[3])

        save_path.parent.mkdir(parents=True, exist_ok=True)
        screenshot = pyautogui.screenshot(region=region)
        screenshot.save(str(save_path))

        return f"Screenshot saved to {save_path} ({screenshot.width}x{screenshot.height}px)."


# ── Open URL ─────────────────────────────────────────────────────────────────


class OpenURLTool(BaseTool):
    """Open a URL in the default browser."""

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="open_url",
            description="Opens a URL in the default web browser.",
            parameters=(
                ToolParameter(
                    name="url",
                    param_type=ParameterType.STRING,
                    description="The URL to open.",
                ),
            ),
            risk_level=ToolRiskLevel.SAFE,
            category="desktop",
        )

    async def execute(self, parameters: dict[str, Any]) -> str:
        import webbrowser

        url = parameters["url"].strip()
        if not url.startswith(("http://", "https://", "file://")):
            url = "https://" + url

        webbrowser.open(url)
        return f"Opened URL: {url}"


# ── List Windows ─────────────────────────────────────────────────────────────


class ListWindowsTool(BaseTool):
    """List visible windows on the desktop."""

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="list_windows",
            description="Lists all visible windows with their titles and positions.",
            parameters=(),
            risk_level=ToolRiskLevel.SAFE,
            category="desktop",
        )

    async def execute(self, parameters: dict[str, Any]) -> str:
        try:
            from pywinauto import Desktop

            desktop = Desktop(backend="uia")
            windows = desktop.windows()
            visible = []
            for win in windows:
                title = win.window_text()
                if title.strip():
                    try:
                        rect = win.rectangle()
                        visible.append(
                            f"  - {title} (pos: {rect.left},{rect.top} size: {rect.width()}x{rect.height()})"
                        )
                    except Exception as exc:
                        log.debug("window_rect_unavailable", title=title[:50], error=str(exc))
                        visible.append(f"  - {title}")

            if not visible:
                return "No visible windows found."
            return f"Open windows ({len(visible)}):\n" + "\n".join(visible[:30])
        except ImportError:
            # Fallback: use psutil process names
            apps = set()
            for proc in psutil.process_iter(["name"]):
                try:
                    name = proc.info["name"]
                    if name:
                        apps.add(name)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            sorted_apps = sorted(apps)[:30]
            return f"Running processes ({len(sorted_apps)}):\n" + "\n".join(
                f"  - {a}" for a in sorted_apps
            )


# ── List Processes ───────────────────────────────────────────────────────────


class ListProcessesTool(BaseTool):
    """List running processes with resource usage."""

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="list_processes",
            description="Lists running processes sorted by CPU or memory usage.",
            parameters=(
                ToolParameter(
                    name="sort_by",
                    param_type=ParameterType.STRING,
                    description="Sort field: 'cpu' or 'memory'.",
                    required=False,
                    default="memory",
                    enum=("cpu", "memory"),
                ),
                ToolParameter(
                    name="limit",
                    param_type=ParameterType.INTEGER,
                    description="Number of processes to return.",
                    required=False,
                    default=15,
                ),
            ),
            risk_level=ToolRiskLevel.SAFE,
            category="desktop",
        )

    async def execute(self, parameters: dict[str, Any]) -> str:
        sort_by = str(parameters.get("sort_by", "memory"))
        limit = int(parameters.get("limit", 15))

        procs: list[dict[str, Any]] = []
        for proc in psutil.process_iter(["name", "pid", "cpu_percent", "memory_info"]):
            try:
                info = proc.info
                mem = info.get("memory_info")
                mem_mb = (mem.rss / 1024 / 1024) if mem else 0.0
                procs.append(
                    {
                        "name": info.get("name", "unknown"),
                        "pid": info.get("pid", 0),
                        "cpu": info.get("cpu_percent", 0.0) or 0.0,
                        "mem_mb": round(mem_mb, 1),
                    }
                )
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        sort_key = "mem_mb" if sort_by == "memory" else "cpu"
        procs.sort(key=lambda p: p[sort_key], reverse=True)
        top = procs[:limit]

        lines = [f"Top {len(top)} processes by {sort_by}:"]
        lines.extend(
            f"  {p['name']:<30s} PID:{p['pid']:<8d} CPU:{p['cpu']:>5.1f}% MEM:{p['mem_mb']:>8.1f}MB"
            for p in top
        )

        return "\n".join(lines)


# ── Focus Window ─────────────────────────────────────────────────────────────


class FocusWindowTool(BaseTool):
    """Bring a window to the foreground by title."""

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="focus_window",
            description="Brings a window to the foreground by its title (partial match supported).",
            parameters=(
                ToolParameter(
                    name="title",
                    param_type=ParameterType.STRING,
                    description="Window title or partial title to match.",
                ),
            ),
            risk_level=ToolRiskLevel.SAFE,
            category="desktop",
        )

    async def execute(self, parameters: dict[str, Any]) -> str:
        title = parameters["title"].strip().lower()

        try:
            from pywinauto import Desktop

            desktop = Desktop(backend="uia")
            windows = desktop.windows()
            for win in windows:
                win_title = win.window_text().lower()
                if title in win_title:
                    win.set_focus()
                    return f"Focused window: '{win.window_text()}'."

            return f"No window found matching '{title}'."
        except ImportError:
            raise DesktopError(
                "pywinauto is required for window management but not installed",
                operation="focus_window",
            ) from None


# ── Get System Info ──────────────────────────────────────────────────────────


class SystemInfoTool(BaseTool):
    """Get system resource information (CPU, RAM, disk, battery)."""

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="system_info",
            description="Returns current system resource usage: CPU, RAM, disk, and battery status.",
            parameters=(),
            risk_level=ToolRiskLevel.SAFE,
            category="desktop",
        )

    async def execute(self, parameters: dict[str, Any]) -> str:
        cpu_pct = psutil.cpu_percent(interval=0.5)
        cpu_count = psutil.cpu_count()
        mem = psutil.virtual_memory()
        disk = psutil.disk_usage("C:\\")
        uptime_sec = time.time() - psutil.boot_time()
        uptime_hours = round(uptime_sec / 3600, 1)

        lines = [
            f"CPU: {cpu_pct}% ({cpu_count} cores)",
            f"RAM: {mem.percent}% used ({round(mem.used / 1024**3, 1)}/{round(mem.total / 1024**3, 1)} GB)",
            f"Disk C: {disk.percent}% used ({round(disk.used / 1024**3, 1)}/{round(disk.total / 1024**3, 1)} GB)",
            f"Uptime: {uptime_hours} hours",
        ]

        try:
            battery = psutil.sensors_battery()
            if battery:
                plug = "plugged in" if battery.power_plugged else "on battery"
                lines.append(f"Battery: {battery.percent}% ({plug})")
        except AttributeError:
            pass

        return "\n".join(lines)


# ── Registration Helper ──────────────────────────────────────────────────────


def register_desktop_tools(registry: Any, settings: AraSettings) -> list[str]:
    """Register all desktop tools with a ToolRegistry.

    Args:
        registry: ToolRegistry instance.
        settings: Application settings.

    Returns:
        List of registered tool names.
    """
    tools: list[BaseTool] = [
        OpenApplicationTool(settings),
        CloseApplicationTool(settings),
        TypeTextTool(settings),
        PressKeyTool(settings),
        ClickTool(settings),
        ScreenshotTool(settings),
        OpenURLTool(settings),
        ListWindowsTool(settings),
        ListProcessesTool(settings),
        FocusWindowTool(settings),
        SystemInfoTool(settings),
    ]

    names: list[str] = []
    for tool in tools:
        registry.register(tool)
        names.append(tool.schema.name)

    return names
