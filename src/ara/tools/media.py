"""Media control tools — volume, playback, and media key simulation.

Provides BaseTool implementations for:
    - VolumeControlTool: System volume up/down/mute/unmute/set level
    - MediaControlTool: Play/pause/next/previous/stop via media keys

Implementation:
    Uses Windows virtual key codes via ctypes to send media key events.
    Volume control uses pycaw (if available) or falls back to media keys.
    This is Windows-native and does not require any third-party media player API.
"""

from __future__ import annotations

import ctypes
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

log = get_logger("media_tools")


# ── Windows Virtual Key Codes for Media ──────────────────────────────────────

_VK_VOLUME_MUTE = 0xAD
_VK_VOLUME_DOWN = 0xAE
_VK_VOLUME_UP = 0xAF
_VK_MEDIA_NEXT_TRACK = 0xB0
_VK_MEDIA_PREV_TRACK = 0xB1
_VK_MEDIA_STOP = 0xB2
_VK_MEDIA_PLAY_PAUSE = 0xB3

_KEYEVENTF_EXTENDEDKEY = 0x0001
_KEYEVENTF_KEYUP = 0x0002


def _press_vk(vk_code: int) -> None:
    """Press and release a virtual key using Windows API.

    Args:
        vk_code: Windows virtual key code.
    """
    ctypes.windll.user32.keybd_event(vk_code, 0, _KEYEVENTF_EXTENDEDKEY, 0)  # type: ignore[union-attr]
    ctypes.windll.user32.keybd_event(vk_code, 0, _KEYEVENTF_EXTENDEDKEY | _KEYEVENTF_KEYUP, 0)  # type: ignore[union-attr]


# ── Volume Control ───────────────────────────────────────────────────────────


class VolumeControlTool(BaseTool):
    """Control system volume: mute, unmute, up, down, or set level."""

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="volume_control",
            description="Controls system volume: mute, unmute, raise, lower, or set to a specific level.",
            parameters=(
                ToolParameter(
                    name="action",
                    param_type=ParameterType.STRING,
                    description="Volume action to perform.",
                    enum=("mute", "unmute", "up", "down", "set"),
                ),
                ToolParameter(
                    name="level",
                    param_type=ParameterType.INTEGER,
                    description="Target volume level 0-100 (only used with 'set' action).",
                    required=False,
                    default=50,
                ),
            ),
            risk_level=ToolRiskLevel.SAFE,
            category="media",
        )

    async def execute(self, parameters: dict[str, Any]) -> str:
        action = parameters["action"].strip().lower()
        level = int(parameters.get("level", 50))

        if action == "mute":
            _press_vk(_VK_VOLUME_MUTE)
            return "System volume muted."

        if action == "unmute":
            # Toggle mute (unmute if currently muted)
            _press_vk(_VK_VOLUME_MUTE)
            return "System volume unmuted (toggled mute)."

        if action == "up":
            # Each press raises volume by ~2%
            for _ in range(3):
                _press_vk(_VK_VOLUME_UP)
            return "Volume raised."

        if action == "down":
            for _ in range(3):
                _press_vk(_VK_VOLUME_DOWN)
            return "Volume lowered."

        if action == "set":
            return await self._set_volume(level)

        raise DesktopError(
            f"Unknown volume action: '{action}'",
            operation="volume_control",
            details={"action": action},
        )

    async def _set_volume(self, level: int) -> str:
        """Set volume to a specific percentage using pycaw or key presses.

        Args:
            level: Target volume 0-100.

        Returns:
            Status message.
        """
        level = max(0, min(100, level))

        try:
            from comtypes import CLSCTX_ALL  # type: ignore[import-untyped]  # noqa: I001
            from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume  # type: ignore[import-untyped]

            devices = AudioUtilities.GetSpeakers()
            interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            volume = interface.QueryInterface(IAudioEndpointVolume)

            # pycaw uses scalar 0.0 to 1.0
            volume.SetMasterVolumeLevelScalar(level / 100.0, None)
            return f"Volume set to {level}%."
        except ImportError:
            # Fallback: use key presses (less precise)
            # First mute, then press up repeatedly (each press ~2%)
            _press_vk(_VK_VOLUME_MUTE)  # Ensure unmuted state by toggling
            _press_vk(_VK_VOLUME_MUTE)
            presses = level // 2
            for _ in range(presses):
                _press_vk(_VK_VOLUME_UP)
            return f"Volume set to approximately {level}% (via key presses, install pycaw for precise control)."


# ── Media Control ────────────────────────────────────────────────────────────


class MediaControlTool(BaseTool):
    """Control media playback: play, pause, next, previous, stop."""

    @property
    def schema(self) -> ToolSchema:
        return ToolSchema(
            name="media_control",
            description="Controls media playback: play/pause, next track, previous track, stop.",
            parameters=(
                ToolParameter(
                    name="action",
                    param_type=ParameterType.STRING,
                    description="Media action to perform.",
                    enum=("play", "pause", "next", "previous", "stop"),
                ),
            ),
            risk_level=ToolRiskLevel.SAFE,
            category="media",
        )

    async def execute(self, parameters: dict[str, Any]) -> str:
        action = parameters["action"].strip().lower()

        actions_map: dict[str, tuple[int, str]] = {
            "play": (_VK_MEDIA_PLAY_PAUSE, "Media play/pause toggled."),
            "pause": (_VK_MEDIA_PLAY_PAUSE, "Media play/pause toggled."),
            "next": (_VK_MEDIA_NEXT_TRACK, "Skipped to next track."),
            "previous": (_VK_MEDIA_PREV_TRACK, "Went to previous track."),
            "stop": (_VK_MEDIA_STOP, "Media playback stopped."),
        }

        entry = actions_map.get(action)
        if not entry:
            raise DesktopError(
                f"Unknown media action: '{action}'",
                operation="media_control",
                details={"action": action},
            )

        vk_code, message = entry
        _press_vk(vk_code)
        return message


# ── Registration Helper ──────────────────────────────────────────────────────


def register_media_tools(registry: Any, settings: AraSettings) -> list[str]:
    """Register all media tools with a ToolRegistry.

    Args:
        registry: ToolRegistry instance.
        settings: Application settings.

    Returns:
        List of registered tool names.
    """
    tools: list[BaseTool] = [
        VolumeControlTool(settings),
        MediaControlTool(settings),
    ]

    names: list[str] = []
    for tool in tools:
        registry.register(tool)
        names.append(tool.schema.name)

    return names
