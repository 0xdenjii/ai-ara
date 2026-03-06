"""Online/offline state manager — network detection, user toggle, connectivity events.

Manages whether ARA uses internet features. Provides:
- Network connectivity detection (ping check)
- User-controlled toggle (persisted via settings)
- State change callbacks for UI notifications
- Graceful degradation when offline

Architecture:
    - Connectivity state is a combination of: user toggle AND network availability
    - User can disable internet even when network is available (privacy mode)
    - Network checks happen on-demand or on a background interval
    - State changes trigger registered callbacks (for UI updates)

Usage:
    manager = ConnectivityManager(settings)
    await manager.start()
    if manager.is_online:
        results = await search(...)
    await manager.stop()
"""

from __future__ import annotations

import asyncio
import contextlib
import socket
import time
from collections.abc import Awaitable, Callable
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from ara.utils.errors import OfflineError
from ara.utils.logging import get_logger

if TYPE_CHECKING:
    from ara.config.settings import InternetSettings

log = get_logger("connectivity")


class ConnectivityState(StrEnum):
    """Internet connectivity state."""

    ONLINE = "online"  # User enabled + network available
    OFFLINE_BY_USER = "offline_by_user"  # User disabled internet toggle
    OFFLINE_NO_NETWORK = "offline_no_network"  # Network unavailable
    CHECKING = "checking"  # Currently verifying connectivity
    UNKNOWN = "unknown"  # Not yet checked


# Well-known hosts for connectivity checks — try multiple to avoid false negatives.
# These are DNS-only checks (socket connect to port 53) — no HTTP traffic.
_CHECK_HOSTS = [
    ("8.8.8.8", 53),  # Google DNS
    ("1.1.1.1", 53),  # Cloudflare DNS
    ("208.67.222.222", 53),  # OpenDNS
]

# Default values used when ConnectivityManager is constructed without settings
_DEFAULT_CHECK_TIMEOUT_S = 3.0
_DEFAULT_MIN_CHECK_INTERVAL_S = 10.0
_DEFAULT_BACKGROUND_CHECK_INTERVAL_S = 60.0


StateChangeCallback = Callable[[ConnectivityState], Awaitable[None] | None]


class ConnectivityManager:
    """Manages online/offline state with network detection and user toggle.

    Args:
        settings: Internet configuration (includes internet_enabled toggle).
        check_interval_s: How often to check connectivity in background.
            Set to 0 to disable background checks.
    """

    def __init__(
        self,
        settings: InternetSettings,
        *,
        check_interval_s: float | None = None,
    ) -> None:
        self._settings = settings
        self._check_interval_s = (
            check_interval_s
            if check_interval_s is not None
            else getattr(
                settings, "connectivity_background_interval_s", _DEFAULT_BACKGROUND_CHECK_INTERVAL_S
            )
        )
        self._check_timeout_s: float = getattr(
            settings, "connectivity_check_timeout_s", _DEFAULT_CHECK_TIMEOUT_S
        )
        self._min_check_interval_s: float = getattr(
            settings, "connectivity_check_interval_s", _DEFAULT_MIN_CHECK_INTERVAL_S
        )
        self._state = ConnectivityState.UNKNOWN
        self._network_available = False
        self._user_enabled = settings.internet_enabled
        self._last_check_time: float = 0.0
        self._callbacks: list[StateChangeCallback] = []
        self._background_task: asyncio.Task[None] | None = None
        self._running = False
        self._check_fn: Callable[[], Awaitable[bool]] | None = None

    @property
    def state(self) -> ConnectivityState:
        """Current connectivity state."""
        return self._state

    @property
    def is_online(self) -> bool:
        """Whether ARA can use internet features right now."""
        return self._state == ConnectivityState.ONLINE

    @property
    def is_user_enabled(self) -> bool:
        """Whether the user has enabled internet access."""
        return self._user_enabled

    @property
    def is_network_available(self) -> bool:
        """Whether the network is physically available (last check result)."""
        return self._network_available

    def set_check_fn(self, fn: Callable[[], Awaitable[bool]]) -> None:
        """Override the default connectivity check function (for testing).

        Args:
            fn: Async function that returns True if network is available.
        """
        self._check_fn = fn

    def on_state_change(self, callback: StateChangeCallback) -> None:
        """Register a callback for connectivity state changes.

        Args:
            callback: Called with the new ConnectivityState when state changes.
                Can be sync or async.
        """
        self._callbacks.append(callback)

    def remove_callback(self, callback: StateChangeCallback) -> None:
        """Remove a previously registered state change callback.

        Args:
            callback: The callback to remove.
        """
        self._callbacks = [cb for cb in self._callbacks if cb is not callback]

    async def start(self) -> None:
        """Start the connectivity manager. Performs initial check and starts background monitoring.

        Does nothing if already running.
        """
        if self._running:
            return

        self._running = True
        log.info("connectivity_manager_started", user_enabled=self._user_enabled)

        # Initial connectivity check
        await self.check_now()

        # Start background monitoring if interval > 0
        if self._check_interval_s > 0:
            self._background_task = asyncio.create_task(self._background_loop())

    async def stop(self) -> None:
        """Stop background connectivity monitoring.

        Does nothing if not running.
        """
        if not self._running:
            return

        self._running = False
        if self._background_task is not None:
            self._background_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._background_task
            self._background_task = None

        log.info("connectivity_manager_stopped")

    async def check_now(self) -> ConnectivityState:
        """Force an immediate connectivity check.

        Returns:
            The current ConnectivityState after checking.
        """
        old_state = self._state

        # Respect minimum check interval to avoid spamming
        now = time.monotonic()
        if (
            now - self._last_check_time
        ) < self._min_check_interval_s and old_state != ConnectivityState.UNKNOWN:
            return self._state

        self._last_check_time = now

        # Check user toggle first
        if not self._user_enabled:
            self._state = ConnectivityState.OFFLINE_BY_USER
            if self._state != old_state:
                await self._notify_state_change()
            return self._state

        # Check network availability
        self._state = ConnectivityState.CHECKING
        self._network_available = await self._check_network()

        if self._network_available:
            self._state = ConnectivityState.ONLINE
        else:
            self._state = ConnectivityState.OFFLINE_NO_NETWORK

        if self._state != old_state:
            log.info(
                "connectivity_state_changed",
                old_state=old_state.value,
                new_state=self._state.value,
                network_available=self._network_available,
            )
            await self._notify_state_change()

        return self._state

    async def set_user_enabled(self, enabled: bool) -> ConnectivityState:
        """Toggle internet access on/off (user control).

        Args:
            enabled: Whether to enable internet access.

        Returns:
            The new ConnectivityState.
        """
        old_enabled = self._user_enabled
        self._user_enabled = enabled
        self._settings.internet_enabled = enabled

        if old_enabled == enabled:
            return self._state

        log.info("internet_toggle_changed", enabled=enabled)

        # Force immediate re-check
        self._last_check_time = 0.0
        return await self.check_now()

    def require_online(self, operation: str) -> None:
        """Assert that we're currently online. Raises OfflineError if not.

        Use this at the start of any internet-dependent operation.

        Args:
            operation: Name of the operation attempting internet access.

        Raises:
            OfflineError: If currently offline.
        """
        if not self.is_online:
            raise OfflineError(operation)

    async def _check_network(self) -> bool:
        """Check network availability by attempting socket connections.

        Tries multiple well-known hosts. Returns True if any succeed.

        Returns:
            True if network is available.
        """
        if self._check_fn is not None:
            try:
                return await self._check_fn()
            except Exception as exc:
                log.debug("custom_check_fn_failed", error=str(exc))
                return False

        loop = asyncio.get_event_loop()

        for host, port in _CHECK_HOSTS:
            try:
                result = await asyncio.wait_for(
                    loop.run_in_executor(None, _socket_check, host, port),
                    timeout=self._check_timeout_s,
                )
                if result:
                    return True
            except Exception as exc:
                log.debug("connectivity_check_failed", host=host, port=port, error=str(exc))
                continue

        return False

    async def _notify_state_change(self) -> None:
        """Notify all registered callbacks of a state change."""
        for callback in self._callbacks:
            try:
                result = callback(self._state)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as exc:
                log.warning("state_change_callback_error", error=str(exc))

    async def _background_loop(self) -> None:
        """Background task that periodically checks connectivity."""
        while self._running:
            try:
                await asyncio.sleep(self._check_interval_s)
                if self._running:
                    # Reset last check time to allow the check to proceed
                    self._last_check_time = 0.0
                    await self.check_now()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.warning("background_check_error", error=str(exc))

    def get_status(self) -> dict[str, Any]:
        """Get current connectivity status as a dict (for API responses).

        Returns:
            Dict with state, user_enabled, network_available, and last_check_time.
        """
        return {
            "state": self._state.value,
            "user_enabled": self._user_enabled,
            "network_available": self._network_available,
            "last_check_time": self._last_check_time,
        }


def _socket_check(host: str, port: int) -> bool:
    """Synchronous socket connectivity check.

    Args:
        host: IP address to connect to.
        port: Port number.

    Returns:
        True if connection succeeds.
    """
    try:
        sock = socket.create_connection((host, port), timeout=_DEFAULT_CHECK_TIMEOUT_S)
        sock.close()
        return True
    except (OSError, TimeoutError):
        return False
