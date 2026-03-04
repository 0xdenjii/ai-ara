"""Global hotkey listener for voice pipeline activation.

Registers a system-wide hotkey (default: Ctrl+Space) that triggers the
voice pipeline to start listening. Uses the `keyboard` library for
cross-platform global hotkey capture on Windows.

The hotkey fires pipeline.activate() which transitions from IDLE to
LISTENING, bypassing wake word detection.

All configuration comes from AraSettings.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import keyboard

from ara.utils.errors import VoiceError
from ara.utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

log = get_logger("hotkey")


class HotkeyListener:
    """Global hotkey listener that triggers voice pipeline activation.

    Registers a system-wide keyboard shortcut. When pressed, calls the
    configured activation callback (typically pipeline.activate).

    The keyboard library runs its own background thread for event
    capture, so this class is non-blocking.

    Usage:
        listener = HotkeyListener(hotkey="ctrl+space")
        listener.set_callback(pipeline.activate)
        listener.start()
        # ... hotkey is now active
        listener.stop()
    """

    def __init__(self, hotkey: str) -> None:
        self._hotkey = hotkey
        self._callback: Callable[[], None] | None = None
        self._registered = False
        self._suppressed = False
        self._lock = threading.Lock()

    @property
    def hotkey(self) -> str:
        """The registered hotkey combination string."""
        return self._hotkey

    @property
    def is_registered(self) -> bool:
        """Whether the hotkey is currently registered and active."""
        return self._registered

    def set_callback(self, callback: Callable[[], None]) -> None:
        """Set the function to call when the hotkey is pressed.

        Args:
            callback: A callable (typically pipeline.activate) invoked
                      on each hotkey press. Must be thread-safe.
        """
        self._callback = callback

    def start(self) -> None:
        """Register the global hotkey.

        Raises:
            VoiceError: If the hotkey cannot be registered.
        """
        if self._registered:
            log.warning("hotkey_already_registered", hotkey=self._hotkey)
            return

        if self._callback is None:
            raise VoiceError(
                "No callback set — call set_callback() before start()",
                operation="hotkey_start",
                details={"hotkey": self._hotkey},
            )

        try:
            keyboard.add_hotkey(
                self._hotkey,
                self._on_hotkey_pressed,
                suppress=False,
                trigger_on_release=False,
            )
            self._registered = True
            log.info("hotkey_registered", hotkey=self._hotkey)
        except Exception as exc:
            raise VoiceError(
                f"Failed to register hotkey '{self._hotkey}': {exc}",
                operation="hotkey_start",
                details={"hotkey": self._hotkey},
            ) from exc

    def stop(self) -> None:
        """Unregister the global hotkey and clean up."""
        if not self._registered:
            return

        import contextlib

        with contextlib.suppress(KeyError, ValueError):
            keyboard.remove_hotkey(self._hotkey)

        self._registered = False
        log.info("hotkey_unregistered", hotkey=self._hotkey)

    def update_hotkey(self, new_hotkey: str) -> None:
        """Change the hotkey combination at runtime.

        Unregisters the old hotkey and registers the new one.

        Args:
            new_hotkey: New hotkey string (e.g. "ctrl+shift+space").

        Raises:
            VoiceError: If the new hotkey cannot be registered.
        """
        was_registered = self._registered

        if was_registered:
            self.stop()

        old_hotkey = self._hotkey
        self._hotkey = new_hotkey

        if was_registered:
            try:
                self.start()
            except VoiceError:
                # Restore old hotkey on failure
                self._hotkey = old_hotkey
                self.start()
                raise

        log.info("hotkey_updated", from_hotkey=old_hotkey, to_hotkey=new_hotkey)

    def _on_hotkey_pressed(self) -> None:
        """Internal handler invoked by the keyboard library on hotkey press."""
        with self._lock:
            if self._suppressed:
                return

        log.debug("hotkey_pressed", hotkey=self._hotkey)

        if self._callback is not None:
            try:
                self._callback()
            except Exception as exc:
                log.warning(
                    "hotkey_callback_error",
                    hotkey=self._hotkey,
                    error=str(exc),
                )

    def suppress(self) -> None:
        """Temporarily suppress hotkey callbacks without unregistering.

        Useful during text input or when the pipeline is already active.
        """
        with self._lock:
            self._suppressed = True
        log.debug("hotkey_suppressed", hotkey=self._hotkey)

    def unsuppress(self) -> None:
        """Resume hotkey callbacks after suppression."""
        with self._lock:
            self._suppressed = False
        log.debug("hotkey_unsuppressed", hotkey=self._hotkey)
