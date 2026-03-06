"""Smart notification queue for proactive features.

Queues notifications from all proactive subsystems (suggestions, recaps,
habits, mood, reviews) and delivers them at natural conversation pauses —
never interrupting the user mid-flow.

Architecture:
    - Priority-based queue (urgent → high → medium → low)
    - TTL-based expiration (stale notifications are pruned)
    - Cooldown enforcement (minimum gap between deliveries)
    - Voice pipeline state awareness (only deliver when IDLE)
    - Callbacks for delivery to WebSocket / voice / frontend

Usage:
    from ara.proactive.notifications import NotificationQueue

    queue = NotificationQueue(settings.proactive)

    # Enqueue from any proactive feature
    queue.enqueue(
        title="Daily Recap",
        message="You had 5 conversations today...",
        priority=NotificationPriority.MEDIUM,
        source="recap",
    )

    # Check if there's something to deliver (called from voice/API idle loop)
    notification = queue.try_deliver()
    if notification:
        # Send via WebSocket or speak aloud
        await send_to_user(notification)
"""

from __future__ import annotations

import threading
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import structlog

from ara.utils.errors import NotificationError
from ara.utils.types import Notification, NotificationPriority, VoicePipelineState

if TYPE_CHECKING:
    from collections.abc import Callable

    from ara.config.settings import ProactiveSettings

log = structlog.get_logger(component="notifications")

# Priority ordering — lower number = higher urgency
_PRIORITY_ORDER: dict[NotificationPriority, int] = {
    NotificationPriority.URGENT: 0,
    NotificationPriority.HIGH: 1,
    NotificationPriority.MEDIUM: 2,
    NotificationPriority.LOW: 3,
}


class NotificationQueue:
    """Priority-based notification queue with delivery timing awareness.

    Notifications are sorted by priority (urgent first), then by creation
    time (oldest first within same priority). Delivery respects:
    - Voice pipeline state (only deliver when user is idle)
    - Cooldown period (minimum gap between successive deliveries)
    - TTL expiration (stale notifications are auto-pruned)
    - Queue size limit (oldest low-priority notifications are evicted)

    Thread-safe via a reentrant lock on all mutations.
    """

    def __init__(self, settings: ProactiveSettings) -> None:
        self._max_size = settings.notification_queue_max
        self._ttl_hours = settings.notification_ttl_hours
        self._cooldown_s = settings.notification_cooldown_s
        self._queue: list[Notification] = []
        self._lock = threading.RLock()
        self._last_delivery_time: datetime | None = None
        self._delivery_callbacks: list[Callable[[Notification], None]] = []
        self._total_enqueued = 0
        self._total_delivered = 0
        self._total_expired = 0
        self._total_dismissed = 0

    # ── Public API ────────────────────────────────────────────────────────

    def enqueue(
        self,
        *,
        title: str,
        message: str,
        priority: NotificationPriority = NotificationPriority.MEDIUM,
        source: str = "",
        metadata: dict[str, object] | None = None,
    ) -> Notification:
        """Add a notification to the queue.

        Args:
            title: Short headline for the notification.
            message: Full notification text.
            priority: Urgency level (affects delivery order).
            source: Which proactive feature generated it.
            metadata: Additional context for rendering.

        Returns:
            The created Notification object.

        Raises:
            NotificationError: If the queue is full and cannot evict.
        """
        ttl_delta = timedelta(hours=self._ttl_hours)
        now = datetime.now()

        notification = Notification(
            priority=priority,
            title=title,
            message=message,
            source=source,
            created_at=now,
            expires_at=now + ttl_delta,
            metadata=metadata or {},
        )

        with self._lock:
            # Prune expired first to make room
            self._prune_expired()

            # If still at capacity, evict lowest-priority oldest notification
            if len(self._queue) >= self._max_size:
                evicted = self._evict_lowest()
                if not evicted:
                    raise NotificationError(
                        f"Notification queue full ({self._max_size} items), "
                        "cannot enqueue new notification",
                        details={"queue_size": self._max_size, "priority": priority.value},
                    )

            self._queue.append(notification)
            self._sort_queue()
            self._total_enqueued += 1

        log.info(
            "notification_enqueued",
            title=title,
            priority=priority.value,
            source=source,
            queue_size=len(self._queue),
        )

        return notification

    def try_deliver(
        self,
        voice_state: VoicePipelineState = VoicePipelineState.IDLE,
    ) -> Notification | None:
        """Try to deliver the next notification if conditions are met.

        Delivery conditions:
        1. Queue is non-empty (after pruning expired)
        2. Voice pipeline is IDLE (user not mid-conversation)
        3. Cooldown period has elapsed since last delivery
        4. For URGENT: skip cooldown check

        Args:
            voice_state: Current state of the voice pipeline.

        Returns:
            The delivered Notification, or None if conditions not met.
        """
        with self._lock:
            self._prune_expired()

            if not self._queue:
                return None

            next_notification = self._queue[0]

            # URGENT notifications bypass voice state and cooldown
            is_urgent = next_notification.priority == NotificationPriority.URGENT

            # Check voice state (only deliver when idle, unless urgent)
            if not is_urgent and voice_state != VoicePipelineState.IDLE:
                return None

            # Check cooldown (unless urgent)
            if not is_urgent and not self._cooldown_elapsed():
                return None

            # Deliver: remove from queue and mark
            self._queue.pop(0)
            now = datetime.now()

            # Notification is mutable, so we can update in place
            next_notification.delivered = True
            next_notification.delivered_at = now

            self._last_delivery_time = now
            self._total_delivered += 1

        log.info(
            "notification_delivered",
            notification_id=next_notification.notification_id,
            title=next_notification.title,
            priority=next_notification.priority.value,
            source=next_notification.source,
        )

        # Fire delivery callbacks (outside lock)
        for callback in self._delivery_callbacks:
            try:
                callback(next_notification)
            except Exception:
                log.warning(
                    "notification_callback_error",
                    notification_id=next_notification.notification_id,
                    exc_info=True,
                )

        return next_notification

    def dismiss(self, notification_id: str) -> bool:
        """Dismiss a notification by ID (user chose to ignore it).

        Args:
            notification_id: The notification to dismiss.

        Returns:
            True if found and dismissed, False if not in queue.
        """
        with self._lock:
            for i, n in enumerate(self._queue):
                if n.notification_id == notification_id:
                    n.dismissed = True
                    self._queue.pop(i)
                    self._total_dismissed += 1
                    log.info("notification_dismissed", notification_id=notification_id)
                    return True
        return False

    def peek(self) -> Notification | None:
        """Peek at the next notification without removing it."""
        with self._lock:
            self._prune_expired()
            return self._queue[0] if self._queue else None

    def get_pending(self) -> list[Notification]:
        """Get all pending (undelivered) notifications.

        Returns a copy of the queue with expired items pruned.
        """
        with self._lock:
            self._prune_expired()
            return list(self._queue)

    def clear(self) -> int:
        """Clear all pending notifications.

        Returns:
            Number of notifications cleared.
        """
        with self._lock:
            count = len(self._queue)
            self._queue.clear()
            log.info("notification_queue_cleared", count=count)
            return count

    def register_callback(self, callback: Callable[[Notification], None]) -> None:
        """Register a callback to be invoked on each delivery.

        Callbacks receive the delivered Notification object. They are called
        outside the queue lock, so they are safe for async operations.

        Args:
            callback: Function to call with the delivered notification.
        """
        self._delivery_callbacks.append(callback)

    def get_stats(self) -> dict[str, object]:
        """Get queue statistics for monitoring.

        Returns:
            Dict with queue_size, total_enqueued, total_delivered,
            total_expired, total_dismissed, priority_breakdown.
        """
        with self._lock:
            priority_breakdown: dict[str, int] = {}
            for n in self._queue:
                key = n.priority.value
                priority_breakdown[key] = priority_breakdown.get(key, 0) + 1

            return {
                "queue_size": len(self._queue),
                "total_enqueued": self._total_enqueued,
                "total_delivered": self._total_delivered,
                "total_expired": self._total_expired,
                "total_dismissed": self._total_dismissed,
                "priority_breakdown": priority_breakdown,
                "cooldown_active": not self._cooldown_elapsed(),
                "last_delivery_time": (
                    self._last_delivery_time.isoformat() if self._last_delivery_time else None
                ),
            }

    @property
    def size(self) -> int:
        """Current number of pending notifications."""
        return len(self._queue)

    @property
    def is_empty(self) -> bool:
        """Whether the queue has no pending notifications."""
        return len(self._queue) == 0

    # ── Internal Helpers ──────────────────────────────────────────────────

    def _sort_queue(self) -> None:
        """Sort queue by priority (urgent first), then by creation time (oldest first)."""
        self._queue.sort(
            key=lambda n: (_PRIORITY_ORDER.get(n.priority, 99), n.created_at),
        )

    def _prune_expired(self) -> None:
        """Remove notifications that have exceeded their TTL."""
        now = datetime.now()
        before_count = len(self._queue)
        self._queue = [n for n in self._queue if n.expires_at is None or n.expires_at > now]
        expired_count = before_count - len(self._queue)
        if expired_count > 0:
            self._total_expired += expired_count
            log.debug("notifications_pruned", expired=expired_count, remaining=len(self._queue))

    def _cooldown_elapsed(self) -> bool:
        """Check whether the cooldown period since last delivery has elapsed."""
        if self._last_delivery_time is None:
            return True
        elapsed = (datetime.now() - self._last_delivery_time).total_seconds()
        return elapsed >= self._cooldown_s

    def _evict_lowest(self) -> bool:
        """Evict the lowest-priority, oldest notification to make room.

        Returns:
            True if an item was evicted, False if queue is empty.
        """
        if not self._queue:
            return False

        # Find the lowest priority notification (last after sorting)
        self._queue.pop(-1)
        log.debug("notification_evicted", queue_size=len(self._queue))
        return True
