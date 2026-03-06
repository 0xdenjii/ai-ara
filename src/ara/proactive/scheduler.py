"""Async scheduler for proactive features.

Runs background tasks on configurable intervals, triggering:
- Idle-state suggestion checks (every N minutes)
- Daily recap generation (at configured hour)
- Habit pattern scans (periodic)
- Mood recording after conversations
- Weekly/monthly reviews (on configured days)
- Notification delivery checks (continuous when idle)

Architecture:
    - Single asyncio task loop with sleep-based scheduling
    - Each feature has its own interval and last-run timestamp
    - Scheduler is started/stopped via start()/stop() lifecycle
    - All features are optional and individually configurable
    - Error isolation: one feature failing doesn't affect others

Usage:
    from ara.proactive.scheduler import ProactiveScheduler

    scheduler = ProactiveScheduler(
        settings=settings.proactive,
        notification_queue=queue,
        suggestion_engine=engine,
        mood_tracker=mood,
        habit_detector=habits,
        recap_generator=recap,
        review_generator=review,
    )

    await scheduler.start()    # begins background loop
    await scheduler.stop()     # graceful shutdown
"""

from __future__ import annotations

import asyncio
import contextlib
from datetime import datetime
from typing import TYPE_CHECKING

import structlog

from ara.utils.types import NotificationPriority, VoicePipelineState

if TYPE_CHECKING:
    from ara.config.settings import ObsidianSettings, ProactiveSettings
    from ara.core.llm import LLMClient
    from ara.obsidian.daily_briefing import BriefingGenerator
    from ara.obsidian.sync import ObsidianSync
    from ara.proactive.habits import HabitDetector
    from ara.proactive.mood import MoodTracker
    from ara.proactive.notifications import NotificationQueue
    from ara.proactive.recap import RecapGenerator
    from ara.proactive.review import ReviewGenerator
    from ara.proactive.suggestions import SuggestionEngine

log = structlog.get_logger(component="scheduler")

# Minimum interval between Obsidian sync checks (seconds)
_OBSIDIAN_SYNC_INTERVAL_S = 300  # 5 minutes (configurable via settings)


class ProactiveScheduler:
    """Async scheduler that coordinates all proactive feature triggers.

    Runs a single background asyncio task that periodically checks
    whether each proactive feature should fire based on its configured
    schedule. Features are isolated — one failure doesn't cascade.

    Lifecycle:
        scheduler = ProactiveScheduler(...)
        await scheduler.start()
        # ... runs in background ...
        await scheduler.stop()
    """

    def __init__(
        self,
        settings: ProactiveSettings,
        *,
        notification_queue: NotificationQueue,
        suggestion_engine: SuggestionEngine | None = None,
        mood_tracker: MoodTracker | None = None,
        habit_detector: HabitDetector | None = None,
        recap_generator: RecapGenerator | None = None,
        review_generator: ReviewGenerator | None = None,
        llm_client: LLMClient | None = None,
        obsidian_sync: ObsidianSync | None = None,
        obsidian_settings: ObsidianSettings | None = None,
        briefing_generator: BriefingGenerator | None = None,
    ) -> None:
        self._settings = settings
        self._enabled = settings.enabled
        self._queue = notification_queue
        self._suggestions = suggestion_engine
        self._mood = mood_tracker
        self._habits = habit_detector
        self._recap = recap_generator
        self._review = review_generator
        self._llm = llm_client

        # Obsidian integration (optional)
        self._obsidian_sync = obsidian_sync
        self._obsidian_settings = obsidian_settings
        self._briefing = briefing_generator

        # Scheduling state
        self._task: asyncio.Task[None] | None = None
        self._running = False
        self._voice_state = VoicePipelineState.IDLE
        self._last_user_activity: datetime = datetime.now()

        # Last-run timestamps for rate limiting
        self._last_suggestion_check: datetime | None = None
        self._last_habit_scan: datetime | None = None
        self._last_recap_date: str = ""  # Date string "YYYY-MM-DD"
        self._last_weekly_review_date: str = ""
        self._last_monthly_review_date: str = ""
        self._last_obsidian_sync: datetime | None = None
        self._last_morning_briefing_date: str = ""
        self._last_eod_recap_date: str = ""

    # ── Lifecycle ─────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start the proactive scheduler background task."""
        if not self._enabled:
            log.info("proactive_scheduler_disabled")
            return

        if self._running:
            log.warning("proactive_scheduler_already_running")
            return

        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        log.info("proactive_scheduler_started")

    async def stop(self) -> None:
        """Stop the proactive scheduler gracefully."""
        self._running = False
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        log.info("proactive_scheduler_stopped")

    @property
    def is_running(self) -> bool:
        """Whether the scheduler is currently running."""
        return self._running

    # ── External State Updates ────────────────────────────────────────────

    def update_voice_state(self, state: VoicePipelineState) -> None:
        """Update the current voice pipeline state.

        Called by the voice pipeline when state changes. Used to
        determine when it's appropriate to deliver notifications.

        Args:
            state: Current voice pipeline state.
        """
        self._voice_state = state

    def record_user_activity(self) -> None:
        """Record that the user was just active (sent a message, etc.).

        Used for idle detection — suggestions are only generated when
        the user has been idle for longer than the configured threshold.
        """
        self._last_user_activity = datetime.now()

    # ── Main Loop ─────────────────────────────────────────────────────────

    async def _run_loop(self) -> None:
        """Main scheduler loop — checks each feature on its schedule."""
        log.info("proactive_loop_started")

        while self._running:
            try:
                now = datetime.now()

                # Check notification delivery (always, when idle)
                if self._voice_state == VoicePipelineState.IDLE:
                    self._check_notification_delivery()

                # Check if user is idle (for suggestions)
                if self._is_user_idle(now):
                    await self._check_suggestions(now)

                # Check scheduled features
                await self._check_daily_recap(now)
                await self._check_habit_scan(now)
                await self._check_weekly_review(now)
                await self._check_monthly_review(now)

                # Obsidian integration checks
                await self._check_obsidian_sync(now)
                await self._check_morning_briefing(now)
                await self._check_eod_recap(now)

            except asyncio.CancelledError:
                break
            except Exception:
                log.warning("proactive_loop_error", exc_info=True)

            # Sleep between checks
            try:
                await asyncio.sleep(self._settings.loop_interval_s)
            except asyncio.CancelledError:
                break

        log.info("proactive_loop_exited")

    # ── Feature Checks ────────────────────────────────────────────────────

    def _check_notification_delivery(self) -> None:
        """Try to deliver the next queued notification."""
        notification = self._queue.try_deliver(voice_state=self._voice_state)
        if notification:
            log.info(
                "proactive_notification_delivered",
                title=notification.title,
                source=notification.source,
            )

    def _is_user_idle(self, now: datetime) -> bool:
        """Check if the user has been idle long enough for suggestions."""
        idle_seconds = (now - self._last_user_activity).total_seconds()
        return idle_seconds >= self._settings.idle_threshold_s

    async def _check_suggestions(self, now: datetime) -> None:
        """Generate and enqueue idle-state suggestions if due."""
        if self._suggestions is None:
            return

        # Rate limit: check at configured interval
        interval = self._settings.idle_check_interval_s
        if (
            self._last_suggestion_check is not None
            and (now - self._last_suggestion_check).total_seconds() < interval
        ):
            return

        try:
            suggestions = self._suggestions.generate()
            if suggestions:
                enqueued = self._suggestions.enqueue_suggestions(
                    suggestions, notification_queue=self._queue
                )
                log.info("proactive_suggestions_enqueued", count=enqueued)

            self._last_suggestion_check = now

        except Exception:
            log.warning("proactive_suggestion_check_failed", exc_info=True)

    async def _check_daily_recap(self, now: datetime) -> None:
        """Generate daily recap if it's the configured hour and not yet done today."""
        if self._recap is None:
            return

        today_str = now.strftime("%Y-%m-%d")
        if today_str == self._last_recap_date:
            return  # Already done today

        if (
            now.hour == self._settings.daily_recap_hour
            and now.minute >= self._settings.daily_recap_minute
        ):
            try:
                recap = await self._recap.generate_daily_recap(
                    llm_client=self._llm, target_date=now
                )
                if recap:
                    self._queue.enqueue(
                        title="Daily Recap",
                        message=recap.summary,
                        priority=NotificationPriority.MEDIUM,
                        source="recap",
                        metadata={
                            "recap_id": recap.recap_id,
                            "conversations": recap.conversation_count,
                            "topics": recap.topics_discussed,
                        },
                    )
                    log.info("daily_recap_enqueued", date=today_str)

                self._last_recap_date = today_str

            except Exception:
                log.warning("daily_recap_failed", exc_info=True)

    async def _check_habit_scan(self, now: datetime) -> None:
        """Scan for habit patterns and enqueue nudges for fading habits."""
        if self._habits is None:
            return

        if (
            self._last_habit_scan is not None
            and (now - self._last_habit_scan).total_seconds() < self._settings.habit_scan_interval_s
        ):
            return

        try:
            fading = self._habits.get_fading_habits()
            for pattern in fading[:2]:  # Max 2 nudges per scan
                days_since = (now - pattern.last_seen).days if pattern.last_seen else 0
                self._queue.enqueue(
                    title="Habit Check-in",
                    message=(
                        f"Your habit of {pattern.description} is fading — "
                        f"it's been {days_since} days. Want to get back on track?"
                    ),
                    priority=NotificationPriority.HIGH,
                    source="habits",
                    metadata={
                        "pattern_id": pattern.pattern_id,
                        "entity_name": pattern.entity_name,
                    },
                )

            self._last_habit_scan = now

        except Exception:
            log.warning("habit_scan_failed", exc_info=True)

    async def _check_weekly_review(self, now: datetime) -> None:
        """Generate weekly review if it's the right day and hour."""
        if self._review is None:
            return

        today_str = now.strftime("%Y-%m-%d")
        if today_str == self._last_weekly_review_date:
            return

        if self._review.is_weekly_review_day() and now.hour >= self._settings.weekly_review_hour:
            try:
                review = await self._review.generate_weekly_review(
                    llm_client=self._llm, end_date=now
                )
                if review:
                    self._queue.enqueue(
                        title="Weekly Review",
                        message=review.summary,
                        priority=NotificationPriority.MEDIUM,
                        source="review",
                        metadata={
                            "review_id": review.review_id,
                            "period": "weekly",
                            "entities": review.key_entities,
                        },
                    )
                    log.info("weekly_review_enqueued", date=today_str)

                self._last_weekly_review_date = today_str

            except Exception:
                log.warning("weekly_review_failed", exc_info=True)

    async def _check_monthly_review(self, now: datetime) -> None:
        """Generate monthly review if it's the right day and hour."""
        if self._review is None:
            return

        month_str = now.strftime("%Y-%m")
        if month_str == self._last_monthly_review_date:
            return

        if self._review.is_monthly_review_day() and now.hour >= self._settings.weekly_review_hour:
            try:
                review = await self._review.generate_monthly_review(
                    llm_client=self._llm, end_date=now
                )
                if review:
                    self._queue.enqueue(
                        title="Monthly Review",
                        message=review.summary,
                        priority=NotificationPriority.MEDIUM,
                        source="review",
                        metadata={
                            "review_id": review.review_id,
                            "period": "monthly",
                            "entities": review.key_entities,
                        },
                    )
                    log.info("monthly_review_enqueued", month=month_str)

                self._last_monthly_review_date = month_str

            except Exception:
                log.warning("monthly_review_failed", exc_info=True)

    # ── Obsidian Integration ─────────────────────────────────────────────

    async def _check_obsidian_sync(self, now: datetime) -> None:
        """Run incremental Obsidian vault sync if due."""
        if self._obsidian_sync is None:
            return

        sync_interval = (
            self._obsidian_settings.sync_interval_s
            if self._obsidian_settings is not None
            else _OBSIDIAN_SYNC_INTERVAL_S
        )

        if (
            self._last_obsidian_sync is not None
            and (now - self._last_obsidian_sync).total_seconds() < sync_interval
        ):
            return

        try:
            state = await self._obsidian_sync.incremental_sync()
            self._last_obsidian_sync = now

            if state.notes_changed > 0:
                log.info(
                    "obsidian_sync_changes_detected",
                    changed=state.notes_changed,
                    ingested=state.notes_ingested,
                )

        except Exception:
            log.warning("obsidian_sync_failed", exc_info=True)

    async def _check_morning_briefing(self, now: datetime) -> None:
        """Generate morning briefing if it's the configured hour."""
        if self._briefing is None or self._obsidian_settings is None:
            return

        if not self._obsidian_settings.morning_briefing_enabled:
            return

        today_str = now.strftime("%Y-%m-%d")
        if today_str == self._last_morning_briefing_date:
            return

        if now.hour == self._obsidian_settings.morning_briefing_hour:
            try:
                briefing = await self._briefing.generate_morning_briefing(
                    llm_client=self._llm, now=now
                )
                self._queue.enqueue(
                    title="Morning Briefing",
                    message=briefing.yesterday_summary,
                    priority=NotificationPriority.HIGH,
                    source="obsidian_briefing",
                    metadata={
                        "briefing_id": briefing.briefing_id,
                        "pending_tasks": len(briefing.pending_tasks),
                        "protocol_score": briefing.protocol_score,
                    },
                )
                self._last_morning_briefing_date = today_str
                log.info("morning_briefing_enqueued", date=today_str)

            except Exception:
                log.warning("morning_briefing_failed", exc_info=True)

    async def _check_eod_recap(self, now: datetime) -> None:
        """Generate end-of-day Obsidian recap if it's the configured hour."""
        if self._briefing is None or self._obsidian_settings is None:
            return

        if not self._obsidian_settings.end_of_day_recap_enabled:
            return

        today_str = now.strftime("%Y-%m-%d")
        if today_str == self._last_eod_recap_date:
            return

        if now.hour == self._obsidian_settings.end_of_day_recap_hour:
            try:
                recap_text = await self._briefing.generate_end_of_day_recap(
                    llm_client=self._llm, now=now
                )
                self._queue.enqueue(
                    title="End of Day Recap",
                    message=recap_text,
                    priority=NotificationPriority.MEDIUM,
                    source="obsidian_eod",
                )
                self._last_eod_recap_date = today_str
                log.info("eod_recap_enqueued", date=today_str)

            except Exception:
                log.warning("eod_recap_failed", exc_info=True)

    # ── Stats ─────────────────────────────────────────────────────────────

    def get_stats(self) -> dict[str, object]:
        """Get scheduler statistics for monitoring."""
        idle_s = (datetime.now() - self._last_user_activity).total_seconds()
        return {
            "running": self._running,
            "enabled": self._enabled,
            "voice_state": self._voice_state.value,
            "user_idle_seconds": round(idle_s, 1),
            "last_suggestion_check": (
                self._last_suggestion_check.isoformat() if self._last_suggestion_check else None
            ),
            "last_habit_scan": (
                self._last_habit_scan.isoformat() if self._last_habit_scan else None
            ),
            "last_recap_date": self._last_recap_date or None,
            "last_weekly_review": self._last_weekly_review_date or None,
            "last_monthly_review": self._last_monthly_review_date or None,
            "last_obsidian_sync": (
                self._last_obsidian_sync.isoformat() if self._last_obsidian_sync else None
            ),
            "last_morning_briefing": self._last_morning_briefing_date or None,
            "last_eod_recap": self._last_eod_recap_date or None,
            "notification_queue": self._queue.get_stats(),
        }
