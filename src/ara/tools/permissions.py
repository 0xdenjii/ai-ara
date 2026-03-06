"""Tool permission system — whitelist, deny list, confirmation, and audit logging.

Provides:
    - PermissionPolicy: Per-tool and per-category permission rules
    - PermissionChecker: Evaluates ToolCalls against policies
    - AuditLogger: SQLite-backed audit trail for all tool executions (P6-07)

Architecture:
    The PermissionChecker is injected into the ToolRegistry as the
    ``permission_checker`` callback.  It evaluates each tool call against
    a PermissionPolicy that can whitelist/deny specific tools, require
    confirmation for categories, and enforce rate limits.

    The AuditLogger is injected into the ToolRegistry as the ``audit_logger``
    callback.  It writes every execution (success or failure) to SQLite with
    full context for post-hoc review.

Usage:
    policy = PermissionPolicy.default()
    checker = PermissionChecker(settings, policy)
    audit = AuditLogger(settings.data_dir / settings.desktop.audit_db_name)
    await audit.connect()
    registry = ToolRegistry(settings, permission_checker=checker, audit_logger=audit.log)
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from ara.utils.logging import get_logger

if TYPE_CHECKING:
    from pathlib import Path

    from ara.config.settings import AraSettings
    from ara.tools.registry import ToolRiskLevel
    from ara.utils.types import ToolCall, ToolResult

log = get_logger("tool_permissions")


# ── Permission Policy ────────────────────────────────────────────────────────


@dataclass
class PermissionPolicy:
    """Declarative permission policy for tool execution.

    Attributes:
        allowed_tools: Explicit whitelist. If non-empty, ONLY these tools can execute.
        denied_tools: Explicit deny list. These tools are always blocked.
        allowed_categories: Whitelist by category. If non-empty, only these categories run.
        denied_categories: Deny list by category.
        max_calls_per_minute: Rate limit (0 = unlimited).
        allow_dangerous: Whether DANGEROUS risk-level tools can execute at all.
        require_confirmation_for_moderate: Whether MODERATE tools need confirmation.
    """

    allowed_tools: set[str] = field(default_factory=set)
    denied_tools: set[str] = field(default_factory=set)
    allowed_categories: set[str] = field(default_factory=set)
    denied_categories: set[str] = field(default_factory=set)
    max_calls_per_minute: int = 0
    allow_dangerous: bool = False
    require_confirmation_for_moderate: bool = True

    @classmethod
    def default(cls) -> PermissionPolicy:
        """Return the default permissive policy.

        All tools allowed, dangerous blocked, moderate requires confirmation.
        """
        return cls(
            allow_dangerous=False,
            require_confirmation_for_moderate=True,
        )

    @classmethod
    def locked_down(cls) -> PermissionPolicy:
        """Return a restrictive policy for untrusted environments."""
        return cls(
            allowed_categories={"desktop"},
            denied_tools={"run_command", "file_write", "file_delete"},
            allow_dangerous=False,
            require_confirmation_for_moderate=True,
            max_calls_per_minute=10,
        )


# ── Permission Checker ───────────────────────────────────────────────────────


class PermissionChecker:
    """Evaluates tool calls against a PermissionPolicy.

    This is the ``permission_checker`` callback injected into ToolRegistry.

    The callable signature is ``(tool_name: str, risk_level: ToolRiskLevel) -> bool``.
    Additional context (category) can be checked via the registry's tool schemas.

    Args:
        settings: Application settings.
        policy: The permission policy to enforce.
    """

    def __init__(self, settings: AraSettings, policy: PermissionPolicy) -> None:
        self._settings = settings
        self._policy = policy
        self._call_timestamps: list[float] = []
        self._tool_categories: dict[str, str] = {}

    def set_tool_category(self, tool_name: str, category: str) -> None:
        """Register a tool's category for category-level permission checks.

        Args:
            tool_name: Tool name.
            category: Category string.
        """
        self._tool_categories[tool_name] = category

    def __call__(self, tool_name: str, risk_level: ToolRiskLevel) -> bool:
        """Check if a tool call is permitted.

        Args:
            tool_name: Name of the tool being called.
            risk_level: Risk level of the tool.

        Returns:
            True if the call is allowed.
        """
        from ara.tools.registry import ToolRiskLevel

        policy = self._policy

        # Desktop globally disabled
        if not self._settings.desktop.enabled:
            log.debug("permission_denied_desktop_disabled", tool=tool_name)
            return False

        # Explicit deny list
        if tool_name in policy.denied_tools:
            log.debug("permission_denied_denylist", tool=tool_name)
            return False

        # Explicit whitelist (if set, tool must be in it)
        if policy.allowed_tools and tool_name not in policy.allowed_tools:
            log.debug("permission_denied_not_whitelisted", tool=tool_name)
            return False

        # Category checks
        category = self._tool_categories.get(tool_name, "")
        if category and policy.denied_categories and category in policy.denied_categories:
            log.debug("permission_denied_category", tool=tool_name, category=category)
            return False
        if policy.allowed_categories and category and category not in policy.allowed_categories:
            log.debug("permission_denied_category_not_allowed", tool=tool_name, category=category)
            return False

        # Risk level checks
        if risk_level == ToolRiskLevel.DANGEROUS and not policy.allow_dangerous:
            log.debug("permission_denied_dangerous", tool=tool_name)
            return False

        # Rate limiting
        if policy.max_calls_per_minute > 0:
            now = time.monotonic()
            # Purge old timestamps
            cutoff = now - 60.0
            self._call_timestamps = [t for t in self._call_timestamps if t > cutoff]
            if len(self._call_timestamps) >= policy.max_calls_per_minute:
                log.warning(
                    "permission_denied_rate_limit",
                    tool=tool_name,
                    limit=policy.max_calls_per_minute,
                )
                return False
            self._call_timestamps.append(now)

        return True

    @property
    def policy(self) -> PermissionPolicy:
        """Return the current policy."""
        return self._policy

    def update_policy(self, policy: PermissionPolicy) -> None:
        """Replace the permission policy.

        Args:
            policy: New policy.
        """
        self._policy = policy


# ── Audit Logger ─────────────────────────────────────────────────────────────

# SQLite schema for the audit trail.
_AUDIT_SCHEMA = """
CREATE TABLE IF NOT EXISTS tool_audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    call_id TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    parameters TEXT NOT NULL,
    success INTEGER NOT NULL,
    output TEXT NOT NULL DEFAULT '',
    error TEXT,
    execution_time_ms REAL NOT NULL DEFAULT 0.0,
    created_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_tool_audit_timestamp ON tool_audit(timestamp);
CREATE INDEX IF NOT EXISTS idx_tool_audit_tool_name ON tool_audit(tool_name);
CREATE INDEX IF NOT EXISTS idx_tool_audit_call_id ON tool_audit(call_id);
"""


class AuditLogger:
    """SQLite-backed audit trail for tool executions.

    Records every tool call with full parameters, results, timing, and errors.

    Args:
        db_path: Path to the SQLite database file.
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._conn: sqlite3.Connection | None = None

    async def connect(self) -> None:
        """Create the database and tables if needed."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(_AUDIT_SCHEMA)
        self._conn.commit()
        log.debug("audit_logger_connected", db_path=str(self._db_path))

    async def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def log(self, tool_call: ToolCall, result: ToolResult) -> None:
        """Log a tool execution to the audit trail.

        This is the ``audit_logger`` callback for ToolRegistry.

        Args:
            tool_call: The call that was made.
            result: The result produced.
        """
        if self._conn is None:
            log.warning("audit_logger_not_connected")
            return

        now = datetime.now(tz=UTC).isoformat()
        try:
            self._conn.execute(
                """INSERT INTO tool_audit
                (timestamp, call_id, tool_name, parameters, success, output, error, execution_time_ms, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    now,
                    tool_call.call_id,
                    tool_call.tool_name,
                    json.dumps(tool_call.parameters),
                    1 if result.success else 0,
                    result.output[:10000],  # Truncate very long outputs
                    result.error,
                    result.execution_time_ms,
                    time.time(),
                ),
            )
            self._conn.commit()
        except sqlite3.Error as exc:
            log.error("audit_log_write_failed", error=str(exc))

    def query_recent(self, limit: int = 50) -> list[dict[str, Any]]:
        """Query the most recent audit entries.

        Args:
            limit: Maximum number of entries to return.

        Returns:
            List of audit entry dicts, newest first.
        """
        if self._conn is None:
            return []

        cursor = self._conn.execute(
            """SELECT timestamp, call_id, tool_name, parameters, success, output, error, execution_time_ms
            FROM tool_audit ORDER BY id DESC LIMIT ?""",
            (limit,),
        )
        rows = cursor.fetchall()
        return [
            {
                "timestamp": row[0],
                "call_id": row[1],
                "tool_name": row[2],
                "parameters": json.loads(row[3]),
                "success": bool(row[4]),
                "output": row[5],
                "error": row[6],
                "execution_time_ms": row[7],
            }
            for row in rows
        ]

    def query_by_tool(self, tool_name: str, limit: int = 20) -> list[dict[str, Any]]:
        """Query audit entries for a specific tool.

        Args:
            tool_name: Tool to filter by.
            limit: Max entries.

        Returns:
            List of audit entry dicts, newest first.
        """
        if self._conn is None:
            return []

        cursor = self._conn.execute(
            """SELECT timestamp, call_id, tool_name, parameters, success, output, error, execution_time_ms
            FROM tool_audit WHERE tool_name = ? ORDER BY id DESC LIMIT ?""",
            (tool_name, limit),
        )
        rows = cursor.fetchall()
        return [
            {
                "timestamp": row[0],
                "call_id": row[1],
                "tool_name": row[2],
                "parameters": json.loads(row[3]),
                "success": bool(row[4]),
                "output": row[5],
                "error": row[6],
                "execution_time_ms": row[7],
            }
            for row in rows
        ]

    def get_stats(self) -> dict[str, Any]:
        """Get audit statistics.

        Returns:
            Dict with total_calls, success_count, error_count, and per-tool breakdown.
        """
        if self._conn is None:
            return {"total_calls": 0, "success_count": 0, "error_count": 0, "by_tool": {}}

        cursor = self._conn.execute(
            "SELECT COUNT(*), SUM(success), SUM(CASE WHEN success=0 THEN 1 ELSE 0 END) FROM tool_audit"
        )
        row = cursor.fetchone()
        total = row[0] or 0
        successes = row[1] or 0
        errors = row[2] or 0

        # Per-tool breakdown
        cursor = self._conn.execute(
            "SELECT tool_name, COUNT(*), SUM(success) FROM tool_audit GROUP BY tool_name"
        )
        by_tool = {}
        for row in cursor.fetchall():
            by_tool[row[0]] = {"total": row[1], "successes": row[2] or 0}

        return {
            "total_calls": total,
            "success_count": successes,
            "error_count": errors,
            "by_tool": by_tool,
        }

    def clear(self) -> None:
        """Delete all audit entries.  Use with caution."""
        if self._conn is not None:
            self._conn.execute("DELETE FROM tool_audit")
            self._conn.commit()
