"""Async HTTP client for Obsidian's Local REST API.

Wraps all vault operations (read, write, append, list, search) through the
Obsidian Local REST API plugin. Uses httpx directly rather than going through
MCP stdio protocol for lower latency and full control over error handling.

The Local REST API exposes:
    GET  /vault/{path}       — read file content
    PUT  /vault/{path}       — overwrite file content
    POST /vault/{path}       — append to file
    GET  /vault/             — list vault root
    POST /search/simple/     — full-text search across vault
    GET  /                   — server status / health check

All endpoints require Bearer token auth via the API key configured in
the Obsidian Local REST API plugin settings.

Usage:
    from ara.obsidian.client import ObsidianClient

    client = ObsidianClient(settings.obsidian)
    await client.connect()
    content = await client.read_note("10_Journal/Daily/2026/03-March/2026-03-06.md")
    await client.append_note("path/to/note.md", "\\n## New Section\\nContent here")
    await client.close()
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import httpx
import structlog

from ara.utils.errors import (
    ObsidianConnectionError,
    ObsidianReadError,
    ObsidianWriteError,
)

if TYPE_CHECKING:
    from ara.config.settings import ObsidianSettings

log = structlog.get_logger(component="obsidian.client")

# HTTP timeout for Obsidian API calls (seconds)
_REQUEST_TIMEOUT_S = 30.0


class ObsidianClient:
    """Async HTTP client for the Obsidian Local REST API.

    Provides typed methods for all vault operations with proper error handling,
    content hashing, and connection lifecycle management.

    Attributes:
        settings: ObsidianSettings configuration.
        connected: Whether the client has a verified connection.
    """

    def __init__(self, settings: ObsidianSettings) -> None:
        self._settings = settings
        self._client: httpx.AsyncClient | None = None
        self._connected = False
        self._base_url = settings.api_url.rstrip("/")

    @property
    def connected(self) -> bool:
        """Whether the client has a verified connection to Obsidian."""
        return self._connected

    @property
    def base_url(self) -> str:
        """The base URL of the Obsidian Local REST API."""
        return self._base_url

    async def connect(self) -> None:
        """Establish connection and verify Obsidian REST API is reachable.

        Raises:
            ObsidianConnectionError: If the API is unreachable or auth fails.
        """
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers={
                "Authorization": f"Bearer {self._settings.api_key}",
                "Accept": "text/markdown",
            },
            verify=self._settings.verify_ssl,
            timeout=httpx.Timeout(_REQUEST_TIMEOUT_S),
        )

        try:
            response = await self._client.get("/")
            if response.status_code == 200:
                self._connected = True
                log.info("obsidian_connected", base_url=self._base_url)
            elif response.status_code == 401:
                raise ObsidianConnectionError(
                    self._base_url, "Authentication failed — check API key"
                )
            else:
                raise ObsidianConnectionError(
                    self._base_url,
                    f"Unexpected status {response.status_code}: {response.text[:200]}",
                )
        except httpx.ConnectError as exc:
            raise ObsidianConnectionError(
                self._base_url,
                f"Connection refused — is Obsidian running with Local REST API? ({exc})",
            ) from exc
        except httpx.TimeoutException as exc:
            raise ObsidianConnectionError(self._base_url, f"Connection timed out ({exc})") from exc
        except ObsidianConnectionError:
            raise
        except Exception as exc:
            raise ObsidianConnectionError(self._base_url, f"Unexpected error: {exc}") from exc

    async def close(self) -> None:
        """Close the HTTP client and release resources."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        self._connected = False
        log.info("obsidian_disconnected")

    def _ensure_connected(self) -> httpx.AsyncClient:
        """Return the active client or raise if not connected."""
        if self._client is None or not self._connected:
            raise ObsidianConnectionError(self._base_url, "Not connected — call connect() first")
        return self._client

    # ── Read Operations ──────────────────────────────────────────────────

    async def read_note(self, path: str) -> str:
        """Read the content of a vault file.

        Args:
            path: Vault-relative path (e.g. "10_Journal/Daily/2026/03-March/2026-03-06.md").

        Returns:
            The raw markdown content of the file.

        Raises:
            ObsidianReadError: If the file doesn't exist or read fails.
        """
        client = self._ensure_connected()
        vault_path = f"/vault/{path.lstrip('/')}"
        try:
            response = await client.get(vault_path)
            if response.status_code == 200:
                return response.text
            if response.status_code == 404:
                raise ObsidianReadError(path, "File not found in vault")
            raise ObsidianReadError(path, f"HTTP {response.status_code}: {response.text[:200]}")
        except ObsidianReadError:
            raise
        except httpx.ConnectError as exc:
            raise ObsidianReadError(path, f"Connection lost: {exc}") from exc
        except httpx.TimeoutException as exc:
            raise ObsidianReadError(path, f"Request timed out: {exc}") from exc
        except Exception as exc:
            raise ObsidianReadError(path, f"Unexpected error: {exc}") from exc

    async def list_files(self, folder: str = "/") -> list[str]:
        """List files in a vault directory.

        Args:
            folder: Vault-relative folder path (e.g. "10_Journal/Daily/2026").
                   Use "/" for vault root.

        Returns:
            List of file paths relative to vault root.

        Raises:
            ObsidianReadError: If the directory doesn't exist or list fails.
        """
        client = self._ensure_connected()
        vault_path = f"/vault/{folder.strip('/')}/"
        try:
            response = await client.get(
                vault_path,
                headers={"Accept": "application/json"},
            )
            if response.status_code == 200:
                data = response.json()
                # API returns {"files": ["path1.md", "path2.md", ...]}
                if isinstance(data, dict) and "files" in data:
                    return sorted(data["files"])
                # Some versions return a flat list
                if isinstance(data, list):
                    return sorted(data)
                return []
            if response.status_code == 404:
                raise ObsidianReadError(folder, "Directory not found in vault")
            raise ObsidianReadError(folder, f"HTTP {response.status_code}: {response.text[:200]}")
        except ObsidianReadError:
            raise
        except httpx.ConnectError as exc:
            raise ObsidianReadError(folder, f"Connection lost: {exc}") from exc
        except httpx.TimeoutException as exc:
            raise ObsidianReadError(folder, f"Request timed out: {exc}") from exc
        except Exception as exc:
            raise ObsidianReadError(folder, f"Unexpected error: {exc}") from exc

    async def search(self, query: str) -> list[dict[str, str]]:
        """Full-text search across the vault.

        Args:
            query: Search query string.

        Returns:
            List of dicts with "filename" and "matches" keys.

        Raises:
            ObsidianReadError: If search fails.
        """
        client = self._ensure_connected()
        try:
            response = await client.post(
                "/search/simple/",
                headers={"Accept": "application/json"},
                content=query,
                # The API expects the query as the raw POST body
            )
            if response.status_code == 200:
                results = response.json()
                if isinstance(results, list):
                    return results
                return []
            raise ObsidianReadError(
                f"search:{query}",
                f"HTTP {response.status_code}: {response.text[:200]}",
            )
        except ObsidianReadError:
            raise
        except httpx.ConnectError as exc:
            raise ObsidianReadError(f"search:{query}", f"Connection lost: {exc}") from exc
        except httpx.TimeoutException as exc:
            raise ObsidianReadError(f"search:{query}", f"Request timed out: {exc}") from exc
        except Exception as exc:
            raise ObsidianReadError(f"search:{query}", f"Unexpected error: {exc}") from exc

    # ── Write Operations ─────────────────────────────────────────────────

    async def write_note(self, path: str, content: str) -> None:
        """Overwrite a vault file with new content (PUT).

        Creates the file if it doesn't exist. Overwrites if it does.

        Args:
            path: Vault-relative path.
            content: Full markdown content to write.

        Raises:
            ObsidianWriteError: If write fails.
        """
        client = self._ensure_connected()
        vault_path = f"/vault/{path.lstrip('/')}"
        try:
            response = await client.put(
                vault_path,
                content=content,
                headers={"Content-Type": "text/markdown"},
            )
            if response.status_code in (200, 204):
                log.debug("obsidian_write", path=path, bytes=len(content))
                return
            raise ObsidianWriteError(path, f"HTTP {response.status_code}: {response.text[:200]}")
        except ObsidianWriteError:
            raise
        except httpx.ConnectError as exc:
            raise ObsidianWriteError(path, f"Connection lost: {exc}") from exc
        except httpx.TimeoutException as exc:
            raise ObsidianWriteError(path, f"Request timed out: {exc}") from exc
        except Exception as exc:
            raise ObsidianWriteError(path, f"Unexpected error: {exc}") from exc

    async def append_note(self, path: str, content: str) -> None:
        """Append content to an existing vault file (POST).

        Args:
            path: Vault-relative path.
            content: Markdown content to append.

        Raises:
            ObsidianWriteError: If append fails (e.g. file doesn't exist).
        """
        client = self._ensure_connected()
        vault_path = f"/vault/{path.lstrip('/')}"
        try:
            response = await client.post(
                vault_path,
                content=content,
                headers={"Content-Type": "text/markdown"},
            )
            if response.status_code in (200, 204):
                log.debug("obsidian_append", path=path, bytes=len(content))
                return
            raise ObsidianWriteError(path, f"HTTP {response.status_code}: {response.text[:200]}")
        except ObsidianWriteError:
            raise
        except httpx.ConnectError as exc:
            raise ObsidianWriteError(path, f"Connection lost: {exc}") from exc
        except httpx.TimeoutException as exc:
            raise ObsidianWriteError(path, f"Request timed out: {exc}") from exc
        except Exception as exc:
            raise ObsidianWriteError(path, f"Unexpected error: {exc}") from exc

    async def delete_note(self, path: str) -> None:
        """Delete a vault file.

        Args:
            path: Vault-relative path.

        Raises:
            ObsidianWriteError: If delete fails.
        """
        client = self._ensure_connected()
        vault_path = f"/vault/{path.lstrip('/')}"
        try:
            response = await client.delete(vault_path)
            if response.status_code in (200, 204):
                log.debug("obsidian_delete", path=path)
                return
            if response.status_code == 404:
                log.debug("obsidian_delete_not_found", path=path)
                return  # Idempotent — already gone
            raise ObsidianWriteError(path, f"HTTP {response.status_code}: {response.text[:200]}")
        except ObsidianWriteError:
            raise
        except httpx.ConnectError as exc:
            raise ObsidianWriteError(path, f"Connection lost: {exc}") from exc
        except httpx.TimeoutException as exc:
            raise ObsidianWriteError(path, f"Request timed out: {exc}") from exc
        except Exception as exc:
            raise ObsidianWriteError(path, f"Unexpected error: {exc}") from exc

    # ── Helpers ───────────────────────────────────────────────────────────

    async def note_exists(self, path: str) -> bool:
        """Check if a vault file exists without reading its full content.

        Args:
            path: Vault-relative path.

        Returns:
            True if the file exists.
        """
        try:
            await self.read_note(path)
            return True
        except ObsidianReadError:
            return False

    async def read_note_with_hash(self, path: str) -> tuple[str, str]:
        """Read a note and return (content, sha256_hex) for change detection.

        Args:
            path: Vault-relative path.

        Returns:
            Tuple of (content, sha256_hex_digest).
        """
        content = await self.read_note(path)
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        return content, content_hash

    def build_daily_note_path(self, date_str: str) -> str:
        """Build the vault path for a daily note from a YYYY-MM-DD date string.

        Uses the configured daily_note_folder and daily_note_pattern.

        Args:
            date_str: Date in YYYY-MM-DD format.

        Returns:
            Full vault-relative path (e.g. "10_Journal/Daily/2026/03-March/2026-03-06.md").
        """
        from datetime import datetime

        dt = datetime.strptime(date_str, "%Y-%m-%d")

        month_names = [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ]
        month_name = month_names[dt.month - 1]
        mm = f"{dt.month:02d}"

        relative = self._settings.daily_note_pattern.format(
            year=dt.year,
            mm=mm,
            month=month_name,
            date=date_str,
        )

        folder = self._settings.daily_note_folder.rstrip("/")
        return f"{folder}/{relative}"

    async def ensure_daily_note(self, date_str: str) -> str:
        """Ensure a daily note exists for the given date, creating if needed.

        Args:
            date_str: Date in YYYY-MM-DD format.

        Returns:
            The vault-relative path of the daily note.
        """
        path = self.build_daily_note_path(date_str)
        if not await self.note_exists(path):
            # Create minimal daily note with frontmatter
            template = f"---\ndate: {date_str}\n---\n\n# {date_str}\n"
            await self.write_note(path, template)
            log.info("obsidian_daily_note_created", path=path, date=date_str)
        return path
