"""Web content caching — avoid re-fetching same URLs within a configurable TTL.

Provides a simple in-memory LRU cache with time-based expiration for web content.
Supports caching search results, page content, and research results.

Architecture:
    - In-memory dict-based cache with TTL per entry
    - Automatic eviction of expired entries on access
    - LRU eviction when cache exceeds max size
    - Thread-safe for async contexts (single-threaded event loop)
    - Cache stats for monitoring

Usage:
    cache = WebCache(ttl_s=3600, max_entries=500)
    cache.set("https://example.com", content)
    result = cache.get("https://example.com")  # Returns content or None
"""

from __future__ import annotations

import time
from typing import Any

from ara.utils.logging import get_logger

log = get_logger("web_cache")


class CacheEntry:
    """A single cache entry with metadata.

    Attributes:
        key: Cache key (usually URL or query hash).
        value: Cached content.
        created_at: When the entry was created (monotonic time).
        last_accessed: When the entry was last accessed.
        access_count: Number of times this entry was accessed.
    """

    __slots__ = ("access_count", "created_at", "key", "last_accessed", "value")

    def __init__(self, key: str, value: Any) -> None:
        self.key = key
        self.value = value
        self.created_at = time.monotonic()
        self.last_accessed = self.created_at
        self.access_count = 0

    def is_expired(self, ttl_s: float) -> bool:
        """Check if this entry has expired.

        Args:
            ttl_s: Time-to-live in seconds.

        Returns:
            True if the entry is older than ttl_s.
        """
        return (time.monotonic() - self.created_at) > ttl_s

    def touch(self) -> None:
        """Update access time and increment counter."""
        self.last_accessed = time.monotonic()
        self.access_count += 1


class WebCache:
    """In-memory LRU web content cache with TTL expiration.

    Args:
        ttl_s: Time-to-live for cache entries in seconds.
        max_entries: Maximum number of entries before LRU eviction.
    """

    def __init__(
        self,
        *,
        ttl_s: float = 3600.0,
        max_entries: int = 500,
    ) -> None:
        self._ttl_s = ttl_s
        self._max_entries = max_entries
        self._entries: dict[str, CacheEntry] = {}
        self._hits = 0
        self._misses = 0

    @property
    def size(self) -> int:
        """Current number of entries in the cache."""
        return len(self._entries)

    @property
    def hit_rate(self) -> float:
        """Cache hit rate as a fraction (0.0 to 1.0)."""
        total = self._hits + self._misses
        if total == 0:
            return 0.0
        return self._hits / total

    def get(self, key: str) -> Any | None:
        """Get a cached value by key.

        Returns None if the key is not found or has expired.
        Expired entries are automatically removed.

        Args:
            key: Cache key.

        Returns:
            Cached value, or None if not found/expired.
        """
        entry = self._entries.get(key)
        if entry is None:
            self._misses += 1
            return None

        if entry.is_expired(self._ttl_s):
            del self._entries[key]
            self._misses += 1
            return None

        entry.touch()
        self._hits += 1
        return entry.value

    def set(self, key: str, value: Any) -> None:
        """Store a value in the cache.

        If the cache is full, evicts the least-recently-used entry.

        Args:
            key: Cache key.
            value: Value to cache.
        """
        # Evict if at capacity (before adding new entry)
        if key not in self._entries and len(self._entries) >= self._max_entries:
            self._evict_lru()

        self._entries[key] = CacheEntry(key, value)

    def has(self, key: str) -> bool:
        """Check if a key exists and is not expired.

        Args:
            key: Cache key.

        Returns:
            True if the key exists and hasn't expired.
        """
        entry = self._entries.get(key)
        if entry is None:
            return False
        if entry.is_expired(self._ttl_s):
            del self._entries[key]
            return False
        return True

    def delete(self, key: str) -> bool:
        """Delete a specific cache entry.

        Args:
            key: Cache key.

        Returns:
            True if the entry was found and deleted.
        """
        if key in self._entries:
            del self._entries[key]
            return True
        return False

    def clear(self) -> int:
        """Clear all cache entries.

        Returns:
            Number of entries cleared.
        """
        count = len(self._entries)
        self._entries.clear()
        self._hits = 0
        self._misses = 0
        log.info("cache_cleared", entries_cleared=count)
        return count

    def evict_expired(self) -> int:
        """Remove all expired entries.

        Returns:
            Number of entries evicted.
        """
        expired_keys = [
            key for key, entry in self._entries.items() if entry.is_expired(self._ttl_s)
        ]
        for key in expired_keys:
            del self._entries[key]

        if expired_keys:
            log.debug("expired_entries_evicted", count=len(expired_keys))

        return len(expired_keys)

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dict with size, hits, misses, hit_rate, ttl_s, max_entries.
        """
        return {
            "size": self.size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self.hit_rate, 3),
            "ttl_s": self._ttl_s,
            "max_entries": self._max_entries,
        }

    def _evict_lru(self) -> None:
        """Evict the least-recently-used entry."""
        if not self._entries:
            return

        # First try to evict expired entries
        expired_count = self.evict_expired()
        if expired_count > 0 and len(self._entries) < self._max_entries:
            return

        # Find LRU entry
        lru_key = min(self._entries, key=lambda k: self._entries[k].last_accessed)
        del self._entries[lru_key]
        log.debug("lru_entry_evicted", key=lru_key[:80])
