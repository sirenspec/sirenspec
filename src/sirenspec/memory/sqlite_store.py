"""SQLite-backed MemoryStore implementation using Python's stdlib sqlite3."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from sirenspec.exceptions import MemoryError

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS memory (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    expires_at REAL
)
"""
_GET = "SELECT value, expires_at FROM memory WHERE key = ?"
_UPSERT = "INSERT OR REPLACE INTO memory (key, value, expires_at) VALUES (?, ?, ?)"
_DELETE = "DELETE FROM memory WHERE key = ?"
_KEYS = "SELECT key FROM memory WHERE expires_at IS NULL OR expires_at > ?"
_PURGE = "DELETE FROM memory WHERE expires_at IS NOT NULL AND expires_at <= ?"


def _now() -> float:
    return time.time()


class SQLiteMemoryStore:
    """SQLite-backed key-value store with per-entry TTL.

    Uses a single ``memory`` table with ``key``, ``value`` (JSON-serialised),
    and ``expires_at`` (Unix timestamp float, NULL means no expiry).  TTL is
    enforced on read and purged lazily on :meth:`keys`.

    :param path: Directory path under which ``memory.db`` is created.
    :param default_ttl: Default TTL in seconds applied when none is supplied
        to :meth:`set`. ``None`` means no expiry.
    """

    def __init__(self, path: str | Path, default_ttl: int | None = None) -> None:
        db_path = Path(path) / "memory.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._default_ttl = default_ttl
        try:
            self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
            self._conn.execute(_CREATE_TABLE)
            self._conn.commit()
        except sqlite3.Error as exc:
            raise MemoryError(f"Failed to initialise SQLite memory store at '{db_path}': {exc}") from exc

    def get(self, key: str) -> Any | None:
        """Return the value for *key*, or ``None`` if absent or expired.

        :param key: The key to look up.
        :returns: The stored Python value, or ``None``.
        """
        try:
            row = self._conn.execute(_GET, (key,)).fetchone()
        except sqlite3.Error as exc:
            raise MemoryError(f"SQLite read error for key '{key}': {exc}") from exc

        if row is None:
            return None
        value_json, expires_at = row
        if expires_at is not None and expires_at <= _now():
            return None
        return json.loads(value_json)

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Write *value* under *key* with an optional TTL.

        :param key: The key to write.
        :param value: A JSON-serialisable value.
        :param ttl: Seconds until expiry; overrides the store default when given.
        """
        effective_ttl = ttl if ttl is not None else self._default_ttl
        expires_at = (_now() + effective_ttl) if effective_ttl is not None else None
        try:
            self._conn.execute(_UPSERT, (key, json.dumps(value), expires_at))
            self._conn.commit()
        except (sqlite3.Error, TypeError) as exc:
            raise MemoryError(f"SQLite write error for key '{key}': {exc}") from exc

    def delete(self, key: str) -> None:
        """Remove *key* from the store; no-op if absent.

        :param key: The key to remove.
        """
        try:
            self._conn.execute(_DELETE, (key,))
            self._conn.commit()
        except sqlite3.Error as exc:
            raise MemoryError(f"SQLite delete error for key '{key}': {exc}") from exc

    def keys(self) -> list[str]:
        """Return all live (non-expired) keys, purging stale rows as a side-effect.

        :returns: Sorted list of live key strings.
        """
        now = _now()
        try:
            self._conn.execute(_PURGE, (now,))
            self._conn.commit()
            rows = self._conn.execute(_KEYS, (now,)).fetchall()
        except sqlite3.Error as exc:
            raise MemoryError(f"SQLite keys error: {exc}") from exc
        return sorted(row[0] for row in rows)

    def close(self) -> None:
        """Close the underlying database connection."""
        self._conn.close()
