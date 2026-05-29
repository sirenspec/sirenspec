"""JSON file-backed MemoryStore implementation."""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from sirenspec.exceptions import MemoryError


def _now() -> float:
    return time.time()


def _load_entries(path: Path) -> dict[str, dict[str, Any]]:
    """Read the JSON store file and return raw entries dict."""
    try:
        text = path.read_text(encoding="utf-8")
        data = json.loads(text)
        if isinstance(data, dict):
            return data
        return {}
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_entries(path: Path, entries: dict[str, dict[str, Any]]) -> None:
    """Atomically write entries to the JSON store file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2)
        os.replace(tmp_path, path)
    except OSError as exc:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise MemoryError(f"Failed to write memory store at '{path}': {exc}") from exc


def _is_live(entry: dict[str, Any]) -> bool:
    """Return True when entry has no expiry or expiry is in the future."""
    expires_at = entry.get("expires_at")
    return expires_at is None or expires_at > _now()


class FileMemoryStore:
    """JSON file-backed key-value store with per-entry TTL.

    All reads load the file fresh; all writes atomically replace it via a
    temp-file-rename so concurrent processes see a consistent view.

    :param path: File path for the JSON store.
    :param default_ttl: Default TTL in seconds applied when none is supplied
        to :meth:`set`. ``None`` means no expiry.
    """

    def __init__(self, path: str | Path, default_ttl: int | None = None) -> None:
        self._path = Path(path).with_suffix(".json")
        self._default_ttl = default_ttl

    def get(self, key: str) -> Any | None:
        """Return the value for *key*, or ``None`` if absent or expired.

        :param key: The key to look up.
        :returns: The stored value, or ``None``.
        """
        entries = _load_entries(self._path)
        entry = entries.get(key)
        if entry is None or not _is_live(entry):
            return None
        return entry["value"]

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Write *value* under *key* with an optional TTL.

        :param key: The key to write.
        :param value: A JSON-serialisable value.
        :param ttl: Seconds until expiry; overrides the store default when given.
        """
        entries = _load_entries(self._path)
        effective_ttl = ttl if ttl is not None else self._default_ttl
        expires_at = (_now() + effective_ttl) if effective_ttl is not None else None
        entries[key] = {"value": value, "expires_at": expires_at}
        _save_entries(self._path, entries)

    def delete(self, key: str) -> None:
        """Remove *key* from the store; no-op if absent.

        :param key: The key to remove.
        """
        entries = _load_entries(self._path)
        if key in entries:
            del entries[key]
            _save_entries(self._path, entries)

    def keys(self) -> list[str]:
        """Return all live (non-expired) keys in the store.

        :returns: Sorted list of live key strings.
        """
        entries = _load_entries(self._path)
        return sorted(k for k, v in entries.items() if _is_live(v))

    def close(self) -> None:
        """No-op for the file store; included for protocol compatibility."""
