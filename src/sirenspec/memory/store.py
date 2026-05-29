"""MemoryStore Protocol defining the interface for all memory backend implementations."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class MemoryStore(Protocol):
    """Structural interface for pluggable memory backends.

    :param get: Retrieve a value by key; returns ``None`` when the key is absent or expired.
    :param set: Write a key/value pair with an optional TTL in seconds.
    :param delete: Remove a key; no-op if the key does not exist.
    :param keys: Return all live (non-expired) keys currently in the store.
    :param close: Flush any pending writes and release resources.
    """

    def get(self, key: str) -> Any | None: ...

    def set(self, key: str, value: Any, ttl: int | None = None) -> None: ...

    def delete(self, key: str) -> None: ...

    def keys(self) -> list[str]: ...

    def close(self) -> None: ...
