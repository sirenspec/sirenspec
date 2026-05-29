"""MemoryManager — injected into the executor to mediate all memory access."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sirenspec.memory.file_store import FileMemoryStore
from sirenspec.memory.sqlite_store import SQLiteMemoryStore

if TYPE_CHECKING:
    from sirenspec.core.models import MemoryConfig
    from sirenspec.memory.store import MemoryStore


def build_store(config: MemoryConfig) -> MemoryStore:
    """Instantiate the correct store backend from *config*.

    :param config: The workflow-level :class:`~sirenspec.core.models.MemoryConfig`.
    :returns: An initialised :class:`~sirenspec.memory.store.MemoryStore`.
    """
    if config.backend == "file":
        return FileMemoryStore(path=config.path, default_ttl=config.ttl)
    return SQLiteMemoryStore(path=config.path, default_ttl=config.ttl)


class MemoryManager:
    """Mediates all memory reads and writes for a single workflow execution.

    Wraps a :class:`~sirenspec.memory.store.MemoryStore` and provides two public
    operations used by the executor:

    * :meth:`read_namespace` — snapshot all live keys into a dict so they can
      be injected into the ``{{ memory.* }}`` template namespace before execution.
    * :meth:`write` — persist a value under *key* with an optional per-write TTL.

    :param store: An initialised :class:`~sirenspec.memory.store.MemoryStore`.
    """

    def __init__(self, store: MemoryStore) -> None:
        self._store = store

    def read_namespace(self) -> dict[str, Any]:
        """Snapshot all live memory keys into a plain dict for template resolution.

        :returns: Dict mapping each live key to its current value.
        """
        return {key: self._store.get(key) for key in self._store.keys()}

    def write(self, key: str, value: Any, ttl: int | None = None) -> None:
        """Persist *value* under *key* with an optional TTL.

        :param key: The memory key to write (dot-path suffix after ``memory.``).
        :param value: A JSON-serialisable value.
        :param ttl: Per-write TTL in seconds; overrides the store default when given.
        """
        self._store.set(key, value, ttl=ttl)

    def close(self) -> None:
        """Close the underlying store and release any held resources."""
        self._store.close()
