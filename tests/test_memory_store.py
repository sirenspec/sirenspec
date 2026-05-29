"""Unit tests for FileMemoryStore and SQLiteMemoryStore."""

from __future__ import annotations

import time

import pytest

from sirenspec.memory.file_store import FileMemoryStore
from sirenspec.memory.sqlite_store import SQLiteMemoryStore

# ---------------------------------------------------------------------------
# Shared behaviour tested against both backends via parametrize
# ---------------------------------------------------------------------------


def make_file_store(tmp_path, default_ttl=None):
    return FileMemoryStore(path=tmp_path / "store", default_ttl=default_ttl)


def make_sqlite_store(tmp_path, default_ttl=None):
    return SQLiteMemoryStore(path=tmp_path / "db", default_ttl=default_ttl)


@pytest.fixture(params=["file", "sqlite"])
def store(request, tmp_path):
    if request.param == "file":
        return make_file_store(tmp_path)
    return make_sqlite_store(tmp_path)


class TestMemoryStoreCommon:
    def test_get_missing_returns_none(self, store) -> None:
        assert store.get("nonexistent") is None

    def test_set_and_get_string(self, store) -> None:
        store.set("greeting", "hello")
        assert store.get("greeting") == "hello"

    def test_set_and_get_dict(self, store) -> None:
        store.set("data", {"x": 1, "y": [2, 3]})
        assert store.get("data") == {"x": 1, "y": [2, 3]}

    def test_set_overwrites_existing(self, store) -> None:
        store.set("key", "first")
        store.set("key", "second")
        assert store.get("key") == "second"

    def test_delete_removes_key(self, store) -> None:
        store.set("key", "value")
        store.delete("key")
        assert store.get("key") is None

    def test_delete_nonexistent_is_noop(self, store) -> None:
        store.delete("missing")  # must not raise

    def test_keys_returns_live_keys(self, store) -> None:
        store.set("a", 1)
        store.set("b", 2)
        assert store.keys() == ["a", "b"]

    def test_keys_empty_when_nothing_set(self, store) -> None:
        assert store.keys() == []

    def test_ttl_expires_entry(self, store) -> None:
        store.set("ephemeral", "gone", ttl=1)
        assert store.get("ephemeral") == "gone"
        time.sleep(1.05)
        assert store.get("ephemeral") is None

    def test_ttl_excludes_expired_from_keys(self, store) -> None:
        store.set("live", "yes")
        store.set("dead", "no", ttl=1)
        time.sleep(1.05)
        assert store.keys() == ["live"]

    def test_no_ttl_persists_indefinitely(self, store) -> None:
        store.set("permanent", "stays")
        assert store.get("permanent") == "stays"

    def test_close_does_not_raise(self, store) -> None:
        store.close()

    def test_set_integer_value(self, store) -> None:
        store.set("count", 42)
        assert store.get("count") == 42

    def test_set_list_value(self, store) -> None:
        store.set("items", [1, "two", 3.0])
        assert store.get("items") == [1, "two", 3.0]

    def test_set_none_value(self, store) -> None:
        store.set("nothing", None)
        assert store.get("nothing") is None


class TestFileMemoryStorePersistence:
    def test_data_persists_across_instances(self, tmp_path) -> None:
        s1 = make_file_store(tmp_path)
        s1.set("key", "value")
        s1.close()

        s2 = make_file_store(tmp_path)
        assert s2.get("key") == "value"

    def test_default_ttl_applied_when_no_per_write_ttl(self, tmp_path) -> None:
        store = make_file_store(tmp_path, default_ttl=1)
        store.set("x", "y")
        time.sleep(1.05)
        assert store.get("x") is None

    def test_per_write_ttl_overrides_default(self, tmp_path) -> None:
        store = make_file_store(tmp_path, default_ttl=1)
        store.set("x", "y", ttl=3600)
        time.sleep(1.05)
        assert store.get("x") == "y"

    def test_atomic_write_creates_file(self, tmp_path) -> None:
        store = make_file_store(tmp_path)
        store.set("k", "v")
        assert (tmp_path / "store.json").exists()


class TestSQLiteMemoryStorePersistence:
    def test_data_persists_across_instances(self, tmp_path) -> None:
        s1 = make_sqlite_store(tmp_path)
        s1.set("key", "value")
        s1.close()

        s2 = make_sqlite_store(tmp_path)
        assert s2.get("key") == "value"

    def test_default_ttl_applied_when_no_per_write_ttl(self, tmp_path) -> None:
        store = make_sqlite_store(tmp_path, default_ttl=1)
        store.set("x", "y")
        time.sleep(1.05)
        assert store.get("x") is None

    def test_per_write_ttl_overrides_default(self, tmp_path) -> None:
        store = make_sqlite_store(tmp_path, default_ttl=1)
        store.set("x", "y", ttl=3600)
        time.sleep(1.05)
        assert store.get("x") == "y"

    def test_db_file_created_in_path(self, tmp_path) -> None:
        store = make_sqlite_store(tmp_path)
        store.set("k", "v")
        store.close()
        assert (tmp_path / "db" / "memory.db").exists()

    def test_purge_on_keys_removes_expired_rows(self, tmp_path) -> None:
        store = make_sqlite_store(tmp_path)
        store.set("live", "yes")
        store.set("dead", "no", ttl=1)
        time.sleep(1.05)
        keys = store.keys()
        assert "dead" not in keys
        assert "live" in keys
