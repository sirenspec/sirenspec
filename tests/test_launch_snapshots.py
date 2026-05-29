"""Tests for native snapshot version control in ``sirenspec launch``."""

from __future__ import annotations

from pathlib import Path

import pytest

from sirenspec.exceptions import SnapshotError
from sirenspec.session.app import LaunchApp
from sirenspec.session.runtime import WorkflowSession
from sirenspec.session.snapshots import SnapshotStore
from sirenspec.session.widgets import CommandInput, StatusBar, Transcript
from sirenspec.yaml.parser import load_workflow

WORKFLOW_YAML = """version: "0.1"
agents:
  a:
    model: openai:gpt-4o-mini
    system: "You are helpful."
nodes:
  answer:
    agent: a
    writes: output.reply
"""


def write_workflow(tmp_path: Path, content: str = WORKFLOW_YAML) -> Path:
    path = tmp_path / "workflow.yaml"
    path.write_text(content)
    return path


# ---------------------------------------------------------------------------
# SnapshotStore unit behaviour
# ---------------------------------------------------------------------------


class TestSnapshotStore:
    def test_create_and_list_increments_version(self, tmp_path: Path) -> None:
        store = SnapshotStore(write_workflow(tmp_path))
        first = store.create(label="initial", trigger="manual")
        second = store.create(trigger="auto")
        assert first.version == 1
        assert second.version == 2
        assert [s.version for s in store.list()] == [1, 2]
        assert store.latest_label() == "v2"

    def test_snapshot_persists_content(self, tmp_path: Path) -> None:
        path = write_workflow(tmp_path)
        store = SnapshotStore(path)
        snap = store.create(label="v-one", trigger="manual")
        path.write_text(WORKFLOW_YAML + "\n# changed\n")
        assert store.read_content(snap) == WORKFLOW_YAML  # snapshot is immutable

    def test_resolve_by_ref_and_label(self, tmp_path: Path) -> None:
        store = SnapshotStore(write_workflow(tmp_path))
        store.create(label="baseline", trigger="manual")
        assert store.resolve("v1").label == "baseline"
        assert store.resolve("1").version == 1
        assert store.resolve("baseline").version == 1

    def test_resolve_unknown_raises(self, tmp_path: Path) -> None:
        store = SnapshotStore(write_workflow(tmp_path))
        with pytest.raises(SnapshotError):
            store.resolve("v99")

    def test_diff_working_vs_snapshot(self, tmp_path: Path) -> None:
        path = write_workflow(tmp_path)
        store = SnapshotStore(path)
        snap = store.create(trigger="manual")
        path.write_text(WORKFLOW_YAML.replace("You are helpful.", "You are terse."))
        lines = store.diff(snap)
        assert any(line.startswith("-") and "helpful" in line for line in lines)
        assert any(line.startswith("+") and "terse" in line for line in lines)

    def test_diff_between_two_snapshots(self, tmp_path: Path) -> None:
        path = write_workflow(tmp_path)
        store = SnapshotStore(path)
        v1 = store.create(trigger="manual")
        path.write_text(WORKFLOW_YAML.replace("You are helpful.", "You are terse."))
        v2 = store.create(trigger="manual")
        lines = store.diff(v1, v2)
        assert any("terse" in line for line in lines)

    def test_rollback_is_reversible(self, tmp_path: Path) -> None:
        path = write_workflow(tmp_path)
        store = SnapshotStore(path)
        v1 = store.create(trigger="manual")  # content A
        path.write_text(WORKFLOW_YAML.replace("helpful", "terse"))  # content B on disk

        safety = store.rollback(v1)  # safety captures B, working becomes A
        assert "helpful" in path.read_text()
        assert "terse" in store.read_content(safety)

        store.rollback(safety)  # restore B again
        assert "terse" in path.read_text()

    def test_retention_keeps_manual_and_last_n_auto(self, tmp_path: Path) -> None:
        store = SnapshotStore(write_workflow(tmp_path), keep_last=2)
        store.create(label="keep-me", trigger="manual")
        for _ in range(4):
            store.create(trigger="auto")
        versions = store.list()
        manual = [s for s in versions if s.trigger == "manual"]
        auto = [s for s in versions if s.trigger == "auto"]
        assert len(manual) == 1  # manual always retained
        assert len(auto) == 2  # only the last 2 auto kept
        # The pruned auto snapshot files are gone from disk.
        assert not (store.versions_dir / "v2.yaml").exists()


# ---------------------------------------------------------------------------
# App integration
# ---------------------------------------------------------------------------


def make_app(tmp_path: Path) -> tuple[LaunchApp, Path]:
    path = write_workflow(tmp_path)
    session = WorkflowSession(load_workflow(str(path)), path, "workflow")
    return LaunchApp(session), path


class TestSnapshotAppIntegration:
    @pytest.mark.asyncio
    async def test_snapshot_command_updates_status_label(self, tmp_path: Path) -> None:
        app, _ = make_app(tmp_path)
        async with app.run_test() as pilot:
            command_input = app.query_one(CommandInput)
            command_input.value = "/snapshot before refactor"
            await command_input.action_submit()
            await pilot.pause()
            assert app.snapshot_label == "v1"
            assert "snapshot v1" in app.query_one(StatusBar).right_text.plain
            assert app.snapshots.resolve("v1").label == "before refactor"

    @pytest.mark.asyncio
    async def test_diff_command_renders_changes(self, tmp_path: Path) -> None:
        app, path = make_app(tmp_path)
        async with app.run_test() as pilot:
            await app.command_snapshot("")
            path.write_text(WORKFLOW_YAML.replace("helpful", "terse"))
            command_input = app.query_one(CommandInput)
            command_input.value = "/diff"
            await command_input.action_submit()
            await pilot.pause()
            transcript_text = "\n".join(str(line) for line in app.query_one(Transcript).lines)
            assert "terse" in transcript_text

    @pytest.mark.asyncio
    async def test_rollback_restores_working_file(self, tmp_path: Path) -> None:
        app, path = make_app(tmp_path)
        async with app.run_test() as pilot:
            await app.command_snapshot("baseline")
            path.write_text(WORKFLOW_YAML.replace("helpful", "terse"))
            await pilot.press("ctrl+r")
            await pilot.pause()
            assert "helpful" in path.read_text()  # restored
            # A safety snapshot of the edited state was taken (rollback is reversible).
            assert any(s.trigger == "rollback" for s in app.snapshots.list())
