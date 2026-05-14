"""Unit tests for WorkflowContext."""

from __future__ import annotations

import pytest

from sirenspec.core.context import WorkflowContext


class TestWorkflowContext:
    def test_empty_init(self) -> None:
        ctx = WorkflowContext()
        assert ctx.working == {}
        assert ctx.output == {}

    def test_initial_state(self) -> None:
        ctx = WorkflowContext(initial_state={"working": {"key": "value"}, "output": {"x": 1}})
        assert ctx.working["key"] == "value"
        assert ctx.output["x"] == 1

    def test_write_to_output(self) -> None:
        ctx = WorkflowContext()
        ctx.write("output.reply", "Hello!")
        assert ctx.output["reply"] == "Hello!"

    def test_write_to_working(self) -> None:
        ctx = WorkflowContext()
        ctx.write("working.intent", "question")
        assert ctx.working["intent"] == "question"

    def test_write_nested_path(self) -> None:
        ctx = WorkflowContext()
        ctx.write("working.analysis.step1", "result1")
        assert ctx.working["analysis"]["step1"] == "result1"

    def test_write_deeply_nested(self) -> None:
        ctx = WorkflowContext()
        ctx.write("working.a.b.c", 42)
        assert ctx.working["a"]["b"]["c"] == 42

    def test_resolve_simple(self) -> None:
        ctx = WorkflowContext()
        ctx.write("output.x", "foo")
        assert ctx.resolve("output.x") == "foo"

    def test_resolve_nested(self) -> None:
        ctx = WorkflowContext()
        ctx.write("working.analysis.step1", "r1")
        assert ctx.resolve("working.analysis.step1") == "r1"

    def test_resolve_missing_raises_keyerror(self) -> None:
        ctx = WorkflowContext()
        with pytest.raises(KeyError):
            ctx.resolve("working.nonexistent")

    def test_resolve_missing_nested_raises_keyerror(self) -> None:
        ctx = WorkflowContext()
        ctx.write("working.a", {})
        with pytest.raises(KeyError):
            ctx.resolve("working.a.b")

    def test_resolve_unknown_root_raises_keyerror(self) -> None:
        ctx = WorkflowContext()
        with pytest.raises(KeyError):
            ctx.resolve("state.x")

    def test_write_unknown_root_raises_keyerror(self) -> None:
        ctx = WorkflowContext()
        with pytest.raises(KeyError):
            ctx.write("state.x", "v")

    def test_snapshot(self) -> None:
        ctx = WorkflowContext()
        ctx.write("output.reply", "hi")
        ctx.write("working.intent", "q")
        snap = ctx.snapshot()
        assert snap["output"]["reply"] == "hi"
        assert snap["working"]["intent"] == "q"

    def test_write_overwrites_existing(self) -> None:
        ctx = WorkflowContext()
        ctx.write("output.x", "first")
        ctx.write("output.x", "second")
        assert ctx.resolve("output.x") == "second"
