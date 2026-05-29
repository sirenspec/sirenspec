"""Tests for the multi-line prompt: submit/newline keys, history, and paste collapsing."""

from __future__ import annotations

from pathlib import Path

import pytest
from textual import events

from sirenspec.core.models import AgentDefinition, AgentNode, Workflow
from sirenspec.session.app import LaunchApp
from sirenspec.session.runtime import WorkflowSession
from sirenspec.session.widgets import (
    CommandInput,
    paste_placeholder,
    should_collapse_paste,
)


def make_app(tmp_path: Path) -> LaunchApp:
    workflow = Workflow(
        version="0.1",
        agents={"a": AgentDefinition(model="openai:gpt-4o-mini", system="s")},
        nodes={"answer": AgentNode(agent="a", writes="output.reply")},
    )
    path = tmp_path / "wf.yaml"
    path.write_text("version: '0.1'\n")
    return LaunchApp(WorkflowSession(workflow, path, "wf"))


# ---------------------------------------------------------------------------
# Paste-collapse helpers
# ---------------------------------------------------------------------------


class TestPasteHelpers:
    def test_short_paste_not_collapsed(self) -> None:
        assert should_collapse_paste("one line") is False
        assert should_collapse_paste("a\nb\nc") is False  # exactly 3 lines

    def test_many_lines_collapses(self) -> None:
        assert should_collapse_paste("a\nb\nc\nd") is True

    def test_many_words_collapses(self) -> None:
        assert should_collapse_paste("word " * 101) is True

    def test_placeholder_format(self) -> None:
        assert paste_placeholder(2, 42) == "[Pasted text #2 (lines 1-42)]"


# ---------------------------------------------------------------------------
# Widget behaviour (headless)
# ---------------------------------------------------------------------------


class TestPromptInput:
    @pytest.mark.asyncio
    async def test_ctrl_j_inserts_newline_without_submitting(self, tmp_path: Path) -> None:
        app = make_app(tmp_path)
        async with app.run_test() as pilot:
            ci = app.query_one(CommandInput)
            ci.value = "line one"
            await pilot.press("ctrl+j")
            ci.insert("line two")
            await pilot.pause()
            assert "\n" in ci.text
            assert ci.text.count("\n") == 1

    @pytest.mark.asyncio
    async def test_enter_submits_and_clears(self, tmp_path: Path) -> None:
        app = make_app(tmp_path)
        async with app.run_test() as pilot:
            ci = app.query_one(CommandInput)
            ci.value = "/help"
            await pilot.press("enter")
            await pilot.pause()
            assert ci.text == ""  # cleared on submit
            assert "/help" in ci.command_history

    @pytest.mark.asyncio
    async def test_large_paste_collapses_and_expands_on_submit(self, tmp_path: Path) -> None:
        app = make_app(tmp_path)
        async with app.run_test() as pilot:
            ci = app.query_one(CommandInput)
            ci.focus()
            big = "\n".join(f"line {i}" for i in range(10))
            await ci._on_paste(events.Paste(big))
            await pilot.pause()
            # The buffer shows a placeholder, not the raw 10 lines.
            assert "[Pasted text #1 (lines 1-10)]" in ci.text
            assert "line 9" not in ci.text
            # Submitting expands the placeholder back to the full text.
            assert "line 9" in ci.expanded_value()

    @pytest.mark.asyncio
    async def test_small_paste_inserts_verbatim(self, tmp_path: Path) -> None:
        app = make_app(tmp_path)
        async with app.run_test() as pilot:
            ci = app.query_one(CommandInput)
            ci.focus()
            await ci._on_paste(events.Paste("short text"))
            await pilot.pause()
            assert "short text" in ci.text
            assert "Pasted text" not in ci.text

    @pytest.mark.asyncio
    async def test_history_navigation(self, tmp_path: Path) -> None:
        app = make_app(tmp_path)
        async with app.run_test() as pilot:
            ci = app.query_one(CommandInput)
            ci.remember("first")
            ci.remember("second")
            ci.focus()
            await pilot.press("up")
            assert ci.text == "second"
            await pilot.press("up")
            assert ci.text == "first"
            await pilot.press("down")
            assert ci.text == "second"
