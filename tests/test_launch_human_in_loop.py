"""Regression tests for human-in-the-loop nodes inside the ``sirenspec launch`` TUI.

The claude-code-mini demo (sirenspec-demos/claude-code-mini/workflow.yaml) uses
``type: human`` nodes (``approve_edits``, ``approve_command``) to gate file edits
and shell commands behind operator approval.  In the TUI those nodes silently fall
back to :func:`sirenspec.core.human_runner.stdin_input` because
:meth:`WorkflowSession.take_turn` / :meth:`run_full` never pass a
``human_input_fn`` through to :func:`execute_streaming`.  In a Textual full-screen
app stdin is captured by the runtime, so the operator can never type the approval
and the whole turn either hangs (when stdin blocks) or fails with
``HumanInputError`` (when stdin is closed by pytest's capture).  Either way the
workflow does not work end-to-end inside the studio.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import pytest

from sirenspec.core.models import (
    AgentDefinition,
    AgentNode,
    Edge,
    HumanNode,
    Workflow,
    WorkflowInput,
)
from sirenspec.core.usage import TokenUsage
from sirenspec.providers.registry import set_provider_override
from sirenspec.session.app import LaunchApp
from sirenspec.session.runtime import WorkflowSession
from sirenspec.session.widgets import CommandInput, Transcript


class FakeProvider:
    """Minimal streaming provider stub."""

    def __init__(self, response: str = "ok") -> None:
        self.response = response
        self.usage = TokenUsage(prompt_tokens=1, completion_tokens=1)

    async def complete(self, messages: list[dict], max_tokens: int | None = None) -> str:
        return self.response

    async def stream(self, messages: list[dict], max_tokens: int | None = None) -> AsyncIterator[str]:
        yield self.response

    @property
    def last_token_usage(self) -> TokenUsage:
        return self.usage

    @property
    def client(self) -> object:
        return None


@pytest.fixture
def fake_provider() -> Iterator[FakeProvider]:
    provider = FakeProvider()
    set_provider_override(lambda _uri: provider)
    try:
        yield provider
    finally:
        set_provider_override(None)


def human_in_loop_workflow() -> Workflow:
    """A 3-node workflow that mirrors claude-code-mini's approval gate."""
    return Workflow(
        version="0.1",
        agents={"a": AgentDefinition(model="openai:gpt-4o-mini", system="s")},
        nodes={
            "propose": AgentNode(agent="a", writes="working.proposal"),
            "approve": HumanNode(
                type="human",
                prompt="Approve? type yes/no",
                writes="working.approval",
                timeout=2.0,
                on_timeout="use_default",
                default_output="no",
            ),
            "summarize": AgentNode(agent="a", writes="output.reply"),
        },
        edges=[
            Edge(from_node="propose", to_node="approve"),
            Edge(from_node="approve", to_node="summarize"),
        ],
        input=WorkflowInput(message="please change foo"),
    )


def write_workflow_file(tmp_path: Path) -> Path:
    """Write any non-empty yaml so :class:`WorkflowSession` mtime/reload work."""
    path = tmp_path / "wf.yaml"
    path.write_text("version: '0.1'\n")
    return path


def build_app(tmp_path: Path) -> LaunchApp:
    workflow = human_in_loop_workflow()
    session = WorkflowSession(workflow, write_workflow_file(tmp_path), "wf")
    return LaunchApp(session)


@pytest.mark.asyncio
async def test_chat_turn_with_human_node_completes_without_blocking(
    tmp_path: Path, fake_provider: FakeProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A chat turn whose workflow has a HumanNode must finish via the TUI's own approval flow.

    Today this fails because the runtime never hands a TUI-aware ``human_input_fn``
    to the executor.  The turn either times out at the executor-level (because
    on_timeout='use_default' is set) — silently writing the default without any
    chance for the user to actually answer — or, when on_timeout='abort', the
    whole turn collapses with a HumanInputError surfaced as a transcript notice.
    We assert the success path so that, once fixed, the studio must present the
    prompt and accept input from the chat box.
    """

    # Replace the executor's stdin reader with one that asserts it is NEVER used:
    # if the TUI routes human input properly we should never call this function.
    async def must_not_call_stdin(_: str) -> str:
        raise AssertionError("TUI fell back to stdin_input for a HumanNode")

    monkeypatch.setattr(
        "sirenspec.core.human_runner.stdin_input",
        must_not_call_stdin,
    )

    app = build_app(tmp_path)

    async with app.run_test() as pilot:
        ci = app.query_one(CommandInput)
        ci.value = "please change foo"
        await pilot.press("enter")

        # Give the TUI a generous window to surface the approval prompt and accept
        # an answer.  When the bug bites this loop spins past the HumanNode's
        # timeout fallback (or the stdin block) without ever showing anything to
        # the user.
        for _ in range(20):
            await pilot.pause(0.1)

        # After fix: an explicit approval UI accepts "yes" from the user.
        ci.value = "yes"
        await pilot.press("enter")

        for _ in range(20):
            await pilot.pause(0.1)

        transcript_text = "\n".join(str(line) for line in app.query_one(Transcript).lines)
        assert "HumanInputError" not in transcript_text
        # The summarizer must have produced a final reply, proving the graph
        # walked past the approval gate driven by real TUI input.
        assert "ok" in transcript_text


@pytest.mark.asyncio
async def test_runtime_does_not_forward_human_input_fn_today(
    tmp_path: Path, fake_provider: FakeProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pin the current (broken) behaviour: ``stream_execution`` is called without ``human_input_fn``.

    Locks in the smoking gun so future refactors are forced to confront it.
    """
    captured: dict[str, Any] = {}

    from sirenspec.core import executor as executor_mod

    original = executor_mod.execute_streaming

    def spy(*args: Any, **kwargs: Any):  # noqa: ANN401
        captured["kwargs"] = kwargs
        return original(*args, **kwargs)

    monkeypatch.setattr("sirenspec.session.runtime.execute_streaming", spy)

    workflow = human_in_loop_workflow()
    session = WorkflowSession(workflow, write_workflow_file(tmp_path), "wf")

    async def consume() -> None:
        # The session swallows HumanInputError into a TurnResult via the executor;
        # we just need ANY call so we can inspect kwargs.
        try:
            async for _ in session.take_turn("hi"):
                pass
        except Exception:
            pass

    await asyncio.wait_for(consume(), timeout=5.0)

    assert "kwargs" in captured, "stream_execution was never invoked"
    assert "human_input_fn" in captured["kwargs"], (
        "WorkflowSession.take_turn must forward human_input_fn to execute_streaming "
        "so HumanNodes never fall back to stdin inside the TUI."
    )
    initial_inputs = captured["kwargs"].get("initial_inputs")
    assert initial_inputs is not None and "history" in initial_inputs, (
        "WorkflowSession.take_turn must seed inputs.history so workflows authored "
        "for hosts like claude-code-mini (which reference {{ inputs.history }}) "
        "do not crash with InterpolationError inside the TUI."
    )
