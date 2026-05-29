"""Tests for the ``sirenspec launch`` production-style testing runtime (WorkflowSession)."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pytest

from sirenspec.core.models import AgentDefinition, AgentNode, Workflow, WorkflowInput
from sirenspec.core.usage import TokenUsage
from sirenspec.exceptions import SessionError
from sirenspec.providers.registry import set_provider_override
from sirenspec.session.app import LaunchApp
from sirenspec.session.runtime import (
    Turn,
    TurnResult,
    WorkflowSession,
    build_turn_input,
    node_attribution_label,
)
from sirenspec.session.widgets import CommandInput, StatusBar, TraceDrawer, Transcript


class FakeProvider:
    """A streaming provider stub that records the messages it is sent."""

    def __init__(self, response: str = "the answer", tokens: int = 12) -> None:
        self.response = response
        self.usage = TokenUsage(prompt_tokens=tokens, completion_tokens=tokens)
        self.calls: list[list[dict]] = []

    async def complete(self, messages: list[dict], max_tokens: int | None = None) -> str:
        self.calls.append(messages)
        return self.response

    async def stream(self, messages: list[dict], max_tokens: int | None = None) -> AsyncIterator[str]:
        self.calls.append(messages)
        # Emit two chunks that concatenate to exactly the response (no added whitespace).
        midpoint = len(self.response) // 2
        for chunk in (self.response[:midpoint], self.response[midpoint:]):
            yield chunk

    @property
    def last_token_usage(self) -> TokenUsage:
        return self.usage

    @property
    def client(self) -> object:
        return None


@pytest.fixture
def fake_provider() -> Iterator[FakeProvider]:
    """Install a shared FakeProvider as the global provider override for a test."""
    provider = FakeProvider()
    set_provider_override(lambda _uri: provider)
    try:
        yield provider
    finally:
        set_provider_override(None)


def single_agent_workflow() -> Workflow:
    """A one-node workflow that echoes the agent's reply into output.reply."""
    return Workflow(
        version="0.1",
        agents={"a": AgentDefinition(model="openai:gpt-4o-mini", system="You are helpful.")},
        nodes={"answer": AgentNode(agent="a", writes="output.reply")},
        input=WorkflowInput(message="run me"),
    )


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

MEMORY_WORKFLOW_YAML = """version: "0.1"
agents:
  a:
    model: openai:gpt-4o-mini
    system: "You are helpful."
nodes:
  answer:
    agent: a
    writes: output.reply
memory:
  backend: file
  path: {path}
"""


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestTurnInput:
    def test_first_turn_is_raw_message(self) -> None:
        assert build_turn_input([], "hello") == "hello"

    def test_later_turn_threads_history(self) -> None:
        history = [Turn(role="user", content="hi"), Turn(role="siren", content="hello there")]
        composed = build_turn_input(history, "and now?")
        assert "Conversation so far:" in composed
        assert "You: hi" in composed
        assert "Siren: hello there" in composed
        assert composed.strip().endswith("You: and now?")


class TestAttributionLabel:
    def test_agent_label_includes_provider(self) -> None:
        label, provider = node_attribution_label(single_agent_workflow(), "answer")
        assert label == "answer (openai)"
        assert provider == "openai"


# ---------------------------------------------------------------------------
# Session turn loop
# ---------------------------------------------------------------------------


class TestTakeTurn:
    @pytest.mark.asyncio
    async def test_turn_returns_result_and_records_history(self, tmp_path: Path, fake_provider: FakeProvider) -> None:
        session = WorkflowSession(single_agent_workflow(), tmp_path / "wf.yaml", "wf")
        chunks: list[str] = []
        result: TurnResult | None = None
        async for event in session.take_turn("what are the risks?", token_callback=chunks.append):
            if isinstance(event, TurnResult):
                result = event
        assert result is not None
        assert result.text == "the answer"
        assert result.tokens == 24  # prompt 12 + completion 12
        assert result.status == "success"
        assert "".join(chunks).strip() == "the answer"  # tokens streamed
        assert session.turns == 1
        assert [t.role for t in session.history] == ["user", "siren"]

    @pytest.mark.asyncio
    async def test_state_persists_across_turns(self, tmp_path: Path, fake_provider: FakeProvider) -> None:
        session = WorkflowSession(single_agent_workflow(), tmp_path / "wf.yaml", "wf")
        async for _ in session.take_turn("first question"):
            pass
        async for _ in session.take_turn("second question"):
            pass
        # The provider call for the second turn must include the first turn in its prompt.
        second_turn_user_msg = fake_provider.calls[-1][-1]["content"]
        assert "first question" in second_turn_user_msg
        assert "second question" in second_turn_user_msg
        assert session.turns == 2

    @pytest.mark.asyncio
    async def test_cost_accounting_accumulates(self, tmp_path: Path, fake_provider: FakeProvider) -> None:
        session = WorkflowSession(single_agent_workflow(), tmp_path / "wf.yaml", "wf")
        async for _ in session.take_turn("q1"):
            pass
        first_cost = session.session_cost_usd
        assert first_cost is not None and first_cost > 0
        async for _ in session.take_turn("q2"):
            pass
        assert session.session_cost_usd > first_cost

    @pytest.mark.asyncio
    async def test_run_full_uses_workflow_input(self, tmp_path: Path, fake_provider: FakeProvider) -> None:
        session = WorkflowSession(single_agent_workflow(), tmp_path / "wf.yaml", "wf")
        result: TurnResult | None = None
        async for event in session.run_full():
            if isinstance(event, TurnResult):
                result = event
        assert result is not None and result.status == "success"
        # run_full does not append to chat history.
        assert session.history == []
        assert fake_provider.calls[-1][-1]["content"] == "run me"


# ---------------------------------------------------------------------------
# Hot-reload
# ---------------------------------------------------------------------------


class TestHotReload:
    def test_reload_round_trip_preserves_history(self, tmp_path: Path) -> None:
        path = tmp_path / "wf.yaml"
        path.write_text(WORKFLOW_YAML)
        from sirenspec.yaml.parser import load_workflow

        session = WorkflowSession(load_workflow(str(path)), path, "wf")
        session.history.append(Turn(role="user", content="remember me"))
        assert session.file_changed() is False

        # Edit the file and bump its mtime so the watcher detects the change.
        path.write_text(WORKFLOW_YAML + "\n# edited\n")
        os.utime(path, (session.mtime + 10, session.mtime + 10))
        assert session.file_changed() is True

        session.reload()
        assert session.file_changed() is False
        assert session.history == [Turn(role="user", content="remember me")]

    def test_reload_invalid_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "wf.yaml"
        path.write_text(WORKFLOW_YAML)
        from sirenspec.yaml.parser import load_workflow

        session = WorkflowSession(load_workflow(str(path)), path, "wf")
        path.write_text("not: [valid")
        with pytest.raises(SessionError):
            session.reload()


# ---------------------------------------------------------------------------
# --session rehydration
# ---------------------------------------------------------------------------


class TestSessionPersistence:
    def test_session_requires_memory_backend(self, tmp_path: Path) -> None:
        with pytest.raises(SessionError):
            WorkflowSession(single_agent_workflow(), tmp_path / "wf.yaml", "wf", session_id="s1")

    @pytest.mark.asyncio
    async def test_session_round_trips_via_memory(self, tmp_path: Path, fake_provider: FakeProvider) -> None:
        from sirenspec.yaml.parser import load_workflow

        path = tmp_path / "wf.yaml"
        path.write_text(MEMORY_WORKFLOW_YAML.format(path=str(tmp_path / "mem")))

        session1 = WorkflowSession(load_workflow(str(path)), path, "wf", session_id="abc")
        async for _ in session1.take_turn("persist this"):
            pass
        session1.close()

        session2 = WorkflowSession(load_workflow(str(path)), path, "wf", session_id="abc")
        assert any(t.content == "persist this" for t in session2.history)
        assert session2.turns == 1
        session2.close()


# ---------------------------------------------------------------------------
# App integration
# ---------------------------------------------------------------------------


class TestAppRuntimeIntegration:
    @pytest.mark.asyncio
    async def test_chat_turn_streams_into_transcript(self, tmp_path: Path, fake_provider: FakeProvider) -> None:
        path = tmp_path / "wf.yaml"
        path.write_text("version: '0.1'\n")
        session = WorkflowSession(single_agent_workflow(), path, "wf")
        app = LaunchApp(session)
        async with app.run_test() as pilot:
            command_input = app.query_one(CommandInput)
            command_input.value = "what are the risks?"
            await command_input.action_submit()
            await pilot.pause()
            assert command_input.value == ""
            assert session.turns == 1
            transcript_text = "\n".join(str(line) for line in app.query_one(Transcript).lines)
            assert "the answer" in transcript_text
            # Trace drawer reflects the completed turn.
            assert "answer (openai)" in app.query_one(TraceDrawer).trace_text.plain

    @pytest.mark.asyncio
    async def test_run_command_executes_full_graph(self, tmp_path: Path, fake_provider: FakeProvider) -> None:
        path = tmp_path / "wf.yaml"
        path.write_text("version: '0.1'\n")
        session = WorkflowSession(single_agent_workflow(), path, "wf")
        app = LaunchApp(session)
        async with app.run_test() as pilot:
            command_input = app.query_one(CommandInput)
            command_input.value = "/run"
            await command_input.action_submit()
            await pilot.pause()
            transcript_text = "\n".join(str(line) for line in app.query_one(Transcript).lines)
            assert "the answer" in transcript_text

    @pytest.mark.asyncio
    async def test_live_indicator_resets_after_turn(self, tmp_path: Path, fake_provider: FakeProvider) -> None:
        path = tmp_path / "wf.yaml"
        path.write_text("version: '0.1'\n")
        session = WorkflowSession(single_agent_workflow(), path, "wf")
        app = LaunchApp(session)
        async with app.run_test() as pilot:
            command_input = app.query_one(CommandInput)
            command_input.value = "hello"
            await command_input.action_submit()
            await pilot.pause()
            assert "idle" in app.query_one(StatusBar).right_text.plain
