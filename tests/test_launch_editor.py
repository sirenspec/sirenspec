"""Tests for the ``/edit`` suggestive editor: assistant engine and the split-screen UI."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pytest

from sirenspec.core.usage import TokenUsage
from sirenspec.exceptions import EditAssistantError
from sirenspec.providers.registry import set_provider_override
from sirenspec.session.app import LaunchApp
from sirenspec.session.edit_screen import EditScreen
from sirenspec.session.editor import EditAssistant, choose_model, strip_code_fences
from sirenspec.session.runtime import WorkflowSession
from sirenspec.yaml.parser import load_workflow

CURRENT_YAML = """version: "0.1"
agents:
  a:
    model: openai:gpt-4o-mini
    system: "You are helpful."
nodes:
  answer:
    agent: a
    writes: output.reply
"""

PROPOSED_YAML = CURRENT_YAML.replace("You are helpful.", "You are concise.")
INVALID_YAML = "not: [valid yaml"


class ScriptedProvider:
    """A provider stub whose completion returns a fixed scripted response."""

    def __init__(self, response: str) -> None:
        self.response = response
        self.usage = TokenUsage(prompt_tokens=5, completion_tokens=5)

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


def use_provider(response: str) -> None:
    """Install a ScriptedProvider returning *response* as the global provider override."""
    set_provider_override(lambda _uri: ScriptedProvider(response))


@pytest.fixture(autouse=True)
def _clear_override() -> Iterator[None]:
    """Ensure the provider override is cleared after each test."""
    yield
    set_provider_override(None)


# ---------------------------------------------------------------------------
# Provider preference / fallback
# ---------------------------------------------------------------------------


class TestChooseModel:
    def test_prefers_anthropic_when_both_present(self) -> None:
        model = choose_model({"ANTHROPIC_API_KEY": "x", "OPENAI_API_KEY": "y"})
        assert model.startswith("anthropic:")

    def test_falls_back_to_openai(self) -> None:
        model = choose_model({"OPENAI_API_KEY": "y"})
        assert model.startswith("openai:")

    def test_raises_without_any_key(self) -> None:
        with pytest.raises(EditAssistantError):
            choose_model({})


class TestStripFences:
    def test_strips_yaml_fence(self) -> None:
        fenced = "```yaml\nversion: '0.1'\n```"
        assert strip_code_fences(fenced) == "version: '0.1'"

    def test_passthrough_without_fence(self) -> None:
        assert strip_code_fences("version: '0.1'") == "version: '0.1'"


# ---------------------------------------------------------------------------
# Assistant engine
# ---------------------------------------------------------------------------


class TestEditAssistant:
    @pytest.mark.asyncio
    async def test_propose_returns_change_and_diff(self) -> None:
        use_provider(PROPOSED_YAML)
        assistant = EditAssistant(model_uri="anthropic:claude-haiku-4-5-20251001")
        change = await assistant.propose(CURRENT_YAML, "make the agent concise")
        assert change.new_yaml.strip() == PROPOSED_YAML.strip()
        assert any("concise" in line and line.startswith("+") for line in change.diff_lines)

    def test_validate_accepts_valid_yaml(self, tmp_path: Path) -> None:
        assistant = EditAssistant(model_uri="anthropic:claude-haiku-4-5-20251001")
        workflow = assistant.validate(PROPOSED_YAML, tmp_path / "workflow.yaml")
        assert "answer" in workflow.nodes

    def test_validate_rejects_invalid_yaml(self, tmp_path: Path) -> None:
        assistant = EditAssistant(model_uri="anthropic:claude-haiku-4-5-20251001")
        with pytest.raises(EditAssistantError):
            assistant.validate(INVALID_YAML, tmp_path / "workflow.yaml")

    def test_validate_cleans_up_temp_file(self, tmp_path: Path) -> None:
        assistant = EditAssistant(model_uri="anthropic:claude-haiku-4-5-20251001")
        assistant.validate(PROPOSED_YAML, tmp_path / "workflow.yaml")
        leftovers = list(tmp_path.glob(".sirenspec-edit-*"))
        assert leftovers == []


# ---------------------------------------------------------------------------
# Edit screen integration
# ---------------------------------------------------------------------------


def make_app(tmp_path: Path) -> tuple[LaunchApp, Path]:
    path = tmp_path / "workflow.yaml"
    path.write_text(CURRENT_YAML)
    session = WorkflowSession(load_workflow(str(path)), path, "workflow")
    app = LaunchApp(session)
    app.editor = EditAssistant(model_uri="anthropic:claude-haiku-4-5-20251001")
    return app, path


class TestEditScreenIntegration:
    @pytest.mark.asyncio
    async def test_edit_command_opens_split_screen(self, tmp_path: Path) -> None:
        app, _ = make_app(tmp_path)
        async with app.run_test() as pilot:
            await app.command_edit("")
            await pilot.pause()
            assert isinstance(app.screen, EditScreen)

    @pytest.mark.asyncio
    async def test_accept_validates_writes_and_snapshots(self, tmp_path: Path) -> None:
        use_provider(PROPOSED_YAML)
        app, path = make_app(tmp_path)
        async with app.run_test() as pilot:
            await app.command_edit("")
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, EditScreen)
            await screen.request_proposal("make the agent concise")
            await pilot.pause()
            assert screen.proposal is not None
            await screen.action_accept()
            await pilot.pause()
            assert "concise" in path.read_text()  # accepted change written
            assert any(s.trigger == "accept" for s in app.snapshots.list())  # snapshot on accept
            assert screen.proposal is None  # cleared after accept

    @pytest.mark.asyncio
    async def test_reject_discards_without_writing(self, tmp_path: Path) -> None:
        use_provider(PROPOSED_YAML)
        app, path = make_app(tmp_path)
        async with app.run_test() as pilot:
            await app.command_edit("")
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, EditScreen)
            await screen.request_proposal("make the agent concise")
            await pilot.pause()
            screen.action_reject()
            assert screen.proposal is None
            assert "helpful" in path.read_text()  # unchanged

    @pytest.mark.asyncio
    async def test_validation_gate_blocks_invalid_accept(self, tmp_path: Path) -> None:
        use_provider(INVALID_YAML)
        app, path = make_app(tmp_path)
        async with app.run_test() as pilot:
            await app.command_edit("")
            await pilot.pause()
            screen = app.screen
            assert isinstance(screen, EditScreen)
            await screen.request_proposal("break it")
            await pilot.pause()
            await screen.action_accept()
            await pilot.pause()
            assert "helpful" in path.read_text()  # invalid proposal not written
            assert not any(s.trigger == "accept" for s in app.snapshots.list())
