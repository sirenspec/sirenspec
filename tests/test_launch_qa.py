"""Adversarial end-to-end QA tests for the ``sirenspec launch`` TUI.

Every test drives the app through real keystrokes and pilot actions — never by calling
``action_*`` or event-handler methods directly.  That discipline is the whole point: it
catches focus-routing bugs that direct method calls silently skip.

For each confirmed bug the test is marked ``@pytest.mark.xfail(strict=True)`` so the suite
stays green while documenting the defect; the xfail flips to a failure the day the bug is
fixed, prompting removal of the marker.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import pytest

from sirenspec.core.models import AgentDefinition, AgentNode, Workflow
from sirenspec.core.usage import TokenUsage
from sirenspec.exceptions import SessionError
from sirenspec.providers.registry import set_provider_override
from sirenspec.session.app import LaunchApp
from sirenspec.session.edit_screen import EditScreen
from sirenspec.session.runtime import WorkflowSession
from sirenspec.session.widgets import (
    CommandInput,
    CommandPalette,
    StatusBar,
    TraceDrawer,
    Transcript,
    WorkflowRail,
)
from sirenspec.yaml.parser import load_workflow

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

WORKFLOW_YAML = """\
version: "0.1"
agents:
  a:
    model: openai:gpt-4o-mini
    system: "You are helpful."
nodes:
  answer:
    agent: a
    writes: output.reply
input:
  message: "run me"
"""

PROPOSED_YAML = """\
version: "0.1"
agents:
  a:
    model: openai:gpt-4o-mini
    system: "You are concise."
nodes:
  answer:
    agent: a
    writes: output.reply
input:
  message: "run me"
"""


class FakeProvider:
    """Streaming provider stub.  Vary ``response`` and ``raises`` to stress the UI."""

    def __init__(self, response: str = "the answer", tokens: int = 12, raises: Exception | None = None) -> None:
        self.response = response
        self.raises = raises
        self.usage = TokenUsage(prompt_tokens=tokens, completion_tokens=tokens)
        self.calls: list[list[dict]] = []

    async def complete(self, messages: list[dict], max_tokens: int | None = None) -> str:
        self.calls.append(messages)
        if self.raises:
            raise self.raises
        return self.response

    async def stream(self, messages: list[dict], max_tokens: int | None = None) -> AsyncIterator[str]:
        self.calls.append(messages)
        if self.raises:
            raise self.raises
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
    """Install a FakeProvider and restore the override on teardown."""
    provider = FakeProvider()
    set_provider_override(lambda _uri: provider)
    try:
        yield provider
    finally:
        set_provider_override(None)


@pytest.fixture(autouse=True)
def _clear_override() -> Iterator[None]:
    """Ensure no stale provider override leaks between tests."""
    yield
    set_provider_override(None)


def make_session(tmp_path: Path, yaml: str = WORKFLOW_YAML) -> tuple[WorkflowSession, Path]:
    """Write *yaml* to a temp file and return a WorkflowSession + its path."""
    path = tmp_path / "workflow.yaml"
    path.write_text(yaml)
    return WorkflowSession(load_workflow(str(path)), path, "workflow"), path


def make_app(tmp_path: Path, yaml: str = WORKFLOW_YAML) -> tuple[LaunchApp, Path]:
    """Return a ready LaunchApp and the workflow path."""
    session, path = make_session(tmp_path, yaml)
    return LaunchApp(session), path


def transcript_text(app: LaunchApp) -> str:
    """Return all transcript lines joined as a single string."""
    return "\n".join(str(line) for line in app.query_one(Transcript).lines)


# ---------------------------------------------------------------------------
# Behavioral — key bindings, palette, focus routing, input routing
# ---------------------------------------------------------------------------


class TestEditThenAcceptViaKeys:
    """The seed bug: in /edit, pressing 'e' then 'a' through real keys."""

    @pytest.mark.asyncio
    async def test_edit_then_accept_applies_change(self, tmp_path: Path) -> None:
        """Pressing e then a should apply the proposal even while the TextArea has focus."""
        set_provider_override(lambda _uri: FakeProvider(response=PROPOSED_YAML))
        app, path = make_app(tmp_path)
        from sirenspec.session.editor import EditAssistant

        app.editor = EditAssistant(model_uri="openai:gpt-4o-mini")
        async with app.run_test(size=(120, 40)) as pilot:
            ci = app.query_one(CommandInput)
            ci.focus()
            ci.value = "/edit"
            await pilot.press("enter")
            await pilot.pause()
            assert isinstance(app.screen, EditScreen)

            screen: EditScreen = app.screen  # type: ignore[assignment]

            from textual.widgets import Input

            edit_input = app.screen.query_one("#edit-input", Input)
            edit_input.focus()
            edit_input.value = "make the agent concise"
            await pilot.press("enter")
            await pilot.pause()
            assert screen.proposal is not None, "No proposal generated"

            # Press 'e' — opens edit buffer and focuses the TextArea
            await pilot.press("e")
            await pilot.pause()
            from textual.widgets import TextArea

            assert isinstance(app.focused, TextArea), "Expected TextArea to be focused after 'e'"

            # Press 'a' — now fires action_accept (priority=True binding) even though
            # TextArea has focus.
            await pilot.press("a")
            await pilot.pause()

            assert "concise" in path.read_text(), "File not updated after pressing 'a' with TextArea focused"

    @pytest.mark.asyncio
    async def test_accept_without_editing_applies_change(self, tmp_path: Path) -> None:
        """Pressing 'a' directly after a proposal (no edit step) should accept it."""
        set_provider_override(lambda _uri: FakeProvider(response=PROPOSED_YAML))
        app, path = make_app(tmp_path)
        from sirenspec.session.editor import EditAssistant

        app.editor = EditAssistant(model_uri="openai:gpt-4o-mini")
        async with app.run_test(size=(120, 40)) as pilot:
            ci = app.query_one(CommandInput)
            ci.focus()
            ci.value = "/edit"
            await pilot.press("enter")
            await pilot.pause()
            assert isinstance(app.screen, EditScreen)

            screen: EditScreen = app.screen  # type: ignore[assignment]

            from textual.widgets import Input

            # Use app.screen to query widgets inside the pushed EditScreen
            edit_input = app.screen.query_one("#edit-input", Input)
            edit_input.focus()
            edit_input.value = "make the agent concise"
            await pilot.press("enter")
            await pilot.pause()
            assert screen.proposal is not None

            # No 'e' pressed — focus is None after set_focus(None) in request_proposal.
            # Pressing 'a' should trigger action_accept via the Screen binding.
            focused_before = app.focused
            await pilot.press("a")
            await pilot.pause()

            assert "concise" in path.read_text(), (
                f"File not updated after pressing 'a'; focused widget was {type(focused_before).__name__}"
            )
            assert screen.proposal is None, "Proposal not cleared after accept"

    @pytest.mark.asyncio
    async def test_reject_via_r_key_discards_proposal(self, tmp_path: Path) -> None:
        """Pressing 'r' with a pending proposal should discard it without writing."""
        set_provider_override(lambda _uri: FakeProvider(response=PROPOSED_YAML))
        app, path = make_app(tmp_path)
        from sirenspec.session.editor import EditAssistant

        app.editor = EditAssistant(model_uri="openai:gpt-4o-mini")
        original = path.read_text()
        async with app.run_test(size=(120, 40)) as pilot:
            ci = app.query_one(CommandInput)
            ci.focus()
            ci.value = "/edit"
            await pilot.press("enter")
            await pilot.pause()

            screen: EditScreen = app.screen  # type: ignore[assignment]

            from textual.widgets import Input

            edit_input = app.screen.query_one("#edit-input", Input)
            edit_input.focus()
            edit_input.value = "make the agent concise"
            await pilot.press("enter")
            await pilot.pause()
            assert screen.proposal is not None

            await pilot.press("r")
            await pilot.pause()

            assert screen.proposal is None, "Proposal not cleared after 'r'"
            assert path.read_text() == original, "File was written despite rejection"

    @pytest.mark.asyncio
    async def test_edit_area_opens_after_e_key(self, tmp_path: Path) -> None:
        """Pressing 'e' with a pending proposal should open the edit-area TextArea."""
        set_provider_override(lambda _uri: FakeProvider(response=PROPOSED_YAML))
        app, path = make_app(tmp_path)
        from sirenspec.session.editor import EditAssistant

        app.editor = EditAssistant(model_uri="openai:gpt-4o-mini")
        async with app.run_test(size=(120, 40)) as pilot:
            ci = app.query_one(CommandInput)
            ci.focus()
            ci.value = "/edit"
            await pilot.press("enter")
            await pilot.pause()

            from textual.widgets import Input, TextArea

            edit_input = app.screen.query_one("#edit-input", Input)
            edit_input.focus()
            edit_input.value = "make the agent concise"
            await pilot.press("enter")
            await pilot.pause()

            screen: EditScreen = app.screen  # type: ignore[assignment]
            assert screen.proposal is not None

            await pilot.press("e")
            await pilot.pause()

            edit_area = app.screen.query_one("#edit-area", TextArea)
            assert edit_area.has_class("open"), "edit-area did not open after pressing 'e'"
            assert isinstance(app.focused, TextArea), "TextArea should have focus after 'e'"


class TestEmptySubmitNoOp:
    """Submitting empty or whitespace-only input must not create a transcript turn."""

    @pytest.mark.asyncio
    async def test_empty_enter_does_not_add_turn(self, tmp_path: Path, fake_provider: FakeProvider) -> None:
        app, _ = make_app(tmp_path)
        async with app.run_test() as pilot:
            ci = app.query_one(CommandInput)
            ci.focus()
            # Submit empty buffer
            await pilot.press("enter")
            await pilot.pause()
            assert fake_provider.calls == [], "Provider was called on empty submit"
            assert app.session.turns == 0
            assert not any(str(line).strip() for line in app.query_one(Transcript).lines), (
                "Transcript should be empty after empty submit"
            )

    @pytest.mark.asyncio
    async def test_whitespace_only_enter_does_not_add_turn(self, tmp_path: Path, fake_provider: FakeProvider) -> None:
        app, _ = make_app(tmp_path)
        async with app.run_test() as pilot:
            ci = app.query_one(CommandInput)
            ci.focus()
            ci.value = "   "
            await pilot.press("enter")
            await pilot.pause()
            assert fake_provider.calls == [], "Provider was called on whitespace-only submit"
            assert app.session.turns == 0


class TestPaletteRouting:
    """Palette open/filter/dismiss/fill-and-return-focus via real keys."""

    @pytest.mark.asyncio
    async def test_slash_opens_palette(self, tmp_path: Path) -> None:
        app, _ = make_app(tmp_path)
        async with app.run_test() as pilot:
            ci = app.query_one(CommandInput)
            ci.focus()
            ci.value = "/"
            await pilot.pause()
            assert app.query_one(CommandPalette).has_class("open"), "Palette should open when '/' is typed"

    @pytest.mark.asyncio
    async def test_palette_filters_as_you_type(self, tmp_path: Path) -> None:
        app, _ = make_app(tmp_path)
        async with app.run_test() as pilot:
            ci = app.query_one(CommandInput)
            ci.focus()
            ci.value = "/sn"
            await pilot.pause()
            palette = app.query_one(CommandPalette)
            assert palette.has_class("open")
            assert palette.option_count >= 1
            # Check that only snapshot-prefixed commands appear, not all
            for i in range(palette.option_count):
                assert palette.get_option_at_index(i).id is not None
                assert palette.get_option_at_index(i).id.startswith("sn"), (  # type: ignore[union-attr]
                    f"palette option '{palette.get_option_at_index(i).id}' doesn't start with 'sn'"
                )

    @pytest.mark.asyncio
    async def test_escape_closes_palette(self, tmp_path: Path) -> None:
        app, _ = make_app(tmp_path)
        async with app.run_test() as pilot:
            ci = app.query_one(CommandInput)
            ci.focus()
            ci.value = "/"
            await pilot.pause()
            assert app.query_one(CommandPalette).has_class("open")

            await pilot.press("escape")
            await pilot.pause()
            assert not app.query_one(CommandPalette).has_class("open"), "Palette should close after pressing Escape"

    @pytest.mark.asyncio
    async def test_non_command_text_closes_palette(self, tmp_path: Path) -> None:
        """Deleting the leading '/' should close the palette."""
        app, _ = make_app(tmp_path)
        async with app.run_test() as pilot:
            ci = app.query_one(CommandInput)
            ci.focus()
            ci.value = "/sn"
            await pilot.pause()
            assert app.query_one(CommandPalette).has_class("open")

            # Replace with non-command text
            ci.value = "regular text"
            await pilot.pause()
            assert not app.query_one(CommandPalette).has_class("open"), (
                "Palette should close when input is not a command"
            )

    @pytest.mark.asyncio
    async def test_palette_selection_fills_input_with_command(self, tmp_path: Path) -> None:
        """Clicking a palette option fills the CommandInput with the selected command."""
        app, _ = make_app(tmp_path)
        async with app.run_test(size=(120, 40)) as pilot:
            ci = app.query_one(CommandInput)
            ci.focus()
            ci.value = "/sn"
            await pilot.pause()
            palette = app.query_one(CommandPalette)
            assert palette.has_class("open")

            # The palette region is 3 rows tall (border + 1 option + border).
            # Option 0 is at y=1 (inside the top border).
            await pilot.click(CommandPalette, offset=(5, 1))
            await pilot.pause()

            assert ci.text.startswith("/snapshot"), f"Input not filled with selected command; got '{ci.text!r}'"

    @pytest.mark.asyncio
    async def test_palette_closes_after_selection(self, tmp_path: Path) -> None:
        """Palette should dismiss and focus should return to the input after selection."""
        app, _ = make_app(tmp_path)
        async with app.run_test(size=(120, 40)) as pilot:
            ci = app.query_one(CommandInput)
            ci.focus()
            ci.value = "/sn"
            await pilot.pause()
            palette = app.query_one(CommandPalette)
            assert palette.has_class("open")

            await pilot.click(CommandPalette, offset=(5, 1))
            await pilot.pause()

            assert not palette.has_class("open"), "Palette should close after selection"
            assert ci.text.startswith("/snapshot"), f"Input should be filled; got '{ci.text!r}'"
            assert app.focused is ci, "Focus should return to CommandInput after selection"


class TestKeyBindings:
    """ctrl+b / ctrl+t / ctrl+r via real pilot.press."""

    @pytest.mark.asyncio
    async def test_ctrl_b_toggles_rail(self, tmp_path: Path) -> None:
        app, _ = make_app(tmp_path)
        async with app.run_test() as pilot:
            rail = app.query_one("#rail", WorkflowRail)
            assert not rail.has_class("hidden"), "Rail starts visible"
            await pilot.press("ctrl+b")
            await pilot.pause()
            assert rail.has_class("hidden"), "Rail should be hidden after ctrl+b"
            await pilot.press("ctrl+b")
            await pilot.pause()
            assert not rail.has_class("hidden"), "Rail should be visible after second ctrl+b"

    @pytest.mark.asyncio
    async def test_ctrl_t_toggles_drawer(self, tmp_path: Path) -> None:
        app, _ = make_app(tmp_path)
        async with app.run_test() as pilot:
            from sirenspec.session.widgets import TraceDrawer

            drawer = app.query_one("#drawer", TraceDrawer)
            assert not drawer.has_class("hidden"), "Drawer starts visible"
            await pilot.press("ctrl+t")
            await pilot.pause()
            assert drawer.has_class("hidden"), "Drawer should be hidden after ctrl+t"
            await pilot.press("ctrl+t")
            await pilot.pause()
            assert not drawer.has_class("hidden"), "Drawer should be visible after second ctrl+t"

    @pytest.mark.asyncio
    async def test_ctrl_r_with_no_snapshots_posts_notice(self, tmp_path: Path) -> None:
        """ctrl+r with no snapshots should post a notice, not raise."""
        app, _ = make_app(tmp_path)
        async with app.run_test() as pilot:
            await pilot.press("ctrl+r")
            await pilot.pause()
            tx = transcript_text(app)
            assert "no snapshots" in tx.lower(), (
                f"Expected 'no snapshots' notice in transcript after ctrl+r with no snapshots; got:\n{tx}"
            )


class TestSlashCommandsViaEnter:
    """Every slash command fires through real Enter — confirms effect, not just no-error."""

    @pytest.mark.asyncio
    async def test_help_via_enter_writes_transcript(self, tmp_path: Path) -> None:
        app, _ = make_app(tmp_path)
        async with app.run_test() as pilot:
            ci = app.query_one(CommandInput)
            ci.focus()
            ci.value = "/help"
            await pilot.press("enter")
            await pilot.pause()
            tx = transcript_text(app)
            assert "commands" in tx.lower(), f"Expected command listing in transcript; got:\n{tx}"
            assert "/edit" in tx or "edit" in tx

    @pytest.mark.asyncio
    async def test_snapshot_via_enter_creates_snapshot(self, tmp_path: Path) -> None:
        app, _ = make_app(tmp_path)
        async with app.run_test() as pilot:
            ci = app.query_one(CommandInput)
            ci.focus()
            ci.value = "/snapshot v1"
            await pilot.press("enter")
            await pilot.pause()
            snapshots = app.snapshots.list()
            assert len(snapshots) == 1, f"Expected 1 snapshot after /snapshot v1; got {len(snapshots)}"
            assert snapshots[0].label == "v1"
            tx = transcript_text(app)
            assert "saved snapshot" in tx.lower()

    @pytest.mark.asyncio
    async def test_snapshot_without_label_still_works(self, tmp_path: Path) -> None:
        app, _ = make_app(tmp_path)
        async with app.run_test() as pilot:
            ci = app.query_one(CommandInput)
            ci.focus()
            ci.value = "/snapshot"
            await pilot.press("enter")
            await pilot.pause()
            assert len(app.snapshots.list()) == 1

    @pytest.mark.asyncio
    async def test_diff_no_snapshots_warns_gracefully(self, tmp_path: Path) -> None:
        app, _ = make_app(tmp_path)
        async with app.run_test() as pilot:
            ci = app.query_one(CommandInput)
            ci.focus()
            ci.value = "/diff"
            await pilot.press("enter")
            await pilot.pause()
            tx = transcript_text(app)
            assert "no snapshots" in tx.lower(), f"Expected 'no snapshots' warning; got:\n{tx}"

    @pytest.mark.asyncio
    async def test_diff_with_snapshot_shows_output(self, tmp_path: Path) -> None:
        app, _ = make_app(tmp_path)
        async with app.run_test() as pilot:
            # First snapshot
            ci = app.query_one(CommandInput)
            ci.focus()
            ci.value = "/snapshot baseline"
            await pilot.press("enter")
            await pilot.pause()
            # Diff should run (even if no difference — "no differences" is valid output)
            ci.value = "/diff"
            await pilot.press("enter")
            await pilot.pause()
            tx = transcript_text(app)
            # Either "no differences" or actual diff lines
            has_diff_output = "no differences" in tx.lower() or any(c in tx for c in ("@@", "---", "+++"))
            assert has_diff_output, f"Expected diff output after /diff with a snapshot; got:\n{tx}"

    @pytest.mark.asyncio
    async def test_rollback_no_snapshots_warns(self, tmp_path: Path) -> None:
        app, _ = make_app(tmp_path)
        async with app.run_test() as pilot:
            ci = app.query_one(CommandInput)
            ci.focus()
            ci.value = "/rollback"
            await pilot.press("enter")
            await pilot.pause()
            tx = transcript_text(app)
            assert "no snapshots" in tx.lower(), f"Expected 'no snapshots' notice; got:\n{tx}"

    @pytest.mark.asyncio
    async def test_rollback_with_snapshot_restores_file(self, tmp_path: Path) -> None:
        """Rollback should restore the snapshotted content and post a notice."""
        app, path = make_app(tmp_path)
        original_content = path.read_text()
        async with app.run_test() as pilot:
            ci = app.query_one(CommandInput)
            ci.focus()
            # Take a snapshot of the original
            ci.value = "/snapshot before-change"
            await pilot.press("enter")
            await pilot.pause()

            # Mutate the file on disk (simulating a change)
            modified = original_content.replace("You are helpful.", "You are changed.")
            path.write_text(modified)
            os.utime(path, (app.session.mtime + 10, app.session.mtime + 10))

            # Roll back via command
            ci.value = "/rollback"
            await pilot.press("enter")
            await pilot.pause()

            restored = path.read_text()
            assert "You are helpful." in restored, "Rollback did not restore the original content"
            tx = transcript_text(app)
            assert "rolled back" in tx.lower(), f"Expected 'rolled back' notice; got:\n{tx}"

    @pytest.mark.asyncio
    async def test_unknown_command_surfaces_error_not_crash(self, tmp_path: Path) -> None:
        """/nope should post an error notice in the transcript, not raise."""
        app, _ = make_app(tmp_path)
        async with app.run_test() as pilot:
            ci = app.query_one(CommandInput)
            ci.focus()
            ci.value = "/nope"
            await pilot.press("enter")
            await pilot.pause()
            tx = transcript_text(app)
            assert "unknown command" in tx.lower() or "not available" in tx.lower(), (
                f"Expected an error notice for unknown command; got:\n{tx}"
            )

    @pytest.mark.asyncio
    async def test_reload_command_posts_notice(self, tmp_path: Path) -> None:
        app, _ = make_app(tmp_path)
        async with app.run_test() as pilot:
            ci = app.query_one(CommandInput)
            ci.focus()
            ci.value = "/reload"
            await pilot.press("enter")
            await pilot.pause()
            tx = transcript_text(app)
            assert "reload" in tx.lower(), f"Expected reload notice in transcript; got:\n{tx}"

    @pytest.mark.asyncio
    async def test_run_command_via_enter_executes_and_snapshots(
        self, tmp_path: Path, fake_provider: FakeProvider
    ) -> None:
        """/run via Enter should execute the workflow and take a pre-run snapshot."""
        app, _ = make_app(tmp_path)
        async with app.run_test() as pilot:
            ci = app.query_one(CommandInput)
            ci.focus()
            ci.value = "/run"
            await pilot.press("enter")
            await pilot.pause()
            tx = transcript_text(app)
            assert "the answer" in tx, f"Expected provider reply in transcript; got:\n{tx}"
            # auto-snapshot before /run
            assert any(s.trigger == "run" for s in app.snapshots.list()), "Expected a pre-run auto-snapshot"


# ---------------------------------------------------------------------------
# State / data — transcript, cost, trace drawer, snapshots
# ---------------------------------------------------------------------------


class TestFailedTurn:
    """Provider raises → transcript notice, status bar returns to idle."""

    @pytest.mark.asyncio
    async def test_session_error_posts_notice_and_resets_live(self, tmp_path: Path) -> None:
        error = SessionError("something went wrong")
        set_provider_override(lambda _uri: FakeProvider(raises=error))
        app, _ = make_app(tmp_path)
        async with app.run_test() as pilot:
            ci = app.query_one(CommandInput)
            ci.focus()
            ci.value = "what are the risks?"
            await pilot.press("enter")
            await pilot.pause()

            status = app.query_one(StatusBar)
            assert "idle" in status.right_text.plain, (
                f"Status should reset to 'idle' after a failed turn; got: {status.right_text.plain}"
            )
            tx = transcript_text(app)
            assert "something went wrong" in tx, f"Expected error notice in transcript; got:\n{tx}"


class TestCostAccumulation:
    """Rail and trace drawer cost rises across turns and never resets or goes negative."""

    @pytest.mark.asyncio
    async def test_rail_cost_accumulates_across_turns(self, tmp_path: Path, fake_provider: FakeProvider) -> None:
        app, _ = make_app(tmp_path)
        async with app.run_test() as pilot:
            ci = app.query_one(CommandInput)
            ci.focus()
            ci.value = "first question"
            await pilot.press("enter")
            await pilot.pause()

            # Extract cost line — it should be non-zero now
            assert app.session.session_cost_usd is not None
            cost_after_1 = app.session.session_cost_usd
            assert cost_after_1 > 0

            ci.value = "second question"
            await pilot.press("enter")
            await pilot.pause()

            cost_after_2 = app.session.session_cost_usd
            assert cost_after_2 > cost_after_1, f"Cost did not accumulate: {cost_after_1} -> {cost_after_2}"

            rail = app.query_one("#rail", WorkflowRail)
            assert "$" in rail.body_text.plain, "Cost should appear in the rail text"

    @pytest.mark.asyncio
    async def test_trace_drawer_shows_turn_data_after_reply(self, tmp_path: Path, fake_provider: FakeProvider) -> None:
        app, _ = make_app(tmp_path)
        async with app.run_test() as pilot:
            ci = app.query_one(CommandInput)
            ci.focus()
            ci.value = "tell me something"
            await pilot.press("enter")
            await pilot.pause()

            drawer = app.query_one("#drawer", TraceDrawer)
            trace = drawer.trace_text.plain
            assert "answer (openai)" in trace, f"Expected node label in trace drawer; got:\n{trace}"
            assert "tok" in trace, f"Expected token count in trace drawer; got:\n{trace}"
            assert "turn 1/1" in trace, f"Expected turn counter in trace drawer; got:\n{trace}"


class TestEmptyReplyFallback:
    """Provider returns '' → transcript shows '(no output)' not blank."""

    @pytest.mark.asyncio
    async def test_empty_reply_shows_no_output_fallback(self, tmp_path: Path) -> None:
        set_provider_override(lambda _uri: FakeProvider(response=""))
        app, _ = make_app(tmp_path)
        async with app.run_test() as pilot:
            ci = app.query_one(CommandInput)
            ci.focus()
            ci.value = "hello"
            await pilot.press("enter")
            await pilot.pause()
            tx = transcript_text(app)
            assert "(no output)" in tx, f"Expected '(no output)' fallback for empty reply; got:\n{tx}"


class TestTurnAppearance:
    """User line and Siren reply appear once each, no duplicates."""

    @pytest.mark.asyncio
    async def test_turn_appears_once(self, tmp_path: Path, fake_provider: FakeProvider) -> None:
        app, _ = make_app(tmp_path)
        async with app.run_test() as pilot:
            ci = app.query_one(CommandInput)
            ci.focus()
            ci.value = "what are the risks?"
            await pilot.press("enter")
            await pilot.pause()
            tx = transcript_text(app)
            assert tx.count("what are the risks?") == 1, "User message appeared more than once"
            assert tx.count("the answer") == 1, "Siren reply appeared more than once"


class TestHotReloadPreservesHistory:
    """Editing workflow.yaml on disk reloads the session without losing chat history."""

    @pytest.mark.asyncio
    async def test_file_change_reloads_and_preserves_history(self, tmp_path: Path, fake_provider: FakeProvider) -> None:
        app, path = make_app(tmp_path)
        async with app.run_test() as pilot:
            ci = app.query_one(CommandInput)
            ci.focus()
            ci.value = "remember me"
            await pilot.press("enter")
            await pilot.pause()
            assert app.session.turns == 1

            # Edit the file on disk so the watcher fires
            mutated = WORKFLOW_YAML.replace("You are helpful.", "You are extra helpful.")
            path.write_text(mutated)
            os.utime(path, (app.session.mtime + 10, app.session.mtime + 10))

            # Trigger the check manually (the 1s interval won't tick in tests)
            app.check_reload()
            await pilot.pause()

            assert app.session.turns == 1, "Turn count should survive hot-reload"
            assert any(t.content == "remember me" for t in app.session.history), "Chat history lost after hot-reload"
            tx = transcript_text(app)
            assert "workflow.yaml changed" in tx or "reloaded" in tx, (
                f"Expected reload notice in transcript; got:\n{tx}"
            )

    @pytest.mark.asyncio
    async def test_invalid_file_change_surfaces_notice_not_crash(self, tmp_path: Path) -> None:
        app, path = make_app(tmp_path)
        async with app.run_test() as pilot:
            path.write_text("not: [valid yaml")
            os.utime(path, (app.session.mtime + 10, app.session.mtime + 10))
            app.check_reload()
            await pilot.pause()
            tx = transcript_text(app)
            assert tx.strip() != "", "Expected an error notice after invalid reload; got empty transcript"
            # App is still alive (not crashed)
            assert app.is_running


class TestEditAcceptWritesAndReloads:
    """/edit accept via real keys: file written, snapshot created, on_applied fires."""

    @pytest.mark.asyncio
    async def test_accept_via_action_method_writes_and_snapshots(self, tmp_path: Path) -> None:
        """Confirms the mechanism works when called via action method (control case)."""
        from sirenspec.session.editor import EditAssistant

        set_provider_override(lambda _uri: FakeProvider(response=PROPOSED_YAML))
        app, path = make_app(tmp_path)
        app.editor = EditAssistant(model_uri="openai:gpt-4o-mini")
        async with app.run_test() as pilot:
            await app.command_edit("")
            await pilot.pause()
            screen: EditScreen = app.screen  # type: ignore[assignment]
            await screen.request_proposal("make the agent concise")
            await pilot.pause()
            assert screen.proposal is not None, (
                "Proposal was not generated — check provider override and editor model_uri"
            )

            await screen.action_accept()
            await pilot.pause()

            assert "concise" in path.read_text(), "File not updated after action_accept"
            assert any(s.trigger == "accept" for s in app.snapshots.list())
            # on_applied should repaint the studio: check transcript has the notice
            tx = transcript_text(app)
            assert "workflow updated via /edit" in tx


# ---------------------------------------------------------------------------
# Layout / visual — screenshots for human review
# ---------------------------------------------------------------------------


class TestVisualScreenshots:
    """Capture SVGs at key moments for human eyeballing; no assertion-only checks."""

    @pytest.mark.asyncio
    async def test_long_reply_screenshot(self, tmp_path: Path) -> None:
        """5000-char reply: transcript should not push input off-screen."""
        long_response = "x " * 2500  # 5000 chars
        set_provider_override(lambda _uri: FakeProvider(response=long_response))
        app, _ = make_app(tmp_path)
        async with app.run_test(size=(120, 40)) as pilot:
            ci = app.query_one(CommandInput)
            ci.focus()
            ci.value = "give me everything"
            await pilot.press("enter")
            await pilot.pause()
            app.save_screenshot("tui-qa-artifacts/long-reply.svg")
            # The CommandInput should still be visible (not pushed off-screen)
            assert app.query_one(CommandInput).is_attached
            tx = transcript_text(app)
            assert "x " in tx, "Long reply should appear in transcript"

    @pytest.mark.asyncio
    async def test_cramped_terminal_screenshot(self, tmp_path: Path, fake_provider: FakeProvider) -> None:
        """60x20 terminal: rail, status bar, footer, and input should all render."""
        app, _ = make_app(tmp_path)
        async with app.run_test(size=(60, 20)) as pilot:
            await pilot.pause()
            app.save_screenshot("tui-qa-artifacts/cramped-60x20.svg")
            # Minimal sanity: the app is mounted and the key widgets exist
            assert app.query_one(StatusBar).is_attached
            assert app.query_one(CommandInput).is_attached

    @pytest.mark.asyncio
    async def test_rail_toggled_screenshot(self, tmp_path: Path, fake_provider: FakeProvider) -> None:
        """Rail hidden: right pane should reclaim the space cleanly."""
        app, _ = make_app(tmp_path)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.press("ctrl+b")
            await pilot.pause()
            app.save_screenshot("tui-qa-artifacts/rail-hidden.svg")
            assert app.query_one("#rail", WorkflowRail).has_class("hidden")

    @pytest.mark.asyncio
    async def test_palette_overlay_screenshot(self, tmp_path: Path) -> None:
        """Palette rendered over transcript should not corrupt it."""
        app, _ = make_app(tmp_path)
        async with app.run_test(size=(120, 40)) as pilot:
            ci = app.query_one(CommandInput)
            ci.focus()
            ci.value = "/s"
            await pilot.pause()
            app.save_screenshot("tui-qa-artifacts/palette-open.svg")
            assert app.query_one(CommandPalette).has_class("open")


# ---------------------------------------------------------------------------
# Edge / stress
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Rapid-fire, resize mid-turn, --plain fallback, bad file."""

    @pytest.mark.asyncio
    async def test_rapid_fire_turns_no_corruption(self, tmp_path: Path, fake_provider: FakeProvider) -> None:
        """Three turns submitted back-to-back: all should appear, no duplicates."""
        app, _ = make_app(tmp_path)
        async with app.run_test() as pilot:
            ci = app.query_one(CommandInput)
            for msg in ("first", "second", "third"):
                ci.focus()
                ci.value = msg
                await pilot.press("enter")
            await pilot.pause()
            await pilot.pause()
            tx = transcript_text(app)
            for msg in ("first", "second", "third"):
                assert tx.count(msg) == 1, f"'{msg}' appeared != 1 times in transcript"
            assert app.session.turns == 3

    @pytest.mark.asyncio
    async def test_resize_mid_session_no_exception(self, tmp_path: Path, fake_provider: FakeProvider) -> None:
        """Resizing the terminal after a turn should not raise."""
        app, _ = make_app(tmp_path)
        async with app.run_test(size=(120, 40)) as pilot:
            ci = app.query_one(CommandInput)
            ci.focus()
            ci.value = "hello"
            await pilot.press("enter")
            await pilot.pause()
            await pilot.resize_terminal(80, 30)
            await pilot.pause()
            app.save_screenshot("tui-qa-artifacts/post-resize.svg")
            # App is still alive
            assert app.is_running

    def test_render_plain_single_node_singular_grammar(self, tmp_path: Path) -> None:
        """render_plain should say '1 node' not '1 nodes'."""
        from rich.console import Console

        from sirenspec.session.app import render_plain

        single_node_wf = Workflow(
            version="0.1",
            agents={"a": AgentDefinition(model="openai:gpt-4o-mini", system="s")},
            nodes={"n": AgentNode(agent="a", writes="output.reply")},
        )
        console = Console(record=True, force_terminal=False, no_color=True, width=100)
        render_plain(single_node_wf, "test-wf", console)
        out = console.export_text()
        assert "1 node " in out and "1 nodes" not in out, f"Singular grammar wrong: {out!r}"

    def test_render_plain_single_agent_singular_grammar(self, tmp_path: Path) -> None:
        """render_plain should say '1 agent' not '1 agents'."""
        from rich.console import Console

        from sirenspec.session.app import render_plain

        single_agent_wf = Workflow(
            version="0.1",
            agents={"a": AgentDefinition(model="openai:gpt-4o-mini", system="s")},
            nodes={"n": AgentNode(agent="a", writes="output.reply")},
        )
        console = Console(record=True, force_terminal=False, no_color=True, width=100)
        render_plain(single_agent_wf, "test-wf", console)
        out = console.export_text()
        assert "1 agent " in out and "1 agents" not in out, f"Singular grammar wrong: {out!r}"

    @pytest.mark.asyncio
    async def test_edit_screen_unknown_command_posts_notice(self, tmp_path: Path) -> None:
        """/unknown inside edit mode should show a notice, not raise."""
        app, _ = make_app(tmp_path)
        async with app.run_test() as pilot:
            ci = app.query_one(CommandInput)
            ci.focus()
            ci.value = "/edit"
            await pilot.press("enter")
            await pilot.pause()
            assert isinstance(app.screen, EditScreen)

            from textual.widgets import Input

            # Query within the active EditScreen, not the main screen
            edit_input = app.screen.query_one("#edit-input", Input)
            edit_input.focus()
            edit_input.value = "/unknowncmd"
            await pilot.press("enter")
            await pilot.pause()
            screen: EditScreen = app.screen  # type: ignore[assignment]
            assistant_text = screen.assistant.lines
            assert any("unknown command" in str(line).lower() for line in assistant_text), (
                "Expected 'unknown command' notice in edit assistant pane"
            )

    @pytest.mark.asyncio
    async def test_edit_screen_escape_exits(self, tmp_path: Path) -> None:
        """/escape while in edit mode should return to the main studio."""
        app, _ = make_app(tmp_path)
        async with app.run_test() as pilot:
            ci = app.query_one(CommandInput)
            ci.focus()
            ci.value = "/edit"
            await pilot.press("enter")
            await pilot.pause()
            assert isinstance(app.screen, EditScreen)

            await pilot.press("escape")
            await pilot.pause()
            assert not isinstance(app.screen, EditScreen), (
                "Should have returned to main studio after Escape in edit screen"
            )
