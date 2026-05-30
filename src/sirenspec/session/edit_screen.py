"""The ``/edit`` split-screen suggestive editor for ``sirenspec launch``.

:class:`EditScreen` is the full-screen split view the design shows: the workflow on the left,
the assistant conversation and the proposed diff on the right.  Every assistant suggestion is
an approve-able diff — accept (validate + write + snapshot), edit (tweak in place), or reject.
``/test`` runs the in-progress edit against the live runtime without leaving edit mode.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

from rich.text import Text
from textual import events
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import Input, RichLog, Static, TextArea

from sirenspec.exceptions import SessionError
from sirenspec.session import theme
from sirenspec.session.commands import parse_command_line
from sirenspec.session.editor import EditAssistant, ProposedChange
from sirenspec.session.runtime import TurnResult, WorkflowSession, stream_execution
from sirenspec.session.snapshots import SnapshotStore


class ProposalEditArea(TextArea):
    """A TextArea that routes 'a' and 'r' to the screen's decision actions.

    Textual strips printable-character bindings from parent widgets when a TextArea has
    focus (``check_consume_key`` returns ``True`` for printable chars), so Screen-level
    ``Binding("a", "accept")`` never fires while this widget is focused.  This subclass
    handles ``a``/``r`` in ``_on_key`` directly so the accept/reject flow always works,
    regardless of focus state.
    """

    async def _on_key(self, event: events.Key) -> None:
        """Route 'a'/'r' as decision keys when a proposal is pending.

        :param event: The key event.
        """
        screen = self.screen
        if getattr(screen, "proposal", None) is not None:
            if event.key == "a":
                event.prevent_default()
                event.stop()
                await screen.action_accept()
                return
            if event.key == "r":
                event.prevent_default()
                event.stop()
                screen.action_reject()
                return
        await super()._on_key(event)


EDIT_CSS = """
EditScreen { background: $background; layers: base; }

#edit-header {
    dock: top;
    height: 2;
    padding: 0 1;
    background: $surface;
    border-bottom: solid $border;
}

#edit-split { height: 1fr; }

#edit-yaml {
    width: 40%;
    padding: 1 1;
    border-right: solid $border;
    background: $surface;
}

#edit-right { width: 1fr; }

#assistant { height: 1fr; padding: 0 1; background: $background; }

#edit-area { height: 12; display: none; border: round $primary; }
#edit-area.open { display: block; }

#edit-input {
    dock: bottom;
    height: 3;
    margin: 0 1;
    border: round $border;
    background: $background;
}
#edit-input:focus { border: round $primary; }

#edit-footer {
    dock: bottom;
    height: 2;
    padding: 0 1;
    background: $surface;
    border-top: solid $border;
}
"""


class EditScreen(Screen):
    """The suggestive ``/edit`` split view: workflow on the left, assistant + diff on the right.

    :param editor: The :class:`~sirenspec.session.editor.EditAssistant` engine.
    :param session: The live :class:`~sirenspec.session.runtime.WorkflowSession`.
    :param snapshots: The :class:`~sirenspec.session.snapshots.SnapshotStore`.
    :param on_applied: Callback invoked after an accepted change is written, so the main
        studio can repaint its chrome.
    """

    CSS = EDIT_CSS
    BINDINGS = [
        Binding("a", "accept", "accept", show=False),
        Binding("e", "edit", "edit", show=False),
        Binding("r", "reject", "reject", show=False),
        Binding("escape", "exit_edit", "exit", show=False),
    ]

    def __init__(
        self,
        editor: EditAssistant,
        session: WorkflowSession,
        snapshots: SnapshotStore,
        on_applied: Callable[[], None],
    ) -> None:
        super().__init__()
        self.editor = editor
        self.session = session
        self.snapshots = snapshots
        self.on_applied = on_applied
        self.proposal: ProposedChange | None = None

    def compose(self) -> ComposeResult:
        """Compose the split editor layout.

        :returns: The screen's child widgets.
        """
        yield Static(id="edit-header")
        with Horizontal(id="edit-split"):
            yield VerticalScroll(Static(id="edit-yaml-body"), id="edit-yaml")
            with Vertical(id="edit-right"):
                yield RichLog(id="assistant", wrap=True, markup=False)
                yield ProposalEditArea(id="edit-area", language="yaml")
                yield Input(placeholder="Describe a change, or /test  /exit-edit", id="edit-input")
        yield Static(id="edit-footer")

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        """Disable accept/reject when the instruction input has focus so typing still works.

        :param action: The action being checked.
        :param parameters: Action parameters (unused).
        :returns: ``False`` to disable the action (key falls through to the focused widget),
            ``True`` to allow it (matches the base-class default).
        """
        if action in ("accept", "reject") and isinstance(self.focused, Input):
            return False
        return True

    def on_mount(self) -> None:
        """Paint the header, footer, and workflow view, then focus the instruction input."""
        provider = self.editor.model_uri or "auto"
        header = Text()
        header.append("─ ", style=theme.TERM_FAINT)
        header.append(f"SirenSpec · /edit {self.session.name}", style=f"bold {theme.LIGHT}")
        header.append(f"   {provider}", style=theme.TERM_DIM)
        self.query_one("#edit-header", Static).update(header)

        footer = Text()
        footer.append("✎ editor · accept auto-snapshots   ", style=theme.CREST_LIGHT)
        footer.append("[a]ccept [e]dit [r]eject   ", style=theme.TERM_DIM)
        footer.append("/test   /exit-edit   ^C", style=theme.TERM_MUTED)
        self.query_one("#edit-footer", Static).update(footer)

        self.refresh_yaml()
        self.assistant.write(Text("Describe a change and I'll propose a diff.", style=theme.TERM_DIM))
        self.query_one("#edit-input", Input).focus()

    @property
    def assistant(self) -> RichLog:
        """Return the assistant conversation log.

        :returns: The assistant :class:`~textual.widgets.RichLog`.
        """
        return self.query_one("#assistant", RichLog)

    def refresh_yaml(self) -> None:
        """Repaint the left pane with the current workflow YAML and snapshot label."""
        body = Text()
        body.append(f"workflow.yaml ◇ {self.snapshots.latest_label()}\n\n", style=theme.TERM_MUTED)
        body.append(self.current_yaml(), style=theme.TERM_FG)
        self.query_one("#edit-yaml-body", Static).update(body)

    def current_yaml(self) -> str:
        """Read the current working workflow YAML.

        :returns: The file's text, or ``""`` if it cannot be read.
        """
        try:
            return self.session.workflow_path.read_text(encoding="utf-8")
        except OSError:
            return ""

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Route a submitted line as an edit-mode command or an edit instruction.

        Stops the message so the parent :class:`LaunchApp.on_input_submitted` does not
        also try to dispatch it (and post a spurious "unknown command" for ``/exit-edit``).

        :param event: The input submission event.
        """
        event.stop()
        line = event.value.strip()
        self.query_one("#edit-input", Input).value = ""
        if not line:
            return
        if line.startswith("/"):
            name, _ = parse_command_line(line)
            if name == "test":
                await self.action_test()
            elif name in ("exit-edit", "exit"):
                self.action_exit_edit()
            else:
                self.assistant.write(Text(f"unknown command /{name}", style=theme.YOU))
            return
        await self.request_proposal(line)

    async def request_proposal(self, instruction: str) -> None:
        """Ask the assistant for a proposed change, streaming progress as it arrives.

        The assistant log gets a live ``▸ thinking · NN chars`` indicator that updates on
        every streamed chunk, so the screen never feels frozen during the round-trip.  Once
        the full YAML is received the indicator is replaced with the unified diff and the
        decision-key hint.

        :param instruction: The user's natural-language edit instruction.
        """
        self.assistant.write(Text(f"You: {instruction}", style=theme.YOU))
        received = 0
        ticks = 0
        self.assistant.write(Text("▸ thinking…", style=theme.CREST_LIGHT))

        def on_chunk(chunk: str) -> None:
            nonlocal received, ticks
            received += len(chunk)
            ticks += 1
            # Cheap throttle: repaint at most every few chunks to keep the log readable.
            if ticks % 5 == 0 or ticks == 1:
                self.assistant.write(Text(f"  · streaming · {received} chars", style=theme.TERM_FAINT))

        try:
            self.proposal = await self.editor.propose_stream(self.current_yaml(), instruction, on_chunk)
        except SessionError as exc:
            self.assistant.write(Text(str(exc), style=theme.YOU))
            return
        self.render_proposal(self.proposal)
        # Blur the Input AND focus the screen so a/e/r reach the screen-level bindings
        # immediately — without an explicit screen focus, the next keypress can race the
        # focus change and land in the still-active Input widget.
        self.query_one("#edit-input", Input).blur()
        self.set_focus(None)
        self.focus()

    def render_proposal(self, proposal: ProposedChange) -> None:
        """Render a proposed change's diff and the decision keys in the assistant pane.

        :param proposal: The proposed change to render.
        """
        self.assistant.write(Text("Proposed change ▾", style=f"bold {theme.LIGHT}"))
        if not proposal.diff_lines:
            self.assistant.write(Text("  (no changes proposed)", style=theme.TERM_FAINT))
        for line in proposal.diff_lines:
            self.assistant.write(Text(line, style=diff_style(line)))
        self.assistant.write(Text("[a]ccept   [e]dit   [r]eject", style=theme.CREST_LIGHT))

    def proposed_content(self) -> str:
        """Return the content to apply: the edited buffer if open, else the proposal.

        :returns: The YAML text to validate and write.
        """
        edit_area = self.query_one("#edit-area", TextArea)
        if edit_area.has_class("open"):
            return edit_area.text
        return self.proposal.new_yaml if self.proposal is not None else ""

    async def action_accept(self) -> None:
        """Validate the pending proposal, then write it and snapshot it on success."""
        if self.proposal is None:
            return
        content = self.proposed_content()
        # Surface "applying…" immediately, then yield so it paints before the synchronous
        # validate + write + reload chain runs (which otherwise looks like a freeze).
        self.assistant.write(Text("▸ applying…", style=theme.CREST_LIGHT))
        await asyncio.sleep(0)
        try:
            self.editor.validate(content, self.session.workflow_path)
        except SessionError as exc:
            self.assistant.write(Text(str(exc), style=theme.YOU))
            return
        self.session.workflow_path.write_text(content, encoding="utf-8")
        snapshot = self.snapshots.create(label=f"accept: {self.proposal.instruction}"[:48], trigger="accept")
        try:
            self.session.reload()
        except SessionError as exc:
            self.assistant.write(Text(str(exc), style=theme.YOU))
            return
        self.clear_proposal()
        self.refresh_yaml()
        self.on_applied()
        self.assistant.write(Text(f"accepted ✓ snapshot {snapshot.ref}", style=theme.LIGHT))

    def action_reject(self) -> None:
        """Discard the pending proposal."""
        if self.proposal is None:
            return
        self.clear_proposal()
        self.assistant.write(Text("rejected", style=theme.TERM_DIM))
        self.query_one("#edit-input", Input).focus()

    def action_edit(self) -> None:
        """Open the proposed YAML in an editable buffer for manual tweaks before accepting."""
        if self.proposal is None:
            return
        edit_area = self.query_one("#edit-area", TextArea)
        edit_area.text = self.proposed_content()
        edit_area.add_class("open")
        edit_area.focus()
        self.assistant.write(Text("editing the proposal — press [a]ccept to apply", style=theme.TERM_DIM))

    def clear_proposal(self) -> None:
        """Drop the pending proposal and close the edit buffer."""
        self.proposal = None
        self.query_one("#edit-area", TextArea).remove_class("open")

    async def action_test(self) -> None:
        """Run the in-progress edit against the live runtime without leaving edit mode."""
        if self.proposal is not None:
            try:
                workflow = self.editor.validate(self.proposed_content(), self.session.workflow_path)
            except SessionError as exc:
                self.assistant.write(Text(str(exc), style=theme.YOU))
                return
        else:
            workflow = self.session.workflow

        self.snapshots.create(label="pre-test", trigger="test")
        try:
            user_input = self.session.resolve_run_input()
        except SessionError as exc:
            self.assistant.write(Text(str(exc), style=theme.YOU))
            return
        self.assistant.write(Text("testing the in-progress edit…", style=theme.CREST_LIGHT))
        async for event in stream_execution(workflow, user_input, None):
            if isinstance(event, TurnResult):
                self.assistant.write(Text(f"Siren: {event.text}", style=theme.LIGHT))

    def action_exit_edit(self) -> None:
        """Leave edit mode and return to the main studio."""
        self.app.pop_screen()


def diff_style(line: str) -> str:
    """Return the Rich style for a unified-diff line.

    :param line: A single unified-diff line.
    :returns: The colour to render it in.
    """
    if line.startswith("+") and not line.startswith("+++"):
        return theme.LIGHT
    if line.startswith("-") and not line.startswith("---"):
        return "#ff5f57"
    if line.startswith("@@"):
        return theme.CREST_LIGHT
    return theme.TERM_DIM
