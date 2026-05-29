"""The ``/edit`` split-screen suggestive editor for ``sirenspec launch``.

:class:`EditScreen` is the full-screen split view the design shows: the workflow on the left,
the assistant conversation and the proposed diff on the right.  Every assistant suggestion is
an approve-able diff — accept (validate + write + snapshot), edit (tweak in place), or reject.
``/test`` runs the in-progress edit against the live runtime without leaving edit mode.
"""

from __future__ import annotations

from collections.abc import Callable

from rich.text import Text
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
                yield TextArea(id="edit-area", language="yaml")
                yield Input(placeholder="Describe a change, or /test  /exit-edit", id="edit-input")
        yield Static(id="edit-footer")

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

        :param event: The input submission event.
        """
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
        """Ask the assistant for a proposed change and render its diff.

        :param instruction: The user's natural-language edit instruction.
        """
        self.assistant.write(Text(f"You: {instruction}", style=theme.YOU))
        try:
            self.proposal = await self.editor.propose(self.current_yaml(), instruction)
        except SessionError as exc:
            self.assistant.write(Text(str(exc), style=theme.YOU))
            return
        self.render_proposal(self.proposal)
        self.set_focus(None)  # blur input so a/e/r act as decisions

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
