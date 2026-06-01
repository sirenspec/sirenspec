"""The full-screen Textual application for ``sirenspec launch``.

:class:`LaunchApp` composes the studio shell from the widgets in
:mod:`sirenspec.session.widgets`, applies the brand theme from
:mod:`sirenspec.session.theme`, and routes input through the slash-command registry in
:mod:`sirenspec.session.commands`.  Non-interactive terminals fall back to a plain Rich
console via :func:`run_launch`.
"""

from __future__ import annotations

import asyncio
import sys
from asyncio import Task
from pathlib import Path

from rich.console import Console
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.events import Resize
from textual.widgets import OptionList, TextArea

from sirenspec.core.models import Workflow
from sirenspec.exceptions import ProviderError, SessionError, SirenSpecError
from sirenspec.session import theme
from sirenspec.session.commands import CommandHandler, CommandRegistry, is_command, make_registry
from sirenspec.session.edit_screen import EditScreen
from sirenspec.session.editor import EditAssistant
from sirenspec.session.runtime import TurnNodeEvent, TurnResult, WorkflowSession
from sirenspec.session.snapshots import Snapshot, SnapshotStore
from sirenspec.session.summary import WorkflowSummary, summarise_workflow
from sirenspec.session.widgets import (
    CommandInput,
    CommandPalette,
    FooterHints,
    HintCycler,
    RightPane,
    SplashHeader,
    StatusBar,
    TraceDrawer,
    Transcript,
    WorkflowRail,
)
from sirenspec.yaml.parser import load_workflow

LAUNCH_CSS = """
Screen {
    layers: base overlay;
    background: $background;
    color: $foreground;
}

StatusBar {
    dock: top;
    height: 2;
    padding: 0 1;
    background: $surface;
    border-bottom: solid $border;
}
StatusBar #status-spacer { width: 1fr; }
StatusBar #status-left, StatusBar #status-right { width: auto; height: 1; }

#body { height: 1fr; }

WorkflowRail {
    width: 32;
    padding: 1 1;
    border-right: solid $border;
    background: $surface;
}
WorkflowRail.hidden { display: none; }
WorkflowRail.narrow { width: 20; }

RightPane { width: 1fr; }

SplashHeader {
    height: auto;
    padding: 1 1 0 2;
}
SplashHeader #crest { width: 30; height: 14; }
SplashHeader #splash-meta { width: 1fr; padding: 5 0 0 2; height: auto; }

Transcript {
    height: 1fr;
    padding: 0 2;
    background: $background;
    scrollbar-size-vertical: 1;
}

HintCycler {
    height: 1;
    padding: 0 0 0 1;
}

CommandInput {
    dock: bottom;
    margin: 0 1 0 1;
    height: auto;
    min-height: 3;
    max-height: 8;
    border: round $border;
    background: $background;
}
CommandInput:focus { border: round $primary; }
CommandInput > .text-area--cursor-line { background: $background; }

CommandPalette {
    layer: overlay;
    dock: bottom;
    margin: 0 0 8 4;
    width: 60;
    max-width: 86%;
    height: auto;
    max-height: 10;
    border: round $primary;
    background: $surface;
    display: none;
}
CommandPalette.open { display: block; }

#bottombars {
    dock: bottom;
    height: auto;
}

TraceDrawer {
    height: 2;
    padding: 0 1;
    background: $surface;
    border-top: solid $border;
}
TraceDrawer.hidden { display: none; }

FooterHints {
    height: 2;
    padding: 0 1;
    background: $background;
    border-top: solid $border;
}
"""


class LaunchApp(App):
    """The ``sirenspec launch`` workflow studio Textual application.

    :param session: The live :class:`~sirenspec.session.runtime.WorkflowSession` the studio
        tests against and renders.
    """

    CSS = LAUNCH_CSS
    BINDINGS = [
        Binding("ctrl+c", "handle_ctrl_c", "quit", priority=True, show=False),
        Binding("ctrl+b", "toggle_rail", "rail"),
        Binding("ctrl+t", "toggle_drawer", "trace"),
        Binding("ctrl+r", "rollback", "rollback", show=False),
        Binding("escape", "dismiss_palette", "dismiss", show=False),
    ]

    def __init__(self, session: WorkflowSession) -> None:
        super().__init__()
        self.session = session
        self.color_mode = theme.ColorMode(enabled=True)
        self.summary: WorkflowSummary = summarise_workflow(session.workflow, session.name)
        self.snapshots = SnapshotStore(session.workflow_path)
        self.editor = EditAssistant()
        self.snapshot_label = self.snapshots.latest_label()
        self.stream_buffer: list[str] = []
        self._palette_selecting: bool = False
        self._pending_approval: asyncio.Future[str] | None = None
        self._pre_palette_buffer: str | None = None
        self._drive_task: Task[None] | None = None
        self._node_statuses: dict[str, str] = {}
        self._last_failed_message: str | None = None
        self.registry: CommandRegistry = make_registry(self.build_command_handlers())

    def build_command_handlers(self) -> dict[str, CommandHandler]:
        """Return the mapping of command name to async handler bound to this app.

        Commands whose feature is not part of this build are intentionally omitted so the
        registry wires them to a handler that reports them as unavailable.

        :returns: A dict of command name to async handler.
        """
        return {
            "help": self.command_help,
            "exit": self.command_exit,
            "reload": self.command_reload,
            "run": self.command_run,
            "snapshot": self.command_snapshot,
            "diff": self.command_diff,
            "rollback": self.command_rollback,
            "edit": self.command_edit,
            "retry": self.command_retry,
        }

    def compose(self) -> ComposeResult:
        """Compose the studio layout.

        :returns: The top-level widgets of the application.
        """
        yield StatusBar(id="statusbar")
        with Horizontal(id="body"):
            yield WorkflowRail(id="rail")
            yield RightPane(id="right")
        yield CommandPalette(id="palette")
        with Vertical(id="bottombars"):
            yield TraceDrawer(id="drawer")
            yield FooterHints(id="footer")

    def on_mount(self) -> None:
        """Apply the theme, paint every region, and start the hot-reload watcher."""
        self.register_theme(theme.build_theme(self.color_mode))
        self.theme = "sirenspec"
        self.refresh_chrome()
        self.query_one("#footer", FooterHints).show_hints(self.registry)
        self.query_one("#splash", SplashHeader).show_splash(self.summary, self.color_mode)
        self.query_one(CommandInput).focus()
        self.set_interval(1.0, self.check_reload)
        self.set_interval(5.0, self.advance_hint)

    def refresh_chrome(self, *, live: bool = False) -> None:
        """Repaint the status bar, rail, and trace drawer from current session state.

        :param live: Whether a turn is currently executing (shows ``● live``).
        """
        self.query_one("#statusbar", StatusBar).update_status(
            name=self.session.name,
            provider=self.summary.primary_provider,
            live=live,
            snapshot_label=self.snapshot_label,
        )
        self.query_one("#rail", WorkflowRail).show_rail(
            self.summary,
            snapshot_label=self.snapshot_label,
            turns=self.session.turns,
            cost_usd=self.session.session_cost_usd,
            node_statuses=self._node_statuses if live else {},
        )
        self.query_one("#drawer", TraceDrawer).show_trace(
            node_label="",
            latency_s=None,
            tokens=0,
            cost_usd=None,
            turn=self.session.turns,
            total_turns=self.session.turns,
            session_cost_usd=self.session.session_cost_usd,
        )

    @property
    def transcript(self) -> Transcript:
        """Return the transcript widget.

        :returns: The :class:`~sirenspec.session.widgets.Transcript` instance.
        """
        return self.query_one("#transcript", Transcript)

    # ------------------------------------------------------------------ #
    # Input + palette routing                                            #
    # ------------------------------------------------------------------ #

    async def on_input_submitted(self, event: CommandInput.Submitted) -> None:
        """Handle a submitted input line as either a slash command or a chat turn.

        When a :class:`~sirenspec.core.models.HumanNode` is waiting on an operator response
        (``self._pending_approval`` is set), the next submission resolves that future
        instead of starting a new chat turn — so an approval prompt and its answer use the
        same input box the user is already typing in.

        When the submission came from a palette-opened command (``_pre_palette_buffer`` is
        set), the pre-palette text is restored to the input after the command runs so the
        user never loses their in-progress message.

        :param event: The input submission event.
        """
        raw = event.value
        command_input = self.query_one(CommandInput)
        command_input.reset_prompt()
        pre_palette = self._pre_palette_buffer
        self._pre_palette_buffer = None
        self.close_palette()
        if self._pending_approval is not None and not self._pending_approval.done():
            response = raw.strip()
            self.transcript.add_user(response)
            command_input.remember(raw)
            self._pending_approval.set_result(response)
            return
        line = raw.strip()
        if not line:
            if pre_palette is not None:
                command_input.value = pre_palette
            return
        command_input.remember(line)
        if is_command(line):
            await self.dispatch_command(line)
            if pre_palette is not None:
                command_input.value = pre_palette
        else:
            await self.handle_turn(line)

    async def await_human_input(self, prompt_text: str) -> str:
        """Surface a HumanNode prompt to the user and await their reply in the chat box.

        Installed as the executor's ``human_input_fn`` so that ``type: human`` nodes inside
        the studio never fall back to stdin (which Textual has captured).  The prompt is
        rendered into the transcript and the next :class:`CommandInput` submission resolves
        the awaiting future.

        :param prompt_text: The fully-rendered prompt to show the operator.
        :returns: The operator's typed response (trailing whitespace stripped).
        """
        for line in prompt_text.splitlines() or [""]:
            self.transcript.add_notice(line, style=theme.CREST_LIGHT)
        self.transcript.add_notice("(type your answer below and press enter)", style=theme.TERM_FAINT)
        loop = asyncio.get_running_loop()
        self._pending_approval = loop.create_future()
        self.query_one(CommandInput).focus()
        try:
            return await self._pending_approval
        finally:
            self._pending_approval = None

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        """Open or filter the command palette while the user types a ``/`` command.

        Only the main prompt drives the palette; changes from other text areas (e.g. the
        ``/edit`` editing buffer) are ignored.  When the change came from a palette
        selection (``_palette_selecting`` is set), close the palette rather than re-opening
        it — the selection already filled the input.

        :param event: The text-area change event.
        """
        if event.text_area is not self.query_one(CommandInput):
            return
        if self._palette_selecting:
            self._palette_selecting = False
            self.close_palette()
            return
        value = event.text_area.text
        if is_command(value):
            palette = self.query_one(CommandPalette)
            palette.show_matches(self.registry, value)
            palette.add_class("open")
        else:
            self.close_palette()

    def on_input_open_palette(self, _: CommandInput.OpenPalette) -> None:
        """Handle the palette-open request from the ``/`` key in :class:`CommandInput`.

        Snapshots the current buffer so it can be restored when the palette is dismissed or
        a command is submitted, then sets the input to ``"/"`` to trigger palette filtering.

        :param _: The :class:`~sirenspec.session.widgets.CommandInput.OpenPalette` message.
        """
        command_input = self.query_one(CommandInput)
        self._pre_palette_buffer = command_input.text
        command_input.value = "/"

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        """Fill the input with the chosen palette command and return focus to it.

        Sets ``_palette_selecting`` before updating the input value so that the
        ``TextArea.Changed`` message processed in the next event-loop tick closes the
        palette instead of re-opening it.

        :param event: The palette selection event.
        """
        if event.option.id is None:
            return
        command_input = self.query_one(CommandInput)
        self._palette_selecting = True
        command_input.value = f"/{event.option.id} "
        command_input.focus()

    def close_palette(self) -> None:
        """Hide the command palette overlay."""
        self.query_one(CommandPalette).remove_class("open")

    async def dispatch_command(self, line: str) -> None:
        """Route *line* through the registry, surfacing errors as transcript notices.

        :param line: The raw command line.
        """
        try:
            await self.registry.dispatch(line)
        except SessionError as exc:
            self.transcript.add_notice(str(exc), style=theme.YOU)

    async def handle_turn(self, message: str) -> None:
        """Run a chat turn live against the workflow, streaming attribution and the reply.

        :param message: The user's message text.
        """
        self.query_one(HintCycler).deactivate()
        self.transcript.add_user(message)
        await self.drive(
            self.session.take_turn(
                message,
                token_callback=self.on_token,
                human_input_fn=self.await_human_input,
            ),
            retry_message=message,
        )

    async def drive(self, events: object, *, retry_message: str | None = None) -> None:
        """Consume a turn event stream in a cancellable asyncio task.

        Wrapping the event loop in a task lets Ctrl+C (``action_handle_ctrl_c``) cancel an
        in-flight turn without blocking the Textual message queue.

        :param events: An async generator of :class:`TurnNodeEvent` / :class:`TurnResult`.
        :param retry_message: The user message that will be retried (stored for ``/retry``).
        """
        self.stream_buffer = []
        self._node_statuses = {}
        self.refresh_chrome(live=True)

        async def run_events() -> None:
            async for event in events:  # type: ignore[attr-defined]  — typed as object but is an async generator; checker cannot infer __aiter__
                if isinstance(event, TurnResult):
                    self.present_result(event)
                elif isinstance(event, TurnNodeEvent):
                    self._node_statuses[event.node_id] = event.status
                    self.query_one("#rail", WorkflowRail).show_rail(
                        self.summary,
                        snapshot_label=self.snapshot_label,
                        turns=self.session.turns,
                        cost_usd=self.session.session_cost_usd,
                        node_statuses=self._node_statuses,
                    )
                    self.transcript.add_attribution(
                        event.label, f"{event.tokens} tok", spinning=event.status == "success"
                    )

        loop = asyncio.get_running_loop()
        self._drive_task = loop.create_task(run_events())
        try:
            await self._drive_task
        except asyncio.CancelledError:
            self.transcript.add_notice("cancelled — Ctrl+C again to quit", style=theme.TERM_FAINT)
        except ProviderError as exc:
            self.transcript.add_notice(str(exc), style=theme.YOU)
        except SessionError as exc:
            self.transcript.add_notice(str(exc), style=theme.YOU)
        finally:
            self._drive_task = None
            self.mark_live(False)
        if retry_message is not None:
            self._last_failed_message = retry_message

    def mark_live(self, live: bool) -> None:
        """Update only the status bar's live indicator, leaving the rail and drawer intact.

        :param live: Whether a turn is currently executing.
        """
        self.query_one("#statusbar", StatusBar).update_status(
            name=self.session.name,
            provider=self.summary.primary_provider,
            live=live,
            snapshot_label=self.snapshot_label,
        )

    def on_token(self, chunk: str) -> None:
        """Collect a streamed token chunk for the in-flight turn.

        :param chunk: The text chunk emitted by the provider.
        """
        self.stream_buffer.append(chunk)

    def present_result(self, result: TurnResult) -> None:
        """Render a completed turn's reply and update the rail and trace drawer.

        :param result: The turn's :class:`TurnResult`.
        """
        streamed = "".join(self.stream_buffer)
        reply = result.text or streamed or "(no output)"
        if result.status == "failed":
            self.transcript.add_notice(reply, style=theme.YOU)
            self.transcript.add_notice(
                "  /retry to rerun  ·  /edit to change the workflow  ·  /help for all commands",
                style=theme.TERM_FAINT,
            )
        else:
            self.transcript.add_siren(reply)
        self.query_one("#drawer", TraceDrawer).show_trace(
            node_label=result.last_label,
            latency_s=result.duration_s,
            tokens=result.tokens,
            cost_usd=result.cost_usd,
            turn=self.session.turns,
            total_turns=self.session.turns,
            session_cost_usd=self.session.session_cost_usd,
        )
        self.query_one("#rail", WorkflowRail).show_rail(
            self.summary,
            snapshot_label=self.snapshot_label,
            turns=self.session.turns,
            cost_usd=self.session.session_cost_usd,
            node_statuses=self._node_statuses,
        )

    def on_resize(self, event: Resize) -> None:
        """Adjust the rail layout to fit the terminal width.

        Three breakpoints:
        - < 100 cols: auto-hide the rail (user can reveal with Ctrl+B).
        - 100–140 cols: narrow rail (20 chars wide).
        - > 140 cols: full rail (32 chars wide).

        :param event: The terminal resize event.
        """
        rail = self.query_one("#rail", WorkflowRail)
        cols = event.size.width
        if cols < 100:
            rail.add_class("hidden")
        else:
            rail.remove_class("hidden")
            if cols <= 140:
                rail.add_class("narrow")
            else:
                rail.remove_class("narrow")

    def advance_hint(self) -> None:
        """Advance the onboarding hint cycler by one step."""
        self.query_one(HintCycler).advance()

    def check_reload(self) -> None:
        """Hot-reload the workflow when its file changes on disk (preserving history)."""
        if self.session.file_changed():
            self.perform_reload("workflow.yaml changed on disk — reloaded")

    def perform_reload(self, notice: str) -> None:
        """Reload the workflow from disk, repaint the chrome, and post *notice*.

        :param notice: The transcript message to show after a successful reload.
        """
        try:
            self.session.reload()
        except SessionError as exc:
            self.transcript.add_notice(str(exc), style=theme.YOU)
            return
        self.summary = summarise_workflow(self.session.workflow, self.session.name)
        self.refresh_chrome()
        self.query_one("#splash", SplashHeader).show_splash(self.summary, self.color_mode)
        self.transcript.add_notice(notice, style=theme.LIGHT)

    # ------------------------------------------------------------------ #
    # Key actions                                                        #
    # ------------------------------------------------------------------ #

    def action_handle_ctrl_c(self) -> None:
        """Cancel an in-flight turn on first press; quit the studio on second press.

        If a workflow turn is currently executing (``_drive_task`` is live), the first
        Ctrl+C cancels it and posts a notice.  Any subsequent Ctrl+C (or a press when no
        turn is running) exits the application.
        """
        if self._drive_task is not None and not self._drive_task.done():
            self._drive_task.cancel()
            return
        self.exit()

    def action_toggle_rail(self) -> None:
        """Toggle visibility of the left workflow rail."""
        self.query_one("#rail", WorkflowRail).toggle_class("hidden")

    def action_toggle_drawer(self) -> None:
        """Toggle visibility of the bottom trace drawer."""
        self.query_one("#drawer", TraceDrawer).toggle_class("hidden")

    def action_dismiss_palette(self) -> None:
        """Dismiss the command palette overlay (Esc) and restore the pre-palette buffer."""
        self.close_palette()
        command_input = self.query_one(CommandInput)
        if self._pre_palette_buffer is not None:
            command_input.value = self._pre_palette_buffer
            self._pre_palette_buffer = None
        command_input.focus()

    # ------------------------------------------------------------------ #
    # Standard command handlers                                          #
    # ------------------------------------------------------------------ #

    async def command_help(self, _: str) -> None:
        """List every registered command in the transcript.

        :param _: Unused argument string.
        """
        self.transcript.add_notice("commands", style=theme.CREST_LIGHT)
        for command in self.registry.all():
            self.transcript.add_notice(f"  {command.display:<12} {command.summary}")

    async def command_exit(self, _: str) -> None:
        """Leave the studio.

        :param _: Unused argument string.
        """
        self.exit()

    async def command_reload(self, _: str) -> None:
        """Reload the workflow from disk on demand and repaint the chrome.

        :param _: Unused argument string.
        """
        self.perform_reload("reloaded workflow.yaml")

    async def command_run(self, _: str) -> None:
        """Execute the full workflow graph end-to-end (vs. turn-by-turn chat).

        Auto-snapshots the working file before running only when its content has diverged
        from the latest snapshot, so repeated ``/run`` calls don't spam the version index.

        :param _: Unused argument string.
        """
        latest = self.snapshots.latest()
        if latest is None or self.snapshots.working_content() != self.snapshots.read_content(latest):
            saved = self.auto_snapshot("run", "pre-run")
            if saved is not None:
                self.transcript.add_notice(f"auto-snapshot {saved.ref} saved before run", style=theme.TERM_FAINT)
        self.transcript.add_notice("running the full workflow graph…", style=theme.CREST_LIGHT)
        await self.drive(
            self.session.run_full(
                token_callback=self.on_token,
                human_input_fn=self.await_human_input,
            )
        )

    def auto_snapshot(self, trigger: str, label: str) -> Snapshot | None:
        """Take a best-effort snapshot before a destructive or notable action.

        Snapshot failures are surfaced as a notice but never block the action.

        :param trigger: The snapshot trigger (e.g. ``"run"``, ``"test"``).
        :param label: A human label for the snapshot.
        :returns: The created :class:`~sirenspec.session.snapshots.Snapshot`, or ``None``
            when the snapshot could not be created.
        """
        try:
            snapshot = self.snapshots.create(label=label, trigger=trigger)
        except SessionError as exc:
            self.transcript.add_notice(f"snapshot skipped: {exc}", style=theme.TERM_FAINT)
            return None
        self.update_snapshot_label()
        return snapshot

    def update_snapshot_label(self) -> None:
        """Refresh the active-snapshot label shown in the status bar and rail."""
        self.snapshot_label = self.snapshots.latest_label()
        self.refresh_chrome()

    async def command_snapshot(self, args: str) -> None:
        """Save a labelled snapshot of the working workflow file.

        :param args: An optional label for the snapshot.
        """
        try:
            snapshot = self.snapshots.create(label=args, trigger="manual")
        except SessionError as exc:
            self.transcript.add_notice(str(exc), style=theme.YOU)
            return
        self.update_snapshot_label()
        suffix = f" ({snapshot.label})" if snapshot.label else ""
        self.transcript.add_notice(f"saved snapshot {snapshot.ref}{suffix}", style=theme.LIGHT)

    async def command_diff(self, args: str) -> None:
        """Render a diff: working vs. latest, working vs. a snapshot, or between two.

        Always prints a header line identifying what is being compared so the user knows
        which snapshots are involved — even when there are no differences.

        :param args: Empty (working vs. latest), one ref (working vs. that snapshot), or
            two refs (snapshot vs. snapshot).
        """
        snapshots = self.snapshots.list()
        if not snapshots:
            self.transcript.add_notice("no snapshots yet — use /snapshot first", style=theme.TERM_FAINT)
            return
        tokens = args.split()
        try:
            if not tokens:
                snap = snapshots[-1]
                lines = self.snapshots.diff(snap)
                header = f"diff workflow.yaml (working) ↔ {snap.ref}"
            elif len(tokens) == 1:
                snap = self.snapshots.resolve(tokens[0])
                lines = self.snapshots.diff(snap)
                header = f"diff workflow.yaml (working) ↔ {snap.ref}"
            else:
                a = self.snapshots.resolve(tokens[0])
                b = self.snapshots.resolve(tokens[1])
                lines = self.snapshots.diff(a, b)
                header = f"diff {a.ref} ↔ {b.ref}"
        except SessionError as exc:
            self.transcript.add_notice(str(exc), style=theme.YOU)
            return
        self.transcript.add_notice(header, style=theme.CREST_LIGHT)
        self.render_diff(lines)

    def render_diff(self, lines: list[str]) -> None:
        """Render unified-diff *lines* into the transcript with +/- colouring.

        :param lines: Unified-diff lines from the snapshot store.
        """
        if not lines:
            self.transcript.add_notice("  (no differences)", style=theme.TERM_FAINT)
            return
        for line in lines:
            if line.startswith("+") and not line.startswith("+++"):
                style = theme.LIGHT
            elif line.startswith("-") and not line.startswith("---"):
                style = "#ff5f57"
            elif line.startswith("@@"):
                style = theme.CREST_LIGHT
            else:
                style = theme.TERM_DIM
            self.transcript.add_line(line, style=style)

    async def command_rollback(self, args: str) -> None:
        """Restore a snapshot (defaulting to the *previous* one), reversibly, and reload.

        With no argument the target is the second-to-latest snapshot, so ``/rollback``
        always undoes the last change rather than rolling back to the current state.
        Refuses when fewer than two snapshots exist and no explicit ref was given.

        :param args: An optional snapshot ref/label to restore; defaults to latest minus one.
        """
        snapshots = self.snapshots.list()
        if not snapshots:
            self.transcript.add_notice("no snapshots to roll back to", style=theme.TERM_FAINT)
            return
        try:
            if args.strip():
                target = self.snapshots.resolve(args)
            elif len(snapshots) < 2:
                only = snapshots[-1]
                if self.snapshots.working_content() == self.snapshots.read_content(only):
                    self.transcript.add_notice(
                        "working file already matches the only snapshot — nothing to restore",
                        style=theme.TERM_FAINT,
                    )
                    return
                target = only
            else:
                target = snapshots[-2]
            safety = self.snapshots.rollback(target)
        except SessionError as exc:
            self.transcript.add_notice(str(exc), style=theme.YOU)
            return
        self.perform_reload(f"rolled back to {target.ref} (state saved as {safety.ref})")
        self.update_snapshot_label()

    async def action_rollback(self) -> None:
        """Roll back to the latest snapshot (^R)."""
        await self.command_rollback("")

    async def command_edit(self, _: str) -> None:
        """Open the suggestive ``/edit`` split editor over the studio.

        :param _: Unused argument string.
        """
        self.push_screen(EditScreen(self.editor, self.session, self.snapshots, self.on_edit_applied))

    async def command_retry(self, _: str) -> None:
        """Re-run the last user message (useful after a node failure).

        :param _: Unused argument string.
        """
        if self._last_failed_message is None:
            self.transcript.add_notice("nothing to retry — no previous message", style=theme.TERM_FAINT)
            return
        message = self._last_failed_message
        self.transcript.add_notice(f"retrying: {message}", style=theme.CREST_LIGHT)
        await self.handle_turn(message)

    def on_edit_applied(self) -> None:
        """Repaint the studio after the editor writes an accepted change."""
        self.summary = summarise_workflow(self.session.workflow, self.session.name)
        self.update_snapshot_label()
        self.query_one("#splash", SplashHeader).show_splash(self.summary, self.color_mode)
        self.transcript.add_notice("workflow updated via /edit", style=theme.LIGHT)


def render_plain(workflow: Workflow, workflow_name: str, console: Console) -> None:
    """Render a plain, colourless summary of the studio for non-interactive terminals.

    Used when ``--plain`` / ``NO_COLOR`` is set or stdout is not a TTY.  Prints the Crest
    silhouette, the workflow headline, and the available commands without entering the
    full-screen app.

    :param workflow: The validated workflow.
    :param workflow_name: Human-facing workflow name.
    :param console: The Rich console to print to.
    """
    summary = summarise_workflow(workflow, workflow_name)
    console.print(theme.render_crest(theme.ColorMode(enabled=False)))
    console.print(f"SirenSpec · {summary.name}")
    nodes_word = "node" if summary.node_count == 1 else "nodes"
    agents_word = "agent" if summary.agent_count == 1 else "agents"
    headline_parts = [
        f"{summary.node_count} {nodes_word}",
        f"{summary.agent_count} {agents_word}",
    ]
    if summary.primary_provider:
        headline_parts.append(summary.primary_provider)
    console.print(" · ".join(headline_parts))
    console.print("")
    console.print("This terminal does not support the full studio UI (NO_COLOR / --plain / non-TTY).")
    console.print("Use `sirenspec run` for plain execution, or launch in an interactive terminal.")


def build_console(*, is_tty: bool) -> Console:
    """Create a Rich console appropriate for the current stdout environment.

    :param is_tty: Whether stdout is an interactive terminal.
    :returns: A :class:`~rich.console.Console` with colour disabled when not a TTY.
    """
    if is_tty:
        return Console()
    return Console(force_terminal=False, no_color=True)


def run_launch(workflow_path: Path, *, plain: bool, is_tty: bool, session_id: str | None = None) -> None:
    """Resolve colour mode and either run the studio app or print the plain fallback.

    :param workflow_path: Path to the workflow YAML file.
    :param plain: Whether the ``--plain`` flag was supplied.
    :param is_tty: Whether stdout is an interactive terminal.
    :param session_id: Optional id for a persistent, resumable session via ``--session``.
    :raises SessionError: If the workflow cannot be loaded or the session cannot be created.
    """
    workflow_dir = str(workflow_path.parent.resolve())
    if workflow_dir not in sys.path:
        sys.path.insert(0, workflow_dir)

    try:
        workflow = load_workflow(str(workflow_path))
    except (SirenSpecError, OSError, ValueError) as exc:
        raise SessionError(f"Could not load workflow '{workflow_path}': {exc}") from exc

    workflow_name = workflow_path.stem
    color_mode = theme.resolve_color_mode(plain, is_tty=is_tty)
    if not color_mode.enabled:
        render_plain(workflow, workflow_name, build_console(is_tty=is_tty))
        return

    session = WorkflowSession(workflow, workflow_path, workflow_name, session_id=session_id)
    try:
        LaunchApp(session).run()
    finally:
        session.close()
