"""The full-screen Textual application for ``sirenspec launch``.

:class:`LaunchApp` composes the studio shell from the widgets in
:mod:`sirenspec.session.widgets`, applies the brand theme from
:mod:`sirenspec.session.theme`, and routes input through the slash-command registry in
:mod:`sirenspec.session.commands`.  Non-interactive terminals fall back to a plain Rich
console via :func:`run_launch`.
"""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import OptionList

from sirenspec.core.models import Workflow
from sirenspec.exceptions import SessionError
from sirenspec.session import theme
from sirenspec.session.commands import CommandRegistry, is_command, make_registry
from sirenspec.session.summary import WorkflowSummary, summarise_workflow
from sirenspec.session.widgets import (
    CommandInput,
    CommandPalette,
    FooterHints,
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

CommandInput {
    dock: bottom;
    margin: 0 1 0 1;
    height: 3;
    border: round $border;
    background: $background;
}
CommandInput:focus { border: round $primary; }

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

    :param workflow: The validated workflow to load into the session.
    :param workflow_path: Filesystem path to the workflow YAML (used by ``/reload``).
    :param workflow_name: Human-facing workflow name (typically the file stem).
    """

    CSS = LAUNCH_CSS
    BINDINGS = [
        Binding("ctrl+c", "quit", "quit", priority=True, show=False),
        Binding("ctrl+b", "toggle_rail", "rail"),
        Binding("ctrl+t", "toggle_drawer", "trace"),
        Binding("escape", "dismiss_palette", "dismiss", show=False),
    ]

    def __init__(self, workflow: Workflow, workflow_path: Path, workflow_name: str) -> None:
        super().__init__()
        self.workflow = workflow
        self.workflow_path = workflow_path
        self.workflow_name = workflow_name
        self.color_mode = theme.ColorMode(enabled=True)
        self.summary: WorkflowSummary = summarise_workflow(workflow, workflow_name)
        self.snapshot_label = "v1"
        self.turns = 0
        self.session_cost_usd: float | None = None
        self.registry: CommandRegistry = make_registry(self.build_command_handlers())

    def build_command_handlers(self) -> dict:
        """Return the mapping of command name to async handler bound to this app.

        Commands whose feature is not part of this shell are intentionally omitted so the
        registry wires them to a handler that reports them as unavailable.

        :returns: A dict of command name to async handler.
        """
        return {
            "help": self.command_help,
            "exit": self.command_exit,
            "reload": self.command_reload,
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
        """Apply the theme and paint every region from the initial workflow state."""
        self.register_theme(theme.build_theme(self.color_mode))
        self.theme = "sirenspec"
        self.refresh_chrome()
        self.query_one("#footer", FooterHints).show_hints(self.registry)
        self.query_one("#splash", SplashHeader).show_splash(self.summary, self.color_mode)
        self.query_one(CommandInput).focus()

    def refresh_chrome(self) -> None:
        """Repaint the status bar, rail, and trace drawer from current session state."""
        provider = self.summary.primary_provider
        self.query_one("#statusbar", StatusBar).update_status(
            name=self.workflow_name,
            provider=provider,
            live=False,
            snapshot_label=self.snapshot_label,
        )
        self.query_one("#rail", WorkflowRail).show_rail(
            self.summary,
            snapshot_label=self.snapshot_label,
            turns=self.turns,
            cost_usd=self.session_cost_usd,
        )
        self.query_one("#drawer", TraceDrawer).show_trace(
            node_label="",
            latency_s=None,
            tokens=0,
            cost_usd=None,
            turn=self.turns,
            total_turns=self.turns,
            session_cost_usd=self.session_cost_usd,
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

        :param event: The input submission event.
        """
        line = event.value.strip()
        command_input = self.query_one(CommandInput)
        command_input.value = ""
        self.close_palette()
        if not line:
            return
        command_input.remember(line)
        if is_command(line):
            await self.dispatch_command(line)
        else:
            await self.handle_turn(line)

    def on_input_changed(self, event: CommandInput.Changed) -> None:
        """Open or filter the command palette while the user types a ``/`` command.

        :param event: The input change event.
        """
        value = event.value
        if is_command(value):
            palette = self.query_one(CommandPalette)
            palette.show_matches(self.registry, value)
            palette.add_class("open")
        else:
            self.close_palette()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        """Fill the input with the chosen palette command and return focus to it.

        :param event: The palette selection event.
        """
        if event.option.id is None:
            return
        command_input = self.query_one(CommandInput)
        command_input.value = f"/{event.option.id} "
        command_input.cursor_position = len(command_input.value)
        self.close_palette()
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
        """Handle a chat turn against the workflow.

        The shell records the user turn and notes that the live testing runtime is not yet
        attached; the runtime sub-feature replaces this with real multi-turn execution.

        :param message: The user's message text.
        """
        self.transcript.add_user(message)
        self.transcript.add_notice("testing runtime not attached in this shell — see /help", style=theme.TERM_FAINT)

    # ------------------------------------------------------------------ #
    # Key actions                                                        #
    # ------------------------------------------------------------------ #

    def action_toggle_rail(self) -> None:
        """Toggle visibility of the left workflow rail."""
        self.query_one("#rail", WorkflowRail).toggle_class("hidden")

    def action_toggle_drawer(self) -> None:
        """Toggle visibility of the bottom trace drawer."""
        self.query_one("#drawer", TraceDrawer).toggle_class("hidden")

    def action_dismiss_palette(self) -> None:
        """Dismiss the command palette overlay (Esc)."""
        self.close_palette()

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
        """Reload the workflow from disk and repaint the chrome.

        :param _: Unused argument string.
        :raises SessionError: If the workflow file can no longer be loaded or validated.
        """
        try:
            self.workflow = load_workflow(str(self.workflow_path))
        except Exception as exc:  # noqa: BLE001 — surface every load failure as a session error
            raise SessionError(f"Reload failed: {exc}") from exc
        self.summary = summarise_workflow(self.workflow, self.workflow_name)
        self.refresh_chrome()
        self.query_one("#splash", SplashHeader).show_splash(self.summary, self.color_mode)
        self.transcript.add_notice("reloaded workflow.yaml", style=theme.LIGHT)


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
    console.print(
        f"{summary.node_count} {nodes_word} · {summary.agent_count} {agents_word} · {summary.primary_provider}"
    )
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


def run_launch(workflow_path: Path, *, plain: bool, is_tty: bool) -> None:
    """Resolve colour mode and either run the studio app or print the plain fallback.

    :param workflow_path: Path to the workflow YAML file.
    :param plain: Whether the ``--plain`` flag was supplied.
    :param is_tty: Whether stdout is an interactive terminal.
    :raises SessionError: If the workflow cannot be loaded.
    """
    try:
        workflow = load_workflow(str(workflow_path))
    except Exception as exc:  # noqa: BLE001 — normalise all load failures into SessionError
        raise SessionError(f"Could not load workflow '{workflow_path}': {exc}") from exc

    workflow_name = workflow_path.stem
    color_mode = theme.resolve_color_mode(plain, is_tty=is_tty)
    if not color_mode.enabled:
        render_plain(workflow, workflow_name, build_console(is_tty=is_tty))
        return

    LaunchApp(workflow=workflow, workflow_path=workflow_path, workflow_name=workflow_name).run()
