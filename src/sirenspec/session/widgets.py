"""Textual widgets composing the ``sirenspec launch`` studio shell.

Each widget owns one region of the layout in ``tui-final-design.html``: the status bar, the
collapsible workflow rail, the Crest splash, the scrolling transcript, the bordered input with
history, the command palette overlay, the trace drawer, and the footer hint row.  All colours
come from :mod:`sirenspec.session.theme`; none are hard-coded here.
"""

from __future__ import annotations

from rich.text import Text
from textual import events
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Input, OptionList, RichLog, Static
from textual.widgets.option_list import Option

from sirenspec.session import theme
from sirenspec.session.commands import CommandRegistry, SlashCommand
from sirenspec.session.summary import WorkflowSummary

SPINNER = "⠙"


class StatusBar(Horizontal):
    """Top status bar: workflow name, live state, active snapshot, and provider."""

    left_text: Text
    right_text: Text

    def compose(self) -> ComposeResult:
        """Yield the left identity cluster, a flexible spacer, and the right status cluster.

        :returns: The child widgets of the status bar.
        """
        yield Static(id="status-left")
        yield Static(id="status-spacer")
        yield Static(id="status-right")

    def update_status(self, *, name: str, provider: str, live: bool, snapshot_label: str) -> None:
        """Repaint the status bar from the current session state.

        :param name: The workflow name shown beside the ``⌁`` glyph.
        :param provider: The active assistant/runtime provider name.
        :param live: ``True`` while a turn is executing (shows ``● live``), else ``◌ idle``.
        :param snapshot_label: The active snapshot label (e.g. ``"v3"``).
        """
        left = Text()
        left.append("⌁ ", style=theme.CREST_LIGHT)
        left.append("SirenSpec", style=f"bold {theme.LIGHT}")
        left.append(" · ", style=theme.TERM_DIM)
        left.append(name, style=f"bold {theme.LIGHT}")
        self.left_text = left
        self.query_one("#status-left", Static).update(left)

        right = Text()
        if live:
            right.append("● ", style=theme.LIGHT)
            right.append("live", style=theme.TERM_DIM)
        else:
            right.append("◌ ", style=theme.TERM_FAINT)
            right.append("idle", style=theme.TERM_DIM)
        right.append("   ")
        right.append("◇ ", style=theme.CREST_LIGHT)
        right.append(f"snapshot {snapshot_label}", style=theme.TERM_DIM)
        right.append("   ")
        right.append(provider or "—", style=theme.TERM_DIM)
        self.right_text = right
        self.query_one("#status-right", Static).update(right)


class CrestBanner(Static):
    """The Crest mascot sprite rendered as half-block ANSI art (or a plain silhouette)."""

    def show_crest(self, color_mode: theme.ColorMode) -> None:
        """Render the Crest into this widget for the given colour mode.

        :param color_mode: The resolved colour mode for the session.
        """
        self.update(theme.render_crest(color_mode))


class SplashHeader(Horizontal):
    """The splash row: Crest banner on the left, workflow meta on the right."""

    meta_text: Text

    def compose(self) -> ComposeResult:
        """Yield the Crest banner and the meta block.

        :returns: The child widgets of the splash header.
        """
        yield CrestBanner(id="crest")
        yield Static(id="splash-meta")

    def show_splash(self, summary: WorkflowSummary, color_mode: theme.ColorMode) -> None:
        """Populate the splash from a workflow summary.

        :param summary: The workflow summary to display.
        :param color_mode: The resolved colour mode for the session.
        """
        self.query_one("#crest", CrestBanner).show_crest(color_mode)
        meta = Text()
        meta.append("SirenSpec\n", style=f"bold {theme.LIGHT}")
        agents_word = "agent" if summary.agent_count == 1 else "agents"
        nodes_word = "node" if summary.node_count == 1 else "nodes"
        meta.append(
            f"{summary.name} · {summary.node_count} {nodes_word} · {summary.agent_count} {agents_word}\n",
            style=theme.TERM_DIM,
        )
        provider = summary.primary_provider or "—"
        meta.append(f"testing in production mode · {provider}", style=theme.TERM_FAINT)
        self.meta_text = meta
        self.query_one("#splash-meta", Static).update(meta)


class WorkflowRail(VerticalScroll):
    """The collapsible left rail: workflow tree, agent roster, and session stats."""

    body_text: Text

    def compose(self) -> ComposeResult:
        """Yield the single Static that holds the rendered rail.

        :returns: The child widgets of the rail.
        """
        yield Static(id="rail-body")

    def show_rail(
        self,
        summary: WorkflowSummary,
        *,
        snapshot_label: str,
        turns: int,
        cost_usd: float | None,
    ) -> None:
        """Repaint the rail from the workflow summary and live session stats.

        :param summary: The workflow summary to display.
        :param snapshot_label: The active snapshot label (e.g. ``"v3"``).
        :param turns: Number of completed turns this session.
        :param cost_usd: Accumulated estimated USD spend, or ``None`` when unavailable.
        """
        body = Text()
        body.append(f"workflow.yaml ◇ {snapshot_label}\n", style=theme.TERM_MUTED)
        for node in summary.nodes:
            body.append("▸ ", style=theme.TERM_FG)
            body.append(node.node_id, style=theme.TERM_FG)
            body.append(f" ({node.kind})\n", style=theme.TERM_DIM)
            for i, child in enumerate(node.children):
                connector = "└" if i == len(node.children) - 1 else "├"
                body.append(f"  {connector} {child.label}", style=theme.TERM_DIM)
                if child.provider:
                    body.append(f"  {child.provider}", style=theme.CREST_LIGHT)
                body.append("\n")

        body.append("\nagents\n", style=theme.TERM_MUTED)
        for agent in summary.agents:
            body.append("● ", style=theme.LIGHT)
            body.append(f"{agent.agent_id} ", style=theme.TERM_DIM)
            body.append("ready\n", style=theme.LIGHT)

        body.append("\nsession\n", style=theme.TERM_MUTED)
        body.append("turns ", style=theme.TERM_DIM)
        body.append(f"{turns}\n", style=theme.CREST_LIGHT)
        body.append("cost ", style=theme.TERM_DIM)
        body.append(f"{format_cost(cost_usd)}\n", style=theme.CREST_LIGHT)
        body.append("\n^B hides this rail", style=theme.TERM_FAINT)

        self.body_text = body
        self.query_one("#rail-body", Static).update(body)


class Transcript(RichLog):
    """The scrolling conversation transcript pane."""

    def add_user(self, message: str) -> None:
        """Append a user turn to the transcript.

        :param message: The user's message text.
        """
        self.write(gutter_line("You", message, theme.YOU))

    def add_siren(self, message: str) -> None:
        """Append an assistant ("Siren") turn to the transcript.

        :param message: The assistant's response text.
        """
        self.write(gutter_line("Siren", message, theme.LIGHT))

    def add_attribution(self, node_label: str, detail: str, *, spinning: bool = True) -> None:
        """Append a per-node attribution line (active node, latency, tokens, cost).

        :param node_label: The active node path, e.g. ``"analyze ▸ risk (anthropic)"``.
        :param detail: The right-hand metrics string, e.g. ``"1.2s · 412 tok · $0.004"``.
        :param spinning: Whether to prefix the spinner glyph.
        """
        line = Text("      ")
        if spinning:
            line.append(f"{SPINNER} ", style=theme.LIGHT)
        line.append(node_label, style=theme.TERM_DIM)
        if detail:
            line.append(f"   {detail}", style=theme.TERM_MUTED)
        self.write(line)

    def add_notice(self, message: str, style: str = theme.TERM_DIM) -> None:
        """Append a dimmed system notice line.

        :param message: The notice text.
        :param style: The Rich style for the notice.
        """
        self.write(Text(f"      {message}", style=style))


def gutter_line(label: str, message: str, label_style: str) -> Text:
    """Build a two-column transcript row: a right-aligned label gutter then the message.

    :param label: The speaker label (e.g. ``"You"`` or ``"Siren"``).
    :param message: The message text (may contain newlines).
    :param label_style: The Rich style applied to the label.
    :returns: A Rich ``Text`` row.
    """
    text = Text()
    text.append(f"{label:>5} ", style=label_style)
    text.append(message, style=theme.TERM_FG)
    return text


def format_cost(cost_usd: float | None) -> str:
    """Format an accumulated USD cost for the rail and trace drawer.

    :param cost_usd: The cost in USD, or ``None`` when pricing is unavailable.
    :returns: A string like ``"$0.004"`` or ``"—"``.
    """
    if cost_usd is None:
        return "—"
    return f"${cost_usd:.3f}"


class CommandInput(Input):
    """The bordered prompt input with up/down command history."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.history: list[str] = []
        self.history_index: int | None = None

    def remember(self, line: str) -> None:
        """Append *line* to the history ring and reset the browse cursor.

        :param line: The submitted line to remember.
        """
        if line and (not self.history or self.history[-1] != line):
            self.history.append(line)
        self.history_index = None

    def history_prev(self) -> None:
        """Move one step backwards through history into the input value."""
        if not self.history:
            return
        if self.history_index is None:
            self.history_index = len(self.history) - 1
        elif self.history_index > 0:
            self.history_index -= 1
        self.value = self.history[self.history_index]
        self.cursor_position = len(self.value)

    def history_next(self) -> None:
        """Move one step forwards through history, clearing past the newest entry."""
        if self.history_index is None:
            return
        if self.history_index < len(self.history) - 1:
            self.history_index += 1
            self.value = self.history[self.history_index]
        else:
            self.history_index = None
            self.value = ""
        self.cursor_position = len(self.value)

    async def _on_key(self, event: events.Key) -> None:
        """Intercept up/down for history before the default cursor handling.

        :param event: The key event.
        """
        if event.key == "up":
            event.prevent_default()
            event.stop()
            self.history_prev()
            return
        if event.key == "down":
            event.prevent_default()
            event.stop()
            self.history_next()
            return
        await super()._on_key(event)


class CommandPalette(OptionList):
    """The floating command palette overlay shown while typing a ``/`` command."""

    def show_matches(self, registry: CommandRegistry, prefix: str) -> None:
        """Repopulate the palette with commands matching *prefix*.

        :param registry: The command registry to query.
        :param prefix: The text typed after ``/`` so far.
        """
        self.clear_options()
        for command in registry.match(prefix):
            self.add_option(Option(palette_row(command), id=command.name))
        if self.option_count:
            self.highlighted = 0

    def selected_command_name(self) -> str | None:
        """Return the id of the currently highlighted option, or ``None``.

        :returns: The highlighted command name, or ``None`` when nothing is highlighted.
        """
        if self.highlighted is None:
            return None
        return self.get_option_at_index(self.highlighted).id


def palette_row(command: SlashCommand) -> Text:
    """Render a single palette row: command name, glyph, and summary.

    :param command: The command to render.
    :returns: A Rich ``Text`` row for the palette.
    """
    row = Text()
    row.append(f"{command.display:<12}", style=f"bold {theme.LIGHT}")
    glyph = command.glyph or " "
    row.append(f"{glyph}  ", style=theme.CREST_LIGHT)
    row.append(command.summary, style=theme.TERM_DIM)
    return row


class TraceDrawer(Static):
    """The bottom trace drawer: active node attribution and running cost segments."""

    trace_text: Text

    def show_trace(
        self,
        *,
        node_label: str,
        latency_s: float | None,
        tokens: int,
        cost_usd: float | None,
        turn: int,
        total_turns: int,
        session_cost_usd: float | None,
    ) -> None:
        """Repaint the trace drawer from the most recent turn's metrics.

        :param node_label: The active node path, e.g. ``"analyze ▸ risk (anthropic)"``.
        :param latency_s: Wall-clock latency of the last turn in seconds, or ``None``.
        :param tokens: Token count for the last turn.
        :param cost_usd: Estimated USD cost of the last turn, or ``None``.
        :param turn: 1-based index of the last completed turn.
        :param total_turns: Total turns this session.
        :param session_cost_usd: Accumulated session USD spend, or ``None``.
        """
        text = Text()
        text.append("▾ trace   ", style=theme.CREST_LIGHT)
        text.append(node_label or "idle", style=theme.TERM_FG)
        text.append("    ")
        latency = f"⠿ {latency_s:.1f}s" if latency_s is not None else "⠿ —"
        text.append(f"{latency}   ", style=theme.TERM_DIM)
        text.append(f"{tokens} tok   ", style=theme.TERM_DIM)
        text.append(f"{format_cost(cost_usd)}   ", style=theme.LIGHT)
        text.append(f"turn {turn}/{total_turns}   ", style=theme.TERM_DIM)
        text.append(f"session {format_cost(session_cost_usd)}", style=theme.TERM_DIM)
        self.trace_text = text
        self.update(text)


class FooterHints(Static):
    """The footer hint row: palette/trace/rail keys plus headline slash commands."""

    def show_hints(self, registry: CommandRegistry) -> None:
        """Repaint the footer from the registry's hint-flagged commands.

        :param registry: The command registry to query for footer hints.
        """
        text = Text()
        for key, label in (("/", "palette"), ("^T", "trace"), ("^B", "rail")):
            text.append(f"{key} ", style=theme.CREST_LIGHT)
            text.append(f"{label}   ", style=theme.TERM_MUTED)
        for command in registry.hints():
            text.append(f"{command.display}  ", style=theme.LIGHT)
        text.append("↑↓ ", style=theme.CREST_LIGHT)
        text.append("history   ", style=theme.TERM_MUTED)
        text.append("^C ", style=theme.CREST_LIGHT)
        text.append("quit", style=theme.TERM_MUTED)
        self.update(text)


class RightPane(Vertical):
    """The right column: splash, transcript, and input."""

    def compose(self) -> ComposeResult:
        """Yield the splash header, transcript, and command input.

        :returns: The child widgets of the right pane.
        """
        yield SplashHeader(id="splash")
        yield Transcript(id="transcript", wrap=True, markup=False)
        yield CommandInput(placeholder="Ask the workflow, or type / for commands", id="input")
