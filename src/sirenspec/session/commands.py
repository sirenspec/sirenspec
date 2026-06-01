"""Slash-command registry and router for the ``sirenspec launch`` studio.

The shell owns a single :class:`CommandRegistry`.  Sub-features (the testing runtime,
``/edit``, snapshots) register their commands into it at startup, so the palette overlay,
footer hints, and dispatcher all stay in sync from one source of truth.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from sirenspec.exceptions import SessionError

# A command handler receives the (possibly empty) argument string typed after the command
# name and performs the action asynchronously.  Handlers return any value the caller wants
# to inspect; the dispatcher itself ignores the return value.
CommandHandler = Callable[[str], Awaitable[Any]]


@dataclass(frozen=True)
class SlashCommand:
    """A single registered slash command.

    :param name: The command keyword without its leading slash (e.g. ``"snapshot"``).
    :param summary: One-line description shown in the palette and ``/help``.
    :param handler: Async callable invoked with the argument string when the command runs.
    :param glyph: Optional single-character glyph shown beside the command in the palette.
    :param hint: Whether the command appears in the compact footer hint row.
    """

    name: str
    summary: str
    handler: CommandHandler
    glyph: str = ""
    hint: bool = False

    @property
    def display(self) -> str:
        """Return the slash-prefixed command name for display.

        :returns: The command name prefixed with ``/`` (e.g. ``"/snapshot"``).
        """
        return f"/{self.name}"


def parse_command_line(line: str) -> tuple[str, str]:
    """Split a raw input line into a command name and its argument string.

    A leading slash is optional and stripped.  The name is the first whitespace-delimited
    token, lower-cased; everything after the first run of whitespace is the argument
    string (leading/trailing whitespace trimmed).

    :param line: The raw text typed by the user (e.g. ``"/snapshot before refactor"``).
    :returns: A ``(name, args)`` tuple, e.g. ``("snapshot", "before refactor")``.
    """
    stripped = line.strip()
    if stripped.startswith("/"):
        stripped = stripped[1:]
    if not stripped:
        return "", ""
    parts = stripped.split(None, 1)
    name = parts[0].lower()
    args = parts[1].strip() if len(parts) > 1 else ""
    return name, args


def is_command(line: str) -> bool:
    """Return whether *line* should be routed as a slash command rather than a chat turn.

    :param line: The raw text typed by the user.
    :returns: ``True`` if the trimmed line begins with ``/``.
    """
    return line.strip().startswith("/")


class CommandRegistry:
    """Registry and dispatcher for the studio's slash commands.

    Commands are registered once at startup and looked up by name.  The registry also
    powers the command palette (prefix matching) and the footer hint row.
    """

    def __init__(self) -> None:
        self._commands: dict[str, SlashCommand] = {}

    def register(
        self,
        name: str,
        summary: str,
        handler: CommandHandler,
        glyph: str = "",
        hint: bool = False,
    ) -> SlashCommand:
        """Register a new slash command.

        :param name: Command keyword without its leading slash.
        :param summary: One-line description for the palette and ``/help``.
        :param handler: Async callable invoked with the argument string.
        :param glyph: Optional single-character glyph for the palette.
        :param hint: Whether the command appears in the footer hint row.
        :raises SessionError: If a command with the same name is already registered.
        :returns: The created :class:`SlashCommand`.
        """
        key = name.lower()
        if key in self._commands:
            raise SessionError(f"Slash command '/{key}' is already registered.")
        command = SlashCommand(name=key, summary=summary, handler=handler, glyph=glyph, hint=hint)
        self._commands[key] = command
        return command

    def get(self, name: str) -> SlashCommand | None:
        """Return the command registered under *name*, or ``None`` if absent.

        :param name: Command keyword (with or without a leading slash).
        :returns: The matching :class:`SlashCommand`, or ``None``.
        """
        return self._commands.get(name.lstrip("/").lower())

    def all(self) -> list[SlashCommand]:
        """Return every registered command in registration order.

        :returns: A list of all :class:`SlashCommand` instances.
        """
        return list(self._commands.values())

    def hints(self) -> list[SlashCommand]:
        """Return the commands flagged for the compact footer hint row.

        :returns: A list of footer-hint :class:`SlashCommand` instances.
        """
        return [c for c in self._commands.values() if c.hint]

    def match(self, prefix: str) -> list[SlashCommand]:
        """Return commands whose name starts with *prefix* (for palette filtering).

        :param prefix: The text typed after ``/`` so far (case-insensitive); empty
            matches all commands.
        :returns: Matching commands in registration order.
        """
        needle = prefix.lstrip("/").lower()
        if not needle:
            return self.all()
        return [c for c in self._commands.values() if c.name.startswith(needle)]

    async def dispatch(self, line: str) -> SlashCommand:
        """Parse *line*, invoke the matching command's handler, and return the command.

        :param line: The raw command line typed by the user.
        :raises SessionError: If the line names no command or an unknown command.
        :returns: The :class:`SlashCommand` that was dispatched.
        """
        name, args = parse_command_line(line)
        if not name:
            raise SessionError("No command was entered.")
        command = self._commands.get(name)
        if command is None:
            raise SessionError(f"Unknown command '/{name}'. Type /help to list commands.")
        await command.handler(args)
        return command


def make_registry(handlers: dict[str, CommandHandler] | None = None) -> CommandRegistry:
    """Build a registry pre-populated with the studio's standard command set.

    Each standard command is wired to a handler from *handlers* when one is supplied for
    that name; commands without a provided handler fall back to :func:`unbound_handler`,
    which raises a clear :class:`SessionError` so a half-wired shell fails loudly rather
    than silently no-op'ing.

    :param handlers: Optional mapping of command name to async handler.
    :returns: A populated :class:`CommandRegistry`.
    """
    handlers = handlers or {}
    registry = CommandRegistry()
    for name, summary, glyph, hint in STANDARD_COMMANDS:
        registry.register(
            name=name,
            summary=summary,
            handler=handlers.get(name, make_unbound_handler(name)),
            glyph=glyph,
            hint=hint,
        )
    return registry


def make_unbound_handler(name: str) -> CommandHandler:
    """Return a handler that raises because command *name* has no implementation wired.

    :param name: The command name that is missing a handler.
    :returns: An async handler that always raises :class:`SessionError`.
    """

    async def handler(_: str) -> None:
        raise SessionError(f"Command '/{name}' is not available in this session.")

    return handler


# Standard command set: (name, summary, glyph, footer-hint).  Handlers are injected by the
# app via :func:`make_registry` so this table stays declarative and feature-agnostic.
STANDARD_COMMANDS: tuple[tuple[str, str, str, bool], ...] = (
    ("edit", "open the suggestive editor", "✎", True),
    ("run", "run the full workflow graph end-to-end", "▸", True),
    ("snapshot", "save a version snapshot of workflow.yaml", "◇", True),
    ("rollback", "restore a snapshot (auto-saves a safety copy first)", "↺", False),
    ("diff", "compare the working tree against a snapshot", "⇄", False),
    ("retry", "re-run the last message (useful after a node failure)", "↩", False),
    ("reload", "rebuild the session from the workflow file", "↻", False),
    ("help", "list the available commands", "?", False),
    ("exit", "leave the studio", "⏏", False),
)
