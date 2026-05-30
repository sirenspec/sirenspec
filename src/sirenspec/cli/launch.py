"""``sirenspec launch`` command — open the terminal workflow studio."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from sirenspec.exceptions import SessionError
from sirenspec.session.app import run_launch

_err = Console(stderr=True)


def stdout_is_tty() -> bool:
    """Detect whether stdout is an interactive terminal with colour permitted.

    :returns: ``True`` if stdout is a TTY and ``NO_COLOR`` is not set.
    """
    return sys.stdout.isatty() and "NO_COLOR" not in os.environ


def launch_command(
    workflow_file: Annotated[str, typer.Argument(help="Path to the workflow YAML file")],
    plain: Annotated[
        bool,
        typer.Option(
            "--plain",
            help="Disable colour and the full-screen UI (plain console fallback)",
        ),
    ] = False,
    session: Annotated[
        str | None,
        typer.Option(
            "--session",
            help="Resume a named session; persists conversation via the workflow's memory backend",
        ),
    ] = None,
) -> None:
    """Launch the interactive full-screen workflow studio (TUI).

    Boots the full-screen Textual studio — a scrolling transcript, the collapsible workflow
    rail, the command palette, and the trace drawer — themed with the SirenSpec brand.  The
    workflow runs live against real providers, one turn at a time; ``--session`` rehydrates a
    prior conversation from the workflow's ``memory:`` backend.  Non-interactive terminals and
    ``--plain`` / ``NO_COLOR`` fall back to a plain console.

    \b
    Slash commands (type / to open the palette):
      /run        run the full workflow graph end-to-end
      /edit       open the suggestive workflow editor
      /snapshot   save a named version snapshot of workflow.yaml
      /diff       compare the working file against a snapshot
      /rollback   restore a snapshot (auto-saves a safety copy first)
      /reload     reload the workflow file without restarting
      /help       list all registered commands
      /exit       leave the studio

    \b
    Keyboard shortcuts:
      Ctrl+B      toggle the workflow rail (left sidebar)
      Ctrl+T      toggle the trace drawer (bottom bar)
      Ctrl+R      roll back to the latest snapshot
      Esc         dismiss the command palette
      Ctrl+C      quit

    :param workflow_file: Path to the workflow YAML file to load into the session.
    :param plain: When ``True``, skip the full-screen UI and print a plain console summary.
    :param session: Optional session id to persist and rehydrate conversation state.
    """
    path = Path(workflow_file)
    if not path.exists():
        _err.print(f"[red]Error:[/red] workflow file not found: {workflow_file}")
        raise typer.Exit(1)
    try:
        run_launch(path, plain=plain, is_tty=stdout_is_tty(), session_id=session)
    except SessionError as exc:
        _err.print(f"[red]Launch error:[/red] {exc}")
        raise typer.Exit(1) from exc
