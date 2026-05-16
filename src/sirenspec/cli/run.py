"""``sirenspec run`` command implementation."""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Annotated

import typer
from rich.console import Console

from sirenspec.core.executor import execute
from sirenspec.yaml.parser import load_env_file, load_workflow

_err = Console(stderr=True)


def run_command(
    workflow_file: Annotated[str, typer.Argument(help="Path to the workflow YAML file")],
    input_message: Annotated[str | None, typer.Option("--input", "-i", help="User input message")] = None,
) -> None:
    """Execute a SirenSpec workflow and print the JSON trace to stdout."""
    try:
        workflow = load_workflow(workflow_file)
    except FileNotFoundError as exc:
        _err.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc
    except ValueError as exc:
        _err.print(f"[red]Validation error:[/red] {exc}")
        raise typer.Exit(1) from exc

    if workflow.env_file is not None:
        from pathlib import Path

        env_path = Path(workflow_file).parent / workflow.env_file
        try:
            load_env_file(env_path)
        except FileNotFoundError as exc:
            _err.print(f"[red]Error:[/red] {exc}")
            raise typer.Exit(1) from exc

    # Resolve user input: CLI flag takes precedence over workflow.input.message
    user_input = input_message
    if user_input is None and workflow.input is not None:
        user_input = workflow.input.message
    if not user_input:
        _err.print("[red]Error:[/red] No input provided. Use --input or define input.message in the workflow.")
        raise typer.Exit(1)

    try:
        trace = asyncio.run(execute(workflow, user_input))
    except Exception as exc:
        _err.print(f"[red]Execution error:[/red] {exc}")
        raise typer.Exit(1) from exc

    print(json.dumps(trace, indent=2))  # noqa: T201
    if trace.get("summary", {}).get("status") == "failed":
        sys.exit(1)
