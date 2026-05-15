"""``sirenspec render`` command implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from sirenspec.render.mermaid import workflow_to_mermaid
from sirenspec.yaml.parser import load_workflow

_SUPPORTED_TARGETS = ("mermaid",)

_err = Console(stderr=True)


def render_command(
    workflow_file: Annotated[str, typer.Argument(help="Path to the workflow YAML file")],
    target: Annotated[str, typer.Option("--target", help="Render target format (mermaid)")],
    output: Annotated[str | None, typer.Option("--output", "-o", help="Output file path (default: stdout)")] = None,
) -> None:
    """Render a SirenSpec workflow as a diagram."""
    if target not in _SUPPORTED_TARGETS:
        _err.print(f"[red]✗ Unsupported target:[/red] '{target}'. Supported: {', '.join(_SUPPORTED_TARGETS)}")
        raise typer.Exit(1)

    try:
        workflow = load_workflow(workflow_file)
    except FileNotFoundError as exc:
        _err.print(f"[red]✗ File not found:[/red] {exc}")
        raise typer.Exit(1) from exc
    except ValueError as exc:
        _err.print(f"[red]✗ Validation failed:[/red] {exc}")
        raise typer.Exit(1) from exc

    diagram = workflow_to_mermaid(workflow)

    if output:
        Path(output).write_text(diagram + "\n", encoding="utf-8")
    else:
        print(diagram)  # noqa: T201
