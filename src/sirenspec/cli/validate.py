"""``sirenspec validate`` command implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from sirenspec.exceptions import WorkflowLintError
from sirenspec.guardrails.registry import build_guardrails
from sirenspec.yaml.parser import load_workflow

_err = Console(stderr=True)


def validate_command(
    workflow_file: Annotated[str, typer.Argument(help="Path to the workflow YAML file")],
) -> None:
    """Validate a SirenSpec workflow YAML file."""
    try:
        workflow = load_workflow(workflow_file)
    except FileNotFoundError as exc:
        _err.print(f"[red]✗ File not found:[/red] {exc}")
        raise typer.Exit(1) from exc
    except WorkflowLintError as exc:
        for issue in exc.issues:
            prefix = "✗ Lint error" if issue.level == "error" else "⚠ Lint warning"
            location = f" ({issue.location})" if issue.location else ""
            _err.print(f"[red]{prefix}[/red]{location}: {issue.message}")
        raise typer.Exit(1) from exc
    except ValueError as exc:
        _err.print(f"[red]✗ Validation failed:[/red] {exc}")
        raise typer.Exit(1) from exc

    try:
        build_guardrails(workflow.guardrails)
    except ValueError as exc:
        _err.print(f"[red]✗ Guardrail config error:[/red] {exc}")
        raise typer.Exit(1) from exc

    n_agents = len(workflow.agents)
    n_nodes = len(workflow.nodes)
    filename = Path(workflow_file).name
    print(
        f"✓ {filename} is valid ({n_agents} agent{'s' if n_agents != 1 else ''}, {n_nodes} node{'s' if n_nodes != 1 else ''})"
    )  # noqa: T201
