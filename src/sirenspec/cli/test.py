"""``sirenspec test`` command implementation."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from sirenspec.testing.runner import FixtureResult, discover_fixtures, run_fixtures

_console = Console()
_err = Console(stderr=True)

_PASS = "[green]PASS[/green]"
_FAIL = "[red]FAIL[/red]"
_ERROR = "[red]ERROR[/red]"


def render_fixture_result(result: FixtureResult) -> None:
    """Print a per-fixture result block to stdout.

    :param result: The fixture result to render.
    """
    label = result.fixture_path.name
    if result.error:
        _console.print(f"{_ERROR} {label}")
        _console.print(f"  [red]{result.error}[/red]")
        return

    status = _PASS if result.passed else _FAIL
    _console.print(f"{status} {label}")
    for outcome in result.assertion_outcomes:
        icon = "[green]✓[/green]" if outcome.result.passed else "[red]✗[/red]"
        _console.print(f"  {icon} assertion[{outcome.index}]: {outcome.result.message}")


def render_summary(results: list[FixtureResult]) -> None:
    """Print a one-line pass/fail summary to stdout.

    :param results: All fixture results from the run.
    """
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    failed = total - passed
    color = "green" if failed == 0 else "red"
    _console.print(f"\n[{color}]{passed}/{total} fixtures passed[/{color}]", end="")
    if failed:
        _console.print(f"  ({failed} failed)")
    else:
        _console.print()


def determine_exit_code(results: list[FixtureResult]) -> int:
    """Compute the process exit code from the fixture results.

    :param results: All fixture results from the run.
    :returns: ``0`` if all passed, ``1`` if any assertion failed or execution error occurred.
    """
    return 0 if all(r.passed for r in results) else 1


def test_command(
    test_path: Annotated[str, typer.Argument(help="Path to a test fixture file or directory of *.test.yaml files")],
    mock: Annotated[
        bool, typer.Option("--mock", help="Replay LLM responses from the cassette (no live API calls)")
    ] = False,
    record: Annotated[
        bool, typer.Option("--record", help="Run live and record LLM responses into the cassette")
    ] = False,
    cassette: Annotated[
        str | None, typer.Option("--cassette", help="Path to the cassette file for --mock or --record")
    ] = None,
) -> None:
    """Discover and run SirenSpec YAML test fixtures."""
    # --record intercepts live calls and saves them; --mock replays saved responses.
    # Both together would replay stale data into the recording stream, corrupting the cassette.
    if mock and record:
        _err.print("[red]Error:[/red] --mock and --record are mutually exclusive")
        raise typer.Exit(2)

    if (mock or record) and cassette is None:
        _err.print("[red]Error:[/red] --cassette is required when using --mock or --record")
        raise typer.Exit(2)

    mode = "mock" if mock else ("record" if record else "live")
    cassette_path = Path(cassette) if cassette else None

    try:
        fixture_paths = discover_fixtures(Path(test_path))
    except FileNotFoundError as exc:
        _err.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(2) from exc

    if not fixture_paths:
        _err.print(f"[yellow]No *.test.yaml fixtures found in '{test_path}'[/yellow]")
        raise typer.Exit(0)

    results = run_fixtures(fixture_paths, cassette_path, mode)

    for result in results:
        render_fixture_result(result)

    render_summary(results)

    exit_code = determine_exit_code(results)
    if exit_code != 0:
        sys.exit(exit_code)
