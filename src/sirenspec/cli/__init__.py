"""SirenSpec CLI entrypoint."""

import importlib.metadata

import typer

app = typer.Typer(name="sirenspec", help="YAML-first agent orchestration SDK.")

from sirenspec.cli.explain import explain_command  # noqa: E402
from sirenspec.cli.render import render_command  # noqa: E402
from sirenspec.cli.run import run_command  # noqa: E402
from sirenspec.cli.test import test_command  # noqa: E402
from sirenspec.cli.validate import validate_command  # noqa: E402

app.command(name="explain")(explain_command)
app.command(name="render")(render_command)
app.command(name="run")(run_command)
app.command(name="test")(test_command)
app.command(name="validate")(validate_command)


def version_callback(value: bool) -> None:
    if value:
        typer.echo(f"sirenspec {importlib.metadata.version('sirenspec')}")
        raise typer.Exit()


@app.callback()
def callback(
    version: bool = typer.Option(
        None, "--version", callback=version_callback, is_eager=True, help="Show version and exit."
    ),
) -> None:
    pass
