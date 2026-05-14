"""SirenSpec CLI entrypoint."""

import typer

app = typer.Typer(name="sirenspec", help="YAML-first agent orchestration SDK.")

from sirenspec.cli.run import run_command  # noqa: E402
from sirenspec.cli.validate import validate_command  # noqa: E402

app.command(name="run")(run_command)
app.command(name="validate")(validate_command)
