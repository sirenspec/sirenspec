"""SirenSpec CLI entrypoint."""

import typer

app = typer.Typer(name="sirenspec", help="YAML-first agent orchestration SDK.")

from sirenspec.cli.explain import explain_command  # noqa: E402
from sirenspec.cli.render import render_command  # noqa: E402
from sirenspec.cli.run import run_command  # noqa: E402
from sirenspec.cli.validate import validate_command  # noqa: E402

app.command(name="explain")(explain_command)
app.command(name="render")(render_command)
app.command(name="run")(run_command)
app.command(name="validate")(validate_command)
