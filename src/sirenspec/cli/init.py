"""``sirenspec init`` interactive scaffolding command."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.prompt import Confirm, Prompt

from sirenspec.guardrails.registry import build_guardrails
from sirenspec.templates import TEMPLATES, WorkflowTemplate
from sirenspec.yaml.parser import load_workflow

# Full "provider:model" strings used as defaults internally.
# The CLI displays only the provider name (the part before ":").
PROVIDER_CHOICES: list[str] = [
    "openai:gpt-4o-mini",
    "anthropic:claude-haiku-4-5-20251001",
]

ENV_CONTENT: dict[str, str] = {
    "openai": "OPENAI_API_KEY=sk-...\n",
    "anthropic": "ANTHROPIC_API_KEY=sk-ant-...\n",
}


def split_provider_model(model_string: str) -> tuple[str, str]:
    """Split a combined provider:model string into its two components.

    :param model_string: Combined string such as ``"openai:gpt-4o-mini"``.
    :returns: A ``(provider, model)`` tuple, e.g. ``("openai", "gpt-4o-mini")``.
    """
    provider, _, model = model_string.partition(":")
    return provider, model


def render_template(template: WorkflowTemplate, model: str, guardrails: bool) -> str:
    """Substitute all placeholders in a template and return the final YAML string.

    :param template: The ``WorkflowTemplate`` whose ``content`` contains placeholders.
    :param model: Combined ``"provider:model"`` string, e.g. ``"openai:gpt-4o-mini"``.
    :param guardrails: When ``True``, injects a ``guardrails: [injection, length]`` block.
    :returns: Rendered YAML string ready to write to disk.
    """
    provider, bare_model = split_provider_model(model)
    guardrails_block = "\nguardrails:\n  - injection\n  - length" if guardrails else ""
    result = template.content
    for placeholder, value in [
        ("__MODEL__", model),
        ("__PROVIDER__", provider),
        ("__BARE_MODEL__", bare_model),
        ("__GUARDRAILS__", guardrails_block),
    ]:
        result = result.replace(placeholder, value)
    return result


def prompt_template(console: Console) -> WorkflowTemplate:
    """Display a numbered template picklist and return the chosen ``WorkflowTemplate``.

    :param console: Rich console used for display output.
    :returns: The selected ``WorkflowTemplate``.
    """
    console.print("\nChoose a template:")
    for i, t in enumerate(TEMPLATES, 1):
        console.print(f"  {i}. {t.label:<24} — {t.description}")
    console.print()
    while True:
        raw = Prompt.ask("Template", default="1")
        if raw.isdigit() and 1 <= int(raw) <= len(TEMPLATES):
            return TEMPLATES[int(raw) - 1]
        console.print(f"[red]Please enter a number between 1 and {len(TEMPLATES)}.[/red]")


def prompt_provider(console: Console) -> str:
    """Display a numbered provider picklist (names only) and return the full ``"provider:model"`` string.

    :param console: Rich console used for display output.
    :returns: The selected entry from ``PROVIDER_CHOICES``, e.g. ``"openai:gpt-4o-mini"``.
    """
    console.print("\nChoose a provider:")
    for i, choice in enumerate(PROVIDER_CHOICES, 1):
        provider, _ = split_provider_model(choice)
        console.print(f"  {i}. {provider}")
    console.print()
    while True:
        raw = Prompt.ask("Provider", default="1")
        if raw.isdigit() and 1 <= int(raw) <= len(PROVIDER_CHOICES):
            return PROVIDER_CHOICES[int(raw) - 1]
        console.print(f"[red]Please enter a number between 1 and {len(PROVIDER_CHOICES)}.[/red]")


def resolve_output_path(output_dir: Path, template_key: str) -> Path:
    """Return the path for the new workflow file, prompting for a name on collision.

    :param output_dir: Directory where the file will be written.
    :param template_key: Template key used as the default alternate filename stem.
    :returns: Resolved ``Path`` for the workflow YAML file.
    """
    default = output_dir / "workflow.yaml"
    if not default.exists():
        return default
    alternate = f"{template_key}.yaml"
    chosen = Prompt.ask("workflow.yaml already exists. Filename", default=alternate)
    return output_dir / chosen


def write_env_example(output_dir: Path, provider: str) -> None:
    """Write a provider-appropriate ``.env.example`` file to ``output_dir``.

    :param output_dir: Directory where ``.env.example`` will be written.
    :param provider: Provider name, e.g. ``"openai"`` or ``"anthropic"``.
    """
    (output_dir / ".env.example").write_text(ENV_CONTENT[provider])


def init_command(
    output_dir: Annotated[
        Path,
        typer.Option("--output", "-o", help="Directory for output files (default: current directory)"),
    ] = Path("."),
) -> None:
    """Scaffold a new SirenSpec workflow interactively.

    Generates a ``workflow.yaml`` and ``.env.example`` in the target directory.
    The generated workflow passes ``sirenspec validate`` immediately.

    :param output_dir: Directory where ``workflow.yaml`` and ``.env.example`` are written.
        Defaults to the current working directory.
    """
    console = Console()
    err = Console(stderr=True)

    console.print("\nWelcome to SirenSpec! Let's scaffold a new workflow.")

    template = prompt_template(console)
    model = prompt_provider(console)
    enable_guardrails = Confirm.ask("Enable guardrails? (injection + length)", default=True)

    provider, _ = split_provider_model(model)
    yaml_path = resolve_output_path(output_dir, template.key)
    content = render_template(template, model, enable_guardrails)

    yaml_path.write_text(content)

    try:
        workflow = load_workflow(str(yaml_path))
        build_guardrails(workflow.guardrails)
    except (ValueError, FileNotFoundError) as exc:
        yaml_path.unlink(missing_ok=True)
        err.print(f"[red]✗ Generated workflow failed validation:[/red] {exc}")
        raise typer.Exit(1) from exc

    write_env_example(output_dir, provider)

    console.print(f"\n✓ {yaml_path.name} created")
    console.print("✓ .env.example created")
    console.print("\nNext steps:")
    console.print("  Copy .env.example → .env and add your API key")
    console.print(f"  sirenspec validate {yaml_path.name}")
    console.print(f"  sirenspec run {yaml_path.name}")
