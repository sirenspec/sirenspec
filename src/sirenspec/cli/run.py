"""``sirenspec run`` command implementation."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.box import ASCII, ROUNDED, Box
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from sirenspec.core.events import NodeCompleteEvent, SummaryEvent
from sirenspec.core.executor import execute, execute_streaming
from sirenspec.core.models import Workflow
from sirenspec.yaml.parser import load_env_file, load_workflow

_err = Console(stderr=True)


def is_tty() -> bool:
    """Detect whether stdout is an interactive terminal.

    :returns: ``True`` if stdout is a TTY and the ``NO_COLOR`` env var is not set.
    """
    return sys.stdout.isatty() and "NO_COLOR" not in os.environ


def build_console() -> Console:
    """Create a Rich console appropriate for the current stdout environment.

    :returns: A :class:`~rich.console.Console` with colour disabled when not a TTY.
    """
    if is_tty():
        return Console()
    return Console(force_terminal=False, no_color=True)


def format_agent_duration(duration_ms: float) -> str:
    """Format a per-agent duration as a compact seconds string.

    :param duration_ms: Duration in milliseconds.
    :returns: A string like ``"0.8s"`` or ``"12.3s"``.
    """
    return f"{duration_ms / 1000:.1f}s"


def render_swarm_agent_panel(agent: dict[str, Any], console: Console, box_style: Box) -> None:
    """Render a single swarm agent's output as a Rich panel.

    Failed agents use a red border with the error message as content.
    Successful agents display the response text with the default border.

    :param agent: Agent trace dict with keys ``id``, ``response_received``, ``duration_ms``, ``error``.
    :param console: The Rich console to print to.
    :param box_style: Rich box style (``ROUNDED`` for TTY, ``ASCII`` for pipes/CI).
    """
    if agent.get("error"):
        content = agent["error"]
        border_style = "red"
    else:
        content = format_output_content(agent.get("response_received") or "")
        border_style = "default"

    terminal_width = shutil.get_terminal_size((80, 24)).columns
    panel = Panel(
        content,
        title=Text(agent["id"], style="bold"),
        border_style=border_style,
        box=box_style,
        width=min(terminal_width, 100),
    )
    console.print(panel)


def format_output_content(output: Any) -> str:
    """Format a node output value as a printable string.

    If *output* is a string that parses as JSON, it is re-formatted with
    indentation.  Dicts and lists are serialised with indentation.  Everything
    else is converted via ``str()``.

    :param output: The raw node output.
    :returns: A human-readable string representation of *output*.
    """
    if isinstance(output, (dict, list)):
        return json.dumps(output, indent=2)
    if isinstance(output, str):
        try:
            parsed = json.loads(output)
            return json.dumps(parsed, indent=2)
        except (json.JSONDecodeError, ValueError):
            return output
    return str(output) if output is not None else ""


def render_swarm_panel(event: NodeCompleteEvent, console: Console, box_style: Box) -> None:
    """Render a swarm node as a parallel execution block.

    Prints an opening rule, a Rich panel per agent, a closing rule with success counts
    and timing, and (when synthesis produced a single string output) a panel for the
    synthesised result followed by the writes arrow.

    :param event: A swarm ``NodeCompleteEvent`` with ``event.agents`` populated.
    :param console: The Rich console to print to.
    :param box_style: Rich box style — controls whether the rule is styled or plain.
    """
    agents: list[dict[str, Any]] = event.agents  # type: ignore[assignment]  # always set when node_type="swrm"
    total = len(agents)
    succeeded = sum(1 for a in agents if not a.get("error"))
    failed = total - succeeded

    rule_style = "bold blue" if box_style is not ASCII else "default"
    header = f"Swarm: {event.node_id} [{total} agent{'s' if total != 1 else ''}]"
    console.rule(header, style=rule_style)

    for agent in agents:
        render_swarm_agent_panel(agent, console, box_style)

    footer_parts = [f"{succeeded}/{total} succeeded"]
    if failed:
        footer_parts.append(f"{failed} failed")
    footer_parts.append(f"({format_agent_duration(event.duration_ms or 0.0)} total)")
    console.rule(f"Swarm complete: {', '.join(footer_parts)}", style=rule_style)

    if event.status == "success" and not isinstance(event.output, list) and event.output is not None:
        terminal_width = shutil.get_terminal_size((80, 24)).columns
        console.print(
            Panel(
                format_output_content(event.output),
                title=Text(event.node_id, style="bold"),
                border_style="default",
                box=box_style,
                width=min(terminal_width, 100),
            )
        )

    if event.status == "success" and event.writes:
        console.print(f"  [dim]↓ {event.writes}[/dim]")


def render_swrm_factory_instance(instance: dict[str, Any], total: int, console: Console, box_style: Box) -> None:
    """Render a swrm factory instance as an expanded per-agent block.

    Prints a labelled divider rule, a Rich panel per agent, and (when synthesis
    is present) a synthesis panel.

    :param instance: Instance trace dict with a populated ``swrm`` key.
    :param total: Total number of factory instances.
    :param console: The Rich console to print to.
    :param box_style: Rich box style (``ROUNDED`` for TTY, ``ASCII`` for pipes/CI).
    """
    idx = instance["index"]
    swrm = instance["swrm"]
    rule_style = "bold blue" if box_style is not ASCII else "default"
    console.rule(f"[{idx + 1}/{total}]", style=rule_style)

    for agent in swrm.get("agents", []):
        render_swarm_agent_panel(agent, console, box_style)

    synthesis_output = swrm.get("output")
    if synthesis_output is not None and not isinstance(synthesis_output, list):
        terminal_width = shutil.get_terminal_size((80, 24)).columns
        console.print(
            Panel(
                format_output_content(synthesis_output),
                title=Text(f"[{idx + 1}/{total}] synthesis", style="bold"),
                border_style="default",
                box=box_style,
                width=min(terminal_width, 100),
            )
        )


def render_factory_instance_panel(instance: dict[str, Any], total: int, console: Console, box_style: Box) -> None:
    """Render a single factory instance's output.

    Swrm factory instances are expanded into per-agent panels via
    :func:`render_swrm_factory_instance`.  Regular agent instances and failed
    instances render as a single Rich panel.

    :param instance: Instance trace dict with keys ``index``, ``response_received``, ``swrm``,
        ``duration_ms``, and ``error``.
    :param total: Total number of instances in the factory run.
    :param console: The Rich console to print to.
    :param box_style: Rich box style (``ROUNDED`` for TTY, ``ASCII`` for pipes/CI).
    """
    if instance.get("swrm") is not None:
        render_swrm_factory_instance(instance, total, console, box_style)
        return

    idx = instance["index"]
    if instance.get("error"):
        content = instance["error"]
        border_style = "red"
    else:
        content = format_output_content(instance.get("response_received") or "")
        border_style = "default"

    terminal_width = shutil.get_terminal_size((80, 24)).columns
    console.print(
        Panel(
            content,
            title=Text(f"[{idx + 1}/{total}]", style="bold"),
            border_style=border_style,
            box=box_style,
            width=min(terminal_width, 100),
        )
    )


def render_factory_panel(event: NodeCompleteEvent, console: Console, box_style: Box) -> None:
    """Render a factory node as a parallel execution block.

    Prints an opening rule, a Rich panel per instance, and a closing rule with
    success counts and timing, followed by the writes arrow.

    :param event: A factory ``NodeCompleteEvent`` with ``event.instances`` populated.
    :param console: The Rich console to print to.
    :param box_style: Rich box style — controls whether the rule is styled or plain.
    """
    instances: list[dict[str, Any]] = event.instances  # type: ignore[assignment]  # always set when node_type="factory"
    total = len(instances)
    succeeded = sum(1 for inst in instances if not inst.get("error"))
    failed = total - succeeded

    rule_style = "bold blue" if box_style is not ASCII else "default"
    header = f"Factory: {event.node_id} [{total} instance{'s' if total != 1 else ''}]"
    console.rule(header, style=rule_style)

    for instance in instances:
        render_factory_instance_panel(instance, total, console, box_style)

    footer_parts = [f"{succeeded}/{total} succeeded"]
    if failed:
        footer_parts.append(f"{failed} failed")
    footer_parts.append(f"({format_agent_duration(event.duration_ms or 0.0)} total)")
    console.rule(f"Factory complete: {', '.join(footer_parts)}", style=rule_style)

    if event.status == "success" and event.writes:
        console.print(f"  [dim]↓ {event.writes}[/dim]")


def render_node_panel(event: NodeCompleteEvent, console: Console, box_style: Box) -> None:
    """Render a completed node as a Rich panel followed by its writes arrow.

    Skipped nodes are printed as a single dimmed line.  Failed nodes use a
    red border.  Successful nodes use the default rounded border.  Swarm and
    factory nodes with per-instance trace data are delegated to their dedicated
    renderers.

    :param event: The node completion event to render.
    :param console: The Rich console to print to.
    :param box_style: The Rich box style to use (``ROUNDED`` for TTY, ``ASCII`` for pipes/CI).
    """
    if event.node_type == "swrm" and event.agents is not None:
        render_swarm_panel(event, console, box_style)
        return

    if event.node_type == "factory" and event.instances is not None:
        render_factory_panel(event, console, box_style)
        return

    if event.status == "skipped":
        console.print(f"  [dim](skipped) {event.node_id}[/dim]")
        return

    content_text = ""
    if event.status == "failed" and event.error:
        content_text = event.error
    elif event.output is not None:
        content_text = format_output_content(event.output)

    title = Text(event.node_id, style="bold")
    border_style = "red" if event.status == "failed" else "default"

    terminal_width = shutil.get_terminal_size((80, 24)).columns
    panel = Panel(
        content_text,
        title=title,
        border_style=border_style,
        box=box_style,
        width=min(terminal_width, 100),
    )
    console.print(panel)

    if event.status == "success" and event.writes:
        arrow_label = event.writes
        console.print(f"  [dim]↓ {arrow_label}[/dim]")


def format_summary_line(event: SummaryEvent) -> str:
    """Produce the one-line run summary string from a :class:`SummaryEvent`.

    :param event: The final summary event from the streaming executor.
    :returns: A formatted summary string including cost if pricing is available.
    """
    nodes_label = f"{event.total_nodes} node{'s' if event.total_nodes != 1 else ''}"
    tokens_label = f"{event.total_tokens:,} token{'s' if event.total_tokens != 1 else ''}"
    cost_label = (
        f"~${event.estimated_usd:.6f}"
        if event.estimated_usd is not None
        else "cost unavailable (pricing not configured)"
    )
    return f"Run complete  │  {nodes_label}  │  {tokens_label}  │  {cost_label}"


async def run_streaming(
    workflow: Workflow,
    user_input: str,
    quiet: bool,
    trace_file: str | None,
) -> int:
    """Stream workflow execution, printing node panels to stdout.

    When *trace_file* is given, the complete trace dict is also written to that
    path as JSON after execution finishes.

    :param workflow: The validated workflow to execute.
    :param user_input: The user input message to pass to the workflow.
    :param quiet: When ``True``, suppress node panels; only print the summary.
    :param trace_file: Optional file path for writing the full JSON trace.
    :returns: Exit code — 0 for success, 1 for failure.
    """
    console = build_console()
    box_style = ROUNDED if is_tty() else ASCII
    trace_nodes: list[dict[str, Any]] = []
    summary_event: SummaryEvent | None = None

    try:
        async for event in execute_streaming(workflow, user_input):
            if isinstance(event, NodeCompleteEvent):
                if not quiet:
                    render_node_panel(event, console, box_style)
                if trace_file is not None:
                    trace_nodes.append(
                        {
                            "id": event.node_id,
                            "type": event.node_type,
                            "output": event.output,
                            "writes": event.writes,
                            "status": event.status,
                            "error": event.error,
                            "tokens": event.tokens,
                        }
                    )
            elif isinstance(event, SummaryEvent):
                summary_event = event

    except Exception as exc:
        _err.print(f"[red]Execution error:[/red] {exc}")
        return 1

    if summary_event is not None:
        console.print(format_summary_line(summary_event))

        if trace_file is not None:
            trace_dict: dict[str, Any] = {
                "workflow": {"version": workflow.version},
                "input": {"message": user_input},
                "nodes": trace_nodes,
                "summary": {
                    "total_nodes": summary_event.total_nodes,
                    "total_tokens": summary_event.total_tokens,
                    "status": summary_event.status,
                    "total_duration_ms": summary_event.duration_ms,
                },
            }
            Path(trace_file).write_text(json.dumps(trace_dict, indent=2), encoding="utf-8")

        return 0 if summary_event.status == "success" else 1

    return 0


def load_workflow_with_env(workflow_file: str) -> Workflow:
    """Load and validate a workflow file, then load its env file if configured.

    :param workflow_file: Path to the workflow YAML file.
    :returns: The validated :class:`~sirenspec.core.models.Workflow` instance.
    :raises typer.Exit: With code 1 on any load or validation error.
    """
    try:
        workflow = load_workflow(workflow_file)
    except FileNotFoundError as exc:
        _err.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc
    except ValueError as exc:
        _err.print(f"[red]Validation error:[/red] {exc}")
        raise typer.Exit(1) from exc

    if workflow.env_file is not None:
        env_path = Path(workflow_file).parent / workflow.env_file
        try:
            load_env_file(env_path)
        except FileNotFoundError as exc:
            _err.print(f"[red]Error:[/red] {exc}")
            raise typer.Exit(1) from exc

    return workflow


def resolve_user_input(workflow: Workflow, input_message: str | None) -> str:
    """Resolve the user input from the CLI flag or workflow default.

    :param workflow: The loaded workflow, which may have a default input message.
    :param input_message: The value of the ``--input`` CLI flag, or ``None``.
    :returns: The resolved user input string.
    :raises typer.Exit: With code 1 when no input can be resolved.
    """
    user_input = input_message
    if user_input is None and workflow.input is not None:
        user_input = workflow.input.message
    if not user_input:
        _err.print("[red]Error:[/red] No input provided. Use --input or define input.message in the workflow.")
        raise typer.Exit(1)
    return user_input


def run_command(
    workflow_file: Annotated[str, typer.Argument(help="Path to the workflow YAML file")],
    input_message: Annotated[str | None, typer.Option("--input", "-i", help="User input message")] = None,
    trace: Annotated[bool, typer.Option("--trace", help="Print full JSON trace to stdout")] = False,
    trace_file: Annotated[
        str | None, typer.Option("--trace-file", help="Write full JSON trace to this file path")
    ] = None,
    quiet: Annotated[bool, typer.Option("--quiet", help="Suppress node panels; print only the summary")] = False,
    output_format: Annotated[
        str | None, typer.Option("--output", help="Output format. Use 'json' for raw JSON trace.")
    ] = None,
) -> None:
    """Execute a SirenSpec workflow with a streaming per-node view."""
    workflow = load_workflow_with_env(workflow_file)
    user_input = resolve_user_input(workflow, input_message)

    if trace or output_format == "json":
        try:
            wf_trace = asyncio.run(execute(workflow, user_input))
        except Exception as exc:
            _err.print(f"[red]Execution error:[/red] {exc}")
            raise typer.Exit(1) from exc

        print(json.dumps(wf_trace, indent=2))  # noqa: T201

        if wf_trace.get("summary", {}).get("status") == "failed":
            sys.exit(1)
        return

    exit_code = asyncio.run(run_streaming(workflow, user_input, quiet, trace_file))
    if exit_code != 0:
        sys.exit(exit_code)
