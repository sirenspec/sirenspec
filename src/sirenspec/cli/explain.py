"""``sirenspec explain`` command — prints a human-readable execution plan without LLM calls."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console

from sirenspec.core.executor import topological_sort
from sirenspec.core.models import (
    AgentNode,
    AnyNode,
    Edge,
    FactoryNode,
    HumanNode,
    SwrmNode,
    ToolNode,
    Workflow,
    WorkflowNode,
)
from sirenspec.yaml.parser import load_workflow

_err = Console(stderr=True)


# ---------------------------------------------------------------------------
# Guardrail label helpers
# ---------------------------------------------------------------------------


def guardrail_labels(entries: list[str] | None) -> list[str]:
    """Convert a list of guardrail entries to display labels.

    :param entries: The raw guardrail list from the model, or None.
    :returns: A list of label strings (empty list when *entries* is None).
    """
    if not entries:
        return []
    return list(entries)


# ---------------------------------------------------------------------------
# Node type info helpers
# ---------------------------------------------------------------------------


def node_type_info(node: AnyNode, workflow: Workflow) -> str:
    """Return a compact type description string for a node.

    :param node: The typed node object.
    :param workflow: The containing workflow (used to look up agent definitions).
    :returns: A string such as ``"agent=router (openai:gpt-4o-mini)"`` or ``"tool=http"``.
    """
    if isinstance(node, AgentNode):
        agent_def = workflow.agents.get(node.agent)
        model_uri = agent_def.model if agent_def else "unknown"
        return f"agent={node.agent} ({model_uri})"
    if isinstance(node, ToolNode):
        return f"tool={node.tool}"
    if isinstance(node, SwrmNode):
        return "swrm"
    if isinstance(node, FactoryNode):
        agent_def = workflow.agents.get(node.agent)
        model_uri = agent_def.model if agent_def else "unknown"
        return f"factory agent={node.agent} ({model_uri})"
    if isinstance(node, HumanNode):
        timeout_label = f", timeout={node.timeout}s" if node.timeout is not None else ""
        return f"human ({node.on_timeout}{timeout_label})"
    if isinstance(node, WorkflowNode):
        return f"workflow ref={node.ref}"
    return "unknown"


def node_writes_path(node: AnyNode) -> str | None:
    """Return the output path that a node writes to, or None when not applicable.

    :param node: The typed node object.
    :returns: A dot-notation path string or None.
    """
    if isinstance(node, AgentNode):
        return node.writes
    if isinstance(node, FactoryNode):
        return node.writes
    if isinstance(node, HumanNode):
        return node.writes
    if isinstance(node, WorkflowNode):
        return node.writes
    if isinstance(node, ToolNode):
        return f"working.<node_id>.{node.output_key}"
    return None


def node_guardrails(node: AnyNode, workflow: Workflow) -> list[str]:
    """Return the effective guardrail labels for a node.

    For AgentNodes, uses agent-level guardrails when set, otherwise workflow-level.
    ToolNodes, SwrmNodes, and FactoryNodes use workflow-level guardrails.

    :param node: The typed node object.
    :param workflow: The containing workflow.
    :returns: A list of guardrail label strings.
    """
    if isinstance(node, AgentNode):
        agent_def = workflow.agents.get(node.agent)
        if agent_def and agent_def.guardrails is not None:
            return guardrail_labels(agent_def.guardrails)
    return guardrail_labels(workflow.guardrails)


# ---------------------------------------------------------------------------
# Graph analysis helpers
# ---------------------------------------------------------------------------


def collect_errors(workflow: Workflow) -> list[str]:
    """Collect structural errors in the workflow graph.

    Checks for edge references to undeclared nodes. This mirrors ``sirenspec validate``
    intentionally — explain is self-contained.

    :param workflow: The workflow to inspect.
    :returns: A list of human-readable error strings.
    """
    node_ids = set(workflow.nodes.keys())
    errors: list[str] = []
    for edge in workflow.edges:
        if edge.from_node not in node_ids:
            errors.append(f'Edge references unknown node "{edge.from_node}"')
        if edge.to_node not in node_ids:
            errors.append(f'Edge references unknown node "{edge.to_node}"')
    return errors


def collect_warnings(workflow: Workflow, execution_order: list[str]) -> list[str]:
    """Collect structural warnings about the workflow graph.

    Current checks:
    - Nodes that have no incoming edges and are not the first node in topo order
      are flagged as potentially unreachable (only reachable via a conditional branch).

    :param workflow: The workflow to inspect.
    :param execution_order: The topological order returned by ``topological_sort``.
    :returns: A list of human-readable warning strings.
    """
    warnings: list[str] = []

    node_ids = set(workflow.nodes.keys())
    # Build in-degree map from edges that reference valid nodes only
    in_degree: dict[str, int] = dict.fromkeys(node_ids, 0)
    for edge in workflow.edges:
        if edge.to_node in node_ids:
            in_degree[edge.to_node] += 1

    first_node = execution_order[0] if execution_order else None

    for node_id in node_ids:
        if in_degree[node_id] == 0 and node_id != first_node:
            warnings.append(
                f'Node "{node_id}" is only reachable via one conditional branch — add a default edge or document intent'
            )

    return warnings


def outgoing_edges_for_node(node_id: str, edges: list[Edge]) -> list[Edge]:
    """Return all edges whose source is *node_id*.

    :param node_id: The source node identifier.
    :param edges: The full list of workflow edges.
    :returns: Filtered list of Edge objects.
    """
    return [e for e in edges if e.from_node == node_id]


# ---------------------------------------------------------------------------
# Plan data assembly
# ---------------------------------------------------------------------------


def build_plan(workflow: Workflow, workflow_name: str) -> dict[str, Any]:
    """Assemble the complete execution plan as a structured dict.

    Runs ``topological_sort`` to determine execution order and detects cycles.
    All analysis (errors, warnings, node details) is collected here.

    :param workflow: The validated workflow model.
    :param workflow_name: Display name for the workflow (from filename or a default).
    :returns: A dict suitable for JSON serialisation or text rendering.
    """
    node_ids = list(workflow.nodes.keys())
    edge_pairs = [(e.from_node, e.to_node) for e in workflow.edges]

    cycle_error: str | None = None
    execution_order: list[str] = []
    try:
        execution_order = topological_sort(node_ids, edge_pairs)
    except ValueError as exc:
        cycle_error = str(exc)

    errors = collect_errors(workflow)
    if cycle_error:
        errors.insert(0, cycle_error)

    warnings = collect_warnings(workflow, execution_order) if execution_order else []

    agent_count = sum(1 for n in workflow.nodes.values() if isinstance(n, AgentNode))

    node_plans: list[dict[str, Any]] = []
    for node_id in execution_order:
        node = workflow.nodes[node_id]
        out_edges = outgoing_edges_for_node(node_id, workflow.edges)
        edge_dicts = [{"to": e.to_node, "when": e.when} for e in out_edges]
        node_plans.append(
            {
                "id": node_id,
                "type": type(node).__name__.lower().removesuffix("node"),
                "type_info": node_type_info(node, workflow),
                "agent": node.agent if isinstance(node, (AgentNode, FactoryNode)) else None,
                "model": (
                    workflow.agents[node.agent].model
                    if isinstance(node, (AgentNode, FactoryNode)) and node.agent in workflow.agents
                    else None
                ),
                "writes": node_writes_path(node),
                "guardrails": node_guardrails(node, workflow),
                "outgoing_edges": edge_dicts,
            }
        )

    return {
        "workflow_name": workflow_name,
        "node_count": len(workflow.nodes),
        "agent_count": agent_count,
        "execution_order": execution_order,
        "nodes": node_plans,
        "workflow_guardrails": guardrail_labels(workflow.guardrails),
        "warnings": warnings,
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# Text rendering
# ---------------------------------------------------------------------------


def format_edge_condition(when: str | None) -> str:
    """Return the bracketed condition label for an edge.

    :param when: The raw ``when:`` expression, or None for an unconditional edge.
    :returns: A string such as ``'[working.intent == "refund"]'`` or ``"[default]"``.
    """
    if when is None:
        return "[default]"
    return f"[{when}]"


def render_text(plan: dict[str, Any]) -> str:
    """Render the execution plan as a multi-line human-readable text string.

    :param plan: The plan dict produced by ``build_plan``.
    :returns: The full text output ready to print.
    """
    lines: list[str] = []

    header = f"Workflow: {plan['workflow_name']}  │  {plan['node_count']} nodes  │  {plan['agent_count']} agents"
    lines.append(header)
    lines.append("")

    for idx, node in enumerate(plan["nodes"], start=1):
        guardrails_part = f"  guardrails=[{', '.join(node['guardrails'])}]" if node["guardrails"] else ""
        lines.append(f"{idx}. {node['id']:<18} {node['type_info']}{guardrails_part}")

        if node["writes"]:
            lines.append(f"   writes → {node['writes']}")

        for edge in node["outgoing_edges"]:
            condition = format_edge_condition(edge["when"])
            lines.append(f"   ↳ {condition:<40} → {edge['to']}")

        lines.append("")

    if plan["workflow_guardrails"]:
        lines.append(f"Guardrails (workflow-level): {', '.join(plan['workflow_guardrails'])}")

    if plan["warnings"]:
        lines.append("Warnings:")
        for w in plan["warnings"]:
            lines.append(f"  ⚠  {w}")

    if plan["errors"]:
        lines.append("Errors:")
        for e in plan["errors"]:
            lines.append(f"  ✗  {e}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI command
# ---------------------------------------------------------------------------


def explain_command(
    workflow_file: Annotated[str, typer.Argument(help="Path to the workflow YAML file")],
    output_format: Annotated[str, typer.Option("--format", "-f", help="Output format: 'text' or 'json'")] = "text",
) -> None:
    """Print a human-readable execution plan for a workflow without making any LLM calls.

    :param workflow_file: Path to the workflow YAML file to explain.
    :param output_format: Output format — ``"text"`` for human-readable output or ``"json"`` for
        machine-readable JSON.
    :returns: None. Exits with code 1 if the workflow has validation errors.
    """
    try:
        workflow = load_workflow(workflow_file)
    except FileNotFoundError as exc:
        _err.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc
    except ValueError as exc:
        _err.print(f"[red]Validation error:[/red] {exc}")
        raise typer.Exit(1) from exc

    workflow_name = Path(workflow_file).stem
    plan = build_plan(workflow, workflow_name)

    if output_format == "json":
        print(json.dumps(plan, indent=2))  # noqa: T201
    else:
        print(render_text(plan))  # noqa: T201

    if plan["errors"]:
        sys.exit(1)


def explain_workflow(workflow: Workflow, workflow_name: str, output_format: str = "text") -> tuple[str, bool]:
    """Explain a pre-loaded workflow and return the rendered output string and a has-errors flag.

    This function is the testable core of ``explain_command``: it accepts an already-validated
    :class:`~sirenspec.core.models.Workflow` object rather than a file path, so tests can
    construct workflows in memory without writing YAML to disk.

    :param workflow: A validated Workflow instance.
    :param workflow_name: Display name used in the plan header.
    :param output_format: Either ``"text"`` or ``"json"``.
    :returns: A tuple of ``(rendered_string, has_errors)`` where ``has_errors`` is True
        when the plan contains any structural errors.
    """
    plan = build_plan(workflow, workflow_name)
    if output_format == "json":
        rendered = json.dumps(plan, indent=2)
    else:
        rendered = render_text(plan)
    return rendered, bool(plan["errors"])
