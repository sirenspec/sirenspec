"""Load-time workflow linter: validates template expressions against known node IDs.

Run automatically by :func:`~sirenspec.yaml.parser.load_workflow` after model
validation.  Blocking issues (level ``"error"``) cause
:class:`~sirenspec.exceptions.WorkflowLintError` to be raised before the workflow
is returned to the caller.  Advisory issues (level ``"warning"``) are collected and
returned without raising.

Current rules
-------------
``working_dot_node_id``
    ``{{ working.<node_id>.* }}`` where ``<node_id>`` is a known node ID is always
    wrong.  The canonical access path is ``{{ <node_id>.output }}``.

``unknown_namespace``
    A top-level name in a ``{{ expr }}`` expression that is not a reserved
    namespace (``inputs``, ``env``, ``item``, ``index``, ``total``) and not a
    known node ID is likely a typo or a broken reference.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from sirenspec.core.models import AgentNode, FactoryNode, SwrmNode, Workflow, WorkflowNode

_TEMPLATE_RE = re.compile(r"\{\{\s*(.+?)\s*\}\}")
_DEFAULT_FILTER_RE = re.compile(r"^(.+?)\s*\|\s*(?:default|json_or_default)\s*\(")
_RESERVED = frozenset({"inputs", "env", "item", "index", "total", "memory"})


@dataclass
class LintIssue:
    """A single finding from the workflow linter.

    :param level: Severity — ``"error"`` (blocks load) or ``"warning"`` (advisory).
    :param rule: Machine-readable rule identifier (e.g. ``"working_dot_node_id"``).
    :param message: Human-readable description of the problem and how to fix it.
    :param location: Optional human-readable source location hint, e.g. ``"agent 'plan' system prompt"``.
    """

    level: Literal["error", "warning"]
    rule: str
    message: str
    location: str = ""


def collect_templates(workflow: Workflow) -> list[tuple[str, str]]:
    """Walk all template-bearing fields in *workflow* and return ``(template, location)`` pairs.

    Covers agent system prompts, swrm agent and synthesis prompts, factory
    ``for_each``/``swarm_size`` expressions, factory ``inputs`` values, and
    workflow node ``inputs`` values.

    :param workflow: The fully-validated workflow to inspect.
    :returns: List of ``(template_string, location_hint)`` pairs.
    """
    pairs: list[tuple[str, str]] = []

    for node_id, node in workflow.nodes.items():
        if isinstance(node, AgentNode):
            agent_def = workflow.agents.get(node.agent)
            if agent_def:
                pairs.append((agent_def.system, f"agent '{node.agent}' system prompt (used by node '{node_id}')"))

        elif isinstance(node, SwrmNode):
            for agent in node.agents:
                pairs.append((agent.prompt, f"swrm node '{node_id}' agent '{agent.id}' prompt"))
            if node.synthesis:
                pairs.append((node.synthesis.prompt, f"swrm node '{node_id}' synthesis prompt"))

        elif isinstance(node, FactoryNode):
            if node.for_each is not None:
                pairs.append((node.for_each, f"factory node '{node_id}' for_each"))
            if isinstance(node.swarm_size, str):
                pairs.append((node.swarm_size, f"factory node '{node_id}' swarm_size"))
            for key, val in node.inputs.items():
                pairs.append((val, f"factory node '{node_id}' inputs['{key}']"))
            if node.swrm is not None:
                for agent in node.swrm.agents:
                    pairs.append((agent.prompt, f"factory node '{node_id}' swrm agent '{agent.id}' prompt"))
                if node.swrm.synthesis is not None:
                    pairs.append((node.swrm.synthesis.prompt, f"factory node '{node_id}' swrm synthesis prompt"))

        elif isinstance(node, WorkflowNode):
            for key, val in node.inputs.items():
                pairs.append((val, f"workflow node '{node_id}' inputs['{key}']"))

    return pairs


def extract_top_level_names(template: str) -> list[tuple[str, str]]:
    """Return ``(full_expr, first_segment)`` for every ``{{ … }}`` placeholder in *template*.

    The ``| default(...)`` and ``| json_or_default(...)`` filter suffixes are
    stripped before the first segment is extracted.

    :param template: A template string containing zero or more ``{{ … }}`` placeholders.
    :returns: List of ``(full_expression, first_segment)`` pairs.
    """
    result: list[tuple[str, str]] = []
    for match in _TEMPLATE_RE.finditer(template):
        expr = match.group(1).strip()
        filter_match = _DEFAULT_FILTER_RE.match(expr)
        if filter_match:
            expr = filter_match.group(1).strip()
        first = expr.split(".")[0].strip()
        result.append((expr, first))
    return result


def check_working_dot_node_id(template: str, location: str, node_ids: set[str]) -> list[LintIssue]:
    """Detect ``{{ working.<node_id>.* }}`` expressions that reference a known node ID.

    These are almost always bugs: ``working`` is an internal context namespace, not
    a node ID.  The correct form is ``{{ <node_id>.output }}``.

    :param template: The template string to scan.
    :param location: Human-readable source location for error messages.
    :param node_ids: Set of valid node IDs in the workflow.
    :returns: List of :class:`LintIssue` instances found.
    """
    issues: list[LintIssue] = []
    for expr, first in extract_top_level_names(template):
        if first != "working":
            continue
        parts = expr.split(".")
        if len(parts) < 2:
            continue
        second = parts[1]
        if second in node_ids:
            suggestion = f"{{{{ {second}.output }}}}" if len(parts) == 2 else f"{{{{ {'.'.join(parts[1:])} }}}}"
            issues.append(
                LintIssue(
                    level="error",
                    rule="working_dot_node_id",
                    message=(
                        f"'{{{{ {expr} }}}}' references node '{second}' via the internal 'working' namespace. "
                        f"Use '{suggestion}' instead."
                    ),
                    location=location,
                )
            )
    return issues


def check_unknown_namespace(template: str, location: str, node_ids: set[str]) -> list[LintIssue]:
    """Warn when a top-level template name is neither a reserved namespace nor a known node ID.

    :param template: The template string to scan.
    :param location: Human-readable source location for warning messages.
    :param node_ids: Set of valid node IDs in the workflow.
    :returns: List of advisory :class:`LintIssue` instances.
    """
    issues: list[LintIssue] = []
    for expr, first in extract_top_level_names(template):
        if first in _RESERVED or first in node_ids:
            continue
        issues.append(
            LintIssue(
                level="warning",
                rule="unknown_namespace",
                message=(
                    f"'{{{{ {expr} }}}}' starts with '{first}', which is not a reserved namespace "
                    f"(inputs, env, item, index, total) nor a known node ID. "
                    f"This will raise InterpolationError at runtime unless a '| default(...)' filter is present."
                ),
                location=location,
            )
        )
    return issues


def lint_workflow(workflow: Workflow) -> list[LintIssue]:
    """Run all lint rules against *workflow* and return every issue found.

    Does not raise — callers that want to block on errors should check
    ``[i for i in issues if i.level == "error"]`` and raise themselves, or use
    :func:`~sirenspec.yaml.parser.load_workflow` which does this automatically.

    :param workflow: A fully-validated :class:`~sirenspec.core.models.Workflow`.
    :returns: List of :class:`LintIssue` instances (may be empty).
    """
    node_ids = set(workflow.nodes.keys())
    issues: list[LintIssue] = []

    for template, location in collect_templates(workflow):
        issues.extend(check_working_dot_node_id(template, location, node_ids))
        issues.extend(check_unknown_namespace(template, location, node_ids))

    return issues
