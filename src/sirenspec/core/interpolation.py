"""Template interpolation engine for ``{{ expr }}`` placeholders in SirenSpec workflows.

Resolves expressions at node execution time against a structured namespace that includes
workflow inputs, node outputs, environment variables, and loop variables (factory nodes).

Supported namespaces::

    {{ inputs.field }}                  # workflow input field
    {{ env.VAR_NAME }}                  # os.environ value at resolution time
    {{ node_id.output }}                # canonical node output slot
    {{ node_id.agents.agent_id.output}} # swrm sub-agent output
    {{ classify.output.sentiment }}     # nested dot access
    {{ item }}                          # current loop item (factory nodes)
    {{ index }}                         # current loop index (factory nodes)
    {{ expr | default('fallback') }}    # optional fallback on missing key

Missing keys raise :class:`~sirenspec.exceptions.InterpolationError` unless a
``| default('value')`` filter is provided.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

from sirenspec.core.models import AgentNode, FactoryNode, SwrmNode, Workflow
from sirenspec.exceptions import InterpolationError

_TEMPLATE_RE = re.compile(r"\{\{\s*(.+?)\s*\}\}")
_DEFAULT_RE = re.compile(r"^(.+?)\s*\|\s*default\(\s*['\"](.+?)['\"]\s*\)\s*$")

_RESERVED_NAMESPACES = frozenset({"inputs", "env", "item", "index"})


@dataclass
class InterpolationContext:
    """All namespaces available during template resolution.

    :param inputs: Workflow input fields, keyed by name. Always includes ``message``.
    :param nodes: Node output data, sourced from the executor's ``working`` dict.
        Accessed by node ID: ``nodes["plan"]["output"]`` → ``{{ plan.output }}``.
    :param env: Environment variables snapshot (from ``os.environ`` at build time).
    :param item: Current loop iteration value. ``None`` outside a factory loop.
    :param index: Current loop iteration index. ``None`` outside a factory loop.
    """

    inputs: dict[str, Any]
    nodes: dict[str, Any]
    env: dict[str, str] = field(default_factory=dict)
    item: Any | None = None
    index: int | None = None


def build_interpolation_context(
    user_input: str,
    working: dict[str, Any],
    item: Any | None = None,
    index: int | None = None,
) -> InterpolationContext:
    """Build an :class:`InterpolationContext` from executor state.

    :param user_input: The original workflow input message.
    :param working: The executor's current ``working`` dict (contains node outputs).
    :param item: Loop item for factory nodes; ``None`` outside a loop.
    :param index: Loop index for factory nodes; ``None`` outside a loop.
    :returns: A fully populated interpolation context.
    """
    return InterpolationContext(
        inputs={"message": user_input},
        nodes=working,
        env=dict(os.environ),
        item=item,
        index=index,
    )


def navigate_dict(data: dict[str, Any], parts: list[str], full_path: str, namespace: str) -> Any:
    """Walk nested dict structure along ``parts``, raising on any missing key.

    :param data: Root dict to traverse.
    :param parts: Ordered list of key segments to follow.
    :param full_path: The original expression (used in error messages).
    :param namespace: Namespace label (used in error messages).
    :raises InterpolationError: If any segment of the path is missing or non-dict.
    :returns: The value at the end of the path.
    """
    current: Any = data
    for part in parts:
        if not isinstance(current, dict) or part not in current:
            raise InterpolationError(full_path, namespace, f"Key '{part}' not found")
        current = current[part]
    return current


def extract_node_refs(template: str, known_node_ids: set[str]) -> set[str]:
    """Return the set of node IDs referenced in ``{{ node_id.* }}`` expressions.

    Ignores reserved namespaces (``inputs``, ``env``, ``item``, ``index``).

    :param template: A prompt or template string to scan.
    :param known_node_ids: Set of valid node IDs in the workflow.
    :returns: Set of node ID strings found in the template.
    """
    refs: set[str] = set()
    for match in _TEMPLATE_RE.finditer(template):
        expr = match.group(1).strip()
        default_match = _DEFAULT_RE.match(expr)
        if default_match:
            expr = default_match.group(1).strip()
        first_segment = expr.split(".")[0].strip()
        if first_segment not in _RESERVED_NAMESPACES and first_segment in known_node_ids:
            refs.add(first_segment)
    return refs


def resolve_expression(
    expr: str,
    context: InterpolationContext,
    redact_env: bool = False,
) -> str:
    """Resolve a single interpolation expression (without braces) to a string value.

    Supports the ``| default('fallback')`` filter: if the expression would raise
    :class:`~sirenspec.exceptions.InterpolationError`, the default value is returned instead.

    :param expr: The raw expression text, e.g. ``inputs.message`` or ``plan.output | default('')``.
    :param context: The current interpolation context.
    :param redact_env: When ``True``, ``env.*`` values are returned as ``'***'`` instead of their
        real values. Used when building trace output to avoid logging credentials.
    :raises InterpolationError: If resolution fails and no default filter is present.
    :returns: The resolved string value.
    """
    default_value: str | None = None
    default_match = _DEFAULT_RE.match(expr)
    if default_match:
        expr = default_match.group(1).strip()
        default_value = default_match.group(2)

    try:
        return _resolve_path(expr, context, redact_env)
    except InterpolationError:
        if default_value is not None:
            return default_value
        raise


def _resolve_path(expr: str, context: InterpolationContext, redact_env: bool) -> str:
    """Resolve a dotted path against the interpolation context namespaces.

    :param expr: Dot-separated path, e.g. ``inputs.message``, ``env.HOME``, ``plan.output``.
    :param context: The interpolation context providing all namespaces.
    :param redact_env: When ``True``, redact env values as ``'***'``.
    :raises InterpolationError: If the path cannot be resolved.
    :returns: String representation of the resolved value.
    """
    stripped = expr.strip()
    parts = stripped.split(".")
    namespace = parts[0]

    if namespace == "item":
        if context.item is None:
            raise InterpolationError(stripped, "item", "Not inside a factory loop context")
        return str(context.item)

    if namespace == "index":
        if context.index is None:
            raise InterpolationError(stripped, "index", "Not inside a factory loop context")
        return str(context.index)

    if namespace == "inputs":
        if len(parts) < 2:
            raise InterpolationError(stripped, "inputs", "Missing field name after 'inputs.'")
        return str(navigate_dict(context.inputs, parts[1:], stripped, "inputs"))

    if namespace == "env":
        if len(parts) < 2:
            raise InterpolationError(stripped, "env", "Missing variable name after 'env.'")
        var_name = parts[1]
        if var_name not in context.env:
            raise InterpolationError(stripped, "env", f"Environment variable '{var_name}' is not set")
        if redact_env:
            return "***"
        return context.env[var_name]

    return str(navigate_dict(context.nodes, parts, stripped, namespace))


def resolve_template(
    template: str,
    context: InterpolationContext,
    redact_env: bool = False,
) -> str:
    """Render all ``{{ expr }}`` placeholders in *template* against *context*.

    Multiple expressions per string are supported. Resolution is strict: any
    unresolvable expression raises :class:`~sirenspec.exceptions.InterpolationError`
    unless a ``| default()`` filter is present.

    :param template: Template string containing zero or more ``{{ … }}`` placeholders.
    :param context: Interpolation context providing all namespaces.
    :param redact_env: When ``True``, ``env.*`` values appear as ``***`` in the output.
        Pass ``True`` when building trace output to avoid logging credentials.
    :raises InterpolationError: If any placeholder cannot be resolved.
    :returns: Fully rendered string.
    """

    def replace(match: re.Match[str]) -> str:
        return resolve_expression(match.group(1).strip(), context, redact_env)

    return _TEMPLATE_RE.sub(replace, template)


def resolve_to_list(template: str, context: InterpolationContext) -> list[Any]:
    """Resolve *template* to a string and parse the result as a JSON list.

    Used by the factory runner to evaluate ``for_each:`` expressions. The
    upstream node must output a valid JSON array string, e.g. ``["a", "b", "c"]``.

    :param template: A template string that resolves to a JSON-encoded list.
    :param context: The interpolation context.
    :raises InterpolationError: If resolution fails or the result is not a JSON list.
    :returns: The parsed Python list.
    """
    resolved = resolve_template(template, context)
    try:
        result = json.loads(resolved)
    except json.JSONDecodeError as exc:
        raise InterpolationError(template, "for_each", f"Result is not valid JSON: {exc}") from exc
    if not isinstance(result, list):
        raise InterpolationError(template, "for_each", f"Expected a JSON list but got {type(result).__name__}")
    return result


def check_circular_template_refs(workflow: Workflow) -> None:
    """Detect circular template dependencies among node prompts at parse time.

    Scans ``AgentDefinition.system`` prompts (via their ``AgentNode`` users),
    ``SwrmAgent.prompt``, ``SwrmSynthesis.prompt``, and ``FactoryNode.for_each``
    and ``inputs`` values for ``{{ node_id.* }}`` references.  Builds a directed
    dependency graph and checks it for cycles using Kahn's algorithm.

    :param workflow: The fully-validated Workflow model.
    :raises InterpolationError: If a circular template reference is detected.
    """
    node_ids = set(workflow.nodes.keys())
    deps: dict[str, set[str]] = {nid: set() for nid in node_ids}

    for node_id, node in workflow.nodes.items():
        templates: list[str] = []

        if isinstance(node, AgentNode):
            agent_def = workflow.agents.get(node.agent)
            if agent_def:
                templates.append(agent_def.system)

        elif isinstance(node, SwrmNode):
            for agent in node.agents:
                templates.append(agent.prompt)
            if node.synthesis:
                templates.append(node.synthesis.prompt)

        elif isinstance(node, FactoryNode):
            templates.append(node.for_each)
            templates.extend(node.inputs.values())

        for tmpl in templates:
            for ref in extract_node_refs(tmpl, node_ids):
                if ref != node_id:
                    deps[node_id].add(ref)

    try:
        _topological_sort_deps(node_ids, deps)
    except ValueError as exc:
        raise InterpolationError(
            "", "circular_ref", "Circular template reference detected among workflow nodes"
        ) from exc


def _topological_sort_deps(node_ids: set[str], deps: dict[str, set[str]]) -> list[str]:
    """Run Kahn's topological sort on the template dependency graph.

    :param node_ids: All node IDs.
    :param deps: Mapping of node_id → set of node_ids it depends on (references).
    :raises ValueError: If a cycle is detected.
    :returns: Topologically sorted list of node IDs.
    """
    from collections import deque

    in_degree: dict[str, int] = dict.fromkeys(node_ids, 0)
    adjacency: dict[str, list[str]] = {nid: [] for nid in node_ids}

    for node_id, referenced_set in deps.items():
        for ref in referenced_set:
            adjacency[ref].append(node_id)
            in_degree[node_id] += 1

    queue: deque[str] = deque(n for n in node_ids if in_degree[n] == 0)
    order: list[str] = []

    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbour in adjacency[node]:
            in_degree[neighbour] -= 1
            if in_degree[neighbour] == 0:
                queue.append(neighbour)

    if len(order) != len(node_ids):
        raise ValueError("Cycle detected in template dependency graph")

    return order
