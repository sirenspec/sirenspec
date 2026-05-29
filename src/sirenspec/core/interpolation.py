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
    {{ expr | default('fallback') }}          # fallback on missing key or empty string
    {{ expr | json_or_default('[]') }}        # fallback on missing key, empty string, or invalid JSON

Missing keys raise :class:`~sirenspec.exceptions.InterpolationError` unless a
``| default('value')`` or ``| json_or_default('value')`` filter is provided.
Both filters also engage when the resolved value is an empty string ``""``.
``json_or_default`` additionally engages when the value is not parseable as JSON.
"""

from __future__ import annotations

import json
import os
import re
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from sirenspec.core.models import AgentNode, FactoryNode, SwrmNode, Workflow, WorkflowNode
from sirenspec.exceptions import InterpolationError

_TEMPLATE_RE = re.compile(r"\{\{\s*(.+?)\s*\}\}")
_DEFAULT_RE = re.compile(r"^(.+?)\s*\|\s*default\(\s*['\"](.+?)['\"]\s*\)\s*$")
_JSON_OR_DEFAULT_RE = re.compile(r"^(.+?)\s*\|\s*json_or_default\(\s*['\"](.+?)['\"]\s*\)\s*$")

_RESERVED_NAMESPACES = frozenset({"inputs", "env", "item", "index", "total", "memory"})


@dataclass
class InterpolationContext:
    """All namespaces available during template resolution.

    :param inputs: Workflow input fields, keyed by name. Always includes ``message``.
    :param nodes: Node output data, sourced from the executor's ``working`` dict.
        Accessed by node ID: ``nodes["plan"]["output"]`` → ``{{ plan.output }}``.
    :param env: Environment variables snapshot (from ``os.environ`` at build time).
    :param item: Current loop iteration value. ``None`` outside a factory loop.
    :param index: Current loop iteration index. ``None`` outside a factory loop.
    :param total: Total number of instances in the current factory loop. ``None`` outside a loop.
    :param memory: Snapshot of live persistent memory keys for ``{{ memory.key }}`` resolution.
    """

    inputs: dict[str, Any]
    nodes: dict[str, Any]
    env: dict[str, str] = field(default_factory=dict)
    item: Any | None = None
    index: int | None = None
    total: int | None = None
    memory: dict[str, Any] = field(default_factory=dict)


def build_interpolation_context(
    user_input: str,
    working: dict[str, Any],
    item: Any | None = None,
    index: int | None = None,
    total: int | None = None,
    extra_inputs: dict[str, Any] | None = None,
    memory: dict[str, Any] | None = None,
) -> InterpolationContext:
    """Build an :class:`InterpolationContext` from executor state.

    :param user_input: The original workflow input message.
    :param working: The executor's current ``working`` dict (contains node outputs).
    :param item: Loop item for factory nodes; ``None`` outside a loop.
    :param index: Loop index for factory nodes; ``None`` outside a loop.
    :param total: Total instance count for factory loops; ``None`` outside a loop.
    :param extra_inputs: Additional key/value pairs merged into the ``inputs`` namespace.
        Used by sub-workflow nodes to inject bound inputs so that ``{{ inputs.key }}``
        resolves correctly inside the sub-workflow.
    :param memory: Snapshot of live persistent memory keys from the MemoryManager.
        When provided, ``{{ memory.key }}`` expressions resolve against this dict.
    :returns: A fully populated interpolation context.
    """
    inputs: dict[str, Any] = {"message": user_input}
    if extra_inputs:
        inputs.update(extra_inputs)
    return InterpolationContext(
        inputs=inputs,
        nodes=working,
        env=dict(os.environ),
        item=item,
        index=index,
        total=total,
        memory=memory or {},
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

    Supports two optional filters:

    ``| default('fallback')``
        Returns *fallback* when the expression raises
        :class:`~sirenspec.exceptions.InterpolationError` **or** when the resolved
        value is the empty string ``""``.

    ``| json_or_default('fallback')``
        Like ``| default`` but also returns *fallback* when the resolved value
        cannot be parsed as JSON.  Useful for ``for_each:`` upstream outputs
        that may legitimately be empty or contain non-JSON text.

    :param expr: The raw expression text, e.g. ``inputs.message`` or
        ``plan.output | default('')``.
    :param context: The current interpolation context.
    :param redact_env: When ``True``, ``env.*`` values are returned as ``'***'`` instead of their
        real values. Used when building trace output to avoid logging credentials.
    :raises InterpolationError: If resolution fails and no default filter is present.
    :returns: The resolved string value.
    """
    default_value: str | None = None
    use_json_or_default = False

    json_match = _JSON_OR_DEFAULT_RE.match(expr)
    if json_match:
        expr = json_match.group(1).strip()
        default_value = json_match.group(2)
        use_json_or_default = True
    else:
        default_match = _DEFAULT_RE.match(expr)
        if default_match:
            expr = default_match.group(1).strip()
            default_value = default_match.group(2)

    try:
        resolved = _resolve_path(expr, context, redact_env)
    except InterpolationError:
        if default_value is not None:
            return default_value
        raise

    if default_value is not None and resolved == "":
        return default_value

    if use_json_or_default and default_value is not None:
        try:
            json.loads(resolved)
        except json.JSONDecodeError:
            return default_value

    return resolved


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

    if namespace == "total":
        if context.total is None:
            raise InterpolationError(stripped, "total", "Not inside a factory loop context")
        return str(context.total)

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

    if namespace == "memory":
        if len(parts) < 2:
            raise InterpolationError(stripped, "memory", "Missing key name after 'memory.'")
        return str(navigate_dict(context.memory, parts[1:], stripped, "memory"))

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


def strip_json_fence(text: str) -> str:
    """Strip a leading `` ```json `` (or bare `` ``` ``) fence and trailing `` ``` `` from *text*.

    Many LLMs wrap JSON output in markdown fences.  This function removes the fence
    so the inner content can be parsed directly.

    :param text: The raw string that may contain a markdown code fence.
    :returns: The de-fenced string, or *text* unchanged if no fence is present.
    """
    stripped = text.strip()
    if not stripped.startswith("```"):
        return text
    first_newline = stripped.find("\n")
    last_fence = stripped.rfind("```")
    if first_newline == -1 or last_fence <= first_newline:
        return text
    return stripped[first_newline:last_fence].strip()


def resolve_to_list(template: str, context: InterpolationContext) -> list[Any]:
    """Resolve *template* and return the result as a Python list.

    Three resolution strategies are tried in order:

    1. **Native list** — if the resolved raw value is already a Python ``list``
       (e.g. when the upstream is a tool node that returned a list object), it
       is returned directly without JSON parsing.
    2. **Plain JSON** — ``json.loads`` on the resolved string.
    3. **Jsonish (fenced) JSON** — if plain parsing fails, a leading
       `` ```json `` or `` ``` `` fence is stripped and parsing is retried.

    :param template: A template expression (e.g. ``{{ plan.output }}``) whose
        resolved value must be a JSON array string or a Python list.
    :param context: The interpolation context.
    :raises InterpolationError: If resolution fails, or the result is not a list
        after all three strategies are exhausted.
    :returns: The resolved Python list.
    """
    raw = resolve_raw_value(template, context)
    if isinstance(raw, list):
        return raw

    resolved = str(raw)
    try:
        result = json.loads(resolved)
    except json.JSONDecodeError:
        de_fenced = strip_json_fence(resolved)
        try:
            result = json.loads(de_fenced)
        except json.JSONDecodeError as exc:
            raise InterpolationError(template, "for_each", f"Result is not valid JSON: {exc}") from exc

    if not isinstance(result, list):
        raise InterpolationError(template, "for_each", f"Expected a JSON list but got {type(result).__name__}")
    return result


def resolve_raw_value(template: str, context: InterpolationContext) -> Any:
    """Resolve a simple ``{{ expr }}`` template and return the raw Python value without stringification.

    If *template* is not a single ``{{ expr }}`` expression (e.g. it contains
    literal text or multiple placeholders), it is returned unchanged as a string.

    This is used by :func:`resolve_to_list` to detect when an upstream tool node
    produced a native Python list rather than a JSON-encoded string.

    :param template: A template string, ideally a single ``{{ expr }}`` placeholder.
    :param context: The interpolation context.
    :raises InterpolationError: If the expression cannot be resolved.
    :returns: The raw Python value (possibly a list, dict, or any type) without ``str()`` coercion.
    """
    m = _TEMPLATE_RE.fullmatch(template.strip()) if template else None
    if m is None:
        return template

    expr = m.group(1).strip()
    default_value: str | None = None

    json_match = _JSON_OR_DEFAULT_RE.match(expr)
    if json_match:
        expr = json_match.group(1).strip()
        default_value = json_match.group(2)
    else:
        default_match = _DEFAULT_RE.match(expr)
        if default_match:
            expr = default_match.group(1).strip()
            default_value = default_match.group(2)

    parts = expr.strip().split(".")
    namespace = parts[0]

    if namespace in _RESERVED_NAMESPACES:
        return resolve_expression(m.group(1).strip(), context)

    try:
        return _navigate_raw(context.nodes, parts)
    except (KeyError, TypeError) as exc:
        if default_value is not None:
            return default_value
        raise InterpolationError(expr, namespace, f"Key '{parts[-1]}' not found") from exc


def _navigate_raw(data: dict[str, Any], parts: list[str]) -> Any:
    """Walk *data* along *parts* and return the raw value at the end without stringification.

    :param data: The root dict to traverse.
    :param parts: Ordered list of key segments.
    :raises KeyError: If any segment is missing.
    :raises TypeError: If an intermediate value is not a dict.
    :returns: The raw Python value at the end of the path.
    """
    current: Any = data
    for part in parts:
        if not isinstance(current, dict):
            raise TypeError(f"Expected dict at '{part}', got {type(current).__name__}")
        current = current[part]
    return current


def check_circular_template_refs(workflow: Workflow) -> None:
    """Detect circular template dependencies among node prompts at parse time.

    Scans ``AgentDefinition.system`` prompts (via their ``AgentNode`` users),
    ``SwrmAgent.prompt``, ``SwrmSynthesis.prompt``, ``FactoryNode.for_each``,
    ``FactoryNode.swarm_size`` (when a template string), and ``inputs`` values
    for ``{{ node_id.* }}`` references.  Builds a directed
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
            if node.for_each is not None:
                templates.append(node.for_each)
            if isinstance(node.swarm_size, str):
                templates.append(node.swarm_size)
            if node.swrm is not None:
                for agent in node.swrm.agents:
                    templates.append(agent.prompt)
                if node.swrm.synthesis is not None:
                    templates.append(node.swrm.synthesis.prompt)
            templates.extend(node.inputs.values())

        elif isinstance(node, WorkflowNode):
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
