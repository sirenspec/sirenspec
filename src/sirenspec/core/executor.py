"""Async workflow executor: orchestrates nodes, guardrails, providers, and trace."""

from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Any

from sirenspec.core.context import WorkflowContext
from sirenspec.core.models import Workflow
from sirenspec.guardrails.base import GuardrailViolation
from sirenspec.guardrails.registry import build_guardrails
from sirenspec.providers.registry import resolve_provider


class DotDict:
    """Wraps a plain dict so that attribute-style (dot) access works in ``when:`` expressions.

    Nested dicts are wrapped recursively, so a path like ``working.triage.intent``
    resolves correctly without any special parsing in the condition string.

    Only non-private attribute lookups are forwarded to the underlying dict;
    any ``AttributeError`` propagates naturally when a key is missing.
    """

    def __init__(self, data: dict[str, Any]) -> None:
        # Store under a mangled name so user keys never shadow it.
        object.__setattr__(self, "_data", data)

    def __getattr__(self, key: str) -> Any:
        data: dict[str, Any] = object.__getattribute__(self, "_data")
        try:
            value = data[key]
        except KeyError as exc:
            raise AttributeError(key) from exc
        # Recursively wrap nested dicts so deep paths keep resolving.
        return DotDict(value) if isinstance(value, dict) else value

    def __eq__(self, other: object) -> bool:
        data: dict[str, Any] = object.__getattribute__(self, "_data")
        if isinstance(other, DotDict):
            return data == object.__getattribute__(other, "_data")
        return data == other

    def __repr__(self) -> str:  # pragma: no cover
        return repr(object.__getattribute__(self, "_data"))


def evaluate_when_condition(condition: str, context: WorkflowContext) -> bool:
    """Evaluate a ``when:`` expression against the current workflow context.

    The expression runs in a tightly-restricted namespace:

    * ``working`` — a :class:`DotDict` wrapping ``context.working``
    * ``output``  — a :class:`DotDict` wrapping ``context.output``
    * ``true`` / ``false`` / ``null`` — YAML boolean/null literals

    No built-ins are available (``__builtins__`` is cleared), so arbitrary
    imports, file access, or other side-effects cannot occur.

    :param condition: A Python expression string, e.g. ``working.triage.intent == "refund"``.
    :param context: The live workflow context providing working and output state.
    :returns: ``True`` if the expression is truthy; ``False`` on any evaluation error.
    """
    try:
        # Build a safe evaluation namespace with only the allowed names.
        namespace: dict[str, Any] = {
            "working": DotDict(context.working),
            "output": DotDict(context.output),
            # Map YAML boolean/null literals so authors can write them naturally.
            "true": True,
            "false": False,
            "null": None,
        }
        # __builtins__ is set to an empty dict to block all built-in functions
        # and prevent code from importing modules or doing other unsafe operations.
        result = eval(condition, {"__builtins__": {}}, namespace)  # noqa: S307
        return bool(result)
    except Exception:
        # Per spec: any evaluation failure (missing key, syntax error, type error)
        # is treated as a false condition so the edge is not traversed.
        return False


def topological_sort(node_ids: list[str], edges: list[tuple[str, str]]) -> list[str]:
    """Return node IDs in topological order (Kahn's algorithm).

    :param node_ids: All node identifiers in the workflow.
    :param edges: Directed (source, destination) pairs — ``when:`` conditions are not
        considered here; they are evaluated at runtime by :func:`execute`.
    :raises ValueError: If the graph contains a cycle.
    :returns: A list of node IDs in a valid execution order.
    """
    in_degree: dict[str, int] = dict.fromkeys(node_ids, 0)
    adjacency: dict[str, list[str]] = defaultdict(list)

    for src, dst in edges:
        adjacency[src].append(dst)
        in_degree[dst] += 1

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
        raise ValueError("Workflow graph contains a cycle")

    return order


async def execute(workflow: Workflow, user_input: str) -> dict[str, Any]:
    """Execute a workflow and return a structured JSON-serialisable trace.

    **Execution model**

    1. A topological sort over all nodes determines a stable iteration order
       and detects cycles up-front.
    2. Only *root* nodes (no incoming edges) are initially marked *active*.
    3. After each node completes, outgoing edges are evaluated:

       * An edge without a ``when:`` condition always activates its target.
       * An edge with a ``when:`` condition activates its target only if the
         expression evaluates to ``True`` against the current context.

    4. Nodes that never become active are silently skipped, so exactly one
       branch of a conditional fork is executed.

    :param workflow: Validated :class:`~sirenspec.core.models.Workflow` instance.
    :param user_input: The initial user message (from CLI ``--input`` or
        ``workflow.input.message``).
    :returns: Execution trace dict with workflow metadata, per-node entries, and summary.
    """
    context = WorkflowContext(initial_state=workflow.state)

    node_ids = list(workflow.nodes.keys())

    # Build a forward-adjacency map that retains each edge's when: condition.
    # This is used after every node execution to activate eligible successors.
    out_edges: dict[str, list[tuple[str, str | None]]] = {n: [] for n in node_ids}
    in_degree: dict[str, int] = dict.fromkeys(node_ids, 0)
    for edge in workflow.edges:
        out_edges[edge.from_node].append((edge.to_node, edge.when))
        in_degree[edge.to_node] += 1

    # Use all edges (without when: semantics) for topological ordering only —
    # the sort gives a deterministic, cycle-free iteration sequence.
    edge_pairs = [(e.from_node, e.to_node) for e in workflow.edges]
    execution_order = topological_sort(node_ids, edge_pairs)

    # Root nodes are unconditionally active; all other nodes start inactive and
    # become active only when an incoming edge's condition is satisfied at runtime.
    active_nodes: set[str] = {n for n in node_ids if in_degree[n] == 0}

    # Track the writes path of the most recently completed node for downstream input resolution.
    last_writes_path: str | None = None

    trace_nodes: list[dict[str, Any]] = []
    total_tokens = 0
    total_duration_ms = 0.0
    status = "success"

    global_guardrail_names = workflow.guardrails

    for node_id in execution_order:
        # Skip nodes that no active edge has routed to yet.
        # This is how conditional branching suppresses the unchosen fork.
        if node_id not in active_nodes:
            continue

        node = workflow.nodes[node_id]
        agent_def = workflow.agents[node.agent]

        # Resolve this node's input: first active node gets raw user input;
        # subsequent nodes receive the output of the most recently executed node.
        if last_writes_path is None:
            node_input = user_input
        else:
            try:
                node_input = str(context.resolve(last_writes_path))
            except KeyError:
                node_input = user_input

        # Agent-level guardrails take precedence over workflow-level guardrails.
        guardrail_names = agent_def.guardrails if agent_def.guardrails is not None else global_guardrail_names
        guardrails = build_guardrails(guardrail_names)

        node_trace: dict[str, Any] = {
            "id": node_id,
            "agent": node.agent,
            "prompt_sent": node_input,
            "response_received": None,
            "writes": node.writes,
            "guardrails_passed": [],
            "tokens": 0,
            "duration_ms": 0,
            "error": None,
        }

        start_time = time.monotonic()
        try:
            # Run input guardrails before sending to the LLM.
            checked_input = node_input
            for g in guardrails:
                checked_input = g.check_input(checked_input)
                node_trace["guardrails_passed"].append(f"{type(g).__name__}.check_input")

            messages = [
                {"role": "system", "content": agent_def.system},
                {"role": "user", "content": checked_input},
            ]

            provider = resolve_provider(agent_def.model)
            response_text = await provider.complete(messages)
            tokens = provider.last_token_count

            # Run output guardrails before writing the response to the context.
            checked_output = response_text
            for g in guardrails:
                checked_output = g.check_output(checked_output)
                node_trace["guardrails_passed"].append(f"{type(g).__name__}.check_output")

            context.write(node.writes, checked_output)
            last_writes_path = node.writes

            # Evaluate outgoing edges now that the context has been updated.
            # An unconditional edge always activates its target; a conditional edge
            # does so only if its when: expression evaluates to True.
            for target, condition in out_edges[node_id]:
                if condition is None or evaluate_when_condition(condition, context):
                    active_nodes.add(target)

            duration_ms = (time.monotonic() - start_time) * 1000
            node_trace.update(
                {
                    "response_received": checked_output,
                    "tokens": tokens,
                    "duration_ms": round(duration_ms, 2),
                }
            )
            total_tokens += tokens
            total_duration_ms += duration_ms

        except GuardrailViolation as exc:
            duration_ms = (time.monotonic() - start_time) * 1000
            node_trace["error"] = f"GuardrailViolation: {exc.reason}"
            node_trace["duration_ms"] = round(duration_ms, 2)
            total_duration_ms += duration_ms
            status = "failed"
            trace_nodes.append(node_trace)
            break

        except Exception as exc:
            duration_ms = (time.monotonic() - start_time) * 1000
            node_trace["error"] = str(exc)
            node_trace["duration_ms"] = round(duration_ms, 2)
            total_duration_ms += duration_ms
            status = "failed"
            trace_nodes.append(node_trace)
            break

        trace_nodes.append(node_trace)

    return {
        "workflow": {"version": workflow.version},
        "input": {"message": user_input},
        "nodes": trace_nodes,
        "output": context.output,
        "summary": {
            "total_tokens": total_tokens,
            "total_duration_ms": round(total_duration_ms, 2),
            "status": status,
        },
    }
