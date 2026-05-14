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


def _topological_sort(node_ids: list[str], edges: list[tuple[str, str]]) -> list[str]:
    """Return node IDs in topological order (Kahn's algorithm)."""
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

    :param workflow: Validated Workflow instance.
    :param user_input: The initial user message (from CLI ``--input`` or ``workflow.input.message``).
    :returns: Execution trace dict with workflow metadata, per-node entries, and summary.
    """
    context = WorkflowContext(initial_state=workflow.state)

    node_ids = list(workflow.nodes.keys())
    edge_pairs = [(e.from_node, e.to_node) for e in workflow.edges]
    execution_order = _topological_sort(node_ids, edge_pairs)

    # Track the writes path of the most recently completed node for downstream input resolution
    last_writes_path: str | None = None

    trace_nodes: list[dict[str, Any]] = []
    total_tokens = 0
    total_duration_ms = 0.0
    status = "success"

    global_guardrail_names = workflow.guardrails

    for node_id in execution_order:
        node = workflow.nodes[node_id]
        agent_def = workflow.agents[node.agent]

        # Resolve this node's input
        if last_writes_path is None:
            node_input = user_input
        else:
            try:
                node_input = str(context.resolve(last_writes_path))
            except KeyError:
                node_input = user_input

        # Build guardrails: agent-level overrides global
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
            # Input guardrails
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

            # Output guardrails
            checked_output = response_text
            for g in guardrails:
                checked_output = g.check_output(checked_output)
                node_trace["guardrails_passed"].append(f"{type(g).__name__}.check_output")

            context.write(node.writes, checked_output)
            last_writes_path = node.writes

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
