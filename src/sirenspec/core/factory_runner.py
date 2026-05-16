"""Factory node execution: dynamic agent fan-out over a runtime list.

A factory node resolves its ``for_each:`` expression to a list at runtime, then
spawns one agent instance per item.  Instances run concurrently up to the
``concurrency`` limit.  Each instance receives its own ``{{ item }}`` and
``{{ index }}`` interpolation context when resolving ``inputs:`` values.

All instance outputs are collected and returned in the structured trace; the
executor writes them to the context as a list at the ``writes`` path.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from sirenspec.core.agent_runner import execute_agent_node
from sirenspec.core.interpolation import (
    InterpolationContext,
    build_interpolation_context,
    resolve_template,
    resolve_to_list,
)
from sirenspec.core.models import AgentDefinition, FactoryNode, RetryPolicy
from sirenspec.exceptions import FactoryNodeError

if TYPE_CHECKING:
    from sirenspec.core.models import Workflow


@dataclass
class FactoryInstanceResult:
    """Result of a single factory instance execution.

    :param index: Zero-based iteration index.
    :param item: The list item this instance processed (as a string).
    :param prompt_sent: The redacted prompt sent to the LLM.
    :param response_received: The LLM's response, or ``None`` if the instance failed.
    :param tokens: Token count for this instance's provider call.
    :param duration_ms: Wall-clock milliseconds for this instance.
    :param error: Error message string, or ``None`` on success.
    """

    index: int
    item: str
    prompt_sent: str
    response_received: str | None
    tokens: int
    duration_ms: float
    error: str | None = None


async def run_factory_instance(
    node_id: str,
    idx: int,
    item: Any,
    node: FactoryNode,
    agent_def: AgentDefinition,
    base_working: dict[str, Any],
    user_input: str,
    guardrail_names: list[str] | None,
) -> tuple[dict[str, Any], FactoryNodeError | None]:
    """Execute a single factory instance and return ``(trace_dict, error_or_None)``.

    Builds a per-instance :class:`~sirenspec.core.interpolation.InterpolationContext`
    with ``item`` and ``index`` populated, resolves ``inputs:`` templates, and calls
    :func:`~sirenspec.core.agent_runner.execute_agent_node` with a timeout.

    :param node_id: The factory node's workflow ID (used in error messages).
    :param idx: Zero-based instance index (populates ``{{ index }}``).
    :param item: The current list item (populates ``{{ item }}``).
    :param node: The validated FactoryNode definition.
    :param agent_def: The resolved AgentDefinition for this factory's agent.
    :param base_working: The executor's working dict snapshot.
    :param user_input: The original workflow input message.
    :param guardrail_names: Guardrail names to apply to this instance.
    :returns: Tuple of (instance trace dict, FactoryNodeError or None on success).
    """
    loop_ctx = build_interpolation_context(user_input, base_working, item=item, index=idx)

    resolved_inputs = {key: resolve_template(val, loop_ctx) for key, val in node.inputs.items()}
    prompt = "\n".join(f"{key}: {val}" for key, val in resolved_inputs.items())
    redacted_prompt = "\n".join(
        f"{key}: {resolve_template(val, loop_ctx, redact_env=True)}" for key, val in node.inputs.items()
    )

    resolved_system = resolve_template(agent_def.system, loop_ctx)

    instance_trace: dict[str, Any] = {
        "index": idx,
        "item": str(item),
        "prompt_sent": redacted_prompt,
        "response_received": None,
        "tokens": 0,
        "duration_ms": 0.0,
        "error": None,
    }

    start = time.monotonic()
    try:
        run_result = await asyncio.wait_for(
            execute_agent_node(
                node_id=f"{node_id}[{idx}]",
                model_uri=agent_def.model,
                system_prompt=resolved_system,
                user_input=prompt,
                guardrail_names=guardrail_names,
                retry_policy=RetryPolicy(max_attempts=1),
            ),
            timeout=float(node.timeout_per_instance),
        )
        duration_ms = (time.monotonic() - start) * 1000
        instance_trace.update(
            {
                "response_received": run_result.output,
                "tokens": run_result.tokens,
                "duration_ms": round(duration_ms, 2),
            }
        )
        return instance_trace, None

    except TimeoutError:
        duration_ms = (time.monotonic() - start) * 1000
        timeout_msg = f"Instance timed out after {node.timeout_per_instance}s"
        instance_trace["error"] = timeout_msg
        instance_trace["duration_ms"] = round(duration_ms, 2)
        return instance_trace, FactoryNodeError(node_id, idx, TimeoutError(timeout_msg))

    except Exception as exc:
        duration_ms = (time.monotonic() - start) * 1000
        instance_trace["error"] = str(exc)
        instance_trace["duration_ms"] = round(duration_ms, 2)
        return instance_trace, FactoryNodeError(node_id, idx, exc)


async def execute_factory_node(
    node_id: str,
    node: FactoryNode,
    workflow: Workflow,
    user_input: str,
    working: dict[str, Any],
    output: dict[str, Any],
    guardrail_names: list[str] | None,
) -> dict[str, Any]:
    """Execute a factory node: fan-out over a runtime list with concurrency control.

    Returns a structured trace dict with the shape::

        {
            "id": "<node_id>",
            "type": "factory",
            "instances": [
                {
                    "index": 0,
                    "item": "task text",
                    "prompt_sent": "task: task text",
                    "response_received": "...",  # None if failed and on_failure=continue
                    "tokens": 42,
                    "duration_ms": 123.4,
                    "error": None,  # or error string
                },
                ...
            ],
            "outputs": ["result1", "result2", ...],  # empty string for failed instances
            "tokens": 100,
            "duration_ms": 250.0,
            "error": None,
        }

    :param node_id: The node's ID as declared in the workflow.
    :param node: The validated FactoryNode.
    :param workflow: The parent workflow (used for agent lookup).
    :param user_input: Raw user input string.
    :param working: Current working context dict (read-only snapshot).
    :param output: Current output context dict (read-only snapshot).
    :param guardrail_names: Workflow-level guardrail names.
    :raises FactoryNodeError: If ``on_failure`` is ``'abort'`` and any instance fails.
    :returns: Structured trace dict for the factory node.
    """
    agent_def = workflow.agents[node.agent]
    agent_guardrail_names = agent_def.guardrails if agent_def.guardrails is not None else guardrail_names

    base_ctx = InterpolationContext(
        inputs={"message": user_input},
        nodes=working,
        env=dict(__import__("os").environ),
    )
    items = resolve_to_list(node.for_each, base_ctx)

    semaphore = asyncio.Semaphore(node.concurrency)

    async def run_with_semaphore(idx: int, item: Any) -> tuple[dict[str, Any], FactoryNodeError | None]:
        async with semaphore:
            return await run_factory_instance(
                node_id=node_id,
                idx=idx,
                item=item,
                node=node,
                agent_def=agent_def,
                base_working=working,
                user_input=user_input,
                guardrail_names=agent_guardrail_names,
            )

    tasks = [run_with_semaphore(idx, item) for idx, item in enumerate(items)]
    gathered: list[tuple[dict[str, Any], FactoryNodeError | None]] = await asyncio.gather(*tasks)

    instance_traces: list[dict[str, Any]] = []
    outputs: list[str] = []
    total_tokens = 0
    total_duration_ms = 0.0
    first_error: FactoryNodeError | None = None

    for instance_trace, exc in gathered:
        instance_traces.append(instance_trace)
        total_tokens += instance_trace["tokens"]
        total_duration_ms += instance_trace["duration_ms"]

        if exc is not None:
            outputs.append("")
            if node.on_failure == "abort" and first_error is None:
                first_error = exc
        else:
            outputs.append(instance_trace["response_received"] or "")

    if first_error is not None:
        raise first_error

    return {
        "id": node_id,
        "type": "factory",
        "instances": instance_traces,
        "outputs": outputs,
        "tokens": total_tokens,
        "duration_ms": round(total_duration_ms, 2),
        "error": None,
    }
