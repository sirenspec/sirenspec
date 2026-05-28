"""Factory node execution: dynamic agent fan-out in list, swarm_size, or swrm mode.

**Agent + for_each** — resolves ``for_each:`` to a list at runtime and spawns one agent
instance per item.  Each instance receives ``{{ item }}`` and ``{{ index }}``.

**Agent + swarm_size** — resolves ``swarm_size:`` to an integer at runtime and spawns
exactly that many identical agent instances.  Each instance receives ``{{ index }}``
and ``{{ total }}``; there is no ``{{ item }}``.

**Swrm + for_each** — resolves ``for_each:`` to a list at runtime and spawns one full
swrm (set of parallel specialist agents with optional synthesis) per item.  Agent
prompts and the synthesis prompt all receive ``{{ item }}``, ``{{ index }}``, and
``{{ total }}``.

Instances in all modes run concurrently up to the ``concurrency`` limit.  All
instance outputs are collected and returned in the structured trace; the executor
writes them to the context as a list at the ``writes`` path.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from typing import Any

from sirenspec.core.agent_runner import execute_agent_node
from sirenspec.core.interpolation import (
    InterpolationContext,
    build_interpolation_context,
    resolve_template,
    resolve_to_list,
)
from sirenspec.core.models import AgentDefinition, FactoryNode, RetryPolicy, Workflow
from sirenspec.core.swrm_runner import execute_swrm_for_item
from sirenspec.exceptions import FactoryNodeError, GuardrailError, InterpolationError

MAX_SWARM_SIZE: int = 100
"""Maximum number of agents that a swarm factory node may spawn in a single execution."""


def resolve_swarm_size(node: FactoryNode, interp_ctx: InterpolationContext) -> int:
    """Resolve the swarm_size field to a concrete integer.

    :param node: The FactoryNode with swarm_size set.
    :param interp_ctx: Current interpolation context for template resolution.
    :raises InterpolationError: If the resolved string cannot be cast to an integer.
    :raises GuardrailError: If the resolved size is < 1 or > :data:`MAX_SWARM_SIZE`.
    :returns: Concrete swarm size as a positive integer.
    """
    raw = node.swarm_size
    if isinstance(raw, int):
        size = raw
    else:
        resolved = resolve_template(raw, interp_ctx)  # type: ignore[arg-type]
        try:
            size = int(resolved)
        except ValueError as exc:
            raise InterpolationError(
                raw,
                "swarm_size",
                f"Template resolved to non-integer value: '{resolved}'",  # type: ignore[arg-type]
            ) from exc
    if size < 1:
        raise GuardrailError(f"swarm_size resolved to {size}; must be >= 1")
    if size > MAX_SWARM_SIZE:
        raise GuardrailError(f"swarm_size resolved to {size}; exceeds MAX_SWARM_SIZE={MAX_SWARM_SIZE}")
    return size


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
    total: int | None = None,
) -> tuple[dict[str, Any], FactoryNodeError | None]:
    """Execute a single factory instance and return ``(trace_dict, error_or_None)``.

    Builds a per-instance :class:`~sirenspec.core.interpolation.InterpolationContext`
    with ``item``, ``index``, and ``total`` populated, resolves ``inputs:`` templates,
    and exposes the resolved input values in the ``inputs`` namespace so that
    ``{{ inputs.key }}`` expressions in the agent's system prompt resolve correctly.
    Then calls :func:`~sirenspec.core.agent_runner.execute_agent_node` with a timeout.

    :param node_id: The factory node's workflow ID (used in error messages).
    :param idx: Zero-based instance index (populates ``{{ index }}``).
    :param item: The current list item (populates ``{{ item }}``). ``None`` in swarm mode.
    :param node: The validated FactoryNode definition.
    :param agent_def: The resolved AgentDefinition for this factory's agent.
    :param base_working: The executor's working dict snapshot.
    :param user_input: The original workflow input message.
    :param guardrail_names: Guardrail names to apply to this instance.
    :param total: Total number of instances in this execution; populates ``{{ total }}``.
    :returns: Tuple of (instance trace dict, FactoryNodeError or None on success).
    """
    loop_ctx = build_interpolation_context(user_input, base_working, item=item, index=idx, total=total)

    resolved_inputs = {key: resolve_template(val, loop_ctx) for key, val in node.inputs.items()}

    # Makes {{ inputs.key }} resolvable in the agent system prompt.
    loop_ctx.inputs.update(resolved_inputs)

    prompt = "\n".join(f"{key}: {val}" for key, val in resolved_inputs.items())
    redacted_prompt = "\n".join(
        f"{key}: {resolve_template(val, loop_ctx, redact_env=True)}" for key, val in node.inputs.items()
    )

    resolved_system = resolve_template(agent_def.system, loop_ctx)

    instance_trace: dict[str, Any] = {
        "index": idx,
        "total": total,
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
                "tokens": run_result.token_usage.total,
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


async def run_swrm_factory_instance(
    node_id: str,
    idx: int,
    item: Any,
    node: FactoryNode,
    base_working: dict[str, Any],
    user_input: str,
    guardrail_names: list[str] | None,
    total: int,
) -> tuple[dict[str, Any], FactoryNodeError | None]:
    """Execute one swrm for a single factory item and return ``(trace_dict, error_or_None)``.

    Builds a per-item :class:`~sirenspec.core.interpolation.InterpolationContext` with
    ``item``, ``index``, and ``total`` populated, then delegates to
    :func:`~sirenspec.core.swrm_runner.execute_swrm_for_item`.  The entire per-item swrm
    execution is wrapped in a timeout governed by ``node.timeout_per_instance``.

    :param node_id: The factory node's workflow ID.
    :param idx: Zero-based item index (populates ``{{ index }}``).
    :param item: The current list item (populates ``{{ item }}``).
    :param node: The validated FactoryNode with ``swrm`` set.
    :param base_working: The executor's working dict snapshot.
    :param user_input: The original workflow input message.
    :param guardrail_names: Workflow-level guardrail names.
    :param total: Total number of items in this execution (populates ``{{ total }}``).
    :returns: Tuple of (instance trace dict, FactoryNodeError or None on success).
    """
    interp_ctx = build_interpolation_context(user_input, base_working, item=item, index=idx, total=total)

    instance_trace: dict[str, Any] = {
        "index": idx,
        "total": total,
        "item": str(item),
        "swrm": None,
        "tokens": 0,
        "duration_ms": 0.0,
        "error": None,
    }

    start = time.monotonic()
    try:
        swrm_trace, swrm_error = await asyncio.wait_for(
            execute_swrm_for_item(node.swrm, interp_ctx, node_id, base_working, guardrail_names),  # type: ignore[arg-type]
            timeout=float(node.timeout_per_instance),
        )
        duration_ms = (time.monotonic() - start) * 1000
        instance_trace.update(
            {
                "swrm": swrm_trace,
                "tokens": swrm_trace["tokens"],
                "duration_ms": round(duration_ms, 2),
                "error": swrm_trace.get("error"),
            }
        )
        if swrm_error is not None:
            return instance_trace, FactoryNodeError(node_id, idx, swrm_error)
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
    base_ctx = InterpolationContext(
        inputs={"message": user_input},
        nodes=working,
        env=dict(os.environ),
    )

    if node.swrm is not None:
        items: list[Any] = resolve_to_list(node.for_each, base_ctx)  # type: ignore[arg-type]
        total = len(items)
        semaphore = asyncio.Semaphore(node.concurrency)

        async def run_swrm_with_semaphore(idx: int, item: Any) -> tuple[dict[str, Any], FactoryNodeError | None]:
            async with semaphore:
                return await run_swrm_factory_instance(
                    node_id=node_id,
                    idx=idx,
                    item=item,
                    node=node,
                    base_working=working,
                    user_input=user_input,
                    guardrail_names=guardrail_names,
                    total=total,
                )

        tasks = [run_swrm_with_semaphore(idx, item) for idx, item in enumerate(items)]
    else:
        agent_def = workflow.agents[node.agent]  # type: ignore[index]
        agent_guardrail_names = agent_def.guardrails if agent_def.guardrails is not None else guardrail_names

        if node.for_each is not None:
            items = resolve_to_list(node.for_each, base_ctx)
            total = len(items)
        else:
            total = resolve_swarm_size(node, base_ctx)
            items = [None] * total

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
                    total=total,
                )

        tasks = [run_with_semaphore(idx, item) for idx, item in enumerate(items)]

    gathered: list[tuple[dict[str, Any], FactoryNodeError | None]] = await asyncio.gather(*tasks)

    instance_traces: list[dict[str, Any]] = []
    outputs: list[Any] = []
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
        elif node.swrm is not None:
            outputs.append(instance_trace["swrm"]["output"])
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
