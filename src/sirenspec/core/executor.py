"""Async workflow executor: orchestrates nodes, context, edges, and trace assembly."""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from collections.abc import AsyncGenerator
from typing import Any

from sirenspec.core.agent_runner import execute_agent_node
from sirenspec.core.context import WorkflowContext
from sirenspec.core.events import NodeCompleteEvent, SummaryEvent
from sirenspec.core.factory_runner import execute_factory_node
from sirenspec.core.interpolation import InterpolationContext, build_interpolation_context, resolve_template
from sirenspec.core.models import (
    AgentNode,
    FactoryNode,
    HttpToolConfig,
    OnFailurePolicy,
    RetryPolicy,
    SwrmNode,
    ToolNode,
    Workflow,
)
from sirenspec.core.pricing import estimate_usd
from sirenspec.core.swrm_runner import execute_swrm
from sirenspec.core.tool_runner import execute_tool_node
from sirenspec.core.usage import TokenUsage
from sirenspec.exceptions import BudgetExceededError, RetryExhaustedError
from sirenspec.guardrails.base import GuardrailViolation, WorkflowGuardrail
from sirenspec.guardrails.registry import build_guardrails

logger = logging.getLogger(__name__)


def interpolate_tool_config(node: ToolNode, ctx: InterpolationContext) -> ToolNode:
    """Return a copy of *node* with template strings in its HTTP config resolved.

    Only :class:`~sirenspec.core.models.HttpToolConfig` fields are interpolated
    (``url``, ``headers``, ``body``). Python tool configs pass through unchanged.

    :param node: The tool node whose config may contain ``{{ expr }}`` placeholders.
    :param ctx: The interpolation context at the time the node executes.
    :returns: A new ToolNode with all resolvable placeholders replaced.
    """
    if not isinstance(node.config, HttpToolConfig):
        return node
    cfg = node.config
    interpolated_url = resolve_template(cfg.url, ctx)
    interpolated_headers = {k: resolve_template(v, ctx) for k, v in cfg.headers.items()} if cfg.headers else None
    interpolated_body = resolve_template(cfg.body, ctx) if cfg.body is not None else None
    new_config = HttpToolConfig(
        url=interpolated_url,
        method=cfg.method,
        headers=interpolated_headers,
        body=interpolated_body,
        timeout=cfg.timeout,
    )
    return ToolNode(
        type=node.type,
        tool=node.tool,
        config=new_config,
        output_key=node.output_key,
        retry=node.retry,
        on_failure=node.on_failure,
    )


class DotDict:
    """Wraps a plain dict so that attribute-style (dot) access works in ``when:`` expressions.

    Nested dicts are wrapped recursively, so a path like ``working.triage.intent``
    resolves correctly without any special parsing in the condition string.

    Only non-private attribute lookups are forwarded to the underlying dict;
    any ``AttributeError`` propagates naturally when a key is missing.

    We use object.__setattr__ and object.__getattribute__ to bypass any
    accidental ``__setattr__``/``__getattr__`` override that Python might add,
    keeping the internal ``_data`` reference isolated from user-defined keys.
    """

    def __init__(self, data: dict[str, Any]) -> None:
        # Store under a name that's very unlikely to collide with workflow keys.
        object.__setattr__(self, "_data", data)

    def __getattr__(self, key: str) -> Any:
        data: dict[str, Any] = object.__getattribute__(self, "_data")
        try:
            value = data[key]
        except KeyError as exc:
            raise AttributeError(key) from exc
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
        namespace: dict[str, Any] = {
            "working": DotDict(context.working),
            "output": DotDict(context.output),
            "true": True,
            "false": False,
            "null": None,
        }
        result = eval(condition, {"__builtins__": {}}, namespace)  # noqa: S307
        return bool(result)
    except (SyntaxError, TypeError, NameError, AttributeError, KeyError) as exc:
        logger.debug("when: condition %r evaluated with error: %s", condition, exc)
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


def extract_workflow_guardrails(names: list | None) -> list[WorkflowGuardrail]:
    """Return only the :class:`~sirenspec.guardrails.base.WorkflowGuardrail` instances from *names*.

    :param names: Raw guardrail name list from the workflow definition (may be ``None``).
    :returns: List of guardrails that implement the WorkflowGuardrail protocol.
    """
    all_guardrails = build_guardrails(names)
    return [g for g in all_guardrails if isinstance(g, WorkflowGuardrail)]


def check_workflow_budget(
    workflow_guardrails: list[WorkflowGuardrail],
    usage: TokenUsage,
    estimated_usd: float | None,
) -> None:
    """Run all workflow-level guardrails against the current accumulated spend.

    :param workflow_guardrails: Guardrail instances that implement ``check_budget``.
    :param usage: Accumulated token usage across all completed nodes.
    :param estimated_usd: Running USD estimate, or ``None`` for local models.
    :raises BudgetExceededError: If any workflow guardrail raises it.
    """
    for guardrail in workflow_guardrails:
        guardrail.check_budget(usage, estimated_usd)


def resolve_retry_policy(workflow: Workflow, node_id: str) -> RetryPolicy:
    """Return the effective :class:`~sirenspec.core.models.RetryPolicy` for *node_id*.

    Node-level policy takes precedence over workflow defaults.  If neither is
    set a policy with ``max_attempts=1`` (i.e. no retries) is returned.

    :param workflow: The workflow containing node and default definitions.
    :param node_id: The identifier of the node being resolved.
    :returns: The effective RetryPolicy for the node.
    """
    node = workflow.nodes[node_id]
    if node.retry is not None:
        return node.retry
    if workflow.defaults and workflow.defaults.retry is not None:
        return workflow.defaults.retry
    return RetryPolicy()


def resolve_on_failure_policy(workflow: Workflow, node_id: str) -> OnFailurePolicy:
    """Return the effective :class:`~sirenspec.core.models.OnFailurePolicy` for *node_id*.

    Node-level policy takes precedence over workflow defaults.  If neither is
    set a policy with ``action='abort'`` is returned.

    :param workflow: The workflow containing node and default definitions.
    :param node_id: The identifier of the node being resolved.
    :returns: The effective OnFailurePolicy for the node.
    """
    node = workflow.nodes[node_id]
    if node.on_failure is not None:
        return node.on_failure
    if workflow.defaults and workflow.defaults.on_failure is not None:
        return workflow.defaults.on_failure
    return OnFailurePolicy()


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
    5. Agent nodes delegate to :func:`~sirenspec.core.agent_runner.execute_agent_node`;
       tool nodes to :func:`~sirenspec.core.tool_runner.execute_tool_node`;
       swrm nodes to :func:`~sirenspec.core.swrm_runner.execute_swrm`.

    :param workflow: Validated :class:`~sirenspec.core.models.Workflow` instance.
    :param user_input: The initial user message.
    :returns: Execution trace dict with workflow metadata, per-node entries, and summary.
    """
    context = WorkflowContext(initial_state=workflow.state)
    node_ids = list(workflow.nodes.keys())

    # Build two graph structures: out_edges for edge traversal after each node
    # completes, and in_degree for identifying root nodes (those with no predecessors).
    out_edges: dict[str, list[tuple[str, str | None]]] = {n: [] for n in node_ids}
    in_degree: dict[str, int] = dict.fromkeys(node_ids, 0)
    for edge in workflow.edges:
        out_edges[edge.from_node].append((edge.to_node, edge.when))
        in_degree[edge.to_node] += 1

    edge_pairs = [(e.from_node, e.to_node) for e in workflow.edges]
    # topological_sort detects cycles up-front and gives a stable node order
    # that guarantees every node runs after all its predecessors.
    execution_order = topological_sort(node_ids, edge_pairs)

    # Only root nodes (no incoming edges) are active at the start.
    # Child nodes become active when their predecessors write to active_nodes.
    active_nodes: set[str] = {n for n in node_ids if in_degree[n] == 0}

    # Tracks the dotted path of the most recent agent output so each subsequent
    # agent in a linear chain automatically receives the prior agent's response.
    # None means "use the original user_input" (no agent has run yet).
    last_writes_path: str | None = None

    trace_nodes: list[dict[str, Any]] = []
    total_usage = TokenUsage(prompt_tokens=0, completion_tokens=0)
    total_estimated_usd: float | None = None
    total_duration_ms = 0.0
    status = "success"

    global_guardrail_names = workflow.guardrails
    workflow_guardrails = extract_workflow_guardrails(global_guardrail_names)

    # Fallback nodes are not reachable via normal edges — they are activated
    # explicitly when a node's on_failure policy is 'fallback'. We track them
    # here and inject them into active_nodes when the execution order reaches them.
    pending_fallback_nodes: set[str] = set()

    for node_id in execution_order:
        # Inject fallback nodes into the active set when the topological order
        # reaches them. This lets fallback routing work without adding edges
        # to the graph (which would change reachability for topological_sort).
        if node_id in pending_fallback_nodes:
            active_nodes.add(node_id)
            pending_fallback_nodes.discard(node_id)

        # Nodes not in active_nodes were never activated by their predecessors
        # (e.g. the other branch of a conditional fork). Skip them silently.
        if node_id not in active_nodes:
            continue

        node = workflow.nodes[node_id]

        # ------------------------------------------------------------------ #
        # Swrm node — parallel agent fan-out with optional synthesis.         #
        # ------------------------------------------------------------------ #
        if isinstance(node, SwrmNode):
            start_time = time.monotonic()
            try:
                swrm_trace = await execute_swrm(
                    node_id=node_id,
                    node=node,
                    user_input=user_input,
                    working=context.working,
                    output=context.output,
                    global_guardrail_names=global_guardrail_names,
                )
            except Exception as exc:
                duration_ms = (time.monotonic() - start_time) * 1000
                error_trace: dict[str, Any] = {
                    "id": node_id,
                    "type": "swrm",
                    "agents": [],
                    "synthesis": None,
                    "output": None,
                    "tokens": 0,
                    "duration_ms": round(duration_ms, 2),
                    "error": str(exc),
                }
                total_duration_ms += duration_ms
                status = "failed"
                trace_nodes.append(error_trace)
                break

            for agent_trace in swrm_trace["agents"]:
                if agent_trace["response_received"] is not None:
                    context.write(
                        f"working.{node_id}.agents.{agent_trace['id']}.output",
                        agent_trace["response_received"],
                    )
            context.write(f"output.{node_id}", swrm_trace["output"])
            context.write(f"working.{node_id}.output", swrm_trace["output"])
            last_writes_path = f"output.{node_id}"

            total_usage += TokenUsage(prompt_tokens=0, completion_tokens=swrm_trace["tokens"])
            total_duration_ms += swrm_trace["duration_ms"]

            context.write("working._budget.total_tokens", total_usage.total)
            context.write("working._budget.estimated_usd", total_estimated_usd)

            try:
                check_workflow_budget(workflow_guardrails, total_usage, total_estimated_usd)
            except BudgetExceededError as exc:
                status = "failed"
                swrm_trace["error"] = str(exc)
                trace_nodes.append(swrm_trace)
                break

            for target, condition in out_edges[node_id]:
                if condition is None or evaluate_when_condition(condition, context):
                    active_nodes.add(target)

            trace_nodes.append(swrm_trace)
            continue

        # ------------------------------------------------------------------ #
        # Tool node — adapter dispatch with simple retry.                     #
        # ------------------------------------------------------------------ #
        if isinstance(node, ToolNode):
            tool_node_trace: dict[str, Any] = {
                "id": node_id,
                "type": "tool",
                "tool": node.tool,
                "output_key": node.output_key,
                "result": None,
                "duration_ms": 0,
                "error": None,
            }
            start_time = time.monotonic()
            try:
                tool_interp_ctx = build_interpolation_context(user_input, context.working)
                interpolated_node = interpolate_tool_config(node, tool_interp_ctx)
                run_result = await execute_tool_node(node_id, interpolated_node)
                context.write(f"working.{node_id}.{node.output_key}", run_result.result)
                tool_node_trace["result"] = run_result.result
                tool_node_trace["duration_ms"] = round(run_result.duration_ms, 2)
                total_duration_ms += run_result.duration_ms

                for target, condition in out_edges[node_id]:
                    if condition is None or evaluate_when_condition(condition, context):
                        active_nodes.add(target)

            except Exception as exc:
                duration_ms = (time.monotonic() - start_time) * 1000
                tool_node_trace["error"] = str(exc)
                tool_node_trace["duration_ms"] = round(duration_ms, 2)
                total_duration_ms += duration_ms
                status = "failed"
                trace_nodes.append(tool_node_trace)
                break

            trace_nodes.append(tool_node_trace)
            continue

        # ------------------------------------------------------------------ #
        # Factory node — dynamic agent fan-out over a runtime list.          #
        # ------------------------------------------------------------------ #
        if isinstance(node, FactoryNode):
            start_time = time.monotonic()
            try:
                factory_trace = await execute_factory_node(
                    node_id=node_id,
                    node=node,
                    workflow=workflow,
                    user_input=user_input,
                    working=context.working,
                    output=context.output,
                    guardrail_names=global_guardrail_names,
                )
            except Exception as exc:
                duration_ms = (time.monotonic() - start_time) * 1000
                error_trace: dict[str, Any] = {
                    "id": node_id,
                    "type": "factory",
                    "instances": [],
                    "outputs": [],
                    "tokens": 0,
                    "duration_ms": round(duration_ms, 2),
                    "error": str(exc),
                }
                total_duration_ms += duration_ms
                status = "failed"
                trace_nodes.append(error_trace)
                break

            outputs_list = factory_trace["outputs"]
            context.write(node.writes, outputs_list)
            context.write(f"working.{node_id}.outputs", outputs_list)
            context.write(f"working.{node_id}.output", "\n".join(s for s in outputs_list if s))
            last_writes_path = node.writes

            total_usage += TokenUsage(prompt_tokens=0, completion_tokens=factory_trace["tokens"])
            total_duration_ms += factory_trace["duration_ms"]

            context.write("working._budget.total_tokens", total_usage.total)
            context.write("working._budget.estimated_usd", total_estimated_usd)

            try:
                check_workflow_budget(workflow_guardrails, total_usage, total_estimated_usd)
            except BudgetExceededError as exc:
                status = "failed"
                factory_trace["error"] = str(exc)
                trace_nodes.append(factory_trace)
                break

            for target, condition in out_edges[node_id]:
                if condition is None or evaluate_when_condition(condition, context):
                    active_nodes.add(target)

            trace_nodes.append(factory_trace)
            continue

        # ------------------------------------------------------------------ #
        # Agent node — guarded LLM call with retry and on-failure routing.   #
        # ------------------------------------------------------------------ #
        assert isinstance(node, AgentNode)
        agent_def = workflow.agents[node.agent]

        if last_writes_path is None:
            # No prior agent has run yet — use the original workflow input.
            node_input = user_input
        else:
            try:
                # Chain style: each agent receives the output of the previous agent.
                # Falls back to user_input if the path doesn't resolve (e.g. a
                # skipped node that never wrote its output key).
                node_input = str(context.resolve(last_writes_path))
            except KeyError:
                node_input = user_input

        # Agent-level guardrails override workflow-level guardrails when set.
        # Passing None (not []) is intentional — build_guardrails(None) applies
        # the default injection guardrail, while build_guardrails([]) disables all.
        guardrail_names = agent_def.guardrails if agent_def.guardrails is not None else global_guardrail_names
        retry_policy = resolve_retry_policy(workflow, node_id)
        on_failure_policy = resolve_on_failure_policy(workflow, node_id)

        # Resolve the system prompt with the current context so {{ inputs.* }} and
        # {{ node_id.output }} expressions in system prompts are evaluated at runtime.
        interp_ctx = build_interpolation_context(user_input, context.working)
        resolved_system = resolve_template(agent_def.system, interp_ctx)
        redacted_system = resolve_template(agent_def.system, interp_ctx, redact_env=True)

        agent_node_trace: dict[str, Any] = {
            "id": node_id,
            "agent": node.agent,
            "system_prompt": redacted_system,
            "prompt_sent": node_input,
            "response_received": None,
            "writes": node.writes,
            "guardrails_passed": [],
            "tokens": 0,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "estimated_usd": None},
            "duration_ms": 0,
            "error": None,
            "retry_attempts": [],
        }

        start_time = time.monotonic()
        try:
            run_result = await execute_agent_node(
                node_id=node_id,
                model_uri=agent_def.model,
                system_prompt=resolved_system,
                user_input=node_input,
                guardrail_names=guardrail_names,
                retry_policy=retry_policy,
            )

            context.write(node.writes, run_result.output)
            context.write(f"working.{node_id}.output", run_result.output)
            last_writes_path = node.writes

            for target, condition in out_edges[node_id]:
                if condition is None or evaluate_when_condition(condition, context):
                    active_nodes.add(target)

            duration_ms = (time.monotonic() - start_time) * 1000
            node_estimated_usd = estimate_usd(
                run_result.token_usage.prompt_tokens,
                run_result.token_usage.completion_tokens,
                agent_def.model,
            )
            agent_node_trace.update(
                {
                    "response_received": run_result.output,
                    "tokens": run_result.token_usage.total,
                    "usage": {
                        "prompt_tokens": run_result.token_usage.prompt_tokens,
                        "completion_tokens": run_result.token_usage.completion_tokens,
                        "estimated_usd": node_estimated_usd,
                    },
                    "duration_ms": round(duration_ms, 2),
                    "guardrails_passed": run_result.guardrails_passed,
                    "retry_attempts": run_result.retry_attempts,
                }
            )
            total_usage += run_result.token_usage
            if node_estimated_usd is not None:
                total_estimated_usd = (total_estimated_usd or 0.0) + node_estimated_usd
            total_duration_ms += duration_ms

            context.write("working._budget.total_tokens", total_usage.total)
            context.write("working._budget.estimated_usd", total_estimated_usd)

            try:
                check_workflow_budget(workflow_guardrails, total_usage, total_estimated_usd)
            except BudgetExceededError as exc:
                agent_node_trace["error"] = str(exc)
                status = "failed"
                trace_nodes.append(agent_node_trace)
                break

        except GuardrailViolation as exc:
            duration_ms = (time.monotonic() - start_time) * 1000
            agent_node_trace["error"] = f"GuardrailViolation: {exc.reason}"
            agent_node_trace["duration_ms"] = round(duration_ms, 2)
            total_duration_ms += duration_ms
            status = "failed"
            trace_nodes.append(agent_node_trace)
            break

        except RetryExhaustedError as exc:
            duration_ms = (time.monotonic() - start_time) * 1000
            agent_node_trace["error"] = str(exc)
            agent_node_trace["duration_ms"] = round(duration_ms, 2)
            total_duration_ms += duration_ms

            action = on_failure_policy.action

            if action == "abort":
                status = "failed"
                trace_nodes.append(agent_node_trace)
                break

            elif action == "fallback":
                agent_node_trace["on_failure_action"] = "fallback"
                agent_node_trace["fallback_node"] = on_failure_policy.fallback_node
                trace_nodes.append(agent_node_trace)
                if on_failure_policy.fallback_node:
                    pending_fallback_nodes.add(on_failure_policy.fallback_node)
                continue

            elif action == "skip":
                agent_node_trace["on_failure_action"] = "skip"
                trace_nodes.append(agent_node_trace)
                for target, condition in out_edges[node_id]:
                    if condition is None or evaluate_when_condition(condition, context):
                        active_nodes.add(target)
                continue

            elif action == "use_default":
                default_val = on_failure_policy.default_output or ""
                context.write(node.writes, default_val)
                last_writes_path = node.writes
                agent_node_trace["on_failure_action"] = "use_default"
                agent_node_trace["response_received"] = default_val
                trace_nodes.append(agent_node_trace)
                for target, condition in out_edges[node_id]:
                    if condition is None or evaluate_when_condition(condition, context):
                        active_nodes.add(target)
                continue

        except Exception as exc:
            duration_ms = (time.monotonic() - start_time) * 1000
            agent_node_trace["error"] = str(exc)
            agent_node_trace["duration_ms"] = round(duration_ms, 2)
            total_duration_ms += duration_ms
            status = "failed"
            trace_nodes.append(agent_node_trace)
            break

        trace_nodes.append(agent_node_trace)

    return {
        "workflow": {"version": workflow.version},
        "input": {"message": user_input},
        "nodes": trace_nodes,
        "output": context.output,
        "summary": {
            "total_tokens": total_usage.total,
            "total_usage": {
                "prompt_tokens": total_usage.prompt_tokens,
                "completion_tokens": total_usage.completion_tokens,
                "total_tokens": total_usage.total,
                "estimated_usd": total_estimated_usd,
            },
            "total_duration_ms": round(total_duration_ms, 2),
            "status": status,
        },
    }


async def execute_streaming(workflow: Workflow, user_input: str) -> AsyncGenerator[NodeCompleteEvent | SummaryEvent]:
    """Execute a workflow and yield typed events as each node completes.

    This is the streaming counterpart to :func:`execute`.  It walks the same
    node/edge graph with identical semantics — topological ordering, conditional
    branching, retry and on-failure policies — but instead of collecting all
    results and returning them at once, it yields a :class:`~sirenspec.core.events.NodeCompleteEvent`
    immediately after each node finishes.  Nodes that are never activated yield a
    ``NodeCompleteEvent`` with ``status="skipped"``.  At the very end a single
    :class:`~sirenspec.core.events.SummaryEvent` is yielded.

    :param workflow: Validated :class:`~sirenspec.core.models.Workflow` instance.
    :param user_input: The initial user message.
    :returns: An async generator of ``NodeCompleteEvent`` (one per node) followed
        by a final ``SummaryEvent``.
    """
    start_wall = time.monotonic()
    context = WorkflowContext(initial_state=workflow.state)
    node_ids = list(workflow.nodes.keys())

    out_edges: dict[str, list[tuple[str, str | None]]] = {n: [] for n in node_ids}
    in_degree: dict[str, int] = dict.fromkeys(node_ids, 0)
    for edge in workflow.edges:
        out_edges[edge.from_node].append((edge.to_node, edge.when))
        in_degree[edge.to_node] += 1

    edge_pairs = [(e.from_node, e.to_node) for e in workflow.edges]
    execution_order = topological_sort(node_ids, edge_pairs)

    active_nodes: set[str] = {n for n in node_ids if in_degree[n] == 0}
    last_writes_path: str | None = None

    total_usage = TokenUsage(prompt_tokens=0, completion_tokens=0)
    total_estimated_usd: float | None = None
    total_duration_ms = 0.0
    status = "success"
    active_node_count = 0

    global_guardrail_names = workflow.guardrails
    workflow_guardrails = extract_workflow_guardrails(global_guardrail_names)
    pending_fallback_nodes: set[str] = set()

    for node_id in execution_order:
        if node_id in pending_fallback_nodes:
            active_nodes.add(node_id)
            pending_fallback_nodes.discard(node_id)

        if node_id not in active_nodes:
            yield NodeCompleteEvent(
                node_id=node_id,
                node_type=type(workflow.nodes[node_id]).__name__.lower().removesuffix("node"),
                status="skipped",
            )
            continue

        active_node_count += 1
        node = workflow.nodes[node_id]

        if isinstance(node, SwrmNode):
            node_start = time.monotonic()
            try:
                swrm_trace = await execute_swrm(
                    node_id=node_id,
                    node=node,
                    user_input=user_input,
                    working=context.working,
                    output=context.output,
                    global_guardrail_names=global_guardrail_names,
                )
            except Exception as exc:
                duration_ms = (time.monotonic() - node_start) * 1000
                total_duration_ms += duration_ms
                status = "failed"
                yield NodeCompleteEvent(
                    node_id=node_id,
                    node_type="swrm",
                    status="failed",
                    error=str(exc),
                )
                break

            for agent_trace in swrm_trace["agents"]:
                if agent_trace["response_received"] is not None:
                    context.write(
                        f"working.{node_id}.agents.{agent_trace['id']}.output",
                        agent_trace["response_received"],
                    )
            context.write(f"output.{node_id}", swrm_trace["output"])
            context.write(f"working.{node_id}.output", swrm_trace["output"])
            last_writes_path = f"output.{node_id}"

            total_usage += TokenUsage(prompt_tokens=0, completion_tokens=swrm_trace["tokens"])
            total_duration_ms += swrm_trace["duration_ms"]

            context.write("working._budget.total_tokens", total_usage.total)
            context.write("working._budget.estimated_usd", total_estimated_usd)

            try:
                check_workflow_budget(workflow_guardrails, total_usage, total_estimated_usd)
            except BudgetExceededError as exc:
                status = "failed"
                yield NodeCompleteEvent(
                    node_id=node_id,
                    node_type="swrm",
                    status="failed",
                    error=str(exc),
                )
                break

            for target, condition in out_edges[node_id]:
                if condition is None or evaluate_when_condition(condition, context):
                    active_nodes.add(target)

            yield NodeCompleteEvent(
                node_id=node_id,
                node_type="swrm",
                output=swrm_trace["output"],
                writes=f"output.{node_id}",
                status="success",
                tokens=swrm_trace["tokens"],
            )
            continue

        if isinstance(node, ToolNode):
            node_start = time.monotonic()
            try:
                tool_interp_ctx = build_interpolation_context(user_input, context.working)
                interpolated_node = interpolate_tool_config(node, tool_interp_ctx)
                run_result = await execute_tool_node(node_id, interpolated_node)
                context.write(f"working.{node_id}.{node.output_key}", run_result.result)
                total_duration_ms += run_result.duration_ms

                for target, condition in out_edges[node_id]:
                    if condition is None or evaluate_when_condition(condition, context):
                        active_nodes.add(target)

                yield NodeCompleteEvent(
                    node_id=node_id,
                    node_type="tool",
                    output=run_result.result,
                    writes=f"working.{node_id}.{node.output_key}",
                    status="success",
                )

            except Exception as exc:
                duration_ms = (time.monotonic() - node_start) * 1000
                total_duration_ms += duration_ms
                status = "failed"
                yield NodeCompleteEvent(
                    node_id=node_id,
                    node_type="tool",
                    status="failed",
                    error=str(exc),
                )
                break
            continue

        if isinstance(node, FactoryNode):
            node_start = time.monotonic()
            try:
                factory_trace = await execute_factory_node(
                    node_id=node_id,
                    node=node,
                    workflow=workflow,
                    user_input=user_input,
                    working=context.working,
                    output=context.output,
                    guardrail_names=global_guardrail_names,
                )
            except Exception as exc:
                duration_ms = (time.monotonic() - node_start) * 1000
                total_duration_ms += duration_ms
                status = "failed"
                yield NodeCompleteEvent(
                    node_id=node_id,
                    node_type="factory",
                    status="failed",
                    error=str(exc),
                )
                break

            outputs_list = factory_trace["outputs"]
            context.write(node.writes, outputs_list)
            context.write(f"working.{node_id}.outputs", outputs_list)
            context.write(f"working.{node_id}.output", "\n".join(s for s in outputs_list if s))
            last_writes_path = node.writes

            total_usage += TokenUsage(prompt_tokens=0, completion_tokens=factory_trace["tokens"])
            total_duration_ms += factory_trace["duration_ms"]

            context.write("working._budget.total_tokens", total_usage.total)
            context.write("working._budget.estimated_usd", total_estimated_usd)

            try:
                check_workflow_budget(workflow_guardrails, total_usage, total_estimated_usd)
            except BudgetExceededError as exc:
                status = "failed"
                yield NodeCompleteEvent(
                    node_id=node_id,
                    node_type="factory",
                    status="failed",
                    error=str(exc),
                )
                break

            for target, condition in out_edges[node_id]:
                if condition is None or evaluate_when_condition(condition, context):
                    active_nodes.add(target)

            yield NodeCompleteEvent(
                node_id=node_id,
                node_type="factory",
                output=outputs_list,
                writes=node.writes,
                status="success",
                tokens=factory_trace["tokens"],
            )
            continue

        if not isinstance(node, AgentNode):
            raise TypeError(f"Unhandled node type: {type(node).__name__}")
        agent_def = workflow.agents[node.agent]

        if last_writes_path is None:
            node_input = user_input
        else:
            try:
                node_input = str(context.resolve(last_writes_path))
            except KeyError:
                node_input = user_input

        guardrail_names = agent_def.guardrails if agent_def.guardrails is not None else global_guardrail_names
        retry_policy = resolve_retry_policy(workflow, node_id)
        on_failure_policy = resolve_on_failure_policy(workflow, node_id)

        interp_ctx = build_interpolation_context(user_input, context.working)
        resolved_system = resolve_template(agent_def.system, interp_ctx)

        node_start = time.monotonic()
        try:
            run_result = await execute_agent_node(
                node_id=node_id,
                model_uri=agent_def.model,
                system_prompt=resolved_system,
                user_input=node_input,
                guardrail_names=guardrail_names,
                retry_policy=retry_policy,
            )

            context.write(node.writes, run_result.output)
            context.write(f"working.{node_id}.output", run_result.output)
            last_writes_path = node.writes

            for target, condition in out_edges[node_id]:
                if condition is None or evaluate_when_condition(condition, context):
                    active_nodes.add(target)

            duration_ms = (time.monotonic() - node_start) * 1000
            total_usage = total_usage + run_result.token_usage
            node_estimated_usd = estimate_usd(
                run_result.token_usage.prompt_tokens,
                run_result.token_usage.completion_tokens,
                agent_def.model,
            )
            if node_estimated_usd is not None:
                total_estimated_usd = (total_estimated_usd or 0.0) + node_estimated_usd
            total_duration_ms += duration_ms

            context.write("working._budget.total_tokens", total_usage.total)
            context.write("working._budget.estimated_usd", total_estimated_usd)

            try:
                check_workflow_budget(workflow_guardrails, total_usage, total_estimated_usd)
            except BudgetExceededError as exc:
                status = "failed"
                yield NodeCompleteEvent(
                    node_id=node_id,
                    node_type="agent",
                    status="failed",
                    error=str(exc),
                )
                break

            yield NodeCompleteEvent(
                node_id=node_id,
                node_type="agent",
                output=run_result.output,
                writes=node.writes,
                status="success",
                tokens=run_result.token_usage.total,
            )

        except GuardrailViolation as exc:
            duration_ms = (time.monotonic() - node_start) * 1000
            total_duration_ms += duration_ms
            status = "failed"
            yield NodeCompleteEvent(
                node_id=node_id,
                node_type="agent",
                status="failed",
                error=f"GuardrailViolation: {exc.reason}",
            )
            break

        except RetryExhaustedError as exc:
            duration_ms = (time.monotonic() - node_start) * 1000
            total_duration_ms += duration_ms
            action = on_failure_policy.action

            if action == "abort":
                status = "failed"
                yield NodeCompleteEvent(
                    node_id=node_id,
                    node_type="agent",
                    status="failed",
                    error=str(exc),
                )
                break

            elif action == "fallback":
                yield NodeCompleteEvent(
                    node_id=node_id,
                    node_type="agent",
                    status="failed",
                    error=str(exc),
                    writes=node.writes,
                )
                if on_failure_policy.fallback_node:
                    pending_fallback_nodes.add(on_failure_policy.fallback_node)
                continue

            elif action == "skip":
                yield NodeCompleteEvent(
                    node_id=node_id,
                    node_type="agent",
                    status="skipped",
                    error=str(exc),
                )
                for target, condition in out_edges[node_id]:
                    if condition is None or evaluate_when_condition(condition, context):
                        active_nodes.add(target)
                continue

            elif action == "use_default":
                default_val = on_failure_policy.default_output or ""
                context.write(node.writes, default_val)
                last_writes_path = node.writes
                for target, condition in out_edges[node_id]:
                    if condition is None or evaluate_when_condition(condition, context):
                        active_nodes.add(target)
                yield NodeCompleteEvent(
                    node_id=node_id,
                    node_type="agent",
                    output=default_val,
                    writes=node.writes,
                    status="success",
                )
                continue

        except Exception as exc:
            duration_ms = (time.monotonic() - node_start) * 1000
            total_duration_ms += duration_ms
            status = "failed"
            yield NodeCompleteEvent(
                node_id=node_id,
                node_type="agent",
                status="failed",
                error=str(exc),
            )
            break

    yield SummaryEvent(
        total_nodes=active_node_count,
        total_tokens=total_usage.total,
        status=status,
        duration_ms=round((time.monotonic() - start_wall) * 1000, 2),
    )
