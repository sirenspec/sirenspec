"""Async workflow executor: orchestrates nodes, guardrails, providers, and trace."""

from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Any

from sirenspec.core.context import WorkflowContext
from sirenspec.core.models import AgentNode, HttpToolConfig, OnFailurePolicy, PythonToolConfig, RetryPolicy, SwrmNode, ToolNode, Workflow
from sirenspec.core.retry import run_with_retry
from sirenspec.core.swrm import execute_swrm
from sirenspec.exceptions import RetryExhaustedError, SwrmAgentError, ToolError
from sirenspec.guardrails.base import GuardrailViolation
from sirenspec.guardrails.registry import build_guardrails
from sirenspec.providers.registry import resolve_provider
from sirenspec.tools.http_adapter import run_http_tool
from sirenspec.tools.python_adapter import run_python_tool


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



async def run_tool_node(node_id: str, node: ToolNode) -> Any:
    """Execute a tool node with retry logic.

    Attempts the tool call up to ``node.retry + 1`` times.  If all attempts fail and
    ``node.on_failure == 'raise'``, the final :class:`~sirenspec.exceptions.ToolError`
    propagates.  If ``node.on_failure == 'skip'``, ``None`` is returned instead.

    :param node_id: The workflow node identifier (used for error messages).
    :param node: The validated :class:`~sirenspec.core.models.ToolNode` to execute.
    :raises ToolError: If the tool fails on all attempts and ``on_failure == 'raise'``.
    :returns: The tool's return value, or ``None`` if skipped on failure.
    """
    max_attempts = node.retry + 1
    last_exc: ToolError | None = None

    for attempt in range(max_attempts):
        try:
            if node.tool == "http" and isinstance(node.config, HttpToolConfig):
                return await run_http_tool(node.config)
            elif node.tool == "python" and isinstance(node.config, PythonToolConfig):
                return await run_python_tool(node.config)
            else:
                raise ToolError(node.tool, f"Config type mismatch for tool '{node.tool}'")
        except ToolError as exc:
            last_exc = exc
            if attempt < max_attempts - 1:
                continue  # will retry

    # All attempts exhausted.
    if node.on_failure == "skip":
        return None

    assert last_exc is not None
    raise last_exc


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
    5. Provider calls are wrapped by the retry engine.  When all retries are
       exhausted the ``on_failure`` policy governs what happens next:

       * ``abort`` — raises :class:`~sirenspec.exceptions.RetryExhaustedError` and halts execution.
       * ``fallback`` — routes execution to the named ``fallback_node``.
       * ``skip`` — silently marks the node as skipped and continues.
       * ``use_default`` — injects ``default_output`` into the node's write path.

    **Tool nodes** (``type: tool``) are handled by the respective adapter instead of
    an LLM provider.  Their output is stored in the workflow context under
    ``working.<node_id>.<output_key>``.  Downstream nodes can reference this value
    as ``{{ node_id.output }}`` (or whatever ``output_key`` was set to).

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

    # Track the writes path of the most recently completed agent node for downstream input resolution.
    last_writes_path: str | None = None

    trace_nodes: list[dict[str, Any]] = []
    total_tokens = 0
    total_duration_ms = 0.0
    status = "success"

    global_guardrail_names = workflow.guardrails

    # Collect node IDs that the on_failure fallback mechanism wants to force-activate.
    # We store them here and activate them at the start of the loop iteration.
    pending_fallback_nodes: set[str] = set()

    for node_id in execution_order:
        # Activate any fallback nodes requested by a previous on_failure policy.
        if node_id in pending_fallback_nodes:
            active_nodes.add(node_id)
            pending_fallback_nodes.discard(node_id)

        # Skip nodes that no active edge has routed to yet.
        # This is how conditional branching suppresses the unchosen fork.
        if node_id not in active_nodes:
            continue

        node = workflow.nodes[node_id]

        # ------------------------------------------------------------------ #
        # Swrm node — fan-out parallel execution with optional synthesis.     #
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
            except SwrmAgentError as exc:
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
            except Exception as exc:
                duration_ms = (time.monotonic() - start_time) * 1000
                error_trace = {
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

            # Write per-agent outputs and the node-level output into context.
            for agent_trace in swrm_trace["agents"]:
                if agent_trace["response_received"] is not None:
                    context.write(
                        f"working.{node_id}.agents.{agent_trace['id']}.output",
                        agent_trace["response_received"],
                    )
            node_output = swrm_trace["output"]
            context.write(f"output.{node_id}", node_output)
            last_writes_path = f"output.{node_id}"

            node_tokens: int = swrm_trace["tokens"]
            node_duration_ms: float = swrm_trace["duration_ms"]
            total_tokens += node_tokens
            total_duration_ms += node_duration_ms

            # Activate successors.
            for target, condition in out_edges[node_id]:
                if condition is None or evaluate_when_condition(condition, context):
                    active_nodes.add(target)

            trace_nodes.append(swrm_trace)
            continue

        # ---------------------------------------------------------------
        # Tool node execution path
        # ---------------------------------------------------------------
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
                result = await run_tool_node(node_id, node)
                context.write(f"working.{node_id}.{node.output_key}", result)
                duration_ms = (time.monotonic() - start_time) * 1000
                tool_node_trace["result"] = result
                tool_node_trace["duration_ms"] = round(duration_ms, 2)
                total_duration_ms += duration_ms

                for target, condition in out_edges[node_id]:
                    if condition is None or evaluate_when_condition(condition, context):
                        active_nodes.add(target)

            except ToolError as exc:
                duration_ms = (time.monotonic() - start_time) * 1000
                tool_node_trace["error"] = str(exc)
                tool_node_trace["duration_ms"] = round(duration_ms, 2)
                total_duration_ms += duration_ms
                status = "failed"
                trace_nodes.append(tool_node_trace)
                break

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
        # Agent node execution path.                                           #
        # ------------------------------------------------------------------ #
        assert isinstance(node, AgentNode)
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

        retry_policy = resolve_retry_policy(workflow, node_id)
        on_failure_policy = resolve_on_failure_policy(workflow, node_id)

        agent_node_trace: dict[str, Any] = {
            "id": node_id,
            "agent": node.agent,
            "prompt_sent": node_input,
            "response_received": None,
            "writes": node.writes,
            "guardrails_passed": [],
            "tokens": 0,
            "duration_ms": 0,
            "error": None,
            "retry_attempts": [],
        }

        start_time = time.monotonic()
        try:
            # Run input guardrails before sending to the LLM.
            checked_input = node_input
            for g in guardrails:
                checked_input = g.check_input(checked_input)
                agent_node_trace["guardrails_passed"].append(f"{type(g).__name__}.check_input")

            messages = [
                {"role": "system", "content": agent_def.system},
                {"role": "user", "content": checked_input},
            ]

            provider = resolve_provider(agent_def.model)

            def log_attempt(attempt_number: int, delay: float, error_message: str) -> None:
                agent_node_trace["retry_attempts"].append(
                    {
                        "attempt": attempt_number,
                        "delay_seconds": round(delay, 3),
                        "error": error_message,
                    }
                )

            async def call() -> str:
                return await provider.complete(messages)

            response_text = await run_with_retry(
                node_id=node_id,
                policy=retry_policy,
                call=call,
                on_attempt=log_attempt,
            )
            tokens = provider.last_token_count

            # Run output guardrails before writing the response to the context.
            checked_output = response_text
            for g in guardrails:
                checked_output = g.check_output(checked_output)
                agent_node_trace["guardrails_passed"].append(f"{type(g).__name__}.check_output")

            context.write(node.writes, checked_output)
            last_writes_path = node.writes

            # Evaluate outgoing edges now that the context has been updated.
            # An unconditional edge always activates its target; a conditional edge
            # does so only if its when: expression evaluates to True.
            for target, condition in out_edges[node_id]:
                if condition is None or evaluate_when_condition(condition, context):
                    active_nodes.add(target)

            duration_ms = (time.monotonic() - start_time) * 1000
            agent_node_trace.update(
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
                # Activate outgoing edges so the graph can continue.
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
            "total_tokens": total_tokens,
            "total_duration_ms": round(total_duration_ms, 2),
            "status": status,
        },
    }
