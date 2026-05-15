"""Tool node execution: adapter dispatch with simple retry."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from sirenspec.core.models import HttpToolConfig, PythonToolConfig, ToolNode
from sirenspec.exceptions import ToolError
from sirenspec.tools.http_adapter import run_http_tool
from sirenspec.tools.python_adapter import run_python_tool


@dataclass
class ToolRunResult:
    """Result of a single tool node execution.

    :param result: The value returned by the tool adapter, or ``None`` when skipped.
    :param duration_ms: Wall-clock milliseconds from first attempt start to completion.
    """

    result: Any
    duration_ms: float


async def dispatch_tool(node: ToolNode) -> Any:
    """Route a tool node to the appropriate adapter and return its result.

    The tool name and config type are always validated together: a mismatch
    (e.g. tool='http' but config is a PythonToolConfig) is a programming error
    and raises ToolError immediately without retrying.

    :param node: The :class:`~sirenspec.core.models.ToolNode` to execute.
    :raises ToolError: If the tool type is unrecognised or the config type does not match.
    :returns: The value returned by the adapter.
    """
    if node.tool == "http" and isinstance(node.config, HttpToolConfig):
        return await run_http_tool(node.config)
    if node.tool == "python" and isinstance(node.config, PythonToolConfig):
        return await run_python_tool(node.config)
    raise ToolError(node.tool, f"Config type mismatch for tool '{node.tool}'")


async def execute_tool_node(node_id: str, node: ToolNode) -> ToolRunResult:
    """Execute a tool node, retrying on failure up to ``node.retry`` additional times.

    Tool nodes use a simpler retry model than agent nodes: every ToolError triggers
    a retry (there is no per-error-type matching like the LLM retry policy). The
    ``on_failure`` behaviour is resolved here after all attempts are exhausted:

    * ``on_failure='skip'`` — returns :class:`ToolRunResult` with ``result=None`` instead of raising.
    * ``on_failure='raise'`` (default) — re-raises the last :class:`~sirenspec.exceptions.ToolError`.

    :param node_id: Node identifier (used in error messages).
    :param node: The :class:`~sirenspec.core.models.ToolNode` to execute.
    :raises ToolError: If all retry attempts fail and ``on_failure`` is ``'raise'``.
    :returns: :class:`ToolRunResult` with the tool's output and elapsed time.
    """
    # node.retry is the number of *extra* attempts beyond the first, so
    # total attempts = retry + 1.  None means no retries (1 attempt total).
    max_attempts = (node.retry or 0) + 1
    last_exc: ToolError | None = None

    start = time.monotonic()
    # The loop variable is named with a leading underscore because it is never
    # used inside the loop body — ruff (rule B007) would flag an unused variable.
    for _attempt_index in range(max_attempts):
        try:
            result = await dispatch_tool(node)
            return ToolRunResult(result=result, duration_ms=(time.monotonic() - start) * 1000)
        except ToolError as exc:
            last_exc = exc

    # We only reach here when every attempt raised ToolError.
    duration_ms = (time.monotonic() - start) * 1000

    if node.on_failure == "skip":
        # Return a sentinel result so the executor can continue the workflow
        # without treating this as a fatal failure.
        return ToolRunResult(result=None, duration_ms=duration_ms)

    # The type checker cannot prove last_exc is non-None here, but the logic
    # guarantees it: the loop always runs at least once (max_attempts >= 1) and
    # we only reach this line when every iteration caught a ToolError.
    assert last_exc is not None
    raise last_exc
