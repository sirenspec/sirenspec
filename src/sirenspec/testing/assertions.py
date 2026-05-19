"""Assertion evaluation: path resolution and operator checks against an execution trace."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from sirenspec.testing.models import Assertion


@dataclass
class AssertionResult:
    """Outcome of evaluating a single assertion.

    :param passed: Whether the assertion passed.
    :param message: Human-readable description of the result (pass or diff on failure).
    """

    passed: bool
    message: str


def resolve_path(trace: dict[str, Any], path: str) -> Any:
    """Resolve a dot-notation path against the execution trace.

    Special path prefixes:
    - ``nodes.<node_id>.<field>`` — find the node with ``id==<node_id>`` in
      ``trace["nodes"]`` and return ``node[<field>]``.
    - ``total_usage.<field>`` — shorthand for ``summary.total_usage.<field>``.
    - All other paths navigate plain dict keys with dot splitting.

    :param trace: The full execution trace dict from :func:`~sirenspec.core.executor.execute`.
    :param path: Dot-notation path string.
    :raises KeyError: If any segment of the path cannot be resolved.
    :returns: The value at the resolved path.
    """
    parts = path.split(".")

    if parts[0] == "nodes" and len(parts) >= 2:
        node_id = parts[1]
        nodes: list[dict[str, Any]] = trace.get("nodes", [])
        node = next((n for n in nodes if n.get("id") == node_id), None)
        if node is None:
            raise KeyError(f"No node with id '{node_id}' in trace")
        if len(parts) == 2:
            return node
        return _navigate(node, parts[2:])

    if parts[0] == "total_usage":
        summary = trace.get("summary", {})
        total_usage = summary.get("total_usage", {})
        return _navigate(total_usage, parts[1:])

    return _navigate(trace, parts)


def _navigate(obj: Any, parts: list[str]) -> Any:
    """Walk *obj* by successive key lookups using *parts*.

    :param obj: Starting object (dict expected at each level).
    :param parts: Sequence of keys to traverse.
    :raises KeyError: If any key is missing.
    :returns: The value at the end of the path.
    """
    current = obj
    for part in parts:
        if not isinstance(current, dict):
            raise KeyError(f"Cannot navigate into non-dict value at key '{part}'")
        if part not in current:
            raise KeyError(f"Key '{part}' not found")
        current = current[part]
    return current


def resolve_node_path(trace: dict[str, Any], node_id: str, field: str | None) -> Any:
    """Resolve a node-scoped path for assertions using the ``node:`` shorthand.

    :param trace: The full execution trace dict.
    :param node_id: The ``id`` of the node to look up in ``trace["nodes"]``.
    :param field: Optional field within the node trace dict.  When ``None``
        the entire node trace dict is returned.
    :raises KeyError: If the node is not found or the field is missing.
    :returns: The value at the resolved path.
    """
    nodes: list[dict[str, Any]] = trace.get("nodes", [])
    node = next((n for n in nodes if n.get("id") == node_id), None)
    if node is None:
        raise KeyError(f"No node with id '{node_id}' in trace")
    if field is None:
        return node
    return _navigate(node, field.split("."))


def evaluate_operator(assertion: Assertion, value: Any) -> AssertionResult:
    """Apply the assertion's operator to *value* and return the result.

    :param assertion: The assertion whose operator should be applied.
    :param value: The resolved value from the execution trace.
    :returns: :class:`AssertionResult` indicating pass/fail and a diff message.
    """
    if assertion.equals is not None:
        passed = value == assertion.equals
        return AssertionResult(passed=passed, message=f"equals {assertion.equals!r}: got {value!r}")

    if assertion.contains is not None:
        if isinstance(value, str):
            passed = assertion.contains in value
        elif isinstance(value, list):
            passed = assertion.contains in value
        else:
            passed = False
        return AssertionResult(passed=passed, message=f"contains {assertion.contains!r}: got {value!r}")

    if assertion.matches is not None:
        if not isinstance(value, str):
            return AssertionResult(passed=False, message=f"matches: expected string, got {type(value).__name__}")
        passed = bool(re.search(assertion.matches, value))
        return AssertionResult(passed=passed, message=f"matches {assertion.matches!r}: got {value!r}")

    if assertion.lt is not None:
        try:
            passed = float(value) < assertion.lt
        except (TypeError, ValueError):
            return AssertionResult(passed=False, message=f"lt: expected numeric, got {value!r}")
        return AssertionResult(passed=passed, message=f"lt {assertion.lt}: got {value!r}")

    if assertion.gt is not None:
        try:
            passed = float(value) > assertion.gt
        except (TypeError, ValueError):
            return AssertionResult(passed=False, message=f"gt: expected numeric, got {value!r}")
        return AssertionResult(passed=passed, message=f"gt {assertion.gt}: got {value!r}")

    if assertion.exists is not None:
        present = value is not None
        passed = present == assertion.exists
        return AssertionResult(passed=passed, message=f"exists={assertion.exists}: got {value!r}")

    if assertion.status is not None:
        passed = value == assertion.status
        return AssertionResult(passed=passed, message=f"status equals {assertion.status!r}: got {value!r}")

    return AssertionResult(passed=False, message="No operator set on assertion")


def evaluate_assertion(assertion: Assertion, trace: dict[str, Any]) -> AssertionResult:
    """Resolve the target value from the trace and evaluate the assertion operator.

    Handles both the ``path:`` form and the ``node:`` shorthand form.

    :param assertion: The assertion to evaluate.
    :param trace: The full execution trace dict.
    :returns: :class:`AssertionResult` with pass/fail and a diff string.
    """
    try:
        if assertion.node is not None:
            field = (
                assertion.path if assertion.path is not None else ("status" if assertion.status is not None else None)
            )
            value = resolve_node_path(trace, assertion.node, field)
        else:
            value = resolve_path(trace, assertion.path)  # type: ignore[arg-type]
    except KeyError as exc:
        return AssertionResult(passed=False, message=f"path resolution failed: {exc}")

    return evaluate_operator(assertion, value)
