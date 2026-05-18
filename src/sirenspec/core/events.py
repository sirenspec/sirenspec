"""Typed streaming events emitted by :func:`~sirenspec.core.executor.execute_streaming`."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class NodeCompleteEvent:
    """Emitted once per node (active or skipped) during a streaming workflow execution.

    :param kind: Discriminator literal — always ``"node_complete"``.
    :param node_id: The identifier of the node that completed or was skipped.
    :param node_type: The node type: ``"agent"``, ``"tool"``, ``"swrm"``, ``"factory"``, or ``"workflow"``.
    :param output: The node's output value (string, dict, or list).  ``None`` for skipped nodes.
    :param writes: The context path written by this node (agent nodes only).
    :param status: Execution result: ``"success"``, ``"skipped"``, or ``"failed"``.
    :param error: Human-readable error message when ``status="failed"``.
    :param tokens: Total tokens consumed by this node (0 for tool and skipped nodes).
    """

    kind: Literal["node_complete"] = field(default="node_complete")
    node_id: str = field(default="")
    node_type: str = field(default="")
    output: Any = field(default=None)
    writes: str = field(default="")
    status: Literal["success", "skipped", "failed"] = field(default="success")
    error: str | None = field(default=None)
    tokens: int = field(default=0)


@dataclass
class SummaryEvent:
    """Emitted once at the end of a streaming workflow execution.

    :param kind: Discriminator literal — always ``"summary"``.
    :param total_nodes: Number of nodes that ran (active nodes only, excluding skipped).
    :param total_tokens: Aggregate token count across all active nodes.
    :param estimated_usd: Estimated total USD cost, or ``None`` if pricing is unavailable.
    :param status: Overall workflow status: ``"success"`` or ``"failed"``.
    :param duration_ms: Wall-clock execution time in milliseconds.
    """

    kind: Literal["summary"] = field(default="summary")
    total_nodes: int = field(default=0)
    total_tokens: int = field(default=0)
    estimated_usd: float | None = field(default=None)
    status: Literal["success", "failed"] = field(default="success")
    duration_ms: float = field(default=0.0)
