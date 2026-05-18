"""Runner for ``workflow`` nodes: loads and executes a sub-workflow inline."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sirenspec.core.interpolation import build_interpolation_context, resolve_template
from sirenspec.core.models import WorkflowNode
from sirenspec.exceptions import ValidationError

if TYPE_CHECKING:
    from sirenspec.core.workflow_registry import WorkflowRegistry


def resolve_sub_workflow(ref: str, registry: WorkflowRegistry | None) -> Any:
    """Return the Workflow object for *ref*, loading from disk or registry as needed.

    :param ref: A relative/absolute file path (starts with ``.`` or ``/``) or a
        registry name.
    :param registry: Optional :class:`~sirenspec.core.workflow_registry.WorkflowRegistry`
        used for named refs.
    :raises ValueError: If *ref* is a named ref and *registry* is ``None``.
    :raises KeyError: If *ref* names a workflow not present in *registry*.
    :raises FileNotFoundError: If *ref* is a file path that does not exist.
    :returns: A validated :class:`~sirenspec.core.models.Workflow` instance.
    """
    # Deferred to avoid circular import: yaml.parser → models, executor → workflow_runner → yaml.parser
    from sirenspec.yaml.parser import load_workflow

    if ref.startswith(".") or ref.startswith("/"):
        return load_workflow(ref)
    if registry is None:
        raise ValueError(
            f"Workflow node references '{ref}' by name but no WorkflowRegistry was provided to the executor."
        )
    return registry.get(ref)


def resolve_node_inputs(
    node: WorkflowNode,
    user_input: str,
    working: dict[str, Any],
) -> dict[str, str]:
    """Resolve each template string in *node.inputs* against the parent context.

    :param node: The :class:`~sirenspec.core.models.WorkflowNode` whose inputs to resolve.
    :param user_input: The parent workflow's user input string.
    :param working: The parent workflow's current ``working`` dict.
    :returns: A dict of input key → resolved string value.
    """
    interp_ctx = build_interpolation_context(user_input, working)
    return {key: resolve_template(value, interp_ctx) for key, value in node.inputs.items()}


async def execute_workflow_node(
    node_id: str,
    node: WorkflowNode,
    user_input: str,
    working: dict[str, Any],
    registry: WorkflowRegistry | None,
    depth: int,
) -> dict[str, Any]:
    """Execute a sub-workflow node and return a structured trace dict.

    The sub-workflow runs inline and blocking.  Its output dict (keyed by sub-node ID)
    is returned as ``output`` in the trace so the executor can write it into the parent
    context.

    :param node_id: The parent node's identifier (used in error messages and the trace).
    :param node: The :class:`~sirenspec.core.models.WorkflowNode` definition.
    :param user_input: The parent workflow's user input string (passed through to the sub-workflow).
    :param working: The parent workflow's current ``working`` dict, used to resolve ``inputs``.
    :param registry: Optional registry for named workflow refs.
    :param depth: Current nesting depth (0 = top-level parent).
    :raises ValidationError: If the nesting depth exceeds ``node.max_depth``.
    :returns: Trace dict with keys ``id``, ``type``, ``ref``, ``inputs``, ``output``,
        ``tokens``, ``duration_ms``, ``sub_trace``, and ``error``.
    """
    # Deferred to avoid circular import: executor imports workflow_runner, which would re-import executor
    from sirenspec.core.executor import execute

    if depth >= node.max_depth:
        raise ValidationError(f"Max workflow nesting depth {node.max_depth} exceeded at node '{node_id}'")

    sub_workflow = resolve_sub_workflow(node.ref, registry)
    resolved_inputs = resolve_node_inputs(node, user_input, working)

    sub_trace = await execute(
        sub_workflow,
        user_input,
        registry=registry,
        depth=depth + 1,
        initial_inputs=resolved_inputs,
    )

    return {
        "id": node_id,
        "type": "workflow",
        "ref": node.ref,
        "inputs": resolved_inputs,
        "output": sub_trace["output"],
        "tokens": sub_trace["summary"]["total_tokens"],
        "duration_ms": sub_trace["summary"]["total_duration_ms"],
        "sub_trace": sub_trace,
        "error": None,
    }
