"""Helpers for ``workflow`` nodes: resolve a sub-workflow ref and its inputs.

The execution itself lives in :mod:`sirenspec.core.executor` as
:func:`~sirenspec.core.executor.execute_workflow_node` — co-located with the
``execute`` function it calls to avoid an import cycle.
"""

from __future__ import annotations

from typing import Any

from sirenspec.core.interpolation import build_interpolation_context, resolve_template
from sirenspec.core.models import WorkflowNode
from sirenspec.core.workflow_registry import WorkflowRegistry
from sirenspec.yaml.parser import load_workflow


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
