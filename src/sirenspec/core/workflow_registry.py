"""Registry for named SirenSpec workflow definitions."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sirenspec.core.models import Workflow


class WorkflowRegistry:
    """Maps string names to :class:`~sirenspec.core.models.Workflow` instances.

    Pass a populated registry to :func:`~sirenspec.core.executor.execute` or
    :func:`~sirenspec.core.executor.execute_streaming` so that workflow nodes
    using a named ``ref`` (e.g. ``ref: my-sub-workflow``) can be resolved at
    runtime.

    File-path refs (starting with ``.`` or ``/``) are resolved by the executor
    directly from disk and do not require a registry entry.
    """

    def __init__(self) -> None:
        self._registry: dict[str, Workflow] = {}

    def register(self, name: str, workflow: Workflow) -> None:
        """Register *workflow* under *name*.

        :param name: The name used in a workflow node's ``ref`` field.
        :param workflow: The :class:`~sirenspec.core.models.Workflow` instance to register.
        """
        self._registry[name] = workflow

    def get(self, name: str) -> Workflow:
        """Return the workflow registered under *name*.

        :param name: The name to look up.
        :raises KeyError: If no workflow is registered under *name*.
        :returns: The registered :class:`~sirenspec.core.models.Workflow` instance.
        """
        if name not in self._registry:
            raise KeyError(f"No workflow registered under '{name}'")
        return self._registry[name]
