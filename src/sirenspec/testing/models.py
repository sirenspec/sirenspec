"""Pydantic models for SirenSpec YAML test fixtures."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, model_validator


class Assertion(BaseModel):
    """A single assertion on the execution trace.

    One of ``path`` or ``node`` must be provided. Exactly one operator
    (``equals``, ``contains``, ``matches``, ``lt``, ``gt``, ``exists``,
    ``status``) must be set.

    When ``node`` is given without ``path``, ``status`` is the only valid
    operator (shorthand for asserting node completion status).  When ``node``
    is given with ``path``, the path is resolved relative to the matching
    node's trace dict.
    """

    path: str | None = None
    node: str | None = None

    equals: Any = None
    contains: str | None = None
    matches: str | None = None
    lt: float | None = None
    gt: float | None = None
    exists: bool | None = None
    status: str | None = None

    @model_validator(mode="after")
    def validate_assertion(self) -> Assertion:
        """Ensure the assertion has a resolvable target and exactly one operator.

        :raises ValueError: If neither ``path`` nor ``node`` is given, or if no
            operator is set.
        :returns: The validated :class:`Assertion`.
        """
        if self.path is None and self.node is None:
            raise ValueError("Each assertion must specify at least one of 'path' or 'node'")
        operators = [
            k for k in ("equals", "contains", "matches", "lt", "gt", "exists", "status") if getattr(self, k) is not None
        ]
        if not operators:
            raise ValueError(
                "Each assertion must specify exactly one operator (equals, contains, matches, lt, gt, exists, status)"
            )
        return self


class WorkflowFixture(BaseModel):
    """A single SirenSpec YAML test fixture.

    :param workflow: Relative or absolute path to the workflow YAML file.
    :param input: User input string to pass to the workflow.
    :param assertions: List of assertions to evaluate against the execution trace.
    """

    workflow: str
    input: str
    assertions: list[Assertion] = []
