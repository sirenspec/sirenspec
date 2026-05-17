"""Abstract base class for guardrails and the GuardrailViolation exception."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol, runtime_checkable

from sirenspec.core.usage import TokenUsage
from sirenspec.exceptions import GuardrailError


class GuardrailViolation(GuardrailError):
    """Raised when a guardrail detects a policy violation."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class Guardrail(ABC):
    """Abstract base class that all guardrails must implement."""

    @abstractmethod
    def check_input(self, text: str) -> str:
        """Validate or transform *text* before it is sent to the LLM.

        :param text: The input text to check.
        :raises GuardrailViolation: If the text violates the guardrail policy.
        :returns: The (possibly transformed) input text.
        """

    @abstractmethod
    def check_output(self, text: str) -> str:
        """Validate or transform *text* after it is received from the LLM.

        :param text: The output text to check.
        :raises GuardrailViolation: If the text violates the guardrail policy.
        :returns: The (possibly transformed) output text.
        """


@runtime_checkable
class WorkflowGuardrail(Protocol):
    """Structural interface for guardrails that operate at the workflow level.

    Unlike :class:`Guardrail`, which checks individual inputs and outputs, a
    ``WorkflowGuardrail`` is checked after each node completes and has access to
    the accumulated :class:`~sirenspec.core.usage.TokenUsage` across all nodes so far.

    Any object that implements :meth:`check_budget` satisfies this protocol.
    """

    def check_budget(self, usage: TokenUsage, estimated_usd: float | None) -> None:
        """Assert that the accumulated spend is within the configured budget.

        :param usage: Combined token usage accumulated across all completed nodes.
        :param estimated_usd: Running USD estimate, or ``None`` if the models in use
            do not have pricing entries (e.g. Ollama/local models).
        :raises BudgetExceededError: If the accumulated spend exceeds the configured ceiling.
        """
        ...
