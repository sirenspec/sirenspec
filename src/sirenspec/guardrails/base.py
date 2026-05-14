"""Abstract base class for guardrails and the GuardrailViolation exception."""

from __future__ import annotations

from abc import ABC, abstractmethod


class GuardrailViolation(Exception):
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
