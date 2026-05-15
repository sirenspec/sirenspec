"""SirenSpec custom exception hierarchy.

All SirenSpec exceptions inherit from :class:`SirenSpecError` so callers can
catch the entire family with a single ``except SirenSpecError`` clause.
"""

from __future__ import annotations


class SirenSpecError(Exception):
    """Base class for all SirenSpec exceptions."""


class ProviderError(SirenSpecError):
    """Raised when an LLM provider call fails."""


class GuardrailError(SirenSpecError):
    """Raised on guardrail configuration or execution errors."""


class ValidationError(SirenSpecError):
    """Raised when workflow YAML fails model validation."""


class SwrmAgentError(SirenSpecError):
    """Raised when one or more agents inside a swrm node fail.

    :param agent_id: The ``id`` of the agent that raised the underlying error.
    :param cause: The original exception raised by the agent.
    """

    def __init__(self, agent_id: str, cause: BaseException) -> None:
        super().__init__(f"swrm agent '{agent_id}' failed: {cause}")
        self.agent_id = agent_id
        self.cause = cause
