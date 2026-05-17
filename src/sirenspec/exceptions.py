"""SirenSpec custom exception hierarchy.

All SirenSpec exceptions inherit from :class:`SirenSpecError` so callers can
catch the entire family with a single ``except SirenSpecError`` clause.
"""

from __future__ import annotations


class SirenSpecError(Exception):
    """Base class for all SirenSpec exceptions."""


class ProviderError(SirenSpecError):
    """Raised when an LLM provider call fails."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class RetryExhaustedError(ProviderError):
    """Raised when all retry attempts for a provider call are exhausted.

    This is a subclass of :class:`ProviderError` and is raised by the executor
    when a node's retry policy runs out of attempts and the on_failure action is
    ``'abort'`` (or no on_failure policy is configured).
    """

    def __init__(self, node_id: str, attempts: int, last_error: Exception) -> None:
        message = f"Node '{node_id}' failed after {attempts} attempt(s): {last_error}"
        super().__init__(message)
        self.node_id = node_id
        self.attempts = attempts
        self.last_error = last_error


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


class ToolError(SirenSpecError):
    """Raised when a tool node execution fails.

    :param tool_name: The name of the tool adapter that failed (e.g. ``'http'``, ``'python'``).
    :param message: A human-readable description of the failure.
    :param cause: The upstream exception that triggered this error, if any.
    :param status_code: HTTP status code, if the failure originated from an HTTP response.
        Used by the retry engine to match against numeric triggers in a RetryPolicy
        (e.g. retry on 429 or 503).  ``None`` for network-level failures (no HTTP response).
    """

    def __init__(
        self,
        tool_name: str,
        message: str,
        cause: BaseException | None = None,
        status_code: int | None = None,
    ) -> None:
        full_message = f"[tool:{tool_name}] {message}"
        super().__init__(full_message)
        self.tool_name = tool_name
        self.cause = cause
        # status_code mirrors the same attribute on ProviderError so that
        # error_matches_policy() in retry.py can treat both error types uniformly.
        self.status_code = status_code


class InterpolationError(SirenSpecError):
    """Raised when ``{{ expr }}`` template resolution fails.

    :param expression: The full expression string (without braces), e.g. ``inputs.message``.
    :param namespace: The namespace prefix that failed (e.g. ``'inputs'``, ``'env'``, ``'node_id'``).
    :param reason: Human-readable description of what went wrong.
    """

    def __init__(self, expression: str, namespace: str, reason: str) -> None:
        super().__init__(f"InterpolationError in '{{{{ {expression} }}}}' [{namespace}]: {reason}")
        self.expression = expression
        self.namespace = namespace
        self.reason = reason


class PIIDetectedError(GuardrailError):
    """Raised when PII is detected and the guardrail action is ``'block'``.

    :param entity_types: List of entity type names that were detected (e.g. ``['email', 'ssn']``).
        Matched values are intentionally omitted from the message to avoid leaking PII.
    """

    def __init__(self, entity_types: list[str]) -> None:
        super().__init__(f"PII detected: {entity_types}")
        self.entity_types = entity_types


class FactoryNodeError(SirenSpecError):
    """Raised when a factory node instance fails and ``on_failure`` is ``'abort'``.

    :param node_id: The factory node's identifier.
    :param instance_index: Zero-based index of the failing instance, or ``None`` for setup failures.
    :param cause: The original exception raised by the instance.
    """

    def __init__(self, node_id: str, instance_index: int | None, cause: BaseException) -> None:
        idx_str = f"[{instance_index}]" if instance_index is not None else ""
        super().__init__(f"Factory node '{node_id}'{idx_str} failed: {cause}")
        self.node_id = node_id
        self.instance_index = instance_index
        self.cause = cause
