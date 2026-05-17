"""Agent node execution: provider dispatch, guardrail checks, and retry orchestration."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from sirenspec.core.models import RetryPolicy
from sirenspec.core.retry import run_with_retry
from sirenspec.core.usage import TokenUsage
from sirenspec.guardrails.registry import build_guardrails
from sirenspec.providers.registry import resolve_provider


@dataclass
class AgentRunResult:
    """Result of a single agent node execution.

    :param output: The guardrail-checked response text.
    :param token_usage: Structured token usage reported by the provider.
    :param duration_ms: Wall-clock milliseconds from first attempt start to final result.
    :param guardrails_passed: Names of each guardrail check that ran, in order.
    :param retry_attempts: Log entries for each retry (empty when the first attempt succeeds).
    """

    output: str
    token_usage: TokenUsage
    duration_ms: float
    guardrails_passed: list[str] = field(default_factory=list)
    retry_attempts: list[dict[str, Any]] = field(default_factory=list)

    @property
    def tokens(self) -> int:
        """Total token count (prompt + completion).

        Provided for backward-compatible access in callers that read ``result.tokens``.

        :returns: Sum of prompt and completion tokens.
        """
        return self.token_usage.total


async def execute_agent_node(
    node_id: str,
    model_uri: str,
    system_prompt: str,
    user_input: str,
    guardrail_names: list[str] | None,
    retry_policy: RetryPolicy,
) -> AgentRunResult:
    """Execute a single agent call: apply guardrails, call the provider with retry, check output.

    :param node_id: Node identifier used in retry error messages.
    :param model_uri: Provider URI in ``'provider:model'`` format (e.g. ``'openai:gpt-4o-mini'``).
    :param system_prompt: System prompt text; omitted from messages when empty.
    :param user_input: Already-resolved user input text.
    :param guardrail_names: Names of guardrails to apply on input and output.
        Pass ``None`` to use the default set (injection detection).
        Pass ``[]`` to explicitly disable all guardrails.
        These two cases are NOT the same — None is never silently coerced to [].
    :param retry_policy: Policy governing retry behaviour on provider failure.
    :raises GuardrailViolation: If any guardrail rejects the input or output.
    :raises RetryExhaustedError: If the provider fails on all retry attempts.
    :returns: :class:`AgentRunResult` with output, token usage, timing, and audit trail.
    """
    # None → default guardrails (injection detection); [] → no guardrails.
    # build_guardrails understands this distinction — do not coerce None to [] here.
    guardrails = build_guardrails(guardrail_names)
    guardrails_passed: list[str] = []
    retry_attempts: list[dict[str, Any]] = []

    # Run each guardrail's input check in order; each check may transform the text
    # (e.g. stripping detected injections) before passing it to the next guardrail.
    checked_input = user_input
    for g in guardrails:
        checked_input = g.check_input(checked_input)
        guardrails_passed.append(f"{type(g).__name__}.check_input")

    # An empty system_prompt means "no system message" — used by swrm agents,
    # which always pass "" so the synthesis step can be stateless.
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": checked_input})

    provider = resolve_provider(model_uri)

    # log_retry is a closure that captures retry_attempts by reference.
    # run_with_retry calls it before each retry (not before the first attempt).
    def log_retry(attempt: int, delay: float, error: str) -> None:
        retry_attempts.append({"attempt": attempt, "delay_seconds": round(delay, 3), "error": error})

    start = time.monotonic()

    # run_with_retry requires a zero-argument async callable so it can re-invoke
    # the provider on each retry without needing to know about messages or the provider.
    async def call() -> str:
        return await provider.complete(messages)

    output = await run_with_retry(node_id=node_id, policy=retry_policy, call=call, on_attempt=log_retry)

    # Read token usage after the retry loop completes; the provider updates
    # last_token_usage after every successful call, so this always reflects the
    # attempt that actually succeeded.
    token_usage = provider.last_token_usage
    duration_ms = (time.monotonic() - start) * 1000

    # Output guardrails run after a successful provider response. They may raise
    # GuardrailViolation (e.g. length exceeded) — this is NOT retried.
    for g in guardrails:
        output = g.check_output(output)
        guardrails_passed.append(f"{type(g).__name__}.check_output")

    return AgentRunResult(
        output=output,
        token_usage=token_usage,
        duration_ms=duration_ms,
        guardrails_passed=guardrails_passed,
        retry_attempts=retry_attempts,
    )
