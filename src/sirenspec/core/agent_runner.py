"""Agent node execution: provider dispatch, guardrail checks, and retry orchestration."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from sirenspec.core.models import RetryPolicy
from sirenspec.core.retry import run_with_retry
from sirenspec.core.usage import TokenUsage
from sirenspec.guardrails.registry import build_guardrails
from sirenspec.providers.base import StreamingLLMProvider
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


async def collect_stream(provider: StreamingLLMProvider, messages: list[dict], callback: Callable[[str], None]) -> str:
    """Consume a provider stream, invoke *callback* for each chunk, and return the full text.

    :param provider: A streaming-capable provider instance.
    :param messages: List of ``{"role": ..., "content": ...}`` dicts passed to the stream.
    :param callback: Called with each text chunk as it arrives from the provider.
    :returns: The fully assembled response text.
    """
    chunks: list[str] = []
    async for chunk in provider.stream(messages):
        callback(chunk)
        chunks.append(chunk)
    return "".join(chunks)


async def execute_agent_node(
    node_id: str,
    model_uri: str,
    system_prompt: str,
    user_input: str,
    guardrail_names: list[str] | None,
    retry_policy: RetryPolicy,
    streaming: bool = True,
    stream_callback: Callable[[str], None] | None = None,
) -> AgentRunResult:
    """Execute a single agent call: apply guardrails, call the provider with retry, check output.

    When *streaming* is ``True`` and the provider supports the
    :class:`~sirenspec.providers.base.StreamingLLMProvider` protocol, tokens are
    streamed incrementally.  Each chunk is forwarded to *stream_callback* as it
    arrives.  Guardrails still run against the fully assembled response after the
    stream completes.

    When *streaming* is ``False``, or when the provider does not implement
    :meth:`~sirenspec.providers.base.StreamingLLMProvider.stream`, the non-streaming
    :meth:`~sirenspec.providers.base.LLMProvider.complete` path is used instead.

    :param node_id: Node identifier used in retry error messages.
    :param model_uri: Provider URI in ``'provider:model'`` format (e.g. ``'openai:gpt-4o-mini'``).
    :param system_prompt: System prompt text; omitted from messages when empty.
    :param user_input: Already-resolved user input text.
    :param guardrail_names: Names of guardrails to apply on input and output.
        Pass ``None`` to use the default set (injection detection).
        Pass ``[]`` to explicitly disable all guardrails.
        These two cases are NOT the same — None is never silently coerced to [].
    :param retry_policy: Policy governing retry behaviour on provider failure.
    :param streaming: Whether to use token-by-token streaming when supported.
    :param stream_callback: Optional callable invoked with each text chunk during streaming.
        Ignored when *streaming* is ``False`` or the provider does not support streaming.
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

    use_streaming = streaming and stream_callback is not None and isinstance(provider, StreamingLLMProvider)

    if use_streaming:
        effective_callback: Callable[[str], None] = stream_callback if stream_callback is not None else lambda _: None

        async def stream_call() -> str:
            return await collect_stream(provider, messages, effective_callback)

        output = await run_with_retry(node_id=node_id, policy=retry_policy, call=stream_call, on_attempt=log_retry)
    else:

        async def complete_call() -> str:
            return await provider.complete(messages)

        output = await run_with_retry(node_id=node_id, policy=retry_policy, call=complete_call, on_attempt=log_retry)

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
