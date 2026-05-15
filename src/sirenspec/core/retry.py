"""Retry engine: backoff calculation and retry-loop execution."""

from __future__ import annotations

import asyncio
import random
from collections.abc import Callable, Coroutine
from typing import Any

from sirenspec.core.models import RetryPolicy
from sirenspec.exceptions import RetryExhaustedError


def compute_delay(policy: RetryPolicy, attempt: int) -> float:
    """Return the delay in seconds for the given attempt index (0-based).

    :param policy: The :class:`~sirenspec.core.models.RetryPolicy` governing backoff.
    :param attempt: Zero-based retry index (0 = first retry after the initial failure).
    :returns: Delay in seconds, clamped to ``policy.max_delay`` and optionally jittered.
    """
    if policy.backoff == "constant":
        delay = policy.base_delay
    elif policy.backoff == "linear":
        delay = policy.base_delay * (attempt + 1)
    else:  # exponential
        delay = policy.base_delay * (2**attempt)

    delay = min(delay, policy.max_delay)

    if policy.jitter:
        # Apply ±20% random variation.
        factor = 1.0 + random.uniform(-0.2, 0.2)  # noqa: S311
        delay = delay * factor

    return delay


def error_matches_policy(exc: Exception, policy: RetryPolicy) -> bool:
    """Return True if *exc* matches any trigger condition in *policy.on*.

    HTTP status-code triggers (e.g. ``'429'``, ``'500'``) are matched against
    the ``status_code`` attribute on the exception (set by :class:`~sirenspec.exceptions.ProviderError`
    and compatible HTTP client exceptions).  The special string ``'network_error'`` matches
    any exception that does not have a numeric status code (i.e. connection-level failures).

    :param exc: The exception raised by the provider call.
    :param policy: The retry policy specifying which errors trigger a retry.
    :returns: True if the exception should trigger a retry.
    """
    status_code: int | None = getattr(exc, "status_code", None)

    for trigger in policy.on:
        if trigger == "network_error":
            if status_code is None:
                return True
        else:
            try:
                if status_code is not None and int(trigger) == status_code:
                    return True
                # Also match 5xx ranges: any trigger like "500", "502" etc.
            except ValueError:
                pass

    return False


async def run_with_retry[T](
    node_id: str,
    policy: RetryPolicy,
    call: Callable[[], Coroutine[Any, Any, T]],
    on_attempt: Callable[[int, float, str], None] | None = None,
) -> T:
    """Execute *call* with retries according to *policy*.

    :param node_id: Identifier of the node being executed (used in error messages and logs).
    :param policy: The :class:`~sirenspec.core.models.RetryPolicy` governing retry behaviour.
    :param call: An async callable (no arguments) that performs the call and returns a value.
    :param on_attempt: Optional callback invoked before each retry with
        ``(attempt_number, delay_seconds, error_message)``.  The first attempt (attempt 1)
        does not trigger this callback; it fires only for retries (attempt 2+).
    :raises RetryExhaustedError: When all attempts are exhausted without a successful result.
    :returns: The value returned by *call* on a successful attempt.
    """
    last_exc: Exception | None = None

    for attempt in range(1, policy.max_attempts + 1):
        try:
            return await call()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc

            is_last = attempt >= policy.max_attempts
            if is_last or not error_matches_policy(exc, policy):
                raise RetryExhaustedError(node_id, attempt, exc) from exc

            # Compute delay for the upcoming retry (0-based index = attempt - 1).
            delay = compute_delay(policy, attempt - 1)

            if on_attempt is not None:
                on_attempt(attempt + 1, delay, str(exc))

            await asyncio.sleep(delay)

    # Unreachable in practice, but satisfies type checker.
    raise RetryExhaustedError(node_id, policy.max_attempts, last_exc or RuntimeError("unknown"))  # pragma: no cover
