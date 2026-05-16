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
        # Delay grows linearly: base * 1, base * 2, base * 3, …
        delay = policy.base_delay * (attempt + 1)
    else:  # exponential
        # Delay doubles with each attempt: base * 1, base * 2, base * 4, …
        delay = policy.base_delay * (2**attempt)

    delay = min(delay, policy.max_delay)

    if policy.jitter:
        # ±20% random variation prevents all retrying clients from hammering
        # a recovering service at the exact same moment (thundering herd).
        factor = 1.0 + random.uniform(-0.2, 0.2)  # noqa: S311
        delay = delay * factor

    return delay


def error_matches_policy(exc: Exception, policy: RetryPolicy) -> bool:
    """Return True if *exc* matches any trigger condition in *policy.on*.

    HTTP status-code triggers (e.g. ``'429'``, ``'500'``) are matched against
    the ``status_code`` attribute on the exception (set by :class:`~sirenspec.exceptions.ProviderError`
    and compatible HTTP client exceptions).  The special string ``'network_error'`` matches
    any exception that does not have a numeric status code (i.e. connection-level failures
    such as DNS errors or dropped TCP connections).

    :param exc: The exception raised by the provider call.
    :param policy: The retry policy specifying which errors trigger a retry.
    :returns: True if the exception should trigger a retry.
    """
    # status_code is set by ProviderError and ToolError on HTTP failures.
    # A value of None means the exception came from a network-level failure,
    # not an HTTP response (e.g. ConnectionError, TimeoutError).
    status_code: int | None = getattr(exc, "status_code", None)

    for trigger in policy.on:
        if trigger == "network_error":
            # "network_error" matches any exception that has no HTTP status code.
            if status_code is None:
                return True
        else:
            try:
                if status_code is not None and int(trigger) == status_code:
                    return True
            except ValueError:
                # Ignore unrecognised trigger strings (e.g. typos in YAML).
                pass

    return False


async def run_with_retry[T](
    node_id: str,
    policy: RetryPolicy,
    call: Callable[[], Coroutine[Any, Any, T]],
    on_attempt: Callable[[int, float, str], None] | None = None,
) -> T:
    """Execute *call* with retries according to *policy*.

    The ``[T]`` type parameter (PEP 695, Python 3.12+) makes this function generic:
    callers can pass any async callable and get back its exact return type.
    This replaces the older ``TypeVar("T")`` style and requires no extra imports.

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

    # attempt is 1-based so that "attempt >= policy.max_attempts" naturally detects
    # the last attempt without an off-by-one adjustment.
    for attempt in range(1, policy.max_attempts + 1):
        try:
            return await call()
        except Exception as exc:  # noqa: BLE001
            last_exc = exc

            is_last = attempt >= policy.max_attempts
            if is_last or not error_matches_policy(exc, policy):
                # Either we've used all attempts, or this error type is not in the
                # retry policy — stop immediately instead of sleeping and retrying.
                raise RetryExhaustedError(node_id, attempt, exc) from exc

            # compute_delay expects a 0-based index, so subtract 1 from the
            # 1-based attempt counter to keep the math consistent.
            delay = compute_delay(policy, attempt - 1)

            if on_attempt is not None:
                # Notify the caller that attempt+1 is about to start after sleeping.
                on_attempt(attempt + 1, delay, str(exc))

            await asyncio.sleep(delay)

    # This line is unreachable: the loop always returns on success or raises on
    # exhaustion. The raise satisfies the type checker's control-flow analysis.
    raise RetryExhaustedError(node_id, policy.max_attempts, last_exc or RuntimeError("unknown"))  # pragma: no cover
