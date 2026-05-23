"""Human-in-the-loop node execution: interactive prompts with timeout and default-output handling."""

from __future__ import annotations

import asyncio
import sys
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from sirenspec.core.models import HumanNode
from sirenspec.exceptions import HumanInputError

InputCoroutine = Callable[[str], Awaitable[str]]


@dataclass
class HumanRunResult:
    """Result of a single :class:`~sirenspec.core.models.HumanNode` execution.

    :param response: The text collected from the operator (or the default fallback).
    :param duration_ms: Wall-clock milliseconds the node spent waiting for input.
    :param timed_out: ``True`` when ``on_timeout`` fired before a response was received.
    :param prompt_text: The fully-rendered prompt that was shown to the operator
        (or an empty string when the node had no prompt).
    """

    response: str
    duration_ms: float
    timed_out: bool
    prompt_text: str


async def stdin_input(prompt_text: str) -> str:
    """Default human-input source: print the prompt to stderr and read one line from stdin.

    Stderr is used for the rendered prompt so the workflow's own structured stdout
    (JSON traces, streamed tokens) is never mixed with the interactive prompt.
    Reading is delegated to a worker thread so the executor's event loop remains
    responsive — :func:`asyncio.wait_for` can therefore cancel the read when a
    timeout fires.

    :param prompt_text: The fully-rendered prompt to display.
    :raises HumanInputError: When stdin is closed (EOF) before any input is received.
    :returns: The line of input with its trailing newline stripped.
    """
    if prompt_text:
        sys.stderr.write(prompt_text)
        if not prompt_text.endswith("\n"):
            sys.stderr.write("\n")
        sys.stderr.flush()

    sys.stderr.write("> ")
    sys.stderr.flush()

    line = await asyncio.to_thread(sys.stdin.readline)
    if line == "":
        raise HumanInputError(node_id="<stdin>", reason="stdin closed before any input was received.")
    return line.rstrip("\n")


def resolve_timeout_fallback(node: HumanNode) -> str:
    """Return the response written when ``timeout`` expires, or raise when ``on_timeout='abort'``.

    :param node: The HumanNode whose ``on_timeout`` policy governs the fallback.
    :raises HumanInputError: When ``on_timeout='abort'``.
    :returns: ``''`` for ``'skip'`` and ``node.default_output`` for ``'use_default'``.
    """
    if node.on_timeout == "abort":
        raise HumanInputError(
            node_id="<human>",
            reason=f"timeout of {node.timeout}s expired without a response.",
            timed_out=True,
        )
    if node.on_timeout == "skip":
        return ""
    # ``use_default`` — validated at model load time, so default_output is never None here.
    assert node.default_output is not None
    return node.default_output


async def execute_human_node(
    node_id: str,
    node: HumanNode,
    prompt_text: str,
    input_fn: InputCoroutine | None = None,
) -> HumanRunResult:
    """Pause execution to collect a response from a human operator.

    The node consumes no LLM tokens.  It displays *prompt_text* via *input_fn*,
    blocks until a line is received (or *node.timeout* expires), and returns the
    response.  Timeouts are handled via :func:`asyncio.wait_for` so cancellation
    propagates cleanly to the input source.

    :param node_id: Node identifier used in error messages and the trace.
    :param node: The :class:`~sirenspec.core.models.HumanNode` definition.
    :param prompt_text: Fully-rendered prompt to show the operator.  May be empty.
    :param input_fn: Coroutine that takes the rendered prompt and returns the
        operator's response.  Defaults to reading a single line from stdin.  Tests
        and webhook integrations inject their own implementation here.
    :raises HumanInputError: When the timeout expires and ``on_timeout='abort'``,
        or when the input source signals an unrecoverable failure.
    :returns: :class:`HumanRunResult` with the response, duration, and timeout flag.
    """
    effective_input = input_fn if input_fn is not None else stdin_input
    start = time.monotonic()

    try:
        if node.timeout is None:
            response = await effective_input(prompt_text)
        else:
            response = await asyncio.wait_for(effective_input(prompt_text), timeout=node.timeout)
        timed_out = False
    except TimeoutError:
        try:
            response = resolve_timeout_fallback(node)
        except HumanInputError as exc:
            raise HumanInputError(node_id=node_id, reason=exc.reason, timed_out=True) from exc
        timed_out = True
    except HumanInputError as exc:
        raise HumanInputError(node_id=node_id, reason=exc.reason, timed_out=exc.timed_out) from exc

    duration_ms = (time.monotonic() - start) * 1000
    return HumanRunResult(
        response=response,
        duration_ms=duration_ms,
        timed_out=timed_out,
        prompt_text=prompt_text,
    )
