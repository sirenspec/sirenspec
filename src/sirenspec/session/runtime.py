"""Production-style testing runtime for ``sirenspec launch``.

:class:`WorkflowSession` holds live conversation state and drives the workflow against real
providers, one turn at a time, exactly as it would run in production.  Each user message is a
turn; prior turns are threaded back into the agent input so agents retain context across the
session.  The runtime streams tokens, reports per-node attribution (provider, tokens, latency,
cost), hot-reloads on file save without losing history, and can rehydrate a session from the
workflow's ``memory:`` backend via ``--session``.
"""

from __future__ import annotations

import time
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sirenspec.core.events import NodeCompleteEvent, SummaryEvent
from sirenspec.core.executor import execute_streaming
from sirenspec.core.human_runner import InputCoroutine
from sirenspec.core.models import AgentNode, SwrmNode, Workflow
from sirenspec.exceptions import SessionError, SirenSpecError
from sirenspec.memory.manager import build_store
from sirenspec.session.summary import provider_of
from sirenspec.yaml.parser import load_workflow

# Key prefix under which a named session's conversation history is persisted in the
# workflow's memory backend, so ``--session <id>`` can rehydrate it on a later launch.
SESSION_KEY_PREFIX = "launch_session_"

TokenCallback = Callable[[str], None]


@dataclass
class Turn:
    """A single conversational turn in a session.

    :param role: ``"user"`` for a person's message or ``"siren"`` for the workflow's reply.
    :param content: The message text.
    """

    role: str
    content: str

    def to_dict(self) -> dict[str, str]:
        """Serialise this turn for persistence.

        :returns: A JSON-serialisable ``{"role", "content"}`` dict.
        """
        return {"role": self.role, "content": self.content}


@dataclass
class TurnNodeEvent:
    """Per-node attribution emitted while a turn executes.

    :param node_id: The node that completed.
    :param node_type: The node type (``"agent"``, ``"swrm"`` …).
    :param label: A display label such as ``"analyze ▸ risk (anthropic)"``.
    :param provider: The provider that served the node, or ``""``.
    :param tokens: Tokens consumed by the node.
    :param status: ``"success"``, ``"skipped"``, or ``"failed"``.
    """

    node_id: str
    node_type: str
    label: str
    provider: str
    tokens: int
    status: str


@dataclass
class TurnResult:
    """The outcome of a completed turn (or full ``/run``).

    :param text: The final reply text (the last successful node output).
    :param tokens: Total tokens consumed during the turn.
    :param cost_usd: Estimated USD cost, or ``None`` when pricing is unavailable.
    :param duration_s: Wall-clock duration of the turn in seconds.
    :param status: ``"success"`` or ``"failed"``.
    :param last_label: Attribution label of the final node, for the trace drawer.
    """

    text: str
    tokens: int
    cost_usd: float | None
    duration_s: float
    status: str
    last_label: str = ""


def build_turn_input(history: list[Turn], message: str) -> str:
    """Compose the effective agent input for a new turn, threading prior turns.

    On the first turn the raw *message* is returned.  On later turns the prior turns are
    prepended as a labelled transcript so agents see the conversation so far — this is how
    agents retain state across turns without changing the executor's single-input contract.

    :param history: The session's prior turns, oldest first.
    :param message: The new user message.
    :returns: The composed input string passed to the workflow.
    """
    if not history:
        return message
    lines = ["Conversation so far:"]
    for turn in history:
        speaker = "You" if turn.role == "user" else "Siren"
        lines.append(f"{speaker}: {turn.content}")
    lines.append("")
    lines.append(f"You: {message}")
    return "\n".join(lines)


def format_history_transcript(history: list[Turn]) -> str:
    """Render prior turns as a plain ``role: content`` transcript for the ``inputs.history`` template slot.

    Mirrors the contract used by demo hosts (e.g. claude-code-mini's ``main.py``) so workflows
    written for those hosts behave the same inside the studio.

    :param history: The session's prior turns, oldest first.
    :returns: A newline-joined transcript, or ``""`` when there are no prior turns.
    """
    if not history:
        return ""
    return "\n".join(f"{turn.role}: {turn.content}" for turn in history)


def node_attribution_label(workflow: Workflow, node_id: str) -> tuple[str, str]:
    """Build a human-facing attribution label and provider for a node.

    :param workflow: The workflow being executed.
    :param node_id: The node's identifier.
    :returns: A ``(label, provider)`` tuple, e.g. ``("analyze ▸ risk (anthropic)", "anthropic")``.
    """
    node = workflow.nodes.get(node_id)
    if isinstance(node, AgentNode):
        agent = workflow.agents.get(node.agent)
        provider = provider_of(agent.model) if agent is not None else ""
        return (f"{node_id} ({provider})" if provider else node_id), provider
    if isinstance(node, SwrmNode) and node.agents:
        first = node.agents[0]
        return f"{node_id} ▸ {first.id} ({first.provider})", first.provider
    return node_id, ""


async def stream_execution(
    workflow: Workflow,
    user_input: str,
    token_callback: TokenCallback | None,
    human_input_fn: InputCoroutine | None = None,
    initial_inputs: dict[str, Any] | None = None,
) -> AsyncGenerator[TurnNodeEvent | TurnResult]:
    """Run *workflow* once against *user_input*, yielding node attribution then a result.

    Tokens are forwarded to *token_callback* as they stream from the provider.  A
    :class:`TurnNodeEvent` is yielded for each node that runs (skipped nodes are omitted),
    and a final :class:`TurnResult` carries the reply text and turn totals.

    :param workflow: The workflow to execute.
    :param user_input: The already-composed input string for this turn.
    :param token_callback: Optional callback invoked with each streamed text chunk.
    :returns: An async generator of node events followed by a single :class:`TurnResult`.
    """
    start = time.monotonic()
    final_text = ""
    last_label = ""
    tokens = 0
    cost_usd: float | None = None
    status = "success"

    async for event in execute_streaming(
        workflow,
        user_input,
        stream_callback=token_callback,
        human_input_fn=human_input_fn,
        initial_inputs=initial_inputs,
    ):
        if isinstance(event, NodeCompleteEvent):
            if event.status == "skipped":
                continue
            label, provider = node_attribution_label(workflow, event.node_id)
            if event.status == "success" and event.output is not None:
                final_text = stringify(event.output)
                last_label = label
            elif event.status == "failed" and event.error:
                final_text = event.error
                last_label = label
            yield TurnNodeEvent(
                node_id=event.node_id,
                node_type=event.node_type,
                label=label,
                provider=provider,
                tokens=event.tokens,
                status=event.status,
            )
        elif isinstance(event, SummaryEvent):
            tokens = event.total_tokens
            cost_usd = event.estimated_usd
            status = event.status

    yield TurnResult(
        text=final_text,
        tokens=tokens,
        cost_usd=cost_usd,
        duration_s=time.monotonic() - start,
        status=status,
        last_label=last_label,
    )


def stringify(output: Any) -> str:
    """Render a node output value as plain text for the transcript.

    :param output: The node output (string, dict, list, or scalar).
    :returns: A printable string.
    """
    if isinstance(output, str):
        return output
    if isinstance(output, list):
        return "\n".join(stringify(item) for item in output)
    return str(output) if output is not None else ""


class WorkflowSession:
    """Live, multi-turn testing session for one workflow.

    Holds conversation history and accumulated cost/token totals across turns, drives the
    workflow against real providers, hot-reloads on file save, and (when ``session_id`` and a
    ``memory:`` backend are present) persists and rehydrates the conversation.

    :param workflow: The validated workflow to test.
    :param workflow_path: Filesystem path to the workflow YAML (for hot-reload).
    :param name: Human-facing workflow name.
    :param session_id: Optional id for persistent, resumable sessions via ``--session``.
    """

    def __init__(
        self,
        workflow: Workflow,
        workflow_path: Path,
        name: str,
        session_id: str | None = None,
    ) -> None:
        self.workflow = workflow
        self.workflow_path = workflow_path
        self.name = name
        self.session_id = session_id
        self.history: list[Turn] = []
        self.turns = 0
        self.total_tokens = 0
        self.session_cost_usd: float | None = None
        self.mtime = self.current_mtime()
        self.store = self.open_store()
        if self.store is not None:
            self.history = self.load_history()
            self.turns = sum(1 for turn in self.history if turn.role == "user")

    def open_store(self) -> Any | None:
        """Open the memory backend used for session persistence, if applicable.

        :raises SessionError: If ``--session`` was requested but the workflow declares no
            ``memory:`` backend to persist into.
        :returns: An initialised memory store, or ``None`` when persistence is not used.
        """
        if self.session_id is None:
            return None
        if self.workflow.memory is None:
            raise SessionError("--session requires the workflow to declare a 'memory:' backend to persist into.")
        return build_store(self.workflow.memory)

    def session_key(self) -> str:
        """Return the memory key under which this session's history is stored.

        :returns: The namespaced session key.
        """
        return f"{SESSION_KEY_PREFIX}{self.session_id}"

    def load_history(self) -> list[Turn]:
        """Load persisted conversation history from the memory backend.

        :returns: The rehydrated turns, or an empty list when none are stored.
        """
        if self.store is None:
            return []
        raw = self.store.get(self.session_key())
        if not isinstance(raw, list):
            return []
        return [Turn(role=item["role"], content=item["content"]) for item in raw]

    def persist(self) -> None:
        """Persist the current conversation history to the memory backend (if any)."""
        if self.store is None:
            return
        self.store.set(self.session_key(), [turn.to_dict() for turn in self.history])

    def record_turn(self, message: str, result: TurnResult) -> None:
        """Fold a completed turn's outcome into session history and running totals.

        :param message: The user message that drove the turn.
        :param result: The turn's :class:`TurnResult`.
        """
        self.history.append(Turn(role="user", content=message))
        self.history.append(Turn(role="siren", content=result.text))
        self.turns += 1
        self.total_tokens += result.tokens
        if result.cost_usd is not None:
            self.session_cost_usd = (self.session_cost_usd or 0.0) + result.cost_usd
        self.persist()

    async def take_turn(
        self,
        message: str,
        token_callback: TokenCallback | None = None,
        human_input_fn: InputCoroutine | None = None,
    ) -> AsyncGenerator[TurnNodeEvent | TurnResult]:
        """Run one chat turn, threading prior turns so agents retain context.

        :param message: The user's message for this turn.
        :param token_callback: Optional callback for streamed text chunks.
        :returns: An async generator of node events followed by a single :class:`TurnResult`.
        """
        effective_input = build_turn_input(self.history, message)
        history_text = format_history_transcript(self.history)
        result: TurnResult | None = None
        async for event in stream_execution(
            self.workflow,
            effective_input,
            token_callback,
            human_input_fn,
            initial_inputs={"history": history_text, "message": message},
        ):
            if isinstance(event, TurnResult):
                result = event
            else:
                yield event
        if result is None:  # pragma: no cover — stream_execution always yields a result
            raise SessionError("Turn produced no result.")
        self.record_turn(message, result)
        yield result

    async def run_full(
        self,
        token_callback: TokenCallback | None = None,
        human_input_fn: InputCoroutine | None = None,
    ) -> AsyncGenerator[TurnNodeEvent | TurnResult]:
        """Execute the full workflow graph end-to-end with its declared input.

        Unlike :meth:`take_turn`, this runs the workflow as it would in production — using
        the workflow's own ``input.message`` (or the most recent user turn as a fallback) —
        and does not append to the chat history.

        :raises SessionError: If no input can be resolved for the run.
        :returns: An async generator of node events followed by a single :class:`TurnResult`.
        """
        user_input = self.resolve_run_input()
        history_text = format_history_transcript(self.history)
        async for event in stream_execution(
            self.workflow,
            user_input,
            token_callback,
            human_input_fn,
            initial_inputs={"history": history_text, "message": user_input},
        ):
            if isinstance(event, TurnResult):
                self.total_tokens += event.tokens
                if event.cost_usd is not None:
                    self.session_cost_usd = (self.session_cost_usd or 0.0) + event.cost_usd
            yield event

    def resolve_run_input(self) -> str:
        """Resolve the input for a full ``/run`` from the workflow or recent history.

        :raises SessionError: If the workflow has no declared input and no prior turn exists.
        :returns: The input string for the run.
        """
        if self.workflow.input is not None and self.workflow.input.message:
            return self.workflow.input.message
        for turn in reversed(self.history):
            if turn.role == "user":
                return turn.content
        raise SessionError("No input to run: define input.message in the workflow or send a message first.")

    def current_mtime(self) -> float:
        """Return the workflow file's last-modified time, or ``0.0`` if it is missing.

        :returns: The modification time in seconds since the epoch.
        """
        try:
            return self.workflow_path.stat().st_mtime
        except OSError:
            return 0.0

    def file_changed(self) -> bool:
        """Return whether the workflow file changed on disk since the last reload.

        :returns: ``True`` if the file's mtime advanced past the recorded value.
        """
        return self.current_mtime() > self.mtime

    def reload(self) -> None:
        """Reload the workflow from disk, preserving conversation history.

        :raises SessionError: If the file can no longer be loaded or validated.
        """
        try:
            self.workflow = load_workflow(str(self.workflow_path))
        except (SirenSpecError, OSError, ValueError) as exc:
            raise SessionError(f"Reload failed: {exc}") from exc
        self.mtime = self.current_mtime()

    def close(self) -> None:
        """Persist and release the session's memory backend, if any."""
        if self.store is not None:
            self.persist()
            self.store.close()


# Re-export for callers that build totals from a TurnResult stream.
__all__ = [
    "Turn",
    "TurnNodeEvent",
    "TurnResult",
    "WorkflowSession",
    "build_turn_input",
    "format_history_transcript",
    "node_attribution_label",
    "stream_execution",
]
