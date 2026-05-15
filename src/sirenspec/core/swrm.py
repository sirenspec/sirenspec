"""Swrm (parallel agent fan-out) execution engine.

A swrm node fans out to a static set of author-defined agents that run
concurrently (up to an optional ``concurrency`` limit).  After all agents
complete, an optional ``synthesis`` step aggregates their outputs.

Template interpolation uses ``{{ variable }}`` syntax.  The context namespace
exposed to agent prompts and the synthesis prompt includes the full workflow
``working`` and ``output`` dicts plus a synthetic ``inputs`` alias for the
user-provided workflow input string (as ``inputs.message``).

Individual agent outputs are stored in the context as::

    working.<node_id>.agents.<agent_id>.output

The node's canonical output is::

    output.<node_id>   — synthesis output, or list of agent outputs if no synthesis
"""

from __future__ import annotations

import asyncio
import re
import time
from typing import Any

from sirenspec.core.models import SwrmAgent, SwrmNode, SwrmSynthesis
from sirenspec.exceptions import SwrmAgentError
from sirenspec.guardrails.registry import build_guardrails
from sirenspec.providers.registry import resolve_provider

_TEMPLATE_RE = re.compile(r"\{\{\s*(.+?)\s*\}\}")


def render_template(template: str, context: dict[str, Any]) -> str:
    """Render a ``{{ variable }}`` template against a flat context dict.

    Supports simple dotted-path lookups (e.g. ``{{ inputs.report }}``,
    ``{{ analyze.agents.sentiment.output }}``).  Unknown paths are left as-is.

    :param template: Template string containing ``{{ … }}`` placeholders.
    :param context: Flat namespace of values to substitute.
    :returns: Rendered string.
    """

    def _resolve(path: str, ctx: dict[str, Any]) -> str:
        parts = path.split(".")
        current: Any = ctx
        for part in parts:
            if not isinstance(current, dict) or part not in current:
                # Return the original placeholder so missing keys are visible.
                return "{{ " + path + " }}"
            current = current[part]
        return str(current)

    return _TEMPLATE_RE.sub(lambda m: _resolve(m.group(1).strip(), context), template)


def build_template_context(
    node_id: str,
    user_input: str,
    working: dict[str, Any],
    output: dict[str, Any],
    agent_results: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build the interpolation context for swrm prompts.

    :param node_id: The swrm node's identifier.
    :param user_input: Raw user input string passed to the workflow.
    :param working: Current ``working`` context dict.
    :param output: Current ``output`` context dict.
    :param agent_results: Mapping of agent id → output string (populated after agents run).
    :returns: Flat namespace dict suitable for :func:`render_template`.
    """
    ctx: dict[str, Any] = {
        "inputs": {"message": user_input},
        "working": working,
        "output": output,
    }
    if agent_results is not None:
        agents_ctx: dict[str, Any] = {aid: {"output": out} for aid, out in agent_results.items()}
        ctx[node_id] = {"agents": agents_ctx}
    return ctx


async def run_single_agent(
    agent: SwrmAgent,
    prompt: str,
    global_guardrail_names: list[str] | None,
) -> tuple[str, int, float]:
    """Execute a single swrm agent and return (output, tokens, duration_ms).

    :param agent: The :class:`~sirenspec.core.models.SwrmAgent` definition.
    :param prompt: The fully-rendered prompt string.
    :param global_guardrail_names: Workflow-level guardrail names (fallback if
        the agent does not specify its own).
    :raises SwrmAgentError: If the provider call or a guardrail raises.
    :returns: Tuple of (response text, token count, elapsed milliseconds).
    """
    guardrail_names = agent.guardrails if agent.guardrails is not None else global_guardrail_names
    guardrails = build_guardrails(guardrail_names)

    model_str = agent.model or "gpt-4o-mini"
    uri = f"{agent.provider}:{model_str}"

    start = time.monotonic()
    try:
        checked_input = prompt
        for g in guardrails:
            checked_input = g.check_input(checked_input)

        messages = [{"role": "user", "content": checked_input}]
        provider = resolve_provider(uri)
        response_text = await provider.complete(messages)
        tokens: int = provider.last_token_count

        checked_output = response_text
        for g in guardrails:
            checked_output = g.check_output(checked_output)

        duration_ms = (time.monotonic() - start) * 1000
        return checked_output, tokens, duration_ms
    except SwrmAgentError:
        raise
    except Exception as exc:
        duration_ms = (time.monotonic() - start) * 1000
        raise SwrmAgentError(agent.id, exc) from exc


async def run_synthesis(
    synthesis: SwrmSynthesis,
    prompt: str,
    global_guardrail_names: list[str] | None,
) -> tuple[str, int, float]:
    """Execute the synthesis step and return (output, tokens, duration_ms).

    :param synthesis: The :class:`~sirenspec.core.models.SwrmSynthesis` definition.
    :param prompt: Fully-rendered synthesis prompt.
    :param global_guardrail_names: Workflow-level guardrail names.
    :raises Exception: Propagates any provider or guardrail error directly.
    :returns: Tuple of (response text, token count, elapsed milliseconds).
    """
    guardrail_names = synthesis.guardrails if synthesis.guardrails is not None else global_guardrail_names
    guardrails = build_guardrails(guardrail_names)

    model_str = synthesis.model or "gpt-4o-mini"
    uri = f"{synthesis.provider}:{model_str}"

    start = time.monotonic()
    checked_input = prompt
    for g in guardrails:
        checked_input = g.check_input(checked_input)

    messages = [{"role": "user", "content": checked_input}]
    provider = resolve_provider(uri)
    response_text = await provider.complete(messages)
    tokens: int = provider.last_token_count

    checked_output = response_text
    for g in guardrails:
        checked_output = g.check_output(checked_output)

    duration_ms = (time.monotonic() - start) * 1000
    return checked_output, tokens, duration_ms


async def execute_swrm(
    node_id: str,
    node: SwrmNode,
    user_input: str,
    working: dict[str, Any],
    output: dict[str, Any],
    global_guardrail_names: list[str] | None,
) -> dict[str, Any]:
    """Execute a swrm node and return a structured trace dict.

    The returned trace has the shape::

        {
            "id": "<node_id>",
            "type": "swrm",
            "agents": [
                {
                    "id": "<agent_id>",
                    "prompt_sent": "...",
                    "response_received": "...",  # None if failed and on_failure=continue
                    "tokens": 42,
                    "duration_ms": 123.4,
                    "error": None,  # or error string
                },
                ...
            ],
            "synthesis": {  # only present when synthesis block is defined
                "prompt_sent": "...",
                "response_received": "...",
                "tokens": 10,
                "duration_ms": 50.0,
                "error": None,
            },
            "output": "...",   # synthesis output, or list of agent outputs
            "tokens": 52,      # total token count
            "duration_ms": 173.4,
            "error": None,
        }

    :param node_id: The node's ID as declared in the workflow.
    :param node: The validated :class:`~sirenspec.core.models.SwrmNode`.
    :param user_input: Raw user input string.
    :param working: Current ``working`` context dict (read-only snapshot).
    :param output: Current ``output`` context dict (read-only snapshot).
    :param global_guardrail_names: Workflow-level guardrail list.
    :raises SwrmAgentError: If ``on_failure`` is ``"abort"`` and any agent fails.
    :returns: Structured trace dict for the swrm node.
    """
    agents = node.agents
    concurrency = node.concurrency if node.concurrency is not None else len(agents)
    semaphore = asyncio.Semaphore(concurrency)

    # Build per-agent prompts from the template context (no agent results yet).
    template_ctx = build_template_context(node_id, user_input, working, output)

    agent_traces: list[dict[str, Any]] = []
    agent_results: dict[str, str] = {}  # agent_id → output

    # Each coroutine returns (agent_trace_dict, exception_or_None).
    # We never raise inside gather so we always collect all results.
    async def _run_with_semaphore(agent: SwrmAgent) -> tuple[dict[str, Any], SwrmAgentError | None]:
        rendered_prompt = render_template(agent.prompt, template_ctx)
        agent_trace: dict[str, Any] = {
            "id": agent.id,
            "prompt_sent": rendered_prompt,
            "response_received": None,
            "tokens": 0,
            "duration_ms": 0.0,
            "error": None,
        }
        async with semaphore:
            try:
                text, tok, dur = await run_single_agent(agent, rendered_prompt, global_guardrail_names)
                agent_trace.update({"response_received": text, "tokens": tok, "duration_ms": round(dur, 2)})
                return agent_trace, None
            except SwrmAgentError as exc:
                agent_trace["error"] = str(exc)
                return agent_trace, exc
            except Exception as exc:
                wrapped = SwrmAgentError(agent.id, exc)
                agent_trace["error"] = str(wrapped)
                return agent_trace, wrapped

    # Gather all agents concurrently; each coroutine returns a (trace, error) tuple.
    tasks = [_run_with_semaphore(agent) for agent in agents]
    gathered: list[tuple[dict[str, Any], SwrmAgentError | None]] = await asyncio.gather(*tasks)

    total_tokens = 0
    total_duration_ms = 0.0
    swrm_error: SwrmAgentError | None = None

    for agent_trace, exc in gathered:
        agent_traces.append(agent_trace)
        if exc is not None:
            if node.on_failure == "abort" and swrm_error is None:
                swrm_error = exc
            # on_failure == "continue": just record the error in the trace.
        else:
            agent_id = agent_trace["id"]
            agent_results[agent_id] = agent_trace["response_received"] or ""
            total_tokens += agent_trace["tokens"]
            total_duration_ms += agent_trace["duration_ms"]

    # Abort policy: re-raise the first agent error after all results are collected.
    if swrm_error is not None:
        raise swrm_error

    # Build synthesis prompt template context now that all agent outputs are known.
    synthesis_trace: dict[str, Any] | None = None
    final_output: Any

    if node.synthesis is not None:
        synth_ctx = build_template_context(node_id, user_input, working, output, agent_results)
        rendered_synthesis_prompt = render_template(node.synthesis.prompt, synth_ctx)
        synthesis_trace = {
            "prompt_sent": rendered_synthesis_prompt,
            "response_received": None,
            "tokens": 0,
            "duration_ms": 0.0,
            "error": None,
        }
        try:
            synth_text, synth_tok, synth_dur = await run_synthesis(
                node.synthesis, rendered_synthesis_prompt, global_guardrail_names
            )
            synthesis_trace.update(
                {
                    "response_received": synth_text,
                    "tokens": synth_tok,
                    "duration_ms": round(synth_dur, 2),
                }
            )
            total_tokens += synth_tok
            total_duration_ms += synth_dur
            final_output = synth_text
        except Exception as exc:
            synthesis_trace["error"] = str(exc)
            raise

    else:
        # No synthesis: output is a list of agent outputs in definition order.
        final_output = [agent_results.get(agent.id, "") for agent in agents]

    return {
        "id": node_id,
        "type": "swrm",
        "agents": agent_traces,
        "synthesis": synthesis_trace,
        "output": final_output,
        "tokens": total_tokens,
        "duration_ms": round(total_duration_ms, 2),
        "error": None,
    }
