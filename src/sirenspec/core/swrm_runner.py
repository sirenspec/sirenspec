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
from typing import Any

from sirenspec.core.agent_runner import execute_agent_node
from sirenspec.core.models import RetryPolicy, SwrmAgent, SwrmNode, SwrmSynthesis
from sirenspec.exceptions import SwrmAgentError

_TEMPLATE_RE = re.compile(r"\{\{\s*(.+?)\s*\}\}")


def render_template(template: str, context: dict[str, Any]) -> str:
    """Render a ``{{ variable }}`` template against a flat context dict.

    Supports simple dotted-path lookups (e.g. ``{{ inputs.report }}``,
    ``{{ analyze.agents.sentiment.output }}``).  Unknown paths are left as-is
    rather than raising, so a typo in a prompt placeholder silently passes through
    instead of crashing the workflow.

    :param template: Template string containing ``{{ … }}`` placeholders.
    :param context: Flat namespace of values to substitute.
    :returns: Rendered string.
    """

    def resolve_path(path: str, ctx: dict[str, Any]) -> str:
        parts = path.split(".")
        current: Any = ctx
        for part in parts:
            if not isinstance(current, dict) or part not in current:
                # Unknown path: preserve the original placeholder so the LLM
                # still sees the template syntax and can signal the missing value.
                return "{{ " + path + " }}"
            current = current[part]
        return str(current)

    return _TEMPLATE_RE.sub(lambda m: resolve_path(m.group(1).strip(), context), template)


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

    Delegates the provider call and guardrail cycle to :func:`execute_agent_node`.
    Wraps any exception in :class:`~sirenspec.exceptions.SwrmAgentError`.

    :param agent: The :class:`~sirenspec.core.models.SwrmAgent` definition.
    :param prompt: The fully-rendered prompt string.
    :param global_guardrail_names: Workflow-level guardrail names (fallback if
        the agent does not specify its own).
    :raises SwrmAgentError: If the provider call or a guardrail raises.
    :returns: Tuple of (response text, token count, elapsed milliseconds).
    """
    # Swrm agents resolve guardrails differently from regular agent nodes:
    # - If the agent defines its own guardrails, use those.
    # - Otherwise, fall back to the workflow-level list (or [] if that's also None).
    # Note: we use `or []` here (not `or None`) because swrm agents are designed
    # to be lightweight fan-out workers — if no guardrails are configured at any
    # level, disabling them is the intended default for swrm sub-agents.
    guardrail_names = agent.guardrails if agent.guardrails is not None else (global_guardrail_names or [])
    model_uri = f"{agent.provider}:{agent.model or 'gpt-4o-mini'}"

    try:
        result = await execute_agent_node(
            node_id=agent.id,
            model_uri=model_uri,
            # Swrm agents have no system prompt — the per-agent prompt is the full context.
            system_prompt="",
            user_input=prompt,
            guardrail_names=guardrail_names,
            # No retries inside swrm fan-out; failure handling is managed by
            # run_with_semaphore, which lets all agents complete before deciding
            # whether to abort or continue.
            retry_policy=RetryPolicy(max_attempts=1),
        )
        return result.output, result.tokens, result.duration_ms
    except SwrmAgentError:
        # Already wrapped — don't double-wrap.
        raise
    except Exception as exc:
        raise SwrmAgentError(agent.id, exc) from exc


async def run_synthesis(
    synthesis: SwrmSynthesis,
    prompt: str,
    global_guardrail_names: list[str] | None,
) -> tuple[str, int, float]:
    """Execute the synthesis step and return (output, tokens, duration_ms).

    Delegates to :func:`execute_agent_node` using the synthesis provider and model.

    :param synthesis: The :class:`~sirenspec.core.models.SwrmSynthesis` definition.
    :param prompt: Fully-rendered synthesis prompt.
    :param global_guardrail_names: Workflow-level guardrail names.
    :raises Exception: Propagates any provider or guardrail error directly.
    :returns: Tuple of (response text, token count, elapsed milliseconds).
    """
    # Same guardrail fallback logic as run_single_agent: synthesis also defaults
    # to [] (not None) because it is a coordinated aggregation step, not a
    # standalone agent node that inherits workflow defaults.
    guardrail_names = synthesis.guardrails if synthesis.guardrails is not None else (global_guardrail_names or [])
    model_uri = f"{synthesis.provider}:{synthesis.model or 'gpt-4o-mini'}"

    result = await execute_agent_node(
        node_id="synthesis",
        model_uri=model_uri,
        system_prompt="",
        user_input=prompt,
        guardrail_names=guardrail_names,
        retry_policy=RetryPolicy(max_attempts=1),
    )
    return result.output, result.tokens, result.duration_ms


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

    template_ctx = build_template_context(node_id, user_input, working, output)

    agent_results: dict[str, str] = {}

    async def run_with_semaphore(agent: SwrmAgent) -> tuple[dict[str, Any], SwrmAgentError | None]:
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
                # Return (trace, None) on success — no error to propagate.
                return agent_trace, None
            except SwrmAgentError as exc:
                agent_trace["error"] = str(exc)
                # Return (trace, error) instead of raising so that asyncio.gather
                # lets all other agents finish before we decide whether to abort.
                return agent_trace, exc
            except Exception as exc:
                wrapped = SwrmAgentError(agent.id, exc)
                agent_trace["error"] = str(wrapped)
                return agent_trace, wrapped

    # asyncio.gather starts all tasks immediately; the semaphore throttles how
    # many run concurrently. gather waits for every task before returning.
    tasks = [run_with_semaphore(agent) for agent in agents]
    gathered: list[tuple[dict[str, Any], SwrmAgentError | None]] = await asyncio.gather(*tasks)

    agent_traces: list[dict[str, Any]] = []
    total_tokens = 0
    total_duration_ms = 0.0
    swrm_error: SwrmAgentError | None = None

    for agent_trace, exc in gathered:
        agent_traces.append(agent_trace)
        if exc is not None:
            # Record only the first failure for the abort decision; subsequent
            # failures are already captured in agent_trace["error"].
            if node.on_failure == "abort" and swrm_error is None:
                swrm_error = exc
        else:
            agent_id = agent_trace["id"]
            agent_results[agent_id] = agent_trace["response_received"] or ""
            total_tokens += agent_trace["tokens"]
            total_duration_ms += agent_trace["duration_ms"]

    # Raise after all agents have completed (not inline in run_with_semaphore)
    # so the trace contains every agent's result even on partial failure.
    if swrm_error is not None:
        raise swrm_error

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
