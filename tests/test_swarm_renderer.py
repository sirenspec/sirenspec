"""Unit tests for swarm node CLI rendering (issue #63)."""

from __future__ import annotations

import io
from typing import Any

from rich.box import ASCII, ROUNDED
from rich.console import Console

from sirenspec.cli.run import (
    format_agent_duration,
    render_node_panel,
    render_swarm_agent_panel,
    render_swarm_panel,
)
from sirenspec.core.events import NodeCompleteEvent


def _make_agent(
    agent_id: str = "agent1",
    *,
    duration_ms: float = 800.0,
    error: str | None = None,
    response: str = "test response",
) -> dict[str, Any]:
    return {
        "id": agent_id,
        "prompt_sent": "test prompt",
        "response_received": None if error else response,
        "tokens": 10,
        "duration_ms": duration_ms,
        "error": error,
    }


def _make_swarm_event(
    *,
    node_id: str = "generate_summaries",
    agents: list[dict[str, Any]] | None = None,
    output: Any = "combined result",
    status: str = "success",
    error: str | None = None,
    duration_ms: float = 1100.0,
    writes: str = "output.generate_summaries",
) -> NodeCompleteEvent:
    if agents is None:
        agents = [
            _make_agent("chunk_1", duration_ms=800.0),
            _make_agent("chunk_2", duration_ms=1100.0),
            _make_agent("chunk_3", duration_ms=900.0),
            _make_agent("chunk_4", duration_ms=1000.0),
        ]
    return NodeCompleteEvent(
        node_id=node_id,
        node_type="swrm",
        output=output,
        writes=writes,
        status=status,  # type: ignore[arg-type]
        error=error,
        tokens=42,
        agents=agents,
        duration_ms=duration_ms,
    )


def _capture(event: NodeCompleteEvent, *, tty: bool = False) -> str:
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=tty, no_color=not tty)
    box_style = ROUNDED if tty else ASCII
    render_node_panel(event, console, box_style)
    return buf.getvalue()


class TestRenderSwarmPanelHeader:
    def test_header_contains_node_id(self) -> None:
        output = _capture(_make_swarm_event(node_id="generate_summaries"))
        assert "generate_summaries" in output

    def test_header_contains_agent_count(self) -> None:
        output = _capture(_make_swarm_event())
        assert "4 agents" in output

    def test_header_singular_agent(self) -> None:
        output = _capture(_make_swarm_event(agents=[_make_agent()]))
        assert "1 agent" in output
        assert "1 agents" not in output


class TestRenderSwarmPanelFooter:
    def test_footer_shows_success_ratio(self) -> None:
        agents = [
            _make_agent("a1"),
            _make_agent("a2"),
            _make_agent("a3"),
            _make_agent("a4", error="GuardrailError"),
        ]
        output = _capture(_make_swarm_event(agents=agents))
        assert "3/4 succeeded" in output

    def test_footer_shows_total_duration(self) -> None:
        output = _capture(_make_swarm_event(duration_ms=1100.0))
        assert "1.1s total" in output

    def test_footer_omits_failed_when_all_succeed(self) -> None:
        output = _capture(_make_swarm_event())
        assert "Swarm complete:" in output
        assert "failed" not in output

    def test_footer_includes_failed_count_when_partial(self) -> None:
        agents = [_make_agent("a1"), _make_agent("a2", error="timeout")]
        output = _capture(_make_swarm_event(agents=agents))
        assert "1 failed" in output


class TestRenderSwarmAgentPanels:
    def test_agent_output_shown_in_panel(self) -> None:
        agents = [_make_agent("worker", response="hello world")]
        output = _capture(_make_swarm_event(agents=agents))
        assert "hello world" in output

    def test_all_agent_ids_present(self) -> None:
        agents = [_make_agent(f"agent_{i}") for i in range(3)]
        output = _capture(_make_swarm_event(agents=agents))
        for i in range(3):
            assert f"agent_{i}" in output

    def test_failed_agent_shows_error_text(self) -> None:
        output = _capture(_make_swarm_event(agents=[_make_agent("worker", error="GuardrailError")]))
        assert "GuardrailError" in output

    def test_failed_agent_id_in_output(self) -> None:
        output = _capture(_make_swarm_event(agents=[_make_agent("bad_agent", error="timeout")]))
        assert "bad_agent" in output


class TestRenderSwarmPanelSynthesis:
    def test_synthesis_panel_shown_when_output_is_string(self) -> None:
        output = _capture(_make_swarm_event(output="synthesised text", status="success"))
        assert "synthesised text" in output

    def test_synthesis_panel_not_shown_when_output_is_list(self) -> None:
        outputs = ["result a", "result b"]
        output = _capture(_make_swarm_event(output=outputs, status="success"))
        assert "synthesised text" not in output

    def test_synthesis_panel_not_shown_on_failure(self) -> None:
        output = _capture(_make_swarm_event(output="some output", status="failed"))
        assert "some output" not in output

    def test_synthesis_panel_uses_node_id_as_title(self) -> None:
        output = _capture(_make_swarm_event(node_id="my_swarm", output="result", status="success"))
        assert output.count("my_swarm") >= 2


class TestRenderSwarmPanelWritesArrow:
    def test_writes_arrow_after_success(self) -> None:
        output = _capture(_make_swarm_event(status="success", writes="output.foo"))
        assert "↓ output.foo" in output

    def test_writes_arrow_omitted_on_failure(self) -> None:
        output = _capture(_make_swarm_event(status="failed", error="boom", writes="output.foo"))
        assert "↓" not in output


class TestRenderNodePanelDispatch:
    def test_dispatches_swarm_when_agents_populated(self) -> None:
        output = _capture(_make_swarm_event())
        assert "Swarm:" in output
        assert "Swarm complete:" in output

    def test_falls_through_to_panel_when_agents_none(self) -> None:
        event = NodeCompleteEvent(
            node_id="analyze",
            node_type="swrm",
            status="failed",
            error="SwrmAgentError: agent failed",
            agents=None,
        )
        output = _capture(event)
        assert "SwrmAgentError" in output
        assert "Swarm:" not in output

    def test_non_swrm_node_unaffected(self) -> None:
        event = NodeCompleteEvent(
            node_id="classify",
            node_type="agent",
            output="positive",
            writes="working.sentiment",
            status="success",
            tokens=15,
        )
        output = _capture(event)
        assert "classify" in output
        assert "Swarm:" not in output


class TestRenderSwarmAgentPanelDirect:
    def test_success_agent_renders_response(self) -> None:
        agent = _make_agent("worker", response="the answer")
        buf = io.StringIO()
        console = Console(file=buf, force_terminal=False, no_color=True)
        render_swarm_agent_panel(agent, console, ASCII)
        output = buf.getvalue()
        assert "the answer" in output
        assert "worker" in output

    def test_failed_agent_renders_error(self) -> None:
        agent = _make_agent("worker", error="LengthError")
        buf = io.StringIO()
        console = Console(file=buf, force_terminal=False, no_color=True)
        render_swarm_agent_panel(agent, console, ASCII)
        output = buf.getvalue()
        assert "LengthError" in output


class TestRenderSwarmPanelDirectCall:
    def test_direct_call_produces_header_and_footer(self) -> None:
        event = _make_swarm_event()
        buf = io.StringIO()
        console = Console(file=buf, force_terminal=False, no_color=True)
        render_swarm_panel(event, console, ASCII)
        output = buf.getvalue()
        assert "Swarm:" in output
        assert "Swarm complete:" in output


class TestFormatAgentDuration:
    def test_sub_second(self) -> None:
        assert format_agent_duration(800.0) == "0.8s"

    def test_multi_second(self) -> None:
        assert format_agent_duration(12300.0) == "12.3s"

    def test_zero(self) -> None:
        assert format_agent_duration(0.0) == "0.0s"
