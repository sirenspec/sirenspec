"""Unit tests for swarm and factory node CLI rendering."""

from __future__ import annotations

import io
from typing import Any

from rich.box import ASCII, ROUNDED
from rich.console import Console

from sirenspec.cli.run import (
    format_agent_duration,
    render_factory_instance_panel,
    render_factory_panel,
    render_node_panel,
    render_swarm_agent_panel,
    render_swarm_panel,
)
from sirenspec.core.events import NodeCompleteEvent


def make_agent(
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


def make_swarm_event(
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
            make_agent("chunk_1", duration_ms=800.0),
            make_agent("chunk_2", duration_ms=1100.0),
            make_agent("chunk_3", duration_ms=900.0),
            make_agent("chunk_4", duration_ms=1000.0),
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


def capture_render(event: NodeCompleteEvent, *, tty: bool = False) -> str:
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=tty, no_color=not tty)
    box_style = ROUNDED if tty else ASCII
    render_node_panel(event, console, box_style)
    return buf.getvalue()


class TestRenderSwarmPanelHeader:
    def test_header_contains_node_id(self) -> None:
        output = capture_render(make_swarm_event(node_id="generate_summaries"))
        assert "generate_summaries" in output

    def test_header_contains_agent_count(self) -> None:
        output = capture_render(make_swarm_event())
        assert "4 agents" in output

    def test_header_singular_agent(self) -> None:
        output = capture_render(make_swarm_event(agents=[make_agent()]))
        assert "1 agent" in output
        assert "1 agents" not in output


class TestRenderSwarmPanelFooter:
    def test_footer_shows_success_ratio(self) -> None:
        agents = [
            make_agent("a1"),
            make_agent("a2"),
            make_agent("a3"),
            make_agent("a4", error="GuardrailError"),
        ]
        output = capture_render(make_swarm_event(agents=agents))
        assert "3/4 succeeded" in output

    def test_footer_shows_total_duration(self) -> None:
        output = capture_render(make_swarm_event(duration_ms=1100.0))
        assert "1.1s total" in output

    def test_footer_omits_failed_when_all_succeed(self) -> None:
        output = capture_render(make_swarm_event())
        assert "Swarm complete:" in output
        assert "failed" not in output

    def test_footer_includes_failed_count_when_partial(self) -> None:
        agents = [make_agent("a1"), make_agent("a2", error="timeout")]
        output = capture_render(make_swarm_event(agents=agents))
        assert "1 failed" in output


class TestRenderSwarmAgentPanels:
    def test_agent_output_shown_in_panel(self) -> None:
        agents = [make_agent("worker", response="hello world")]
        output = capture_render(make_swarm_event(agents=agents))
        assert "hello world" in output

    def test_all_agent_ids_present(self) -> None:
        agents = [make_agent(f"agent_{i}") for i in range(3)]
        output = capture_render(make_swarm_event(agents=agents))
        for i in range(3):
            assert f"agent_{i}" in output

    def test_failed_agent_shows_error_text(self) -> None:
        output = capture_render(make_swarm_event(agents=[make_agent("worker", error="GuardrailError")]))
        assert "GuardrailError" in output

    def test_failed_agent_id_in_output(self) -> None:
        output = capture_render(make_swarm_event(agents=[make_agent("bad_agent", error="timeout")]))
        assert "bad_agent" in output


class TestRenderSwarmPanelSynthesis:
    def test_synthesis_panel_shown_when_output_is_string(self) -> None:
        output = capture_render(make_swarm_event(output="synthesised text", status="success"))
        assert "synthesised text" in output

    def test_synthesis_panel_not_shown_when_output_is_list(self) -> None:
        output = capture_render(make_swarm_event(output=["result a", "result b"], status="success"))
        assert "synthesised text" not in output

    def test_synthesis_panel_not_shown_on_failure(self) -> None:
        output = capture_render(make_swarm_event(output="some output", status="failed"))
        assert "some output" not in output

    def test_synthesis_panel_uses_node_id_as_title(self) -> None:
        output = capture_render(make_swarm_event(node_id="my_swarm", output="result", status="success"))
        assert output.count("my_swarm") >= 2


class TestRenderSwarmPanelWritesArrow:
    def test_writes_arrow_after_success(self) -> None:
        output = capture_render(make_swarm_event(status="success", writes="output.foo"))
        assert "↓ output.foo" in output

    def test_writes_arrow_omitted_on_failure(self) -> None:
        output = capture_render(make_swarm_event(status="failed", error="boom", writes="output.foo"))
        assert "↓" not in output


class TestRenderNodePanelDispatch:
    def test_dispatches_swarm_when_agents_populated(self) -> None:
        output = capture_render(make_swarm_event())
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
        output = capture_render(event)
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
        output = capture_render(event)
        assert "classify" in output
        assert "Swarm:" not in output


class TestRenderSwarmAgentPanelDirect:
    def test_success_agent_renders_response(self) -> None:
        agent = make_agent("worker", response="the answer")
        buf = io.StringIO()
        console = Console(file=buf, force_terminal=False, no_color=True)
        render_swarm_agent_panel(agent, console, ASCII)
        output = buf.getvalue()
        assert "the answer" in output
        assert "worker" in output

    def test_failed_agent_renders_error(self) -> None:
        agent = make_agent("worker", error="LengthError")
        buf = io.StringIO()
        console = Console(file=buf, force_terminal=False, no_color=True)
        render_swarm_agent_panel(agent, console, ASCII)
        output = buf.getvalue()
        assert "LengthError" in output


class TestRenderSwarmPanelDirectCall:
    def test_direct_call_produces_header_and_footer(self) -> None:
        event = make_swarm_event()
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


# ---------------------------------------------------------------------------
# Factory renderer tests
# ---------------------------------------------------------------------------


def make_instance(
    index: int = 0,
    *,
    item: str = "test item",
    response: str = "test output",
    duration_ms: float = 500.0,
    error: str | None = None,
) -> dict[str, Any]:
    return {
        "index": index,
        "item": item,
        "prompt_sent": "test prompt",
        "response_received": None if error else response,
        "tokens": 10,
        "duration_ms": duration_ms,
        "error": error,
    }


def make_swrm_factory_instance(
    index: int = 0,
    *,
    agents: list[dict[str, Any]] | None = None,
    synthesis_output: str = "synthesised result",
    error: str | None = None,
) -> dict[str, Any]:
    if agents is None:
        agents = [make_agent("editor"), make_agent("grader")]
    return {
        "index": index,
        "item": "test paper",
        "tokens": 30,
        "duration_ms": 800.0,
        "error": error,
        "swrm": None
        if error
        else {
            "agents": agents,
            "output": synthesis_output,
            "tokens": 30,
            "duration_ms": 800.0,
        },
    }


def make_factory_event(
    *,
    node_id: str = "grade_papers",
    instances: list[dict[str, Any]] | None = None,
    output: Any = None,
    status: str = "success",
    duration_ms: float = 900.0,
    writes: str = "working.grade_reports",
) -> NodeCompleteEvent:
    if instances is None:
        instances = [
            make_instance(0, response="Grade: A"),
            make_instance(1, response="Grade: D"),
            make_instance(2, response="Grade: C"),
        ]
    if output is None:
        output = [inst.get("response_received") or "" for inst in instances]
    return NodeCompleteEvent(
        node_id=node_id,
        node_type="factory",
        output=output,
        writes=writes,
        status=status,  # type: ignore[arg-type]
        tokens=30,
        instances=instances,
        duration_ms=duration_ms,
    )


def capture_factory_render(event: NodeCompleteEvent, *, tty: bool = False) -> str:
    buf = io.StringIO()
    console = Console(file=buf, force_terminal=tty, no_color=not tty)
    box_style = ROUNDED if tty else ASCII
    render_node_panel(event, console, box_style)
    return buf.getvalue()


class TestRenderFactoryPanelHeader:
    def test_header_contains_node_id(self) -> None:
        output = capture_factory_render(make_factory_event(node_id="grade_papers"))
        assert "grade_papers" in output

    def test_header_contains_instance_count(self) -> None:
        output = capture_factory_render(make_factory_event())
        assert "3 instances" in output

    def test_header_singular_instance(self) -> None:
        output = capture_factory_render(make_factory_event(instances=[make_instance(0)]))
        assert "1 instance" in output
        assert "1 instances" not in output


class TestRenderFactoryPanelFooter:
    def test_footer_shows_success_ratio(self) -> None:
        instances = [make_instance(0), make_instance(1), make_instance(2, error="timeout")]
        output = capture_factory_render(make_factory_event(instances=instances))
        assert "2/3 succeeded" in output

    def test_footer_shows_total_duration(self) -> None:
        output = capture_factory_render(make_factory_event(duration_ms=900.0))
        assert "0.9s total" in output

    def test_footer_omits_failed_when_all_succeed(self) -> None:
        output = capture_factory_render(make_factory_event())
        assert "Factory complete:" in output
        assert "failed" not in output

    def test_footer_includes_failed_count_when_partial(self) -> None:
        instances = [make_instance(0), make_instance(1, error="boom")]
        output = capture_factory_render(make_factory_event(instances=instances))
        assert "1 failed" in output


class TestRenderFactoryInstancePanels:
    def test_instance_output_shown(self) -> None:
        instances = [make_instance(0, response="Grade: A")]
        output = capture_factory_render(make_factory_event(instances=instances))
        assert "Grade: A" in output

    def test_all_instance_outputs_present(self) -> None:
        instances = [make_instance(i, response=f"result_{i}") for i in range(3)]
        output = capture_factory_render(make_factory_event(instances=instances))
        for i in range(3):
            assert f"result_{i}" in output

    def test_failed_instance_shows_error(self) -> None:
        instances = [make_instance(0, error="GuardrailError")]
        output = capture_factory_render(make_factory_event(instances=instances))
        assert "GuardrailError" in output

    def test_index_labels_shown(self) -> None:
        instances = [make_instance(i) for i in range(3)]
        output = capture_factory_render(make_factory_event(instances=instances))
        assert "[1/3]" in output
        assert "[3/3]" in output


class TestRenderSwrmFactoryInstance:
    def test_swrm_instance_shows_agent_outputs(self) -> None:
        agents = [make_agent("editor", response="good writing"), make_agent("grader", response="A")]
        instance = make_swrm_factory_instance(0, agents=agents, synthesis_output="Grade: A")
        instances = [instance]
        output = capture_factory_render(make_factory_event(instances=instances))
        assert "good writing" in output
        assert "A" in output

    def test_swrm_instance_shows_all_agent_ids(self) -> None:
        agents = [make_agent("editor"), make_agent("grader"), make_agent("ai_detector")]
        instance = make_swrm_factory_instance(0, agents=agents)
        output = capture_factory_render(make_factory_event(instances=[instance]))
        assert "editor" in output
        assert "grader" in output
        assert "ai_detector" in output

    def test_swrm_instance_shows_synthesis_output(self) -> None:
        instance = make_swrm_factory_instance(0, synthesis_output="Grade: A\nIntegrity: LIKELY_HUMAN")
        output = capture_factory_render(make_factory_event(instances=[instance]))
        assert "LIKELY_HUMAN" in output

    def test_swrm_instance_shows_index_rule(self) -> None:
        instances = [make_swrm_factory_instance(0), make_swrm_factory_instance(1)]
        output = capture_factory_render(make_factory_event(instances=instances))
        assert "[1/2]" in output
        assert "[2/2]" in output


class TestRenderFactoryPanelWritesArrow:
    def test_writes_arrow_after_success(self) -> None:
        output = capture_factory_render(make_factory_event(status="success", writes="working.results"))
        assert "↓ working.results" in output

    def test_writes_arrow_omitted_on_failure(self) -> None:
        event = NodeCompleteEvent(
            node_id="grade_papers",
            node_type="factory",
            status="failed",
            error="FactoryNodeError",
            instances=None,
        )
        output = capture_factory_render(event)
        assert "↓" not in output


class TestRenderNodePanelFactoryDispatch:
    def test_dispatches_factory_when_instances_populated(self) -> None:
        output = capture_factory_render(make_factory_event())
        assert "Factory:" in output
        assert "Factory complete:" in output

    def test_falls_through_to_panel_when_instances_none(self) -> None:
        event = NodeCompleteEvent(
            node_id="grade_papers",
            node_type="factory",
            status="failed",
            error="FactoryNodeError: something failed",
            instances=None,
        )
        output = capture_factory_render(event)
        assert "FactoryNodeError" in output
        assert "Factory:" not in output


class TestRenderFactoryInstancePanelDirect:
    def test_success_instance_renders_response(self) -> None:
        instance = make_instance(0, response="the grade")
        buf = io.StringIO()
        console = Console(file=buf, force_terminal=False, no_color=True)
        render_factory_instance_panel(instance, 3, console, ASCII)
        output = buf.getvalue()
        assert "the grade" in output
        assert "[1/3]" in output

    def test_failed_instance_renders_error(self) -> None:
        instance = make_instance(1, error="LengthError")
        buf = io.StringIO()
        console = Console(file=buf, force_terminal=False, no_color=True)
        render_factory_instance_panel(instance, 3, console, ASCII)
        output = buf.getvalue()
        assert "LengthError" in output

    def test_swrm_factory_instance_renders_agent_panels(self) -> None:
        agents = [make_agent("editor", response="clear prose"), make_agent("grader", response="Grade: B")]
        instance = make_swrm_factory_instance(0, agents=agents, synthesis_output="Grade: B")
        buf = io.StringIO()
        console = Console(file=buf, force_terminal=False, no_color=True)
        render_factory_instance_panel(instance, 2, console, ASCII)
        output = buf.getvalue()
        assert "editor" in output
        assert "clear prose" in output
        assert "grader" in output
        assert "[1/2]" in output


class TestRenderFactoryPanelDirectCall:
    def test_direct_call_produces_header_and_footer(self) -> None:
        event = make_factory_event()
        buf = io.StringIO()
        console = Console(file=buf, force_terminal=False, no_color=True)
        render_factory_panel(event, console, ASCII)
        output = buf.getvalue()
        assert "Factory:" in output
        assert "Factory complete:" in output
