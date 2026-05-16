"""Tests for the render CLI command and Mermaid renderer."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from sirenspec.cli import app
from sirenspec.core.models import AgentDefinition, Edge, Node, SwrmAgent, SwrmNode, SwrmSynthesis, Workflow
from sirenspec.render.mermaid import workflow_to_mermaid

runner = CliRunner()

_AGENT = AgentDefinition(model="openai:gpt-4o-mini", system="sys")


def _write_workflow(tmp_path: Path, content: str) -> Path:
    f = tmp_path / "wf.yaml"
    f.write_text(content)
    return f


def _make_workflow(
    nodes: dict[str, Node] | None = None,
    edges: list[Edge] | None = None,
) -> Workflow:
    nodes = nodes or {"n1": Node(agent="a", writes="output.x")}
    return Workflow(
        version="0.1",
        agents={"a": _AGENT},
        nodes=nodes,
        edges=edges or [],
    )


MINIMAL_YAML = """\
version: "0.1"
agents:
  assistant:
    model: "openai:gpt-4o-mini"
    system: "You are helpful."
nodes:
  answer:
    agent: assistant
    writes: output.reply
"""

CONDITIONAL_YAML = """\
version: "0.1"
agents:
  a:
    model: "openai:gpt-4o-mini"
    system: "sys"
nodes:
  triage:
    agent: a
    writes: working.intent
  handle:
    agent: a
    writes: output.reply
edges:
  - from: triage
    to: handle
    when: "working.intent == 'yes'"
"""


class TestWorkflowToMermaid:
    def test_graph_header(self) -> None:
        result = workflow_to_mermaid(_make_workflow())
        assert result.startswith("graph TD")

    def test_node_present(self) -> None:
        result = workflow_to_mermaid(_make_workflow())
        assert "n1[n1]" in result

    def test_multiple_nodes(self) -> None:
        nodes = {
            "n1": Node(agent="a", writes="output.x"),
            "n2": Node(agent="a", writes="output.y"),
        }
        result = workflow_to_mermaid(_make_workflow(nodes=nodes))
        assert "n1[n1]" in result
        assert "n2[n2]" in result

    def test_unconditional_edge(self) -> None:
        nodes = {"n1": Node(agent="a", writes="output.x"), "n2": Node(agent="a", writes="output.y")}
        edges = [Edge(**{"from": "n1", "to": "n2"})]
        result = workflow_to_mermaid(_make_workflow(nodes=nodes, edges=edges))
        assert "n1 --> n2" in result

    def test_conditional_edge_includes_label(self) -> None:
        nodes = {"n1": Node(agent="a", writes="output.x"), "n2": Node(agent="a", writes="output.y")}
        edges = [Edge(**{"from": "n1", "to": "n2", "when": "output.x == 'yes'"})]
        result = workflow_to_mermaid(_make_workflow(nodes=nodes, edges=edges))
        assert "n1 -->" in result
        assert "n2" in result
        assert "output.x == 'yes'" in result

    def test_no_edges_produces_only_header_and_nodes(self) -> None:
        result = workflow_to_mermaid(_make_workflow())
        lines = result.strip().split("\n")
        assert len(lines) == 2


def _make_swrm_node(
    agent_ids: list[str],
    with_synthesis: bool = True,
) -> SwrmNode:
    agents = [SwrmAgent(id=aid, provider="openai", model="gpt-4o-mini", prompt="p") for aid in agent_ids]
    synthesis = SwrmSynthesis(provider="openai", model="gpt-4o-mini", prompt="s") if with_synthesis else None
    return SwrmNode(agents=agents, synthesis=synthesis)


def _make_swrm_workflow(
    node_id: str,
    agent_ids: list[str],
    with_synthesis: bool = True,
) -> Workflow:
    swrm = _make_swrm_node(agent_ids, with_synthesis=with_synthesis)
    return Workflow(version="0.1", agents={}, nodes={node_id: swrm}, edges=[])


class TestSwrmMermaid:
    def test_swrm_renders_as_subgraph(self) -> None:
        result = workflow_to_mermaid(_make_swrm_workflow("fan", ["a", "b"]))
        assert "subgraph fan" in result

    def test_swrm_agents_appear_inside_subgraph(self) -> None:
        result = workflow_to_mermaid(_make_swrm_workflow("fan", ["a", "b"]))
        assert "fan_a[a]" in result
        assert "fan_b[b]" in result

    def test_swrm_synthesis_node_present(self) -> None:
        result = workflow_to_mermaid(_make_swrm_workflow("fan", ["a", "b"], with_synthesis=True))
        assert "fan_synthesis" in result

    def test_swrm_agents_connect_to_synthesis(self) -> None:
        result = workflow_to_mermaid(_make_swrm_workflow("fan", ["a", "b"], with_synthesis=True))
        assert "fan_a --> fan_synthesis" in result
        assert "fan_b --> fan_synthesis" in result

    def test_swrm_without_synthesis_has_no_synthesis_node(self) -> None:
        result = workflow_to_mermaid(_make_swrm_workflow("fan", ["a"], with_synthesis=False))
        assert "synthesis" not in result

    def test_workflow_edge_from_swrm_with_synthesis_exits_synthesis(self) -> None:
        swrm = _make_swrm_node(["a"], with_synthesis=True)
        downstream = Node(agent="a", writes="output.x")
        wf = Workflow(
            version="0.1",
            agents={"a": _AGENT},
            nodes={"fan": swrm, "next": downstream},
            edges=[Edge(**{"from": "fan", "to": "next"})],
        )
        result = workflow_to_mermaid(wf)
        assert "fan_synthesis --> next" in result

    def test_workflow_edge_from_swrm_without_synthesis_uses_node_id(self) -> None:
        swrm = _make_swrm_node(["a"], with_synthesis=False)
        downstream = Node(agent="a", writes="output.x")
        wf = Workflow(
            version="0.1",
            agents={"a": _AGENT},
            nodes={"fan": swrm, "next": downstream},
            edges=[Edge(**{"from": "fan", "to": "next"})],
        )
        result = workflow_to_mermaid(wf)
        assert "fan --> next" in result


class TestRenderCommand:
    def test_render_mermaid_exits_zero(self, tmp_path: Path) -> None:
        f = _write_workflow(tmp_path, MINIMAL_YAML)
        result = runner.invoke(app, ["render", str(f), "--target", "mermaid"])
        assert result.exit_code == 0

    def test_render_mermaid_output_contains_graph(self, tmp_path: Path) -> None:
        f = _write_workflow(tmp_path, MINIMAL_YAML)
        result = runner.invoke(app, ["render", str(f), "--target", "mermaid"])
        assert "graph TD" in result.output
        assert "answer[answer]" in result.output

    def test_render_conditional_edges(self, tmp_path: Path) -> None:
        f = _write_workflow(tmp_path, CONDITIONAL_YAML)
        result = runner.invoke(app, ["render", str(f), "--target", "mermaid"])
        assert result.exit_code == 0
        assert "triage -->" in result.output
        assert "handle" in result.output
        assert "working.intent == 'yes'" in result.output

    def test_render_unsupported_target_exits_nonzero(self, tmp_path: Path) -> None:
        f = _write_workflow(tmp_path, MINIMAL_YAML)
        result = runner.invoke(app, ["render", str(f), "--target", "png"])
        assert result.exit_code != 0

    def test_render_missing_file_exits_nonzero(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["render", "nope.yaml", "--target", "mermaid"])
        assert result.exit_code != 0

    def test_render_output_to_file(self, tmp_path: Path) -> None:
        f = _write_workflow(tmp_path, MINIMAL_YAML)
        out = tmp_path / "diagram.md"
        result = runner.invoke(app, ["render", str(f), "--target", "mermaid", "--output", str(out)])
        assert result.exit_code == 0
        assert out.exists()
        content = out.read_text()
        assert "graph TD" in content

    def test_render_output_to_file_nothing_on_stdout(self, tmp_path: Path) -> None:
        f = _write_workflow(tmp_path, MINIMAL_YAML)
        out = tmp_path / "diagram.md"
        result = runner.invoke(app, ["render", str(f), "--target", "mermaid", "--output", str(out)])
        assert result.output.strip() == ""
