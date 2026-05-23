"""Tests for the ``sirenspec explain`` command and its supporting functions."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from sirenspec.cli import app
from sirenspec.cli.explain import build_plan, explain_workflow
from sirenspec.core.models import Workflow

runner = CliRunner()


# ---------------------------------------------------------------------------
# Workflow fixture builders
# ---------------------------------------------------------------------------


def linear_workflow() -> Workflow:
    """Return a simple linear A→B→C workflow."""
    return Workflow.model_validate(
        {
            "version": "0.1",
            "agents": {
                "bot": {"model": "openai:gpt-4o-mini", "system": "You are helpful."},
            },
            "nodes": {
                "step_a": {"agent": "bot", "writes": "working.a"},
                "step_b": {"agent": "bot", "writes": "working.b"},
                "step_c": {"agent": "bot", "writes": "output.result"},
            },
            "edges": [
                {"from": "step_a", "to": "step_b"},
                {"from": "step_b", "to": "step_c"},
            ],
        }
    )


def branching_workflow() -> Workflow:
    """Return a workflow with a conditional branch: A→B (when condition), A→C (default)."""
    return Workflow.model_validate(
        {
            "version": "0.1",
            "agents": {
                "router": {"model": "openai:gpt-4o-mini", "system": "Route the request."},
                "specialist": {"model": "anthropic:claude-haiku-4-5-20251001", "system": "Handle."},
            },
            "nodes": {
                "classify": {"agent": "router", "writes": "working.intent"},
                "handle_refund": {"agent": "specialist", "writes": "output.reply"},
                "handle_inquiry": {"agent": "specialist", "writes": "output.reply"},
            },
            "edges": [
                {"from": "classify", "to": "handle_refund", "when": 'working.intent == "refund"'},
                {"from": "classify", "to": "handle_inquiry"},
            ],
        }
    )


def unreachable_node_workflow() -> Workflow:
    """Return a workflow where node D exists but nothing points to it (and it is not root)."""
    return Workflow.model_validate(
        {
            "version": "0.1",
            "agents": {
                "bot": {"model": "openai:gpt-4o-mini", "system": "sys"},
            },
            "nodes": {
                "start": {"agent": "bot", "writes": "working.a"},
                "end": {"agent": "bot", "writes": "output.result"},
                "orphan": {"agent": "bot", "writes": "working.orphan"},
            },
            "edges": [
                {"from": "start", "to": "end"},
            ],
        }
    )


def no_edges_workflow() -> Workflow:
    """Return a workflow where nodes have no connecting edges."""
    return Workflow.model_validate(
        {
            "version": "0.1",
            "agents": {
                "bot": {"model": "openai:gpt-4o-mini", "system": "sys"},
            },
            "nodes": {
                "alpha": {"agent": "bot", "writes": "working.a"},
                "beta": {"agent": "bot", "writes": "working.b"},
            },
            "edges": [],
        }
    )


def workflow_with_guardrails() -> Workflow:
    """Return a workflow with both workflow-level and agent-level guardrails."""
    return Workflow.model_validate(
        {
            "version": "0.1",
            "agents": {
                "bot": {
                    "model": "openai:gpt-4o-mini",
                    "system": "sys",
                    "guardrails": ["injection", "schema"],
                },
            },
            "nodes": {
                "classify": {"agent": "bot", "writes": "working.intent"},
            },
            "edges": [],
            "guardrails": ["injection", "cost_cap"],
        }
    )


# ---------------------------------------------------------------------------
# Unit tests: build_plan / explain_workflow
# ---------------------------------------------------------------------------


class TestLinearWorkflow:
    def test_execution_order_is_linear(self) -> None:
        wf = linear_workflow()
        plan = build_plan(wf, "linear")
        assert plan["execution_order"] == ["step_a", "step_b", "step_c"]

    def test_no_warnings_for_linear_chain(self) -> None:
        wf = linear_workflow()
        plan = build_plan(wf, "linear")
        assert plan["warnings"] == []

    def test_no_errors_for_valid_workflow(self) -> None:
        wf = linear_workflow()
        plan = build_plan(wf, "linear")
        assert plan["errors"] == []

    def test_node_count(self) -> None:
        wf = linear_workflow()
        plan = build_plan(wf, "linear")
        assert plan["node_count"] == 3

    def test_text_output_contains_node_ids(self) -> None:
        wf = linear_workflow()
        rendered, has_errors = explain_workflow(wf, "linear")
        assert "step_a" in rendered
        assert "step_b" in rendered
        assert "step_c" in rendered
        assert not has_errors

    def test_text_output_contains_writes(self) -> None:
        wf = linear_workflow()
        rendered, _ = explain_workflow(wf, "linear")
        assert "working.a" in rendered
        assert "output.result" in rendered

    def test_text_output_shows_edges(self) -> None:
        wf = linear_workflow()
        rendered, _ = explain_workflow(wf, "linear")
        assert "→ step_b" in rendered
        assert "→ step_c" in rendered


class TestBranchingWorkflow:
    def test_conditional_edge_shown(self) -> None:
        wf = branching_workflow()
        rendered, _ = explain_workflow(wf, "branching")
        assert 'working.intent == "refund"' in rendered
        assert "handle_refund" in rendered

    def test_default_edge_shown(self) -> None:
        wf = branching_workflow()
        rendered, _ = explain_workflow(wf, "branching")
        assert "[default]" in rendered
        assert "handle_inquiry" in rendered

    def test_json_edge_has_when_and_null(self) -> None:
        wf = branching_workflow()
        rendered, _ = explain_workflow(wf, "branching", output_format="json")
        data = json.loads(rendered)
        classify = next(n for n in data["nodes"] if n["id"] == "classify")
        whens = {e["to"]: e["when"] for e in classify["outgoing_edges"]}
        assert whens["handle_refund"] == 'working.intent == "refund"'
        assert whens["handle_inquiry"] is None

    def test_agent_count(self) -> None:
        wf = branching_workflow()
        plan = build_plan(wf, "branching")
        assert plan["agent_count"] == 3  # three AgentNode instances


class TestUnreachableNode:
    def test_warning_emitted_for_orphan(self) -> None:
        wf = unreachable_node_workflow()
        plan = build_plan(wf, "unreachable")
        warning_text = " ".join(plan["warnings"])
        assert "orphan" in warning_text

    def test_no_error_for_orphan(self) -> None:
        wf = unreachable_node_workflow()
        plan = build_plan(wf, "unreachable")
        assert plan["errors"] == []

    def test_rendered_text_contains_warning_symbol(self) -> None:
        wf = unreachable_node_workflow()
        rendered, _ = explain_workflow(wf, "unreachable")
        assert "⚠" in rendered


class TestNoEdgesWorkflow:
    def test_nodes_shown_standalone(self) -> None:
        wf = no_edges_workflow()
        rendered, _ = explain_workflow(wf, "standalone")
        assert "alpha" in rendered
        assert "beta" in rendered

    def test_no_edge_arrows_in_output(self) -> None:
        wf = no_edges_workflow()
        rendered, _ = explain_workflow(wf, "standalone")
        # There are no edges so no arrow lines should appear
        assert "↳" not in rendered

    def test_no_errors(self) -> None:
        wf = no_edges_workflow()
        plan = build_plan(wf, "standalone")
        assert plan["errors"] == []


class TestJsonFormat:
    def test_json_is_valid(self) -> None:
        wf = linear_workflow()
        rendered, _ = explain_workflow(wf, "linear", output_format="json")
        data = json.loads(rendered)
        assert isinstance(data, dict)

    def test_json_has_required_keys(self) -> None:
        wf = linear_workflow()
        rendered, _ = explain_workflow(wf, "linear", output_format="json")
        data = json.loads(rendered)
        for key in (
            "workflow_name",
            "node_count",
            "agent_count",
            "execution_order",
            "nodes",
            "workflow_guardrails",
            "warnings",
            "errors",
        ):
            assert key in data, f"Missing key: {key}"

    def test_json_execution_order(self) -> None:
        wf = linear_workflow()
        rendered, _ = explain_workflow(wf, "linear", output_format="json")
        data = json.loads(rendered)
        assert data["execution_order"] == ["step_a", "step_b", "step_c"]

    def test_json_nodes_have_required_keys(self) -> None:
        wf = linear_workflow()
        rendered, _ = explain_workflow(wf, "linear", output_format="json")
        data = json.loads(rendered)
        for node in data["nodes"]:
            for key in ("id", "type", "agent", "model", "writes", "guardrails", "outgoing_edges"):
                assert key in node, f"Node missing key: {key}"

    def test_json_node_model_populated(self) -> None:
        wf = linear_workflow()
        rendered, _ = explain_workflow(wf, "linear", output_format="json")
        data = json.loads(rendered)
        assert data["nodes"][0]["model"] == "openai:gpt-4o-mini"

    def test_json_workflow_guardrails(self) -> None:
        wf = workflow_with_guardrails()
        rendered, _ = explain_workflow(wf, "guarded", output_format="json")
        data = json.loads(rendered)
        assert "injection" in data["workflow_guardrails"]
        assert "cost_cap" in data["workflow_guardrails"]


class TestGuardrailDisplay:
    def test_agent_guardrails_shown_in_text(self) -> None:
        wf = workflow_with_guardrails()
        rendered, _ = explain_workflow(wf, "guarded")
        assert "injection" in rendered
        assert "schema" in rendered

    def test_workflow_guardrails_shown_at_bottom(self) -> None:
        wf = workflow_with_guardrails()
        rendered, _ = explain_workflow(wf, "guarded")
        assert "Guardrails (workflow-level)" in rendered
        assert "cost_cap" in rendered


class TestCycleDetection:
    def test_cycle_produces_error_and_has_errors_flag(self) -> None:
        # Build a workflow with a cycle bypassing model validator (which catches reference errors)
        # by providing a cycle A→B→A without validation errors from the model itself.
        # The model validator only checks node references, not cycles, so this is fine.
        wf = Workflow.model_validate(
            {
                "version": "0.1",
                "agents": {
                    "bot": {"model": "openai:gpt-4o-mini", "system": "sys"},
                },
                "nodes": {
                    "a": {"agent": "bot", "writes": "working.a"},
                    "b": {"agent": "bot", "writes": "working.b"},
                },
                "edges": [
                    {"from": "a", "to": "b"},
                    {"from": "b", "to": "a"},
                ],
            }
        )
        plan = build_plan(wf, "cyclic")
        assert any("cycle" in e.lower() for e in plan["errors"])

    def test_cycle_sets_has_errors_true(self) -> None:
        wf = Workflow.model_validate(
            {
                "version": "0.1",
                "agents": {
                    "bot": {"model": "openai:gpt-4o-mini", "system": "sys"},
                },
                "nodes": {
                    "a": {"agent": "bot", "writes": "working.a"},
                    "b": {"agent": "bot", "writes": "working.b"},
                },
                "edges": [
                    {"from": "a", "to": "b"},
                    {"from": "b", "to": "a"},
                ],
            }
        )
        _, has_errors = explain_workflow(wf, "cyclic")
        assert has_errors


# ---------------------------------------------------------------------------
# CLI integration tests
# ---------------------------------------------------------------------------


class TestExplainCLI:
    MINIMAL_YAML = """\
version: "0.1"
agents:
  bot:
    model: openai:gpt-4o-mini
    system: "You are helpful."
nodes:
  step_a:
    agent: bot
    writes: working.a
  step_b:
    agent: bot
    writes: output.result
edges:
  - from: step_a
    to: step_b
"""

    def test_explain_exits_zero_for_valid_workflow(self, tmp_path):  # type: ignore[no-untyped-def]
        f = tmp_path / "wf.yaml"
        f.write_text(self.MINIMAL_YAML)
        result = runner.invoke(app, ["explain", str(f)])
        assert result.exit_code == 0, result.output

    def test_explain_text_output_contains_workflow_name(self, tmp_path):  # type: ignore[no-untyped-def]
        f = tmp_path / "wf.yaml"
        f.write_text(self.MINIMAL_YAML)
        result = runner.invoke(app, ["explain", str(f)])
        assert "wf" in result.output

    def test_explain_json_format_flag(self, tmp_path):  # type: ignore[no-untyped-def]
        f = tmp_path / "wf.yaml"
        f.write_text(self.MINIMAL_YAML)
        result = runner.invoke(app, ["explain", str(f), "--format", "json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "execution_order" in data

    def test_explain_missing_file_exits_nonzero(self, tmp_path):  # type: ignore[no-untyped-def]
        result = runner.invoke(app, ["explain", str(tmp_path / "nope.yaml")])
        assert result.exit_code != 0

    def test_explain_short_format_flag(self, tmp_path):  # type: ignore[no-untyped-def]
        f = tmp_path / "wf.yaml"
        f.write_text(self.MINIMAL_YAML)
        result = runner.invoke(app, ["explain", str(f), "-f", "json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["node_count"] == 2
