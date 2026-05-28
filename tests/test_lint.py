"""Unit tests for the load-time workflow linter (core/lint.py)."""

from __future__ import annotations

import pytest

from sirenspec.core.lint import (
    check_unknown_namespace,
    check_working_dot_node_id,
    extract_top_level_names,
    lint_workflow,
)
from sirenspec.core.models import (
    Workflow,
)
from sirenspec.exceptions import WorkflowLintError
from sirenspec.yaml.parser import load_workflow

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

NODE_IDS = {"plan", "execute", "summarise"}


def _make_workflow(system_prompt: str) -> Workflow:
    """Build a minimal single-agent workflow with the given system prompt."""
    return Workflow.model_validate(
        {
            "version": "0.1",
            "agents": {"a": {"model": "openai:gpt-4o-mini", "system": system_prompt}},
            "nodes": {"n": {"type": "agent", "agent": "a", "writes": "output.result"}},
            "edges": [],
        }
    )


def _make_two_node_workflow(system_prompt: str) -> Workflow:
    """Build a two-node workflow where the second agent prompt may reference the first node."""
    return Workflow.model_validate(
        {
            "version": "0.1",
            "agents": {
                "planner": {"model": "openai:gpt-4o-mini", "system": "You plan."},
                "executor": {"model": "openai:gpt-4o-mini", "system": system_prompt},
            },
            "nodes": {
                "plan": {"type": "agent", "agent": "planner", "writes": "output.plan"},
                "execute": {"type": "agent", "agent": "executor", "writes": "output.result"},
            },
            "edges": [{"from": "plan", "to": "execute"}],
        }
    )


# ---------------------------------------------------------------------------
# extract_top_level_names
# ---------------------------------------------------------------------------


class TestExtractTopLevelNames:
    def test_simple_expression(self) -> None:
        result = extract_top_level_names("{{ plan.output }}")
        assert result == [("plan.output", "plan")]

    def test_multiple_expressions(self) -> None:
        result = extract_top_level_names("{{ inputs.x }} and {{ env.HOME }}")
        assert result == [("inputs.x", "inputs"), ("env.HOME", "env")]

    def test_default_filter_stripped(self) -> None:
        result = extract_top_level_names("{{ plan.output | default('none') }}")
        assert result == [("plan.output", "plan")]

    def test_json_or_default_filter_stripped(self) -> None:
        result = extract_top_level_names("{{ plan.output | json_or_default('[]') }}")
        assert result == [("plan.output", "plan")]

    def test_no_expressions(self) -> None:
        assert extract_top_level_names("just plain text") == []

    def test_bare_name(self) -> None:
        result = extract_top_level_names("{{ item }}")
        assert result == [("item", "item")]


# ---------------------------------------------------------------------------
# check_working_dot_node_id
# ---------------------------------------------------------------------------


class TestCheckWorkingDotNodeId:
    def test_flags_working_dot_known_node(self) -> None:
        issues = check_working_dot_node_id("{{ working.plan.output }}", "loc", NODE_IDS)
        assert len(issues) == 1
        assert issues[0].level == "error"
        assert issues[0].rule == "working_dot_node_id"
        assert "plan" in issues[0].message
        assert "{{ plan.output }}" in issues[0].message

    def test_flags_working_dot_node_two_segments(self) -> None:
        issues = check_working_dot_node_id("{{ working.plan }}", "loc", NODE_IDS)
        assert len(issues) == 1
        assert "{{ plan.output }}" in issues[0].message

    def test_ignores_working_dot_unknown_name(self) -> None:
        issues = check_working_dot_node_id("{{ working.someRandomKey }}", "loc", NODE_IDS)
        assert issues == []

    def test_ignores_non_working_prefix(self) -> None:
        issues = check_working_dot_node_id("{{ plan.output }}", "loc", NODE_IDS)
        assert issues == []

    def test_multiple_violations_reported(self) -> None:
        template = "{{ working.plan.output }} and {{ working.execute.output }}"
        issues = check_working_dot_node_id(template, "loc", NODE_IDS)
        assert len(issues) == 2

    def test_location_propagated(self) -> None:
        issues = check_working_dot_node_id("{{ working.plan.output }}", "agent 'foo' prompt", NODE_IDS)
        assert issues[0].location == "agent 'foo' prompt"


# ---------------------------------------------------------------------------
# check_unknown_namespace
# ---------------------------------------------------------------------------


class TestCheckUnknownNamespace:
    def test_reserved_namespaces_are_allowed(self) -> None:
        for ns in ("inputs", "env", "item", "index", "total"):
            assert check_unknown_namespace(f"{{{{ {ns}.foo }}}}", "loc", NODE_IDS) == []

    def test_known_node_ids_are_allowed(self) -> None:
        assert check_unknown_namespace("{{ plan.output }}", "loc", NODE_IDS) == []

    def test_unknown_name_generates_warning(self) -> None:
        issues = check_unknown_namespace("{{ typo.output }}", "loc", NODE_IDS)
        assert len(issues) == 1
        assert issues[0].level == "warning"
        assert issues[0].rule == "unknown_namespace"
        assert "typo" in issues[0].message

    def test_default_filter_does_not_suppress_warning(self) -> None:
        issues = check_unknown_namespace("{{ typo.output | default('x') }}", "loc", NODE_IDS)
        assert len(issues) == 1

    def test_bare_item_is_reserved(self) -> None:
        assert check_unknown_namespace("{{ item }}", "loc", NODE_IDS) == []


# ---------------------------------------------------------------------------
# lint_workflow — integration
# ---------------------------------------------------------------------------


class TestLintWorkflow:
    def test_clean_workflow_returns_no_issues(self) -> None:
        wf = _make_two_node_workflow("Here is the plan: {{ plan.output }}")
        assert lint_workflow(wf) == []

    def test_working_dot_node_in_agent_prompt_raises_error(self) -> None:
        wf = _make_two_node_workflow("Here is the plan: {{ working.plan.output }}")
        issues = lint_workflow(wf)
        errors = [i for i in issues if i.level == "error"]
        assert len(errors) == 1
        assert errors[0].rule == "working_dot_node_id"

    def test_unknown_namespace_produces_warning(self) -> None:
        wf = _make_two_node_workflow("Here is the plan: {{ plann.output }}")
        issues = lint_workflow(wf)
        warnings = [i for i in issues if i.level == "warning"]
        assert len(warnings) == 1
        assert warnings[0].rule == "unknown_namespace"

    def test_valid_inputs_namespace_is_clean(self) -> None:
        wf = _make_workflow("User said: {{ inputs.message }}")
        assert lint_workflow(wf) == []

    def test_valid_env_namespace_is_clean(self) -> None:
        wf = _make_workflow("Token: {{ env.API_KEY }}")
        assert lint_workflow(wf) == []


# ---------------------------------------------------------------------------
# load_workflow raises WorkflowLintError on errors
# ---------------------------------------------------------------------------


class TestLoadWorkflowLintIntegration:
    def test_working_dot_node_id_raises_on_load(self, tmp_path: pytest.TempDir) -> None:
        yaml_content = """\
version: "0.1"
agents:
  planner:
    model: openai:gpt-4o-mini
    system: "plan here"
  executor:
    model: openai:gpt-4o-mini
    system: "execute: {{ working.plan.output }}"
nodes:
  plan:
    type: agent
    agent: planner
    writes: output.plan
  execute:
    type: agent
    agent: executor
    writes: output.result
edges:
  - from: plan
    to: execute
"""
        wf_file = tmp_path / "bad.yaml"
        wf_file.write_text(yaml_content)
        with pytest.raises(WorkflowLintError) as exc_info:
            load_workflow(wf_file)
        assert "working_dot_node_id" in str(exc_info.value)

    def test_clean_workflow_loads_successfully(self, tmp_path: pytest.TempDir) -> None:
        yaml_content = """\
version: "0.1"
agents:
  planner:
    model: openai:gpt-4o-mini
    system: "plan here"
  executor:
    model: openai:gpt-4o-mini
    system: "execute: {{ plan.output }}"
nodes:
  plan:
    type: agent
    agent: planner
    writes: output.plan
  execute:
    type: agent
    agent: executor
    writes: output.result
edges:
  - from: plan
    to: execute
"""
        wf_file = tmp_path / "good.yaml"
        wf_file.write_text(yaml_content)
        wf = load_workflow(wf_file)
        assert "execute" in wf.nodes
