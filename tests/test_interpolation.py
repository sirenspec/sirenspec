"""Unit tests for the ``{{ }}`` template interpolation engine."""

from __future__ import annotations

import pytest

from sirenspec.core.interpolation import (
    InterpolationContext,
    build_interpolation_context,
    check_circular_template_refs,
    extract_node_refs,
    resolve_expression,
    resolve_template,
    resolve_to_list,
)
from sirenspec.core.models import (
    AgentDefinition,
    AgentNode,
    FactoryNode,
    SwrmAgent,
    SwrmNode,
    SwrmSynthesis,
    Workflow,
)
from sirenspec.exceptions import InterpolationError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ctx(
    *,
    inputs: dict | None = None,
    nodes: dict | None = None,
    env: dict | None = None,
    item: object = None,
    index: int | None = None,
) -> InterpolationContext:
    return InterpolationContext(
        inputs=inputs or {"message": "hello"},
        nodes=nodes or {},
        env=env if env is not None else {},
        item=item,
        index=index,
    )


# ---------------------------------------------------------------------------
# resolve_expression: inputs namespace
# ---------------------------------------------------------------------------


class TestResolveExpressionInputs:
    def test_resolves_message(self) -> None:
        ctx = _ctx(inputs={"message": "hello world"})
        assert resolve_expression("inputs.message", ctx) == "hello world"

    def test_resolves_nested_field(self) -> None:
        ctx = _ctx(inputs={"user": {"name": "Alice"}})
        assert resolve_expression("inputs.user.name", ctx) == "Alice"

    def test_missing_top_level_field_raises(self) -> None:
        ctx = _ctx(inputs={"message": "hi"})
        with pytest.raises(InterpolationError) as exc_info:
            resolve_expression("inputs.missing", ctx)
        assert exc_info.value.namespace == "inputs"
        assert "missing" in exc_info.value.reason

    def test_bare_inputs_without_field_raises(self) -> None:
        ctx = _ctx()
        with pytest.raises(InterpolationError) as exc_info:
            resolve_expression("inputs", ctx)
        assert exc_info.value.namespace == "inputs"

    def test_error_captures_expression(self) -> None:
        ctx = _ctx()
        with pytest.raises(InterpolationError) as exc_info:
            resolve_expression("inputs.nonexistent", ctx)
        assert exc_info.value.expression == "inputs.nonexistent"


# ---------------------------------------------------------------------------
# resolve_expression: env namespace
# ---------------------------------------------------------------------------


class TestResolveExpressionEnv:
    def test_resolves_set_var(self) -> None:
        ctx = _ctx(env={"MY_VAR": "my_value"})
        assert resolve_expression("env.MY_VAR", ctx) == "my_value"

    def test_unset_var_raises(self) -> None:
        ctx = _ctx(env={})
        with pytest.raises(InterpolationError) as exc_info:
            resolve_expression("env.UNSET_VAR", ctx)
        assert exc_info.value.namespace == "env"
        assert "UNSET_VAR" in exc_info.value.reason

    def test_redact_env_returns_stars(self) -> None:
        ctx = _ctx(env={"SECRET_KEY": "super_secret"})
        result = resolve_expression("env.SECRET_KEY", ctx, redact_env=True)
        assert result == "***"

    def test_redact_env_still_raises_if_unset(self) -> None:
        ctx = _ctx(env={})
        with pytest.raises(InterpolationError):
            resolve_expression("env.MISSING", ctx, redact_env=True)

    def test_bare_env_without_var_raises(self) -> None:
        ctx = _ctx()
        with pytest.raises(InterpolationError) as exc_info:
            resolve_expression("env", ctx)
        assert exc_info.value.namespace == "env"


# ---------------------------------------------------------------------------
# resolve_expression: node namespace
# ---------------------------------------------------------------------------


class TestResolveExpressionNodes:
    def test_resolves_node_output(self) -> None:
        ctx = _ctx(nodes={"plan": {"output": "the plan"}})
        assert resolve_expression("plan.output", ctx) == "the plan"

    def test_resolves_deeply_nested(self) -> None:
        ctx = _ctx(nodes={"classify": {"output": {"sentiment": "positive"}}})
        assert resolve_expression("classify.output.sentiment", ctx) == "positive"

    def test_resolves_swrm_subagent_output(self) -> None:
        ctx = _ctx(nodes={"analyze": {"agents": {"sentiment": {"output": "bullish"}}}})
        assert resolve_expression("analyze.agents.sentiment.output", ctx) == "bullish"

    def test_missing_node_raises(self) -> None:
        ctx = _ctx(nodes={})
        with pytest.raises(InterpolationError) as exc_info:
            resolve_expression("unknown_node.output", ctx)
        assert exc_info.value.namespace == "unknown_node"

    def test_missing_key_within_node_raises(self) -> None:
        ctx = _ctx(nodes={"plan": {}})
        with pytest.raises(InterpolationError):
            resolve_expression("plan.output", ctx)


# ---------------------------------------------------------------------------
# resolve_expression: item and index namespaces
# ---------------------------------------------------------------------------


class TestResolveExpressionLoopVars:
    def test_resolves_item(self) -> None:
        ctx = _ctx(item="task A")
        assert resolve_expression("item", ctx) == "task A"

    def test_resolves_index(self) -> None:
        ctx = _ctx(index=3)
        assert resolve_expression("index", ctx) == "3"

    def test_item_outside_loop_raises(self) -> None:
        ctx = _ctx()
        with pytest.raises(InterpolationError) as exc_info:
            resolve_expression("item", ctx)
        assert exc_info.value.namespace == "item"
        assert "loop" in exc_info.value.reason

    def test_index_outside_loop_raises(self) -> None:
        ctx = _ctx()
        with pytest.raises(InterpolationError) as exc_info:
            resolve_expression("index", ctx)
        assert exc_info.value.namespace == "index"


# ---------------------------------------------------------------------------
# resolve_expression: | default() filter
# ---------------------------------------------------------------------------


class TestDefaultFilter:
    def test_default_not_used_when_resolved(self) -> None:
        ctx = _ctx(inputs={"message": "real value"})
        result = resolve_expression("inputs.message | default('fallback')", ctx)
        assert result == "real value"

    def test_default_used_on_missing_key(self) -> None:
        ctx = _ctx(inputs={"message": "hi"})
        result = resolve_expression("inputs.missing | default('fallback')", ctx)
        assert result == "fallback"

    def test_default_used_on_missing_node(self) -> None:
        ctx = _ctx(nodes={})
        result = resolve_expression("plan.output | default('none')", ctx)
        assert result == "none"

    def test_default_single_quotes(self) -> None:
        ctx = _ctx()
        result = resolve_expression("inputs.x | default('hello world')", ctx)
        assert result == "hello world"

    def test_default_double_quotes(self) -> None:
        ctx = _ctx()
        result = resolve_expression('inputs.x | default("hello")', ctx)
        assert result == "hello"

    def test_no_default_raises_on_missing(self) -> None:
        ctx = _ctx()
        with pytest.raises(InterpolationError):
            resolve_expression("inputs.missing", ctx)

    def test_default_fires_on_empty_string(self) -> None:
        ctx = _ctx(inputs={"message": ""})
        result = resolve_expression("inputs.message | default('fallback')", ctx)
        assert result == "fallback"

    def test_default_does_not_fire_on_whitespace(self) -> None:
        ctx = _ctx(inputs={"message": "  "})
        result = resolve_expression("inputs.message | default('fallback')", ctx)
        assert result == "  "

    def test_default_does_not_fire_on_zero(self) -> None:
        ctx = _ctx(inputs={"message": "0"})
        result = resolve_expression("inputs.message | default('fallback')", ctx)
        assert result == "0"


class TestJsonOrDefaultFilter:
    def test_fires_on_interpolation_error(self) -> None:
        ctx = _ctx(nodes={})
        result = resolve_expression("plan.output | json_or_default('[]')", ctx)
        assert result == "[]"

    def test_fires_on_empty_string(self) -> None:
        ctx = _ctx(nodes={"plan": {"output": ""}})
        result = resolve_expression("plan.output | json_or_default('[]')", ctx)
        assert result == "[]"

    def test_fires_on_non_json_value(self) -> None:
        ctx = _ctx(nodes={"plan": {"output": "not json at all"}})
        result = resolve_expression("plan.output | json_or_default('[]')", ctx)
        assert result == "[]"

    def test_returns_value_when_valid_json(self) -> None:
        ctx = _ctx(nodes={"plan": {"output": '["a", "b"]'}})
        result = resolve_expression("plan.output | json_or_default('[]')", ctx)
        assert result == '["a", "b"]'

    def test_returns_json_object(self) -> None:
        ctx = _ctx(nodes={"plan": {"output": '{"key": "val"}'}})
        result = resolve_expression("plan.output | json_or_default('{}')", ctx)
        assert result == '{"key": "val"}'

    def test_json_or_default_in_full_template(self) -> None:
        ctx = _ctx(nodes={"plan": {"output": ""}})
        result = resolve_template("items: {{ plan.output | json_or_default('[]') }}", ctx)
        assert result == "items: []"


# ---------------------------------------------------------------------------
# resolve_template: full string rendering
# ---------------------------------------------------------------------------


class TestResolveTemplate:
    def test_no_placeholders(self) -> None:
        ctx = _ctx()
        assert resolve_template("plain text", ctx) == "plain text"

    def test_single_placeholder(self) -> None:
        ctx = _ctx(inputs={"message": "world"})
        assert resolve_template("Hello {{ inputs.message }}", ctx) == "Hello world"

    def test_multiple_placeholders(self) -> None:
        ctx = _ctx(nodes={"a": {"v": "first"}, "b": {"v": "second"}})
        assert resolve_template("{{ a.v }} + {{ b.v }}", ctx) == "first + second"

    def test_raises_on_first_missing_key(self) -> None:
        ctx = _ctx()
        with pytest.raises(InterpolationError):
            resolve_template("{{ missing.key }}", ctx)

    def test_redact_env_in_template(self) -> None:
        ctx = _ctx(env={"API_KEY": "sk-abc123"})
        result = resolve_template("Key: {{ env.API_KEY }}", ctx, redact_env=True)
        assert result == "Key: ***"

    def test_real_env_in_template_without_redact(self) -> None:
        ctx = _ctx(env={"API_KEY": "sk-abc123"})
        result = resolve_template("Key: {{ env.API_KEY }}", ctx, redact_env=False)
        assert result == "Key: sk-abc123"

    def test_default_filter_in_template(self) -> None:
        ctx = _ctx()
        result = resolve_template("Value: {{ inputs.x | default('none') }}", ctx)
        assert result == "Value: none"

    def test_whitespace_around_expression(self) -> None:
        ctx = _ctx(inputs={"message": "val"})
        assert resolve_template("{{inputs.message}}", ctx) == "val"
        assert resolve_template("{{  inputs.message  }}", ctx) == "val"


# ---------------------------------------------------------------------------
# resolve_to_list
# ---------------------------------------------------------------------------


class TestResolveToList:
    def test_resolves_json_array_from_node(self) -> None:
        ctx = _ctx(nodes={"plan": {"output": '["task A", "task B", "task C"]'}})
        result = resolve_to_list("{{ plan.output }}", ctx)
        assert result == ["task A", "task B", "task C"]

    def test_resolves_inline_json_list(self) -> None:
        ctx = _ctx()
        result = resolve_to_list('["x", "y"]', ctx)
        assert result == ["x", "y"]

    def test_invalid_json_raises(self) -> None:
        ctx = _ctx(nodes={"plan": {"output": "not json at all"}})
        with pytest.raises(InterpolationError) as exc_info:
            resolve_to_list("{{ plan.output }}", ctx)
        assert exc_info.value.namespace == "for_each"

    def test_json_but_not_list_raises(self) -> None:
        ctx = _ctx(nodes={"plan": {"output": '{"key": "value"}'}})
        with pytest.raises(InterpolationError) as exc_info:
            resolve_to_list("{{ plan.output }}", ctx)
        assert "list" in exc_info.value.reason

    def test_missing_key_raises(self) -> None:
        ctx = _ctx(nodes={})
        with pytest.raises(InterpolationError):
            resolve_to_list("{{ missing.output }}", ctx)


# ---------------------------------------------------------------------------
# build_interpolation_context
# ---------------------------------------------------------------------------


class TestBuildInterpolationContext:
    def test_inputs_message(self) -> None:
        ctx = build_interpolation_context("hello", {})
        assert ctx.inputs["message"] == "hello"

    def test_nodes_is_working_dict(self) -> None:
        working = {"plan": {"output": "plan text"}}
        ctx = build_interpolation_context("input", working)
        assert ctx.nodes is working

    def test_env_snapshot(self) -> None:
        ctx = build_interpolation_context("x", {})
        assert isinstance(ctx.env, dict)

    def test_item_and_index_passed_through(self) -> None:
        ctx = build_interpolation_context("x", {}, item="task", index=0)
        assert ctx.item == "task"
        assert ctx.index == 0

    def test_item_index_default_none(self) -> None:
        ctx = build_interpolation_context("x", {})
        assert ctx.item is None
        assert ctx.index is None


# ---------------------------------------------------------------------------
# extract_node_refs
# ---------------------------------------------------------------------------


class TestExtractNodeRefs:
    def test_finds_node_ref(self) -> None:
        known = {"plan", "execute"}
        refs = extract_node_refs("{{ plan.output }}", known)
        assert refs == {"plan"}

    def test_ignores_inputs(self) -> None:
        known = {"inputs"}
        refs = extract_node_refs("{{ inputs.message }}", known)
        assert refs == set()

    def test_ignores_env(self) -> None:
        known = {"env"}
        refs = extract_node_refs("{{ env.HOME }}", known)
        assert refs == set()

    def test_ignores_item_and_index(self) -> None:
        known = {"item", "index"}
        refs = extract_node_refs("{{ item }} {{ index }}", known)
        assert refs == set()

    def test_multiple_refs(self) -> None:
        known = {"plan", "execute", "aggregate"}
        refs = extract_node_refs("{{ plan.output }} {{ execute.outputs }}", known)
        assert refs == {"plan", "execute"}

    def test_default_filter_handled(self) -> None:
        known = {"plan"}
        refs = extract_node_refs("{{ plan.output | default('none') }}", known)
        assert refs == {"plan"}

    def test_unknown_segment_not_included(self) -> None:
        known = {"plan"}
        refs = extract_node_refs("{{ unknown.output }}", known)
        assert refs == set()


# ---------------------------------------------------------------------------
# check_circular_template_refs
# ---------------------------------------------------------------------------


def _simple_workflow(system_a: str = "sys A", system_b: str = "sys B") -> Workflow:
    return Workflow(
        version="0.1",
        agents={
            "agent_a": AgentDefinition(model="openai:gpt-4o-mini", system=system_a),
            "agent_b": AgentDefinition(model="openai:gpt-4o-mini", system=system_b),
        },
        nodes={
            "node_a": AgentNode(agent="agent_a", writes="working.node_a.output"),
            "node_b": AgentNode(agent="agent_b", writes="working.node_b.output"),
        },
    )


class TestCheckCircularTemplateRefs:
    def test_no_cycle_passes(self) -> None:
        wf = _simple_workflow(
            system_a="You are A.",
            system_b="Use {{ node_a.output }} to reply.",
        )
        check_circular_template_refs(wf)  # Should not raise

    def test_no_refs_passes(self) -> None:
        wf = _simple_workflow(system_a="Simple system prompt.", system_b="Another prompt.")
        check_circular_template_refs(wf)

    def test_direct_cycle_raises(self) -> None:
        wf = _simple_workflow(
            system_a="Use {{ node_b.output }}.",
            system_b="Use {{ node_a.output }}.",
        )
        with pytest.raises(InterpolationError) as exc_info:
            check_circular_template_refs(wf)
        assert exc_info.value.namespace == "circular_ref"

    def test_swrm_cycle_raises(self) -> None:
        wf = Workflow(
            version="0.1",
            agents={
                "agent_a": AgentDefinition(
                    model="openai:gpt-4o-mini",
                    system="Use {{ node_b.output }}.",
                ),
            },
            nodes={
                "node_a": AgentNode(
                    agent="agent_a",
                    writes="working.node_a.output",
                ),
                "node_b": SwrmNode(
                    type="swrm",
                    agents=[
                        SwrmAgent(
                            id="sub",
                            provider="openai",
                            model="gpt-4o-mini",
                            prompt="Use {{ node_a.output }}.",
                        )
                    ],
                    synthesis=SwrmSynthesis(
                        provider="openai",
                        model="gpt-4o-mini",
                        prompt="Summary using {{ node_b.output }}.",
                    ),
                ),
            },
        )
        with pytest.raises(InterpolationError):
            check_circular_template_refs(wf)

    def test_factory_node_refs_scanned(self) -> None:
        wf = Workflow(
            version="0.1",
            agents={
                "worker": AgentDefinition(model="openai:gpt-4o-mini", system="Do task."),
            },
            nodes={
                "plan": AgentNode(agent="worker", writes="working.plan.output"),
                "execute": FactoryNode(
                    agent="worker",
                    for_each="{{ plan.output }}",
                    inputs={"task": "{{ item }}"},
                    writes="working.execute.outputs",
                ),
            },
        )
        check_circular_template_refs(wf)  # One-directional reference — no cycle


# ---------------------------------------------------------------------------
# Integration: resolve_template with os.environ
# ---------------------------------------------------------------------------


class TestEnvVarIntegration:
    def test_resolves_real_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SIREN_TEST_VAR", "test_value_123")
        ctx = build_interpolation_context("input", {})
        result = resolve_template("Value: {{ env.SIREN_TEST_VAR }}", ctx)
        assert result == "Value: test_value_123"

    def test_redacts_real_env_var(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SIREN_SECRET", "do_not_log_me")
        ctx = build_interpolation_context("input", {})
        result = resolve_template("Secret: {{ env.SIREN_SECRET }}", ctx, redact_env=True)
        assert result == "Secret: ***"
        assert "do_not_log_me" not in result
