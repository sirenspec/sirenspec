"""Unit tests for the sirenspec test framework."""

from __future__ import annotations

import asyncio
import textwrap
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sirenspec.testing.assertions import (
    evaluate_assertion,
    evaluate_operator,
    resolve_node_path,
    resolve_path,
)
from sirenspec.testing.cassette import (
    Cassette,
    CassetteError,
    CassetteInteraction,
    RecordingProvider,
    ReplayProvider,
    interaction_key,
    load_cassette,
    save_cassette,
)
from sirenspec.testing.models import Assertion, WorkflowFixture
from sirenspec.testing.runner import (
    discover_fixtures,
    load_fixture,
    resolve_workflow_path,
    run_fixture,
    run_fixtures,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

MINIMAL_TRACE: dict[str, Any] = {
    "workflow": {"version": "1.0"},
    "input": {"message": "hello"},
    "nodes": [
        {"id": "classify", "type": "agent", "status": "success", "output": "refund", "tokens": 10},
        {"id": "reply", "type": "agent", "status": "success", "output": "Here is your refund", "tokens": 20},
    ],
    "output": {"reply": "Here is your refund"},
    "summary": {
        "total_tokens": 30,
        "total_usage": {
            "prompt_tokens": 15,
            "completion_tokens": 15,
            "total_tokens": 30,
            "estimated_usd": 0.001,
        },
        "total_duration_ms": 500.0,
        "status": "success",
    },
}

MINIMAL_WORKFLOW_YAML = textwrap.dedent("""\
    version: "1.0"
    agents:
      responder:
        model: "openai:gpt-4o-mini"
        system: "You are a helpful assistant."
    nodes:
      respond:
        type: agent
        agent: responder
        writes: output.reply
    edges: []
""")

MINIMAL_FIXTURE_YAML = textwrap.dedent("""\
    workflow: workflow.yaml
    input: "What is the speed of light?"
    assertions:
      - path: output.reply
        contains: "299"
""")


# ---------------------------------------------------------------------------
# resolve_path
# ---------------------------------------------------------------------------


class TestResolvePath:
    def test_resolves_output_key(self) -> None:
        result = resolve_path(MINIMAL_TRACE, "output.reply")
        assert result == "Here is your refund"

    def test_resolves_total_usage_shorthand(self) -> None:
        result = resolve_path(MINIMAL_TRACE, "total_usage.estimated_usd")
        assert result == 0.001

    def test_resolves_node_field(self) -> None:
        result = resolve_path(MINIMAL_TRACE, "nodes.classify.status")
        assert result == "success"

    def test_resolves_node_output(self) -> None:
        result = resolve_path(MINIMAL_TRACE, "nodes.classify.output")
        assert result == "refund"

    def test_missing_node_raises(self) -> None:
        with pytest.raises(KeyError, match="No node with id 'missing'"):
            resolve_path(MINIMAL_TRACE, "nodes.missing.status")

    def test_missing_top_level_key_raises(self) -> None:
        with pytest.raises(KeyError):
            resolve_path(MINIMAL_TRACE, "nonexistent.key")

    def test_resolves_summary_field(self) -> None:
        result = resolve_path(MINIMAL_TRACE, "summary.status")
        assert result == "success"


# ---------------------------------------------------------------------------
# resolve_node_path
# ---------------------------------------------------------------------------


class TestResolveNodePath:
    def test_resolves_status_field(self) -> None:
        result = resolve_node_path(MINIMAL_TRACE, "classify", "status")
        assert result == "success"

    def test_returns_full_node_when_field_is_none(self) -> None:
        result = resolve_node_path(MINIMAL_TRACE, "classify", None)
        assert result["id"] == "classify"

    def test_missing_node_raises(self) -> None:
        with pytest.raises(KeyError, match="No node with id 'ghost'"):
            resolve_node_path(MINIMAL_TRACE, "ghost", "status")


# ---------------------------------------------------------------------------
# evaluate_operator
# ---------------------------------------------------------------------------


class TestEvaluateOperator:
    def test_equals_pass(self) -> None:
        a = Assertion(path="x", equals="refund")
        assert evaluate_operator(a, "refund").passed

    def test_equals_fail(self) -> None:
        a = Assertion(path="x", equals="refund")
        assert not evaluate_operator(a, "inquiry").passed

    def test_contains_string_pass(self) -> None:
        a = Assertion(path="x", contains="299")
        assert evaluate_operator(a, "The answer is 299,792 km/s").passed

    def test_contains_string_fail(self) -> None:
        a = Assertion(path="x", contains="299")
        assert not evaluate_operator(a, "unknown speed").passed

    def test_contains_list_pass(self) -> None:
        a = Assertion(path="x", contains="b")
        assert evaluate_operator(a, ["a", "b", "c"]).passed

    def test_matches_pass(self) -> None:
        a = Assertion(path="x", matches=r"^\d+$")
        assert evaluate_operator(a, "42").passed

    def test_matches_fail(self) -> None:
        a = Assertion(path="x", matches=r"^\d+$")
        assert not evaluate_operator(a, "abc").passed

    def test_matches_non_string_fails(self) -> None:
        a = Assertion(path="x", matches=r"\d+")
        result = evaluate_operator(a, 42)
        assert not result.passed

    def test_lt_pass(self) -> None:
        a = Assertion(path="x", lt=0.01)
        assert evaluate_operator(a, 0.001).passed

    def test_lt_fail(self) -> None:
        a = Assertion(path="x", lt=0.01)
        assert not evaluate_operator(a, 0.05).passed

    def test_gt_pass(self) -> None:
        a = Assertion(path="x", gt=100)
        assert evaluate_operator(a, 200).passed

    def test_gt_fail(self) -> None:
        a = Assertion(path="x", gt=100)
        assert not evaluate_operator(a, 50).passed

    def test_exists_true_pass(self) -> None:
        a = Assertion(path="x", exists=True)
        assert evaluate_operator(a, "something").passed

    def test_exists_true_fail(self) -> None:
        a = Assertion(path="x", exists=True)
        assert not evaluate_operator(a, None).passed

    def test_exists_false_pass(self) -> None:
        a = Assertion(path="x", exists=False)
        assert evaluate_operator(a, None).passed

    def test_status_pass(self) -> None:
        a = Assertion(node="classify", status="success")
        assert evaluate_operator(a, "success").passed

    def test_status_fail(self) -> None:
        a = Assertion(node="classify", status="success")
        assert not evaluate_operator(a, "failed").passed


# ---------------------------------------------------------------------------
# evaluate_assertion
# ---------------------------------------------------------------------------


class TestEvaluateAssertion:
    def test_path_contains_pass(self) -> None:
        a = Assertion(path="output.reply", contains="refund")
        result = evaluate_assertion(a, MINIMAL_TRACE)
        assert result.passed

    def test_path_contains_fail(self) -> None:
        a = Assertion(path="output.reply", contains="MISSING_TOKEN")
        result = evaluate_assertion(a, MINIMAL_TRACE)
        assert not result.passed

    def test_node_status_pass(self) -> None:
        a = Assertion(node="classify", status="success")
        result = evaluate_assertion(a, MINIMAL_TRACE)
        assert result.passed

    def test_node_path_output_matches(self) -> None:
        a = Assertion(node="classify", path="output", matches=r"^(refund|inquiry)$")
        result = evaluate_assertion(a, MINIMAL_TRACE)
        assert result.passed

    def test_missing_path_returns_failure(self) -> None:
        a = Assertion(path="output.nonexistent", equals="x")
        result = evaluate_assertion(a, MINIMAL_TRACE)
        assert not result.passed
        assert "path resolution failed" in result.message

    def test_total_usage_lt(self) -> None:
        a = Assertion(path="total_usage.estimated_usd", lt=1.0)
        result = evaluate_assertion(a, MINIMAL_TRACE)
        assert result.passed


# ---------------------------------------------------------------------------
# WorkflowFixture model validation
# ---------------------------------------------------------------------------


class TestWorkflowFixtureModel:
    def test_valid_fixture(self) -> None:
        fixture = WorkflowFixture(
            workflow="workflow.yaml",
            input="hello",
            assertions=[Assertion(path="output.reply", contains="hi")],
        )
        assert fixture.workflow == "workflow.yaml"

    def test_assertion_requires_path_or_node(self) -> None:
        with pytest.raises(ValueError):
            Assertion(equals="x")

    def test_assertion_requires_operator(self) -> None:
        with pytest.raises(ValueError):
            Assertion(path="output.reply")

    def test_assertion_node_status_valid(self) -> None:
        a = Assertion(node="classify", status="success")
        assert a.node == "classify"
        assert a.status == "success"


# ---------------------------------------------------------------------------
# interaction_key
# ---------------------------------------------------------------------------


class TestInteractionKey:
    def test_same_inputs_produce_same_key(self) -> None:
        messages = [{"role": "user", "content": "hello"}]
        assert interaction_key("openai:gpt-4o-mini", messages) == interaction_key("openai:gpt-4o-mini", messages)

    def test_different_messages_produce_different_keys(self) -> None:
        m1 = [{"role": "user", "content": "hello"}]
        m2 = [{"role": "user", "content": "bye"}]
        assert interaction_key("openai:gpt-4o-mini", m1) != interaction_key("openai:gpt-4o-mini", m2)

    def test_different_uris_produce_different_keys(self) -> None:
        messages = [{"role": "user", "content": "hello"}]
        assert interaction_key("openai:gpt-4o-mini", messages) != interaction_key(
            "anthropic:claude-haiku-4-5-20251001", messages
        )


# ---------------------------------------------------------------------------
# Cassette serialisation
# ---------------------------------------------------------------------------


class TestCassetteSerialization:
    def test_round_trip(self, tmp_path: Path) -> None:
        cassette = Cassette(
            interactions=[
                CassetteInteraction(key="abc123", response="hello world", prompt_tokens=10, completion_tokens=5)
            ]
        )
        cassette_path = tmp_path / "test.cassette.yaml"
        save_cassette(cassette, cassette_path)
        loaded = load_cassette(cassette_path)
        assert len(loaded.interactions) == 1
        assert loaded.interactions[0].key == "abc123"
        assert loaded.interactions[0].response == "hello world"
        assert loaded.interactions[0].prompt_tokens == 10

    def test_load_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(CassetteError, match="not found"):
            load_cassette(tmp_path / "nonexistent.yaml")

    def test_load_malformed_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text("not a mapping: [1, 2, 3\n", encoding="utf-8")
        with pytest.raises(CassetteError):
            load_cassette(bad)

    def test_load_missing_interactions_key_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text("foo: bar\n", encoding="utf-8")
        with pytest.raises(CassetteError, match="interactions"):
            load_cassette(bad)


# ---------------------------------------------------------------------------
# ReplayProvider
# ---------------------------------------------------------------------------


class TestReplayProvider:
    @pytest.mark.asyncio
    async def test_replay_returns_recorded_response(self) -> None:
        messages = [{"role": "user", "content": "speed of light?"}]
        key = interaction_key("openai:gpt-4o-mini", messages)
        cassette = Cassette(
            interactions=[CassetteInteraction(key=key, response="299,792 km/s", prompt_tokens=8, completion_tokens=6)]
        )
        provider = ReplayProvider(uri="openai:gpt-4o-mini", cassette=cassette)
        response = await provider.complete(messages)
        assert response == "299,792 km/s"
        assert provider.last_token_usage.prompt_tokens == 8

    @pytest.mark.asyncio
    async def test_replay_raises_on_missing_key(self) -> None:
        cassette = Cassette(interactions=[])
        provider = ReplayProvider(uri="openai:gpt-4o-mini", cassette=cassette)
        with pytest.raises(CassetteError, match="No cassette interaction"):
            await provider.complete([{"role": "user", "content": "hello"}])


# ---------------------------------------------------------------------------
# RecordingProvider
# ---------------------------------------------------------------------------


class TestRecordingProvider:
    @pytest.mark.asyncio
    async def test_records_interaction(self) -> None:
        from sirenspec.core.usage import TokenUsage

        real = MagicMock()
        real.complete = AsyncMock(return_value="The answer is 42")
        real.last_token_usage = TokenUsage(prompt_tokens=5, completion_tokens=3)
        real.client = None

        cassette = Cassette()
        provider = RecordingProvider(uri="openai:gpt-4o-mini", real_provider=real, cassette=cassette)
        messages = [{"role": "user", "content": "what is the answer?"}]
        response = await provider.complete(messages)
        assert response == "The answer is 42"
        assert len(cassette.interactions) == 1
        assert cassette.interactions[0].response == "The answer is 42"
        assert cassette.interactions[0].prompt_tokens == 5


# ---------------------------------------------------------------------------
# discover_fixtures
# ---------------------------------------------------------------------------


class TestDiscoverFixtures:
    def test_discovers_test_yaml_files(self, tmp_path: Path) -> None:
        (tmp_path / "a.test.yaml").write_text("", encoding="utf-8")
        (tmp_path / "b.test.yaml").write_text("", encoding="utf-8")
        (tmp_path / "not_a_test.yaml").write_text("", encoding="utf-8")
        found = discover_fixtures(tmp_path)
        names = [f.name for f in found]
        assert "a.test.yaml" in names
        assert "b.test.yaml" in names
        assert "not_a_test.yaml" not in names

    def test_returns_single_file_directly(self, tmp_path: Path) -> None:
        fixture = tmp_path / "single.test.yaml"
        fixture.write_text("", encoding="utf-8")
        found = discover_fixtures(fixture)
        assert found == [fixture]

    def test_missing_path_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            discover_fixtures(tmp_path / "nonexistent")

    def test_discovers_recursively(self, tmp_path: Path) -> None:
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "nested.test.yaml").write_text("", encoding="utf-8")
        found = discover_fixtures(tmp_path)
        assert any(f.name == "nested.test.yaml" for f in found)


# ---------------------------------------------------------------------------
# load_fixture
# ---------------------------------------------------------------------------


class TestLoadFixture:
    def test_loads_valid_fixture(self, tmp_path: Path) -> None:
        f = tmp_path / "basic.test.yaml"
        f.write_text(MINIMAL_FIXTURE_YAML, encoding="utf-8")
        fixture = load_fixture(f)
        assert fixture.input == "What is the speed of light?"
        assert len(fixture.assertions) == 1

    def test_raises_on_missing_required_field(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.test.yaml"
        f.write_text("input: hello\n", encoding="utf-8")
        with pytest.raises(ValueError, match="Fixture validation failed"):
            load_fixture(f)

    def test_raises_on_invalid_yaml(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.test.yaml"
        f.write_text("key: [unclosed\n", encoding="utf-8")
        with pytest.raises(ValueError, match="YAML parse error"):
            load_fixture(f)


# ---------------------------------------------------------------------------
# resolve_workflow_path
# ---------------------------------------------------------------------------


class TestResolveWorkflowPath:
    def test_resolves_relative_path(self, tmp_path: Path) -> None:
        fixture = tmp_path / "tests" / "basic.test.yaml"
        fixture.parent.mkdir(parents=True)
        result = resolve_workflow_path(fixture, "workflow.yaml")
        assert result == (tmp_path / "tests" / "workflow.yaml").resolve()

    def test_passes_through_absolute_path(self, tmp_path: Path) -> None:
        fixture = tmp_path / "basic.test.yaml"
        abs_path = tmp_path / "workflow.yaml"
        result = resolve_workflow_path(fixture, str(abs_path))
        assert result == abs_path


# ---------------------------------------------------------------------------
# run_fixture (integration — mocked provider)
# ---------------------------------------------------------------------------


class TestRunFixture:
    def _write_workflow(self, path: Path) -> None:
        path.write_text(MINIMAL_WORKFLOW_YAML, encoding="utf-8")

    def _write_fixture(self, path: Path, content: str) -> None:
        path.write_text(content, encoding="utf-8")

    def test_passing_assertion_with_mock_cassette(self, tmp_path: Path) -> None:
        wf = tmp_path / "workflow.yaml"
        self._write_workflow(wf)

        fixture_text = textwrap.dedent("""\
            workflow: workflow.yaml
            input: "What is the speed of light?"
            assertions:
              - path: output.reply
                contains: "speed"
        """)
        fx = tmp_path / "basic.test.yaml"
        self._write_fixture(fx, fixture_text)

        messages_key = interaction_key(
            "openai:gpt-4o-mini",
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the speed of light?"},
            ],
        )
        cassette = Cassette(
            interactions=[CassetteInteraction(key=messages_key, response="The speed of light is 299,792 km/s")]
        )

        cassette_path = tmp_path / "cassette.yaml"
        save_cassette(cassette, cassette_path)

        result = asyncio.run(run_fixture(fx, cassette_path, mode="mock"))
        assert result.passed
        assert result.error is None

    def test_failing_assertion_returns_false(self, tmp_path: Path) -> None:
        wf = tmp_path / "workflow.yaml"
        self._write_workflow(wf)

        fixture_text = textwrap.dedent("""\
            workflow: workflow.yaml
            input: "hello"
            assertions:
              - path: output.reply
                equals: "EXPECTED_EXACT_MATCH"
        """)
        fx = tmp_path / "basic.test.yaml"
        self._write_fixture(fx, fixture_text)

        messages_key = interaction_key(
            "openai:gpt-4o-mini",
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "hello"},
            ],
        )
        cassette = Cassette(interactions=[CassetteInteraction(key=messages_key, response="Hi there!")])
        cassette_path = tmp_path / "cassette.yaml"
        save_cassette(cassette, cassette_path)

        result = asyncio.run(run_fixture(fx, cassette_path, mode="mock"))
        assert not result.passed

    def test_missing_workflow_returns_error(self, tmp_path: Path) -> None:
        fx = tmp_path / "basic.test.yaml"
        self._write_fixture(fx, MINIMAL_FIXTURE_YAML)
        result = asyncio.run(run_fixture(fx, None, mode="live"))
        assert not result.passed
        assert result.error is not None

    def test_mock_with_missing_cassette_returns_error(self, tmp_path: Path) -> None:
        wf = tmp_path / "workflow.yaml"
        self._write_workflow(wf)
        fx = tmp_path / "basic.test.yaml"
        self._write_fixture(fx, MINIMAL_FIXTURE_YAML)
        missing = tmp_path / "missing.cassette.yaml"
        result = asyncio.run(run_fixture(fx, missing, mode="mock"))
        assert not result.passed
        assert result.error is not None

    def test_record_mode_writes_cassette(self, tmp_path: Path) -> None:
        wf = tmp_path / "workflow.yaml"
        self._write_workflow(wf)

        fixture_text = textwrap.dedent("""\
            workflow: workflow.yaml
            input: "hello"
            assertions:
              - path: output.reply
                exists: true
        """)
        fx = tmp_path / "basic.test.yaml"
        self._write_fixture(fx, fixture_text)
        cassette_path = tmp_path / "new.cassette.yaml"

        from sirenspec.core.usage import TokenUsage

        mock_provider = MagicMock()
        mock_provider.complete = AsyncMock(return_value="Hello back!")
        mock_provider.last_token_usage = TokenUsage(prompt_tokens=5, completion_tokens=3)
        mock_provider.client = None

        with patch("sirenspec.providers.registry._PROVIDER_FACTORIES", {"openai": lambda m: mock_provider}):
            asyncio.run(run_fixture(fx, cassette_path, mode="record"))

        assert cassette_path.exists()
        loaded = load_cassette(cassette_path)
        assert len(loaded.interactions) == 1
        assert loaded.interactions[0].response == "Hello back!"

    def test_invalid_fixture_yaml_returns_error(self, tmp_path: Path) -> None:
        fx = tmp_path / "bad.test.yaml"
        fx.write_text("not valid yaml: [unclosed\n", encoding="utf-8")
        result = asyncio.run(run_fixture(fx, None, mode="live"))
        assert not result.passed
        assert result.error is not None


# ---------------------------------------------------------------------------
# run_fixtures (batch)
# ---------------------------------------------------------------------------


class TestRunFixtures:
    def test_returns_one_result_per_fixture(self, tmp_path: Path) -> None:
        wf = tmp_path / "workflow.yaml"
        wf.write_text(MINIMAL_WORKFLOW_YAML, encoding="utf-8")

        for name in ("a.test.yaml", "b.test.yaml"):
            (tmp_path / name).write_text(
                textwrap.dedent("""\
                    workflow: workflow.yaml
                    input: "hi"
                    assertions: []
                """),
                encoding="utf-8",
            )

        from sirenspec.core.usage import TokenUsage

        mock_provider = MagicMock()
        mock_provider.complete = AsyncMock(return_value="hi back")
        mock_provider.last_token_usage = TokenUsage(prompt_tokens=2, completion_tokens=2)
        mock_provider.client = None

        with patch("sirenspec.providers.registry._PROVIDER_FACTORIES", {"openai": lambda m: mock_provider}):
            results = run_fixtures(
                [tmp_path / "a.test.yaml", tmp_path / "b.test.yaml"],
                cassette_path=None,
                mode="live",
            )

        assert len(results) == 2
