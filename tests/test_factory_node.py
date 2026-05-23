"""Unit tests for the factory node type and runner."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from sirenspec.core.executor import execute
from sirenspec.core.factory_runner import MAX_SWARM_SIZE, execute_factory_node, run_factory_instance
from sirenspec.core.models import (
    AgentDefinition,
    Edge,
    FactoryNode,
    FactorySwrm,
    Node,
    SwrmAgent,
    SwrmSynthesis,
    Workflow,
)
from sirenspec.core.usage import TokenUsage
from sirenspec.exceptions import FactoryNodeError, GuardrailError, InterpolationError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_provider_mock(response_text: str = "mock result", tokens: int = 10) -> MagicMock:
    """Create a mock LLM provider with a fixed response and token count.

    :param response_text: Text returned by the mock provider's ``complete`` call.
    :param tokens: Completion token count reported by ``last_token_usage``.
    :returns: Configured MagicMock with async ``complete`` and ``last_token_usage``.
    """
    mock = MagicMock()
    mock.complete = AsyncMock(return_value=response_text)
    mock.last_token_usage = TokenUsage(prompt_tokens=0, completion_tokens=tokens)
    return mock


def _make_workflow_with_factory(
    *,
    for_each: str = '["task A", "task B"]',
    inputs: dict[str, str] | None = None,
    concurrency: int = 4,
    timeout_per_instance: int = 60,
    on_failure: str = "abort",
) -> Workflow:
    """Build a minimal workflow with one agent+for_each factory node for testing.

    :param for_each: JSON list string used as the ``for_each`` expression.
    :param inputs: Input bindings for the factory node; defaults to task+index.
    :param concurrency: Max concurrent factory instances.
    :param timeout_per_instance: Per-instance timeout in seconds.
    :param on_failure: Failure policy (``'abort'`` or ``'continue'``).
    :returns: Workflow containing a single ``execute`` factory node.
    """
    return Workflow(
        version="0.1",
        agents={
            "worker": AgentDefinition(model="openai:gpt-4o-mini", system="Do the task."),
        },
        nodes={
            "execute": FactoryNode(
                agent="worker",
                for_each=for_each,
                inputs=inputs or {"task": "{{ item }}", "index": "{{ index }}"},
                concurrency=concurrency,
                timeout_per_instance=timeout_per_instance,
                on_failure=on_failure,  # type: ignore[arg-type]
                writes="working.execute.outputs",
            )
        },
    )


# ---------------------------------------------------------------------------
# FactoryNode model validation
# ---------------------------------------------------------------------------


class TestFactoryNodeModel:
    def test_minimal_factory_node(self) -> None:
        node = FactoryNode(
            agent="worker",
            for_each='["a"]',
            writes="working.execute.outputs",
        )
        assert node.type == "factory"
        assert node.agent == "worker"

    def test_defaults(self) -> None:
        node = FactoryNode(agent="w", for_each="[]", writes="working.out")
        assert node.concurrency == 1
        assert node.timeout_per_instance == 60
        assert node.on_failure == "abort"
        assert node.inputs == {}

    def test_factory_node_in_workflow(self) -> None:
        wf = _make_workflow_with_factory()
        assert isinstance(wf.nodes["execute"], FactoryNode)

    def test_factory_node_invalid_agent_raises(self) -> None:
        import pydantic

        with pytest.raises(pydantic.ValidationError):
            Workflow(
                version="0.1",
                agents={},
                nodes={
                    "execute": FactoryNode(
                        agent="nonexistent_agent",
                        for_each='["x"]',
                        writes="working.execute.outputs",
                    )
                },
            )


# ---------------------------------------------------------------------------
# run_factory_instance
# ---------------------------------------------------------------------------


class TestRunFactoryInstance:
    @pytest.mark.asyncio
    async def test_successful_instance(self) -> None:
        mock_provider = _make_provider_mock("task result", tokens=15)
        node = FactoryNode(agent="worker", for_each="[]", writes="working.out")
        agent_def = AgentDefinition(model="openai:gpt-4o-mini", system="Do task.")

        with patch("sirenspec.core.agent_runner.resolve_provider", return_value=mock_provider):
            trace, error = await run_factory_instance(
                node_id="execute",
                idx=0,
                item="task A",
                node=node,
                agent_def=agent_def,
                base_working={},
                user_input="goal",
                guardrail_names=None,
            )

        assert error is None
        assert trace["response_received"] == "task result"
        assert trace["tokens"] == 15
        assert trace["index"] == 0
        assert trace["item"] == "task A"
        assert trace["error"] is None

    @pytest.mark.asyncio
    async def test_item_and_index_resolved_in_inputs(self) -> None:
        captured_messages: list[list[dict]] = []

        async def capture(messages: list[dict]) -> str:
            captured_messages.append(messages)
            return "done"

        mock_provider = MagicMock()
        mock_provider.complete = capture
        mock_provider.last_token_usage = TokenUsage(prompt_tokens=0, completion_tokens=5)

        node = FactoryNode(
            agent="worker",
            for_each="[]",
            inputs={"task": "{{ item }}", "idx": "{{ index }}"},
            writes="working.out",
        )
        agent_def = AgentDefinition(model="openai:gpt-4o-mini", system="Process.")

        with patch("sirenspec.core.agent_runner.resolve_provider", return_value=mock_provider):
            trace, error = await run_factory_instance(
                node_id="execute",
                idx=2,
                item="my task",
                node=node,
                agent_def=agent_def,
                base_working={},
                user_input="goal",
                guardrail_names=None,
            )

        assert error is None
        user_message = captured_messages[0][-1]["content"]
        assert "my task" in user_message
        assert "2" in user_message

    @pytest.mark.asyncio
    async def test_timeout_returns_factory_node_error(self) -> None:
        async def slow_complete(messages: list[dict]) -> str:
            await asyncio.sleep(10)
            return "too late"

        mock_provider = MagicMock()
        mock_provider.complete = slow_complete
        mock_provider.last_token_usage = TokenUsage(prompt_tokens=0, completion_tokens=0)

        node = FactoryNode(
            agent="worker",
            for_each="[]",
            timeout_per_instance=1,
            writes="working.out",
        )
        agent_def = AgentDefinition(model="openai:gpt-4o-mini", system="Slow.")

        with patch("sirenspec.core.agent_runner.resolve_provider", return_value=mock_provider):
            trace, error = await run_factory_instance(
                node_id="execute",
                idx=0,
                item="task",
                node=node,
                agent_def=agent_def,
                base_working={},
                user_input="goal",
                guardrail_names=[],
            )

        assert error is not None
        assert isinstance(error, FactoryNodeError)
        assert error.instance_index == 0
        assert trace["error"] is not None
        assert trace["response_received"] is None

    @pytest.mark.asyncio
    async def test_provider_error_returns_factory_node_error(self) -> None:
        mock_provider = MagicMock()
        mock_provider.complete = AsyncMock(side_effect=RuntimeError("provider boom"))
        mock_provider.last_token_usage = TokenUsage(prompt_tokens=0, completion_tokens=0)

        node = FactoryNode(agent="worker", for_each="[]", writes="working.out")
        agent_def = AgentDefinition(model="openai:gpt-4o-mini", system="Do.")

        with patch("sirenspec.core.agent_runner.resolve_provider", return_value=mock_provider):
            trace, error = await run_factory_instance(
                node_id="execute",
                idx=1,
                item="task B",
                node=node,
                agent_def=agent_def,
                base_working={},
                user_input="goal",
                guardrail_names=[],
            )

        assert error is not None
        assert error.instance_index == 1
        assert "provider boom" in trace["error"]


# ---------------------------------------------------------------------------
# execute_factory_node
# ---------------------------------------------------------------------------


class TestExecuteFactoryNode:
    def _make_workflow(self) -> Workflow:
        return _make_workflow_with_factory()

    @pytest.mark.asyncio
    async def test_all_instances_succeed(self) -> None:
        mock_provider = _make_provider_mock("result", tokens=10)
        wf = _make_workflow_with_factory(for_each='["a", "b", "c"]')
        node = wf.nodes["execute"]
        assert isinstance(node, FactoryNode)

        with patch("sirenspec.core.agent_runner.resolve_provider", return_value=mock_provider):
            trace = await execute_factory_node(
                node_id="execute",
                node=node,
                workflow=wf,
                user_input="goal",
                working={},
                output={},
                guardrail_names=None,
            )

        assert trace["error"] is None
        assert len(trace["instances"]) == 3
        assert trace["outputs"] == ["result", "result", "result"]
        assert trace["tokens"] == 30

    @pytest.mark.asyncio
    async def test_for_each_not_list_raises_interpolation_error(self) -> None:
        wf = _make_workflow_with_factory(for_each='{"not": "a list"}')
        node = wf.nodes["execute"]
        assert isinstance(node, FactoryNode)

        with pytest.raises(InterpolationError):
            await execute_factory_node(
                node_id="execute",
                node=node,
                workflow=wf,
                user_input="goal",
                working={},
                output={},
                guardrail_names=None,
            )

    @pytest.mark.asyncio
    async def test_on_failure_abort_raises(self) -> None:
        mock_provider = MagicMock()
        mock_provider.complete = AsyncMock(side_effect=RuntimeError("fail"))
        mock_provider.last_token_usage = TokenUsage(prompt_tokens=0, completion_tokens=0)

        wf = _make_workflow_with_factory(on_failure="abort")
        node = wf.nodes["execute"]
        assert isinstance(node, FactoryNode)

        with patch("sirenspec.core.agent_runner.resolve_provider", return_value=mock_provider):
            with pytest.raises(FactoryNodeError):
                await execute_factory_node(
                    node_id="execute",
                    node=node,
                    workflow=wf,
                    user_input="goal",
                    working={},
                    output={},
                    guardrail_names=[],
                )

    @pytest.mark.asyncio
    async def test_on_failure_continue_collects_all_results(self) -> None:
        call_num = [0]

        async def partial_fail(messages: list[dict]) -> str:
            call_num[0] += 1
            if call_num[0] == 1:
                raise RuntimeError("first fails")
            return "success"

        mock_provider = MagicMock()
        mock_provider.complete = partial_fail
        mock_provider.last_token_usage = TokenUsage(prompt_tokens=0, completion_tokens=5)

        wf = _make_workflow_with_factory(for_each='["x", "y"]', on_failure="continue")
        node = wf.nodes["execute"]
        assert isinstance(node, FactoryNode)

        with patch("sirenspec.core.agent_runner.resolve_provider", return_value=mock_provider):
            trace = await execute_factory_node(
                node_id="execute",
                node=node,
                workflow=wf,
                user_input="goal",
                working={},
                output={},
                guardrail_names=[],
            )

        assert trace["error"] is None
        assert len(trace["instances"]) == 2
        failed = next(i for i in trace["instances"] if i["error"] is not None)
        assert failed is not None
        assert "" in trace["outputs"]
        assert "success" in trace["outputs"]

    @pytest.mark.asyncio
    async def test_concurrency_limit_respected(self) -> None:
        active = [0]
        max_active = [0]

        async def probe(messages: list[dict]) -> str:
            active[0] += 1
            max_active[0] = max(max_active[0], active[0])
            await asyncio.sleep(0.01)
            active[0] -= 1
            return "ok"

        mock_provider = MagicMock()
        mock_provider.complete = probe
        mock_provider.last_token_usage = TokenUsage(prompt_tokens=0, completion_tokens=1)

        wf = _make_workflow_with_factory(
            for_each='["a", "b", "c", "d"]',
            concurrency=2,
        )
        node = wf.nodes["execute"]
        assert isinstance(node, FactoryNode)

        with patch("sirenspec.core.agent_runner.resolve_provider", return_value=mock_provider):
            await execute_factory_node(
                node_id="execute",
                node=node,
                workflow=wf,
                user_input="goal",
                working={},
                output={},
                guardrail_names=[],
            )

        assert max_active[0] <= 2

    @pytest.mark.asyncio
    async def test_item_resolves_in_for_each_from_working(self) -> None:
        mock_provider = _make_provider_mock("done", tokens=3)
        items_json = json.dumps(["alpha", "beta"])
        working = {"plan": {"output": items_json}}

        wf = Workflow(
            version="0.1",
            agents={"worker": AgentDefinition(model="openai:gpt-4o-mini", system="Do.")},
            nodes={
                "execute": FactoryNode(
                    agent="worker",
                    for_each="{{ plan.output }}",
                    inputs={"task": "{{ item }}"},
                    writes="working.execute.outputs",
                )
            },
        )
        node = wf.nodes["execute"]
        assert isinstance(node, FactoryNode)

        with patch("sirenspec.core.agent_runner.resolve_provider", return_value=mock_provider):
            trace = await execute_factory_node(
                node_id="execute",
                node=node,
                workflow=wf,
                user_input="goal",
                working=working,
                output={},
                guardrail_names=[],
            )

        assert len(trace["instances"]) == 2
        assert trace["instances"][0]["item"] == "alpha"
        assert trace["instances"][1]["item"] == "beta"


# ---------------------------------------------------------------------------
# Integration: factory node within execute()
# ---------------------------------------------------------------------------


class TestFactoryNodeInWorkflow:
    @pytest.mark.asyncio
    async def test_factory_node_trace_present(self) -> None:
        mock_provider = _make_provider_mock("result", tokens=5)
        wf = _make_workflow_with_factory(for_each='["task1", "task2"]')

        with patch("sirenspec.core.agent_runner.resolve_provider", return_value=mock_provider):
            trace = await execute(wf, "do it")

        assert trace["summary"]["status"] == "success"
        factory_trace = next(n for n in trace["nodes"] if n.get("type") == "factory")
        assert factory_trace is not None
        assert len(factory_trace["instances"]) == 2

    @pytest.mark.asyncio
    async def test_factory_outputs_accessible_in_downstream_node(self) -> None:
        call_num = [0]
        items = ["task A", "task B"]
        captured: list[str] = []

        async def mock_complete(messages: list[dict]) -> str:
            call_num[0] += 1
            n = call_num[0]
            if n <= len(items):
                return f"result {n}"
            captured.append(messages[-1]["content"])
            return "aggregated"

        mock_provider = MagicMock()
        mock_provider.complete = mock_complete
        mock_provider.last_token_usage = TokenUsage(prompt_tokens=0, completion_tokens=5)

        wf = Workflow(
            version="0.1",
            agents={
                "worker": AgentDefinition(model="openai:gpt-4o-mini", system="Do."),
                "aggregator": AgentDefinition(model="openai:gpt-4o-mini", system="Aggregate."),
            },
            nodes={
                "execute": FactoryNode(
                    agent="worker",
                    for_each='["task A", "task B"]',
                    inputs={"task": "{{ item }}"},
                    writes="working.execute.outputs",
                ),
                "aggregate": Node(agent="aggregator", writes="output.report"),
            },
            edges=[Edge(**{"from": "execute", "to": "aggregate"})],
        )

        with patch("sirenspec.core.agent_runner.resolve_provider", return_value=mock_provider):
            trace = await execute(wf, "run all")

        assert trace["summary"]["status"] == "success"
        node_ids = [n["id"] for n in trace["nodes"]]
        assert "execute" in node_ids
        assert "aggregate" in node_ids


# ---------------------------------------------------------------------------
# Swarm factory helpers
# ---------------------------------------------------------------------------


def _make_workflow_with_swarm(
    *,
    swarm_size: int | str = 3,
    inputs: dict[str, str] | None = None,
    concurrency: int = 4,
    on_failure: str = "abort",
) -> Workflow:
    return Workflow(
        version="0.1",
        agents={
            "worker": AgentDefinition(model="openai:gpt-4o-mini", system="Do the task."),
        },
        nodes={
            "execute": FactoryNode(
                agent="worker",
                swarm_size=swarm_size,
                inputs=inputs or {"position": "{{ index }} of {{ total }}"},
                concurrency=concurrency,
                on_failure=on_failure,  # type: ignore[arg-type]
                writes="working.execute.outputs",
            )
        },
    )


# ---------------------------------------------------------------------------
# Swarm factory model validation
# ---------------------------------------------------------------------------


class TestSwarmFactoryModel:
    def test_swarm_size_int_valid(self) -> None:
        node = FactoryNode(agent="worker", swarm_size=5, writes="out")
        assert node.swarm_size == 5
        assert node.for_each is None

    def test_swarm_size_template_string_valid(self) -> None:
        node = FactoryNode(agent="worker", swarm_size="{{ inputs.count }}", writes="out")
        assert node.swarm_size == "{{ inputs.count }}"
        assert node.for_each is None

    def test_both_set_raises_validation_error(self) -> None:
        import pydantic

        with pytest.raises(pydantic.ValidationError, match="Exactly one"):
            FactoryNode(agent="worker", for_each='["a"]', swarm_size=3, writes="out")

    def test_neither_set_raises_validation_error(self) -> None:
        import pydantic

        with pytest.raises(pydantic.ValidationError, match="Exactly one"):
            FactoryNode(agent="worker", writes="out")

    def test_existing_for_each_node_still_valid(self) -> None:
        node = FactoryNode(agent="worker", for_each='["a", "b"]', writes="out")
        assert node.for_each == '["a", "b"]'
        assert node.swarm_size is None


# ---------------------------------------------------------------------------
# Swarm factory execution
# ---------------------------------------------------------------------------


class TestSwarmFactoryExecution:
    @pytest.mark.asyncio
    async def test_swarm_spawns_correct_count(self) -> None:
        mock_provider = _make_provider_mock("ok", tokens=1)
        wf = _make_workflow_with_swarm(swarm_size=3)
        node = wf.nodes["execute"]
        assert isinstance(node, FactoryNode)

        with patch("sirenspec.core.agent_runner.resolve_provider", return_value=mock_provider):
            trace = await execute_factory_node(
                node_id="execute",
                node=node,
                workflow=wf,
                user_input="go",
                working={},
                output={},
                guardrail_names=None,
            )

        assert len(trace["instances"]) == 3
        assert len(trace["outputs"]) == 3

    @pytest.mark.asyncio
    async def test_swarm_instances_have_index_and_total(self) -> None:
        mock_provider = _make_provider_mock("ok", tokens=1)
        wf = _make_workflow_with_swarm(swarm_size=4)
        node = wf.nodes["execute"]
        assert isinstance(node, FactoryNode)

        with patch("sirenspec.core.agent_runner.resolve_provider", return_value=mock_provider):
            trace = await execute_factory_node(
                node_id="execute",
                node=node,
                workflow=wf,
                user_input="go",
                working={},
                output={},
                guardrail_names=None,
            )

        indices = [inst["index"] for inst in trace["instances"]]
        totals = [inst["total"] for inst in trace["instances"]]
        assert indices == [0, 1, 2, 3]
        assert all(t == 4 for t in totals)

    @pytest.mark.asyncio
    async def test_swarm_instance_item_is_none(self) -> None:
        mock_provider = _make_provider_mock("ok", tokens=1)
        wf = _make_workflow_with_swarm(swarm_size=2)
        node = wf.nodes["execute"]
        assert isinstance(node, FactoryNode)

        with patch("sirenspec.core.agent_runner.resolve_provider", return_value=mock_provider):
            trace = await execute_factory_node(
                node_id="execute",
                node=node,
                workflow=wf,
                user_input="go",
                working={},
                output={},
                guardrail_names=None,
            )

        assert len(trace["instances"]) == 2
        for inst in trace["instances"]:
            assert inst["item"] == "None"

    @pytest.mark.asyncio
    async def test_swarm_exceeds_max_raises_guardrail_error(self) -> None:
        wf = _make_workflow_with_swarm(swarm_size=MAX_SWARM_SIZE + 1)
        node = wf.nodes["execute"]
        assert isinstance(node, FactoryNode)
        assert node.swarm_size == MAX_SWARM_SIZE + 1

        with pytest.raises(GuardrailError, match="MAX_SWARM_SIZE"):
            await execute_factory_node(
                node_id="execute",
                node=node,
                workflow=wf,
                user_input="go",
                working={},
                output={},
                guardrail_names=None,
            )

    @pytest.mark.asyncio
    async def test_swarm_size_zero_raises_guardrail_error(self) -> None:
        node = FactoryNode(agent="worker", swarm_size=0, writes="out")
        assert node.swarm_size == 0
        wf = Workflow(
            version="0.1",
            agents={"worker": AgentDefinition(model="openai:gpt-4o-mini", system="Do.")},
            nodes={"execute": node},
        )

        with pytest.raises(GuardrailError, match="must be >= 1"):
            await execute_factory_node(
                node_id="execute",
                node=node,
                workflow=wf,
                user_input="go",
                working={},
                output={},
                guardrail_names=None,
            )

    @pytest.mark.asyncio
    async def test_dynamic_swarm_size_resolved_from_context(self) -> None:
        mock_provider = _make_provider_mock("ok", tokens=1)
        wf = _make_workflow_with_swarm(swarm_size="{{ plan.output }}")
        node = wf.nodes["execute"]
        assert isinstance(node, FactoryNode)

        working = {"plan": {"output": "3"}}

        with patch("sirenspec.core.agent_runner.resolve_provider", return_value=mock_provider):
            trace = await execute_factory_node(
                node_id="execute",
                node=node,
                workflow=wf,
                user_input="go",
                working=working,
                output={},
                guardrail_names=None,
            )

        assert len(trace["instances"]) == 3

    @pytest.mark.asyncio
    async def test_on_failure_abort_in_swarm_mode(self) -> None:
        mock_provider = _make_provider_mock("ok", tokens=1)
        mock_provider.complete = AsyncMock(side_effect=RuntimeError("boom"))
        wf = _make_workflow_with_swarm(swarm_size=2, on_failure="abort")
        node = wf.nodes["execute"]
        assert isinstance(node, FactoryNode)

        with patch("sirenspec.core.agent_runner.resolve_provider", return_value=mock_provider):
            with pytest.raises(FactoryNodeError):
                await execute_factory_node(
                    node_id="execute",
                    node=node,
                    workflow=wf,
                    user_input="go",
                    working={},
                    output={},
                    guardrail_names=None,
                )
        assert mock_provider.complete.call_count >= 1

    @pytest.mark.asyncio
    async def test_on_failure_continue_in_swarm_mode(self) -> None:
        call_count = [0]

        async def alternating_response(messages: list) -> str:
            call_count[0] += 1
            if call_count[0] % 2 == 1:
                raise RuntimeError("odd failure")
            return "even success"

        mock_provider = MagicMock()
        mock_provider.complete = alternating_response
        mock_provider.last_token_usage = TokenUsage(prompt_tokens=0, completion_tokens=1)

        wf = _make_workflow_with_swarm(swarm_size=4, on_failure="continue")
        node = wf.nodes["execute"]
        assert isinstance(node, FactoryNode)

        with patch("sirenspec.core.agent_runner.resolve_provider", return_value=mock_provider):
            trace = await execute_factory_node(
                node_id="execute",
                node=node,
                workflow=wf,
                user_input="go",
                working={},
                output={},
                guardrail_names=None,
            )

        assert len(trace["instances"]) == 4
        assert len(trace["outputs"]) == 4
        errors = [inst["error"] for inst in trace["instances"]]
        assert sum(1 for e in errors if e is not None) == 2


# ---------------------------------------------------------------------------
# Swarm factory — property-based tests
# ---------------------------------------------------------------------------


class TestSwarmFactoryHypothesis:
    @given(swarm_size=st.integers(min_value=1, max_value=MAX_SWARM_SIZE))
    @settings(max_examples=20)
    @pytest.mark.asyncio
    async def test_swarm_always_produces_n_results(self, swarm_size: int) -> None:
        mock_provider = _make_provider_mock("ok", tokens=1)
        wf = _make_workflow_with_swarm(swarm_size=swarm_size, concurrency=swarm_size)
        node = wf.nodes["execute"]
        assert isinstance(node, FactoryNode)

        with patch("sirenspec.core.agent_runner.resolve_provider", return_value=mock_provider):
            trace = await execute_factory_node(
                node_id="execute",
                node=node,
                workflow=wf,
                user_input="go",
                working={},
                output={},
                guardrail_names=None,
            )

        assert len(trace["instances"]) == swarm_size
        assert len(trace["outputs"]) == swarm_size

    @given(swarm_size=st.integers(min_value=1, max_value=MAX_SWARM_SIZE))
    @settings(max_examples=20)
    @pytest.mark.asyncio
    async def test_continue_mode_output_length_equals_swarm_size(self, swarm_size: int) -> None:
        mock_provider = _make_provider_mock("ok", tokens=1)
        mock_provider.complete = AsyncMock(side_effect=RuntimeError("always fails"))

        wf = _make_workflow_with_swarm(swarm_size=swarm_size, concurrency=swarm_size, on_failure="continue")
        node = wf.nodes["execute"]
        assert isinstance(node, FactoryNode)

        with patch("sirenspec.core.agent_runner.resolve_provider", return_value=mock_provider):
            trace = await execute_factory_node(
                node_id="execute",
                node=node,
                workflow=wf,
                user_input="go",
                working={},
                output={},
                guardrail_names=None,
            )

        assert len(trace["outputs"]) == swarm_size
        assert all(out == "" for out in trace["outputs"])


# ---------------------------------------------------------------------------
# Swrm factory helpers
# ---------------------------------------------------------------------------


def _make_swrm_spec(*, with_synthesis: bool = False) -> FactorySwrm:
    """Construct a FactorySwrm with two test agents and optional synthesis.

    :param with_synthesis: When ``True``, attaches a synthesis step that references
        both agent outputs via ``{{ grade_papers.agents.<id>.output }}``.
    :returns: FactorySwrm with editor and grader agents.
    """
    agents = [
        SwrmAgent(id="editor", provider="openai", model="gpt-4o-mini", prompt="Edit: {{ item }}"),
        SwrmAgent(id="grader", provider="openai", model="gpt-4o-mini", prompt="Grade: {{ item }}"),
    ]
    synthesis = None
    if with_synthesis:
        synthesis = SwrmSynthesis(
            provider="openai",
            model="gpt-4o-mini",
            prompt=(
                "Editor: {{ grade_papers.agents.editor.output }}\n"
                "Grader: {{ grade_papers.agents.grader.output }}\n"
                "Final grade:"
            ),
        )
    return FactorySwrm(agents=agents, synthesis=synthesis)


def _make_workflow_with_swrm_factory(
    *,
    for_each: str = '["paper A", "paper B"]',
    with_synthesis: bool = False,
    concurrency: int = 4,
    on_failure: str = "abort",
) -> Workflow:
    """Build a minimal workflow with one swrm+for_each factory node for testing.

    :param for_each: JSON list string used as the ``for_each`` expression.
    :param with_synthesis: When ``True``, the swrm spec includes a synthesis step.
    :param concurrency: Max concurrent swrm instances (one per item).
    :param on_failure: Failure policy (``'abort'`` or ``'continue'``).
    :returns: Workflow containing a single ``grade_papers`` swrm factory node.
    """
    return Workflow(
        version="0.1",
        agents={},
        nodes={
            "grade_papers": FactoryNode(
                swrm=_make_swrm_spec(with_synthesis=with_synthesis),
                for_each=for_each,
                concurrency=concurrency,
                on_failure=on_failure,  # type: ignore[arg-type]
                writes="working.grades",
            )
        },
    )


# ---------------------------------------------------------------------------
# Swrm factory model validation
# ---------------------------------------------------------------------------


class TestSwrmFactoryNodeModel:
    def test_swrm_field_valid(self) -> None:
        node = FactoryNode(swrm=_make_swrm_spec(), for_each='["a"]', writes="out")
        assert node.swrm is not None
        assert node.agent is None

    def test_agent_and_swrm_both_set_raises(self) -> None:
        import pydantic

        with pytest.raises(pydantic.ValidationError, match="Exactly one"):
            FactoryNode(agent="worker", swrm=_make_swrm_spec(), for_each='["a"]', writes="out")

    def test_neither_agent_nor_swrm_raises(self) -> None:
        import pydantic

        with pytest.raises(pydantic.ValidationError, match="Exactly one"):
            FactoryNode(for_each='["a"]', writes="out")

    def test_swrm_with_swarm_size_raises(self) -> None:
        import pydantic

        with pytest.raises(pydantic.ValidationError, match="swarm_size"):
            FactoryNode(swrm=_make_swrm_spec(), swarm_size=3, writes="out")

    def test_swrm_requires_for_each(self) -> None:
        import pydantic

        with pytest.raises(pydantic.ValidationError, match="Exactly one"):
            FactoryNode(swrm=_make_swrm_spec(), writes="out")

    def test_existing_agent_for_each_still_valid(self) -> None:
        node = FactoryNode(agent="worker", for_each='["a"]', writes="out")
        assert node.agent == "worker"
        assert node.swrm is None


# ---------------------------------------------------------------------------
# Swrm factory execution
# ---------------------------------------------------------------------------


class TestSwrmFactoryExecution:
    @pytest.mark.asyncio
    async def test_swrm_factory_spawns_swrm_per_item(self) -> None:
        mock_provider = _make_provider_mock("ok", tokens=1)
        wf = _make_workflow_with_swrm_factory(for_each='["p1", "p2", "p3"]')
        node = wf.nodes["grade_papers"]
        assert isinstance(node, FactoryNode)

        with patch("sirenspec.core.agent_runner.resolve_provider", return_value=mock_provider):
            trace = await execute_factory_node(
                node_id="grade_papers",
                node=node,
                workflow=wf,
                user_input="grade",
                working={},
                output={},
                guardrail_names=None,
            )

        assert len(trace["instances"]) == 3
        assert len(trace["outputs"]) == 3

    @pytest.mark.asyncio
    async def test_swrm_factory_instance_trace_shape(self) -> None:
        mock_provider = _make_provider_mock("result", tokens=2)
        wf = _make_workflow_with_swrm_factory(for_each='["paper A"]')
        node = wf.nodes["grade_papers"]
        assert isinstance(node, FactoryNode)

        with patch("sirenspec.core.agent_runner.resolve_provider", return_value=mock_provider):
            trace = await execute_factory_node(
                node_id="grade_papers",
                node=node,
                workflow=wf,
                user_input="grade",
                working={},
                output={},
                guardrail_names=None,
            )

        inst = trace["instances"][0]
        assert inst["index"] == 0
        assert inst["item"] == "paper A"
        assert inst["swrm"] is not None
        assert "agents" in inst["swrm"]
        assert inst["swrm"]["output"] is not None

    @pytest.mark.asyncio
    async def test_swrm_factory_item_available_in_agent_prompt(self) -> None:
        captured: list[list[dict]] = []

        async def capture_complete(messages: list[dict]) -> str:
            captured.append(messages)
            return "reviewed"

        mock_provider = MagicMock()
        mock_provider.complete = capture_complete
        mock_provider.last_token_usage = TokenUsage(prompt_tokens=0, completion_tokens=1)

        wf = _make_workflow_with_swrm_factory(for_each='["essay text"]')
        node = wf.nodes["grade_papers"]
        assert isinstance(node, FactoryNode)

        with patch("sirenspec.core.agent_runner.resolve_provider", return_value=mock_provider):
            await execute_factory_node(
                node_id="grade_papers",
                node=node,
                workflow=wf,
                user_input="grade",
                working={},
                output={},
                guardrail_names=None,
            )

        all_prompts = " ".join(msg[-1]["content"] for msg in captured)
        assert "essay text" in all_prompts

    @pytest.mark.asyncio
    async def test_swrm_factory_synthesis_runs_per_item(self) -> None:
        mock_provider = _make_provider_mock("synthesized", tokens=1)
        wf = _make_workflow_with_swrm_factory(for_each='["paper A", "paper B"]', with_synthesis=True)
        node = wf.nodes["grade_papers"]
        assert isinstance(node, FactoryNode)

        with patch("sirenspec.core.agent_runner.resolve_provider", return_value=mock_provider):
            trace = await execute_factory_node(
                node_id="grade_papers",
                node=node,
                workflow=wf,
                user_input="grade",
                working={},
                output={},
                guardrail_names=None,
            )

        for inst in trace["instances"]:
            assert inst["swrm"]["synthesis"] is not None
            assert inst["swrm"]["synthesis"]["response_received"] == "synthesized"

    @pytest.mark.asyncio
    async def test_swrm_factory_on_failure_abort(self) -> None:
        mock_provider = _make_provider_mock("ok", tokens=1)
        mock_provider.complete = AsyncMock(side_effect=RuntimeError("agent boom"))
        wf = _make_workflow_with_swrm_factory(on_failure="abort")
        node = wf.nodes["grade_papers"]
        assert isinstance(node, FactoryNode)

        with patch("sirenspec.core.agent_runner.resolve_provider", return_value=mock_provider):
            with pytest.raises(FactoryNodeError):
                await execute_factory_node(
                    node_id="grade_papers",
                    node=node,
                    workflow=wf,
                    user_input="grade",
                    working={},
                    output={},
                    guardrail_names=None,
                )
        assert mock_provider.complete.call_count >= 1

    @pytest.mark.asyncio
    async def test_swrm_factory_on_failure_continue(self) -> None:
        call_count = [0]

        async def alternating(messages: list) -> str:
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("first agent of first paper fails")
            return "ok"

        mock_provider = MagicMock()
        mock_provider.complete = alternating
        mock_provider.last_token_usage = TokenUsage(prompt_tokens=0, completion_tokens=1)

        wf = _make_workflow_with_swrm_factory(for_each='["paper A", "paper B"]', on_failure="continue")
        node = wf.nodes["grade_papers"]
        assert isinstance(node, FactoryNode)

        with patch("sirenspec.core.agent_runner.resolve_provider", return_value=mock_provider):
            trace = await execute_factory_node(
                node_id="grade_papers",
                node=node,
                workflow=wf,
                user_input="grade",
                working={},
                output={},
                guardrail_names=None,
            )

        assert len(trace["instances"]) == 2
        assert len(trace["outputs"]) == 2

    @pytest.mark.asyncio
    async def test_swrm_factory_index_and_total_in_context(self) -> None:
        captured_prompts: list[str] = []

        async def capture(messages: list[dict]) -> str:
            captured_prompts.append(messages[-1]["content"])
            return "done"

        mock_provider = MagicMock()
        mock_provider.complete = capture
        mock_provider.last_token_usage = TokenUsage(prompt_tokens=0, completion_tokens=1)

        spec = FactorySwrm(
            agents=[SwrmAgent(id="a", provider="openai", model="gpt-4o-mini", prompt="{{ index }} of {{ total }}")],
        )
        wf = Workflow(
            version="0.1",
            agents={},
            nodes={"gp": FactoryNode(swrm=spec, for_each='["x", "y", "z"]', writes="out")},
        )
        node = wf.nodes["gp"]
        assert isinstance(node, FactoryNode)

        with patch("sirenspec.core.agent_runner.resolve_provider", return_value=mock_provider):
            await execute_factory_node(
                node_id="gp",
                node=node,
                workflow=wf,
                user_input="go",
                working={},
                output={},
                guardrail_names=None,
            )

        assert any("0 of 3" in p for p in captured_prompts)
        assert any("1 of 3" in p for p in captured_prompts)
        assert any("2 of 3" in p for p in captured_prompts)


# ---------------------------------------------------------------------------
# Swrm factory — property-based tests
# ---------------------------------------------------------------------------


class TestSwrmFactoryHypothesis:
    @given(n=st.integers(min_value=1, max_value=10))
    @settings(max_examples=15)
    @pytest.mark.asyncio
    async def test_swrm_factory_always_produces_n_instances(self, n: int) -> None:
        mock_provider = _make_provider_mock("ok", tokens=1)
        papers = json.dumps([f"paper {i}" for i in range(n)])
        wf = _make_workflow_with_swrm_factory(for_each=papers, concurrency=n)
        node = wf.nodes["grade_papers"]
        assert isinstance(node, FactoryNode)

        with patch("sirenspec.core.agent_runner.resolve_provider", return_value=mock_provider):
            trace = await execute_factory_node(
                node_id="grade_papers",
                node=node,
                workflow=wf,
                user_input="grade",
                working={},
                output={},
                guardrail_names=None,
            )

        assert len(trace["instances"]) == n
        assert len(trace["outputs"]) == n

    @given(n=st.integers(min_value=1, max_value=10))
    @settings(max_examples=15)
    @pytest.mark.asyncio
    async def test_swrm_factory_continue_mode_output_length_equals_n(self, n: int) -> None:
        mock_provider = _make_provider_mock("ok", tokens=1)
        mock_provider.complete = AsyncMock(side_effect=RuntimeError("all fail"))
        papers = json.dumps([f"paper {i}" for i in range(n)])
        wf = _make_workflow_with_swrm_factory(for_each=papers, concurrency=n, on_failure="continue")
        node = wf.nodes["grade_papers"]
        assert isinstance(node, FactoryNode)

        with patch("sirenspec.core.agent_runner.resolve_provider", return_value=mock_provider):
            trace = await execute_factory_node(
                node_id="grade_papers",
                node=node,
                workflow=wf,
                user_input="grade",
                working={},
                output={},
                guardrail_names=None,
            )

        assert len(trace["outputs"]) == n
