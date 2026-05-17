"""Unit tests for the factory node type and runner."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sirenspec.core.executor import execute
from sirenspec.core.factory_runner import execute_factory_node, run_factory_instance
from sirenspec.core.models import AgentDefinition, Edge, FactoryNode, Node, Workflow
from sirenspec.core.usage import TokenUsage
from sirenspec.exceptions import FactoryNodeError, InterpolationError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_provider_mock(response_text: str = "mock result", tokens: int = 10) -> MagicMock:
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
