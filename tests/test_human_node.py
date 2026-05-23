"""Tests for the HumanNode model, human runner, and executor integration (issue #76)."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sirenspec.core.executor import execute, execute_streaming
from sirenspec.core.human_runner import HumanRunResult, execute_human_node, resolve_timeout_fallback
from sirenspec.core.models import AgentDefinition, AgentNode, Edge, HumanNode, Workflow
from sirenspec.core.usage import TokenUsage
from sirenspec.exceptions import HumanInputError

# ---------------------------------------------------------------------------
# HumanNode model
# ---------------------------------------------------------------------------


class TestHumanNodeModel:
    def test_minimal_human_node(self) -> None:
        node = HumanNode(type="human", writes="working.approval")
        assert node.type == "human"
        assert node.writes == "working.approval"
        assert node.timeout is None
        assert node.on_timeout == "abort"
        assert node.prompt is None
        assert node.default_output is None

    def test_full_human_node(self) -> None:
        node = HumanNode(
            type="human",
            prompt="Approve? {{ draft.output }}",
            writes="working.approval",
            timeout=60.0,
            on_timeout="use_default",
            default_output="approved",
        )
        assert node.timeout == 60.0
        assert node.on_timeout == "use_default"
        assert node.default_output == "approved"

    def test_use_default_requires_default_output(self) -> None:
        with pytest.raises(ValueError, match="default_output"):
            HumanNode(type="human", writes="working.x", on_timeout="use_default")

    def test_skip_without_default_is_valid(self) -> None:
        node = HumanNode(type="human", writes="working.x", timeout=1.0, on_timeout="skip")
        assert node.default_output is None

    def test_abort_without_default_is_valid(self) -> None:
        node = HumanNode(type="human", writes="working.x", timeout=1.0, on_timeout="abort")
        assert node.default_output is None

    def test_negative_timeout_rejected(self) -> None:
        with pytest.raises(ValueError):
            HumanNode(type="human", writes="working.x", timeout=-1.0)

    def test_zero_timeout_rejected(self) -> None:
        with pytest.raises(ValueError):
            HumanNode(type="human", writes="working.x", timeout=0)


# ---------------------------------------------------------------------------
# Parse_node discriminator
# ---------------------------------------------------------------------------


class TestParseNodeDiscriminator:
    def test_workflow_loads_human_node_from_dict(self) -> None:
        from sirenspec.core.models import parse_node

        raw = {"type": "human", "writes": "working.x", "prompt": "hi"}
        node = parse_node(raw)
        assert isinstance(node, HumanNode)
        assert node.prompt == "hi"

    def test_workflow_with_human_node_validates(self) -> None:
        wf = Workflow(
            version="0.1",
            agents={"a": AgentDefinition(model="openai:gpt-4o-mini", system="hi")},
            nodes={
                "approve": HumanNode(type="human", writes="working.approval"),
                "act": AgentNode(agent="a", writes="output.result"),
            },
            edges=[Edge(**{"from": "approve", "to": "act"})],
        )
        assert isinstance(wf.nodes["approve"], HumanNode)


# ---------------------------------------------------------------------------
# resolve_timeout_fallback
# ---------------------------------------------------------------------------


class TestResolveTimeoutFallback:
    def test_abort_raises(self) -> None:
        node = HumanNode(type="human", writes="working.x", timeout=1.0, on_timeout="abort")
        with pytest.raises(HumanInputError) as exc_info:
            resolve_timeout_fallback(node)
        assert exc_info.value.timed_out is True

    def test_skip_returns_empty_string(self) -> None:
        node = HumanNode(type="human", writes="working.x", timeout=1.0, on_timeout="skip")
        assert resolve_timeout_fallback(node) == ""

    def test_use_default_returns_default_output(self) -> None:
        node = HumanNode(
            type="human",
            writes="working.x",
            timeout=1.0,
            on_timeout="use_default",
            default_output="auto-approved",
        )
        assert resolve_timeout_fallback(node) == "auto-approved"


# ---------------------------------------------------------------------------
# execute_human_node
# ---------------------------------------------------------------------------


class TestExecuteHumanNode:
    @pytest.mark.asyncio
    async def test_returns_response_from_input_fn(self) -> None:
        node = HumanNode(type="human", writes="working.x", prompt="ok?")

        async def fake_input(_prompt: str) -> str:
            return "yes"

        result = await execute_human_node("approve", node, "ok?", input_fn=fake_input)
        assert isinstance(result, HumanRunResult)
        assert result.response == "yes"
        assert result.timed_out is False
        assert result.duration_ms >= 0

    @pytest.mark.asyncio
    async def test_timeout_abort_raises_human_input_error(self) -> None:
        node = HumanNode(type="human", writes="working.x", timeout=0.05, on_timeout="abort")

        async def slow_input(_prompt: str) -> str:
            await asyncio.sleep(1.0)
            return "late"

        with pytest.raises(HumanInputError) as exc_info:
            await execute_human_node("approve", node, "ok?", input_fn=slow_input)
        assert exc_info.value.timed_out is True
        assert exc_info.value.node_id == "approve"

    @pytest.mark.asyncio
    async def test_timeout_use_default_returns_default(self) -> None:
        node = HumanNode(
            type="human",
            writes="working.x",
            timeout=0.05,
            on_timeout="use_default",
            default_output="default-value",
        )

        async def slow_input(_prompt: str) -> str:
            await asyncio.sleep(1.0)
            return "late"

        result = await execute_human_node("approve", node, "ok?", input_fn=slow_input)
        assert result.response == "default-value"
        assert result.timed_out is True

    @pytest.mark.asyncio
    async def test_timeout_skip_returns_empty(self) -> None:
        node = HumanNode(type="human", writes="working.x", timeout=0.05, on_timeout="skip")

        async def slow_input(_prompt: str) -> str:
            await asyncio.sleep(1.0)
            return "late"

        result = await execute_human_node("approve", node, "ok?", input_fn=slow_input)
        assert result.response == ""
        assert result.timed_out is True

    @pytest.mark.asyncio
    async def test_no_timeout_waits_indefinitely(self) -> None:
        node = HumanNode(type="human", writes="working.x")  # no timeout

        async def quick(_prompt: str) -> str:
            await asyncio.sleep(0.01)
            return "ok"

        result = await execute_human_node("approve", node, "", input_fn=quick)
        assert result.response == "ok"
        assert result.timed_out is False


# ---------------------------------------------------------------------------
# Executor integration — execute()
# ---------------------------------------------------------------------------


def _provider(response: str = "mocked", tokens: int = 10) -> MagicMock:
    mock = MagicMock()
    mock.complete = AsyncMock(return_value=response)
    mock.last_token_usage = TokenUsage(prompt_tokens=0, completion_tokens=tokens)
    return mock


def _build_workflow_with_human() -> Workflow:
    return Workflow(
        version="0.1",
        agents={
            "drafter": AgentDefinition(model="openai:gpt-4o-mini", system="Draft."),
            "publisher": AgentDefinition(model="openai:gpt-4o-mini", system="Publish."),
        },
        nodes={
            "draft": AgentNode(agent="drafter", writes="working.draft"),
            "approve": HumanNode(
                type="human",
                prompt="Approve: {{ draft.output }}",
                writes="working.approval",
            ),
            "publish": AgentNode(agent="publisher", writes="output.final"),
        },
        edges=[
            Edge(**{"from": "draft", "to": "approve"}),
            Edge(**{"from": "approve", "to": "publish", "when": 'working.approval == "yes"'}),
        ],
    )


class TestExecutorHumanNodeIntegration:
    @pytest.mark.asyncio
    async def test_human_node_writes_response_to_context(self) -> None:
        wf = _build_workflow_with_human()
        provider = _provider("draft body")

        async def fake_input(_prompt: str) -> str:
            return "yes"

        with patch("sirenspec.core.agent_runner.resolve_provider", return_value=provider):
            trace = await execute(wf, "topic", human_input_fn=fake_input)

        assert trace["summary"]["status"] == "success"
        approve_trace = next(n for n in trace["nodes"] if n["id"] == "approve")
        assert approve_trace["type"] == "human"
        assert approve_trace["response_received"] == "yes"
        assert approve_trace["writes"] == "working.approval"
        assert approve_trace["tokens"] == 0
        assert approve_trace["duration_ms"] >= 0

    @pytest.mark.asyncio
    async def test_human_node_gates_downstream_via_when_condition(self) -> None:
        wf = _build_workflow_with_human()
        provider = _provider("draft body")

        async def reject(_prompt: str) -> str:
            return "no"

        with patch("sirenspec.core.agent_runner.resolve_provider", return_value=provider):
            trace = await execute(wf, "topic", human_input_fn=reject)

        node_ids = [n["id"] for n in trace["nodes"]]
        assert "draft" in node_ids
        assert "approve" in node_ids
        assert "publish" not in node_ids
        assert trace["summary"]["status"] == "success"

    @pytest.mark.asyncio
    async def test_human_node_prompt_template_resolved(self) -> None:
        wf = _build_workflow_with_human()
        provider = _provider("the draft body")
        captured: list[str] = []

        async def capture(prompt: str) -> str:
            captured.append(prompt)
            return "yes"

        with patch("sirenspec.core.agent_runner.resolve_provider", return_value=provider):
            await execute(wf, "topic", human_input_fn=capture)

        assert captured[0] == "Approve: the draft body"

    @pytest.mark.asyncio
    async def test_human_node_timeout_abort_fails_workflow(self) -> None:
        wf = Workflow(
            version="0.1",
            agents={},
            nodes={
                "approve": HumanNode(
                    type="human",
                    writes="working.approval",
                    timeout=0.05,
                    on_timeout="abort",
                ),
            },
        )

        async def slow(_prompt: str) -> str:
            await asyncio.sleep(1.0)
            return "late"

        trace = await execute(wf, "topic", human_input_fn=slow)

        assert trace["summary"]["status"] == "failed"
        approve_trace = trace["nodes"][0]
        assert approve_trace["timed_out"] is True
        assert "Human node" in approve_trace["error"]

    @pytest.mark.asyncio
    async def test_human_node_timeout_use_default_writes_default(self) -> None:
        wf = Workflow(
            version="0.1",
            agents={},
            nodes={
                "approve": HumanNode(
                    type="human",
                    writes="output.decision",
                    timeout=0.05,
                    on_timeout="use_default",
                    default_output="auto-approved",
                ),
            },
        )

        async def slow(_prompt: str) -> str:
            await asyncio.sleep(1.0)
            return "late"

        trace = await execute(wf, "topic", human_input_fn=slow)

        assert trace["summary"]["status"] == "success"
        assert trace["output"]["decision"] == "auto-approved"
        assert trace["nodes"][0]["timed_out"] is True

    @pytest.mark.asyncio
    async def test_human_node_consumes_no_tokens(self) -> None:
        wf = _build_workflow_with_human()
        provider = _provider("draft body", tokens=10)

        async def yes(_prompt: str) -> str:
            return "yes"

        with patch("sirenspec.core.agent_runner.resolve_provider", return_value=provider):
            trace = await execute(wf, "topic", human_input_fn=yes)

        # Three nodes ran: draft (10 tokens), approve (0 tokens), publish (10 tokens).
        assert trace["summary"]["total_tokens"] == 20


# ---------------------------------------------------------------------------
# Streaming executor integration
# ---------------------------------------------------------------------------


class TestStreamingHumanNode:
    @pytest.mark.asyncio
    async def test_streaming_emits_human_event(self) -> None:
        from sirenspec.core.events import NodeCompleteEvent

        wf = _build_workflow_with_human()
        provider = _provider("draft")

        async def approve(_prompt: str) -> str:
            return "yes"

        events: list[NodeCompleteEvent] = []
        with patch("sirenspec.core.agent_runner.resolve_provider", return_value=provider):
            async for event in execute_streaming(wf, "topic", human_input_fn=approve):
                if isinstance(event, NodeCompleteEvent):
                    events.append(event)

        approve_events = [e for e in events if e.node_id == "approve"]
        assert len(approve_events) == 1
        assert approve_events[0].node_type == "human"
        assert approve_events[0].status == "success"
        assert approve_events[0].output == "yes"
