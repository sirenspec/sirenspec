"""Unit tests for the swrm parallel agent primitive."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sirenspec.core.executor import execute
from sirenspec.core.models import AgentDefinition, Edge, Node, SwrmAgent, SwrmNode, SwrmSynthesis, Workflow
from sirenspec.core.swrm import build_template_context, render_template, execute_swrm
from sirenspec.exceptions import SwrmAgentError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_provider_mock(response_text: str = "mock response", tokens: int = 10) -> MagicMock:
    mock = MagicMock()
    mock.complete = AsyncMock(return_value=response_text)
    mock.last_token_count = tokens
    return mock


def _make_swrm_workflow(
    *,
    concurrency: int | None = None,
    on_failure: str = "abort",
    with_synthesis: bool = True,
) -> Workflow:
    """Build a minimal swrm-only workflow for testing."""
    synthesis = None
    if with_synthesis:
        synthesis = SwrmSynthesis(
            provider="openai",
            model="gpt-4o-mini",
            prompt="Sentiment: {{ analyze.agents.sentiment.output }}\nRisk: {{ analyze.agents.risk.output }}",
        )
    return Workflow(
        version="0.1",
        agents={},
        nodes={
            "analyze": SwrmNode(
                type="swrm",
                concurrency=concurrency,
                on_failure=on_failure,  # type: ignore[arg-type]
                agents=[
                    SwrmAgent(
                        id="sentiment", provider="openai", model="gpt-4o-mini", prompt="Analyze: {{ inputs.message }}"
                    ),
                    SwrmAgent(
                        id="risk",
                        provider="anthropic",
                        model="claude-haiku-4-5-20251001",
                        prompt="Risks: {{ inputs.message }}",
                    ),
                ],
                synthesis=synthesis,
            )
        },
    )


# ---------------------------------------------------------------------------
# Template rendering
# ---------------------------------------------------------------------------


class TestRenderTemplate:
    def test_simple_substitution(self) -> None:
        result = render_template("Hello {{ name }}", {"name": "World"})
        assert result == "Hello World"

    def test_dotted_path(self) -> None:
        ctx = {"inputs": {"message": "test report"}}
        result = render_template("Report: {{ inputs.message }}", ctx)
        assert result == "Report: test report"

    def test_deeply_nested_path(self) -> None:
        ctx = {"analyze": {"agents": {"sentiment": {"output": "bullish"}}}}
        result = render_template("Sentiment: {{ analyze.agents.sentiment.output }}", ctx)
        assert result == "Sentiment: bullish"

    def test_missing_key_leaves_placeholder(self) -> None:
        result = render_template("{{ missing.key }}", {})
        assert "missing.key" in result

    def test_no_placeholders(self) -> None:
        result = render_template("No placeholders here", {"x": 1})
        assert result == "No placeholders here"

    def test_multiple_placeholders(self) -> None:
        ctx = {"a": "first", "b": "second"}
        result = render_template("{{ a }} and {{ b }}", ctx)
        assert result == "first and second"


class TestBuildTemplateContext:
    def test_includes_inputs(self) -> None:
        ctx = build_template_context("node1", "hello world", {}, {})
        assert ctx["inputs"]["message"] == "hello world"

    def test_includes_working_and_output(self) -> None:
        working = {"x": 1}
        output = {"y": 2}
        ctx = build_template_context("node1", "input", working, output)
        assert ctx["working"]["x"] == 1
        assert ctx["output"]["y"] == 2

    def test_agent_results_injected(self) -> None:
        agent_results = {"sentiment": "bullish", "risk": "high"}
        ctx = build_template_context("analyze", "input", {}, {}, agent_results)
        assert ctx["analyze"]["agents"]["sentiment"]["output"] == "bullish"
        assert ctx["analyze"]["agents"]["risk"]["output"] == "high"

    def test_no_agent_results_by_default(self) -> None:
        ctx = build_template_context("node1", "input", {}, {})
        assert "node1" not in ctx


# ---------------------------------------------------------------------------
# execute_swrm: all agents succeed
# ---------------------------------------------------------------------------


class TestSwrmAllAgentsSucceed:
    @pytest.mark.asyncio
    async def test_returns_synthesis_output(self) -> None:
        """When synthesis is present, the node output is the synthesis text."""
        call_count = 0

        async def mock_complete(messages: list[dict]) -> str:
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return f"agent output {call_count}"
            return "synthesis output"

        mock_provider = MagicMock()
        mock_provider.complete = mock_complete
        mock_provider.last_token_count = 5

        with patch("sirenspec.core.swrm.resolve_provider", return_value=mock_provider):
            node = SwrmNode(
                type="swrm",
                agents=[
                    SwrmAgent(id="a1", provider="openai", model="gpt-4o-mini", prompt="p1"),
                    SwrmAgent(id="a2", provider="openai", model="gpt-4o-mini", prompt="p2"),
                ],
                synthesis=SwrmSynthesis(provider="openai", model="gpt-4o-mini", prompt="synth"),
            )
            trace = await execute_swrm("test", node, "user input", {}, {}, None)

        assert trace["output"] == "synthesis output"
        assert trace["error"] is None
        assert len(trace["agents"]) == 2
        assert trace["synthesis"] is not None
        assert trace["synthesis"]["response_received"] == "synthesis output"

    @pytest.mark.asyncio
    async def test_no_synthesis_output_is_list(self) -> None:
        """Without a synthesis block, output is a list of agent outputs in order."""
        responses = ["response A", "response B"]
        idx = 0

        async def mock_complete(messages: list[dict]) -> str:
            nonlocal idx
            val = responses[idx % len(responses)]
            idx += 1
            return val

        mock_provider = MagicMock()
        mock_provider.complete = mock_complete
        mock_provider.last_token_count = 3

        with patch("sirenspec.core.swrm.resolve_provider", return_value=mock_provider):
            node = SwrmNode(
                type="swrm",
                agents=[
                    SwrmAgent(id="a1", provider="openai", model="gpt-4o-mini", prompt="p1"),
                    SwrmAgent(id="a2", provider="openai", model="gpt-4o-mini", prompt="p2"),
                ],
                synthesis=None,
            )
            trace = await execute_swrm("test", node, "input", {}, {}, None)

        assert isinstance(trace["output"], list)
        assert len(trace["output"]) == 2
        assert trace["synthesis"] is None

    @pytest.mark.asyncio
    async def test_agent_outputs_in_trace(self) -> None:
        """Each agent's output is accessible in the agents list."""
        mock_provider = _make_provider_mock("agent result", tokens=7)
        with patch("sirenspec.core.swrm.resolve_provider", return_value=mock_provider):
            node = SwrmNode(
                type="swrm",
                agents=[SwrmAgent(id="only", provider="openai", model="gpt-4o-mini", prompt="do something")],
                synthesis=None,
            )
            trace = await execute_swrm("mynode", node, "hi", {}, {}, None)

        assert trace["agents"][0]["id"] == "only"
        assert trace["agents"][0]["response_received"] == "agent result"
        assert trace["agents"][0]["tokens"] == 7

    @pytest.mark.asyncio
    async def test_token_and_duration_totals(self) -> None:
        """Total tokens and duration are summed across all agents + synthesis."""
        mock_provider = _make_provider_mock("result", tokens=10)
        with patch("sirenspec.core.swrm.resolve_provider", return_value=mock_provider):
            node = SwrmNode(
                type="swrm",
                agents=[
                    SwrmAgent(id="a", provider="openai", model="gpt-4o-mini", prompt="p"),
                    SwrmAgent(id="b", provider="openai", model="gpt-4o-mini", prompt="q"),
                ],
                synthesis=SwrmSynthesis(provider="openai", model="gpt-4o-mini", prompt="s"),
            )
            trace = await execute_swrm("n", node, "input", {}, {}, None)

        # 2 agents + 1 synthesis = 30 tokens total.
        assert trace["tokens"] == 30

    @pytest.mark.asyncio
    async def test_synthesis_can_interpolate_agent_outputs(self) -> None:
        """The synthesis prompt receives agent outputs via {{ node.agents.<id>.output }}."""
        call_args: list[list[dict]] = []

        async def capture_complete(messages: list[dict]) -> str:
            call_args.append(messages)
            return "captured"

        mock_provider = MagicMock()
        mock_provider.complete = capture_complete
        mock_provider.last_token_count = 1

        # Sequence: first two calls are agents, third is synthesis.
        call_num = [0]

        async def sequenced_complete(messages: list[dict]) -> str:
            call_num[0] += 1
            n = call_num[0]
            if n == 1:
                return "sentiment result"
            if n == 2:
                return "risk result"
            # Third call is synthesis — capture its prompt.
            call_args.append(messages)
            return "synthesis result"

        mock_provider.complete = sequenced_complete

        with patch("sirenspec.core.swrm.resolve_provider", return_value=mock_provider):
            node = SwrmNode(
                type="swrm",
                agents=[
                    SwrmAgent(id="sentiment", provider="openai", model="gpt-4o-mini", prompt="sent prompt"),
                    SwrmAgent(id="risk", provider="openai", model="gpt-4o-mini", prompt="risk prompt"),
                ],
                synthesis=SwrmSynthesis(
                    provider="openai",
                    model="gpt-4o-mini",
                    prompt="Sentiment: {{ test_node.agents.sentiment.output }} Risk: {{ test_node.agents.risk.output }}",
                ),
            )
            trace = await execute_swrm("test_node", node, "input", {}, {}, None)

        # The synthesis prompt_sent should contain the interpolated agent outputs.
        assert "sentiment result" in trace["synthesis"]["prompt_sent"]
        assert "risk result" in trace["synthesis"]["prompt_sent"]


# ---------------------------------------------------------------------------
# execute_swrm: agent failure with abort
# ---------------------------------------------------------------------------


class TestSwrmAgentFailureAbort:
    @pytest.mark.asyncio
    async def test_abort_raises_swrm_agent_error(self) -> None:
        """When on_failure=abort and an agent fails, SwrmAgentError is raised."""
        call_num = [0]

        async def failing_complete(messages: list[dict]) -> str:
            call_num[0] += 1
            if call_num[0] == 1:
                raise RuntimeError("provider timeout")
            return "ok"

        mock_provider = MagicMock()
        mock_provider.complete = failing_complete
        mock_provider.last_token_count = 0

        with patch("sirenspec.core.swrm.resolve_provider", return_value=mock_provider):
            node = SwrmNode(
                type="swrm",
                on_failure="abort",
                agents=[
                    SwrmAgent(id="bad", provider="openai", model="gpt-4o-mini", prompt="p"),
                    SwrmAgent(id="good", provider="openai", model="gpt-4o-mini", prompt="q"),
                ],
                synthesis=None,
            )
            with pytest.raises(SwrmAgentError) as exc_info:
                await execute_swrm("n", node, "input", {}, {}, None)

        assert exc_info.value.agent_id == "bad"
        assert "provider timeout" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_abort_surfaced_in_workflow_trace(self) -> None:
        """When the swrm node aborts, the workflow trace reflects failure status."""
        mock_provider = MagicMock()
        mock_provider.complete = AsyncMock(side_effect=RuntimeError("boom"))
        mock_provider.last_token_count = 0

        wf = Workflow(
            version="0.1",
            agents={},
            nodes={
                "analyze": SwrmNode(
                    type="swrm",
                    on_failure="abort",
                    agents=[SwrmAgent(id="a", provider="openai", model="gpt-4o-mini", prompt="p")],
                    synthesis=None,
                )
            },
        )

        with patch("sirenspec.core.swrm.resolve_provider", return_value=mock_provider):
            trace = await execute(wf, "input")

        assert trace["summary"]["status"] == "failed"
        assert trace["nodes"][0]["error"] is not None


# ---------------------------------------------------------------------------
# execute_swrm: agent failure with continue
# ---------------------------------------------------------------------------


class TestSwrmAgentFailureContinue:
    @pytest.mark.asyncio
    async def test_continue_does_not_raise(self) -> None:
        """When on_failure=continue, a single agent failure does not abort the node."""
        call_num = [0]

        async def partial_fail(messages: list[dict]) -> str:
            call_num[0] += 1
            if call_num[0] == 1:
                raise RuntimeError("agent failed")
            return "good output"

        mock_provider = MagicMock()
        mock_provider.complete = partial_fail
        mock_provider.last_token_count = 5

        with patch("sirenspec.core.swrm.resolve_provider", return_value=mock_provider):
            node = SwrmNode(
                type="swrm",
                on_failure="continue",
                agents=[
                    SwrmAgent(id="bad", provider="openai", model="gpt-4o-mini", prompt="p"),
                    SwrmAgent(id="good", provider="openai", model="gpt-4o-mini", prompt="q"),
                ],
                synthesis=None,
            )
            trace = await execute_swrm("n", node, "input", {}, {}, None)

        # No exception raised; trace records the error for the failed agent.
        assert trace["error"] is None
        failed_agent = next(a for a in trace["agents"] if a["id"] == "bad")
        assert failed_agent["error"] is not None
        good_agent = next(a for a in trace["agents"] if a["id"] == "good")
        assert good_agent["response_received"] == "good output"

    @pytest.mark.asyncio
    async def test_continue_output_omits_failed_agent(self) -> None:
        """With continue, the failed agent's output is '' in the list output."""
        mock_provider = MagicMock()
        mock_provider.complete = AsyncMock(side_effect=RuntimeError("fail"))
        mock_provider.last_token_count = 0

        with patch("sirenspec.core.swrm.resolve_provider", return_value=mock_provider):
            node = SwrmNode(
                type="swrm",
                on_failure="continue",
                agents=[SwrmAgent(id="a", provider="openai", model="gpt-4o-mini", prompt="p")],
                synthesis=None,
            )
            trace = await execute_swrm("n", node, "input", {}, {}, None)

        assert isinstance(trace["output"], list)
        assert trace["output"] == [""]


# ---------------------------------------------------------------------------
# Full workflow integration: swrm within execute()
# ---------------------------------------------------------------------------


class TestSwrmInWorkflow:
    @pytest.mark.asyncio
    async def test_swrm_node_writes_output_to_context(self) -> None:
        """After a swrm node runs, output is accessible in workflow context."""
        mock_provider = _make_provider_mock("agent text", tokens=5)
        wf = Workflow(
            version="0.1",
            agents={},
            nodes={
                "analyze": SwrmNode(
                    type="swrm",
                    agents=[SwrmAgent(id="a", provider="openai", model="gpt-4o-mini", prompt="p")],
                    synthesis=None,
                )
            },
        )
        with patch("sirenspec.core.swrm.resolve_provider", return_value=mock_provider):
            trace = await execute(wf, "hello")

        assert trace["summary"]["status"] == "success"
        assert "analyze" in trace["output"]

    @pytest.mark.asyncio
    async def test_swrm_followed_by_agent_node(self) -> None:
        """A swrm node can feed into a regular agent node via an edge."""
        call_num = [0]

        async def mock_complete(messages: list[dict]) -> str:
            call_num[0] += 1
            if call_num[0] == 1:
                return "parallel result"
            return "downstream result"

        mock_swrm_provider = MagicMock()
        mock_swrm_provider.complete = mock_complete
        mock_swrm_provider.last_token_count = 5

        mock_agent_provider = MagicMock()
        mock_agent_provider.complete = mock_complete
        mock_agent_provider.last_token_count = 8

        wf = Workflow(
            version="0.1",
            agents={"writer": AgentDefinition(model="openai:gpt-4o-mini", system="Summarise.")},
            nodes={
                "fan_out": SwrmNode(
                    type="swrm",
                    agents=[SwrmAgent(id="a", provider="openai", model="gpt-4o-mini", prompt="p")],
                    synthesis=None,
                ),
                "summary": Node(agent="writer", writes="output.final"),
            },
            edges=[Edge(**{"from": "fan_out", "to": "summary"})],
        )

        def resolve_side_effect(uri: str) -> MagicMock:
            return mock_swrm_provider

        with (
            patch("sirenspec.core.swrm.resolve_provider", side_effect=resolve_side_effect),
            patch("sirenspec.core.executor.resolve_provider", side_effect=resolve_side_effect),
        ):
            trace = await execute(wf, "input")

        node_ids = [n["id"] for n in trace["nodes"]]
        assert "fan_out" in node_ids
        assert "summary" in node_ids

    @pytest.mark.asyncio
    async def test_swrm_concurrency_limit_respected(self) -> None:
        """With concurrency=1, agents run serially (one at a time)."""
        import asyncio

        active = [0]
        max_active = [0]

        async def concurrency_probe(messages: list[dict]) -> str:
            active[0] += 1
            max_active[0] = max(max_active[0], active[0])
            await asyncio.sleep(0.01)
            active[0] -= 1
            return "ok"

        mock_provider = MagicMock()
        mock_provider.complete = concurrency_probe
        mock_provider.last_token_count = 1

        with patch("sirenspec.core.swrm.resolve_provider", return_value=mock_provider):
            node = SwrmNode(
                type="swrm",
                concurrency=1,
                agents=[
                    SwrmAgent(id="a", provider="openai", model="gpt-4o-mini", prompt="p"),
                    SwrmAgent(id="b", provider="openai", model="gpt-4o-mini", prompt="q"),
                    SwrmAgent(id="c", provider="openai", model="gpt-4o-mini", prompt="r"),
                ],
                synthesis=None,
            )
            await execute_swrm("n", node, "input", {}, {}, None)

        assert max_active[0] == 1


# ---------------------------------------------------------------------------
# Model validation
# ---------------------------------------------------------------------------


class TestSwrmModelValidation:
    def test_swrm_node_requires_at_least_one_agent(self) -> None:
        """SwrmNode must have at least one agent."""
        import pydantic

        with pytest.raises(pydantic.ValidationError):
            SwrmNode(type="swrm", agents=[])

    def test_swrm_node_default_on_failure_is_abort(self) -> None:
        node = SwrmNode(
            type="swrm",
            agents=[SwrmAgent(id="a", provider="openai", prompt="p")],
        )
        assert node.on_failure == "abort"

    def test_swrm_node_default_concurrency_is_none(self) -> None:
        node = SwrmNode(
            type="swrm",
            agents=[SwrmAgent(id="a", provider="openai", prompt="p")],
        )
        assert node.concurrency is None

    def test_workflow_with_swrm_node_validates(self) -> None:
        """A Workflow containing a SwrmNode passes validation."""
        wf = Workflow(
            version="0.1",
            agents={},
            nodes={
                "analyze": SwrmNode(
                    type="swrm",
                    agents=[SwrmAgent(id="a", provider="openai", prompt="p")],
                )
            },
        )
        assert wf.nodes["analyze"].type == "swrm"

    def test_workflow_with_mixed_nodes_validates(self) -> None:
        """A Workflow can contain both regular and swrm nodes."""
        wf = Workflow(
            version="0.1",
            agents={"bot": AgentDefinition(model="openai:gpt-4o-mini", system="sys")},
            nodes={
                "parallel": SwrmNode(
                    type="swrm",
                    agents=[SwrmAgent(id="a", provider="openai", prompt="p")],
                ),
                "final": Node(agent="bot", writes="output.result"),
            },
            edges=[Edge(**{"from": "parallel", "to": "final"})],
        )
        assert len(wf.nodes) == 2
