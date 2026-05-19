"""Integration tests for all docs/cookbook workflow examples.

Each test loads the real workflow YAML and exercises the full executor path
with mocked LLM providers (and mocked HTTP for tool nodes). No real API calls
are made.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sirenspec.core.executor import execute
from sirenspec.core.usage import TokenUsage
from sirenspec.yaml.parser import load_workflow

COOKBOOK = Path(__file__).parent.parent / "docs" / "cookbook"


def _provider(response: str = "mock response", tokens: int = 10) -> MagicMock:
    mock = MagicMock()
    mock.complete = AsyncMock(return_value=response)
    mock.last_token_usage = TokenUsage(prompt_tokens=0, completion_tokens=tokens)
    return mock


def _sequential_provider(*responses: str, tokens: int = 10) -> MagicMock:
    """Return a mock provider whose calls cycle through *responses* in order."""
    idx = [0]

    async def _complete(messages: list[dict[str, Any]]) -> str:
        val = responses[idx[0] % len(responses)]
        idx[0] += 1
        return val

    mock = MagicMock()
    mock.complete = _complete
    mock.last_token_usage = TokenUsage(prompt_tokens=0, completion_tokens=tokens)
    return mock


# ---------------------------------------------------------------------------
# simple-agent
# ---------------------------------------------------------------------------


class TestSimpleAgent:
    @pytest.mark.asyncio
    async def test_runs_successfully(self) -> None:
        wf = load_workflow(COOKBOOK / "simple-agent" / "workflow.yaml")
        provider = _provider("Paris is the capital of France.")
        with patch("sirenspec.core.agent_runner.resolve_provider", return_value=provider):
            trace = await execute(wf, wf.input.message if wf.input else "What is the capital of France?")
        assert trace["summary"]["status"] == "success"
        assert len(trace["nodes"]) == 1
        assert trace["nodes"][0]["id"] == "answer"
        assert trace["output"]["reply"] == "Paris is the capital of France."


# ---------------------------------------------------------------------------
# sequential-pipeline
# ---------------------------------------------------------------------------


class TestSequentialPipeline:
    @pytest.mark.asyncio
    async def test_runs_two_nodes(self) -> None:
        wf = load_workflow(COOKBOOK / "sequential-pipeline" / "workflow.yaml")
        provider = _sequential_provider("question", "Here is a helpful answer.")
        with patch("sirenspec.core.agent_runner.resolve_provider", return_value=provider):
            trace = await execute(wf, "How do I reset my password?")
        assert trace["summary"]["status"] == "success"
        node_ids = [n["id"] for n in trace["nodes"]]
        assert node_ids == ["classify", "reply"]
        assert trace["output"]["reply"] == "Here is a helpful answer."


# ---------------------------------------------------------------------------
# telephone-game
# ---------------------------------------------------------------------------


class TestTelephoneGame:
    @pytest.mark.asyncio
    async def test_five_hops(self) -> None:
        wf = load_workflow(COOKBOOK / "telephone-game" / "workflow.yaml")
        provider = _sequential_provider("hop1", "hop2", "hop3", "hop4", "hop5")
        with patch("sirenspec.core.agent_runner.resolve_provider", return_value=provider):
            trace = await execute(wf, wf.input.message if wf.input else "start message")
        assert trace["summary"]["status"] == "success"
        node_ids = [n["id"] for n in trace["nodes"]]
        assert node_ids == ["hop_1", "hop_2", "hop_3", "hop_4", "hop_5"]
        assert trace["output"]["final"] == "hop5"


# ---------------------------------------------------------------------------
# adversarial-pair
# ---------------------------------------------------------------------------


class TestAdversarialPair:
    @pytest.mark.asyncio
    async def test_three_nodes(self) -> None:
        wf = load_workflow(COOKBOOK / "adversarial-pair" / "workflow.yaml")
        provider = _sequential_provider("arg for", "arg against", "side_a wins")
        with patch("sirenspec.core.agent_runner.resolve_provider", return_value=provider):
            trace = await execute(wf, wf.input.message if wf.input else "tabs vs spaces")
        assert trace["summary"]["status"] == "success"
        node_ids = [n["id"] for n in trace["nodes"]]
        assert node_ids == ["argue_a", "argue_b", "adjudicate"]
        assert trace["output"]["verdict"] == "side_a wins"


# ---------------------------------------------------------------------------
# blind-code-review
# ---------------------------------------------------------------------------


class TestBlindCodeReview:
    @pytest.mark.asyncio
    async def test_three_stage_pipeline(self) -> None:
        wf = load_workflow(COOKBOOK / "blind-code-review" / "workflow.yaml")
        provider = _sequential_provider("def merge(): ...", "looks good", "def merge_v2(): ...")
        with patch("sirenspec.core.agent_runner.resolve_provider", return_value=provider):
            trace = await execute(wf, wf.input.message if wf.input else "Write a merge function.")
        assert trace["summary"]["status"] == "success"
        node_ids = [n["id"] for n in trace["nodes"]]
        assert node_ids == ["write_code", "review_code", "revise_code"]
        assert trace["output"]["code_final"] == "def merge_v2(): ..."


# ---------------------------------------------------------------------------
# compression-gauntlet
# ---------------------------------------------------------------------------


class TestCompressionGauntlet:
    @pytest.mark.asyncio
    async def test_four_rounds(self) -> None:
        wf = load_workflow(COOKBOOK / "compression-gauntlet" / "workflow.yaml")
        provider = _sequential_provider("summary1", "summary2", "summary3", "summary4")
        with patch("sirenspec.core.agent_runner.resolve_provider", return_value=provider):
            trace = await execute(wf, wf.input.message if wf.input else "Long document text here.")
        assert trace["summary"]["status"] == "success"
        node_ids = [n["id"] for n in trace["nodes"]]
        assert node_ids == ["round_1", "round_2", "round_3", "round_4"]
        assert trace["output"]["final"] == "summary4"


# ---------------------------------------------------------------------------
# conditional-pipeline
# ---------------------------------------------------------------------------


class TestConditionalPipeline:
    @pytest.mark.asyncio
    async def test_routes_to_refund_handler(self) -> None:
        wf = load_workflow(COOKBOOK / "conditional-pipeline" / "workflow.yaml")
        provider = _sequential_provider("refund", "Your refund is processing.")
        with patch("sirenspec.core.agent_runner.resolve_provider", return_value=provider):
            trace = await execute(wf, "I want a refund for my order.")
        assert trace["summary"]["status"] == "success"
        node_ids = [n["id"] for n in trace["nodes"]]
        assert "triage" in node_ids
        assert "handle_refund" in node_ids
        assert "handle_general" not in node_ids
        assert trace["output"]["reply"] == "Your refund is processing."

    @pytest.mark.asyncio
    async def test_routes_to_general_handler(self) -> None:
        wf = load_workflow(COOKBOOK / "conditional-pipeline" / "workflow.yaml")
        provider = _sequential_provider("general", "Here is an answer to your question.")
        with patch("sirenspec.core.agent_runner.resolve_provider", return_value=provider):
            trace = await execute(wf, "What are your store hours?")
        assert trace["summary"]["status"] == "success"
        node_ids = [n["id"] for n in trace["nodes"]]
        assert "handle_general" in node_ids
        assert "handle_refund" not in node_ids


# ---------------------------------------------------------------------------
# graphic-design-firm
# ---------------------------------------------------------------------------


class TestGraphicDesignFirm:
    @pytest.mark.asyncio
    async def test_five_node_pipeline(self) -> None:
        wf = load_workflow(COOKBOOK / "graphic-design-firm" / "workflow.yaml")
        provider = _sequential_provider("brief", "copy", "<html>design</html>", "critique", "<html>revised</html>")
        with patch("sirenspec.core.agent_runner.resolve_provider", return_value=provider):
            trace = await execute(wf, wf.input.message if wf.input else "A new app called Canopy.")
        assert trace["summary"]["status"] == "success"
        node_ids = [n["id"] for n in trace["nodes"]]
        assert node_ids == ["brief", "copy", "design", "critique", "revise"]
        assert trace["output"]["final_html"] == "<html>revised</html>"


# ---------------------------------------------------------------------------
# news-desk
# ---------------------------------------------------------------------------


class TestNewsDesk:
    @pytest.mark.asyncio
    async def test_four_node_pipeline(self) -> None:
        wf = load_workflow(COOKBOOK / "news-desk" / "workflow.yaml")
        provider = _sequential_provider("raw draft", "edited draft", "1. Headline A\n2. Headline B", "final article")
        with patch("sirenspec.core.agent_runner.resolve_provider", return_value=provider):
            trace = await execute(wf, wf.input.message if wf.input else "A study on walking and cognition.")
        assert trace["summary"]["status"] == "success"
        node_ids = [n["id"] for n in trace["nodes"]]
        assert node_ids == ["report", "edit", "headline", "publish"]
        assert trace["output"]["article"] == "final article"


# ---------------------------------------------------------------------------
# 1000-monkeys (swrm with synthesis)
# ---------------------------------------------------------------------------


class TestThousandMonkeys:
    @pytest.mark.asyncio
    async def test_all_agents_succeed_synthesis_runs(self) -> None:
        wf = load_workflow(COOKBOOK / "1000-monkeys" / "workflow.yaml")
        call_num = [0]

        async def _complete(messages: list[dict[str, Any]]) -> str:
            call_num[0] += 1
            if call_num[0] <= 5:
                return f"poem attempt {call_num[0]}"
            return "chosen poem"

        provider = MagicMock()
        provider.complete = _complete
        provider.last_token_usage = TokenUsage(prompt_tokens=0, completion_tokens=5)

        with patch("sirenspec.core.agent_runner.resolve_provider", return_value=provider):
            trace = await execute(wf, wf.input.message if wf.input else "Write a three-line poem.")

        assert trace["summary"]["status"] == "success"
        assert len(trace["nodes"]) == 1
        swrm_node = trace["nodes"][0]
        assert swrm_node["type"] == "swrm"
        assert len(swrm_node["agents"]) == 5
        assert swrm_node["synthesis"]["response_received"] == "chosen poem"
        assert swrm_node["output"] == "chosen poem"

    @pytest.mark.asyncio
    async def test_all_agents_fail_synthesis_error_recorded(self) -> None:
        """When all agents fail with on_failure=continue, synthesis records an
        InterpolationError in synthesis_trace rather than propagating it."""
        wf = load_workflow(COOKBOOK / "1000-monkeys" / "workflow.yaml")
        provider = MagicMock()
        provider.complete = AsyncMock(side_effect=RuntimeError("no api key"))
        provider.last_token_usage = TokenUsage(prompt_tokens=0, completion_tokens=0)

        with patch("sirenspec.core.agent_runner.resolve_provider", return_value=provider):
            trace = await execute(wf, wf.input.message if wf.input else "Write a poem.")

        swrm_node = trace["nodes"][0]
        assert swrm_node["type"] == "swrm"
        assert len(swrm_node["agents"]) == 5
        for agent in swrm_node["agents"]:
            assert agent["error"] is not None
        assert swrm_node["synthesis"]["error"] is not None
        assert swrm_node["synthesis"]["error"] != ""


# ---------------------------------------------------------------------------
# market-analysis (swrm with synthesis)
# ---------------------------------------------------------------------------


class TestMarketAnalysis:
    @pytest.mark.asyncio
    async def test_three_agents_synthesis(self) -> None:
        wf = load_workflow(COOKBOOK / "market-analysis" / "workflow.yaml")
        call_num = [0]

        async def _complete(messages: list[dict[str, Any]]) -> str:
            call_num[0] += 1
            if call_num[0] == 1:
                return "bullish sentiment"
            if call_num[0] == 2:
                return "rising rates risk"
            if call_num[0] == 3:
                return "APAC opportunity"
            return "investment recommendation"

        provider = MagicMock()
        provider.complete = _complete
        provider.last_token_usage = TokenUsage(prompt_tokens=0, completion_tokens=8)

        with patch("sirenspec.core.agent_runner.resolve_provider", return_value=provider):
            trace = await execute(wf, wf.input.message if wf.input else "Q3 earnings exceeded expectations.")

        assert trace["summary"]["status"] == "success"
        swrm_node = trace["nodes"][0]
        assert len(swrm_node["agents"]) == 3
        assert swrm_node["synthesis"]["response_received"] == "investment recommendation"

    @pytest.mark.asyncio
    async def test_synthesis_interpolates_agent_outputs(self) -> None:
        """The synthesis prompt for market-analysis references all three agent IDs."""
        wf = load_workflow(COOKBOOK / "market-analysis" / "workflow.yaml")
        captured: list[str] = []
        call_num = [0]

        async def _complete(messages: list[dict[str, Any]]) -> str:
            call_num[0] += 1
            if call_num[0] == 1:
                return "SENTIMENT_OUTPUT"
            if call_num[0] == 2:
                return "RISK_OUTPUT"
            if call_num[0] == 3:
                return "OPPORTUNITY_OUTPUT"
            captured.append(messages[-1]["content"])
            return "recommendation"

        provider = MagicMock()
        provider.complete = _complete
        provider.last_token_usage = TokenUsage(prompt_tokens=0, completion_tokens=5)

        with patch("sirenspec.core.agent_runner.resolve_provider", return_value=provider):
            await execute(wf, "some report")

        assert captured, "synthesis was never called"
        assert "SENTIMENT_OUTPUT" in captured[0]
        assert "RISK_OUTPUT" in captured[0]
        assert "OPPORTUNITY_OUTPUT" in captured[0]


# ---------------------------------------------------------------------------
# pr-summarizer (tool node + agent, env interpolation in headers)
# ---------------------------------------------------------------------------


class TestPrSummarizer:
    @pytest.mark.asyncio
    async def test_tool_node_headers_interpolated(self) -> None:
        """The Authorization header must be interpolated from env.GITHUB_TOKEN."""
        wf = load_workflow(COOKBOOK / "pr-summarizer" / "workflow.yaml")
        provider = _provider("PR summary text")

        captured_headers: dict[str, str] = {}

        def fake_sync_request(config: Any) -> str:
            if config.headers:
                captured_headers.update(config.headers)
            return "raw diff content"

        with (
            patch("sirenspec.core.agent_runner.resolve_provider", return_value=provider),
            patch("sirenspec.tools.http_adapter.sync_request", side_effect=fake_sync_request),
            patch.dict("os.environ", {"GITHUB_TOKEN": "ghp_test_token"}),
        ):
            trace = await execute(wf, "Summarise this PR.")

        assert trace["summary"]["status"] == "success"
        assert captured_headers.get("Authorization") == "Bearer ghp_test_token"
        assert trace["output"]["summary"] == "PR summary text"

    @pytest.mark.asyncio
    async def test_tool_node_fails_when_token_missing(self) -> None:
        """When GITHUB_TOKEN is not set, interpolation raises and the node fails."""
        wf = load_workflow(COOKBOOK / "pr-summarizer" / "workflow.yaml")
        provider = _provider("summary")

        import os

        env_without_token = {k: v for k, v in os.environ.items() if k != "GITHUB_TOKEN"}

        with (
            patch("sirenspec.core.agent_runner.resolve_provider", return_value=provider),
            patch.dict("os.environ", env_without_token, clear=True),
        ):
            trace = await execute(wf, "Summarise this PR.")

        assert trace["summary"]["status"] == "failed"
        assert trace["nodes"][0]["error"] is not None
