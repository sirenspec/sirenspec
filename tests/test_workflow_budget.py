"""Tests for the workflow-level ``budget:`` block and per-node ``max_tokens_per_call`` (issue #77)."""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sirenspec.core.budget import BudgetState, build_budget_status, detect_violation, enforce_budget
from sirenspec.core.executor import execute
from sirenspec.core.models import AgentDefinition, AgentNode, BudgetConfig, Edge, Workflow
from sirenspec.core.usage import TokenUsage
from sirenspec.exceptions import BudgetExceededError

# ---------------------------------------------------------------------------
# BudgetConfig model
# ---------------------------------------------------------------------------


class TestBudgetConfigModel:
    def test_tokens_only_valid(self) -> None:
        b = BudgetConfig(max_tokens=50000)
        assert b.max_tokens == 50000
        assert b.on_exceeded == "abort"

    def test_usd_only_valid(self) -> None:
        b = BudgetConfig(max_cost_usd=5.0)
        assert b.max_cost_usd == 5.0

    def test_duration_only_valid(self) -> None:
        b = BudgetConfig(max_duration_s=300.0)
        assert b.max_duration_s == 300.0

    def test_empty_block_rejected(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            BudgetConfig()

    def test_explicit_skip_remaining_action(self) -> None:
        b = BudgetConfig(max_tokens=1, on_exceeded="skip_remaining")
        assert b.on_exceeded == "skip_remaining"

    def test_negative_tokens_rejected(self) -> None:
        with pytest.raises(ValueError):
            BudgetConfig(max_tokens=-5)

    def test_zero_usd_rejected(self) -> None:
        with pytest.raises(ValueError):
            BudgetConfig(max_cost_usd=0)


# ---------------------------------------------------------------------------
# detect_violation
# ---------------------------------------------------------------------------


class TestDetectViolation:
    def test_within_token_budget(self) -> None:
        cfg = BudgetConfig(max_tokens=1000)
        assert detect_violation(cfg, TokenUsage(prompt_tokens=400, completion_tokens=200), None, 0) is None

    def test_token_ceiling_exceeded(self) -> None:
        cfg = BudgetConfig(max_tokens=100)
        result = detect_violation(cfg, TokenUsage(prompt_tokens=80, completion_tokens=30), None, 0)
        assert result is not None
        assert "110" in result and "100" in result

    def test_usd_ceiling_skipped_when_estimate_none(self) -> None:
        cfg = BudgetConfig(max_cost_usd=0.001)
        assert detect_violation(cfg, TokenUsage(prompt_tokens=10000, completion_tokens=10000), None, 0) is None

    def test_usd_ceiling_exceeded(self) -> None:
        cfg = BudgetConfig(max_cost_usd=0.01)
        result = detect_violation(cfg, TokenUsage(prompt_tokens=100, completion_tokens=50), 0.5, 0)
        assert result is not None
        assert "USD" in result or "$" in result

    def test_duration_ceiling_exceeded(self) -> None:
        cfg = BudgetConfig(max_duration_s=1.0)
        result = detect_violation(cfg, TokenUsage(prompt_tokens=10, completion_tokens=5), None, 5.0)
        assert result is not None
        assert "Duration" in result or "duration" in result


# ---------------------------------------------------------------------------
# enforce_budget
# ---------------------------------------------------------------------------


class TestEnforceBudget:
    def test_no_config_is_noop(self) -> None:
        state = BudgetState(config=None, start_time=0)
        # Must not raise even with massive usage.
        enforce_budget(state, TokenUsage(prompt_tokens=999_999, completion_tokens=999_999), 999_999)

    def test_abort_raises_when_over(self) -> None:
        cfg = BudgetConfig(max_tokens=10, on_exceeded="abort")
        state = BudgetState(config=cfg, start_time=0)
        with pytest.raises(BudgetExceededError):
            enforce_budget(state, TokenUsage(prompt_tokens=100, completion_tokens=50), None)

    def test_warn_does_not_raise(self, caplog: pytest.LogCaptureFixture) -> None:
        cfg = BudgetConfig(max_tokens=10, on_exceeded="warn")
        state = BudgetState(config=cfg, start_time=0)
        with caplog.at_level(logging.WARNING, logger="sirenspec.core.budget"):
            enforce_budget(state, TokenUsage(prompt_tokens=100, completion_tokens=50), None)
        assert state.skip_remaining is False
        assert any("budget warning" in r.message.lower() for r in caplog.records)

    def test_skip_remaining_latches_flag(self) -> None:
        cfg = BudgetConfig(max_tokens=10, on_exceeded="skip_remaining")
        state = BudgetState(config=cfg, start_time=0)
        enforce_budget(state, TokenUsage(prompt_tokens=100, completion_tokens=50), None)
        assert state.skip_remaining is True

    def test_identical_violation_deduplicated(self) -> None:
        """Repeating the exact same ceiling violation must record it only once."""

        cfg = BudgetConfig(max_tokens=10, on_exceeded="warn")
        state = BudgetState(config=cfg, start_time=0)
        # Two calls with identical totals produce the same violation message; the second is a no-op.
        usage = TokenUsage(prompt_tokens=100, completion_tokens=50)
        enforce_budget(state, usage, None)
        enforce_budget(state, usage, None)
        assert state.violations == ["Token ceiling exceeded: 150 tokens used, limit is 10"]


# ---------------------------------------------------------------------------
# build_budget_status
# ---------------------------------------------------------------------------


class TestBudgetStatus:
    def test_no_config_returns_none(self) -> None:
        state = BudgetState(config=None, start_time=0)
        assert build_budget_status(state, TokenUsage(prompt_tokens=0, completion_tokens=0), None) is None

    def test_status_contains_limits_and_observed(self) -> None:
        cfg = BudgetConfig(max_tokens=100, max_cost_usd=1.0, on_exceeded="abort")
        state = BudgetState(config=cfg, start_time=0)
        status = build_budget_status(state, TokenUsage(prompt_tokens=30, completion_tokens=20), 0.5)
        assert status is not None
        assert status["max_tokens"] == 100
        assert status["max_cost_usd"] == 1.0
        assert status["tokens_used"] == 50
        assert status["estimated_usd"] == 0.5
        assert status["on_exceeded"] == "abort"
        assert status["exceeded"] is False

    def test_status_reflects_violations(self) -> None:
        cfg = BudgetConfig(max_tokens=10, on_exceeded="warn")
        state = BudgetState(config=cfg, start_time=0)
        enforce_budget(state, TokenUsage(prompt_tokens=20, completion_tokens=20), None)
        status = build_budget_status(state, TokenUsage(prompt_tokens=20, completion_tokens=20), None)
        assert status is not None
        assert status["exceeded"] is True
        assert len(status["violations"]) >= 1


# ---------------------------------------------------------------------------
# Executor integration
# ---------------------------------------------------------------------------


def _provider(response: str = "ok", tokens: int = 10) -> MagicMock:
    mock = MagicMock()
    mock.complete = AsyncMock(return_value=response)
    mock.last_token_usage = TokenUsage(prompt_tokens=0, completion_tokens=tokens)
    return mock


def _build_three_node_workflow(budget: BudgetConfig | None) -> Workflow:
    return Workflow(
        version="0.1",
        agents={
            "writer": AgentDefinition(model="openai:gpt-4o-mini", system="Write."),
            "editor": AgentDefinition(model="openai:gpt-4o-mini", system="Edit."),
            "reviewer": AgentDefinition(model="openai:gpt-4o-mini", system="Review."),
        },
        nodes={
            "draft": AgentNode(agent="writer", writes="working.draft"),
            "edit": AgentNode(agent="editor", writes="working.edited"),
            "review": AgentNode(agent="reviewer", writes="output.final"),
        },
        edges=[
            Edge(**{"from": "draft", "to": "edit"}),
            Edge(**{"from": "edit", "to": "review"}),
        ],
        budget=budget,
    )


class TestExecutorBudgetAbort:
    @pytest.mark.asyncio
    async def test_under_budget_completes(self) -> None:
        wf = _build_three_node_workflow(BudgetConfig(max_tokens=99999, on_exceeded="abort"))
        provider = _provider(tokens=10)
        with patch("sirenspec.core.agent_runner.resolve_provider", return_value=provider):
            trace = await execute(wf, "hello")
        assert trace["summary"]["status"] == "success"

    @pytest.mark.asyncio
    async def test_token_ceiling_abort_fails_workflow(self) -> None:
        # 10 tokens per node × 3 nodes = 30 tokens; ceiling at 15 fires on node 2.
        wf = _build_three_node_workflow(BudgetConfig(max_tokens=15, on_exceeded="abort"))
        provider = _provider(tokens=10)
        with patch("sirenspec.core.agent_runner.resolve_provider", return_value=provider):
            trace = await execute(wf, "hello")
        assert trace["summary"]["status"] == "failed"
        # The third node should not have started.
        node_ids = [n["id"] for n in trace["nodes"]]
        assert "review" not in node_ids


class TestExecutorBudgetSkipRemaining:
    @pytest.mark.asyncio
    async def test_skip_remaining_finishes_with_success(self) -> None:
        wf = _build_three_node_workflow(BudgetConfig(max_tokens=15, on_exceeded="skip_remaining"))
        provider = _provider(tokens=10)
        with patch("sirenspec.core.agent_runner.resolve_provider", return_value=provider):
            trace = await execute(wf, "hello")
        # Skip_remaining is an intentional cap, not a failure.
        assert trace["summary"]["status"] == "success"

    @pytest.mark.asyncio
    async def test_skip_remaining_records_skipped_nodes(self) -> None:
        wf = _build_three_node_workflow(BudgetConfig(max_tokens=15, on_exceeded="skip_remaining"))
        provider = _provider(tokens=10)
        with patch("sirenspec.core.agent_runner.resolve_provider", return_value=provider):
            trace = await execute(wf, "hello")

        # All three nodes should appear in the trace.
        node_ids = [n["id"] for n in trace["nodes"]]
        assert node_ids == ["draft", "edit", "review"]

        # First two nodes ran normally; the third was budget-skipped.
        review_trace = next(n for n in trace["nodes"] if n["id"] == "review")
        assert review_trace["status"] == "skipped"
        assert "budget" in review_trace["skipped_reason"].lower()

    @pytest.mark.asyncio
    async def test_skip_remaining_budget_status_in_summary(self) -> None:
        wf = _build_three_node_workflow(BudgetConfig(max_tokens=15, on_exceeded="skip_remaining"))
        provider = _provider(tokens=10)
        with patch("sirenspec.core.agent_runner.resolve_provider", return_value=provider):
            trace = await execute(wf, "hello")

        budget_status = trace["summary"]["budget"]
        assert budget_status is not None
        assert budget_status["exceeded"] is True
        assert budget_status["skipped_remaining"] is True
        assert budget_status["max_tokens"] == 15
        assert budget_status["tokens_used"] == 20  # Two nodes ran before the cap was hit.


class TestExecutorBudgetWarn:
    @pytest.mark.asyncio
    async def test_warn_lets_all_nodes_run(self, caplog: pytest.LogCaptureFixture) -> None:
        wf = _build_three_node_workflow(BudgetConfig(max_tokens=1, on_exceeded="warn"))
        provider = _provider(tokens=10)
        with patch("sirenspec.core.agent_runner.resolve_provider", return_value=provider):
            with caplog.at_level(logging.WARNING, logger="sirenspec.core.budget"):
                trace = await execute(wf, "hello")

        assert trace["summary"]["status"] == "success"
        node_ids = [n["id"] for n in trace["nodes"]]
        assert node_ids == ["draft", "edit", "review"]
        budget_status = trace["summary"]["budget"]
        assert budget_status["exceeded"] is True


class TestExecutorBudgetStatus:
    @pytest.mark.asyncio
    async def test_no_budget_no_status_in_summary(self) -> None:
        wf = _build_three_node_workflow(None)
        provider = _provider(tokens=10)
        with patch("sirenspec.core.agent_runner.resolve_provider", return_value=provider):
            trace = await execute(wf, "hello")
        assert "budget" not in trace["summary"]

    @pytest.mark.asyncio
    async def test_with_budget_status_in_summary(self) -> None:
        wf = _build_three_node_workflow(BudgetConfig(max_tokens=99999))
        provider = _provider(tokens=10)
        with patch("sirenspec.core.agent_runner.resolve_provider", return_value=provider):
            trace = await execute(wf, "hello")
        budget_status = trace["summary"]["budget"]
        assert budget_status["max_tokens"] == 99999
        assert budget_status["tokens_used"] == 30
        assert budget_status["exceeded"] is False


# ---------------------------------------------------------------------------
# Per-node max_tokens_per_call
# ---------------------------------------------------------------------------


class TestMaxTokensPerCall:
    @pytest.mark.asyncio
    async def test_max_tokens_forwarded_to_provider(self) -> None:
        wf = Workflow(
            version="0.1",
            agents={"a": AgentDefinition(model="openai:gpt-4o-mini", system="hi")},
            nodes={"answer": AgentNode(agent="a", writes="output.x", max_tokens_per_call=42)},
        )

        captured: dict[str, int | None] = {"max_tokens": None}

        async def fake_complete(messages: list[dict], max_tokens: int | None = None) -> str:
            captured["max_tokens"] = max_tokens
            return "ok"

        provider = MagicMock()
        provider.complete = fake_complete
        provider.last_token_usage = TokenUsage(prompt_tokens=0, completion_tokens=5)

        with patch("sirenspec.core.agent_runner.resolve_provider", return_value=provider):
            await execute(wf, "hello")

        assert captured["max_tokens"] == 42

    @pytest.mark.asyncio
    async def test_no_max_tokens_omits_kwarg(self) -> None:
        """When ``max_tokens_per_call`` is unset, the provider must be invoked without the kwarg
        so user-defined providers without the parameter remain compatible."""

        wf = Workflow(
            version="0.1",
            agents={"a": AgentDefinition(model="openai:gpt-4o-mini", system="hi")},
            nodes={"answer": AgentNode(agent="a", writes="output.x")},
        )

        call_kwargs: list[dict] = []

        async def fake_complete(messages: list[dict], **kwargs) -> str:
            call_kwargs.append(kwargs)
            return "ok"

        provider = MagicMock()
        provider.complete = fake_complete
        provider.last_token_usage = TokenUsage(prompt_tokens=0, completion_tokens=5)

        with patch("sirenspec.core.agent_runner.resolve_provider", return_value=provider):
            await execute(wf, "hello")

        assert call_kwargs == [{}]

    def test_negative_max_tokens_rejected(self) -> None:
        with pytest.raises(ValueError):
            AgentNode(agent="a", writes="out.x", max_tokens_per_call=-5)

    def test_zero_max_tokens_rejected(self) -> None:
        with pytest.raises(ValueError):
            AgentNode(agent="a", writes="out.x", max_tokens_per_call=0)
