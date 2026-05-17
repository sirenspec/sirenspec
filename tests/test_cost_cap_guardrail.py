"""Tests for the cost_cap guardrail (issue #38)."""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sirenspec.core.pricing import ModelPricing, estimate_usd, lookup_pricing
from sirenspec.core.usage import TokenUsage
from sirenspec.exceptions import BudgetExceededError
from sirenspec.guardrails.base import WorkflowGuardrail
from sirenspec.guardrails.cost_cap import CostCapGuardrail
from sirenspec.guardrails.registry import build_guardrails, make_cost_cap_guardrail

# ---------------------------------------------------------------------------
# Pricing table
# ---------------------------------------------------------------------------


class TestPricingTable:
    def test_known_openai_model_returns_pricing(self) -> None:
        pricing = lookup_pricing("openai/gpt-4o-mini")
        assert pricing is not None
        assert isinstance(pricing, ModelPricing)
        assert pricing.prompt_usd_per_1k > 0
        assert pricing.completion_usd_per_1k > 0

    def test_known_anthropic_model_returns_pricing(self) -> None:
        pricing = lookup_pricing("anthropic/claude-haiku-4-5-20251001")
        assert pricing is not None
        assert isinstance(pricing, ModelPricing)

    def test_unknown_model_returns_none(self) -> None:
        assert lookup_pricing("ollama/llama3") is None
        assert lookup_pricing("unknown/model") is None

    def test_estimate_usd_known_model(self) -> None:
        cost = estimate_usd(1000, 1000, "openai/gpt-4o-mini")
        assert cost is not None
        assert cost > 0

    def test_estimate_usd_zero_tokens(self) -> None:
        cost = estimate_usd(0, 0, "openai/gpt-4o-mini")
        assert cost == 0.0

    def test_estimate_usd_unknown_model_returns_none(self) -> None:
        assert estimate_usd(1000, 500, "ollama/mistral") is None

    def test_estimate_usd_scales_linearly(self) -> None:
        cost_1k = estimate_usd(1000, 0, "openai/gpt-4o-mini")
        cost_2k = estimate_usd(2000, 0, "openai/gpt-4o-mini")
        assert cost_1k is not None and cost_2k is not None
        assert abs(cost_2k - 2 * cost_1k) < 1e-10


# ---------------------------------------------------------------------------
# CostCapGuardrail construction
# ---------------------------------------------------------------------------


class TestCostCapGuardrailInit:
    def test_requires_at_least_one_ceiling(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            CostCapGuardrail()

    def test_max_usd_only_is_valid(self) -> None:
        g = CostCapGuardrail(max_usd=1.0)
        assert g.max_usd == 1.0
        assert g.max_tokens is None

    def test_max_tokens_only_is_valid(self) -> None:
        g = CostCapGuardrail(max_tokens=50000)
        assert g.max_tokens == 50000
        assert g.max_usd is None

    def test_both_ceilings_valid(self) -> None:
        g = CostCapGuardrail(max_usd=0.50, max_tokens=100000)
        assert g.max_usd == 0.50
        assert g.max_tokens == 100000

    def test_invalid_action_raises(self) -> None:
        with pytest.raises(ValueError, match="abort.*warn"):
            CostCapGuardrail(max_usd=1.0, action="invalid")

    def test_default_action_is_abort(self) -> None:
        g = CostCapGuardrail(max_usd=1.0)
        assert g.action == "abort"


# ---------------------------------------------------------------------------
# WorkflowGuardrail protocol
# ---------------------------------------------------------------------------


class TestWorkflowGuardrailProtocol:
    def test_cost_cap_satisfies_protocol(self) -> None:
        g = CostCapGuardrail(max_usd=1.0)
        assert isinstance(g, WorkflowGuardrail)

    def test_check_input_passthrough(self) -> None:
        g = CostCapGuardrail(max_usd=1.0)
        assert g.check_input("hello") == "hello"

    def test_check_output_passthrough(self) -> None:
        g = CostCapGuardrail(max_usd=1.0)
        assert g.check_output("world") == "world"


# ---------------------------------------------------------------------------
# CostCapGuardrail.detect_violation
# ---------------------------------------------------------------------------


class TestDetectViolation:
    def test_no_violation_within_token_budget(self) -> None:
        g = CostCapGuardrail(max_tokens=1000)
        usage = TokenUsage(prompt_tokens=400, completion_tokens=200)
        assert g.detect_violation(usage, None) is None

    def test_token_ceiling_exceeded(self) -> None:
        g = CostCapGuardrail(max_tokens=500)
        usage = TokenUsage(prompt_tokens=400, completion_tokens=200)
        result = g.detect_violation(usage, None)
        assert result is not None
        assert "600" in result
        assert "500" in result

    def test_usd_ceiling_not_checked_when_estimate_is_none(self) -> None:
        g = CostCapGuardrail(max_usd=0.01)
        usage = TokenUsage(prompt_tokens=10000, completion_tokens=10000)
        assert g.detect_violation(usage, None) is None

    def test_usd_ceiling_exceeded(self) -> None:
        g = CostCapGuardrail(max_usd=0.001)
        usage = TokenUsage(prompt_tokens=1000, completion_tokens=1000)
        result = g.detect_violation(usage, 0.005)
        assert result is not None
        assert "USD" in result or "$" in result

    def test_both_ceilings_token_checked_first(self) -> None:
        g = CostCapGuardrail(max_usd=0.001, max_tokens=100)
        usage = TokenUsage(prompt_tokens=60, completion_tokens=60)
        result = g.detect_violation(usage, 0.005)
        assert result is not None
        assert "Token" in result or "token" in result


# ---------------------------------------------------------------------------
# CostCapGuardrail.check_budget — abort mode
# ---------------------------------------------------------------------------


class TestCheckBudgetAbort:
    def test_under_budget_does_not_raise(self) -> None:
        g = CostCapGuardrail(max_tokens=1000, action="abort")
        usage = TokenUsage(prompt_tokens=200, completion_tokens=100)
        g.check_budget(usage, None)  # should not raise

    def test_token_ceiling_hit_raises_budget_exceeded(self) -> None:
        g = CostCapGuardrail(max_tokens=100, action="abort")
        usage = TokenUsage(prompt_tokens=80, completion_tokens=30)
        with pytest.raises(BudgetExceededError) as exc_info:
            g.check_budget(usage, None)
        assert exc_info.value.tokens_used == 110

    def test_usd_ceiling_hit_raises_budget_exceeded(self) -> None:
        g = CostCapGuardrail(max_usd=0.001, action="abort")
        usage = TokenUsage(prompt_tokens=100, completion_tokens=50)
        with pytest.raises(BudgetExceededError) as exc_info:
            g.check_budget(usage, 0.005)
        assert exc_info.value.estimated_usd == 0.005

    def test_budget_exceeded_is_guardrail_error(self) -> None:
        from sirenspec.exceptions import GuardrailError

        g = CostCapGuardrail(max_tokens=1, action="abort")
        usage = TokenUsage(prompt_tokens=5, completion_tokens=5)
        with pytest.raises(GuardrailError):
            g.check_budget(usage, None)


# ---------------------------------------------------------------------------
# CostCapGuardrail.check_budget — warn mode
# ---------------------------------------------------------------------------


class TestCheckBudgetWarn:
    def test_warn_mode_does_not_raise(self) -> None:
        g = CostCapGuardrail(max_tokens=10, action="warn")
        usage = TokenUsage(prompt_tokens=50, completion_tokens=50)
        g.check_budget(usage, None)  # must not raise

    def test_warn_mode_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        g = CostCapGuardrail(max_tokens=10, action="warn")
        usage = TokenUsage(prompt_tokens=50, completion_tokens=50)
        with caplog.at_level(logging.WARNING, logger="sirenspec.guardrails.cost_cap"):
            g.check_budget(usage, None)
        assert len(caplog.records) == 1
        assert "budget warning" in caplog.records[0].message.lower() or "warning" in caplog.records[0].message.lower()

    def test_warn_mode_no_log_when_under_budget(self, caplog: pytest.LogCaptureFixture) -> None:
        g = CostCapGuardrail(max_tokens=1000, action="warn")
        usage = TokenUsage(prompt_tokens=10, completion_tokens=5)
        with caplog.at_level(logging.WARNING, logger="sirenspec.guardrails.cost_cap"):
            g.check_budget(usage, None)
        assert len(caplog.records) == 0


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------


class TestCostCapRegistry:
    def test_make_cost_cap_guardrail_none_config_raises(self) -> None:
        with pytest.raises(ValueError):
            make_cost_cap_guardrail(None)

    def test_make_cost_cap_guardrail_empty_config_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            make_cost_cap_guardrail({})

    def test_make_cost_cap_guardrail_valid_config(self) -> None:
        g = make_cost_cap_guardrail({"max_usd": 0.5, "action": "warn"})
        assert isinstance(g, CostCapGuardrail)
        assert g.max_usd == 0.5
        assert g.action == "warn"

    def test_build_guardrails_resolves_cost_cap(self) -> None:
        from sirenspec.core.models import GuardrailSpec

        spec = GuardrailSpec(name="cost_cap", config={"max_tokens": 50000})
        guardrails = build_guardrails([spec])
        assert len(guardrails) == 1
        assert isinstance(guardrails[0], CostCapGuardrail)

    def test_cost_cap_in_supported_list(self) -> None:
        try:
            build_guardrails([])  # no-op, just checking registry doesn't blow up
        except Exception:
            pass
        # Verify cost_cap is a valid name by building one
        from sirenspec.core.models import GuardrailSpec

        g = build_guardrails([GuardrailSpec(name="cost_cap", config={"max_tokens": 1000})])
        assert isinstance(g[0], CostCapGuardrail)


# ---------------------------------------------------------------------------
# Executor integration: under-budget run passes
# ---------------------------------------------------------------------------


class TestExecutorCostCapIntegration:
    @pytest.mark.asyncio
    async def test_under_budget_run_completes(self, minimal_workflow) -> None:
        from sirenspec.core.executor import execute
        from sirenspec.core.models import GuardrailSpec

        minimal_workflow.guardrails = [GuardrailSpec(name="cost_cap", config={"max_tokens": 999999})]

        mock_provider = MagicMock()
        mock_provider.complete = AsyncMock(return_value="response")
        mock_provider.last_token_usage = TokenUsage(prompt_tokens=10, completion_tokens=5)

        with patch("sirenspec.core.agent_runner.resolve_provider", return_value=mock_provider):
            trace = await execute(minimal_workflow, "hello")

        assert trace["summary"]["status"] == "success"

    @pytest.mark.asyncio
    async def test_token_ceiling_hit_aborts(self, minimal_workflow) -> None:
        from sirenspec.core.executor import execute
        from sirenspec.core.models import GuardrailSpec

        minimal_workflow.guardrails = [GuardrailSpec(name="cost_cap", config={"max_tokens": 1})]

        mock_provider = MagicMock()
        mock_provider.complete = AsyncMock(return_value="response")
        mock_provider.last_token_usage = TokenUsage(prompt_tokens=10, completion_tokens=5)

        with patch("sirenspec.core.agent_runner.resolve_provider", return_value=mock_provider):
            trace = await execute(minimal_workflow, "hello")

        assert trace["summary"]["status"] == "failed"
        node = trace["nodes"][0]
        assert node["error"] is not None
        assert "Token" in node["error"] or "token" in node["error"] or "BudgetExceeded" in node["error"]

    @pytest.mark.asyncio
    async def test_usd_ceiling_hit_aborts(self, minimal_workflow) -> None:
        from sirenspec.core.executor import execute
        from sirenspec.core.models import GuardrailSpec

        minimal_workflow.guardrails = [GuardrailSpec(name="cost_cap", config={"max_usd": 0.000001})]

        mock_provider = MagicMock()
        mock_provider.complete = AsyncMock(return_value="response")
        mock_provider.last_token_usage = TokenUsage(prompt_tokens=1000, completion_tokens=500)

        with patch("sirenspec.core.agent_runner.resolve_provider", return_value=mock_provider):
            with patch("sirenspec.core.executor.estimate_usd", return_value=0.01):
                trace = await execute(minimal_workflow, "hello")

        assert trace["summary"]["status"] == "failed"

    @pytest.mark.asyncio
    async def test_warn_mode_continues_execution(self, minimal_workflow) -> None:
        from sirenspec.core.executor import execute
        from sirenspec.core.models import GuardrailSpec

        minimal_workflow.guardrails = [GuardrailSpec(name="cost_cap", config={"max_tokens": 1, "action": "warn"})]

        mock_provider = MagicMock()
        mock_provider.complete = AsyncMock(return_value="response")
        mock_provider.last_token_usage = TokenUsage(prompt_tokens=100, completion_tokens=50)

        with patch("sirenspec.core.agent_runner.resolve_provider", return_value=mock_provider):
            trace = await execute(minimal_workflow, "hello")

        assert trace["summary"]["status"] == "success"

    @pytest.mark.asyncio
    async def test_budget_visible_in_trace(self, minimal_workflow) -> None:
        from sirenspec.core.executor import execute
        from sirenspec.core.models import GuardrailSpec

        minimal_workflow.guardrails = [GuardrailSpec(name="cost_cap", config={"max_tokens": 999999})]

        mock_provider = MagicMock()
        mock_provider.complete = AsyncMock(return_value="response")
        mock_provider.last_token_usage = TokenUsage(prompt_tokens=20, completion_tokens=10)

        with patch("sirenspec.core.agent_runner.resolve_provider", return_value=mock_provider):
            trace = await execute(minimal_workflow, "hello")

        assert trace["summary"]["total_usage"]["total_tokens"] == 30


# ---------------------------------------------------------------------------
# validate CLI: catches missing cost_cap config
# ---------------------------------------------------------------------------


_COST_CAP_NO_CEILING_YAML = """\
version: "1.0"
agents:
  a:
    model: openai/gpt-4o-mini
    system: You are helpful.
nodes:
  n:
    agent: a
    writes: output.result
guardrails:
  - name: cost_cap
    config: {}
"""

_COST_CAP_VALID_YAML = """\
version: "1.0"
agents:
  a:
    model: openai/gpt-4o-mini
    system: You are helpful.
nodes:
  n:
    agent: a
    writes: output.result
guardrails:
  - name: cost_cap
    config:
      max_tokens: 100000
      action: abort
"""


class TestValidateCostCapConfig:
    def test_validate_rejects_cost_cap_without_ceiling(self, tmp_path) -> None:
        from typer.testing import CliRunner

        from sirenspec.cli import app

        wf_file = tmp_path / "wf.yaml"
        wf_file.write_text(_COST_CAP_NO_CEILING_YAML)

        runner = CliRunner()
        result = runner.invoke(app, ["validate", str(wf_file)])
        assert result.exit_code != 0

    def test_validate_accepts_valid_cost_cap(self, tmp_path) -> None:
        from typer.testing import CliRunner

        from sirenspec.cli import app

        wf_file = tmp_path / "wf.yaml"
        wf_file.write_text(_COST_CAP_VALID_YAML)

        runner = CliRunner()
        result = runner.invoke(app, ["validate", str(wf_file)])
        assert result.exit_code == 0
