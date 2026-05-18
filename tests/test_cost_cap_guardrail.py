"""Tests for the cost_cap guardrail (issue #38)."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import sirenspec.core.pricing as pricing_module
from sirenspec.core.pricing import (
    ModelPricing,
    estimate_usd,
    fetch_remote,
    load_cache,
    load_pricing_data,
    load_snapshot,
    lookup_pricing,
    parse_litellm_data,
    parse_litellm_entry,
    reset_pricing_cache,
    save_cache,
)
from sirenspec.core.usage import TokenUsage
from sirenspec.exceptions import BudgetExceededError
from sirenspec.guardrails.base import WorkflowGuardrail
from sirenspec.guardrails.cost_cap import CostCapGuardrail
from sirenspec.guardrails.registry import build_guardrails, make_cost_cap_guardrail


@pytest.fixture(autouse=True)
def clear_pricing_cache():
    """Reset the in-process pricing cache before each test."""
    reset_pricing_cache()
    yield
    reset_pricing_cache()


# ---------------------------------------------------------------------------
# Pricing table — LiteLLM dynamic fetch
# ---------------------------------------------------------------------------


class TestPricingTable:
    @pytest.fixture(autouse=True)
    def force_snapshot(self):
        """Force snapshot usage so pricing tests are deterministic regardless of network."""
        with patch("sirenspec.core.pricing.load_pricing_data", new=load_snapshot):
            yield

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

    def test_provider_prefix_stripped(self) -> None:
        with_prefix = lookup_pricing("openai/gpt-4o")
        without_prefix = lookup_pricing("gpt-4o")
        assert with_prefix == without_prefix

    def test_model_uri_without_slash_resolves(self) -> None:
        pricing = lookup_pricing("gpt-4o-mini")
        assert pricing is not None


# ---------------------------------------------------------------------------
# parse_litellm_entry
# ---------------------------------------------------------------------------


class TestParseLitellmEntry:
    def test_valid_entry_returns_model_pricing(self) -> None:
        entry = {"input_cost_per_token": 0.000001, "output_cost_per_token": 0.000003}
        result = parse_litellm_entry(entry)
        assert result is not None
        assert abs(result.prompt_usd_per_1k - 0.001) < 1e-10
        assert abs(result.completion_usd_per_1k - 0.003) < 1e-10

    def test_missing_input_cost_returns_none(self) -> None:
        assert parse_litellm_entry({"output_cost_per_token": 0.001}) is None

    def test_missing_output_cost_returns_none(self) -> None:
        assert parse_litellm_entry({"input_cost_per_token": 0.001}) is None

    def test_empty_entry_returns_none(self) -> None:
        assert parse_litellm_entry({}) is None

    def test_costs_scaled_from_per_token_to_per_1k(self) -> None:
        entry = {"input_cost_per_token": 0.000005, "output_cost_per_token": 0.00002}
        result = parse_litellm_entry(entry)
        assert result is not None
        assert abs(result.prompt_usd_per_1k - 0.005) < 1e-10
        assert abs(result.completion_usd_per_1k - 0.020) < 1e-10


# ---------------------------------------------------------------------------
# parse_litellm_data
# ---------------------------------------------------------------------------


class TestParseLitellmData:
    def test_parses_valid_entries(self) -> None:
        data = {
            "gpt-4o": {"input_cost_per_token": 0.0000025, "output_cost_per_token": 0.00001},
            "gpt-4o-mini": {"input_cost_per_token": 0.00000015, "output_cost_per_token": 0.0000006},
        }
        result = parse_litellm_data(data)
        assert "gpt-4o" in result
        assert "gpt-4o-mini" in result
        assert isinstance(result["gpt-4o"], ModelPricing)

    def test_skips_non_dict_entries(self) -> None:
        data = {
            "_source": "https://example.com",
            "_snapshot_date": "2026-01-01",
            "gpt-4o": {"input_cost_per_token": 0.0000025, "output_cost_per_token": 0.00001},
        }
        result = parse_litellm_data(data)
        assert "_source" not in result
        assert "_snapshot_date" not in result
        assert "gpt-4o" in result

    def test_skips_entries_missing_cost_fields(self) -> None:
        data = {
            "model-no-cost": {"max_tokens": 4096, "litellm_provider": "openai"},
            "gpt-4o": {"input_cost_per_token": 0.0000025, "output_cost_per_token": 0.00001},
        }
        result = parse_litellm_data(data)
        assert "model-no-cost" not in result
        assert "gpt-4o" in result

    def test_empty_dict_returns_empty(self) -> None:
        assert parse_litellm_data({}) == {}


# ---------------------------------------------------------------------------
# load_snapshot
# ---------------------------------------------------------------------------


class TestLoadSnapshot:
    def test_snapshot_loads_successfully(self) -> None:
        data = load_snapshot()
        assert isinstance(data, dict)
        assert len(data) > 0

    def test_snapshot_contains_openai_models(self) -> None:
        data = load_snapshot()
        assert "gpt-4o" in data or "gpt-4o-mini" in data

    def test_snapshot_contains_anthropic_models(self) -> None:
        data = load_snapshot()
        assert any("claude" in k for k in data)

    def test_snapshot_entries_parseable(self) -> None:
        data = load_snapshot()
        parsed = parse_litellm_data(data)
        assert len(parsed) > 0


# ---------------------------------------------------------------------------
# load_cache / save_cache
# ---------------------------------------------------------------------------


class TestFilesystemCache:
    def test_load_cache_returns_none_when_absent(self, tmp_path: Path) -> None:
        with patch.object(pricing_module, "_CACHE_PATH", tmp_path / "pricing.json"):
            assert load_cache() is None

    def test_save_and_load_cache_roundtrip(self, tmp_path: Path) -> None:
        cache_path = tmp_path / "pricing.json"
        with patch.object(pricing_module, "_CACHE_PATH", cache_path):
            payload = {"gpt-4o": {"input_cost_per_token": 0.0000025, "output_cost_per_token": 0.00001}}
            save_cache(payload)
            result = load_cache()
        assert result == payload

    def test_load_cache_returns_none_when_expired(self, tmp_path: Path) -> None:
        cache_path = tmp_path / "pricing.json"
        stale_timestamp = time.time() - (pricing_module._CACHE_TTL_SECONDS + 1)
        cache_path.write_text(json.dumps({"timestamp": stale_timestamp, "data": {"gpt-4o": {}}}))
        with patch.object(pricing_module, "_CACHE_PATH", cache_path):
            assert load_cache() is None

    def test_load_cache_returns_data_when_fresh(self, tmp_path: Path) -> None:
        cache_path = tmp_path / "pricing.json"
        payload = {"gpt-4o": {"input_cost_per_token": 0.0000025, "output_cost_per_token": 0.00001}}
        fresh_timestamp = time.time() - 60  # 1 minute ago
        cache_path.write_text(json.dumps({"timestamp": fresh_timestamp, "data": payload}))
        with patch.object(pricing_module, "_CACHE_PATH", cache_path):
            result = load_cache()
        assert result == payload

    def test_load_cache_returns_none_on_corrupt_file(self, tmp_path: Path) -> None:
        cache_path = tmp_path / "pricing.json"
        cache_path.write_text("not valid json{{{")
        with patch.object(pricing_module, "_CACHE_PATH", cache_path):
            assert load_cache() is None

    def test_save_cache_silently_ignores_write_error(self, tmp_path: Path) -> None:
        read_only = tmp_path / "ro"
        read_only.mkdir(mode=0o444)
        with patch.object(pricing_module, "_CACHE_PATH", read_only / "pricing.json"):
            save_cache({"x": 1})  # must not raise


# ---------------------------------------------------------------------------
# fetch_remote
# ---------------------------------------------------------------------------


class TestFetchRemote:
    def test_fetch_remote_returns_none_on_network_error(self) -> None:
        with patch("urllib.request.urlopen", side_effect=OSError("no network")):
            result = fetch_remote()
        assert result is None

    def test_fetch_remote_returns_data_on_success(self, tmp_path: Path) -> None:
        payload = {"gpt-4o": {"input_cost_per_token": 0.0000025, "output_cost_per_token": 0.00001}}
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = json.dumps(payload).encode()
        with (
            patch("urllib.request.urlopen", return_value=mock_resp),
            patch.object(pricing_module, "_CACHE_PATH", tmp_path / "pricing.json"),
        ):
            result = fetch_remote()
        assert result == payload

    def test_fetch_remote_saves_to_cache_on_success(self, tmp_path: Path) -> None:
        payload = {"gpt-4o": {"input_cost_per_token": 0.0000025, "output_cost_per_token": 0.00001}}
        mock_resp = MagicMock()
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        mock_resp.read.return_value = json.dumps(payload).encode()
        cache_path = tmp_path / "pricing.json"
        with (
            patch("urllib.request.urlopen", return_value=mock_resp),
            patch.object(pricing_module, "_CACHE_PATH", cache_path),
        ):
            fetch_remote()
        assert cache_path.exists()


# ---------------------------------------------------------------------------
# load_pricing_data — priority chain
# ---------------------------------------------------------------------------


class TestLoadPricingData:
    def test_uses_cache_when_fresh(self, tmp_path: Path) -> None:
        payload = {"gpt-4o": {"input_cost_per_token": 0.0000025, "output_cost_per_token": 0.00001}}
        cache_path = tmp_path / "pricing.json"
        cache_path.write_text(json.dumps({"timestamp": time.time(), "data": payload}))
        with (
            patch.object(pricing_module, "_CACHE_PATH", cache_path),
            patch("sirenspec.core.pricing.fetch_remote") as mock_fetch,
        ):
            result = load_pricing_data()
            mock_fetch.assert_not_called()
        assert result == payload

    def test_fetches_remote_when_cache_stale(self, tmp_path: Path) -> None:
        remote_payload = {"gpt-4o-mini": {"input_cost_per_token": 0.00000015, "output_cost_per_token": 0.0000006}}
        stale_cache = tmp_path / "pricing.json"
        stale_cache.write_text(json.dumps({"timestamp": 0, "data": {"old": {}}}))
        with (
            patch.object(pricing_module, "_CACHE_PATH", stale_cache),
            patch("sirenspec.core.pricing.fetch_remote", return_value=remote_payload),
        ):
            result = load_pricing_data()
        assert result == remote_payload

    def test_falls_back_to_snapshot_when_remote_fails(self, tmp_path: Path) -> None:
        with (
            patch.object(pricing_module, "_CACHE_PATH", tmp_path / "pricing.json"),
            patch("sirenspec.core.pricing.fetch_remote", return_value=None),
        ):
            result = load_pricing_data()
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_snapshot_fallback_logs_warning(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        with (
            patch.object(pricing_module, "_CACHE_PATH", tmp_path / "pricing.json"),
            patch("sirenspec.core.pricing.fetch_remote", return_value=None),
            caplog.at_level(logging.WARNING, logger="sirenspec.core.pricing"),
        ):
            load_pricing_data()
        assert any("bundled" in r.message or "snapshot" in r.message for r in caplog.records)


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
