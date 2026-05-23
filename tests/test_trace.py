"""Unit tests for execution trace structure and JSON-serializability."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sirenspec.core.executor import execute
from sirenspec.core.models import Workflow
from sirenspec.core.usage import TokenUsage


def _make_mock_provider(response: str = "resp", tokens: int = 5) -> MagicMock:
    m = MagicMock()
    m.complete = AsyncMock(return_value=response)
    m.last_token_usage = TokenUsage(prompt_tokens=0, completion_tokens=tokens)
    return m


class TestTraceStructure:
    @pytest.mark.asyncio
    async def test_top_level_keys(self, minimal_workflow: Workflow) -> None:
        with patch("sirenspec.core.agent_runner.resolve_provider", return_value=_make_mock_provider()):
            trace = await execute(minimal_workflow, "hi")

        assert set(trace.keys()) >= {"workflow", "input", "nodes", "output", "summary"}

    @pytest.mark.asyncio
    async def test_summary_keys(self, minimal_workflow: Workflow) -> None:
        with patch("sirenspec.core.agent_runner.resolve_provider", return_value=_make_mock_provider(tokens=7)):
            trace = await execute(minimal_workflow, "hi")

        summary = trace["summary"]
        assert "total_tokens" in summary
        assert "total_duration_ms" in summary
        assert "status" in summary
        assert summary["total_tokens"] == 7

    @pytest.mark.asyncio
    async def test_node_entry_keys(self, minimal_workflow: Workflow) -> None:
        with patch("sirenspec.core.agent_runner.resolve_provider", return_value=_make_mock_provider()):
            trace = await execute(minimal_workflow, "hi")

        node = trace["nodes"][0]
        expected = {
            "id",
            "agent",
            "prompt_sent",
            "response_received",
            "writes",
            "guardrails_passed",
            "tokens",
            "duration_ms",
            "error",
        }
        assert set(node.keys()) >= expected

    @pytest.mark.asyncio
    async def test_trace_is_json_serializable(self, minimal_workflow: Workflow) -> None:
        with patch("sirenspec.core.agent_runner.resolve_provider", return_value=_make_mock_provider()):
            trace = await execute(minimal_workflow, "hi")

        # Should not raise
        serialized = json.dumps(trace)
        assert isinstance(serialized, str)

    @pytest.mark.asyncio
    async def test_error_recorded_on_failure(self, minimal_workflow: Workflow) -> None:
        mock_provider = MagicMock()
        mock_provider.complete = AsyncMock(side_effect=RuntimeError("API down"))

        with patch("sirenspec.core.agent_runner.resolve_provider", return_value=mock_provider):
            trace = await execute(minimal_workflow, "hi")

        assert trace["summary"]["status"] == "failed"
        assert trace["nodes"][0]["error"] is not None
        assert "API down" in trace["nodes"][0]["error"]

    @pytest.mark.asyncio
    async def test_duration_ms_is_numeric(self, minimal_workflow: Workflow) -> None:
        with patch("sirenspec.core.agent_runner.resolve_provider", return_value=_make_mock_provider()):
            trace = await execute(minimal_workflow, "hi")

        assert isinstance(trace["nodes"][0]["duration_ms"], (int, float))
        assert trace["nodes"][0]["duration_ms"] >= 0

    @pytest.mark.asyncio
    async def test_summary_status_success(self, minimal_workflow: Workflow) -> None:
        with patch("sirenspec.core.agent_runner.resolve_provider", return_value=_make_mock_provider()):
            trace = await execute(minimal_workflow, "hi")

        assert trace["summary"]["status"] == "success"
