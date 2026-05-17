"""Unit tests for the retry engine and on_failure handling."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sirenspec.core.executor import execute
from sirenspec.core.models import (
    AgentDefinition,
    Edge,
    Node,
    OnFailurePolicy,
    RetryPolicy,
    Workflow,
    WorkflowDefaults,
)
from sirenspec.core.usage import TokenUsage
from sirenspec.core.retry import compute_delay, run_with_retry
from sirenspec.exceptions import ProviderError, RetryExhaustedError

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_provider_mock(response_text: str = "mock response", tokens: int = 10) -> MagicMock:
    mock = MagicMock()
    mock.complete = AsyncMock(return_value=response_text)
    mock.last_token_usage = TokenUsage(prompt_tokens=0, completion_tokens=tokens)
    return mock


def _minimal_workflow(
    retry: RetryPolicy | None = None,
    on_failure: OnFailurePolicy | None = None,
    defaults: WorkflowDefaults | None = None,
) -> Workflow:
    return Workflow(
        version="0.1",
        agents={"a": AgentDefinition(model="openai:gpt-4o-mini", system="sys")},
        nodes={"n1": Node(agent="a", writes="output.reply", retry=retry, on_failure=on_failure)},
        defaults=defaults,
    )


# ---------------------------------------------------------------------------
# compute_delay
# ---------------------------------------------------------------------------


class TestComputeDelay:
    def test_constant_returns_base_delay(self) -> None:
        policy = RetryPolicy(backoff="constant", base_delay=2.0, max_delay=30.0)
        assert compute_delay(policy, 0) == pytest.approx(2.0)
        assert compute_delay(policy, 5) == pytest.approx(2.0)

    def test_linear_grows_linearly(self) -> None:
        policy = RetryPolicy(backoff="linear", base_delay=1.0, max_delay=100.0)
        assert compute_delay(policy, 0) == pytest.approx(1.0)
        assert compute_delay(policy, 1) == pytest.approx(2.0)
        assert compute_delay(policy, 2) == pytest.approx(3.0)
        assert compute_delay(policy, 4) == pytest.approx(5.0)

    def test_exponential_doubles_each_retry(self) -> None:
        policy = RetryPolicy(backoff="exponential", base_delay=1.0, max_delay=100.0)
        assert compute_delay(policy, 0) == pytest.approx(1.0)
        assert compute_delay(policy, 1) == pytest.approx(2.0)
        assert compute_delay(policy, 2) == pytest.approx(4.0)
        assert compute_delay(policy, 3) == pytest.approx(8.0)

    def test_delay_clamped_to_max_delay(self) -> None:
        policy = RetryPolicy(backoff="exponential", base_delay=1.0, max_delay=5.0)
        assert compute_delay(policy, 10) == pytest.approx(5.0)

    def test_jitter_within_bounds(self) -> None:
        policy = RetryPolicy(backoff="constant", base_delay=10.0, max_delay=100.0, jitter=True)
        for _ in range(50):
            delay = compute_delay(policy, 0)
            # ±20% of 10.0 → [8.0, 12.0]
            assert 8.0 <= delay <= 12.0, f"jitter out of bounds: {delay}"

    def test_no_jitter_is_exact(self) -> None:
        policy = RetryPolicy(backoff="constant", base_delay=5.0, max_delay=100.0, jitter=False)
        assert compute_delay(policy, 0) == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# run_with_retry
# ---------------------------------------------------------------------------


class TestRunWithRetry:
    @pytest.mark.asyncio
    async def test_success_on_first_attempt(self) -> None:
        async def _call() -> str:
            return "ok"

        policy = RetryPolicy(max_attempts=3, on=["429"])
        result = await run_with_retry("node1", policy, _call)
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_retries_on_matching_status_code(self) -> None:
        """Should succeed on the third attempt after two 429 failures."""
        attempt_count = 0

        async def _call() -> str:
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                exc = ProviderError("rate limited", status_code=429)
                raise exc
            return "success"

        policy = RetryPolicy(max_attempts=3, backoff="constant", base_delay=0.0, on=["429"])
        logged: list[dict[str, Any]] = []

        def _log(attempt: int, delay: float, error: str) -> None:
            logged.append({"attempt": attempt, "delay": delay, "error": error})

        result = await run_with_retry("node1", policy, _call, on_attempt=_log)
        assert result == "success"
        assert attempt_count == 3
        assert len(logged) == 2  # retries 2 and 3 were logged

    @pytest.mark.asyncio
    async def test_exhausted_raises_retry_exhausted_error(self) -> None:
        async def _call() -> str:
            raise ProviderError("server error", status_code=500)

        policy = RetryPolicy(max_attempts=2, backoff="constant", base_delay=0.0, on=["500"])
        with pytest.raises(RetryExhaustedError) as exc_info:
            await run_with_retry("node1", policy, _call)

        assert exc_info.value.node_id == "node1"
        assert exc_info.value.attempts == 2

    @pytest.mark.asyncio
    async def test_non_matching_error_not_retried(self) -> None:
        """A 404 error should not be retried when only 429 is in the trigger list."""
        call_count = 0

        async def _call() -> str:
            nonlocal call_count
            call_count += 1
            raise ProviderError("not found", status_code=404)

        policy = RetryPolicy(max_attempts=3, backoff="constant", base_delay=0.0, on=["429"])
        with pytest.raises(RetryExhaustedError):
            await run_with_retry("node1", policy, _call)

        # Should fail immediately without retrying.
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_network_error_retried(self) -> None:
        """A plain exception (no status_code) should be retried when 'network_error' is in on."""
        attempt_count = 0

        async def _call() -> str:
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 2:
                raise ConnectionError("connection refused")
            return "recovered"

        policy = RetryPolicy(max_attempts=3, backoff="constant", base_delay=0.0, on=["network_error"])
        result = await run_with_retry("node1", policy, _call)
        assert result == "recovered"
        assert attempt_count == 2

    @pytest.mark.asyncio
    async def test_on_attempt_callback_receives_correct_data(self) -> None:
        attempt_count = 0

        async def _call() -> str:
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count == 1:
                raise ProviderError("oops", status_code=429)
            return "done"

        policy = RetryPolicy(max_attempts=2, backoff="constant", base_delay=0.5, on=["429"])
        logged: list[dict[str, Any]] = []

        def _log(attempt: int, delay: float, error: str) -> None:
            logged.append({"attempt": attempt, "delay": delay, "error": error})

        await run_with_retry("node1", policy, _call, on_attempt=_log)
        assert len(logged) == 1
        assert logged[0]["attempt"] == 2
        assert logged[0]["delay"] == pytest.approx(0.5)
        assert "oops" in logged[0]["error"]


# ---------------------------------------------------------------------------
# Executor integration: retry + on_failure
# ---------------------------------------------------------------------------


class TestExecutorRetryIntegration:
    @pytest.mark.asyncio
    async def test_successful_retry_on_429(self) -> None:
        """Executor should succeed when provider fails once with 429 then succeeds."""
        call_count = 0

        async def _complete(messages: list[dict]) -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ProviderError("rate limited", status_code=429)
            return "recovered response"

        mock_provider = MagicMock()
        mock_provider.complete = _complete
        mock_provider.last_token_usage = TokenUsage(prompt_tokens=0, completion_tokens=5)

        wf = _minimal_workflow(retry=RetryPolicy(max_attempts=2, backoff="constant", base_delay=0.0, on=["429"]))

        with patch("sirenspec.core.agent_runner.resolve_provider", return_value=mock_provider):
            trace = await execute(wf, "hello")

        assert trace["summary"]["status"] == "success"
        assert trace["nodes"][0]["response_received"] == "recovered response"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_retry_attempts_logged_in_trace(self) -> None:
        """Retry attempt metadata should appear in the node trace."""
        call_count = 0

        async def _complete(messages: list[dict]) -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ProviderError("server error", status_code=500)
            return "ok"

        mock_provider = MagicMock()
        mock_provider.complete = _complete
        mock_provider.last_token_usage = TokenUsage(prompt_tokens=0, completion_tokens=5)

        wf = _minimal_workflow(
            retry=RetryPolicy(max_attempts=3, backoff="constant", base_delay=0.0, on=["500"]),
            on_failure=OnFailurePolicy(action="abort"),
        )

        with patch("sirenspec.core.agent_runner.resolve_provider", return_value=mock_provider):
            trace = await execute(wf, "hello")

        node = trace["nodes"][0]
        assert node["retry_attempts"]  # should have retry entries
        assert all("attempt" in r for r in node["retry_attempts"])
        assert all("delay_seconds" in r for r in node["retry_attempts"])

    @pytest.mark.asyncio
    async def test_on_failure_abort_stops_execution(self) -> None:
        """When retries are exhausted and action='abort', status should be 'failed'."""
        mock_provider = MagicMock()
        mock_provider.complete = AsyncMock(side_effect=ProviderError("boom", status_code=500))
        mock_provider.last_token_usage = TokenUsage(prompt_tokens=0, completion_tokens=0)

        wf = _minimal_workflow(
            retry=RetryPolicy(max_attempts=2, backoff="constant", base_delay=0.0, on=["500"]),
            on_failure=OnFailurePolicy(action="abort"),
        )

        with patch("sirenspec.core.agent_runner.resolve_provider", return_value=mock_provider):
            trace = await execute(wf, "hello")

        assert trace["summary"]["status"] == "failed"
        assert trace["nodes"][0]["error"] is not None

    @pytest.mark.asyncio
    async def test_on_failure_skip_continues_execution(self) -> None:
        """When action='skip', the failing node is skipped and execution continues."""
        wf = Workflow(
            version="0.1",
            agents={
                "a": AgentDefinition(model="openai:gpt-4o-mini", system="sys a"),
                "b": AgentDefinition(model="openai:gpt-4o-mini", system="sys b"),
            },
            nodes={
                "n1": Node(
                    agent="a",
                    writes="working.x",
                    retry=RetryPolicy(max_attempts=1, on=["500"]),
                    on_failure=OnFailurePolicy(action="skip"),
                ),
                "n2": Node(agent="b", writes="output.reply"),
            },
            edges=[Edge(**{"from": "n1", "to": "n2"})],
        )

        call_count = 0

        async def _complete(messages: list[dict]) -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ProviderError("error", status_code=500)
            return "n2 result"

        mock_provider = MagicMock()
        mock_provider.complete = _complete
        mock_provider.last_token_usage = TokenUsage(prompt_tokens=0, completion_tokens=5)

        with patch("sirenspec.core.agent_runner.resolve_provider", return_value=mock_provider):
            trace = await execute(wf, "hello")

        node_ids = [n["id"] for n in trace["nodes"]]
        assert "n1" in node_ids
        assert "n2" in node_ids
        assert trace["summary"]["status"] == "success"

    @pytest.mark.asyncio
    async def test_on_failure_use_default_injects_value(self) -> None:
        """When action='use_default', default_output is written to the context."""
        mock_provider = MagicMock()
        mock_provider.complete = AsyncMock(side_effect=ProviderError("error", status_code=500))
        mock_provider.last_token_usage = TokenUsage(prompt_tokens=0, completion_tokens=0)

        wf = _minimal_workflow(
            retry=RetryPolicy(max_attempts=1, on=["500"]),
            on_failure=OnFailurePolicy(action="use_default", default_output="unknown"),
        )

        with patch("sirenspec.core.agent_runner.resolve_provider", return_value=mock_provider):
            trace = await execute(wf, "hello")

        assert trace["output"]["reply"] == "unknown"
        assert trace["nodes"][0]["on_failure_action"] == "use_default"

    @pytest.mark.asyncio
    async def test_on_failure_fallback_routes_to_named_node(self) -> None:
        """When action='fallback', execution continues at fallback_node."""
        wf = Workflow(
            version="0.1",
            agents={
                "primary_agent": AgentDefinition(model="openai:gpt-4o-mini", system="primary"),
                "fallback_agent": AgentDefinition(model="openai:gpt-4o-mini", system="fallback"),
            },
            nodes={
                "primary": Node(
                    agent="primary_agent",
                    writes="output.reply",
                    retry=RetryPolicy(max_attempts=1, on=["500"]),
                    on_failure=OnFailurePolicy(action="fallback", fallback_node="safe_fallback"),
                ),
                "safe_fallback": Node(agent="fallback_agent", writes="output.reply"),
            },
            edges=[],
        )

        call_count = 0

        async def _complete(messages: list[dict]) -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ProviderError("error", status_code=500)
            return "fallback response"

        mock_provider = MagicMock()
        mock_provider.complete = _complete
        mock_provider.last_token_usage = TokenUsage(prompt_tokens=0, completion_tokens=5)

        with patch("sirenspec.core.agent_runner.resolve_provider", return_value=mock_provider):
            trace = await execute(wf, "hello")

        node_ids = [n["id"] for n in trace["nodes"]]
        assert "primary" in node_ids
        assert "safe_fallback" in node_ids
        assert trace["output"]["reply"] == "fallback response"

    @pytest.mark.asyncio
    async def test_workflow_defaults_retry_applied_to_node(self) -> None:
        """Workflow-level defaults.retry should apply when a node has no node-level retry."""
        call_count = 0

        async def _complete(messages: list[dict]) -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ProviderError("rate limited", status_code=429)
            return "ok"

        mock_provider = MagicMock()
        mock_provider.complete = _complete
        mock_provider.last_token_usage = TokenUsage(prompt_tokens=0, completion_tokens=5)

        # No node-level retry, but workflow defaults provide one.
        wf = _minimal_workflow(
            defaults=WorkflowDefaults(retry=RetryPolicy(max_attempts=2, backoff="constant", base_delay=0.0, on=["429"]))
        )

        with patch("sirenspec.core.agent_runner.resolve_provider", return_value=mock_provider):
            trace = await execute(wf, "hello")

        assert trace["summary"]["status"] == "success"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_node_level_retry_overrides_workflow_defaults(self) -> None:
        """Node-level retry should override workflow-level defaults."""
        call_count = 0

        async def _complete(messages: list[dict]) -> str:
            nonlocal call_count
            call_count += 1
            # Always fail with 500
            raise ProviderError("error", status_code=500)

        mock_provider = MagicMock()
        mock_provider.complete = _complete
        mock_provider.last_token_usage = TokenUsage(prompt_tokens=0, completion_tokens=0)

        # Workflow default would retry on 429 up to 3 times,
        # but node-level policy only retries on 500 with max_attempts=1.
        wf = _minimal_workflow(
            retry=RetryPolicy(max_attempts=1, backoff="constant", base_delay=0.0, on=["500"]),
            on_failure=OnFailurePolicy(action="abort"),
            defaults=WorkflowDefaults(
                retry=RetryPolicy(max_attempts=3, backoff="constant", base_delay=0.0, on=["429"])
            ),
        )

        with patch("sirenspec.core.agent_runner.resolve_provider", return_value=mock_provider):
            trace = await execute(wf, "hello")

        assert trace["summary"]["status"] == "failed"
        # Only 1 attempt because node-level max_attempts=1
        assert call_count == 1
