"""Tests for per-node and per-run token usage accounting (issue #40)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sirenspec.core.usage import TokenUsage
from sirenspec.providers.anthropic_provider import AnthropicProvider
from sirenspec.providers.openai_provider import OpenAIProvider

# ---------------------------------------------------------------------------
# TokenUsage dataclass
# ---------------------------------------------------------------------------


class TestTokenUsage:
    def test_total_returns_sum(self) -> None:
        usage = TokenUsage(prompt_tokens=30, completion_tokens=12)
        assert usage.total == 42

    def test_total_zero(self) -> None:
        usage = TokenUsage(prompt_tokens=0, completion_tokens=0)
        assert usage.total == 0

    def test_add_combines_counts(self) -> None:
        a = TokenUsage(prompt_tokens=10, completion_tokens=5)
        b = TokenUsage(prompt_tokens=20, completion_tokens=15)
        result = a + b
        assert result.prompt_tokens == 30
        assert result.completion_tokens == 20
        assert result.total == 50

    def test_add_returns_new_instance(self) -> None:
        a = TokenUsage(prompt_tokens=1, completion_tokens=1)
        b = TokenUsage(prompt_tokens=2, completion_tokens=2)
        result = a + b
        assert result is not a
        assert result is not b

    def test_prompt_and_completion_stored_separately(self) -> None:
        usage = TokenUsage(prompt_tokens=100, completion_tokens=200)
        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 200


# ---------------------------------------------------------------------------
# OpenAI provider: TokenUsage from mocked response
# ---------------------------------------------------------------------------


class TestOpenAIProviderTokenUsage:
    @pytest.fixture
    def provider(self, monkeypatch: pytest.MonkeyPatch) -> OpenAIProvider:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-dummy")
        return OpenAIProvider(model="gpt-4o-mini")

    @pytest.mark.asyncio
    async def test_returns_correct_token_usage(self, provider: OpenAIProvider) -> None:
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello"
        mock_response.usage.prompt_tokens = 40
        mock_response.usage.completion_tokens = 15

        with patch.object(provider._client.chat.completions, "create", new=AsyncMock(return_value=mock_response)):
            await provider.complete([{"role": "user", "content": "hi"}])

        usage = provider.last_token_usage
        assert isinstance(usage, TokenUsage)
        assert usage.prompt_tokens == 40
        assert usage.completion_tokens == 15
        assert usage.total == 55

    @pytest.mark.asyncio
    async def test_no_usage_falls_back_to_zero(self, provider: OpenAIProvider) -> None:
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "ok"
        mock_response.usage = None

        with patch.object(provider._client.chat.completions, "create", new=AsyncMock(return_value=mock_response)):
            await provider.complete([{"role": "user", "content": "test"}])

        assert provider.last_token_usage.prompt_tokens == 0
        assert provider.last_token_usage.completion_tokens == 0
        assert provider.last_token_usage.total == 0


# ---------------------------------------------------------------------------
# Anthropic provider: input_tokens → prompt_tokens, output_tokens → completion_tokens
# ---------------------------------------------------------------------------


class TestAnthropicProviderTokenUsage:
    @pytest.fixture
    def provider(self, monkeypatch: pytest.MonkeyPatch) -> AnthropicProvider:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-dummy")
        return AnthropicProvider(model="claude-haiku-4-5-20251001")

    @pytest.mark.asyncio
    async def test_maps_input_tokens_to_prompt_tokens(self, provider: AnthropicProvider) -> None:
        mock_response = MagicMock()
        mock_response.content = [MagicMock()]
        mock_response.content[0].text = "response"
        mock_response.usage.input_tokens = 50
        mock_response.usage.output_tokens = 25

        with patch.object(provider._client.messages, "create", new=AsyncMock(return_value=mock_response)):
            await provider.complete([{"role": "user", "content": "hello"}])

        usage = provider.last_token_usage
        assert isinstance(usage, TokenUsage)
        assert usage.prompt_tokens == 50
        assert usage.completion_tokens == 25
        assert usage.total == 75

    @pytest.mark.asyncio
    async def test_last_token_usage_reflects_most_recent_call(self, provider: AnthropicProvider) -> None:
        def _mock_response(input_t: int, output_t: int) -> MagicMock:
            r = MagicMock()
            r.content = [MagicMock()]
            r.content[0].text = "ok"
            r.usage.input_tokens = input_t
            r.usage.output_tokens = output_t
            return r

        with patch.object(provider._client.messages, "create", new=AsyncMock(return_value=_mock_response(10, 5))):
            await provider.complete([{"role": "user", "content": "first"}])
        assert provider.last_token_usage.prompt_tokens == 10
        assert provider.last_token_usage.completion_tokens == 5

        with patch.object(provider._client.messages, "create", new=AsyncMock(return_value=_mock_response(20, 8))):
            await provider.complete([{"role": "user", "content": "second"}])
        assert provider.last_token_usage.prompt_tokens == 20
        assert provider.last_token_usage.completion_tokens == 8


# ---------------------------------------------------------------------------
# Executor trace: per-node usage dict and total_usage summary
# ---------------------------------------------------------------------------


class TestExecutorTokenUsageTrace:
    @pytest.mark.asyncio
    async def test_per_node_usage_dict_present(self, minimal_workflow) -> None:
        from sirenspec.core.executor import execute

        mock_provider = MagicMock()
        mock_provider.complete = AsyncMock(return_value="test response")
        mock_provider.last_token_usage = TokenUsage(prompt_tokens=20, completion_tokens=10)

        with patch("sirenspec.core.agent_runner.resolve_provider", return_value=mock_provider):
            trace = await execute(minimal_workflow, "hello")

        node = trace["nodes"][0]
        assert "usage" in node
        assert node["usage"]["prompt_tokens"] == 20
        assert node["usage"]["completion_tokens"] == 10
        assert node["usage"]["estimated_usd"] is None

    @pytest.mark.asyncio
    async def test_summary_contains_total_usage(self, minimal_workflow) -> None:
        from sirenspec.core.executor import execute

        mock_provider = MagicMock()
        mock_provider.complete = AsyncMock(return_value="test response")
        mock_provider.last_token_usage = TokenUsage(prompt_tokens=30, completion_tokens=15)

        with patch("sirenspec.core.agent_runner.resolve_provider", return_value=mock_provider):
            trace = await execute(minimal_workflow, "hello")

        summary = trace["summary"]
        assert "total_usage" in summary
        assert summary["total_usage"]["prompt_tokens"] == 30
        assert summary["total_usage"]["completion_tokens"] == 15
        assert summary["total_usage"]["total_tokens"] == 45
        assert summary["total_usage"]["estimated_usd"] is None

    @pytest.mark.asyncio
    async def test_total_tokens_accumulated_across_nodes(self, sequential_workflow) -> None:
        from sirenspec.core.executor import execute

        mock_provider = MagicMock()
        mock_provider.complete = AsyncMock(return_value="response")
        mock_provider.last_token_usage = TokenUsage(prompt_tokens=10, completion_tokens=5)

        with patch("sirenspec.core.agent_runner.resolve_provider", return_value=mock_provider):
            trace = await execute(sequential_workflow, "hello")

        # Two nodes each contributing 10 prompt + 5 completion = 30 total
        summary = trace["summary"]
        assert summary["total_usage"]["prompt_tokens"] == 20
        assert summary["total_usage"]["completion_tokens"] == 10
        assert summary["total_usage"]["total_tokens"] == 30
        assert summary["total_tokens"] == 30
