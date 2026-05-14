"""Unit tests for Anthropic provider (mocked)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sirenspec.providers.anthropic_provider import AnthropicProvider


@pytest.fixture
def provider(monkeypatch: pytest.MonkeyPatch) -> AnthropicProvider:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-dummy")
    return AnthropicProvider(model="claude-haiku-4-5-20251001")


class TestAnthropicProvider:
    @pytest.mark.asyncio
    async def test_complete_returns_string(self, provider: AnthropicProvider) -> None:
        mock_response = MagicMock()
        mock_response.content = [MagicMock()]
        mock_response.content[0].text = "Anthropic response"
        mock_response.usage.input_tokens = 10
        mock_response.usage.output_tokens = 20

        with patch.object(provider._client.messages, "create", new=AsyncMock(return_value=mock_response)):
            result = await provider.complete([{"role": "user", "content": "hello"}])

        assert result == "Anthropic response"
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_token_count_is_sum(self, provider: AnthropicProvider) -> None:
        mock_response = MagicMock()
        mock_response.content = [MagicMock()]
        mock_response.content[0].text = "ok"
        mock_response.usage.input_tokens = 15
        mock_response.usage.output_tokens = 25

        with patch.object(provider._client.messages, "create", new=AsyncMock(return_value=mock_response)):
            await provider.complete([{"role": "user", "content": "test"}])

        assert provider.last_token_count == 40

    @pytest.mark.asyncio
    async def test_system_message_extracted(self, provider: AnthropicProvider) -> None:
        mock_response = MagicMock()
        mock_response.content = [MagicMock()]
        mock_response.content[0].text = "ok"
        mock_response.usage.input_tokens = 5
        mock_response.usage.output_tokens = 5

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        mock_create = AsyncMock(return_value=mock_response)
        with patch.object(provider._client.messages, "create", new=mock_create):
            await provider.complete(messages)

        call_kwargs = mock_create.call_args.kwargs
        assert call_kwargs["system"] == "You are helpful."
        assert call_kwargs["messages"] == [{"role": "user", "content": "Hello"}]

    @pytest.mark.asyncio
    async def test_no_system_message_omitted(self, provider: AnthropicProvider) -> None:
        mock_response = MagicMock()
        mock_response.content = [MagicMock()]
        mock_response.content[0].text = "ok"
        mock_response.usage.input_tokens = 5
        mock_response.usage.output_tokens = 5

        mock_create = AsyncMock(return_value=mock_response)
        with patch.object(provider._client.messages, "create", new=mock_create):
            await provider.complete([{"role": "user", "content": "hi"}])

        call_kwargs = mock_create.call_args.kwargs
        assert "system" not in call_kwargs
