"""Unit tests for OpenAI provider (mocked)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sirenspec.providers.openai_provider import OpenAIProvider


@pytest.fixture
def provider(monkeypatch: pytest.MonkeyPatch) -> OpenAIProvider:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-dummy")
    return OpenAIProvider(model="gpt-4o-mini")


class TestOpenAIProvider:
    @pytest.mark.asyncio
    async def test_complete_returns_string(self, provider: OpenAIProvider) -> None:
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello, world!"
        mock_response.usage.total_tokens = 10

        with patch.object(provider._client.chat.completions, "create", new=AsyncMock(return_value=mock_response)):
            result = await provider.complete([{"role": "user", "content": "hi"}])

        assert result == "Hello, world!"
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_token_count_captured(self, provider: OpenAIProvider) -> None:
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Response text"
        mock_response.usage.total_tokens = 42

        with patch.object(provider._client.chat.completions, "create", new=AsyncMock(return_value=mock_response)):
            await provider.complete([{"role": "user", "content": "test"}])

        assert provider.last_token_count == 42

    @pytest.mark.asyncio
    async def test_messages_passed_to_api(self, provider: OpenAIProvider) -> None:
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "ok"
        mock_response.usage.total_tokens = 5

        messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "q"}]
        mock_create = AsyncMock(return_value=mock_response)
        with patch.object(provider._client.chat.completions, "create", new=mock_create):
            await provider.complete(messages)

        mock_create.assert_called_once_with(model="gpt-4o-mini", messages=messages)

    @pytest.mark.asyncio
    async def test_no_usage_defaults_zero(self, provider: OpenAIProvider) -> None:
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "ok"
        mock_response.usage = None

        with patch.object(provider._client.chat.completions, "create", new=AsyncMock(return_value=mock_response)):
            await provider.complete([{"role": "user", "content": "test"}])

        assert provider.last_token_count == 0
