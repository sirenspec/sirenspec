"""Unit tests for OpenAI provider (mocked)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sirenspec.core.usage import TokenUsage
from sirenspec.providers.base import StreamingLLMProvider
from sirenspec.providers.openai_provider import OpenAIProvider


async def _make_async_iter(items: list) -> AsyncIterator:
    """Helper to produce an async iterator from a list."""
    for item in items:
        yield item


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
        mock_response.usage.prompt_tokens = 7
        mock_response.usage.completion_tokens = 3

        with patch.object(provider._client.chat.completions, "create", new=AsyncMock(return_value=mock_response)):
            result = await provider.complete([{"role": "user", "content": "hi"}])

        assert result == "Hello, world!"
        assert isinstance(result, str)

    @pytest.mark.asyncio
    async def test_token_usage_captured(self, provider: OpenAIProvider) -> None:
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Response text"
        mock_response.usage.prompt_tokens = 30
        mock_response.usage.completion_tokens = 12

        with patch.object(provider._client.chat.completions, "create", new=AsyncMock(return_value=mock_response)):
            await provider.complete([{"role": "user", "content": "test"}])

        assert isinstance(provider.last_token_usage, TokenUsage)
        assert provider.last_token_usage.prompt_tokens == 30
        assert provider.last_token_usage.completion_tokens == 12
        assert provider.last_token_usage.total == 42

    @pytest.mark.asyncio
    async def test_messages_passed_to_api(self, provider: OpenAIProvider) -> None:
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "ok"
        mock_response.usage.prompt_tokens = 3
        mock_response.usage.completion_tokens = 2

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

        assert provider.last_token_usage.prompt_tokens == 0
        assert provider.last_token_usage.completion_tokens == 0
        assert provider.last_token_usage.total == 0

    def test_satisfies_streaming_protocol(self, provider: OpenAIProvider) -> None:
        assert isinstance(provider, StreamingLLMProvider)

    @pytest.mark.asyncio
    async def test_stream_yields_chunks(self, provider: OpenAIProvider) -> None:
        chunk1 = MagicMock()
        chunk1.choices = [MagicMock()]
        chunk1.choices[0].delta.content = "Hello"
        chunk1.usage = None

        chunk2 = MagicMock()
        chunk2.choices = [MagicMock()]
        chunk2.choices[0].delta.content = " world"
        chunk2.usage = None

        final_chunk = MagicMock()
        final_chunk.choices = []
        final_chunk.usage = MagicMock()
        final_chunk.usage.total_tokens = 5

        mock_stream = _make_async_iter([chunk1, chunk2, final_chunk])

        with patch.object(provider._client.chat.completions, "create", new=AsyncMock(return_value=mock_stream)):
            chunks = []
            async for chunk in provider.stream([{"role": "user", "content": "hi"}]):
                chunks.append(chunk)

        assert chunks == ["Hello", " world"]
        assert provider.last_token_usage.total == 5

    @pytest.mark.asyncio
    async def test_stream_falls_back_to_char_estimate_when_no_usage(self, provider: OpenAIProvider) -> None:
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta.content = "abcdefgh"  # 8 chars → 8 // 4 = 2 estimated tokens
        chunk.usage = None

        mock_stream = _make_async_iter([chunk])

        with patch.object(provider._client.chat.completions, "create", new=AsyncMock(return_value=mock_stream)):
            chunks = [c async for c in provider.stream([{"role": "user", "content": "hi"}])]

        assert chunks == ["abcdefgh"]
        assert provider.last_token_usage.total == 2
