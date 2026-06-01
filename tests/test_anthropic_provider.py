"""Unit tests for Anthropic provider (mocked)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sirenspec.core.usage import TokenUsage
from sirenspec.providers.anthropic_provider import AnthropicProvider
from sirenspec.providers.base import StreamingLLMProvider


async def _make_async_iter(items: list) -> AsyncIterator:
    """Helper to produce an async iterator from a list."""
    for item in items:
        yield item


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
    async def test_token_usage_captured(self, provider: AnthropicProvider) -> None:
        mock_response = MagicMock()
        mock_response.content = [MagicMock()]
        mock_response.content[0].text = "ok"
        mock_response.usage.input_tokens = 15
        mock_response.usage.output_tokens = 25

        with patch.object(provider._client.messages, "create", new=AsyncMock(return_value=mock_response)):
            await provider.complete([{"role": "user", "content": "test"}])

        assert isinstance(provider.last_token_usage, TokenUsage)
        assert provider.last_token_usage.prompt_tokens == 15
        assert provider.last_token_usage.completion_tokens == 25
        assert provider.last_token_usage.total == 40

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

    def test_satisfies_streaming_protocol(self, provider: AnthropicProvider) -> None:
        assert isinstance(provider, StreamingLLMProvider)

    @pytest.mark.asyncio
    async def test_stream_yields_chunks(self, provider: AnthropicProvider) -> None:
        final_message = MagicMock()
        final_message.usage.input_tokens = 10
        final_message.usage.output_tokens = 20

        mock_stream_ctx = MagicMock()
        mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_stream_ctx)
        mock_stream_ctx.__aexit__ = AsyncMock(return_value=False)
        mock_stream_ctx.text_stream = _make_async_iter(["Hello", " world"])
        mock_stream_ctx.get_final_message = AsyncMock(return_value=final_message)

        with patch.object(provider._client.messages, "stream", return_value=mock_stream_ctx):
            chunks = []
            async for chunk in provider.stream([{"role": "user", "content": "hi"}]):
                chunks.append(chunk)

        assert chunks == ["Hello", " world"]
        assert provider.last_token_usage.total == 30

    @pytest.mark.asyncio
    async def test_auth_error_wrapped_as_provider_error(self, provider: AnthropicProvider) -> None:
        """AuthenticationError from the Anthropic SDK is re-raised as ProviderError."""
        from anthropic import AuthenticationError as AnthropicAuthError

        from sirenspec.exceptions import ProviderError

        exc = AnthropicAuthError(message="invalid x-api-key", response=MagicMock(status_code=401), body={})
        with patch.object(provider._client.messages, "create", new=AsyncMock(side_effect=exc)):
            with pytest.raises(ProviderError, match="authentication failed"):
                await provider.complete([{"role": "user", "content": "hi"}])

    @pytest.mark.asyncio
    async def test_api_status_error_wrapped_as_provider_error(self, provider: AnthropicProvider) -> None:
        """APIStatusError from the Anthropic SDK is re-raised as ProviderError."""
        from anthropic import APIStatusError

        from sirenspec.exceptions import ProviderError

        mock_response = MagicMock()
        mock_response.status_code = 429
        exc = APIStatusError(message="rate limit", response=mock_response, body={})
        with patch.object(provider._client.messages, "create", new=AsyncMock(side_effect=exc)):
            with pytest.raises(ProviderError, match="429"):
                await provider.complete([{"role": "user", "content": "hi"}])
