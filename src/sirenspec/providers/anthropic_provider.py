"""Anthropic LLM provider implementation."""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator

from anthropic import AsyncAnthropic

from sirenspec.core.usage import TokenUsage


class AnthropicProvider:
    """Wraps anthropic.AsyncAnthropic to satisfy the StreamingLLMProvider protocol."""

    def __init__(self, model: str) -> None:
        self.model = model
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        self._client = AsyncAnthropic(api_key=api_key)
        self._last_token_usage: TokenUsage = TokenUsage(prompt_tokens=0, completion_tokens=0)

    @property
    def last_token_usage(self) -> TokenUsage:
        """Structured token usage from the most recent call.

        :returns: A :class:`~sirenspec.core.usage.TokenUsage` with prompt and completion counts.
        """
        return self._last_token_usage

    @property
    def client(self) -> AsyncAnthropic:
        """Return the underlying AsyncAnthropic client.

        :returns: The AsyncAnthropic client instance.
        """
        return self._client

    async def complete(self, messages: list[dict], max_tokens: int | None = None) -> str:
        """Call the Anthropic messages API and return the response text.

        :param messages: List of ``{"role": ..., "content": ...}`` dicts; any system
            message is extracted and passed separately.
        :param max_tokens: Optional ceiling forwarded as the ``max_tokens`` API param;
            defaults to 4096 when omitted (Anthropic requires the field).
        :returns: The assistant reply text.
        """
        system_prompt = ""
        user_messages: list[dict] = []

        for msg in messages:
            if msg.get("role") == "system":
                system_prompt = msg.get("content", "")
            else:
                user_messages.append(msg)

        kwargs: dict = {
            "model": self.model,
            "max_tokens": max_tokens if max_tokens is not None else 4096,
            "messages": user_messages,
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        response = await self.client.messages.create(**kwargs)
        self._last_token_usage = TokenUsage(
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
        )
        return response.content[0].text

    async def stream(self, messages: list[dict], max_tokens: int | None = None) -> AsyncIterator[str]:
        """Stream the Anthropic messages API and yield text chunks.

        Extracts any system message from *messages* and passes it as the
        ``system`` kwarg, matching the same behaviour as :meth:`complete`.
        Token counts are captured from the stream's usage metadata after the
        stream completes.

        :param messages: List of ``{"role": ..., "content": ...}`` dicts; any system
            message is extracted and passed separately.
        :param max_tokens: Optional ceiling forwarded as the ``max_tokens`` API param;
            defaults to 4096 when omitted (Anthropic requires the field).
        :returns: An async iterator that yields text chunks as they arrive.
        """
        system_prompt = ""
        user_messages: list[dict] = []

        for msg in messages:
            if msg.get("role") == "system":
                system_prompt = msg.get("content", "")
            else:
                user_messages.append(msg)

        kwargs: dict = {
            "model": self.model,
            "max_tokens": max_tokens if max_tokens is not None else 4096,
            "messages": user_messages,
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        async with self.client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                yield text
            usage = (await stream.get_final_message()).usage
            self._last_token_usage = TokenUsage(
                prompt_tokens=usage.input_tokens,
                completion_tokens=usage.output_tokens,
            )


if __name__ == "__main__":
    provider = AnthropicProvider(model="claude-haiku-4-5-20251001")
    reply = asyncio.run(provider.complete([{"role": "user", "content": "Say hello in one sentence."}]))
    print(reply)
    print(f"tokens: {provider.last_token_usage.total}")
