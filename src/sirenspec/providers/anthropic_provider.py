"""Anthropic LLM provider implementation."""

from __future__ import annotations

import asyncio
import os

from anthropic import AsyncAnthropic

from sirenspec.core.usage import TokenUsage


class AnthropicProvider:
    """Wraps anthropic.AsyncAnthropic to satisfy the LLMProvider protocol."""

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

    async def complete(self, messages: list[dict]) -> str:
        """Call the Anthropic messages API and return the response text.

        :param messages: List of ``{"role": ..., "content": ...}`` dicts; any system
            message is extracted and passed separately.
        :returns: The assistant reply text.
        """
        system_prompt = ""
        user_messages: list[dict] = []

        for msg in messages:
            if msg.get("role") == "system":
                system_prompt = msg.get("content", "")
            else:
                user_messages.append(msg)

        kwargs: dict = {"model": self.model, "max_tokens": 4096, "messages": user_messages}
        if system_prompt:
            kwargs["system"] = system_prompt

        response = await self.client.messages.create(**kwargs)
        self._last_token_usage = TokenUsage(
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
        )
        return response.content[0].text


if __name__ == "__main__":
    provider = AnthropicProvider(model="claude-haiku-4-5-20251001")
    reply = asyncio.run(provider.complete([{"role": "user", "content": "Say hello in one sentence."}]))
    print(reply)
    print(f"tokens: {provider.last_token_usage.total}")
