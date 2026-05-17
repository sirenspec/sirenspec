"""OpenAI LLM provider implementation."""

from __future__ import annotations

import asyncio
import os

from openai import AsyncOpenAI

from sirenspec.core.usage import TokenUsage


class OpenAIProvider:
    """Wraps openai.AsyncOpenAI to satisfy the LLMProvider protocol."""

    def __init__(self, model: str) -> None:
        self.model = model
        api_key = os.environ.get("OPENAI_API_KEY")
        self._client = AsyncOpenAI(api_key=api_key)
        self._last_token_usage: TokenUsage = TokenUsage(prompt_tokens=0, completion_tokens=0)

    @property
    def last_token_usage(self) -> TokenUsage:
        """Structured token usage from the most recent call.

        :returns: A :class:`~sirenspec.core.usage.TokenUsage` with prompt and completion counts.
        """
        return self._last_token_usage

    @property
    def client(self) -> AsyncOpenAI:
        """Return the underlying AsyncOpenAI client.

        :returns: The AsyncOpenAI client instance.
        """
        return self._client

    async def complete(self, messages: list[dict]) -> str:
        """Call the OpenAI chat completions API and return the response text.

        :param messages: List of ``{"role": ..., "content": ...}`` dicts.
        :returns: The assistant reply text.
        """
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        if response.usage:
            self._last_token_usage = TokenUsage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
            )
        else:
            self._last_token_usage = TokenUsage(prompt_tokens=0, completion_tokens=0)
        content = response.choices[0].message.content
        return content or ""


if __name__ == "__main__":
    provider = OpenAIProvider(model="gpt-4o-mini")
    reply = asyncio.run(provider.complete([{"role": "user", "content": "Say hello in one sentence."}]))
    print(reply)
    print(f"tokens: {provider.last_token_usage.total}")
