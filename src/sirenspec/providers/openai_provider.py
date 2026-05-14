"""OpenAI LLM provider implementation."""

from __future__ import annotations

import asyncio
import os

from openai import AsyncOpenAI


class OpenAIProvider:
    """Wraps openai.AsyncOpenAI to satisfy the LLMProvider protocol."""

    def __init__(self, model: str) -> None:
        self.model = model
        api_key = os.environ.get("OPENAI_API_KEY")
        self._client = AsyncOpenAI(api_key=api_key)
        self._last_token_count: int = 0

    @property
    def last_token_count(self) -> int:
        return self._last_token_count

    @property
    def client(self) -> AsyncOpenAI:
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
        self._last_token_count = response.usage.total_tokens if response.usage else 0
        content = response.choices[0].message.content
        return content or ""


if __name__ == "__main__":
    provider = OpenAIProvider(model="gpt-4o-mini")
    reply = asyncio.run(provider.complete([{"role": "user", "content": "Say hello in one sentence."}]))
    print(reply)
    print(f"tokens: {provider.last_token_count}")
