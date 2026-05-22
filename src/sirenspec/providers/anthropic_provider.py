"""Anthropic LLM provider implementation."""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator

from anthropic import AsyncAnthropic


class AnthropicProvider:
    """Wraps anthropic.AsyncAnthropic to satisfy the StreamingLLMProvider protocol."""

    def __init__(self, model: str) -> None:
        self.model = model
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        self._client = AsyncAnthropic(api_key=api_key)
        self._last_token_count: int = 0

    @property
    def last_token_count(self) -> int:
        return self._last_token_count

    @property
    def client(self) -> AsyncAnthropic:
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
        self._last_token_count = response.usage.input_tokens + response.usage.output_tokens
        return response.content[0].text

    async def stream(self, messages: list[dict]) -> AsyncIterator[str]:
        """Stream the Anthropic messages API and yield text chunks.

        Extracts any system message from *messages* and passes it as the
        ``system`` kwarg, matching the same behaviour as :meth:`complete`.
        Token counts are captured from the stream's usage metadata after the
        stream completes.

        :param messages: List of ``{"role": ..., "content": ...}`` dicts; any system
            message is extracted and passed separately.
        :returns: An async iterator that yields text chunks as they arrive.
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

        async with self.client.messages.stream(**kwargs) as stream:
            async for text in stream.text_stream:
                yield text
            usage = (await stream.get_final_message()).usage
            self._last_token_count = usage.input_tokens + usage.output_tokens


if __name__ == "__main__":
    provider = AnthropicProvider(model="claude-haiku-4-5-20251001")
    reply = asyncio.run(provider.complete([{"role": "user", "content": "Say hello in one sentence."}]))
    print(reply)
    print(f"tokens: {provider.last_token_count}")
