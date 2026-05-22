"""Ollama LLM provider implementation."""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator

from openai import AsyncOpenAI

from sirenspec.core.usage import TokenUsage

_DEFAULT_BASE_URL = "http://localhost:11434/v1"


class OllamaProvider:
    """Wraps openai.AsyncOpenAI pointed at a local Ollama server to satisfy the LLMProvider protocol."""

    def __init__(self, model: str) -> None:
        self.model = model
        base_url = os.environ.get("OLLAMA_BASE_URL", _DEFAULT_BASE_URL)
        # Ollama doesn't require a real key; the env var is a passthrough for auth-protected deployments.
        api_key = os.environ.get("OLLAMA_API_KEY", "ollama")
        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self._last_token_usage: TokenUsage = TokenUsage(prompt_tokens=0, completion_tokens=0)

    @property
    def last_token_usage(self) -> TokenUsage:
        """Structured token usage from the most recent call.

        :returns: A :class:`~sirenspec.core.usage.TokenUsage` with prompt and completion counts.
        """
        return self._last_token_usage

    @property
    def client(self) -> AsyncOpenAI:
        """Return the underlying AsyncOpenAI client (pointed at Ollama).

        :returns: The AsyncOpenAI client instance.
        """
        return self._client

    async def complete(self, messages: list[dict]) -> str:
        """Call the Ollama chat completions API and return the response text.

        Ollama exposes ``prompt_eval_count`` for prompt tokens and ``eval_count`` for
        completion tokens via the OpenAI-compatible usage object. Both fall back to 0
        if the Ollama version does not report them.

        :param messages: List of ``{"role": ..., "content": ...}`` dicts.
        :returns: The assistant reply text.
        """
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        if response.usage:
            self._last_token_usage = TokenUsage(
                prompt_tokens=response.usage.prompt_tokens or 0,
                completion_tokens=response.usage.completion_tokens or 0,
            )
        else:
            self._last_token_usage = TokenUsage(prompt_tokens=0, completion_tokens=0)
        content = response.choices[0].message.content
        return content or ""

    async def stream(self, messages: list[dict]) -> AsyncIterator[str]:
        """Stream the Ollama chat completions API and yield text chunks.

        Accumulates total character length as a token-count approximation
        (chars / 4) stored in ``last_token_count`` after the stream completes.
        If the final chunk includes a ``usage`` field, that value is used instead.

        :param messages: List of ``{"role": ..., "content": ...}`` dicts.
        :returns: An async iterator that yields text chunks as they arrive.
        """
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
        )
        char_count = 0
        usage_tokens: int | None = None
        async for chunk in response:
            if chunk.usage is not None:
                usage_tokens = chunk.usage.total_tokens
            if chunk.choices:
                delta = chunk.choices[0].delta.content
                if delta:
                    char_count += len(delta)
                    yield delta
        total = usage_tokens if usage_tokens is not None else char_count // 4
        self._last_token_usage = TokenUsage(prompt_tokens=0, completion_tokens=total)


if __name__ == "__main__":
    provider = OllamaProvider(model="llama3.2")
    reply = asyncio.run(provider.complete([{"role": "user", "content": "Say hello in one sentence."}]))
    print(reply)
    print(f"tokens: {provider.last_token_usage.total}")
