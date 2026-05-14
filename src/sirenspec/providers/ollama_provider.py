"""Ollama LLM provider implementation."""

from __future__ import annotations

import asyncio
import os

from openai import AsyncOpenAI

_DEFAULT_BASE_URL = "http://localhost:11434/v1"


class OllamaProvider:
    """Wraps openai.AsyncOpenAI pointed at a local Ollama server to satisfy the LLMProvider protocol."""

    def __init__(self, model: str) -> None:
        self.model = model
        base_url = os.environ.get("OLLAMA_BASE_URL", _DEFAULT_BASE_URL)
        # Ollama doesn't require a real key; the env var is a passthrough for auth-protected deployments.
        api_key = os.environ.get("OLLAMA_API_KEY", "ollama")
        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self._last_token_count: int = 0

    @property
    def last_token_count(self) -> int:
        return self._last_token_count

    @property
    def client(self) -> AsyncOpenAI:
        return self._client

    async def complete(self, messages: list[dict]) -> str:
        """Call the Ollama chat completions API and return the response text.

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
    provider = OllamaProvider(model="llama3.2")
    reply = asyncio.run(provider.complete([{"role": "user", "content": "Say hello in one sentence."}]))
    print(reply)
    print(f"tokens: {provider.last_token_count}")
