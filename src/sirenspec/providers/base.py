"""LLMProvider Protocol — all providers implement this interface."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from sirenspec.core.usage import TokenUsage


@runtime_checkable
class LLMProvider(Protocol):
    """Async LLM provider interface."""

    async def complete(self, messages: list[dict], max_tokens: int | None = None) -> str:
        """Send *messages* to the LLM and return the response text.

        :param messages: List of ``{"role": ..., "content": ...}`` dicts.
        :param max_tokens: Optional ceiling on completion tokens for this call.
            ``None`` means use the provider's default.  When supported, the provider
            forwards this to the underlying API so the model truncates its own response.
        :returns: The assistant reply text.
        """
        ...

    @property
    def last_token_usage(self) -> TokenUsage:
        """Structured token usage from the most recent call.

        :returns: A :class:`~sirenspec.core.usage.TokenUsage` with prompt and completion counts.
        """
        ...

    @property
    def client(self) -> object:
        """Return the underlying client object for direct access."""
        ...


class StreamingLLMProvider(LLMProvider, Protocol):
    """Extension of LLMProvider for providers that support token streaming.

    Providers that implement this protocol yield text chunks incrementally.
    Callers can detect streaming support with ``isinstance(provider, StreamingLLMProvider)``.
    """

    async def stream(self, messages: list[dict], max_tokens: int | None = None) -> AsyncIterator[str]:
        """Stream *messages* to the LLM and yield response text chunks.

        :param messages: List of ``{"role": ..., "content": ...}`` dicts.
        :param max_tokens: Optional ceiling on completion tokens for this call.
            ``None`` means use the provider's default.
        :returns: An async iterator that yields text chunks as they arrive.
        """
        ...
