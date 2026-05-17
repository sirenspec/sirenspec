"""LLMProvider Protocol — all providers implement this interface."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from sirenspec.core.usage import TokenUsage


@runtime_checkable
class LLMProvider(Protocol):
    """Async LLM provider interface."""

    async def complete(self, messages: list[dict]) -> str:
        """Send *messages* to the LLM and return the response text."""
        ...

    @property
    def last_token_usage(self) -> TokenUsage:
        """Structured token usage from the most recent call."""
        ...

    @property
    def client(self) -> object:
        """Return the underlying client object for direct access."""
        ...
