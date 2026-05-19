"""Token usage accounting for SirenSpec provider calls."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TokenUsage:
    """Structured token count from a single LLM provider call.

    :param prompt_tokens: Tokens consumed by the prompt (input).
    :param completion_tokens: Tokens generated in the completion (output).
    """

    prompt_tokens: int
    completion_tokens: int

    @property
    def total(self) -> int:
        """Sum of prompt and completion tokens.

        :returns: Total token count for this call.
        """
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: TokenUsage) -> TokenUsage:
        """Return a new TokenUsage that is the sum of self and other.

        :param other: Another TokenUsage instance to add.
        :returns: A new TokenUsage with combined prompt and completion counts.
        """
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
        )
