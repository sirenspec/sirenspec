"""Output length guardrail — truncates or raises on over-long LLM responses."""

from __future__ import annotations

from sirenspec.guardrails.base import Guardrail, GuardrailViolation


class LengthGuardrail(Guardrail):
    """Enforces a maximum character length on LLM output.

    :param max_chars: Maximum allowed output length (default 4000).
    :param mode: ``'truncate'`` appends ``'...'`` and trims; ``'raise'`` raises GuardrailViolation.
    """

    def __init__(self, max_chars: int = 4000, mode: str = "truncate") -> None:
        if mode not in {"truncate", "raise"}:
            raise ValueError(f"Invalid mode '{mode}'; expected 'truncate' or 'raise'")
        self.max_chars = max_chars
        self.mode = mode

    def check_input(self, text: str) -> str:
        """Input is not length-limited; pass through unchanged.

        :param text: Input text.
        :returns: The original text unchanged.
        """
        return text

    def check_output(self, text: str) -> str:
        """Enforce max_chars on output text.

        :param text: Output text to check.
        :raises GuardrailViolation: In ``'raise'`` mode when length is exceeded.
        :returns: The (possibly truncated) output text.
        """
        if len(text) <= self.max_chars:
            return text

        if self.mode == "raise":
            raise GuardrailViolation(f"Output length {len(text)} exceeds max_chars={self.max_chars}")

        return text[: self.max_chars] + "..."
