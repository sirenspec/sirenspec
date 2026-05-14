"""Prompt-injection detection guardrail."""

from __future__ import annotations

import re

from sirenspec.guardrails.base import Guardrail, GuardrailViolation

_INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|all|prior)\s+(previous\s+)?instructions?",
    r"disregard\s+(your\s+)?(previous|all|prior|your)\s+(previous\s+)?instructions?",
    r"you\s+are\s+now\s+",
    r"forget\s+(your|all|previous)\s+instructions?",
    r"new\s+instructions?:",
    r"override\s+(previous|all|prior)\s+instructions?",
    r"act\s+as\s+(if\s+you\s+are|a\s+)",
    r"pretend\s+(you\s+are|to\s+be)\s+",
    r"your\s+(new\s+)?role\s+is\s+",
    r"system\s*:\s*you\s+are\s+",
]

_COMPILED = [re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS]


class InjectionGuardrail(Guardrail):
    """Detects common prompt-injection signatures in input and output text."""

    def _check(self, text: str, direction: str) -> str:
        for pattern in _COMPILED:
            if pattern.search(text):
                raise GuardrailViolation(f"Injection pattern detected in {direction}: '{pattern.pattern}'")
        return text

    def check_input(self, text: str) -> str:
        """Raise GuardrailViolation if *text* contains an injection pattern.

        :param text: Input text to check.
        :raises GuardrailViolation: If an injection signature is found.
        :returns: The original text if no violation.
        """
        return self._check(text, "input")

    def check_output(self, text: str) -> str:
        """Raise GuardrailViolation if *text* contains an injection pattern.

        :param text: Output text to check.
        :raises GuardrailViolation: If an injection signature is found.
        :returns: The original text if no violation.
        """
        return self._check(text, "output")
