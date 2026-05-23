"""PII detection and redaction guardrail."""

from __future__ import annotations

import re
from typing import Literal

from sirenspec.exceptions import PIIDetectedError
from sirenspec.guardrails.base import Guardrail

# ---------------------------------------------------------------------------
# Entity patterns
# ---------------------------------------------------------------------------

_EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")

# US phone: optional +1 country code, 7/10/11 digits with common separators ()-. and space.
_PHONE_PATTERN = re.compile(
    r"(?<!\d)"
    r"(\+1[\s.-]?)?"
    r"(?:\(?\d{3}\)?[\s.-]?)?"
    r"\d{3}[\s.-]?"
    r"\d{4}"
    r"(?!\d)"
)

_SSN_PATTERN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")

# 13–19 consecutive digits with optional spaces or dashes between groups of 4.
_CREDIT_CARD_PATTERN = re.compile(r"\b(?:\d{4}[\s-]?){3}\d{1,7}\b")

_KNOWN_ENTITIES: dict[str, re.Pattern[str]] = {
    "email": _EMAIL_PATTERN,
    "phone": _PHONE_PATTERN,
    "ssn": _SSN_PATTERN,
    "credit_card": _CREDIT_CARD_PATTERN,
}


# ---------------------------------------------------------------------------
# Luhn validation
# ---------------------------------------------------------------------------


def luhn_valid(number: str) -> bool:
    """Return ``True`` if *number* passes the Luhn checksum algorithm.

    Non-digit characters are stripped before checking, so formatted strings
    like ``'4111 1111 1111 1111'`` are accepted.

    :param number: A string containing the card number (may include spaces/dashes).
    :returns: ``True`` if the number satisfies the Luhn formula, ``False`` otherwise.
    """
    digits = [int(ch) for ch in number if ch.isdigit()]
    if len(digits) < 13:
        return False
    total = 0
    for i, digit in enumerate(reversed(digits)):
        if i % 2 == 1:
            doubled = digit * 2
            total += doubled - 9 if doubled > 9 else doubled
        else:
            total += digit
    return total % 10 == 0


# ---------------------------------------------------------------------------
# Match helpers
# ---------------------------------------------------------------------------


def find_credit_card_matches(text: str) -> list[re.Match[str]]:
    """Return all credit card pattern matches in *text* that pass Luhn validation.

    :param text: The text to search.
    :returns: List of regex match objects for valid credit card numbers.
    """
    return [m for m in _CREDIT_CARD_PATTERN.finditer(text) if luhn_valid(m.group())]


def find_entity_matches(text: str, entities: list[str]) -> dict[str, list[re.Match[str]]]:
    """Return all regex matches grouped by entity type for the given *entities* list.

    Credit card matches are additionally filtered through Luhn validation.

    :param text: The text to scan.
    :param entities: Entity type names to check (e.g. ``['email', 'ssn']``).
    :returns: Mapping of entity name to list of match objects.
    """
    results: dict[str, list[re.Match[str]]] = {}
    for entity in entities:
        if entity == "credit_card":
            matches = find_credit_card_matches(text)
        else:
            matches = list(_KNOWN_ENTITIES[entity].finditer(text))
        if matches:
            results[entity] = matches
    return results


def redact_text(text: str, matches_by_entity: dict[str, list[re.Match[str]]], replacement: str) -> str:
    """Replace all matched spans in *text* with *replacement*.

    Spans from all entity types are merged and sorted so that overlapping or
    adjacent replacements are applied in a single left-to-right pass.

    :param text: The original text.
    :param matches_by_entity: Match objects keyed by entity type.
    :param replacement: The string to substitute for each matched span.
    :returns: Text with all matched spans replaced.
    """
    all_spans: list[tuple[int, int]] = []
    for match_list in matches_by_entity.values():
        for m in match_list:
            all_spans.append((m.start(), m.end()))

    all_spans.sort()

    parts: list[str] = []
    cursor = 0
    for start, end in all_spans:
        if start < cursor:
            # Overlapping span — skip (already covered).
            continue
        parts.append(text[cursor:start])
        parts.append(replacement)
        cursor = end
    parts.append(text[cursor:])
    return "".join(parts)


# ---------------------------------------------------------------------------
# Guardrail
# ---------------------------------------------------------------------------


class PIIGuardrail(Guardrail):
    """Detects and handles personally identifiable information in text.

    Supports three actions:

    * ``'redact'`` — replace each matched span with *replacement* and return the sanitised text.
    * ``'block'`` — raise :class:`~sirenspec.exceptions.PIIDetectedError` if any PII is found.
    * ``'flag'`` — return the text unchanged (annotation is the executor's responsibility).

    :param entities: Entity types to detect. ``None`` means all supported types
        (``email``, ``phone``, ``ssn``, ``credit_card``).
    :param action: What to do when PII is detected.
    :param replacement: Replacement string used when *action* is ``'redact'``.
    """

    def __init__(
        self,
        entities: list[str] | None = None,
        action: Literal["redact", "block", "flag"] = "redact",
        replacement: str = "[REDACTED]",
    ) -> None:
        self.entities: list[str] = entities if entities is not None else list(_KNOWN_ENTITIES.keys())
        self.action = action
        self.replacement = replacement

    def apply(self, text: str) -> str:
        """Detect PII in *text* and apply the configured action.

        :param text: Text to inspect.
        :raises PIIDetectedError: When *action* is ``'block'`` and PII is found.
        :returns: The (possibly transformed) text.
        """
        matches = find_entity_matches(text, self.entities)
        if not matches:
            return text

        if self.action == "redact":
            return redact_text(text, matches, self.replacement)

        if self.action == "block":
            raise PIIDetectedError(entity_types=sorted(matches.keys()))

        # action == "flag": return unchanged
        return text

    def check_input(self, text: str) -> str:
        """Detect or redact PII in *text* before it is sent to the LLM.

        :param text: Input text to check.
        :raises PIIDetectedError: If *action* is ``'block'`` and PII is found.
        :returns: The (possibly transformed) input text.
        """
        return self.apply(text)

    def check_output(self, text: str) -> str:
        """Detect or redact PII in *text* after it is received from the LLM.

        :param text: Output text to check.
        :raises PIIDetectedError: If *action* is ``'block'`` and PII is found.
        :returns: The (possibly transformed) output text.
        """
        return self.apply(text)
