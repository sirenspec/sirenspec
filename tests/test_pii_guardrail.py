"""Tests for PIIGuardrail — entity detection, actions, config, and property tests."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from sirenspec.exceptions import GuardrailError, PIIDetectedError, SirenSpecError
from sirenspec.guardrails.pii import PIIGuardrail, luhn_valid
from sirenspec.guardrails.registry import build_guardrails, make_pii_guardrail

# ---------------------------------------------------------------------------
# Luhn validation unit tests
# ---------------------------------------------------------------------------


class TestLuhnValid:
    def test_valid_visa_test_number(self) -> None:
        assert luhn_valid("4111111111111111") is True

    def test_invalid_fails_luhn(self) -> None:
        assert luhn_valid("4111111111111110") is False

    def test_strips_spaces(self) -> None:
        assert luhn_valid("4111 1111 1111 1111") is True

    def test_strips_dashes(self) -> None:
        assert luhn_valid("4111-1111-1111-1111") is True

    def test_too_short_returns_false(self) -> None:
        assert luhn_valid("12345") is False


# ---------------------------------------------------------------------------
# True-positive detection (redact action)
# ---------------------------------------------------------------------------


class TestRedactTruePositives:
    def test_redacts_email(self) -> None:
        guardrail = PIIGuardrail(action="redact")
        result = guardrail.check_output("Contact user@example.com for details.")
        assert "user@example.com" not in result
        assert "[REDACTED]" in result

    def test_redacts_us_phone_parens(self) -> None:
        guardrail = PIIGuardrail(action="redact")
        result = guardrail.check_output("Call (555) 123-4567 now.")
        assert "(555) 123-4567" not in result
        assert "[REDACTED]" in result

    def test_redacts_us_phone_dashes(self) -> None:
        guardrail = PIIGuardrail(action="redact")
        result = guardrail.check_output("Reach us at 555-123-4567.")
        assert "555-123-4567" not in result
        assert "[REDACTED]" in result

    def test_redacts_us_phone_plus_country_code(self) -> None:
        guardrail = PIIGuardrail(action="redact")
        result = guardrail.check_output("My number is +15551234567.")
        assert "+15551234567" not in result
        assert "[REDACTED]" in result

    def test_redacts_ssn(self) -> None:
        guardrail = PIIGuardrail(action="redact")
        result = guardrail.check_output("SSN: 123-45-6789")
        assert "123-45-6789" not in result
        assert "[REDACTED]" in result

    def test_redacts_valid_credit_card(self) -> None:
        guardrail = PIIGuardrail(action="redact")
        # 4111111111111111 is the canonical Visa test number (passes Luhn)
        result = guardrail.check_output("Card: 4111111111111111")
        assert "4111111111111111" not in result
        assert "[REDACTED]" in result

    def test_check_input_also_redacts(self) -> None:
        guardrail = PIIGuardrail(action="redact")
        result = guardrail.check_input("My email is user@example.com.")
        assert "user@example.com" not in result
        assert "[REDACTED]" in result


# ---------------------------------------------------------------------------
# True-negative (should NOT be redacted)
# ---------------------------------------------------------------------------


class TestRedactTrueNegatives:
    def test_invalid_luhn_not_redacted_as_credit_card(self) -> None:
        guardrail = PIIGuardrail(entities=["credit_card"], action="redact")
        text = "Number: 4111111111111110"
        assert guardrail.check_output(text) == text

    def test_short_number_not_treated_as_ssn(self) -> None:
        guardrail = PIIGuardrail(entities=["ssn"], action="redact")
        text = "Code: 12345"
        assert guardrail.check_output(text) == text

    def test_no_pii_returns_unchanged(self) -> None:
        guardrail = PIIGuardrail(action="redact")
        text = "Hello, world! Nothing sensitive here."
        assert guardrail.check_output(text) == text


# ---------------------------------------------------------------------------
# Action: block
# ---------------------------------------------------------------------------


class TestBlockAction:
    def test_block_raises_pii_detected_error(self) -> None:
        guardrail = PIIGuardrail(action="block")
        with pytest.raises(PIIDetectedError) as exc_info:
            guardrail.check_output("Email: admin@example.com")
        assert "email" in exc_info.value.entity_types

    def test_block_error_does_not_contain_matched_value(self) -> None:
        guardrail = PIIGuardrail(action="block")
        with pytest.raises(PIIDetectedError) as exc_info:
            guardrail.check_output("SSN: 123-45-6789")
        assert "123-45-6789" not in str(exc_info.value)

    def test_block_reports_entity_types(self) -> None:
        guardrail = PIIGuardrail(action="block")
        with pytest.raises(PIIDetectedError) as exc_info:
            guardrail.check_output("user@example.com and 123-45-6789")
        assert set(exc_info.value.entity_types) == {"email", "ssn"}

    def test_block_no_pii_does_not_raise(self) -> None:
        guardrail = PIIGuardrail(action="block")
        result = guardrail.check_output("No PII here.")
        assert result == "No PII here."


# ---------------------------------------------------------------------------
# Action: flag
# ---------------------------------------------------------------------------


class TestFlagAction:
    def test_flag_returns_text_unchanged(self) -> None:
        guardrail = PIIGuardrail(action="flag")
        text = "Email: user@example.com"
        assert guardrail.check_output(text) == text

    def test_flag_input_returns_text_unchanged(self) -> None:
        guardrail = PIIGuardrail(action="flag")
        text = "SSN: 123-45-6789"
        assert guardrail.check_input(text) == text


# ---------------------------------------------------------------------------
# Entity filtering via config
# ---------------------------------------------------------------------------


class TestEntityFiltering:
    def test_email_only_redacts_email_not_phone(self) -> None:
        guardrail = PIIGuardrail(entities=["email"], action="redact")
        text = "Email: user@example.com Phone: 555-123-4567"
        result = guardrail.check_output(text)
        assert "user@example.com" not in result
        assert "555-123-4567" in result

    def test_ssn_only_leaves_email_untouched(self) -> None:
        guardrail = PIIGuardrail(entities=["ssn"], action="redact")
        text = "SSN: 123-45-6789 Email: user@example.com"
        result = guardrail.check_output(text)
        assert "123-45-6789" not in result
        assert "user@example.com" in result


# ---------------------------------------------------------------------------
# Custom replacement string
# ---------------------------------------------------------------------------


class TestCustomReplacement:
    def test_custom_replacement_string(self) -> None:
        guardrail = PIIGuardrail(action="redact", replacement="***")
        result = guardrail.check_output("user@example.com")
        assert "***" in result
        assert "user@example.com" not in result


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_bare_string_builds_default_pii_guardrail(self) -> None:
        guardrails = build_guardrails(["pii"])
        assert len(guardrails) == 1
        assert isinstance(guardrails[0], PIIGuardrail)

    def test_guardrail_spec_passes_config(self) -> None:
        from sirenspec.core.models import GuardrailSpec

        spec = GuardrailSpec(name="pii", config={"entities": ["email"], "action": "block"})
        guardrails = build_guardrails([spec])
        assert len(guardrails) == 1
        g = guardrails[0]
        assert isinstance(g, PIIGuardrail)
        assert g.action == "block"
        assert g.entities == ["email"]

    def test_unknown_entity_in_config_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown PII entity"):
            make_pii_guardrail(config={"entities": ["email", "passport"]})

    def test_pii_not_in_defaults(self) -> None:
        # Default guardrails (None) should NOT include pii.
        guardrails = build_guardrails(None)
        assert not any(isinstance(g, PIIGuardrail) for g in guardrails)


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------


class TestExceptionHierarchy:
    def test_pii_detected_error_is_guardrail_error(self) -> None:
        err = PIIDetectedError(["email"])
        assert isinstance(err, GuardrailError)

    def test_pii_detected_error_is_siren_spec_error(self) -> None:
        err = PIIDetectedError(["email"])
        assert isinstance(err, SirenSpecError)

    def test_guardrail_violation_is_guardrail_error(self) -> None:
        from sirenspec.guardrails.base import GuardrailViolation

        err = GuardrailViolation("test")
        assert isinstance(err, GuardrailError)

    def test_guardrail_violation_is_siren_spec_error(self) -> None:
        from sirenspec.guardrails.base import GuardrailViolation

        err = GuardrailViolation("test")
        assert isinstance(err, SirenSpecError)

    def test_pii_detected_error_entity_types_stored(self) -> None:
        err = PIIDetectedError(["ssn", "email"])
        assert err.entity_types == ["ssn", "email"]

    def test_pii_detected_error_message_excludes_values(self) -> None:
        err = PIIDetectedError(["email"])
        # Message should mention entity types, not matched values.
        assert "email" in str(err)


# ---------------------------------------------------------------------------
# Hypothesis property tests
# ---------------------------------------------------------------------------

# Characters that cannot form PII patterns — safe for generating "clean" text.
_SAFE_ALPHABET = st.text(
    alphabet=st.characters(
        whitelist_categories=("Lu", "Ll", "Zs"),
        whitelist_characters=" ",
        blacklist_characters="@0123456789.-+()_",
    ),
    min_size=0,
    max_size=200,
)


@given(text=_SAFE_ALPHABET)
@settings(max_examples=200)
def test_no_pii_text_unchanged_in_redact_mode(text: str) -> None:
    """Any string with no PII patterns is returned unchanged in redact mode."""
    guardrail = PIIGuardrail(action="redact")
    assert guardrail.check_output(text) == text


@given(text=st.text(min_size=0, max_size=300))
@settings(max_examples=300)
def test_redact_never_contains_original_email(text: str) -> None:
    """After redaction, no original email address should remain in the output."""
    import re

    email_pattern = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
    emails_in_input = email_pattern.findall(text)

    guardrail = PIIGuardrail(entities=["email"], action="redact")
    result = guardrail.check_output(text)

    for email in emails_in_input:
        assert email not in result, f"Email {email!r} was not redacted from output"
