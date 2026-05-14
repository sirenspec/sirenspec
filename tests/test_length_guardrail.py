"""Unit tests for LengthGuardrail."""

from __future__ import annotations

import pytest

from sirenspec.guardrails.base import GuardrailViolation
from sirenspec.guardrails.length import LengthGuardrail


class TestLengthGuardrailTruncate:
    @pytest.fixture
    def g(self) -> LengthGuardrail:
        return LengthGuardrail(max_chars=10, mode="truncate")

    def test_short_text_passes(self, g: LengthGuardrail) -> None:
        assert g.check_output("hello") == "hello"

    def test_exact_limit_passes(self, g: LengthGuardrail) -> None:
        assert g.check_output("1234567890") == "1234567890"

    def test_over_limit_truncates(self, g: LengthGuardrail) -> None:
        result = g.check_output("12345678901234")
        assert result == "1234567890..."

    def test_truncated_length(self, g: LengthGuardrail) -> None:
        result = g.check_output("x" * 100)
        assert len(result) == 10 + 3  # max_chars + len("...")

    def test_check_input_passes_through(self, g: LengthGuardrail) -> None:
        long_text = "a" * 10000
        assert g.check_input(long_text) == long_text


class TestLengthGuardrailRaise:
    @pytest.fixture
    def g(self) -> LengthGuardrail:
        return LengthGuardrail(max_chars=10, mode="raise")

    def test_short_text_passes(self, g: LengthGuardrail) -> None:
        assert g.check_output("hello") == "hello"

    def test_over_limit_raises(self, g: LengthGuardrail) -> None:
        with pytest.raises(GuardrailViolation, match="max_chars"):
            g.check_output("12345678901")

    def test_violation_has_reason(self, g: LengthGuardrail) -> None:
        with pytest.raises(GuardrailViolation) as exc_info:
            g.check_output("x" * 100)
        assert "max_chars" in exc_info.value.reason


class TestLengthGuardrailDefaults:
    def test_default_max_chars(self) -> None:
        g = LengthGuardrail()
        assert g.max_chars == 4000
        assert g.mode == "truncate"

    def test_invalid_mode_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid mode"):
            LengthGuardrail(mode="skip")
