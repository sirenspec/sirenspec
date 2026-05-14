"""Unit tests for guardrail registry."""

from __future__ import annotations

import pytest

from sirenspec.guardrails.injection import InjectionGuardrail
from sirenspec.guardrails.length import LengthGuardrail
from sirenspec.guardrails.registry import build_guardrails


class TestBuildGuardrails:
    def test_none_returns_defaults(self) -> None:
        guardrails = build_guardrails(None)
        assert len(guardrails) == 1
        assert isinstance(guardrails[0], InjectionGuardrail)

    def test_empty_list_returns_none(self) -> None:
        guardrails = build_guardrails([])
        assert guardrails == []

    def test_injection_only(self) -> None:
        guardrails = build_guardrails(["injection"])
        assert len(guardrails) == 1
        assert isinstance(guardrails[0], InjectionGuardrail)

    def test_length_only(self) -> None:
        guardrails = build_guardrails(["length"])
        assert len(guardrails) == 1
        assert isinstance(guardrails[0], LengthGuardrail)

    def test_both_guardrails(self) -> None:
        guardrails = build_guardrails(["injection", "length"])
        assert len(guardrails) == 2
        assert isinstance(guardrails[0], InjectionGuardrail)
        assert isinstance(guardrails[1], LengthGuardrail)

    def test_unknown_guardrail_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown guardrail"):
            build_guardrails(["pii"])

    def test_unknown_in_list_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown guardrail"):
            build_guardrails(["injection", "bogus"])
