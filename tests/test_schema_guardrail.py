"""Tests for the SchemaGuardrail and related registry integration."""

from __future__ import annotations

import json
from typing import Any

import pytest

from sirenspec.core.models import GuardrailSpec
from sirenspec.exceptions import GuardrailError, SirenSpecError
from sirenspec.guardrails.base import GuardrailViolation
from sirenspec.guardrails.registry import build_guardrails
from sirenspec.guardrails.schema import SchemaGuardrail

_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["intent", "confidence"],
    "properties": {
        "intent": {
            "type": "string",
            "enum": ["refund", "inquiry", "complaint"],
        },
        "confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
        },
    },
}

_VALID_OUTPUT = json.dumps({"intent": "refund", "confidence": 0.9})


class TestSchemaGuardrailCheckOutput:
    """Tests for SchemaGuardrail.check_output."""

    def test_valid_json_matching_schema_passes(self) -> None:
        """Valid JSON that conforms to the schema is returned unchanged."""
        guardrail = SchemaGuardrail(schema=_SCHEMA)
        result = guardrail.check_output(_VALID_OUTPUT)
        assert result == _VALID_OUTPUT
        assert isinstance(result, str)

    def test_missing_required_field_raises_violation(self) -> None:
        """Output missing a required field raises GuardrailViolation with the field path."""
        guardrail = SchemaGuardrail(schema=_SCHEMA)
        output = json.dumps({"intent": "refund"})  # missing 'confidence'
        with pytest.raises(GuardrailViolation) as exc_info:
            guardrail.check_output(output)
        assert "confidence" in str(exc_info.value)

    def test_wrong_type_raises_violation(self) -> None:
        """Output with the wrong type for a field raises GuardrailViolation."""
        guardrail = SchemaGuardrail(schema=_SCHEMA)
        output = json.dumps({"intent": "refund", "confidence": "high"})  # confidence should be number
        with pytest.raises(GuardrailViolation) as exc_info:
            guardrail.check_output(output)
        assert "confidence" in str(exc_info.value) or "high" in str(exc_info.value)

    def test_enum_violation_raises_violation(self) -> None:
        """Output with a value not in the enum raises GuardrailViolation."""
        guardrail = SchemaGuardrail(schema=_SCHEMA)
        output = json.dumps({"intent": "unknown", "confidence": 0.5})
        with pytest.raises(GuardrailViolation) as exc_info:
            guardrail.check_output(output)
        assert "intent" in str(exc_info.value) or "unknown" in str(exc_info.value)

    def test_non_json_output_raises_violation_with_parse_error(self) -> None:
        """Non-JSON output raises GuardrailViolation with a clear parse error message."""
        guardrail = SchemaGuardrail(schema=_SCHEMA)
        with pytest.raises(GuardrailViolation) as exc_info:
            guardrail.check_output("this is not json")
        assert "not valid JSON" in str(exc_info.value)
        assert exc_info.type is GuardrailViolation


class TestSchemaGuardrailCheckInput:
    """Tests for SchemaGuardrail.check_input."""

    def test_check_input_passes_through_any_string(self) -> None:
        """check_input returns the input string unchanged regardless of content."""
        guardrail = SchemaGuardrail(schema=_SCHEMA)
        text = "some arbitrary input {{ not json }}"
        result = guardrail.check_input(text)
        assert result == text
        assert isinstance(result, str)

    def test_check_input_passes_through_empty_string(self) -> None:
        """check_input returns an empty string unchanged."""
        guardrail = SchemaGuardrail(schema=_SCHEMA)
        result = guardrail.check_input("")
        assert result == ""
        assert isinstance(result, str)


class TestBuildGuardrailsSchemaIntegration:
    """Tests for build_guardrails with schema guardrail entries."""

    def test_build_guardrails_bare_schema_name_raises_value_error(self) -> None:
        """build_guardrails(['schema']) raises ValueError because no config is provided."""
        with pytest.raises(ValueError, match="config") as exc_info:
            build_guardrails(["schema"])
        assert "schema" in str(exc_info.value)

    def test_build_guardrails_with_guardrail_spec_returns_schema_guardrail(self) -> None:
        """build_guardrails with a GuardrailSpec returns a SchemaGuardrail instance."""
        spec = GuardrailSpec(name="schema", config={"schema": _SCHEMA})
        result = build_guardrails([spec])
        assert len(result) == 1
        assert isinstance(result[0], SchemaGuardrail)

    def test_build_guardrails_schema_spec_none_config_raises_value_error(self) -> None:
        """build_guardrails with GuardrailSpec(name='schema', config=None) raises ValueError."""
        spec = GuardrailSpec(name="schema", config=None)
        with pytest.raises(ValueError, match="config") as exc_info:
            build_guardrails([spec])
        assert "schema" in str(exc_info.value)

    def test_build_guardrails_schema_spec_missing_schema_key_raises_value_error(self) -> None:
        """build_guardrails with config dict lacking 'schema' key raises ValueError."""
        spec = GuardrailSpec(name="schema", config={"not_schema": {}})
        with pytest.raises(ValueError, match="schema") as exc_info:
            build_guardrails([spec])
        assert "config" in str(exc_info.value)


class TestGuardrailViolationHierarchy:
    """Tests for the GuardrailViolation exception hierarchy."""

    def test_guardrail_violation_is_guardrail_error(self) -> None:
        """GuardrailViolation is an instance of GuardrailError."""
        exc = GuardrailViolation("test reason")
        assert isinstance(exc, GuardrailError)
        assert isinstance(exc, SirenSpecError)

    def test_guardrail_violation_is_siren_spec_error(self) -> None:
        """GuardrailViolation is an instance of SirenSpecError."""
        exc = GuardrailViolation("test reason")
        assert isinstance(exc, SirenSpecError)
        assert isinstance(exc, GuardrailError)

    def test_guardrail_violation_raised_by_schema_guardrail_is_guardrail_error(self) -> None:
        """GuardrailViolation raised by SchemaGuardrail is catchable as GuardrailError."""
        guardrail = SchemaGuardrail(schema=_SCHEMA)
        with pytest.raises(GuardrailError) as exc_info:
            guardrail.check_output("not json")
        assert exc_info.type is GuardrailViolation
