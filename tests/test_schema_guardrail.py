"""Unit and property-based tests for SchemaGuardrail."""

from __future__ import annotations

import json

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from sirenspec.guardrails.base import GuardrailViolation
from sirenspec.guardrails.schema import SchemaGuardrail, parse_json, validate_against_schema

# ---------------------------------------------------------------------------
# Module-level helper function tests
# ---------------------------------------------------------------------------


class TestParseJson:
    def test_valid_object(self) -> None:
        result = parse_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_valid_array(self) -> None:
        result = parse_json("[1, 2, 3]")
        assert result == [1, 2, 3]

    def test_valid_string(self) -> None:
        result = parse_json('"hello"')
        assert result == "hello"

    def test_valid_number(self) -> None:
        result = parse_json("42")
        assert result == 42

    def test_valid_null(self) -> None:
        result = parse_json("null")
        assert result is None

    def test_valid_boolean(self) -> None:
        assert parse_json("true") is True
        assert parse_json("false") is False

    def test_invalid_json_raises(self) -> None:
        with pytest.raises(GuardrailViolation, match="not valid JSON"):
            parse_json("not json at all")

    def test_truncated_json_raises(self) -> None:
        with pytest.raises(GuardrailViolation, match="not valid JSON"):
            parse_json('{"unclosed": ')

    def test_violation_has_reason(self) -> None:
        with pytest.raises(GuardrailViolation) as exc_info:
            parse_json("bad")
        assert "not valid JSON" in exc_info.value.reason


class TestValidateAgainstSchema:
    def test_valid_object_passes(self) -> None:
        schema = {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}
        validate_against_schema({"name": "Alice"}, schema)  # should not raise

    def test_missing_required_raises(self) -> None:
        schema = {"type": "object", "required": ["name"]}
        with pytest.raises(GuardrailViolation, match="JSON Schema validation"):
            validate_against_schema({}, schema)

    def test_wrong_type_raises(self) -> None:
        schema = {"type": "string"}
        with pytest.raises(GuardrailViolation, match="JSON Schema validation"):
            validate_against_schema(123, schema)

    def test_violation_has_reason(self) -> None:
        schema = {"type": "object", "required": ["x"]}
        with pytest.raises(GuardrailViolation) as exc_info:
            validate_against_schema({}, schema)
        assert "JSON Schema validation" in exc_info.value.reason

    def test_empty_schema_accepts_anything(self) -> None:
        validate_against_schema({"anything": True}, {})
        validate_against_schema([1, 2], {})
        validate_against_schema("text", {})


# ---------------------------------------------------------------------------
# SchemaGuardrail class tests
# ---------------------------------------------------------------------------


class TestSchemaGuardrailCheckInput:
    def test_passes_through_unchanged(self) -> None:
        g = SchemaGuardrail(schema={"type": "object"})
        text = "some arbitrary input text"
        assert g.check_input(text) == text

    def test_does_not_validate_input_as_json(self) -> None:
        g = SchemaGuardrail(schema={"type": "object"})
        # Non-JSON input must not raise — input is never validated
        assert g.check_input("not json") == "not json"


class TestSchemaGuardrailCheckOutputValid:
    @pytest.fixture
    def g(self) -> SchemaGuardrail:
        schema = {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "confidence": {"type": "number"},
            },
            "required": ["answer", "confidence"],
            "additionalProperties": False,
        }
        return SchemaGuardrail(schema=schema)

    def test_valid_output_returns_original_text(self, g: SchemaGuardrail) -> None:
        text = '{"answer": "Paris", "confidence": 0.99}'
        assert g.check_output(text) == text

    def test_valid_output_with_whitespace(self, g: SchemaGuardrail) -> None:
        text = json.dumps({"answer": "London", "confidence": 0.8}, indent=2)
        assert g.check_output(text) == text


class TestSchemaGuardrailCheckOutputInvalid:
    @pytest.fixture
    def g(self) -> SchemaGuardrail:
        return SchemaGuardrail(schema={"type": "object", "required": ["result"]})

    def test_non_json_raises(self, g: SchemaGuardrail) -> None:
        with pytest.raises(GuardrailViolation, match="not valid JSON"):
            g.check_output("plain text answer")

    def test_schema_violation_raises(self, g: SchemaGuardrail) -> None:
        with pytest.raises(GuardrailViolation, match="JSON Schema validation"):
            g.check_output('{"wrong_key": "value"}')

    def test_wrong_root_type_raises(self, g: SchemaGuardrail) -> None:
        with pytest.raises(GuardrailViolation, match="JSON Schema validation"):
            g.check_output('"just a string"')

    def test_violation_exposes_reason(self, g: SchemaGuardrail) -> None:
        with pytest.raises(GuardrailViolation) as exc_info:
            g.check_output("{}")
        assert exc_info.value.reason


class TestSchemaGuardrailStoresSchema:
    def test_schema_is_accessible(self) -> None:
        schema = {"type": "array"}
        g = SchemaGuardrail(schema=schema)
        assert g.schema is schema


class TestSchemaGuardrailArraySchema:
    def test_valid_array_passes(self) -> None:
        g = SchemaGuardrail(schema={"type": "array", "items": {"type": "integer"}})
        text = "[1, 2, 3]"
        assert g.check_output(text) == text

    def test_array_with_wrong_items_raises(self) -> None:
        g = SchemaGuardrail(schema={"type": "array", "items": {"type": "integer"}})
        with pytest.raises(GuardrailViolation, match="JSON Schema validation"):
            g.check_output('["not", "integers"]')


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------


class TestSchemaGuardrailRegistry:
    def test_schema_name_is_registered(self) -> None:
        from sirenspec.guardrails.registry import build_guardrails

        guardrails = build_guardrails(["schema"])
        assert len(guardrails) == 1
        assert isinstance(guardrails[0], SchemaGuardrail)

    def test_registry_schema_guardrail_has_empty_schema(self) -> None:
        from sirenspec.guardrails.registry import build_guardrails

        guardrail = build_guardrails(["schema"])[0]
        assert isinstance(guardrail, SchemaGuardrail)
        assert guardrail.schema == {}

    def test_registry_schema_guardrail_accepts_any_valid_json(self) -> None:
        from sirenspec.guardrails.registry import build_guardrails

        guardrail = build_guardrails(["schema"])[0]
        assert guardrail.check_output('{"anything": true}') == '{"anything": true}'

    def test_registry_schema_guardrail_rejects_non_json(self) -> None:
        from sirenspec.guardrails.registry import build_guardrails

        guardrail = build_guardrails(["schema"])[0]
        with pytest.raises(GuardrailViolation, match="not valid JSON"):
            guardrail.check_output("plain text")


# ---------------------------------------------------------------------------
# Property-based tests
# ---------------------------------------------------------------------------


@given(st.text())
@settings(max_examples=200)
def test_parse_json_never_returns_non_json_silently(text: str) -> None:
    """parse_json either returns a valid Python value or raises GuardrailViolation."""
    try:
        result = parse_json(text)
        # Round-trip: re-serialising and re-parsing must be stable
        assert json.loads(json.dumps(result)) == result
    except GuardrailViolation:
        pass  # expected for non-JSON input


@given(st.text())
@settings(max_examples=200)
def test_check_input_always_returns_same_text(text: str) -> None:
    """check_input is always a no-op regardless of content."""
    g = SchemaGuardrail(schema={"type": "object"})
    assert g.check_input(text) == text


@given(st.dictionaries(st.text(min_size=1), st.integers()))
@settings(max_examples=100)
def test_valid_object_passes_object_schema(data: dict[str, int]) -> None:
    """Any dict of str->int passes an object schema with no additional constraints."""
    g = SchemaGuardrail(schema={"type": "object"})
    text = json.dumps(data)
    assert g.check_output(text) == text
