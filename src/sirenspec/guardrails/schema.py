"""JSON Schema output validation guardrail."""

from __future__ import annotations

import json
from typing import Any

import jsonschema

from sirenspec.guardrails.base import Guardrail, GuardrailViolation


def parse_json(text: str) -> Any:
    """Parse *text* as JSON, raising GuardrailViolation on parse failure.

    :param text: Raw text to parse.
    :raises GuardrailViolation: If *text* is not valid JSON.
    :returns: The parsed JSON value.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise GuardrailViolation(f"Output is not valid JSON: {exc}") from exc


def validate_against_schema(value: Any, schema: dict[str, Any]) -> None:
    """Validate *value* against *schema* using jsonschema.

    :param value: The parsed JSON value to validate.
    :param schema: A JSON Schema dict.
    :raises GuardrailViolation: If *value* does not conform to *schema*.
    """
    try:
        jsonschema.validate(instance=value, schema=schema)
    except jsonschema.ValidationError as exc:
        raise GuardrailViolation(f"Output failed JSON Schema validation: {exc.message}") from exc


class SchemaGuardrail(Guardrail):
    """Validates LLM output against a JSON Schema.

    The output text must be parseable as JSON and must conform to the provided
    schema. Input text is passed through without modification.

    :param schema: A JSON Schema dict describing the expected output structure.
    """

    def __init__(self, schema: dict[str, Any]) -> None:
        self.schema = schema

    def check_input(self, text: str) -> str:
        """Pass input through unchanged; schema validation applies only to output.

        :param text: Input text.
        :returns: The original text unchanged.
        """
        return text

    def check_output(self, text: str) -> str:
        """Parse *text* as JSON and validate it against the configured schema.

        :param text: LLM output text to validate.
        :raises GuardrailViolation: If *text* is not valid JSON or does not conform to the schema.
        :returns: The original text if validation passes.
        """
        value = parse_json(text)
        validate_against_schema(value, self.schema)
        return text
