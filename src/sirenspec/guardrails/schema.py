"""JSON Schema output validation guardrail."""

from __future__ import annotations

import json
from typing import Any

import jsonschema

from sirenspec.guardrails.base import Guardrail, GuardrailViolation


def parse_json_output(text: str) -> Any:
    """Attempt to parse *text* as JSON.

    :param text: The raw text to parse.
    :raises GuardrailViolation: If the text is not valid JSON.
    :returns: The parsed Python object.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise GuardrailViolation(f"Output is not valid JSON: {exc}") from exc


def validate_against_schema(data: Any, schema: dict[str, Any]) -> None:
    """Validate *data* against *schema* using JSON Schema Draft 7.

    :param data: The parsed Python object to validate.
    :param schema: A JSON Schema Draft 7 dict.
    :raises GuardrailViolation: If *data* does not conform to *schema*, with a message
        that includes the JSON Pointer path and the violated constraint.
    """
    validator = jsonschema.Draft7Validator(schema)
    errors = sorted(validator.iter_errors(data), key=lambda e: e.path)
    if not errors:
        return
    first = errors[0]
    if first.absolute_path:
        path = "$." + ".".join(str(p) for p in first.absolute_path)
    else:
        path = "$"
    raise GuardrailViolation(f"Schema violation at {path!r}: {first.message}")


class SchemaGuardrail(Guardrail):
    """Validates LLM output against a JSON Schema Draft 7 definition.

    The guardrail parses the output text as JSON and checks it against the
    provided schema.  Input text is passed through unchanged — schema validation
    is output-only.

    :param schema: A JSON Schema Draft 7 dict.  Required; there is no default schema.
    """

    def __init__(self, schema: dict[str, Any]) -> None:
        self.schema = schema

    def check_input(self, text: str) -> str:
        """Pass input text through unchanged — schema validation is output-only.

        :param text: Input text to check.
        :returns: The original text, unmodified.
        """
        return text

    def check_output(self, text: str) -> str:
        """Parse *text* as JSON and validate it against the guardrail's schema.

        :param text: Output text to validate.
        :raises GuardrailViolation: If the text is not valid JSON or does not
            conform to the schema.
        :returns: The original text if validation passes.
        """
        data = parse_json_output(text)
        validate_against_schema(data, self.schema)
        return text
