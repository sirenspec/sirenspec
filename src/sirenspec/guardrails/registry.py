"""Guardrail registry: maps guardrail names to Guardrail instances."""

from __future__ import annotations

from collections.abc import Callable

from sirenspec.guardrails.base import Guardrail
from sirenspec.guardrails.injection import InjectionGuardrail
from sirenspec.guardrails.length import LengthGuardrail
from sirenspec.guardrails.schema import SchemaGuardrail

# Names of guardrails that are applied when a node does not configure guardrails
# explicitly (i.e. when guardrail_names is None, not []).
_DEFAULT_GUARDRAILS = ["injection"]


def make_injection_guardrail() -> Guardrail:
    """Instantiate an InjectionGuardrail.

    :returns: Configured :class:`~sirenspec.guardrails.injection.InjectionGuardrail`.
    """
    return InjectionGuardrail()


def make_length_guardrail() -> Guardrail:
    """Instantiate a LengthGuardrail.

    :returns: Configured :class:`~sirenspec.guardrails.length.LengthGuardrail`.
    """
    return LengthGuardrail()


def make_schema_guardrail() -> Guardrail:
    """Instantiate a SchemaGuardrail with an empty (pass-all) schema.

    When used via the registry by name only, the schema accepts any valid JSON.
    For schema-constrained validation, instantiate :class:`~sirenspec.guardrails.schema.SchemaGuardrail`
    directly with the desired schema dict.

    :returns: Configured :class:`~sirenspec.guardrails.schema.SchemaGuardrail`.
    """
    return SchemaGuardrail(schema={})


# Maps guardrail names (as used in YAML) to factory functions.
# To add a new guardrail, add one factory function above and one entry here.
_GUARDRAIL_FACTORIES: dict[str, Callable[[], Guardrail]] = {
    "injection": make_injection_guardrail,
    "length": make_length_guardrail,
    "schema": make_schema_guardrail,
}


def build_guardrails(names: list[str] | None) -> list[Guardrail]:
    """Resolve a list of guardrail names to Guardrail instances.

    There is an important distinction between ``None`` and ``[]``:

    * ``None`` — the caller didn't specify guardrails; apply the defaults (injection detection).
    * ``[]`` — the caller explicitly disabled all guardrails.

    This distinction is preserved throughout the stack: executor.py and swrm_runner.py
    pass ``None`` through unchanged rather than coercing it to ``[]``.

    :param names: Guardrail identifiers, or ``None`` to use defaults.
    :raises ValueError: If any name is not a known guardrail.
    :returns: List of configured Guardrail instances.
    """
    if names is None:
        # list() makes a copy so the caller cannot mutate _DEFAULT_GUARDRAILS.
        names = list(_DEFAULT_GUARDRAILS)

    # Collect all unknown names in a single pass before raising, so the error
    # message reports every typo at once instead of one at a time.
    unknown = [n for n in names if n not in _GUARDRAIL_FACTORIES]
    if unknown:
        raise ValueError(f"Unknown guardrail(s) {unknown!r}; supported: {sorted(_GUARDRAIL_FACTORIES)}")

    return [_GUARDRAIL_FACTORIES[name]() for name in names]
