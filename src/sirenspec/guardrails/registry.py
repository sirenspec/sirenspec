"""Guardrail registry: maps guardrail names to Guardrail instances."""

from __future__ import annotations

from collections.abc import Callable

from sirenspec.core.models import GuardrailSpec
from sirenspec.guardrails.base import Guardrail
from sirenspec.guardrails.cost_cap import CostCapGuardrail
from sirenspec.guardrails.injection import InjectionGuardrail
from sirenspec.guardrails.length import LengthGuardrail
from sirenspec.guardrails.pii import PIIGuardrail
from sirenspec.guardrails.schema import SchemaGuardrail

# Names of guardrails that are applied when a node does not configure guardrails
# explicitly (i.e. when guardrail_names is None, not []).
_DEFAULT_GUARDRAILS = ["injection"]


def make_injection_guardrail(config: dict | None = None) -> Guardrail:
    """Instantiate an InjectionGuardrail.

    :param config: Unused; accepted for interface uniformity.
    :returns: Configured :class:`~sirenspec.guardrails.injection.InjectionGuardrail`.
    """
    return InjectionGuardrail()


def make_length_guardrail(config: dict | None = None) -> Guardrail:
    """Instantiate a LengthGuardrail.

    :param config: Unused; accepted for interface uniformity.
    :returns: Configured :class:`~sirenspec.guardrails.length.LengthGuardrail`.
    """
    return LengthGuardrail()


def make_schema_guardrail(config: dict | None) -> Guardrail:
    """Instantiate a SchemaGuardrail from a config dict containing a ``schema`` key.

    :param config: Must be a dict with a ``schema`` key containing a JSON Schema dict.
    :raises ValueError: If ``config`` is ``None`` or does not contain a ``schema`` key.
    :returns: Configured :class:`~sirenspec.guardrails.schema.SchemaGuardrail`.
    """
    if config is None or "schema" not in config:
        raise ValueError("schema guardrail requires config with a 'schema' key")
    return SchemaGuardrail(schema=config["schema"])


_KNOWN_PII_ENTITIES = {"email", "phone", "ssn", "credit_card"}


def make_pii_guardrail(config: dict | None = None) -> Guardrail:
    """Instantiate a PIIGuardrail from an optional config dict.

    :param config: Optional dict with keys ``entities``, ``action``, and/or ``replacement``.
        If ``None``, all defaults are used (all entities, redact action, ``[REDACTED]``).
    :raises ValueError: If any entity name in ``config['entities']`` is not a known PII entity.
    :returns: Configured :class:`~sirenspec.guardrails.pii.PIIGuardrail`.
    """
    if config is None:
        return PIIGuardrail()

    entities = config.get("entities", None)
    action = config.get("action", "redact")
    replacement = config.get("replacement", "[REDACTED]")

    if entities is not None:
        unknown = set(entities) - _KNOWN_PII_ENTITIES
        if unknown:
            raise ValueError(
                f"Unknown PII entity type(s) {sorted(unknown)!r}; supported: {sorted(_KNOWN_PII_ENTITIES)}"
            )

    return PIIGuardrail(entities=entities, action=action, replacement=replacement)


def make_cost_cap_guardrail(config: dict | None) -> Guardrail:
    """Instantiate a CostCapGuardrail from a config dict.

    :param config: Must be a dict with at least one of ``max_usd`` or ``max_tokens``.
        Optional key ``action`` controls behaviour (``'abort'`` or ``'warn'``).
    :raises ValueError: If ``config`` is ``None`` or lacks both ``max_usd`` and ``max_tokens``.
    :returns: Configured :class:`~sirenspec.guardrails.cost_cap.CostCapGuardrail`.
    """
    if config is None:
        raise ValueError("cost_cap guardrail requires a config dict with 'max_usd' and/or 'max_tokens'")
    return CostCapGuardrail(
        max_usd=config.get("max_usd"),
        max_tokens=config.get("max_tokens"),
        action=config.get("action", "abort"),
    )


# Maps guardrail names (as used in YAML) to factory functions.
# To add a new guardrail, add one factory function above and one entry here.
_GUARDRAIL_FACTORIES: dict[str, Callable[[dict | None], Guardrail]] = {
    "cost_cap": make_cost_cap_guardrail,
    "injection": make_injection_guardrail,
    "length": make_length_guardrail,
    "pii": make_pii_guardrail,
    "schema": make_schema_guardrail,
}


def resolve_guardrail_name_and_config(entry: str | object) -> tuple[str, dict | None]:
    """Extract the guardrail name and optional config from a string or GuardrailSpec entry.

    :param entry: Either a plain string guardrail name or a GuardrailSpec instance.
    :returns: A ``(name, config)`` tuple.
    """
    if isinstance(entry, GuardrailSpec):
        return entry.name, entry.config
    return str(entry), None


def build_guardrails(names: list[str | object] | None) -> list[Guardrail]:
    """Resolve a list of guardrail names (or GuardrailSpec objects) to Guardrail instances.

    There is an important distinction between ``None`` and ``[]``:

    * ``None`` — the caller didn't specify guardrails; apply the defaults (injection detection).
    * ``[]`` — the caller explicitly disabled all guardrails.

    This distinction is preserved throughout the stack: executor.py and swrm_runner.py
    pass ``None`` through unchanged rather than coercing it to ``[]``.

    :param names: Guardrail identifiers or :class:`~sirenspec.core.models.GuardrailSpec`
        objects, or ``None`` to use defaults.
    :raises ValueError: If any name is not a known guardrail or a required config is absent.
    :returns: List of configured Guardrail instances.
    """
    if names is None:
        # list() makes a copy so the caller cannot mutate _DEFAULT_GUARDRAILS.
        names = list(_DEFAULT_GUARDRAILS)

    resolved = [resolve_guardrail_name_and_config(entry) for entry in names]

    # Collect all unknown names in a single pass before raising, so the error
    # message reports every typo at once instead of one at a time.
    unknown = [name for name, _ in resolved if name not in _GUARDRAIL_FACTORIES]
    if unknown:
        raise ValueError(f"Unknown guardrail(s) {unknown!r}; supported: {sorted(_GUARDRAIL_FACTORIES)}")

    return [_GUARDRAIL_FACTORIES[name](config) for name, config in resolved]
