"""Guardrail registry: maps guardrail names to Guardrail instances."""

from __future__ import annotations

from sirenspec.guardrails.base import Guardrail

_DEFAULT_GUARDRAILS = ["injection"]
_KNOWN_GUARDRAILS = {"injection", "length"}


def build_guardrails(names: list[str] | None) -> list[Guardrail]:
    """Resolve a list of guardrail names to Guardrail instances.

    If *names* is ``None``, the default set (``['injection']``) is applied.
    An empty list disables all guardrails.

    :param names: Guardrail identifiers, or ``None`` to use defaults.
    :raises ValueError: If any name is not a known guardrail.
    :returns: List of configured Guardrail instances.
    """
    if names is None:
        names = list(_DEFAULT_GUARDRAILS)

    for name in names:
        if name not in _KNOWN_GUARDRAILS:
            raise ValueError(f"Unknown guardrail '{name}'; supported: {sorted(_KNOWN_GUARDRAILS)}")

    guardrails: list[Guardrail] = []
    for name in names:
        if name == "injection":
            from sirenspec.guardrails.injection import InjectionGuardrail

            guardrails.append(InjectionGuardrail())
        elif name == "length":
            from sirenspec.guardrails.length import LengthGuardrail

            guardrails.append(LengthGuardrail())

    return guardrails
