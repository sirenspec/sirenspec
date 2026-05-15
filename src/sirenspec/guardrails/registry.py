"""Guardrail registry: maps guardrail names to Guardrail instances."""

from __future__ import annotations

from collections.abc import Callable

from sirenspec.guardrails.base import Guardrail

_DEFAULT_GUARDRAILS = ["injection"]


def make_injection_guardrail() -> Guardrail:
    """Instantiate an InjectionGuardrail.

    :returns: Configured :class:`~sirenspec.guardrails.injection.InjectionGuardrail`.
    """
    from sirenspec.guardrails.injection import InjectionGuardrail

    return InjectionGuardrail()


def make_length_guardrail() -> Guardrail:
    """Instantiate a LengthGuardrail.

    :returns: Configured :class:`~sirenspec.guardrails.length.LengthGuardrail`.
    """
    from sirenspec.guardrails.length import LengthGuardrail

    return LengthGuardrail()


_GUARDRAIL_FACTORIES: dict[str, Callable[[], Guardrail]] = {
    "injection": make_injection_guardrail,
    "length": make_length_guardrail,
}


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

    unknown = [n for n in names if n not in _GUARDRAIL_FACTORIES]
    if unknown:
        raise ValueError(f"Unknown guardrail(s) {unknown!r}; supported: {sorted(_GUARDRAIL_FACTORIES)}")

    return [_GUARDRAIL_FACTORIES[name]() for name in names]
