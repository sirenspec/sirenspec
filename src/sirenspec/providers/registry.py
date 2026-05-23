"""Provider registry: maps 'provider:model' URIs to LLMProvider instances."""

from __future__ import annotations

from collections.abc import Callable

from sirenspec.exceptions import ProviderError
from sirenspec.providers.anthropic_provider import AnthropicProvider
from sirenspec.providers.base import LLMProvider
from sirenspec.providers.ollama_provider import OllamaProvider
from sirenspec.providers.openai_provider import OpenAIProvider


def make_openai_provider(model: str) -> LLMProvider:
    """Instantiate an OpenAIProvider for the given model.

    :param model: Model identifier (e.g. ``'gpt-4o-mini'``).
    :returns: Configured :class:`~sirenspec.providers.openai_provider.OpenAIProvider`.
    """
    return OpenAIProvider(model=model)


def make_anthropic_provider(model: str) -> LLMProvider:
    """Instantiate an AnthropicProvider for the given model.

    :param model: Model identifier (e.g. ``'claude-haiku-4-5-20251001'``).
    :returns: Configured :class:`~sirenspec.providers.anthropic_provider.AnthropicProvider`.
    """
    return AnthropicProvider(model=model)


def make_ollama_provider(model: str) -> LLMProvider:
    """Instantiate an OllamaProvider for the given model.

    :param model: Model identifier (e.g. ``'llama3'``).
    :returns: Configured :class:`~sirenspec.providers.ollama_provider.OllamaProvider`.
    """
    return OllamaProvider(model=model)


# Maps provider names (the part before ':' in a URI) to factory functions.
# To add a new provider, add one factory function above and one entry here.
_PROVIDER_FACTORIES: dict[str, Callable[[str], LLMProvider]] = {
    "openai": make_openai_provider,
    "anthropic": make_anthropic_provider,
    "ollama": make_ollama_provider,
}

# When set, all resolve_provider calls return the result of this function instead
# of the normal factory lookup. Used by the test framework to inject mock providers.
_provider_override: Callable[[str], LLMProvider] | None = None


def set_provider_override(factory: Callable[[str], LLMProvider] | None) -> None:
    """Install or clear a global provider override for testing.

    When *factory* is not ``None``, every subsequent :func:`resolve_provider`
    call returns ``factory(uri)`` instead of the real provider.  Pass ``None``
    to restore normal dispatch.

    :param factory: A callable ``(uri: str) -> LLMProvider``, or ``None`` to clear.
    """
    global _provider_override
    _provider_override = factory


def resolve_provider(uri: str) -> LLMProvider:
    """Parse a *provider:model* URI and return a configured LLMProvider.

    :param uri: Provider URI in the form ``'provider:model'`` (e.g., ``'openai:gpt-4o-mini'``).
    :raises ProviderError: If the URI is malformed or the provider name is not registered.
    :returns: A configured LLMProvider instance.
    """
    if _provider_override is not None:
        return _provider_override(uri)

    if ":" not in uri:
        raise ProviderError(f"Malformed provider URI '{uri}'; expected 'provider:model' format")

    # partition returns exactly 3 parts and never raises, unlike split which
    # would require extra handling when the delimiter is absent.
    provider_name, _, model = uri.partition(":")
    if not provider_name or not model:
        raise ProviderError(f"Malformed provider URI '{uri}'; expected 'provider:model' format")

    factory = _PROVIDER_FACTORIES.get(provider_name)
    if factory is None:
        raise ProviderError(f"Unknown provider '{provider_name}'; supported: {sorted(_PROVIDER_FACTORIES)}")

    return factory(model)
