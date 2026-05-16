"""Provider registry: maps 'provider:model' URIs to LLMProvider instances."""

from __future__ import annotations

from collections.abc import Callable

from sirenspec.exceptions import ProviderError
from sirenspec.providers.base import LLMProvider


def make_openai_provider(model: str) -> LLMProvider:
    """Instantiate an OpenAIProvider for the given model.

    The import is inside the function so that the ``openai`` SDK is only loaded
    when this provider is actually used — avoiding a startup-time import error
    if the package is not installed.

    :param model: Model identifier (e.g. ``'gpt-4o-mini'``).
    :returns: Configured :class:`~sirenspec.providers.openai_provider.OpenAIProvider`.
    """
    from sirenspec.providers.openai_provider import OpenAIProvider

    return OpenAIProvider(model=model)


def make_anthropic_provider(model: str) -> LLMProvider:
    """Instantiate an AnthropicProvider for the given model.

    The import is inside the function so that the ``anthropic`` SDK is only loaded
    when this provider is actually used.

    :param model: Model identifier (e.g. ``'claude-haiku-4-5-20251001'``).
    :returns: Configured :class:`~sirenspec.providers.anthropic_provider.AnthropicProvider`.
    """
    from sirenspec.providers.anthropic_provider import AnthropicProvider

    return AnthropicProvider(model=model)


def make_ollama_provider(model: str) -> LLMProvider:
    """Instantiate an OllamaProvider for the given model.

    The import is inside the function so that the ``ollama`` SDK is only loaded
    when this provider is actually used.

    :param model: Model identifier (e.g. ``'llama3'``).
    :returns: Configured :class:`~sirenspec.providers.ollama_provider.OllamaProvider`.
    """
    from sirenspec.providers.ollama_provider import OllamaProvider

    return OllamaProvider(model=model)


# Maps provider names (the part before ':' in a URI) to factory functions.
# To add a new provider, add one factory function above and one entry here.
_PROVIDER_FACTORIES: dict[str, Callable[[str], LLMProvider]] = {
    "openai": make_openai_provider,
    "anthropic": make_anthropic_provider,
    "ollama": make_ollama_provider,
}


def resolve_provider(uri: str) -> LLMProvider:
    """Parse a *provider:model* URI and return a configured LLMProvider.

    :param uri: Provider URI in the form ``'provider:model'`` (e.g., ``'openai:gpt-4o-mini'``).
    :raises ProviderError: If the URI is malformed or the provider name is not registered.
    :returns: A configured LLMProvider instance.
    """
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
