"""Provider registry: maps 'provider:model' URIs to LLMProvider instances."""

from __future__ import annotations

from sirenspec.providers.base import LLMProvider

_KNOWN_PROVIDERS = {"openai", "anthropic", "ollama"}


def resolve_provider(uri: str) -> LLMProvider:
    """Parse a *provider:model* URI and return a configured LLMProvider.

    :param uri: Provider URI in the form ``'provider:model'`` (e.g., ``'openai:gpt-4o-mini'``).
    :raises ValueError: If the URI is malformed or the provider is unknown.
    :returns: A configured LLMProvider instance.
    """
    if ":" not in uri:
        raise ValueError(f"Malformed provider URI '{uri}'; expected 'provider:model' format")

    provider_name, _, model = uri.partition(":")
    if not provider_name or not model:
        raise ValueError(f"Malformed provider URI '{uri}'; expected 'provider:model' format")

    if provider_name not in _KNOWN_PROVIDERS:
        raise ValueError(f"Unknown provider '{provider_name}'; supported: {sorted(_KNOWN_PROVIDERS)}")

    if provider_name == "openai":
        from sirenspec.providers.openai_provider import OpenAIProvider

        return OpenAIProvider(model=model)

    if provider_name == "anthropic":
        from sirenspec.providers.anthropic_provider import AnthropicProvider

        return AnthropicProvider(model=model)

    if provider_name == "ollama":
        from sirenspec.providers.ollama_provider import OllamaProvider

        return OllamaProvider(model=model)

    raise ValueError(f"Unknown provider '{provider_name}'")  # pragma: no cover
