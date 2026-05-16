"""Unit tests for provider registry."""

from __future__ import annotations

import pytest

from sirenspec.exceptions import ProviderError
from sirenspec.providers.registry import resolve_provider


class TestResolveProvider:
    def test_openai_provider(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-dummy")
        provider = resolve_provider("openai:gpt-4o-mini")
        from sirenspec.providers.openai_provider import OpenAIProvider

        assert isinstance(provider, OpenAIProvider)
        assert provider.model == "gpt-4o-mini"

    def test_anthropic_provider(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-dummy")
        provider = resolve_provider("anthropic:claude-haiku-4-5-20251001")
        from sirenspec.providers.anthropic_provider import AnthropicProvider

        assert isinstance(provider, AnthropicProvider)
        assert provider.model == "claude-haiku-4-5-20251001"

    def test_unknown_provider_raises(self) -> None:
        with pytest.raises(ProviderError, match="Unknown provider"):
            resolve_provider("vertex:gemini-pro")

    def test_malformed_no_colon_raises(self) -> None:
        with pytest.raises(ProviderError, match="Malformed"):
            resolve_provider("openai-gpt4")

    def test_malformed_empty_model_raises(self) -> None:
        with pytest.raises(ProviderError, match="Malformed"):
            resolve_provider("openai:")

    def test_malformed_empty_provider_raises(self) -> None:
        with pytest.raises(ProviderError, match="Malformed"):
            resolve_provider(":gpt-4o")
