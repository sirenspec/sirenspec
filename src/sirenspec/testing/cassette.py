"""VCR-style cassette: record and replay LLM provider responses."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML

from sirenspec.core.usage import TokenUsage
from sirenspec.exceptions import SirenSpecError
from sirenspec.providers.base import LLMProvider


class CassetteError(SirenSpecError):
    """Raised when a cassette cannot be loaded or a replay key is missing."""


@dataclass
class CassetteInteraction:
    """A single recorded provider interaction.

    :param key: Stable hash of ``(uri, messages)`` used for lookup during replay.
    :param response: The text response returned by the provider.
    :param prompt_tokens: Token count for the prompt.
    :param completion_tokens: Token count for the completion.
    """

    key: str
    response: str
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class Cassette:
    """An ordered collection of :class:`CassetteInteraction` records.

    :param interactions: All recorded interactions; order matches invocation order.
    """

    interactions: list[CassetteInteraction] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        """Serialise the cassette to a plain dict for YAML output.

        :returns: Dict representation suitable for ``ruamel.yaml`` serialisation.
        """
        return {
            "interactions": [
                {
                    "key": i.key,
                    "response": i.response,
                    "tokens": {"prompt": i.prompt_tokens, "completion": i.completion_tokens},
                }
                for i in self.interactions
            ]
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Cassette:
        """Deserialise a cassette from a plain dict (loaded from YAML).

        :param data: Raw dict with an ``interactions`` key.
        :raises CassetteError: If the data is malformed.
        :returns: Populated :class:`Cassette`.
        """
        if not isinstance(data, dict) or "interactions" not in data:
            raise CassetteError("Cassette file is missing an 'interactions' key")
        interactions = []
        for raw in data["interactions"]:
            tokens = raw.get("tokens", {})
            interactions.append(
                CassetteInteraction(
                    key=raw["key"],
                    response=raw["response"],
                    prompt_tokens=tokens.get("prompt", 0),
                    completion_tokens=tokens.get("completion", 0),
                )
            )
        return cls(interactions=interactions)


def interaction_key(uri: str, messages: list[dict]) -> str:
    """Compute a stable hash key for a (uri, messages) pair.

    :param uri: Provider URI (e.g. ``'openai:gpt-4o-mini'``).
    :param messages: OpenAI-style message list.
    :returns: A 16-character hex digest string.
    """
    payload = json.dumps({"uri": uri, "messages": messages}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def load_cassette(path: Path) -> Cassette:
    """Load a cassette YAML file from *path*.

    :param path: Path to the cassette YAML file.
    :raises CassetteError: If the file is missing or malformed.
    :returns: Populated :class:`Cassette`.
    """
    if not path.exists():
        raise CassetteError(f"Cassette file not found: {path}")
    yaml = YAML(typ="safe")
    try:
        data: Any = yaml.load(path)
    except Exception as exc:
        raise CassetteError(f"Failed to parse cassette '{path}': {exc}") from exc
    return Cassette.from_dict(data)


def save_cassette(cassette: Cassette, path: Path) -> None:
    """Write *cassette* to *path* as YAML, creating parent directories if needed.

    :param cassette: The cassette to serialise.
    :param path: Destination file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    yaml = YAML()
    yaml.default_flow_style = False
    with path.open("w", encoding="utf-8") as fh:
        yaml.dump(cassette.as_dict(), fh)


class ReplayProvider:
    """An :class:`~sirenspec.providers.base.LLMProvider` that replays from a cassette.

    Interactions are matched by the hash of ``(uri, messages)``.  If a matching
    interaction is not found, a :class:`CassetteError` is raised so the test
    fails loudly rather than silently falling through to a live API call.

    :param uri: The provider URI this instance was created for (used as the hash key).
    :param cassette: The cassette to replay from.
    """

    def __init__(self, uri: str, cassette: Cassette) -> None:
        self._uri = uri
        self._cassette = cassette
        self._last_token_usage = TokenUsage(prompt_tokens=0, completion_tokens=0)
        self._client: object = None

    async def complete(self, messages: list[dict]) -> str:
        """Return the recorded response for *messages*, or raise if not found.

        :param messages: The messages being sent.
        :raises CassetteError: If no matching interaction exists in the cassette.
        :returns: The recorded response text.
        """
        key = interaction_key(self._uri, messages)
        interaction = next((i for i in self._cassette.interactions if i.key == key), None)
        if interaction is None:
            raise CassetteError(
                f"No cassette interaction found for key '{key}' (uri='{self._uri}'). "
                "Re-record the cassette with --record."
            )
        self._last_token_usage = TokenUsage(
            prompt_tokens=interaction.prompt_tokens,
            completion_tokens=interaction.completion_tokens,
        )
        return interaction.response

    @property
    def last_token_usage(self) -> TokenUsage:
        """Token usage from the last replayed interaction.

        :returns: A :class:`~sirenspec.core.usage.TokenUsage` instance.
        """
        return self._last_token_usage

    @property
    def client(self) -> object:
        """Return the underlying client (always ``None`` for replay providers).

        :returns: ``None``.
        """
        return self._client


class RecordingProvider:
    """An :class:`~sirenspec.providers.base.LLMProvider` that records interactions into a cassette.

    Wraps a real provider; every successful call appends an interaction to *cassette*.

    :param uri: The provider URI (used as part of the interaction hash key).
    :param real_provider: The underlying real provider to delegate calls to.
    :param cassette: The cassette to append recorded interactions to.
    """

    def __init__(self, uri: str, real_provider: LLMProvider, cassette: Cassette) -> None:
        self._uri = uri
        self._real = real_provider
        self._cassette = cassette

    async def complete(self, messages: list[dict]) -> str:
        """Call the real provider and record the interaction before returning.

        :param messages: The messages to send.
        :returns: The real provider's response text.
        """
        response = await self._real.complete(messages)
        usage = self._real.last_token_usage
        key = interaction_key(self._uri, messages)
        self._cassette.interactions.append(
            CassetteInteraction(
                key=key,
                response=response,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
            )
        )
        return response

    @property
    def last_token_usage(self) -> TokenUsage:
        """Token usage from the most recent real provider call.

        :returns: A :class:`~sirenspec.core.usage.TokenUsage` instance.
        """
        return self._real.last_token_usage

    @property
    def client(self) -> object:
        """Return the underlying real provider's client.

        :returns: The real provider's client object.
        """
        return self._real.client
