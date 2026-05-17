"""Dynamic pricing lookup via LiteLLM's model_prices_and_context_window.json.

Pricing is resolved in priority order:
1. In-process memory cache (warm after first call in a process)
2. Filesystem cache at ~/.cache/sirenspec/pricing.json (fresh within 24 hours)
3. Fresh fetch from LiteLLM's published JSON (https://github.com/BerriAI/litellm)
4. Bundled snapshot shipped with the package (offline fallback)

Source: https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

logger = logging.getLogger(__name__)

_LITELLM_URL = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"
_CACHE_PATH = Path.home() / ".cache" / "sirenspec" / "pricing.json"
_CACHE_TTL_SECONDS = 86400  # 24 hours
_FETCH_TIMEOUT_SECONDS = 5

# In-process cache: populated on first lookup, shared for the lifetime of the process.
_PRICING: dict[str, ModelPricing] | None = None


@dataclass(frozen=True)
class ModelPricing:
    """Per-1k-token pricing for a single model.

    :param prompt_usd_per_1k: Cost in USD per 1,000 prompt (input) tokens.
    :param completion_usd_per_1k: Cost in USD per 1,000 completion (output) tokens.
    """

    prompt_usd_per_1k: float
    completion_usd_per_1k: float


def parse_litellm_entry(entry: dict) -> ModelPricing | None:
    """Convert a single LiteLLM pricing entry dict to a ModelPricing instance.

    LiteLLM stores costs as USD per *token* (not per 1,000 tokens), so the values
    are multiplied by 1,000 to match SirenSpec's per-1k convention.

    :param entry: Raw dict from the LiteLLM pricing JSON for one model.
    :returns: ModelPricing if both cost fields are present and non-None, else None.
    """
    input_cost = entry.get("input_cost_per_token")
    output_cost = entry.get("output_cost_per_token")
    if input_cost is None or output_cost is None:
        return None
    return ModelPricing(
        prompt_usd_per_1k=float(input_cost) * 1000.0,
        completion_usd_per_1k=float(output_cost) * 1000.0,
    )


def parse_litellm_data(data: dict) -> dict[str, ModelPricing]:
    """Parse a full LiteLLM pricing JSON dict into a model-name → ModelPricing map.

    Entries without both cost fields (e.g. metadata keys starting with ``_``) are
    silently skipped.

    :param data: Raw dict loaded from LiteLLM's model_prices_and_context_window.json.
    :returns: Mapping of LiteLLM model name (no provider prefix) to ModelPricing.
    """
    result: dict[str, ModelPricing] = {}
    for key, entry in data.items():
        if not isinstance(entry, dict):
            continue
        pricing = parse_litellm_entry(entry)
        if pricing is not None:
            result[key] = pricing
    return result


def load_snapshot() -> dict:
    """Load the bundled LiteLLM pricing snapshot shipped with the package.

    :returns: Raw dict from the bundled JSON snapshot.
    """
    snapshot_bytes = files("sirenspec.core.data").joinpath("litellm_pricing_snapshot.json").read_bytes()
    return json.loads(snapshot_bytes)


def load_cache() -> dict | None:
    """Load raw pricing data from the filesystem cache if it is fresh (< 24 h old).

    :returns: Raw LiteLLM pricing dict, or None if the cache is absent, expired, or corrupt.
    """
    if not _CACHE_PATH.exists():
        return None
    try:
        envelope = json.loads(_CACHE_PATH.read_bytes())
        age = time.time() - float(envelope.get("timestamp", 0))
        if age > _CACHE_TTL_SECONDS:
            return None
        return envelope.get("data")
    except (json.JSONDecodeError, ValueError, OSError):
        return None


def save_cache(data: dict) -> None:
    """Persist freshly fetched pricing data to the filesystem cache.

    Failures (e.g. read-only filesystem) are silently swallowed so that a cache
    write error never prevents normal execution.

    :param data: Raw LiteLLM pricing dict to store.
    """
    try:
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _CACHE_PATH.write_text(json.dumps({"timestamp": time.time(), "data": data}))
    except Exception as exc:
        logger.debug("Could not write pricing cache: %s", exc)


def fetch_remote() -> dict | None:
    """Fetch fresh pricing data from LiteLLM's published JSON and cache it.

    The result is saved to the filesystem cache on success so subsequent calls
    within the 24-hour window skip the network round-trip.

    :returns: Raw LiteLLM pricing dict, or None if the fetch fails for any reason.
    """
    try:
        with urllib.request.urlopen(_LITELLM_URL, timeout=_FETCH_TIMEOUT_SECONDS) as resp:
            data = json.loads(resp.read())
        save_cache(data)
        return data
    except (OSError, json.JSONDecodeError, urllib.error.URLError) as exc:
        logger.debug("Could not fetch LiteLLM pricing from remote: %s", exc)
        return None


def load_pricing_data() -> dict:
    """Resolve raw LiteLLM pricing data using the cache → remote → snapshot chain.

    :returns: Raw LiteLLM pricing dict from the first available source.
    """
    cached = load_cache()
    if cached is not None:
        return cached
    remote = fetch_remote()
    if remote is not None:
        return remote
    logger.warning("sirenspec: using bundled pricing snapshot; prices may be out of date")
    return load_snapshot()


def get_pricing() -> dict[str, ModelPricing]:
    """Return the in-process pricing map, loading it lazily on the first call.

    :returns: Mapping of LiteLLM model name to ModelPricing.
    """
    global _PRICING
    if _PRICING is None:
        _PRICING = parse_litellm_data(load_pricing_data())
    return _PRICING


def reset_pricing_cache() -> None:
    """Clear the in-process pricing cache so it is reloaded on the next lookup.

    Intended for tests and CLI tooling that needs a fresh load after a cache write.
    """
    global _PRICING
    _PRICING = None


def lookup_pricing(model_uri: str) -> ModelPricing | None:
    """Return pricing for *model_uri*, or ``None`` if the model is not known.

    SirenSpec model URIs use the form ``"<provider>/<model-id>"``.  The provider
    prefix is stripped to obtain the LiteLLM key, so ``"openai/gpt-4o-mini"``
    resolves to LiteLLM key ``"gpt-4o-mini"``.

    :param model_uri: A SirenSpec model URI of the form ``"<provider>/<model-id>"``.
    :returns: A ModelPricing instance, or None for unknown or local models.
    """
    litellm_key = model_uri.split("/", 1)[-1] if "/" in model_uri else model_uri
    return get_pricing().get(litellm_key)


def estimate_usd(prompt_tokens: int, completion_tokens: int, model_uri: str) -> float | None:
    """Estimate the USD cost for a single LLM call.

    :param prompt_tokens: Number of prompt (input) tokens consumed.
    :param completion_tokens: Number of completion (output) tokens generated.
    :param model_uri: SirenSpec model URI, e.g. ``"openai/gpt-4o-mini"``.
    :returns: Estimated cost in USD, or None if the model has no pricing entry.
    """
    pricing = lookup_pricing(model_uri)
    if pricing is None:
        return None
    return (prompt_tokens / 1000.0) * pricing.prompt_usd_per_1k + (
        completion_tokens / 1000.0
    ) * pricing.completion_usd_per_1k
