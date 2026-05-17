"""Static pricing tables for LLM provider cost estimation.

Source: https://openai.com/api/pricing/ and https://www.anthropic.com/pricing#anthropic-api
Last updated: 2025-05-17

Prices are in USD per 1,000 tokens. These tables are versioned data — update this
file (and the last-updated date above) when providers publish new pricing.

Ollama and other local providers are not present here; callers receive ``None``
for any model URI that is not found, indicating cost cannot be estimated.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPricing:
    """Per-1k-token pricing for a single model.

    :param prompt_usd_per_1k: Cost in USD per 1,000 prompt (input) tokens.
    :param completion_usd_per_1k: Cost in USD per 1,000 completion (output) tokens.
    """

    prompt_usd_per_1k: float
    completion_usd_per_1k: float


# ---------------------------------------------------------------------------
# Pricing tables
# Keys are "<provider>/<model-id>" as used in SirenSpec model URIs.
# ---------------------------------------------------------------------------

_PRICING: dict[str, ModelPricing] = {
    # OpenAI — https://openai.com/api/pricing/
    "openai/gpt-4o": ModelPricing(prompt_usd_per_1k=0.0025, completion_usd_per_1k=0.010),
    "openai/gpt-4o-mini": ModelPricing(prompt_usd_per_1k=0.000150, completion_usd_per_1k=0.000600),
    "openai/gpt-4-turbo": ModelPricing(prompt_usd_per_1k=0.010, completion_usd_per_1k=0.030),
    "openai/gpt-4": ModelPricing(prompt_usd_per_1k=0.030, completion_usd_per_1k=0.060),
    "openai/gpt-3.5-turbo": ModelPricing(prompt_usd_per_1k=0.0005, completion_usd_per_1k=0.0015),
    "openai/o1": ModelPricing(prompt_usd_per_1k=0.015, completion_usd_per_1k=0.060),
    "openai/o1-mini": ModelPricing(prompt_usd_per_1k=0.001, completion_usd_per_1k=0.004),
    "openai/o3": ModelPricing(prompt_usd_per_1k=0.010, completion_usd_per_1k=0.040),
    "openai/o3-mini": ModelPricing(prompt_usd_per_1k=0.0011, completion_usd_per_1k=0.0044),
    "openai/o4-mini": ModelPricing(prompt_usd_per_1k=0.0011, completion_usd_per_1k=0.0044),
    # Anthropic — https://www.anthropic.com/pricing#anthropic-api
    "anthropic/claude-opus-4-7": ModelPricing(prompt_usd_per_1k=0.015, completion_usd_per_1k=0.075),
    "anthropic/claude-sonnet-4-6": ModelPricing(prompt_usd_per_1k=0.003, completion_usd_per_1k=0.015),
    "anthropic/claude-haiku-4-5-20251001": ModelPricing(prompt_usd_per_1k=0.00025, completion_usd_per_1k=0.00125),
    "anthropic/claude-opus-4-5": ModelPricing(prompt_usd_per_1k=0.015, completion_usd_per_1k=0.075),
    "anthropic/claude-sonnet-4-5": ModelPricing(prompt_usd_per_1k=0.003, completion_usd_per_1k=0.015),
    "anthropic/claude-haiku-3-5": ModelPricing(prompt_usd_per_1k=0.00025, completion_usd_per_1k=0.00125),
}


def lookup_pricing(model_uri: str) -> ModelPricing | None:
    """Return pricing for *model_uri*, or ``None`` if the model is not in the table.

    :param model_uri: A SirenSpec model URI of the form ``"<provider>/<model-id>"``.
    :returns: A :class:`ModelPricing` instance, or ``None`` for unknown/local models.
    """
    return _PRICING.get(model_uri)


def estimate_usd(prompt_tokens: int, completion_tokens: int, model_uri: str) -> float | None:
    """Estimate the USD cost for a single LLM call.

    :param prompt_tokens: Number of prompt (input) tokens consumed.
    :param completion_tokens: Number of completion (output) tokens generated.
    :param model_uri: SirenSpec model URI, e.g. ``"openai/gpt-4o-mini"``.
    :returns: Estimated cost in USD, or ``None`` if the model has no pricing entry.
    """
    pricing = lookup_pricing(model_uri)
    if pricing is None:
        return None
    prompt_cost = (prompt_tokens / 1000.0) * pricing.prompt_usd_per_1k
    completion_cost = (completion_tokens / 1000.0) * pricing.completion_usd_per_1k
    return prompt_cost + completion_cost
