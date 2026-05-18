"""Cost-cap guardrail: aborts or warns when a workflow exceeds a token or USD budget."""

from __future__ import annotations

import logging

from sirenspec.core.usage import TokenUsage
from sirenspec.exceptions import BudgetExceededError
from sirenspec.guardrails.base import Guardrail

logger = logging.getLogger(__name__)


class CostCapGuardrail(Guardrail):
    """Tracks accumulated token spend and enforces a configurable budget ceiling.

    Implements both :class:`~sirenspec.guardrails.base.Guardrail` (for registry
    compatibility) and the :class:`~sirenspec.guardrails.base.WorkflowGuardrail`
    protocol (``check_budget``).  The executor calls :meth:`check_budget` after
    each node; ``check_input`` and ``check_output`` are pass-throughs.

    At least one of *max_usd* or *max_tokens* must be provided.

    :param max_usd: Maximum estimated USD spend for the entire run.  Checked only
        when a USD estimate is available (i.e. the model has a pricing entry).
    :param max_tokens: Hard token ceiling independent of USD.
    :param action: ``'abort'`` raises :class:`~sirenspec.exceptions.BudgetExceededError`;
        ``'warn'`` logs a structured warning and allows execution to continue.
    """

    def __init__(
        self,
        max_usd: float | None = None,
        max_tokens: int | None = None,
        action: str = "abort",
    ) -> None:
        if max_usd is None and max_tokens is None:
            raise ValueError("cost_cap guardrail requires at least one of 'max_usd' or 'max_tokens'")
        if action not in {"abort", "warn"}:
            raise ValueError(f"cost_cap action must be 'abort' or 'warn', got '{action}'")
        self.max_usd = max_usd
        self.max_tokens = max_tokens
        self.action = action

    def check_input(self, text: str) -> str:
        """Pass-through: cost cap does not inspect individual inputs.

        :param text: Input text.
        :returns: The original text unchanged.
        """
        return text

    def check_output(self, text: str) -> str:
        """Pass-through: cost cap does not inspect individual outputs.

        :param text: Output text.
        :returns: The original text unchanged.
        """
        return text

    def check_budget(self, usage: TokenUsage, estimated_usd: float | None) -> None:
        """Assert accumulated spend is within budget; raise or warn if exceeded.

        :param usage: Combined token usage across all completed nodes.
        :param estimated_usd: Running USD estimate, or ``None`` for local models.
        :raises BudgetExceededError: In ``'abort'`` mode when any ceiling is exceeded.
        """
        violation = self.detect_violation(usage, estimated_usd)
        if violation is None:
            return

        if self.action == "warn":
            logger.warning(
                "cost_cap budget warning: %s (tokens_used=%d, estimated_usd=%s)",
                violation,
                usage.total,
                estimated_usd,
            )
            return

        raise BudgetExceededError(
            reason=violation,
            tokens_used=usage.total,
            estimated_usd=estimated_usd,
        )

    def detect_violation(self, usage: TokenUsage, estimated_usd: float | None) -> str | None:
        """Return a human-readable violation message if a ceiling is exceeded, else ``None``.

        :param usage: Accumulated token usage.
        :param estimated_usd: Running USD estimate, or ``None``.
        :returns: Violation description string, or ``None`` if within budget.
        """
        if self.max_tokens is not None and usage.total > self.max_tokens:
            return f"Token ceiling exceeded: {usage.total} tokens used, limit is {self.max_tokens}"
        if self.max_usd is not None and estimated_usd is not None and estimated_usd > self.max_usd:
            return f"USD ceiling exceeded: ${estimated_usd:.6f} spent, limit is ${self.max_usd:.6f}"
        return None
