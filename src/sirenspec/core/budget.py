"""Workflow-level budget enforcement: token, USD, and wall-clock ceilings."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from sirenspec.core.models import BudgetConfig
from sirenspec.core.usage import TokenUsage
from sirenspec.exceptions import BudgetExceededError

logger = logging.getLogger(__name__)


@dataclass
class BudgetState:
    """Mutable budget bookkeeping carried alongside the executor.

    Wraps a :class:`~sirenspec.core.models.BudgetConfig` with running totals so the
    executor can ask a single object whether the next node should run, whether the
    remainder should be skipped, or whether the workflow should abort.

    :param config: The workflow's declared budget block, or ``None`` when no budget
        was configured (in which case every check is a no-op).
    :param start_time: ``time.monotonic()`` captured when the run began.  Used to
        evaluate ``max_duration_s``.
    :param skip_remaining: Latch set ``True`` after the first violation under
        ``on_exceeded='skip_remaining'``; remains set until the run ends.
    :param violations: Per-violation descriptions captured for the trace summary.
    """

    config: BudgetConfig | None
    start_time: float
    skip_remaining: bool = False
    violations: list[str] | None = None

    def __post_init__(self) -> None:
        """Initialise the violations list lazily so it is always a fresh per-run list."""
        if self.violations is None:
            self.violations = []


def detect_violation(
    config: BudgetConfig,
    usage: TokenUsage,
    estimated_usd: float | None,
    elapsed_s: float,
) -> str | None:
    """Return a human-readable violation message if any ceiling is exceeded, else ``None``.

    :param config: The active budget configuration.
    :param usage: Accumulated token usage across all completed nodes.
    :param estimated_usd: Running USD estimate, or ``None`` for local/unknown models.
    :param elapsed_s: Wall-clock seconds since the workflow started.
    :returns: Violation description string, or ``None`` when within budget.
    """
    if config.max_tokens is not None and usage.total > config.max_tokens:
        return f"Token ceiling exceeded: {usage.total} tokens used, limit is {config.max_tokens}"
    if config.max_cost_usd is not None and estimated_usd is not None and estimated_usd > config.max_cost_usd:
        return f"USD ceiling exceeded: ${estimated_usd:.6f} spent, limit is ${config.max_cost_usd:.6f}"
    if config.max_duration_s is not None and elapsed_s > config.max_duration_s:
        return f"Duration ceiling exceeded: {elapsed_s:.2f}s elapsed, limit is {config.max_duration_s:.2f}s"
    return None


def enforce_budget(
    state: BudgetState,
    usage: TokenUsage,
    estimated_usd: float | None,
) -> None:
    """Check the running spend against the workflow budget; raise, warn, or latch skip.

    No-ops when ``state.config`` is ``None``.  Otherwise computes the elapsed wall-clock
    seconds, queries :func:`detect_violation`, and dispatches on ``state.config.on_exceeded``:

    * ``'abort'`` raises :class:`~sirenspec.exceptions.BudgetExceededError`.
    * ``'warn'`` logs a structured warning the first time a given ceiling is hit and
      returns; the run keeps going.
    * ``'skip_remaining'`` sets ``state.skip_remaining = True`` so the executor stops
      issuing new LLM calls but still finalises the trace cleanly.

    :param state: The mutable :class:`BudgetState` carried across node executions.
    :param usage: Accumulated token usage across all completed nodes.
    :param estimated_usd: Running USD estimate, or ``None`` for local/unknown models.
    :raises BudgetExceededError: When ``on_exceeded='abort'`` and any ceiling is hit.
    """
    config = state.config
    if config is None:
        return

    elapsed_s = time.monotonic() - state.start_time
    violation = detect_violation(config, usage, estimated_usd, elapsed_s)
    if violation is None:
        return

    assert state.violations is not None
    if violation not in state.violations:
        state.violations.append(violation)

    if config.on_exceeded == "warn":
        logger.warning(
            "workflow budget warning: %s (tokens_used=%d, estimated_usd=%s, elapsed_s=%.2f)",
            violation,
            usage.total,
            estimated_usd,
            elapsed_s,
        )
        return

    if config.on_exceeded == "skip_remaining":
        if not state.skip_remaining:
            logger.warning(
                "workflow budget skipping remaining nodes: %s (tokens_used=%d, estimated_usd=%s)",
                violation,
                usage.total,
                estimated_usd,
            )
        state.skip_remaining = True
        return

    raise BudgetExceededError(reason=violation, tokens_used=usage.total, estimated_usd=estimated_usd)


def build_budget_status(
    state: BudgetState,
    usage: TokenUsage,
    estimated_usd: float | None,
) -> dict[str, Any] | None:
    """Build the ``budget`` block embedded in the workflow trace summary.

    Returns ``None`` when no budget was configured so the trace shape stays
    backward-compatible for workflows that don't use the budget feature.

    :param state: The final :class:`BudgetState` after the run completes.
    :param usage: Final aggregated token usage.
    :param estimated_usd: Final aggregated USD estimate, or ``None``.
    :returns: Trace dict with declared limits, observed totals, and the on_exceeded
        action, or ``None`` when no budget was configured.
    """
    config = state.config
    if config is None:
        return None

    elapsed_s = time.monotonic() - state.start_time
    return {
        "max_tokens": config.max_tokens,
        "max_cost_usd": config.max_cost_usd,
        "max_duration_s": config.max_duration_s,
        "on_exceeded": config.on_exceeded,
        "tokens_used": usage.total,
        "estimated_usd": estimated_usd,
        "duration_s": round(elapsed_s, 3),
        "exceeded": bool(state.violations),
        "violations": list(state.violations or []),
        "skipped_remaining": state.skip_remaining,
    }
