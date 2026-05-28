"""SirenSpec — YAML-first agent orchestration SDK."""

__version__ = "0.1.2"

from sirenspec.core.events import NodeCompleteEvent, SummaryEvent
from sirenspec.core.executor import execute, execute_streaming
from sirenspec.core.models import BudgetConfig, GuardrailSpec, HumanNode, Workflow
from sirenspec.core.usage import TokenUsage
from sirenspec.core.workflow_registry import WorkflowRegistry
from sirenspec.exceptions import (
    BudgetExceededError,
    GuardrailError,
    HumanInputError,
    ProviderError,
    RetryExhaustedError,
    SirenSpecError,
    SwrmAgentError,
    ToolError,
    ValidationError,
    WorkflowLintError,
)
from sirenspec.guardrails.base import Guardrail, GuardrailViolation, WorkflowGuardrail
from sirenspec.providers.base import LLMProvider
from sirenspec.yaml.parser import load_workflow

__all__ = [
    # Package metadata
    "__version__",
    # Execution
    "execute",
    "execute_streaming",
    "load_workflow",
    # Core models
    "Workflow",
    "GuardrailSpec",
    "HumanNode",
    "BudgetConfig",
    "WorkflowRegistry",
    # Streaming events
    "NodeCompleteEvent",
    "SummaryEvent",
    # Provider & guardrail extension points
    "LLMProvider",
    "Guardrail",
    "WorkflowGuardrail",
    # Value types
    "TokenUsage",
    # Exceptions
    "SirenSpecError",
    "ProviderError",
    "RetryExhaustedError",
    "GuardrailError",
    "GuardrailViolation",
    "BudgetExceededError",
    "HumanInputError",
    "ValidationError",
    "SwrmAgentError",
    "ToolError",
    "WorkflowLintError",
]
