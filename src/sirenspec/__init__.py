"""SirenSpec — YAML-first agent orchestration SDK."""

__version__ = "0.1.0"

from sirenspec.core.events import NodeCompleteEvent, SummaryEvent
from sirenspec.core.executor import execute, execute_streaming
from sirenspec.core.models import GuardrailSpec, Workflow
from sirenspec.core.usage import TokenUsage
from sirenspec.core.workflow_registry import WorkflowRegistry
from sirenspec.exceptions import (
    BudgetExceededError,
    GuardrailError,
    ProviderError,
    RetryExhaustedError,
    SirenSpecError,
    SwrmAgentError,
    ToolError,
    ValidationError,
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
    "ValidationError",
    "SwrmAgentError",
    "ToolError",
]
