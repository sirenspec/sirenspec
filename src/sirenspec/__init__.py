"""SirenSpec — YAML-first agent orchestration SDK."""

__version__ = "0.1.0"

from sirenspec.core.events import NodeCompleteEvent, SummaryEvent
from sirenspec.core.executor import execute, execute_streaming
from sirenspec.core.models import Workflow
from sirenspec.exceptions import (
    GuardrailError,
    ProviderError,
    RetryExhaustedError,
    SirenSpecError,
    SwrmAgentError,
    ToolError,
    ValidationError,
)
from sirenspec.yaml.parser import load_workflow

__all__ = [
    "execute",
    "execute_streaming",
    "load_workflow",
    "Workflow",
    "NodeCompleteEvent",
    "SummaryEvent",
    "__version__",
    "SirenSpecError",
    "ProviderError",
    "RetryExhaustedError",
    "GuardrailError",
    "ValidationError",
    "SwrmAgentError",
    "ToolError",
]
