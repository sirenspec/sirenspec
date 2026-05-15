"""SirenSpec — YAML-first agent orchestration SDK."""

__version__ = "0.1.0"

from sirenspec.core.executor import execute
from sirenspec.core.models import Workflow
from sirenspec.exceptions import SwrmAgentError
from sirenspec.yaml.parser import load_workflow

__all__ = ["execute", "load_workflow", "Workflow", "SwrmAgentError", "__version__"]
