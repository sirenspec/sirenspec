"""YAML workflow loader: parse file → validate into Workflow model."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import ValidationError
from ruamel.yaml import YAML
from ruamel.yaml.constructor import DuplicateKeyError

from sirenspec.core.models import Workflow


def load_workflow(filepath: str | Path) -> Workflow:
    """Load and validate a SirenSpec YAML workflow file.

    :param filepath: Path to the ``.yaml`` workflow file.
    :raises FileNotFoundError: If the file does not exist.
    :raises ValueError: If the YAML is malformed or fails schema validation.
    :returns: A validated Workflow instance.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Workflow file not found: {filepath}")

    yaml = YAML(typ="safe")
    try:
        raw: Any = yaml.load(path)
    except DuplicateKeyError as exc:
        raise ValueError(f"Duplicate key in YAML: {exc}") from exc
    except Exception as exc:
        raise ValueError(f"YAML parse error in '{filepath}': {exc}") from exc

    if not isinstance(raw, dict):
        raise ValueError(f"Expected a YAML mapping at the top level of '{filepath}'")

    try:
        return Workflow.model_validate(raw)
    except ValidationError as exc:
        field_errors = "; ".join(f"{'.'.join(str(loc) for loc in e['loc'])}: {e['msg']}" for e in exc.errors())
        raise ValueError(f"Workflow validation failed in '{filepath}': {field_errors}") from exc
