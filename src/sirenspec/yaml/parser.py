"""YAML workflow loader: parse file → validate into Workflow model."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from pydantic import ValidationError
from ruamel.yaml import YAML
from ruamel.yaml.constructor import DuplicateKeyError

from sirenspec.core.interpolation import check_circular_template_refs
from sirenspec.core.lint import lint_workflow
from sirenspec.core.models import Workflow
from sirenspec.exceptions import WorkflowLintError

_ENV_LINE_RE = re.compile(r"""^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$""")


def load_env_file(path: Path) -> None:
    """Parse a ``.env`` file and set variables in :data:`os.environ`.

    Supports ``KEY=VALUE``, quoted values (single or double), inline comments,
    and blank lines. Only sets variables that are not already present in the
    environment so shell exports always take precedence.

    :param path: Absolute path to the ``.env`` file.
    :raises FileNotFoundError: If *path* does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f".env file not found: {path}")

    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        m = _ENV_LINE_RE.match(stripped)
        if not m:
            continue
        key, raw_value = m.group(1), m.group(2)
        # Strip inline comments (unquoted # preceded by whitespace).
        value = raw_value
        if value and value[0] in ('"', "'"):
            quote = value[0]
            end = value.find(quote, 1)
            value = value[1:end] if end != -1 else value[1:]
        else:
            comment = re.search(r"\s+#", value)
            if comment:
                value = value[: comment.start()]
        if key not in os.environ:
            os.environ[key] = value


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
        workflow = Workflow.model_validate(raw)
    except ValidationError as exc:
        field_errors = "; ".join(f"{'.'.join(str(loc) for loc in e['loc'])}: {e['msg']}" for e in exc.errors())
        raise ValueError(f"Workflow validation failed in '{filepath}': {field_errors}") from exc

    check_circular_template_refs(workflow)

    lint_issues = lint_workflow(workflow)
    errors = [i for i in lint_issues if i.level == "error"]
    if errors:
        raise WorkflowLintError(errors)

    return workflow
