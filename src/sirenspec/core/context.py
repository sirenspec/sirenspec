"""Mutable workflow execution context with dot-notation state access."""

from __future__ import annotations

from typing import Any


class WorkflowContext:
    """Holds mutable working and output dicts for a workflow execution."""

    def __init__(self, initial_state: dict[str, Any] | None = None) -> None:
        state = initial_state or {}
        self.working: dict[str, Any] = dict(state.get("working", {}))
        self.output: dict[str, Any] = dict(state.get("output", {}))

    def write(self, path: str, value: Any) -> None:
        """Write *value* to *path* using dot notation (e.g., ``working.key.subkey``)."""
        parts = path.split(".")
        root = parts[0]
        if root == "working":
            target = self.working
        elif root == "output":
            target = self.output
        else:
            raise KeyError(f"Unknown context root '{root}'; expected 'working' or 'output'")

        for part in parts[1:-1]:
            if part not in target or not isinstance(target[part], dict):
                target[part] = {}
            target = target[part]

        target[parts[-1]] = value

    def resolve(self, path: str) -> Any:
        """Resolve *path* using dot notation; raises KeyError if missing."""
        parts = path.split(".")
        root = parts[0]
        if root == "working":
            current: Any = self.working
        elif root == "output":
            current = self.output
        else:
            raise KeyError(f"Unknown context root '{root}'; expected 'working' or 'output'")

        for part in parts[1:]:
            if not isinstance(current, dict) or part not in current:
                raise KeyError(f"Path '{path}' not found in context")
            current = current[part]

        return current

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-serialisable copy of the current context."""
        return {"working": dict(self.working), "output": dict(self.output)}
