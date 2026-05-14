"""Mermaid flowchart renderer for SirenSpec workflows."""

from __future__ import annotations

from sirenspec.core.models import Workflow


def workflow_to_mermaid(workflow: Workflow) -> str:
    """Render a Workflow as a Mermaid flowchart string.

    :param workflow: A validated Workflow instance.
    :returns: Mermaid ``graph TD`` diagram source.
    """
    lines: list[str] = ["graph TD"]

    for node_id in workflow.nodes:
        lines.append(f"    {node_id}[{node_id}]")

    for edge in workflow.edges:
        if edge.when:
            label = edge.when.replace('"', "'")
            lines.append(f'    {edge.from_node} -->|"{label}"| {edge.to_node}')
        else:
            lines.append(f"    {edge.from_node} --> {edge.to_node}")

    return "\n".join(lines)
