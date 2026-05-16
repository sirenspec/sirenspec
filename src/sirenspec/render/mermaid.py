"""Mermaid flowchart renderer for SirenSpec workflows."""

from __future__ import annotations

from sirenspec.core.models import SwrmNode, Workflow


def swrm_subgraph_lines(node_id: str, node: SwrmNode) -> list[str]:
    """Render a SwrmNode as a Mermaid subgraph with agents and optional synthesis.

    :param node_id: The workflow-level node identifier.
    :param node: The SwrmNode to render.
    :returns: Lines to append to the Mermaid diagram.
    """
    lines: list[str] = [f"    subgraph {node_id}"]
    agent_mermaid_ids = [f"{node_id}_{a.id}" for a in node.agents]
    for agent, aid in zip(node.agents, agent_mermaid_ids):
        lines.append(f"        {aid}[{agent.id}]")
    if node.synthesis:
        synth_id = swrm_synthesis_id(node_id)
        lines.append(f"        {synth_id}([synthesis])")
        for aid in agent_mermaid_ids:
            lines.append(f"        {aid} --> {synth_id}")
    lines.append("    end")
    return lines


def swrm_synthesis_id(node_id: str) -> str:
    """Return the Mermaid node ID for a swrm node's synthesis step.

    :param node_id: The workflow-level node identifier.
    :returns: The synthesis node's Mermaid ID.
    """
    return f"{node_id}_synthesis"


def edge_source_id(node_id: str, workflow: Workflow) -> str:
    """Return the Mermaid ID to use as the source of a workflow edge.

    For swrm nodes with synthesis, edges exit from the synthesis node.
    For all other node types, the node's own ID is used.

    :param node_id: The workflow-level node identifier.
    :param workflow: The workflow containing the node.
    :returns: The Mermaid node ID to use as the edge source.
    """
    node = workflow.nodes.get(node_id)
    if isinstance(node, SwrmNode) and node.synthesis:
        return swrm_synthesis_id(node_id)
    return node_id


def workflow_to_mermaid(workflow: Workflow) -> str:
    """Render a Workflow as a Mermaid flowchart string.

    SwrmNodes are rendered as subgraphs with their agents and optional synthesis
    step shown as internal nodes. Workflow-level edges from a swrm node with
    synthesis exit from the synthesis node.

    :param workflow: A validated Workflow instance.
    :returns: Mermaid ``graph TD`` diagram source.
    """
    lines: list[str] = ["graph TD"]

    for node_id, node in workflow.nodes.items():
        if isinstance(node, SwrmNode):
            lines.extend(swrm_subgraph_lines(node_id, node))
        else:
            lines.append(f"    {node_id}[{node_id}]")

    for edge in workflow.edges:
        from_id = edge_source_id(edge.from_node, workflow)
        to_id = edge.to_node
        if edge.when:
            label = edge.when.replace('"', "'")
            lines.append(f'    {from_id} -->|"{label}"| {to_id}')
        else:
            lines.append(f"    {from_id} --> {to_id}")

    return "\n".join(lines)
