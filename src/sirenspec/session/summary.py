"""Pure functions that summarise a :class:`~sirenspec.core.models.Workflow` for display.

These build the data the left rail and splash header render — the node tree, the agent
roster, and headline counts — without any Textual or I/O dependency, so they are trivial
to unit-test and reuse from the plain-console fallback.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from sirenspec.core.models import (
    AgentNode,
    FactoryNode,
    HumanNode,
    SwrmNode,
    ToolNode,
    Workflow,
    WorkflowNode,
)


def provider_of(model_uri: str) -> str:
    """Return the provider name (the part before ``:`` or ``/``) of a model URI.

    :param model_uri: A model URI such as ``"anthropic:claude-haiku-4-5-20251001"``.
    :returns: The provider prefix (e.g. ``"anthropic"``), or the whole string if no
        separator is present.
    """
    for sep in (":", "/"):
        if sep in model_uri:
            return model_uri.split(sep, 1)[0]
    return model_uri


@dataclass(frozen=True)
class ChildSummary:
    """A leaf under a node in the rail tree (e.g. a swrm agent).

    :param label: The display label for the child (e.g. an agent id).
    :param provider: The provider name shown right-aligned, or ``""`` when not applicable.
    """

    label: str
    provider: str = ""


@dataclass(frozen=True)
class NodeSummary:
    """A single node row in the rail tree.

    :param node_id: The node's identifier.
    :param kind: A short parenthetical descriptor (e.g. ``"swrm"`` or ``"tool: http"``).
    :param children: Ordered leaf rows shown indented under the node.
    """

    node_id: str
    kind: str
    children: list[ChildSummary] = field(default_factory=list)


@dataclass(frozen=True)
class AgentSummary:
    """An entry in the agent roster section of the rail.

    :param agent_id: The agent's identifier.
    :param provider: The provider name resolved from the agent's model URI.
    """

    agent_id: str
    provider: str


@dataclass(frozen=True)
class WorkflowSummary:
    """Everything the rail and splash need to describe a workflow at a glance.

    :param name: Human-facing workflow name (typically the file stem).
    :param version: The workflow's declared version string.
    :param node_count: Total number of nodes.
    :param agent_count: Total number of declared agents.
    :param primary_provider: The most common provider across declared agents, or ``""``.
    :param nodes: Ordered node rows for the rail tree.
    :param agents: Ordered agent roster rows.
    """

    name: str
    version: str
    node_count: int
    agent_count: int
    primary_provider: str
    nodes: list[NodeSummary] = field(default_factory=list)
    agents: list[AgentSummary] = field(default_factory=list)


def summarise_node(node_id: str, node: object, workflow: Workflow) -> NodeSummary:
    """Build the rail row for a single node.

    :param node_id: The node's identifier.
    :param node: The typed node model instance.
    :param workflow: The owning workflow (used to resolve agent providers).
    :returns: The :class:`NodeSummary` describing this node.
    """
    if isinstance(node, SwrmNode):
        children = [ChildSummary(label=a.id, provider=a.provider) for a in node.agents]
        return NodeSummary(node_id=node_id, kind="swrm", children=children)
    if isinstance(node, ToolNode):
        return NodeSummary(node_id=node_id, kind=f"tool: {node.tool}")
    if isinstance(node, FactoryNode):
        kind = "factory: swrm" if node.swrm is not None else f"factory: {node.agent}"
        return NodeSummary(node_id=node_id, kind=kind)
    if isinstance(node, WorkflowNode):
        return NodeSummary(node_id=node_id, kind=f"workflow: {node.ref}")
    if isinstance(node, HumanNode):
        return NodeSummary(node_id=node_id, kind="human")
    if isinstance(node, AgentNode):
        agent = workflow.agents.get(node.agent)
        provider = provider_of(agent.model) if agent is not None else ""
        return NodeSummary(node_id=node_id, kind="agent", children=[ChildSummary(label=node.agent, provider=provider)])
    return NodeSummary(node_id=node_id, kind="node")


def summarise_workflow(workflow: Workflow, name: str) -> WorkflowSummary:
    """Build a full :class:`WorkflowSummary` for the rail and splash.

    :param workflow: The validated workflow.
    :param name: Human-facing workflow name (typically the workflow file stem).
    :returns: The assembled :class:`WorkflowSummary`.
    """
    nodes = [summarise_node(node_id, node, workflow) for node_id, node in workflow.nodes.items()]
    agents = [
        AgentSummary(agent_id=agent_id, provider=provider_of(definition.model))
        for agent_id, definition in workflow.agents.items()
    ]
    provider_counts = Counter(a.provider for a in agents if a.provider)
    primary_provider = provider_counts.most_common(1)[0][0] if provider_counts else ""
    return WorkflowSummary(
        name=name,
        version=workflow.version,
        node_count=len(workflow.nodes),
        agent_count=len(workflow.agents),
        primary_provider=primary_provider,
        nodes=nodes,
        agents=agents,
    )
