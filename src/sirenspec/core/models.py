"""Pydantic v2 data models for SirenSpec workflow definitions."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator


class AgentDefinition(BaseModel):
    """Defines an LLM agent: model URI, system prompt, and optional guardrails."""

    model: str
    system: str
    guardrails: list[str] | None = None


class Node(BaseModel):
    """Binds an agent to an output context path."""

    agent: str
    writes: str


class Edge(BaseModel):
    """Connects two nodes in the workflow graph."""

    from_node: str = Field(alias="from")
    to_node: str = Field(alias="to")
    when: str | None = None

    model_config = {"populate_by_name": True}


class WorkflowInput(BaseModel):
    """Optional static input message for the first node."""

    message: str | None = None


class Workflow(BaseModel):
    """Top-level workflow definition loaded from YAML."""

    version: str
    agents: dict[str, AgentDefinition]
    nodes: dict[str, Node]
    edges: list[Edge] = Field(default_factory=list)
    input: WorkflowInput | None = None
    state: dict[str, Any] | None = None
    guardrails: list[str] | None = None

    @model_validator(mode="after")
    def validate_edges_reference_valid_nodes(self) -> Workflow:
        node_ids = set(self.nodes.keys())
        for edge in self.edges:
            if edge.from_node not in node_ids:
                raise ValueError(f"Edge references unknown node '{edge.from_node}' in 'from'")
            if edge.to_node not in node_ids:
                raise ValueError(f"Edge references unknown node '{edge.to_node}' in 'to'")
        return self

    @model_validator(mode="after")
    def validate_nodes_reference_valid_agents(self) -> Workflow:
        agent_ids = set(self.agents.keys())
        for node_id, node in self.nodes.items():
            if node.agent not in agent_ids:
                raise ValueError(f"Node '{node_id}' references unknown agent '{node.agent}'")
        return self
