"""Pydantic v2 data models for SirenSpec workflow definitions."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class AgentDefinition(BaseModel):
    """Defines an LLM agent: model URI, system prompt, and optional guardrails."""

    model: str
    system: str
    guardrails: list[str] | None = None


class Node(BaseModel):
    """Binds an agent to an output context path."""

    type: Literal["agent"] = "agent"
    agent: str
    writes: str


class SwrmAgent(BaseModel):
    """Defines a single agent within a swrm node.

    Each swrm agent has its own provider, model, and prompt template.
    The prompt may use Jinja-style ``{{ variable }}`` interpolation against the
    workflow context (e.g. ``{{ inputs.report }}``).
    """

    id: str
    provider: str
    model: str | None = None
    prompt: str
    guardrails: list[str] | None = None


class SwrmSynthesis(BaseModel):
    """Optional synthesis step that runs after all swrm agents complete.

    The synthesis prompt may reference individual agent outputs via
    ``{{ <swrm_node_id>.agents.<agent_id>.output }}``.  Its output becomes
    the value of ``{{ <swrm_node_id>.output }}``.
    """

    provider: str
    model: str | None = None
    prompt: str
    guardrails: list[str] | None = None


class SwrmNode(BaseModel):
    """A parallel swrm node that fans out to multiple agents concurrently.

    All agents run concurrently up to the ``concurrency`` limit.  When a
    ``synthesis`` block is present its output becomes the node's canonical
    ``output``; otherwise the node's ``output`` is a list of agent outputs in
    definition order.

    :Example::

        nodes:
          analyze:
            type: swrm
            concurrency: 3
            on_failure: abort
            agents:
              - id: sentiment
                provider: openai
                model: gpt-4o-mini
                prompt: "Analyze sentiment in: {{ inputs.report }}"
            synthesis:
              provider: anthropic
              model: claude-haiku-4-5-20251001
              prompt: |
                Sentiment: {{ analyze.agents.sentiment.output }}
                Produce a recommendation.
    """

    type: Literal["swrm"] = "swrm"
    concurrency: int | None = Field(default=None, description="Max agents to run concurrently. Default: all agents.")
    agents: list[SwrmAgent] = Field(..., min_length=1, description="Ordered list of agents to run in parallel.")
    synthesis: SwrmSynthesis | None = Field(default=None, description="Optional synthesis step run after all agents.")
    on_failure: Literal["abort", "continue"] = Field(
        default="abort", description="Policy when an agent fails: abort (raise) or continue (skip)."
    )


AnyNode = Node | SwrmNode


class Edge(BaseModel):
    """Connects two nodes in the workflow graph.

    When ``when`` is omitted the edge is always traversed.  When it is set, the
    executor evaluates it as a Python expression against the live workflow context
    after the source node completes; the target node is activated only if the
    expression returns a truthy value.

    The expression namespace contains only ``working`` and ``output`` (plus the
    YAML literals ``true``, ``false``, and ``null``).  No built-ins are available,
    so arbitrary imports or side-effects are blocked.  Any evaluation error is
    treated as ``False`` so the edge is silently skipped.

    Example::

        edges:
          - from: triage
            to: handle_refund
            when: working.triage.intent == "refund"
          - from: triage
            to: handle_general
            when: working.triage.intent == "general"
    """

    from_node: str = Field(alias="from")
    to_node: str = Field(alias="to")
    when: str | None = Field(
        default=None,
        description=(
            "Optional Python expression evaluated against the workflow context after the source node "
            "completes.  Only 'working' and 'output' are in scope.  Evaluation failure is treated as False."
        ),
    )

    model_config = {"populate_by_name": True}


class WorkflowInput(BaseModel):
    """Optional static input message for the first node."""

    message: str | None = None


class Workflow(BaseModel):
    """Top-level workflow definition loaded from YAML."""

    version: str
    agents: dict[str, AgentDefinition] = Field(default_factory=dict)
    nodes: dict[str, AnyNode]
    edges: list[Edge] = Field(default_factory=list)
    input: WorkflowInput | None = None
    state: dict[str, Any] | None = None
    guardrails: list[str] | None = None

    @model_validator(mode="after")
    def validate_edges_reference_valid_nodes(self) -> Workflow:
        """Validate that all edge endpoints reference declared node IDs."""
        node_ids = set(self.nodes.keys())
        for edge in self.edges:
            if edge.from_node not in node_ids:
                raise ValueError(f"Edge references unknown node '{edge.from_node}' in 'from'")
            if edge.to_node not in node_ids:
                raise ValueError(f"Edge references unknown node '{edge.to_node}' in 'to'")
        return self

    @model_validator(mode="after")
    def validate_nodes_reference_valid_agents(self) -> Workflow:
        """Validate that agent-type nodes reference declared agent IDs."""
        agent_ids = set(self.agents.keys())
        for node_id, node in self.nodes.items():
            if isinstance(node, Node) and node.agent not in agent_ids:
                raise ValueError(f"Node '{node_id}' references unknown agent '{node.agent}'")
        return self
