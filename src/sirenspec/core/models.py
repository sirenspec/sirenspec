"""Pydantic v2 data models for SirenSpec workflow definitions."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class GuardrailSpec(BaseModel):
    """A named guardrail with optional configuration dict.

    Used when a guardrail requires configuration parameters (e.g. a JSON Schema
    for the ``schema`` guardrail).  Zero-config guardrails may still be specified
    as bare strings anywhere a ``list[str | GuardrailSpec]`` is accepted.

    :param name: The guardrail identifier as registered in the guardrail registry.
    :param config: Optional dict of configuration values passed to the guardrail factory.
    """

    name: str
    config: dict[str, Any] | None = None


class RetryPolicy(BaseModel):
    """Retry policy for a provider node or workflow-level defaults.

    Controls how many times the executor will retry a failing LLM call, which
    backoff strategy to use, how long to wait between attempts, and which error
    codes or categories trigger a retry.
    """

    max_attempts: int = Field(default=1, ge=1, description="Maximum number of total attempts (including the first).")
    backoff: Literal["exponential", "linear", "constant"] = Field(
        default="constant",
        description="Backoff strategy: 'exponential' doubles the delay each retry, "
        "'linear' adds base_delay each retry, 'constant' keeps the delay fixed.",
    )
    base_delay: float = Field(default=1.0, ge=0.0, description="Initial delay in seconds before the first retry.")
    max_delay: float = Field(default=30.0, ge=0.0, description="Upper bound on the computed delay in seconds.")
    jitter: bool = Field(default=False, description="When True, applies ±20% random variation to the computed delay.")
    on: list[str] = Field(
        default_factory=lambda: ["429", "network_error"],
        description=(
            "List of trigger conditions. HTTP status codes are written as strings (e.g. '429', '500'); "
            "use 'network_error' to also retry on connection-level failures; "
            "use 'guardrail_violation' to retry when an output guardrail rejects the response."
        ),
    )
    retry_on_guardrail: bool = Field(
        default=False,
        description=(
            "When True, output guardrail violations trigger a retry rather than an immediate failure. "
            "Equivalent to including 'guardrail_violation' in the ``on`` list and wrapping the output "
            "guardrail check inside the retry loop."
        ),
    )


class OnFailurePolicy(BaseModel):
    """Specifies what the executor should do when all retry attempts are exhausted."""

    action: Literal["abort", "fallback", "skip", "use_default"] = Field(
        default="abort",
        description=(
            "'abort' raises RetryExhaustedError; "
            "'fallback' routes execution to fallback_node; "
            "'skip' silently marks the node as skipped; "
            "'use_default' injects default_output into the context."
        ),
    )
    fallback_node: str | None = Field(
        default=None,
        description="Node ID to route execution to when action is 'fallback'.",
    )
    default_output: str | None = Field(
        default=None,
        description="Static string written to the node's 'writes' path when action is 'use_default'.",
    )


class WorkflowDefaults(BaseModel):
    """Workflow-level defaults applied to every node that does not override them."""

    retry: RetryPolicy | None = None
    on_failure: OnFailurePolicy | None = None


class BudgetConfig(BaseModel):
    """Top-level workflow budget for token, USD, and wall-clock spend.

    The executor checks cumulative spend after every node and enforces the ceilings
    declared here.  At least one of ``max_tokens``, ``max_cost_usd``, or ``max_duration_s``
    must be set — an empty budget block is rejected at validation time.

    :param max_tokens: Hard ceiling on the total token count across all nodes in a run.
    :param max_cost_usd: Hard ceiling on the estimated USD cost across all nodes.
        Cost estimation falls back to ``None`` for models without pricing entries
        (e.g. Ollama/local models); the ceiling is then effectively unenforced.
    :param max_duration_s: Wall-clock timeout in seconds for the entire workflow run.
    :param on_exceeded: What the executor does when a ceiling is hit: ``'abort'`` raises
        :class:`~sirenspec.exceptions.BudgetExceededError`; ``'warn'`` logs a structured
        warning and lets execution continue; ``'skip_remaining'`` marks all remaining
        nodes as skipped without issuing further LLM calls and finishes with a
        successful status.
    """

    max_tokens: int | None = Field(default=None, ge=1, description="Maximum total tokens across all nodes in a run.")
    max_cost_usd: float | None = Field(default=None, gt=0, description="Maximum estimated USD spend across all nodes.")
    max_duration_s: float | None = Field(default=None, gt=0, description="Maximum wall-clock seconds for the full run.")
    on_exceeded: Literal["abort", "warn", "skip_remaining"] = Field(
        default="abort",
        description=(
            "Action when any ceiling is hit: 'abort' raises BudgetExceededError; "
            "'warn' logs and continues; 'skip_remaining' finishes without new LLM calls."
        ),
    )

    @model_validator(mode="after")
    def validate_at_least_one_ceiling(self) -> BudgetConfig:
        """Reject empty budget blocks — at least one ceiling must be configured."""
        if self.max_tokens is None and self.max_cost_usd is None and self.max_duration_s is None:
            raise ValueError("budget block requires at least one of 'max_tokens', 'max_cost_usd', or 'max_duration_s'.")
        return self


class MemoryConfig(BaseModel):
    """Top-level workflow memory configuration.

    Declares a persistent key-value store that survives process restarts.  Node
    ``writes:`` paths starting with ``memory.`` are routed to this store instead of
    the in-process :class:`~sirenspec.core.context.WorkflowContext`.  All live keys
    are snapshotted before execution and exposed as ``{{ memory.key }}`` in templates.

    :param backend: Storage backend — ``'sqlite'`` (default, safer for concurrent writes)
        or ``'file'`` (plain JSON file, sufficient for single-process use).
    :param path: Directory (SQLite) or file base-path (JSON) for the store.
        Relative paths are resolved against the current working directory.
    :param ttl: Default time-to-live in seconds for all entries.  ``None`` means
        entries persist indefinitely.  Individual node writes may override this.
    """

    backend: Literal["file", "sqlite"] = "sqlite"
    path: str = ".sirenspec/memory"
    ttl: int | None = Field(default=None, ge=1, description="Default TTL in seconds; None means no expiry.")


class AgentDefinition(BaseModel):
    """Defines an LLM agent: model URI, system prompt, and optional guardrails."""

    model: str
    system: str
    guardrails: list[str | GuardrailSpec] | None = None


# ---------------------------------------------------------------------------
# Tool node models
# ---------------------------------------------------------------------------


class HttpToolConfig(BaseModel):
    """Configuration for the HTTP tool adapter.

    Supports GET and POST requests with optional headers, a request body, and a timeout.
    Template placeholders (``{{ expr }}``) in ``url``, ``headers``, and ``body`` are
    resolved against the workflow context before the request is made.
    """

    url: str
    method: Literal["GET", "POST"] = "GET"
    headers: dict[str, str] | None = None
    body: str | None = None
    timeout: int = Field(default=10, ge=1, description="Request timeout in seconds.")


class PythonToolConfig(BaseModel):
    """Configuration for the Python callable adapter.

    The executor imports ``module`` at runtime (relative to the user's environment, not
    the sirenspec package) and calls ``function`` with the keyword arguments in ``args``.
    """

    module: str
    function: str
    args: dict[str, Any] | None = None


class ToolNode(BaseModel):
    """A node that invokes an external tool (HTTP or Python callable) instead of an LLM agent.

    .. code-block:: yaml

        nodes:
          fetch_context:
            type: tool
            tool: http
            config:
              url: "{{ inputs.context_url }}"
              method: GET
              timeout: 10
            output_key: context_data
    """

    type: Literal["tool"]
    tool: Literal["http", "python"]
    config: HttpToolConfig | PythonToolConfig
    output_key: str = Field(default="output", description="Key under which the tool result is stored in node output.")
    retry: int = Field(default=0, ge=0, description="Number of times to retry the tool on failure.")
    on_failure: Literal["raise", "skip"] = Field(
        default="raise",
        description="What to do when the tool fails after all retries: 'raise' (default) or 'skip'.",
    )

    @model_validator(mode="before")
    @classmethod
    def coerce_config(cls, values: Any) -> Any:
        """Parse the raw ``config`` dict into the correct typed config model based on ``tool``."""
        if not isinstance(values, dict):
            return values
        tool = values.get("tool")
        config = values.get("config")
        if isinstance(config, dict):
            if tool == "http":
                values["config"] = HttpToolConfig(**config)
            elif tool == "python":
                values["config"] = PythonToolConfig(**config)
        return values


class AgentNode(BaseModel):
    """Binds an agent to an output context path (classic node type)."""

    type: Literal["agent"] | None = None  # ``None`` means the field is omitted (backward compat)
    agent: str
    writes: str
    streaming: bool = Field(default=True, description="When True, use token streaming if the provider supports it.")
    retry: RetryPolicy | None = None
    on_failure: OnFailurePolicy | None = None
    max_tokens_per_call: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Optional ceiling on the number of completion tokens the provider may produce for "
            "a single call.  When set, it is forwarded to the provider as the ``max_tokens`` "
            "parameter so the LLM truncates its own response rather than running unbounded."
        ),
    )


# ---------------------------------------------------------------------------
# Swrm (parallel agent fan-out) node models
# ---------------------------------------------------------------------------


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
    guardrails: list[str | GuardrailSpec] | None = None


class SwrmSynthesis(BaseModel):
    """Optional synthesis step that runs after all swrm agents complete.

    The synthesis prompt may reference individual agent outputs via
    ``{{ <swrm_node_id>.agents.<agent_id>.output }}``.  Its output becomes
    the value of ``{{ <swrm_node_id>.output }}``.
    """

    provider: str
    model: str | None = None
    prompt: str
    guardrails: list[str | GuardrailSpec] | None = None


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


class FactorySwrm(BaseModel):
    """Swrm specification executed per item in a factory fan-out.

    Reuses :class:`SwrmAgent` and :class:`SwrmSynthesis` — agent prompts support
    ``{{ item }}``, ``{{ index }}``, and ``{{ total }}`` in addition to all standard
    interpolation namespaces.  Each factory item spawns one full swrm of these agents
    running in parallel; the swrm result (synthesis output or list of agent outputs) is
    collected as one entry in the factory output list.

    :param agents: One or more agents to run in parallel for each factory item.
    :param synthesis: Optional synthesis step run after all agents complete.
    :param concurrency: Max agents to run concurrently per item. Default: all agents.
    """

    agents: list[SwrmAgent] = Field(..., min_length=1)
    synthesis: SwrmSynthesis | None = None
    concurrency: int | None = Field(default=None, ge=1)


class FactoryNode(BaseModel):
    """A factory node that fans out work across multiple parallel instances.

    Supports three mutually exclusive execution modes determined by which fields are set:

    **Agent + for_each** — one agent call per item in a runtime JSON list.
    ``{{ item }}`` and ``{{ index }}`` available in ``inputs:`` templates.

    **Agent + swarm_size** — N identical agent calls on the same input.
    ``{{ index }}`` (0-based) and ``{{ total }}`` available; no ``{{ item }}``.

    **Swrm + for_each** — one full :class:`FactorySwrm` (parallel specialist agents
    with optional synthesis) per item in a runtime JSON list.  Agent prompts and the
    synthesis prompt all receive ``{{ item }}``, ``{{ index }}``, and ``{{ total }}``.

    All instance outputs are collected and written to the ``writes`` path as a list.

    :Example (agent + for_each)::

        nodes:
          execute:
            type: factory
            agent: worker_agent
            for_each: "{{ plan.output }}"
            inputs:
              task: "{{ item }}"
            concurrency: 4
            writes: working.execute.outputs

    :Example (agent + swarm_size)::

        nodes:
          execute:
            type: factory
            agent: worker_agent
            swarm_size: 5
            inputs:
              position: "{{ index }} of {{ total }}"
            concurrency: 5
            writes: working.execute.outputs

    :Example (swrm + for_each)::

        nodes:
          grade_papers:
            type: factory
            swrm:
              agents:
                - id: editor
                  provider: openai
                  model: gpt-4o-mini
                  prompt: "Review: {{ item }}"
                - id: grader
                  provider: anthropic
                  model: claude-haiku-4-5-20251001
                  prompt: "Grade this paper: {{ item }}"
              synthesis:
                provider: anthropic
                model: claude-haiku-4-5-20251001
                prompt: |
                  Editor: {{ grade_papers.agents.editor.output }}
                  Grader: {{ grade_papers.agents.grader.output }}
                  Return final grade.
            for_each: "{{ inputs.papers }}"
            concurrency: 3
            writes: working.grades
    """

    type: Literal["factory"] = "factory"
    agent: str | None = Field(
        default=None,
        description="Named agent from the workflow's top-level agents map. Mutually exclusive with swrm.",
    )
    swrm: FactorySwrm | None = Field(
        default=None,
        description="Inline swrm spec executed per item. Mutually exclusive with agent.",
    )
    for_each: str | None = Field(
        default=None,
        description="Template expression that resolves to a JSON list at runtime. Mutually exclusive with swarm_size.",
    )
    swarm_size: int | str | None = Field(
        default=None,
        description=(
            "Static count or template expression (e.g. '{{ inputs.count }}') for parallel agent instances. "
            "Mutually exclusive with for_each. Only valid with agent mode."
        ),
    )
    inputs: dict[str, str] = Field(
        default_factory=dict,
        description="Template strings for each input. Supports {{ item }}, {{ index }}, and {{ total }}.",
    )
    concurrency: int = Field(default=1, ge=1, description="Max parallel worker instances.")
    timeout_per_instance: int = Field(default=60, ge=1, description="Per-instance timeout in seconds.")
    on_failure: Literal["abort", "continue"] = Field(
        default="abort", description="Policy when an instance fails: abort (raise) or continue (skip)."
    )
    writes: str = Field(..., description="Dot-notation path where the outputs list is stored.")

    @model_validator(mode="after")
    def validate_mode_fields(self) -> FactoryNode:
        """Enforce execution target and fan-out mode constraints."""
        has_agent = self.agent is not None
        has_swrm = self.swrm is not None
        has_for_each = self.for_each is not None
        has_swarm_size = self.swarm_size is not None

        if has_agent == has_swrm:
            raise ValueError("Exactly one of 'agent' or 'swrm' must be set.")
        if has_for_each == has_swarm_size:
            raise ValueError("Exactly one of 'for_each' or 'swarm_size' must be set.")
        if has_swrm and has_swarm_size:
            raise ValueError("'swrm' mode requires 'for_each'; 'swarm_size' is only valid with 'agent'.")
        return self


class WorkflowNode(BaseModel):
    """A node that executes another SirenSpec workflow inline (sub-workflow composition).

    The referenced sub-workflow runs blocking inside the parent workflow.  Its output
    dict (keyed by sub-node ID) is written into the parent context so downstream nodes
    can reference it via ``{{ <node_id>.output.<sub_node_id> }}``.

    ``ref`` accepts either a relative/absolute file path (``./path/to/b.yaml``) resolved
    at execution time, or a named string resolved from a
    :class:`~sirenspec.core.workflow_registry.WorkflowRegistry` passed to the executor.

    The sub-workflow's context is initialised with only the keys declared in ``inputs``;
    it does not inherit the parent's full working context.  Values in ``inputs`` are
    template strings resolved against the parent context before the sub-workflow starts.

    :Example::

        nodes:
          run_b:
            type: workflow
            ref: ./workflows/b.yaml
            inputs:
              topic: "{{ extract.output }}"
              max_tokens: "500"
    """

    type: Literal["workflow"] = "workflow"
    ref: str = Field(..., description="File path or registry name for the sub-workflow to execute.")
    inputs: dict[str, str] = Field(
        default_factory=dict,
        description="Template strings bound to the sub-workflow's input context.",
    )
    writes: str | None = Field(
        default=None,
        description=(
            "Optional dot-notation path to write sub-workflow output in the parent context. "
            "Defaults to 'output.<node_id>' when omitted."
        ),
    )
    max_depth: int = Field(
        default=10,
        ge=1,
        description="Maximum nesting depth before a ValidationError is raised.",
    )


class HumanNode(BaseModel):
    """A node that pauses execution to collect input from a human operator.

    Human nodes consume no LLM tokens.  They render an optional prompt, block until
    a response is received (or a timeout expires), and write the collected text to
    the workflow context like any other node.  Downstream nodes consume the response
    via the normal ``{{ <node_id>.output }}`` chain.

    .. code-block:: yaml

        nodes:
          approve_draft:
            type: human
            prompt: |
              {{ draft.output }}

              Approve this draft? (yes/edit/reject)
            writes: working.approval
            timeout: 3600          # seconds before auto-action
            on_timeout: use_default
            default_output: "approved"

    Execution semantics:

    * The ``prompt`` field is resolved against the workflow context before being
      shown — it can reference upstream outputs and inputs.
    * ``timeout`` is wall-clock seconds; ``None`` means wait indefinitely.
    * When the timeout expires, ``on_timeout`` decides the response: ``'abort'``
      raises :class:`~sirenspec.exceptions.HumanInputError`, ``'skip'`` writes an
      empty string and continues, ``'use_default'`` writes ``default_output``.
    """

    type: Literal["human"]
    prompt: str | None = Field(
        default=None,
        description=(
            "Optional template string shown to the operator before they respond.  Supports "
            "all standard ``{{ expr }}`` interpolation.  When omitted, only the node id is shown."
        ),
    )
    writes: str = Field(
        ...,
        description="Dot-notation path where the collected response is stored in the workflow context.",
    )
    timeout: float | None = Field(
        default=None,
        gt=0,
        description="Wall-clock seconds before the on_timeout action is taken.  ``None`` waits indefinitely.",
    )
    on_timeout: Literal["abort", "skip", "use_default"] = Field(
        default="abort",
        description=(
            "Action on timeout: 'abort' raises HumanInputError; 'skip' writes empty string and continues; "
            "'use_default' writes ``default_output``."
        ),
    )
    default_output: str | None = Field(
        default=None,
        description="Static fallback string written when on_timeout='use_default'.",
    )

    @model_validator(mode="after")
    def validate_default_required_for_use_default(self) -> HumanNode:
        """Ensure ``default_output`` is supplied when ``on_timeout='use_default'``."""
        if self.on_timeout == "use_default" and self.default_output is None:
            raise ValueError("HumanNode with on_timeout='use_default' requires 'default_output' to be set.")
        return self


# Backward-compatible alias so existing code using ``Node(agent=..., writes=...)`` keeps working.
Node = AgentNode
AnyNode = AgentNode | ToolNode | SwrmNode | FactoryNode | WorkflowNode | HumanNode


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


# Discriminator helper: dispatches on the raw ``type`` field with AgentNode as the fallback.
def parse_node(raw: Any) -> AnyNode:
    """Parse a raw node dict into a typed node model.

    :param raw: The raw YAML mapping for a single node.
    :returns: A typed node instance (AgentNode, ToolNode, SwrmNode, FactoryNode, WorkflowNode, or HumanNode).
    """
    if isinstance(raw, (AgentNode, ToolNode, SwrmNode, FactoryNode, WorkflowNode, HumanNode)):
        return raw
    if isinstance(raw, dict):
        t = raw.get("type")
        if t == "tool":
            return ToolNode.model_validate(raw)
        if t == "swrm":
            return SwrmNode.model_validate(raw)
        if t == "factory":
            return FactoryNode.model_validate(raw)
        if t == "workflow":
            return WorkflowNode.model_validate(raw)
        if t == "human":
            return HumanNode.model_validate(raw)
    return AgentNode.model_validate(raw)


class Workflow(BaseModel):
    """Top-level workflow definition loaded from YAML."""

    version: str
    agents: dict[str, AgentDefinition] = Field(default_factory=dict)
    nodes: dict[str, AgentNode | ToolNode | SwrmNode | FactoryNode | WorkflowNode | HumanNode]
    edges: list[Edge] = Field(default_factory=list)
    input: WorkflowInput | None = None
    state: dict[str, Any] | None = None
    guardrails: list[str | GuardrailSpec] | None = None
    defaults: WorkflowDefaults | None = None
    budget: BudgetConfig | None = Field(
        default=None,
        description=(
            "Optional cumulative budget for the run.  When set, the executor enforces "
            "token/USD/duration ceilings after every node and the budget status is "
            "included in the trace summary."
        ),
    )
    env_file: str | None = Field(
        default=None,
        description=(
            "Path to a .env file to load before execution, relative to the workflow file. "
            "Variables are set in os.environ so provider clients pick them up automatically."
        ),
    )
    memory: MemoryConfig | None = Field(
        default=None,
        description=(
            "Optional persistent memory store.  When set, node writes to 'memory.*' paths "
            "are persisted across runs and '{{ memory.key }}' resolves in all templates."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def parse_nodes(cls, values: Any) -> Any:
        """Convert each raw node mapping into the correct typed node model."""
        if not isinstance(values, dict):
            return values
        raw_nodes = values.get("nodes")
        if isinstance(raw_nodes, dict):
            values["nodes"] = {node_id: parse_node(raw) for node_id, raw in raw_nodes.items()}
        return values

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
    def validate_agent_nodes_reference_valid_agents(self) -> Workflow:
        """Validate that agent-type and factory-type nodes reference declared agent IDs."""
        agent_ids = set(self.agents.keys())
        for node_id, node in self.nodes.items():
            if isinstance(node, AgentNode) and node.agent not in agent_ids:
                raise ValueError(f"Node '{node_id}' references unknown agent '{node.agent}'")
            if isinstance(node, FactoryNode) and node.agent is not None and node.agent not in agent_ids:
                raise ValueError(f"Node '{node_id}' references unknown agent '{node.agent}'")
        return self

    # Keep backward-compat alias: ``node.agent`` and ``node.writes`` still work for AgentNode.
    # ToolNode uses ``node.tool``, ``node.config``, and ``node.output_key``.
