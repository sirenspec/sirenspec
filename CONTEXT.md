# SirenSpec

A YAML-first agent orchestration SDK for defining and executing multi-agent workflows against LLM backends.

## Language

**Workflow**:
A YAML document that declares agents, nodes, edges, and guardrails — the full specification of what runs.
_Avoid_: Pipeline, spec, script

**Recipe**:
A self-contained, runnable workflow in its own subdirectory under `docs/cookbook/`, consisting of a `workflow.yaml` and a Mintlify `README.mdx` with frontmatter, that demonstrates one specific named capability of SirenSpec.
_Avoid_: Example, demo, sample

**Cookbook**:
The `docs/cookbook/` directory — the canonical, Mintlify-served collection of recipes. Runnable via `sirenspec run docs/cookbook/<name>/workflow.yaml`.
_Avoid_: Examples directory, samples directory

**Node**:
A single unit of execution in a workflow — either an agent call, a tool invocation, or a swrm fan-out.
_Avoid_: Step, task, stage

**Agent**:
An LLM-backed actor declared in a workflow, identified by a model URI and a system prompt.
_Avoid_: Model, bot, actor

**Edge**:
A directed connection between two nodes that controls execution order; may carry an optional `when` condition.
_Avoid_: Link, transition, arrow

**Guardrail**:
A validation layer applied to agent input and output — e.g. injection detection, PII redaction, schema validation. Declared as a list of names (bare strings for zero-config guardrails) or inline objects (`name:` + `config:`) for guardrails that require or accept configuration. Zero-config guardrails (injection, length, pii) have sensible defaults and work as bare strings; config-required guardrails (schema) always need a config block.
_Avoid_: Filter, check, validator

**Context**:
The live key-value store of `working.*` and `output.*` values that accumulates as nodes execute.
_Avoid_: State, memory, store

**Swrm**:
A node type that fans out to multiple agents concurrently and optionally synthesizes their outputs.
_Avoid_: Parallel node, fan-out, multi-agent node

## Relationships

- A **Workflow** contains one or more **Nodes** connected by **Edges**
- A **Node** references an **Agent** (or a tool, or is a **Swrm**) and writes its output to a **Context** path
- A **Recipe** is a **Workflow** that lives in the **Cookbook** and demonstrates a single named capability
- A **Guardrail** is declared at the workflow level or per-agent and is applied after each agent produces output

**GuardrailSpec**:
A Pydantic model (`name: str`, `config: dict[str, Any] | None`) defined in `models.py` alongside other YAML schema types. Replaces the bare `str` in `Workflow.guardrails` and `AgentDefinition.guardrails` — both become `list[str | GuardrailSpec] | None`. Zero-config guardrails may still be declared as bare strings.
_Avoid_: Defining this type in `guardrails/base.py` (creates a circular dependency and splits schema types across modules).

**Cassette**:
A YAML file recording `(model_uri, messages_hash) → response_text` mappings used by `sirenspec test --mock` to replay LLM responses without making real API calls. Written by `--record`, consumed by `--mock`. Human-readable and stable across SDK version changes.
_Avoid_: HTTP-level cassette formats (VCR-style) that couple tests to internal request/response wire details.

**Pricing Cache**:
Model token pricing is fetched from LiteLLM's open-source pricing JSON (`model_prices_and_context_window.json`) with a 24-hour TTL cache stored at `~/.cache/sirenspec/pricing.json`. On fetch failure (offline/rate-limited), execution falls back to a bundled snapshot shipped with the package. Never hardcoded inline.
_Avoid_: Hardcoding token prices in source code or data files owned by this repo.

**Streaming Execution**:
The executor exposes two functions: `execute()` (returns a complete trace dict — used by tests and programmatic callers) and `execute_streaming()` (async generator that yields a typed `NodeEvent` after each node completes, followed by a `SummaryEvent`). The CLI uses `execute_streaming()`; nothing else changes.
_Avoid_: Adding rendering callbacks to `execute()` or making `execute()` aware of the CLI layer.

**TokenUsage**:
A typed dataclass (`prompt_tokens: int`, `completion_tokens: int`, `total` computed property) returned by `LLMProvider.last_token_usage`. Replaces the bare `last_token_count: int` property. Used by the executor, trace, cost-cap guardrail, and CLI summary line.
_Avoid_: Passing prompt and completion token counts as separate ints across module boundaries.

**GuardrailViolation**:
The exception raised by a guardrail's `check_input` / `check_output` methods when a policy is violated. A subclass of `GuardrailError` (which is a subclass of `SirenSpecError`), so callers can catch at any level of the hierarchy.
_Avoid_: Treating it as a standalone exception outside the SirenSpecError hierarchy.

## Flagged ambiguities

- "example" and "recipe" were used interchangeably — resolved: **Recipe** is the canonical term for a file in the cookbook.
- "v0.1 constraints (sequential only)" in issue #24 are stale — resolved: recipes should use all currently available node types (agent, tool, swrm, conditional edges).
