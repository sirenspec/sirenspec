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
A validation layer applied to agent output before it is written to the context — e.g. injection detection, length limits.
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

## Flagged ambiguities

- "example" and "recipe" were used interchangeably — resolved: **Recipe** is the canonical term for a file in the cookbook.
- "v0.1 constraints (sequential only)" in issue #24 are stale — resolved: recipes should use all currently available node types (agent, tool, swrm, conditional edges).
