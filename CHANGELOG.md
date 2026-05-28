# Changelog

All notable changes to SirenSpec are documented here.
This project follows [Semantic Versioning](https://semver.org/).

---

## [0.1.2] — 2026-05-28

A reliability and authoring-ergonomics release driven by early-user feedback. No breaking changes.

### Added

- **Load-time workflow linter** (`core/lint.py`) — `load_workflow()` now runs a static linter that surfaces problems before execution:
  - `working_dot_node_id` (error) — rejects `{{ working.<node_id>.* }}` when `<node_id>` is a known node; the canonical form is `{{ <node_id>.output }}`.
  - `unknown_namespace` (warning) — flags top-level names that are neither a reserved namespace (`inputs`/`env`/`item`/`index`/`total`) nor a known node ID, catching typos before they raise `InterpolationError` at runtime.
- **`| json_or_default('...')` filter** — engages both when a key is missing *and* when the resolved value cannot be parsed as JSON, covering LLM outputs that return `""` or fenced markdown instead of a JSON array.
- **Safe builtins in `when:` expressions** — `len`, `bool`, `str`, `int`, `float`, `abs`, `min`, `max` are now available so authors can write conditions like `len(working.items) > 0` without a `NameError`. `__builtins__` remains otherwise blank.
- **`retry_on_guardrail` on `RetryPolicy`** — when `true`, output guardrail checks run inside the retry loop so a `GuardrailViolation` triggers another provider call rather than an immediate failure. `guardrail_violation` is also accepted as a retry trigger in `error_matches_policy`.
- **Factory `inputs:` exposed as `{{ inputs.key }}`** — resolved factory `inputs:` values are now available as named template vars in the spawned agent's system prompt, in addition to the existing user-message join. Strictly additive and backward-compatible.

### Changed

- **`| default('...')` now fires on empty strings** — previously it engaged only on `InterpolationError` (missing key); it now also fires when the resolved value is `""`.
- **`for_each` accepts native lists and fenced JSON** — `resolve_to_list` tries native list passthrough first, then plain JSON, then strips ` ```json ` fences, so upstream tool outputs and fenced agent outputs work as loop sources without post-processing.
- **`env_file:` is loaded eagerly in `load_workflow()`** — environment values are applied before the workflow is returned, so provider clients that read `os.environ` at init time see the correct values regardless of call order. An `EnvFileShadowWarning` is emitted when an `env_file` key is already present in `os.environ` as an empty string.
- **CLI trace output suppresses branch-not-taken nodes** — `NodeCompleteEvent` gains a `skip_reason` field (`branch_not_taken` / `budget_exceeded`); branches that were intentionally not traversed no longer appear as `[-]` noise in the terminal, while `budget_exceeded` skips still print clearly.

### Fixed

- **Python tool config interpolation** — `interpolate_tool_config` now resolves `{{ ... }}` templates inside `PythonToolConfig.module`, `.function`, and `.args` (recursively through nested dicts and lists), matching HTTP tool config behaviour. This unblocks dynamic dispatch where a Python tool's inputs depend on prior node outputs or workflow inputs.

## [0.1.1] — 2026-05-23

Maintenance release — CI/CD and release-tooling fixes, README rebrand copy, and badge/link updates. No library behaviour changes.

## [0.1.0] — 2026-05-23

Initial public release.

### Features

- **YAML-first workflow engine** — define multi-agent pipelines as human-readable YAML documents
- **Node types** — `agent`, `swrm` (parallel fan-out/synthesis), `factory` (dynamic node generation), `tool` (HTTP and Python), `workflow` (nested sub-workflow execution), and `human` (human-in-the-loop pauses)
- **LLM providers** — OpenAI, Anthropic, and Ollama adapters with a pluggable `Protocol`-based interface
- **Streaming output** — per-node streaming with Rich-formatted console display and a public `execute_streaming` async generator that yields `NodeCompleteEvent` / `SummaryEvent`
- **Template interpolation** — reference upstream node outputs with `{{ node_id.output }}` syntax
- **Guardrails** — prompt injection detection, length limits, PII redaction, JSON Schema output validation, and cost cap enforcement
- **Workflow `budget:` block** — declarative token, USD, and wall-clock ceilings on a run, with `abort` / `warn` / `skip_remaining` actions. Budget state is exposed in `when:` expressions as `_budget` and embedded in the trace summary.
- **Per-node `max_tokens_per_call`** — optional per-call ceiling forwarded to the provider so the LLM truncates its own response.
- **Retry policies** — configurable per-node retry with backoff
- **Token usage accounting** — per-node and per-run token tracking via the public `TokenUsage` dataclass
- **CLI** — six commands: `init`, `run`, `validate`, `explain`, `render`, `test`
- **Cassette-based testing** — deterministic, reproducible agent tests via YAML fixture recording (`sirenspec test --record` / `--mock`)
- **JSON Schema artifact** — `sirenspec.schema.json` for IDE autocomplete on workflow files (the `version` field is now constrained to `"0.1"`)
- **21 cookbook recipes** — runnable examples spanning sequential pipelines, swrm fan-out, factory iteration, tool nodes, human-in-the-loop, budgets, and guardrail demos

### Public API

The top-level `sirenspec` package exports:

- Execution: `execute`, `execute_streaming`, `load_workflow`
- Models: `Workflow`, `BudgetConfig`, `HumanNode`, `GuardrailSpec`, `WorkflowRegistry`
- Streaming events: `NodeCompleteEvent`, `SummaryEvent`
- Extension points: `LLMProvider`, `Guardrail`, `WorkflowGuardrail`
- Value types: `TokenUsage`
- Exceptions: `SirenSpecError`, `ProviderError`, `RetryExhaustedError`, `GuardrailError`, `GuardrailViolation`, `BudgetExceededError`, `HumanInputError`, `ValidationError`, `SwrmAgentError`, `ToolError`
