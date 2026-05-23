# Changelog

All notable changes to SirenSpec are documented here.
This project follows [Semantic Versioning](https://semver.org/).

---

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
