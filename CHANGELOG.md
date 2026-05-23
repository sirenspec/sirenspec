# Changelog

All notable changes to SirenSpec are documented here.
This project follows [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

### Features

- **Human-in-the-loop (`HumanNode`)** — new node type that pauses execution to collect input from a human operator (issue #76). Supports template-rendered prompts, configurable `timeout`, and `on_timeout` actions (`abort` / `skip` / `use_default`). Stdin is the default input source; a pluggable `human_input_fn` parameter on `execute()` and `execute_streaming()` enables webhook and test integrations.
- **Workflow `budget:` block** — declarative token, USD, and wall-clock ceilings on a workflow run (issue #77). Supports three `on_exceeded` actions: `abort` (raises `BudgetExceededError`), `warn` (logs and continues), and `skip_remaining` (finishes without further LLM calls). The full budget status — declared limits, observed totals, violations, and the skip flag — is embedded in the trace summary.
- **Per-node `max_tokens_per_call`** — agent nodes accept an optional `max_tokens_per_call` field that is forwarded to the provider so the LLM truncates its own response.
- **Two new cookbook recipes** — `content-approval` (HumanNode demo) and `budget-guarded` (workflow budget demo).

### API surface

- `HumanNode`, `BudgetConfig`, `HumanInputError` exported from the top-level package.
- `LLMProvider.complete()` / `LLMProvider.stream()` now accept an optional `max_tokens` kwarg (backward compatible — only forwarded when set).
- `SummaryEvent.budget` field added for streaming consumers.

---

## [0.1.0] — 2026-05-18

Initial public release.

### Features

- **YAML-first workflow engine** — define multi-agent pipelines as human-readable YAML documents
- **Node types** — `agent`, `swrm` (parallel fan-out/synthesis), `factory` (dynamic node generation), `tool` (HTTP and Python), `workflow` (nested sub-workflow execution)
- **LLM providers** — OpenAI, Anthropic, and Ollama adapters with a pluggable `Protocol`-based interface
- **Streaming output** — per-node streaming with Rich-formatted console display
- **Template interpolation** — reference upstream node outputs with `{{ node_id.output }}` syntax
- **Guardrails** — prompt injection detection, length limits, PII redaction, JSON Schema output validation, and cost cap enforcement
- **Retry policies** — configurable per-node retry with backoff
- **Token usage accounting** — per-node and per-run token tracking
- **CLI** — six commands: `run`, `validate`, `test`, `explain`, `render`, `init`
- **Cassette-based testing** — deterministic, reproducible agent tests via YAML fixture recording (`sirenspec test`)
- **JSON Schema artifact** — `sirenspec.schema.json` for IDE autocomplete on workflow files
- **12 cookbook recipes** — runnable examples from simple pipelines to adversarial agent pairs
