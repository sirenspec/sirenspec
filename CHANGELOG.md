# Changelog

All notable changes to SirenSpec are documented here.
This project follows [Semantic Versioning](https://semver.org/).

---

## [0.1.0] — 2025-05-17

Initial public release.

### Features

- **YAML-first workflow engine** — define multi-agent pipelines as human-readable YAML documents
- **Node types** — `agent`, `swrm` (parallel fan-out/synthesis), `factory` (dynamic node generation), `tool` (HTTP)
- **LLM providers** — OpenAI, Anthropic, and Ollama adapters with a pluggable `Protocol`-based interface
- **Streaming output** — per-node streaming with Rich-formatted console display
- **Template interpolation** — reference upstream node outputs with `{{ node_id.output }}` syntax
- **Guardrails** — prompt injection detection, length limits, PII redaction, JSON Schema output validation, and cost cap enforcement
- **Retry policies** — configurable per-node retry with backoff
- **Token usage accounting** — per-node and per-run token tracking
- **CLI** — five commands: `run`, `validate`, `test`, `explain`, `render`
- **Cassette-based testing** — deterministic, reproducible agent tests via YAML fixture recording (`sirenspec test`)
- **JSON Schema artifact** — `sirenspec.schema.json` for IDE autocomplete on workflow files
- **12 cookbook recipes** — runnable examples from simple pipelines to adversarial agent pairs
