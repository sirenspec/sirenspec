---
name: sirenspec
description: Build SirenSpec YAML workflows — define agents, nodes, edges, guardrails, tool nodes, and swrm fan-outs for multi-agent AI pipelines. Use when the user wants to create, edit, validate, or run a SirenSpec workflow, or when code imports/invokes sirenspec.
---

# SirenSpec

SirenSpec executes YAML-defined multi-agent workflows against OpenAI, Anthropic, or Ollama backends. A workflow declares **agents** (LLM actors), **nodes** (execution units), **edges** (ordered connections), and **guardrails** (validation layers).

## Quick start

```yaml
version: "0.1"

agents:
  assistant:
    model: "openai:gpt-4o-mini"
    system: "You are a helpful assistant."

nodes:
  answer:
    agent: assistant
    writes: output.reply

input:
  message: "What is the capital of France?"
```

```bash
sirenspec run workflow.yaml
sirenspec validate workflow.yaml      # schema check + load-time linter, without running
sirenspec run workflow.yaml --trace   # full JSON execution trace
sirenspec run workflow.yaml --input "Override message"
```

`load_workflow()` / `validate` run a static linter: it errors on
`{{ working.<node_id>.* }}` references and warns on unknown template namespaces
(likely typos) before any LLM call is made.

## Install & auth

```bash
uv add sirenspec   # or: pip install sirenspec
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
```

## Model URIs

| Provider | URI format | Examples |
|----------|-----------|---------|
| OpenAI | `openai:<model>` | `openai:gpt-4o-mini`, `openai:gpt-4o` |
| Anthropic | `anthropic:<model>` | `anthropic:claude-haiku-4-5-20251001`, `anthropic:claude-sonnet-4-6` |
| Ollama | `ollama:<model>` | `ollama:llama3.2` |

## Node types

| Type | Purpose | Key fields |
|------|---------|-----------|
| agent (default) | LLM call | `agent`, `writes` |
| `type: tool` / `tool: http` | HTTP request | `config.url`, `config.method`, `output_key` |
| `type: tool` / `tool: python` | Python callable | `config.module`, `config.function` |
| `type: swrm` | Parallel fan-out | `agents`, `synthesis`, `concurrency` |

## Context paths (`writes:`)

- `output.*` — included in the final trace output
- `working.*` — intermediate state, readable downstream but not in final output

## Template interpolation (`{{ expr }}`)

- `{{ inputs.message }}` — workflow input
- `{{ node_id.output }}` — upstream agent output (canonical form)
- `{{ node_id.output_key }}` — upstream tool output (e.g. `{{ fetch.diff }}`)
- `{{ env.VAR_NAME }}` — environment variable (redacted in traces)
- `{{ working.path }}` — custom context paths you wrote to (e.g. seeded `state`)
- `{{ value | default('fallback') }}` — fallback on missing key or empty string
- `{{ value | json_or_default('[]') }}` — fallback on missing key, empty string, or invalid JSON

> Reference an upstream node by its ID (`{{ node_id.output }}`), **not** via the
> internal `working` namespace. The load-time linter rejects
> `{{ working.<node_id>.* }}` and tells you the canonical form.

## Best practices

- Write system prompts as `|` multiline literals for readability
- Use `working.*` paths for intermediate state; reserve `output.*` for final results
- Declare `guardrails: [injection]` at the workflow level (it's the default — don't omit)
- Use `schema` guardrail when an agent must return structured JSON
- Set `defaults.retry` at the workflow level for production resilience
- Use `when:` on edges for conditional branching — keep expressions simple (one comparison). Safe builtins (`len`, `bool`, `str`, `int`, `float`, `abs`, `min`, `max`) are available, e.g. `len(working.items) > 0`
- Prefer `type: swrm` over manual parallel wiring for concurrent multi-agent patterns
- Set `retry.retry_on_guardrail: true` when an agent must satisfy an output guardrail (e.g. `schema`) — a violation re-runs the LLM call instead of failing
- For `for_each` factories, the source may be a native list, plain JSON, or fenced ```json``` output — no manual unwrapping needed

## Reference files

- [REFERENCE.md](REFERENCE.md) — complete field-by-field YAML reference and guardrail config
- [EXAMPLES.md](EXAMPLES.md) — full workflow examples for common patterns
