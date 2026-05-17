# SirenSpec YAML Reference

## Top-level structure

```yaml
version: "0.1"    # required — only "0.1" supported
agents: {}         # required
nodes: {}          # required
edges: []          # optional
input: {}          # optional — static default input
state: {}          # optional — seed the context before execution
defaults: {}       # optional — workflow-wide retry/failure defaults
guardrails: []     # optional — defaults to ["injection"] if omitted
```

---

## `agents`

```yaml
agents:
  my_agent:
    model: "openai:gpt-4o-mini"   # required — provider:model URI
    system: "You are helpful."    # required — system prompt
    guardrails: ["injection"]     # optional — overrides workflow-level list for this agent
```

An agent's `guardrails` list **replaces** (does not merge with) the workflow-level list.

---

## `nodes`

### Agent node (default)

```yaml
nodes:
  classify:
    agent: triage_agent           # required — agent ID from agents map
    writes: working.intent        # required — dot-notation context path
    retry:                        # optional — overrides defaults.retry
      max_attempts: 3
      backoff: exponential
      base_delay: 1.0
    on_failure:                   # optional — overrides defaults.on_failure
      action: fallback
      fallback_node: safe_classify
```

### Tool node — HTTP

```yaml
nodes:
  fetch_data:
    type: tool
    tool: http
    config:
      url: "https://api.example.com/data"
      method: GET                  # GET | POST
      headers:
        Authorization: "Bearer {{ env.API_TOKEN }}"
      body: '{"key": "value"}'    # POST only
      timeout: 15                  # seconds, default 10
    output_key: data               # stored at working.fetch_data.data
    retry: 2                       # extra attempts on failure
    on_failure: skip               # "raise" (default) | "skip"
```

### Tool node — Python

```yaml
nodes:
  transform:
    type: tool
    tool: python
    config:
      module: "mypackage.utils"    # dotted import path
      function: "process"          # callable name
      kwargs:                      # passed as keyword arguments
        value: "{{ working.prev.result }}"
    output_key: result
```

### Swrm node (parallel fan-out)

```yaml
nodes:
  analyze:
    type: swrm
    concurrency: 3                  # max parallel agents; default = all
    on_failure: abort               # "abort" (default) | "continue"
    agents:
      - id: sentiment               # unique within swrm
        provider: openai
        model: gpt-4o-mini
        prompt: "Analyze sentiment: {{ inputs.message }}"
        guardrails: [injection]
      - id: risk
        provider: anthropic
        model: claude-haiku-4-5-20251001
        prompt: "Identify risks: {{ inputs.message }}"
    synthesis:                      # optional — runs after all agents complete
      provider: anthropic
      model: claude-haiku-4-5-20251001
      prompt: |
        Sentiment: {{ analyze.agents.sentiment.output }}
        Risk: {{ analyze.agents.risk.output }}
        Write a summary.
```

---

## `edges`

```yaml
edges:
  - from: classify          # required — source node ID
    to: handle_refund       # required — target node ID
    when: working.intent == "refund"   # optional — Python expression
```

**`when` expression rules:**
- Evaluated after source node completes
- Namespace: `working`, `output`, `true`, `false`, `null`
- No imports, no built-ins
- Errors → treated as `false` (edge not traversed)
- Edges without `when` always traverse

---

## `defaults`

```yaml
defaults:
  retry:
    max_attempts: 3
    backoff: exponential       # constant | linear | exponential
    base_delay: 1.0            # seconds before first retry
    max_delay: 60.0            # cap on computed delay
    jitter: true               # adds ±20% random variation
    on: [429, network_error]   # trigger conditions (HTTP codes or "network_error")
  on_failure:
    action: abort              # abort | fallback | skip | use_default
```

Per-node `retry`/`on_failure` fully replaces defaults — no field-level merging.

### `on_failure` actions

| Action | Behaviour |
|--------|-----------|
| `abort` | Raises `RetryExhaustedError`, stops workflow |
| `fallback` | Routes to `fallback_node`; failed node marked skipped |
| `skip` | Skips node silently; downstream receives nothing |
| `use_default` | Writes `default_output` to `writes` path, continues |

---

## `guardrails`

### Simple (zero-config)

```yaml
guardrails:
  - injection    # default — prompt injection detection (case-insensitive)
  - length       # truncates output at 4000 chars (mode: truncate)
```

### With configuration

```yaml
guardrails:
  - name: length
    config:
      max_chars: 2000
      mode: raise              # truncate (default) | raise

  - name: schema
    config:
      schema:
        type: "object"
        properties:
          intent:
            type: "string"
            enum: ["refund", "general"]
        required: ["intent"]
```

| Guardrail | Checks | Config keys |
|-----------|--------|-------------|
| `injection` | input + output | none |
| `length` | output only | `max_chars` (default 4000), `mode` (`truncate`\|`raise`) |
| `schema` | output only | `schema` (required — JSON Schema Draft 7 dict) |

---

## `state`

Seed context values before any node executes:

```yaml
state:
  working:
    seed_value: "initial"
  output:
    default_reply: "No answer yet."
```

---

## Template interpolation

| Namespace | Example | Notes |
|-----------|---------|-------|
| `inputs.*` | `{{ inputs.message }}` | Workflow input fields |
| `working.*` | `{{ working.triage.intent }}` | Intermediate context |
| `output.*` | `{{ output.reply }}` | Final output context |
| `env.*` | `{{ env.APP_ENV }}` | Env var; redacted in traces |
| `<swrm_id>.agents.<agent_id>.output` | `{{ analyze.agents.risk.output }}` | Swrm agent outputs |
| `\| default(v)` | `{{ working.x \| default('none') }}` | Safe fallback |

---

## Python SDK

```python
import asyncio
from sirenspec import load_workflow, execute
from sirenspec.core.executor import execute_streaming

workflow = load_workflow("workflow.yaml")

# One-shot — returns full trace dict
trace = asyncio.run(execute(workflow, user_input="Hello"))
print(trace["output"])       # {"reply": "..."}
print(trace["summary"]["status"])  # "success"

# Streaming — yields NodeEvent then SummaryEvent
async def stream():
    async for event in execute_streaming(workflow, "Hello"):
        print(event)
```

---

## Exceptions

```python
from sirenspec.exceptions import (
    SirenSpecError,       # base
    ProviderError,        # invalid URI, unknown provider, API failure
    RetryExhaustedError,  # all retries failed (subclass of ProviderError)
    GuardrailError,       # base guardrail exception
    GuardrailViolation,   # policy violated (subclass of GuardrailError)
    ValidationError,      # workflow schema/model validation
)
```
