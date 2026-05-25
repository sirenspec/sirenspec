<div align="center">
  <h1>SirenSpec</h1>
  <img src="docs/logo/crest.svg" alt="Crest, the SirenSpec mascot" width="180"/>

  [![CI](https://github.com/sirenspec/sirenspec/actions/workflows/ci.yml/badge.svg)](https://github.com/sirenspec/sirenspec/actions/workflows/ci.yml)
</div>

YAML-first agent orchestration SDK. Define multi-agent workflows in human-readable YAML and execute them against OpenAI, Anthropic, or Ollama backends.

📚 Documentation: [docs.sirenspec.dev](https://docs.sirenspec.dev)

## Why SirenSpec

Most production "agents" are really just workflows — a fixed sequence of LLM calls with some branching. The problem isn't the logic. It's that the logic lives inside Python files nobody else on the team can read.

SirenSpec is a YAML format and runtime for AI workflows. Write the pipeline once. It's readable by your PM, diffable in git, validatable in CI, and auditable by anyone who needs to understand what your AI is doing — without reading code.

**It is not a dynamic agent framework.** No autonomous tool selection, no open-ended loops. For those, use LangGraph or the OpenAI Agents SDK. SirenSpec is for the 90% of AI workflows that should be deterministic by design.

## Installation

```bash
# curl — easiest, auto-detects uv / pipx / pip
curl -fsSL https://sirenspec.dev/install.sh | sh
```

Or install directly with your preferred Python tool:

```bash
uv add sirenspec          # uv
pipx install sirenspec    # pipx (isolated global install)
pip install sirenspec     # pip
```

Python 3.11 or later is required.

## Quick Start

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...

# Scaffold a workflow interactively
sirenspec init

# Or run a cookbook example
sirenspec run docs/cookbook/simple-agent/workflow.yaml
```

## Setup (development)

```bash
uv sync --extra dev
source .venv/bin/activate
```

The repository pins Python 3.13 in `.python-version` for development. Users running the published package only need Python 3.11 or later.

## CLI

SirenSpec ships six commands. See the [CLI Reference](https://docs.sirenspec.dev/cli-reference) for the full surface.

| Command | Purpose |
|---------|---------|
| `sirenspec init` | Interactive scaffolding for a new `workflow.yaml` and `.env.example`. |
| `sirenspec run` | Execute a workflow with streaming node output (or `--trace` for JSON). |
| `sirenspec validate` | Parse and schema-check a workflow without making LLM calls. |
| `sirenspec explain` | Print a dry-run execution plan (text or JSON) — no LLM calls. |
| `sirenspec render` | Render a workflow as a Mermaid diagram. |
| `sirenspec test` | Discover and run YAML test fixtures, with optional cassette replay. |

### `sirenspec run`

By default, each node renders as a Rich-formatted panel as it completes, followed by a summary line. Use `--trace` (or `--output json`) for a machine-readable JSON trace, and `--no-stream` to suppress live token streaming inside each panel.

```bash
sirenspec run workflow.yaml
sirenspec run workflow.yaml --input "What is the speed of light?"
sirenspec run workflow.yaml --trace                # full JSON trace to stdout
sirenspec run workflow.yaml --trace | jq '.output' # machine-readable output
sirenspec run workflow.yaml --no-stream            # panels without per-token streaming
```

Options:
- `--input / -i` — User message (overrides `input.message` in the YAML)
- `--trace` — Print full JSON trace to stdout (suppresses node panels)
- `--output` — Output format; use `json` for a raw JSON trace (equivalent to `--trace`)
- `--trace-file` — Write the JSON trace to a file alongside the streaming output
- `--quiet` — Suppress node panels; print only the summary
- `--no-stream` — Disable per-token streaming inside panels

Exit code `0` on success, `1` on failure.

### `sirenspec validate`

```bash
sirenspec validate workflow.yaml
# ✓ workflow.yaml is valid (2 agents, 2 nodes)
```

### `sirenspec init`

Interactively scaffold a workflow. Picks a template, prompts for a provider, and writes `workflow.yaml` plus `.env.example` ready to run.

```bash
sirenspec init                  # current directory
sirenspec init --output ./my-workflow
```

### `sirenspec explain`

Print a human-readable execution plan — node order, agents, guardrails, and edges — without making LLM calls. Pass `--format json` for machine-readable output.

```bash
sirenspec explain workflow.yaml
sirenspec explain workflow.yaml --format json
```

### `sirenspec render`

Render a workflow as a Mermaid diagram. Conditional edges are labelled with their `when:` expression.

```bash
sirenspec render workflow.yaml --target mermaid
sirenspec render workflow.yaml --target mermaid --output diagram.md
```

### `sirenspec test`

Run YAML test fixtures (`*.test.yaml`). Use `--record` / `--mock` with a cassette file to capture and replay LLM responses deterministically.

```bash
sirenspec test tests/
sirenspec test tests/ --record --cassette cassettes/responses.yaml
sirenspec test tests/ --mock   --cassette cassettes/responses.yaml
```

## YAML Workflow Format

```yaml
version: "0.1"
env_file: .env                           # optional: load API keys from a .env file

agents:
  assistant:
    model: "openai:gpt-4o-mini"          # provider:model URI
    system: "You are a helpful assistant."
    guardrails: ["injection", "length"]  # optional agent-level override

nodes:
  answer:
    agent: assistant
    writes: output.reply                 # dot-notation context path

edges:
  - from: classify                       # optional: control flow
    to: reply
    when: working.intent == "refund"     # optional: conditional edge

input:
  message: "What is AI?"                 # optional static default input

guardrails:                              # workflow-level guardrails
  - injection
  - length
```

### Provider URIs

Credentials are read from environment variables. All three built-in providers support token streaming.

| Provider | URI format | Environment variable | Streaming |
|----------|-----------|----------------------|-----------|
| OpenAI | `openai:gpt-4o-mini` | `OPENAI_API_KEY` | Yes |
| Anthropic | `anthropic:claude-haiku-4-5-20251001` | `ANTHROPIC_API_KEY` | Yes |
| Ollama | `ollama:llama3` | _(none required)_ | Yes |

### `env_file`

Point a workflow at a `.env` file (path relative to the workflow file) to load API keys automatically at run time. Variables already set in the environment take precedence.

```yaml
env_file: .env
```

Variables are set in `os.environ` before execution, so provider clients pick them up without any extra configuration.

### Context paths

Nodes write to dot-notation paths in the workflow context:

- `output.reply` — final output (included in the trace `output` field)
- `working.intent` — intermediate state readable by downstream nodes via `{{ working.intent }}`

### Template interpolation

Use `{{ expr }}` in system prompts and agent prompts to reference runtime values:

```yaml
{{ inputs.message }}              # original user input
{{ env.GITHUB_TOKEN }}            # environment variable
{{ node_id.output }}              # another node's output
{{ node_id.agents.x.output }}    # swrm sub-agent output
{{ value | default('fallback') }} # optional fallback
```

## Node Types

### Agent node

Classic single-agent node. Runs one LLM call and writes the output to a context path.

Streaming is **on by default** (`streaming: true`). Each agent node streams tokens to stdout when run via `sirenspec run`. Set `streaming: false` on individual nodes to opt out, or use `--no-stream` at the CLI level.

Guardrails always apply to the **fully assembled response** after streaming completes — they are not applied per-chunk.

```yaml
nodes:
  classify:
    agent: my_agent
    writes: working.intent
    streaming: true          # default — stream tokens to stdout
    retry:
      max_attempts: 3
      backoff: exponential
    on_failure:
      action: fallback
      fallback_node: handle_error
```

### Swrm node

Fan-out to multiple agents running concurrently, then optionally synthesise their outputs.

```yaml
nodes:
  analyze:
    type: swrm
    concurrency: 3
    on_failure: continue          # or abort
    agents:
      - id: sentiment
        provider: openai
        model: gpt-4o-mini
        prompt: "Analyze: {{ inputs.message }}"
      - id: risk
        provider: anthropic
        model: claude-haiku-4-5-20251001
        prompt: "List risks in: {{ inputs.message }}"
    synthesis:
      provider: anthropic
      model: claude-haiku-4-5-20251001
      prompt: |
        Sentiment: {{ analyze.agents.sentiment.output }}
        Risk: {{ analyze.agents.risk.output }}
        Produce a recommendation.
```

### Factory node

Dynamically spawns one agent instance per item in a runtime list.

```yaml
nodes:
  execute:
    type: factory
    agent: worker
    for_each: "{{ plan.output }}"   # must resolve to a JSON array
    inputs:
      task: "{{ item }}"
      index: "{{ index }}"
    concurrency: 4
    writes: working.results
```

### Tool node

Calls an HTTP endpoint or Python callable instead of an LLM.

```yaml
nodes:
  fetch:
    type: tool
    tool: http
    config:
      url: "https://api.example.com/data"
      method: GET
      headers:
        Authorization: "Bearer {{ env.API_TOKEN }}"
      timeout: 15
    output_key: data
```

### Workflow node

Executes another SirenSpec workflow inline as a single node. The sub-workflow's output is written back into the parent context.

```yaml
nodes:
  summarize:
    type: workflow
    ref: ./workflows/summarize.yaml
    inputs:
      topic: "{{ extract.output }}"
    writes: working.summary
```

### Human node

Pauses execution to collect input from a human operator. Consumes no LLM tokens. Supports a `timeout` with `on_timeout` actions (`abort` / `skip` / `use_default`).

```yaml
nodes:
  approve_draft:
    type: human
    prompt: |
      {{ draft.output }}

      Approve this draft? (yes/edit/reject)
    writes: working.approval
    timeout: 3600
    on_timeout: use_default
    default_output: "approved"
```

## Guardrails

| Name | Behaviour | Config |
|------|-----------|--------|
| `injection` | Detects prompt-injection patterns. Applied by default. | None |
| `length` | Truncates output to 4000 chars. | `max_chars`, `mode` |
| `pii` | Detects and redacts email, phone, SSN, and credit-card data. | `entities`, `action`, `replacement` |
| `schema` | Validates output as JSON against a JSON Schema Draft 7 dict. | `schema` (required) |
| `cost_cap` | Enforces token and/or USD ceilings across the run. | `max_tokens` and/or `max_usd`, `action` |

Specify at the workflow level (`guardrails:`) or per-agent. Configurable guardrails use a `{name, config}` form:

```yaml
guardrails:
  - injection
  - name: cost_cap
    config:
      max_usd: 5.0
      action: abort
```

An empty list (`[]`) disables all guardrails. See the [Guardrails docs](https://docs.sirenspec.dev/guardrails) for full configuration details.

## Budget controls

Cap the entire run's spend with a workflow-level `budget:` block. At least one ceiling must be set.

```yaml
budget:
  max_tokens: 50000
  max_cost_usd: 5.00
  max_duration_s: 300
  on_exceeded: abort    # abort | warn | skip_remaining
```

Per-call ceilings are available on agent nodes via `max_tokens_per_call`.

## Retry & on_failure

```yaml
nodes:
  answer:
    agent: assistant
    writes: output.reply
    retry:
      max_attempts: 3
      backoff: exponential    # exponential | linear | constant
      base_delay: 1.0
      on: ["429", "network_error"]
    on_failure:
      action: use_default     # abort | fallback | skip | use_default
      default_output: "Sorry, I could not process your request."
```

## SDK Usage

Everything is exported at the top level — see the [Python SDK docs](https://docs.sirenspec.dev/sdk) for the full surface.

```python
import asyncio
from sirenspec import load_workflow, execute

workflow = load_workflow("workflow.yaml")
trace = asyncio.run(execute(workflow, user_input="Hello"))
print(trace["output"])
```

Streaming, budget enforcement, nested workflows, and custom guardrails are all exported alongside `execute`:

```python
from sirenspec import (
    execute_streaming,            # async generator of per-node events
    NodeCompleteEvent, SummaryEvent,
    BudgetConfig, BudgetExceededError,
    HumanNode, HumanInputError,
    WorkflowRegistry,             # for named sub-workflow refs
    Guardrail, GuardrailViolation,
    LLMProvider, TokenUsage,
)
```

## Cookbook

See [`docs/cookbook/`](docs/cookbook/) for 21 runnable examples:

### Basics
| Example | What it demonstrates |
|---------|----------------------|
| [simple-agent](docs/cookbook/simple-agent/) | Single agent, minimal config |
| [sequential-pipeline](docs/cookbook/sequential-pipeline/) | Two-node chain |
| [conditional-pipeline](docs/cookbook/conditional-pipeline/) | `when:` edge routing |

### Multi-agent patterns
| Example | What it demonstrates |
|---------|----------------------|
| [adversarial-pair](docs/cookbook/adversarial-pair/) | Debate + judge pattern |
| [blind-code-review](docs/cookbook/blind-code-review/) | Multi-turn code refinement |
| [graphic-design-firm](docs/cookbook/graphic-design-firm/) | 5-node creative pipeline |
| [news-desk](docs/cookbook/news-desk/) | Reporter → editor → publisher chain |
| [content-moderation-pipeline](docs/cookbook/content-moderation-pipeline/) | Multi-stage moderation chain |

### Swrm & fan-out
| Example | What it demonstrates |
|---------|----------------------|
| [1000-monkeys](docs/cookbook/1000-monkeys/) | Swrm fan-out + curator synthesis |
| [market-analysis](docs/cookbook/market-analysis/) | Parallel specialist agents + synthesis |
| [email-triage](docs/cookbook/email-triage/) | Parallel triage with synthesised verdict |

### Factory & iteration
| Example | What it demonstrates |
|---------|----------------------|
| [changelog-annotator](docs/cookbook/changelog-annotator/) | Annotate each commit in a list |
| [github-issues-triage](docs/cookbook/github-issues-triage/) | Per-issue triage via factory |
| [grading-factory](docs/cookbook/grading-factory/) | Factory + per-item swrm |

### Guardrails & budgets
| Example | What it demonstrates |
|---------|----------------------|
| [structured-bug-reporter](docs/cookbook/structured-bug-reporter/) | JSON Schema guardrail enforcement |
| [budget-guarded](docs/cookbook/budget-guarded/) | Workflow `budget:` block in action |

### Human in the loop
| Example | What it demonstrates |
|---------|----------------------|
| [content-approval](docs/cookbook/content-approval/) | Human approval gate before publish |

### Stress tests
| Example | What it demonstrates |
|---------|----------------------|
| [telephone-game](docs/cookbook/telephone-game/) | Semantic drift across 5 hops |
| [compression-gauntlet](docs/cookbook/compression-gauntlet/) | 4-round summarisation loop |

### Tool nodes
| Example | What it demonstrates |
|---------|----------------------|
| [pr-summarizer](docs/cookbook/pr-summarizer/) | HTTP tool node + LLM summariser |
| [code-health-report](docs/cookbook/code-health-report/) | HTTP tool + multi-stage analysis |
