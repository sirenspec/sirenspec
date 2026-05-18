<div align="center">
  <h1>SirenSpec</h1>
  <img src="docs/logo/crest.svg" alt="Crest, the SirenSpec mascot" width="180"/>

  [![CI](https://github.com/TJLSmith0831/sirenspec/actions/workflows/ci.yml/badge.svg)](https://github.com/TJLSmith0831/sirenspec/actions/workflows/ci.yml)
</div>

YAML-first agent orchestration SDK. Define multi-agent workflows in human-readable YAML and execute them against OpenAI or Anthropic backends.

## Quick Start

```bash
pip install sirenspec
# or
uv add sirenspec
```

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...

sirenspec run docs/cookbook/simple-agent/workflow.yaml
```

## Setup (development)

```bash
uv sync --extra dev
source .venv/bin/activate
```

Python 3.13 is required.

## CLI

### `sirenspec run`

Execute a workflow and print a JSON execution trace to stdout.

```bash
sirenspec run workflow.yaml
sirenspec run workflow.yaml --input "What is the speed of light?"
sirenspec run workflow.yaml | jq -r '.output[]'
```

Options:
- `--input / -i` — User message (overrides `input.message` in the YAML)

Exit code `0` on success, `1` on failure.

### `sirenspec validate`

Validate a workflow YAML file without executing it.

```bash
sirenspec validate workflow.yaml
# ✓ workflow.yaml is valid (2 agents, 2 nodes)
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

Credentials are read from environment variables:

| Provider | URI format | Environment variable |
|----------|-----------|----------------------|
| OpenAI | `openai:gpt-4o-mini` | `OPENAI_API_KEY` |
| Anthropic | `anthropic:claude-haiku-4-5-20251001` | `ANTHROPIC_API_KEY` |
| Ollama | `ollama:llama3` | _(none required)_ |

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

```yaml
nodes:
  classify:
    agent: my_agent
    writes: working.intent
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

## Guardrails

| Name | Behaviour |
|------|-----------|
| `injection` | Detects prompt-injection patterns |
| `length` | Truncates output to 4 000 chars (configurable) |

Specify at the workflow level (`guardrails:`) or per-agent. An empty list (`[]`) disables all guardrails.

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

```python
import asyncio
from sirenspec.yaml.parser import load_workflow
from sirenspec.core.executor import execute

workflow = load_workflow("workflow.yaml")
trace = asyncio.run(execute(workflow, user_input="Hello"))
print(trace["output"])
```

## Cookbook

See [`docs/cookbook/`](docs/cookbook/) for 12 runnable examples:

| Example | What it demonstrates |
|---------|----------------------|
| [simple-agent](docs/cookbook/simple-agent/) | Single agent, minimal config |
| [sequential-pipeline](docs/cookbook/sequential-pipeline/) | Two-node chain |
| [conditional-pipeline](docs/cookbook/conditional-pipeline/) | `when:` edge routing |
| [telephone-game](docs/cookbook/telephone-game/) | Semantic drift across 5 hops |
| [adversarial-pair](docs/cookbook/adversarial-pair/) | Debate + judge pattern |
| [blind-code-review](docs/cookbook/blind-code-review/) | Multi-turn code refinement |
| [compression-gauntlet](docs/cookbook/compression-gauntlet/) | 4-round summarisation loop |
| [graphic-design-firm](docs/cookbook/graphic-design-firm/) | 5-node creative pipeline |
| [news-desk](docs/cookbook/news-desk/) | Reporter → editor → publisher chain |
| [1000-monkeys](docs/cookbook/1000-monkeys/) | Swrm fan-out + curator synthesis |
| [market-analysis](docs/cookbook/market-analysis/) | Parallel specialist agents + synthesis |
| [pr-summarizer](docs/cookbook/pr-summarizer/) | HTTP tool node + LLM summariser |
