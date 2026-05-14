# SirenSpec

YAML-first agent orchestration SDK. Define multi-agent workflows in human-readable YAML and execute them against OpenAI or Anthropic backends.

## Quick Start

```bash
pip install sirenspec
# or
uv add sirenspec

export OPENAI_API_KEY=sk-...
sirenspec run examples/simple-agent.yaml
```

## Setup (development)

```bash
uv sync --extra dev
source .venv/bin/activate
```

Python 3.13 is required.

## CLI Commands

### `sirenspec run`

Execute a workflow and print a JSON execution trace to stdout.

```bash
sirenspec run workflow.yaml
sirenspec run workflow.yaml --input "What is the speed of light?"
```

Options:
- `--input / -i` — User message (overrides `input.message` in the YAML)

### `sirenspec validate`

Validate a workflow YAML file without executing it.

```bash
sirenspec validate workflow.yaml
# ✓ workflow.yaml is valid (2 agents, 2 nodes)
```

Exit code `0` on success, `1` on failure.

## YAML Workflow Format

```yaml
version: "0.1"

agents:
  assistant:
    model: "openai:gpt-4o-mini"        # provider:model URI
    system: "You are a helpful assistant."
    guardrails: ["injection", "length"] # optional agent-level override

nodes:
  answer:
    agent: assistant
    writes: output.reply               # dot-notation context path

edges:
  - from: classify                     # optional: control flow
    to: reply

input:
  message: "What is AI?"              # optional static input

state:
  working:
    seed: "initial value"             # optional initial state

guardrails:                            # global guardrails (default: ["injection"])
  - injection
  - length
```

### Required fields

| Field | Description |
|-------|-------------|
| `version` | Schema version (currently `"0.1"`) |
| `agents` | Named agent definitions with `model` and `system` |
| `nodes` | Named nodes binding an agent to an output path |

### Provider URIs

Credentials are read from environment variables:

| Provider | URI format | Environment variable |
|----------|-----------|----------------------|
| OpenAI | `openai:gpt-4o-mini` | `OPENAI_API_KEY` |
| Anthropic | `anthropic:claude-haiku-4-5-20251001` | `ANTHROPIC_API_KEY` |
| Ollama | `gemma4:31b-cloud` | `OLLAMA_API_KEY` |

### Context paths

Nodes write to dot-notation paths in the workflow context:

- `output.reply` — final output (included in the trace `output` field)
- `working.intent` — intermediate state readable by downstream nodes

## Guardrails

Built-in guardrails protect every agent call:

| Name | Behaviour |
|------|-----------|
| `injection` | Detects prompt-injection patterns (always on by default) |
| `length` | Truncates output to 4000 chars (configurable) |

Specify at the workflow level (`guardrails:`) or per-agent. An empty list (`[]`) disables all guardrails.

## SDK Usage

```python
import asyncio
from sirenspec import load_workflow, execute

workflow = load_workflow("workflow.yaml")
trace = asyncio.run(execute(workflow, user_input="Hello"))
print(trace["output"])
```

## Examples

See [examples/](examples/) for runnable reference workflows.

## Limitations & Phase 2 Roadmap

v0.1 implements sequential pipelines only. The following are deferred to Phase 2:

- **Conditional edges** — `when:` expressions are accepted but not evaluated
- **Parallel node execution** — all nodes run sequentially
- **Tool nodes / function calling** — agents respond with text only
- **Additional providers** — Azure, Vertex planned
- **Retry logic** — no automatic retry on provider failures
- **PII redaction / output moderation** — beyond built-in guardrails

> **Note**: `SafeLoader` prevents code injection from YAML files, but always audit untrusted workflows before running them.
