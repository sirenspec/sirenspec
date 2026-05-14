# SirenSpec Examples

## Prerequisites

```bash
uv sync --extra dev
source .venv/bin/activate
export OPENAI_API_KEY=sk-...       # for openai: providers
export ANTHROPIC_API_KEY=sk-...    # for anthropic: providers
export OLLAMA_API_KEY=ollama       # for ollama: providers
```

## simple-agent.yaml

A single-agent workflow that answers a static question.

```bash
sirenspec run examples/simple-agent.yaml
# or override the question:
sirenspec run examples/simple-agent.yaml --input "What is the speed of light?"
```

**Expected output**: A JSON trace with one node entry and `output.reply` containing the answer.

## sequential-pipeline.yaml

A two-agent pipeline: `classify` detects the user's intent, then `reply` crafts a response.

```bash
sirenspec run examples/sequential-pipeline.yaml --input "My order hasn't arrived yet."
```

**Expected output**: A JSON trace with two node entries. The classifier writes `working.intent`; the replier reads that value and writes `output.reply`.

## Validating examples

```bash
sirenspec validate examples/simple-agent.yaml
sirenspec validate examples/sequential-pipeline.yaml
```
