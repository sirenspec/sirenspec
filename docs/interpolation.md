# Template Interpolation

SirenSpec uses `{{ expr }}` syntax to embed dynamic values in workflow YAML.  Expressions are
resolved at **node execution time** — values from upstream nodes are available once those nodes
complete.

---

## Syntax

### Basic placeholder

```yaml
prompt: "Summarize this: {{ inputs.message }}"
```

### Default filter

Use `| default('value')` to supply a fallback when a key might be missing:

```yaml
prompt: "Context: {{ working.context | default('none provided') }}"
```

### Multiple placeholders per string

```yaml
prompt: "Analyze {{ node_a.output }} against {{ node_b.output }}"
```

---

## Namespaces

### `inputs.*` — Workflow input fields

Resolves against the workflow's top-level input.

```yaml
prompt: "Analyze: {{ inputs.message }}"
```

The `message` field is always present.  Additional fields defined in `input:` are also available.

---

### `env.*` — Environment variables

Reads `os.environ` **at node execution time**, not at parse time.

```yaml
config:
  url: "https://api.example.com?key={{ env.API_KEY }}"
```

> **Security note:** `env.*` values are **redacted** (`***`) in execution traces to prevent
> credentials from appearing in logs.  The resolved value is passed to the LLM as normal.

Unset variables raise `InterpolationError`.  Use a default if the variable is optional:

```yaml
system: "Mode: {{ env.DEBUG_MODE | default('production') }}"
```

---

### `node_id.output` — Canonical node output

Every node (agent, swrm, factory) writes its primary output to `working.{node_id}.output`
automatically.  Reference it in downstream nodes:

```yaml
prompt: "Summarize: {{ classifier.output }}"
```

---

### `node_id.*` — Arbitrary node context access

Any value written to the `working` dict is accessible by its path:

```yaml
# If the 'classify' node writes: working.classify.output.sentiment
prompt: "Sentiment was {{ classify.output.sentiment }}"
```

---

### `node_id.agents.agent_id.output` — SWRM sub-agent output

Inside a `swrm` synthesis prompt, each sub-agent's output is available:

```yaml
synthesis:
  prompt: |
    Sentiment: {{ analyze.agents.sentiment.output }}
    Risk:      {{ analyze.agents.risk.output }}
    Synthesize an investment recommendation.
```

---

### `item` — Loop iteration value (factory nodes)

Inside a `factory` node's `inputs:` values, `{{ item }}` refers to the current list element:

```yaml
nodes:
  execute:
    type: factory
    for_each: "{{ plan.output }}"
    inputs:
      task: "{{ item }}"
```

Using `{{ item }}` outside a factory loop raises `InterpolationError`.

---

### `index` — Loop iteration index (factory nodes)

Zero-based position of the current item in the list:

```yaml
inputs:
  task: "{{ item }}"
  position: "{{ index }}"
```

---

## Error Behaviour

Any unresolvable expression raises `InterpolationError` with three attributes:

| Attribute | Description |
|---|---|
| `expression` | The full `{{ expr }}` string (without braces) |
| `namespace` | The first segment of the path (e.g. `inputs`, `env`, `plan`) |
| `reason` | Human-readable description of what went wrong |

Example:

```
InterpolationError in '{{ plan.output }}' [plan]: Key 'output' not found
```

Use `| default('fallback')` to suppress the error and return a static value instead.

---

## Circular Reference Detection

At workflow load time, SirenSpec scans all prompt templates for `{{ node_id.* }}` references
and builds a dependency graph.  If node A's prompt references node B **and** node B's prompt
references node A, loading raises `InterpolationError` with `namespace="circular_ref"`.

This check runs automatically in `load_workflow()` — no configuration needed.

---

## Security

- `env.*` values are **never** written to execution traces.  The trace shows `***` instead.
- `when:` edge conditions use a separate, restricted `eval()` namespace — not the interpolation engine.
- YAML is loaded with `SafeLoader`, so no code can be injected via YAML tags.

---

## Examples

### Sequential pipeline with node output chaining

```yaml
nodes:
  classify:
    agent: classifier_agent
    writes: working.classify.output

  reply:
    agent: reply_agent   # system: "The intent was: {{ classify.output }}"
    writes: output.reply
```

### SWRM synthesis referencing sub-agent outputs

```yaml
nodes:
  analyze:
    type: swrm
    agents:
      - id: sentiment
        provider: openai
        model: gpt-4o-mini
        prompt: "Analyze sentiment: {{ inputs.message }}"
      - id: risk
        provider: anthropic
        model: claude-haiku-4-5-20251001
        prompt: "Identify risks: {{ inputs.message }}"
    synthesis:
      provider: anthropic
      model: claude-haiku-4-5-20251001
      prompt: |
        Sentiment: {{ analyze.agents.sentiment.output }}
        Risk:      {{ analyze.agents.risk.output }}
        Write a recommendation.
```

### Factory node with loop variables

```yaml
nodes:
  plan:
    agent: planner
    writes: working.plan.output     # outputs JSON list: ["task A", "task B"]

  execute:
    type: factory
    agent: worker
    for_each: "{{ plan.output }}"
    inputs:
      task: "{{ item }}"
      position: "{{ index }}"
    concurrency: 4
    writes: working.execute.outputs

  aggregate:
    agent: summarizer               # system: "Summarize: {{ execute.output }}"
    writes: output.report
```

### Environment variable injection

```yaml
agents:
  api_caller:
    model: "openai:gpt-4o-mini"
    system: |
      You have access to the analytics API.
      Auth header: Bearer {{ env.ANALYTICS_API_KEY }}
```
