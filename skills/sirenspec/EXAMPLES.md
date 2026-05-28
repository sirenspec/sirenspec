# SirenSpec Workflow Examples

## 1. Single-agent workflow

```yaml
version: "0.1"

agents:
  assistant:
    model: "openai:gpt-4o-mini"
    system: "You are a helpful assistant. Answer concisely."

nodes:
  answer:
    agent: assistant
    writes: output.reply

input:
  message: "What is the speed of light?"
```

---

## 2. Sequential pipeline (classify → reply)

```yaml
version: "0.1"

agents:
  classifier:
    model: "openai:gpt-4o-mini"
    system: |
      Classify the user's message as "question" or "complaint".
      Reply with ONLY one word.

  responder:
    model: "anthropic:claude-haiku-4-5-20251001"
    system: "You are a support agent. Reply helpfully in 2–3 sentences."

nodes:
  classify:
    agent: classifier
    writes: working.intent

  reply:
    agent: responder
    writes: output.reply

edges:
  - from: classify
    to: reply
```

---

## 3. Conditional branching

```yaml
version: "0.1"

agents:
  triage_agent:
    model: "openai:gpt-4o-mini"
    system: |
      Classify the user's message as "refund" or "general".
      Reply with ONLY one word.

  refund_handler:
    model: "openai:gpt-4o-mini"
    system: "You are a refund specialist. Acknowledge and explain the process."

  general_handler:
    model: "openai:gpt-4o-mini"
    system: "You are a support agent. Answer helpfully."

nodes:
  triage:
    agent: triage_agent
    writes: working.triage.intent

  handle_refund:
    agent: refund_handler
    writes: output.reply

  handle_general:
    agent: general_handler
    writes: output.reply

edges:
  - from: triage
    to: handle_refund
    when: working.triage.intent == "refund"

  - from: triage
    to: handle_general
    when: working.triage.intent == "general"

guardrails:
  - injection
  - length
```

---

## 4. Structured JSON output with schema guardrail

```yaml
version: "0.1"

agents:
  extractor:
    model: "openai:gpt-4o-mini"
    system: |
      Extract the person's name and age from the text.
      Reply ONLY with valid JSON: {"name": "...", "age": <integer>}
    guardrails:
      - name: schema
        config:
          schema:
            type: "object"
            properties:
              name:
                type: "string"
              age:
                type: "integer"
                minimum: 0
                maximum: 150
            required: ["name", "age"]

nodes:
  extract:
    agent: extractor
    writes: output.person
```

---

## 5. Workflow-level retry defaults

```yaml
version: "0.1"

defaults:
  retry:
    max_attempts: 3
    backoff: exponential
    base_delay: 1.0
    max_delay: 30.0
    jitter: true
    on: [429, 503, network_error]
  on_failure:
    action: abort

agents:
  assistant:
    model: "anthropic:claude-haiku-4-5-20251001"
    system: "You are a helpful assistant."

nodes:
  answer:
    agent: assistant
    writes: output.reply
    # inherits defaults.retry and defaults.on_failure
```

---

## 6. Tool node — HTTP fetch into agent prompt

```yaml
version: "0.1"

agents:
  summarizer:
    model: "openai:gpt-4o-mini"
    system: |
      Summarize the following content in 3 bullet points:
      {{ fetch_page.content }}

nodes:
  fetch_page:
    type: tool
    tool: http
    config:
      url: "https://example.com/api/article"
      method: GET
      headers:
        Authorization: "Bearer {{ env.API_TOKEN }}"
      timeout: 10
    output_key: content

  summarize:
    agent: summarizer
    writes: output.summary

edges:
  - from: fetch_page
    to: summarize
```

---

## 7. Swrm parallel fan-out with synthesis

```yaml
version: "0.1"

agents: {}  # swrm nodes define their agents inline

nodes:
  analyze:
    type: swrm
    concurrency: 3
    agents:
      - id: sentiment
        provider: openai
        model: gpt-4o-mini
        prompt: |
          Analyze the market sentiment in:
          {{ inputs.message }}
      - id: risk
        provider: anthropic
        model: claude-haiku-4-5-20251001
        prompt: |
          Identify the top 3 risks in:
          {{ inputs.message }}
      - id: opportunity
        provider: openai
        model: gpt-4o-mini
        prompt: |
          Find the top 3 opportunities in:
          {{ inputs.message }}
    synthesis:
      provider: anthropic
      model: claude-haiku-4-5-20251001
      prompt: |
        Sentiment: {{ analyze.agents.sentiment.output }}
        Risks: {{ analyze.agents.risk.output }}
        Opportunities: {{ analyze.agents.opportunity.output }}

        Write a 3-paragraph investment recommendation.

  write_report:
    agent: reporter
    writes: output.report

edges:
  - from: analyze
    to: write_report

# reporter agent needs to be defined if write_report uses it
```

---

## 8. Mixed providers in one workflow

```yaml
version: "0.1"

agents:
  classifier:
    model: "openai:gpt-4o-mini"
    system: "Classify intent as 'question' or 'complaint'. One word only."

  responder:
    model: "anthropic:claude-sonnet-4-6"
    system: "You are a senior support agent. Compose a thorough, empathetic reply."

nodes:
  classify:
    agent: classifier
    writes: working.intent

  reply:
    agent: responder
    writes: output.reply

edges:
  - from: classify
    to: reply

guardrails:
  - injection
```

---

## 9. Seeded state + per-node fallback

```yaml
version: "0.1"

state:
  working:
    context: "No prior context."
  output:
    reply: "Service unavailable."  # default if all nodes skip

agents:
  assistant:
    model: "anthropic:claude-haiku-4-5-20251001"
    system: |
      Context: {{ working.context }}
      Answer the user's question.

nodes:
  answer:
    agent: assistant
    writes: output.reply
    retry:
      max_attempts: 2
      backoff: linear
      base_delay: 1.0
      on: [429, 503, network_error]
    on_failure:
      action: use_default
      default_output: "I'm unable to answer right now. Please try again later."
```
