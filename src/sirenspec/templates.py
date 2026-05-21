"""Workflow scaffold templates for ``sirenspec init``."""

from __future__ import annotations

from dataclasses import dataclass

SIMPLE_AGENT: str = """\
version: "0.1"

agents:
  assistant:
    model: "__MODEL__"
    system: "You are a helpful assistant."

nodes:
  answer:
    agent: assistant
    writes: output.reply
__GUARDRAILS__
input:
  message: "Hello, how can you help me?"
"""

SEQUENTIAL: str = """\
version: "0.1"

agents:
  processor:
    model: "__MODEL__"
    system: "You are a text processor. Summarize the input concisely."

  formatter:
    model: "__MODEL__"
    system: "You are a formatter. Present the processed content clearly."

nodes:
  process:
    agent: processor
    writes: working.summary

  format:
    agent: formatter
    writes: output.result
__GUARDRAILS__
edges:
  - from: process
    to: format

input:
  message: "Enter your text here."
"""

CONDITIONAL: str = """\
version: "0.1"

agents:
  triage:
    model: "__MODEL__"
    system: "Classify the user request. Respond with exactly one word: question or task."

  question_handler:
    model: "__MODEL__"
    system: "Answer the user's question clearly and helpfully."

  task_handler:
    model: "__MODEL__"
    system: "Complete the requested task step by step."

nodes:
  classify:
    agent: triage
    writes: working.intent

  handle_question:
    agent: question_handler
    writes: output.reply

  handle_task:
    agent: task_handler
    writes: output.reply
__GUARDRAILS__
edges:
  - from: classify
    to: handle_question
    when: "working.intent == 'question'"

  - from: classify
    to: handle_task
    when: "working.intent == 'task'"

input:
  message: "What is the capital of France?"
"""

PARALLEL_SWRM: str = """\
version: "0.1"

nodes:
  analyze:
    type: swrm
    agents:
      - id: researcher
        provider: __PROVIDER__
        model: __BARE_MODEL__
        prompt: "Research this topic thoroughly: {{ inputs.message }}"

      - id: critic
        provider: __PROVIDER__
        model: __BARE_MODEL__
        prompt: "Critically evaluate this topic: {{ inputs.message }}"

    synthesis:
      provider: __PROVIDER__
      model: __BARE_MODEL__
      prompt: |
        Research findings: {{ analyze.agents.researcher.output }}
        Critical evaluation: {{ analyze.agents.critic.output }}
        Synthesize a balanced, comprehensive response.
__GUARDRAILS__
input:
  message: "Explain the pros and cons of microservices architecture."
"""

FACTORY: str = """\
version: "0.1"

agents:
  worker:
    model: "__MODEL__"
    system: "Process the task provided and return a detailed result."

  aggregator:
    model: "__MODEL__"
    system: "Combine all individual results into a cohesive final summary."

nodes:
  process_items:
    type: factory
    swarm_size: 3
    concurrency: 2
    writes: working.results
    agent: worker

  compile_results:
    agent: aggregator
    writes: output.summary
__GUARDRAILS__
edges:
  - from: process_items
    to: compile_results

input:
  message: "Analyze three different perspectives on artificial intelligence in healthcare."
"""


@dataclass
class WorkflowTemplate:
    """Metadata and YAML content for a scaffold template.

    :param key: Machine-readable identifier used as default filename on collision.
    :param label: Human-readable name shown in the interactive picklist.
    :param description: One-line description shown alongside the label.
    :param content: YAML string containing ``__MODEL__``, ``__PROVIDER__``,
        ``__BARE_MODEL__``, and ``__GUARDRAILS__`` placeholders.
    """

    key: str
    label: str
    description: str
    content: str


TEMPLATES: list[WorkflowTemplate] = [
    WorkflowTemplate(
        key="simple-agent",
        label="Simple Agent",
        description="Single agent, one node, no edges. Fastest way to get started.",
        content=SIMPLE_AGENT,
    ),
    WorkflowTemplate(
        key="sequential",
        label="Sequential Pipeline",
        description="Two agents chained in sequence. Good for multi-step processing.",
        content=SEQUENTIAL,
    ),
    WorkflowTemplate(
        key="conditional",
        label="Conditional Pipeline",
        description="Triage agent that routes to different handlers based on intent.",
        content=CONDITIONAL,
    ),
    WorkflowTemplate(
        key="parallel-swrm",
        label="Parallel SWRM",
        description="Multiple agents fan out in parallel with an optional synthesis step.",
        content=PARALLEL_SWRM,
    ),
    WorkflowTemplate(
        key="factory",
        label="Factory",
        description="Planner that dynamically spawns worker agents over a list of tasks.",
        content=FACTORY,
    ),
]
