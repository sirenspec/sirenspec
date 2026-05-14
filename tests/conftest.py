"""Shared pytest fixtures for SirenSpec tests."""

from __future__ import annotations

from typing import Any

import pytest

from sirenspec.core.models import AgentDefinition, Edge, Node, Workflow


@pytest.fixture
def minimal_workflow() -> Workflow:
    """A minimal valid single-node workflow."""
    return Workflow(
        version="0.1",
        agents={"assistant": AgentDefinition(model="openai:gpt-4o-mini", system="You are helpful.")},
        nodes={"answer": Node(agent="assistant", writes="output.reply")},
    )


@pytest.fixture
def sequential_workflow() -> Workflow:
    """A two-node sequential workflow with one edge."""
    return Workflow(
        version="0.1",
        agents={
            "classifier": AgentDefinition(model="openai:gpt-4o-mini", system="Classify intent."),
            "replier": AgentDefinition(model="anthropic:claude-haiku-4-5-20251001", system="Reply helpfully."),
        },
        nodes={
            "classify": Node(agent="classifier", writes="working.intent"),
            "reply": Node(agent="replier", writes="output.reply"),
        },
        edges=[Edge(**{"from": "classify", "to": "reply"})],
    )


@pytest.fixture
def mock_provider_response() -> dict[str, Any]:
    return {"content": "Mock LLM response", "tokens": 42}
