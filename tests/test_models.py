"""Unit tests for Pydantic data models."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from sirenspec.core.models import AgentDefinition, Edge, MemoryConfig, Node, Workflow, WorkflowInput


class TestAgentDefinition:
    def test_minimal(self) -> None:
        agent = AgentDefinition(model="openai:gpt-4o-mini", system="Help me.")
        assert agent.model == "openai:gpt-4o-mini"
        assert agent.system == "Help me."
        assert agent.guardrails is None

    def test_with_guardrails(self) -> None:
        agent = AgentDefinition(model="openai:gpt-4o", system="sys", guardrails=["injection", "length"])
        assert agent.guardrails == ["injection", "length"]

    def test_missing_model_raises(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            AgentDefinition(system="sys")  # type: ignore[call-arg]
        assert "model" in str(exc_info.value)

    def test_missing_system_raises(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            AgentDefinition(model="openai:gpt-4o")  # type: ignore[call-arg]
        assert "system" in str(exc_info.value)


class TestNode:
    def test_valid_node(self) -> None:
        node = Node(agent="assistant", writes="output.reply")
        assert node.agent == "assistant"
        assert node.writes == "output.reply"

    def test_working_path(self) -> None:
        node = Node(agent="classifier", writes="working.intent")
        assert node.writes == "working.intent"

    def test_missing_agent_raises(self) -> None:
        with pytest.raises(ValidationError):
            Node(writes="output.reply")  # type: ignore[call-arg]


class TestEdge:
    def test_valid_edge(self) -> None:
        edge = Edge(**{"from": "a", "to": "b"})
        assert edge.from_node == "a"
        assert edge.to_node == "b"
        assert edge.when is None

    def test_edge_with_when(self) -> None:
        edge = Edge(**{"from": "a", "to": "b", "when": "intent == 'question'"})
        assert edge.when == "intent == 'question'"

    def test_missing_from_raises(self) -> None:
        with pytest.raises(ValidationError):
            Edge(**{"to": "b"})  # type: ignore[call-arg]


class TestWorkflowInput:
    def test_with_message(self) -> None:
        wi = WorkflowInput(message="hello")
        assert wi.message == "hello"

    def test_no_message(self) -> None:
        wi = WorkflowInput()
        assert wi.message is None


class TestWorkflow:
    def test_minimal_valid_workflow(self) -> None:
        wf = Workflow(
            version="0.1",
            agents={"a": AgentDefinition(model="openai:gpt-4o-mini", system="sys")},
            nodes={"n": Node(agent="a", writes="output.x")},
        )
        assert wf.version == "0.1"
        assert len(wf.agents) == 1
        assert len(wf.nodes) == 1
        assert wf.edges == []

    def test_full_workflow(self) -> None:
        wf = Workflow(
            version="0.1",
            agents={
                "a": AgentDefinition(model="openai:gpt-4o-mini", system="sys"),
                "b": AgentDefinition(model="anthropic:claude-haiku-4-5-20251001", system="sys2"),
            },
            nodes={
                "n1": Node(agent="a", writes="working.x"),
                "n2": Node(agent="b", writes="output.y"),
            },
            edges=[Edge(**{"from": "n1", "to": "n2"})],
            input=WorkflowInput(message="hi"),
            guardrails=["injection"],
        )
        assert len(wf.edges) == 1
        assert wf.input is not None
        assert wf.input.message == "hi"

    def test_missing_version_raises(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            Workflow(
                agents={"a": AgentDefinition(model="openai:gpt-4o-mini", system="s")},  # type: ignore[call-arg]
                nodes={"n": Node(agent="a", writes="output.x")},
            )
        assert "version" in str(exc_info.value)

    def test_edge_references_unknown_node_raises(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            Workflow(
                version="0.1",
                agents={"a": AgentDefinition(model="openai:gpt-4o-mini", system="s")},
                nodes={"n1": Node(agent="a", writes="output.x")},
                edges=[Edge(**{"from": "n1", "to": "nonexistent"})],
            )
        assert "nonexistent" in str(exc_info.value)

    def test_node_references_unknown_agent_raises(self) -> None:
        with pytest.raises(ValidationError) as exc_info:
            Workflow(
                version="0.1",
                agents={"a": AgentDefinition(model="openai:gpt-4o-mini", system="s")},
                nodes={"n1": Node(agent="unknown_agent", writes="output.x")},
            )
        assert "unknown_agent" in str(exc_info.value)

    def test_workflow_from_dict(self) -> None:
        data = {
            "version": "0.1",
            "agents": {"a": {"model": "openai:gpt-4o-mini", "system": "sys"}},
            "nodes": {"n": {"agent": "a", "writes": "output.x"}},
        }
        wf = Workflow.model_validate(data)
        assert wf.version == "0.1"

    @given(version=st.text(min_size=1))
    @settings(max_examples=20)
    def test_version_is_any_nonempty_string(self, version: str) -> None:
        wf = Workflow(
            version=version,
            agents={"a": AgentDefinition(model="openai:gpt-4o-mini", system="s")},
            nodes={"n": Node(agent="a", writes="output.x")},
        )
        assert wf.version == version

    def test_workflow_with_memory_sqlite(self) -> None:
        wf = Workflow(
            version="0.1",
            agents={"a": AgentDefinition(model="openai:gpt-4o-mini", system="s")},
            nodes={"n": Node(agent="a", writes="memory.result")},
            memory=MemoryConfig(backend="sqlite", path=".sirenspec/memory", ttl=3600),
        )
        assert wf.memory is not None
        assert wf.memory.backend == "sqlite"
        assert wf.memory.ttl == 3600

    def test_workflow_with_memory_file(self) -> None:
        wf = Workflow(
            version="0.1",
            agents={"a": AgentDefinition(model="openai:gpt-4o-mini", system="s")},
            nodes={"n": Node(agent="a", writes="output.x")},
            memory=MemoryConfig(backend="file", path="/tmp/mem"),
        )
        assert wf.memory is not None
        assert wf.memory.backend == "file"
        assert wf.memory.ttl is None

    def test_workflow_without_memory_is_none(self) -> None:
        wf = Workflow(
            version="0.1",
            agents={"a": AgentDefinition(model="openai:gpt-4o-mini", system="s")},
            nodes={"n": Node(agent="a", writes="output.x")},
        )
        assert wf.memory is None


class TestMemoryConfig:
    def test_defaults(self) -> None:
        cfg = MemoryConfig()
        assert cfg.backend == "sqlite"
        assert cfg.path == ".sirenspec/memory"
        assert cfg.ttl is None

    def test_file_backend(self) -> None:
        cfg = MemoryConfig(backend="file")
        assert cfg.backend == "file"

    def test_sqlite_backend(self) -> None:
        cfg = MemoryConfig(backend="sqlite")
        assert cfg.backend == "sqlite"

    def test_ttl_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            MemoryConfig(ttl=0)

    def test_ttl_one_is_valid(self) -> None:
        cfg = MemoryConfig(ttl=1)
        assert cfg.ttl == 1

    def test_invalid_backend_raises(self) -> None:
        with pytest.raises(ValidationError):
            MemoryConfig(backend="redis")  # type: ignore[arg-type]

    def test_from_dict(self) -> None:
        cfg = MemoryConfig.model_validate({"backend": "file", "path": "/data/mem", "ttl": 86400})
        assert cfg.backend == "file"
        assert cfg.path == "/data/mem"
        assert cfg.ttl == 86400
