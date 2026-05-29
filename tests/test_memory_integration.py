"""Integration tests: persistent memory across two executor runs."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from sirenspec.core.executor import execute
from sirenspec.core.models import AgentDefinition, AgentNode, MemoryConfig, Workflow


def make_workflow(memory_config: MemoryConfig) -> Workflow:
    return Workflow(
        version="0.1",
        agents={"writer": AgentDefinition(model="openai:gpt-4o-mini", system="You are a writer.")},
        nodes={"write_node": AgentNode(agent="writer", writes="memory.summary")},
        memory=memory_config,
    )


def make_reader_workflow(memory_config: MemoryConfig) -> Workflow:
    return Workflow(
        version="0.1",
        agents={
            "reader": AgentDefinition(
                model="openai:gpt-4o-mini",
                system="Prior summary: {{ memory.summary }}",
            )
        },
        nodes={"read_node": AgentNode(agent="reader", writes="output.result")},
        memory=memory_config,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["file", "sqlite"])
async def test_memory_persists_across_runs(tmp_path, backend) -> None:
    memory_config = MemoryConfig(backend=backend, path=str(tmp_path / "mem"))

    writer_wf = make_workflow(memory_config)
    mock_result = AsyncMock()
    mock_result.output = "key finding from run 1"
    mock_result.token_usage.prompt_tokens = 10
    mock_result.token_usage.completion_tokens = 5
    mock_result.token_usage.total = 15
    mock_result.guardrails_passed = []
    mock_result.retry_attempts = []

    with patch("sirenspec.core.executor.execute_agent_node", return_value=mock_result):
        await execute(writer_wf, "run 1")

    reader_wf = make_reader_workflow(memory_config)
    captured_system: list[str] = []

    async def capturing_agent(node_id, model_uri, system_prompt, **kwargs):
        captured_system.append(system_prompt)
        return mock_result

    with patch("sirenspec.core.executor.execute_agent_node", side_effect=capturing_agent):
        await execute(reader_wf, "run 2")

    assert len(captured_system) == 1
    assert "key finding from run 1" in captured_system[0]


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["file", "sqlite"])
async def test_memory_namespace_available_in_template(tmp_path, backend) -> None:
    memory_config = MemoryConfig(backend=backend, path=str(tmp_path / "mem"))

    writer_wf = make_workflow(memory_config)
    mock_result = AsyncMock()
    mock_result.output = "stored value"
    mock_result.token_usage.prompt_tokens = 5
    mock_result.token_usage.completion_tokens = 5
    mock_result.token_usage.total = 10
    mock_result.guardrails_passed = []
    mock_result.retry_attempts = []

    with patch("sirenspec.core.executor.execute_agent_node", return_value=mock_result):
        await execute(writer_wf, "initial")

    from sirenspec.memory.manager import MemoryManager, build_store

    manager = MemoryManager(build_store(memory_config))
    namespace = manager.read_namespace()
    manager.close()

    assert namespace.get("summary") == "stored value"


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ["file", "sqlite"])
async def test_memory_write_does_not_pollute_context_output(tmp_path, backend) -> None:
    memory_config = MemoryConfig(backend=backend, path=str(tmp_path / "mem"))
    writer_wf = make_workflow(memory_config)

    mock_result = AsyncMock()
    mock_result.output = "value"
    mock_result.token_usage.prompt_tokens = 5
    mock_result.token_usage.completion_tokens = 5
    mock_result.token_usage.total = 10
    mock_result.guardrails_passed = []
    mock_result.retry_attempts = []

    with patch("sirenspec.core.executor.execute_agent_node", return_value=mock_result):
        trace = await execute(writer_wf, "test")

    # memory.summary is not a regular context path, so output dict should be empty
    assert trace["output"] == {}


@pytest.mark.asyncio
async def test_no_memory_block_does_not_error(tmp_path) -> None:
    wf = Workflow(
        version="0.1",
        agents={"a": AgentDefinition(model="openai:gpt-4o-mini", system="sys")},
        nodes={"n": AgentNode(agent="a", writes="output.x")},
    )
    mock_result = AsyncMock()
    mock_result.output = "ok"
    mock_result.token_usage.prompt_tokens = 5
    mock_result.token_usage.completion_tokens = 5
    mock_result.token_usage.total = 10
    mock_result.guardrails_passed = []
    mock_result.retry_attempts = []

    with patch("sirenspec.core.executor.execute_agent_node", return_value=mock_result):
        trace = await execute(wf, "hi")

    assert trace["output"]["x"] == "ok"
