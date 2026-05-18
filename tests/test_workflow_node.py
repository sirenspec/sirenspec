"""Tests for WorkflowNode model, WorkflowRegistry, and execute_workflow_node."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError as PydanticValidationError

from sirenspec.core.models import WorkflowNode, parse_node
from sirenspec.core.workflow_registry import WorkflowRegistry
from sirenspec.core.workflow_runner import execute_workflow_node, resolve_node_inputs, resolve_sub_workflow
from sirenspec.exceptions import ValidationError

# ---------------------------------------------------------------------------
# WorkflowNode model
# ---------------------------------------------------------------------------


class TestWorkflowNodeModel:
    def test_parse_from_dict(self) -> None:
        node = WorkflowNode.model_validate({"type": "workflow", "ref": "./b.yaml"})
        assert node.type == "workflow"
        assert node.ref == "./b.yaml"
        assert node.inputs == {}
        assert node.writes is None
        assert node.max_depth == 10

    def test_parse_node_dispatches_workflow_type(self) -> None:
        raw = {"type": "workflow", "ref": "./b.yaml", "inputs": {"topic": "hello"}}
        node = parse_node(raw)
        assert isinstance(node, WorkflowNode)
        assert node.inputs == {"topic": "hello"}

    def test_max_depth_default_is_ten(self) -> None:
        node = WorkflowNode.model_validate({"type": "workflow", "ref": "my-sub"})
        assert node.max_depth == 10

    def test_custom_max_depth(self) -> None:
        node = WorkflowNode.model_validate({"type": "workflow", "ref": "my-sub", "max_depth": 3})
        assert node.max_depth == 3

    def test_max_depth_must_be_positive(self) -> None:
        with pytest.raises(PydanticValidationError):
            WorkflowNode.model_validate({"type": "workflow", "ref": "my-sub", "max_depth": 0})

    def test_optional_writes_field(self) -> None:
        node = WorkflowNode.model_validate({"type": "workflow", "ref": "./b.yaml", "writes": "output.summary"})
        assert node.writes == "output.summary"

    def test_ref_is_required(self) -> None:
        with pytest.raises(PydanticValidationError):
            WorkflowNode.model_validate({"type": "workflow"})


# ---------------------------------------------------------------------------
# WorkflowRegistry
# ---------------------------------------------------------------------------


class TestWorkflowRegistry:
    def test_register_and_get(self) -> None:
        registry = WorkflowRegistry()
        workflow = MagicMock()
        registry.register("my-sub", workflow)
        assert registry.get("my-sub") is workflow

    def test_get_missing_raises_key_error(self) -> None:
        registry = WorkflowRegistry()
        with pytest.raises(KeyError, match="not-there"):
            registry.get("not-there")

    def test_register_overwrites_existing(self) -> None:
        registry = WorkflowRegistry()
        wf1, wf2 = MagicMock(), MagicMock()
        registry.register("sub", wf1)
        registry.register("sub", wf2)
        assert registry.get("sub") is wf2


# ---------------------------------------------------------------------------
# resolve_sub_workflow
# ---------------------------------------------------------------------------


class TestResolveSubWorkflow:
    def test_file_path_dot_slash(self, tmp_path: Path) -> None:
        yaml_content = """
version: "0.1"
agents:
  a:
    model: "openai:gpt-4o-mini"
    system: "Hi"
nodes:
  step:
    agent: a
    writes: output.result
"""
        wf_file = tmp_path / "sub.yaml"
        wf_file.write_text(yaml_content)
        result = resolve_sub_workflow(str(wf_file), registry=None)
        assert result.version == "0.1"

    def test_named_ref_uses_registry(self) -> None:
        registry = WorkflowRegistry()
        wf = MagicMock()
        registry.register("child", wf)
        result = resolve_sub_workflow("child", registry)
        assert result is wf

    def test_named_ref_without_registry_raises(self) -> None:
        with pytest.raises(ValueError, match="no WorkflowRegistry"):
            resolve_sub_workflow("child", registry=None)

    def test_named_ref_not_in_registry_raises_key_error(self) -> None:
        registry = WorkflowRegistry()
        with pytest.raises(KeyError):
            resolve_sub_workflow("missing", registry)


# ---------------------------------------------------------------------------
# resolve_node_inputs
# ---------------------------------------------------------------------------


class TestResolveNodeInputs:
    def test_static_values(self) -> None:
        node = WorkflowNode.model_validate({"type": "workflow", "ref": "./b.yaml", "inputs": {"key": "value"}})
        result = resolve_node_inputs(node, "user msg", {})
        assert result == {"key": "value"}

    def test_template_resolved_from_working(self) -> None:
        node = WorkflowNode.model_validate(
            {
                "type": "workflow",
                "ref": "./b.yaml",
                "inputs": {"topic": "{{ extract.output }}"},
            }
        )
        working = {"extract": {"output": "climate change"}}
        result = resolve_node_inputs(node, "user", working)
        assert result == {"topic": "climate change"}

    def test_empty_inputs_returns_empty_dict(self) -> None:
        node = WorkflowNode.model_validate({"type": "workflow", "ref": "./b.yaml"})
        result = resolve_node_inputs(node, "user", {})
        assert result == {}


# ---------------------------------------------------------------------------
# execute_workflow_node
# ---------------------------------------------------------------------------


class TestExecuteWorkflowNode:
    @pytest.mark.asyncio
    async def test_depth_limit_raises_validation_error(self) -> None:
        node = WorkflowNode.model_validate({"type": "workflow", "ref": "./b.yaml", "max_depth": 3})
        with pytest.raises(ValidationError, match="Max workflow nesting depth 3 exceeded"):
            await execute_workflow_node(
                node_id="run_b",
                node=node,
                user_input="hello",
                working={},
                registry=None,
                depth=3,
            )

    @pytest.mark.asyncio
    async def test_successful_execution_returns_trace(self) -> None:
        node = WorkflowNode.model_validate(
            {
                "type": "workflow",
                "ref": "./b.yaml",
                "inputs": {"topic": "AI"},
            }
        )
        sub_trace = {
            "output": {"step1": "result text"},
            "summary": {"total_tokens": 42, "total_duration_ms": 100.0},
        }
        mock_workflow = MagicMock()
        with (
            patch("sirenspec.core.workflow_runner.resolve_sub_workflow", return_value=mock_workflow),
            patch("sirenspec.core.executor.execute", new_callable=AsyncMock, return_value=sub_trace),
        ):
            trace = await execute_workflow_node(
                node_id="run_b",
                node=node,
                user_input="hello",
                working={},
                registry=None,
                depth=0,
            )

        assert trace["id"] == "run_b"
        assert trace["type"] == "workflow"
        assert trace["ref"] == "./b.yaml"
        assert trace["output"] == {"step1": "result text"}
        assert trace["tokens"] == 42
        assert trace["duration_ms"] == 100.0
        assert trace["error"] is None

    @pytest.mark.asyncio
    async def test_resolved_inputs_passed_to_sub_execute(self) -> None:
        node = WorkflowNode.model_validate(
            {
                "type": "workflow",
                "ref": "./b.yaml",
                "inputs": {"topic": "AI"},
            }
        )
        sub_trace = {"output": {}, "summary": {"total_tokens": 0, "total_duration_ms": 0.0}}
        mock_workflow = MagicMock()
        mock_execute = AsyncMock(return_value=sub_trace)

        with (
            patch("sirenspec.core.workflow_runner.resolve_sub_workflow", return_value=mock_workflow),
            patch("sirenspec.core.executor.execute", mock_execute),
        ):
            await execute_workflow_node(
                node_id="run_b",
                node=node,
                user_input="hello",
                working={},
                registry=None,
                depth=0,
            )

        call_kwargs = mock_execute.call_args.kwargs
        assert call_kwargs["initial_inputs"] == {"topic": "AI"}
        assert call_kwargs["depth"] == 1

    @pytest.mark.asyncio
    async def test_depth_increments_for_sub_execute(self) -> None:
        node = WorkflowNode.model_validate({"type": "workflow", "ref": "./b.yaml"})
        sub_trace = {"output": {}, "summary": {"total_tokens": 0, "total_duration_ms": 0.0}}
        mock_execute = AsyncMock(return_value=sub_trace)

        with (
            patch("sirenspec.core.workflow_runner.resolve_sub_workflow", return_value=MagicMock()),
            patch("sirenspec.core.executor.execute", mock_execute),
        ):
            await execute_workflow_node(
                node_id="run_b",
                node=node,
                user_input="hello",
                working={},
                registry=None,
                depth=2,
            )

        assert mock_execute.call_args.kwargs["depth"] == 3


# ---------------------------------------------------------------------------
# Integration: executor wires WorkflowNode output into parent context
# ---------------------------------------------------------------------------


class TestExecutorWorkflowNodeIntegration:
    @pytest.mark.asyncio
    async def test_output_written_to_parent_context(self) -> None:
        from sirenspec.core.executor import execute
        from sirenspec.core.models import Workflow

        sub_output = {"sub_step": "sub result"}
        sub_trace = {
            "workflow": {"version": "0.1"},
            "input": {"message": "hello"},
            "nodes": [],
            "output": sub_output,
            "summary": {"total_tokens": 10, "total_usage": {}, "total_duration_ms": 50.0, "status": "success"},
        }

        wf_raw = {
            "version": "0.1",
            "agents": {},
            "nodes": {
                "run_b": {
                    "type": "workflow",
                    "ref": "child",
                }
            },
            "edges": [],
        }
        workflow = Workflow.model_validate(wf_raw)

        registry = WorkflowRegistry()
        registry.register("child", MagicMock())

        with patch("sirenspec.core.executor.execute", new_callable=AsyncMock, return_value=sub_trace):
            result = await execute(workflow, "hello", registry=registry)

        assert result["output"]["run_b"] == sub_output

    @pytest.mark.asyncio
    async def test_explicit_writes_also_populated(self) -> None:
        from sirenspec.core.executor import execute
        from sirenspec.core.models import Workflow

        sub_output = {"sub_step": "result"}
        sub_trace = {
            "workflow": {"version": "0.1"},
            "input": {"message": "hello"},
            "nodes": [],
            "output": sub_output,
            "summary": {"total_tokens": 5, "total_usage": {}, "total_duration_ms": 20.0, "status": "success"},
        }

        wf_raw = {
            "version": "0.1",
            "agents": {},
            "nodes": {
                "run_b": {
                    "type": "workflow",
                    "ref": "child",
                    "writes": "output.summary",
                }
            },
            "edges": [],
        }
        workflow = Workflow.model_validate(wf_raw)

        registry = WorkflowRegistry()
        registry.register("child", MagicMock())

        with patch("sirenspec.core.executor.execute", new_callable=AsyncMock, return_value=sub_trace):
            result = await execute(workflow, "hello", registry=registry)

        assert result["output"]["run_b"] == sub_output
        assert result["output"]["summary"] == sub_output
