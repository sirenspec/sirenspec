"""Unit tests for YAML workflow parser."""

from __future__ import annotations

from pathlib import Path

import pytest

from sirenspec.core.models import Workflow
from sirenspec.yaml.parser import load_workflow


class TestLoadWorkflow:
    def test_load_minimal_yaml(self, tmp_path: Path) -> None:
        yaml_content = """\
version: "0.1"
agents:
  assistant:
    model: "openai:gpt-4o-mini"
    system: "You are helpful."
nodes:
  answer:
    agent: assistant
    writes: output.reply
"""
        f = tmp_path / "wf.yaml"
        f.write_text(yaml_content)
        wf = load_workflow(f)
        assert isinstance(wf, Workflow)
        assert wf.version == "0.1"

    def test_load_with_edges(self, tmp_path: Path) -> None:
        yaml_content = """\
version: "0.1"
agents:
  a:
    model: "openai:gpt-4o-mini"
    system: "sys"
  b:
    model: "anthropic:claude-haiku-4-5-20251001"
    system: "sys2"
nodes:
  n1:
    agent: a
    writes: working.x
  n2:
    agent: b
    writes: output.y
edges:
  - from: n1
    to: n2
"""
        f = tmp_path / "wf.yaml"
        f.write_text(yaml_content)
        wf = load_workflow(f)
        assert len(wf.edges) == 1
        assert wf.edges[0].from_node == "n1"

    def test_file_not_found_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_workflow(tmp_path / "nonexistent.yaml")

    def test_invalid_yaml_raises_value_error(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.yaml"
        f.write_text("{ invalid yaml ][")
        with pytest.raises(ValueError, match="YAML parse error"):
            load_workflow(f)

    def test_missing_required_field_raises(self, tmp_path: Path) -> None:
        yaml_content = """\
agents:
  a:
    model: openai:gpt-4o-mini
    system: sys
nodes:
  n:
    agent: a
    writes: output.x
"""
        f = tmp_path / "wf.yaml"
        f.write_text(yaml_content)
        with pytest.raises(ValueError, match="validation failed"):
            load_workflow(f)

    def test_load_simple_agent_example(self) -> None:
        example = Path(__file__).parent.parent / "docs" / "cookbook" / "simple-agent" / "workflow.yaml"
        wf = load_workflow(example)
        assert wf.version == "0.1"
        assert "assistant" in wf.agents

    def test_load_sequential_pipeline_example(self) -> None:
        example = Path(__file__).parent.parent / "docs" / "cookbook" / "sequential-pipeline" / "workflow.yaml"
        wf = load_workflow(example)
        assert len(wf.edges) == 1
        assert "classifier" in wf.agents
        assert "replier" in wf.agents

    def test_non_mapping_yaml_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "list.yaml"
        f.write_text("- item1\n- item2\n")
        with pytest.raises(ValueError, match="mapping"):
            load_workflow(f)
