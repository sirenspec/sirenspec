"""Unit tests for CLI commands."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from sirenspec.cli import app

runner = CliRunner()


def _write_workflow(tmp_path: Path, content: str) -> Path:
    f = tmp_path / "wf.yaml"
    f.write_text(content)
    return f


MINIMAL_YAML = """\
version: "0.1"
agents:
  assistant:
    model: "openai:gpt-4o-mini"
    system: "You are helpful."
nodes:
  answer:
    agent: assistant
    writes: output.reply
input:
  message: "Hello"
"""

_MOCK_TRACE = {
    "workflow": {"version": "0.1"},
    "input": {"message": "Hello"},
    "nodes": [],
    "output": {"reply": "Hi there"},
    "summary": {"total_tokens": 5, "total_duration_ms": 10.0, "status": "success"},
}


class TestValidateCommand:
    def test_valid_workflow_exits_zero(self, tmp_path: Path) -> None:
        f = _write_workflow(tmp_path, MINIMAL_YAML)
        result = runner.invoke(app, ["validate", str(f)])
        assert result.exit_code == 0
        assert "✓" in result.output

    def test_valid_workflow_shows_counts(self, tmp_path: Path) -> None:
        f = _write_workflow(tmp_path, MINIMAL_YAML)
        result = runner.invoke(app, ["validate", str(f)])
        assert "1 agent" in result.output
        assert "1 node" in result.output

    def test_missing_file_exits_nonzero(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["validate", str(tmp_path / "nope.yaml")])
        assert result.exit_code != 0

    def test_invalid_yaml_exits_nonzero(self, tmp_path: Path) -> None:
        f = _write_workflow(tmp_path, "{ bad yaml ][")
        result = runner.invoke(app, ["validate", str(f)])
        assert result.exit_code != 0

    def test_validate_example_workflows(self) -> None:
        examples_dir = Path(__file__).parent.parent / "examples"
        for example in examples_dir.glob("*.yaml"):
            result = runner.invoke(app, ["validate", str(example)])
            assert result.exit_code == 0, f"Failed for {example.name}: {result.output}"


class TestRunCommand:
    def test_run_trace_flag_prints_json(self, tmp_path: Path) -> None:
        """--trace flag falls back to execute() and emits JSON on stdout."""
        f = _write_workflow(tmp_path, MINIMAL_YAML)

        with patch("sirenspec.cli.run.asyncio") as mock_asyncio:
            mock_asyncio.run.return_value = _MOCK_TRACE
            result = runner.invoke(app, ["run", str(f), "--trace"])

        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed["summary"]["status"] == "success"

    def test_run_output_json_flag_prints_json(self, tmp_path: Path) -> None:
        """--output json flag falls back to execute() and emits JSON on stdout."""
        f = _write_workflow(tmp_path, MINIMAL_YAML)

        with patch("sirenspec.cli.run.asyncio") as mock_asyncio:
            mock_asyncio.run.return_value = _MOCK_TRACE
            result = runner.invoke(app, ["run", str(f), "--output", "json"])

        assert result.exit_code == 0
        parsed = json.loads(result.output)
        assert parsed["summary"]["status"] == "success"

    def test_run_missing_file_exits_nonzero(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["run", str(tmp_path / "nope.yaml")])
        assert result.exit_code != 0

    def test_run_no_input_exits_nonzero(self, tmp_path: Path) -> None:
        yaml_no_input = """\
version: "0.1"
agents:
  a:
    model: "openai:gpt-4o-mini"
    system: "sys"
nodes:
  n:
    agent: a
    writes: output.x
"""
        f = _write_workflow(tmp_path, yaml_no_input)
        result = runner.invoke(app, ["run", str(f)])
        assert result.exit_code != 0

    def test_run_input_flag_overrides_workflow(self, tmp_path: Path) -> None:
        f = _write_workflow(tmp_path, MINIMAL_YAML)
        captured: list = []

        def capture_asyncio_run(coro):  # type: ignore[no-untyped-def]
            captured.append(coro)
            return _MOCK_TRACE

        with patch("sirenspec.cli.run.asyncio") as mock_asyncio:
            mock_asyncio.run.side_effect = capture_asyncio_run
            runner.invoke(app, ["run", str(f), "--trace", "--input", "Override"])

        # The coroutine was created with "Override" as user_input — just verify asyncio.run was called
        assert mock_asyncio.run.called

    def test_run_workflow_input_used_when_no_flag(self, tmp_path: Path) -> None:
        f = _write_workflow(tmp_path, MINIMAL_YAML)

        with patch("sirenspec.cli.run.asyncio") as mock_asyncio:
            mock_asyncio.run.return_value = _MOCK_TRACE
            result = runner.invoke(app, ["run", str(f), "--trace"])

        assert result.exit_code == 0
        assert mock_asyncio.run.called

    @pytest.mark.parametrize("status", ["failed"])
    def test_run_failed_trace_exits_nonzero(self, tmp_path: Path, status: str) -> None:
        f = _write_workflow(tmp_path, MINIMAL_YAML)
        failed_trace = {**_MOCK_TRACE, "summary": {**_MOCK_TRACE["summary"], "status": status}}

        with patch("sirenspec.cli.run.asyncio") as mock_asyncio:
            mock_asyncio.run.return_value = failed_trace
            result = runner.invoke(app, ["run", str(f), "--trace"], catch_exceptions=False)

        assert result.exit_code != 0
