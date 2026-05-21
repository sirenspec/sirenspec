"""Tests for ``sirenspec init`` scaffolding command."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from sirenspec.cli import app
from sirenspec.cli.init import (
    PROVIDER_CHOICES,
    render_template,
    resolve_output_path,
    split_provider_model,
)
from sirenspec.templates import TEMPLATES

runner = CliRunner()


class TestSplitProviderModel:
    def test_openai(self) -> None:
        assert split_provider_model("openai:gpt-4o-mini") == ("openai", "gpt-4o-mini")

    def test_anthropic(self) -> None:
        assert split_provider_model("anthropic:claude-haiku-4-5-20251001") == (
            "anthropic",
            "claude-haiku-4-5-20251001",
        )

    def test_all_provider_choices_split_cleanly(self) -> None:
        for choice in PROVIDER_CHOICES:
            provider, model = split_provider_model(choice)
            assert provider
            assert model
            assert ":" not in provider
            assert ":" not in model


class TestRenderTemplate:
    @pytest.mark.parametrize("template", TEMPLATES, ids=lambda t: t.key)
    @pytest.mark.parametrize("model_str", PROVIDER_CHOICES, ids=lambda m: split_provider_model(m)[0])
    @pytest.mark.parametrize("guardrails", [True, False], ids=["guardrails", "no_guardrails"])
    def test_rendered_yaml_passes_validate(
        self,
        template,
        model_str: str,
        guardrails: bool,
        tmp_path: Path,
    ) -> None:
        content = render_template(template, model_str, guardrails)
        path = tmp_path / "workflow.yaml"
        path.write_text(content)
        result = runner.invoke(app, ["validate", str(path)])
        assert result.exit_code == 0, (
            f"validate failed for {template.key}/{model_str}/guardrails={guardrails}:\n{result.output}"
        )

    def test_guardrails_block_present_when_enabled(self) -> None:
        template = TEMPLATES[0]
        content = render_template(template, "openai:gpt-4o-mini", guardrails=True)
        assert "guardrails:" in content
        assert "- injection" in content
        assert "- length" in content

    def test_guardrails_block_absent_when_disabled(self) -> None:
        template = TEMPLATES[0]
        content = render_template(template, "openai:gpt-4o-mini", guardrails=False)
        assert "guardrails:" not in content

    def test_no_placeholders_remain_after_render(self) -> None:
        for template in TEMPLATES:
            content = render_template(template, "openai:gpt-4o-mini", guardrails=True)
            for placeholder in ("__MODEL__", "__PROVIDER__", "__BARE_MODEL__", "__GUARDRAILS__"):
                assert placeholder not in content, f"{placeholder} not substituted in {template.key}"

    def test_swrm_template_uses_separate_provider_and_model_fields(self) -> None:
        swrm = next(t for t in TEMPLATES if t.key == "parallel-swrm")
        content = render_template(swrm, "anthropic:claude-haiku-4-5-20251001", guardrails=False)
        assert "provider: anthropic" in content
        assert "model: claude-haiku-4-5-20251001" in content
        assert "anthropic:claude-haiku-4-5-20251001" not in content


class TestResolveOutputPath:
    def test_no_collision_returns_workflow_yaml(self, tmp_path: Path) -> None:
        result = resolve_output_path(tmp_path, "simple-agent")
        assert result == tmp_path / "workflow.yaml"

    def test_collision_prompts_for_alternate_name(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        (tmp_path / "workflow.yaml").write_text("exists")
        monkeypatch.setattr("sirenspec.cli.init.Prompt.ask", lambda *a, **kw: "my-workflow.yaml")
        result = resolve_output_path(tmp_path, "simple-agent")
        assert result == tmp_path / "my-workflow.yaml"

    def test_collision_default_uses_template_key(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        (tmp_path / "workflow.yaml").write_text("exists")
        captured: dict[str, str] = {}

        def capture_ask(*args: object, default: str = "", **kwargs: object) -> str:
            captured["default"] = default
            return default

        monkeypatch.setattr("sirenspec.cli.init.Prompt.ask", capture_ask)
        resolve_output_path(tmp_path, "sequential")
        assert captured["default"] == "sequential.yaml"
