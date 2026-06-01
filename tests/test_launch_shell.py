"""Tests for the ``sirenspec launch`` TUI shell: theme, commands, summary, and app boot."""

from __future__ import annotations

from pathlib import Path

import pytest

from sirenspec.core.models import (
    AgentDefinition,
    AgentNode,
    SwrmAgent,
    SwrmNode,
    ToolNode,
    Workflow,
)
from sirenspec.exceptions import SessionError
from sirenspec.session import theme
from sirenspec.session.app import LaunchApp, render_plain
from sirenspec.session.commands import (
    CommandRegistry,
    is_command,
    make_registry,
    parse_command_line,
)
from sirenspec.session.runtime import WorkflowSession
from sirenspec.session.summary import provider_of, summarise_workflow
from sirenspec.session.widgets import (
    CommandInput,
    CommandPalette,
    StatusBar,
    Transcript,
    WorkflowRail,
)


def build_workflow() -> Workflow:
    """Build a workflow exercising agent, swrm, and tool nodes for the rail."""
    return Workflow(
        version="0.3",
        agents={
            "risk": AgentDefinition(model="anthropic:claude-haiku-4-5-20251001", system="Assess risk."),
            "writer": AgentDefinition(model="anthropic:claude-haiku-4-5-20251001", system="Write."),
        },
        nodes={
            "analyze": SwrmNode(
                type="swrm",
                agents=[
                    SwrmAgent(
                        id="risk", provider="anthropic", model="claude-haiku-4-5-20251001", prompt="{{ inputs }}"
                    ),
                    SwrmAgent(id="sentiment", provider="openai", model="gpt-4o-mini", prompt="{{ inputs }}"),
                ],
            ),
            "draft": AgentNode(agent="writer", writes="output.reply"),
            "store": ToolNode(type="tool", tool="http", config={"url": "https://example.com"}),
        },
    )


# ---------------------------------------------------------------------------
# Theme & colour-mode resolution
# ---------------------------------------------------------------------------


class TestColorMode:
    def test_default_tty_enables_color(self) -> None:
        assert theme.resolve_color_mode(plain=False, is_tty=True).enabled is True

    def test_plain_disables_color(self) -> None:
        assert theme.resolve_color_mode(plain=True, is_tty=True).enabled is False

    def test_non_tty_disables_color(self) -> None:
        assert theme.resolve_color_mode(plain=False, is_tty=False).enabled is False

    def test_no_color_env_disables_color(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("NO_COLOR", "1")
        assert theme.resolve_color_mode(plain=False, is_tty=True).enabled is False


class TestCrestBanner:
    def test_pixel_map_places_known_pixels(self) -> None:
        pixels = theme.build_crest_pixels()
        # Body fill is crest-light; the eyes overwrite two body cells with the eye colour.
        assert pixels[(6, 8)] == theme.CREST_LIGHT
        assert pixels[(8, 10)] == theme.CREST_EYE
        assert pixels[(8, 16)] == theme.CREST_EYE
        # A left-wing pixel is crest-dark.
        assert pixels[(6, 0)] == theme.CREST_DARK
        # The sprite stays within the 28×28 grid.
        assert all(0 <= r < theme.CREST_GRID and 0 <= c < theme.CREST_GRID for (r, c) in pixels)

    def test_color_banner_uses_half_blocks(self) -> None:
        rendered = theme.render_crest(theme.ColorMode(enabled=True)).plain
        assert "▀" in rendered
        assert rendered.count("\n") == theme.CREST_GRID // 2 - 1

    def test_plain_banner_is_monochrome_silhouette(self) -> None:
        text = theme.render_crest(theme.ColorMode(enabled=False))
        # No colour styles applied in the plain fallback.
        assert all(span.style is None for span in text.spans)
        assert "█" in text.plain or "▀" in text.plain

    def test_build_theme_named_sirenspec(self) -> None:
        assert theme.build_theme(theme.ColorMode(enabled=True)).name == "sirenspec"


# ---------------------------------------------------------------------------
# Command parsing & registry
# ---------------------------------------------------------------------------


class TestCommandParsing:
    def test_parse_strips_slash_and_splits_args(self) -> None:
        assert parse_command_line("/snapshot before refactor") == ("snapshot", "before refactor")

    def test_parse_no_args(self) -> None:
        assert parse_command_line("/help") == ("help", "")

    def test_parse_empty(self) -> None:
        assert parse_command_line("/") == ("", "")

    def test_is_command(self) -> None:
        assert is_command("/run")
        assert not is_command("what are the risks?")


class TestCommandRegistry:
    @pytest.mark.asyncio
    async def test_dispatch_invokes_handler_with_args(self) -> None:
        captured: list[str] = []

        async def handler(args: str) -> None:
            captured.append(args)

        registry = CommandRegistry()
        registry.register("snapshot", "save", handler, glyph="◇")
        await registry.dispatch("/snapshot my label")
        assert captured == ["my label"]

    @pytest.mark.asyncio
    async def test_dispatch_unknown_raises(self) -> None:
        registry = CommandRegistry()
        with pytest.raises(SessionError):
            await registry.dispatch("/nope")

    def test_duplicate_registration_raises(self) -> None:
        registry = CommandRegistry()
        registry.register("run", "run it", _noop)
        with pytest.raises(SessionError):
            registry.register("run", "again", _noop)

    def test_match_filters_by_prefix(self) -> None:
        registry = make_registry()
        names = {c.name for c in registry.match("/s")}
        assert "snapshot" in names
        assert "edit" not in names

    def test_hints_are_subset(self) -> None:
        registry = make_registry()
        hint_names = {c.name for c in registry.hints()}
        assert {"edit", "run", "snapshot"} <= hint_names

    @pytest.mark.asyncio
    async def test_unbound_handler_raises(self) -> None:
        registry = make_registry()  # no handlers supplied → all unbound
        with pytest.raises(SessionError):
            await registry.dispatch("/snapshot")


async def _noop(_: str) -> None:
    return None


# ---------------------------------------------------------------------------
# Workflow summary
# ---------------------------------------------------------------------------


class TestSummary:
    def test_provider_of(self) -> None:
        assert provider_of("anthropic:claude-haiku-4-5-20251001") == "anthropic"
        assert provider_of("openai/gpt-4o-mini") == "openai"
        assert provider_of("llama3") == "llama3"

    def test_summarise_counts_and_tree(self) -> None:
        summary = summarise_workflow(build_workflow(), "market-analysis")
        assert summary.name == "market-analysis"
        assert summary.node_count == 3
        assert summary.agent_count == 2
        assert summary.primary_provider == "anthropic"

        by_id = {n.node_id: n for n in summary.nodes}
        assert by_id["analyze"].kind == "swrm"
        assert [c.label for c in by_id["analyze"].children] == ["risk", "sentiment"]
        assert by_id["analyze"].children[1].provider == "openai"
        assert by_id["store"].kind == "tool: http"


# ---------------------------------------------------------------------------
# Plain console fallback
# ---------------------------------------------------------------------------


class TestPlainFallback:
    def test_render_plain_prints_summary(self) -> None:
        from rich.console import Console

        console = Console(record=True, force_terminal=False, no_color=True, width=100)
        render_plain(build_workflow(), "market-analysis", console)
        out = console.export_text()
        assert "SirenSpec · market-analysis" in out
        assert "3 nodes · 2 agents" in out


# ---------------------------------------------------------------------------
# App boot (headless Textual pilot)
# ---------------------------------------------------------------------------


def make_app(tmp_path: Path) -> LaunchApp:
    path = tmp_path / "market-analysis.yaml"
    path.write_text("version: '0.3'\n")  # content unused; the session is given the workflow directly
    session = WorkflowSession(build_workflow(), path, "market-analysis")
    return LaunchApp(session)


class TestAppBoot:
    @pytest.mark.asyncio
    async def test_boots_and_paints_chrome(self, tmp_path: Path) -> None:
        app = make_app(tmp_path)
        async with app.run_test() as pilot:
            await pilot.pause()
            status = app.query_one(StatusBar)
            assert "market-analysis" in status.left_text.plain
            rail = app.query_one(WorkflowRail)
            assert "analyze" in rail.body_text.plain
            assert "risk" in rail.body_text.plain

    @pytest.mark.asyncio
    async def test_slash_opens_palette(self, tmp_path: Path) -> None:
        app = make_app(tmp_path)
        async with app.run_test() as pilot:
            command_input = app.query_one(CommandInput)
            command_input.value = "/s"
            await pilot.pause()
            palette = app.query_one(CommandPalette)
            assert palette.has_class("open")
            assert palette.option_count >= 1

    @pytest.mark.asyncio
    async def test_help_command_writes_to_transcript(self, tmp_path: Path) -> None:
        app = make_app(tmp_path)
        async with app.run_test() as pilot:
            command_input = app.query_one(CommandInput)
            command_input.value = "/help"
            await command_input.action_submit()
            await pilot.pause()
            transcript = app.query_one(Transcript)
            assert any("commands" in str(line) for line in transcript.lines) or transcript.lines

    @pytest.mark.asyncio
    async def test_toggle_rail_hides_it(self, tmp_path: Path) -> None:
        app = make_app(tmp_path)
        async with app.run_test(size=(160, 40)) as pilot:
            await pilot.press("ctrl+b")
            assert app.query_one("#rail", WorkflowRail).has_class("hidden")
