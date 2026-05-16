"""Unit tests for tool node models, adapters, and executor integration."""

from __future__ import annotations

import json
from io import BytesIO
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sirenspec.core.models import (
    AgentNode,
    HttpToolConfig,
    PythonToolConfig,
    ToolNode,
    Workflow,
)
from sirenspec.exceptions import ToolError

# ---------------------------------------------------------------------------
# Model validation
# ---------------------------------------------------------------------------


class TestToolNodeModels:
    def test_http_tool_node_defaults(self) -> None:
        node = ToolNode.model_validate(
            {
                "type": "tool",
                "tool": "http",
                "config": {"url": "https://example.com/api"},
            }
        )
        assert node.tool == "http"
        assert isinstance(node.config, HttpToolConfig)
        assert node.config.method == "GET"
        assert node.config.timeout == 10
        assert node.output_key == "output"
        assert node.retry == 0
        assert node.on_failure == "raise"

    def test_http_tool_node_full(self) -> None:
        node = ToolNode.model_validate(
            {
                "type": "tool",
                "tool": "http",
                "config": {
                    "url": "https://api.example.com/data",
                    "method": "POST",
                    "headers": {"Authorization": "Bearer token"},
                    "body": '{"key": "value"}',
                    "timeout": 30,
                },
                "output_key": "api_data",
                "retry": 2,
                "on_failure": "skip",
            }
        )
        assert node.config.method == "POST"
        assert node.config.timeout == 30
        assert node.output_key == "api_data"
        assert node.retry == 2
        assert node.on_failure == "skip"

    def test_python_tool_node(self) -> None:
        node = ToolNode.model_validate(
            {
                "type": "tool",
                "tool": "python",
                "config": {
                    "module": "mypackage.tools",
                    "function": "parse_diff",
                    "args": {"raw": "some diff"},
                },
            }
        )
        assert node.tool == "python"
        assert isinstance(node.config, PythonToolConfig)
        assert node.config.module == "mypackage.tools"
        assert node.config.function == "parse_diff"
        assert node.config.args == {"raw": "some diff"}

    def test_workflow_with_tool_node(self) -> None:
        wf = Workflow.model_validate(
            {
                "version": "0.1",
                "agents": {
                    "assistant": {"model": "openai:gpt-4o-mini", "system": "Help."},
                },
                "nodes": {
                    "fetch": {
                        "type": "tool",
                        "tool": "http",
                        "config": {"url": "https://api.example.com"},
                    },
                    "answer": {"agent": "assistant", "writes": "output.reply"},
                },
                "edges": [{"from": "fetch", "to": "answer"}],
            }
        )
        assert isinstance(wf.nodes["fetch"], ToolNode)
        assert isinstance(wf.nodes["answer"], AgentNode)

    def test_workflow_agent_node_still_works_without_type(self) -> None:
        wf = Workflow.model_validate(
            {
                "version": "0.1",
                "agents": {"a": {"model": "openai:gpt-4o-mini", "system": "S"}},
                "nodes": {"n": {"agent": "a", "writes": "output.x"}},
            }
        )
        assert isinstance(wf.nodes["n"], AgentNode)


# ---------------------------------------------------------------------------
# HTTP adapter
# ---------------------------------------------------------------------------


def _make_http_response(
    body: str | bytes,
    status: int = 200,
    content_type: str = "text/plain",
) -> MagicMock:
    """Build a fake urllib response context-manager."""
    if isinstance(body, str):
        body = body.encode("utf-8")
    mock_resp = MagicMock()
    mock_resp.read.return_value = body
    mock_resp.headers.get.return_value = content_type
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


class TestHttpAdapter:
    @pytest.mark.asyncio
    async def test_get_returns_text_body(self) -> None:
        from sirenspec.tools.http_adapter import run_http_tool

        config = HttpToolConfig(url="https://example.com/text")
        mock_resp = _make_http_response("hello world")

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = await run_http_tool(config)

        assert result == "hello world"

    @pytest.mark.asyncio
    async def test_get_parses_json_response(self) -> None:
        from sirenspec.tools.http_adapter import run_http_tool

        config = HttpToolConfig(url="https://api.example.com/data")
        payload = {"status": "ok", "value": 42}
        mock_resp = _make_http_response(json.dumps(payload), content_type="application/json")

        with patch("urllib.request.urlopen", return_value=mock_resp):
            result = await run_http_tool(config)

        assert result == payload

    @pytest.mark.asyncio
    async def test_http_4xx_raises_tool_error(self) -> None:
        import urllib.error

        from sirenspec.tools.http_adapter import run_http_tool

        config = HttpToolConfig(url="https://api.example.com/missing")

        http_error = urllib.error.HTTPError(
            url="https://api.example.com/missing",
            code=404,
            msg="Not Found",
            hdrs=MagicMock(),  # type: ignore[arg-type]
            fp=BytesIO(b"resource not found"),
        )

        with patch("urllib.request.urlopen", side_effect=http_error):
            with pytest.raises(ToolError) as exc_info:
                await run_http_tool(config)

        assert "404" in str(exc_info.value)
        assert exc_info.value.tool_name == "http"

    @pytest.mark.asyncio
    async def test_http_5xx_raises_tool_error(self) -> None:
        import urllib.error

        from sirenspec.tools.http_adapter import run_http_tool

        config = HttpToolConfig(url="https://api.example.com/broken")

        http_error = urllib.error.HTTPError(
            url="https://api.example.com/broken",
            code=503,
            msg="Service Unavailable",
            hdrs=MagicMock(),  # type: ignore[arg-type]
            fp=BytesIO(b"service down"),
        )

        with patch("urllib.request.urlopen", side_effect=http_error):
            with pytest.raises(ToolError) as exc_info:
                await run_http_tool(config)

        assert "503" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_network_error_raises_tool_error(self) -> None:
        import urllib.error

        from sirenspec.tools.http_adapter import run_http_tool

        config = HttpToolConfig(url="https://unreachable.example.com/")

        url_error = urllib.error.URLError(reason="Name or service not known")
        with patch("urllib.request.urlopen", side_effect=url_error):
            with pytest.raises(ToolError) as exc_info:
                await run_http_tool(config)

        assert "Network error" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_post_request_sends_body(self) -> None:
        from sirenspec.tools.http_adapter import run_http_tool

        config = HttpToolConfig(url="https://api.example.com/post", method="POST", body='{"hello": "world"}')
        mock_resp = _make_http_response('{"ok": true}', content_type="application/json")

        captured_requests: list[Any] = []

        def fake_urlopen(req: Any, timeout: int = 10) -> Any:
            captured_requests.append(req)
            return mock_resp

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            result = await run_http_tool(config)

        assert result == {"ok": True}
        req = captured_requests[0]
        assert req.get_method() == "POST"


# ---------------------------------------------------------------------------
# Python callable adapter
# ---------------------------------------------------------------------------


class TestPythonAdapter:
    @pytest.mark.asyncio
    async def test_calls_function_with_args(self) -> None:
        from sirenspec.tools.python_adapter import run_python_tool

        # ``math.sqrt`` is a builtin and does NOT accept keyword args.
        # Use ``json.dumps`` instead which accepts keyword args.
        config2 = PythonToolConfig(module="json", function="dumps", args={"obj": {"a": 1}})
        result = await run_python_tool(config2)
        assert result == '{"a": 1}'

    @pytest.mark.asyncio
    async def test_module_not_found_raises_tool_error(self) -> None:
        from sirenspec.tools.python_adapter import run_python_tool

        config = PythonToolConfig(module="nonexistent_module_xyz", function="some_func")

        with pytest.raises(ToolError) as exc_info:
            await run_python_tool(config)

        assert "Cannot import module" in str(exc_info.value)
        assert exc_info.value.tool_name == "python"

    @pytest.mark.asyncio
    async def test_missing_function_raises_tool_error(self) -> None:
        from sirenspec.tools.python_adapter import run_python_tool

        config = PythonToolConfig(module="json", function="nonexistent_function_xyz")

        with pytest.raises(ToolError) as exc_info:
            await run_python_tool(config)

        assert "no attribute" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_function_exception_raises_tool_error(self) -> None:
        from sirenspec.tools.python_adapter import run_python_tool

        # json.loads("not-json") will raise a JSONDecodeError.
        config = PythonToolConfig(module="json", function="loads", args={"s": "not valid json !!!"})

        with pytest.raises(ToolError) as exc_info:
            await run_python_tool(config)

        assert "JSONDecodeError" in str(exc_info.value) or "json" in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_async_function_is_awaited(self) -> None:
        """Verify that async callables are properly awaited by the adapter."""
        import types

        from sirenspec.tools.python_adapter import run_python_tool

        # Create a minimal fake module with an async function.
        fake_module = types.ModuleType("fake_async_module")

        async def async_greet(name: str) -> str:
            return f"hello {name}"

        fake_module.greet = async_greet  # type: ignore[attr-defined]

        config = PythonToolConfig(module="fake_async_module", function="greet", args={"name": "world"})

        with patch("importlib.import_module", return_value=fake_module):
            result = await run_python_tool(config)

        assert result == "hello world"


# ---------------------------------------------------------------------------
# Executor integration: tool nodes
# ---------------------------------------------------------------------------


def _make_provider_mock(response_text: str = "mock response", tokens: int = 10) -> MagicMock:
    mock = MagicMock()
    mock.complete = AsyncMock(return_value=response_text)
    mock.last_token_count = tokens
    return mock


class TestExecutorToolNodes:
    @pytest.mark.asyncio
    async def test_tool_node_result_stored_in_context(self) -> None:
        from sirenspec.core.executor import execute

        wf = Workflow.model_validate(
            {
                "version": "0.1",
                "agents": {},
                "nodes": {
                    "fetch": {
                        "type": "tool",
                        "tool": "http",
                        "config": {"url": "https://api.example.com/data"},
                        "output_key": "api_data",
                    }
                },
            }
        )

        mock_resp = _make_http_response('{"result": "42"}', content_type="application/json")
        with patch("urllib.request.urlopen", return_value=mock_resp):
            trace = await execute(wf, "start")

        assert trace["summary"]["status"] == "success"
        tool_node = trace["nodes"][0]
        assert tool_node["id"] == "fetch"
        assert tool_node["type"] == "tool"
        assert tool_node["result"] == {"result": "42"}

    @pytest.mark.asyncio
    async def test_tool_node_failure_raises_and_stops_execution(self) -> None:
        import urllib.error

        from sirenspec.core.executor import execute

        wf = Workflow.model_validate(
            {
                "version": "0.1",
                "agents": {},
                "nodes": {
                    "fetch": {
                        "type": "tool",
                        "tool": "http",
                        "config": {"url": "https://api.example.com/fail"},
                    }
                },
            }
        )

        http_error = urllib.error.HTTPError(
            url="https://api.example.com/fail",
            code=500,
            msg="Internal Server Error",
            hdrs=MagicMock(),  # type: ignore[arg-type]
            fp=BytesIO(b"server error"),
        )

        with patch("urllib.request.urlopen", side_effect=http_error):
            trace = await execute(wf, "start")

        assert trace["summary"]["status"] == "failed"
        assert trace["nodes"][0]["error"] is not None
        assert "500" in trace["nodes"][0]["error"]

    @pytest.mark.asyncio
    async def test_tool_node_on_failure_skip_continues(self) -> None:
        import urllib.error

        from sirenspec.core.executor import execute

        wf = Workflow.model_validate(
            {
                "version": "0.1",
                "agents": {
                    "assistant": {"model": "openai:gpt-4o-mini", "system": "Help."},
                },
                "nodes": {
                    "fetch": {
                        "type": "tool",
                        "tool": "http",
                        "config": {"url": "https://api.example.com/fail"},
                        "on_failure": "skip",
                    },
                    "answer": {"agent": "assistant", "writes": "output.reply"},
                },
                "edges": [{"from": "fetch", "to": "answer"}],
            }
        )

        http_error = urllib.error.HTTPError(
            url="https://api.example.com/fail",
            code=404,
            msg="Not Found",
            hdrs=MagicMock(),  # type: ignore[arg-type]
            fp=BytesIO(b"not found"),
        )

        mock_provider = _make_provider_mock("fallback response")
        with patch("urllib.request.urlopen", side_effect=http_error):
            with patch("sirenspec.core.agent_runner.resolve_provider", return_value=mock_provider):
                trace = await execute(wf, "start")

        assert trace["summary"]["status"] == "success"
        # Fetch node should have null result but no error stopping execution.
        fetch_node = next(n for n in trace["nodes"] if n["id"] == "fetch")
        assert fetch_node["result"] is None

    @pytest.mark.asyncio
    async def test_tool_node_retry_on_failure(self) -> None:
        import urllib.error

        from sirenspec.core.executor import execute

        wf = Workflow.model_validate(
            {
                "version": "0.1",
                "agents": {},
                "nodes": {
                    "fetch": {
                        "type": "tool",
                        "tool": "http",
                        "config": {"url": "https://api.example.com/flaky"},
                        "retry": 2,
                        "on_failure": "raise",
                    }
                },
            }
        )

        call_count = 0

        def flaky_urlopen(req: Any, timeout: int = 10) -> Any:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise urllib.error.HTTPError(
                    url="https://api.example.com/flaky",
                    code=503,
                    msg="Service Unavailable",
                    hdrs=MagicMock(),  # type: ignore[arg-type]
                    fp=BytesIO(b"retry later"),
                )
            return _make_http_response("ok")

        with patch("urllib.request.urlopen", side_effect=flaky_urlopen):
            trace = await execute(wf, "start")

        assert trace["summary"]["status"] == "success"
        # 3 total calls: 2 failures + 1 success (retry=2 means 3 attempts total)
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_python_tool_node_in_executor(self) -> None:
        import types

        from sirenspec.core.executor import execute

        fake_module = types.ModuleType("fake_tools")

        def compute(x: int) -> int:
            return x * 2

        fake_module.compute = compute  # type: ignore[attr-defined]

        wf = Workflow.model_validate(
            {
                "version": "0.1",
                "agents": {},
                "nodes": {
                    "calc": {
                        "type": "tool",
                        "tool": "python",
                        "config": {
                            "module": "fake_tools",
                            "function": "compute",
                            "args": {"x": 21},
                        },
                        "output_key": "result",
                    }
                },
            }
        )

        with patch("importlib.import_module", return_value=fake_module):
            trace = await execute(wf, "start")

        assert trace["summary"]["status"] == "success"
        calc_node = trace["nodes"][0]
        assert calc_node["result"] == 42

    @pytest.mark.asyncio
    async def test_missing_output_key_defaults_to_output(self) -> None:
        from sirenspec.core.executor import execute

        wf = Workflow.model_validate(
            {
                "version": "0.1",
                "agents": {},
                "nodes": {
                    "fetch": {
                        "type": "tool",
                        "tool": "http",
                        "config": {"url": "https://api.example.com"},
                        # output_key omitted — should default to "output"
                    }
                },
            }
        )

        mock_resp = _make_http_response("plain text")
        with patch("urllib.request.urlopen", return_value=mock_resp):
            trace = await execute(wf, "start")

        assert trace["summary"]["status"] == "success"
        # Result stored under working.fetch.output (default output_key).
        assert trace["nodes"][0]["output_key"] == "output"
