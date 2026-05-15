"""HTTP tool adapter: executes an HTTP request and returns the response body."""

from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.request
from typing import Any

from sirenspec.core.models import HttpToolConfig
from sirenspec.exceptions import ToolError


async def run_http_tool(config: HttpToolConfig) -> Any:
    """Execute an HTTP request defined by *config* and return a parsed JSON body or raw text.

    The request is performed using ``urllib`` from the standard library so that no
    additional HTTP dependency is required.  The function raises :class:`~sirenspec.exceptions.ToolError`
    on network errors, non-2xx status codes, and request timeouts.

    :param config: A validated :class:`~sirenspec.core.models.HttpToolConfig` instance.
    :raises ToolError: On any I/O error, timeout, or non-2xx HTTP response.
    :returns: Parsed JSON response (dict or list) if the ``Content-Type`` is JSON, otherwise the raw response text.
    """
    try:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, sync_request, config)
    except ToolError:
        raise
    except Exception as exc:
        raise ToolError("http", f"Unexpected error during HTTP request: {exc}", cause=exc) from exc


def sync_request(config: HttpToolConfig) -> Any:
    """Execute the HTTP request synchronously.

    This is called via ``run_in_executor`` so it does not block the event loop.

    :param config: Validated HTTP tool configuration.
    :raises ToolError: On HTTP errors, timeouts, or network failures.
    :returns: Parsed JSON or raw text response body.
    """
    url = config.url
    method = config.method.upper()
    timeout = config.timeout

    # Build the request body bytes (only relevant for POST).
    body_bytes: bytes | None = None
    if config.body is not None:
        body_bytes = config.body.encode("utf-8")

    # Build headers dict (caller can override Content-Type).
    headers: dict[str, str] = {}
    if config.headers:
        headers.update(config.headers)

    # Default Content-Type for POST with a body.
    if method == "POST" and body_bytes is not None and "Content-Type" not in headers:
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url, data=body_bytes, headers=headers, method=method)

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            raw_body = resp.read().decode("utf-8", errors="replace")
            content_type = resp.headers.get("Content-Type", "")
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
        raise ToolError(
            "http",
            f"HTTP {exc.code} {exc.reason} from {url}: {error_body[:200]}",
            cause=exc,
            status_code=exc.code,
        ) from exc
    except urllib.error.URLError as exc:
        raise ToolError("http", f"Network error reaching {url}: {exc.reason}", cause=exc) from exc
    except TimeoutError as exc:
        raise ToolError("http", f"Request to {url} timed out after {timeout}s", cause=exc) from exc

    # Attempt JSON parse when the response content-type indicates JSON.
    if "application/json" in content_type or "text/json" in content_type:
        try:
            return json.loads(raw_body)
        except json.JSONDecodeError:
            pass  # Fall back to raw text if JSON parse fails.

    return raw_body
