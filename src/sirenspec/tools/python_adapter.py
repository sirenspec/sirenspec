"""Python callable tool adapter: imports a module at runtime and calls a function."""

from __future__ import annotations

import asyncio
import importlib
import inspect
from typing import Any

from sirenspec.core.models import PythonToolConfig
from sirenspec.exceptions import ToolError


async def run_python_tool(config: PythonToolConfig) -> Any:
    """Import *config.module* at runtime and call *config.function* with *config.args*.

    The module is resolved relative to the user's runtime environment (i.e. ``sys.path``),
    not relative to the sirenspec package.  This allows users to point at any importable
    Python module in their project without installing it as part of sirenspec.

    :param config: A validated :class:`~sirenspec.core.models.PythonToolConfig` instance.
    :raises ToolError: If the module cannot be imported, the function does not exist on the
        module, or the function raises an exception during execution.
    :returns: The return value of the called function.
    """
    try:
        module = importlib.import_module(config.module)
    except ModuleNotFoundError as exc:
        raise ToolError(
            "python",
            f"Cannot import module '{config.module}': {exc}",
            cause=exc,
        ) from exc
    except Exception as exc:
        raise ToolError(
            "python",
            f"Error importing module '{config.module}': {exc}",
            cause=exc,
        ) from exc

    fn = getattr(module, config.function, None)
    if fn is None:
        raise ToolError(
            "python",
            f"Module '{config.module}' has no attribute '{config.function}'",
        )
    if not callable(fn):
        raise ToolError(
            "python",
            f"'{config.module}.{config.function}' is not callable",
        )

    kwargs: dict[str, Any] = config.args or {}

    try:
        if inspect.iscoroutinefunction(fn):
            # Async functions are awaited directly.
            result = await fn(**kwargs)
        else:
            # Sync callables run in a thread pool executor to avoid blocking the event loop.
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, lambda: fn(**kwargs))
    except ToolError:
        raise
    except Exception as exc:
        raise ToolError(
            "python",
            f"'{config.module}.{config.function}' raised {type(exc).__name__}: {exc}",
            cause=exc,
        ) from exc

    return result
