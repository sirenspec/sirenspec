"""Static analysis module for the code-health-report cookbook recipe."""

import ast
import os
import pathlib
from typing import Any


def run_analysis() -> dict[str, Any]:
    """Analyze a Python source file and return code quality metrics.

    Reads from CODE_HEALTH_FILE env var if set, otherwise falls back to
    the bundled sample.py in this directory.

    :returns: Dict of static analysis metrics.
    """
    source_path = os.environ.get("CODE_HEALTH_FILE")
    if source_path:
        source = pathlib.Path(source_path).read_text()
    else:
        source = (pathlib.Path(__file__).parent / "sample.py").read_text()

    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return {"error": f"Syntax error — cannot parse file: {exc}"}

    lines = source.splitlines()
    functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
    classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]

    avg_fn_length = (
        round(sum(len(ast.unparse(f).splitlines()) for f in functions) / len(functions), 1) if functions else 0.0
    )

    return {
        "total_lines": len(lines),
        "blank_lines": sum(1 for line in lines if not line.strip()),
        "comment_lines": sum(1 for line in lines if line.strip().startswith("#")),
        "functions": len(functions),
        "classes": len(classes),
        "avg_function_length_lines": avg_fn_length,
        "max_function_args": max((len(f.args.args) for f in functions), default=0),
    }
