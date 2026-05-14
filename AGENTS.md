# AGENTS.md

## Quick Setup
- **Python version**: 3.13 (enforced via `.python-version`)
- **Package manager**: `uv` (faster, deterministic than pip/poetry)
- **Activate venv**: `source .venv/bin/activate`
- **Install deps**: `uv sync` (or `uv sync --extra dev` for dev tools)

## Verification Commands
Run these in order. Always activate venv first.

```bash
source .venv/bin/activate
ruff format .                    # Format code
ruff check . --fix               # Lint and fix
pytest                           # Run tests (currently 0 tests)
```

## OpenSpec Workflow
Project uses OpenSpec framework (`.opencode/skills/` and `.claude/commands/`):
- Proposed changes in `openspec/`
- Use `/opsx` commands via OpenCode for change management
- Changes tracked in structured format, not direct git commits initially

## Project Structure
- `main.py` - Minimal entry point (prints "Hello from siren-spec!")
- `openspec/` - OpenSpec change management and specs (gitignored)
- `.venv/` - Python virtual environment (Python 3.13)
- No test files yet; pytest configured but 0 tests collected

## Dependencies
- **Core**: Anthropic SDK, OpenAI SDK, Pydantic, Rich, Typer, ruamel-yaml
- **Dev**: Ruff, Pytest, pytest-asyncio, Hypothesis

## Python Standards
- **Type hints**: all function signatures and public variables must be annotated
- **Async**: `async/await` for all I/O-bound work; no sync wrappers
- **Errors**: `SirenSpecError` base class; subtypes per subsystem (`ProviderError`, `GuardrailError`, etc.)
- **Tests**: pytest unit tests + Hypothesis property tests; mock LLM providers
- **Docstrings** - Use Sphinx style docstrings without :rtype, and :type. Each docstring should have a return type hint.
- **Line length** - Maximum line length is 120 characters.

## Important Quirks / Gotchas
- Virtual environment activation **required** for all Python commands (ruff, pytest, etc.)
- No CI workflows yet (no `.github/actions`).
- README is empty; project just initialized.
