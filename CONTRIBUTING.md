# Contributing to SirenSpec

Thanks for your interest in contributing. This document covers how to get set up, what the standards are, and how to submit changes.

## Getting started

```bash
git clone https://github.com/sirenspec/sirenspec.git
cd sirenspec
uv sync --extra dev
source .venv/bin/activate
```

Python 3.13 is required (enforced via `.python-version`).

## Running tests

```bash
pytest                                          # full suite
pytest tests/path/test_foo.py::test_name        # single test
```

Tests mock all LLM providers — no API keys required.

## Code style

```bash
ruff format .         # format
ruff check . --fix    # lint and auto-fix
```

The CI gate runs both. PRs that fail linting will not be merged.

Key conventions:
- Every public function and variable must have type annotations
- All I/O-bound work is `async/await`
- No `_underscore` private helper methods — if logic needs a name, it belongs on a public method or a module-level function
- Use the `SirenSpecError` exception hierarchy; avoid raising built-ins for domain errors
- Docstrings follow Sphinx style (`:param`, `:returns:` — no `:type:` or `:rtype:`)

## Submitting a pull request

1. Open an issue first for non-trivial changes so we can discuss the approach
2. Create a branch off `develop` (not `main`)
3. Keep PRs focused — one feature or fix per PR
4. Add or update tests for any changed behavior
5. Update `CHANGELOG.md` under an `[Unreleased]` section

## What we're looking for

- Bug fixes with a clear reproduction case
- New guardrail types
- New node types (see issue tracker for planned additions)
- Additional provider adapters
- Cookbook recipes demonstrating real-world use cases

## What we're not looking for (right now)

- Large-scale refactors without prior discussion
- New dependencies without justification — the core dependency list is intentionally small

## Questions?

Open a [GitHub Discussion](https://github.com/sirenspec/sirenspec/discussions) or file an issue.
