# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SirenSpec is a YAML-first agent orchestration SDK. It lets developers define multi-agent workflows in human-readable YAML and execute them against OpenAI or Anthropic backends. The MVP consists of:

- A Pydantic v2 data model layer for YAML workflow definitions
- An async execution engine that walks the node/edge graph
- A pluggable LLM provider layer (OpenAI + Anthropic)
- A guardrail framework (injection detection, length limits, extensible)
- A Typer CLI (`sirenspec run` / `sirenspec validate`)
- A JSON Schema artifact for IDE autocomplete

## Setup

```bash
uv sync --extra dev       # install all deps including dev tools
source .venv/bin/activate # required before running any Python command
```

Python 3.13 is enforced via `.python-version`. Always activate the venv; ruff, pytest, and the CLI are not on the system PATH otherwise.

## Common Commands

```bash
ruff format .             # format code
ruff check . --fix        # lint and auto-fix
pytest                    # run all tests
pytest tests/path/test_foo.py::test_name  # run a single test
```

## Python Standards

**Type annotations** — every function signature and public variable must have type hints. No unannotated public surface.

**Async** — all I/O-bound work is `async/await`. Never wrap async code in sync shims.

**Error handling** — use a custom exception hierarchy: `SirenSpecError` as the base, with subtypes per subsystem (e.g., `ProviderError`, `GuardrailError`, `ValidationError`). Raise built-ins only for programmer errors (e.g., `TypeError` for bad arguments).

**Testing** — pytest for unit tests; Hypothesis for property-based tests on models, guardrails, and schema validation. Mock LLM providers rather than making real API calls.

**Style** — Ruff enforces formatting (120-char line length) and linting (E/W/F/I/B/C4/UP rule sets). Run `ruff format` before `ruff check`.

**Docstrings** - Use Sphinx style docstrings without :rtype, and :type. Each docstring should have a return type hint.

**Line length** - Maximum line length is 120 characters.

## OOP Principles

**Single responsibility** — each class has one reason to change. A class that handles parsing, validation, and execution is three classes.

**Composition over inheritance** — prefer composing objects with injected dependencies over deep inheritance chains. Inherit only when there is a genuine is-a relationship (e.g., provider subtypes extending a base provider).

**Dependency injection** — classes receive their dependencies via `__init__`, never instantiate external collaborators internally. This keeps classes testable without patching.

**DRY** — shared logic lives in one place. Before adding a method, check if the behavior already exists elsewhere. Extract reusable logic to a shared module or base class rather than duplicating it across classes.

**Interfaces via Protocol** — define structural interfaces with `typing.Protocol` rather than ABCs unless default implementations are needed. Provider and guardrail interfaces should be Protocols.

## Methods and Functions

**No private helper methods** — do not use `_underscore` methods to hide internal logic inside a class. If a piece of logic needs a name, it belongs as a public method on an appropriate class or as a module-level function in a focused utility module. A class with several `_private` helpers is a signal the class is doing too much — split it.

**Public methods should read as intent** — a method body should be a clear sequence of calls to other well-named methods or functions, not a block of inline logic. Decompose by extracting to other classes or module-level functions, not by adding private methods.

**Module-level functions for stateless logic** — pure transformations with no dependency on class state belong at module scope, not hidden as `_helpers` on a class.

## Anti-Patterns

- No `_private` or `__dunder` methods for encapsulation of helper logic
- No God classes — if a class has more than ~5 public methods, question whether it has a single responsibility
- No business logic inside Pydantic models — models are data containers, not orchestrators
- No bare `dict` or `Any` passed between module boundaries — use typed models
- No inline logic duplication — if you've written it twice, extract it

## OpenSpec Workflow

Changes are managed through OpenSpec (`.claude/commands/` and `.opencode/`). Proposed changes live in `openspec/changes/<name>/` as structured YAML + markdown before being applied to code. Use `/opsx` commands to propose, explore, and apply changes.