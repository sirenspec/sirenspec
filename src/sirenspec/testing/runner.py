"""Test discovery, execution, and result collection for SirenSpec YAML test fixtures."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import ValidationError
from ruamel.yaml import YAML

from sirenspec.core.executor import execute
from sirenspec.exceptions import ProviderError
from sirenspec.providers.registry import _PROVIDER_FACTORIES, set_provider_override
from sirenspec.testing.assertions import AssertionResult, evaluate_assertion
from sirenspec.testing.cassette import (
    Cassette,
    CassetteError,
    RecordingProvider,
    ReplayProvider,
    load_cassette,
    save_cassette,
)
from sirenspec.testing.models import WorkflowFixture
from sirenspec.yaml.parser import load_workflow


@dataclass
class AssertionOutcome:
    """The result of one assertion within a fixture.

    :param index: Zero-based index of the assertion in the fixture's list.
    :param result: The pass/fail result from :func:`~sirenspec.testing.assertions.evaluate_assertion`.
    """

    index: int
    result: AssertionResult


@dataclass
class FixtureResult:
    """The aggregated result of running a single test fixture.

    :param fixture_path: Path to the ``.test.yaml`` file.
    :param passed: ``True`` if all assertions passed and no errors occurred.
    :param error: A human-readable error message if the fixture could not run.
    :param assertion_outcomes: Per-assertion pass/fail outcomes.
    """

    fixture_path: Path
    passed: bool
    error: str | None = None
    assertion_outcomes: list[AssertionOutcome] = field(default_factory=list)


def discover_fixtures(path: Path) -> list[Path]:
    """Discover all ``*.test.yaml`` files under *path*.

    If *path* is a file, it is returned directly (regardless of naming convention).
    If *path* is a directory, it is walked recursively for ``*.test.yaml`` files.

    :param path: File or directory to search.
    :returns: Sorted list of discovered fixture paths.
    :raises FileNotFoundError: If *path* does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Test path not found: {path}")
    if path.is_file():
        return [path]
    return sorted(path.rglob("*.test.yaml"))


def load_fixture(fixture_path: Path) -> WorkflowFixture:
    """Parse and validate a ``.test.yaml`` fixture file.

    :param fixture_path: Path to the fixture file.
    :raises ValueError: If the YAML is malformed or fails model validation.
    :returns: Validated :class:`~sirenspec.testing.models.WorkflowFixture`.
    """
    yaml = YAML(typ="safe")
    try:
        raw: Any = yaml.load(fixture_path)
    except Exception as exc:
        raise ValueError(f"YAML parse error in '{fixture_path}': {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a YAML mapping in '{fixture_path}'")
    try:
        return WorkflowFixture.model_validate(raw)
    except ValidationError as exc:
        field_errors = "; ".join(f"{'.'.join(str(loc) for loc in e['loc'])}: {e['msg']}" for e in exc.errors())
        raise ValueError(f"Fixture validation failed in '{fixture_path}': {field_errors}") from exc


def resolve_workflow_path(fixture_path: Path, workflow_ref: str) -> Path:
    """Resolve the workflow path relative to the fixture file's directory.

    :param fixture_path: Path to the ``.test.yaml`` file.
    :param workflow_ref: The ``workflow:`` value from the fixture (relative or absolute).
    :returns: Resolved absolute path to the workflow YAML file.
    """
    ref = Path(workflow_ref)
    if ref.is_absolute():
        return ref
    return (fixture_path.parent / ref).resolve()


def make_replay_factory(cassette: Cassette):
    """Return a provider factory function that replays from *cassette*.

    :param cassette: The cassette to replay interactions from.
    :returns: A ``(uri: str) -> LLMProvider`` callable suitable for :func:`~sirenspec.providers.registry.set_provider_override`.
    """

    def factory(uri: str) -> ReplayProvider:
        return ReplayProvider(uri=uri, cassette=cassette)

    return factory


def make_recording_factory(cassette: Cassette):
    """Return a provider factory function that records into *cassette* while delegating to real providers.

    :param cassette: The cassette to append recorded interactions to.
    :returns: A ``(uri: str) -> LLMProvider`` callable suitable for :func:`~sirenspec.providers.registry.set_provider_override`.
    """

    def factory(uri: str) -> RecordingProvider:
        if ":" not in uri:
            raise ProviderError(f"Malformed provider URI '{uri}'")
        provider_name, _, model = uri.partition(":")
        real_factory = _PROVIDER_FACTORIES.get(provider_name)
        if real_factory is None:
            raise ProviderError(f"Unknown provider '{provider_name}'")
        real = real_factory(model)
        return RecordingProvider(uri=uri, real_provider=real, cassette=cassette)

    return factory


async def run_fixture(
    fixture_path: Path,
    cassette_path: Path | None,
    mode: str,
) -> FixtureResult:
    """Run a single test fixture and return its result.

    :param fixture_path: Path to the ``.test.yaml`` file.
    :param cassette_path: Path to the cassette file (required when *mode* is
        ``'mock'`` or ``'record'``; ignored when *mode* is ``'live'``).
    :param mode: One of ``'live'``, ``'mock'``, or ``'record'``.
    :returns: :class:`FixtureResult` with pass/fail outcomes for every assertion.
    """
    try:
        fixture = load_fixture(fixture_path)
    except ValueError as exc:
        return FixtureResult(fixture_path=fixture_path, passed=False, error=str(exc))

    workflow_path = resolve_workflow_path(fixture_path, fixture.workflow)
    try:
        workflow = load_workflow(workflow_path)
    except (FileNotFoundError, ValueError) as exc:
        return FixtureResult(fixture_path=fixture_path, passed=False, error=f"Workflow error: {exc}")

    cassette: Cassette | None = None

    if mode == "mock":
        if cassette_path is None:
            return FixtureResult(fixture_path=fixture_path, passed=False, error="--cassette required with --mock")
        try:
            cassette = load_cassette(cassette_path)
        except CassetteError as exc:
            return FixtureResult(fixture_path=fixture_path, passed=False, error=str(exc))
        set_provider_override(make_replay_factory(cassette))

    elif mode == "record":
        if cassette_path is None:
            return FixtureResult(fixture_path=fixture_path, passed=False, error="--cassette required with --record")
        cassette = Cassette()
        set_provider_override(make_recording_factory(cassette))

    try:
        trace = await execute(workflow, fixture.input)
    except Exception as exc:
        set_provider_override(None)
        return FixtureResult(fixture_path=fixture_path, passed=False, error=f"Execution error: {exc}")
    finally:
        set_provider_override(None)

    if mode == "record" and cassette is not None and cassette_path is not None:
        save_cassette(cassette, cassette_path)

    outcomes: list[AssertionOutcome] = []
    all_passed = True
    for i, assertion in enumerate(fixture.assertions):
        result = evaluate_assertion(assertion, trace)
        outcomes.append(AssertionOutcome(index=i, result=result))
        if not result.passed:
            all_passed = False

    return FixtureResult(fixture_path=fixture_path, passed=all_passed, assertion_outcomes=outcomes)


def run_fixtures(
    paths: list[Path],
    cassette_path: Path | None,
    mode: str,
) -> list[FixtureResult]:
    """Run all fixtures sequentially and collect results.

    :param paths: Fixture file paths to run.
    :param cassette_path: Cassette file path (for mock/record modes).
    :param mode: ``'live'``, ``'mock'``, or ``'record'``.
    :returns: One :class:`FixtureResult` per fixture.
    """
    results = []
    for fixture_path in paths:
        result = asyncio.run(run_fixture(fixture_path, cassette_path, mode))
        results.append(result)
    return results
