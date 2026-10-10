"""Named, versioned case sets a launch can target, and history epochs labelled by them (#676).

A set is append-only (``smoke v1`` cannot be rewritten; a change mints ``v2``), a launch naming one runs exactly
its cases and records it, and history labels each epoch by the set its runs were launched against while still
deciding the epoch from the frozen case ids.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from pydantic import ValidationError

from threetears.evals.contracts import (
    CaseSet,
    CaseSetRef,
    ConflictError,
    EvalRun,
    EvalStorage,
    NotFoundError,
    ValidationFailedError,
)
from threetears.evals.ops import LaunchArguments, case_set_mint, case_sets_list, history_text, scope_history
from threetears.evals.ops import CaseSetMint
from threetears.evals.run import LaunchHost, mint_case_set, start_run
from threetears.evals.storage import InMemoryDocumentStore
from packages.evals.tests.fixtures.toyhost.corpus import TOYHOST_SCOPE, TOYHOST_SUBJECT
from packages.evals.tests.fixtures.toyhost.launch import toyhost_launch_host
from packages.evals.tests.fixtures.toyhost.run import RUN_MODELS, toyhost_template, toyhost_test_cases

TEMPLATE = toyhost_template()
CASES = [case.id for case in toyhost_test_cases(TEMPLATE)]


def _launching() -> tuple[LaunchHost, EvalStorage]:
    storage = EvalStorage(InMemoryDocumentStore())
    storage.save_template(TEMPLATE)
    for case in toyhost_test_cases(TEMPLATE):
        storage.save_test_case(case)
    host, _client = toyhost_launch_host(storage=storage)
    return host, storage


def _mint(storage: EvalStorage, case_ids: list[str], *, name: str = "smoke") -> CaseSet:
    return mint_case_set(storage, scope_id=TOYHOST_SCOPE, name=name, template_id=TEMPLATE.id, test_case_ids=case_ids)


async def _launch(host: LaunchHost, storage: EvalStorage, **arguments: Any) -> EvalRun:
    (run,) = await start_run(
        host,
        template_id=TEMPLATE.id,
        scope_id=TOYHOST_SCOPE,
        subject_id=TOYHOST_SUBJECT.subject_id,
        models=[RUN_MODELS[0]],
        k_runs=1,
        **arguments,
    )
    async with asyncio.timeout(10):
        while host.job_manager.is_active(run.id):
            await asyncio.sleep(0.01)
    stored = storage.load_eval_run(run.id, TOYHOST_SCOPE)
    assert stored is not None and stored.status == "completed"
    return stored


# =============================================================================
# Append-only, minted versions
# =============================================================================


def test_a_set_mints_from_v1_and_each_change_is_the_next_version() -> None:
    _host, storage = _launching()
    v1 = _mint(storage, CASES[:2])
    v2 = _mint(storage, CASES)
    assert (v1.ref.label, v2.ref.label) == ("smoke v1", "smoke v2")
    assert storage.load_case_set("smoke", 1, TOYHOST_SCOPE) == v1, "v1 still names its own cases"
    assert [s.version for s in storage.query_case_sets(TOYHOST_SCOPE, name="smoke")] == [2, 1]


def test_rewriting_a_stored_version_is_refused() -> None:
    _host, storage = _launching()
    v1 = _mint(storage, CASES[:2])
    rewritten = CaseSet(scope_id=TOYHOST_SCOPE, name="smoke", version=1, template_id=TEMPLATE.id, test_case_ids=CASES)
    with pytest.raises(ConflictError, match="append-only"):
        storage.save_case_set(rewritten)
    assert storage.load_case_set("smoke", 1, TOYHOST_SCOPE) == v1


def test_minting_refuses_an_unchanged_list_a_missing_case_and_another_templates_name() -> None:
    _host, storage = _launching()
    _mint(storage, CASES[:2])
    with pytest.raises(ValidationFailedError, match="already lists exactly these cases"):
        _mint(storage, CASES[:2])
    with pytest.raises(ValidationFailedError, match="no longer resolve in this scope: ghost"):
        _mint(storage, [CASES[0], "ghost"])
    with pytest.raises(ValidationFailedError, match="not 'other-template''s"):
        mint_case_set(storage, scope_id=TOYHOST_SCOPE, name="smoke", template_id="other-template", test_case_ids=CASES)


def test_a_set_lists_each_case_once() -> None:
    with pytest.raises(ValidationError, match="lists each case once"):
        CaseSet(scope_id=TOYHOST_SCOPE, name="s", version=1, template_id="t", test_case_ids=["a", "a"])


# =============================================================================
# A launch runs the set's cases and records it
# =============================================================================


async def test_a_launch_against_a_set_runs_exactly_its_cases_and_records_it() -> None:
    host, storage = _launching()
    _mint(storage, CASES[:2])

    run = await _launch(host, storage, case_set=CaseSetRef(name="smoke", version=1))

    assert run.case_set == CaseSetRef(name="smoke", version=1)
    assert run.test_case_ids == CASES[:2]
    assert {r.test_case_id for r in storage.query_eval_results_by_run(run.id, TOYHOST_SCOPE)} == set(CASES[:2])


async def test_a_launch_against_a_set_whose_case_is_gone_is_refused_naming_it() -> None:
    host, storage = _launching()
    _mint(storage, CASES[:2])
    storage.delete_test_case(CASES[1], TOYHOST_SCOPE)

    with pytest.raises(ValidationFailedError, match=f"no longer resolve in this scope: {CASES[1]}"):
        await _launch(host, storage, case_set=CaseSetRef(name="smoke", version=1))
    with pytest.raises(NotFoundError):
        await _launch(host, storage, case_set=CaseSetRef(name="smoke", version=9))


def test_the_launch_arguments_name_a_set_and_its_version_together() -> None:
    base = {"template_id": TEMPLATE.id, "subject_id": TOYHOST_SUBJECT.subject_id}
    with pytest.raises(ValidationError, match="case_set_name and case_set_version"):
        LaunchArguments.model_validate({**base, "case_set_name": "smoke"})
    named = LaunchArguments.model_validate({**base, "case_set_name": "smoke", "case_set_version": 2})
    assert named.case_set == CaseSetRef(name="smoke", version=2)


def test_the_operations_mint_and_list() -> None:
    host, _storage = _launching()
    minted = case_set_mint(
        host.eval_host,
        CaseSetMint(case_set="smoke", template_id=TEMPLATE.id, test_case_ids=CASES[:1], tracked=False),
        TOYHOST_SCOPE,
    )
    assert (minted.label, minted.tracked) == ("smoke v1", False)
    assert [line.label for line in case_sets_list(host.eval_host, TOYHOST_SCOPE).case_sets] == ["smoke v1"]


# =============================================================================
# History labels epochs by the set, and still epochs a set-less run by its ids
# =============================================================================


async def test_runs_against_two_versions_are_an_epoch_boundary_labelled_with_both() -> None:
    host, storage = _launching()
    _mint(storage, CASES[:2])
    _mint(storage, CASES)
    first = await _launch(host, storage, case_set=CaseSetRef(name="smoke", version=1))
    second = await _launch(host, storage, case_set=CaseSetRef(name="smoke", version=2))

    history = scope_history(host.eval_host, TOYHOST_SCOPE, metric="cost_usd")

    (series,) = history.series
    points = {point.run_id: point for point in series.points}
    assert (points[first.id].epoch, points[first.id].epoch_label) == (1, "smoke v1")
    assert (points[second.id].epoch, points[second.id].epoch_boundary, points[second.id].epoch_label) == (
        2,
        True,
        "smoke v2",
    )
    assert "suite changed here (smoke v1 to smoke v2)" in history_text(history)


async def test_a_run_without_a_set_still_epochs_by_its_frozen_ids() -> None:
    host, storage = _launching()
    _mint(storage, CASES[:2])
    with_set = await _launch(host, storage, case_set=CaseSetRef(name="smoke", version=1))
    without = await _launch(host, storage)
    again = await _launch(host, storage)

    (series,) = scope_history(host.eval_host, TOYHOST_SCOPE, metric="cost_usd").series
    points = {point.run_id: point for point in series.points}
    assert without.case_set is None and without.test_case_ids == CASES
    assert points[without.id].epoch_boundary, "the frozen ids changed, so the suite changed, set or no set"
    assert points[without.id].epoch == points[again.id].epoch == 2
    assert not points[again.id].epoch_boundary, "the same ids are one suite version"
    assert points[with_set.id].epoch_label == "smoke v1"
    assert points[without.id].epoch_label is None and points[again.id].epoch_label is None
