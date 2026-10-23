"""A run whose holding ended before its matrix did is taken up again, and ends as the same run.

Pinned:

- **The acceptance**: a run killed part-way, its lease lapsing with cells in flight, is taken up from a fresh process
  and ends exactly as an uninterrupted control run of it does: the same cells, one row each, the same pass^k,
  ``completed``, its context and variant keys unchanged — and it records one lifetime per process that held it. A
  run an operator cancelled, whose cut-short cells left ``cancelled`` rows, is taken up the same way, each of those
  rows written over; and so is one taken up by the very process that held it.
- **Only the same measurement**: a take-up that would measure otherwise is refused, naming every difference, and
  leaves the run as it was.
- **Its budget**: a run its budget stopped is taken up only under a cap above what it spent, and each holding
  records the cap it ran under.
- **Old records**: a run an older build stored and lost — no epochs, minted result ids — is taken up, its done rows
  kept and its cut-short rows written over under their own ids.
- **One holding**: of two take-ups racing, exactly one wins.
"""

from __future__ import annotations

import asyncio
import dataclasses
import uuid
from datetime import timedelta
from typing import Any

import pytest

from threetears.evals.analysis.completeness import completeness_disclosure
from threetears.evals.kernel import ConflictError, ValidationFailedError
from threetears.evals.kernel.scoring import compute_pass_hat_k
from threetears.evals.ops import OpsHost, job_poll, run_job_id
from threetears.evals.run import ArmPrice, ArmQuote
from threetears.evals.run.resume import counted_spend, resume_drift, resume_run
from threetears.evals.schema import EvalResult, EvalRun, JobLease
from threetears.evals.schema.models import cell_result_id, eval_trace_doc_id
from packages.evals.tests.factories import make_eval_run, make_template
from packages.evals.tests.fixtures.toyhost.contract import ExtractorSpec
from packages.evals.tests.fixtures.toyhost.corpus import TOYHOST_COST_CEILING_USD, TOYHOST_SUBJECT
from packages.evals.tests.fixtures.toyhost.launch import TOYHOST_LAUNCH_SETTINGS, TOYHOST_SUBJECTS
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile
from packages.evals.tests.fixtures.toyhost.run import toyhost_template
from packages.evals.tests.lease_support import T0, ClockedStore, held_extraction, launched_run, other_process, until
from packages.evals.tests.off_loop import off_loop
from packages.evals.tests.ops_support import TOYHOST_SCOPE, ops_fixture, settled

#: Repeats per case: three cases, nine cells.
K = 3

#: The cells every interrupted run stores before its holding ends.
DONE_BEFORE = 3


def _run(host: OpsHost, run_id: str) -> EvalRun:
    run, _ = host.eval_host.storage.load_job_record(EvalRun, run_id, TOYHOST_SCOPE)
    assert run is not None
    return run


def _rows(host: OpsHost, run_id: str) -> list[EvalResult]:
    return off_loop(host.eval_host.storage.query_eval_results_by_run, run_id, TOYHOST_SCOPE)


def _cells(rows: list[EvalResult]) -> list[tuple[str, str, int]]:
    return sorted((row.test_case_id, row.model, row.k_iteration) for row in rows)


def _holder_id(host: OpsHost) -> str:
    leases = host.launch.job_manager.leases
    assert leases is not None
    return leases.holder_id


async def _interrupted(monkeypatch: pytest.MonkeyPatch, how: str) -> tuple[OpsHost, str, EvalRun, ClockedStore]:
    """A run of nine cells, three stored and the rest in flight or not started, whose holding then ended ``how``.

    ``lost``: the holder stops renewing, as a killed process does; the store's clock passes the lease, another process
    reads the run lost, and then the dead holder's cells in flight finish and try to write. ``cancelled``: an operator
    cancels it, and the cells the cancel cuts short are recorded ``cancelled``.

    :return: The holder, the run's id, the run as launched, and the store.
    """
    gate = held_extraction(monkeypatch, passing=DONE_BEFORE)
    store = ClockedStore()
    holder = off_loop(ops_fixture, store=store).host
    run_id = await launched_run(holder, K)
    await until(lambda: len(_rows(holder, run_id)) == DONE_BEFORE and gate.held >= 1)
    launched = off_loop(_run, holder, run_id)
    if how == "lost":
        store.at = T0 + timedelta(seconds=61)
        assert (await job_poll(other_process(holder), run_job_id(run_id), TOYHOST_SCOPE)).state == "lost"
        in_flight = gate.held
        assert in_flight >= 1, "cells were in flight when the lease lapsed"
        gate.set()
        await until(lambda: gate.released >= in_flight and not holder.launch.job_manager.is_active(run_id))
        assert len(_rows(holder, run_id)) == DONE_BEFORE, "the dead holder's cells in flight finished and wrote nothing"
    else:
        assert holder.launch.job_manager.cancel_job(run_id, reason="shipping a fix")
        await until(lambda: not holder.launch.job_manager.is_active(run_id))
        assert "cancelled" in {row.termination for row in _rows(holder, run_id)}, "the cancel cut cells short"
        gate.set()
    return holder, run_id, launched, store


#: What each cell a cancel cut short had spent, set on its row: the toy's cells are cut short waiting on their one
#: call, which spends nothing, while a real cell cut short has usually paid for some of its calls.
CUT_SHORT_USD = 0.05


def _cut_short_after_spending(store: ClockedStore, run_id: str) -> None:
    """Give every cut-short row of the run what it had spent (:data:`CUT_SHORT_USD`)."""
    for document in list(store.documents.values()):
        if document.get("eval_run_id") == run_id and document.get("termination") == "cancelled":
            store.upsert(document | {"cost_usd": CUT_SHORT_USD})


async def _control() -> tuple[OpsHost, str]:
    """The same launch on a store of its own, never interrupted."""
    control = off_loop(ops_fixture, store=ClockedStore()).host
    run_id = await launched_run(control, K)
    assert (await settled(control, run_job_id(run_id))).state == "completed"
    return control, run_id


async def _assert_ends_as_the_control(holder: OpsHost, run_id: str, launched: EvalRun) -> EvalRun:
    """The run taken up ended exactly as an uninterrupted run of it does; the run, as it ended."""
    control, control_id = await _control()
    run, rows = off_loop(_run, holder, run_id), _rows(holder, run_id)
    control_run, control_rows = off_loop(_run, control, control_id), _rows(control, control_id)
    assert _cells(rows) == _cells(control_rows), "the same cells"
    assert len(rows) == len(control_rows) == 3 * K, "each stored once"
    assert list(compute_pass_hat_k(rows).values()) == list(compute_pass_hat_k(control_rows).values()), "pass^k"
    assert "cancelled" not in {row.termination for row in rows}, "no abandoned attempt is a result"
    assert run.status == control_run.status == "completed"
    assert run.progress == control_run.progress, "its progress counts every cell, the earlier holding's too"
    assert (run.context_key, run.variant_levers) == (launched.context_key, launched.variant_levers)
    assert run.context_key == control_run.context_key
    assert {row.variant_key for row in rows} == {row.variant_key for row in control_rows}
    assert run.completeness is not None and completeness_disclosure(run.completeness) is None, "never partial"
    assert (run.completeness.expected_cells, run.completeness.persisted_cells) == (3 * K, 3 * K)
    assert (run.lost, run.cancellation_reason, run.budget_stop_reason) == (None, None, None)
    return run


# =============================================================================
# The acceptance
# =============================================================================


@pytest.mark.parametrize("how", ["lost", "cancelled"])
async def test_a_run_taken_up_again_from_a_fresh_process_ends_as_an_uninterrupted_run_does(
    monkeypatch: pytest.MonkeyPatch, how: str
) -> None:
    holder, run_id, launched, store = await _interrupted(monkeypatch, how)
    _cut_short_after_spending(store, run_id)
    before = _rows(holder, run_id)
    cut_short = [row for row in before if row.termination == "cancelled"]
    resumer = other_process(holder)

    taken_up = await resume_run(resumer.launch, run_id, TOYHOST_SCOPE)
    assert taken_up.status == "running"
    assert (await settled(resumer, run_job_id(run_id))).state == "completed"

    run = await _assert_ends_as_the_control(holder, run_id, launched)
    assert {row.id for row in _rows(holder, run_id)} == {
        cell_result_id(run_id, *cell) for cell in _cells(_rows(holder, run_id))
    }
    first, second = run.lifetimes
    assert (first.holder, first.epoch, first.status) == (_holder_id(holder), 0, "cancelled")
    assert first.lost is not None if how == "lost" else first.reason == "shipping a fix"
    assert first.spend_lower_bound is (how == "lost")
    assert first.cells_stored == len(before)
    assert (second.holder, second.epoch, second.status) == (_holder_id(resumer), 1, "completed")
    assert second.completeness == run.completeness
    after = {row.id: row for row in _rows(holder, run_id)}
    for row in cut_short:
        assert after[row.id].superseded_spend_usd == pytest.approx(CUT_SHORT_USD), "its cut-short attempt's spend moved"
    assert sum(amount or 0.0 for amount in counted_spend(list(after.values()))) == pytest.approx(
        sum(row.cost_usd or 0.0 for row in before)
        + sum(row.cost_usd or 0.0 for row in after.values() if row.lease_epoch)
    ), "every attempt's spend, once"


async def test_an_operator_cancelled_run_is_taken_up_again_by_the_process_that_held_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    holder, run_id, launched, _ = await _interrupted(monkeypatch, "cancelled")

    await resume_run(holder.launch, run_id, TOYHOST_SCOPE)
    assert (await settled(holder, run_job_id(run_id))).state == "completed"

    run = await _assert_ends_as_the_control(holder, run_id, launched)
    assert [(lifetime.holder, lifetime.epoch) for lifetime in run.lifetimes] == [
        (_holder_id(holder), 0),
        (_holder_id(holder), 1),
    ]


# =============================================================================
# Only the same measurement
# =============================================================================


def test_a_take_up_that_would_measure_otherwise_names_every_difference_and_no_label() -> None:
    stored = make_eval_run(
        judge_model="judge-a",
        judge_config_ids={"triage.tone": "cfg-1"},
        apparatus_settings={"reviewer_pool": "pool-a"},
        variant_levers={},
        identity_version=1,
        created_at="2026-06-01T00:00:00+00:00",
    )
    relabelled = stored.subject_snapshot.model_copy(update={"labels": {"app_version": "after-the-deploy"}})
    fresh = stored.model_copy(
        update={
            "judge_config_ids": {"triage.tone": "cfg-2"},
            "apparatus_settings": {"reviewer_pool": "pool-b"},
            "subject_snapshot": relabelled,
            "labels": ["baseline"],
        }
    )
    template = make_template(created_at="2026-01-01T00:00:00+00:00", updated_at="2026-01-01T00:00:00+00:00")

    differences = resume_drift(stored, fresh, template, toyhost_profile())

    assert [difference.split(" ", 1)[0] for difference in differences] == ["judge_config_ids", "apparatus_settings"]
    assert "cfg-2" in differences[0] and "pool-b" in differences[1]
    assert (
        resume_drift(stored, stored.model_copy(update={"subject_snapshot": relabelled}), template, toyhost_profile())
        == []
    )


async def test_a_run_whose_template_moved_is_refused_naming_every_difference_and_left_as_it_was(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    holder, run_id, _, _ = await _interrupted(monkeypatch, "cancelled")
    template = toyhost_template()
    edited = template.model_copy(
        update={
            "kind_spec": {"graded_fields": list(ExtractorSpec().graded_fields[:1])},
            "updated_at": "2099-01-01T00:00:00+00:00",
        }
    )
    off_loop(holder.eval_host.storage.save_template, edited)
    before, rows = off_loop(_run, holder, run_id), _rows(holder, run_id)

    with pytest.raises(ValidationFailedError) as refused:
        await resume_run(other_process(holder).launch, run_id, TOYHOST_SCOPE)

    assert "template" in str(refused.value) and "kind_spec" in str(refused.value), refused.value
    assert refused.value.remedy
    assert off_loop(_run, holder, run_id) == before, "nothing was written"
    assert _rows(holder, run_id) == rows


# =============================================================================
# Its budget
# =============================================================================


def _cheaply_priced(host: OpsHost) -> OpsHost:
    """``host`` with a pricer that predicts almost nothing, so a launch under a tiny cap is admitted and the cap
    then stops it."""

    def price(quote: ArmQuote) -> ArmPrice:
        return ArmPrice(predicted_usd=0.000001, basis="a test's under-estimate")

    return dataclasses.replace(host, launch=dataclasses.replace(host.launch, launch_pricer=price))


async def test_a_run_its_budget_stopped_is_taken_up_only_under_a_cap_above_its_spend() -> None:
    holder = _cheaply_priced(off_loop(ops_fixture, store=ClockedStore()).host)
    run_id = await launched_run(holder, K, max_cost_usd=0.000001)
    assert (await settled(holder, run_job_id(run_id))).status == "budget_stopped"
    stopped = off_loop(_run, holder, run_id)
    spent = sum(row.cost_usd or 0.0 for row in _rows(holder, run_id))
    resumer = other_process(holder)

    with pytest.raises(ValidationFailedError, match="max_cost_usd above"):
        await resume_run(resumer.launch, run_id, TOYHOST_SCOPE)
    with pytest.raises(ValidationFailedError, match="max_cost_usd above"):
        await resume_run(resumer.launch, run_id, TOYHOST_SCOPE, max_cost_usd=spent)
    assert off_loop(_run, holder, run_id) == stopped, "a refused take-up writes nothing"

    # Just above what it spent: what it spent counts toward the cap, so the first cell it lands stops it again —
    # counted from nothing, it would have run every cell left.
    barely = spent + 0.0000001
    await resume_run(resumer.launch, run_id, TOYHOST_SCOPE, max_cost_usd=barely)
    assert (await settled(resumer, run_job_id(run_id))).status == "budget_stopped"
    assert len(_rows(holder, run_id)) < 3 * K
    await resume_run(other_process(holder).launch, run_id, TOYHOST_SCOPE, max_cost_usd=2.0)
    assert (await settled(resumer, run_job_id(run_id))).state == "completed"

    run = off_loop(_run, holder, run_id)
    assert len(_rows(holder, run_id)) == 3 * K
    assert (run.max_cost_usd, run.budget_stops, run.budget_stop_reason) == (2.0, None, None)
    first, second, third = run.lifetimes
    assert (first.status, first.max_cost_usd) == ("budget_stopped", 0.000001)
    assert first.budget_stops and first.reason == stopped.budget_stop_reason
    assert (second.status, second.max_cost_usd, second.epoch) == ("budget_stopped", barely, 1)
    assert (third.status, third.max_cost_usd, third.epoch) == ("completed", 2.0, 2)


# =============================================================================
# Old records
# =============================================================================


def _as_an_older_build_stored_it(store: ClockedStore, run_id: str) -> dict[str, str]:
    """Rewrite a run and its rows as a build before epochs stored them: minted result ids, no epoch anywhere, and
    the run still reading running under a lease an old pod held. Returns each cell's old id by its derived one."""
    minted: dict[str, str] = {}
    for (scope, doc_id), document in list(store.documents.items()):
        if document.get("doc_type") != "eval_result" or document.get("eval_run_id") != run_id:
            continue
        old = str(uuid.uuid7())
        minted[doc_id] = old
        trace = store.get(eval_trace_doc_id(doc_id), scope)
        store.delete(doc_id, scope)
        document = {k: v for k, v in document.items() if k != "lease_epoch"} | {"id": old, "schema_minor": 6}
        store.upsert(document)
        if trace is not None:
            store.delete(eval_trace_doc_id(doc_id), scope)
            store.upsert(trace | {"id": eval_trace_doc_id(old), "result_id": old, "schema_minor": 6})
    run = store.get(run_id, TOYHOST_SCOPE)
    assert run is not None
    old_lease = {"holder": "an-old-pod", "expires_at": (T0 + timedelta(seconds=1)).isoformat()}
    stripped = {key: value for key, value in run.items() if key != "lifetimes"}
    store.upsert(
        stripped
        | {
            "status": "running",
            "cancellation_reason": None,
            "completed_at": None,
            "completeness": None,
            "lost": None,
            "lease": old_lease,
            "schema_minor": 6,
        }
    )
    return minted


async def test_a_run_an_older_build_stored_and_lost_is_taken_up_keeping_its_rows_and_their_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    holder, run_id, launched, store = await _interrupted(monkeypatch, "cancelled")
    minted = _as_an_older_build_stored_it(store, run_id)
    old_rows = {row.id: row for row in _rows(holder, run_id)}
    assert set(old_rows) == set(minted.values()) and not any(row.lease_epoch for row in old_rows.values())
    store.at = T0 + timedelta(seconds=61)
    resumer = other_process(holder)
    assert (await job_poll(resumer, run_job_id(run_id), TOYHOST_SCOPE)).state == "lost"

    await resume_run(resumer.launch, run_id, TOYHOST_SCOPE)
    assert (await settled(resumer, run_job_id(run_id))).state == "completed"

    rows = {row.id: row for row in _rows(holder, run_id)}
    assert len(rows) == 3 * K and len(_cells(list(rows.values()))) == len(set(_cells(list(rows.values())))), (
        "each cell has one row"
    )
    for old_id, old in old_rows.items():
        assert old_id in rows, "a row an older build stored keeps its id"
        if old.termination == "cancelled":
            assert (rows[old_id].termination, rows[old_id].lease_epoch) == ("completed", 1), "written over in place"
        else:
            assert rows[old_id] == old, "a done row is left as it was"
    run = off_loop(_run, holder, run_id)
    assert run.status == "completed" and run.context_key == launched.context_key
    assert [(lifetime.holder, lifetime.epoch, lifetime.status) for lifetime in run.lifetimes] == [
        ("an-old-pod", 0, "cancelled"),
        (_holder_id(resumer), 1, "completed"),
    ]


# =============================================================================
# One holding
# =============================================================================


async def test_of_two_take_ups_racing_exactly_one_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    holder, run_id, _, _ = await _interrupted(monkeypatch, "lost")
    first, second = other_process(holder), other_process(holder)

    outcomes: list[Any] = await asyncio.gather(
        resume_run(first.launch, run_id, TOYHOST_SCOPE),
        resume_run(second.launch, run_id, TOYHOST_SCOPE),
        return_exceptions=True,
    )

    won = [outcome for outcome in outcomes if isinstance(outcome, EvalRun)]
    lost = [outcome for outcome in outcomes if not isinstance(outcome, EvalRun)]
    assert len(won) == 1 and len(lost) == 1 and isinstance(lost[0], ConflictError), outcomes
    winner = first if isinstance(outcomes[0], EvalRun) else second
    assert (await settled(winner, run_job_id(run_id))).state == "completed"
    run = off_loop(_run, holder, run_id)
    assert [lifetime.holder for lifetime in run.lifetimes] == [_holder_id(holder), _holder_id(winner)]
    assert len(_rows(holder, run_id)) == 3 * K


# =============================================================================
# The spend a cut-short cell held
# =============================================================================


async def test_a_cut_short_cell_s_spend_counts_once_before_and_after_its_cell_is_run_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancelled, taken up and stopped by its cap part-way, then taken up again: at each take-up the cap counts every
    attempt the run paid for once — a cut-short row's own spend while it is still stored, and the same spend on the
    row written over it once its cell has run again."""
    holder, run_id, _, store = await _interrupted(monkeypatch, "cancelled")
    _cut_short_after_spending(store, run_id)
    first = _rows(holder, run_id)
    first_total = sum(row.cost_usd or 0.0 for row in first)
    cut_short = {row.id: row for row in first if row.termination == "cancelled"}
    assert sum(amount or 0.0 for amount in counted_spend(first)) == pytest.approx(first_total)
    resumer = _cheaply_priced(other_process(holder))

    # Just above what it spent, so the take-up stops once its first cell lands: some cells run again, the rest wait.
    await resume_run(resumer.launch, run_id, TOYHOST_SCOPE, max_cost_usd=first_total + 0.0000001)
    assert (await settled(resumer, run_job_id(run_id))).status == "budget_stopped"
    second = {row.id: row for row in _rows(holder, run_id)}
    run_again = [row_id for row_id in cut_short if second[row_id].lease_epoch == 1]
    assert run_again, "a cut-short cell was run again before the cap stopped the run"
    for row_id, row in cut_short.items():
        if row_id in run_again:
            assert second[row_id].superseded_spend_usd == pytest.approx(CUT_SHORT_USD)
        else:
            assert second[row_id] == row, "a cut-short cell not reached keeps its row and its spend"
    second_total = first_total + sum(row.cost_usd or 0.0 for row in second.values() if row.lease_epoch == 1)
    assert sum(amount or 0.0 for amount in counted_spend(list(second.values()))) == pytest.approx(second_total)
    with pytest.raises(ValidationFailedError, match=f"spent \\${second_total:.4f}"):
        await resume_run(resumer.launch, run_id, TOYHOST_SCOPE, max_cost_usd=second_total - 0.000000001)

    await resume_run(resumer.launch, run_id, TOYHOST_SCOPE, max_cost_usd=2.0)
    assert (await settled(resumer, run_job_id(run_id))).state == "completed"
    last = _rows(holder, run_id)
    assert len(last) == 3 * K
    assert sum(amount or 0.0 for amount in counted_spend(last)) == pytest.approx(
        second_total + sum(row.cost_usd or 0.0 for row in last if row.lease_epoch == 2)
    )


# =============================================================================
# What is not taken up
# =============================================================================


def _seeded(storage: Any, **fields: Any) -> EvalRun:
    run = make_eval_run(
        **{"scope_id": TOYHOST_SCOPE, "template_id": toyhost_template().id, "test_case_ids": ["gone-case"], **fields}
    )
    off_loop(storage.save_job_record, run.to_dict(), if_match=None, create_only=True)
    return run


@pytest.mark.parametrize(
    ("fields", "refusal"),
    [
        ({"status": "completed"}, "completed; there is nothing left"),
        ({"status": "failed", "failed_because": "nothing_measured"}, "failed"),
        (
            {
                "status": "running",
                "lease": JobLease(holder="another-process", expires_at=(T0 + timedelta(hours=1)).isoformat()),
            },
            "still running by process another-process",
        ),
        ({"status": "cancelled", "template_id": None}, "ad hoc"),
        ({"status": "cancelled"}, "gone-case, which scope"),
    ],
    ids=["completed", "failed", "live", "ad-hoc", "cases-gone"],
)
async def test_a_run_that_cannot_be_taken_up_is_refused_with_a_remedy_and_left_as_it_was(
    fields: dict[str, Any], refusal: str
) -> None:
    host = off_loop(ops_fixture, store=ClockedStore()).host
    run = _seeded(host.eval_host.storage, **fields)
    stored = off_loop(_run, host, run.id)

    with pytest.raises(ValidationFailedError, match=refusal) as refused:
        await resume_run(host.launch, run.id, TOYHOST_SCOPE)

    assert refused.value.remedy
    assert off_loop(_run, host, run.id) == stored


async def test_a_run_whose_job_is_still_unwinding_in_this_process_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    gate = held_extraction(monkeypatch)
    store = ClockedStore()
    holder = off_loop(ops_fixture, store=store).host
    run_id = await launched_run(holder, K)
    await until(lambda: gate.held >= 1)
    store.at = T0 + timedelta(seconds=61)
    assert (await job_poll(other_process(holder), run_job_id(run_id), TOYHOST_SCOPE)).state == "lost"
    lost = off_loop(_run, holder, run_id)

    with pytest.raises(ValidationFailedError, match="still unwinding"):
        await resume_run(holder.launch, run_id, TOYHOST_SCOPE)

    assert off_loop(_run, holder, run_id) == lost
    gate.set()
    await until(lambda: not holder.launch.job_manager.is_active(run_id))


# =============================================================================
# What a take-up records and runs under
# =============================================================================


async def test_a_take_up_across_a_deploy_goes_ahead_and_its_holding_records_the_new_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    holder, run_id, launched, _ = await _interrupted(monkeypatch, "cancelled")
    deployed = TOYHOST_SUBJECT.model_copy(update={"labels": {**TOYHOST_SUBJECT.labels, "app_version": "2.0.1"}})
    monkeypatch.setitem(TOYHOST_SUBJECTS, TOYHOST_SUBJECT.subject_id, deployed)
    resumer = other_process(holder)

    await resume_run(resumer.launch, run_id, TOYHOST_SCOPE)
    assert (await settled(resumer, run_job_id(run_id))).state == "completed"

    run = off_loop(_run, holder, run_id)
    first, second = run.lifetimes
    assert (first.labels, second.labels) == (dict(TOYHOST_SUBJECT.labels), dict(deployed.labels))
    assert run.subject_snapshot == launched.subject_snapshot, "the run keeps the subject it was launched on"
    assert run.context_key == launched.context_key


async def test_a_run_its_provider_account_stopped_is_taken_up_again(monkeypatch: pytest.MonkeyPatch) -> None:
    holder, run_id, _, store = await _interrupted(monkeypatch, "cancelled")
    refusal = "the provider account refused a candidate call: out of credit"
    document = store.get(run_id, TOYHOST_SCOPE)
    assert document is not None
    store.upsert(document | {"status": "exhausted", "cancellation_reason": None, "error_details": [refusal]})
    resumer = other_process(holder)

    await resume_run(resumer.launch, run_id, TOYHOST_SCOPE)
    assert (await settled(resumer, run_job_id(run_id))).state == "completed"

    first, second = off_loop(_run, holder, run_id).lifetimes
    assert (first.status, first.reason, second.status) == ("exhausted", refusal, "completed")
    assert len(_rows(holder, run_id)) == 3 * K


async def test_a_take_up_runs_under_the_cap_the_run_recorded_not_the_hosts_current_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    holder, run_id, launched, _ = await _interrupted(monkeypatch, "cancelled")
    assert (launched.max_cost_usd, launched.max_cost_usd_origin) == (TOYHOST_COST_CEILING_USD, "inherited")
    raised = TOYHOST_LAUNCH_SETTINGS.model_copy(update={"max_cost_usd": TOYHOST_COST_CEILING_USD + 2.0})
    resumer = other_process(holder)
    resumer = dataclasses.replace(resumer, launch=dataclasses.replace(resumer.launch, settings=lambda: raised))

    await resume_run(resumer.launch, run_id, TOYHOST_SCOPE)
    assert (await settled(resumer, run_job_id(run_id))).state == "completed"

    run = off_loop(_run, holder, run_id)
    assert (run.max_cost_usd, run.max_cost_usd_origin) == (TOYHOST_COST_CEILING_USD, "inherited")
    assert [lifetime.max_cost_usd for lifetime in run.lifetimes] == [TOYHOST_COST_CEILING_USD] * 2
