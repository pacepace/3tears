"""A run whose holding ended before its matrix did is taken up again, and ends as the same run.

Pinned:

- **The acceptance**: a run killed part-way, its lease lapsing with cells in flight, is taken up from a fresh process
  and ends exactly as an uninterrupted control run of it does: the same cells, one row each, the same pass^k,
  ``completed``, its context and variant keys unchanged — and it records one lifetime per process that held it. A
  run an operator cancelled, whose cut-short cells left ``cancelled`` rows, is taken up the same way, each of those
  rows written over; and so is one taken up by the very process that held it.
- **Only the same measurement**: a take-up that would measure otherwise is refused, naming every difference, and
  leaves the run as it was. A template edited after the run began is refused when its measured content moved (the
  digest the run froze at launch: a scoring guide, the intent, a goal check); an edit in nothing measured (its
  description) is taken up, and disclosed on the holding that began after it. A run stored before it froze that
  digest is refused once its template was edited, and taken up while it is not.
- **Its budget**: a run its budget stopped is taken up only under a cap above what it spent, and each holding
  records the cap it ran under. A run its metered-call ceiling stopped is taken up only under a ceiling above the
  calls its stored cells landed, which count toward it. A run its provider account stopped runs the cells the account
  refused again once taken up.
- **Across an identity version**: a run stamped under an older one, its inputs unchanged, is taken up; one whose host
  can no longer place its variant names that as a difference.
- **A group that never starts**: a run taken up whose group is settled before it starts ends its holding with it.
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
from threetears.evals.kernel import ConflictError, EvalStorage, ValidationFailedError
from threetears.evals.kernel.candidate_kind import CandidateOutput, CellSink
from threetears.evals.kernel.errors import StorageError
from threetears.evals.kernel.host import HostProfile
from threetears.evals.kernel.identity import IDENTITY_VERSION
from threetears.evals.kernel.scoring import compute_pass_hat_k
from threetears.evals.ops import OpsHost, job_poll, run_job_id
from threetears.evals.run import (
    ArmPrice,
    ArmQuote,
    KindWiring,
    LaunchHost,
    LaunchRequest,
    LaunchSettings,
    MeteredCallLedger,
    ResumableJob,
    RunResumption,
    StoredCells,
    launch_run,
    resumable_jobs,
    resume_run,
    start_run,
)
from threetears.evals.run.definition_digests import measured_template_digest
from threetears.evals.run.jobs import EvalJobManager
from threetears.evals.schema import EvalResult, EvalRun, EvalTemplate, JobLease
from threetears.evals.schema.models import RunLifetime, cell_result_id, eval_trace_doc_id
from threetears.evals.storage import InMemoryDocumentStore
from packages.evals.tests.factories import make_eval_run, make_template
from packages.evals.tests.fixtures.toyhost.corpus import TOYHOST_COST_CEILING_USD, TOYHOST_SUBJECT
from packages.evals.tests.fixtures.toyhost.host import with_registry
from packages.evals.tests.fixtures.toyhost.kind import TOY_EXTRACTOR_KIND, ScriptedExtractionClient, ToyExtractorKind
from packages.evals.tests.fixtures.toyhost.launch import (
    TOYHOST_LAUNCH_SETTINGS,
    TOYHOST_SUBJECTS,
    toyhost_launch_host,
)
from packages.evals.tests.fixtures.toyhost.profile import toyhost_profile
from packages.evals.tests.fixtures.toyhost.run import RUN_MODELS, toyhost_template, toyhost_test_cases
from packages.evals.tests.job_support import InMemoryRunStore
from packages.evals.tests.lease_support import (
    T0,
    ClockedStore,
    held_extraction,
    launched_arm,
    launched_run,
    other_process,
    until,
)
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


def _offered(host: OpsHost, run_id: str) -> ResumableJob:
    """The run as ``resumable_jobs`` offers it to a resume, on its own or among its launch's runs."""
    listed = off_loop(resumable_jobs, host.eval_host.storage, [TOYHOST_SCOPE])
    (offered,) = [
        job for job in [*listed.jobs, *(run for group in listed.groups for run in group.runs)] if job.run_id == run_id
    ]
    return offered


def _spend_of_every_attempt(rows: list[EvalResult]) -> float:
    """What a run's rows record it spent, each attempt once: a row's own spend and the spend of the attempts it was
    written over (``EvalResult.superseded_spend_usd``)."""
    return sum((row.cost_usd or 0.0) + (row.superseded_spend_usd or 0.0) for row in rows)


def _measured_otherwise(stored: EvalRun, fresh: EvalRun, template: EvalTemplate, profile: HostProfile) -> str | None:
    """Every difference a take-up of ``stored`` names when its launch assembles ``fresh`` from ``template``, as the
    launch hands the two to ``RunResumption.take_up``; ``None`` when the take-up goes ahead."""
    resuming = RunResumption(
        run=stored,
        cases=(),
        stored=StoredCells(),
        rows=(),
        holder=None,
        epoch=1,
        cell_order_seed=None,
        etag=None,
        max_cost_usd=None,
    )
    try:
        resuming.take_up(fresh, template, profile)
    except ValidationFailedError as refused:
        _, said, differences = refused.message.partition("it would measure otherwise than it did — ")
        assert said and differences, refused.message
        return differences
    return None


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
    launch_id, run_id = await launched_arm(holder, K)
    await until(lambda: len(_rows(holder, run_id)) == DONE_BEFORE and gate.held >= 1)
    launched = off_loop(_run, holder, run_id)
    manager = holder.launch.job_manager
    if how == "lost":
        store.at = T0 + timedelta(seconds=61)
        assert (await job_poll(other_process(holder), run_job_id(run_id), TOYHOST_SCOPE)).state == "lost"
        in_flight = gate.held
        assert in_flight >= 1, "cells were in flight when the lease lapsed"
        gate.set()
        await until(lambda: gate.released >= in_flight and not manager.is_active(run_id))
        assert len(_rows(holder, run_id)) == DONE_BEFORE, "the dead holder's cells in flight finished and wrote nothing"
    else:
        assert manager.cancel_job(run_id, reason="shipping a fix")
        await until(lambda: not manager.is_active(run_id))
        assert "cancelled" in {row.termination for row in _rows(holder, run_id)}, "the cancel cut cells short"
        gate.set()
    # Its launch has ended too, so nothing of the first holding is still writing when the test takes the run up.
    await until(lambda: not manager.is_task_active(launch_id))
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
    assert _spend_of_every_attempt(list(after.values())) == pytest.approx(
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

    said = _measured_otherwise(stored, fresh, template, toyhost_profile())

    assert said is not None
    differences = said.split("; ")
    # Read at the current identity version on both sides, the apparatus moves a context component as well as the field.
    assert [difference.split(" ", 1)[0] for difference in differences] == [
        "context",
        "judge_config_ids",
        "apparatus_settings",
    ]
    assert differences[0].startswith("context component apparatus_settings")
    assert "cfg-2" in differences[1] and "pool-b" in differences[2]
    unlabelled = stored.model_copy(update={"subject_snapshot": relabelled})
    assert _measured_otherwise(stored, unlabelled, template, toyhost_profile()) is None


def _edited(template: EvalTemplate, **fields: Any) -> EvalTemplate:
    return template.model_copy(update={**fields, "updated_at": "2099-01-01T00:00:00+00:00"})


def test_a_template_edit_moves_its_measured_digest_only_where_a_run_measures_it() -> None:
    template = make_template()
    (dim,) = template.rubric
    digest = measured_template_digest(template)

    measured = {
        "scoring_guide": _edited(template, rubric=[dim.model_copy(update={"scoring_guide": {"5": "Warm and plain."}})]),
        "rubric description": _edited(template, rubric=[dim.model_copy(update={"description": "Sounds curt"})]),
        "intent": _edited(template, intent="Probe the subject's patience with a slow shopper."),
        "conversation": _edited(template, conversation=None),
    }
    unmeasured = {
        "description": _edited(template, description="Reworded for the catalogue."),
        "name": _edited(template, name="category_bundle_renamed"),
        "archived": _edited(template, archived=True),
    }

    assert {what: measured_template_digest(edit) != digest for what, edit in measured.items()} == dict.fromkeys(
        measured, True
    )
    assert {what: measured_template_digest(edit) == digest for what, edit in unmeasured.items()} == dict.fromkeys(
        unmeasured, True
    )


def test_a_scoring_guide_edited_since_the_run_launched_refuses_its_take_up_naming_the_template() -> None:
    template = make_template(created_at="2026-01-01T00:00:00+00:00", updated_at="2026-01-01T00:00:00+00:00")
    (dim,) = template.rubric
    stored = make_eval_run(created_at="2026-06-01T00:00:00+00:00", template_digest=measured_template_digest(template))
    edited = _edited(template, rubric=[dim.model_copy(update={"scoring_guide": {"5": "Warm and plain."}})])
    fresh = stored.model_copy(update={"template_digest": measured_template_digest(edited)})

    difference = _measured_otherwise(stored, fresh, edited, toyhost_profile())

    assert difference is not None
    assert difference.startswith(f"template {template.id!r}: its measured content changed since the run started")
    assert difference.endswith("its name and description are not)"), (
        f"the template's is the one difference: {difference}"
    )
    reworded = _edited(template, description="Reworded for the catalogue.")
    assert _measured_otherwise(stored, stored.model_copy(), reworded, toyhost_profile()) is None, (
        "an unmeasured edit passes"
    )


#: When the tests' template edits land: after every run they interrupt was created.
EDITED_AT = "2099-01-01T00:00:00+00:00"


async def test_a_run_whose_template_was_edited_in_what_it_measures_is_refused_naming_it_and_left_as_it_was(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    holder, run_id, _, _ = await _interrupted(monkeypatch, "cancelled")
    edited = toyhost_template().model_copy(
        update={"goal_state_checks": [], "goal_check_controls": None, "updated_at": EDITED_AT}
    )
    off_loop(holder.eval_host.storage.save_template, edited)
    before, rows = off_loop(_run, holder, run_id), _rows(holder, run_id)

    with pytest.raises(ValidationFailedError) as refused:
        await resume_run(other_process(holder).launch, run_id, TOYHOST_SCOPE)

    assert "goal_check_proofs" in refused.value.message, refused.value.message
    assert "edited at" not in refused.value.message, "the edit is refused for what it changed, never for its time"
    assert refused.value.remedy
    assert off_loop(_run, holder, run_id) == before, "nothing was written"
    assert _rows(holder, run_id) == rows


async def test_a_run_whose_template_intent_was_edited_is_refused_as_measuring_otherwise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    holder, run_id, _, _ = await _interrupted(monkeypatch, "cancelled")
    edited = toyhost_template().model_copy(update={"intent": "Extract only the totals.", "updated_at": EDITED_AT})
    off_loop(holder.eval_host.storage.save_template, edited)
    before = off_loop(_run, holder, run_id)
    assert before.template_digest is not None

    with pytest.raises(ValidationFailedError) as refused:
        await resume_run(other_process(holder).launch, run_id, TOYHOST_SCOPE)

    assert "its measured content changed since the run started" in refused.value.message, refused.value.message
    assert off_loop(_run, holder, run_id) == before, "nothing was written"


def _as_stored_before_template_digests(store: ClockedStore, run_id: str) -> None:
    """Rewrite a run's record as a build before runs froze their template's measured content stored it."""
    document = store.get(run_id, TOYHOST_SCOPE)
    assert document is not None
    document.pop("template_digest", None)
    store.upsert(document)


async def test_a_run_stored_before_it_froze_its_template_is_refused_once_its_template_was_edited(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    holder, run_id, _, store = await _interrupted(monkeypatch, "cancelled")
    _as_stored_before_template_digests(store, run_id)
    edited = toyhost_template().model_copy(
        update={"description": "Reworded for the catalogue.", "updated_at": EDITED_AT}
    )
    off_loop(holder.eval_host.storage.save_template, edited)
    before = off_loop(_run, holder, run_id)

    with pytest.raises(ValidationFailedError) as refused:
        await resume_run(other_process(holder).launch, run_id, TOYHOST_SCOPE)

    message = refused.value.message
    assert f"edited at {EDITED_AT}" in message and "whether the edit changed the measurement cannot be told" in message
    assert "taken up only while its template is unedited since the run was created" in refused.value.remedy
    assert off_loop(_run, holder, run_id) == before, "nothing was written"


async def test_a_run_stored_before_it_froze_its_template_is_taken_up_while_its_template_is_unedited(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    holder, run_id, launched, store = await _interrupted(monkeypatch, "cancelled")
    _as_stored_before_template_digests(store, run_id)
    resumer = other_process(holder)

    await resume_run(resumer.launch, run_id, TOYHOST_SCOPE)
    assert (await settled(resumer, run_job_id(run_id))).state == "completed"

    run = await _assert_ends_as_the_control(holder, run_id, launched)
    assert run.template_digest == launched.template_digest, "frozen from the take-up on"
    assert [holding.template_edited_at for holding in run.lifetimes] == [None, None]


async def test_a_run_whose_template_was_edited_in_nothing_it_measures_is_taken_up_and_the_edit_disclosed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    holder, run_id, launched, _ = await _interrupted(monkeypatch, "cancelled")
    edited = toyhost_template().model_copy(
        update={"description": "Reworded for the catalogue.", "updated_at": EDITED_AT}
    )
    off_loop(holder.eval_host.storage.save_template, edited)
    resumer = other_process(holder)

    await resume_run(resumer.launch, run_id, TOYHOST_SCOPE)
    assert (await settled(resumer, run_job_id(run_id))).state == "completed"

    run = await _assert_ends_as_the_control(holder, run_id, launched)
    first, second = run.lifetimes
    assert (first.template_edited_at, second.template_edited_at) == (None, EDITED_AT), "the holding after it says so"


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
    assert _offered(holder, run_id).spent_usd == pytest.approx(first_total)
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
    assert _offered(holder, run_id).spent_usd == pytest.approx(second_total)
    with pytest.raises(ValidationFailedError, match=f"spent \\${second_total:.4f}"):
        await resume_run(resumer.launch, run_id, TOYHOST_SCOPE, max_cost_usd=second_total - 0.000000001)

    await resume_run(resumer.launch, run_id, TOYHOST_SCOPE, max_cost_usd=2.0)
    assert (await settled(resumer, run_job_id(run_id))).state == "completed"
    last = _rows(holder, run_id)
    assert len(last) == 3 * K
    assert _spend_of_every_attempt(last) == pytest.approx(
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


#: What the provider says of a call the account paying for the run cannot pay for.
OUT_OF_CREDIT = "the provider account refused the extractor's call: out of credit"


def _account_refused_after(monkeypatch: pytest.MonkeyPatch, served: int) -> dict[str, bool]:
    """Serve the first ``served`` extractions, then refuse every later one for the account, until the returned flag's
    ``funded`` is set: the account topped up."""
    invoke = ToyExtractorKind.invoke
    state = {"funded": False}
    left = [served]

    async def refusing(self: ToyExtractorKind, instance: Any, test_case: Any, sink: Any) -> CandidateOutput:
        if state["funded"] or left[0] > 0:
            left[0] -= 1
            return await invoke(self, instance, test_case, sink)
        return CandidateOutput(output=[], infra_errors=[OUT_OF_CREDIT], account_refused=True)

    monkeypatch.setattr(ToyExtractorKind, "invoke", refusing)
    return state


async def test_a_run_its_provider_account_stopped_runs_its_refused_cells_again_once_taken_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    account = _account_refused_after(monkeypatch, served=DONE_BEFORE)
    holder = off_loop(ops_fixture, store=ClockedStore()).host
    run_id = await launched_run(holder, K)
    assert (await settled(holder, run_job_id(run_id))).status == "exhausted"
    launched = off_loop(_run, holder, run_id)
    refused = [row for row in _rows(holder, run_id) if row.account_refused]
    assert refused, "the account's refusal is on the cells it refused"
    assert all(row.infra_error and OUT_OF_CREDIT in row.infra_error for row in refused)
    resumer = other_process(holder)

    account["funded"] = True
    await resume_run(resumer.launch, run_id, TOYHOST_SCOPE)
    assert (await settled(resumer, run_job_id(run_id))).state == "completed"

    await _assert_ends_as_the_control(holder, run_id, launched)
    after = {row.id: row for row in _rows(holder, run_id)}
    for row in refused:
        assert (after[row.id].account_refused, after[row.id].lease_epoch) == (False, 1), "run again, written over"
    first, second = off_loop(_run, holder, run_id).lifetimes
    assert (first.status, second.status) == ("exhausted", "completed")
    assert first.reason is not None and OUT_OF_CREDIT in first.reason


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


# =============================================================================
# Across an identity version, and a run its host can no longer place
# =============================================================================


async def test_a_run_stamped_under_an_older_identity_version_with_its_inputs_unchanged_is_taken_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bump of the identity version moves no run's measurement: the stored run is read at today's version."""
    holder, run_id, launched, store = await _interrupted(monkeypatch, "cancelled")
    document = store.get(run_id, TOYHOST_SCOPE)
    assert document is not None
    # As an older build stamped it: another version, whose predicate digested the same inputs to another key.
    store.upsert(document | {"identity_version": IDENTITY_VERSION - 1, "context_key": "an-older-predicates-key"})

    await resume_run(other_process(holder).launch, run_id, TOYHOST_SCOPE)
    assert (await settled(holder, run_job_id(run_id))).state == "completed"

    assert off_loop(_run, holder, run_id).variant_levers == launched.variant_levers


def test_a_run_that_recorded_no_lever_map_its_host_can_no_longer_place_is_a_difference_named() -> None:
    template = make_template(created_at="2026-01-01T00:00:00+00:00", updated_at="2026-01-01T00:00:00+00:00")
    stored = make_eval_run(variant_levers=None, created_at="2026-06-01T00:00:00+00:00")
    fresh = stored.model_copy(update={"variant_levers": {}, "identity_version": IDENTITY_VERSION})
    unplaceable = dataclasses.replace(toyhost_profile(), variant_levers=lambda run: {"an_undeclared_axis": "x"})

    differences = _measured_otherwise(stored, fresh, template, unplaceable)

    assert differences is not None
    assert [d for d in differences.split("; ") if d.startswith("variant levers (the run recorded none")], differences


# =============================================================================
# A take-up whose group never starts
# =============================================================================


class _SiblingSaveFails(InMemoryRunStore):
    """A store that refuses to save one run as its group starts, so the runs saved before it are settled."""

    def __init__(self, refused: str) -> None:
        super().__init__()
        self.refused = refused

    def save_eval_run(self, run: EvalRun, *, if_match: str | None = None) -> None:
        if run.id == self.refused:
            raise StorageError(f"run {run.id} could not be saved")
        super().save_eval_run(run, if_match=if_match)


async def test_a_run_taken_up_whose_group_never_starts_ends_its_holding_with_it() -> None:
    """Settled cancelled before it started, the run's current holding reads how it ended, as the run does."""
    earlier = RunLifetime(holder="p1", epoch=0, started_at="2026-10-01T00:00:00+00:00", status="cancelled")
    holding = RunLifetime(holder="p2", epoch=1, started_at="2026-10-02T00:00:00+00:00")
    taken_up = make_eval_run(status="running", lifetimes=[earlier, holding])
    sibling = make_eval_run()
    storage = _SiblingSaveFails(sibling.id)
    manager = EvalJobManager(storage)

    async def work(progress: Any) -> None:
        raise AssertionError("no run of the group starts")

    with pytest.raises(StorageError):
        await manager.start_group([(taken_up, work, None), (sibling, work, None)])

    run = storage.runs[taken_up.id]
    assert run.status == "cancelled" and run.cancellation_reason is not None
    assert run.lifetimes[0] == earlier
    last = run.lifetimes[-1]
    assert (last.status, last.reason, last.epoch) == ("cancelled", run.cancellation_reason, 1)
    assert last.ended_at is not None


def test_a_run_ended_by_a_writer_that_never_closes_its_holding_reads_its_holding_ended_as_the_run_did() -> None:
    """The run's own fields are the one record of how its current holding ended: a run read with a terminal status
    and an open last lifetime reads that lifetime closed from them, and setting the fields in any order lands the same."""
    earlier = RunLifetime(holder="p1", epoch=0, started_at="2026-10-01T00:00:00+00:00", status="cancelled")
    holding = RunLifetime(holder="p2", epoch=1, started_at="2026-10-02T00:00:00+00:00")
    stored = make_eval_run(status="running", lifetimes=[earlier, holding]).to_dict() | {
        "status": "budget_stopped",
        "budget_stop_reason": "the cost cap was reached",
        "completed_at": "2026-10-03T00:00:00+00:00",
    }

    run = EvalRun.from_dict(stored)

    assert run.lifetimes[0] == earlier
    last = run.lifetimes[-1]
    assert (last.status, last.ended_at, last.reason) == (
        "budget_stopped",
        "2026-10-03T00:00:00+00:00",
        "the cost cap was reached",
    )
    set_in_turn = make_eval_run(status="running", lifetimes=[earlier, holding])
    set_in_turn.status = "cancelled"
    set_in_turn.cancellation_reason = "shipping a fix"
    set_in_turn.completed_at = "2026-10-03T00:00:00+00:00"
    assert (set_in_turn.lifetimes[-1].status, set_in_turn.lifetimes[-1].reason) == ("cancelled", "shipping a fix")
    assert set_in_turn.lifetimes[-1].ended_at == "2026-10-03T00:00:00+00:00"
    reopened = make_eval_run(status="running", lifetimes=[earlier, holding])
    assert reopened.lifetimes[-1] == holding, "a running run's holding is open"


# =============================================================================
# The metered-call ceiling
# =============================================================================


class _OneMeteredCallKind(ToyExtractorKind):
    """A toy extractor whose every cell makes one metered call through the run's ledger, its cells overlapping.

    A cell past the first ``hold.passing`` waits on ``hold.gate`` after its call when a test set one, so a cancel cuts
    it short with its call made.
    """

    def __init__(self, *args: Any, ledger: MeteredCallLedger | None, hold: _MeteredHold, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.ledger = ledger
        self.hold = hold

    async def invoke(self, instance: Any, test_case: Any, sink: CellSink) -> CandidateOutput:
        await asyncio.sleep(0.01)
        assert self.ledger is not None and self.ledger.admit(tool="web_search", action="search", spend=None)
        if self.hold.gate is not None:
            if self.hold.passing > 0:
                self.hold.passing -= 1
            else:
                self.hold.held += 1
                await self.hold.gate.wait()
        sink.add_spend("candidate", 0.0)
        return CandidateOutput(output=[{"admitted": 1}])


@dataclasses.dataclass
class _MeteredHold:
    """Which of the metered kind's cells wait after their call, and on what: none, while ``gate`` is None."""

    gate: asyncio.Event | None = None
    passing: int = 0
    #: How many cells have waited on the gate, their call made.
    held: int = 0


#: Cells of the metered run: three cases, four repeats.
METERED_CELLS = 12


@dataclasses.dataclass
class _MeteredHost:
    """The toy launch host with every cell making one metered call, under a metered-call ceiling a test moves."""

    launching: LaunchHost
    storage: EvalStorage
    store: InMemoryDocumentStore
    ceiling: list[int]
    hold: _MeteredHold

    async def ended(self, run_id: str) -> tuple[EvalRun, list[EvalResult]]:
        """The run once its job has ended, and its rows."""
        await until(lambda: not self.launching.job_manager.is_active(run_id), timeout=20)
        run = off_loop(self.storage.load_eval_run, run_id, TOYHOST_SCOPE)
        assert run is not None
        return run, off_loop(self.storage.query_eval_results_by_run, run_id, TOYHOST_SCOPE)

    async def launch(self) -> EvalRun:
        """Launch one arm of the metered run, four repeats over the toy cases."""
        (launched,) = await start_run(
            self.launching,
            template_id=toyhost_template().id,
            scope_id=TOYHOST_SCOPE,
            subject_id=TOYHOST_SUBJECT.subject_id,
            models=[RUN_MODELS[0]],
            k_runs=4,
        )
        return launched


def _metered_host(ceiling: int) -> _MeteredHost:
    store = InMemoryDocumentStore()
    storage = EvalStorage(store)
    ceilings = [ceiling]
    hold = _MeteredHold()

    def settings() -> LaunchSettings:
        return TOYHOST_LAUNCH_SETTINGS.model_copy(update={"max_metered_calls": ceilings[0]})

    host, _client = toyhost_launch_host(storage=storage, settings=settings)
    world = host.eval_host.profile.world
    assert world is not None

    async def launch(request: LaunchRequest) -> EvalRun:
        def factory(context: Any) -> ToyExtractorKind:
            return _OneMeteredCallKind(
                client=ScriptedExtractionClient(),
                world=world,
                goal_checks=(),
                ledger=context.options.metered_calls,
                hold=hold,
            )

        return await launch_run(
            launching,
            request,
            KindWiring(
                kind_factory=factory,
                subject=TOYHOST_SUBJECT,
                test_cases=toyhost_test_cases(request.template),
                max_concurrent_cells=2,
            ),
        )

    (launchable,) = host.kinds.values()
    launching = dataclasses.replace(
        host,
        eval_host=with_registry(host.eval_host, {TOY_EXTRACTOR_KIND: dataclasses.replace(launchable, launch=launch)}),
    )
    template = toyhost_template().model_copy(update={"goal_state_checks": []})
    storage.save_template(template)
    # A take-up reads the run's frozen cases from its scope.
    for case in toyhost_test_cases(template):
        storage.save_test_case(case)
    return _MeteredHost(launching=launching, storage=storage, store=store, ceiling=ceilings, hold=hold)


async def test_a_run_its_metered_ceiling_stopped_is_taken_up_only_under_a_ceiling_above_what_it_landed() -> None:
    metered = off_loop(_metered_host, 3)
    launching, storage, ceiling, ended = metered.launching, metered.storage, metered.ceiling, metered.ended

    launched = await metered.launch()
    stopped, rows = await ended(launched.id)
    assert stopped.status == "budget_stopped" and len(rows) < METERED_CELLS
    landed = sum(row.metered_calls or 0 for row in rows)
    assert landed == len(rows), "each row records its cell's one metered call"

    with pytest.raises(ValidationFailedError, match="metered-call ceiling of 3") as refused:
        await resume_run(launching, launched.id, TOYHOST_SCOPE)
    assert "max_metered_calls" in refused.value.remedy
    assert off_loop(storage.load_eval_run, launched.id, TOYHOST_SCOPE) == stopped, "a refused take-up writes nothing"

    # Just above what it landed: its landed calls count, so the first cells it lands stop it again — counted from
    # nothing, it would have run every cell left.
    ceiling[0] = landed + 1
    await resume_run(launching, launched.id, TOYHOST_SCOPE)
    again, rows = await ended(launched.id)
    assert again.status == "budget_stopped" and len(rows) < METERED_CELLS
    assert again.max_metered_calls == landed + 1, "the run records the ceiling it ran under"
    assert again.budget_stops is not None
    (stop,) = again.budget_stops
    assert (stop.currency, stop.landed_at_stop >= landed + 1) == ("metered_calls", True)

    ceiling[0] = 100
    await resume_run(launching, launched.id, TOYHOST_SCOPE)
    done, rows = await ended(launched.id)
    assert done.status == "completed" and len(rows) == METERED_CELLS
    assert sum(row.metered_calls or 0 for row in rows) == METERED_CELLS, "every cell's call, counted once"
    assert done.metered_calls_refused == 0


async def test_a_cut_short_cell_s_metered_calls_and_a_stored_cell_s_refused_calls_carry_into_its_take_up() -> None:
    """A cell cut short after its metered call, run again on the take-up: its row counts both attempts' calls. A
    stored cell's refused calls count toward the run's refusals once it ends taken up, beside the holding's own."""
    metered = off_loop(_metered_host, 100)
    metered.hold.gate, metered.hold.passing = asyncio.Event(), 3
    launched = await metered.launch()
    await until(lambda: len(off_loop(metered.storage.query_eval_results_by_run, launched.id, TOYHOST_SCOPE)) >= 3)
    # Both cells in flight (the run's width) have made their call and wait.
    await until(lambda: metered.hold.held == 2)
    assert metered.launching.job_manager.cancel_job(launched.id, reason="shipping a fix")
    cancelled, rows = await metered.ended(launched.id)
    assert cancelled.status == "cancelled"
    cut_short = {row.id for row in rows if row.termination == "cancelled"}
    assert cut_short, "the cancel cut a cell short"
    assert all(row.metered_calls == 1 for row in rows), "each row records its attempt's one call"
    # A stored cell some of whose calls dispatch refused: a host declaring no metered tools, where every call is.
    done = next(row for row in rows if row.termination != "cancelled")
    document = metered.store.get(done.id, TOYHOST_SCOPE)
    assert document is not None
    metered.store.upsert(document | {"metered_calls_refused": 3})

    metered.hold.gate.set()
    await resume_run(metered.launching, launched.id, TOYHOST_SCOPE)
    run, rows = await metered.ended(launched.id)

    assert run.status == "completed" and len(rows) == METERED_CELLS
    by_id = {row.id: row for row in rows}
    assert {by_id[row_id].metered_calls for row_id in cut_short} == {2}, "the cut-short attempt's call, and its own"
    assert sum(row.metered_calls or 0 for row in rows) == METERED_CELLS + len(cut_short)
    assert run.metered_calls_refused == 3, "the stored cell's refused calls, and none of this holding's"
