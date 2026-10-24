"""Resuming an interrupted run through the actions an agent and a host call: ``job_resume`` and ``jobs_resumable``.

Pinned:

- **``job_resume`` takes a lost run up again**, through the catalogue every transport calls, and the run's own job
  then reads it ``completed`` with every cell, having spanned two process lifetimes.
- **Every refusal reaches the caller with its remedy**: a run that would measure otherwise (every difference
  named), one its budget stopped at its cap when no higher cap is named, one that completed; a sweep that completed,
  failed or is live; a sweep or launch id naming nothing; and a job of a kind not resumed (a judge pass's), which
  names the action that starts it again.
- **``jobs_resumable`` lists exactly what ``job_resume`` would take up** — lost, operator-cancelled, budget-stopped
  and exhausted runs, a run whose lease lapsed unwritten included — with how each ended, its cells, its spend, its
  cap and its lifetimes; never a completed, failed or live run, an ad hoc one, nor an archived one, nor a sweep or
  launch whose every run is archived. A lost sweep and a lost launch are
  listed once each, their arms under them and not again as runs. Each matching row it cannot read is counted. It
  reads the same on every store backend.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from datetime import timedelta
from typing import Any

import pytest

from threetears.evals.actions import eval_catalogue, standard_tools
from threetears.evals.kernel import EvalStorage, EvalSweep, SweepArmArguments, SweepArmRecord, SweepLaunchArguments
from threetears.evals.ops import JobsStarted, OpsHost, job_poll, run_job_id
from threetears.evals.run import ArmPrice, ArmQuote, EvalLaunch, resumable_jobs
from threetears.evals.schema import DocumentStore, EvalRun, JobLease, JobLoss
from threetears.evals.storage import InMemoryDocumentStore, SqliteDocumentStore
from packages.evals.tests.factories import make_eval_result, make_eval_run
from packages.evals.tests.fixtures.toyhost.contract import ExtractorSpec
from packages.evals.tests.fixtures.toyhost.run import toyhost_template
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
from packages.evals.tests.ops_support import CALLER, TOYHOST_SCOPE, TOYHOST_SUBJECT, ops_fixture, settled

#: Repeats per case: three cases, nine cells.
K = 3

#: The cells the interrupted run stores before its holder dies.
DONE_BEFORE = 3


@pytest.fixture
def tool() -> Any:
    """The ``evals`` tool, as every transport mounts it."""
    (evals, _admin) = eval_catalogue().mount_all(standard_tools())
    return evals


async def _call(tool: Any, host: OpsHost, **arguments: Any) -> Any:
    return await tool.call(arguments, host=host, caller=CALLER)


def _run(host: OpsHost, run_id: str) -> EvalRun:
    run = off_loop(host.eval_host.storage.load_eval_run, run_id, TOYHOST_SCOPE)
    assert run is not None
    return run


async def _lost_run(monkeypatch: pytest.MonkeyPatch) -> tuple[OpsHost, str]:
    """A run of nine cells whose holder died with three stored and the rest in flight or not started, read lost."""
    gate = held_extraction(monkeypatch, passing=DONE_BEFORE)
    store = ClockedStore()
    holder = off_loop(ops_fixture, store=store).host
    launch_id, run_id = await launched_arm(holder, K)
    rows = holder.eval_host.storage.query_eval_results_by_run
    await until(lambda: len(off_loop(rows, run_id, TOYHOST_SCOPE)) == DONE_BEFORE and gate.held >= 1)
    store.at = T0 + timedelta(seconds=61)
    assert (await job_poll(other_process(holder), run_job_id(run_id), TOYHOST_SCOPE)).state == "lost"
    gate.set()
    manager = holder.launch.job_manager
    # The launch too: one still running when a later read finds its lease lapsed is written lost, which lists the run
    # under it rather than on its own.
    await until(lambda: not manager.is_active(run_id) and not manager.is_task_active(launch_id))
    return holder, run_id


# =============================================================================
# job_resume
# =============================================================================


async def test_job_resume_takes_a_lost_run_up_again_and_its_job_reads_completed(
    tool: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    holder, run_id = await _lost_run(monkeypatch)
    resumer = other_process(holder)

    outcome = await _call(tool, resumer, action="job_resume", job_id=run_job_id(run_id))

    assert not outcome.is_error, outcome.text
    (job,) = JobsStarted.model_validate(outcome.structured).jobs
    assert (job.job_id, job.kind, job.target_id) == (run_job_id(run_id), "run", run_id)
    assert (await settled(resumer, job.job_id)).state == "completed"
    run = off_loop(_run, holder, run_id)
    assert run.status == "completed" and len(run.lifetimes) == 2
    assert len(off_loop(holder.eval_host.storage.query_eval_results_by_run, run_id, TOYHOST_SCOPE)) == 3 * K


async def test_job_resume_refuses_a_run_that_would_measure_otherwise_naming_the_difference_and_how_to_proceed(
    tool: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    holder, run_id = await _lost_run(monkeypatch)
    edited = toyhost_template().model_copy(
        update={
            "kind_spec": {"graded_fields": list(ExtractorSpec().graded_fields[:1])},
            "updated_at": "2099-01-01T00:00:00+00:00",
        }
    )
    off_loop(holder.eval_host.storage.save_template, edited)
    before = off_loop(_run, holder, run_id)

    outcome = await _call(tool, other_process(holder), action="job_resume", job_id=run_job_id(run_id))

    assert outcome.is_error and outcome.text.startswith("refused: job_resume: ")
    refusal = outcome.text.splitlines()[0]
    assert "kind_spec (" in refusal, outcome.text
    assert "its measured content changed since the run started" in refusal, outcome.text
    assert "remedy:" in outcome.text and "run_launch" in outcome.text.splitlines()[-1]
    assert off_loop(_run, holder, run_id) == before, "a refused resume writes nothing"


async def test_job_resume_refuses_a_run_its_budget_stopped_until_a_cap_above_its_spend_is_named(tool: Any) -> None:
    def price(quote: ArmQuote) -> ArmPrice:
        return ArmPrice(predicted_usd=0.000001, basis="a test's under-estimate")

    host = off_loop(ops_fixture, store=ClockedStore()).host
    holder = dataclasses.replace(host, launch=dataclasses.replace(host.launch, launch_pricer=price))
    run_id = await launched_run(holder, K, max_cost_usd=0.000001)
    assert (await settled(holder, run_job_id(run_id))).status == "budget_stopped"

    refused = await _call(tool, holder, action="job_resume", job_id=run_job_id(run_id))

    assert refused.is_error and "stopped by its budget" in refused.text
    assert refused.text.splitlines()[-1].startswith("remedy: call job_resume with max_cost_usd above $")

    resumed = await _call(tool, holder, action="job_resume", job_id=run_job_id(run_id), max_cost_usd=2.0)
    assert not resumed.is_error, resumed.text
    assert (await settled(holder, run_job_id(run_id))).state == "completed"
    assert off_loop(_run, holder, run_id).max_cost_usd == 2.0


async def test_job_resume_refuses_a_completed_run_with_its_remedy(tool: Any) -> None:
    host = off_loop(ops_fixture, store=ClockedStore()).host
    run_id = await launched_run(host, 1)
    assert (await settled(host, run_job_id(run_id))).state == "completed"

    outcome = await _call(tool, host, action="job_resume", job_id=run_job_id(run_id))

    assert outcome.is_error and "completed; there is nothing left of it to run" in outcome.text
    assert outcome.text.splitlines()[-1] == "remedy: launch its template again (run_launch) to measure again"


@pytest.mark.parametrize(
    ("job_id", "refusal", "remedy"),
    [
        ("sweep:0193a1b2-sweep", "sweep '0193a1b2-sweep' not found", "a job's id is the one the call that started it"),
        ("launch:0193a1b2-launch", "launch '0193a1b2-launch' not found", "a job's id is the one the call"),
        (
            "second-judge:0193a1b2-pass",
            "only a run's, a sweep's or a launch's job is",
            "start the work again with the action that started it",
        ),
    ],
    ids=["sweep", "launch", "second-judge"],
)
async def test_job_resume_refuses_a_job_it_cannot_find_or_does_not_resume_naming_what_to_do_instead(
    tool: Any, job_id: str, refusal: str, remedy: str
) -> None:
    host = off_loop(ops_fixture, store=ClockedStore()).host

    outcome = await _call(tool, host, action="job_resume", job_id=job_id)

    assert outcome.is_error and refusal in outcome.text, outcome.text
    assert remedy in outcome.text.splitlines()[-1]


def _ended_sweep(host: OpsHost, outcome: str, **fields: Any) -> EvalSweep:
    sweep = EvalSweep(
        scope_id=TOYHOST_SCOPE,
        campaign_id="0193a1b2-campaign",
        template_id=toyhost_template().id,
        subject_id=TOYHOST_SUBJECT.subject_id,
        arms=[SweepArmRecord(label="only", model="extractor-v2")],
        max_concurrent_arms=1,
        outcome=outcome,  # type: ignore[arg-type]
        **fields,
    )
    off_loop(host.eval_host.storage.save_sweep, sweep)
    return sweep


@pytest.mark.parametrize(
    ("outcome", "fields", "refusal", "remedy"),
    [
        ("completed", {}, "completed; there is nothing left of it to run", "start it again (sweep_launch)"),
        (
            "running",
            {"lease": JobLease(holder="another-process", expires_at="2099-01-01T00:00:00+00:00")},
            "is still running; a live sweep is never taken up",
            "cancel it (job_cancel) or wait for it to end",
        ),
        ("failed", {"detail": "arm 'only' was refused"}, "failed (arm 'only' was refused)", "fix what failed it"),
    ],
    ids=["completed", "live", "failed"],
)
async def test_job_resume_refuses_a_sweep_that_completed_failed_or_is_live_with_its_remedy(
    tool: Any, outcome: str, fields: dict[str, Any], refusal: str, remedy: str
) -> None:
    host = off_loop(ops_fixture, store=ClockedStore()).host
    sweep = _ended_sweep(host, outcome, **fields)

    refused = await _call(tool, host, action="job_resume", job_id=f"sweep:{sweep.id}")

    assert refused.is_error and refusal in refused.text, refused.text
    assert remedy in refused.text.splitlines()[-1]


# =============================================================================
# jobs_resumable
# =============================================================================

#: The scope the listing's runs are seeded in.
SCOPE = "scope-resumable"

#: Far enough ahead that a lease reads live on any clock a test runs under.
LIVE_UNTIL = "2099-01-01T00:00:00+00:00"


@pytest.fixture(params=["memory", "sqlite"])
def store(request: pytest.FixtureRequest) -> Iterator[DocumentStore]:
    """Each backend the listing's store query must answer the same on."""
    if request.param == "memory":
        yield InMemoryDocumentStore()
        return
    with SqliteDocumentStore(":memory:") as sqlite:
        yield sqlite


def _seed(storage: EvalStorage, run_id: str, **fields: Any) -> EvalRun:
    run = make_eval_run(id=run_id, scope_id=SCOPE, test_case_ids=["tc-1", "tc-2"], k_runs=2, **fields)
    storage.save_eval_run(run)
    return run


def _row(storage: EvalStorage, run_id: str, test_case_id: str, k: int, **fields: Any) -> None:
    storage.save_eval_result(
        make_eval_result(
            scope_id=SCOPE, eval_run_id=run_id, test_case_id=test_case_id, model="sonnet", k_iteration=k, **fields
        )
    )


def test_jobs_resumable_lists_exactly_the_runs_job_resume_would_take_up(store: DocumentStore) -> None:
    storage = EvalStorage(store)
    live = JobLease(holder="another-process", expires_at=LIVE_UNTIL)
    lapsed = JobLease(holder="a-dead-pod", expires_at="2020-01-01T00:00:00+00:00")
    _seed(storage, "completed", status="completed")
    _seed(storage, "failed", status="failed", failed_because="nothing_measured")
    _seed(storage, "live", status="running", lease=live)
    _seed(storage, "ad-hoc", status="cancelled", template_id=None)
    _seed(
        storage,
        "lost",
        status="cancelled",
        lost=JobLoss(at="2026-10-01T00:00:00+00:00", cause="its lease lapsed"),
        created_at="2026-10-01T00:00:00+00:00",
    )
    _seed(storage, "lapsed", status="running", lease=lapsed, created_at="2026-10-02T00:00:00+00:00")
    _seed(
        storage,
        "operator",
        status="cancelled",
        cancellation_reason="shipping a fix",
        created_at="2026-10-03T00:00:00+00:00",
        max_cost_usd=1.0,
    )
    _row(storage, "operator", "tc-1", 1, cost_usd=0.25)
    _row(storage, "operator", "tc-1", 2, cost_usd=None)
    _row(storage, "operator", "tc-2", 1, termination="cancelled", cost_usd=0.05)
    _seed(
        storage,
        "budget",
        status="budget_stopped",
        budget_stop_reason="the cost cap was reached",
        created_at="2026-10-04T00:00:00+00:00",
        max_cost_usd=0.02,
    )
    _row(storage, "budget", "tc-1", 1, cost_usd=0.03)
    _seed(
        storage,
        "exhausted",
        status="exhausted",
        error_details=["the provider account refused a candidate call: out of credit"],
        created_at="2026-10-05T00:00:00+00:00",
    )

    listing = resumable_jobs(storage, [SCOPE])

    jobs = {job.run_id: job for job in listing.jobs}
    assert [job.run_id for job in listing.jobs] == ["exhausted", "budget", "operator", "lapsed", "lost"], "newest first"
    assert (listing.unreadable, listing.drift) == (0, "checked_on_resume")
    assert {run_id: (job.status, job.ended) for run_id, job in jobs.items()} == {
        "exhausted": ("exhausted", "exhausted"),
        "budget": ("budget_stopped", "budget_stopped"),
        "operator": ("cancelled", "cancelled"),
        "lapsed": ("cancelled", "lost"),
        "lost": ("cancelled", "lost"),
    }
    operator = jobs["operator"]
    assert operator.job_id == "run:operator" and operator.reason == "shipping a fix"
    assert (operator.cells_done, operator.cells_expected) == (2, 4), "a cut-short row is not done"
    assert (operator.spent_usd, operator.unpriced, operator.max_cost_usd) == (pytest.approx(0.30), 1, 1.0)
    assert (operator.lifetimes, operator.note) == (1, None)
    budget = jobs["budget"]
    assert budget.reason == "the cost cap was reached"
    assert budget.note is not None and "max_cost_usd above $0.0300" in budget.note
    assert jobs["lost"].reason == "its lease lapsed"
    assert jobs["exhausted"].reason == "the provider account refused a candidate call: out of credit"
    assert jobs["exhausted"].note is None


def test_jobs_resumable_counts_a_run_resumed_before_by_its_lifetimes_and_reads_each_scope_named(
    store: DocumentStore,
) -> None:
    storage = EvalStorage(store)
    resumed = make_eval_run(id="resumed", scope_id=SCOPE, status="cancelled", cancellation_reason="again")
    holdings = [
        {"holder": "p1", "epoch": 0, "started_at": "2026-10-01T00:00:00+00:00", "status": "cancelled"},
        {"holder": "p2", "epoch": 1, "started_at": "2026-10-02T00:00:00+00:00", "status": "cancelled"},
    ]
    storage.save_eval_run(EvalRun.model_validate(resumed.model_dump() | {"lifetimes": holdings}))
    storage.save_eval_run(make_eval_run(id="elsewhere", scope_id="another-scope", status="cancelled"))

    (job,) = resumable_jobs(storage, [SCOPE]).jobs
    assert (job.run_id, job.lifetimes) == ("resumed", 2)
    assert [job.run_id for job in resumable_jobs(storage, [SCOPE, "another-scope"]).jobs] == ["resumed", "elsewhere"]


async def test_the_jobs_resumable_action_lists_a_lost_run_for_its_callers_scope(
    tool: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    holder, run_id = await _lost_run(monkeypatch)

    outcome = await _call(tool, other_process(holder), action="jobs_resumable")

    assert not outcome.is_error, outcome.text
    (job,) = outcome.structured["jobs"]
    assert (job["job_id"], job["ended"], job["cells_done"], job["cells_expected"]) == (
        run_job_id(run_id),
        "lost",
        DONE_BEFORE,
        3 * K,
    )
    assert outcome.structured["drift"] == "checked_on_resume"


#: The loss every lost record in the listing's seeds carries.
LOST = JobLoss(at="2026-10-06T00:00:00+00:00", cause="its lease lapsed")


def _sweep_of(storage: EvalStorage, sweep_id: str, arms: list[SweepArmRecord], **fields: Any) -> EvalSweep:
    sweep = EvalSweep(
        id=sweep_id,
        scope_id=SCOPE,
        campaign_id="campaign",
        template_id="tpl-1",
        subject_id="subject",
        arms=arms,
        max_concurrent_arms=1,
        arm_schedule="sequential",
        **fields,
    )
    storage.save_sweep(sweep)
    return sweep


def test_jobs_resumable_lists_a_lost_sweep_and_a_lost_launch_once_with_their_arms_under_them(
    store: DocumentStore,
) -> None:
    storage = EvalStorage(store)
    lost = {"status": "cancelled", "lost": LOST}
    for run_id in ("arm-lost", "launch-lost", "old-arm", "alone"):
        _seed(storage, run_id, **lost)
    for run_id in ("arm-done", "spent-arm"):
        _seed(storage, run_id, status="completed")
    _seed(storage, "launch-done", status="completed", candidate_model="opus")
    _row(storage, "arm-lost", "tc-1", 1, cost_usd=0.1)
    recorded = {"arguments": SweepLaunchArguments(k_runs=2)}
    sweep = _sweep_of(
        storage,
        "sweep-lost",
        [
            SweepArmRecord(label="a", model="sonnet", run_id="arm-lost"),
            SweepArmRecord(label="b", model="sonnet", run_id="arm-done"),
            SweepArmRecord(label="c", model="sonnet", arguments=SweepArmArguments(overlays={"style": "c"})),
        ],
        outcome="cancelled",
        lost=LOST,
        detail=f"lost: {LOST.cause}",
        created_at="2026-10-08T00:00:00+00:00",
        **recorded,
    )
    old = _sweep_of(
        storage,
        "sweep-old",
        [SweepArmRecord(label="x", model="sonnet", run_id="old-arm"), SweepArmRecord(label="y", model="sonnet")],
        outcome="timed_out",
        detail="its budget ran out",
        created_at="2026-10-06T00:00:00+00:00",
    )
    _sweep_of(
        storage, "sweep-completed", [SweepArmRecord(label="z", model="sonnet", run_id="arm-done")], outcome="completed"
    )
    _sweep_of(
        storage,
        "sweep-spent",
        [SweepArmRecord(label="s", model="sonnet", run_id="spent-arm")],
        outcome="cancelled",
        lost=LOST,
        **recorded,
    )
    launch = EvalLaunch(
        id="launch-1",
        scope_id=SCOPE,
        template_id="tpl-1",
        subject_id="subject",
        models=["sonnet", "opus"],
        k_runs=2,
        n_variations=0,
        run_ids=["launch-lost", "launch-done"],
        outcome="cancelled",
        lost=LOST,
        created_at="2026-10-07T00:00:00+00:00",
    )
    storage.save_job_record(launch.to_dict(), if_match=None)

    listing = resumable_jobs(storage, [SCOPE])

    assert [job.run_id for job in listing.jobs] == ["alone"], "an arm's run is listed under its sweep or launch only"
    assert [group.job_id for group in listing.groups] == [
        f"sweep:{sweep.id}",
        "launch:launch-1",
        f"sweep:{old.id}",
    ], "newest first; a completed sweep, and a lost one with nothing left to take up, are not listed"
    lost_sweep, lost_launch, old_sweep = listing.groups
    assert (lost_sweep.kind, lost_sweep.ended, lost_sweep.reason, lost_sweep.note) == (
        "sweep",
        "lost",
        LOST.cause,
        None,
    )
    assert lost_sweep.arms == {"a": "cancelled", "b": "completed", "c": "not_launched"}
    (arm,) = lost_sweep.runs
    assert (arm.job_id, arm.cells_done, arm.spent_usd) == ("run:arm-lost", 1, pytest.approx(0.1))
    assert (lost_launch.kind, lost_launch.ended) == ("launch", "lost")
    assert lost_launch.arms == {"sonnet": "cancelled", "opus": "completed"}
    assert [run.run_id for run in lost_launch.runs] == ["launch-lost"]
    assert (old_sweep.ended, old_sweep.reason, old_sweep.arms) == (
        "timed_out",
        "its budget ran out",
        {"x": "cancelled", "y": "not_launched"},
    )
    assert old_sweep.note is not None and "'y'" in old_sweep.note and "run_launch" in old_sweep.note


def test_jobs_resumable_leaves_out_an_archived_run_and_a_sweep_or_launch_whose_every_run_is_archived(
    store: DocumentStore,
) -> None:
    """Archiving a run is how an operator declines it, so it is not offered again at every boot."""
    storage = EvalStorage(store)
    lost = {"status": "cancelled", "lost": LOST}
    _seed(storage, "kept", **lost)
    for run_id in ("declined", "declined-arm", "declined-launch-run"):
        _seed(storage, run_id, archived=True, **lost)
    _sweep_of(
        storage,
        "sweep-declined",
        [SweepArmRecord(label="a", model="sonnet", run_id="declined-arm"), SweepArmRecord(label="b", model="sonnet")],
        outcome="cancelled",
        lost=LOST,
        arguments=SweepLaunchArguments(k_runs=2),
    )
    launch = EvalLaunch(
        id="launch-declined",
        scope_id=SCOPE,
        template_id="tpl-1",
        subject_id="subject",
        models=["sonnet"],
        k_runs=2,
        n_variations=0,
        run_ids=["declined-launch-run"],
        outcome="cancelled",
        lost=LOST,
    )
    storage.save_job_record(launch.to_dict(), if_match=None)

    listing = resumable_jobs(storage, [SCOPE])

    assert [job.run_id for job in listing.jobs] == ["kept"]
    assert listing.groups == []


def test_jobs_resumable_counts_each_matching_row_it_cannot_read_and_lists_the_rest(store: DocumentStore) -> None:
    storage = EvalStorage(store)
    _seed(storage, "readable", status="cancelled")
    retired = {"retired_knob": 1}
    store.upsert(make_eval_run(id="unreadable-lost", scope_id=SCOPE, status="cancelled").to_dict() | retired)
    store.upsert(make_eval_run(id="unreadable-stopped", scope_id=SCOPE, status="budget_stopped").to_dict() | retired)
    store.upsert(make_eval_run(id="unreadable-completed", scope_id=SCOPE, status="completed").to_dict() | retired)
    sweep = EvalSweep(
        scope_id=SCOPE,
        campaign_id="campaign",
        template_id="tpl-1",
        subject_id="subject",
        arms=[SweepArmRecord(label="a", model="sonnet")],
        max_concurrent_arms=1,
        outcome="cancelled",
    )
    store.upsert(sweep.to_dict() | retired)

    listing = resumable_jobs(storage, [SCOPE])

    assert [job.run_id for job in listing.jobs] == ["readable"]
    assert listing.groups == []
    assert listing.unreadable == 3, "two runs and a sweep that would be offered; a completed run is not counted"


def test_jobs_resumable_reads_a_sweep_and_a_launch_still_stored_running_with_a_lapsed_lease_as_lost(
    store: DocumentStore,
) -> None:
    storage = EvalStorage(store)
    lapsed = JobLease(holder="a-dead-pod", expires_at="2020-01-01T00:00:00+00:00")
    live = JobLease(holder="another-process", expires_at=LIVE_UNTIL)
    to_launch = SweepArmRecord(label="a", model="sonnet", arguments=SweepArmArguments())
    recorded = {"arguments": SweepLaunchArguments(k_runs=2), "outcome": "running"}
    _sweep_of(storage, "sweep-lapsed", [to_launch], lease=lapsed, **recorded)
    _sweep_of(storage, "sweep-live", [to_launch], lease=live, **recorded)
    for name, lease in (("lapsed", lapsed), ("live", live)):
        _seed(storage, f"run-of-{name}", status="running", lease=lease)
        launch = EvalLaunch(
            id=f"launch-{name}",
            scope_id=SCOPE,
            template_id="tpl-1",
            subject_id="subject",
            models=["sonnet"],
            k_runs=2,
            n_variations=0,
            run_ids=[f"run-of-{name}"],
            lease=lease,
        )
        storage.save_job_record(launch.to_dict(), if_match=None)

    listing = resumable_jobs(storage, [SCOPE])

    assert sorted((group.job_id, group.ended) for group in listing.groups) == [
        ("launch:launch-lapsed", "lost"),
        ("sweep:sweep-lapsed", "lost"),
    ], "a lapsed record is read lost and listed; a live one is not"
    launch_group = next(group for group in listing.groups if group.kind == "launch")
    assert launch_group.arms == {"sonnet": "cancelled"} and [run.run_id for run in launch_group.runs] == [
        "run-of-lapsed"
    ], "the launch's run is settled lost with it"
    assert listing.jobs == []
    stored, _ = storage.load_job_record(EvalSweep, "sweep-lapsed", SCOPE)
    assert stored is not None and stored.lost is not None, "the loss was written, as any reader writes it"


@pytest.mark.parametrize(
    ("run_ids", "refusal"),
    [
        ([], "ended before it started any run"),
        (["finished"], "every run launch 'launch-ended' started has ended completed or failed"),
    ],
    ids=["no-run", "every-run-ended"],
)
async def test_job_resume_refuses_a_launch_with_nothing_left_to_take_up(
    tool: Any, run_ids: list[str], refusal: str
) -> None:
    host = off_loop(ops_fixture, store=ClockedStore()).host
    storage = host.eval_host.storage
    off_loop(storage.save_eval_run, make_eval_run(id="finished", scope_id=TOYHOST_SCOPE, status="completed"))
    launch = EvalLaunch(
        id="launch-ended",
        scope_id=TOYHOST_SCOPE,
        template_id=toyhost_template().id,
        subject_id=TOYHOST_SUBJECT.subject_id,
        models=["extractor-v2"],
        k_runs=1,
        n_variations=0,
        run_ids=run_ids,
        outcome="cancelled",
        detail="lost: its lease lapsed",
    )
    off_loop(storage.save_job_record, launch.to_dict(), if_match=None)

    outcome = await _call(tool, host, action="job_resume", job_id="launch:launch-ended")

    assert outcome.is_error and refusal in outcome.text, outcome.text
    assert outcome.text.splitlines()[-1].startswith("remedy: ")
