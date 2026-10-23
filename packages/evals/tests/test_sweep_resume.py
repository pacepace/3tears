"""A sweep or a launch whose holding ended part-way is taken up again as a whole, its arm schedule kept.

Pinned:

- **The acceptance**: an interleaved two-arm sweep lost with both arms' cells in flight is taken up from a fresh
  process with ``job_resume sweep:<id>``, and each arm ends exactly as an uninterrupted control sweep's does — the
  same cells, one row each, the same pass^k, ``completed``. Its arms ran as one group under the sweep's original
  launch group, so each arm's cells kept the order they had; the sweep ends ``completed`` and records two lifetimes.
- **A sequential sweep keeps its order**: an arm that completed is left alone, the arm that was running is taken up,
  and the arm that never launched is launched from what the sweep recorded, after it.
- **However it ended**: a sweep an operator cancelled, and one its own budget stopped, are taken up the same way;
  and a sweep taken up again that is then lost itself closes its open holding with the loss.
- **Old records**: a sweep stored before sweeps recorded their arms' arguments is taken up when every arm launched,
  and refused, naming the arm, when one never did.
- **One arm's drift refuses the whole sweep** before anything is spent, naming it, and leaves every record as it was.
- **A launch**: ``job_resume launch:<id>`` takes a lost interleaved two-model launch's runs up together, as one group
  under the launch's id, in their original order.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import timedelta
from typing import Any

import pytest

from threetears.evals.kernel import EvalSweep, SweepArmRecord, ValidationFailedError
from threetears.evals.kernel.scoring import compute_pass_hat_k
from threetears.evals.ops import LaunchArguments, OpsHost, job_cancel, job_poll, job_resume, run_launch, sweep_launch
from threetears.evals.run import EvalLaunch, runner
from threetears.evals.schema import EvalResult, EvalRun
from threetears.evals.schema.models import ArmSchedule
from packages.evals.tests.fixtures.toyhost.contract import ExtractorSpec
from packages.evals.tests.fixtures.toyhost.run import RUN_MODELS, toyhost_template
from packages.evals.tests.lease_support import (
    T0,
    ClockedStore,
    HeldGate,
    held_extraction,
    other_process,
    style_sweep,
    until,
)
from packages.evals.tests.off_loop import off_loop
from packages.evals.tests.ops_support import TOYHOST_SCOPE, TOYHOST_SUBJECT, ops_fixture, settled

#: Repeats per case: three cases, nine cells an arm.
K = 3

#: The interleaved sweep's arms, by their prompt style.
TWO_STYLES = ("terse", "verbose")

#: The sequential sweep's arms, in launch order.
THREE_STYLES = ("terse", "standard", "verbose")

#: The cost cap the sequential sweep is taken up under, raised from none.
RAISED_CAP = 2.0

#: Each cell's order as the runner computed it for a run, with the seed it was ordered by.
Orders = list[tuple[str, str | None, list[tuple[int, str]]]]


def _sweep(host: OpsHost, sweep_id: str) -> EvalSweep:
    sweep, _ = off_loop(host.eval_host.storage.load_job_record, EvalSweep, sweep_id, TOYHOST_SCOPE)
    assert sweep is not None
    return sweep


def _run(host: OpsHost, run_id: str) -> EvalRun:
    run, _ = off_loop(host.eval_host.storage.load_job_record, EvalRun, run_id, TOYHOST_SCOPE)
    assert run is not None
    return run


def _rows(host: OpsHost, run_id: str) -> list[EvalResult]:
    return off_loop(host.eval_host.storage.query_eval_results_by_run, run_id, TOYHOST_SCOPE)


def _cells(rows: list[EvalResult]) -> list[tuple[str, int]]:
    return sorted((row.test_case_id, row.k_iteration) for row in rows)


def _arm_runs(host: OpsHost, sweep_id: str) -> dict[str, str | None]:
    return {arm.label: arm.run_id for arm in _sweep(host, sweep_id).arms}


def _holder_id(host: OpsHost) -> str:
    leases = host.launch.job_manager.leases
    assert leases is not None
    return leases.holder_id


def _spied_orders(monkeypatch: pytest.MonkeyPatch) -> Orders:
    """Record every order the runner computes for a run's cells, with the seed it ordered them by."""
    orders: Orders = []
    order_cells = runner.cell_execution_order

    def spy(run: EvalRun, test_cases: list[Any], **kwargs: Any) -> Any:
        order = order_cells(run, test_cases, **kwargs)
        orders.append((run.id, kwargs.get("seed"), [(k, case.id) for k, case in order]))
        return order

    monkeypatch.setattr(runner, "cell_execution_order", spy)
    return orders


def _spied_groups(monkeypatch: pytest.MonkeyPatch, host: OpsHost) -> list[int]:
    """Record how many runs each group the host's job manager starts holds."""
    manager = host.launch.job_manager
    sizes: list[int] = []
    start_group = manager.start_group

    async def spy(members: Any, **kwargs: Any) -> Any:
        sizes.append(len(members))
        return await start_group(members, **kwargs)

    monkeypatch.setattr(manager, "start_group", spy)
    return sizes


async def _lost_sweep(
    monkeypatch: pytest.MonkeyPatch,
    styles: tuple[str, ...],
    *,
    arm_schedule: ArmSchedule,
    k_runs: int,
    passing: int,
    ready: Callable[[OpsHost, str, HeldGate], bool],
) -> tuple[OpsHost, ClockedStore, str]:
    """A sweep whose holder dies once ``ready`` holds, read lost from another process, its dead holder's cells in flight
    then let through to find themselves fenced.

    :return: The holder, the store, and the sweep's id.
    """
    gate = held_extraction(monkeypatch, passing=passing)
    store = ClockedStore()
    holder = off_loop(ops_fixture, store=store).host
    arguments = style_sweep(styles, arm_schedule=arm_schedule, k_runs=k_runs)
    (job,) = (await sweep_launch(holder, arguments, TOYHOST_SCOPE, created_by="agent:test")).jobs
    sweep_id = job.job_id.removeprefix("sweep:")
    await until(lambda: ready(holder, sweep_id, gate))
    store.at = T0 + timedelta(seconds=61)
    assert (await job_poll(other_process(holder), job.job_id, TOYHOST_SCOPE)).state == "lost"
    gate.set()
    manager = holder.launch.job_manager
    await until(
        lambda: (
            not manager.is_task_active(sweep_id)
            and not any(manager.is_active(run_id) for run_id in _arm_runs(holder, sweep_id).values() if run_id)
        )
    )
    return holder, store, sweep_id


def _interleaved_ready(stored: int) -> Callable[[OpsHost, str, HeldGate], bool]:
    """Both arms launched, ``stored`` cells stored between them, and cells of the two held in flight."""

    def ready(host: OpsHost, sweep_id: str, gate: HeldGate) -> bool:
        runs = [run_id for run_id in _arm_runs(host, sweep_id).values() if run_id is not None]
        return len(runs) == 2 and sum(len(_rows(host, run_id)) for run_id in runs) == stored and gate.held >= 2

    return ready


async def _lost_interleaved_sweep(
    monkeypatch: pytest.MonkeyPatch, *, k_runs: int, stored: int
) -> tuple[OpsHost, ClockedStore, str]:
    return await _lost_sweep(
        monkeypatch,
        TWO_STYLES,
        arm_schedule="interleaved",
        k_runs=k_runs,
        passing=stored,
        ready=_interleaved_ready(stored),
    )


# =============================================================================
# The acceptance
# =============================================================================


async def test_a_lost_interleaved_sweep_taken_up_again_ends_as_an_uninterrupted_sweep_does_in_its_original_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orders = _spied_orders(monkeypatch)
    holder, _store, sweep_id = await _lost_interleaved_sweep(monkeypatch, k_runs=K, stored=4)
    lost = _sweep(holder, sweep_id)
    group_id = lost.launch_group_id
    arm_runs = {arm.label: arm.run_id for arm in lost.arms}
    assert group_id is not None and None not in arm_runs.values()
    done = {label: _cells(_rows(holder, run_id)) for label, run_id in arm_runs.items() if run_id is not None}
    assert sum(len(cells) for cells in done.values()) == 4, "the dead holder's cells in flight wrote nothing"
    assert all(len(cells) < 3 * K for cells in done.values()), "neither arm finished before its sweep was lost"
    resumer = other_process(holder)
    groups = _spied_groups(monkeypatch, resumer)

    (job,) = (await job_resume(resumer, f"sweep:{sweep_id}", TOYHOST_SCOPE)).jobs

    assert (job.job_id, job.kind) == (f"sweep:{sweep_id}", "sweep")
    assert (await settled(resumer, job.job_id)).state == "completed"
    assert groups == [2], "both arms were taken up as one group"
    control = off_loop(ops_fixture, store=ClockedStore()).host
    (control_job,) = (
        await sweep_launch(
            control, style_sweep(TWO_STYLES, arm_schedule="interleaved", k_runs=K), TOYHOST_SCOPE, created_by="agent:c"
        )
    ).jobs
    assert (await settled(control, control_job.job_id)).state == "completed"
    control_runs = _arm_runs(control, control_job.job_id.removeprefix("sweep:"))
    for label, run_id in arm_runs.items():
        assert run_id is not None
        rows, control_id = _rows(holder, run_id), control_runs[label]
        assert control_id is not None
        control_rows = _rows(control, control_id)
        assert _cells(rows) == _cells(control_rows) and len(rows) == len(control_rows) == 3 * K, label
        assert list(compute_pass_hat_k(rows).values()) == list(compute_pass_hat_k(control_rows).values()), label
        run = _run(holder, run_id)
        assert run.status == _run(control, control_id).status == "completed"
        assert run.launch_group_id == group_id and len(run.lifetimes) == 2
        # Ordered twice, by each holding, under the sweep's one launch group, to one order.
        (first, seed_first), (second, seed_second) = [
            (order, seed) for ordered, seed, order in orders if ordered == run_id
        ]
        assert seed_first == seed_second == group_id
        assert first == second, "the arm kept the order its launch gave it"
        resumed = {(row.test_case_id, row.k_iteration) for row in rows if row.lease_epoch == 1}
        assert resumed == {(case_id, k) for k, case_id in second} - set(done[label]), "only what was missing ran"
    sweep = _sweep(holder, sweep_id)
    assert (sweep.outcome, sweep.lost, sweep.launch_group_id) == ("completed", None, group_id)
    first_holding, second_holding = sweep.lifetimes
    assert (first_holding.holder, first_holding.epoch, first_holding.outcome) == (_holder_id(holder), 0, "cancelled")
    assert first_holding.lost is not None
    assert (second_holding.holder, second_holding.epoch, second_holding.outcome) == (
        _holder_id(resumer),
        1,
        "completed",
    )


async def test_a_lost_sequential_sweep_leaves_its_finished_arm_takes_up_the_running_one_and_launches_the_rest_after(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def ready(host: OpsHost, sweep_id: str, gate: HeldGate) -> bool:
        runs = _arm_runs(host, sweep_id)
        first, second = runs[THREE_STYLES[0]], runs[THREE_STYLES[1]]
        return (
            first is not None
            and _run(host, first).status == "completed"
            and second is not None
            and len(_rows(host, second)) == 1
            and gate.held >= 1
        )

    # The first arm's three cells, and one of the second's.
    holder, _store, sweep_id = await _lost_sweep(
        monkeypatch, THREE_STYLES, arm_schedule="sequential", k_runs=1, passing=4, ready=ready
    )
    before = _arm_runs(holder, sweep_id)
    finished, running = before[THREE_STYLES[0]], before[THREE_STYLES[1]]
    assert finished is not None and running is not None and before[THREE_STYLES[2]] is None
    finished_rows = _rows(holder, finished)
    resumer = other_process(holder)

    (job,) = (await job_resume(resumer, f"sweep:{sweep_id}", TOYHOST_SCOPE, max_cost_usd=RAISED_CAP)).jobs

    assert (await settled(resumer, job.job_id)).state == "completed"
    after = _arm_runs(holder, sweep_id)
    assert (after[THREE_STYLES[0]], after[THREE_STYLES[1]]) == (finished, running), "the same runs, taken up"
    launched = after[THREE_STYLES[2]]
    assert launched is not None
    left, taken_up, last = _run(holder, finished), _run(holder, running), _run(holder, launched)
    assert left.lifetimes == [] and _rows(holder, finished) == finished_rows, "the finished arm was left alone"
    assert (taken_up.status, len(taken_up.lifetimes), len(_rows(holder, running))) == ("completed", 2, 3)
    assert (last.status, last.lifetimes, len(_rows(holder, launched))) == ("completed", [], 3)
    assert last.overlays.get("prompt_style") == THREE_STYLES[2], "launched from what the sweep recorded"
    assert (last.k_runs, last.judge_model) == (left.k_runs, left.judge_model), "pinned to the first arm's judge"
    assert taken_up.completed_at is not None and last.created_at >= taken_up.completed_at, "launched after, in turn"
    assert (taken_up.max_cost_usd, last.max_cost_usd) == (RAISED_CAP, RAISED_CAP), "it caps each arm it ran"
    assert left.max_cost_usd != RAISED_CAP, "the finished arm keeps its own"
    assert taken_up.lifetimes[1].max_cost_usd == RAISED_CAP
    sweep = _sweep(holder, sweep_id)
    assert (sweep.outcome, len(sweep.lifetimes)) == ("completed", 2)
    assert sweep.arguments is not None and sweep.arguments.max_cost_usd == RAISED_CAP
    assert [holding.max_cost_usd for holding in sweep.lifetimes] == [None, RAISED_CAP], "each holding's cap"
    campaign = off_loop(holder.eval_host.storage.load_campaign, sweep.campaign_id, TOYHOST_SCOPE)
    assert campaign is not None and set(campaign.run_ids) >= {finished, running, launched}


@pytest.mark.parametrize("ended", ["cancelled", "timed_out"])
async def test_a_sweep_an_operator_cancelled_or_its_own_budget_stopped_is_taken_up_and_completes(
    monkeypatch: pytest.MonkeyPatch, ended: str
) -> None:
    gate = held_extraction(monkeypatch)
    store = ClockedStore()
    holder = off_loop(ops_fixture, store=store).host
    (job,) = (
        await sweep_launch(holder, style_sweep(TWO_STYLES, arm_schedule="sequential"), TOYHOST_SCOPE, created_by="t")
    ).jobs
    sweep_id = job.job_id.removeprefix("sweep:")
    await until(lambda: gate.held >= 1)
    await job_cancel(holder, job.job_id, TOYHOST_SCOPE, reason="shipping a fix")
    assert (await settled(holder, job.job_id)).state == "cancelled"
    gate.set()
    manager = holder.launch.job_manager
    await until(
        lambda: (
            not manager.is_task_active(sweep_id)
            and not any(manager.is_active(run_id) for run_id in _arm_runs(holder, sweep_id).values() if run_id)
        )
    )
    if ended == "timed_out":
        # What a sweep its own budget stopped stores: its arms in flight cancelled, none launched after, timed out.
        document = store.get(sweep_id, TOYHOST_SCOPE)
        assert document is not None
        store.upsert(document | {"outcome": "timed_out", "detail": "the sweep ran out of its budget"})
    assert _arm_runs(holder, sweep_id)[TWO_STYLES[1]] is None, "its second arm never launched"

    (resumed,) = (await job_resume(holder, job.job_id, TOYHOST_SCOPE)).jobs

    assert (await settled(holder, resumed.job_id)).state == "completed"
    sweep = _sweep(holder, sweep_id)
    assert [holding.outcome for holding in sweep.lifetimes] == [ended, "completed"]
    for run_id in _arm_runs(holder, sweep_id).values():
        assert run_id is not None and _run(holder, run_id).status == "completed"


async def test_a_sweep_taken_up_again_and_then_lost_closes_its_holding_with_the_loss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    holder, store, sweep_id = await _lost_interleaved_sweep(monkeypatch, k_runs=1, stored=2)
    again = held_extraction(monkeypatch)  # the second holding's cells wait in flight until it too is lost
    resumer = other_process(holder)
    (job,) = (await job_resume(resumer, f"sweep:{sweep_id}", TOYHOST_SCOPE)).jobs
    await until(lambda: again.held >= 1)

    # The second holding's lease, granted at the store's T0+61s, lapses in turn.
    store.at = T0 + timedelta(seconds=200)
    assert (await job_poll(other_process(holder), job.job_id, TOYHOST_SCOPE)).state == "lost"
    again.set()
    await until(lambda: not resumer.launch.job_manager.is_task_active(sweep_id))

    sweep = _sweep(holder, sweep_id)
    assert (sweep.outcome, sweep.lost is not None) == ("cancelled", True)
    first, second = sweep.lifetimes
    assert (first.outcome, first.lost is not None) == ("cancelled", True)
    assert (second.holder, second.epoch, second.outcome) == (_holder_id(resumer), 1, "cancelled")
    assert second.lost is not None and second.lost == sweep.lost and second.ended_at == sweep.ended_at


# =============================================================================
# Old records
# =============================================================================


def _as_stored_before_arguments(store: ClockedStore, sweep_id: str) -> None:
    """Rewrite a sweep's record as a build before sweeps recorded their arms' arguments stored it."""
    document = store.get(sweep_id, TOYHOST_SCOPE)
    assert document is not None
    for field in ("arguments", "launch_group_id", "lifetimes"):
        document.pop(field, None)
    for arm in document["arms"]:
        arm.pop("arguments", None)
    store.upsert(document)


async def test_a_sweep_stored_before_it_recorded_its_arms_is_taken_up_when_every_arm_launched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orders = _spied_orders(monkeypatch)
    holder, store, sweep_id = await _lost_interleaved_sweep(monkeypatch, k_runs=1, stored=2)
    group_id = _sweep(holder, sweep_id).launch_group_id
    _as_stored_before_arguments(store, sweep_id)
    assert _sweep(holder, sweep_id).arguments is None
    resumer = other_process(holder)

    (job,) = (await job_resume(resumer, f"sweep:{sweep_id}", TOYHOST_SCOPE)).jobs

    assert (await settled(resumer, job.job_id)).state == "completed"
    sweep = _sweep(holder, sweep_id)
    assert sweep.launch_group_id == group_id, "its group, read off its arms' runs"
    for run_id in _arm_runs(holder, sweep_id).values():
        assert run_id is not None and _run(holder, run_id).status == "completed"
        assert [seed for ordered, seed, _ in orders if ordered == run_id] == [group_id, group_id]


async def test_a_sweep_stored_before_it_recorded_its_arms_refuses_an_arm_it_never_launched_naming_it() -> None:
    fixture = off_loop(ops_fixture, store=ClockedStore())
    host, storage = fixture.host, fixture.host.eval_host.storage
    finished = _run(host, fixture.campaign.run_ids[0])
    sweep = EvalSweep(
        scope_id=TOYHOST_SCOPE,
        campaign_id=fixture.campaign.id,
        template_id=toyhost_template().id,
        subject_id=TOYHOST_SUBJECT.subject_id,
        arms=[
            SweepArmRecord(label="first", model=finished.candidate_model, run_id=finished.id),
            SweepArmRecord(label="never", model=RUN_MODELS[1]),
        ],
        max_concurrent_arms=1,
        arm_schedule="sequential",
        outcome="cancelled",
        detail="lost: its process went away",
        ended_at=T0.isoformat(),
    )
    off_loop(storage.save_sweep, sweep)

    with pytest.raises(ValidationFailedError) as refused:
        await job_resume(other_process(host), f"sweep:{sweep.id}", TOYHOST_SCOPE)

    assert "arm 'never' never launched" in refused.value.message, refused.value.message
    assert "arm 'first'" not in refused.value.message, "an arm that ended is left alone, and refused nothing"
    assert "run_launch" in refused.value.remedy
    assert _sweep(host, sweep.id) == sweep, "a refused take-up writes nothing"


# =============================================================================
# Drift
# =============================================================================


async def test_one_arm_that_would_measure_otherwise_refuses_the_whole_sweep_before_anything_is_spent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    holder, store, sweep_id = await _lost_interleaved_sweep(monkeypatch, k_runs=1, stored=2)
    arm_runs = _arm_runs(holder, sweep_id)
    drifted = arm_runs["verbose"]
    assert drifted is not None
    document = store.get(drifted, TOYHOST_SCOPE)
    assert document is not None
    store.upsert(document | {"kind_spec": {"graded_fields": list(ExtractorSpec().graded_fields[:1])}})
    sweep_before = _sweep(holder, sweep_id)
    runs_before = {run_id: _run(holder, run_id) for run_id in arm_runs.values() if run_id is not None}
    rows_before = {run_id: _cells(_rows(holder, run_id)) for run_id in runs_before}

    with pytest.raises(ValidationFailedError) as refused:
        await job_resume(other_process(holder), f"sweep:{sweep_id}", TOYHOST_SCOPE)

    message = refused.value.message
    assert message.startswith(f"sweep '{sweep_id}' is not taken up again"), message
    assert "arm 'verbose'" in message and "kind_spec" in message and "arm 'terse'" not in message, message
    assert "run_launch" in refused.value.remedy
    assert _sweep(holder, sweep_id) == sweep_before, "the sweep's record is as it was"
    assert {run_id: _run(holder, run_id) for run_id in runs_before} == runs_before, "so is every arm's run"
    assert {run_id: _cells(_rows(holder, run_id)) for run_id in runs_before} == rows_before, "nothing ran"


# =============================================================================
# A launch
# =============================================================================


async def test_a_lost_interleaved_launch_is_taken_up_as_one_group_under_its_id_in_its_original_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orders = _spied_orders(monkeypatch)
    gate = held_extraction(monkeypatch, passing=4)
    store = ClockedStore()
    holder = off_loop(ops_fixture, store=store).host
    arguments = LaunchArguments(
        template_id=toyhost_template().id, subject_id=TOYHOST_SUBJECT.subject_id, models=list(RUN_MODELS), k_runs=K
    )
    (job,) = (await run_launch(holder, arguments, TOYHOST_SCOPE)).jobs
    launch_id = job.job_id.removeprefix("launch:")

    def launched() -> EvalLaunch:
        launch, _ = off_loop(holder.eval_host.storage.load_job_record, EvalLaunch, launch_id, TOYHOST_SCOPE)
        assert launch is not None
        return launch

    await until(
        lambda: (
            len(launched().run_ids) == 2
            and sum(len(_rows(holder, run_id)) for run_id in launched().run_ids) == 4
            and gate.held >= 2
        )
    )
    run_ids = launched().run_ids
    store.at = T0 + timedelta(seconds=61)
    assert (await job_poll(other_process(holder), job.job_id, TOYHOST_SCOPE)).state == "lost"
    gate.set()
    manager = holder.launch.job_manager
    await until(lambda: not manager.is_task_active(launch_id) and not any(manager.is_active(i) for i in run_ids))
    resumer = other_process(holder)
    groups = _spied_groups(monkeypatch, resumer)

    (resumed,) = (await job_resume(resumer, job.job_id, TOYHOST_SCOPE)).jobs

    assert (resumed.job_id, resumed.kind) == (job.job_id, "launch")
    assert (await settled(resumer, job.job_id)).state == "completed"
    assert groups == [2], "its runs were taken up as one group"
    launch = launched()
    assert (launch.outcome, launch.run_ids, launch.lost) == ("completed", run_ids, None)
    for run_id in run_ids:
        run = _run(holder, run_id)
        assert (run.status, len(run.lifetimes), len(_rows(holder, run_id))) == ("completed", 2, 3 * K)
        (first, seed_first), (second, seed_second) = [
            (order, seed) for ordered, seed, order in orders if ordered == run_id
        ]
        assert seed_first == seed_second == launch_id == run.launch_group_id
        assert first == second, "the run kept the order its launch gave it"
