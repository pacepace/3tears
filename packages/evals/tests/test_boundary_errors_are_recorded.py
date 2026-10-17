"""A boundary that contains an error writes it where an operator reads it, when there is a record to write it on.

Some handlers are isolation boundaries: they keep an error from changing an outcome that is not theirs to change (a
run that executed its matrix, a job whose lease machinery failed). Each used to put the error in the log and nowhere
else. Where a run or job record exists at that point, the error is now also written on it — ``EvalRun.error_details``
for a run's own work, a job record's ``faults`` for its lease (every kind alike, kept beside the ending a loss writes
over), the loss's cause for a job record ended lost — naming where it happened, its kind and its message. What the
caller receives is unchanged, and a record write that fails as well is logged with both errors.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from collections.abc import Callable
from typing import Any

import pytest
from pydantic import BaseModel

from threetears.evals.kernel.candidate_kind import CandidateOutput, CellSink, VariantConfig
from threetears.evals.kernel.cassettes import ActionSeam, CellCassettes, DeliverySeam, ToolWrap
from threetears.evals.actions.render import render_job_status
from threetears.evals.kernel import ANALYSIS_JOB, RUN_JOB, SWEEP_JOB, EvalSweep, JobLeases, SweepArmRecord
from threetears.evals.kernel.job_leases import LeasedJobKind, settle_job_record
from threetears.evals.ops import job_poll, sweep_job_id
from threetears.evals.run import LAUNCH_JOB, EvalLaunch
from threetears.evals.kernel.errors import ConflictError, StorageError
from threetears.evals.kernel.host.eval_host import EvalHost
from threetears.evals.kernel.scoring import CellSummary
from threetears.evals.kernel.storage import EvalStorage
from threetears.evals.run.jobs import EvalJobManager
from threetears.evals.run.lifecycle import record_completeness
from threetears.evals.run.runner import RunnerOptions, execute_run
from threetears.evals.schema.base import EvalDocumentModel
from threetears.evals.schema.models import EvalCassette, EvalRun, JudgedArtifact
from threetears.evals.storage import InMemoryDocumentStore
from packages.evals.tests.factories import (
    DEFAULT_KIND,
    make_analysis_attempt,
    make_eval_result,
    make_eval_run,
    make_template,
    make_test_case,
)
from packages.evals.tests.fixtures.toyhost.host import toyhost_host
from packages.evals.tests.job_support import InMemoryRunStore
from packages.evals.tests.ops_support import TOYHOST_SCOPE, ops_fixture
from packages.evals.tests.off_loop import off_loop

# =============================================================================
# record_completeness: a record that is not written says so on the run
# =============================================================================


class _CompletenessRefusingStore(InMemoryRunStore):
    """Refuses every write of a run carrying a completeness record, as a store another writer keeps winning against."""

    def __init__(self, *, refuse_everything: bool = False) -> None:
        """Start with no runs; ``refuse_everything`` refuses the follow-up write too."""
        super().__init__()
        self.refuse_everything = refuse_everything

    def save_eval_run(self, run: EvalRun, *, if_match: str | None = None) -> None:
        """Refuse a completeness write (or every write); keep anything else."""
        if self.refuse_everything or run.completeness is not None:
            raise ConflictError(f"run {run.id} moved under the write")
        super().save_eval_run(run, if_match=if_match)


def _cells(run: EvalRun) -> list[CellSummary]:
    return [CellSummary.from_result(make_eval_result(eval_run_id=run.id, test_case_id="tc-1"), persisted=True)]


@pytest.mark.expect_contained(
    TypeError,
    why="the refusing store builds ConflictError without its remedy, so its write raises TypeError into the generic write-fault path",
)
def test_a_completeness_record_that_is_not_written_says_so_on_the_run() -> None:
    """Before: the run's completeness read unknown, and why was only in the log. Now it is an ``error_details`` entry."""
    storage = _CompletenessRefusingStore()
    run = make_eval_run()
    storage.runs[run.id] = run

    record_completeness(storage, run.id, run.scope_id, _cells(run))

    stored = storage.runs[run.id]
    assert stored.completeness is None, "the record itself was refused, as the store was made to"
    [entry] = stored.error_details
    assert entry.startswith("completeness record not written"), entry
    assert "no attempt of its conditional write completed" in entry


def test_a_completeness_record_that_raises_names_the_error_on_the_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """An exception inside the record's own write is named by its type and message on the run."""
    storage = InMemoryRunStore()
    run = make_eval_run()
    storage.runs[run.id] = run

    def broken(current: EvalRun, cells: Any) -> Any:
        raise ValueError("a cell summary this build cannot count")

    monkeypatch.setattr("threetears.evals.run.lifecycle.summarize_completeness", broken)

    record_completeness(storage, run.id, run.scope_id, _cells(run))

    [entry] = storage.runs[run.id].error_details
    assert entry == (
        "completeness record not written, so the run's completeness reads unknown: ValueError: a cell summary this "
        "build cannot count"
    )


@pytest.mark.expect_contained(
    TypeError,
    why="the refusing store builds ConflictError without its remedy, so its write raises TypeError into the generic write-fault path",
)
def test_a_completeness_fault_that_cannot_be_written_either_is_logged_with_both_errors(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The run is left as it was, nothing is raised into the work function, and the log carries both failures."""
    storage = _CompletenessRefusingStore(refuse_everything=True)
    run = make_eval_run()
    storage.runs[run.id] = run

    with caplog.at_level(logging.ERROR):
        record_completeness(storage, run.id, run.scope_id, _cells(run))

    assert storage.runs[run.id].error_details == []
    [line] = [r for r in caplog.records if "could not be written on the run either" in r.getMessage()]
    assert line.levelno == logging.ERROR
    assert "refused" in line.getMessage() and "completeness record not written" in line.getMessage()


# =============================================================================
# execute_run: what its loop contained is written on the run as the loop ends
# =============================================================================


class _Roll(BaseModel):
    total: int


# parity-with: threetears.evals.kernel.cassettes.ToolLike
class _FakeDice:
    """One recordable tool: a roll."""

    @property
    def name(self) -> str:
        return "roll"

    def can_dispatch(self, action: str) -> bool:
        return action == "roll"

    async def act(self, action: str, parameters: dict[str, Any]) -> _Roll:
        await asyncio.sleep(0)
        return _Roll(total=7)


# parity-with: threetears.evals.kernel.cassettes.CassetteSeams
class _FakeSession:
    """A prepared candidate exposing its one tool through an action seam."""

    def __init__(self) -> None:
        self.tools: dict[str, Any] = {"roll": _FakeDice()}

    @property
    def action_seam(self) -> ActionSeam | None:
        return self  # type: ignore[return-value]

    @property
    def delivery_seams(self) -> Mapping[str, DeliverySeam]:
        return {}

    @property
    def recorded_tools(self) -> Mapping[str, type[_Roll]]:
        return {"roll": _Roll}

    def arm_tools(self, wrap: ToolWrap) -> None:
        self.tools = wrap(self.tools)


# parity-with: threetears.evals.kernel.candidate_kind.CandidateKind
class _FakeRollingKind:
    """Wires whatever cassettes it is handed, then rolls once per cell."""

    judged_artifact = JudgedArtifact.UNJUDGED

    async def prepare(
        self,
        *,
        subject_snapshot: Any,
        variant_config: VariantConfig,
        world_seed: Any,
        span_window: Any,
        cassettes: CellCassettes | None,
        world: Any,
    ) -> _FakeSession:
        session = _FakeSession()
        if cassettes is not None:
            cassettes.wire(session)
        return session

    async def invoke(self, instance: _FakeSession, test_case: Any, sink: CellSink) -> CandidateOutput:
        roll = await instance.tools["roll"].act("roll", {"dice": "1d20"})
        return CandidateOutput(output=[{"roll": roll.total}])


class _CassetteRefusingStorage(EvalStorage):
    """An eval store that refuses every cassette recording, and keeps everything else."""

    def save_cassette(self, cassette: EvalCassette) -> None:
        """Refuse the recording."""
        raise StorageError(f"the store refused recording {cassette.id}")


_TEMPLATE = make_template(id="tpl-roll", candidate_kind=DEFAULT_KIND, goal_state_checks=[])
_CASES = [make_test_case(id="case-1", template_id=_TEMPLATE.id)]


async def _execute(host: EvalHost, run: EvalRun, options: RunnerOptions) -> EvalRun:
    off_loop(host.storage.save_eval_run, run)
    await execute_run(host, run=run, template=_TEMPLATE, test_cases=_CASES, judge_service=None, options=options)
    # The etag read, which settles nothing: the run here carries no lease, which a settling read would write lost.
    stored, _ = off_loop(host.storage.load_eval_run_with_etag, run.id, run.scope_id)
    assert stored is not None
    return stored


def _rolling_run(**fields: Any) -> EvalRun:
    return make_eval_run(
        template_id=_TEMPLATE.id, test_case_ids=[case.id for case in _CASES], candidate_kind=DEFAULT_KIND, **fields
    )


async def test_a_capture_recording_that_is_not_written_is_named_on_the_capture_run() -> None:
    """Before: the corpus hole was a log line; a later replay found it as a miss. Now the capture run says so."""
    host = toyhost_host(storage=_CassetteRefusingStorage(InMemoryDocumentStore()))
    run = _rolling_run(cassette_mode="capture")

    stored = await _execute(host, run, RunnerOptions(candidate_kinds={DEFAULT_KIND: lambda _cell: _FakeRollingKind()}))

    [entry] = stored.error_details
    assert entry.startswith("cassette capture: recording ")
    assert "this run's corpus is incomplete" in entry
    recording = entry.split()[3]
    assert f"re-capture before replaying it: StorageError: the store refused recording {recording}" in entry
    assert stored.status == run.status, "the run's status is the job manager's, untouched"


async def test_a_raising_concurrency_probe_is_named_on_the_run_with_the_samples_it_cost() -> None:
    """Before: each raise was logged and the cells read execution_mode unmeasured, with nothing on the run."""
    host = toyhost_host()
    run = _rolling_run()

    def probe() -> int:
        raise RuntimeError("the job manager is gone")

    options = RunnerOptions(
        candidate_kinds={DEFAULT_KIND: lambda _cell: _FakeRollingKind()}, concurrent_eval_jobs_probe=probe
    )
    stored = await _execute(host, run, options)

    [entry] = stored.error_details
    assert entry.startswith("concurrency probe (RunnerOptions.concurrent_eval_jobs_probe) raised")
    assert ": RuntimeError: the job manager is gone (" in entry and entry.endswith(" samples)")
    [result] = off_loop(host.storage.query_eval_results_by_run, run.id, run.scope_id)
    assert "execution_mode" not in result.covariates, "the cell still reads unmeasured, as it did"


async def test_a_run_whose_loop_contained_nothing_carries_no_entry() -> None:
    host = toyhost_host()
    run = _rolling_run(cassette_mode="capture")

    stored = await _execute(host, run, RunnerOptions(candidate_kinds={DEFAULT_KIND: lambda _cell: _FakeRollingKind()}))

    assert stored.error_details == []


# =============================================================================
# Job leases: a holder whose renewals fail says why in its record's faults; a failed task's record says what ended it
# =============================================================================

#: When the store's clock starts; the tests move it inside the lease and hold it there, so only the holder's own
#: count can find its lease gone.
_T0 = datetime(2026, 10, 10, 12, 0, tzinfo=UTC)


class _RenewalRefusingStore(InMemoryDocumentStore):
    """A clocked store that refuses, once told to, every write moving a record's lease — or, told so, every write.

    Its clock is moved inside the lease and held there, so no reader, and no renewal, finds the lease lapsed: only
    the holder's own count can tell it its lease is gone.
    """

    def __init__(self) -> None:
        """Accept every write, at a clock that does not move."""
        super().__init__()
        self.at = _T0
        self.refusing = False
        self.unreachable = False

    def now(self) -> datetime:
        """The store's time, as the test set it."""
        return self.at

    def upsert(self, document: dict[str, Any], *, if_match: str | None = None, create_only: bool = False) -> None:
        """Refuse a renewal while refusing, and every write while unreachable; write anything else."""
        if self.unreachable:
            raise RuntimeError("the store is unreachable")
        if self.refusing and document.get("lease") is not None:
            stored = self.get(str(document["id"]), str(document["scope_id"]))
            if stored is not None and stored.get("lease") != document["lease"]:
                raise RuntimeError("the store refused the renewal")
        super().upsert(document, if_match=if_match, create_only=create_only)


async def _held_run(store: _RenewalRefusingStore) -> tuple[EvalStorage, JobLeases, EvalRun, asyncio.Event]:
    storage = EvalStorage(store)
    leases = JobLeases(storage, lease_s=0.3)
    run = make_eval_run(status="running")
    run.lease = off_loop(leases.grant)
    off_loop(storage.save_eval_run, run)
    stopped = asyncio.Event()
    leases.hold(RUN_JOB, run.id, run.scope_id, cancel=lambda _reason: None, stop=stopped.set)
    store.at = _T0 + timedelta(seconds=0.1)  # inside the lease, so every renewal moves it, for the whole test
    return storage, leases, run, stopped


async def test_a_run_whose_renewals_fail_for_a_whole_lease_is_told_why_on_its_record() -> None:
    """Before: the holder stopped the run and logged; the run later read lost as a process that went away."""
    store = _RenewalRefusingStore()
    storage, _leases, run, stopped = await _held_run(store)
    store.refusing = True

    await asyncio.wait_for(stopped.wait(), timeout=5)

    record, _ = off_loop(storage.load_job_record, EvalRun, run.id, run.scope_id)
    assert record is not None and (record.status, record.lost) == ("running", None), "no ending of its own"
    assert record.error_details == [], "a lease fault is the job's, kept in faults as on every job record"
    [entry] = record.faults
    assert entry.startswith("job lease: its holder, process ")
    assert "could not renew it for 0.3s, its whole lease, and stopped the job" in entry
    assert "— the last renewal failed: StorageError: " in entry and "the store refused the renewal" in entry


async def test_a_run_whose_renewals_fail_and_whose_record_cannot_be_written_logs_both_errors(
    caplog: pytest.LogCaptureFixture,
) -> None:
    store = _RenewalRefusingStore()
    storage, _leases, run, stopped = await _held_run(store)
    store.unreachable = True

    with caplog.at_level(logging.ERROR):
        await asyncio.wait_for(stopped.wait(), timeout=5)

    [line] = [r for r in caplog.records if "could not be written on its record either" in r.getMessage()]
    assert line.levelno == logging.ERROR
    assert "the renewal failure: the last renewal failed: StorageError: " in line.getMessage()
    assert line.exc_info is not None and "the store is unreachable" in str(line.exc_info[1])
    store.unreachable = False
    record, _ = off_loop(storage.load_job_record, EvalRun, run.id, run.scope_id)
    assert record is not None and record.faults == []


def _launch(scope_id: str) -> EvalLaunch:
    return EvalLaunch(
        id="launch-1",
        scope_id=scope_id,
        template_id="tpl-1",
        subject_id="subject-1",
        models=["m1"],
        k_runs=1,
        n_variations=0,
    )


#: Each job record kind as it is held, and the field its own ending is written in — the one a loss writes over.
_HELD_KINDS: dict[str, tuple[LeasedJobKind[Any], Callable[[str], EvalDocumentModel], Callable[[Any], str | None]]] = {
    "run": (
        RUN_JOB,
        lambda scope_id: make_eval_run(status="running", scope_id=scope_id),
        lambda r: r.cancellation_reason,
    ),
    "sweep": (SWEEP_JOB, lambda scope_id: _sweep(scope_id), lambda r: r.detail),
    "launch": (LAUNCH_JOB, _launch, lambda r: r.detail),
    "analysis attempt": (
        ANALYSIS_JOB,
        lambda scope_id: make_analysis_attempt(scope_id=scope_id, outcome="running", analysis_id=None),
        lambda r: r.error,
    ),
}


async def _expired(
    store: _RenewalRefusingStore, storage: EvalStorage, kind: LeasedJobKind[Any], record: EvalDocumentModel
) -> str:
    """Hold ``record`` under a 0.3s lease whose every renewal the store refuses, until its holder stops it; its fault."""
    leases = JobLeases(storage, lease_s=0.3)
    held = record.model_copy(update={"lease": off_loop(leases.grant)})
    off_loop(storage.save_job_record, held.to_dict(), if_match=None, create_only=True)
    stopped = asyncio.Event()
    leases.hold(kind, held.id, held.scope_id, cancel=lambda _reason: None, stop=stopped.set)
    store.at = _T0 + timedelta(seconds=0.1)  # inside the lease, so every renewal moves it and is refused
    store.refusing = True
    await asyncio.wait_for(stopped.wait(), timeout=5)
    store.refusing = False
    stored, _ = off_loop(storage.load_job_record, kind.model, held.id, held.scope_id)
    assert stored is not None and stored.lost is None and not kind.ended(stored), "no ending of its own yet"
    [fault] = stored.faults
    return fault


@pytest.mark.parametrize("name", sorted(_HELD_KINDS))
async def test_a_job_whose_renewals_fail_keeps_the_renewal_error_in_its_faults_after_it_is_written_lost(
    name: str,
) -> None:
    """Before: a sweep's, a launch's or an attempt's record had no field for it beside its ending, so the renewal
    error reached only the log, and the loss a reader wrote later read as a process that simply went away."""
    kind, make, ending = _HELD_KINDS[name]
    store = _RenewalRefusingStore()
    storage = EvalStorage(store)
    record = make("scope-1")

    fault = await _expired(store, storage, kind, record)
    store.at = _T0 + timedelta(seconds=5)  # past the expiry: the next reader writes the loss
    lost, wrote = off_loop(settle_job_record, storage, kind, record.id, record.scope_id)

    assert fault.startswith("job lease: its holder, process ")
    assert "could not renew it for 0.3s, its whole lease, and stopped the job" in fault
    assert "— the last renewal failed: StorageError: " in fault and "the store refused the renewal" in fault
    assert wrote and lost is not None and lost.lost is not None
    assert ending(lost) == f"lost: {lost.lost.cause}", "the loss wrote the record's ending"
    assert lost.faults == [fault], "and kept the fault beside it"


async def test_job_poll_shows_the_faults_of_a_job_read_lost() -> None:
    store = _RenewalRefusingStore()
    fixture = off_loop(ops_fixture, store=store)
    storage = fixture.host.eval_host.storage
    sweep = _sweep(TOYHOST_SCOPE)

    fault = await _expired(store, storage, SWEEP_JOB, sweep)
    store.at = _T0 + timedelta(seconds=5)
    status = await job_poll(fixture.host, sweep_job_id(sweep.id), TOYHOST_SCOPE)

    assert (status.state, status.faults) == ("lost", [fault])
    assert f"fault: {fault}" in render_job_status(status).splitlines()


def _sweep(scope_id: str = "scope-1") -> EvalSweep:
    return EvalSweep(
        scope_id=scope_id,
        campaign_id="campaign-1",
        template_id="tpl-1",
        subject_id="subject-1",
        arms=[SweepArmRecord(label="a", model="m1")],
        max_concurrent_arms=1,
    )


async def test_a_task_that_raises_past_its_own_ending_has_its_record_written_lost_naming_the_error() -> None:
    """Before: the task's error was logged and its record read running until its lease lapsed, then lost, causeless."""
    storage = EvalStorage(InMemoryDocumentStore())
    leases = JobLeases(storage)
    manager = EvalJobManager(storage, leases=leases)
    sweep = _sweep()
    sweep.lease = off_loop(leases.grant)
    off_loop(storage.save_sweep, sweep)

    async def work() -> None:
        raise RuntimeError("the arm list could not be read")

    manager.start_task(sweep.id, work, budget_s=None, record=(SWEEP_JOB, sweep.scope_id))
    await asyncio.wait_for(_task_ended(manager, sweep.id), timeout=5)

    record, _ = off_loop(storage.load_job_record, EvalSweep, sweep.id, sweep.scope_id)
    assert record is not None and record.lost is not None
    assert record.lost.cause == "its task raised past its own ending: RuntimeError: the arm list could not be read"
    assert (record.outcome, record.detail) == ("cancelled", f"lost: {record.lost.cause}")
    assert record.lease is None


async def test_a_task_whose_work_recorded_its_own_ending_keeps_it() -> None:
    storage = EvalStorage(InMemoryDocumentStore())
    leases = JobLeases(storage)
    manager = EvalJobManager(storage, leases=leases)
    sweep = _sweep()
    sweep.lease = off_loop(leases.grant)
    off_loop(storage.save_sweep, sweep)

    async def work() -> None:
        off_loop(storage.save_sweep, sweep.model_copy(update={"outcome": "failed", "detail": "arm 'a' was refused"}))
        raise RuntimeError("raised after the ending was written")

    manager.start_task(sweep.id, work, budget_s=None, record=(SWEEP_JOB, sweep.scope_id))
    await asyncio.wait_for(_task_ended(manager, sweep.id), timeout=5)

    record, _ = off_loop(storage.load_job_record, EvalSweep, sweep.id, sweep.scope_id)
    assert record is not None and (record.outcome, record.detail, record.lost) == (
        "failed",
        "arm 'a' was refused",
        None,
    )


async def _task_ended(manager: EvalJobManager, job_id: str) -> None:
    while manager.is_task_active(job_id):
        await asyncio.sleep(0.01)
