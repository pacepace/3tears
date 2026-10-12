"""A job whose holder died reads lost on every surface, not only through ``job_poll`` on its own job.

After a hard kill, ``run_get`` and ``runs_list`` kept reporting a lost run as ``running``: only a ``job_poll`` on that
run's own job, or a boot sweep, settled it, and a launch read lost left the runs it started running. A dead process
could hold work that counted as running for as long as nobody polled it.

Now every store read of a leased record kind goes through one settling path — the lease read ``job_poll`` uses — so
a record that has not ended and whose lease lapsed (or which carries none) reads lost wherever it is read, and a
launch or a sweep written lost settles the runs it names. The kinds are found on the public roots, so a kind added
without its reads here fails; the store's readers of each are found off ``EvalStorage``, so a reader added without
being held here fails too.
"""

from __future__ import annotations

import importlib
import inspect
import re
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from pydantic import ValidationError

from threetears.evals import PUBLIC_ROOTS
from threetears.evals.actions import eval_catalogue, standard_tools
from threetears.evals.kernel import EvalStorage, LeasedJobKind, settle_job_record
from threetears.evals.kernel.campaign import EvalAnalysisAttempt, EvalSweep, SweepArmRecord
from threetears.evals.run import (
    LAUNCH_JOB,
    EvalActionJob,
    EvalJudgeTemperatureCheck,
    EvalLaunch,
    EvalSecondJudgePass,
)
from threetears.evals.schema import JobLease, SecondJudge
from threetears.evals.schema.models import EvalRun
from threetears.evals.schema.store_port import StoreConflict
from threetears.evals.storage import InMemoryDocumentStore
from packages.evals.tests.factories import make_analysis_attempt, make_eval_result, make_eval_run
from packages.evals.tests.ops_support import CALLER, TOYHOST_SCOPE, ops_fixture
from packages.evals.tests.off_loop import off_loop

_SCOPE = "uni-1"


def _leased_kinds() -> dict[type, LeasedJobKind[Any]]:
    found: dict[type, LeasedJobKind[Any]] = {}
    for root in PUBLIC_ROOTS:
        module = importlib.import_module(root)
        for name in module.__all__:
            value = getattr(module, name)
            if isinstance(value, LeasedJobKind):
                found[value.model] = value
    return found


#: A record of each leased kind that has not ended and carries no lease: what a holder that died before it wrote
#: one, or a build from before leases, leaves behind. Every process reads it lost.
_OPEN: dict[type, Callable[[], Any]] = {
    EvalRun: lambda: make_eval_run(scope_id=_SCOPE, status="running"),
    EvalSweep: lambda: EvalSweep(
        scope_id=_SCOPE, campaign_id="c-1", template_id="t-1", subject_id="s-1",
        arms=[SweepArmRecord(label="a", model="m")], max_concurrent_arms=1,
    ),
    EvalAnalysisAttempt: lambda: make_analysis_attempt(scope_id=_SCOPE, outcome="running", analysis_id=None),
    EvalLaunch: lambda: EvalLaunch(id="launch-1", scope_id=_SCOPE, template_id="t-1", subject_id="s-1", models=["m"], k_runs=1, n_variations=0),
    EvalSecondJudgePass: lambda: EvalSecondJudgePass(
        scope_id=_SCOPE, run_id="run-1", asked_by="judge_second", judge=SecondJudge(model="judge/b"),
        sample_fraction=0.5, sample_seed=0,
    ),
    EvalJudgeTemperatureCheck: lambda: EvalJudgeTemperatureCheck(scope_id=_SCOPE, run_id="run-1", selection="all", repeats=2),
    EvalActionJob: lambda: EvalActionJob(scope_id=_SCOPE, action="rubric_propose"),
}  # fmt: skip

#: Every store read of each kind's records a surface goes through. A kind with none is read only through ``job_poll``,
#: which settles it (``test_model_calling_jobs``).
_READS: dict[type, list[Callable[[EvalStorage, Any], Any]]] = {
    EvalRun: [
        lambda storage, run: storage.load_eval_run(run.id, _SCOPE),
        lambda storage, run: next(r for r in storage.query_eval_runs(_SCOPE) if r.id == run.id),
        lambda storage, run: next(r for r in storage.query_eval_runs(_SCOPE, status="cancelled") if r.id == run.id),
        lambda storage, run: next(iter(storage.load_eval_runs([run.id], _SCOPE))),
    ],
    EvalSweep: [lambda storage, sweep: storage.load_sweep(sweep.id, _SCOPE)],
    EvalAnalysisAttempt: [
        lambda storage, attempt: next(iter(storage.list_analysis_attempts_by_campaign(attempt.campaign_id, _SCOPE)))
    ],
    EvalLaunch: [],
    EvalSecondJudgePass: [],
    EvalJudgeTemperatureCheck: [],
    EvalActionJob: [],
}

#: ``EvalStorage``'s reads that are not settled, on purpose: the lease machinery's own (what settling is made of), the
#: reads a read-modify-write takes an etag from, and the boot reclaim's scan, which counts what it writes lost itself.
_RAW_READS = {
    "load_job_record",
    "load_eval_run_with_etag",
    "query_non_terminal_eval_runs",
    "load_generation_slot_with_etag",
}

#: Each settled reader of a leased model on ``EvalStorage``, by name, as ``_READS`` exercises them.
_SETTLED_READERS = {
    "load_eval_run",
    "query_eval_runs",
    "load_eval_runs",
    "load_sweep",
    "list_analysis_attempts_by_campaign",
}


def test_every_leased_kind_on_the_public_roots_is_read_here() -> None:
    assert set(_leased_kinds()) == set(_OPEN) == set(_READS)


def test_every_store_reader_of_a_leased_record_is_settled_or_named_raw() -> None:
    names = {model.__name__ for model in _leased_kinds()}
    readers = {
        name
        for name, member in inspect.getmembers(EvalStorage, inspect.isfunction)
        if not name.startswith("_")
        and any(re.search(rf"\b{model}\b", str(inspect.signature(member).return_annotation)) for model in names)
    }
    assert readers - _RAW_READS == _SETTLED_READERS


@pytest.mark.parametrize("model", list(_OPEN), ids=lambda model: model.__name__)
def test_a_record_whose_holder_died_reads_lost_through_every_store_read(model: type) -> None:
    storage = EvalStorage(InMemoryDocumentStore())
    record = _OPEN[model]()
    storage.save_job_record(record.to_dict(), if_match=None)

    for read in _READS[model]:
        read_back = read(storage, record)
        assert read_back.lost is not None, f"{model.__name__} read through {read} as not lost"
    if model is EvalRun:
        assert [r.id for r in storage.query_eval_runs(_SCOPE, status="running")] == []


def _live_lease() -> JobLease:
    return JobLease(holder="another-process", expires_at=(datetime.now(UTC) + timedelta(hours=1)).isoformat())


@pytest.mark.parametrize("named_by", ["launch", "sweep"])
def test_a_launch_or_sweep_written_lost_settles_the_runs_it_names(named_by: str) -> None:
    storage = EvalStorage(InMemoryDocumentStore())
    orphan = make_eval_run(scope_id=_SCOPE, status="pending")
    held = make_eval_run(scope_id=_SCOPE, status="running", lease=_live_lease())
    for run in (orphan, held):
        storage.save_job_record(run.to_dict(), if_match=None)
    if named_by == "launch":
        launch = _OPEN[EvalLaunch]().model_copy(update={"run_ids": [orphan.id, held.id]})
        storage.save_job_record(launch.to_dict(), if_match=None)
        settle_job_record(storage, LAUNCH_JOB, launch.id, _SCOPE)
    else:
        arms = [
            SweepArmRecord(label="a", model="m", run_id=orphan.id),
            SweepArmRecord(label="b", model="m", run_id=held.id),
        ]
        sweep = _OPEN[EvalSweep]().model_copy(update={"arms": arms})
        storage.save_job_record(sweep.to_dict(), if_match=None)
        storage.load_sweep(sweep.id, _SCOPE)

    # Read raw, as the lease machinery reads: nothing but the launch's or the sweep's settling has read the runs.
    orphaned, _ = storage.load_job_record(EvalRun, orphan.id, _SCOPE)
    still_held, _ = storage.load_job_record(EvalRun, held.id, _SCOPE)
    assert orphaned is not None and orphaned.lost is not None and orphaned.status == "cancelled"
    assert still_held is not None and still_held.lost is None and still_held.status == "running"


async def test_after_a_hard_kill_run_get_and_runs_list_alone_report_the_run_lost() -> None:
    """The reported repro: the holder killed mid-run, its lease lapsed; nothing polls the run's job."""
    fixture = off_loop(ops_fixture)
    storage = fixture.host.eval_host.storage
    expired = JobLease(holder="killed-process", expires_at=(datetime.now(UTC) - timedelta(minutes=5)).isoformat())
    run = make_eval_run(scope_id=TOYHOST_SCOPE, status="running", lease=expired)
    off_loop(storage.save_job_record, run.to_dict(), if_match=None)
    evals, _admin = eval_catalogue().mount_all(standard_tools())

    got = await evals.call({"action": "run_get", "run_id": run.id}, host=fixture.host, caller=CALLER)
    listed = await evals.call({"action": "runs_list"}, host=fixture.host, caller=CALLER)

    assert not got.is_error and got.structured is not None and got.structured["status"] == "cancelled", got.text
    assert not listed.is_error and listed.structured is not None
    (line,) = [line for line in listed.structured["runs"] if line["id"] == run.id]
    assert line["status"] == "cancelled", line


def _unreadable_result(storage: EvalStorage, store: InMemoryDocumentStore, run: EvalRun) -> None:
    """Store one readable result of ``run`` and one this build cannot reconstruct (a corpus that outlived its model)."""
    readable = make_eval_result(id="res-ok", scope_id=_SCOPE, eval_run_id=run.id)
    storage.save_eval_result(readable)
    store.upsert(readable.to_dict() | {"id": "res-bad", "turns_delivered": "several"})


def test_a_lost_run_whose_result_cannot_be_read_settles_with_its_completeness_unknown() -> None:
    """One unreadable result broke runs_list for the whole scope, run_get for its run, and the run never settled."""
    store = InMemoryDocumentStore()
    storage = EvalStorage(store)
    run = make_eval_run(scope_id=_SCOPE, status="running")
    beside = make_eval_run(scope_id=_SCOPE, status="completed")
    for record in (run, beside):
        storage.save_job_record(record.to_dict(), if_match=None)
    _unreadable_result(storage, store, run)

    got = storage.load_eval_run(run.id, _SCOPE)
    assert got is not None and got.status == "cancelled" and got.lost is not None and got.completeness is None
    assert {listed.id: listed.status for listed in storage.query_eval_runs(_SCOPE)} == {
        run.id: "cancelled",
        beside.id: "completed",
    }
    stored, _ = storage.load_job_record(EvalRun, run.id, _SCOPE)
    assert stored is not None and stored.lost is not None, "the run was written lost, not merely read so"


def test_a_lost_run_whose_completeness_is_unknown_says_why_on_its_own_record() -> None:
    """The unreadable result was only logged: the record read as an unknown completeness with no fault named on it."""
    store = InMemoryDocumentStore()
    storage = EvalStorage(store)
    run = make_eval_run(scope_id=_SCOPE, status="running")
    storage.save_job_record(run.to_dict(), if_match=None)
    _unreadable_result(storage, store, run)

    assert storage.load_eval_run(run.id, _SCOPE) is not None  # the read that settles it
    stored, _ = storage.load_job_record(EvalRun, run.id, _SCOPE)

    assert stored is not None and stored.completeness is None
    (detail,) = stored.error_details
    assert "completeness unknown" in detail and "turns_delivered" in detail, detail


class _ContestedStore(InMemoryDocumentStore):
    """Every conditional write of one record loses its race, as a record another writer keeps moving does."""

    def __init__(self, contested: str) -> None:
        super().__init__()
        self.contested = contested

    def upsert(self, document: dict[str, Any], *, if_match: str | None = None, create_only: bool = False) -> None:
        if document["id"] == self.contested and if_match is not None:
            raise StoreConflict(f"document {self.contested!r} keeps moving")
        super().upsert(document, if_match=if_match, create_only=create_only)


def test_one_run_that_cannot_be_settled_never_breaks_a_list_or_its_neighbours() -> None:
    """The boot reclaim's per-row tolerance, on every settling read: the contested run reads as stored, the rest lost."""
    contested = make_eval_run(scope_id=_SCOPE, status="running")
    store = _ContestedStore(contested.id)
    storage = EvalStorage(store)
    other = make_eval_run(scope_id=_SCOPE, status="running")
    for record in (contested, other):
        storage.save_job_record(record.to_dict(), if_match=None)

    statuses = {listed.id: listed.status for listed in storage.query_eval_runs(_SCOPE)}
    assert statuses == {contested.id: "running", other.id: "cancelled"}
    got = storage.load_eval_run(contested.id, _SCOPE)
    assert got is not None and got.status == "running"


def test_a_launch_written_lost_settles_the_runs_it_can_and_leaves_the_rest() -> None:
    """A run of a lost launch that cannot be settled does not keep the launch, or its other runs, from settling."""
    contested = make_eval_run(scope_id=_SCOPE, status="pending")
    store = _ContestedStore(contested.id)
    storage = EvalStorage(store)
    orphan = make_eval_run(scope_id=_SCOPE, status="pending")
    unreadable = make_eval_run(scope_id=_SCOPE, status="running")
    for run in (contested, orphan, unreadable):
        storage.save_job_record(run.to_dict(), if_match=None)
    _unreadable_result(storage, store, unreadable)
    launch = _OPEN[EvalLaunch]().model_copy(update={"run_ids": [contested.id, unreadable.id, orphan.id]})
    storage.save_job_record(launch.to_dict(), if_match=None)

    settled, wrote_lost = settle_job_record(storage, LAUNCH_JOB, launch.id, _SCOPE)

    assert wrote_lost and settled is not None and settled.lost is not None
    read = {run.id: storage.load_job_record(EvalRun, run.id, _SCOPE)[0] for run in (contested, orphan, unreadable)}
    assert read[contested.id] is not None and read[contested.id].lost is None  # type: ignore[union-attr]
    assert read[orphan.id] is not None and read[orphan.id].lost is not None  # type: ignore[union-attr]
    assert read[unreadable.id] is not None and read[unreadable.id].lost is not None  # type: ignore[union-attr]


def test_a_run_this_build_cannot_read_is_not_settled_as_if_it_were_still_running() -> None:
    """A named run whose document would not validate was left as stored and logged; the launch read as all settled."""
    store = InMemoryDocumentStore()
    storage = EvalStorage(store)
    orphan = make_eval_run(scope_id=_SCOPE, status="pending")
    storage.save_job_record(orphan.to_dict(), if_match=None)
    unreadable = make_eval_run(scope_id=_SCOPE, status="running")
    store.upsert(unreadable.to_dict() | {"k_runs": "several"})
    launch = _OPEN[EvalLaunch]().model_copy(update={"run_ids": [orphan.id, unreadable.id]})
    storage.save_job_record(launch.to_dict(), if_match=None)

    with pytest.raises(ValidationError, match="k_runs"):
        settle_job_record(storage, LAUNCH_JOB, launch.id, _SCOPE)

    settled, wrote_lost = settle_job_record(storage, LAUNCH_JOB, launch.id, _SCOPE)
    assert settled is not None and settled.lost is not None and not wrote_lost, "the launch's own loss was written"
    read, _ = storage.load_job_record(EvalRun, orphan.id, _SCOPE)
    assert read is not None and read.lost is not None, "the run named before it was settled"
