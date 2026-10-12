"""Admitted work runs once: what admission priced is spent by one run, and no run writes over another's record.

- **A second start of the same work is refused** before anything is written or spent — as a job and in the
  foreground alike — and the first run's record is left as it ended.
- **A job's first write is create-only**: a record already stored under the work's id is never written over, and
  what admission built is released.
- **A cancel during a write keeps what the write stored**: the job's copy of its record is what the store holds, so
  the cancelled ending, built from it, drops nothing the work recorded.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any

import pytest
from pydantic import Field

from threetears.evals.kernel import ConflictError, JobLeases
from threetears.evals.run import EvalActionJob, JobRecorder, action_job_work, run_job_work, start_job_work
from threetears.evals.schema.base import EvalBaseModel
from packages.evals.tests.ops_support import TOYHOST_SCOPE, ops_fixture, settled
from packages.evals.tests.off_loop import off_loop


class _Draft(EvalBaseModel):
    """What a drafting action produces."""

    text: str = Field(description="The draft.")


class _Spend:
    """Counts what a piece of work paid for, and what admission released."""

    def __init__(self) -> None:
        """Nothing yet."""
        self.calls = 0
        self.released = 0

    async def draft(self) -> _Draft:
        """One paid call."""
        self.calls += 1
        return _Draft(text=f"draft {self.calls}")

    async def discard(self) -> None:
        """Admission's holdings handed back."""
        self.released += 1


async def test_the_same_admitted_work_started_twice_is_refused_and_spends_once() -> None:
    fixture = off_loop(ops_fixture)
    storage, executor = fixture.host.eval_host.storage, fixture.host.eval_host.blocking_executor
    spend = _Spend()
    work = action_job_work(TOYHOST_SCOPE, action="draft", run=spend.draft, discard=spend.discard, budget_s=60.0)

    await start_job_work(fixture.host.launch.job_manager, work, store=storage, executor=executor)
    first = await settled(fixture.host, f"action:{work.record_id}")
    with pytest.raises(RuntimeError, match="was already started; admitted work runs once"):
        await start_job_work(fixture.host.launch.job_manager, work, store=storage, executor=executor)
    with pytest.raises(RuntimeError, match="was already started"):
        await run_job_work(work, store=storage, executor=executor, leases=fixture.host.launch.job_leases)

    assert spend.calls == 1, "the second start paid for nothing"
    assert first.result == {"text": "draft 1"}
    stored, _ = off_loop(storage.load_job_record, EvalActionJob, work.record_id, TOYHOST_SCOPE)
    assert stored is not None and (stored.outcome, stored.result) == ("completed", {"text": "draft 1"})


async def test_a_record_already_stored_under_the_works_id_is_never_written_over() -> None:
    fixture = off_loop(ops_fixture)
    storage, executor = fixture.host.eval_host.storage, fixture.host.eval_host.blocking_executor
    spend = _Spend()
    work = action_job_work(TOYHOST_SCOPE, action="draft", run=spend.draft, discard=spend.discard, budget_s=60.0)
    there = work.record.model_copy(update={"outcome": "completed", "result": {"text": "an earlier run's"}})
    off_loop(storage.save_job_record, there.to_dict(), if_match=None)
    leases = JobLeases(storage, blocking_executor=executor)

    with pytest.raises(ConflictError):
        await run_job_work(work, store=storage, executor=executor, leases=leases)

    stored, _ = off_loop(storage.load_job_record, EvalActionJob, work.record_id, TOYHOST_SCOPE)
    assert stored == there, "the stored record was left as it was"
    assert (spend.calls, spend.released) == (0, 1), "nothing ran, and what admission built was released"


class _GatedStore:
    """A job record store whose writes wait on a gate, so a test can cancel while one is on a worker."""

    def __init__(self, store: Any) -> None:
        """Wrap ``store``; every other call goes straight through."""
        self._store = store
        self.entered = threading.Event()
        self.release = threading.Event()

    def save_job_record(self, *args: Any, **kwargs: Any) -> Any:
        """Say the write has started, wait for the gate, then write."""
        self.entered.set()
        assert self.release.wait(5), "the test never opened the gate"
        return self._store.save_job_record(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        """Everything else is the wrapped store's."""
        return getattr(self._store, name)


async def test_a_cancel_while_a_write_is_on_a_worker_keeps_what_the_write_stored() -> None:
    """The cancel cannot stop the write, so the job's copy must be what was written, or the ending writes over it."""
    fixture = off_loop(ops_fixture)
    storage, executor = fixture.host.eval_host.storage, fixture.host.eval_host.blocking_executor
    spend = _Spend()
    work = action_job_work(TOYHOST_SCOPE, action="draft", run=spend.draft, discard=spend.discard, budget_s=60.0)
    gated = _GatedStore(storage)
    recorder = JobRecorder(gated, executor, work.kind, work.record, leases=None)
    progressed = work.record.model_copy(update={"result": {"text": "paid for"}})

    saving = asyncio.create_task(recorder.save(progressed))
    assert await asyncio.to_thread(gated.entered.wait, 5), "the write reached its worker"
    saving.cancel()
    await asyncio.sleep(0)
    gated.release.set()
    with pytest.raises(asyncio.CancelledError):
        await saving

    stored, _ = off_loop(storage.load_job_record, EvalActionJob, work.record_id, TOYHOST_SCOPE)
    assert stored is not None and stored.result == {"text": "paid for"}, "the write landed despite the cancel"
    assert recorder.record.result == {"text": "paid for"}, "the job's copy is what the store holds"
