"""Admitted work runs once: what admission priced is spent by one run, and no run writes over another's record.

- **A second start of the same work is refused** before anything is written or spent — as a job and in the
  foreground alike — and the first run's record is left as it ended.
- **A job's first write is create-only**: a record already stored under the work's id is never written over, and
  what admission built is released.
"""

from __future__ import annotations

import pytest
from pydantic import Field

from threetears.evals.kernel import ConflictError, JobLeases
from threetears.evals.run import EvalActionJob, action_job_work, run_job_work, start_job_work
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
