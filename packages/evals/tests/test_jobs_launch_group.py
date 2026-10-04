"""A launch group's runs share one concurrency slot and start together.

The arms of one campaign launch are one experiment; measured apart in time, a difference between
them can be when they ran rather than what they ran. These pin that the group starts every member
at once even under a concurrency limit below its size, that it holds exactly one slot while any
member lives, and that a group cancelled while it waits for a slot never takes one.
"""

from __future__ import annotations

import asyncio

import pytest

from threetears.evals.run.jobs import EvalJobManager
from packages.evals.tests.factories import make_eval_run
from packages.evals.tests.job_support import InMemoryRunStore, blocked_work, start_tracked


def _released_work():
    """A work fn that parks until released, plus its started event and its release."""
    started, release = asyncio.Event(), asyncio.Event()

    async def work(progress):
        started.set()
        await release.wait()

    return work, started, release


async def test_every_member_starts_even_under_a_limit_below_the_group_size():
    manager = EvalJobManager(InMemoryRunStore(), max_concurrent=1)
    members = [_released_work() for _ in range(3)]

    await manager.start_group([(make_eval_run(), work, None) for work, _, _ in members])
    await asyncio.wait_for(asyncio.gather(*(started.wait() for _, started, _ in members)), timeout=2)

    for _, _, release in members:
        release.set()
    await manager.shutdown()


async def test_the_group_holds_one_slot_until_its_last_member_ends():
    storage = InMemoryRunStore()
    manager = EvalJobManager(storage, max_concurrent=1)
    first, second = _released_work(), _released_work()
    await manager.start_group([(make_eval_run(), first[0], None), (make_eval_run(), second[0], None)])
    await asyncio.wait_for(asyncio.gather(first[1].wait(), second[1].wait()), timeout=2)
    outsider, outsider_started = blocked_work()
    await manager.start_group([(make_eval_run(), outsider, None)])

    first[2].set()
    await asyncio.sleep(0.05)
    assert not outsider_started.is_set(), "one member ending must not hand the slot away while another runs"

    second[2].set()
    await asyncio.wait_for(outsider_started.wait(), timeout=2)
    await manager.shutdown()


async def test_a_group_queued_behind_a_job_starts_together_when_the_slot_frees():
    manager = EvalJobManager(InMemoryRunStore(), max_concurrent=1)
    blocker, blocker_started, release_blocker = _released_work()
    await manager.start_group([(make_eval_run(), blocker, None)])
    await asyncio.wait_for(blocker_started.wait(), timeout=2)
    members = [_released_work() for _ in range(2)]
    await manager.start_group([(make_eval_run(), work, None) for work, _, _ in members])
    await asyncio.sleep(0.05)
    assert not any(started.is_set() for _, started, _ in members), "the group waits for a slot like any job"

    release_blocker.set()
    await asyncio.wait_for(asyncio.gather(*(started.wait() for _, started, _ in members)), timeout=2)
    for _, _, release in members:
        release.set()
    await manager.shutdown()


async def test_a_group_cancelled_while_queued_takes_no_slot():
    """Nothing was acquired for a group that never started, so the next job is not held behind it."""
    storage = InMemoryRunStore()
    manager = EvalJobManager(storage, max_concurrent=1)
    blocker, blocker_started, release_blocker = _released_work()
    await manager.start_group([(make_eval_run(), blocker, None)])
    await asyncio.wait_for(blocker_started.wait(), timeout=2)
    queued = [make_eval_run(), make_eval_run()]
    member_work = [_released_work() for _ in queued]
    tasks = await start_tracked(
        manager, [(run, work, None) for run, (work, _, _) in zip(queued, member_work, strict=True)]
    )
    await asyncio.sleep(0)
    for run in queued:
        manager.cancel_job(run.id, reason="launch abandoned")
    outcomes = await asyncio.gather(*tasks, return_exceptions=True)
    assert all(isinstance(outcome, asyncio.CancelledError) for outcome in outcomes), outcomes
    release_blocker.set()
    outsider, outsider_started = blocked_work()

    await manager.start_group([(make_eval_run(), outsider, None)])

    await asyncio.wait_for(outsider_started.wait(), timeout=2)
    assert all(storage.runs[run.id].status == "cancelled" for run in queued)
    await manager.shutdown()


async def test_every_member_is_saved_before_any_starts():
    storage = InMemoryRunStore()
    manager = EvalJobManager(storage)
    runs = [make_eval_run(), make_eval_run()]
    seen: list[set[str]] = []

    async def work(progress):
        seen.append(set(storage.runs))

    tasks = await start_tracked(manager, [(run, work, None) for run in runs])
    await asyncio.wait_for(asyncio.gather(*tasks), timeout=2)

    assert all({run.id for run in runs} <= saved for saved in seen)
    await manager.shutdown()


async def test_an_empty_group_is_refused():
    with pytest.raises(ValueError, match="at least one run"):
        await EvalJobManager(InMemoryRunStore()).start_group([])


async def test_a_member_cancelled_before_its_first_step_still_gives_its_share_back():
    """A task cancelled before it runs never executes its body, so release cannot live in the body."""
    manager = EvalJobManager(InMemoryRunStore(), max_concurrent=1)
    runs = [make_eval_run(), make_eval_run()]
    first, first_started, release_first = _released_work()
    second, _, _ = _released_work()
    await manager.start_group([(runs[0], first, None), (runs[1], second, None)])
    assert manager.cancel_job(runs[1].id) is True  # before the loop has given it a single step
    await asyncio.wait_for(first_started.wait(), timeout=2)
    release_first.set()
    outsider, outsider_started = blocked_work()

    await manager.start_group([(make_eval_run(), outsider, None)])

    await asyncio.wait_for(outsider_started.wait(), timeout=2)
    await manager.shutdown()


async def test_a_sibling_that_cannot_be_saved_cancels_the_runs_already_saved():
    """A run left ``pending`` with no task would wait for a slot forever."""

    class _FailingSecondSave(InMemoryRunStore):
        def __init__(self) -> None:
            super().__init__()
            self.saves = 0

        def save_eval_run(self, run, *, if_match=None):
            self.saves += 1
            if self.saves == 2:
                raise OSError("disk full")
            return super().save_eval_run(run, if_match=if_match)

    storage = _FailingSecondSave()
    manager = EvalJobManager(storage)
    runs = [make_eval_run(), make_eval_run()]

    before = asyncio.all_tasks()
    with pytest.raises(OSError):
        await manager.start_group([(run, _released_work()[0], None) for run in runs])

    assert storage.runs[runs[0].id].status == "cancelled"
    assert asyncio.all_tasks() - before == set(), "a job task started for a group that never launched"
    assert manager.get_active_job_ids() == []


async def test_a_cancellation_that_cannot_be_saved_neither_stops_the_rest_nor_masks_the_cause():
    from threetears.evals.contracts.errors import StorageError

    runs = [make_eval_run(), make_eval_run(), make_eval_run()]

    class _Storage(InMemoryRunStore):
        def __init__(self) -> None:
            super().__init__()
            self.saves = 0

        def save_eval_run(self, run, *, if_match=None):
            self.saves += 1
            if self.saves == 3:
                raise StorageError("the third member could not be saved")
            if self.saves == 4:  # the first member's cancellation
                raise StorageError("the cancellation could not be saved")
            return super().save_eval_run(run, if_match=if_match)

    storage = _Storage()
    manager = EvalJobManager(storage)

    before = asyncio.all_tasks()
    with pytest.raises(StorageError, match="third member"):
        await manager.start_group([(run, _released_work()[0], None) for run in runs])

    assert storage.runs[runs[1].id].status == "cancelled"
    assert asyncio.all_tasks() - before == set(), "a job task started for a group that never launched"
    assert manager.get_active_job_ids() == []
