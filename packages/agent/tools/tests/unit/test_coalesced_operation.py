"""A pod's operation coordinated across replicas: one unit at a time, requests coalesced, outcome and progress shared."""

from __future__ import annotations

import asyncio
import json
from datetime import timedelta
from typing import Any

import pytest
from threetears.core.coordination.lease import KVLease
from threetears.core.testing.kv import FakeKvBucket, FakeNatsClient

from threetears.agent.tools.background_operation import OperationStatusTool, RequestOperationTool
from threetears.agent.tools.coalesced_operation import CoalescedOperation, Drained, rendered_drain

#: the scope every replica of one pod keys its leases under, and the operation's lease key
SCOPE = "pod1"
KEY = "enr/refresh"
TTL = timedelta(seconds=60)
RENEW = timedelta(seconds=20)


class _NotReachable(Exception):
    """the tables are out of reach until the hub grants them."""


class _Unit:
    """a unit of work the test can hold, fail and count; each call returns its own number.

    :param held: whether each call waits for ``release``
    :ptype held: bool
    """

    def __init__(self, *, held: bool = False) -> None:
        self.calls = 0
        self.running = 0
        self.most_running = 0
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.fail_with: Exception | None = None
        self.cancelled = 0
        if not held:
            self.release.set()

    async def __call__(self) -> dict[str, int]:
        self.calls += 1
        number = self.calls
        self.running += 1
        self.most_running = max(self.most_running, self.running)
        self.entered.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        finally:
            self.running -= 1
        if self.fail_with is not None:
            raise self.fail_with
        return {"unit": number}


def _lease(nats: FakeNatsClient, replica: str) -> KVLease:
    return KVLease(nats, pod_id=replica, key_scope=SCOPE)  # type: ignore[arg-type]


def _operation(nats: FakeNatsClient, replica: str, unit: Any, **options: Any) -> CoalescedOperation[dict[str, int]]:
    """one replica's operation over the pod's shared leases."""
    return CoalescedOperation("enr-load", _lease(nats, replica), KEY, unit, ttl=TTL, renew_every=RENEW, **options)


def _status_tool(operation: CoalescedOperation[Any]) -> OperationStatusTool:
    return OperationStatusTool(
        name="enr.load_status",
        description="Report the load.",
        operation=lambda: operation.operation,
        render_result=operation.render_result,
    )


async def _entered(unit: _Unit) -> None:
    """wait until a unit is running; one that never does fails the test rather than hang it."""
    await asyncio.wait_for(unit.entered.wait(), timeout=5)


async def _bucket(nats: FakeNatsClient) -> FakeKvBucket:
    return await nats.kv_bucket(name="leases")


async def _stored(nats: FakeNatsClient, suffix: str = "") -> Any:
    """the JSON an entry beside the lease holds; None when there is none."""
    value = await (await _bucket(nats)).get(key=f"{SCOPE}.{KEY}{suffix}")
    return None if value is None else json.loads(value)


async def _progress(nats: FakeNatsClient) -> Any:
    """the progress entry beside the lease, without the holding it names; None when there is none."""
    entry = await _stored(nats, ".progress")
    return None if entry is None else {name: item for name, item in entry.items() if name != "holding"}


# -- one replica -----------------------------------------------------------------------------------


async def test_a_requested_unit_runs_in_the_background_and_leaves_its_own_result() -> None:
    nats, unit = FakeNatsClient(), _Unit(held=True)
    load = _operation(nats, "replica-a", unit)

    assert await load.request()
    await _entered(unit)
    assert load.operation.running and load.operation.state == "running"
    unit.release.set()
    await load.operation.wait()

    last = load.operation.last
    assert last is not None and last.result == Drained(results=({"unit": 1},))
    assert last.result.runs == 1
    assert load.operation.state == "succeeded"
    assert not await load.requested()

    # a later run's result is its own units', not the run before's with them
    assert await load.request()
    await load.operation.wait()
    later = load.operation.last
    assert later is not None and later.result == Drained(results=({"unit": 2},))


async def test_requests_while_a_unit_runs_are_one_more_unit_after_it_and_each_units_result_is_kept() -> None:
    nats, unit = FakeNatsClient(), _Unit(held=True)
    load = _operation(nats, "replica-a", unit)
    reload = RequestOperationTool(
        name="enr.reload", description="Reload.", operation=lambda: load.operation, status_tool="enr.load_status"
    )

    assert (await reload.run()).metadata == {"started": True}
    await _entered(unit)
    for _ in range(3):
        again = await reload.run()
        assert again.success and again.metadata == {"started": False}
    unit.release.set()
    await load.operation.wait()

    assert unit.calls == 2
    last = load.operation.last
    # the run that found the request already taken ran nothing, and does not replace the one that ran
    assert last is not None and last.result == Drained(results=({"unit": 1}, {"unit": 2}))
    assert rendered_drain(last.result) == {"runs": 2, "results": [{"unit": 1}, {"unit": 2}]}


async def test_a_unit_that_fails_leaves_why_and_a_run_with_nothing_left_does_not_hide_it() -> None:
    nats, unit = FakeNatsClient(), _Unit()
    unit.fail_with = ConnectionError("datasource unreachable")
    load = _operation(nats, "replica-a", unit)
    status = _status_tool(load)

    await load.request()
    await load.operation.wait()
    failed = load.operation.last
    assert failed is not None and failed.error == "ConnectionError: datasource unreachable"

    # a run started with no request recorded: it runs no unit
    assert load.operation.start()
    await load.operation.wait()

    assert unit.calls == 1
    assert load.operation.last is failed
    assert load.operation.state == "failed"
    answer = await status.run()
    assert answer.metadata is not None
    assert answer.metadata["state"] == "failed"
    assert answer.metadata["last"]["error"] == "ConnectionError: datasource unreachable"
    recorded = await _stored(nats, ".outcome")
    assert recorded["error"] == "ConnectionError: datasource unreachable"


async def test_the_first_run_waits_until_the_tables_answer_and_runs_only_when_needed() -> None:
    nats, unit = FakeNatsClient(), _Unit()
    load = _operation(nats, "replica-a", unit)
    answers: list[Any] = [_NotReachable("not granted yet"), _NotReachable("not granted yet"), True]

    async def needed() -> bool:
        answer = answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return bool(answer)

    assert load.start_when_needed(needed, retry_on=(_NotReachable,), retry_seconds=0)
    await load.operation.wait_until_settled()

    assert answers == [] and unit.calls == 1
    last = load.operation.last
    assert last is not None and last.result == Drained(results=({"unit": 1},))

    idle_unit = _Unit()
    idle = _operation(FakeNatsClient(), "replica-a", idle_unit)

    async def not_needed() -> bool:
        return False

    idle.start_when_needed(not_needed, retry_on=(_NotReachable,), retry_seconds=0)
    await idle.operation.wait_until_settled()
    assert idle_unit.calls == 0 and idle.operation.state == "idle"
    assert not await idle.requested()


async def test_a_request_while_the_first_run_waits_for_the_tables_is_the_only_unit() -> None:
    nats, unit = FakeNatsClient(), _Unit(held=True)
    load = _operation(nats, "replica-a", unit)
    reachable = asyncio.Event()

    async def needed() -> bool:
        if not reachable.is_set():
            raise _NotReachable("not granted yet")
        return True

    load.start_when_needed(needed, retry_on=(_NotReachable,), retry_seconds=0.01)
    assert await load.request()
    await _entered(unit)
    # the tables answer while the requested run is in progress: it is the first run, so no second request
    reachable.set()
    await asyncio.sleep(0.05)
    unit.release.set()
    await load.operation.wait_until_settled()

    assert unit.calls == 1
    assert not await load.requested()


# -- stop ------------------------------------------------------------------------------------------


async def test_stopping_cancels_the_unit_leaves_its_request_and_the_next_replica_to_start_runs_it() -> None:
    nats, unit = FakeNatsClient(), _Unit(held=True)
    load = _operation(nats, "replica-a", unit)
    await load.request()
    await _entered(unit)

    assert await load.stop() is True

    assert load.stopping and unit.cancelled == 1
    assert not load.operation.running
    assert await load.requested()
    assert await _lease(nats, "observer").holder(KEY) is None

    # the next replica finds every table loaded, and the request the stopped unit left
    next_unit = _Unit()
    following = _operation(nats, "replica-b", next_unit)

    async def not_needed() -> bool:
        return False

    following.start_when_needed(not_needed, retry_on=(_NotReachable,), retry_seconds=0)
    await following.operation.wait_until_settled()
    assert next_unit.calls == 1
    assert not await following.requested()


async def test_a_run_started_after_the_stop_takes_neither_the_lease_nor_the_request() -> None:
    nats, unit = FakeNatsClient(), _Unit()
    load = _operation(nats, "replica-a", unit)
    assert await load.stop()

    assert await load.request()
    await load.operation.wait()

    assert unit.calls == 0
    # the run that ran nothing is not a run: this replica has none to report
    assert load.operation.last is None and load.operation.state == "idle"
    assert await load.requested()
    assert await _stored(nats) is None


async def test_a_stop_given_a_bound_says_whether_everything_ended_and_goes_on_past_it() -> None:
    nats = FakeNatsClient()
    let_go = asyncio.Event()
    entered = asyncio.Event()

    async def slow_to_end() -> dict[str, int]:
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            # work that cannot be dropped mid-write: the lease is not let go until it has ended
            await asyncio.shield(let_go.wait())
        return {"unit": 1}

    load = _operation(nats, "replica-a", slow_to_end)
    await load.request()
    await asyncio.wait_for(entered.wait(), timeout=5)

    assert await load.stop(within=0.05) is False
    # the unit is still ending, so the lease is still held: no other replica starts under it
    assert load.operation.running
    assert await _lease(nats, "observer").holder(KEY) == "replica-a"

    let_go.set()
    assert await load.stop(within=5) is True
    assert not load.operation.running
    assert await _lease(nats, "observer").holder(KEY) is None
    assert await load.requested()


async def test_stopping_cancels_the_wait_for_the_tables_and_the_wait_to_try_a_held_lease_again() -> None:
    nats, unit = FakeNatsClient(), _Unit()
    held = await _lease(nats, "other-replica").acquire(KEY, ttl_seconds=60)
    load = _operation(nats, "replica-a", unit, lease_retry_seconds=0.05)

    async def never_reachable() -> bool:
        raise _NotReachable("not granted yet")

    load.start_when_needed(never_reachable, retry_on=(_NotReachable,), retry_seconds=0.01)
    assert await load.request()
    await load.operation.wait()
    assert await load.stop() is True

    # the lease is free and the request still recorded: only a wait that outlived the stop would run
    await held.release()
    assert await load.requested()
    assert not [task for task in asyncio.all_tasks() if task.get_name().startswith("enr-load")]
    await asyncio.sleep(0.3)
    assert unit.calls == 0 and not load.operation.running


async def test_a_wait_to_try_a_held_lease_again_that_ends_while_the_pod_is_stopping_starts_nothing() -> None:
    nats, unit = FakeNatsClient(), _Unit()
    held = await _lease(nats, "other-replica").acquire(KEY, ttl_seconds=60)
    load = _operation(nats, "replica-a", unit, lease_retry_seconds=0.05)
    await load.request()
    await load.operation.wait()
    # the stop is held up before it reaches the wait (as it is while a unit is still ending)
    stopped = asyncio.Event()
    real_stop = load.operation.stop

    async def slow_stop() -> None:
        await stopped.wait()
        await real_stop()

    load.operation.stop = slow_stop  # type: ignore[method-assign]

    assert await load.stop(within=0.01) is False
    await held.release()
    await asyncio.sleep(0.3)

    assert unit.calls == 0 and await load.requested()
    stopped.set()
    assert await load.stop() is True


async def test_a_run_that_found_the_lease_held_looks_again_after_the_leases_lifetime_unless_told_another_wait() -> None:
    nats, unit = FakeNatsClient(), _Unit()
    held = await _lease(nats, "other-replica").acquire(KEY, ttl_seconds=60)
    load: CoalescedOperation[dict[str, int]] = CoalescedOperation(
        "enr-load",
        _lease(nats, "replica-a"),
        KEY,
        unit,
        ttl=timedelta(seconds=1),
        renew_every=timedelta(seconds=0.3),
    )
    await load.request()
    await load.operation.wait()
    await held.release()

    await asyncio.sleep(0.5)
    assert unit.calls == 0
    await asyncio.wait_for(unit.entered.wait(), timeout=2)
    await load.operation.wait()
    assert unit.calls == 1
    await load.stop()


# -- two replicas on one lease ---------------------------------------------------------------------


async def test_one_unit_runs_at_a_time_across_replicas_and_a_request_on_the_other_is_run_once_after() -> None:
    nats, unit = FakeNatsClient(), _Unit(held=True)
    holder = _operation(nats, "replica-a", unit)
    waiting = _operation(nats, "replica-b", unit, lease_retry_seconds=0.2)

    await holder.request()
    await _entered(unit)
    await waiting.request()
    await waiting.request()
    await waiting.operation.wait()
    assert waiting.operation.last is None and waiting.operation.state == "idle"
    assert await waiting.requested()

    unit.release.set()
    await holder.operation.wait()

    assert unit.calls == 2 and unit.most_running == 1
    last = holder.operation.last
    assert last is not None and last.result == Drained(results=({"unit": 1}, {"unit": 2}))
    assert not await holder.requested()
    await waiting.stop()


class _FirstLoad:
    """a first load: needed until one has ended well, as a table is empty until it is loaded."""

    def __init__(self) -> None:
        self.loaded = False
        self.calls = 0
        self.running = 0
        self.most_running = 0
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.failures = 0

    async def needed(self) -> bool:
        await asyncio.sleep(0)
        return not self.loaded

    async def __call__(self) -> dict[str, int]:
        self.calls += 1
        number = self.calls
        self.running += 1
        self.most_running = max(self.most_running, self.running)
        self.entered.set()
        try:
            await self.release.wait()
        finally:
            self.running -= 1
        if self.failures:
            self.failures -= 1
            raise ConnectionError("datasource unreachable")
        self.loaded = True
        return {"unit": number}


@pytest.mark.parametrize("together", [True, False])
async def test_replicas_that_start_needing_a_first_run_run_exactly_one_unit_and_never_two_at_once(
    together: bool,
) -> None:
    nats, load = FakeNatsClient(), _FirstLoad()
    replicas = [_operation(nats, f"replica-{name}", load, lease_retry_seconds=0.05) for name in "abc"]

    replicas[0].start_when_needed(load.needed, retry_on=(_NotReachable,), retry_seconds=0)
    if not together:
        # the others start while the first load runs: their tables are still empty
        await asyncio.wait_for(load.entered.wait(), timeout=5)
    for replica in replicas[1:]:
        replica.start_when_needed(load.needed, retry_on=(_NotReachable,), retry_seconds=0)
    await asyncio.wait_for(load.entered.wait(), timeout=5)
    await asyncio.sleep(0.15)
    load.release.set()
    for replica in replicas:
        await asyncio.wait_for(replica.operation.wait_until_settled(), timeout=5)
    await asyncio.sleep(0.2)

    assert load.calls == 1, "a replica that started beside the first load asked for another"
    assert load.most_running == 1
    assert not await replicas[0].requested()
    for replica in replicas:
        assert not replica.operation.running
        await replica.stop()


async def test_a_replica_that_waited_on_anothers_first_run_runs_it_itself_when_that_run_failed() -> None:
    nats, load = FakeNatsClient(), _FirstLoad()
    load.failures = 1
    first = _operation(nats, "replica-a", load, lease_retry_seconds=0.05)
    waiting = _operation(nats, "replica-b", load, lease_retry_seconds=0.05)

    first.start_when_needed(load.needed, retry_on=(_NotReachable,), retry_seconds=0)
    await asyncio.wait_for(load.entered.wait(), timeout=5)
    waiting.start_when_needed(load.needed, retry_on=(_NotReachable,), retry_seconds=0)
    await asyncio.sleep(0.15)
    # it recorded nothing and ran nothing while the first run was in progress
    assert load.calls == 1 and not await waiting.requested() and not waiting.operation.running

    load.release.set()
    await first.operation.wait_until_settled()
    assert first.operation.state == "failed"
    # no live holder remains and the tables are still empty: the first run is still needed, and is run
    await asyncio.wait_for(waiting.operation.wait_until_settled(), timeout=5)

    assert load.calls == 2 and load.loaded
    last = waiting.operation.last
    assert last is not None and last.result == Drained(results=({"unit": 2},))
    await first.stop()
    await waiting.stop()


async def test_a_bucket_that_cannot_be_reached_while_the_first_run_is_decided_does_not_end_the_wait(
    caplog: pytest.LogCaptureFixture,
) -> None:
    nats, load = FakeNatsClient(), _FirstLoad()
    load.release.set()
    operation = _operation(nats, "replica-a", load)
    bucket = await _bucket(nats)
    real_put = bucket.put
    refused = 0

    async def refusing_once(**kwargs: Any) -> int:
        nonlocal refused
        if kwargs["key"].endswith(".requested") and not refused:
            refused += 1
            raise TimeoutError("the put was not answered")
        return await real_put(**kwargs)

    bucket.put = refusing_once  # type: ignore[method-assign]

    # the pod names the errors of its own check; it does not know the bucket's
    operation.start_when_needed(load.needed, retry_on=(_NotReachable,), retry_seconds=0.01)
    await asyncio.wait_for(operation.operation.wait_until_settled(), timeout=5)

    assert refused == 1 and load.calls == 1
    assert operation.operation.state == "succeeded"
    assert any("the put was not answered" in str(getattr(record, "extra_data", "")) for record in caplog.records)

    # an error of the pod's own check that it did not name still ends the wait, and is reported
    async def broken() -> bool:
        raise ValueError("the check itself is broken")

    other = _operation(FakeNatsClient(), "replica-a", _Unit())
    other.start_when_needed(broken, retry_on=(_NotReachable,), retry_seconds=0.01)
    with pytest.raises(ValueError, match="the check itself is broken"):
        await other.operation.wait_until_settled()


async def test_a_unit_whose_lease_is_lost_is_run_again_by_the_replica_that_lost_it_with_nobody_asking() -> None:
    nats, unit = FakeNatsClient(), _Unit(held=True)
    load: CoalescedOperation[dict[str, int]] = CoalescedOperation(
        "enr-load",
        _lease(nats, "replica-a"),
        KEY,
        unit,
        ttl=timedelta(seconds=1),
        renew_every=timedelta(seconds=0.05),
        lease_retry_seconds=0.2,
    )
    await load.request()
    await _entered(unit)
    bucket = await _bucket(nats)
    entry = await bucket.get_entry(key=f"{SCOPE}.{KEY}")
    assert entry is not None
    # the lease is taken from under the unit (its renewals did not arrive in time): the next finds it gone
    stolen = entry[0].replace(b"replica-a", b"replica-z")
    assert await bucket.update(key=f"{SCOPE}.{KEY}", value=stolen, revision=entry[1]) is not None
    await asyncio.wait_for(load.operation.wait(), timeout=5)

    lost = load.operation.last
    assert lost is not None and lost.error is not None and lost.error.startswith("LeaseLost")
    assert unit.cancelled == 1 and await load.requested()

    # no replica is draining and nobody asks again: once the lease that took it has lapsed, this one runs it
    unit.entered.clear()
    unit.release.set()
    await asyncio.wait_for(unit.entered.wait(), timeout=5)
    await load.operation.wait()
    assert unit.calls == 2
    last = load.operation.last
    assert last is not None and last.result == Drained(results=({"unit": 2},))
    assert not await load.requested()
    await load.stop()


async def test_every_replica_reports_the_newest_finished_run_and_says_whose_it_was() -> None:
    nats, unit = FakeNatsClient(), _Unit()
    ran = _operation(nats, "replica-a", unit)
    idle = _operation(nats, "replica-b", _Unit())

    await ran.request()
    await ran.operation.wait()
    assert ran.operation.last is not None
    here = await _status_tool(ran).run()
    elsewhere = await _status_tool(idle).run()

    assert here.metadata is not None and elsewhere.metadata is not None
    assert here.metadata["last"]["result"] == {"runs": 1, "results": [{"unit": 1}]}
    assert elsewhere.metadata["last"] == {**here.metadata["last"], "other_replica": True}
    assert elsewhere.metadata["last"]["replica"] == "replica-a"
    assert "the last run, on another replica (replica-a), succeeded at" in elsewhere.content
    assert (await _stored(nats, ".outcome"))["result"] == {"runs": 1, "results": [{"unit": 1}]}


async def test_a_units_result_is_recorded_as_the_pods_own_render_gives_it() -> None:
    nats, unit = FakeNatsClient(), _Unit()

    def render(drained: Drained[dict[str, int]]) -> dict[str, Any]:
        return {"loads": drained.runs, "last": drained.results[-1]["unit"] if drained.results else None}

    ran = _operation(nats, "replica-a", unit, render_result=render)
    idle = _operation(nats, "replica-b", _Unit(), render_result=render)
    await ran.request()
    await ran.operation.wait()

    assert ran.render_result is render
    answer = await _status_tool(idle).run()
    assert answer.metadata is not None
    assert answer.metadata["last"]["result"] == {"loads": 1, "last": 1}


async def test_runs_that_ran_nothing_while_a_holder_runs_are_never_recorded_and_hide_no_real_run() -> None:
    nats = FakeNatsClient()
    first, unit, waiting_unit = _Unit(), _Unit(held=True), _Unit()
    earlier = _operation(nats, "replica-a", first)
    first.fail_with = RuntimeError("part way")
    await earlier.request()
    await earlier.operation.wait()
    failed = await _stored(nats, ".outcome")
    assert failed["error"] == "RuntimeError: part way"

    holder = _operation(nats, "replica-a2", unit)
    waiting = _operation(nats, "replica-b", waiting_unit, lease_retry_seconds=0.05)
    await holder.request()
    await _entered(unit)
    for _ in range(3):
        await waiting.request()
        await waiting.operation.wait()

    # three runs on the waiting replica ran nothing: the record is the failed run still, on every replica
    assert waiting_unit.calls == 0
    # and it waits once to look again, not once for each
    assert len([task for task in asyncio.all_tasks() if task.get_name() == "enr-load-retry"]) == 1
    assert await _stored(nats, ".outcome") == failed
    answer = await _status_tool(waiting).run()
    assert answer.metadata is not None
    assert answer.metadata["last"]["error"] == "RuntimeError: part way"
    assert answer.metadata["last"]["other_replica"] is True

    unit.release.set()
    await holder.operation.wait()
    await waiting.stop()


# -- the request left behind a holder --------------------------------------------------------------


async def test_a_request_left_behind_a_dead_holder_is_run_once_its_lease_has_lapsed_with_nobody_asking_again() -> None:
    nats, unit = FakeNatsClient(), _Unit()
    told: list[str] = []
    await _lease(nats, "dead-replica").acquire(KEY, ttl_seconds=1)
    load = _operation(nats, "replica-a", unit, lease_retry_seconds=0.4, ran_nothing=lambda: told.append("nothing"))

    await load.request()
    await load.operation.wait()

    assert load.operation.last is None and load.operation.state == "idle"
    assert unit.calls == 0 and told == ["nothing"]
    assert await load.requested()

    # the dead holder's lease says when it lapses; nothing here ends it sooner, and nobody asks again
    await _entered(unit)
    await load.operation.wait()
    assert unit.calls == 1
    last = load.operation.last
    assert last is not None and last.result == Drained(results=({"unit": 1},))
    assert not await load.requested()
    await load.stop()


async def test_a_replica_that_waited_on_a_live_holder_starts_nothing_while_it_runs_or_after_it_took_the_request() -> (
    None
):
    nats, unit, waiting_unit = FakeNatsClient(), _Unit(held=True), _Unit()
    told: list[str] = []
    holder = _operation(nats, "replica-a", unit)
    waiting = _operation(
        nats, "replica-b", waiting_unit, lease_retry_seconds=0.05, ran_nothing=lambda: told.append("nothing")
    )
    await holder.request()
    await _entered(unit)
    await waiting.request()
    await waiting.operation.wait()
    assert waiting.operation.last is None
    assert told == ["nothing"]

    # many lifetimes of the wait pass while the holder is alive: the waiting replica starts no run
    starts: list[bool] = []
    real_start = waiting.operation.start
    waiting.operation.start = lambda: starts.append(True) or real_start()  # type: ignore[method-assign]
    await asyncio.sleep(0.3)
    assert starts == [] and waiting.operation.last is None

    unit.release.set()
    await holder.operation.wait()
    assert unit.calls == 2
    # past the wait: the holder took the request, so the waiting replica finds none and starts nothing
    await asyncio.sleep(0.3)
    assert starts == [] and waiting_unit.calls == 0
    assert waiting.operation.last is None and not waiting.operation.running
    assert told == ["nothing", "nothing"]
    await waiting.stop()


async def test_a_bucket_that_cannot_be_asked_does_not_end_the_wait_to_try_a_held_lease_again(
    caplog: pytest.LogCaptureFixture,
) -> None:
    nats, unit = FakeNatsClient(), _Unit()
    held = await _lease(nats, "other-replica").acquire(KEY, ttl_seconds=60)
    load = _operation(nats, "replica-a", unit, lease_retry_seconds=0.05)
    await load.request()
    await load.operation.wait()
    bucket = await _bucket(nats)

    bucket.become_unreachable(ConnectionError("the broker is away"))
    await asyncio.sleep(0.2)
    assert unit.calls == 0
    bucket.become_reachable()
    await held.release()

    await _entered(unit)
    await load.operation.wait()
    assert unit.calls == 1
    assert any("asking again" in record.message for record in caplog.records)
    await load.stop()


async def test_a_ran_nothing_callback_that_raises_ends_neither_the_run_nor_the_wait(
    caplog: pytest.LogCaptureFixture,
) -> None:
    nats, unit = FakeNatsClient(), _Unit()
    held = await _lease(nats, "other-replica").acquire(KEY, ttl_seconds=60)

    def broken() -> None:
        raise LookupError("no copy yet")

    load = _operation(nats, "replica-a", unit, lease_retry_seconds=0.05, ran_nothing=broken)
    await load.request()
    await load.operation.wait()

    assert load.operation.state == "idle" and not load.operation.running
    await held.release()
    await _entered(unit)
    await load.operation.wait()
    assert unit.calls == 1
    assert any("no copy yet" in str(getattr(record, "extra_data", "")) for record in caplog.records)
    await load.stop()


# -- the keys --------------------------------------------------------------------------------------


async def test_the_lease_the_request_the_outcome_and_the_progress_are_keyed_beside_each_other_under_the_scope() -> None:
    nats, unit = FakeNatsClient(), _Unit(held=True)
    load = _operation(nats, "replica-a", unit)
    await load.request()
    await _entered(unit)
    load.report_progress({"summary": "Loading"})
    await asyncio.sleep(0.01)
    await load.request()

    bucket = await _bucket(nats)
    assert set(bucket.keys()) == {
        f"{SCOPE}.{KEY}",
        f"{SCOPE}.{KEY}.requested",
        f"{SCOPE}.{KEY}.progress",
    }
    unit.release.set()
    await load.operation.wait()
    assert f"{SCOPE}.{KEY}.outcome" in set(bucket.keys())
    assert (await _stored(nats, ".outcome"))["replica"] == "replica-a"


# -- progress --------------------------------------------------------------------------------------


async def test_a_run_in_progress_is_shown_by_the_other_replica_with_its_progress_and_gone_when_it_ends() -> None:
    nats, unit = FakeNatsClient(), _Unit(held=True)
    holder = _operation(nats, "replica-a", unit)
    idle = _operation(nats, "replica-b", _Unit())
    idle_status, holder_status = _status_tool(idle), _status_tool(holder)

    before = await idle_status.run()
    assert before.metadata is not None
    assert before.metadata["state"] == "idle" and before.metadata["elsewhere"] is None

    await holder.request()
    await _entered(unit)
    started = await idle_status.run()
    assert started.metadata is not None
    assert started.metadata["state"] == "running" and started.metadata["running"] is True
    assert started.metadata["started_at"] is None
    assert started.metadata["elsewhere"] == {"replica": "replica-a", "progress": {}}
    assert started.content == "running on another replica (replica-a)"

    holder.report_progress({"summary": "Loading · 37 of 51 states", "states": {"left": ["VA"]}})
    await asyncio.sleep(0.01)
    moving = await idle_status.run()
    assert moving.metadata is not None
    assert moving.metadata["elsewhere"] == {
        "replica": "replica-a",
        "progress": {"summary": "Loading · 37 of 51 states", "states": {"left": ["VA"]}},
    }
    assert moving.content == "running on another replica (replica-a): Loading · 37 of 51 states"
    assert await idle.running_elsewhere() == await idle.operation.running_elsewhere()

    # the replica that runs it reports its own run, and asks nobody
    own = await holder_status.run()
    assert own.metadata is not None
    assert own.metadata["elsewhere"] is None and own.content.startswith("running since ")
    assert await holder.running_elsewhere() is None

    unit.release.set()
    await holder.operation.wait()
    after = await idle_status.run()
    assert after.metadata is not None
    assert after.metadata["state"] == "succeeded" and after.metadata["running"] is False
    assert after.metadata["elsewhere"] is None
    assert await _progress(nats) is None
    assert "running" not in after.content


async def test_only_the_latest_progress_is_written_and_none_outside_a_unit_running_here() -> None:
    nats, unit = FakeNatsClient(), _Unit(held=True)
    load = _operation(nats, "replica-a", unit)
    bucket = await _bucket(nats)
    puts: list[dict[str, Any]] = []
    real_put = bucket.put

    async def counted(**kwargs: Any) -> int:
        if kwargs["key"].endswith(".progress"):
            puts.append({name: item for name, item in json.loads(kwargs["value"]).items() if name != "holding"})
        return await real_put(**kwargs)

    bucket.put = counted  # type: ignore[method-assign]

    load.report_progress({"summary": "before any unit"})
    await asyncio.sleep(0.01)
    assert puts == []

    await load.request()
    await _entered(unit)
    await asyncio.sleep(0.01)
    for done in range(50):
        load.report_progress({"summary": f"{done} done"})
    await asyncio.sleep(0.05)

    # the unit's first entry, then the first reported and the latest: never one write per report
    assert puts[0] == {"holder": "replica-a"}
    assert puts[-1] == {"summary": "49 done", "holder": "replica-a"}
    assert len(puts) <= 3

    # a report made while a write is in flight is written after it
    during = {"reported": False}
    plain_put = bucket.put

    async def reporting(**kwargs: Any) -> int:
        if kwargs["key"].endswith(".progress") and not during["reported"]:
            during["reported"] = True
            load.report_progress({"summary": "reported during a write"})
        return await plain_put(**kwargs)

    bucket.put = reporting  # type: ignore[method-assign]
    load.report_progress({"summary": "a write in flight"})
    await asyncio.sleep(0.05)
    assert (await _progress(nats))["summary"] == "reported during a write"

    unit.release.set()
    await load.operation.wait()
    written = len(puts)
    load.report_progress({"summary": "after the unit"})
    await asyncio.sleep(0.01)
    assert len(puts) == written
    assert await _progress(nats) is None


async def test_a_unit_whose_progress_cannot_be_shared_says_so_and_finishes(caplog: pytest.LogCaptureFixture) -> None:
    nats, unit = FakeNatsClient(), _Unit(held=True)
    load = _operation(nats, "replica-a", unit)
    bucket = await _bucket(nats)
    real_put = bucket.put

    async def refusing(**kwargs: Any) -> int:
        if kwargs["key"].endswith(".progress"):
            raise ConnectionError("the progress write was refused")
        return await real_put(**kwargs)

    bucket.put = refusing  # type: ignore[method-assign]
    await load.request()
    await _entered(unit)
    load.report_progress({"summary": "Loading"})
    load.report_progress({"summary": object()})
    await asyncio.sleep(0.01)
    unit.release.set()
    await load.operation.wait()

    last = load.operation.last
    assert last is not None and last.result == Drained(results=({"unit": 1},))
    assert any(
        "progress could not be shared" in record.message
        and "the progress write was refused" in str(getattr(record, "extra_data", ""))
        for record in caplog.records
    )


async def test_stopping_starts_no_progress_write_after_it() -> None:
    nats = FakeNatsClient()
    load: CoalescedOperation[dict[str, int]]

    async def reports_as_it_is_stopped() -> dict[str, int]:
        try:
            await asyncio.Event().wait()
        finally:
            # the unit goes on a moment as it is stopped, and reports: a write would have time to land
            load.report_progress({"summary": "stopping"})
            await asyncio.sleep(0.02)
        return {"unit": 1}

    load = _operation(nats, "replica-a", reports_as_it_is_stopped)
    await load.request()
    await asyncio.sleep(0.01)
    load.report_progress({"summary": "Loading"})
    await asyncio.sleep(0.01)
    assert (await _progress(nats))["summary"] == "Loading"
    bucket = await _bucket(nats)
    written: list[bytes] = []
    real_put = bucket.put

    async def counted(**kwargs: Any) -> int:
        written.append(kwargs["value"])
        return await real_put(**kwargs)

    bucket.put = counted  # type: ignore[method-assign]

    assert await load.stop() is True

    # the unit reported as it was stopped: nothing of it was written
    assert not [value for value in written if b"stopping" in value]
    assert await _progress(nats) is None
    load.report_progress({"summary": "after the stop"})
    await asyncio.sleep(0.01)
    assert await _progress(nats) is None


async def test_a_dead_holders_progress_is_shown_while_its_lease_names_it_and_never_after() -> None:
    nats = FakeNatsClient()
    reader = _operation(nats, "replica-b", _Unit())
    bucket = await _bucket(nats)
    progress = {"summary": "Loading · 12 of 51 states", "holder": "dead-replica"}
    await bucket.put(key=f"{SCOPE}.{KEY}.progress", value=json.dumps(progress).encode())

    # no lease: an entry alone is nobody's run, however lately it was written
    assert await reader.running_elsewhere() is None

    await _lease(nats, "dead-replica").acquire(KEY, ttl_seconds=1)
    shown = await reader.running_elsewhere()
    assert shown is not None and shown.replica == "dead-replica"
    assert shown.progress == {"summary": "Loading · 12 of 51 states"}

    await asyncio.sleep(1.1)
    # the lease has lapsed: the entry is still there, and counts for nothing
    assert await bucket.get(key=f"{SCOPE}.{KEY}.progress") is not None
    assert await reader.running_elsewhere() is None
    answer = await _status_tool(reader).run()
    assert answer.metadata is not None
    assert answer.metadata["state"] == "idle" and answer.metadata["elsewhere"] is None

    # the next holder's run is never shown with the dead holder's progress
    await _lease(nats, "replica-c").acquire(KEY, ttl_seconds=30)
    taken = await reader.running_elsewhere()
    assert taken is not None and taken.replica == "replica-c" and taken.progress is None


async def test_progress_written_by_code_that_does_not_name_its_writer_is_the_lease_holders() -> None:
    nats = FakeNatsClient()
    reader = _operation(nats, "replica-b", _Unit())
    bucket = await _bucket(nats)
    older = {"phase": "reading", "summary": "Loading · 3 of 51 states", "written_at": 1.0, "writer": "abc"}
    await bucket.put(key=f"{SCOPE}.{KEY}.progress", value=json.dumps(older).encode())
    await _lease(nats, "older-replica").acquire(KEY, ttl_seconds=30)

    shown = await reader.running_elsewhere()

    assert shown is not None and shown.replica == "older-replica"
    assert shown.progress == older


async def test_a_lease_or_an_entry_that_cannot_be_read_leaves_the_status_answering(
    caplog: pytest.LogCaptureFixture,
) -> None:
    nats, unit = FakeNatsClient(), _Unit()
    load = _operation(nats, "replica-a", unit)
    await load.request()
    await load.operation.wait()
    bucket = await _bucket(nats)
    await _lease(nats, "replica-b").acquire(KEY, ttl_seconds=30)
    await bucket.put(key=f"{SCOPE}.{KEY}.progress", value=b"[not, an, object")

    unreadable = await _status_tool(load).run()
    assert unreadable.metadata is not None
    assert unreadable.metadata["elsewhere"] == {"replica": "replica-b", "progress": None}

    bucket.become_unreachable(TimeoutError("no answer from the broker"))
    answer = await _status_tool(load).run()

    assert answer.success and answer.metadata is not None
    assert answer.metadata["state"] == "succeeded" and answer.metadata["elsewhere"] is None
    assert answer.metadata["last"]["result"] == {"runs": 1, "results": [{"unit": 1}]}
    assert any("cannot tell whether another replica runs" in record.message for record in caplog.records)


# -- failures that must change nothing -------------------------------------------------------------


async def test_a_stop_that_fails_says_it_did_not_finish_and_raises_nothing(caplog: pytest.LogCaptureFixture) -> None:
    load = _operation(FakeNatsClient(), "replica-a", _Unit())

    async def failing_stop() -> None:
        raise RuntimeError("the task would not end")

    load.operation.stop = failing_stop  # type: ignore[method-assign]

    assert await load.stop() is False

    assert load.stopping
    assert any(
        "failed while stopping" in record.message and "the task would not end" in str(getattr(record, "extra_data", ""))
        for record in caplog.records
    )


async def test_a_run_that_cannot_tell_whether_its_request_was_left_looks_again_as_if_it_was() -> None:
    nats, unit = FakeNatsClient(), _Unit()
    held = await _lease(nats, "other-replica").acquire(KEY, ttl_seconds=60)
    load = _operation(nats, "replica-a", unit, lease_retry_seconds=0.05)
    bucket = await _bucket(nats)
    real_get_entry = bucket.get_entry
    asked = 0

    async def second_ask_fails(**kwargs: Any) -> tuple[bytes, int] | None:
        nonlocal asked
        if kwargs["key"].endswith(".requested"):
            asked += 1
            if asked == 2:
                raise ConnectionError("the broker is away")
        return await real_get_entry(**kwargs)

    bucket.get_entry = second_ask_fails  # type: ignore[method-assign]
    await load.request()
    await load.operation.wait()

    assert asked == 2 and load.operation.state == "idle"
    await held.release()
    await _entered(unit)
    await load.operation.wait()
    assert unit.calls == 1
    await load.stop()


async def test_progress_that_cannot_be_removed_when_the_unit_ends_changes_nothing_of_how_it_ended(
    caplog: pytest.LogCaptureFixture,
) -> None:
    nats, unit = FakeNatsClient(), _Unit()
    load = _operation(nats, "replica-a", unit)
    bucket = await _bucket(nats)
    real_delete = bucket.delete

    async def refusing(**kwargs: Any) -> bool:
        if kwargs["key"].endswith(".progress"):
            raise ConnectionError("the delete was refused")
        return await real_delete(**kwargs)

    bucket.delete = refusing  # type: ignore[method-assign]
    await load.request()
    await load.operation.wait()

    last = load.operation.last
    assert last is not None and last.result == Drained(results=({"unit": 1},))
    assert any("progress could not be removed" in record.message for record in caplog.records)
    # the entry it left counts for nothing: the lease is let go
    assert await _progress(nats) == {"holder": "replica-a"}
    assert await _operation(nats, "replica-b", _Unit()).running_elsewhere() is None


# -- the same line whichever replica answers, said once --------------------------------------------


async def test_the_replica_that_runs_says_how_far_it_is_as_the_other_replicas_do() -> None:
    nats, unit = FakeNatsClient(), _Unit(held=True)
    holder = _operation(nats, "replica-a", unit)
    idle = _operation(nats, "replica-b", _Unit())
    holder_status, idle_status = _status_tool(holder), _status_tool(idle)

    await holder.request()
    await _entered(unit)
    started = await holder_status.run()
    assert started.metadata is not None
    # nothing reported yet: the run in progress has an empty progress, here as there
    assert started.metadata["run_progress"] == {} and started.metadata["elsewhere"] is None
    assert started.content == f"running since {started.metadata['started_at']}"
    assert (await idle_status.run()).metadata["run_progress"] == {}  # type: ignore[index]

    holder.report_progress({"summary": "Loading · 37 of 51 states", "states": {"left": ["VA"]}})
    await asyncio.sleep(0.01)
    here = await holder_status.run()
    there = await idle_status.run()

    assert here.metadata is not None and there.metadata is not None
    progress = {"summary": "Loading · 37 of 51 states", "states": {"left": ["VA"]}}
    assert here.metadata["run_progress"] == progress == there.metadata["run_progress"]
    assert "progress" not in here.metadata
    assert here.content == f"running since {here.metadata['started_at']}: Loading · 37 of 51 states"
    assert there.content == "running on another replica (replica-a): Loading · 37 of 51 states"

    unit.release.set()
    await holder.operation.wait()
    after = await holder_status.run()
    assert after.metadata is not None and after.metadata["run_progress"] is None
    assert (await idle_status.run()).metadata["run_progress"] is None  # type: ignore[index]


async def test_the_sentence_says_the_runs_line_once_and_the_pods_own_line_only_when_it_is_another() -> None:
    nats, unit = FakeNatsClient(), _Unit(held=True)
    holder = _operation(nats, "replica-a", unit)
    idle = _operation(nats, "replica-b", _Unit())
    line = {"summary": "Loading · 1 of 2 states"}

    def tool(operation: CoalescedOperation[Any]) -> OperationStatusTool:
        return OperationStatusTool(
            name="enr.load_status",
            description="Report the load.",
            operation=lambda: operation.operation,
            render_result=operation.render_result,
            progress=lambda: dict(line),
        )

    await holder.request()
    await _entered(unit)
    holder.report_progress({"summary": "Loading · 1 of 2 states"})
    await asyncio.sleep(0.01)

    here, there = await tool(holder).run(), await tool(idle).run()
    assert here.metadata is not None and there.metadata is not None
    # the pod's own progress says the line the run already said: it is not said again
    assert there.content == "running on another replica (replica-a): Loading · 1 of 2 states"
    assert here.content == f"running since {here.metadata['started_at']}: Loading · 1 of 2 states"
    assert here.metadata["progress"] == line and there.metadata["progress"] == line

    # a line of the pod's own that is not the run's is said after it
    line["summary"] = "Ready"
    here, there = await tool(holder).run(), await tool(idle).run()
    assert here.metadata is not None
    assert there.content == "running on another replica (replica-a): Loading · 1 of 2 states; now: Ready"
    assert here.content == f"running since {here.metadata['started_at']}: Loading · 1 of 2 states; now: Ready"

    unit.release.set()
    await holder.operation.wait()
    assert (await tool(idle).run()).content.startswith("succeeded; now: Ready; the last run, on another replica")


async def test_one_status_call_reads_the_lease_and_the_progress_once() -> None:
    nats, unit = FakeNatsClient(), _Unit(held=True)
    holder = _operation(nats, "replica-a", unit)
    idle = _operation(nats, "replica-b", _Unit())
    await holder.request()
    await _entered(unit)
    holder.report_progress({"summary": "Loading"})
    await asyncio.sleep(0.01)
    bucket = await _bucket(nats)
    reads: list[str] = []
    real_get = bucket.get

    async def counted(**kwargs: Any) -> bytes | None:
        reads.append(kwargs["key"])
        return await real_get(**kwargs)

    bucket.get = counted  # type: ignore[method-assign]

    answer = await _status_tool(idle).run()

    assert answer.content == "running on another replica (replica-a): Loading"
    assert sorted(reads) == [f"{SCOPE}.{KEY}", f"{SCOPE}.{KEY}.outcome", f"{SCOPE}.{KEY}.progress"]
    reads.clear()
    await _status_tool(holder).run()
    # the replica that runs asks nobody how far it is
    assert reads == [f"{SCOPE}.{KEY}.outcome"]
    unit.release.set()
    await holder.operation.wait()


# -- a unit whose work is over while it still holds the lease --------------------------------------


async def test_a_unit_that_says_its_work_is_over_is_no_run_in_progress_to_the_others_though_it_holds_the_lease() -> (
    None
):
    nats = FakeNatsClient()
    working, waiting = asyncio.Event(), asyncio.Event()
    entered, over = asyncio.Event(), asyncio.Event()
    holder: CoalescedOperation[dict[str, int]]

    async def unit() -> dict[str, int]:
        entered.set()
        await working.wait()
        await holder.work_over()
        holder.report_progress({"summary": "reported after the work was over"})
        over.set()
        await waiting.wait()
        return {"unit": 1}

    holder = _operation(nats, "replica-a", unit)
    idle = _operation(nats, "replica-b", _Unit())
    holder_status, idle_status = _status_tool(holder), _status_tool(idle)
    observer = _lease(nats, "observer")

    # the lease held, nothing reported yet: a run in progress, with an empty progress
    await holder.request()
    await asyncio.wait_for(entered.wait(), timeout=5)
    held = await idle_status.run()
    assert held.metadata is not None
    assert held.metadata["state"] == "running" and held.metadata["elsewhere"] == {
        "replica": "replica-a",
        "progress": {},
    }
    assert held.content == "running on another replica (replica-a)"

    # reporting: a run in progress, with what was reported
    holder.report_progress({"summary": "Publishing"})
    await asyncio.sleep(0.01)
    reporting = await idle_status.run()
    assert reporting.content == "running on another replica (replica-a): Publishing"

    # said by a replica whose unit it is not, it is nothing: the run is still in progress
    await idle.work_over()
    assert (await idle_status.run()).content == "running on another replica (replica-a): Publishing"

    # work over, the lease still held: no run in progress to the others
    working.set()
    await asyncio.wait_for(over.wait(), timeout=5)
    await asyncio.sleep(0.01)
    assert await observer.holder(KEY) == "replica-a"
    assert await _progress(nats) == {"holder": "replica-a", "work_over": True}
    assert await idle.running_elsewhere() is None
    ended = await idle_status.run()
    assert ended.metadata is not None
    assert ended.metadata["state"] == "idle" and ended.metadata["running"] is False
    assert ended.metadata["elsewhere"] is None and ended.metadata["run_progress"] is None
    assert ended.content == "idle"
    # the replica whose unit it is still runs it, and reports nothing more of it
    own = await holder_status.run()
    assert own.metadata is not None
    assert own.metadata["running"] is True and own.metadata["run_progress"] is None
    assert own.content == f"running since {own.metadata['started_at']}"

    # from the unit's return until the lease is let go, the entry still says the work is over
    bucket = await _bucket(nats)
    under_the_lease: list[Any] = []
    real_get_entry = bucket.get_entry

    async def watched(**kwargs: Any) -> tuple[bytes, int] | None:
        if kwargs["key"].endswith(".requested") and await observer.holder(KEY) == "replica-a":
            under_the_lease.append(await _progress(nats))
        return await real_get_entry(**kwargs)

    bucket.get_entry = watched  # type: ignore[method-assign]

    # the lease let go: no run in progress, the entry gone, and the run recorded
    waiting.set()
    await holder.operation.wait()
    assert under_the_lease and all(entry == {"holder": "replica-a", "work_over": True} for entry in under_the_lease)
    assert await observer.holder(KEY) is None
    assert await _progress(nats) is None
    released = await idle_status.run()
    assert released.metadata is not None
    assert released.metadata["state"] == "succeeded" and released.metadata["elsewhere"] is None
    assert released.metadata["last"]["replica"] == "replica-a"


async def test_a_unit_run_after_one_whose_work_was_over_is_a_run_in_progress_again() -> None:
    nats = FakeNatsClient()
    gates = [asyncio.Event(), asyncio.Event()]
    overs = [asyncio.Event(), asyncio.Event()]
    calls = 0
    holder: CoalescedOperation[dict[str, int]]

    async def unit() -> dict[str, int]:
        nonlocal calls
        number = calls
        calls += 1
        if number == 0:
            await holder.work_over()
        overs[number].set()
        await gates[number].wait()
        return {"unit": number + 1}

    holder = _operation(nats, "replica-a", unit)
    idle = _operation(nats, "replica-b", _Unit())
    await holder.request()
    await asyncio.wait_for(overs[0].wait(), timeout=5)
    assert await idle.running_elsewhere() is None

    # asked again while the first unit holds the lease past its work: the unit that follows is a run
    await idle.request()
    await idle.operation.wait()
    gates[0].set()
    await asyncio.wait_for(overs[1].wait(), timeout=5)
    again = await idle.running_elsewhere()
    assert again is not None and again.replica == "replica-a" and again.progress == {}

    gates[1].set()
    await holder.operation.wait()
    assert await _progress(nats) is None
    await idle.stop()


async def test_only_the_holders_own_word_ends_its_run_early() -> None:
    nats = FakeNatsClient()
    reader = _operation(nats, "replica-b", _Unit())
    bucket = await _bucket(nats)
    await _lease(nats, "replica-a").acquire(KEY, ttl_seconds=30)

    # an entry naming no writer, or another's, may be one a holder before this one left
    for body in ({"work_over": True}, {"holder": "replica-z", "work_over": True}):
        await bucket.put(key=f"{SCOPE}.{KEY}.progress", value=json.dumps(body).encode())
        assert await reader.running_elsewhere() is not None

    await bucket.put(
        key=f"{SCOPE}.{KEY}.progress", value=json.dumps({"holder": "replica-a", "work_over": "yes"}).encode()
    )
    assert await reader.running_elsewhere() is not None
    await bucket.put(
        key=f"{SCOPE}.{KEY}.progress", value=json.dumps({"holder": "replica-a", "work_over": True}).encode()
    )
    assert await reader.running_elsewhere() is None


async def test_a_unit_that_cannot_say_its_work_is_over_goes_on_and_ends_as_it_ends(
    caplog: pytest.LogCaptureFixture,
) -> None:
    nats = FakeNatsClient()
    holder: CoalescedOperation[dict[str, int]]
    bucket = await _bucket(nats)
    real_put = bucket.put

    async def refusing(**kwargs: Any) -> int:
        if b"work_over" in kwargs["value"]:
            raise ConnectionError("the over write was refused")
        return await real_put(**kwargs)

    bucket.put = refusing  # type: ignore[method-assign]

    async def unit() -> dict[str, int]:
        await holder.work_over()
        return {"unit": 1}

    holder = _operation(nats, "replica-a", unit)
    await holder.request()
    await holder.operation.wait()

    last = holder.operation.last
    assert last is not None and last.result == Drained(results=({"unit": 1},))
    assert any("the over write was refused" in str(getattr(record, "extra_data", "")) for record in caplog.records)
    # the entry it could not replace is removed once the lease is let go
    assert await _progress(nats) is None


# -- progress given as what answers it -------------------------------------------------------------


async def test_progress_given_as_a_callable_is_asked_only_when_it_is_written_or_shown() -> None:
    nats, unit = FakeNatsClient(), _Unit(held=True)
    load = _operation(nats, "replica-a", unit)
    idle = _operation(nats, "replica-b", _Unit())
    state = {"done": 0, "asked": 0}

    def render() -> dict[str, Any]:
        state["asked"] += 1
        return {"summary": f"{state['done']} done", "holder": "the pod's own", "work_over": True}

    # no unit runs here: the call is dropped, and the progress never rendered
    for _ in range(20):
        load.report_progress(render)
    idle.report_progress(render)
    await asyncio.sleep(0.01)
    assert state["asked"] == 0

    await load.request()
    await _entered(unit)
    for done in range(1, 31):
        state["done"] = done
        load.report_progress(render)
    await asyncio.sleep(0.05)

    # thirty reports are a write or two, each rendered as it is written: the latest state, not the first
    assert 1 <= state["asked"] <= 2
    assert await _progress(nats) == {"summary": "30 done", "holder": "replica-a"}
    there = await idle.running_elsewhere()
    assert there is not None and there.progress == {"summary": "30 done"}

    # the replica that runs shows it as it is now, asked anew
    state["done"] = 31
    asked = state["asked"]
    here = await _status_tool(load).run()
    assert here.metadata is not None
    assert here.metadata["run_progress"] == {"summary": "31 done"} and state["asked"] == asked + 1

    unit.release.set()
    await load.operation.wait()
    asked = state["asked"]
    load.report_progress(render)
    await _status_tool(load).run()
    assert state["asked"] == asked


async def test_progress_that_cannot_be_rendered_fails_neither_the_unit_nor_the_status(
    caplog: pytest.LogCaptureFixture,
) -> None:
    nats, unit = FakeNatsClient(), _Unit(held=True)
    load = _operation(nats, "replica-a", unit)

    def broken() -> dict[str, Any]:
        raise LookupError("no states yet")

    await load.request()
    await _entered(unit)
    load.report_progress(broken)
    await asyncio.sleep(0.01)

    answer = await _status_tool(load).run()
    assert answer.success and answer.metadata is not None
    assert answer.metadata["running"] is True and answer.metadata["run_progress"] is None
    assert answer.content == f"running since {answer.metadata['started_at']}"

    unit.release.set()
    await load.operation.wait()
    last = load.operation.last
    assert last is not None and last.result == Drained(results=({"unit": 1},))
    messages = [
        record.message for record in caplog.records if "no states yet" in str(getattr(record, "extra_data", ""))
    ]
    assert any("could not be shared" in message for message in messages)
    assert any("could not be asked" in message for message in messages)


async def test_between_a_units_end_and_the_runs_the_replica_that_ran_it_shows_no_progress() -> None:
    nats, unit = FakeNatsClient(), _Unit(held=True)
    load = _operation(nats, "replica-a", unit)
    bucket = await _bucket(nats)
    recording, recorded = asyncio.Event(), asyncio.Event()
    real_create = bucket.create

    async def slow_record(**kwargs: Any) -> int | None:
        # the run's own write is held; the one the status makes for it goes through
        if kwargs["key"].endswith(".outcome") and not recording.is_set():
            recording.set()
            await recorded.wait()
        return await real_create(**kwargs)

    bucket.create = slow_record  # type: ignore[method-assign]
    await load.request()
    await _entered(unit)
    load.report_progress({"summary": "Loading"})
    await asyncio.sleep(0.01)
    unit.release.set()
    await asyncio.wait_for(recording.wait(), timeout=5)

    # the unit has ended and the run has not (its outcome is being recorded): nothing is in progress to show
    answer = await _status_tool(load).run()
    assert answer.metadata is not None
    assert answer.metadata["running"] is True and answer.metadata["run_progress"] is None
    assert answer.content.startswith(f"running since {answer.metadata['started_at']}; the last run succeeded")

    recorded.set()
    await load.operation.wait()


# -- an entry is one holding's ---------------------------------------------------------------------


async def test_an_entry_names_the_holding_of_the_lease_it_was_written_under() -> None:
    nats, unit = FakeNatsClient(), _Unit(held=True)
    load = _operation(nats, "replica-a", unit)
    await load.request()
    await _entered(unit)
    holding = await _lease(nats, "observer").held(KEY)
    assert holding is not None

    stamp = holding.acquired_at.isoformat(timespec="microseconds")
    assert await _stored(nats, ".progress") == {"holder": "replica-a", "holding": stamp}
    load.report_progress({"summary": "Loading", "holding": "the pod's own"})
    await asyncio.sleep(0.01)
    assert await _stored(nats, ".progress") == {"summary": "Loading", "holder": "replica-a", "holding": stamp}

    unit.release.set()
    await load.operation.wait()


async def test_a_run_that_has_just_taken_the_lease_is_in_progress_whatever_entry_its_replica_left_before() -> None:
    nats = FakeNatsClient()
    reader = _operation(nats, "replica-b", _Unit())
    bucket = await _bucket(nats)
    for left in (
        {"holder": "replica-a", "holding": "2026-10-12T12:00:00.000000+00:00", "work_over": True},
        {"holder": "replica-a", "holding": "2026-10-12T12:00:00.000000+00:00", "summary": "Publishing"},
    ):
        await bucket.put(key=f"{SCOPE}.{KEY}.progress", value=json.dumps(left).encode())
        handle = await _lease(nats, "replica-a").acquire(KEY, ttl_seconds=30)

        # the lease is this replica's again, under another holding: the entry is not this run's
        running = await reader.running_elsewhere()
        assert running is not None and running.replica == "replica-a" and running.progress is None
        await handle.release()


async def test_a_work_over_entry_that_could_not_be_removed_does_not_hide_the_same_replicas_next_run() -> None:
    nats = FakeNatsClient()
    waiting, entered = asyncio.Event(), asyncio.Event()
    holder: CoalescedOperation[dict[str, int]]
    quiet = {"writes": True}

    async def unit() -> dict[str, int]:
        await holder.work_over()
        entered.set()
        await waiting.wait()
        return {"unit": 1}

    holder = _operation(nats, "replica-a", unit)
    reader = _operation(nats, "replica-b", _Unit())
    bucket = await _bucket(nats)
    real_delete, real_put = bucket.delete, bucket.put

    async def refusing_delete(**kwargs: Any) -> bool:
        if kwargs["key"].endswith(".progress"):
            raise ConnectionError("the delete was refused")
        return await real_delete(**kwargs)

    async def refusing_put(**kwargs: Any) -> int:
        if kwargs["key"].endswith(".progress") and not quiet["writes"]:
            raise ConnectionError("the write was refused")
        return await real_put(**kwargs)

    bucket.delete = refusing_delete  # type: ignore[method-assign]
    bucket.put = refusing_put  # type: ignore[method-assign]

    waiting.set()
    await holder.request()
    await holder.operation.wait()
    left = await _progress(nats)
    assert left == {"holder": "replica-a", "work_over": True}
    assert await reader.running_elsewhere() is None

    # its next run writes nothing at all (every progress write is refused), and never says its work is over
    quiet["writes"] = False
    waiting.clear()
    entered.clear()

    async def silent() -> dict[str, int]:
        entered.set()
        await waiting.wait()
        return {"unit": 2}

    again = _operation(nats, "replica-a", silent)
    await again.request()
    await asyncio.wait_for(entered.wait(), timeout=5)

    assert await _progress(nats) == left
    running = await reader.running_elsewhere()
    assert running is not None and running.replica == "replica-a" and running.progress is None
    answer = await _status_tool(reader).run()
    assert answer.content.startswith("running on another replica (replica-a)")
    waiting.set()
    await again.operation.wait()


async def test_a_unit_that_follows_one_whose_work_was_over_ends_as_any_unit_does() -> None:
    nats = FakeNatsClient()
    gates = [asyncio.Event(), asyncio.Event()]
    begun = [asyncio.Event(), asyncio.Event()]
    calls = 0
    holder: CoalescedOperation[dict[str, int]]

    async def unit() -> dict[str, int]:
        nonlocal calls
        number = calls
        calls += 1
        if number == 0:
            await holder.work_over()
        else:
            holder.report_progress({"summary": "the second unit"})
            await asyncio.sleep(0.01)
        begun[number].set()
        await gates[number].wait()
        return {"unit": number + 1}

    holder = _operation(nats, "replica-a", unit)
    observer = _lease(nats, "observer")
    await holder.request()
    await asyncio.wait_for(begun[0].wait(), timeout=5)
    await holder.request()
    gates[0].set()
    await asyncio.wait_for(begun[1].wait(), timeout=5)
    assert await _progress(nats) == {"summary": "the second unit", "holder": "replica-a"}

    bucket = await _bucket(nats)
    under_the_lease: list[Any] = []
    real_get_entry = bucket.get_entry

    async def watched(**kwargs: Any) -> tuple[bytes, int] | None:
        if kwargs["key"].endswith(".requested") and await observer.holder(KEY) == "replica-a":
            under_the_lease.append(await _progress(nats))
        return await real_get_entry(**kwargs)

    bucket.get_entry = watched  # type: ignore[method-assign]
    gates[1].set()
    await holder.operation.wait()

    # the second unit's progress is removed when it ends, under the lease, as any unit's is
    assert under_the_lease and all(entry is None for entry in under_the_lease)


async def test_a_unit_that_cannot_read_its_lease_runs_and_its_progress_names_no_holding(
    caplog: pytest.LogCaptureFixture,
) -> None:
    nats, unit = FakeNatsClient(), _Unit()
    load = _operation(nats, "replica-a", unit)
    reader = _operation(nats, "replica-b", _Unit())
    # a unit before it ran under a holding it could read: that one is not this unit's
    await load.request()
    await load.operation.wait()
    unit.release.clear()
    unit.entered.clear()
    bucket = await _bucket(nats)
    real_get = bucket.get
    refused = 0

    async def refusing_once(**kwargs: Any) -> bytes | None:
        nonlocal refused
        if kwargs["key"] == f"{SCOPE}.{KEY}" and not refused:
            refused += 1
            raise TimeoutError("the lease read was not answered")
        return await real_get(**kwargs)

    bucket.get = refusing_once  # type: ignore[method-assign]
    await load.request()
    await _entered(unit)
    load.report_progress({"summary": "Loading"})
    await asyncio.sleep(0.01)

    assert refused == 1
    assert await _stored(nats, ".progress") == {"summary": "Loading", "holder": "replica-a"}
    # an entry naming no holding is its holder's present run's
    running = await reader.running_elsewhere()
    assert running is not None and running.progress == {"summary": "Loading"}
    unit.release.set()
    await load.operation.wait()
    last = load.operation.last
    assert last is not None and last.result == Drained(results=({"unit": 2},))
    assert any("names no holding" in record.message for record in caplog.records)
