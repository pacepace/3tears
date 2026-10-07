"""``CoalescedRun``: one run at a time across replicas, and a request during a run runs it once more.

Over a real ``KVLease`` and the in-memory bucket, so the lease's compare-and-swap and the request
key's revisions are the genuine article. Two ``KVLease`` factories with different holder ids on one
client stand in for two replicas of one pod: they share the bucket and the key, as replicas do.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta

import pytest

from threetears.core.coordination.coalesced_run import CoalescedRun
from threetears.core.coordination.lease import KVLease, LeaseLost
from threetears.core.testing.kv import FakeKvBucket, FakeNatsClient

_KEY = "enr/refresh"
_TTL = timedelta(seconds=30)
_RENEW = timedelta(seconds=10)


class _Body:
    """the operation: counts its runs, and can be held open until a test lets it finish."""

    def __init__(self) -> None:
        self.runs = 0
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.release.set()

    def hold_open(self) -> None:
        self.release.clear()

    async def __call__(self) -> None:
        self.runs += 1
        self.started.set()
        await self.release.wait()


def _replica(client: FakeNatsClient, holder: str, body: _Body) -> CoalescedRun:
    lease = KVLease(client, pod_id=holder)  # type: ignore[arg-type]
    return CoalescedRun(lease, _KEY, body, ttl=_TTL, renew_every=_RENEW)


async def test_a_request_is_run_once() -> None:
    body = _Body()
    run = _replica(FakeNatsClient(), "pod-a", body)

    await run.request()
    assert await run.drain() == 1

    assert body.runs == 1


async def test_nothing_runs_without_a_request() -> None:
    body = _Body()
    run = _replica(FakeNatsClient(), "pod-a", body)

    assert await run.drain() == 0
    assert body.runs == 0


async def test_requests_during_a_run_run_it_once_more_and_no_more() -> None:
    body = _Body()
    run = _replica(FakeNatsClient(), "pod-a", body)
    await run.request()
    body.hold_open()
    draining = asyncio.create_task(run.drain())
    await body.started.wait()

    for _ in range(3):
        await run.request()
    body.release.set()

    assert await draining == 2
    assert body.runs == 2


async def test_a_request_on_another_replica_during_a_run_is_run_by_the_holder() -> None:
    client = FakeNatsClient()
    body_a, body_b = _Body(), _Body()
    a, b = _replica(client, "pod-a", body_a), _replica(client, "pod-b", body_b)
    await a.request()
    body_a.hold_open()
    draining = asyncio.create_task(a.drain())
    await body_a.started.wait()

    await b.request()
    await b.request()
    assert await b.drain() == 0, "a second replica ran while the first held the lease"
    body_a.release.set()

    assert await draining == 2
    assert (body_a.runs, body_b.runs) == (2, 0)


async def test_a_request_landing_as_the_holder_lets_go_is_still_run() -> None:
    client = FakeNatsClient()
    body_a, body_b = _Body(), _Body()
    a, b = _replica(client, "pod-a", body_a), _replica(client, "pod-b", body_b)
    await a.request()
    assert await a.drain() == 1
    bucket: FakeKvBucket = await client.kv_bucket(name="leases")
    read_request = bucket.get_entry
    asked_late = False

    async def get_entry(*, key: str) -> tuple[bytes, int] | None:
        nonlocal asked_late
        entry = await read_request(key=key)
        lease_held = await read_request(key=_KEY) is not None
        if key != _KEY and entry is None and lease_held and not asked_late:
            # the holder has just found no request, and still holds the lease: b asks now, and
            # cannot hold it, so only the holder can run what b asked for
            asked_late = True
            await b.request()
            assert await b.drain() == 0
        return entry

    bucket.get_entry = get_entry  # type: ignore[method-assign]
    await a.request()

    assert await a.drain() == 2
    assert asked_late
    assert (body_a.runs, body_b.runs) == (3, 0)


async def test_after_a_holder_finishes_another_replica_runs_its_own_request() -> None:
    client = FakeNatsClient()
    body_a, body_b = _Body(), _Body()
    a, b = _replica(client, "pod-a", body_a), _replica(client, "pod-b", body_b)
    await a.request()
    assert await a.drain() == 1

    await b.request()
    assert await b.drain() == 1
    assert (body_a.runs, body_b.runs) == (1, 1)


async def test_a_failed_run_is_raised_and_not_retried() -> None:
    client = FakeNatsClient()

    async def fails() -> None:
        raise ConnectionError("the warehouse went away")

    run = CoalescedRun(KVLease(client, pod_id="pod-a"), _KEY, fails, ttl=_TTL, renew_every=_RENEW)  # type: ignore[arg-type]
    await run.request()
    with pytest.raises(ConnectionError, match="went away"):
        await run.drain()

    assert await run.drain() == 0, "a failed run was retried without a new request"


async def test_a_run_whose_lease_is_lost_is_cancelled_and_asked_for_again() -> None:
    client = FakeNatsClient()
    body = _Body()
    lease = KVLease(client, pod_id="pod-a")  # type: ignore[arg-type]
    run = CoalescedRun(lease, _KEY, body, ttl=timedelta(seconds=1), renew_every=timedelta(seconds=0.05))
    await run.request()
    body.hold_open()
    draining = asyncio.create_task(run.drain())
    await body.started.wait()

    bucket: FakeKvBucket = await client.kv_bucket(name="leases")
    entry = await bucket.get_entry(key=_KEY)
    assert entry is not None
    # another replica reclaimed the key: the holder's next renewal finds it gone
    assert await bucket.update(key=_KEY, value=entry[0].replace(b"pod-a", b"pod-b"), revision=entry[1]) is not None

    with pytest.raises(LeaseLost):
        await asyncio.wait_for(draining, timeout=10)
    assert body.runs == 1
    assert await run.requested(), "the lost run's request was dropped"


def test_the_renewal_must_be_shorter_than_the_lease() -> None:
    with pytest.raises(ValueError, match="renew_every"):
        CoalescedRun(KVLease(FakeNatsClient()), _KEY, _Body(), ttl=_TTL, renew_every=_TTL)  # type: ignore[arg-type]
