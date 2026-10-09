"""``KVLease.hold``: a lease kept alive in the background that says when it has been lost.

``acquire`` hands back a lease the caller must refresh by hand, so every consumer that holds one
across real work writes the same renewal loop -- scrape's session claim, and in scriob a git-writer
lease and a chat-turn guard, neither of which ever learned it had been lost. These tests pin the one
loop they all move onto, over a real ``KVLease`` and an in-memory bucket, so compare-and-swap,
holder identity and expiry are the genuine article.

**Every test runs in VIRTUAL time.** ``HeldLease`` times everything on the event loop's clock --
renewals every 50ms against a one-second TTL, an expiry timer, bounded waits -- so on the wall clock
these tests measured the host: under a loaded CI runner a renewal scheduled at 50ms ran late enough
to cross the TTL and the lease was reported lost. :class:`_VirtualTimeLoop` is a stock selector loop
whose clock stands still while work is ready and jumps straight to the next timer when nothing is:
the production code runs unchanged, and every interleaving is a function of the schedule alone.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import selectors
from collections.abc import Callable, Coroutine
from datetime import timedelta
from typing import Any

import pytest

from threetears.core.coordination.lease import HeldLease, KVLease, LeaseUnavailable
from threetears.core.testing.kv import FakeKvBucket, FakeNatsClient

_BUCKET = "held"
_TTL = timedelta(seconds=1)
_RENEW = timedelta(seconds=0.05)

#: how long, on the REAL clock, the virtual loop waits for I/O when nothing at all is scheduled
#: before calling the test deadlocked -- rather than hanging the run.
_IDLE_REAL_SECONDS = 5.0


class _VirtualClock:
    """the one clock a :class:`_VirtualTimeLoop` and its selector share."""

    def __init__(self) -> None:
        self.now = 0.0


# parity-exempt: a selector that delegates every registration to the platform's default selector and only changes how long select() waits
class _AdvancingSelector(selectors.BaseSelector):
    """polls the real selector without blocking, and advances virtual time instead of waiting.

    The loop asks ``select(timeout)`` with ``timeout`` the time until its next timer (``0`` when
    work is ready). Nothing arriving on a real file descriptor means nothing could happen before that
    timer, so the clock moves to it at once.
    """

    def __init__(self, clock: _VirtualClock) -> None:
        self._inner = selectors.DefaultSelector()
        self._clock = clock

    def register(self, fileobj: Any, events: int, data: Any = None) -> selectors.SelectorKey:
        return self._inner.register(fileobj, events, data)

    def unregister(self, fileobj: Any) -> selectors.SelectorKey:
        return self._inner.unregister(fileobj)

    def modify(self, fileobj: Any, events: int, data: Any = None) -> selectors.SelectorKey:
        return self._inner.modify(fileobj, events, data)

    def select(self, timeout: float | None = None) -> list[tuple[selectors.SelectorKey, int]]:
        ready = self._inner.select(0)
        if not ready and timeout is not None and timeout > 0:
            self._clock.now += timeout
        elif not ready and timeout is None:
            # nothing scheduled and nothing ready: only another thread could wake the loop now.
            ready = self._inner.select(_IDLE_REAL_SECONDS)
            if not ready:
                raise RuntimeError("virtual-time loop is idle with nothing scheduled: the test deadlocked")
        return ready

    def close(self) -> None:
        self._inner.close()

    def get_key(self, fileobj: Any) -> selectors.SelectorKey:
        return self._inner.get_key(fileobj)

    def get_map(self) -> Any:
        return self._inner.get_map()


class _VirtualTimeLoop(asyncio.SelectorEventLoop):
    """a selector event loop whose ``time()`` is virtual and skips every idle wait."""

    def __init__(self) -> None:
        self.virtual_clock = _VirtualClock()
        super().__init__(selector=_AdvancingSelector(self.virtual_clock))

    def time(self) -> float:
        return self.virtual_clock.now


def _in_virtual_time(test: Callable[..., Coroutine[Any, Any, None]]) -> Callable[..., None]:
    """run an async test body to completion on a fresh :class:`_VirtualTimeLoop`.

    :param test: the async test function
    :ptype test: Callable[..., Coroutine[Any, Any, None]]
    :return: a synchronous test pytest collects with the same fixtures
    :rtype: Callable[..., None]
    """

    @functools.wraps(test)
    def run(*args: Any, **kwargs: Any) -> None:
        asyncio.run(test(*args, **kwargs), loop_factory=_VirtualTimeLoop)

    return run


def _lease(client: FakeNatsClient, pod: str) -> KVLease:
    return KVLease(nats_client=client, bucket_name=_BUCKET, pod_id=pod)  # type: ignore[arg-type]


async def _bucket(client: FakeNatsClient) -> FakeKvBucket:
    return await client.kv_bucket(name=_BUCKET)


async def _take_over(client: FakeNatsClient, key: str) -> None:
    """Rewrite the entry's holder by compare-and-swap, as a stale reclaim by another pod leaves it."""
    bucket = await _bucket(client)
    entry = await bucket.get_entry(key=key)
    assert entry is not None
    value, revision = entry
    assert await bucket.update(key=key, value=value.replace(b"pod-a", b"pod-b"), revision=revision) is not None


def _outage() -> ConnectionError:
    """the error every operation raises while the bucket is unreachable, as a lost broker answers."""
    return ConnectionError("kv is unreachable")


@_in_virtual_time
async def test_a_held_lease_is_renewed_past_several_ttls() -> None:
    client = FakeNatsClient()
    held = await _lease(client, "pod-a").hold("job", ttl=_TTL, renew_every=_RENEW)
    try:
        await asyncio.sleep(_TTL.total_seconds() * 2.5)
        assert held.held
        with pytest.raises(LeaseUnavailable):
            await _lease(client, "pod-b").hold("job", ttl=_TTL, renew_every=_RENEW)
    finally:
        await held.release()


@_in_virtual_time
async def test_a_takeover_is_noticed_on_the_next_renewal() -> None:
    client = FakeNatsClient()
    held = await _lease(client, "pod-a").hold("job", ttl=_TTL, renew_every=_RENEW)
    try:
        await _take_over(client, "job")
        await asyncio.wait_for(held.until_lost(), timeout=5)
        assert not held.held
        assert held.lost.is_set()
    finally:
        await held.release()


@_in_virtual_time
async def test_releasing_a_lost_lease_leaves_the_new_owners_entry() -> None:
    client = FakeNatsClient()
    held = await _lease(client, "pod-a").hold("job", ttl=_TTL, renew_every=_RENEW)
    await _take_over(client, "job")
    await asyncio.wait_for(held.until_lost(), timeout=5)
    await held.release()
    surviving = await (await _bucket(client)).get_entry(key="job")
    assert surviving is not None and b"pod-b" in surviving[0]


@_in_virtual_time
async def test_a_brief_transport_failure_is_not_a_lost_lease() -> None:
    client = FakeNatsClient()
    held = await _lease(client, "pod-a").hold("job", ttl=_TTL, renew_every=_RENEW)
    bucket = await _bucket(client)
    try:
        bucket.become_unreachable(_outage())
        await asyncio.sleep(_RENEW.total_seconds() * 4)
        bucket.become_reachable()
        assert held.held, "a transient renewal failure was treated as losing the lease"
        await asyncio.sleep(_RENEW.total_seconds() * 3)
        assert held.held, "renewal did not resume once the bucket came back"
    finally:
        await held.release()


@_in_virtual_time
async def test_a_failure_lasting_past_the_ttl_gives_the_lease_up() -> None:
    client = FakeNatsClient()
    held = await _lease(client, "pod-a").hold("job", ttl=_TTL, renew_every=_RENEW)
    try:
        (await _bucket(client)).become_unreachable(_outage())
        await asyncio.wait_for(held.until_lost(), timeout=5)
        assert not held.held
    finally:
        await held.release()


@_in_virtual_time
async def test_release_frees_the_key_and_is_idempotent() -> None:
    client = FakeNatsClient()
    held = await _lease(client, "pod-a").hold("job", ttl=_TTL, renew_every=_RENEW)
    await held.release()
    await held.release()
    assert await (await _bucket(client)).get_entry(key="job") is None
    other = await _lease(client, "pod-b").hold("job", ttl=_TTL, renew_every=_RENEW)
    await other.release()


@_in_virtual_time
async def test_release_stops_renewal() -> None:
    client = FakeNatsClient()
    held = await _lease(client, "pod-a").hold("job", ttl=_TTL, renew_every=_RENEW)
    await held.release()
    await asyncio.sleep(_RENEW.total_seconds() * 3)
    assert await (await _bucket(client)).get_entry(key="job") is None, "a renewal after release put the entry back"


@_in_virtual_time
async def test_as_a_context_manager_it_releases_on_exit_even_when_the_body_fails() -> None:
    client = FakeNatsClient()
    with pytest.raises(RuntimeError, match="body"):
        async with await _lease(client, "pod-a").hold("job", ttl=_TTL, renew_every=_RENEW) as held:
            assert isinstance(held, HeldLease)
            raise RuntimeError("body")
    assert await (await _bucket(client)).get_entry(key="job") is None


@_in_virtual_time
async def test_a_release_that_cannot_reach_the_bucket_does_not_raise() -> None:
    client = FakeNatsClient()
    held = await _lease(client, "pod-a").hold("job", ttl=_TTL, renew_every=_RENEW)
    (await _bucket(client)).become_unreachable(_outage())
    await held.release()  # the TTL frees the entry; a cleanup error must not replace the caller's outcome


@_in_virtual_time
async def test_wait_bound_is_honoured() -> None:
    client = FakeNatsClient()
    held = await _lease(client, "pod-a").hold("job", ttl=_TTL, renew_every=_RENEW)
    try:
        with pytest.raises(LeaseUnavailable):
            await _lease(client, "pod-b").hold("job", ttl=_TTL, renew_every=_RENEW, max_wait_seconds=0)
    finally:
        await held.release()


@pytest.mark.parametrize(
    ("ttl", "renew_every", "match"),
    [
        (timedelta(seconds=0.5), timedelta(seconds=0.1), "whole number of seconds"),
        (timedelta(seconds=1), timedelta(seconds=1), "shorter than ttl"),
        (timedelta(seconds=1), timedelta(0), "renew_every must be positive"),
    ],
)
@_in_virtual_time
async def test_configuration_that_cannot_hold_is_refused(ttl: timedelta, renew_every: timedelta, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        await _lease(FakeNatsClient(), "pod-a").hold("job", ttl=ttl, renew_every=renew_every)


@_in_virtual_time
async def test_loss_is_reported_no_later_than_the_entry_could_expire() -> None:
    """Another pod may take the key the moment the entry expires, so ``lost`` must be set by then --
    not up to a renewal interval later."""
    client = FakeNatsClient()
    loop = asyncio.get_running_loop()
    started = loop.time()
    held = await _lease(client, "pod-a").hold("job", ttl=_TTL, renew_every=timedelta(seconds=0.4))
    try:
        (await _bucket(client)).become_unreachable(_outage())
        await asyncio.wait_for(held.until_lost(), timeout=5)
        assert loop.time() - started <= _TTL.total_seconds() + 0.05
        assert held.lost.is_set()
    finally:
        await held.release()


@_in_virtual_time
async def test_a_hanging_renewal_does_not_delay_the_loss(monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeNatsClient()
    loop = asyncio.get_running_loop()
    started = loop.time()
    held = await _lease(client, "pod-a").hold("job", ttl=_TTL, renew_every=_RENEW)
    bucket = await _bucket(client)

    async def hang(*, key: str) -> tuple[bytes, int] | None:
        await asyncio.sleep(3600)
        return None

    monkeypatch.setattr(bucket, "get_entry", hang)
    try:
        await asyncio.wait_for(held.until_lost(), timeout=5)
        assert loop.time() - started <= _TTL.total_seconds() + 0.05
    finally:
        monkeypatch.undo()
        await held.release()


@_in_virtual_time
async def test_release_during_an_in_flight_renewal_still_frees_the_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cancelling a renewal between the server applying it and the handle recording the new revision
    would leave the delete keyed on a stale revision -- and the entry held for a full TTL."""
    client = FakeNatsClient()
    held = await _lease(client, "pod-a").hold("job", ttl=_TTL, renew_every=_RENEW)
    bucket = await _bucket(client)
    real_update = bucket.update
    in_flight = asyncio.Event()

    async def slow_update(**kwargs: Any) -> int | None:
        in_flight.set()
        revision = await real_update(**kwargs)
        await asyncio.sleep(0.1)  # the write has landed; the handle has not heard yet
        return revision

    monkeypatch.setattr(bucket, "update", slow_update)
    await asyncio.wait_for(in_flight.wait(), timeout=5)
    await held.release()
    assert await bucket.get_entry(key="job") is None


@_in_virtual_time
async def test_until_lost_returns_once_the_lease_is_released() -> None:
    """A caller racing work against the lease must not hang after it lets the lease go."""
    client = FakeNatsClient()
    held = await _lease(client, "pod-a").hold("job", ttl=_TTL, renew_every=_RENEW)
    waiter = asyncio.create_task(held.until_lost())
    await held.release()
    await asyncio.wait_for(waiter, timeout=1)
    assert not held.lost.is_set(), "a release is not a loss"
    assert not held.held


class _Records(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@_in_virtual_time
async def test_the_callers_log_context_rides_every_lease_log() -> None:
    # A handler on the lease's own logger, not caplog: another test configuring logging can stop the
    # records propagating to the root, which made a caplog version pass alone and fail in the full run.
    lease_log = logging.getLogger("threetears.core.coordination.lease")
    capture = _Records()
    lease_log.addHandler(capture)
    previous = lease_log.level
    lease_log.setLevel(logging.INFO)
    client = FakeNatsClient()
    held = await _lease(client, "pod-a").hold("job", ttl=_TTL, renew_every=_RENEW, log_extra={"session_id": "s-1"})
    try:
        await _take_over(client, "job")
        await asyncio.wait_for(held.until_lost(), timeout=5)
        # the held lease's own lines ("KVLease: ..."), not the factory's bucket-binding line
        lease_logs = [r for r in capture.records if r.getMessage().startswith("KVLease:")]
        assert lease_logs and all(getattr(r, "extra_data", {}).get("session_id") == "s-1" for r in lease_logs)
    finally:
        await held.release()
        lease_log.removeHandler(capture)
        lease_log.setLevel(previous)


@_in_virtual_time
async def test_a_lease_won_after_waiting_is_not_reported_lost_on_arrival() -> None:
    """Its life counts from the write that won it, not from when the caller started waiting."""
    client = FakeNatsClient()
    first = await _lease(client, "pod-a").hold("job", ttl=_TTL, renew_every=_RENEW)
    waiter = asyncio.create_task(_lease(client, "pod-b").hold("job", ttl=_TTL, renew_every=_RENEW, max_wait_seconds=5))
    await asyncio.sleep(1.5)  # longer than the TTL the waiter will be granted
    await first.release()
    second = await asyncio.wait_for(waiter, timeout=5)
    try:
        await asyncio.sleep(0.2)
        assert second.held, "a lease won after a long wait was reported lost the moment it arrived"
    finally:
        await second.release()


# --- a lease on a bucket another owner declared, whose readers test whether a key exists ---------
#
# A writer's claim on what it writes lives in a bucket the writer did not declare (its pod's pointer
# bucket), under a key a reader recognises by its grammar alone, and the reader -- an orphan purge --
# asks only whether the key EXISTS. So the lease must key exactly as told, in the bucket it was
# handed, and a holder that stops renewing must lose the key itself: an expiry inside the value would
# leave the key standing forever, and the purge with it.


def _claims_lease(bucket: FakeKvBucket, *, expire_entries: bool = True) -> KVLease:
    return KVLease(None, bucket=bucket, pod_id="pod-a", expire_entries=expire_entries)


async def _advance_with_the_bucket(bucket: FakeKvBucket, seconds: float, *, step: float = 0.1) -> None:
    """let ``seconds`` pass on the loop's clock and the bucket's together, so renewals and per-key
    TTLs run against the same time."""
    for _ in range(round(seconds / step)):
        await asyncio.sleep(step)
        bucket.advance_clock(timedelta(seconds=step))


@_in_virtual_time
async def test_a_lease_handed_a_bucket_keys_exactly_as_told_and_opens_none() -> None:
    client = FakeNatsClient()
    pointers = await client.kv_bucket(name="pointers")
    held = await _claims_lease(pointers).hold("enr.w.3.writer", ttl=_TTL, renew_every=_RENEW)
    try:
        assert await pointers.list_keys() == ["enr.w.3.writer"]
        assert _claims_lease(pointers).bucket_name == pointers.name
    finally:
        await held.release()
    assert await pointers.list_keys() == []


def test_a_lease_takes_a_client_or_a_bucket_never_both_nor_neither() -> None:
    with pytest.raises(ValueError, match="one of the two"):
        KVLease(None)
    bucket = asyncio.run(FakeNatsClient().kv_bucket(name="pointers"))
    with pytest.raises(ValueError, match="one of the two"):
        KVLease(FakeNatsClient(), bucket=bucket)  # type: ignore[arg-type]


@_in_virtual_time
async def test_an_expiring_entry_lives_while_renewed() -> None:
    pointers = await FakeNatsClient().kv_bucket(name="pointers")
    held = await _claims_lease(pointers).hold("enr.w.3.writer", ttl=_TTL, renew_every=_RENEW)
    try:
        await _advance_with_the_bucket(pointers, _TTL.total_seconds() * 3)
        assert await pointers.list_keys() == ["enr.w.3.writer"], "a renewal did not carry the per-key TTL forward"
        assert held.held
    finally:
        await held.release()


@pytest.mark.parametrize("expire_entries", [True, False])
@_in_virtual_time
async def test_a_holder_that_stops_renewing_loses_its_key_only_when_entries_expire(expire_entries: bool) -> None:
    pointers = await FakeNatsClient().kv_bucket(name="pointers")
    held = await _claims_lease(pointers, expire_entries=expire_entries).hold(
        "enr.w.3.writer", ttl=_TTL, renew_every=_RENEW
    )
    pointers.become_unreachable(_outage())  # its renewals stop landing, as a dead holder's do
    await asyncio.wait_for(held.until_lost(), timeout=5)
    pointers.advance_clock(_TTL * 2)
    pointers.become_reachable()
    keys = await pointers.list_keys()
    if expire_entries:
        assert keys == [], "a holder that stopped renewing still holds its key"
    else:
        assert keys == ["enr.w.3.writer"], "an envelope-expiry lease's key outlives its holder, as it always has"
    await held.release()


@_in_virtual_time
async def test_a_retaking_lease_takes_back_an_entry_its_bucket_lost() -> None:
    pointers = await FakeNatsClient().kv_bucket(name="pointers")
    held = await _claims_lease(pointers).hold("enr.w.3.writer", ttl=_TTL, renew_every=_RENEW, retake=True)
    try:
        pointers.vanish()  # NATS lost the bucket, and the entry with it
        await asyncio.sleep(_RENEW.total_seconds() * 3)
        assert await pointers.list_keys() == ["enr.w.3.writer"], "the lost entry was not taken again"
        assert held.held and not held.lost.is_set()
        await _advance_with_the_bucket(pointers, _TTL.total_seconds() * 2)
        assert await pointers.list_keys() == ["enr.w.3.writer"], "the retaken entry is not renewed"
    finally:
        await held.release()
    assert await pointers.list_keys() == []


@_in_virtual_time
async def test_a_retaking_lease_rides_out_an_outage_that_lapsed_its_key() -> None:
    pointers = await FakeNatsClient().kv_bucket(name="pointers")
    held = await _claims_lease(pointers).hold("enr.w.3.writer", ttl=_TTL, renew_every=_RENEW, retake=True)
    try:
        pointers.become_unreachable(_outage())
        await asyncio.sleep(_TTL.total_seconds() * 2)
        pointers.advance_clock(_TTL * 2)  # the key lapsed while nothing could renew it
        assert held.held, "a lapse alone lost a lease that retakes"
        pointers.become_reachable()
        await asyncio.sleep(_RENEW.total_seconds() * 3)
        assert await pointers.list_keys() == ["enr.w.3.writer"], "the lapsed entry was not taken again"
        assert held.held
    finally:
        await held.release()


@_in_virtual_time
async def test_without_retake_an_entry_its_bucket_lost_is_a_lost_lease() -> None:
    pointers = await FakeNatsClient().kv_bucket(name="pointers")
    held = await _claims_lease(pointers).hold("enr.w.3.writer", ttl=_TTL, renew_every=_RENEW)
    pointers.vanish()
    await asyncio.wait_for(held.until_lost(), timeout=5)
    assert await pointers.list_keys() == []
    await held.release()


@_in_virtual_time
async def test_a_retaking_lease_still_yields_to_another_holder() -> None:
    client = FakeNatsClient()
    held = await _lease(client, "pod-a").hold("job", ttl=_TTL, renew_every=_RENEW, retake=True)
    await _take_over(client, "job")
    await asyncio.wait_for(held.until_lost(), timeout=5)
    await held.release()
    surviving = await (await _bucket(client)).get_entry(key="job")
    assert surviving is not None and b"pod-b" in surviving[0], "a lease that retakes took another holder's entry"


@_in_virtual_time
async def test_a_retaking_lease_whose_own_renewal_landed_unseen_keeps_its_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A renewal the server applied whose reply was lost leaves the handle a revision behind its own
    entry: that is still this holder's lease, not another holder's."""
    pointers = await FakeNatsClient().kv_bucket(name="pointers")
    held = await _claims_lease(pointers).hold("enr.w.3.writer", ttl=_TTL, renew_every=_RENEW, retake=True)
    real_update = pointers.update
    dropped = [False]

    async def reply_lost(**kwargs: Any) -> int | None:
        revision = await real_update(**kwargs)
        if not dropped[0]:
            dropped[0] = True
            raise ConnectionError("the reply was lost after the write landed")
        return revision

    monkeypatch.setattr(pointers, "update", reply_lost)
    try:
        await _advance_with_the_bucket(pointers, _TTL.total_seconds() * 3)
        assert dropped[0]
        assert held.held and not held.lost.is_set(), "its own unseen renewal was taken for another holder"
        assert await pointers.list_keys() == ["enr.w.3.writer"], "the entry stopped being renewed"
    finally:
        await held.release()


@_in_virtual_time
async def test_closing_a_lease_ends_every_hold_it_handed_out() -> None:
    pointers = await FakeNatsClient().kv_bucket(name="pointers")
    lease = _claims_lease(pointers)
    holds = [await lease.hold(key, ttl=_TTL, renew_every=_RENEW, retake=True) for key in ("enr.w.3.a", "enr.w.3.b")]

    await lease.close()

    assert await pointers.list_keys() == [], "a hold outlived its lease's close"
    assert all(not held.held and not held.lost.is_set() for held in holds), "a close is a release, not a loss"
    await _advance_with_the_bucket(pointers, _RENEW.total_seconds() * 4)
    assert await pointers.list_keys() == [], "a closed hold renewed its entry back"
    after = await lease.hold("enr.w.4.c", ttl=_TTL, renew_every=_RENEW)
    try:
        await _advance_with_the_bucket(pointers, _TTL.total_seconds() * 2)
        assert after.held, "a hold taken after the close was ended by it"
    finally:
        for held in [after, *holds]:
            await held.release()


@_in_virtual_time
async def test_an_entry_the_lease_cannot_read_is_anothers_never_an_error() -> None:
    """A key written in another format (an older holder's raw id) is held by somebody: the lease neither
    fails on it nor reclaims it."""
    pointers = await FakeNatsClient().kv_bucket(name="pointers")
    await pointers.create(key="enr.rebuild", value=b"another replica")
    with pytest.raises(LeaseUnavailable):
        await _claims_lease(pointers).hold("enr.rebuild", ttl=_TTL, renew_every=_RENEW)
    assert await pointers.get(key="enr.rebuild") == b"another replica"


@_in_virtual_time
async def test_a_release_cancelled_while_it_waits_still_frees_the_entry(monkeypatch: pytest.MonkeyPatch) -> None:
    """An owner stopping cancels the task that is releasing: the entry must still go, or every other
    pod waits out its TTL."""
    client = FakeNatsClient()
    held = await _lease(client, "pod-a").hold("job", ttl=_TTL, renew_every=_RENEW)
    bucket = await _bucket(client)
    real_update = bucket.update
    in_flight = asyncio.Event()

    async def slow_update(**kwargs: Any) -> int | None:
        in_flight.set()
        await asyncio.sleep(0.5)
        return await real_update(**kwargs)

    monkeypatch.setattr(bucket, "update", slow_update)
    await asyncio.wait_for(in_flight.wait(), timeout=5)
    releasing = asyncio.create_task(held.release())
    await asyncio.sleep(0.1)  # the release waits on the renewal in flight
    releasing.cancel()
    await asyncio.wait([releasing])
    assert await bucket.get_entry(key="job") is None, "a cancelled release left the entry behind"
