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


def _make_unreachable(bucket: FakeKvBucket, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Make every read of ``bucket`` raise while ``state["down"]`` is true."""
    state: dict[str, Any] = {"down": False}
    real = bucket.get_entry

    async def get_entry(*, key: str) -> tuple[bytes, int] | None:
        if state["down"]:
            raise ConnectionError("kv is unreachable")
        return await real(key=key)

    monkeypatch.setattr(bucket, "get_entry", get_entry)
    return state


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
async def test_a_brief_transport_failure_is_not_a_lost_lease(monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeNatsClient()
    held = await _lease(client, "pod-a").hold("job", ttl=_TTL, renew_every=_RENEW)
    try:
        state = _make_unreachable(await _bucket(client), monkeypatch)
        state["down"] = True
        await asyncio.sleep(_RENEW.total_seconds() * 4)
        state["down"] = False
        assert held.held, "a transient renewal failure was treated as losing the lease"
        await asyncio.sleep(_RENEW.total_seconds() * 3)
        assert held.held, "renewal did not resume once the bucket came back"
    finally:
        await held.release()


@_in_virtual_time
async def test_a_failure_lasting_past_the_ttl_gives_the_lease_up(monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeNatsClient()
    held = await _lease(client, "pod-a").hold("job", ttl=_TTL, renew_every=_RENEW)
    try:
        _make_unreachable(await _bucket(client), monkeypatch)["down"] = True
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
async def test_a_release_that_cannot_reach_the_bucket_does_not_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    client = FakeNatsClient()
    held = await _lease(client, "pod-a").hold("job", ttl=_TTL, renew_every=_RENEW)
    _make_unreachable(await _bucket(client), monkeypatch)["down"] = True
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
async def test_loss_is_reported_no_later_than_the_entry_could_expire(monkeypatch: pytest.MonkeyPatch) -> None:
    """Another pod may take the key the moment the entry expires, so ``lost`` must be set by then --
    not up to a renewal interval later."""
    client = FakeNatsClient()
    loop = asyncio.get_running_loop()
    started = loop.time()
    held = await _lease(client, "pod-a").hold("job", ttl=_TTL, renew_every=timedelta(seconds=0.4))
    try:
        _make_unreachable(await _bucket(client), monkeypatch)["down"] = True
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
