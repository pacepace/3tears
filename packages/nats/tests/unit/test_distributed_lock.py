"""Unit tests for :func:`threetears.nats.nats_distributed_lock`.

Substitutes a fake KV bucket + a minimal NatsClient stand-in so the
lock lifecycle (acquire, heartbeat, release, cancellation cleanup) can
be exercised without a live JetStream broker. Real-broker round-trips
live in ``tests/integration/test_distributed_lock_round_trip.py``.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
import inspect
from datetime import timedelta

import pytest
from nats.js.errors import KeyNotFoundError, KeyWrongLastSequenceError

from threetears.nats import LockHeld, LockHold, LockLossReason, LockLost, NatsKvBucket, nats_distributed_lock
from threetears.nats.errors import KvError


# Heartbeat cadences in these tests are milliseconds, and a fixed
# ``asyncio.sleep`` long enough to cover N of them on an idle machine is
# not long enough under a loaded event loop -- the whole workspace suite
# running alongside is exactly that. Waiting for the CONDITION instead of
# for a duration makes the outcome independent of how promptly the
# scheduler gets round to the heartbeat task, while still finishing in
# milliseconds when it does.
#
# Only positive waits ("this must happen") use this. A negative assertion
# ("this must NOT happen") cannot wait for its condition and keeps an
# explicit sleep, which load can only make more generous.
async def _wait_until(
    predicate: Callable[[], bool],
    *,
    what: str,
    timeout: float = 5.0,
) -> None:
    """Await until ``predicate`` holds, or fail saying what never happened.

    :param predicate: condition to poll
    :ptype predicate: Callable[[], bool]
    :param what: description used in the timeout message
    :ptype what: str
    :param timeout: seconds to wait before failing
    :ptype timeout: float
    :return: nothing
    :rtype: None
    :raises AssertionError: when the predicate does not hold in time
    """
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            msg = f"timed out after {timeout}s waiting for {what}"
            raise AssertionError(msg)
        await asyncio.sleep(0.001)


# parity-exempt: minimal Entry dataclass for the distributed-lock unit tests carrying only value+revision; mirrors test_kv.py:_FakeEntry exemption
class _FakeEntry:
    def __init__(self, value: bytes | None, revision: int | None) -> None:
        self.value = value
        self.revision = revision


# parity-exempt: subset shim for nats.js.KeyValue used by the distributed-lock unit tests; tracks call/state for assertions, same surface as test_kv.py's _FakeKv
class _FakeKv:
    """Fake nats-py KeyValue handle backing a NatsKvBucket.

    Tracks put/create/delete counts so heartbeat-refresh and cleanup
    behavior can be asserted directly.
    """

    def __init__(self) -> None:
        self.store: dict[str, tuple[bytes, int]] = {}
        self.next_revision = 0
        self.put_calls: list[tuple[str, bytes]] = []
        self.update_calls: list[tuple[str, bytes, int]] = []
        self.create_calls: list[tuple[str, bytes]] = []
        self.delete_calls: list[str] = []
        self.fail_next_create: BaseException | None = None
        self.fail_next_delete: BaseException | None = None

    async def get(self, key: str) -> _FakeEntry:
        entry = self.store.get(key)
        if entry is None:
            raise KeyNotFoundError()
        value, rev = entry
        return _FakeEntry(value=value, revision=rev)

    async def put(self, key: str, value: bytes) -> int:
        self.put_calls.append((key, value))
        self.next_revision += 1
        self.store[key] = (value, self.next_revision)
        return self.next_revision

    async def create(self, key: str, value: bytes) -> int:
        if self.fail_next_create is not None:
            exc = self.fail_next_create
            self.fail_next_create = None
            raise exc
        self.create_calls.append((key, value))
        if key in self.store:
            raise KeyWrongLastSequenceError()
        self.next_revision += 1
        self.store[key] = (value, self.next_revision)
        return self.next_revision

    async def update(self, key: str, value: bytes, revision: int) -> int:
        self.update_calls.append((key, value, revision))
        existing = self.store.get(key)
        if existing is None or existing[1] != revision:
            raise KeyWrongLastSequenceError()
        self.next_revision += 1
        self.store[key] = (value, self.next_revision)
        return self.next_revision

    async def delete(self, key: str, last: int | None = None) -> None:
        # `last` mirrors nats-py's own KeyValue.delete, which NatsKvBucket passes through for a
        # CAS delete. Omitting it here let a revision-fenced release look untested: the fenced
        # call raised TypeError before this fake recorded anything, so `delete_calls` stayed
        # empty and the failure read as "release never happened".
        self.delete_calls.append(key)
        if self.fail_next_delete is not None:
            exc = self.fail_next_delete
            self.fail_next_delete = None
            raise exc
        existing = self.store.get(key)
        if existing is None:
            raise KeyNotFoundError()
        if last is not None and existing[1] != last:
            # The fence refusing: this holder's revision is stale, so the entry now belongs to
            # somebody else and must survive.
            raise KeyWrongLastSequenceError()
        del self.store[key]


# parity-exempt: minimal NatsClient stand-in exposing only kv_bucket(); the distributed-lock body touches only that surface, so full NatsClient parity would be over-mocking
class _FakeClient:
    """Minimal :class:`NatsClient` stand-in exposing only ``kv_bucket``.

    Replays the public surface the lock body touches; constructed with
    a pre-built fake KV so tests can assert against shared state.
    """

    def __init__(self, fake_kv: _FakeKv) -> None:
        self._fake_kv = fake_kv
        self.kv_bucket_calls: list[tuple[str, timedelta | None]] = []

    async def kv_bucket(
        self,
        *,
        name: str,
        ttl: timedelta | None = None,
        storage: str = "memory",
        create_if_missing: bool = True,
        history: int = 1,
    ) -> NatsKvBucket:
        del storage, create_if_missing, history
        self.kv_bucket_calls.append((name, ttl))
        return NatsKvBucket(
            client=None,  # type: ignore[arg-type]
            full_name=f"itest-{name}",
            kv=self._fake_kv,  # type: ignore[arg-type]
            ttl=ttl,
        )


@pytest.mark.asyncio
async def test_yields_immediately_when_client_is_none() -> None:
    """``client=None`` is the single-pod-dev no-op shortcut."""
    entered = False
    async with nats_distributed_lock(None, "job"):
        entered = True
    assert entered is True


@pytest.mark.asyncio
async def test_heartbeat_geq_ttl_raises_value_error() -> None:
    """``heartbeat`` must be strictly less than ``ttl`` or the lock
    would expire under a live holder."""
    fake_kv = _FakeKv()
    client = _FakeClient(fake_kv)
    with pytest.raises(ValueError, match="heartbeat"):
        async with nats_distributed_lock(
            client,  # type: ignore[arg-type]
            "job",
            ttl=timedelta(seconds=5),
            heartbeat=timedelta(seconds=5),
        ):
            pass


@pytest.mark.asyncio
async def test_acquire_and_release() -> None:
    """Body runs, key is deleted on exit, bucket name carries the default."""
    fake_kv = _FakeKv()
    client = _FakeClient(fake_kv)
    async with nats_distributed_lock(
        client,  # type: ignore[arg-type]
        "backup",
        ttl=timedelta(seconds=1),
        heartbeat=timedelta(milliseconds=10),
    ):
        # inside the lock, the key is owned by us
        assert "backup" in fake_kv.store
    assert fake_kv.delete_calls == ["backup"]
    assert client.kv_bucket_calls == [("scheduler-locks", timedelta(seconds=1))]


@pytest.mark.asyncio
async def test_lock_held_raises_when_key_exists() -> None:
    """``create`` returning conflict surfaces as ``LockHeld``."""
    fake_kv = _FakeKv()
    fake_kv.store["job"] = (b"1", 1)  # someone else holds it
    fake_kv.next_revision = 1
    client = _FakeClient(fake_kv)
    with pytest.raises(LockHeld, match="job"):
        async with nats_distributed_lock(
            client,  # type: ignore[arg-type]
            "job",
            ttl=timedelta(seconds=1),
            heartbeat=timedelta(milliseconds=10),
        ):
            pass  # pragma: no cover - never enters
    # we did NOT delete a key we never owned
    assert fake_kv.delete_calls == []


@pytest.mark.asyncio
async def test_create_kv_error_propagates() -> None:
    """A KV-create transport failure surfaces as ``KvError`` (distinct from ``LockHeld``)."""
    fake_kv = _FakeKv()
    fake_kv.fail_next_create = RuntimeError("nats broker died")
    client = _FakeClient(fake_kv)
    with pytest.raises(KvError):
        async with nats_distributed_lock(
            client,  # type: ignore[arg-type]
            "job",
            ttl=timedelta(seconds=1),
            heartbeat=timedelta(milliseconds=10),
        ):
            pass  # pragma: no cover


@pytest.mark.asyncio
async def test_heartbeat_refreshes_key() -> None:
    """The background heartbeat puts the key on cadence while the body runs."""
    fake_kv = _FakeKv()
    client = _FakeClient(fake_kv)
    async with nats_distributed_lock(
        client,  # type: ignore[arg-type]
        "job",
        ttl=timedelta(seconds=1),
        heartbeat=timedelta(milliseconds=30),
    ):
        await _wait_until(
            lambda: len(fake_kv.update_calls) >= 2,
            what="two heartbeat refreshes",
        )
    assert len(fake_kv.update_calls) >= 2
    # Every renewal rewrites the SAME value the acquire created: that value is
    # the holder's identity, and the release fences on it, so a heartbeat that
    # wrote anything else would hand the lock away mid-hold.
    created_token = fake_kv.create_calls[0][1]
    assert all(call[:2] == ("job", created_token) for call in fake_kv.update_calls)
    # And never unconditionally: a blind put is how a stalled holder overwrites its successor.
    assert fake_kv.put_calls == []


@pytest.mark.asyncio
async def test_body_exception_still_releases_lock() -> None:
    """An exception inside the body cancels the heartbeat and releases the key."""
    fake_kv = _FakeKv()
    client = _FakeClient(fake_kv)

    class Boom(Exception):
        pass

    with pytest.raises(Boom):
        async with nats_distributed_lock(
            client,  # type: ignore[arg-type]
            "job",
            ttl=timedelta(seconds=1),
            heartbeat=timedelta(milliseconds=20),
        ):
            raise Boom()
    assert fake_kv.delete_calls == ["job"]
    assert "job" not in fake_kv.store


@pytest.mark.asyncio
async def test_release_swallows_kverror_during_cleanup() -> None:
    """A delete failure during cleanup is logged at debug, not re-raised."""
    fake_kv = _FakeKv()
    fake_kv.fail_next_delete = RuntimeError("transient")
    client = _FakeClient(fake_kv)
    # the cleanup is best-effort: a transient delete failure must not
    # mask normal completion of the lock body.
    async with nats_distributed_lock(
        client,  # type: ignore[arg-type]
        "job",
        ttl=timedelta(seconds=1),
        heartbeat=timedelta(milliseconds=20),
    ):
        pass
    # delete was attempted exactly once
    assert fake_kv.delete_calls == ["job"]


@pytest.mark.asyncio
async def test_external_cancellation_during_body_cleans_up() -> None:
    """If the surrounding task is cancelled, the heartbeat task is reaped
    and the key is deleted."""
    fake_kv = _FakeKv()
    client = _FakeClient(fake_kv)
    started = asyncio.Event()
    cleanup_done = asyncio.Event()

    async def hold_lock() -> None:
        try:
            async with nats_distributed_lock(
                client,  # type: ignore[arg-type]
                "job",
                ttl=timedelta(seconds=1),
                heartbeat=timedelta(milliseconds=20),
            ):
                started.set()
                await asyncio.sleep(5.0)  # would block forever; cancelled below
        finally:
            cleanup_done.set()

    task = asyncio.create_task(hold_lock())
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert cleanup_done.is_set()
    assert fake_kv.delete_calls == ["job"]
    # heartbeat task should not leak as a pending task in the event loop
    pending = [t for t in asyncio.all_tasks() if "nats-lock-heartbeat" in (t.get_name() or "")]
    assert pending == []


@pytest.mark.asyncio
async def test_custom_bucket_name_threads_through() -> None:
    """Caller-supplied ``bucket_name`` is what we ask the client for."""
    fake_kv = _FakeKv()
    client = _FakeClient(fake_kv)
    async with nats_distributed_lock(
        client,  # type: ignore[arg-type]
        "k",
        bucket_name="custom-locks",
        ttl=timedelta(seconds=1),
        heartbeat=timedelta(milliseconds=10),
    ):
        pass
    assert client.kv_bucket_calls == [("custom-locks", timedelta(seconds=1))]


@pytest.mark.asyncio
async def test_ttl_mismatch_against_existing_bucket_raises() -> None:
    """A second caller against a cached bucket with a different ``ttl``
    raises ``ValueError`` rather than silently inheriting the first
    caller's TTL.

    Pins the Critic-flagged footgun: ``NatsClient.kv_bucket`` caches
    buckets by name and JetStream KV bucket TTL is fixed-at-creation,
    so passing a different ``ttl`` against an already-materialised
    bucket would silently use the old TTL. The lock now refuses
    instead of pretending the new TTL took effect.
    """
    fake_kv = _FakeKv()

    class _CachingClient:
        def __init__(self) -> None:
            self._cached: NatsKvBucket | None = None
            self.kv_bucket_calls: list[tuple[str, timedelta | None]] = []

        async def kv_bucket(
            self,
            *,
            name: str,
            ttl: timedelta | None = None,
            storage: str = "memory",
            create_if_missing: bool = True,
            history: int = 1,
        ) -> NatsKvBucket:
            del storage, create_if_missing, history
            self.kv_bucket_calls.append((name, ttl))
            # mimic NatsClient.kv_bucket: first-caller's ttl pins the
            # bucket; subsequent calls ignore the new ttl and return
            # the cached bucket.
            if self._cached is None:
                self._cached = NatsKvBucket(
                    client=None,  # type: ignore[arg-type]
                    full_name=f"itest-{name}",
                    kv=fake_kv,  # type: ignore[arg-type]
                    ttl=ttl,
                )
            return self._cached

    client = _CachingClient()

    # first caller pins ttl=10s on the bucket
    async with nats_distributed_lock(
        client,  # type: ignore[arg-type]
        "first",
        ttl=timedelta(seconds=10),
        heartbeat=timedelta(milliseconds=10),
    ):
        pass

    # second caller against the same default bucket passes ttl=30s.
    # the cached bucket still reports ttl=10s; the lock must reject
    # rather than silently use 10s.
    with pytest.raises(ValueError, match="bucket"):
        async with nats_distributed_lock(
            client,  # type: ignore[arg-type]
            "second",
            ttl=timedelta(seconds=30),
            heartbeat=timedelta(milliseconds=10),
        ):
            pass  # pragma: no cover - never enters


@pytest.mark.asyncio
async def test_ttl_match_against_existing_bucket_acquires() -> None:
    """Identical ``ttl`` against the cached bucket is fine: the check
    is mismatch-only, not first-time-only."""
    fake_kv = _FakeKv()

    class _CachingClient:
        def __init__(self) -> None:
            self._cached: NatsKvBucket | None = None
            self.kv_bucket_calls: list[tuple[str, timedelta | None]] = []

        async def kv_bucket(
            self,
            *,
            name: str,
            ttl: timedelta | None = None,
            storage: str = "memory",
            create_if_missing: bool = True,
            history: int = 1,
        ) -> NatsKvBucket:
            del storage, create_if_missing, history
            self.kv_bucket_calls.append((name, ttl))
            if self._cached is None:
                self._cached = NatsKvBucket(
                    client=None,  # type: ignore[arg-type]
                    full_name=f"itest-{name}",
                    kv=fake_kv,  # type: ignore[arg-type]
                    ttl=ttl,
                )
            return self._cached

    client = _CachingClient()

    async with nats_distributed_lock(
        client,  # type: ignore[arg-type]
        "first",
        ttl=timedelta(seconds=10),
        heartbeat=timedelta(milliseconds=10),
    ):
        pass

    # same ttl on second call: no raise, lock acquires normally
    async with nats_distributed_lock(
        client,  # type: ignore[arg-type]
        "second",
        ttl=timedelta(seconds=10),
        heartbeat=timedelta(milliseconds=10),
    ):
        pass
    assert fake_kv.delete_calls == ["first", "second"]


@pytest.mark.asyncio
async def test_heartbeat_failure_does_not_break_release() -> None:
    """A heartbeat-side broker failure is logged + swallowed; release proceeds."""
    fake_kv = _FakeKv()

    original_update = fake_kv.update
    attempted: list[str] = []

    async def failing_update(key: str, value: bytes, revision: int) -> int:
        # the heartbeat's compare-and-swap renewal; raise so the branch that
        # logs and retries fires. the attempt is recorded because the raise
        # means ``update_calls`` never grows, so it is the only evidence the
        # heartbeat actually ran.
        attempted.append(key)
        raise RuntimeError("broker down")

    fake_kv.update = failing_update  # type: ignore[assignment, method-assign]
    client = _FakeClient(fake_kv)
    async with nats_distributed_lock(
        client,  # type: ignore[arg-type]
        "job",
        ttl=timedelta(seconds=1),
        heartbeat=timedelta(milliseconds=20),
    ):
        await _wait_until(lambda: bool(attempted), what="the heartbeat to fail once")
    # release path still runs
    assert fake_kv.delete_calls == ["job"]
    # restore (defensive; the fake is per-test anyway)
    fake_kv.update = original_update  # type: ignore[method-assign]


class TestALockAStuckHolderCannotKeepForever:
    """One wedged pod must not starve a fleet, and a stale holder must not steal a lock.

    A holder that DIES stops heartbeating and the TTL hands the lock on. A holder that
    WEDGES keeps a perfectly healthy heartbeat task renewing a lock whose body makes no
    progress -- which is how one stuck pod blocked every other pod's tick, with the lock
    behaving exactly as designed.
    """

    @pytest.mark.asyncio
    async def test_renewal_stops_once_the_holder_has_held_too_long(self) -> None:
        """Past the maximum hold the heartbeat stops renewing and lets the TTL take over.

        :return: nothing
        :rtype: None
        """
        fake_kv = _FakeKv()
        client = _FakeClient(fake_kv)

        async with nats_distributed_lock(
            client,
            "wedged",
            heartbeat=timedelta(seconds=0.01),
            ttl=timedelta(seconds=1),
            cancel_on_loss=False,
            max_hold=timedelta(seconds=0),
        ) as hold:
            # a negative assertion cannot wait for its condition, so this
            # one stays a sleep: several heartbeat intervals of slack, and
            # a loaded loop only ever grants MORE of them.
            await asyncio.sleep(0.05)
            renewals_while_wedged = len(fake_kv.update_calls) + len(fake_kv.put_calls)

        assert renewals_while_wedged == 0, (
            "the heartbeat renewed a lock whose holder was past its maximum hold, which is "
            "what keeps a wedged pod's lock alive while every other pod waits"
        )
        # and the holder is TOLD: a body that keeps going believing it still holds the lock
        # is the damage the stop exists to bound.
        assert hold.lost.is_set()
        assert hold.lost_reason is LockLossReason.MAX_HOLD

    @pytest.mark.asyncio
    async def test_an_ordinary_holder_still_gets_its_heartbeats(self) -> None:
        """The negative half: the cap must not interrupt a healthy long body.

        Losing a lock a running body still believes it holds is a worse failure than the
        one the cap prevents, so this pins that the cap does not fire in normal use.

        :return: nothing
        :rtype: None
        """
        fake_kv = _FakeKv()
        client = _FakeClient(fake_kv)

        async with nats_distributed_lock(
            client, "healthy", heartbeat=timedelta(seconds=0.01), ttl=timedelta(seconds=1)
        ):
            await _wait_until(
                lambda: bool(fake_kv.update_calls),
                what="a heartbeat from a healthy holder",
            )

        assert fake_kv.update_calls, "a healthy holder inside the maximum hold stopped being renewed"

    @pytest.mark.asyncio
    async def test_a_stale_holder_does_not_delete_its_successors_lock(self) -> None:
        """The release is fenced on the revision this holder last wrote.

        Whenever the lock did not survive the body -- heartbeat died, or renewal stopped at
        the cap -- the TTL expires the key and another pod acquires it. An unconditional
        delete then removes the SUCCESSOR's lock, handing the same key to a third holder
        while the second still believes it owns it. That is a worse bug than the wedge, and
        it is the one a self-release would have introduced without this fence.

        :return: nothing
        :rtype: None
        """
        fake_kv = _FakeKv()
        client = _FakeClient(fake_kv)

        async with nats_distributed_lock(client, "handover", heartbeat=timedelta(seconds=30)):
            # Simulate the TTL expiring and a successor taking the key: same key, new revision.
            fake_kv.next_revision += 1
            fake_kv.store["handover"] = (b"successor", fake_kv.next_revision)
            successor_revision = fake_kv.next_revision

        assert fake_kv.store.get("handover") == (b"successor", successor_revision), (
            "the departing holder deleted a key that had already moved to another holder"
        )

    @pytest.mark.asyncio
    async def test_a_holder_that_kept_its_lock_still_releases_it(self) -> None:
        """The fence must not turn every ordinary release into a no-op.

        :return: nothing
        :rtype: None
        """
        fake_kv = _FakeKv()
        client = _FakeClient(fake_kv)

        async with nats_distributed_lock(client, "ordinary", heartbeat=timedelta(seconds=30)):
            pass

        assert "ordinary" not in fake_kv.store

    @pytest.mark.asyncio
    async def test_the_fence_follows_the_heartbeats_revision(self) -> None:
        """Each renewal writes a new revision, so the fence must track it, not the first one.

        Fencing on the acquisition revision alone would make every release after the first
        heartbeat a silent no-op -- the lock would then linger until its TTL on every run.

        :return: nothing
        :rtype: None
        """
        fake_kv = _FakeKv()
        client = _FakeClient(fake_kv)

        async with nats_distributed_lock(
            client, "renewed", heartbeat=timedelta(seconds=0.01), ttl=timedelta(seconds=1)
        ):
            await _wait_until(
                lambda: bool(fake_kv.update_calls),
                what="a heartbeat to renew the entry",
            )

        assert "renewed" not in fake_kv.store, (
            "the release was fenced on a stale revision, so a renewed lock was never cleaned up"
        )


# parity-exempt: KeyValue subset whose put lands the write and THEN suspends, so a cancellation can be delivered between the two; used only to force the release race
class _SlowAckKv(_FakeKv):
    """A KV whose renewal ``update`` writes and then waits for an acknowledgement.

    Real brokers behave this way -- the write lands and the ack travels back
    over a network the caller is suspended on -- and the in-memory
    :class:`_FakeKv` cannot express the gap because its ``update`` never
    suspends. Every cancellation-versus-write ordering question needs it.
    """

    def __init__(self) -> None:
        super().__init__()
        self.put_landed = asyncio.Event()
        self.release_ack = asyncio.Event()

    async def update(self, key: str, value: bytes, revision: int) -> int:
        self.update_calls.append((key, value, revision))
        existing = self.store.get(key)
        if existing is None or existing[1] != revision:
            raise KeyWrongLastSequenceError()
        self.next_revision += 1
        self.store[key] = (value, self.next_revision)
        self.put_landed.set()
        await self.release_ack.wait()
        return self.next_revision


class TestAHolderReleasesALockItRenewedMidFlight:
    """The release must survive a heartbeat cancelled between write and ack."""

    @pytest.mark.asyncio
    async def test_the_lock_does_not_outlive_a_body_that_exited_mid_renewal(self) -> None:
        """A body finishing while a renewal is in flight still releases the lock.

        The holder recorded the revision it had written by assigning the result
        of the renewal, so a cancellation delivered after the write landed but
        before that assignment left the holder one revision behind the entry it
        owned. The fenced delete then refused, and the lock sat there for its
        whole TTL with every other pod waiting -- the exact outcome the fence
        exists to prevent, reached from the other side.

        In production the gap is a network round trip, so any body finishing
        near a heartbeat boundary lands in it.

        :return: nothing
        :rtype: None
        """
        fake_kv = _SlowAckKv()
        client = _FakeClient(fake_kv)

        async with nats_distributed_lock(
            client,  # type: ignore[arg-type]
            "raced",
            heartbeat=timedelta(seconds=0.01),
            ttl=timedelta(seconds=5),
        ):
            await _wait_until(
                fake_kv.put_landed.is_set,
                what="a renewal write to land but not yet be acknowledged",
            )
        fake_kv.release_ack.set()
        await _wait_until(
            lambda: "raced" not in fake_kv.store,
            what="the lock to be released",
            timeout=1.0,
        )

    @pytest.mark.asyncio
    async def test_it_still_refuses_to_delete_a_successors_lock(self) -> None:
        """The fix must not weaken the fence into an unconditional delete.

        Identity, not sequence, is what the release checks -- so a key that
        moved to another holder survives however many revisions ago this holder
        last wrote.
        """
        fake_kv = _FakeKv()
        client = _FakeClient(fake_kv)

        async with nats_distributed_lock(
            client,  # type: ignore[arg-type]
            "handover",
            heartbeat=timedelta(seconds=30),
        ):
            fake_kv.next_revision += 1
            fake_kv.store["handover"] = (b"a-successors-token", fake_kv.next_revision)
            successor_revision = fake_kv.next_revision

        assert fake_kv.store.get("handover") == (b"a-successors-token", successor_revision), (
            "the departing holder deleted a key that had already moved to another holder"
        )


# ---------------------------------------------------------------------------
# losing the lock: compare-and-swap renewal, and telling the holder
# ---------------------------------------------------------------------------


# parity-exempt: KeyValue subset whose renewal read suspends until the test lets it continue, so another holder can take the key exactly where a real broker round trip leaves room
class _StallingKv(_FakeKv):
    """A KV whose ``get`` (the renewal's read) suspends until the test releases it.

    A holder that stalls -- a blocked event loop, a GC pause, a network partition -- sits
    somewhere between its reads and its writes while the world moves on. The in-memory
    fake never suspends, so it cannot put a successor's acquisition inside that gap; this
    one parks the renewal's read until the test has done exactly that.
    """

    def __init__(self) -> None:
        super().__init__()
        self.stall_reads = False
        self.read_parked = asyncio.Event()
        self.resume_read = asyncio.Event()

    async def get(self, key: str) -> _FakeEntry:
        if self.stall_reads:
            self.read_parked.set()
            await self.resume_read.wait()
        return await super().get(key)


def _successor_takes(fake_kv: _FakeKv, key: str) -> tuple[bytes, int]:
    """expire the holder's entry and let another pod acquire the key.

    :param fake_kv: the KV the lock lives in
    :ptype fake_kv: _FakeKv
    :param key: the lock key
    :ptype key: str
    :return: the successor's entry
    :rtype: tuple[bytes, int]
    """
    fake_kv.next_revision += 1
    fake_kv.store[key] = (b"the-successors-token", fake_kv.next_revision)
    return fake_kv.store[key]


class TestAHolderThatLostTheLockNeverTakesItBack:
    """Renewal is a compare-and-swap on this holder's token, never a blind write."""

    @pytest.mark.asyncio
    async def test_a_stalled_holder_does_not_overwrite_the_successor_that_acquired_after_expiry(self) -> None:
        fake_kv = _StallingKv()
        client = _FakeClient(fake_kv)

        async with nats_distributed_lock(
            client,  # type: ignore[arg-type]
            "stalled",
            heartbeat=timedelta(seconds=0.01),
            ttl=timedelta(seconds=1),
            cancel_on_loss=False,
        ) as hold:
            fake_kv.stall_reads = True
            await _wait_until(fake_kv.read_parked.is_set, what="the renewal to stall mid-read")
            successor = _successor_takes(fake_kv, "stalled")
            fake_kv.stall_reads = False
            fake_kv.resume_read.set()
            await asyncio.wait_for(hold.lost.wait(), timeout=5)

            assert fake_kv.store["stalled"] == successor, "the stalled holder overwrote its successor's lock"
            assert hold.lost_reason is LockLossReason.TAKEN
        assert fake_kv.store["stalled"] == successor, "the stalled holder released its successor's lock"

    @pytest.mark.asyncio
    async def test_a_successor_writing_between_the_read_and_the_swap_wins(self) -> None:
        """The swap is fenced on the revision just read, so a write in between is a loss, not a race."""
        fake_kv = _FakeKv()
        client = _FakeClient(fake_kv)
        original_get = fake_kv.get
        taken: list[tuple[bytes, int]] = []
        reads: list[str] = []

        async def _get_then_lose_it(key: str) -> _FakeEntry:
            reads.append(key)
            entry = await original_get(key)
            if not taken:
                taken.append(_successor_takes(fake_kv, key))
            return entry

        fake_kv.get = _get_then_lose_it  # type: ignore[method-assign]

        async with nats_distributed_lock(
            client,  # type: ignore[arg-type]
            "raced",
            heartbeat=timedelta(seconds=0.01),
            ttl=timedelta(seconds=1),
            cancel_on_loss=False,
        ) as hold:
            await asyncio.wait_for(hold.lost.wait(), timeout=5)
            reads_until_loss = len(reads)

        assert fake_kv.store["raced"] == taken[0]
        assert hold.lost_reason is LockLossReason.TAKEN
        assert reads_until_loss == 1, "the refused swap was taken for a renewal; the loss surfaced a heartbeat late"

    @pytest.mark.asyncio
    async def test_an_entry_that_expired_with_nobody_waiting_is_a_loss_too(self) -> None:
        fake_kv = _FakeKv()
        client = _FakeClient(fake_kv)

        async with nats_distributed_lock(
            client,  # type: ignore[arg-type]
            "expired",
            heartbeat=timedelta(seconds=0.01),
            ttl=timedelta(seconds=1),
            cancel_on_loss=False,
        ) as hold:
            del fake_kv.store["expired"]
            await asyncio.wait_for(hold.lost.wait(), timeout=5)

        assert hold.lost_reason is LockLossReason.EXPIRED
        assert "expired" not in fake_kv.store, "a holder that lost its entry wrote it back"

    @pytest.mark.asyncio
    async def test_a_lock_still_held_renews_and_is_never_reported_lost(self) -> None:
        fake_kv = _FakeKv()
        client = _FakeClient(fake_kv)

        async with nats_distributed_lock(
            client,  # type: ignore[arg-type]
            "held",
            heartbeat=timedelta(seconds=0.01),
            ttl=timedelta(seconds=1),
        ) as hold:
            await _wait_until(lambda: len(fake_kv.update_calls) >= 3, what="three renewals")
            hold.raise_if_lost()

        assert not hold.lost.is_set()
        assert hold.lost_reason is None


class TestTheHolderIsToldWhenTheLockIsLost:
    @pytest.mark.asyncio
    async def test_renewals_that_keep_failing_until_the_entry_may_have_expired_are_a_loss(self) -> None:
        fake_kv = _FakeKv()
        client = _FakeClient(fake_kv)

        async def _broker_down(key: str, value: bytes, revision: int) -> int:
            raise RuntimeError("broker down")

        fake_kv.update = _broker_down  # type: ignore[method-assign]

        async with nats_distributed_lock(
            client,  # type: ignore[arg-type]
            "unreachable",
            heartbeat=timedelta(seconds=0.02),
            ttl=timedelta(seconds=0.2),
            cancel_on_loss=False,
        ) as hold:
            await asyncio.wait_for(hold.lost.wait(), timeout=5)

        assert hold.lost_reason is LockLossReason.RENEWAL_FAILED

    @pytest.mark.asyncio
    async def test_one_failed_renewal_inside_the_ttl_is_retried_not_a_loss(self) -> None:
        """A single broker blip must not interrupt a healthy body while the entry cannot have expired."""
        fake_kv = _FakeKv()
        client = _FakeClient(fake_kv)
        original_update = fake_kv.update
        failures: list[int] = []

        async def _one_blip(key: str, value: bytes, revision: int) -> int:
            if not failures:
                failures.append(revision)
                raise RuntimeError("broker blip")
            return await original_update(key, value, revision)

        fake_kv.update = _one_blip  # type: ignore[method-assign]

        async with nats_distributed_lock(
            client,  # type: ignore[arg-type]
            "blip",
            heartbeat=timedelta(seconds=0.01),
            ttl=timedelta(seconds=1),
        ) as hold:
            await _wait_until(lambda: len(fake_kv.update_calls) >= 2, what="renewals to resume after the blip")

        assert failures
        assert not hold.lost.is_set()

    @pytest.mark.asyncio
    async def test_by_default_a_lost_lock_interrupts_the_body_and_raises_lock_lost(self) -> None:
        fake_kv = _FakeKv()
        client = _FakeClient(fake_kv)
        body_finished = False

        with pytest.raises(LockLost) as lost:
            async with nats_distributed_lock(
                client,  # type: ignore[arg-type]
                "interrupted",
                heartbeat=timedelta(seconds=0.01),
                ttl=timedelta(seconds=1),
            ):
                _successor_takes(fake_kv, "interrupted")
                await asyncio.sleep(5)
                body_finished = True

        assert not body_finished, "the body kept running after its lock was taken"
        assert lost.value.reason is LockLossReason.TAKEN
        assert lost.value.key == "interrupted"
        current = asyncio.current_task()
        assert current is not None and current.cancelling() == 0, "the loss left a cancellation request pending"

    @pytest.mark.asyncio
    async def test_an_outside_cancellation_is_still_a_cancellation(self) -> None:
        fake_kv = _FakeKv()
        client = _FakeClient(fake_kv)
        entered = asyncio.Event()

        async def _hold() -> None:
            async with nats_distributed_lock(
                client,  # type: ignore[arg-type]
                "cancelled",
                heartbeat=timedelta(seconds=0.01),
                ttl=timedelta(seconds=1),
            ):
                entered.set()
                await asyncio.sleep(5)

        task = asyncio.create_task(_hold())
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    @pytest.mark.asyncio
    async def test_raise_if_lost_names_the_reason(self) -> None:
        fake_kv = _FakeKv()
        client = _FakeClient(fake_kv)

        async with nats_distributed_lock(
            client,  # type: ignore[arg-type]
            "checked",
            heartbeat=timedelta(seconds=0.01),
            ttl=timedelta(seconds=1),
            cancel_on_loss=False,
            max_hold=timedelta(seconds=0),
        ) as hold:
            await asyncio.wait_for(hold.lost.wait(), timeout=5)
            with pytest.raises(LockLost) as lost:
                hold.raise_if_lost()

        assert lost.value.reason is LockLossReason.MAX_HOLD

    @pytest.mark.asyncio
    async def test_the_single_pod_no_op_yields_a_hold_that_is_never_lost(self) -> None:
        async with nats_distributed_lock(None, "dev") as hold:
            assert isinstance(hold, LockHold)
            hold.raise_if_lost()
        assert not hold.lost.is_set()


class TestTheMaximumHoldIsTheCallersToSet:
    """a caller whose body legitimately runs longer than the default, or that wants a wedge
    noticed sooner, sets its own ``max_hold`` rather than living with one number."""

    def test_the_default_is_six_hours(self) -> None:
        default = inspect.signature(nats_distributed_lock).parameters["max_hold"].default
        assert default == timedelta(hours=6)

    @pytest.mark.asyncio
    async def test_a_callers_max_hold_interrupts_a_wedged_body(self) -> None:
        fake_kv = _FakeKv()
        client = _FakeClient(fake_kv)

        with pytest.raises(LockLost) as lost:
            async with nats_distributed_lock(
                client,  # type: ignore[arg-type]
                "wedged-by-its-own-limit",
                heartbeat=timedelta(seconds=0.01),
                ttl=timedelta(seconds=1),
                max_hold=timedelta(seconds=0.03),
            ):
                await asyncio.sleep(5)

        assert lost.value.reason is LockLossReason.MAX_HOLD

    @pytest.mark.asyncio
    async def test_a_negative_max_hold_is_refused(self) -> None:
        client = _FakeClient(_FakeKv())

        with pytest.raises(ValueError, match="max_hold"):
            async with nats_distributed_lock(
                client,  # type: ignore[arg-type]
                "never",
                max_hold=timedelta(seconds=-1),
            ):
                pass
