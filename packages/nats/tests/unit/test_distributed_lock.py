"""Unit tests for :func:`threetears.nats.nats_distributed_lock`.

Substitutes a fake KV bucket + a minimal NatsClient stand-in so the
lock lifecycle (acquire, heartbeat, release, cancellation cleanup) can
be exercised without a live JetStream broker. Real-broker round-trips
live in ``tests/integration/test_distributed_lock_round_trip.py``.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta

import pytest
from nats.js.errors import KeyNotFoundError, KeyWrongLastSequenceError

from threetears.nats import LockHeld, NatsKvBucket, nats_distributed_lock
from threetears.nats.errors import KvError


class _FakeEntry:
    def __init__(self, value: bytes | None, revision: int | None) -> None:
        self.value = value
        self.revision = revision


class _FakeKv:
    """Fake nats-py KeyValue handle backing a NatsKvBucket.

    Tracks put/create/delete counts so heartbeat-refresh and cleanup
    behavior can be asserted directly.
    """

    def __init__(self) -> None:
        self.store: dict[str, tuple[bytes, int]] = {}
        self.next_revision = 0
        self.put_calls: list[tuple[str, bytes]] = []
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
        existing = self.store.get(key)
        if existing is None or existing[1] != revision:
            raise KeyWrongLastSequenceError()
        self.next_revision += 1
        self.store[key] = (value, self.next_revision)
        return self.next_revision

    async def delete(self, key: str) -> None:
        self.delete_calls.append(key)
        if self.fail_next_delete is not None:
            exc = self.fail_next_delete
            self.fail_next_delete = None
            raise exc
        if key not in self.store:
            raise KeyNotFoundError()
        del self.store[key]


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
        storage: str = "file",
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
        await asyncio.sleep(0.12)
    # at least 2 refreshes in 120ms with 30ms cadence
    assert len(fake_kv.put_calls) >= 2
    assert all(call == ("job", b"1") for call in fake_kv.put_calls)


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
async def test_heartbeat_failure_does_not_break_release() -> None:
    """A heartbeat-side broker failure is logged + swallowed; release proceeds."""
    fake_kv = _FakeKv()

    original_put = fake_kv.put

    async def failing_put(key: str, value: bytes) -> int:
        # first put after acquire is the heartbeat refresh; raise so the
        # branch that logs + lets the TTL expire fires.
        raise RuntimeError("broker down")

    fake_kv.put = failing_put  # type: ignore[assignment, method-assign]
    client = _FakeClient(fake_kv)
    async with nats_distributed_lock(
        client,  # type: ignore[arg-type]
        "job",
        ttl=timedelta(seconds=1),
        heartbeat=timedelta(milliseconds=20),
    ):
        # give the heartbeat enough time to fail at least once
        await asyncio.sleep(0.05)
    # release path still runs
    assert fake_kv.delete_calls == ["job"]
    # restore (defensive; the fake is per-test anyway)
    fake_kv.put = original_put  # type: ignore[method-assign]
