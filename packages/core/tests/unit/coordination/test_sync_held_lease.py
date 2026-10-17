"""``KVLease.hold_sync``: a lease held across synchronous work, renewed on the shared loop.

A CLI or batch script holds a lease for a ``with`` block while its own thread does the work; the
renewal is :class:`HeldLease`'s, running on :mod:`threetears.core.sync`'s loop. These tests drive it
from a plain thread over a real ``KVLease`` and an in-memory bucket, as such a caller does.

Unlike ``test_held_lease.py`` they run on the REAL clock, because the point is a renewal that keeps
going while the caller's thread is busy elsewhere. The margins are wide (renewals every 0.1s against
a two-second TTL) so a loaded runner cannot make a healthy lease read as lost.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator

import pytest

from threetears.core import sync
from threetears.core.coordination import KVLease, LeaseLossReason, LeaseLost, LeaseUnavailable, SyncHeldLease
from threetears.core.testing import drain_and_shutdown_bridge
from threetears.core.testing.kv import FakeNatsClient

_BUCKET = "held"
_TTL = 2
_RENEW = 0.1


@pytest.fixture(autouse=True)
def _fresh_bridge() -> Iterator[None]:
    """stop the shared loop after each test, so none inherits another's renewals."""
    yield
    drain_and_shutdown_bridge()


def _lease(client: FakeNatsClient, pod: str) -> KVLease:
    return KVLease(nats_client=client, bucket_name=_BUCKET, pod_id=pod)  # type: ignore[arg-type]


def _entry(client: FakeNatsClient, key: str) -> bytes | None:
    async def read() -> bytes | None:
        entry = await (await client.kv_bucket(name=_BUCKET)).get_entry(key=key)
        return None if entry is None else entry[0]

    return sync.run(read)


def _take_over(client: FakeNatsClient, key: str) -> None:
    """rewrite the entry's holder by compare-and-swap, as a stale reclaim by another pod leaves it."""

    async def take() -> None:
        bucket = await client.kv_bucket(name=_BUCKET)
        entry = await bucket.get_entry(key=key)
        assert entry is not None
        value, revision = entry
        assert await bucket.update(key=key, value=value.replace(b"pod-a", b"pod-b"), revision=revision) is not None

    sync.run(take)


def test_a_lease_is_held_for_the_block_and_freed_after() -> None:
    client = FakeNatsClient()
    with _lease(client, "pod-a").hold_sync("job", ttl=_TTL, renew_every=_RENEW) as held:
        assert isinstance(held, SyncHeldLease)
        assert held.held
        assert held.key == "job"
        with pytest.raises(LeaseUnavailable):
            _lease(client, "pod-b").hold_sync("job", ttl=_TTL, renew_every=_RENEW)
    assert not held.held
    assert held.lost_reason is None, "a release is not a loss"
    assert _entry(client, "job") is None
    held.check()  # released, not lost: nothing to raise


def test_it_is_renewed_while_the_callers_thread_is_busy() -> None:
    client = FakeNatsClient()
    with _lease(client, "pod-a").hold_sync("job", ttl=_TTL, renew_every=_RENEW) as held:
        before = _entry(client, "job")
        time.sleep(0.5)  # the caller's thread works; the shared loop renews
        assert _entry(client, "job") != before, "no renewal reached the bucket while the caller worked"
        assert held.held
        held.check()


def test_a_takeover_sets_lost_and_check_raises() -> None:
    client = FakeNatsClient()
    reasons: list[LeaseLossReason] = []
    held = _lease(client, "pod-a").hold_sync("job", ttl=_TTL, renew_every=_RENEW, on_lost=reasons.append)
    try:
        _take_over(client, "job")
        assert held.lost.wait(timeout=5), "the takeover was never noticed"
        assert not held.held
        assert held.lost_reason is LeaseLossReason.TAKEN
        assert reasons == [LeaseLossReason.TAKEN]
        with pytest.raises(LeaseLost) as raised:
            held.check()
        assert raised.value.reason is LeaseLossReason.TAKEN
    finally:
        held.release()
    surviving = _entry(client, "job")
    assert surviving is not None and b"pod-b" in surviving, "releasing a lost lease must leave the new owner's entry"


def test_the_body_failing_still_releases() -> None:
    client = FakeNatsClient()
    with pytest.raises(ZeroDivisionError):
        with _lease(client, "pod-a").hold_sync("job", ttl=_TTL, renew_every=_RENEW):
            _ = 1 / 0
    assert _entry(client, "job") is None


def test_release_is_idempotent() -> None:
    client = FakeNatsClient()
    held = _lease(client, "pod-a").hold_sync("job", ttl=_TTL, renew_every=_RENEW)
    held.release()
    held.release()
    assert not held.held


def test_waiting_for_a_held_key_ends_when_its_holder_lets_go() -> None:
    client = FakeNatsClient()
    first = _lease(client, "pod-a").hold_sync("job", ttl=_TTL, renew_every=_RENEW)
    threading.Timer(0.3, first.release).start()
    with _lease(client, "pod-b").hold_sync("job", ttl=_TTL, renew_every=_RENEW, max_wait_seconds=10) as second:
        assert second.held
        assert not first.held


async def test_inside_a_running_loop_it_is_refused() -> None:
    with pytest.raises(RuntimeError, match="await the work instead"):
        _lease(FakeNatsClient(), "pod-a").hold_sync("job", ttl=_TTL, renew_every=_RENEW)
