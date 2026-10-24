"""A released lease leaves nothing in its bucket once its marker lapses, and never costs a live lease.

The shared ``leases`` bucket keeps one revision per key and no expiry, and a KV delete publishes a marker. A
host that takes a lease per unique key (a build lock per computed answer) left one marker per key forever, and
the bounded bucket filled until every lease write on the platform was refused. The release's marker now lapses
(a few seconds); the bucket's own clock moves it on here.
"""

from __future__ import annotations

from datetime import timedelta

from threetears.core.coordination import KVLease
from threetears.core.testing.kv import FakeKvBucket

#: enough unique keys that their markers alone would overfill the bucket below several times over
_CYCLES = 400

#: a bound that holds a few dozen live leases and their markers, but not ``_CYCLES`` markers
_MAX_BYTES = 16 * 1024


#: past any lifetime a release gives its marker
_MARKER_LAPSED = timedelta(minutes=1)


def _lapse(bucket: FakeKvBucket) -> None:
    bucket.advance_clock(_MARKER_LAPSED)


async def test_acquiring_and_releasing_unique_keys_never_fills_the_bucket() -> None:
    bucket = FakeKvBucket("ns-leases", max_bytes=_MAX_BYTES)
    lease = KVLease(None, bucket=bucket, pod_id="reports-pod")

    for n in range(_CYCLES):
        handle = await lease.acquire(f"enr_answers.answer-{n}", ttl_seconds=30, max_wait_seconds=0)
        await handle.release()
        if n % 10 == 9:
            _lapse(bucket)

    _lapse(bucket)
    assert bucket.stored_messages == 0, f"{bucket.stored_messages} messages left after every lease was released"
    assert bucket.stored_bytes == 0


async def test_a_held_lease_survives_every_other_release_and_its_lapse() -> None:
    bucket = FakeKvBucket("ns-leases", max_bytes=_MAX_BYTES)
    lease = KVLease(None, bucket=bucket, pod_id="reports-pod")
    held = await lease.acquire("refresh.enr", ttl_seconds=30, max_wait_seconds=0)

    for n in range(_CYCLES):
        handle = await lease.acquire(f"enr_answers.answer-{n}", ttl_seconds=30, max_wait_seconds=0)
        await handle.release()
        if n % 10 == 9:
            _lapse(bucket)
            await held.refresh()

    _lapse(bucket)
    assert await lease.holder("refresh.enr") == "reports-pod"
    assert bucket.stored_messages == 1
    await held.release()


# parity-exempt: one FakeKvBucket that records each delete a release sends
class _RecordingDeletes(FakeKvBucket):
    """records every delete: on a real bucket a delete at revision 0 is unfenced (nats-py drops ``last=0``), so a
    release that sends one could take a successor's lease, whatever this double's own fence makes of it."""

    def __init__(self) -> None:
        super().__init__("ns-leases")
        self.deletes: list[tuple[str, int | None]] = []

    async def delete(self, *, key: str, revision: int | None = None, marker_ttl: timedelta | None = None) -> bool:
        self.deletes.append((key, revision))
        return await super().delete(key=key, revision=revision, marker_ttl=marker_ttl)


async def test_a_release_with_no_revision_to_fence_on_leaves_the_successors_lease() -> None:
    bucket = _RecordingDeletes()
    lease = KVLease(None, bucket=bucket, pod_id="reports-pod")
    stale = await lease.acquire("refresh.enr", ttl_seconds=30, max_wait_seconds=0)
    # the key went and this holder's other handle took it: same holder, a revision the stale one never saw
    await bucket.delete(key="refresh.enr", revision=stale.revision)
    successor = await lease.acquire("refresh.enr", ttl_seconds=30, max_wait_seconds=0)
    stale.revision = 0
    bucket.deletes.clear()

    await stale.release()
    _lapse(bucket)

    assert bucket.deletes == [], f"a release with no revision to fence on sent {bucket.deletes}"
    assert await lease.holder("refresh.enr") == "reports-pod"
    await successor.refresh()


# parity-exempt: one FakeKvBucket whose first create loses as a real create over a lapsing marker does
class _MarkerLapsedUnderCreate(FakeKvBucket):
    """the first create loses with the key holding nothing: nats-py's create found a released key's marker,
    and the marker lapsed before its swap against the marker's revision landed."""

    def __init__(self) -> None:
        super().__init__("ns-leases")
        self.creates = 0

    async def create(self, *, key: str, value: bytes, ttl: timedelta | None = None) -> int | None:
        self.creates += 1
        if self.creates == 1:
            return None
        return await super().create(key=key, value=value, ttl=ttl)


async def test_an_acquire_that_loses_to_a_lapsing_marker_takes_the_free_key_at_once() -> None:
    bucket = _MarkerLapsedUnderCreate()
    lease = KVLease(None, bucket=bucket, pod_id="reports-pod")

    handle = await lease.acquire("enr_answers.answer-1", ttl_seconds=30, max_wait_seconds=0)

    assert await lease.holder("enr_answers.answer-1") == "reports-pod"
    await handle.release()
