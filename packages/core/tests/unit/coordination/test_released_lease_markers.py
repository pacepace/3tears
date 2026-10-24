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
