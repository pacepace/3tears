"""Integration test: ``nats_distributed_lock`` against a real NATS broker.

Exercises the actual JetStream KV ``create``-as-CAS semantics + TTL
expiry that the in-process unit tests cannot validate (the fake KV
ignores TTL). Uses the canonical session-scoped ``nats_container``
fixture from :mod:`threetears.core.testing.fixtures` -- a fresh
checkout without docker skips cleanly via the fixture's
``check_docker_available`` gate.
"""

from __future__ import annotations

import asyncio
import dataclasses
from datetime import timedelta

import pytest
from nats.js.api import StorageType

from threetears.nats import (
    LockHeld,
    LockLossReason,
    LockLost,
    NatsClient,
    NatsKvBucket,
    nats_distributed_lock,
    set_default_namespace,
)
from threetears.nats.kv import build_kv_stream_config

pytestmark = pytest.mark.integration


async def test_acquire_release_round_trip(nats_container: str) -> None:
    """A held-and-released lock can be re-acquired immediately."""
    set_default_namespace("locktest")
    async with await NatsClient.connect(
        nats_url=nats_container,
        nats_subject_namespace="locktest",
        client_name="lock-rr",
    ) as nc:
        async with nats_distributed_lock(
            nc,
            "round-trip",
            bucket_name="rt-locks",
            ttl=timedelta(seconds=2),
            heartbeat=timedelta(milliseconds=500),
        ):
            pass
        # second acquisition works because the first released cleanly
        async with nats_distributed_lock(
            nc,
            "round-trip",
            bucket_name="rt-locks",
            ttl=timedelta(seconds=2),
            heartbeat=timedelta(milliseconds=500),
        ):
            pass


async def test_concurrent_acquire_one_wins(nats_container: str) -> None:
    """Two concurrent acquires of the same key: one body runs, the other raises LockHeld."""
    set_default_namespace("locktest")
    async with await NatsClient.connect(
        nats_url=nats_container,
        nats_subject_namespace="locktest",
        client_name="lock-concurrent",
    ) as nc:
        wins: list[str] = []
        losses: list[str] = []
        first_in_lock = asyncio.Event()
        release_first = asyncio.Event()

        async def holder(name: str, hold: bool) -> None:
            try:
                async with nats_distributed_lock(
                    nc,
                    "contended",
                    bucket_name="contended-locks",
                    ttl=timedelta(seconds=3),
                    heartbeat=timedelta(milliseconds=500),
                ):
                    wins.append(name)
                    if hold:
                        first_in_lock.set()
                        await release_first.wait()
            except LockHeld:
                losses.append(name)

        first = asyncio.create_task(holder("first", hold=True))
        await first_in_lock.wait()
        # second attempt while the first holder is still inside the body
        await holder("second", hold=False)
        release_first.set()
        await first
        assert wins == ["first"]
        assert losses == ["second"]


async def test_ttl_expires_orphaned_lock(nats_container: str) -> None:
    """If a holder dies without cleanup, the TTL expires the key so another claimer can win."""
    set_default_namespace("locktest")
    async with await NatsClient.connect(
        nats_url=nats_container,
        nats_subject_namespace="locktest",
        client_name="lock-ttl",
    ) as nc:
        bucket = await nc.kv_bucket(name="orphan-locks", ttl=timedelta(seconds=1))
        # Simulate a dead holder: put the key directly + don't refresh.
        acquired = await bucket.create(key="orphan", value=b"1")
        assert acquired is not None

        # Immediately trying to re-acquire under the lock helper should fail.
        with pytest.raises(LockHeld):
            async with nats_distributed_lock(
                nc,
                "orphan",
                bucket_name="orphan-locks",
                ttl=timedelta(seconds=1),
                heartbeat=timedelta(milliseconds=200),
            ):
                pass  # pragma: no cover

        # After TTL expiry the key auto-deletes and the lock is winnable.
        # 1s ttl + a small grace window for the broker reaper.
        await asyncio.sleep(1.5)
        async with nats_distributed_lock(
            nc,
            "orphan",
            bucket_name="orphan-locks",
            ttl=timedelta(seconds=1),
            heartbeat=timedelta(milliseconds=200),
        ):
            pass


async def test_a_held_lock_renews_by_compare_and_swap_past_its_ttl(nats_container: str) -> None:
    """Renewal on a real broker: the revision the renewal reads is the one its swap is fenced on."""
    set_default_namespace("locktest")
    async with await NatsClient.connect(
        nats_url=nats_container,
        nats_subject_namespace="locktest",
        client_name="lock-renew",
    ) as nc:
        async with nats_distributed_lock(
            nc,
            "long-body",
            bucket_name="renew-locks",
            ttl=timedelta(seconds=1),
            heartbeat=timedelta(milliseconds=200),
        ) as hold:
            await asyncio.sleep(2.5)  # well past the ttl: only renewals keep the entry alive
            hold.raise_if_lost()
            with pytest.raises(LockHeld):
                async with nats_distributed_lock(
                    nc,
                    "long-body",
                    bucket_name="renew-locks",
                    ttl=timedelta(seconds=1),
                    heartbeat=timedelta(milliseconds=200),
                ):
                    pass  # pragma: no cover


async def test_a_holder_whose_key_changed_hands_stops_and_leaves_the_successor_alone(nats_container: str) -> None:
    """The stalled-holder race on a real broker: the key expires, a second pod takes it, the first wakes."""
    set_default_namespace("locktest")
    async with await NatsClient.connect(
        nats_url=nats_container,
        nats_subject_namespace="locktest",
        client_name="lock-handover",
    ) as nc:
        bucket = await nc.kv_bucket(name="handover-locks", ttl=timedelta(seconds=2))
        body_finished = False

        with pytest.raises(LockLost) as lost:
            async with nats_distributed_lock(
                nc,
                "handover",
                bucket_name="handover-locks",
                ttl=timedelta(seconds=2),
                heartbeat=timedelta(milliseconds=200),
            ):
                # what the stall looks like from the broker: the entry is gone and another
                # pod's create has won the key.
                assert await bucket.delete(key="handover")
                assert await bucket.create(key="handover", value=b"the-successors-token") is not None
                await asyncio.sleep(5)
                body_finished = True

        assert not body_finished
        assert lost.value.reason is LockLossReason.TAKEN
        entry = await bucket.get_entry(key="handover")
        assert entry is not None and entry[0] == b"the-successors-token", (
            "the first holder overwrote or released the successor's lock"
        )


# ---------------------------------------------------------------------------
# rolling upgrade, on a real broker: old raw-token entries and lease envelopes in one bucket
# ---------------------------------------------------------------------------

#: what releases up to 0.66 stored as a lock's value: the holder's ``secrets.token_hex(16)``. An
#: all-digit token is the one a JSON parser accepts (as a number), so it is the sharper case.
_OLD_TOKEN = b"12345678901234567890123456789012"


async def _old_acquire(bucket: NatsKvBucket, key: str) -> bool:
    """the pre-KVLease lock's acquisition: put-if-absent of its raw token."""
    return await bucket.create(key=key, value=_OLD_TOKEN) is not None


async def _old_release(bucket: NatsKvBucket, key: str) -> None:
    """the pre-KVLease lock's release: delete only while the entry still carries its token."""
    entry = await bucket.get_entry(key=key)
    if entry is not None and entry[0] == _OLD_TOKEN:
        await bucket.delete(key=key, revision=entry[1])


async def test_an_old_replicas_entry_holds_off_a_new_replica_until_it_is_released(nats_container: str) -> None:
    set_default_namespace("locktest")
    async with await NatsClient.connect(
        nats_url=nats_container,
        nats_subject_namespace="locktest",
        client_name="lock-upgrade-old-first",
    ) as nc:
        # the bucket exactly as the old lock declared it
        bucket = await nc.kv_bucket(name="upgrade-locks", ttl=timedelta(seconds=30))
        assert await _old_acquire(bucket, "tick")

        with pytest.raises(LockHeld):
            async with nats_distributed_lock(nc, "tick", bucket_name="upgrade-locks", ttl=timedelta(seconds=30)):
                pass  # pragma: no cover - never enters
        entry = await bucket.get_entry(key="tick")
        assert entry is not None and entry[0] == _OLD_TOKEN, "the new replica touched an old replica's held lock"

        await _old_release(bucket, "tick")
        async with nats_distributed_lock(nc, "tick", bucket_name="upgrade-locks", ttl=timedelta(seconds=30)):
            pass


async def test_a_new_replicas_entry_holds_off_an_old_replica(nats_container: str) -> None:
    set_default_namespace("locktest")
    async with await NatsClient.connect(
        nats_url=nats_container,
        nats_subject_namespace="locktest",
        client_name="lock-upgrade-new-first",
    ) as nc:
        bucket = await nc.kv_bucket(name="upgrade-locks-2", ttl=timedelta(seconds=30))
        async with nats_distributed_lock(nc, "tick", bucket_name="upgrade-locks-2", ttl=timedelta(seconds=30)):
            held = await bucket.get_entry(key="tick")
            assert held is not None
            assert not await _old_acquire(bucket, "tick"), "an old replica acquired a lock a new replica holds"
            await _old_release(bucket, "tick")
            assert await bucket.get_entry(key="tick") == held, "an old replica released a new replica's lock"
        assert await _old_acquire(bucket, "tick"), "the new replica's release left its entry behind"


async def test_a_bucket_an_old_release_created_is_used_as_it_stands(nats_container: str) -> None:
    """A bucket created before per-entry TTLs existed: bucket-wide ``max_age``, no ``allow_msg_ttl``.

    The lock writes no per-entry TTL, so it holds, renews past the TTL and releases on that bucket,
    and its bucket-wide expiry is left exactly as it was.
    """
    set_default_namespace("locktest")
    async with await NatsClient.connect(
        nats_url=nats_container,
        nats_subject_namespace="locktest",
        client_name="lock-upgrade-legacy-bucket",
    ) as nc:
        js = nc.jetstream_context()
        # the KV stream shape with per-entry TTLs refused, as buckets created before they existed
        await js.add_stream(
            dataclasses.replace(
                build_kv_stream_config(
                    bucket="locktest-legacy-locks",
                    ttl_seconds=2,
                    history=1,
                    storage_type=StorageType.MEMORY,
                    direct=None,
                ),
                allow_msg_ttl=False,
            )
        )
        before = await js.stream_info("KV_locktest-legacy-locks")
        assert not before.config.allow_msg_ttl

        async with nats_distributed_lock(
            nc, "legacy", bucket_name="legacy-locks", ttl=timedelta(seconds=2), heartbeat=timedelta(milliseconds=400)
        ) as hold:
            await asyncio.sleep(3)  # past the bucket's max_age: only renewals keep the entry
            hold.raise_if_lost()
        after = await js.stream_info("KV_locktest-legacy-locks")
        assert after.config.max_age == before.config.max_age
        async with nats_distributed_lock(
            nc, "legacy", bucket_name="legacy-locks", ttl=timedelta(seconds=2), heartbeat=timedelta(milliseconds=400)
        ):
            pass
