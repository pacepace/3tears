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
from datetime import timedelta

import pytest

from threetears.nats import LockHeld, LockLossReason, LockLost, NatsClient, nats_distributed_lock, set_default_namespace

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
