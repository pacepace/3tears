"""Integration test: ``watch_prefix`` against a real nats-server.

A watch over a family of keys: their current state first, a marker once it has all arrived, then
every change; keys outside the prefix never; and a true picture again after the server lost the
bucket's contents, which leaves no delete marker behind.

Uses the session-scoped ``nats_container`` fixture; a checkout without docker skips cleanly.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncGenerator
from contextlib import aclosing
from datetime import timedelta

import pytest

from threetears.nats import NatsClient, set_default_namespace
from threetears.nats.kv_watch import KvKeyUpdate

pytestmark = pytest.mark.integration

_WAIT = 10.0


async def _next(watch: AsyncGenerator[KvKeyUpdate | None]) -> KvKeyUpdate | None:
    """the watch's next item, within a bound.

    :param watch: the watch
    :ptype watch: AsyncGenerator[KvKeyUpdate | None]
    :return: the next item
    :rtype: KvKeyUpdate | None
    """
    return await asyncio.wait_for(anext(watch), _WAIT)


async def _until_caught_up(watch: AsyncGenerator[KvKeyUpdate | None]) -> list[KvKeyUpdate]:
    """every update before the next caught-up marker.

    :param watch: the watch
    :ptype watch: AsyncGenerator[KvKeyUpdate | None]
    :return: the updates
    :rtype: list[KvKeyUpdate]
    """
    updates: list[KvKeyUpdate] = []
    while (item := await _next(watch)) is not None:
        updates.append(item)
    return updates


async def test_a_prefix_watch_delivers_state_then_changes(nats_container: str) -> None:
    namespace = f"wp{uuid.uuid4().hex[:6]}"
    set_default_namespace(namespace)
    async with await NatsClient.connect(
        nats_url=nats_container, nats_subject_namespace=namespace, client_name="watch-prefix"
    ) as nc:
        bucket = await nc.kv_bucket(name="pointers")
        await bucket.put(key="snap.TX", value=b"1")
        await bucket.put(key="snap.CA", value=b"4")
        await bucket.put(key="other.TX", value=b"9")
        await bucket.put(key="snap.DE", value=b"2")
        assert await bucket.delete(key="snap.DE")
        async with aclosing(bucket.watch_prefix(prefix="snap.")) as watch:
            state = await _until_caught_up(watch)
            assert {(u.key, u.value) for u in state} == {("snap.TX", b"1"), ("snap.CA", b"4")}
            await bucket.put(key="other.CA", value=b"x")
            await bucket.put(key="snap.TX", value=b"2")
            changed = await _next(watch)
            assert changed is not None and (changed.key, changed.value) == ("snap.TX", b"2")
            assert await bucket.delete(key="snap.CA")
            removed = await _next(watch)
            assert removed is not None and removed.key == "snap.CA" and removed.deleted


async def test_an_empty_prefix_is_caught_up_at_once(nats_container: str) -> None:
    namespace = f"wp{uuid.uuid4().hex[:6]}"
    set_default_namespace(namespace)
    async with await NatsClient.connect(
        nats_url=nats_container, nats_subject_namespace=namespace, client_name="watch-prefix-empty"
    ) as nc:
        bucket = await nc.kv_bucket(name="pointers")
        async with aclosing(bucket.watch_prefix(prefix="snap.")) as watch:
            assert await _next(watch) is None
            await bucket.put(key="snap.TX", value=b"1")
            first = await _next(watch)
            assert first is not None and first.key == "snap.TX"


async def test_a_wiped_bucket_reads_as_deleted_keys_once_the_watch_recovers(nats_container: str) -> None:
    """what a memory-storage restart does: the bucket's stream, and the consumer on it, are gone."""
    namespace = f"wp{uuid.uuid4().hex[:6]}"
    set_default_namespace(namespace)
    async with (
        await NatsClient.connect(
            nats_url=nats_container, nats_subject_namespace=namespace, client_name="watch-prefix-wiped"
        ) as nc,
        await NatsClient.connect(
            nats_url=nats_container, nats_subject_namespace=namespace, client_name="wiper"
        ) as wiper,
    ):
        bucket = await nc.ensure_kv_bucket(name="pointers", owns_bucket=True)
        await bucket.put(key="snap.TX", value=b"1")
        async with aclosing(
            bucket.watch_prefix(prefix="snap.", heartbeat=timedelta(seconds=0.5), retry=timedelta(seconds=0.2))
        ) as watch:
            assert [u.key for u in await _until_caught_up(watch)] == ["snap.TX"]
            await wiper.jetstream_context().delete_stream(f"KV_{bucket.name}")
            # the declaration brings the bucket back, empty; the watch replaces its consumer
            await nc.reconnect()
            gone = await _next(watch)
            assert gone is not None and gone.key == "snap.TX" and gone.deleted
            assert await _next(watch) is None


@pytest.mark.parametrize("prefix", ["snap", "sn*.", "snap.>", "a b."])
async def test_a_prefix_must_be_literal_and_end_on_a_token(nats_container: str, prefix: str) -> None:
    namespace = f"wp{uuid.uuid4().hex[:6]}"
    set_default_namespace(namespace)
    async with await NatsClient.connect(
        nats_url=nats_container, nats_subject_namespace=namespace, client_name="watch-prefix-bad"
    ) as nc:
        bucket = await nc.kv_bucket(name="pointers")
        with pytest.raises(ValueError):
            await anext(bucket.watch_prefix(prefix=prefix))


async def test_a_key_rewritten_at_the_same_revision_after_a_wipe_is_delivered(nats_container: str) -> None:
    """a wiped bucket restarts its sequence: a new value can carry the revision the old one had."""
    namespace = f"wp{uuid.uuid4().hex[:6]}"
    set_default_namespace(namespace)
    async with (
        await NatsClient.connect(nats_url=nats_container, nats_subject_namespace=namespace, client_name="wp-seq") as nc,
        await NatsClient.connect(
            nats_url=nats_container, nats_subject_namespace=namespace, client_name="wiper"
        ) as wiper,
    ):
        bucket = await nc.ensure_kv_bucket(name="pointers", owns_bucket=True)
        first_revision = await bucket.put(key="snap.TX", value=b"old")
        async with aclosing(
            bucket.watch_prefix(prefix="snap.", heartbeat=timedelta(seconds=1), retry=timedelta(seconds=0.2))
        ) as watch:
            assert [(u.key, u.value) for u in await _until_caught_up(watch)] == [("snap.TX", b"old")]
            # wiped and written again before the watch notices: the new value lands on revision 1 again
            await wiper.jetstream_context().delete_stream(f"KV_{bucket.name}")
            again = await wiper.ensure_kv_bucket(name="pointers", owns_bucket=True)
            assert await again.put(key="snap.TX", value=b"new") == first_revision
            seen: list[tuple[str, bytes | None]] = []
            deadline = asyncio.get_running_loop().time() + 20
            while (("snap.TX", b"new") not in seen) and asyncio.get_running_loop().time() < deadline:
                item = await asyncio.wait_for(anext(watch), 20)
                if item is not None:
                    seen.append((item.key, item.value))
            assert ("snap.TX", b"new") in seen, f"the rewritten value was never delivered: {seen}"
