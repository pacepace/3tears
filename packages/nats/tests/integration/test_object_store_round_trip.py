"""Integration test: the Object Store wrapper against a real nats-server.

What a fake can only assert: chunks and metadata land on the stream in the NATS Object Store's wire
shape (so the ``nats`` CLI and nats-py read them), an object larger than one chunk comes back whole
with its digest checked, a name is written once, a deleted object leaves the listing, the declarer's
sweep removes chunks no object names, and a declared bucket comes back after a restart wiped it.

Uses the session-scoped ``nats_container`` fixture; a checkout without docker skips cleanly.
"""

from __future__ import annotations

import os
import uuid
from datetime import timedelta

import pytest

from threetears.nats import (
    NatsClient,
    ObjectExistsError,
    ObjectNotFoundError,
    ObjectStoreNotFoundError,
    set_default_namespace,
)

pytestmark = pytest.mark.integration


def _namespace() -> str:
    """a fresh namespace per test, so buckets never collide across tests on the shared server.

    :return: the namespace
    :rtype: str
    """
    namespace = f"obj{uuid.uuid4().hex[:8]}"
    set_default_namespace(namespace)
    return namespace


async def test_an_object_larger_than_a_chunk_comes_back_whole(nats_container: str) -> None:
    namespace = _namespace()
    async with await NatsClient.connect(
        nats_url=nats_container, nats_subject_namespace=namespace, client_name="objects"
    ) as nc:
        store = await nc.ensure_object_store(name="objects", max_bytes=64 * 1024 * 1024)
        data = os.urandom(3 * 1024 * 1024 + 17)
        info = await store.put("tables/TX/7", data)
        assert info.size == len(data)
        assert info.chunks > 1, "the object should span several chunks"
        assert info.digest.startswith("SHA-256=")
        assert await store.get("tables/TX/7") == data
        described = await store.info("tables/TX/7")
        assert described is not None
        assert (described.size, described.chunks, described.digest) == (info.size, info.chunks, info.digest)


async def test_an_empty_object_round_trips(nats_container: str) -> None:
    namespace = _namespace()
    async with await NatsClient.connect(
        nats_url=nats_container, nats_subject_namespace=namespace, client_name="objects"
    ) as nc:
        store = await nc.ensure_object_store(name="objects", max_bytes=1024 * 1024)
        info = await store.put("empty", b"")
        assert (info.size, info.chunks) == (0, 0)
        assert await store.get("empty") == b""


async def test_the_wire_shape_is_the_nats_object_store(nats_container: str) -> None:
    """nats-py's own Object Store reads what the wrapper wrote, so operator tools see it too."""
    namespace = _namespace()
    async with await NatsClient.connect(
        nats_url=nats_container, nats_subject_namespace=namespace, client_name="objects"
    ) as nc:
        store = await nc.ensure_object_store(name="objects", max_bytes=8 * 1024 * 1024)
        data = os.urandom(700_000)
        await store.put("a/b", data)
        stock = await nc.jetstream_context().object_store(store.name)
        assert (await stock.get("a/b")).data == data


async def test_a_name_is_written_once(nats_container: str) -> None:
    namespace = _namespace()
    async with await NatsClient.connect(
        nats_url=nats_container, nats_subject_namespace=namespace, client_name="objects"
    ) as nc:
        store = await nc.ensure_object_store(name="objects", max_bytes=1024 * 1024)
        await store.put("once", b"first")
        with pytest.raises(ObjectExistsError):
            await store.put("once", b"second")
        assert await store.get("once") == b"first"


async def test_an_absent_object_is_answered_as_absent(nats_container: str) -> None:
    namespace = _namespace()
    async with await NatsClient.connect(
        nats_url=nats_container, nats_subject_namespace=namespace, client_name="objects"
    ) as nc:
        store = await nc.ensure_object_store(name="objects", max_bytes=1024 * 1024)
        assert await store.info("missing") is None
        with pytest.raises(ObjectNotFoundError):
            await store.get("missing")


async def test_listing_names_the_live_objects_under_a_prefix(nats_container: str) -> None:
    namespace = _namespace()
    async with await NatsClient.connect(
        nats_url=nats_container, nats_subject_namespace=namespace, client_name="objects"
    ) as nc:
        store = await nc.ensure_object_store(name="objects", max_bytes=1024 * 1024)
        assert await store.list_objects() == []
        for name in ("snap/TX/1", "snap/TX/2", "snap/CA/1", "other/x"):
            await store.put(name, name.encode())
        assert sorted(i.name for i in await store.list_objects(prefix="snap/")) == [
            "snap/CA/1",
            "snap/TX/1",
            "snap/TX/2",
        ]
        assert await store.delete("snap/TX/1")
        assert sorted(i.name for i in await store.list_objects(prefix="snap/")) == ["snap/CA/1", "snap/TX/2"]
        assert await store.info("snap/TX/1") is None
        with pytest.raises(ObjectNotFoundError):
            await store.get("snap/TX/1")
        assert not await store.delete("snap/TX/1"), "deleting an absent object reports that it was absent"


async def test_the_declarers_sweep_removes_chunks_no_object_names(nats_container: str) -> None:
    namespace = _namespace()
    async with await NatsClient.connect(
        nats_url=nats_container, nats_subject_namespace=namespace, client_name="objects"
    ) as nc:
        store = await nc.ensure_object_store(name="objects", max_bytes=8 * 1024 * 1024)
        await store.put("kept", os.urandom(300_000))
        # a writer that died between its chunks and its metadata leaves chunks no object names
        await nc.jetstream_context().publish(f"$O.{store.name}.C.orphanednuid", b"x" * 1000)
        before = await store.bytes_held()
        # the default grace keeps a put in progress; a fresh orphan is kept until it is old enough
        assert await store.purge_orphan_chunks() == 0
        assert await store.purge_orphan_chunks(older_than=timedelta(0)) == 1
        assert await store.bytes_held() < before
        assert len(await store.get("kept")) == 300_000


async def test_binding_an_undeclared_bucket_is_answered_as_absent(nats_container: str) -> None:
    namespace = _namespace()
    async with await NatsClient.connect(
        nats_url=nats_container, nats_subject_namespace=namespace, client_name="objects"
    ) as nc:
        with pytest.raises(ObjectStoreNotFoundError):
            await nc.object_store(name="never-declared")
        await nc.ensure_object_store(name="declared", max_bytes=1024 * 1024)
        bound = await nc.object_store(name="declared")
        await bound.put("x", b"y")
        assert await bound.get("x") == b"y"


async def test_a_declaration_is_bounded_and_on_memory(nats_container: str) -> None:
    namespace = _namespace()
    async with await NatsClient.connect(
        nats_url=nats_container, nats_subject_namespace=namespace, client_name="objects"
    ) as nc:
        store = await nc.ensure_object_store(name="objects", max_bytes=2 * 1024 * 1024)
        info = await nc.jetstream_context().stream_info(store.stream)
        assert info.config.max_bytes == 2 * 1024 * 1024
        assert getattr(info.config.storage, "value", info.config.storage) == "memory"
        assert info.config.allow_direct
        # declaring again, larger, reconciles the live bucket in place and keeps what it holds
        await store.put("x", b"y")
        again = await nc.ensure_object_store(name="objects", max_bytes=4 * 1024 * 1024)
        assert (await nc.jetstream_context().stream_info(again.stream)).config.max_bytes == 4 * 1024 * 1024
        assert await again.get("x") == b"y"


async def test_a_declared_bucket_comes_back_after_a_restart_wiped_it(nats_container: str) -> None:
    """what a memory-storage restart does -- the stream is deleted -- then nats-py's real reconnect path."""
    import asyncio

    from nats.js.errors import NotFoundError

    namespace = _namespace()
    async with (
        await NatsClient.connect(nats_url=nats_container, nats_subject_namespace=namespace, client_name="owner") as nc,
        await NatsClient.connect(
            nats_url=nats_container, nats_subject_namespace=namespace, client_name="wiper"
        ) as wiper,
    ):
        store = await nc.ensure_object_store(name="objects", max_bytes=1024 * 1024)
        await store.put("x", b"y")
        await wiper.jetstream_context().delete_stream(store.stream)
        await nc.reconnect()
        deadline = asyncio.get_running_loop().time() + 15.0
        while True:
            try:
                await wiper.jetstream_context().stream_info(store.stream)
                break
            except NotFoundError:
                assert asyncio.get_running_loop().time() < deadline, "the declared bucket did not come back"
                await asyncio.sleep(0.1)
        # back, and empty: the restart took what it held
        assert await store.info("x") is None
        await store.put("x", b"again")
        assert await store.get("x") == b"again"
