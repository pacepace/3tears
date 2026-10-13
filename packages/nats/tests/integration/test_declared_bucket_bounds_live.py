"""Integration test: a declared bucket's byte bound, and a declaration withdrawn.

A bucket a pod writes into is bounded by its declarer, so one pod can fill only its own bucket and
never the server's memory store; and a declarer that withdraws a bucket (its pod was deleted) deletes
it and stops putting it back after a reconnect.

Uses the session-scoped ``nats_container`` fixture; a checkout without docker skips cleanly.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest
from nats.js.errors import NotFoundError

from threetears.nats import KvError, NatsClient, ObjectStoreError, set_default_namespace

pytestmark = pytest.mark.integration


def _namespace() -> str:
    namespace = f"bnd{uuid.uuid4().hex[:8]}"
    set_default_namespace(namespace)
    return namespace


async def test_a_bounded_kv_bucket_refuses_writes_past_its_bound_and_nothing_else(nats_container: str) -> None:
    namespace = _namespace()
    async with await NatsClient.connect(
        nats_url=nats_container, nats_subject_namespace=namespace, client_name="bounds"
    ) as nc:
        bounded = await nc.ensure_kv_bucket(name="pointers", owns_bucket=True, max_bytes=4096)
        neighbour = await nc.ensure_kv_bucket(name="neighbour", owns_bucket=True)
        info = await nc.jetstream_context().stream_info(f"KV_{bounded.name}")
        assert info.config.max_bytes == 4096
        refused = False
        for index in range(64):
            try:
                await bounded.put(key=f"k{index}", value=b"x" * 200)
            except KvError:
                refused = True
                break
        assert refused, "a write past the bucket's bound was accepted"
        # the bucket keeps what it held, and every other bucket still takes writes
        assert await bounded.get(key="k0") == b"x" * 200
        await neighbour.put(key="k", value=b"y" * 10_000)
        assert await neighbour.get(key="k") == b"y" * 10_000
        # its owner may raise the bound in place
        await nc.ensure_kv_bucket(name="pointers", owns_bucket=True, max_bytes=65536)
        assert (await nc.jetstream_context().stream_info(f"KV_{bounded.name}")).config.max_bytes == 65536


async def test_a_withdrawn_declaration_is_deleted_and_not_put_back(nats_container: str) -> None:
    namespace = _namespace()
    async with (
        await NatsClient.connect(nats_url=nats_container, nats_subject_namespace=namespace, client_name="owner") as nc,
        await NatsClient.connect(nats_url=nats_container, nats_subject_namespace=namespace, client_name="look") as look,
    ):
        store = await nc.ensure_object_store(name="objects", max_bytes=1024 * 1024)
        pointers = await nc.ensure_kv_bucket(name="pointers", owns_bucket=True)
        await nc.delete_object_store(name="objects")
        await nc.delete_kv_bucket(name="pointers")
        js = look.jetstream_context()
        for stream in (store.stream, f"KV_{pointers.name}"):
            with pytest.raises(NotFoundError):
                await js.stream_info(stream)
        await nc.reconnect()
        await asyncio.sleep(2.0)
        for stream in (store.stream, f"KV_{pointers.name}"):
            with pytest.raises(NotFoundError):
                await js.stream_info(stream)
        # deleting what is already gone is not an error
        await nc.delete_object_store(name="objects")
        await nc.delete_kv_bucket(name="pointers")


async def test_an_object_store_refuses_a_rollup(nats_container: str) -> None:
    """a rollup header purges a subject's (or the whole stream's) history: a purge by publish."""
    namespace = _namespace()
    async with await NatsClient.connect(
        nats_url=nats_container, nats_subject_namespace=namespace, client_name="rollup"
    ) as nc:
        store = await nc.ensure_object_store(name="objects", max_bytes=1024 * 1024)
        await store.put("kept", b"x" * 1000)
        info = await nc.jetstream_context().stream_info(store.stream)
        assert not info.config.allow_rollup_hdrs
        with pytest.raises(Exception):  # noqa: B017 -- nats-py's APIError for a refused rollup
            await nc.jetstream_context().publish(f"$O.{store.name}.C.other", b"y", headers={"Nats-Rollup": "all"})
        assert await store.get("kept") == b"x" * 1000
        assert ObjectStoreError is not None


async def test_a_declaration_no_longer_wanted_is_not_put_back(nats_container: str) -> None:
    """a declarer whose owner went away (its pod was deleted on another replica) stops restoring it."""
    namespace = _namespace()
    wanted = {"objects": True, "pointers": True}

    async def want(name: str) -> bool:
        return wanted[name]

    async with (
        await NatsClient.connect(nats_url=nats_container, nats_subject_namespace=namespace, client_name="guard") as nc,
        await NatsClient.connect(nats_url=nats_container, nats_subject_namespace=namespace, client_name="wipe") as wipe,
    ):
        store = await nc.ensure_object_store(
            name="objects", max_bytes=1024 * 1024, still_wanted=lambda: want("objects")
        )
        kept = await nc.ensure_object_store(name="kept", max_bytes=1024 * 1024)
        pointers = await nc.ensure_kv_bucket(name="pointers", owns_bucket=True, still_wanted=lambda: want("pointers"))
        wanted["objects"] = False
        wanted["pointers"] = False
        js = wipe.jetstream_context()
        for stream in (store.stream, kept.stream, f"KV_{pointers.name}"):
            await js.delete_stream(stream)
        await nc.reconnect()
        deadline = asyncio.get_running_loop().time() + 15
        while True:
            try:
                await js.stream_info(kept.stream)
                break
            except NotFoundError:
                assert asyncio.get_running_loop().time() < deadline, "the wanted bucket did not come back"
                await asyncio.sleep(0.1)
        await asyncio.sleep(1.0)
        for stream in (store.stream, f"KV_{pointers.name}"):
            with pytest.raises(NotFoundError):
                await js.stream_info(stream)
