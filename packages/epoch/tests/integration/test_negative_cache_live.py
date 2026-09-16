"""Integration test: generation-stamped negative caching against a real broker.

What a fake cannot prove, proved here with two ``NatsClient`` connections standing in for two pods,
each running the real cache-invalidation listener and the real epoch-bucket generation source:

- a peer listener deleting a writer's fresh L2 value cannot revive an absence recorded before the
  write -- the reviewed failure, end to end;
- an L2 marker carries a server-side lifetime and leaves the bucket when it passes;
- an expired row is replaced by a marker through a compare-and-swap that carries a lifetime, which
  the wrapper sends itself because nats-py's public update takes none;
- a collections bucket created before per-entry lifetimes were allowed is enabled in place by the
  declaring opener, keeping its existing entries -- and a file-backed one stays file-backed;
- peers in one L2 scope, each running the listener, keep a write-behind compare-and-swap counter
  rather than evicting it on every broadcast.

Uses the session-scoped ``nats_container`` fixture; a checkout without docker skips cleanly.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any, ClassVar, Literal

import pytest
from nats.js.api import StorageType
from sqlalchemy import Column, DateTime, MetaData, String, Table

from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections.base import BaseCollection
from threetears.core.collections.flush import WriteBuffer
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.entities.base import BaseEntity
from threetears.epoch import EpochGenerationSource
from threetears.nats import NatsClient, set_default_namespace
from threetears.nats.kv import build_kv_stream_config

pytestmark = pytest.mark.integration

_SCOPE = "live-principal"
_TABLE = "live_denylist"


def _metadata() -> MetaData:
    metadata = MetaData()
    Table(
        _TABLE,
        metadata,
        Column("id", String(255), primary_key=True),
        Column("reason", String(255)),
        Column("expires_at", DateTime(timezone=True)),
        Column("date_created", DateTime(timezone=True)),
        Column("date_updated", DateTime(timezone=True)),
    )
    return metadata


class _Entry(BaseEntity):
    primary_key_field = "id"


class _LiveDenylist(BaseCollection[_Entry]):
    """a negative-caching, expiring collection over an in-process L3 shared by both pods."""

    datetime_columns: ClassVar[frozenset[str]] = frozenset({"expires_at", "date_created", "date_updated"})
    negative_cache_max_age: ClassVar[timedelta | None] = timedelta(seconds=2)
    expires_at_column: ClassVar[str | None] = "expires_at"

    def __init__(self, registry: CollectionRegistry, rows: dict[str, dict[str, Any]]) -> None:
        self._rows = rows
        self.fetches = 0
        super().__init__(registry, DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables=""))

    @property
    def table_name(self) -> str:
        return _TABLE

    @property
    def entity_class(self) -> type[_Entry]:
        return _Entry

    async def fetch_from_store(self, entity_id: Any) -> dict[str, Any] | None:
        self.fetches += 1
        row = self._rows.get(str(entity_id))
        return dict(row) if row is not None else None

    async def save_to_store(
        self, data: dict[str, Any], original_timestamp: datetime | None = None, *, conn: Any = None
    ) -> int:
        self._rows[str(data["id"])] = dict(data)
        return 1

    async def delete_from_store(self, entity_id: Any) -> None:
        self._rows.pop(str(entity_id), None)

    def serialize(self, data: dict[str, Any]) -> bytes:
        return json.dumps(data, default=str).encode("utf-8")

    def deserialize(self, data: bytes) -> dict[str, Any]:
        row: dict[str, Any] = json.loads(data)
        return row


async def _pod(nc: NatsClient, rows: dict[str, dict[str, Any]]) -> tuple[_LiveDenylist, CollectionRegistry]:
    """one pod: its own L1 and registry and listener, the shared broker and L3."""
    l1 = SQLiteBackend(db_name=f"live_negcache_{uuid.uuid4().hex[:8]}")
    l1.initialize(_metadata())
    registry = CollectionRegistry()
    registry.configure(l1_backend=l1, l2_client=nc, l3_pool=object(), kv_key_scope=_SCOPE)  # type: ignore[arg-type]
    registry.set_generation_source(EpochGenerationSource(nc))
    collection = _LiveDenylist(registry, rows)
    await registry.start_invalidation_listener(nc)
    return collection, registry


async def _connect(url: str, name: str, namespace: str) -> NatsClient:
    return await NatsClient.connect(nats_url=url, nats_subject_namespace=namespace, client_name=name)


async def _write(collection: _LiveDenylist, entity_id: str, *, expires_at: datetime | None = None) -> None:
    entity = collection.create({"id": entity_id, "reason": "revoked", "expires_at": expires_at})
    await collection.save_entity(entity)


async def test_a_peer_listener_deleting_the_writers_value_cannot_revive_an_absence(nats_container: str) -> None:
    namespace = f"negc{uuid.uuid4().hex[:6]}"
    set_default_namespace(namespace)
    rows: dict[str, dict[str, Any]] = {}
    async with (
        await _connect(nats_container, "reader", namespace) as reader_nc,
        await _connect(nats_container, "writer", namespace) as writer_nc,
    ):
        reader, reader_registry = await _pod(reader_nc, rows)
        writer, writer_registry = await _pod(writer_nc, rows)
        try:
            assert await reader.get("tok") is None  # absence recorded in the reader's L1 and in L2
            await _write(writer, "tok")
            # the reader's listener receives the broadcast and deletes the writer's fresh L2 value
            await asyncio.sleep(0.5)
            key = reader.l2_key("tok")
            bucket = await reader_nc.kv_bucket(name="collections")
            assert await bucket.get(key=key) is None, "precondition: the listener evicted the writer's value"
            fetched = await reader.get("tok")
            assert fetched is not None, "an absence recorded before the write answered after it"
        finally:
            await reader_registry.stop_invalidation_listener()
            await writer_registry.stop_invalidation_listener()


async def test_a_marker_leaves_the_bucket_when_its_lifetime_passes(nats_container: str) -> None:
    namespace = f"negc{uuid.uuid4().hex[:6]}"
    set_default_namespace(namespace)
    async with await _connect(nats_container, "pod", namespace) as nc:
        coll, registry = await _pod(nc, {})
        try:
            assert await coll.get("never") is None
            bucket = await nc.kv_bucket(name="collections")
            key = coll.l2_key("never")
            assert await bucket.get(key=key) is not None
            await asyncio.sleep(3.5)
            assert await bucket.get(key=key) is None, "a marker nobody reads again stayed in the bucket"
        finally:
            await registry.stop_invalidation_listener()


async def test_an_expired_row_is_replaced_by_a_marker_that_carries_a_lifetime(nats_container: str) -> None:
    namespace = f"negc{uuid.uuid4().hex[:6]}"
    set_default_namespace(namespace)
    rows: dict[str, dict[str, Any]] = {}
    async with await _connect(nats_container, "pod", namespace) as nc:
        coll, registry = await _pod(nc, rows)
        try:
            await _write(coll, "old", expires_at=datetime.now(UTC) - timedelta(seconds=1))
            cold, cold_registry = await _pod(nc, rows)
            try:
                assert await cold.get("old") is None
                bucket = await nc.kv_bucket(name="collections")
                key = cold.l2_key("old")
                raw = await bucket.get(key=key)
                assert raw is not None and raw.startswith(b"\x00threetears.collections.absent\x00"), (
                    "the expired row was not replaced by a marker through the compare-and-swap"
                )
                await asyncio.sleep(3.5)
                assert await bucket.get(key=key) is None, "the compare-and-swapped marker carried no lifetime"
            finally:
                await cold_registry.stop_invalidation_listener()
        finally:
            await registry.stop_invalidation_listener()


async def test_a_legacy_collections_bucket_is_enabled_in_place_and_keeps_its_entries(nats_container: str) -> None:
    namespace = f"negc{uuid.uuid4().hex[:6]}"
    set_default_namespace(namespace)
    async with await _connect(nats_container, "pod", namespace) as nc:
        full_name = f"{namespace}-collections"
        legacy = build_kv_stream_config(
            bucket=full_name, ttl_seconds=0, history=1, storage_type=StorageType.MEMORY, direct=None
        )
        legacy.allow_msg_ttl = False
        js = nc.jetstream_context()
        await js.add_stream(legacy)
        raw_kv = await js.key_value(full_name)
        await raw_kv.put("pre-existing", b"kept")

        coll, registry = await _pod(nc, {})
        try:
            assert await coll.get("never") is None  # opens the bucket as its declarer, then writes a marker
            info = await js.stream_info(f"KV_{full_name}")
            assert info.config.allow_msg_ttl is True, "the declaring opener did not enable per-entry lifetimes"
            assert (await raw_kv.get("pre-existing")).value == b"kept"
            bucket = await nc.kv_bucket(name="collections")
            assert await bucket.get(key=coll.l2_key("never")) is not None, "the marker write was refused"
        finally:
            await registry.stop_invalidation_listener()


async def test_a_legacy_file_backed_bucket_is_enabled_in_place_and_stays_file_backed(nats_container: str) -> None:
    # the server refuses to change a stream's storage; an update that asked for memory would fail
    # the whole reconcile, and with it every open of the bucket.
    namespace = f"negc{uuid.uuid4().hex[:6]}"
    set_default_namespace(namespace)
    async with await _connect(nats_container, "pod", namespace) as nc:
        full_name = f"{namespace}-collections"
        legacy = build_kv_stream_config(
            bucket=full_name, ttl_seconds=0, history=1, storage_type=StorageType.FILE, direct=None
        )
        legacy.allow_msg_ttl = False
        js = nc.jetstream_context()
        await js.add_stream(legacy)
        try:
            coll, registry = await _pod(nc, {})
            try:
                assert await coll.get("never") is None  # opens as declarer asking for memory, then writes a marker
                info = await js.stream_info(f"KV_{full_name}")
                assert info.config.allow_msg_ttl is True, "the reconcile did not enable per-entry lifetimes"
                assert info.config.storage == StorageType.FILE
                bucket = await nc.kv_bucket(name="collections")
                assert await bucket.get(key=coll.l2_key("never")) is not None, "the marker write was refused"
            finally:
                await registry.stop_invalidation_listener()
        finally:
            await js.delete_stream(f"KV_{full_name}")


class _LiveCounter(_LiveDenylist):
    """a write-behind attempt counter: L3 holds nothing until a flush, so L2 is the only current count."""

    negative_cache_max_age: ClassVar[timedelta | None] = None
    l3_write_policy: ClassVar[Literal["synchronous", "write_behind"] | None] = "write_behind"

    def __init__(self, registry: CollectionRegistry, rows: dict[str, dict[str, Any]]) -> None:
        self._rows = rows
        self.fetches = 0
        BaseCollection.__init__(
            self,
            registry,
            DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables=""),
            write_buffer=WriteBuffer(),
        )


async def test_listening_peers_in_one_scope_keep_a_compare_and_swap_counter(nats_container: str) -> None:
    namespace = f"negc{uuid.uuid4().hex[:6]}"
    set_default_namespace(namespace)
    rows: dict[str, dict[str, Any]] = {}

    def _increment(row: dict[str, Any] | None) -> tuple[Literal["upsert", "delete", "noop"], dict[str, Any] | None]:
        count = 0 if row is None else int(row["reason"])
        return "upsert", {"id": "acct", "reason": str(count + 1), "expires_at": None}

    async with (
        await _connect(nats_container, "first", namespace) as first_nc,
        await _connect(nats_container, "second", namespace) as second_nc,
    ):
        pods = []
        for nc in (first_nc, second_nc):
            l1 = SQLiteBackend(db_name=f"live_counter_{uuid.uuid4().hex[:8]}")
            l1.initialize(_metadata())
            registry = CollectionRegistry()
            registry.configure(l1_backend=l1, l2_client=nc, l3_pool=object(), kv_key_scope=_SCOPE)  # type: ignore[arg-type]
            await registry.start_invalidation_listener(nc)
            pods.append((_LiveCounter(registry, rows), registry))
        try:
            for n in range(6):
                await pods[n % 2][0].l2_cas_mutate("acct", _increment)
                await asyncio.sleep(0.1)  # let the other pod's listener act on the broadcast
            bucket = await first_nc.kv_bucket(name="collections")
            raw = await bucket.get(key=pods[0][0].l2_key("acct"))
            assert raw is not None, "a peer's listener evicted the compare-and-swap key"
            assert json.loads(raw)["reason"] == "6"
        finally:
            for _, registry in pods:
                await registry.stop_invalidation_listener()
