"""A row read from L3 never replaces a newer value in L2, on any collection.

A read that misses L2 fetches the row from L3 and seeds L2 with it. When a writer's
``save_entity`` commits to L3 and puts its row into L2 between that fetch and the seed, an
unconditional put lands the OLDER row over the newer one, and every reader on every replica is then
served it until the next write or the entry's lifetime -- the write was correct and every reader is
wrong. The same holds for ``reload_entity``, which refreshes L2 from L3 explicitly.

The race lives at the L3 read's suspension point, so the store below can hold a read after it has
taken its snapshot: exactly a response still in flight while a writer commits.

What this pins:

- a read racing a ``save_entity`` never leaves L2 older than the save, and answers with the newer
  row it lost to;
- a ``reload_entity`` racing a ``save_entity`` never leaves L2 older than the save;
- a read that finds nothing in L2 still seeds it, and a reload still refreshes an L2 value the L3
  row is newer than (a synchronous table, whose L3 is never behind L2).
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime
from typing import Any, ClassVar

import pytest
from sqlalchemy import Column, DateTime, MetaData, String, Table

from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections.base import BaseCollection
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.entities.base import BaseEntity
from threetears.core.testing.kv import FakeKvBucket, FakeNatsClient

_SCOPE = "seed-principal"
_TABLE = "profiles"
_ID = "p-1"
_KEY = f"{_SCOPE}.{_TABLE}.{_ID}"


def _metadata() -> MetaData:
    metadata = MetaData()
    Table(
        _TABLE,
        metadata,
        Column("id", String(255), primary_key=True),
        Column("name", String(255)),
        Column("date_created", DateTime(timezone=True)),
        Column("date_updated", DateTime(timezone=True)),
    )
    return metadata


class _Row(BaseEntity):
    primary_key_field = "id"


class _Store:
    """an in-process L3 whose next read can be held after it has taken its snapshot."""

    def __init__(self) -> None:
        self.rows: dict[str, dict[str, Any]] = {}
        self.held = asyncio.Event()
        self._fetch_holds: list[asyncio.Event] = []

    def hold_next_fetch(self) -> asyncio.Event:
        release = asyncio.Event()
        self._fetch_holds.append(release)
        return release

    async def fetch(self, entity_id: Any) -> dict[str, Any] | None:
        row = self.rows.get(str(entity_id))
        snapshot = dict(row) if row is not None else None
        if self._fetch_holds:
            release = self._fetch_holds.pop(0)
            self.held.set()
            await release.wait()
        return snapshot


class _Profiles(BaseCollection[_Row]):
    """an ordinary three-tier collection: no compare-and-swap, synchronous L3 writes."""

    datetime_columns: ClassVar[frozenset[str]] = frozenset({"date_created", "date_updated"})

    def __init__(self, registry: CollectionRegistry, store: _Store) -> None:
        self._store = store
        super().__init__(registry, DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables=""))

    @property
    def table_name(self) -> str:
        return _TABLE

    @property
    def entity_class(self) -> type[_Row]:
        return _Row

    async def fetch_from_store(self, entity_id: Any) -> dict[str, Any] | None:
        return await self._store.fetch(entity_id)

    async def save_to_store(
        self, data: dict[str, Any], original_timestamp: datetime | None = None, *, conn: Any = None
    ) -> int:
        self._store.rows[str(data["id"])] = dict(data)
        return 1

    async def delete_from_store(self, entity_id: Any) -> None:
        self._store.rows.pop(str(entity_id), None)

    def serialize(self, data: dict[str, Any]) -> bytes:
        return json.dumps(data, default=str).encode("utf-8")

    def deserialize(self, data: bytes) -> dict[str, Any]:
        row: dict[str, Any] = json.loads(data)
        return row


def _replica(nats: FakeNatsClient, store: _Store) -> _Profiles:
    l1 = SQLiteBackend(db_name=f"seed_{uuid.uuid4().hex[:8]}")
    l1.initialize(_metadata())
    registry = CollectionRegistry()
    registry.configure(l1_backend=l1, l2_client=nats, l3_pool=object(), kv_key_scope=_SCOPE)  # type: ignore[arg-type]
    return _Profiles(registry, store)


async def _l2_name(nats: FakeNatsClient) -> str | None:
    bucket: FakeKvBucket = await nats.kv_bucket(name="collections")
    raw = await bucket.get(key=_KEY)
    return None if raw is None else str(json.loads(raw)["name"])


async def _rename(writer: _Profiles, name: str) -> None:
    entity = await writer.get(_ID)
    assert entity is not None
    entity.set_data({**entity.to_dict(), "name": name})
    await writer.save_entity(entity)


class TestAReadNeverRegressesL2:
    @pytest.mark.asyncio
    async def test_a_read_racing_a_save_never_leaves_l2_older_than_the_save(self) -> None:
        nats, store = FakeNatsClient(), _Store()
        store.rows[_ID] = {"id": _ID, "name": "old"}
        reader, writer = _replica(nats, store), _replica(nats, store)
        await writer.ensure(_ID)  # the writer's own copy, so its rename reads nothing from L3
        (await nats.kv_bucket(name="collections")).wipe()  # L2 lost the key; the reader must seed it

        release = store.hold_next_fetch()
        store.held.clear()
        read = asyncio.create_task(reader.ensure(_ID))
        await store.held.wait()  # the reader has L3's "old" row in flight
        await _rename(writer, "new")
        release.set()
        answered = await read

        assert await _l2_name(nats) == "new", "a read put L3's older row over the save's newer one in L2"
        assert answered is not None and answered["name"] == "new", "the read answered with the row it lost to"

    @pytest.mark.asyncio
    async def test_a_read_racing_a_save_and_its_eviction_never_recreates_the_older_row(self) -> None:
        # the save's broadcast makes every peer in its scope delete the key it just wrote, so by
        # the time the read seeds, the key can be empty again. A create would land there; only a
        # write fenced on the key's history as the read found it refuses.
        nats, store = FakeNatsClient(), _Store()
        store.rows[_ID] = {"id": _ID, "name": "old"}
        reader, writer = _replica(nats, store), _replica(nats, store)
        await writer.ensure(_ID)
        bucket = await nats.kv_bucket(name="collections")
        bucket.wipe()

        release = store.hold_next_fetch()
        store.held.clear()
        read = asyncio.create_task(reader.ensure(_ID))
        await store.held.wait()
        await _rename(writer, "new")
        await bucket.delete(key=_KEY)  # a peer's listener evicting on the save's broadcast
        release.set()
        await read

        assert await _l2_name(nats) in {None, "new"}, "a read recreated L3's older row after the save was evicted"

    @pytest.mark.asyncio
    async def test_a_read_that_finds_l2_empty_still_seeds_it(self) -> None:
        nats, store = FakeNatsClient(), _Store()
        store.rows[_ID] = {"id": _ID, "name": "stored"}
        reader = _replica(nats, store)
        assert (await reader.ensure(_ID) or {}).get("name") == "stored"
        assert await _l2_name(nats) == "stored"


class TestAReloadNeverRegressesL2:
    @pytest.mark.asyncio
    async def test_a_reload_racing_a_save_never_leaves_l2_older_than_the_save(self) -> None:
        nats, store = FakeNatsClient(), _Store()
        store.rows[_ID] = {"id": _ID, "name": "old"}
        reloader, writer = _replica(nats, store), _replica(nats, store)
        entity = await reloader.get(_ID)
        assert entity is not None
        await writer.ensure(_ID)

        release = store.hold_next_fetch()
        store.held.clear()
        reload = asyncio.create_task(reloader.reload_entity(entity))
        await store.held.wait()  # the reload has L3's "old" row in flight
        await _rename(writer, "new")
        release.set()
        await reload

        assert await _l2_name(nats) == "new", "a reload put L3's older row over the save's newer one in L2"

    @pytest.mark.asyncio
    async def test_a_reload_still_refreshes_an_l2_value_l3_has_moved_past(self) -> None:
        # a synchronous table's L3 is never behind L2, so a reload that finds L2 unchanged since
        # before its L3 read may -- and must -- replace it.
        nats, store = FakeNatsClient(), _Store()
        store.rows[_ID] = {"id": _ID, "name": "old"}
        coll = _replica(nats, store)
        entity = await coll.get(_ID)
        assert entity is not None and await _l2_name(nats) == "old"
        store.rows[_ID] = {"id": _ID, "name": "repaired"}  # an L3 write that bypassed the cache
        await coll.reload_entity(entity)
        assert await _l2_name(nats) == "repaired"
