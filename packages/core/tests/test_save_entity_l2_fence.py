"""``save_entity`` leaves no row older than L3's cached in L1 or L2 after its L3 round trip.

The hole this pins shut: ``save_entity`` commits to L3 and only then caches the row in L1 and
writes it to L2. A later save of the same row can complete inside that round trip -- this
replica's own, or a peer's, whose broadcast has already evicted this replica's L1 and deleted the
shared L2 key. An unconditional write of the earlier row after that leaves it in L1 (and, with no
peer left to delete it, in L2) behind L3, with nothing left to evict it: the replica serves the
older value until the row is written again.

The race lives at the L3 write's suspension point, so the store below can hold a commit's answer
while a later save lands, then release it: exactly the interleaving a real database round trip
allows.

What this pins, for a collection that stores the compare-and-swap order and for one that does not:

- a save answered after this replica's own later save leaves the later row served;
- a save answered after a peer's later save and broadcast leaves the later row served on both;
- whichever order L3 took two saves in, the replica serves the row L3 holds;
- a save whose L2 state cannot be read before its L3 write caches nothing, and the next read
  serves L3's row;
- an uncontended save still serves its own write from L1 and L2.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime
from typing import Any, ClassVar, Literal

import pytest
from sqlalchemy import BigInteger, Column, DateTime, MetaData, String, Table, Text

from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections.base import BaseCollection
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.entities.base import BaseEntity
from threetears.core.testing.kv import FakeKvBucket, FakeNatsClient
from threetears.nats.errors import KvError

_SCOPE = "save-fence-principal"
_TABLE = "saved_rows"
_ID = "row-1"
_KEY = f"{_SCOPE}.{_TABLE}.{_ID}"


def _metadata(*, ordered: bool) -> MetaData:
    metadata = MetaData()
    order: list[Column[Any]] = (
        [Column("l2_epoch", DateTime(timezone=True)), Column("l2_revision", BigInteger)] if ordered else []
    )
    Table(
        _TABLE,
        metadata,
        Column("id", String(255), primary_key=True),
        Column("members", Text),
        *order,
        Column("date_created", DateTime(timezone=True)),
        Column("date_updated", DateTime(timezone=True)),
    )
    return metadata


class _Row(BaseEntity):
    primary_key_field = "id"


class _Store:
    """an in-process L3 whose next write can be held before it lands, or after it lands."""

    def __init__(self) -> None:
        self.rows: dict[str, dict[str, Any]] = {}
        self.held = asyncio.Event()
        self._write_holds: list[asyncio.Event] = []
        self._ack_holds: list[asyncio.Event] = []

    def hold_next_write(self) -> asyncio.Event:
        """hold the next write before it lands: a commit still on its way to the database."""
        release = asyncio.Event()
        self._write_holds.append(release)
        return release

    def hold_next_write_ack(self) -> asyncio.Event:
        """land the next write at once, then hold its answer: a commit whose response is in flight."""
        release = asyncio.Event()
        self._ack_holds.append(release)
        return release

    async def _pass(self, holds: list[asyncio.Event]) -> None:
        if holds:
            release = holds.pop(0)
            self.held.set()
            await release.wait()

    async def fetch(self, entity_id: Any) -> dict[str, Any] | None:
        row = self.rows.get(str(entity_id))
        return dict(row) if row is not None else None

    async def write(self, data: dict[str, Any]) -> int:
        await self._pass(self._write_holds)
        self.rows[str(data["id"])] = dict(data)
        await self._pass(self._ack_holds)
        return 1


class _Plain(BaseCollection[_Row]):
    """a three-tier collection with no compare-and-swap order columns."""

    ordered: ClassVar[bool] = False
    datetime_columns: ClassVar[frozenset[str]] = frozenset({"l2_epoch", "date_created", "date_updated"})
    l3_write_policy: ClassVar[Literal["synchronous", "write_behind"] | None] = "synchronous"

    def __init__(self, registry: CollectionRegistry, config: DefaultCoreConfig, store: _Store) -> None:
        self._store = store
        super().__init__(registry, config)

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
        return await self._store.write(data)

    async def delete_from_store(self, entity_id: Any) -> None:
        self._store.rows.pop(str(entity_id), None)

    def serialize(self, data: dict[str, Any]) -> bytes:
        return json.dumps(data, default=str).encode("utf-8")

    def deserialize(self, data: bytes) -> dict[str, Any]:
        row: dict[str, Any] = json.loads(data)
        return row


class _Ordered(_Plain):
    """a three-tier collection that stores the order its compare-and-swaps win."""

    ordered: ClassVar[bool] = True

    @property
    def persists_l2_order(self) -> bool:
        return True

    async def save_ordered_to_store(self, data: dict[str, Any], *, conn: Any = None) -> int:
        return await self._store.write(data)


#: the shared collections bucket, and the invalidation broadcast delivered to every listener.
_Nats = FakeNatsClient


async def _replica(cls: type[_Plain], nats: _Nats, store: _Store) -> _Plain:
    l1 = SQLiteBackend(db_name=f"save_fence_{uuid.uuid4().hex[:8]}")
    l1.initialize(_metadata(ordered=cls.ordered))
    registry = CollectionRegistry()
    registry.configure(l1_backend=l1, l2_client=nats, l3_pool=object(), kv_key_scope=_SCOPE)  # type: ignore[arg-type]
    collection = cls(registry, DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables=""), store)
    await registry.start_invalidation_listener(nats)  # type: ignore[arg-type]
    return collection


def _row(members: str) -> dict[str, Any]:
    return {"id": _ID, "members": members}


async def _bucket(nats: _Nats) -> FakeKvBucket:
    return await nats.kv_bucket(name="collections")


async def _l2_members(nats: _Nats) -> str | None:
    raw = await (await _bucket(nats)).get(key=_KEY)
    return None if raw is None else str(json.loads(raw)["members"])


async def _served(coll: _Plain) -> str:
    entity = await coll.get(_ID)
    assert entity is not None
    return str(entity.to_dict()["members"])


async def _answered_late(slow: _Plain, other: _Plain, store: _Store, first: str, second: str) -> None:
    """``slow``'s save of ``first`` lands in L3 and its answer is held while ``other`` saves ``second``."""
    release = store.hold_next_write_ack()
    store.held.clear()
    held = asyncio.create_task(slow.save_entity(slow.create(_row(first))))
    await store.held.wait()  # "first" is committed in L3; its answer is in flight
    await other.save_entity(other.create(_row(second)))
    release.set()
    await held


_KINDS = pytest.mark.parametrize("cls", [_Plain, _Ordered], ids=["without-order-columns", "with-order-columns"])


class TestASaveAnsweredLateLeavesNoOlderRowCached:
    @_KINDS
    @pytest.mark.asyncio
    async def test_this_replicas_own_later_save_stays_served(self, cls: type[_Plain]) -> None:
        nats, store = _Nats(), _Store()
        coll = await _replica(cls, nats, store)
        await _answered_late(coll, coll, store, "x", "y")
        assert store.rows[_ID]["members"] == "y"
        assert await _served(coll) == "y", "the earlier save cached its row over the later one"
        assert await _l2_members(nats) in {None, "y"}, "the earlier save wrote its row over the later one in L2"

    @_KINDS
    @pytest.mark.asyncio
    async def test_a_peers_later_save_and_broadcast_are_not_undone(self, cls: type[_Plain]) -> None:
        nats, store = _Nats(), _Store()
        slow = await _replica(cls, nats, store)
        peer = await _replica(cls, nats, store)
        await _answered_late(slow, peer, store, "x", "y")
        assert store.rows[_ID]["members"] == "y"
        assert await _served(slow) == "y", "a late answer cached a row the peer's broadcast retracted"
        assert await _served(peer) == "y", "a late answer's broadcast left the peer on the older row"
        assert await _l2_members(nats) in {None, "y"}, "the earlier save wrote its row over the later one in L2"

    @_KINDS
    @pytest.mark.asyncio
    async def test_whichever_order_l3_took_the_replica_serves_l3s_row(self, cls: type[_Plain]) -> None:
        nats, store = _Nats(), _Store()
        coll = await _replica(cls, nats, store)
        release = store.hold_next_write()
        store.held.clear()
        held = asyncio.create_task(coll.save_entity(coll.create(_row("x"))))
        await store.held.wait()  # "x" has not reached L3 yet
        await coll.save_entity(coll.create(_row("y")))
        release.set()
        await held
        assert store.rows[_ID]["members"] == "x", "the harness no longer lands the held write last"
        assert await _served(coll) == "x", "the replica serves a row L3 no longer holds"
        assert await _l2_members(nats) in {None, "x"}, "L2 holds a row L3 no longer holds"


class TestWhenL2CannotBeRead:
    @_KINDS
    @pytest.mark.asyncio
    async def test_a_save_whose_l2_state_is_unreadable_caches_nothing_and_still_succeeds(
        self, cls: type[_Plain]
    ) -> None:
        nats, store = _Nats(), _Store()
        coll = await _replica(cls, nats, store)
        bucket = await _bucket(nats)
        await bucket.put(key=_KEY, value=json.dumps(_row("stale")).encode())
        real_get_latest = bucket.get_latest

        async def _unreadable(**_: Any) -> tuple[bytes | None, int]:
            raise KvError("broker unreachable")

        bucket.get_latest = _unreadable  # type: ignore[method-assign]
        await coll.save_entity(coll.create(_row("x")))
        bucket.get_latest = real_get_latest  # type: ignore[method-assign]
        assert store.rows[_ID]["members"] == "x"
        assert not coll.exists_in_cache_sync(_ID), "a save that could not fence L2 cached its row in L1"
        assert await _l2_members(nats) is None, "a save that could not fence L2 left the older L2 row in place"
        assert await _served(coll) == "x"


class TestAnUncontendedSaveStillCaches:
    @_KINDS
    @pytest.mark.asyncio
    async def test_the_saved_row_is_served_from_l1_and_l2(self, cls: type[_Plain]) -> None:
        nats, store = _Nats(), _Store()
        coll = await _replica(cls, nats, store)
        await coll.save_entity(coll.create(_row("x")))
        assert coll.exists_in_cache_sync(_ID), "an uncontended save no longer caches its own row"
        assert await _l2_members(nats) == "x", "an uncontended save no longer writes its row to L2"

    @_KINDS
    @pytest.mark.asyncio
    async def test_consecutive_saves_each_replace_the_cached_row(self, cls: type[_Plain]) -> None:
        nats, store = _Nats(), _Store()
        coll = await _replica(cls, nats, store)
        await coll.save_entity(coll.create(_row("x")))
        await coll.save_entity(coll.create(_row("y")))
        assert coll.exists_in_cache_sync(_ID)
        assert await _served(coll) == "y"
        assert await _l2_members(nats) == "y"
