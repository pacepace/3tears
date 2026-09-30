"""no cache tier is written in an order, or at a time, that lets it disagree with L3 indefinitely.

The rule every write path now keeps: L3 first; then L2 as a compare-and-swap at the revision read
before the L3 write; L1 only when L2 took the row and no other write or eviction of the key
overlapped this one in this process; then the broadcast. A write that cannot know it is the newest
drops the key instead, and the next read takes whichever row L3 kept.

What this pins, each against the interleaving a real round trip allows (the store and the bucket
below can hold an operation at its suspension point):

- a subscript write whose L3 write lands last leaves L2 on L3's row, not on the row a later save
  wrote to L2 in between;
- a save that joins a caller's transaction caches nothing the transaction rolled back, and leaves
  no row a reader cached before the commit once the commit lands;
- two overlapping saves on a collection with an L3 pool and no L2 leave L1 on L3's row, whichever
  order L3 took them in;
- a read that fetched a row before a write or eviction of the key does not cache it after;
- a compare-and-swap whose L3 persist is cancelled withdraws its L2 value, as a failed one does.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime
from types import TracebackType
from typing import Any, ClassVar, Literal

import pytest
from sqlalchemy import BigInteger, Column, DateTime, MetaData, String, Table, Text

from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections import CallerTransaction
from threetears.core.collections.base import BaseCollection
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.entities.base import BaseEntity
from threetears.core.testing.kv import FakeKvBucket, FakeNatsClient

_SCOPE = "tier-order-principal"
_TABLE = "tier_rows"
_ID = "row-1"
_KEY = f"{_SCOPE}.{_TABLE}.{_ID}"


def _metadata() -> MetaData:
    metadata = MetaData()
    Table(
        _TABLE,
        metadata,
        Column("id", String(255), primary_key=True),
        Column("members", Text),
        Column("l2_epoch", DateTime(timezone=True)),
        Column("l2_revision", BigInteger),
        Column("date_created", DateTime(timezone=True)),
        Column("date_updated", DateTime(timezone=True)),
    )
    return metadata


class _Row(BaseEntity):
    primary_key_field = "id"


class _Transaction:
    """one level of a connection's transaction: a savepoint when nested, as asyncpg's is."""

    def __init__(self, conn: _Conn) -> None:
        self._conn = conn

    async def __aenter__(self) -> _Transaction:
        self._conn.levels.append({})
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> bool:
        level = self._conn.levels.pop()
        if exc_type is None:
            target = self._conn.levels[-1] if self._conn.levels else self._conn.store.rows
            target.update(level)
        return False


class _Conn:
    """a caller's connection: writes made on it land in the store only when its transaction commits."""

    def __init__(self, store: _Store) -> None:
        self.store = store
        self.levels: list[dict[str, dict[str, Any]]] = []

    def transaction(self) -> _Transaction:
        return _Transaction(self)


class _Store:
    """an in-process L3 whose next write can be held before or after it lands, and next read held after it reads."""

    def __init__(self) -> None:
        self.rows: dict[str, dict[str, Any]] = {}
        self.held = asyncio.Event()
        self._write_holds: list[asyncio.Event] = []
        self._ack_holds: list[asyncio.Event] = []
        self._read_holds: list[asyncio.Event] = []
        self._ordered_write_holds: list[asyncio.Event] = []

    def hold_next_write(self) -> asyncio.Event:
        release = asyncio.Event()
        self._write_holds.append(release)
        return release

    def hold_next_write_ack(self) -> asyncio.Event:
        release = asyncio.Event()
        self._ack_holds.append(release)
        return release

    def hold_next_read(self) -> asyncio.Event:
        """read the row at once, then hold the answer: a query whose response is in flight."""
        release = asyncio.Event()
        self._read_holds.append(release)
        return release

    def hold_next_ordered_write(self) -> asyncio.Event:
        release = asyncio.Event()
        self._ordered_write_holds.append(release)
        return release

    async def _pass(self, holds: list[asyncio.Event]) -> None:
        if holds:
            release = holds.pop(0)
            self.held.set()
            await release.wait()

    async def fetch(self, entity_id: Any) -> dict[str, Any] | None:
        row = self.rows.get(str(entity_id))
        answer = dict(row) if row is not None else None
        await self._pass(self._read_holds)
        return answer

    async def write(self, data: dict[str, Any], conn: Any = None) -> int:
        await self._pass(self._write_holds)
        if isinstance(conn, _Conn):
            conn.levels[-1][str(data["id"])] = dict(data)
        else:
            self.rows[str(data["id"])] = dict(data)
        await self._pass(self._ack_holds)
        return 1

    async def write_ordered(self, data: dict[str, Any]) -> int:
        await self._pass(self._ordered_write_holds)
        self.rows[str(data["id"])] = dict(data)
        return 1


class _Tiered(BaseCollection[_Row]):
    """a three-tier collection over the store above."""

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

    @property
    def persists_l2_order(self) -> bool:
        return True

    async def fetch_from_store(self, entity_id: Any) -> dict[str, Any] | None:
        return await self._store.fetch(entity_id)

    async def save_to_store(
        self, data: dict[str, Any], original_timestamp: datetime | None = None, *, conn: Any = None
    ) -> int:
        return await self._store.write(data, conn)

    async def save_ordered_to_store(self, data: dict[str, Any], *, conn: Any = None) -> int:
        return await self._store.write_ordered(data)

    async def delete_from_store(self, entity_id: Any) -> None:
        self._store.rows.pop(str(entity_id), None)

    def serialize(self, data: dict[str, Any]) -> bytes:
        return json.dumps(data, default=str).encode("utf-8")

    def deserialize(self, data: bytes) -> dict[str, Any]:
        row: dict[str, Any] = json.loads(data)
        return row


async def _replica(nats: FakeNatsClient | None, store: _Store) -> _Tiered:
    l1 = SQLiteBackend(db_name=f"tier_order_{uuid.uuid4().hex[:8]}")
    l1.initialize(_metadata())
    registry = CollectionRegistry()
    if nats is None:
        registry.configure(l1_backend=l1, l3_pool=object())  # type: ignore[arg-type]
    else:
        registry.configure(l1_backend=l1, l2_client=nats, l3_pool=object(), kv_key_scope=_SCOPE)  # type: ignore[arg-type]
    collection = _Tiered(registry, DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables=""), store)
    if nats is not None:
        await registry.start_invalidation_listener(nats)  # type: ignore[arg-type]
    return collection


def _row(members: str) -> dict[str, Any]:
    return {"id": _ID, "members": members}


async def _bucket(nats: FakeNatsClient) -> FakeKvBucket:
    return await nats.kv_bucket(name="collections")


async def _l2_members(nats: FakeNatsClient) -> str | None:
    raw = await (await _bucket(nats)).get(key=_KEY)
    return None if raw is None else str(json.loads(raw)["members"])


async def _served(coll: _Tiered) -> str:
    entity = await coll.get(_ID)
    assert entity is not None
    return str(entity.to_dict()["members"])


def _l1_members(coll: _Tiered) -> str | None:
    row = coll.get_row_sync(_ID)
    return None if row is None else str(row["members"])


async def _settle_background() -> None:
    """wait for every task this test scheduled -- the fire-and-forget propagation of a subscript write."""
    pending = [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]
    await asyncio.gather(*pending)


class TestASubscriptWriteFollowsL3:
    @pytest.mark.asyncio
    async def test_an_assignment_that_lands_in_l3_last_leaves_l2_on_l3s_row(self) -> None:
        nats, store = FakeNatsClient(), _Store()
        coll = await _replica(nats, store)
        peer = await _replica(nats, store)
        release = store.hold_next_write()
        store.held.clear()
        coll[_ID] = _row("x")
        await store.held.wait()  # "x" is on its way to L3 and has not landed
        await peer.save_entity(peer.create(_row("y")))
        release.set()
        await _settle_background()
        assert store.rows[_ID]["members"] == "x", "the harness no longer lands the assignment last"
        assert await _l2_members(nats) in {None, "x"}, "L2 holds a row L3 no longer holds"
        assert await _served(coll) == "x"
        assert await _served(peer) == "x"

    @pytest.mark.asyncio
    async def test_an_uncontended_assignment_is_cached_in_l1_and_l2(self) -> None:
        nats, store = FakeNatsClient(), _Store()
        coll = await _replica(nats, store)
        coll[_ID] = _row("x")
        await _settle_background()
        assert store.rows[_ID]["members"] == "x"
        assert await _l2_members(nats) == "x"
        assert _l1_members(coll) == "x"


class TestASaveInACallersTransaction:
    @pytest.mark.asyncio
    async def test_a_rolled_back_save_leaves_no_tier_holding_its_row(self) -> None:
        nats, store = FakeNatsClient(), _Store()
        coll = await _replica(nats, store)
        peer = await _replica(nats, store)
        await coll.save_entity(coll.create(_row("base")))
        conn = _Conn(store)

        class _Abort(Exception):
            pass

        with pytest.raises(_Abort):
            async with CallerTransaction(conn):
                entity = await coll.get(_ID)
                assert entity is not None
                entity.members = "x"
                await coll.save_entity(entity, conn=conn)
                raise _Abort
        assert store.rows[_ID]["members"] == "base"
        assert await _l2_members(nats) in {None, "base"}, "L2 holds a row the transaction rolled back"
        assert _l1_members(coll) in {None, "base"}, "L1 holds a row the transaction rolled back"
        assert await _served(coll) == "base"
        assert await _served(peer) == "base"

    @pytest.mark.asyncio
    async def test_a_row_read_before_the_commit_is_not_served_after_it(self) -> None:
        nats, store = FakeNatsClient(), _Store()
        coll = await _replica(nats, store)
        peer = await _replica(nats, store)
        await coll.save_entity(coll.create(_row("base")))
        await coll.invalidate_cache(_ID)
        conn = _Conn(store)
        async with CallerTransaction(conn):
            await coll.save_entity(coll.create(_row("x")), conn=conn)
            # a reader between the save and the commit reads L3's committed row and caches it
            assert await _served(peer) == "base"
            assert await _served(coll) == "base"
        assert store.rows[_ID]["members"] == "x"
        assert await _l2_members(nats) in {None, "x"}, "L2 kept a row read before the commit"
        assert await _served(coll) == "x"
        assert await _served(peer) == "x"

    @pytest.mark.asyncio
    async def test_nothing_is_settled_until_the_outermost_transaction_ends(self) -> None:
        nats, store = FakeNatsClient(), _Store()
        coll = await _replica(nats, store)
        await coll.save_entity(coll.create(_row("base")))
        conn = _Conn(store)
        async with CallerTransaction(conn):
            async with CallerTransaction(conn):
                await coll.save_entity(coll.create(_row("x")), conn=conn)
            assert store.rows[_ID]["members"] == "base", "the savepoint released straight to L3"
            assert await _l2_members(nats) in {None, "base"}, "a save cached a row before its transaction ended"
            assert await _served(coll) == "base"
        assert await _served(coll) == "x"

    @pytest.mark.asyncio
    async def test_a_connection_whose_transaction_no_caller_transaction_opened_is_refused(self) -> None:
        nats, store = FakeNatsClient(), _Store()
        coll = await _replica(nats, store)
        conn = _Conn(store)
        async with conn.transaction():
            with pytest.raises(ValueError, match="CallerTransaction"):
                await coll.save_entity(coll.create(_row("x")), conn=conn)
        assert _ID not in store.rows
        assert await _l2_members(nats) is None


class TestOverlappingSavesWithNoL2:
    @pytest.mark.asyncio
    async def test_a_save_answered_late_does_not_cache_over_the_later_one(self) -> None:
        store = _Store()
        coll = await _replica(None, store)
        release = store.hold_next_write_ack()
        store.held.clear()
        held = asyncio.create_task(coll.save_entity(coll.create(_row("x"))))
        await store.held.wait()  # "x" is in L3; its answer is in flight
        await coll.save_entity(coll.create(_row("y")))
        release.set()
        await held
        assert store.rows[_ID]["members"] == "y"
        assert _l1_members(coll) in {None, "y"}, "the earlier save cached its row over the later one"
        assert await _served(coll) == "y"

    @pytest.mark.asyncio
    async def test_whichever_order_l3_took_the_saves_in_l1_serves_l3s_row(self) -> None:
        store = _Store()
        coll = await _replica(None, store)
        release = store.hold_next_write()
        store.held.clear()
        held = asyncio.create_task(coll.save_entity(coll.create(_row("x"))))
        await store.held.wait()  # "x" has not reached L3 yet
        await coll.save_entity(coll.create(_row("y")))
        release.set()
        await held
        assert store.rows[_ID]["members"] == "x", "the harness no longer lands the held write last"
        assert _l1_members(coll) in {None, "x"}, "L1 serves a row L3 no longer holds"
        assert await _served(coll) == "x"

    @pytest.mark.asyncio
    async def test_consecutive_saves_each_replace_the_cached_row(self) -> None:
        store = _Store()
        coll = await _replica(None, store)
        await coll.save_entity(coll.create(_row("x")))
        await coll.save_entity(coll.create(_row("y")))
        assert _l1_members(coll) == "y"


class TestAReadInFlightDuringAWrite:
    @pytest.mark.asyncio
    async def test_a_row_read_from_l3_before_a_local_save_is_not_cached_after_it(self) -> None:
        store = _Store()
        coll = await _replica(None, store)
        await coll.save_entity(coll.create(_row("base")))
        coll.evict_from_cache_sync(_ID)
        release = store.hold_next_read()
        store.held.clear()
        reading = asyncio.create_task(coll.get(_ID))
        await store.held.wait()  # the read has L3's "base" in hand
        await coll.save_entity(coll.create(_row("y")))
        release.set()
        await reading
        assert _l1_members(coll) in {None, "y"}, "a read cached the row it fetched before a later save"
        assert await _served(coll) == "y"

    @pytest.mark.asyncio
    async def test_a_row_read_from_l2_before_a_peers_save_is_not_cached_after_it(self) -> None:
        nats, store = FakeNatsClient(), _Store()
        coll = await _replica(nats, store)
        peer = await _replica(nats, store)
        await coll.save_entity(coll.create(_row("base")))
        coll.evict_from_cache_sync(_ID)
        bucket = await _bucket(nats)
        # the save's own broadcast made the peer drop the shared key; a read seeds it again
        await bucket.put(key=_KEY, value=json.dumps(_row("base")).encode("utf-8"))
        real_get_latest = bucket.get_latest
        release = asyncio.Event()
        reached = asyncio.Event()

        async def _answer_late(**kwargs: Any) -> tuple[bytes | None, int]:
            answer = await real_get_latest(**kwargs)
            reached.set()
            await release.wait()
            return answer

        bucket.get_latest = _answer_late  # type: ignore[method-assign]
        reading = asyncio.create_task(coll.get(_ID))
        await reached.wait()  # the read has L2's "base" in hand
        bucket.get_latest = real_get_latest  # type: ignore[method-assign]
        await peer.save_entity(peer.create(_row("y")))  # its broadcast evicts coll's L1 and L2 key
        release.set()
        await reading
        assert store.rows[_ID]["members"] == "y"
        assert _l1_members(coll) in {None, "y"}, "a read cached the row it fetched before a peer's save"
        assert await _served(coll) == "y"


class TestACancelledCompareAndSwapPersist:
    @pytest.mark.asyncio
    async def test_the_won_l2_value_is_withdrawn(self) -> None:
        nats, store = FakeNatsClient(), _Store()
        coll = await _replica(nats, store)
        store.hold_next_ordered_write()  # never released: the persist is cancelled mid-write
        store.held.clear()

        def _set(row: dict[str, Any] | None) -> tuple[Literal["upsert", "delete", "noop"], dict[str, Any] | None]:
            del row
            return "upsert", _row("x")

        mutating = asyncio.create_task(coll.l2_cas_mutate(_ID, _set))
        await store.held.wait()  # the swap won in L2; its L3 persist is in flight
        assert await _l2_members(nats) == "x"
        mutating.cancel()
        with pytest.raises(asyncio.CancelledError):
            await mutating
        assert _ID not in store.rows
        assert await _l2_members(nats) is None, "a cancelled persist left L2 holding a value L3 never took"
