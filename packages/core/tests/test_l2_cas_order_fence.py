"""``l2_cas_mutate`` persists to L3 in the order L2 decided, whatever order the persists arrive in.

The hole this pins shut: each compare-and-swap winner persists its row to L3 on its own, so two
winners of consecutive revisions can reach L3 in reverse. Unfenced, the earlier row lands last and
stays; once L2 loses the key -- a broker restart on memory storage -- the next mutation seeds from
that stale row and the later change is lost for good.

The race lives at the persist's suspension point, so the store below can hold one persist while a
later winner's lands, then release it: exactly the interleaving a real database round trip allows.

What this pins:

- two consecutive winners whose L3 writes arrive in reverse leave L3 holding the NEWER row, for a
  synchronous persist and for a write-behind flush, and the superseded persist is not an error;
- after an L2 wipe the next mutation seeds from the newest row: a set keeps every member and a
  counter keeps every increment;
- a recreated bucket restarts its revisions and its writes are still admitted, because the order
  carries the bucket's creation time;
- a three-tier collection that cannot store the order is refused before L2 is touched, and so is
  a delete, which no stored order can fence against a late persist;
- a stored order ahead of anything the bucket can write is refused loudly, not dropped silently;
- the write buffer keeps the newer of two orders for one row, whichever arrived last;
- a read that seeds L2 from L3 never replaces a value a compare-and-swap put there meanwhile;
- a write that did not win a compare-and-swap stores no order;
- a winner whose persist answers after a later swap -- this replica's own, or a peer's whose
  broadcast already evicted this L1 -- leaves no older row cached in L1, while an uncontended
  winner still serves its own write from L1.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any, ClassVar, Literal

import pytest
from sqlalchemy import BigInteger, Column, DateTime, Integer, MetaData, String, Table, Text

from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections.base import BaseCollection
from threetears.core.collections.flush import WriteBuffer, flush_pending
from threetears.core.collections.l2_order import L2Order, l2_order_of, with_l2_order
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.core.entities.base import BaseEntity
from threetears.core.exceptions import L2EpochRegressedError
from threetears.core.testing.kv import FakeKvBucket, FakeNatsClient

_SCOPE = "fence-principal"
_TABLE = "member_sets"
_ID = "set-1"
_KEY = f"{_SCOPE}.{_TABLE}.{_ID}"

_Decision = tuple[Literal["upsert", "delete", "noop"], dict[str, Any] | None]


def _metadata() -> MetaData:
    metadata = MetaData()
    Table(
        _TABLE,
        metadata,
        Column("id", String(255), primary_key=True),
        Column("members", Text),
        Column("count", Integer),
        Column("l2_epoch", DateTime(timezone=True)),
        Column("l2_revision", BigInteger),
        Column("date_created", DateTime(timezone=True)),
        Column("date_updated", DateTime(timezone=True)),
    )
    return metadata


class _Row(BaseEntity):
    primary_key_field = "id"


class _Store:
    """an in-process L3 whose next write can be held at its suspension point.

    ``write`` is an unconditional upsert; ``write_ordered`` lands only over a strictly older
    stored order, which is the conditional write the SQL backend generates.
    """

    def __init__(self) -> None:
        self.rows: dict[str, dict[str, Any]] = {}
        self.held = asyncio.Event()
        self._write_holds: list[asyncio.Event] = []
        self._ack_holds: list[asyncio.Event] = []
        self._fetch_holds: list[asyncio.Event] = []

    def hold_next_write(self) -> asyncio.Event:
        release = asyncio.Event()
        self._write_holds.append(release)
        return release

    def hold_next_write_ack(self) -> asyncio.Event:
        """land the next write at once, then hold its answer: a commit whose response is in flight."""
        release = asyncio.Event()
        self._ack_holds.append(release)
        return release

    def hold_next_fetch(self) -> asyncio.Event:
        release = asyncio.Event()
        self._fetch_holds.append(release)
        return release

    async def _pass(self, holds: list[asyncio.Event]) -> None:
        if holds:
            release = holds.pop(0)
            self.held.set()
            await release.wait()

    async def fetch(self, entity_id: Any) -> dict[str, Any] | None:
        # read, THEN suspend: the row a held read returns is the one the store held when the
        # query ran, which is what a response still in flight carries.
        row = self.rows.get(str(entity_id))
        snapshot = dict(row) if row is not None else None
        await self._pass(self._fetch_holds)
        return snapshot

    async def write(self, data: dict[str, Any]) -> int:
        await self._pass(self._write_holds)
        self.rows[str(data["id"])] = dict(data)
        return 1

    async def write_ordered(self, data: dict[str, Any]) -> int:
        await self._pass(self._write_holds)
        incoming = l2_order_of(data)
        assert incoming is not None, "an ordered write arrived without its order"
        stored = self.rows.get(str(data["id"]))
        stored_order = None if stored is None else l2_order_of(stored)
        if stored_order is not None and stored_order >= incoming:
            return 0
        self.rows[str(data["id"])] = dict(data)
        await self._pass(self._ack_holds)
        return 1


class _Unordered(BaseCollection[_Row]):
    """a three-tier collection whose L3 cannot store the compare-and-swap order."""

    datetime_columns: ClassVar[frozenset[str]] = frozenset({"l2_epoch", "date_created", "date_updated"})
    l3_write_policy: ClassVar[Literal["synchronous", "write_behind"] | None] = "synchronous"

    def __init__(
        self, registry: CollectionRegistry, config: DefaultCoreConfig, store: _Store, buffer: WriteBuffer | None
    ) -> None:
        self._store = store
        super().__init__(registry, config, write_buffer=buffer)

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


class _Synchronous(_Unordered):
    """a three-tier collection that stores the order and writes L3 before returning."""

    @property
    def persists_l2_order(self) -> bool:
        return True

    async def save_ordered_to_store(self, data: dict[str, Any], *, conn: Any = None) -> int:
        return await self._store.write_ordered(data)


class _WriteBehind(_Synchronous):
    l3_write_policy: ClassVar[Literal["synchronous", "write_behind"] | None] = "write_behind"


class _Nats(FakeNatsClient):
    """the shared collections bucket; no listener runs in these tests."""


def _replica(
    cls: type[_Unordered], nats: _Nats, store: _Store, *, buffer: WriteBuffer | None = None
) -> tuple[_Unordered, CollectionRegistry]:
    l1 = SQLiteBackend(db_name=f"fence_{uuid.uuid4().hex[:8]}")
    l1.initialize(_metadata())
    registry = CollectionRegistry()
    registry.configure(l1_backend=l1, l2_client=nats, l3_pool=object(), kv_key_scope=_SCOPE)  # type: ignore[arg-type]
    collection = cls(registry, DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables=""), store, buffer)
    return collection, registry


def _add(member: str):  # type: ignore[no-untyped-def]
    def _decide(row: dict[str, Any] | None) -> _Decision:
        members = [] if row is None else list(json.loads(row["members"]))
        if member in members:
            return "noop", None
        return "upsert", {**(row or {}), "id": _ID, "members": json.dumps([*members, member])}

    return _decide


def _increment(row: dict[str, Any] | None) -> _Decision:
    count = 0 if row is None else int(row["count"])
    return "upsert", {**(row or {}), "id": _ID, "members": "[]", "count": count + 1}


async def _bucket(nats: _Nats) -> FakeKvBucket:
    return await nats.kv_bucket(name="collections")


async def _reverse_delivery(
    first: _Unordered, second: _Unordered, store: _Store, first_change: Any, second_change: Any
) -> None:
    """``first`` wins L2 and is held at its L3 write; ``second`` then wins and lands; ``first`` lands last."""
    release = store.hold_next_write()
    store.held.clear()
    held = asyncio.create_task(first.l2_cas_mutate(_ID, first_change))
    await store.held.wait()
    await second.l2_cas_mutate(_ID, second_change)
    release.set()
    await held


def _members(row: dict[str, Any]) -> list[str]:
    members: list[str] = json.loads(row["members"])
    return members


class TestTheNewerWinnerStaysInL3:
    @pytest.mark.asyncio
    async def test_a_synchronous_persist_delivered_last_does_not_overwrite_a_newer_one(self) -> None:
        nats, store = _Nats(), _Store()
        first, _ = _replica(_Synchronous, nats, store)
        second, _ = _replica(_Synchronous, nats, store)
        await _reverse_delivery(first, second, store, _add("x"), _add("y"))
        assert _members(store.rows[_ID]) == ["x", "y"], "the earlier winner's late persist overwrote the later one"

    @pytest.mark.asyncio
    async def test_the_superseded_persist_is_reported_as_the_success_it_was(self) -> None:
        nats, store = _Nats(), _Store()
        first, _ = _replica(_Synchronous, nats, store)
        second, _ = _replica(_Synchronous, nats, store)
        release = store.hold_next_write()
        store.held.clear()
        held = asyncio.create_task(first.l2_cas_mutate(_ID, _add("x")))
        await store.held.wait()
        await second.l2_cas_mutate(_ID, _add("y"))
        release.set()
        outcome = await held
        assert outcome.action == "created"
        assert outcome.row is not None and _members(outcome.row) == ["x"]

    @pytest.mark.asyncio
    async def test_write_behind_buffers_flushed_in_reverse_leave_the_newer_row(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        nats, store = _Nats(), _Store()
        first_buffer, second_buffer = WriteBuffer(), WriteBuffer()
        first, first_registry = _replica(_WriteBehind, nats, store, buffer=first_buffer)
        second, second_registry = _replica(_WriteBehind, nats, store, buffer=second_buffer)
        await first.l2_cas_mutate(_ID, _add("x"))
        await second.l2_cas_mutate(_ID, _add("y"))
        assert _ID not in store.rows
        await flush_pending(second_buffer, second_registry)
        with caplog.at_level("DEBUG"):
            await flush_pending(first_buffer, first_registry)
        assert _members(store.rows[_ID]) == ["x", "y"], "the earlier winner's late flush overwrote the later one"
        # this harness's L3 handle has no usable transaction, so every flush logs its fall back to
        # the per-entity loop; what must not appear is the per-write "took nothing" report.
        lost = [
            r for r in caplog.records if r.levelname in {"WARNING", "ERROR"} and "Deferred L3 write" in r.getMessage()
        ]
        assert not lost, "a flush superseded by a newer order was reported as a lost write"

    @pytest.mark.asyncio
    async def test_the_persisted_row_carries_the_order_its_swap_won(self) -> None:
        nats, store = _Nats(), _Store()
        coll, _ = _replica(_Synchronous, nats, store)
        await coll.l2_cas_mutate(_ID, _add("x"))
        await coll.l2_cas_mutate(_ID, _add("y"))
        bucket = await _bucket(nats)
        entry = await bucket.get_entry(key=_KEY)
        assert entry is not None
        assert l2_order_of(store.rows[_ID]) == L2Order(await bucket.date_created(), entry[1])


class TestAWipeLosesNothingThatReachedL3:
    @pytest.mark.asyncio
    async def test_a_set_keeps_every_member_across_a_wipe(self) -> None:
        nats, store = _Nats(), _Store()
        first, _ = _replica(_Synchronous, nats, store)
        second, _ = _replica(_Synchronous, nats, store)
        await _reverse_delivery(first, second, store, _add("x"), _add("y"))
        (await _bucket(nats)).wipe()
        third, _ = _replica(_Synchronous, nats, store)
        outcome = await third.l2_cas_mutate(_ID, _add("z"))
        assert outcome.row is not None and _members(outcome.row) == ["x", "y", "z"], "the wipe lost a member"
        assert _members(store.rows[_ID]) == ["x", "y", "z"]

    @pytest.mark.asyncio
    async def test_a_counter_keeps_every_increment_across_a_wipe(self) -> None:
        nats, store = _Nats(), _Store()
        first, _ = _replica(_Synchronous, nats, store)
        second, _ = _replica(_Synchronous, nats, store)
        for _ in range(3):
            await _reverse_delivery(first, second, store, _increment, _increment)
        (await _bucket(nats)).wipe()
        outcome = await first.l2_cas_mutate(_ID, _increment)
        assert outcome.row is not None and outcome.row["count"] == 7, "the wipe lost an increment"
        assert store.rows[_ID]["count"] == 7

    @pytest.mark.asyncio
    async def test_a_write_behind_set_keeps_every_flushed_member_across_a_wipe(self) -> None:
        nats, store = _Nats(), _Store()
        first_buffer, second_buffer = WriteBuffer(), WriteBuffer()
        first, first_registry = _replica(_WriteBehind, nats, store, buffer=first_buffer)
        second, second_registry = _replica(_WriteBehind, nats, store, buffer=second_buffer)
        await first.l2_cas_mutate(_ID, _add("x"))
        await second.l2_cas_mutate(_ID, _add("y"))
        await flush_pending(second_buffer, second_registry)
        await flush_pending(first_buffer, first_registry)
        (await _bucket(nats)).wipe()
        outcome = await first.l2_cas_mutate(_ID, _add("z"))
        assert outcome.row is not None and _members(outcome.row) == ["x", "y", "z"], "the wipe lost a member"


class TestARecreatedBucketIsAdmitted:
    @pytest.mark.asyncio
    async def test_writes_after_a_recreation_land_although_their_revisions_restart(self) -> None:
        nats, store = _Nats(), _Store()
        coll, _ = _replica(_Synchronous, nats, store)
        for _ in range(3):
            await coll.l2_cas_mutate(_ID, _increment)
        bucket = await _bucket(nats)
        before = l2_order_of(store.rows[_ID])
        assert before is not None and before.revision == 3
        recreated = before.epoch + timedelta(seconds=1)
        bucket.wipe(date_created=recreated)
        outcome = await coll.l2_cas_mutate(_ID, _increment)
        assert outcome.row is not None and outcome.row["count"] == 4
        after = l2_order_of(store.rows[_ID])
        assert after == L2Order(recreated, 1), "a recreated bucket's write was refused or mis-ordered"
        assert store.rows[_ID]["count"] == 4

    @pytest.mark.asyncio
    async def test_a_swap_landing_in_a_bucket_recreated_under_it_takes_the_new_creation_time(self) -> None:
        nats, store = _Nats(), _Store()
        coll, _ = _replica(_Synchronous, nats, store)
        bucket = await _bucket(nats)
        recreated = await bucket.date_created() + timedelta(seconds=1)
        real_create = bucket.create

        async def _create_after_recreation(**kwargs: Any) -> int | None:
            bucket.wipe(date_created=recreated)  # the broker restarted between the read and the write
            return await real_create(**kwargs)

        bucket.create = _create_after_recreation  # type: ignore[method-assign]
        await coll.l2_cas_mutate(_ID, _add("x"))
        assert l2_order_of(store.rows[_ID]) == L2Order(recreated, 1), (
            "a swap in the new stream was ordered under the old one, below every later write it has"
        )

    @pytest.mark.asyncio
    async def test_a_swap_whose_stream_died_after_it_keeps_the_old_creation_time(self) -> None:
        nats, store = _Nats(), _Store()
        coll, _ = _replica(_Synchronous, nats, store)
        bucket = await _bucket(nats)
        created = await bucket.date_created()
        real_create = bucket.create

        async def _create_then_restart(**kwargs: Any) -> int | None:
            won = await real_create(**kwargs)
            bucket.wipe(date_created=created + timedelta(seconds=1))  # the stream died with the swap in it
            return won

        bucket.create = _create_then_restart  # type: ignore[method-assign]
        await coll.l2_cas_mutate(_ID, _add("x"))
        assert l2_order_of(store.rows[_ID]) == L2Order(created, 1), (
            "a swap from a dead stream was ordered above the new stream's writes, which restart at 1"
        )


class TestWhatCannotBeFencedIsRefused:
    @pytest.mark.asyncio
    async def test_a_three_tier_collection_without_the_order_columns_is_refused_before_l2(self) -> None:
        nats, store = _Nats(), _Store()
        coll, _ = _replica(_Unordered, nats, store)
        with pytest.raises(ValueError, match="l2_epoch"):
            await coll.l2_cas_mutate(_ID, _add("x"))
        assert await (await _bucket(nats)).get(key=_KEY) is None, "L2 was written before the refusal"
        assert _ID not in store.rows

    @pytest.mark.asyncio
    async def test_a_delete_on_a_three_tier_collection_is_refused_before_l2(self) -> None:
        nats, store = _Nats(), _Store()
        coll, _ = _replica(_Synchronous, nats, store)
        await coll.l2_cas_mutate(_ID, _add("x"))
        with pytest.raises(ValueError, match="delete"):
            await coll.l2_cas_mutate(_ID, lambda _row: ("delete", None))
        assert await (await _bucket(nats)).get(key=_KEY) is not None, "L2 was deleted before the refusal"
        assert _ID in store.rows

    @pytest.mark.asyncio
    async def test_a_stored_order_ahead_of_the_bucket_is_refused_loudly(self) -> None:
        nats, store = _Nats(), _Store()
        created = await (await _bucket(nats)).date_created()
        store.rows[_ID] = with_l2_order({"id": _ID, "members": '["x"]'}, L2Order(created + timedelta(hours=1), 5))
        coll, _ = _replica(_Synchronous, nats, store)
        with pytest.raises(L2EpochRegressedError):
            await coll.l2_cas_mutate(_ID, _add("y"))
        assert await (await _bucket(nats)).get(key=_KEY) is None, "L2 took a write L3 would refuse"
        assert _members(store.rows[_ID]) == ["x"]


class TestTheWriteBufferKeepsTheNewerOrder:
    @pytest.mark.asyncio
    async def test_an_older_order_arriving_last_does_not_replace_a_newer_one(self) -> None:
        buffer = WriteBuffer()
        epoch = datetime(2026, 9, 26, tzinfo=UTC)
        newer = with_l2_order({"id": _ID, "members": '["x", "y"]'}, L2Order(epoch, 7))
        older = with_l2_order({"id": _ID, "members": '["x"]'}, L2Order(epoch, 6))
        await buffer.add(_TABLE, _ID, newer)
        await buffer.add(_TABLE, _ID, older)
        (pending,) = await buffer.drain()
        assert pending.data["members"] == '["x", "y"]', "the buffer replaced a newer order with an older one"

    @pytest.mark.asyncio
    async def test_a_newer_order_still_replaces_an_older_one(self) -> None:
        buffer = WriteBuffer()
        epoch = datetime(2026, 9, 26, tzinfo=UTC)
        await buffer.add(_TABLE, _ID, with_l2_order({"id": _ID, "members": '["x"]'}, L2Order(epoch, 6)))
        await buffer.add(
            _TABLE, _ID, with_l2_order({"id": _ID, "members": '["x", "y"]'}, L2Order(epoch + timedelta(seconds=1), 1))
        )
        (pending,) = await buffer.drain()
        assert pending.data["members"] == '["x", "y"]'


class TestReadsDoNotReorderL2:
    @pytest.mark.asyncio
    async def test_a_read_seeding_l2_from_l3_never_replaces_a_swapped_value(self) -> None:
        nats, store = _Nats(), _Store()
        store.rows[_ID] = {"id": _ID, "members": '["x"]'}
        reader, _ = _replica(_Synchronous, nats, store)
        writer, _ = _replica(_Synchronous, nats, store)
        release = store.hold_next_fetch()
        store.held.clear()
        read = asyncio.create_task(reader.get(_ID))
        await store.held.wait()  # the reader has missed L2 and is reading L3
        await writer.l2_cas_mutate(_ID, _add("y"))
        release.set()
        await read
        raw = await (await _bucket(nats)).get(key=_KEY)
        assert raw is not None
        assert _members(json.loads(raw)) == ["x", "y"], "a read put L3's older row over a swapped L2 value"
        outcome = await writer.l2_cas_mutate(_ID, _add("z"))
        assert outcome.row is not None and _members(outcome.row) == ["x", "y", "z"]


class TestOnlyASwapStoresAnOrder:
    @pytest.mark.asyncio
    async def test_save_entity_on_an_ordered_collection_stores_no_order(self) -> None:
        nats, store = _Nats(), _Store()
        coll, _ = _replica(_Synchronous, nats, store)
        copied = with_l2_order({"id": _ID, "members": '["x"]'}, L2Order(datetime(2026, 9, 26, tzinfo=UTC), 9))
        await coll.save_entity(coll.create(copied))
        assert l2_order_of(store.rows[_ID]) is None, "a write that won no swap stored an order it copied"


class _BroadcastingNats(_Nats):
    """the shared bucket, and the invalidation broadcast delivered to every listening replica."""

    def __init__(self) -> None:
        super().__init__()
        self._subscribers: list[tuple[Any, Any]] = []

    async def publish(self, *, subject: Any, message: Any, reply_to: Any = None) -> None:
        for cb, message_type in list(self._subscribers):
            await cb(message_type.model_validate_json(message.model_dump_json()))

    async def subscribe_typed(self, *, subject: Any, cb: Any, message_type: Any, **_: Any) -> object:
        self._subscribers.append((cb, message_type))
        return object()


class TestASwapAnsweredLateLeavesNoOlderRowInL1:
    """a winner whose persist answers after a later swap must not cache its row over the later one.

    The L1 write follows the L3 persist, and the persist is a round trip. A later swap -- this
    replica's own, or a peer's whose broadcast evicts this replica's L1 -- can complete inside that
    round trip. Caching the earlier winner's row after it leaves L1 behind L2 with nothing left to
    evict it: every read on this replica is served the older value.
    """

    @pytest.mark.asyncio
    async def test_a_superseded_winner_on_the_same_replica_leaves_the_later_row_served(self) -> None:
        nats, store = _Nats(), _Store()
        coll, _ = _replica(_Synchronous, nats, store)
        await _reverse_delivery(coll, coll, store, _add("x"), _add("y"))
        served = await coll.get(_ID)
        assert served is not None
        assert _members(served.to_dict()) == ["x", "y"], "the earlier winner cached its row over the later one"

    @pytest.mark.asyncio
    async def test_a_winner_whose_commit_answers_late_leaves_the_later_row_served(self) -> None:
        nats, store = _Nats(), _Store()
        coll, _ = _replica(_Synchronous, nats, store)
        release = store.hold_next_write_ack()
        store.held.clear()
        held = asyncio.create_task(coll.l2_cas_mutate(_ID, _add("x")))
        await store.held.wait()  # "x" is committed in L3; its answer is in flight
        await coll.l2_cas_mutate(_ID, _add("y"))
        release.set()
        await held
        assert _members(store.rows[_ID]) == ["x", "y"]
        served = await coll.get(_ID)
        assert served is not None
        assert _members(served.to_dict()) == ["x", "y"], "the earlier winner cached its row over the later one"

    @pytest.mark.asyncio
    async def test_a_peers_later_swap_is_not_undone_by_this_replicas_late_answer(self) -> None:
        nats, store = _BroadcastingNats(), _Store()
        slow, slow_registry = _replica(_Synchronous, nats, store)
        peer, peer_registry = _replica(_Synchronous, nats, store)
        await slow_registry.start_invalidation_listener(nats)  # type: ignore[arg-type]
        await peer_registry.start_invalidation_listener(nats)  # type: ignore[arg-type]
        await _reverse_delivery(slow, peer, store, _add("x"), _add("y"))
        served = await slow.get(_ID)
        assert served is not None
        assert _members(served.to_dict()) == ["x", "y"], "a late answer cached a row the peer's broadcast retracted"

    @pytest.mark.asyncio
    async def test_the_newest_winner_still_serves_its_own_write_from_l1(self) -> None:
        nats, store = _Nats(), _Store()
        coll, _ = _replica(_Synchronous, nats, store)
        await coll.l2_cas_mutate(_ID, _add("x"))
        assert coll.exists_in_cache_sync(_ID), "an uncontended winner no longer caches its own row"
