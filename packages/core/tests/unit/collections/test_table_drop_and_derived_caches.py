"""what a registry does with the row messages it hears, and what dropping a table does.

Two registries on one in-process bus stand in for two pods. The contract this pins:

- a heard row message evicts its row exactly as before, is counted against the generation it
  names, and reaches every cache derived from the table, row by row;
- a message from a sender that names no generation is handled as it always was and counts nothing;
- a writer's own derived caches hear its writes, which its listener skips;
- a derived cache is told to drop everything ONLY when the table drops;
- dropping a table removes its rows from L1, its cached scans, and stops the pod trusting its own
  L2 entries for the table until each has been read through from L3 again.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Any, ClassVar, Literal

import pytest
from sqlalchemy import BigInteger, Column, DateTime, MetaData, String, Table

from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections import (
    WRITE_GENERATION,
    BaseCollection,
    CacheInvalidationMessage,
    CollectionRegistry,
    GenerationVerdict,
    WriteBuffer,
)
from threetears.core.config import DefaultCoreConfig
from threetears.core.entities.base import BaseEntity
from threetears.core.testing.kv import FakeNatsClient
from threetears.nats import Subjects, set_default_namespace

_TABLE = "group_members"


@pytest.fixture(autouse=True)
def _namespace() -> None:
    set_default_namespace("dropprobe")


def _metadata() -> MetaData:
    metadata = MetaData()
    Table(
        _TABLE,
        metadata,
        Column("id", String(255), primary_key=True),
        Column("member_id", String(255)),
        Column("date_created", DateTime(timezone=True)),
        Column("date_updated", DateTime(timezone=True)),
        # the order a compare-and-swap's row carries, for the collection whose rows one orders
        Column("l2_epoch", DateTime(timezone=True)),
        Column("l2_revision", BigInteger),
    )
    return metadata


class _Member(BaseEntity):
    primary_key_field = "id"


class _Source:
    """one generation shared by every pod of a test, counted in memory."""

    def __init__(self) -> None:
        self.count = 0

    async def current(self, table_name: str) -> str:
        return f"inc:{self.count}"

    async def advance(self, table_name: str) -> str:
        self.count += 1
        return f"inc:{self.count}"


class _Members(BaseCollection[_Member]):
    """a switched-on membership table over an L3 dict every pod of the test shares."""

    datetime_columns: ClassVar[frozenset[str]] = frozenset({"date_created", "date_updated"})
    write_generation = WRITE_GENERATION
    invalidation_columns: ClassVar[tuple[str, ...]] = ("member_id",)

    def __init__(
        self, registry: CollectionRegistry, rows: dict[str, dict[str, Any]], *, write_buffer: WriteBuffer | None = None
    ) -> None:
        self._rows = rows
        self.fetches = 0
        super().__init__(registry, DefaultCoreConfig(), write_buffer=write_buffer)

    @property
    def table_name(self) -> str:
        return _TABLE

    @property
    def entity_class(self) -> type[_Member]:
        return _Member

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


class _WriteBehindMembers(_Members):
    """L2 is ahead of L3 by design: a save reaches L3 only at the next flush."""

    l3_write_policy: ClassVar[Literal["synchronous", "write_behind"] | None] = "write_behind"


class _OrderedMembers(_Members):
    """rows a compare-and-swap orders: L2 holds the newest swap, L3 catches up behind it."""

    datetime_columns: ClassVar[frozenset[str]] = frozenset({"date_created", "date_updated", "l2_epoch"})

    @property
    def persists_l2_order(self) -> bool:
        return True

    async def save_ordered_to_store(self, data: dict[str, Any], *, conn: Any = None) -> int:
        self._rows[str(data["id"])] = dict(data)
        return 1


class _L2OnlyMembers(_Members):
    """no durable tier: L2 is the source of truth, and nothing behind it holds a row."""

    async def fetch_from_store(self, entity_id: Any) -> dict[str, Any] | None:
        self.fetches += 1
        return None

    async def save_to_store(
        self, data: dict[str, Any], original_timestamp: datetime | None = None, *, conn: Any = None
    ) -> int:
        return 1


class _ForgetfulMembers(_Members):
    """remembers only two keys read through since a drop, so the third pushes the first out."""

    L2_READ_THROUGH_LIMIT: ClassVar[int] = 2


class _PersonCache:
    """a cache derived from the membership table, keyed by the person, not by the row."""

    def __init__(self, registry: CollectionRegistry, people: tuple[str, ...]) -> None:
        self.entries = {person: f"access-of-{person}" for person in people}
        self.dropped_all = 0
        self.unmapped: list[CacheInvalidationMessage] = []
        registry.register_derived_cache(_TABLE, on_row=self._on_row, on_table_dropped=self._on_table_dropped)

    def _on_row(self, message: CacheInvalidationMessage) -> None:
        if message.columns is None:
            # a row this cache has no index for
            self.unmapped.append(message)
        else:
            self.entries.pop(str(message.columns["member_id"]), None)

    def _on_table_dropped(self) -> None:
        self.dropped_all += 1
        self.entries.clear()


class _Pod:
    """one registry with its own L1 and listener, on the shared bus, L3 and generation source."""

    def __init__(
        self,
        bus: FakeNatsClient,
        rows: dict[str, dict[str, Any]],
        source: _Source,
        *,
        scope: str,
        cls: type[_Members] = _Members,
        l3: bool = True,
        write_buffer: WriteBuffer | None = None,
    ) -> None:
        l1 = SQLiteBackend(db_name=f"table_drop_{uuid.uuid4().hex[:8]}")
        l1.initialize(_metadata())
        self.bus = bus
        self.registry = CollectionRegistry()
        self.registry.configure(
            l1_backend=l1,
            l2_client=bus,
            l3_pool=object() if l3 else None,  # type: ignore[arg-type]
            kv_key_scope=scope,
        )
        self.registry.set_generation_source(source)
        self.members = cls(self.registry, rows, write_buffer=write_buffer)

    async def listen(self) -> None:
        await self.registry.start_invalidation_listener(self.bus)  # type: ignore[arg-type]

    async def follow(self, source: _Source) -> None:
        """follow the table and take the first mark, as a pod does once at startup."""
        self.registry.follow_generation(_TABLE)
        self.registry.settle_generation(_TABLE, await source.current(_TABLE))


async def _save(pod: _Pod, entity_id: str, member_id: str) -> None:
    await pod.members.save_entity(pod.members.create({"id": entity_id, "member_id": member_id}))


async def _member_id(pod: _Pod, entity_id: str) -> str | None:
    """the member a pod's three-tier read of one row names, or ``None`` when no tier holds the row."""
    row = await pod.members.ensure(entity_id)
    return None if row is None else str(row["member_id"])


class _Lossy(FakeNatsClient):
    """a bus that can be told to lose the row messages published while it is deaf."""

    def __init__(self) -> None:
        super().__init__()
        self.deaf = False

    async def publish(self, *, subject: Any, message: Any, reply_to: Any = None) -> None:
        if self.deaf:
            return
        await super().publish(subject=subject, message=message, reply_to=reply_to)


@pytest.fixture
def bus() -> _Lossy:
    return _Lossy()


@pytest.fixture
def source() -> _Source:
    return _Source()


class TestAHeardRowIsEvictedCountedAndPassedOnRowByRow:
    async def test_a_peers_write_evicts_exactly_the_entry_it_reaches(self, bus: _Lossy, source: _Source) -> None:
        rows: dict[str, dict[str, Any]] = {}
        writer = _Pod(bus, rows, source, scope="writer")
        reader = _Pod(bus, rows, source, scope="reader")
        await reader.listen()
        await reader.follow(source)
        cache = _PersonCache(reader.registry, ("ada", "grace", "edsger"))

        await _save(writer, "m1", "grace")

        assert cache.entries == {"ada": "access-of-ada", "edsger": "access-of-edsger"}
        assert cache.dropped_all == 0
        # every row of the advance was heard, so the reader is not behind and drops nothing
        assert reader.registry.settle_generation(_TABLE, await source.current(_TABLE)) is GenerationVerdict.CURRENT
        assert cache.dropped_all == 0

    async def test_a_delete_reaches_the_entry_of_the_row_it_removed(self, bus: _Lossy, source: _Source) -> None:
        rows: dict[str, dict[str, Any]] = {}
        writer = _Pod(bus, rows, source, scope="writer")
        reader = _Pod(bus, rows, source, scope="reader")
        await reader.listen()
        await _save(writer, "m1", "grace")
        cache = _PersonCache(reader.registry, ("ada", "grace"))

        await writer.members.delete("m1")

        assert cache.entries == {"ada": "access-of-ada"}
        assert cache.dropped_all == 0

    async def test_a_writers_own_derived_cache_hears_its_writes(self, bus: _Lossy, source: _Source) -> None:
        writer = _Pod(bus, {}, source, scope="writer")
        await writer.listen()
        cache = _PersonCache(writer.registry, ("ada", "grace"))
        await _save(writer, "m1", "ada")
        assert cache.entries == {"grace": "access-of-grace"}
        assert cache.dropped_all == 0

    async def test_a_cache_on_a_pod_that_holds_no_collection_for_the_table_still_hears(
        self, bus: _Lossy, source: _Source
    ) -> None:
        # an agent pod holds caches derived from the access tables and none of the tables
        writer = _Pod(bus, {}, source, scope="writer")
        bare = CollectionRegistry()
        await bare.start_invalidation_listener(bus)  # type: ignore[arg-type]
        cache = _PersonCache(bare, ("ada", "grace"))
        await _save(writer, "m1", "ada")
        assert cache.entries == {"grace": "access-of-grace"}

    async def test_a_row_without_columns_is_handed_over_as_it_is(self, bus: _Lossy, source: _Source) -> None:
        writer = _Pod(bus, {}, source, scope="writer")
        reader = _Pod(bus, {}, source, scope="reader")
        await reader.listen()
        cache = _PersonCache(reader.registry, ("ada",))
        await writer.members.invalidate_cache("m9")
        assert [message.ids for message in cache.unmapped] == [["m9"]]
        assert cache.dropped_all == 0

    async def test_an_unregistered_cache_hears_nothing_more(self, bus: _Lossy, source: _Source) -> None:
        writer = _Pod(bus, {}, source, scope="writer")
        heard: list[str] = []
        remove = writer.registry.register_derived_cache(
            _TABLE, on_row=lambda message: heard.append(message.ids[0]), on_table_dropped=lambda: heard.append("ALL")
        )
        await _save(writer, "m1", "ada")
        remove()
        await _save(writer, "m2", "ada")
        writer.registry.drop_table(_TABLE, reason="test")
        assert heard == ["m1"]


class TestAMessageFromAnOlderSenderIsHandledAsItAlwaysWas:
    async def test_it_evicts_its_row_and_counts_nothing(self, bus: _Lossy, source: _Source) -> None:
        rows: dict[str, dict[str, Any]] = {}
        writer = _Pod(bus, rows, source, scope="writer")
        reader = _Pod(bus, rows, source, scope="reader")
        await reader.listen()
        await reader.follow(source)
        await _save(writer, "m1", "ada")
        assert await reader.members.get("m1") is not None
        assert reader.members.exists_in_cache_sync("m1")
        mark = reader.registry.generation_marks.recorded(_TABLE)
        cache = _PersonCache(reader.registry, ("ada",))

        # exactly the bytes a sender one release back puts on the wire: no generation, no row
        # count, no columns
        old = json.dumps({"table": _TABLE, "ids": ["m1"], "origin": "an-older-pod", "l2_current_scope": None})
        await bus.publish(
            subject=Subjects.cache_invalidate(), message=CacheInvalidationMessage.model_validate_json(old)
        )

        assert not reader.members.exists_in_cache_sync("m1")
        assert reader.registry.generation_marks.recorded(_TABLE) == mark
        assert [message.ids for message in cache.unmapped] == [["m1"]]
        assert cache.dropped_all == 0

    def test_a_receiver_one_release_back_ignores_the_new_fields(self) -> None:
        from pydantic import BaseModel

        class _MessageAsItWas(BaseModel):
            table: str
            ids: list[str]
            origin: str | None = None
            l2_current_scope: str | None = None

        new = CacheInvalidationMessage(
            table=_TABLE, ids=["m1"], origin="o", generation="inc:4", bump_rows=3, columns={"member_id": "ada"}
        )
        parsed = _MessageAsItWas.model_validate_json(new.model_dump_json())
        assert (parsed.table, parsed.ids, parsed.origin) == (_TABLE, ["m1"], "o")

    def test_a_message_without_the_new_fields_parses_with_none_of_them_set(self) -> None:
        parsed = CacheInvalidationMessage.model_validate_json(json.dumps({"table": _TABLE, "ids": ["m1"]}))
        assert (parsed.generation, parsed.bump_rows, parsed.columns) == (None, None, None)


class TestDroppingATable:
    async def test_a_missed_row_drops_the_table_and_everything_derived_from_it(
        self, bus: _Lossy, source: _Source
    ) -> None:
        rows: dict[str, dict[str, Any]] = {}
        writer = _Pod(bus, rows, source, scope="writer")
        reader = _Pod(bus, rows, source, scope="reader")
        await reader.listen()
        await reader.follow(source)
        await _save(writer, "m1", "ada")
        await _save(writer, "m2", "grace")
        assert await _member_id(reader, "m1") == "ada"
        assert await _member_id(reader, "m2") == "grace"
        cache = _PersonCache(reader.registry, ("ada", "grace", "edsger"))

        bus.deaf = True
        await _save(writer, "m1", "edsger")
        bus.deaf = False
        # nothing told the reader, so it still serves the row it cached
        assert await _member_id(reader, "m1") == "ada"

        verdict = reader.registry.settle_generation(_TABLE, await source.current(_TABLE))

        assert verdict is GenerationVerdict.MISSED
        assert not reader.members.exists_in_cache_sync("m1")
        assert not reader.members.exists_in_cache_sync("m2")
        assert cache.entries == {}
        assert cache.dropped_all == 1
        assert await _member_id(reader, "m1") == "edsger"

    async def test_the_pods_own_stale_l2_entry_is_not_served_after_a_drop(self, bus: _Lossy, source: _Source) -> None:
        # the reader's L2 key is under its OWN scope: the writer's save never touches it, and the
        # broadcast that would have made the reader delete it is the one that was lost
        rows: dict[str, dict[str, Any]] = {}
        writer = _Pod(bus, rows, source, scope="writer")
        reader = _Pod(bus, rows, source, scope="reader")
        await reader.listen()
        await reader.follow(source)
        await _save(writer, "m1", "ada")
        await reader.members.get("m1")
        l2 = await bus.kv_bucket(name="collections")
        stale_key = reader.members.l2_key("m1")
        assert json.loads(await l2.get(key=stale_key))["member_id"] == "ada"  # type: ignore[arg-type]

        bus.deaf = True
        await _save(writer, "m1", "edsger")
        bus.deaf = False
        reader.registry.drop_table(_TABLE, reason="test")

        fetches = reader.members.fetches
        assert await _member_id(reader, "m1") == "edsger"
        assert reader.members.fetches == fetches + 1, "the stale L2 entry was read past, to L3"
        # and replaced, so the tier is useful again: the next cold read is answered by L2
        assert json.loads(await l2.get(key=stale_key))["member_id"] == "edsger"  # type: ignore[arg-type]
        reader.members.evict_from_cache_sync("m1")
        assert await _member_id(reader, "m1") == "edsger"
        assert reader.members.fetches == fetches + 1

    async def test_a_stale_l2_entry_of_a_row_deleted_since_is_removed(self, bus: _Lossy, source: _Source) -> None:
        rows: dict[str, dict[str, Any]] = {}
        writer = _Pod(bus, rows, source, scope="writer")
        reader = _Pod(bus, rows, source, scope="reader")
        await reader.listen()
        await _save(writer, "m1", "ada")
        await reader.members.get("m1")
        l2 = await bus.kv_bucket(name="collections")

        bus.deaf = True
        await writer.members.delete("m1")
        bus.deaf = False
        reader.registry.drop_table(_TABLE, reason="test")

        assert await reader.members.get("m1") is None
        assert await l2.get(key=reader.members.l2_key("m1")) is None

    async def test_a_drop_stops_a_read_in_flight_caching_what_it_read(self, bus: _Lossy, source: _Source) -> None:
        rows: dict[str, dict[str, Any]] = {"m1": {"id": "m1", "member_id": "ada"}}
        reader = _Pod(bus, rows, source, scope="reader")
        inner = reader.members.fetch_from_store

        async def fetch_then_drop(entity_id: Any) -> dict[str, Any] | None:
            row = await inner(entity_id)
            # the table drops while this read is between its L3 query and its L1 write
            reader.registry.drop_table(_TABLE, reason="test")
            return row

        reader.members.fetch_from_store = fetch_then_drop  # type: ignore[method-assign]
        assert await _member_id(reader, "m1") == "ada"
        assert not reader.members.exists_in_cache_sync("m1")

    async def test_dropping_drops_the_tables_cached_scans(
        self, bus: _Lossy, source: _Source, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        reader = _Pod(bus, {}, source, scope="reader")
        dropped: list[str] = []
        scans = reader.registry.scan_cache
        monkeypatch.setattr(type(scans), "drop_for_table", lambda self, table: dropped.append(table))
        reader.registry.drop_table(_TABLE, reason="test")
        assert dropped == [_TABLE]

    async def test_a_table_this_pod_holds_no_collection_for_still_drops_its_derived_caches(
        self, source: _Source
    ) -> None:
        bare = CollectionRegistry()
        cache = _PersonCache(bare, ("ada",))
        bare.follow_generation(_TABLE)
        assert bare.settle_generation(_TABLE, "inc:3") is GenerationVerdict.FIRST_SIGHT
        assert cache.dropped_all == 1
        assert bare.settle_generation(_TABLE, "inc:3") is GenerationVerdict.CURRENT
        assert cache.dropped_all == 1

    async def test_every_l1_change_listener_hears_each_row_go(self, bus: _Lossy, source: _Source) -> None:
        rows: dict[str, dict[str, Any]] = {}
        reader = _Pod(bus, rows, source, scope="reader")
        await _save(reader, "m1", "ada")
        await _save(reader, "m2", "grace")
        gone: list[Any] = []
        reader.members.add_l1_change_listener(gone.append)
        reader.registry.drop_table(_TABLE, reason="test")
        assert sorted(gone) == ["m1", "m2"]


class TestADropDoesNotPutAnOlderL3RowOverANewerL2Row:
    """where L2 is ahead of L3 by design, a drop leaves L2 trusted: reading L3 would move readers back."""

    async def _l2_member_id(self, pod: _Pod, entity_id: str) -> str:
        l2 = await pod.bus.kv_bucket(name="collections")
        raw = await l2.get(key=pod.members.l2_key(entity_id))
        assert raw is not None
        return str(json.loads(raw)["member_id"])

    async def test_a_write_behind_table(self, bus: _Lossy, source: _Source) -> None:
        rows: dict[str, dict[str, Any]] = {"m1": {"id": "m1", "member_id": "older"}}
        pod = _Pod(bus, rows, source, scope="reader", cls=_WriteBehindMembers, write_buffer=WriteBuffer())
        await _save(pod, "m1", "newer")
        assert rows["m1"]["member_id"] == "older", "not flushed yet: L3 is behind L2"

        pod.registry.drop_table(_TABLE, reason="test")

        assert await _member_id(pod, "m1") == "newer"
        assert await self._l2_member_id(pod, "m1") == "newer"
        assert rows["m1"]["member_id"] == "older"

    async def test_a_table_whose_rows_a_compare_and_swap_orders(self, bus: _Lossy, source: _Source) -> None:
        rows: dict[str, dict[str, Any]] = {}
        pod = _Pod(bus, rows, source, scope="reader", cls=_OrderedMembers)

        def newer(row: dict[str, Any] | None) -> tuple[Literal["upsert"], dict[str, Any]]:
            del row
            return "upsert", {"id": "m1", "member_id": "newer"}

        await pod.members.l2_cas_mutate("m1", newer)
        # an earlier winner's persist landing late, so L3 is behind the swap L2 holds
        rows["m1"] = {**rows["m1"], "member_id": "older"}

        pod.registry.drop_table(_TABLE, reason="test")

        assert await _member_id(pod, "m1") == "newer"
        assert await self._l2_member_id(pod, "m1") == "newer"

    async def test_a_table_with_no_l3(self, bus: _Lossy, source: _Source) -> None:
        pod = _Pod(bus, {}, source, scope="reader", cls=_L2OnlyMembers, l3=False)
        await _save(pod, "m1", "ada")
        pod.members.evict_from_cache_sync("m1")

        pod.registry.drop_table(_TABLE, reason="test")

        # L2 is the only tier there is; distrusting it would answer that the row does not exist
        assert await _member_id(pod, "m1") == "ada"


class TestWhenAKeyIsTrustedAgainAfterADrop:
    async def test_a_key_read_through_once_is_answered_by_l2_again(self, bus: _Lossy, source: _Source) -> None:
        rows: dict[str, dict[str, Any]] = {"m1": {"id": "m1", "member_id": "ada"}}
        pod = _Pod(bus, rows, source, scope="reader")
        assert await _member_id(pod, "m1") == "ada"
        pod.registry.drop_table(_TABLE, reason="test")
        fetches = pod.members.fetches
        assert await _member_id(pod, "m1") == "ada"
        assert pod.members.fetches == fetches + 1
        pod.members.evict_from_cache_sync("m1")
        assert await _member_id(pod, "m1") == "ada"
        assert pod.members.fetches == fetches + 1, "read through once, the key is trusted again"

    async def test_a_drop_during_the_read_through_leaves_the_key_distrusted(self, bus: _Lossy, source: _Source) -> None:
        rows: dict[str, dict[str, Any]] = {"m1": {"id": "m1", "member_id": "ada"}}
        pod = _Pod(bus, rows, source, scope="reader")
        assert await _member_id(pod, "m1") == "ada"
        pod.registry.drop_table(_TABLE, reason="test")
        inner = pod.members.fetch_from_store
        dropped_again = False

        async def fetch_then_drop(entity_id: Any) -> dict[str, Any] | None:
            nonlocal dropped_again
            row = await inner(entity_id)
            if not dropped_again:
                # a second drop lands while this read is between L3 and re-trusting the key: what
                # it read predates the second drop, so it cannot vouch for the key after it
                dropped_again = True
                pod.registry.drop_table(_TABLE, reason="test")
            return row

        pod.members.fetch_from_store = fetch_then_drop  # type: ignore[method-assign]
        assert await _member_id(pod, "m1") == "ada"
        pod.members.evict_from_cache_sync("m1")
        fetches = pod.members.fetches
        assert await _member_id(pod, "m1") == "ada"
        assert pod.members.fetches == fetches + 1, "the key is still distrusted, so the read went to L3"

    async def test_the_keys_remembered_since_a_drop_are_bounded(self, bus: _Lossy, source: _Source) -> None:
        rows: dict[str, dict[str, Any]] = {f"m{i}": {"id": f"m{i}", "member_id": f"p{i}"} for i in range(3)}
        pod = _Pod(bus, rows, source, scope="reader", cls=_ForgetfulMembers)
        for i in range(3):
            assert await _member_id(pod, f"m{i}") == f"p{i}"
        pod.registry.drop_table(_TABLE, reason="test")
        for i in range(3):
            assert await _member_id(pod, f"m{i}") == f"p{i}"
            pod.members.evict_from_cache_sync(f"m{i}")
        fetches = pod.members.fetches

        # the newest two are remembered and answered by L2; the oldest was forgotten, which only
        # costs it one more read through L3 -- the safe direction
        assert await _member_id(pod, "m2") == "p2"
        assert await _member_id(pod, "m1") == "p1"
        assert pod.members.fetches == fetches
        assert await _member_id(pod, "m0") == "p0"
        assert pod.members.fetches == fetches + 1


class TestADerivedCacheThatFails:
    async def test_it_does_not_cost_the_row_its_eviction_and_the_row_is_not_counted_as_heard(
        self, bus: _Lossy, source: _Source
    ) -> None:
        rows: dict[str, dict[str, Any]] = {}
        writer = _Pod(bus, rows, source, scope="writer")
        reader = _Pod(bus, rows, source, scope="reader")
        await reader.listen()
        await reader.follow(source)
        await _save(writer, "m1", "ada")
        assert await _member_id(reader, "m1") == "ada"

        def broken(message: CacheInvalidationMessage) -> None:
            raise RuntimeError("a bug in a derived cache")

        reader.registry.register_derived_cache(_TABLE, on_row=broken, on_table_dropped=lambda: None)
        with pytest.raises(RuntimeError, match="derived cache"):
            await _save(writer, "m1", "grace")

        # the row was evicted all the same, so the reader does not go on serving the old one
        assert not reader.members.exists_in_cache_sync("m1")
        # and the message was not counted, so the next pass drops the table
        assert reader.registry.settle_generation(_TABLE, await source.current(_TABLE)) is GenerationVerdict.MISSED
