"""a collection switched on to carry a write generation, and one that is not.

The contract this pins:

- a collection that declares nothing writes exactly as it always did: no advance, and a row
  message with none of the new fields set;
- a switched-on collection advances once per commit on EVERY write path -- a save, a delete, a
  subscript write, an eviction, a caller's transaction settling, a cache-bypassing write, a flush
  of the write buffer -- never once per row, and every row message of that commit names the
  generation and the number of rows;
- an opted-out collection and one built with ``NO_L2`` advance nothing;
- a failed advance after a committed write raises, once the rest of the write path has run;
- declared invalidation columns ride on the message, from the row the write saw;
- a registry refuses two collections for one table that disagree, and a class refuses absence
  caching together with an opt-out.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Any, ClassVar, Literal

import pytest
from sqlalchemy import Column, DateTime, MetaData, String, Table

from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections import (
    NO_L2,
    WRITE_GENERATION,
    WRITE_GENERATION_UNDECLARED,
    BaseCollection,
    CacheInvalidationMessage,
    CallerTransaction,
    CollectionRegistry,
    NoWriteGeneration,
    WriteBuffer,
    flush_pending,
)
from threetears.core.collections.generation import GenerationSource
from threetears.core.config import DefaultCoreConfig
from threetears.core.entities.base import BaseEntity
from threetears.core.exceptions import GenerationUnavailableError
from threetears.core.testing.kv import FakeNatsClient

_TABLE = "group_members"
_SCOPE = "test-principal"


def _metadata(table: str = _TABLE) -> MetaData:
    metadata = MetaData()
    Table(
        table,
        metadata,
        Column("id", String(255), primary_key=True),
        Column("member_type", String(255)),
        Column("member_id", String(255)),
        Column("date_created", DateTime(timezone=True)),
        Column("date_updated", DateTime(timezone=True)),
    )
    return metadata


class _Member(BaseEntity):
    primary_key_field = "id"


#: build the collection with whatever L2 client its registry offers
_FROM_REGISTRY = object()


async def _background_writes() -> None:
    """wait for the subscript writes this test started; they run as tasks on its own loop."""
    pending = [task for task in asyncio.all_tasks() if task is not asyncio.current_task()]
    if pending:
        await asyncio.gather(*pending)


class _CountingSource:
    """a generation source that counts its advances, and can be made to refuse them."""

    def __init__(self) -> None:
        self.counts: dict[str, int] = {}
        self.failing = False

    async def current(self, table_name: str) -> str:
        return f"inc:{self.counts.get(table_name, 0)}"

    async def advance(self, table_name: str) -> str:
        if self.failing:
            raise GenerationUnavailableError(f"generation for {table_name!r} is unreachable")
        self.counts[table_name] = self.counts.get(table_name, 0) + 1
        return f"inc:{self.counts[table_name]}"


class _Members(BaseCollection[_Member]):
    """an in-process three-tier collection: SQLite L1, the fake bus's L2, a dict standing in for L3."""

    datetime_columns: ClassVar[frozenset[str]] = frozenset({"date_created", "date_updated"})

    def __init__(
        self,
        registry: CollectionRegistry,
        rows: dict[str, dict[str, Any]],
        *,
        nats_client: Any = _FROM_REGISTRY,
        write_buffer: WriteBuffer | None = None,
        table: str = _TABLE,
    ) -> None:
        self._table = table
        self._rows = rows
        if nats_client is _FROM_REGISTRY:
            super().__init__(registry, DefaultCoreConfig(), write_buffer=write_buffer)
        else:
            super().__init__(registry, DefaultCoreConfig(), nats_client, write_buffer=write_buffer)

    @property
    def table_name(self) -> str:
        return self._table

    @property
    def entity_class(self) -> type[_Member]:
        return _Member

    async def fetch_from_store(self, entity_id: Any) -> dict[str, Any] | None:
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


class _UndeclaredMembers(_Members):
    """as every collection is until it declares otherwise."""


class _SwitchedOnMembers(_Members):
    write_generation = WRITE_GENERATION
    invalidation_columns: ClassVar[tuple[str, ...]] = ("member_type", "member_id")


class _OptedOutMembers(_Members):
    write_generation = NoWriteGeneration(reason="written on every turn; nothing reads a row by key")


class _WriteBehindMembers(_SwitchedOnMembers):
    l3_write_policy: ClassVar[Literal["synchronous", "write_behind"] | None] = "write_behind"


class _Pod:
    """one registry with its own L1, on a bus and an L3 it shares with every other pod of the test."""

    def __init__(
        self,
        cls: type[_Members],
        bus: FakeNatsClient,
        rows: dict[str, dict[str, Any]],
        *,
        source: GenerationSource | None,
        nats_client: Any = _FROM_REGISTRY,
        write_buffer: WriteBuffer | None = None,
    ) -> None:
        l1 = SQLiteBackend(db_name=f"write_generation_{uuid.uuid4().hex[:8]}")
        l1.initialize(_metadata())
        self.registry = CollectionRegistry()
        self.registry.configure(l1_backend=l1, l2_client=bus, l3_pool=object(), kv_key_scope=_SCOPE)  # type: ignore[arg-type]
        if source is not None:
            self.registry.set_generation_source(source)
        self.collection = cls(self.registry, rows, nats_client=nats_client, write_buffer=write_buffer)


def _messages(bus: FakeNatsClient) -> list[CacheInvalidationMessage]:
    return [message for message in bus.published if isinstance(message, CacheInvalidationMessage)]


async def _save(collection: _Members, entity_id: str, member_id: str = "person-1") -> None:
    await collection.save_entity(collection.create({"id": entity_id, "member_type": "user", "member_id": member_id}))


class _Conn:
    """a connection whose transaction does nothing, for a CallerTransaction to open."""

    @asynccontextmanager
    async def _transaction(self) -> AsyncIterator[None]:
        yield

    def transaction(self, **options: Any) -> Any:
        return self._transaction()


@pytest.fixture
def bus() -> FakeNatsClient:
    return FakeNatsClient()


@pytest.fixture
def source() -> _CountingSource:
    return _CountingSource()


class TestACollectionThatDeclaresNothingWritesAsItAlwaysDid:
    def test_undeclared_is_the_default(self) -> None:
        assert BaseCollection.write_generation is WRITE_GENERATION_UNDECLARED
        assert _UndeclaredMembers.write_generation is WRITE_GENERATION_UNDECLARED
        assert BaseCollection.invalidation_columns == ()

    async def test_no_write_path_advances_and_no_message_carries_a_new_field(
        self, bus: FakeNatsClient, source: _CountingSource
    ) -> None:
        pod = _Pod(_UndeclaredMembers, bus, {}, source=source)
        members = pod.collection
        await _save(members, "m1")
        members["m1", "member_id"] = "person-2"
        await _background_writes()
        await members.invalidate_cache("m1")
        await members.invalidate_cache_many(["m1", "m2"])
        async with members.bypassing_write("m1"):
            pass
        async with CallerTransaction(_Conn()) as transaction:
            transaction.enroll(members, "m1")
        await members.delete("m1")

        assert source.counts == {}
        published = _messages(bus)
        assert published, "the writes still broadcast their rows"
        for message in published:
            assert message.model_dump(exclude={"origin"}) == {
                "table": _TABLE,
                "ids": message.ids,
                "l2_current_scope": None,
                "generation": None,
                "bump_rows": None,
                "columns": None,
            }


class TestASwitchedOnCollectionAdvancesOncePerCommit:
    async def test_a_save_advances_once_and_its_message_names_the_advance(
        self, bus: FakeNatsClient, source: _CountingSource
    ) -> None:
        members = _Pod(_SwitchedOnMembers, bus, {}, source=source).collection
        await _save(members, "m1")
        assert source.counts == {_TABLE: 1}
        (message,) = _messages(bus)
        assert (message.generation, message.bump_rows) == ("inc:1", 1)
        assert message.columns == {"member_type": "user", "member_id": "person-1"}

    async def test_a_delete_advances_and_carries_the_row_it_removed(
        self, bus: FakeNatsClient, source: _CountingSource
    ) -> None:
        members = _Pod(_SwitchedOnMembers, bus, {}, source=source).collection
        await _save(members, "m1", member_id="person-7")
        bus.published.clear()
        await members.delete("m1")
        assert source.counts == {_TABLE: 2}
        (message,) = _messages(bus)
        assert (message.generation, message.bump_rows) == ("inc:2", 1)
        # the row is gone from every tier; the message is the only place left that says whose it was
        assert message.columns == {"member_type": "user", "member_id": "person-7"}
        assert await members.get("m1") is None

    async def test_a_subscript_write_advances(self, bus: FakeNatsClient, source: _CountingSource) -> None:
        members = _Pod(_SwitchedOnMembers, bus, {}, source=source).collection
        await _save(members, "m1")
        bus.published.clear()
        members["m1", "member_id"] = "person-2"
        await _background_writes()
        assert source.counts == {_TABLE: 2}
        (message,) = _messages(bus)
        assert (message.generation, message.bump_rows) == ("inc:2", 1)
        assert message.columns == {"member_type": "user", "member_id": "person-2"}

    async def test_an_eviction_of_one_key_advances(self, bus: FakeNatsClient, source: _CountingSource) -> None:
        members = _Pod(_SwitchedOnMembers, bus, {}, source=source).collection
        await members.invalidate_cache("m1")
        assert source.counts == {_TABLE: 1}
        (message,) = _messages(bus)
        assert (message.generation, message.bump_rows) == ("inc:1", 1)
        # an eviction names a key and saw no row
        assert message.columns is None

    async def test_many_keys_evicted_together_are_one_advance(
        self, bus: FakeNatsClient, source: _CountingSource
    ) -> None:
        members = _Pod(_SwitchedOnMembers, bus, {}, source=source).collection
        await members.invalidate_cache_many([f"m{i}" for i in range(7)])
        assert source.counts == {_TABLE: 1}
        published = _messages(bus)
        assert sorted(message.ids[0] for message in published) == sorted(f"m{i}" for i in range(7))
        assert {(message.generation, message.bump_rows) for message in published} == {("inc:1", 7)}

    async def test_evicting_no_keys_advances_nothing(self, bus: FakeNatsClient, source: _CountingSource) -> None:
        members = _Pod(_SwitchedOnMembers, bus, {}, source=source).collection
        await members.invalidate_cache_many([])
        assert source.counts == {}

    async def test_a_transaction_saving_many_rows_is_one_advance(
        self, bus: FakeNatsClient, source: _CountingSource
    ) -> None:
        members = _Pod(_SwitchedOnMembers, bus, {}, source=source).collection
        conn = _Conn()
        async with CallerTransaction(conn):
            for index in range(5):
                entity = members.create({"id": f"m{index}", "member_type": "user", "member_id": f"person-{index}"})
                await members.save_entity(entity, conn=conn)
            assert source.counts == {}, "nothing advances before the transaction ends"
        assert source.counts == {_TABLE: 1}
        published = _messages(bus)
        assert len(published) == 5
        assert {(message.generation, message.bump_rows) for message in published} == {("inc:1", 5)}
        assert {message.ids[0]: message.columns for message in published} == {
            f"m{index}": {"member_type": "user", "member_id": f"person-{index}"} for index in range(5)
        }

    async def test_a_cache_bypassing_write_is_one_advance_when_its_body_ends(
        self, bus: FakeNatsClient, source: _CountingSource
    ) -> None:
        members = _Pod(_SwitchedOnMembers, bus, {}, source=source).collection
        async with members.bypassing_write("m1", "m2") as write:
            write.touches("m3")
            assert source.counts == {}
        assert source.counts == {_TABLE: 1}
        assert {(message.generation, message.bump_rows) for message in _messages(bus)} == {("inc:1", 3)}

    async def test_a_cache_bypassing_write_that_changed_nothing_advances_nothing(
        self, bus: FakeNatsClient, source: _CountingSource
    ) -> None:
        members = _Pod(_SwitchedOnMembers, bus, {}, source=source).collection
        async with members.bypassing_write("m1") as write:
            write.unchanged()
        assert source.counts == {}

    async def test_a_flush_advances_once_per_table_and_not_at_save_time(
        self, bus: FakeNatsClient, source: _CountingSource
    ) -> None:
        rows: dict[str, dict[str, Any]] = {}
        buffer = WriteBuffer()
        pod = _Pod(_WriteBehindMembers, bus, rows, source=source, write_buffer=buffer)
        for index in range(4):
            await _save(pod.collection, f"m{index}", member_id=f"person-{index}")
        assert source.counts == {}, "a write-behind save is not in L3 yet, so it advances nothing"
        assert all(message.generation is None for message in _messages(bus))
        bus.published.clear()

        assert await flush_pending(buffer, pod.registry) == 4

        assert sorted(rows) == ["m0", "m1", "m2", "m3"]
        assert source.counts == {_TABLE: 1}
        announced = _messages(bus)
        assert sorted(message.ids[0] for message in announced) == ["m0", "m1", "m2", "m3"]
        assert {(message.generation, message.bump_rows) for message in announced} == {("inc:1", 4)}
        # the writer's own L2 entry took the row at save time and is still current
        assert {message.l2_current_scope for message in announced} == {_SCOPE}

    async def test_a_flush_with_nothing_pending_advances_nothing(
        self, bus: FakeNatsClient, source: _CountingSource
    ) -> None:
        buffer = WriteBuffer()
        pod = _Pod(_WriteBehindMembers, bus, {}, source=source, write_buffer=buffer)
        assert await flush_pending(buffer, pod.registry) == 0
        assert source.counts == {}

    async def test_a_collection_with_no_bus_of_its_own_still_advances(
        self, bus: FakeNatsClient, source: _CountingSource
    ) -> None:
        # an explicit ``None`` is not NO_L2: the caches to invalidate are other pods'
        registry = CollectionRegistry()
        registry.configure(l3_pool=object())  # type: ignore[arg-type]
        registry.set_generation_source(source)
        members = _SwitchedOnMembers(registry, {}, nats_client=None)
        await _save(members, "m1")
        await members.invalidate_cache_many(["m1", "m2"])
        assert source.counts == {_TABLE: 2}

    async def test_a_collection_with_no_bus_still_tells_the_caches_derived_from_it(
        self, source: _CountingSource
    ) -> None:
        registry = CollectionRegistry()
        registry.configure(l3_pool=object())  # type: ignore[arg-type]
        registry.set_generation_source(source)
        members = _SwitchedOnMembers(registry, {}, nats_client=None)
        told: list[tuple[str, str | None, int | None]] = []
        registry.register_derived_cache(
            _TABLE,
            on_row=lambda message: told.append((message.ids[0], message.generation, message.bump_rows)),
            on_table_dropped=lambda: told.append(("ALL", None, None)),
        )
        await _save(members, "m1")
        await members.invalidate_cache_many(["m2", "m3"])
        assert told == [("m1", "inc:1", 1), ("m2", "inc:2", 2), ("m3", "inc:2", 2)]

    async def test_without_a_generation_source_nothing_can_advance_and_the_write_stands(
        self, bus: FakeNatsClient
    ) -> None:
        members = _Pod(_SwitchedOnMembers, bus, {}, source=None).collection
        await _save(members, "m1")
        (message,) = _messages(bus)
        assert (message.generation, message.bump_rows) == (None, None)
        # the declared columns ride on the message whether or not a generation does
        assert message.columns == {"member_type": "user", "member_id": "person-1"}


class TestWhatAdvancesNothing:
    async def test_an_opted_out_collection(self, bus: FakeNatsClient, source: _CountingSource) -> None:
        members = _Pod(_OptedOutMembers, bus, {}, source=source).collection
        await _save(members, "m1")
        await members.invalidate_cache_many(["m1", "m2"])
        await members.delete("m1")
        assert source.counts == {}
        assert all(message.generation is None for message in _messages(bus))

    async def test_a_switched_on_collection_built_with_no_l2(
        self, bus: FakeNatsClient, source: _CountingSource
    ) -> None:
        members = _Pod(_SwitchedOnMembers, bus, {}, source=source, nats_client=NO_L2).collection
        await _save(members, "m1")
        await members.invalidate_cache_many(["m1", "m2"])
        await members.delete("m1")
        assert source.counts == {}


class TestAFailedAdvanceAfterACommittedWriteRaises:
    async def test_a_save_raises_after_the_row_is_stored_and_broadcast(
        self, bus: FakeNatsClient, source: _CountingSource
    ) -> None:
        rows: dict[str, dict[str, Any]] = {}
        members = _Pod(_SwitchedOnMembers, bus, rows, source=source).collection
        source.failing = True
        with pytest.raises(GenerationUnavailableError):
            await _save(members, "m1")
        assert "m1" in rows
        (message,) = _messages(bus)
        assert message.ids == ["m1"]
        assert message.generation is None

    async def test_a_delete_raises_after_the_row_is_gone(self, bus: FakeNatsClient, source: _CountingSource) -> None:
        rows: dict[str, dict[str, Any]] = {}
        members = _Pod(_SwitchedOnMembers, bus, rows, source=source).collection
        await _save(members, "m1")
        source.failing = True
        with pytest.raises(GenerationUnavailableError):
            await members.delete("m1")
        assert rows == {}

    async def test_a_settled_transaction_raises_after_every_row_is_evicted(
        self, bus: FakeNatsClient, source: _CountingSource
    ) -> None:
        members = _Pod(_SwitchedOnMembers, bus, {}, source=source).collection
        source.failing = True
        with pytest.raises(GenerationUnavailableError):
            async with CallerTransaction(_Conn()) as transaction:
                transaction.enroll(members, "m1")
                transaction.enroll(members, "m2")
        assert sorted(message.ids[0] for message in _messages(bus)) == ["m1", "m2"]

    async def test_a_transaction_whose_body_raised_keeps_its_own_error(
        self, bus: FakeNatsClient, source: _CountingSource
    ) -> None:
        members = _Pod(_SwitchedOnMembers, bus, {}, source=source).collection
        source.failing = True
        with pytest.raises(RuntimeError, match="the body"):
            async with CallerTransaction(_Conn()) as transaction:
                transaction.enroll(members, "m1")
                raise RuntimeError("the body")
        assert [message.ids[0] for message in _messages(bus)] == ["m1"]

    async def test_a_flush_raises_after_every_row_landed_and_was_announced(
        self, bus: FakeNatsClient, source: _CountingSource
    ) -> None:
        rows: dict[str, dict[str, Any]] = {}
        buffer = WriteBuffer()
        pod = _Pod(_WriteBehindMembers, bus, rows, source=source, write_buffer=buffer)
        await _save(pod.collection, "m1")
        bus.published.clear()
        source.failing = True
        with pytest.raises(GenerationUnavailableError):
            await flush_pending(buffer, pod.registry)
        assert "m1" in rows
        assert buffer.pending_count() == 0, "a row L3 took is not replayed for an advance that failed"
        assert [message.ids[0] for message in _messages(bus)] == ["m1"]

    async def test_a_subscript_write_has_nobody_to_raise_to_and_still_lands(
        self, bus: FakeNatsClient, source: _CountingSource
    ) -> None:
        rows: dict[str, dict[str, Any]] = {}
        members = _Pod(_SwitchedOnMembers, bus, rows, source=source).collection
        await _save(members, "m1")
        source.failing = True
        members["m1", "member_id"] = "person-2"
        await _background_writes()
        assert rows["m1"]["member_id"] == "person-2"


class TestADeclarationThatCannotWorkIsRefused:
    def test_absence_caching_with_an_opt_out_is_refused_at_class_definition(self) -> None:
        with pytest.raises(TypeError, match="caches absences"):

            class _Contradiction(_Members):
                negative_cache_max_age: ClassVar[timedelta | None] = timedelta(seconds=30)
                write_generation = NoWriteGeneration(reason="hot")

    def test_a_declaration_of_another_type_is_refused(self) -> None:
        with pytest.raises(TypeError, match="write_generation"):

            class _Flag(_Members):
                write_generation = False  # type: ignore[assignment]

    def test_invalidation_columns_must_be_a_tuple_of_names(self) -> None:
        with pytest.raises(TypeError, match="invalidation_columns"):

            class _Listed(_Members):
                invalidation_columns = ["member_id"]  # type: ignore[assignment]

    def test_a_registry_refuses_two_collections_for_one_table_that_disagree(
        self, bus: FakeNatsClient, source: _CountingSource
    ) -> None:
        pod = _Pod(_SwitchedOnMembers, bus, {}, source=source)
        with pytest.raises(ValueError, match="disagree on write_generation"):
            _OptedOutMembers(pod.registry, {})
        with pytest.raises(ValueError, match="disagree on write_generation"):
            _UndeclaredMembers(pod.registry, {})

    def test_a_registry_takes_a_second_collection_for_the_table_that_agrees(
        self, bus: FakeNatsClient, source: _CountingSource
    ) -> None:
        pod = _Pod(_SwitchedOnMembers, bus, {}, source=source)

        class _HubMembers(_SwitchedOnMembers):
            """a subclass inherits the declaration, as an admin-only subclass of a framework class does."""

        replacement = _HubMembers(pod.registry, {})
        assert pod.registry.get_collection(_TABLE) is replacement

    def test_a_registry_takes_two_opt_outs_whatever_their_reasons(
        self, bus: FakeNatsClient, source: _CountingSource
    ) -> None:
        pod = _Pod(_OptedOutMembers, bus, {}, source=source)

        class _AlsoOptedOut(_Members):
            write_generation = NoWriteGeneration(reason="a different sentence saying the same thing")

        _AlsoOptedOut(pod.registry, {})


class TestConcurrentCommitsEachAdvance:
    async def test_two_saves_at_once_are_two_advances_each_named_on_its_own_row(
        self, bus: FakeNatsClient, source: _CountingSource
    ) -> None:
        members = _Pod(_SwitchedOnMembers, bus, {}, source=source).collection
        await asyncio.gather(_save(members, "m1"), _save(members, "m2"))
        assert source.counts == {_TABLE: 2}
        assert sorted((message.generation, message.bump_rows) for message in _messages(bus)) == [
            ("inc:1", 1),
            ("inc:2", 1),
        ]
