"""Settling many written keys at once: one call per collection, no bus work without a bus, never one key at a time."""

from __future__ import annotations

import asyncio
import sqlite3
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import pytest
from pydantic import BaseModel
from sqlalchemy import MetaData

from threetears.core.cache.duckdb import DuckDBBackend
from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections.caller_transaction import CallerTransaction
from threetears.core.collections.delta_files import entity_key_batch
from threetears.core.collections.epoch_change import key_text
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.collections.schema_backed import (
    DATETIMETZ_TYPE,
    STRING_TYPE,
    Column,
    TableSchema,
    collection_for_schema,
)
from threetears.core.config import DefaultCoreConfig
from threetears.core.testing.kv import FakeKvBucket, FakeNatsClient
from threetears.core.testing.l1_tables import TEST_L1_TABLE


def _schema(name: str) -> TableSchema:
    return TableSchema(
        name=name,
        primary_key="id",
        columns=[
            Column("id", STRING_TYPE),
            Column("date_created", DATETIMETZ_TYPE, immutable=True),
            Column("date_updated", DATETIMETZ_TYPE),
        ],
        l1_eviction=TEST_L1_TABLE,
    )


def _collection(name: str = "rows", nats_client: Any = None) -> Any:
    registry = CollectionRegistry()
    # every L2 key leads with the registry's principal scope
    registry.configure(kv_key_scope="test-principal")
    return collection_for_schema(_schema(name))(registry, DefaultCoreConfig(), nats_client)


class _Conn:
    @asynccontextmanager
    async def _transaction(self) -> AsyncIterator[None]:
        yield

    def transaction(self, **options: Any) -> Any:
        return self._transaction()


async def test_without_a_bus_settling_drops_the_table_s_scans_once(monkeypatch: pytest.MonkeyPatch) -> None:
    collection = _collection()
    dropped: list[str] = []
    scans = collection.registry.scan_cache
    monkeypatch.setattr(type(scans), "drop_for_table", lambda self, table: dropped.append(table))
    await collection.invalidate_cache_many([f"k{i}" for i in range(100)])
    # this process's cached scans of the table go, once: the write changed which rows match
    assert dropped == ["rows"]


class _CountingNats(FakeNatsClient):
    """a NATS client that counts how many publishes are in flight at once."""

    def __init__(self) -> None:
        super().__init__()
        self.in_flight = 0
        self.most = 0

    async def publish(
        self, *args: Any, subject: Any = None, message: BaseModel | None = None, reply_to: Any = None
    ) -> None:
        self.in_flight += 1
        self.most = max(self.most, self.in_flight)
        await asyncio.sleep(0)
        self.in_flight -= 1
        await super().publish(subject=subject, message=message, reply_to=reply_to)


async def test_with_a_bus_every_key_leaves_l2_side_by_side_and_one_broadcast_names_them_all(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Flips if a write of many keys deletes their L2 copies one after another, or broadcasts them as other than one
    message naming every key (a message a key was a pydantic message a key decoded on every peer's loop and turned
    into Arrow one at a time by a copy: ~80 ms of the loop per 200,000 keys, 2026-11-25)."""
    nats = _CountingNats()
    collection = _collection(nats_client=nats)
    keys = [f"k{i}" for i in range(50)]
    deleting = {"now": 0, "most": 0}
    real_delete = FakeKvBucket.delete

    async def counted(self: FakeKvBucket, **kwargs: Any) -> bool:
        deleting["now"] += 1
        deleting["most"] = max(deleting["most"], deleting["now"])
        await asyncio.sleep(0)
        deleting["now"] -= 1
        return await real_delete(self, **kwargs)

    monkeypatch.setattr(FakeKvBucket, "delete", counted)
    await collection.invalidate_cache_many(keys)
    (message,) = nats.published
    assert sorted(row.ids[0] for row in message.each_row()) == sorted(keys)
    assert deleting["most"] > 1


async def test_a_transaction_settles_each_collection_s_keys_in_one_call() -> None:
    first, second = _collection("a"), _collection("b")
    calls: list[tuple[str, list[Any]]] = []

    def spy(name: str) -> Any:
        async def settle(keys: Any) -> None:
            calls.append((name, list(keys)))

        return settle

    first.invalidate_cache_many = spy("a")
    second.invalidate_cache_many = spy("b")
    conn = _Conn()
    async with CallerTransaction(conn) as transaction:
        for i in range(3):
            transaction.enroll(first, f"a{i}")
            transaction.enroll(second, f"b{i}")
    assert calls == [("a", ["a0", "a1", "a2"]), ("b", ["b0", "b1", "b2"])]


async def test_without_a_bus_settling_drops_scans_through_the_registry_s_local_half(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """one owner of "a local write evicts local scans": the registry, as every single-key write uses."""
    collection = _collection()
    local: list[str] = []
    monkeypatch.setattr(type(collection.registry), "drop_local_scans", lambda self, table: local.append(table))
    await collection.invalidate_cache_many(["a", "b"])
    assert local == ["rows"]


async def test_a_write_on_the_collection_s_own_pool_settles_its_rows_in_one_call() -> None:
    collection = _collection()
    calls: list[list[Any]] = []

    async def many(entity_ids: Any) -> None:
        calls.append(list(entity_ids))

    collection.invalidate_cache_many = many
    async with collection.bypassing_write("a", "b", "c"):
        pass
    assert calls == [["a", "b", "c"]]


def _cached_collection(name: str = "rows") -> tuple[Any, SQLiteBackend]:
    """a collection over a real SQLite L1 holding its table, with no bus."""
    schema = _schema(name)
    metadata = MetaData()
    schema.to_sqlalchemy_table(metadata)
    l1 = SQLiteBackend(db_name=f"invalidate_many_{uuid.uuid4().hex[:8]}")
    l1.initialize(metadata)
    registry = CollectionRegistry()
    registry.configure(l1_backend=l1, kv_key_scope="test-principal")
    return collection_for_schema(schema)(registry, DefaultCoreConfig(), None), l1


def _cache_rows(collection: Any, keys: list[str]) -> None:
    for key in keys:
        assert collection.write_to_cache_sync({"id": key})


class TestEvictionIsOneL1Write:
    """settling N keys evicts them from L1 in one write transaction, not one transaction a key."""

    async def test_n_keys_take_one_transaction_and_every_key_leaves(self, l1_statements: Any) -> None:
        collection, l1 = _cached_collection()
        keys = [f"k{i}" for i in range(60)]
        _cache_rows(collection, keys)
        statements = l1_statements(l1)

        await collection.invalidate_cache_many(keys)

        assert statements.count("BEGIN IMMEDIATE") == 1
        assert statements.count("COMMIT") == 1
        assert [collection.exists_in_cache_sync(key) for key in keys] == [False] * len(keys)

    async def test_every_l1_change_listener_hears_each_key(self) -> None:
        collection, _ = _cached_collection()
        keys = [f"k{i}" for i in range(5)]
        _cache_rows(collection, keys)
        heard: list[Any] = []
        collection.add_l1_change_listener(heard.append)

        await collection.invalidate_cache_many(keys)

        assert heard == keys

    async def test_a_failed_eviction_raises_unchanged_and_still_fences_reads_in_flight(self) -> None:
        collection, l1 = _cached_collection()
        keys = [f"k{i}" for i in range(5)]
        _cache_rows(collection, keys)
        l1.get_connection().execute(
            "CREATE TRIGGER refuse BEFORE DELETE ON rows WHEN OLD.id = 'k2' BEGIN SELECT RAISE(ABORT, 'refused'); END"
        )
        ticket = collection.scan_ticket()

        with pytest.raises(sqlite3.IntegrityError, match="refused"):
            await collection.invalidate_cache_many(keys)

        # a read that began before the eviction may not cache what it read, though the delete failed
        assert collection.write_to_cache_sync({"id": "k9"}, read_since=ticket) is False
        assert not l1.get_connection().in_transaction

    def test_dropping_the_cached_table_is_one_transaction(self, l1_statements: Any) -> None:
        collection, l1 = _cached_collection()
        keys = [f"k{i}" for i in range(30)]
        _cache_rows(collection, keys)
        statements = l1_statements(l1)

        assert collection.drop_cached_table() == len(keys)

        assert statements.count("BEGIN IMMEDIATE") == 1
        assert [collection.exists_in_cache_sync(key) for key in keys] == [False] * len(keys)


class _RefusingDuckDBBackend(DuckDBBackend):
    """a DuckDB L1 whose delete of one key fails, after the keys before it have gone."""

    def delete_by_id(self, table: str, entity_id: Any, primary_key: str | tuple[str, ...] = "id") -> None:
        if entity_id in ("k2", ("k2",)):
            raise RuntimeError("refused")
        super().delete_by_id(table, entity_id, primary_key)


class TestAFailedEvictionStillTellsListeners:
    """a batch eviction that fails partway tells every listener of every key that may have left L1."""

    async def test_a_duckdb_l1_failing_partway_tells_listeners_of_the_keys_already_gone(self) -> None:
        schema = _schema("rows")
        metadata = MetaData()
        schema.to_sqlalchemy_table(metadata)
        l1 = _RefusingDuckDBBackend()
        l1.initialize(metadata)
        try:
            registry = CollectionRegistry()
            registry.configure(l1_backend=l1, kv_key_scope="test-principal")
            collection = collection_for_schema(schema)(registry, DefaultCoreConfig(), None)
            keys = [f"k{i}" for i in range(5)]
            _cache_rows(collection, keys)
            heard: list[Any] = []
            collection.add_l1_change_listener(heard.append)

            with pytest.raises(RuntimeError, match="refused"):
                await collection.invalidate_cache_many(keys)

            # k0 and k1 left before the failure: a complete copy over this L1 must hear them go
            assert not collection.exists_in_cache_sync("k0")
            assert not collection.exists_in_cache_sync("k1")
            assert set(heard) == set(keys)
        finally:
            l1.reset()


class _KeyTaker:
    """a cache derived from a table that takes key batches (as a copy follower does), and one that takes row messages."""

    def __init__(self, registry: CollectionRegistry, table: str) -> None:
        self.batches: list[tuple[Any, str | None]] = []
        self.rows_to_taker: list[Any] = []
        self.rows_to_other: list[Any] = []
        registry.register_derived_cache(
            table,
            on_row=self.rows_to_taker.append,
            on_table_dropped=lambda: None,
            on_partition=lambda partition: None,
            on_keys=lambda keys, epoch: self.batches.append((keys, epoch)),
        )
        registry.register_derived_cache(
            table, on_row=self.rows_to_other.append, on_table_dropped=lambda: None, on_partition=lambda partition: None
        )


@pytest.mark.parametrize("bus", [False, True])
async def test_a_write_s_keys_reach_a_cache_taking_key_batches_once_as_arrow_never_a_row_message_a_key(
    bus: bool,
) -> None:
    """Flips if a local write of many keys hands a copy (a cache that takes key batches) a row message a key, which
    the copy turned into Python lists a key on the event loop (~10 s of loop stalls over a 40-minute local burst
    replay, 2026-11-24): the copy is told the write's keys once, as one Arrow batch spelled as each row message spells
    its key; a cache that takes no key batches still hears every row, and with a bus every row is still broadcast."""
    pa = pytest.importorskip("pyarrow")
    nats = _CountingNats() if bus else None
    collection = _collection(nats_client=nats)
    caches = _KeyTaker(collection.registry, "rows")
    keys = [f"k{i}" for i in range(300)]
    await collection.invalidate_cache_many(keys)
    assert caches.rows_to_taker == [], "the cache taking key batches was handed a row message a key"
    assert len(caches.batches) == 1
    batch, _epoch = caches.batches[0]
    assert batch.equals(pa.table({"k0": pa.array(keys, pa.string())}))
    assert [message.ids for message in caches.rows_to_other] == [[key] for key in keys]
    if nats is not None:
        (message,) = nats.published
        assert [row.ids for row in message.each_row()] == [[key] for key in keys], "every row is broadcast, once"


def test_a_key_batch_spells_each_value_as_its_row_message_does() -> None:
    """Flips if the Arrow batch a copy is told a write's keys in spells a value otherwise than the row message would
    (``key_text``): text and integers by Arrow's cast, every other type (a time, a decimal's scale, mixed types) by
    ``key_text`` itself."""
    pa = pytest.importorskip("pyarrow")
    keys: list[tuple[Any, ...]] = [
        ("VA", 12, datetime(2026, 11, 3, 21, 45, tzinfo=UTC), Decimal("1.5"), True),
        ("IL", -3, datetime(2026, 11, 4, 1, 0, 0, 5, tzinfo=UTC), Decimal("1.50"), False),
    ]
    batch = entity_key_batch(keys)
    assert batch.column_names == ["k0", "k1", "k2", "k3", "k4"]
    for at in range(5):
        assert batch.column(at).to_pylist() == [key_text(key[at]) for key in keys]
    mixed = entity_key_batch([1, "a"])
    assert mixed.column(0).to_pylist() == ["1", "a"]
    assert batch.column(0).type == pa.string()
