"""Settling many written keys at once: one call per collection, no bus work without a bus, never one key at a time."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import pytest

from threetears.core.collections.caller_transaction import CallerTransaction
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.collections.schema_backed import (
    DATETIMETZ_TYPE,
    STRING_TYPE,
    Column,
    TableSchema,
    collection_for_schema,
)
from threetears.core.config import DefaultCoreConfig
from threetears.core.testing.kv import FakeNatsClient


def _schema(name: str) -> TableSchema:
    return TableSchema(
        name=name,
        primary_key="id",
        columns=[
            Column("id", STRING_TYPE),
            Column("date_created", DATETIMETZ_TYPE, immutable=True),
            Column("date_updated", DATETIMETZ_TYPE),
        ],
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

    async def publish(self, *, subject: Any, message: Any, reply_to: Any = None) -> None:
        self.in_flight += 1
        self.most = max(self.most, self.in_flight)
        await asyncio.sleep(0)
        self.in_flight -= 1
        await super().publish(subject=subject, message=message, reply_to=reply_to)


async def test_with_a_bus_every_key_is_broadcast_side_by_side_not_one_after_another() -> None:
    nats = _CountingNats()
    collection = _collection(nats_client=nats)
    keys = [f"k{i}" for i in range(50)]
    await collection.invalidate_cache_many(keys)
    assert sorted(m.ids[0] for m in nats.published) == sorted(keys)
    assert nats.most > 1


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
