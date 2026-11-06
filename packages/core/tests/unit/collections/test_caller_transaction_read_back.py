"""A save inside a caller's transaction reads its row back on the caller's connection, and only where it can."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

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

_SCHEMA = TableSchema(
    name="rows",
    primary_key="id",
    columns=[
        Column("id", STRING_TYPE),
        Column("geo", STRING_TYPE, nullable=True),
        Column("date_created", DATETIMETZ_TYPE, immutable=True),
        Column("date_updated", DATETIMETZ_TYPE),
    ],
)


class _Reads:
    """a connection, or a pool, that answers every row read with one row and records it."""

    def __init__(self) -> None:
        self.reads: list[str] = []

    async def fetchrow(self, query: str, *params: Any) -> dict[str, Any]:
        self.reads.append(query)
        return {"id": params[0], "geo": "county", "date_created": None, "date_updated": None}


def _collection(cls: type[Any] | None = None) -> Any:
    registry = CollectionRegistry()
    registry.configure(kv_key_scope="test-principal")
    collection = (cls or collection_for_schema(_SCHEMA))(registry, DefaultCoreConfig(), None)
    collection.l3_pool = _Reads()
    return collection


async def test_the_read_back_runs_on_the_caller_s_connection_not_the_pool() -> None:
    collection = _collection()
    conn = _Reads()
    row = await collection.fetch_from_store_in_transaction("k", conn=conn)
    assert row is not None and row["geo"] == "county"
    assert len(conn.reads) == 1
    assert collection.l3_pool.reads == [], "the read went to the pool, which cannot see the transaction"


async def test_a_subclass_that_reads_its_own_way_reads_nothing_in_a_transaction() -> None:
    class _OwnRead(collection_for_schema(_SCHEMA)):  # type: ignore[misc]
        async def fetch_from_store(self, entity_id: Any) -> dict[str, Any] | None:
            return {"id": entity_id, "geo": "its own"}

    collection = _collection(_OwnRead)
    conn = _Reads()
    assert await collection.fetch_from_store_in_transaction("k", conn=conn) is None
    assert conn.reads == []


class _Conn:
    @asynccontextmanager
    async def _transaction(self) -> AsyncIterator[None]:
        yield

    def transaction(self, **options: Any) -> Any:
        return self._transaction()


class _Announcing(collection_for_schema(_SCHEMA)):  # type: ignore[misc]
    """a collection whose eviction broadcasts carry ``geo``, recording the rows it is settled with."""

    invalidation_columns = ("geo",)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.settled: list[tuple[list[Any], list[dict[str, Any] | None]]] = []

    async def invalidate_cache_many(self, keys: list[Any], *, rows: list[dict[str, Any] | None] | None = None) -> None:
        self.settled.append((list(keys), list(rows or [])))


async def test_settling_broadcasts_the_row_as_restated_for_that_key_only() -> None:
    first, second = _collection(_Announcing), _collection(_Announcing)
    conn = _Conn()
    async with CallerTransaction(conn) as transaction:
        transaction.enroll(first, "k", row={"id": "k"})
        transaction.enroll(first, "other", row={"id": "other"})
        transaction.enroll(second, "k", row={"id": "k"})
        transaction.enroll(first, "k", row={"id": "k", "sent": True})
        transaction.restate(first, "k", row={"id": "k", "geo": "county"})

    assert first.settled == [
        (["k", "other", "k"], [{"id": "k"}, {"id": "other"}, {"id": "k", "geo": "county"}]),
    ]
    assert second.settled == [(["k"], [{"id": "k"}])]
