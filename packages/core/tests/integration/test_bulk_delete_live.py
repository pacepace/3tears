"""Integration: rows deleted, and read, by many keys in key-led statements.

What the unit tests' recording connections cannot prove: that Postgres accepts the typed arrays
(``= ANY($1::timestamptz[])`` beside the rest of the key), deletes exactly the keys named and keeps
them when the transaction rolls back, and that a key-led read cut at a row cap, read again in halves
and paged by the rest of the key, answers every row the values hold and no other.

Uses the session-scoped ``db_container`` fixture; a checkout without docker skips cleanly.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from typing import Any

import asyncpg
import pytest

from threetears.core.backends.sql import SqlL3Backend
from threetears.core.collections.asyncpg_init import init_connection
from threetears.core.collections.caller_transaction import CallerTransaction
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.collections.schema_backed import (
    BIGINT_TYPE,
    DATETIMETZ_TYPE,
    JSONB_TYPE,
    STRING_TYPE,
    Column,
    TableSchema,
    collection_for_schema,
)
from threetears.core.config import DefaultCoreConfig
from threetears.core.entities.base import BaseEntity

pytestmark = pytest.mark.integration

_START = datetime(2026, 11, 4, 1, 0, tzinfo=UTC)

_SCHEMA = TableSchema(
    name="snapshots",
    primary_key=("race", "reported_at"),
    columns=[
        Column("race", STRING_TYPE),
        Column("reported_at", DATETIMETZ_TYPE),
        Column("votes", BIGINT_TYPE, nullable=True),
    ],
    on_conflict="update",
)


class _Snapshot(BaseEntity):
    primary_key_field = "reported_at"


@pytest.fixture
async def pool(db_container: str) -> AsyncIterator[asyncpg.Pool]:
    schema = f"bulkdel_{uuid.uuid4().hex[:8]}"
    admin = await asyncpg.connect(db_container)
    try:
        await admin.execute(f"CREATE SCHEMA {schema}")
    finally:
        await admin.close()
    held = await asyncpg.create_pool(db_container, min_size=1, max_size=2, server_settings={"search_path": schema})
    assert held is not None
    try:
        await held.execute(
            "CREATE TABLE snapshots (race TEXT, reported_at TIMESTAMPTZ, votes BIGINT, PRIMARY KEY (race, reported_at))"
        )
        for minute in range(6):
            await held.execute(
                "INSERT INTO snapshots VALUES ($1, $2, $3)",
                f"r{minute % 2}",
                _START + timedelta(minutes=minute),
                minute,
            )
        yield held
    finally:
        await held.close()


def _collection(pool: asyncpg.Pool) -> Any:
    registry = CollectionRegistry()
    registry.configure(l3_pool=SqlL3Backend(pool))
    return collection_for_schema(_SCHEMA, entity_class=_Snapshot)(registry, DefaultCoreConfig(), None)


async def _held(pool: asyncpg.Pool) -> list[int]:
    return [r["votes"] for r in await pool.fetch("SELECT votes FROM snapshots ORDER BY votes")]


async def test_exactly_the_keys_named_are_deleted_across_statements(pool: asyncpg.Pool) -> None:
    collection = _collection(pool)
    keys = [
        ("r1", _START + timedelta(minutes=1)),
        ("r0", _START + timedelta(minutes=4)),
        ("r1", _START + timedelta(minutes=5)),
        ("nope", _START),
    ]

    async with pool.acquire() as conn, CallerTransaction(conn):
        assert await collection.delete_rows(keys, conn=conn, max_rows=2) == 4

    assert await _held(pool) == [0, 2, 3]


async def test_a_rolled_back_delete_keeps_every_row(pool: asyncpg.Pool) -> None:
    collection = _collection(pool)

    with pytest.raises(ConnectionError):
        async with pool.acquire() as conn, CallerTransaction(conn):
            await collection.delete_rows([("r0", _START)], conn=conn)
            raise ConnectionError("the write after it failed")

    assert await _held(pool) == [0, 1, 2, 3, 4, 5]


def _capped(pool: asyncpg.Pool, cap: int) -> Any:
    """the collection, reading as if the rail answered at most ``cap`` rows a statement."""
    registry = CollectionRegistry()
    registry.configure(l3_pool=SqlL3Backend(_CuttingPool(pool, cap)))
    return collection_for_schema(_SCHEMA, entity_class=_Snapshot)(registry, DefaultCoreConfig(), None)


class _CuttingPool:
    """a pool that answers at most ``cap`` rows a statement and says so, as the L3 rail's client does."""

    def __init__(self, pool: asyncpg.Pool, cap: int) -> None:
        self._pool = pool
        self._cap = cap
        self.rows_per_statement = cap
        self.statements: list[str] = []

    async def fetch(self, query: str, *params: Any) -> list[Any]:
        self.statements.append(query)
        return list(await self._pool.fetch(query, *params))[: self._cap]


async def test_a_key_led_read_answers_every_row_the_values_hold(pool: asyncpg.Pool) -> None:
    for minute in range(6, 30):
        await pool.execute(
            "INSERT INTO snapshots VALUES ($1, $2, $3)", "r2", _START + timedelta(minutes=minute), minute
        )

    # r0 and r1 hold three rows each, r2 twenty-four: the batch is cut, then r2 alone is
    held = await _capped(pool, 4).read_rows_led_by(["r0", "r2", "r1", "nope"], columns=["votes"])

    assert sorted(row["votes"] for row in held) == list(range(30))
    assert all(isinstance(row["reported_at"], datetime) for row in held)


async def test_a_key_led_read_takes_one_statement_when_nothing_is_cut(pool: asyncpg.Pool) -> None:
    collection = _capped(pool, 1000)

    held = await collection.read_rows_led_by(["r1"])

    assert sorted(row["reported_at"] for row in held) == [_START + timedelta(minutes=m) for m in (1, 3, 5)]
    assert set(held[0]) == {"race", "reported_at"}


async def test_keys_sharing_no_value_are_deleted_whole_a_batch_a_statement(pool: asyncpg.Pool) -> None:
    collection = _collection(pool)
    keys = [("r0", _START), ("r1", _START + timedelta(minutes=3)), ("r0", _START + timedelta(minutes=4))]

    async with pool.acquire() as conn, CallerTransaction(conn):
        assert await collection.delete_rows(keys, conn=conn) == 3

    assert await _held(pool) == [1, 2, 5]


_DOCS = TableSchema(
    name="docs",
    primary_key=("doc_id", "shape"),
    columns=[Column("doc_id", STRING_TYPE), Column("shape", JSONB_TYPE)],
    on_conflict="update",
)


async def test_a_jsonb_key_column_held_fixed_deletes_the_keys_named(pool: asyncpg.Pool, db_container: str) -> None:
    await pool.execute("CREATE TABLE docs (doc_id TEXT, shape JSONB, PRIMARY KEY (doc_id, shape))")
    for doc, kind in (("d1", 1), ("d2", 1), ("d3", 1), ("d1", 2)):
        await pool.execute("INSERT INTO docs VALUES ($1, $2::jsonb)", doc, f'{{"k": {kind}}}')
    # a 3tears pool registers the jsonb codec every jsonb value is bound through
    search_path = await pool.fetchval("SHOW search_path")
    coded = await asyncpg.create_pool(
        db_container, min_size=1, max_size=1, init=init_connection, server_settings={"search_path": search_path}
    )
    assert coded is not None
    try:
        registry = CollectionRegistry()
        registry.configure(l3_pool=SqlL3Backend(coded))
        collection = collection_for_schema(_DOCS, entity_class=_Snapshot)(registry, DefaultCoreConfig(), None)
        async with coded.acquire() as conn, CallerTransaction(conn):
            assert await collection.delete_rows([("d1", {"k": 1}), ("d2", {"k": 1})], conn=conn) == 2
    finally:
        await coded.close()

    held = await pool.fetch("SELECT doc_id, shape::text AS shape FROM docs ORDER BY doc_id, shape")
    assert [(r["doc_id"], r["shape"]) for r in held] == [("d1", '{"k": 2}'), ("d3", '{"k": 1}')]
