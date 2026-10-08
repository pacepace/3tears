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
from threetears.core.collections.caller_transaction import CallerTransaction
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.collections.schema_backed import (
    BIGINT_TYPE,
    DATETIMETZ_TYPE,
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
    cls = collection_for_schema(_SCHEMA, entity_class=_Snapshot)
    cls.L3_ROW_CAP = cap
    registry = CollectionRegistry()
    registry.configure(l3_pool=SqlL3Backend(_CuttingPool(pool, cap)))
    return cls(registry, DefaultCoreConfig(), None)


class _CuttingPool:
    """a pool that answers at most ``cap`` rows a statement, as the L3 rail does, without saying so."""

    def __init__(self, pool: asyncpg.Pool, cap: int) -> None:
        self._pool = pool
        self._cap = cap
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
