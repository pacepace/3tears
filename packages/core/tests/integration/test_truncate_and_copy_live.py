"""Integration: a table emptied with TRUNCATE and loaded again with COPY, on a real Postgres.

What the unit tests' recording connections cannot prove: that ``TRUNCATE`` runs under the DDL
lock's session lock and empties the table; that asyncpg's ``copy_records_to_table``, on a pool
initialised as every 3tears pool is, accepts the values an insert binds -- a timestamptz, a uuid,
bytes, a numeric, a bool -- and writes what an upsert would; and that a copy naming a key already
held fails and rolls the whole transaction back.

Uses the session-scoped ``db_container`` fixture; a checkout without docker skips cleanly.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from decimal import Decimal
from typing import Any

import asyncpg
import pytest

from threetears.core.backends.sql import SqlL3Backend
from threetears.core.collections.asyncpg_init import init_connection
from threetears.core.collections.caller_transaction import CallerTransaction
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.collections.schema_backed import (
    BYTES_TYPE,
    DATETIMETZ_TYPE,
    BOOL_TYPE,
    INT_TYPE,
    NUMERIC_TYPE,
    STRING_TYPE,
    UUID_TYPE,
    Column,
    TableSchema,
    collection_for_schema,
)
from threetears.core.config import DefaultCoreConfig
from threetears.core.entities.base import BaseEntity

pytestmark = pytest.mark.integration

_SCHEMA = TableSchema(
    name="lsf_rows",
    primary_key=("race", "geo"),
    columns=[
        Column("race", STRING_TYPE),
        Column("geo", STRING_TYPE),
        Column("votes", INT_TYPE, nullable=True),
        Column("source_id", UUID_TYPE, nullable=True),
        Column("shape", BYTES_TYPE, nullable=True),
        Column("share", NUMERIC_TYPE, nullable=True, precision=38, scale=12),
        Column("final", BOOL_TYPE, nullable=True),
        Column("date_created", DATETIMETZ_TYPE, immutable=True),
        Column("date_updated", DATETIMETZ_TYPE),
    ],
)


class _Row(BaseEntity):
    primary_key_field = "geo"


@pytest.fixture
async def pool(db_container: str) -> AsyncIterator[asyncpg.Pool]:
    schema = f"truncopy_{uuid.uuid4().hex[:8]}"
    admin = await asyncpg.connect(db_container)
    try:
        await admin.execute(f"CREATE SCHEMA {schema}")
    finally:
        await admin.close()
    held = await asyncpg.create_pool(
        db_container, min_size=1, max_size=2, init=init_connection, server_settings={"search_path": schema}
    )
    assert held is not None
    try:
        await held.execute(
            "CREATE TABLE lsf_rows (race TEXT, geo TEXT, votes INT, source_id UUID, shape BYTEA, share NUMERIC(38, 12), "
            "final BOOLEAN, "
            "date_created TIMESTAMPTZ NOT NULL, date_updated TIMESTAMPTZ NOT NULL, PRIMARY KEY (race, geo))"
        )
        yield held
    finally:
        await held.close()


def _collection(pool: asyncpg.Pool) -> Any:
    registry = CollectionRegistry()
    registry.configure(l3_pool=SqlL3Backend(pool))
    return collection_for_schema(_SCHEMA, entity_class=_Row)(registry, DefaultCoreConfig(), None)


def _rows(race: str, count: int) -> list[dict[str, Any]]:
    return [
        {
            "race": race,
            "geo": f"{i:05d}",
            "votes": i,
            "source_id": uuid.UUID(int=i),
            "shape": bytes([i % 256]) * 3,
            "share": Decimal(i) / 8,
            "final": i % 2 == 0,
        }
        for i in range(count)
    ]


async def test_a_table_is_emptied_and_loaded_again_with_copy(pool: asyncpg.Pool) -> None:
    collection = _collection(pool)
    async with pool.acquire() as conn, CallerTransaction(conn):
        await collection.save_rows(_rows("old", 50), conn=conn)

    await collection.truncate()
    assert await pool.fetchval("SELECT count(*) FROM lsf_rows") == 0
    # the DDL lock was released: no session still holds an advisory lock
    assert await pool.fetchval("SELECT count(*) FROM pg_locks WHERE locktype = 'advisory'") == 0

    async with pool.acquire() as conn, CallerTransaction(conn):
        assert await collection.save_rows(list(reversed(_rows("new", 250))), conn=conn, copy=True, max_rows=100) == 250

    assert await pool.fetchval("SELECT count(*) FROM lsf_rows") == 250
    row = await pool.fetchrow("SELECT * FROM lsf_rows WHERE race = 'new' AND geo = '00007'")
    assert row is not None
    assert row["votes"] == 7
    assert row["source_id"] == uuid.UUID(int=7)
    assert row["shape"] == b"\x07\x07\x07"
    assert row["share"] == Decimal("0.875")
    assert row["final"] is False
    assert row["date_created"] == row["date_updated"]


async def test_a_copy_and_an_upsert_write_the_same_row(pool: asyncpg.Pool) -> None:
    collection = _collection(pool)
    async with pool.acquire() as conn, CallerTransaction(conn):
        await collection.save_rows(_rows("upserted", 3), conn=conn)
        await collection.save_rows(_rows("copied", 3), conn=conn, copy=True)
    columns = "geo, votes, source_id, shape, share, final"
    upserted = await pool.fetch(f"SELECT {columns} FROM lsf_rows WHERE race = 'upserted' ORDER BY geo")
    copied = await pool.fetch(f"SELECT {columns} FROM lsf_rows WHERE race = 'copied' ORDER BY geo")
    assert [dict(r) for r in upserted] == [dict(r) for r in copied]


async def test_a_copy_naming_a_key_already_held_rolls_the_whole_transaction_back(pool: asyncpg.Pool) -> None:
    collection = _collection(pool)
    async with pool.acquire() as conn, CallerTransaction(conn):
        await collection.save_rows(_rows("r", 1), conn=conn)

    with pytest.raises(asyncpg.UniqueViolationError):
        async with pool.acquire() as conn, CallerTransaction(conn):
            await collection.save_rows(_rows("other", 2), conn=conn, copy=True)
            await collection.save_rows(_rows("r", 2), conn=conn, copy=True)

    assert await pool.fetchval("SELECT count(*) FROM lsf_rows") == 1


async def test_truncating_an_empty_table_twice_is_safe(pool: asyncpg.Pool) -> None:
    collection = _collection(pool)
    await collection.truncate()
    await collection.truncate()
    assert await pool.fetchval("SELECT count(*) FROM lsf_rows") == 0
