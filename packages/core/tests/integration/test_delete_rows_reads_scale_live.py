"""Integration: a bulk delete reads the rows it removes a batch of keys a statement, not a key a statement.

A collection whose row messages carry invalidation columns reads each row before deleting it, so the
message can say whose it was. That read was one ``fetch_one`` a key: a round trip each, on the L3 rail a
NATS request each. It is now one statement a batch of keys (``fetch_many``), and every key's message
carries exactly the row a ``fetch_one`` would have read. On YugabyteDB the whole delete keeps within twice
the database's own keyed delete of as many rows.

Against the session ``db_container``; set ``THREETEARS_TEST_POSTGRES_URL`` to a YugabyteDB URL to run it there.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator
from typing import Any, ClassVar

import asyncpg
import pytest

from threetears.core.backends.sql import SqlL3Backend
from threetears.core.collections.caller_transaction import CallerTransaction
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.collections.schema_backed import (
    BIGINT_TYPE,
    STRING_TYPE,
    Column,
    TableSchema,
    collection_for_schema,
)
from threetears.core.config import DefaultCoreConfig
from threetears.core.entities.base import BaseEntity
from threetears.core.testing.calibrate import budget_problem, plain_keyed_delete_seconds
from threetears.core.testing.plans import on_yugabyte

pytestmark = pytest.mark.integration

_SCHEMA = TableSchema(
    name="results",
    primary_key=("race", "geo"),
    columns=[
        Column("race", STRING_TYPE),
        Column("geo", STRING_TYPE),
        Column("state", STRING_TYPE),
        Column("votes", BIGINT_TYPE, nullable=True),
    ],
    on_conflict="update",
)

#: a county reload's rows gone from the source
_ROWS = 5_000


class _Result(BaseEntity):
    primary_key_field = "geo"


class _Counting:
    """a connection that counts the reads made on it; everything else is the connection's."""

    def __init__(self, conn: Any) -> None:
        self._conn = conn
        self.reads = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._conn, name)

    async def fetch(self, query: str, *args: Any) -> Any:
        self.reads += 1
        return await self._conn.fetch(query, *args)

    async def fetchrow(self, query: str, *args: Any) -> Any:
        self.reads += 1
        return await self._conn.fetchrow(query, *args)


def _collection(backend: Any) -> Any:
    base = collection_for_schema(_SCHEMA, entity_class=_Result)

    class _Noting(base):  # type: ignore[valid-type,misc]
        invalidation_columns: ClassVar[tuple[str, ...]] = ("state",)

    registry = CollectionRegistry()
    registry.configure(l3_pool=backend)
    return _Noting(registry, DefaultCoreConfig(), None)


@pytest.fixture
async def pool(db_container: str) -> AsyncIterator[asyncpg.Pool]:
    schema = f"delrows_{uuid.uuid4().hex[:8]}"
    held = await asyncpg.create_pool(db_container, min_size=1, max_size=2, server_settings={"search_path": schema})
    assert held is not None
    try:
        await held.execute(f"CREATE SCHEMA {schema}")
        await held.execute(
            "CREATE TABLE results (race TEXT, geo TEXT, state TEXT NOT NULL, votes BIGINT, PRIMARY KEY (race, geo))"
        )
        yield held
    finally:
        await held.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        await held.close()


async def _fill(pool: asyncpg.Pool) -> list[tuple[str, str]]:
    await pool.execute("TRUNCATE results")
    await pool.execute(
        "INSERT INTO results SELECT 'r' || (g % 50), 'g' || g, 'S' || (g % 7), g FROM generate_series(1, $1::int) g",
        _ROWS,
    )
    return [(f"r{g % 50}", f"g{g}") for g in range(1, _ROWS + 1)] + [("nope", "nowhere")]


async def test_every_message_carries_the_row_a_key_read_would_and_the_reads_are_batched(
    pool: asyncpg.Pool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Flips if a bulk delete reads a key a statement, or a key's message carries other than its own row.

    A batch's one statement reads and deletes (``DELETE ... RETURNING``), so the statements counted are the deletes.
    """
    keys = await _fill(pool)
    backend = SqlL3Backend(pool)
    expected = {key: await backend.fetch_one("results", {"race": key[0], "geo": key[1]}) for key in keys}
    collection = _collection(backend)
    enrolled: dict[Any, Any] = {}

    async with pool.acquire() as raw:
        conn = _Counting(raw)
        real_enroll = CallerTransaction.enroll

        def enroll(self: CallerTransaction, owner: Any, key: Any, *, row: Any = None) -> None:
            enrolled[key] = row
            real_enroll(self, owner, key, row=row)

        real_restate_many = CallerTransaction.restate_many

        def restate_many(self: CallerTransaction, owner: Any, rows: Any) -> None:
            # the row a key's message carries is its enrollment's last word: a delete that returns its rows
            # enrolls each key before it runs and restates it with the row it deleted
            enrolled.update(rows)
            real_restate_many(self, owner, rows)

        monkeypatch.setattr(CallerTransaction, "enroll", enroll)
        monkeypatch.setattr(CallerTransaction, "restate_many", restate_many)
        async with CallerTransaction(conn):
            assert await collection.delete_rows(keys, conn=conn, max_rows=1_000) == len(keys)
        statements = conn.reads

    assert enrolled == expected
    assert statements == -(-len(keys) // 1_000)
    assert await pool.fetchval("SELECT count(*) FROM results") == 0


async def test_a_bulk_delete_keeps_within_twice_a_keyed_delete(pool: asyncpg.Pool) -> None:
    """Flips if deleting and reading a reload's rows takes over twice the database's keyed delete of them."""
    async with pool.acquire() as conn:
        if not await on_yugabyte(conn):
            pytest.skip("the budget is YugabyteDB's, the platform's database; Postgres checks the rows only")
        baseline = await plain_keyed_delete_seconds(conn, _ROWS)
    collection = _collection(SqlL3Backend(pool))
    best = float("inf")
    for _ in range(3):
        keys = await _fill(pool)
        async with pool.acquire() as conn:
            began = time.perf_counter()
            async with CallerTransaction(conn):
                await collection.delete_rows(keys, conn=conn, max_rows=1_000)
            best = min(best, time.perf_counter() - began)
    problem = budget_problem(f"deleting {_ROWS:,} rows read for their messages", best, baseline)
    assert problem is None, problem
