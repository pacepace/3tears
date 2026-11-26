"""Integration: whole key prefixes deleted at a cost that does not grow with the rows held, seen by a copy follower.

Reports drops the LSF runs the warehouse no longer has (a state's runs relabelled 'test'): 23 runs, 1.44M rows deleted
row by row after 1.64M were read back, took 794 s on the 2026-11-02 replay. ``delete_prefixes`` deletes a prefix's
rows in one key-led statement and reads none back; and a copy follower reads each prefix again as a key range, never
the table. These check, on a real database:

- the statements sent and the rows read back are the same for a prefix of 1,000 rows and one of 100,000;
- a copy follower (DuckDB) learns the delete from its epoch and drops exactly the prefix's rows, with no whole read.

Against the session ``db_container`` (Postgres); set ``THREETEARS_TEST_POSTGRES_URL`` to a YugabyteDB URL to run them
there.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import asyncpg
import pytest
from prometheus_client import REGISTRY
from sqlalchemy import BigInteger, Column as SaColumn, MetaData, String, Table

from threetears.core.backends.sql import SqlL3Backend
from threetears.core.collections import WRITE_GENERATION, CollectionRegistry
from threetears.core.collections.caller_transaction import CallerTransaction
from threetears.core.collections.epoch_change import ChangedRow, EpochChange, EpochWalk, walk_by_count
from threetears.core.collections.schema_backed import (
    BIGINT_TYPE,
    STRING_TYPE,
    Column,
    TableSchema,
    collection_for_schema,
)
from threetears.core.config import DefaultCoreConfig
from threetears.core.entities.base import BaseEntity
from threetears.core.testing.l1_tables import L1_EVICTION_INFO, TEST_L1_TABLE

duckdb = pytest.importorskip("duckdb")
pytest.importorskip("pyarrow")

from threetears.core.cache.duckdb import DuckDBBackend  # noqa: E402
from threetears.core.collections.copy_follower import CopyFollower, FollowedTable  # noqa: E402
from threetears.core.testing.published_copy import PublishedCopy, close_published_copies  # noqa: E402

pytestmark = pytest.mark.integration

_TABLE = "lsf_runs"
_KEY = ("state", "run_id", "geo")
_COLUMNS = (
    Column(name="state", column_type=STRING_TYPE),
    Column(name="run_id", column_type=BIGINT_TYPE),
    Column(name="geo", column_type=STRING_TYPE),
    Column(name="votes", column_type=BIGINT_TYPE, nullable=True),
)
_SCHEMA = TableSchema(name=_TABLE, primary_key=_KEY, columns=list(_COLUMNS), l1_eviction=TEST_L1_TABLE)

#: the gone run, and a run of the same state and one of another state that stay
_GONE = {"state": "PA", "run_id": 7}
_KEPT_ROWS = 500


@pytest.fixture(autouse=True)
async def _published_copies_closed() -> AsyncIterator[None]:
    """every copy a test published (``testing.published_copy``) closed after it: its query workers and its store."""
    yield
    await close_published_copies()


class _Row(BaseEntity):
    primary_key_field = "geo"


class _Runs(collection_for_schema(_SCHEMA, entity_class=_Row)):  # type: ignore[misc]
    write_generation = WRITE_GENERATION


class _Log:
    """the table's write generation and its records, as the epoch bucket keeps them: written by the writer's
    advance, walked by the follower's catch-up."""

    def __init__(self) -> None:
        self.count = 0
        self.records: dict[str, EpochChange] = {}

    async def current(self, table: str) -> str:
        return f"inc:{self.count}"

    async def advance(self, table: str) -> str:
        self.count += 1
        return f"inc:{self.count}"

    async def advance_with_change(
        self,
        table_name: str,
        rows: Sequence[ChangedRow] | None,
        *,
        keys_unknown: bool = False,
        partition: Mapping[str, str] | None = None,
        prefixes: Sequence[Mapping[str, str]] | None = None,
        reason: str | None = None,
    ) -> str:
        token = await self.advance(table_name)
        self.records[token] = EpochChange(
            table=table_name,
            token=token,
            whole_table=rows is None and not keys_unknown,
            keys_unknown=keys_unknown,
            rows=list(rows) if rows is not None else [],
            partition=dict(partition) if partition else None,
            prefixes=[dict(prefix) for prefix in prefixes] if prefixes else None,
            reason=reason,
        )
        return token

    async def change(self, table_name: str, token: str) -> EpochChange | None:
        return self.records.get(token)

    async def changes_after(self, table_name: str, after: str, upto: str) -> EpochWalk:
        return await walk_by_count(self, table_name, after, upto)


@dataclass
class _Db:
    pool: asyncpg.Pool
    schema: str


class _Counting:
    """a connection that counts the statements sent on it and the rows they answered; the rest is the connection's."""

    def __init__(self, conn: Any) -> None:
        self._conn = conn
        self.statements = 0
        self.rows_read = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self._conn, name)

    async def execute(self, query: str, *args: Any) -> Any:
        self.statements += 1
        return await self._conn.execute(query, *args)

    async def fetch(self, query: str, *args: Any) -> Any:
        self.statements += 1
        rows = await self._conn.fetch(query, *args)
        self.rows_read += len(rows)
        return rows

    async def fetchrow(self, query: str, *args: Any) -> Any:
        self.statements += 1
        row = await self._conn.fetchrow(query, *args)
        self.rows_read += row is not None
        return row


@pytest.fixture
async def db(db_container: str) -> AsyncIterator[_Db]:
    """the runs table."""
    schema = f"prefixdel_{uuid.uuid4().hex[:8]}"
    conn = await asyncpg.connect(db_container, command_timeout=600)
    try:
        await conn.execute(f"CREATE SCHEMA {schema}")
        await conn.execute(
            f"CREATE TABLE {schema}.{_TABLE} (state TEXT, run_id BIGINT, geo TEXT, votes BIGINT, "
            "PRIMARY KEY (state, run_id, geo))"
        )
        pool = await asyncpg.create_pool(
            db_container, min_size=1, max_size=4, command_timeout=600, server_settings={"search_path": schema}
        )
        try:
            yield _Db(pool=pool, schema=schema)
        finally:
            await pool.close()
    finally:
        await conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        await conn.close()


async def _fill(db: _Db, gone_rows: int) -> None:
    """the gone run's rows, and the kept runs' rows."""
    async with db.pool.acquire() as conn, conn.transaction():
        await conn.execute(f"DELETE FROM {_TABLE}")  # noqa: S608 - the test's own table, emptied between fills
        await conn.execute(
            f"INSERT INTO {_TABLE} SELECT 'PA', 7, 'g' || g, g FROM generate_series(1, $1::int) g",  # noqa: S608
            gone_rows,
        )
        await conn.execute(
            f"INSERT INTO {_TABLE} SELECT s, r, 'g' || g, g FROM (VALUES ('PA', 6), ('OH', 7)) AS k(s, r), "  # noqa: S608
            "generate_series(1, $1::int) g",
            _KEPT_ROWS,
        )


def _collection(db: _Db, log: _Log | None = None) -> Any:
    registry = CollectionRegistry()
    registry.configure(l3_pool=SqlL3Backend(db.pool))
    if log is not None:
        registry.set_generation_source(log)
    return _Runs(registry, DefaultCoreConfig(), None)


async def _counts(db: _Db) -> dict[tuple[str, int], int]:
    rows = await db.pool.fetch(f"SELECT state, run_id, count(*) AS n FROM {_TABLE} GROUP BY 1, 2")  # noqa: S608
    return {(row["state"], row["run_id"]): row["n"] for row in rows}


async def test_the_statements_and_the_rows_read_are_the_same_for_1k_and_100k_rows(db: _Db) -> None:
    """Flips if deleting a prefix reads its rows back, or sends more statements as the prefix holds more rows."""
    costs = []
    for gone_rows in (1_000, 100_000):
        await _fill(db, gone_rows)
        collection = _collection(db)
        async with db.pool.acquire() as raw:
            conn = _Counting(raw)
            async with CallerTransaction(conn):
                assert await collection.delete_prefixes([_GONE], conn=conn) == gone_rows
            costs.append((conn.statements, conn.rows_read))
        assert await _counts(db) == {("PA", 6): _KEPT_ROWS, ("OH", 7): _KEPT_ROWS}
    # the marks' one row, the delete, the marks put back: nothing of the table read
    assert costs[0] == costs[1] == (3, 1)


def _metadata() -> MetaData:
    metadata = MetaData()
    Table(
        _TABLE,
        metadata,
        SaColumn("state", String(8), primary_key=True),
        SaColumn("run_id", BigInteger, primary_key=True),
        SaColumn("geo", String(16), primary_key=True),
        SaColumn("votes", BigInteger),
        info={L1_EVICTION_INFO: TEST_L1_TABLE},
    )
    return metadata


def _whole_reads(why: str) -> float:
    return REGISTRY.get_sample_value("copy_follower_whole_reads_total", {"table": _TABLE, "why": why}) or 0.0


async def test_a_copy_follower_drops_the_prefix_from_its_epoch_and_reads_no_table_whole(db: _Db) -> None:
    """Flips if a follower misses the delete, drops other rows, or reads the table whole to learn it."""
    await _fill(db, 3_000)
    log, version = _Log(), [0]
    backend = DuckDBBackend()
    backend.initialize(_metadata())
    registry = CollectionRegistry()
    registry.follow_generation(_TABLE)

    async def settled() -> int:
        return version[0]

    published = PublishedCopy(backend)
    follower = CopyFollower(
        registry,
        [FollowedTable(name=_TABLE, key=_KEY, columns=_COLUMNS)],
        published,
        db.pool,
        settled=settled,
        generation=log.current,
        changes=log,
        queries=published.queries,
    )
    follower.register()
    await follower.warm()
    assert await follower.catch_up()
    whole = {why: _whole_reads(why) for why in ("cold_start", "reset", "legacy_whole_record")}

    collection = _collection(db, log)
    async with db.pool.acquire() as conn, CallerTransaction(conn):
        assert await collection.delete_prefixes([_GONE], conn=conn) == 3_000
    [record] = log.records.values()
    assert record.whole_table and record.prefixes == [{"state": "PA", "run_id": "7"}]
    version[0] += 1
    assert await follower.catch_up()

    with backend.read_snapshot() as cursor:
        copy = cursor.execute(f"SELECT state, run_id, count(*) FROM {_TABLE} GROUP BY 1, 2").fetchall()  # noqa: S608
    assert {(state, run): n for state, run, n in copy} == await _counts(db)
    assert {why: _whole_reads(why) for why in whole} == whole
