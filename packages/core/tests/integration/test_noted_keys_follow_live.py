"""Integration: a commit too large to record reaches a copy follower exactly, at the cost of a keyed read of its keys.

A table carries the change-note trigger; one transaction updates more rows than sixteen change records hold, and its
notes bracket is read just before its COMMIT, as the broker reads it. Its epoch is recorded keys unknown with that
bracket. A copy follower (DuckDB) whose registry names the keys from the notes (:func:`noted_keys_page`, the read the
broker's ``l3.noted_keys`` door answers) is checked to:

- hold the table exactly as Postgres does after its catch-up;
- take, end to end (the notes' pages, the keyed read and the apply), within 2x the same follower applying a record
  that names the same keys (a bare keyed read and the apply), the budget calibrated in the test.

Timed runs belong under measure-lock. Against the session ``db_container`` (Postgres); set
``THREETEARS_TEST_POSTGRES_URL`` to a YugabyteDB URL to run them there.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import asyncpg
import pytest
from sqlalchemy import BigInteger, Column as SaColumn, MetaData, String, Table

duckdb = pytest.importorskip("duckdb")
pytest.importorskip("pyarrow")

from threetears.core.cache.duckdb import DuckDBBackend  # noqa: E402
from threetears.core.collections import CollectionRegistry  # noqa: E402
from threetears.core.collections.change_notes import NotedKeysCursor, NotedKeysPage, noted_keys_page  # noqa: E402
from threetears.core.collections.copy_follower import CopyFollower, FollowedTable  # noqa: E402
from threetears.core.collections.epoch_change import ChangedRow, EpochChange, EpochWalk, walk_by_count  # noqa: E402
from threetears.core.collections.schema_backed import BIGINT_TYPE, STRING_TYPE, Column  # noqa: E402
from threetears.core.ddl.change_notes import note_trigger_sql, trigger_function_sql  # noqa: E402
from threetears.core.testing.calibrate import budget_problem  # noqa: E402
from threetears.core.testing.note_ring import install_note_ring  # noqa: E402

pytestmark = pytest.mark.integration

_TABLE = "votes"
_KEY = ("state", "geo_id")
_COLUMNS = (
    Column(name="state", column_type=STRING_TYPE),
    Column(name="geo_id", column_type=BIGINT_TYPE),
    Column(name="votes", column_type=BIGINT_TYPE),
)
#: the rows the table holds, and how many one commit changes: past sixteen 256 KiB records at ~60 bytes a row
_HELD = 150_000
_CHANGED = 70_000
#: timed runs of each path, the best kept
_RUNS = 3


def _metadata() -> MetaData:
    metadata = MetaData()
    Table(
        _TABLE,
        metadata,
        SaColumn("state", String(8), primary_key=True),
        SaColumn("geo_id", BigInteger, primary_key=True),
        SaColumn("votes", BigInteger),
    )
    return metadata


@dataclass
class _Db:
    conn: asyncpg.Connection
    pool: asyncpg.Pool
    notes: str
    tenant: str | None = None


class _Log:
    """the table's write generation and its records, as the epoch bucket keeps them."""

    def __init__(self) -> None:
        self.count = 0
        self.records: dict[str, EpochChange] = {}

    async def current(self, table: str) -> str:
        return f"inc:{self.count}"

    def record(self, **reach: Any) -> None:
        self.count += 1
        token = f"inc:{self.count}"
        self.records[token] = EpochChange(table=_TABLE, token=token, **reach)

    async def change(self, table_name: str, token: str) -> EpochChange | None:
        return self.records.get(token)

    async def changes_after(self, table_name: str, after: str, upto: str) -> EpochWalk:
        return await walk_by_count(self, table_name, after, upto)


@dataclass
class _PoolNotes:
    """the broker's door, in process: the table's notes of the tenant, a page at a time."""

    db: _Db

    async def noted_keys(
        self,
        table: str,
        *,
        noted_between: tuple[datetime, datetime],
        cursor: NotedKeysCursor | None,
        limit: int,
        max_bytes: int,
    ) -> NotedKeysPage | str:
        return await noted_keys_page(
            self.db.pool,
            notes_schema=self.db.notes,
            table=table,
            tenant_token=self.db.tenant,
            noted_between=noted_between,
            cursor=cursor,
            limit=limit,
            max_bytes=max_bytes,
        )


@pytest.fixture
async def db(db_container: str) -> AsyncIterator[_Db]:
    """a table carrying the change-note trigger, its ring beside it, and its rows."""
    schema = f"follow_{uuid.uuid4().hex[:8]}"
    conn = await asyncpg.connect(db_container, command_timeout=1200, server_settings={"search_path": schema})
    try:
        await conn.execute(f"CREATE SCHEMA {schema}")
        await install_note_ring(conn, schema)
        await conn.execute(
            f"CREATE TABLE {schema}.{_TABLE} (state TEXT, geo_id BIGINT, votes BIGINT, PRIMARY KEY (state, geo_id))"
        )
        await conn.execute(
            trigger_function_sql(notes_schema=schema, table_schema=schema, table=_TABLE, keys=_KEY, columns=())
        )
        for statement in note_trigger_sql(notes_schema=schema, table_schema=schema, table=_TABLE).values():
            await conn.execute(statement)
        await conn.execute(
            f"INSERT INTO {schema}.{_TABLE} SELECT 'VA', g, 1 FROM generate_series(1, $1) g",  # noqa: S608
            _HELD,
        )
        pool = await asyncpg.create_pool(
            db_container, min_size=1, max_size=4, command_timeout=1200, server_settings={"search_path": schema}
        )
        try:
            yield _Db(conn=conn, pool=pool, notes=schema)
        finally:
            await pool.close()
    finally:
        await conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        await conn.close()


async def _commit(db: _Db, votes: int) -> tuple[datetime, datetime]:
    """one transaction updating ``_CHANGED`` rows, its notes bracket read just before its COMMIT, as the broker does."""
    async with db.conn.transaction():
        await db.conn.execute(f"UPDATE {_TABLE} SET votes = $1 WHERE geo_id <= $2", votes, _CHANGED)  # noqa: S608
        row = await db.conn.fetchrow("SELECT now() AS since, clock_timestamp() AS until")
    return row["since"], row["until"]


async def _warm_follower(db: _Db, log: _Log, version: list[int]) -> tuple[CopyFollower, CollectionRegistry, Any]:
    backend = DuckDBBackend()
    backend.initialize(_metadata())
    registry = CollectionRegistry()
    registry.follow_generation(_TABLE)

    async def settled() -> int:
        return version[0]

    follower = CopyFollower(
        registry,
        [FollowedTable(name=_TABLE, key=_KEY, columns=_COLUMNS)],
        backend,
        db.pool,
        settled=settled,
        generation=log.current,
        changes=log,
    )
    follower.register()
    await follower.warm()
    assert await follower.catch_up()
    return follower, registry, backend


def _copy(backend: Any) -> list[tuple[Any, ...]]:
    with backend.read_snapshot() as cursor:
        return sorted(cursor.execute(f"SELECT state, geo_id, votes FROM {_TABLE}").fetchall())


async def _held(db: _Db) -> list[tuple[Any, ...]]:
    return sorted(tuple(row) for row in await db.conn.fetch(f"SELECT state, geo_id, votes FROM {_TABLE}"))  # noqa: S608


async def _timed_pass(db: _Db, *, noted: bool, votes: int) -> tuple[float, bool]:
    """one warm follower's catch-up of one large commit: by its notes, or by a record naming its keys."""
    log, version = _Log(), [0]
    follower, registry, backend = await _warm_follower(db, log, version)
    since_until = await _commit(db, votes)
    if noted:
        registry.set_noted_keys_reader(_PoolNotes(db))
        log.record(keys_unknown=True, noted_between=since_until)
    else:
        log.record(rows=[ChangedRow(ids=["VA", str(g)]) for g in range(1, _CHANGED + 1)])
    version[0] += 1
    began = time.perf_counter()
    assert await follower.catch_up()
    took = time.perf_counter() - began
    return took, _copy(backend) == await _held(db)


def _best(times: Sequence[float]) -> float:
    return min(times)


class TestALargeCommitReachesAFollowerFromItsNotes:
    async def test_the_copy_equals_l3_after_the_catch_up(self, db: _Db) -> None:
        """Flips if a page skips a key, or the follower reads other keys than the commit's."""
        _, equal = await _timed_pass(db, noted=True, votes=2)
        assert equal

    async def test_end_to_end_within_twice_a_bare_keyed_read_of_the_same_keys(self, db: _Db) -> None:
        """Flips if naming the keys from the notes costs more than the keyed read and apply it feeds."""
        noted, bare = [], []
        for run in range(_RUNS):
            took, equal = await _timed_pass(db, noted=False, votes=10 + run)
            assert equal
            bare.append(took)
            took, equal = await _timed_pass(db, noted=True, votes=20 + run)
            assert equal
            noted.append(took)
        print(
            f"notes {_best(noted) * 1000:.1f} ms, keyed {_best(bare) * 1000:.1f} ms, {_best(noted) / _best(bare):.2f}x"
        )
        problem = budget_problem("a large commit from its notes, end to end", _best(noted), _best(bare))
        assert problem is None, problem
