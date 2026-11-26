"""a copy follower's keyed staging of a scope-in-row table, at a realistic size, never holds the event loop.

A reports pod's copy follower froze its loop 12.2 s and 24.3 s (2026-11-25) in ``_stage_keyed`` ->
``arrow_keys.distinct_keys``: each page's scopes found by key (the copy's rows beside L3's) were made distinct on the
loop, a pass over every row of the page. Fine at a test's fifty keys, a frozen pod at a commit's hundreds of thousands.
This runs one large commit through the follower's keyed staging, sized so the pass is quick on a laptop yet a turn
that grows with the rows passes the loop-stall guard's limit (``threetears.core.testing.loop_stall``), which fails
the test naming what held the loop. Nothing here asserts on time: the guard is the check.

L3 is a DuckDB read on a thread, as the real rail's reply is off the loop, so the only work this puts on the loop is
the follower's own.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import pytest
from sqlalchemy import BigInteger, Column as SaColumn, MetaData, String, Table
from threetears.core.testing.l1_tables import L1_EVICTION_INFO, TEST_L1_TABLE

pa = pytest.importorskip("pyarrow")
duckdb = pytest.importorskip("duckdb")

from threetears.core.cache.duckdb import DuckDBBackend  # noqa: E402
from threetears.core.cache.duckdb_binds import run_bound  # noqa: E402
from threetears.core.collections import CollectionRegistry  # noqa: E402
from packages.core.tests.copy_workers_support import COPY_WORKERS  # noqa: E402
from threetears.core.collections.copy_follower import CopyFollower, FollowedTable  # noqa: E402
from threetears.core.testing.published_copy import PublishedCopy  # noqa: E402
from threetears.core.collections.epoch_change import ChangedRow, EpochChange, EpochWalk, walk_by_count  # noqa: E402
from threetears.core.collections.schema_backed import BIGINT_TYPE, STRING_TYPE, Column  # noqa: E402

_TABLE = "votes"
#: the scope (``state``) is in the row only, not the key: each page's scopes are read beside its rows and made distinct
_KEY = ("geo_id",)
_COLUMNS = (
    Column(name="state", column_type=STRING_TYPE),
    Column(name="geo_id", column_type=BIGINT_TYPE),
    Column(name="votes", column_type=BIGINT_TYPE),
)
#: keys a commit changes: a large commit's, scaled so the pass runs in seconds (the night's commits were larger)
_KEYS = 600_000
#: keys a change record names: a record holds at most 256 KiB, about sixty bytes a named row
_RECORD_KEYS = 4_000
#: the states the keys spread over, each a scope
_STATES = ("VA", "IL", "PA", "GA", "AZ", "MI", "WI", "NV")


def _metadata() -> MetaData:
    metadata = MetaData()
    Table(
        _TABLE,
        metadata,
        SaColumn("state", String(8)),
        SaColumn("geo_id", BigInteger, primary_key=True),
        SaColumn("votes", BigInteger),
        info={L1_EVICTION_INFO: TEST_L1_TABLE},
    )
    return metadata


def _rows(votes: int) -> Any:
    """every key's row: ``_KEYS`` keys spread over the states."""
    geo = pa.array(range(_KEYS), pa.int64())
    states = pa.array([_STATES[i % len(_STATES)] for i in range(_KEYS)], pa.string())
    return pa.table({"state": states, "geo_id": geo, "votes": pa.array([votes] * _KEYS, pa.int64())})


class _L3:
    """L3 as a DuckDB, read on a thread (the rail's reply is off the loop)."""

    def __init__(self) -> None:
        self.db = duckdb.connect(":memory:")
        self.db.execute(f"CREATE TABLE {_TABLE} (state VARCHAR, geo_id BIGINT PRIMARY KEY, votes BIGINT)")

    async def answers_from_yugabyte(self) -> bool:
        return False

    def put(self, rows: Any) -> None:
        self.db.register("_put", rows)
        try:
            self.db.execute(f"INSERT OR REPLACE INTO {_TABLE} SELECT * FROM _put")  # noqa: S608
        finally:
            self.db.unregister("_put")

    def _fetch(self, sql: str, params: tuple[Any, ...]) -> list[dict[str, Any]]:
        def dict_rows(result: Any) -> list[dict[str, Any]]:
            names = [d[0] for d in result.description]
            return [dict(zip(names, row, strict=True)) for row in result.fetchall()]

        return run_bound(self.db.cursor(), sql, params, dict_rows)

    async def fetch(self, sql: str, *params: Any) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self._fetch, sql, params)


class _Log:
    """the table's write generation and its change records."""

    def __init__(self) -> None:
        self.count = 0
        self.records: dict[str, EpochChange] = {}

    async def current(self, table: str) -> str:
        return f"inc:{self.count}"

    def commit_whole(self) -> None:
        self.count += 1
        token = f"inc:{self.count}"
        self.records[token] = EpochChange(table=_TABLE, token=token, whole_table=True)

    def commit_every_key(self) -> None:
        """one commit naming every key, as the records the writer keeps for it: ``_RECORD_KEYS`` keys a record. Built
        on a thread by the test: a test's own setup belongs off the loop."""
        for start in range(0, _KEYS, _RECORD_KEYS):
            self.count += 1
            token = f"inc:{self.count}"
            rows = [ChangedRow(ids=[str(i)]) for i in range(start, min(start + _RECORD_KEYS, _KEYS))]
            self.records[token] = EpochChange(table=_TABLE, token=token, rows=rows)

    async def change(self, table_name: str, token: str) -> EpochChange | None:
        return self.records.get(token)

    async def changes_after(self, table_name: str, after: str, upto: str) -> EpochWalk:
        return await walk_by_count(self, table_name, after, upto)


class _Scopes:
    """the writer's record: every state's epoch, nothing held."""

    def __init__(self) -> None:
        self.version = 1
        self.epochs = {f"state:{state}": 1 for state in _STATES}

    async def settled(self) -> Any:
        return SimpleNamespace(version=self.version, epochs=dict(self.epochs), pending=frozenset())

    def commit(self) -> None:
        self.version += 1
        self.epochs = {scope: self.version for scope in self.epochs}


@pytest.fixture
def world() -> Iterator[tuple[_L3, _Log, _Scopes, DuckDBBackend, CopyFollower]]:
    """a follower of one scope-in-row table, its page as large as the commit (a page of the night's size)."""
    l3, log, scopes = _L3(), _Log(), _Scopes()
    l3.put(_rows(1))
    log.commit_whole()
    backend = DuckDBBackend()
    backend.initialize(_metadata())
    registry = CollectionRegistry()
    registry.follow_generation(_TABLE)
    published = PublishedCopy(backend)
    follower = CopyFollower(
        registry,
        [FollowedTable(name=_TABLE, key=_KEY, columns=_COLUMNS, scope=("state", "state:"))],
        published,
        l3,
        settled=scopes.settled,
        generation=log.current,
        changes=log,
        page_size=_KEYS,
        stamp_table="scope_epochs",
        drift_workers=COPY_WORKERS,
        queries=published.queries,
    )
    follower.register()
    try:
        yield l3, log, scopes, backend, follower
    finally:
        backend.reset()
        l3.db.close()


async def test_a_large_commit_is_staged_by_key_without_holding_the_loop(world: Any) -> None:
    """Flips (through the loop-stall guard) if staging a commit's keys runs a pass over its rows on the event loop:
    the scopes made distinct there (``distinct_keys``), or any other per-row work."""
    l3, log, scopes, backend, follower = world
    await follower.warm()
    assert await follower.catch_up()
    # the 600k rows are built on the thread too: building them on the loop held it 0.25 s on a slow pool machine
    await asyncio.to_thread(lambda: l3.put(_rows(2)))
    await asyncio.to_thread(log.commit_every_key)
    scopes.commit()
    assert await follower.catch_up()
    with backend.read_snapshot() as cursor:
        (total,) = cursor.execute(f"SELECT sum(votes) FROM {_TABLE}").fetchone()  # noqa: S608
    assert total == 2 * _KEYS
