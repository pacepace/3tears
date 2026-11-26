"""a copy follower keeps a versioned table's copy one row a logical key, each key's newest live version (append-only L3,
chunk 4).

L3 keeps a version of a key a write (its stored key the key and then the epoch, newest first); a delete appends a
"gone" marker; a whole run deleted is one run marker, "this prefix of the key gone at epoch E". The copy holds none of
that: one row a logical key, the newest live version's, and no row of a key whose newest version is a marker or lies
under a newer run marker. The contract this pins:

- a gone marker reaches the copy as a delete;
- a run marker deletes every key the copy holds under its prefix;
- a noted key whose read comes back empty (its newest version a marker, or under a newer run marker) is deleted;
- a rewrite of a key (a newer version) replaces the copy's row, never a second row of the key, and the newest is the
  highest epoch, never the version that landed last;
- a cold start over a table holding several versions of each key holds one row a key;
- the copy is keyed by the logical key, never L3's stored key (the key and then the epoch); and no row it holds is a
  marker.

L3 is a DuckDB behind a reader that names no dialect, so every shared read takes it for L3 and reads each key's newest
live version (``threetears.interfaces.keyed.keys.newest_relation``); the run markers are written by the writer's own
statement (``run_marker_insert_sql``), so the read's text of a prefix is the write's.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

import pytest

pa = pytest.importorskip("pyarrow")
duckdb = pytest.importorskip("duckdb")

from threetears.core.cache.duckdb_binds import run_bound  # noqa: E402
from threetears.core.collections import CollectionRegistry  # noqa: E402
from threetears.core.collections.copy_follower import CopyFollower, FollowedTable  # noqa: E402
from threetears.core.collections.epoch_change import ChangedRow, EpochChange, EpochWalk, walk_by_count  # noqa: E402
from threetears.core.collections.schema_backed import BIGINT_TYPE, STRING_TYPE, Column, TableSchema  # noqa: E402
from threetears.core.ddl.row_epoch import EPOCH_COLUMN, GONE_COLUMN, row_versions  # noqa: E402
from threetears.core.ddl.run_markers import RUN_MARKERS_TABLE, run_marker_insert_sql  # noqa: E402
from threetears.core.testing.l1_tables import TEST_L1_TABLE  # noqa: E402
from threetears.core.testing.published_copy import SharedCopy, copy_rows  # noqa: E402
from packages.core.tests.copy_workers_support import COPY_WORKERS  # noqa: E402

_TABLE = "votes_versioned"

_SCHEMA = TableSchema(
    name=_TABLE,
    primary_key=("state", "geo_id"),
    columns=[
        Column(name="state", column_type=STRING_TYPE),
        Column(name="geo_id", column_type=BIGINT_TYPE),
        Column(name="votes", column_type=BIGINT_TYPE),
    ],
    row_epoch=row_versions("state"),
    l1_eviction=TEST_L1_TABLE,
)

_DDL = (
    f"CREATE TABLE {_TABLE} (state VARCHAR, geo_id BIGINT, votes BIGINT, epoch BIGINT NOT NULL, "
    "gone BOOLEAN NOT NULL DEFAULT false, PRIMARY KEY (state, geo_id, epoch))"
)


def _dict_rows(result: Any) -> list[dict[str, Any]]:
    names = [d[0] for d in result.description]
    return [dict(zip(names, row, strict=True)) for row in result.fetchall()]


class _L3:
    """L3 as a DuckDB holding the versioned table and its run markers, answering ``fetch`` with ``$n`` placeholders."""

    async def answers_from_yugabyte(self) -> bool:
        """a PostgreSQL reader's: its keyed batches bounded beside their keys, and the question sent nowhere."""
        return False

    def __init__(self) -> None:
        self.db = duckdb.connect(":memory:")
        self.db.execute(_DDL)
        self.db.execute(
            f"CREATE TABLE {RUN_MARKERS_TABLE} (table_name VARCHAR, scope VARCHAR, prefix VARCHAR, epoch BIGINT)"
        )
        self.epoch = 0
        self.statements: list[str] = []

    def _next(self) -> int:
        self.epoch += 10
        return self.epoch

    def write(self, rows: Sequence[tuple[str, int, int]], *, at: int | None = None) -> None:
        """each row a new version of its key, at one epoch (an append: never an update of the version held); ``at``
        names an epoch already passed (another writer's commit landing late, its clock behind)."""
        epoch = self._next() if at is None else at
        put = pa.table(
            {
                "state": pa.array([r[0] for r in rows], pa.string()),
                "geo_id": pa.array([r[1] for r in rows], pa.int64()),
                "votes": pa.array([r[2] for r in rows], pa.int64()),
                "epoch": pa.array([epoch] * len(rows), pa.int64()),
            }
        )
        self.db.register("_put", put)
        try:
            self.db.execute(f"INSERT INTO {_TABLE} (state, geo_id, votes, epoch) SELECT * FROM _put")  # noqa: S608
        finally:
            self.db.unregister("_put")

    def gone(self, keys: Sequence[tuple[str, int]]) -> None:
        """each key's "gone" marker at one epoch."""
        epoch = self._next()
        for state, geo_id in keys:
            self.db.execute(
                f"INSERT INTO {_TABLE} (state, geo_id, votes, epoch, gone) VALUES (?, ?, NULL, ?, true)",  # noqa: S608
                [state, geo_id, epoch],
            )

    def run_gone(self, state: str) -> None:
        """a whole run (a state's every key) gone, one run marker, by the writer's own statement."""
        self.db.execute(run_marker_insert_sql(["text"], 0), [_TABLE, self._next(), state])

    def newest(self) -> list[tuple[str, int, int]]:
        """each key's newest live version, as the copy must hold it."""
        rows = self.db.execute(
            f"SELECT state, geo_id, votes, gone, epoch FROM {_TABLE} ORDER BY state, geo_id, epoch DESC"  # noqa: S608
        ).fetchall()
        markers = self.db.execute(f"SELECT prefix, epoch FROM {RUN_MARKERS_TABLE}").fetchall()  # noqa: S608
        held: list[tuple[str, int, int]] = []
        seen: set[tuple[str, int]] = set()
        for state, geo_id, votes, gone, epoch in rows:
            if (state, geo_id) in seen:
                continue
            seen.add((state, geo_id))
            if gone or any(prefix == state and marked > epoch for prefix, marked in markers):
                continue
            held.append((state, geo_id, votes))
        return held

    async def fetch(self, sql: str, *params: Any) -> list[dict[str, Any]]:
        await asyncio.sleep(0)
        cursor = self.db.cursor()
        rows = run_bound(cursor, sql, params, _dict_rows)
        if not sql.startswith("SELECT version()"):
            self.statements.append(sql)
        return rows

    def close(self) -> None:
        self.db.close()


class _Log:
    """the table's write generation and its change records, as the epoch bucket keeps them."""

    def __init__(self) -> None:
        self.count = 0
        self.records: dict[str, EpochChange] = {}

    async def current(self, table: str) -> str:
        return f"inc:{self.count}"

    def commit(
        self, keys: Sequence[tuple[str, int]] = (), *, prefixes: Sequence[dict[str, str]] = (), s_min: int | None = None
    ) -> str:
        self.count += 1
        token = f"inc:{self.count}"
        self.records[token] = EpochChange(
            table=_TABLE,
            token=token,
            whole_table=bool(prefixes),
            rows=[ChangedRow(ids=[s, str(g)]) for s, g in keys],
            prefixes=list(prefixes) or None,
            s_min=s_min,
        )
        return token

    async def change(self, table_name: str, token: str) -> EpochChange | None:
        return self.records.get(token)

    async def changes_after(self, table_name: str, after: str, upto: str) -> EpochWalk:
        return await walk_by_count(self, table_name, after, upto)


class _Seqlock:
    """the writer's stamp: a settled version."""

    def __init__(self) -> None:
        self.version = 0

    async def settled(self) -> Any:
        return self.version


class _World:
    def __init__(self) -> None:
        self.l3, self.log, self.lock = _L3(), _Log(), _Seqlock()
        self.table = FollowedTable.of(_SCHEMA)
        self.shared = SharedCopy([self.table])
        registry = CollectionRegistry()
        registry.follow_generation(_TABLE)
        self.follower = CopyFollower(
            registry,
            [self.table],
            self.shared.copy,
            self.l3,
            settled=self.lock.settled,
            generation=self.log.current,
            changes=self.log,
            page_size=50,
            retry_waits=(1.0, 60.0),
            drift_workers=COPY_WORKERS,
            queries=self.shared.queries,
        )
        self.follower.register()

    def copy(self) -> list[tuple[Any, ...]]:
        held = copy_rows(self.shared.store, _TABLE, ("state", "geo_id", "votes", GONE_COLUMN))
        assert not any(row[3] for row in held), "the copy holds a gone marker as a row"
        return [row[:3] for row in held]

    async def caught_up(self) -> None:
        self.lock.version += 1
        assert await self.follower.catch_up()
        assert self.copy() == self.l3.newest()


@pytest.fixture
async def world() -> Any:
    made = _World()
    yield made
    made.follower.stop()
    made.l3.close()


async def _warmed(world: _World, rows: Sequence[tuple[str, int, int]]) -> None:
    world.l3.write(rows)
    world.log.commit([(s, g) for s, g, _ in rows])
    await world.follower.warm()
    assert await world.follower.catch_up()
    assert world.copy() == world.l3.newest()


def _rows(state: str, count: int, votes: int = 1) -> list[tuple[str, int, int]]:
    return [(state, i, votes) for i in range(count)]


def test_the_copy_of_a_versioned_table_is_keyed_by_its_logical_key_never_the_stored_one() -> None:
    """L3 stores a version of a key a write (its primary key the key and then the epoch); the copy holds one row a
    logical key. A followed table keyed by L3's stored key would hold a row a version: refused."""
    table = FollowedTable.of(_SCHEMA)
    assert table.key == ("state", "geo_id")
    assert _SCHEMA.stored_key == ("state", "geo_id", EPOCH_COLUMN)
    with pytest.raises(ValueError, match="logical key"):
        FollowedTable(name=_TABLE, key=_SCHEMA.stored_key, columns=table.columns)
    with pytest.raises(ValueError, match="logical key"):
        FollowedTable(name=_TABLE, key=("state",), columns=table.columns)


async def test_a_gone_marker_reaches_the_copy_as_a_delete(world: _World) -> None:
    await _warmed(world, _rows("VA", 10))
    world.l3.gone([("VA", 3), ("VA", 7)])
    world.log.commit([("VA", 3), ("VA", 7)])
    await world.caught_up()
    assert ("VA", 3, 1) not in world.copy()
    assert len(world.copy()) == 8


async def test_a_run_marker_deletes_every_key_under_its_prefix(world: _World) -> None:
    await _warmed(world, _rows("VA", 10) + _rows("OH", 5))
    world.l3.run_gone("VA")
    # the run's delete is recorded as its prefix cleared; a restore rebuilds it so from the marker, naming its floor
    world.log.commit(prefixes=[{"state": "VA"}], s_min=world.l3.epoch)
    await world.caught_up()
    assert world.copy() == _rows("OH", 5)


async def test_a_noted_key_that_reads_back_empty_is_deleted(world: _World) -> None:
    """a record names keys whose newest versions L3's newest read hides: one a gone marker, the others under a newer
    run marker. Each reads back empty, and leaves the copy."""
    await _warmed(world, _rows("VA", 4) + _rows("OH", 3))
    world.l3.gone([("VA", 1)])
    world.l3.run_gone("OH")
    # the restored log names the run's keys one by one (a gone marker is a version like any other) and the marker's key
    world.log.commit([("VA", 1), ("OH", 0), ("OH", 2)])
    world.lock.version += 1
    assert await world.follower.catch_up()
    held = world.copy()
    assert ("VA", 1, 1) not in held
    assert [row for row in held if row[0] == "OH"] == [("OH", 1, 1)]


async def test_a_rewrite_of_a_key_replaces_the_copys_row_with_no_duplicate(world: _World) -> None:
    await _warmed(world, _rows("VA", 5))
    world.l3.write([("VA", 2, 20)])
    world.log.commit([("VA", 2)])
    await world.caught_up()
    world.l3.write([("VA", 2, 30), ("VA", 4, 40)])
    newest = world.l3.epoch
    # a version of the key committed after the newest but stamped below it: newest wins by epoch, never by arrival
    world.l3.write([("VA", 2, 25)], at=newest - 5)
    world.log.commit([("VA", 2), ("VA", 4)])
    await world.caught_up()
    held = world.copy()
    assert [row for row in held if row[1] == 2] == [("VA", 2, 30)]
    assert len(held) == 5


async def test_a_cold_start_over_five_versions_a_key_holds_one_row_a_key(world: _World) -> None:
    for votes in range(1, 6):
        world.l3.write(_rows("VA", 30, votes) + _rows("OH", 20, votes))
    world.l3.gone([("VA", 0)])
    world.l3.run_gone("OH")
    world.l3.write([("OH", 5, 99)])  # a version newer than the run's marker: live
    await world.follower.warm()
    assert await world.follower.catch_up()
    held = world.copy()
    assert held == world.l3.newest()
    assert len(held) == len({(state, geo_id) for state, geo_id, _ in held}) == 30
    assert ("VA", 1, 5) in held and ("OH", 5, 99) in held
