"""a running copy that hears a reset or a schema change of a table that opts in to delta files never reads it from L3.

Decisions 12 and 14 of ``design-pod-delta-files.md``: on a reset or a schema change the writer rebuilds that table's
full once from L3 and publishes it; copies wait for it. Election night failed with L3 on the hot path, and a reset read
back whole by every pod copy is the worst of it. The registry, the follower, the delta files and the fulls are the real
ones; L3 is a DuckDB that counts the statements it answers for each table, kept equal to what the writes leave in L3,
so "the copy equals L3" and "no statement reached L3" are both checked. The contract this pins:

- a running copy hearing a reset of an opted-in table issues no L3 statement: it says in its status that the table
  waits, keeps serving it as it held it, then reloads it alone from the full the writer publishes, applies the deltas
  newer than its parts from their files, and ends equal to L3;
- every other table keeps being followed while it waits;
- a full from before the reset (or of none of its generation) is never taken;
- a schema change of an opted-in table waits the same way;
- a table that does not opt in still reads L3 whole on a reset, as before.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterator
from pathlib import Path
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
from threetears.core.collections.copy_follower import CopyFollower, FollowedTable  # noqa: E402
from threetears.core.collections.delta_files import TableDelta, write_delta_file  # noqa: E402
from threetears.core.collections.epoch_change import (  # noqa: E402
    RESET_REASON,
    ChangedRow,
    EpochChange,
    EpochWalk,
    walk_by_count,
)
from threetears.core.collections.schema_backed import BIGINT_TYPE, STRING_TYPE, Column, rail_arrow_types  # noqa: E402
from threetears.core.collections.volume_fulls import (  # noqa: E402
    FullTable,
    VolumeFulls,
    publish_full_manifest,
    write_full_part,
)

#: opted in to delta files: reset in these tests
_RESET = "results"
#: opted in too: keeps updating while ``results`` waits
_OTHER = "tallies"
#: not opted in: read from L3 as before
_PLAIN = "plain"
_OPTED = (_RESET, _OTHER)
_KEY = ("state", "geo_id")
_COLUMNS = (
    Column(name="state", column_type=STRING_TYPE),
    Column(name="geo_id", column_type=BIGINT_TYPE),
    Column(name="votes", column_type=BIGINT_TYPE),
)
_NAMES = tuple(column.name for column in _COLUMNS)
_TYPES = rail_arrow_types(_COLUMNS)

_OPENED: list[Callable[[], None]] = []


@pytest.fixture(autouse=True)
def _close_what_the_test_opened() -> Iterator[None]:
    """close every DuckDB the test opened (each runs a worker thread a core; test_copy_follower has the account)."""
    yield
    while _OPENED:
        _OPENED.pop()()


@pytest.fixture(autouse=True)
def _volume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """the shared volume the delta files and the fulls are saved on and read off."""
    monkeypatch.setenv("THREETEARS_DELTA_FILES_DIR", str(tmp_path))
    return tmp_path


def _followed(name: str) -> FollowedTable:
    return FollowedTable(name=name, key=_KEY, columns=_COLUMNS, scope=("state", "state:"), delta_files=name in _OPTED)


def _metadata() -> MetaData:
    metadata = MetaData()
    for name in (_RESET, _OTHER, _PLAIN):
        Table(
            name,
            metadata,
            SaColumn("state", String(8), primary_key=True),
            SaColumn("geo_id", BigInteger, primary_key=True),
            SaColumn("votes", BigInteger),
            info={L1_EVICTION_INFO: TEST_L1_TABLE},
        )
    return metadata


def _dict_rows(result: Any) -> list[dict[str, Any]]:
    names = [d[0] for d in result.description]
    return [dict(zip(names, row, strict=True)) for row in result.fetchall()]


class _L3:
    """L3 as a DuckDB answering ``fetch``; counts the statements that read each table."""

    def __init__(self) -> None:
        self.db = duckdb.connect(":memory:")
        _OPENED.append(self.db.close)
        for name in (_RESET, _OTHER, _PLAIN):
            self.db.execute(
                f"CREATE TABLE {name} (state VARCHAR, geo_id BIGINT, votes BIGINT, PRIMARY KEY (state, geo_id))"
            )
        self.statements: dict[str, int] = dict.fromkeys((_RESET, _OTHER, _PLAIN), 0)

    def put(self, table: str, rows: list[tuple[str, int, int]]) -> None:
        for row in rows:
            self.db.execute(f"INSERT OR REPLACE INTO {table} VALUES (?, ?, ?)", list(row))  # noqa: S608

    def reset(self, table: str, rows: list[tuple[str, int, int]]) -> None:
        """an operator's reset: the table emptied and loaded again."""
        self.db.execute(f"DELETE FROM {table}")  # noqa: S608
        self.put(table, rows)

    def held(self, table: str) -> list[tuple[Any, ...]]:
        return sorted(self.db.execute(f"SELECT {', '.join(_NAMES)} FROM {table}").fetchall())  # noqa: S608

    async def fetch(self, sql: str, *params: Any) -> list[dict[str, Any]]:
        await asyncio.sleep(0)
        rows = run_bound(self.db.cursor(), sql, params, _dict_rows)
        for name in self.statements:
            if f'"{name}"' in sql or f" {name} " in sql or f" {name}\n" in sql or sql.rstrip().endswith(name):
                self.statements[name] += 1
        return rows


class _Log:
    """each table's write generation and its change records, as the broker's change reader answers them."""

    def __init__(self) -> None:
        self.count: dict[str, int] = dict.fromkeys((_RESET, _OTHER, _PLAIN), 1)
        self.records: dict[tuple[str, str], EpochChange] = {}

    async def current(self, table: str) -> str:
        return f"inc:{self.count[table]}"

    def _next(self, table: str) -> str:
        self.count[table] += 1
        return f"inc:{self.count[table]}"

    def commit(self, table: str, keys: list[tuple[str, int]], delta_file: str | None = None) -> str:
        token = self._next(table)
        self.records[(table, token)] = EpochChange(
            table=table, token=token, rows=[ChangedRow(ids=[s, str(g)]) for s, g in keys], delta_file=delta_file
        )
        return token

    def reset(self, table: str) -> str:
        token = self._next(table)
        self.records[(table, token)] = EpochChange(table=table, token=token, whole_table=True, reason=RESET_REASON)
        return token

    async def change(self, table_name: str, token: str) -> EpochChange | None:
        return self.records.get((table_name, token))

    async def changes_after(self, table_name: str, after: str, upto: str) -> EpochWalk:
        return await walk_by_count(self, table_name, after, upto)


class _Scopes:
    """the writer's record, settled: each scope's epoch."""

    def __init__(self) -> None:
        self.version = 1
        self.epochs = {"state:VA": 1, "state:IL": 1}

    async def settled(self) -> Any:
        return SimpleNamespace(version=self.version, epochs=dict(self.epochs), pending=frozenset())

    def commit(self, *scopes: str) -> None:
        self.version += 1
        for scope in scopes:
            self.epochs[scope] = self.version


def _part_rows(rows: list[tuple[str, int, int]]) -> Any:
    return pa.table(
        {
            "state": pa.array([r[0] for r in rows], pa.string()),
            "geo_id": pa.array([r[1] for r in rows], pa.int64()),
            "votes": pa.array([r[2] for r in rows], pa.int64()),
        }
    )


def _entry(volume: Path, table: str, rows: list[tuple[str, int, int]], version: str) -> FullTable:
    """the table's entry in a full: a part a state, each at ``version``."""
    parts = tuple(
        write_full_part(volume, table, f"state:{state}", _part_rows([r for r in rows if r[0] == state]), version)
        for state in sorted({r[0] for r in rows})
    )
    return FullTable(version=version, parts=parts)


class _World:
    def __init__(self, volume: Path) -> None:
        self.volume = volume
        self.l3, self.log, self.lock = _L3(), _Log(), _Scopes()
        self.backend = DuckDBBackend()
        _OPENED.append(self.backend.reset)
        self.backend.initialize(_metadata())
        self.registry = CollectionRegistry()
        for name in (_RESET, _OTHER, _PLAIN):
            self.registry.follow_generation(name)
        self.follower = CopyFollower(
            self.registry,
            [_followed(_RESET), _followed(_OTHER), _followed(_PLAIN)],
            self.backend,
            self.l3,
            settled=self.lock.settled,
            generation=self.log.current,
            changes=self.log,
            page_size=50,
            stamp_table="scope_epochs",
            fulls=VolumeFulls(volume, [_followed(_RESET), _followed(_OTHER)], poll_seconds=0.01),
        )
        self.writes = 0

    async def started(self) -> None:
        """L3 and the writer's first full at inc:1; the copy starts from it (``plain`` read whole, a cold start)."""
        rows = [("VA", i, 10 + i) for i in range(4)] + [("IL", i, 20 + i) for i in range(4)]
        for name in (_RESET, _OTHER, _PLAIN):
            self.l3.put(name, rows)
        publish_full_manifest(
            self.volume, term=1, tables={name: _entry(self.volume, name, rows, "inc:1") for name in _OPTED}
        )
        await self.follower.start()
        assert await self.follower.catch_up()
        assert self.same()
        self.l3.statements = dict.fromkeys(self.l3.statements, 0)

    def write(self, table: str, rows: list[tuple[str, int, int]]) -> None:
        """a commit: L3 changed, and for an opted-in table its delta file saved before the record names it."""
        self.l3.put(table, rows)
        delta_file = None
        if table in _OPTED:
            self.writes += 1
            delta = TableDelta(name=table, types=_TYPES, key=_KEY)
            delta.rows([dict(zip(_NAMES, row, strict=True)) for row in rows], list(_NAMES))
            delta_file = write_delta_file(self.volume, delta, f"w{self.writes}")
        self.log.commit(table, [(r[0], r[1]) for r in rows], delta_file=delta_file)
        self.lock.commit(*{f"state:{r[0]}" for r in rows})

    def copy(self, table: str) -> list[tuple[Any, ...]]:
        with self.backend.read_snapshot() as cursor:
            return sorted(cursor.execute(f"SELECT {', '.join(_NAMES)} FROM {table}").fetchall())  # noqa: S608

    def same(self, *tables: str) -> bool:
        return all(self.copy(name) == self.l3.held(name) for name in tables or (_RESET, _OTHER, _PLAIN))

    async def reloaded(self, table: str) -> None:
        for _ in range(500):
            if table not in self.follower.status().waiting_for_full:
                return
            await asyncio.sleep(0.01)
        raise AssertionError(f"{table} was never reloaded: {self.follower.status().waiting_for_full}")


_AFTER_RESET = [("VA", i, 100 + i) for i in range(3)] + [("TX", 1, 500)]


async def test_a_running_copy_hearing_a_reset_of_an_opted_in_table_reads_no_l3_and_reloads_it_from_the_full(
    _volume: Path,
) -> None:
    """Flips if a running copy reads an opted-in table from L3 when it hears the table reset (the old ``_on_reset``
    read it whole from L3: decisions 12 and 14 say the writer rebuilds it once and copies wait); if it takes the full
    from before the reset; if it stops following the other tables while it waits; or if, once the writer publishes the
    reset table's full, it does not reload that table alone from it and apply the later deltas from their files,
    ending equal to L3."""
    world = _World(_volume)
    await world.started()
    old = world.copy(_RESET)
    world.l3.reset(_RESET, _AFTER_RESET)
    reset_at = world.log.reset(_RESET)
    world.lock.commit("state:VA", "state:IL", "state:TX")
    await world.follower.catch_up()
    # the reset heard: nothing read L3, the table waits for its full, and the copy serves it as it held it
    assert world.l3.statements[_RESET] == 0
    assert _RESET in world.follower.status().waiting_for_full
    assert world.copy(_RESET) == old
    await asyncio.sleep(0.05)  # several looks at the volume: the full from before the reset is not taken
    assert "from before the reset at " + reset_at in world.follower.status().waiting_for_full[_RESET]
    assert world.copy(_RESET) == old
    # writes go on: the other table is followed, and the reset table's change waits for its full
    world.write(_OTHER, [("VA", 0, 77), ("IL", 9, 9)])
    world.write(_RESET, [("VA", 1, 222)])
    assert await world.follower.catch_up()
    assert world.same(_OTHER, _PLAIN)
    assert world.copy(_RESET) == old
    # the writer's rebuild, at the reset: L3 as the reset left it, before the later write
    publish_full_manifest(
        _volume,
        term=2,
        tables={
            _RESET: _entry(_volume, _RESET, _AFTER_RESET, reset_at),
            _OTHER: _entry(_volume, _OTHER, world.l3.held(_OTHER), f"inc:{world.log.count[_OTHER]}"),
        },
    )
    await world.reloaded(_RESET)
    assert await world.follower.catch_up()
    assert world.same()
    assert ("VA", 1, 222) in world.copy(_RESET)
    assert world.l3.statements[_RESET] == 0
    assert world.l3.statements[_OTHER] == 0
    assert world.registry.generation_marks.recorded(_RESET) == f"inc:{world.log.count[_RESET]}"


async def test_a_full_holding_the_table_from_before_the_reset_is_not_taken(_volume: Path) -> None:
    """Flips if a waiting copy takes a full that holds the reset table at a generation before the reset (it would load
    the table the reset replaced): a newer manifest naming an older version of the table is waited past."""
    world = _World(_volume)
    await world.started()
    old = world.copy(_RESET)
    world.l3.reset(_RESET, _AFTER_RESET)
    reset_at = world.log.reset(_RESET)
    world.lock.commit("state:VA")
    await world.follower.catch_up()
    # a newer manifest (term 2), the reset table in it still at the version before the reset
    publish_full_manifest(
        _volume, term=2, tables={name: _entry(_volume, name, world.l3.held(_OTHER), "inc:1") for name in _OPTED}
    )
    await asyncio.sleep(0.05)
    assert "from before the reset" in world.follower.status().waiting_for_full[_RESET]
    assert world.copy(_RESET) == old
    publish_full_manifest(
        _volume,
        term=2,
        tables={
            _RESET: _entry(_volume, _RESET, _AFTER_RESET, reset_at),
            _OTHER: _entry(_volume, _OTHER, world.l3.held(_OTHER), "inc:1"),
        },
    )
    await world.reloaded(_RESET)
    assert await world.follower.catch_up()
    assert world.same(_RESET)
    assert world.l3.statements[_RESET] == 0


async def test_a_schema_change_of_an_opted_in_table_waits_for_its_full_and_reads_no_l3(_volume: Path) -> None:
    """Flips if a schema change of an opted-in table reads it whole from L3 instead of waiting for the writer's
    rebuild, as a reset does; or if, told no generation for it, the copy takes a full holding the table at a
    generation it had already heard (from before the change)."""
    world = _World(_volume)
    await world.started()
    old = world.copy(_RESET)
    world.l3.reset(_RESET, _AFTER_RESET)
    world.follower.schema_changed(_RESET)
    assert await world.follower.catch_up()
    assert _RESET in world.follower.status().waiting_for_full
    await asyncio.sleep(0.05)  # several looks: the full at inc:1, heard before the change, is not taken
    assert world.copy(_RESET) == old
    # the recreated table's first write, then the writer's rebuild of it
    world.write(_RESET, [("VA", 1, 222)])
    assert await world.follower.catch_up()
    assert world.copy(_RESET) == old
    publish_full_manifest(
        _volume,
        term=2,
        tables={
            _RESET: _entry(_volume, _RESET, world.l3.held(_RESET), f"inc:{world.log.count[_RESET]}"),
            _OTHER: _entry(_volume, _OTHER, world.l3.held(_OTHER), "inc:1"),
        },
    )
    await world.reloaded(_RESET)
    assert await world.follower.catch_up()
    assert world.same()
    assert world.l3.statements[_RESET] == 0


async def test_a_table_not_opted_in_still_reads_l3_whole_on_a_reset(_volume: Path) -> None:
    """Flips if a table that does not opt in to delta files stops being read again whole from L3 on a reset (its one
    way back), or waits for a full it never has."""
    world = _World(_volume)
    await world.started()
    world.l3.reset(_PLAIN, _AFTER_RESET)
    world.log.reset(_PLAIN)
    world.lock.commit("state:VA", "state:IL", "state:TX")
    assert await world.follower.catch_up()
    assert _PLAIN not in world.follower.status().waiting_for_full
    assert world.same(_PLAIN)
    assert world.l3.statements[_PLAIN] >= 1
    assert world.l3.statements[_RESET] == 0
