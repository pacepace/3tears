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
- a table that does not opt in still reads L3 whole on a reset, as before;
- a copy that can no longer walk an opted-in table's epoch log (the log lost and replaced under a new incarnation, or a
  record of it missing) reads nothing from L3 either: it waits for the writer's recovered full (decision 28), never a
  full from before the loss, reloads the table from it without applying again a change the full already holds, and
  keeps following every other table meanwhile.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterator, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from sqlalchemy import BigInteger, Column as SaColumn, MetaData, String, Table
from threetears.core.testing.l1_tables import L1_EVICTION_INFO, TEST_L1_TABLE

pa = pytest.importorskip("pyarrow")
duckdb = pytest.importorskip("duckdb")

from prometheus_client import REGISTRY  # noqa: E402
from threetears.core.cache.duckdb import DuckDBBackend  # noqa: E402
from threetears.core.cache.duckdb_binds import run_bound  # noqa: E402
from threetears.core.collections import CollectionRegistry  # noqa: E402
import threetears.core.collections.copy_follower as copy_follower  # noqa: E402
from threetears.core.collections.change_notes import NotedKeysPage  # noqa: E402
from threetears.core.collections.copy_checkpoint import Checkpoint, CopyCheckpoints  # noqa: E402
from threetears.core.collections.copy_follower import CopyFollower, FollowedTable, PartialStamp  # noqa: E402
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
        self.incarnation: dict[str, str] = dict.fromkeys((_RESET, _OTHER, _PLAIN), "inc")
        self.records: dict[tuple[str, str], EpochChange] = {}

    async def current(self, table: str) -> str:
        return f"{self.incarnation[table]}:{self.count[table]}"

    def _next(self, table: str) -> str:
        self.count[table] += 1
        return f"{self.incarnation[table]}:{self.count[table]}"

    def lose(self, table: str, incarnation: str) -> str:
        """the table's log lost (every replica, or past its retention): its records gone, its generation started again
        under a new incarnation."""
        self.records = {key: change for key, change in self.records.items() if key[0] != table}
        self.incarnation[table] = incarnation
        self.count[table] = 1
        return f"{incarnation}:1"

    def commit(self, table: str, keys: list[tuple[str, int]], delta_file: str | None = None) -> str:
        token = self._next(table)
        self.records[(table, token)] = EpochChange(
            table=table, token=token, rows=[ChangedRow(ids=[s, str(g)]) for s, g in keys], delta_file=delta_file
        )
        return token

    def clear(self, table: str, prefixes: list[dict[str, str]], delta_file: str) -> str:
        """a key-prefix delete's record, naming its delta file (decision 23)."""
        token = self._next(table)
        self.records[(table, token)] = EpochChange(
            table=table, token=token, whole_table=True, prefixes=prefixes, delta_file=delta_file
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
    """the writer's record, settled: each scope's epoch. ``on_read`` (its read's number, a call) runs a call at that
    read, before it answers: a commit landing between a pass's two reads of the stamp."""

    def __init__(self) -> None:
        self.version = 1
        self.epochs = {"state:VA": 1, "state:IL": 1}
        self.reads = 0
        self.on_read: tuple[int, Callable[[], None]] | None = None

    async def settled(self) -> Any:
        self.reads += 1
        if self.on_read is not None and self.on_read[0] == self.reads:
            self.on_read[1]()
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

    def clear(self, table: str, state: str) -> str:
        """a delete of one state's rows (a key prefix): L3's rows of it gone, its delta file saved before the record
        names it; the record's epoch."""
        self.writes += 1
        delta = TableDelta(name=table, types=_TYPES, key=_KEY)
        delta.clears([{"state": state}])
        delta_file = write_delta_file(self.volume, delta, f"w{self.writes}")
        self.l3.db.execute(f"DELETE FROM {table} WHERE state = ?", [state])  # noqa: S608
        token = self.log.clear(table, [{"state": state}], delta_file)
        self.lock.commit(f"state:{state}")
        return token

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


async def test_a_table_waiting_for_a_file_holds_changes_up_to_its_cap_then_waits_for_the_full(
    _volume: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Flips if a table waiting for a delta file holds every change heard meanwhile with no bound (each about 400 bytes,
    with no end if the file never comes), if the held count is not on its gauge, or if past the cap it reads anything
    from L3 rather than waiting for the writer's next full: past the cap it forgets what it held and the file, reloads
    from the full at or past the newest epoch heard, and ends equal to L3."""
    monkeypatch.setattr(copy_follower, "HELD_WHILE_WAITING_CAP", 6)
    world = _World(_volume)
    await world.started()
    world.write(_RESET, [("VA", 0, 900)])
    lost = world.log.records[(_RESET, f"inc:{world.log.count[_RESET]}")].delta_file
    assert lost is not None
    (_volume / lost).unlink()
    await world.follower.catch_up()
    assert lost in world.follower.status().waiting_for_file[_RESET]

    def held() -> float:
        return REGISTRY.get_sample_value("copy_follower_held_while_waiting_changes", {"table": _RESET}) or 0.0

    for geo in range(1, 7):
        world.write(_RESET, [("VA", geo, 900 + geo)])
        await world.follower.catch_up()
    assert held() == 6
    assert _RESET in world.follower.status().waiting_for_file

    world.write(_RESET, [("VA", 7, 907)])
    await world.follower.catch_up()
    assert held() == 0, "past the cap nothing is held"
    assert _RESET not in world.follower.status().waiting_for_file
    assert "held_past_cap" in world.follower.status().waiting_for_full[_RESET]

    newest = f"inc:{world.log.count[_RESET]}"
    publish_full_manifest(
        _volume,
        term=2,
        tables={
            _RESET: _entry(
                _volume, _RESET, [(state, geo, votes) for state, geo, votes in world.l3.held(_RESET)], newest
            )
        },
    )
    await world.reloaded(_RESET)
    assert await world.follower.catch_up()
    assert world.same(_RESET)
    assert world.l3.statements[_RESET] == 0


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


async def test_a_cleared_prefix_heard_while_a_reset_table_waits_is_held_for_its_full(_volume: Path) -> None:
    """Flips if a cleared prefix of an opted-in table (a slice delete's file) queued before a reset survives it (its
    keys would be named over the full, clearing the reset table's rows of that state), or if one heard while the table
    waits for its full is lost instead of held and applied over the full once it is reloaded: either way the copy then
    differs from L3."""
    world = _World(_volume)
    await world.started()
    world.clear(_RESET, "IL")
    reset_rows = [("VA", 0, 100), ("VA", 1, 101), ("IL", 7, 700)]
    world.l3.reset(_RESET, reset_rows)
    reset_at = world.log.reset(_RESET)
    world.lock.commit("state:VA", "state:IL")
    await world.follower.catch_up()
    assert _RESET in world.follower.status().waiting_for_full
    # a delete of VA while the table waits: held, never named against the old copy
    world.clear(_RESET, "VA")
    await world.follower.catch_up()
    assert _RESET in world.follower.status().waiting_for_full
    publish_full_manifest(
        _volume,
        term=2,
        tables={
            _RESET: _entry(_volume, _RESET, reset_rows, reset_at),
            _OTHER: _entry(_volume, _OTHER, world.l3.held(_OTHER), f"inc:{world.log.count[_OTHER]}"),
        },
    )
    await world.reloaded(_RESET)
    assert await world.follower.catch_up()
    assert world.same(_RESET), "a clear from before the reset was applied over the full, or the later one was lost"
    assert world.l3.statements[_RESET] == 0


async def test_a_copy_whose_log_was_replaced_reads_no_l3_and_reloads_the_table_from_the_recovered_full(
    _volume: Path,
) -> None:
    """Flips if a copy that finds an opted-in table's epoch log replaced under a new incarnation (every NATS replica
    lost) stops following the other tables, reads the table from L3 (by the change notes or whole), takes a full from
    before the loss, or does not reload the table from the writer's recovered full and walk the new log from it, ending
    equal to L3 (decision 28 of the delta-files design)."""
    world = _World(_volume)
    await world.started()
    world.write(_RESET, [("VA", 1, 111)])
    assert await world.follower.catch_up()
    held = world.copy(_RESET)
    # the log is lost while a commit lands: L3 changed, its delta file left on the volume, no record of it anywhere
    world.l3.put(_RESET, [("IL", 2, 222)])
    recovered_at = world.log.lose(_RESET, "new")
    await world.follower.catch_up()
    # nothing read L3; the table waits for the writer's recovery and is served as it was held
    assert world.l3.statements[_RESET] == 0
    assert _RESET in world.follower.status().waiting_for_full
    assert _RESET not in world.follower.status().behind
    assert world.copy(_RESET) == held
    await asyncio.sleep(0.05)  # several looks: the full from before the loss is never taken
    assert "incarnation new" in world.follower.status().waiting_for_full[_RESET]
    assert world.copy(_RESET) == held
    # every other table keeps being followed meanwhile
    world.write(_OTHER, [("VA", 0, 77), ("IL", 9, 9)])
    assert await world.follower.catch_up()
    assert world.same(_OTHER, _PLAIN)
    # the writer's recovery: the kept full and one recovery delta, published at the new log's generation; then a commit
    publish_full_manifest(
        _volume,
        term=2,
        tables={
            _RESET: _entry(_volume, _RESET, world.l3.held(_RESET), recovered_at),
            _OTHER: _entry(_volume, _OTHER, world.l3.held(_OTHER), f"inc:{world.log.count[_OTHER]}"),
        },
    )
    world.write(_RESET, [("VA", 3, 333)])
    await world.reloaded(_RESET)
    assert await world.follower.catch_up()
    assert world.same()
    assert world.l3.statements[_RESET] == 0
    assert world.l3.statements[_OTHER] == 0
    assert world.registry.generation_marks.recorded(_RESET) == f"new:{world.log.count[_RESET]}"


async def test_a_copy_missing_a_record_waits_for_the_recovered_full_and_applies_no_change_it_holds(
    _volume: Path,
) -> None:
    """Flips if a copy whose walk of an opted-in table meets a record the log no longer holds reads L3 or stops
    following the other tables; takes a full no newer than its mark; or, reloading from the recovered full, applies
    again a change it heard before the loss that the full already holds (an older value over the full's newer one)."""
    world = _World(_volume)
    await world.started()
    world.write(_RESET, [("VA", 1, 111)])
    # a second commit to the same row, its record lost: the walk applies the first and stops at the hole
    world.write(_RESET, [("VA", 1, 112)])
    lost = f"inc:{world.log.count[_RESET]}"
    del world.log.records[(_RESET, lost)]
    await world.follower.catch_up()
    assert world.l3.statements[_RESET] == 0
    assert _RESET in world.follower.status().waiting_for_full
    publish_full_manifest(  # a full at the mark, from before the hole: never taken
        _volume,
        term=2,
        tables={name: _entry(_volume, name, world.l3.held(name), "inc:1") for name in _OPTED},
    )
    world.write(_OTHER, [("IL", 9, 9)])
    assert await world.follower.catch_up()
    assert world.same(_OTHER, _PLAIN)
    await asyncio.sleep(0.05)
    assert _RESET in world.follower.status().waiting_for_full
    # the writer's recovery, at the generation past the hole
    publish_full_manifest(
        _volume,
        term=2,
        tables={
            _RESET: _entry(_volume, _RESET, world.l3.held(_RESET), lost),
            _OTHER: _entry(_volume, _OTHER, world.l3.held(_OTHER), f"inc:{world.log.count[_OTHER]}"),
        },
    )
    await world.reloaded(_RESET)
    world.write(_RESET, [("IL", 3, 333)])
    assert await world.follower.catch_up()
    assert world.same()
    assert ("VA", 1, 112) in world.copy(_RESET)
    assert world.l3.statements[_RESET] == 0


# --- a waiting table's scopes are labelled at the epochs the copy holds them at ---------------------------------------


def _held_at(world: _World, epochs: dict[str, int]) -> None:
    """the copy's label keeps every scope at ``epochs``: the rows a waiting table holds are not labelled current."""
    stamp = world.follower.status().stamp
    assert isinstance(stamp, PartialStamp), f"labelled {stamp!r} while a table waits"
    assert {scope: stamp.epoch(scope) for scope in epochs} == epochs
    assert set(epochs) <= stamp.held


async def test_a_table_waiting_for_its_full_after_a_reset_keeps_every_scope_at_the_epoch_the_copy_holds(
    _volume: Path,
) -> None:
    """Flips if, while a reset table waits for its full (its rows as the copy held them before the reset), the copy is
    labelled at the writer's new stamp: a reader would read the replaced rows as current. Once the full is reloaded the
    label is the writer's stamp again."""
    world = _World(_volume)
    await world.started()
    epochs = dict(world.lock.epochs)
    world.l3.reset(_RESET, _AFTER_RESET)
    reset_at = world.log.reset(_RESET)
    world.lock.commit("state:VA", "state:IL")
    await world.follower.catch_up()
    world.write(_OTHER, [("IL", 9, 9)])
    assert await world.follower.catch_up()
    assert world.same(_OTHER, _PLAIN)
    _held_at(world, epochs)
    publish_full_manifest(
        _volume,
        term=2,
        tables={
            _RESET: _entry(_volume, _RESET, _AFTER_RESET, reset_at),
            _OTHER: _entry(_volume, _OTHER, world.l3.held(_OTHER), f"inc:{world.log.count[_OTHER]}"),
        },
    )
    await world.reloaded(_RESET)
    world.lock.commit("state:IL")
    assert await world.follower.catch_up()
    stamp = world.follower.status().stamp
    assert not isinstance(stamp, PartialStamp) and stamp.epochs == world.lock.epochs


async def test_a_table_waiting_for_the_writers_recovery_of_a_lost_log_keeps_every_scope_at_the_epoch_the_copy_holds(
    _volume: Path,
) -> None:
    """Flips if, while a table whose log was lost waits for the writer's recovered full (decision 28), the copy is
    labelled at the writer's new stamp though the change the lost log named is not in its rows."""
    world = _World(_volume)
    await world.started()
    epochs = dict(world.lock.epochs)
    world.l3.put(_RESET, [("IL", 2, 222)])
    world.lock.commit("state:IL")
    recovered_at = world.log.lose(_RESET, "new")
    await world.follower.catch_up()
    assert _RESET in world.follower.status().waiting_for_full
    world.write(_OTHER, [("VA", 0, 77)])
    assert await world.follower.catch_up()
    assert world.same(_OTHER, _PLAIN)
    _held_at(world, epochs)
    publish_full_manifest(
        _volume,
        term=2,
        tables={
            _RESET: _entry(_volume, _RESET, world.l3.held(_RESET), recovered_at),
            _OTHER: _entry(_volume, _OTHER, world.l3.held(_OTHER), f"inc:{world.log.count[_OTHER]}"),
        },
    )
    await world.reloaded(_RESET)
    assert await world.follower.catch_up()
    assert world.same()
    stamp = world.follower.status().stamp
    assert not isinstance(stamp, PartialStamp) and stamp.epochs == world.lock.epochs


# --- the owner's races (decision 37) -------------------------------------------------------------------------------
#
# Each test below injects a message at an await of the work it names: a reset of the table heard while the owner
# awaits a thread, a file read or a record. Before decision 37 every such message changed the follower's state there
# and then, under the work that was still using it; now each is a message the owner judges once its work hands back.


def _reset_now(world: _World, rows: list[tuple[str, int, int]]) -> str:
    """an operator's reset of ``results``, told to the registry as its record tells it (``on_table_reset``)."""
    world.l3.reset(_RESET, rows)
    token = world.log.reset(_RESET)
    world.registry.drop_table(_RESET, reason="the test's reset", why="reset", token=token)
    return token


def _on_the_loop(call: Callable[[], Any]) -> Callable[[], Any]:
    """``call`` run on the event loop from a worker thread, the thread waiting until it ran: a message heard on the
    loop while the owner's work runs on a thread. Made on the loop."""
    loop = asyncio.get_running_loop()

    def run() -> Any:
        async def on_loop() -> Any:
            return call()

        return asyncio.run_coroutine_threadsafe(on_loop(), loop).result()

    return run


async def test_a_second_reset_heard_while_the_full_stages_raises_the_bar_and_never_leaves_the_older_full(
    _volume: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Flips if a reset heard while a waiting table's full stages is lost: the old reload applied the full it had
    judged, then forgot the newer reset's bar and its held changes, so the table stayed on the full from before the
    second reset for good. Now the owner judges the second reset before the apply: the staged full is dropped, the
    table waits for a full at or past the second reset, and reloads from that one, equal to L3."""
    world = _World(_volume)
    await world.started()
    first = [("VA", i, 300 + i) for i in range(3)]
    second = [("VA", i, 700 + i) for i in range(2)] + [("IL", 5, 55)]
    world.l3.reset(_RESET, first)
    first_at = world.log.reset(_RESET)
    world.lock.commit("state:VA", "state:IL")
    await world.follower.catch_up()
    assert _RESET in world.follower.status().waiting_for_full
    stage = VolumeFulls.stage
    reset: list[str] = []

    async def staged(self: VolumeFulls, *args: Any, **kwargs: Any) -> Any:
        found = await stage(self, *args, **kwargs)
        if not reset:
            reset.append(_reset_now(world, second))  # heard while the full of the first reset staged
        return found

    monkeypatch.setattr(VolumeFulls, "stage", staged)
    publish_full_manifest(
        _volume,
        term=2,
        tables={
            _RESET: _entry(_volume, _RESET, first, first_at),
            _OTHER: _entry(_volume, _OTHER, world.l3.held(_OTHER), f"inc:{world.log.count[_OTHER]}"),
        },
    )
    for _ in range(500):
        if reset:
            break
        await asyncio.sleep(0.01)
    await asyncio.sleep(0.05)
    assert reset, "the full was never staged"
    assert _RESET in world.follower.status().waiting_for_full, "the second reset's bar was lost"
    assert world.copy(_RESET) != first, "the full from before the second reset was applied"
    publish_full_manifest(
        _volume,
        term=3,
        tables={
            _RESET: _entry(_volume, _RESET, second, reset[0]),
            _OTHER: _entry(_volume, _OTHER, world.l3.held(_OTHER), f"inc:{world.log.count[_OTHER]}"),
        },
    )
    await world.reloaded(_RESET)
    assert await world.follower.catch_up()
    assert world.same()
    assert world.l3.statements[_RESET] == 0


async def test_a_reset_heard_while_a_pass_leaves_scopes_out_applies_every_other_table(
    _volume: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Flips if a reset heard while a pass takes the keys of a moved scope out of a stage fails the pass (the old
    ``_leave_out`` found the reset table's bookkeeping gone: a KeyError, read as the stages lost, so every table's stage
    was dropped and read again). Now the reset is judged before the apply: the reset table's stage is forgotten and
    it waits for its full, and every other table's change applies."""
    world = _World(_volume)
    await world.started()
    old = world.copy(_RESET)
    world.write(_RESET, [("VA", 0, 900), ("IL", 0, 901)])
    world.write(_OTHER, [("VA", 1, 902)])
    # IL commits again between the pass's two reads of the stamp: its keys leave the stage
    world.lock.on_read = (world.lock.reads + 2, lambda: world.lock.commit("state:IL"))
    unstage = world.backend.unstage_keys
    reset = _on_the_loop(lambda: _reset_now(world, _AFTER_RESET))

    def unstaged(stages: Any, table: str, key: Any, keys: Any) -> int:
        left = unstage(stages, table, key, keys)
        if table == _RESET:
            reset()
        return left

    monkeypatch.setattr(world.backend, "unstage_keys", unstaged)
    await world.follower.catch_up()
    assert world.same(_OTHER), "the other table's change was not applied"
    assert _RESET in world.follower.status().waiting_for_full
    assert world.copy(_RESET) == old, "the reset table was applied from its stage of the table the reset replaced"
    assert world.l3.statements[_RESET] == 0


async def test_a_reset_heard_while_a_pass_stages_the_table_never_applies_its_stage(
    _volume: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Flips if a reset heard while a pass stages a table lets the stage be applied (the old pass forgot the table's
    stage at the reset, then wrote it back once its read returned: rows of the table the reset replaced applied over
    the copy). Now the reset is judged before the apply: the stage is forgotten, never applied, and the table waits
    for its full; the other table's change applies."""
    world = _World(_volume)
    await world.started()
    old = world.copy(_RESET)
    world.write(_RESET, [("VA", 0, 900)])
    world.write(_OTHER, [("VA", 1, 902)])
    stage_rows = world.backend.stage_rows
    reset = _on_the_loop(lambda: _reset_now(world, _AFTER_RESET))

    def staged(stage: str, table: str, rows: Any, key: Any) -> int:
        count = stage_rows(stage, table, rows, key)
        if table == _RESET:
            reset()
        return count

    monkeypatch.setattr(world.backend, "stage_rows", staged)
    await world.follower.catch_up()
    assert world.copy(_RESET) == old, "the stage of the table the reset replaced was applied"
    assert _RESET in world.follower.status().waiting_for_full
    assert world.same(_OTHER)
    assert world.l3.statements[_RESET] == 0


async def test_a_reset_heard_while_a_pass_reads_a_missing_file_leaves_no_wait_on_the_replaced_table(
    _volume: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Flips if a reset heard while a pass looks up a change's record leaves the table waiting for a delta file of the
    table the reset replaced (the old pass judged the missing file after the reset had ended the table's waits). Now
    the reset is judged after the pass's look: the reset ends every wait for a file of the replaced table, and the
    table waits for its full alone."""
    world = _World(_volume)
    await world.started()
    world.write(_RESET, [("VA", 0, 900)])
    lost = world.log.records[(_RESET, f"inc:{world.log.count[_RESET]}")].delta_file
    assert lost is not None
    (_volume / lost).unlink()
    change = world.log.change
    told: list[str] = []
    looked: list[str] = []

    async def looked_up(table_name: str, token: str) -> Any:
        record = await change(table_name, token)
        if record is not None and record.delta_file == lost:
            looked.append(token)
            # the registry's walk looks the record up first; the follower's own look-up as it stages the table second
            if len(looked) == 2 and not told:
                told.append(_reset_now(world, _AFTER_RESET))
        return record

    monkeypatch.setattr(world.log, "change", looked_up)
    await world.follower.catch_up()
    assert told
    assert _RESET in world.follower.status().waiting_for_full
    assert _RESET not in world.follower.status().waiting_for_file, "a wait for a file of the replaced table was left"
    assert world.l3.statements[_RESET] == 0


#: keys of one commit past the old snapshot's run (2,000 a turn): its run let the loop answer a reset mid-snapshot
_PAST_A_RUN = 2_500


async def test_a_reset_heard_while_a_pass_takes_its_keys_epochs_reads_nothing_from_l3(
    _volume: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Flips if a reset heard while a pass snapshots its keys' epochs leaves the rest of them with no epoch (the old
    pass read each key's epochs from the live map in runs, letting the loop run between; the reset emptied the map, so
    every later key was read from L3 by key, ``no_epoch``). Now the pass works on the owner's immutable batches: the
    reset is judged before the apply, nothing reads L3, and the table waits for its full."""
    world = _World(_volume)
    await world.started()
    old = world.copy(_RESET)
    world.write(_RESET, [("VA", 100 + i, i) for i in range(_PAST_A_RUN)])
    real = asyncio
    told: list[str] = []

    class _Asyncio:
        """asyncio, but the first yield or thread of the pass's merge first hears the reset."""

        def __getattr__(self, name: str) -> Any:
            return getattr(real, name)

        async def sleep(self, delay: float, *args: Any) -> Any:
            if delay == 0 and not told:
                told.append(_reset_now(world, _AFTER_RESET))
            return await real.sleep(delay, *args)

        async def to_thread(self, call: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
            if getattr(call, "__name__", "") == "_epoch_plan" and not told:
                told.append(_reset_now(world, _AFTER_RESET))
            return await real.to_thread(call, *args, **kwargs)

    monkeypatch.setattr(copy_follower, "asyncio", _Asyncio())
    await world.follower.catch_up()
    assert told
    assert world.l3.statements[_RESET] == 0, "keys of the reset table were read from L3 by key"
    assert world.copy(_RESET) == old
    assert _RESET in world.follower.status().waiting_for_full


async def test_a_reset_heard_while_a_cleared_prefix_is_named_puts_no_key_of_the_old_table_back(
    _volume: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Flips if a reset heard while a pass names the keys a cleared prefix reaches writes the old table's keys back
    as pending (the old pass held the live map across its read of the copy, and filled it after the reset emptied it),
    so a table waiting for its full held keys of the table the reset replaced, and the pass applied them to it. Now the
    reset is judged after the naming, and forgets every key heard of the replaced table: the copy serves the table as
    it held it until the full is reloaded."""
    world = _World(_volume)
    await world.started()
    old = world.copy(_RESET)
    world.clear(_RESET, "IL")
    reset = _on_the_loop(lambda: _reset_now(world, _AFTER_RESET))
    snapshot = world.backend.read_snapshot
    told: list[str] = []

    def reading() -> Any:
        # the pass's first read of the copy is the one naming the cleared prefix's keys
        if not told:
            told.append(reset())
        return snapshot()

    monkeypatch.setattr(world.backend, "read_snapshot", reading)
    await world.follower.catch_up()
    monkeypatch.setattr(world.backend, "read_snapshot", snapshot)
    assert told
    assert _RESET in world.follower.status().waiting_for_full
    assert _RESET not in world.follower.status().pending, "keys of the replaced table were put back pending"
    assert world.copy(_RESET) == old, "keys of the replaced table were applied to the copy while it waits"
    assert world.l3.statements[_RESET] == 0


# --- an epoch taken from the change notes makes a checkpoint due only for a table no full on the volume holds --------


class _Notes:
    """the change notes as the broker's l3.noted_keys door answers them: one page of the keys given, for any table."""

    def __init__(self, keys: Sequence[tuple[str, ...]]) -> None:
        self.keys = tuple(keys)

    async def noted_keys(self, table: str, **_: Any) -> NotedKeysPage:
        return NotedKeysPage(keys=self.keys, whole_table=False, next=None)


class _Store:
    """an Object Store in memory."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    async def put(self, name: str, data: bytes) -> None:
        self.objects[name] = data

    async def get(self, name: str) -> bytes:
        return self.objects[name]

    async def require_room(self, objects: Sequence[tuple[str, int]]) -> int:
        return sum(size for _, size in objects)

    async def list_names(self, *, prefix: str = "") -> list[str]:
        return [n for n in self.objects if n.startswith(prefix)]

    async def retire(self, names: list[str]) -> None:
        for name in names:
            self.objects.pop(name, None)


class _Pointers:
    """a pointer bucket in memory: create-if-absent and compare-and-set by revision."""

    def __init__(self) -> None:
        self.entries: dict[str, tuple[bytes, int]] = {}
        self.revision = 0

    async def get_entry(self, *, key: str) -> tuple[bytes, int] | None:
        return self.entries.get(key)

    async def create(self, *, key: str, value: bytes, ttl: Any = None) -> int | None:
        if key in self.entries:
            return None
        self.revision += 1
        self.entries[key] = (value, self.revision)
        return self.revision

    async def update(self, *, key: str, value: bytes, revision: int, ttl: Any = None) -> int | None:
        held = self.entries.get(key)
        if held is None or held[1] != revision:
            return None
        self.revision += 1
        self.entries[key] = (value, self.revision)
        return self.revision

    async def delete(self, *, key: str, revision: int | None = None, marker_ttl: Any = None) -> bool:
        return self.entries.pop(key, None) is not None

    async def list_keys(self, *, prefix: str = "") -> list[str]:
        return [k for k in self.entries if k.startswith(prefix)]


class _Checkpointed(_World):
    """the world, its follower also keeping a checkpoint of ``plain`` (the opted-in tables' full is on the volume
    alone) and reading the change notes."""

    def __init__(self, volume: Path) -> None:
        super().__init__(volume)
        self.store, self.pointers = _Store(), _Pointers()
        self.checkpoints = CopyCheckpoints(
            name="noted",
            tables=[_followed(_PLAIN)],
            backend=self.backend,
            store=self.store,
            pointers=self.pointers,
            retire=self.store.retire,
            owner="a",
            encode_stamp=lambda stamp: {"version": stamp.version, "epochs": stamp.epochs},
            decode_stamp=lambda data: SimpleNamespace(**data, pending=frozenset()),
        )
        self.registry.set_noted_keys_reader(_Notes([("VA", "1")]))
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
            checkpoints=self.checkpoints,
            fulls=VolumeFulls(volume, [_followed(_RESET), _followed(_OTHER)], poll_seconds=0.01),
        )

    def seq(self) -> int:
        return Checkpoint.decode(next(iter(self.pointers.entries.values()))[0]).seq

    def too_large(self, table: str, since: datetime) -> None:
        """a commit too large to record its keys, its record naming no delta file: its keys are in the change notes,
        bracketed from ``since``."""
        self.l3.put(table, [("VA", 1, 999)])
        self.log.count[table] += 1
        token = f"{self.log.incarnation[table]}:{self.log.count[table]}"
        self.log.records[(table, token)] = EpochChange(
            table=table, token=token, keys_unknown=True, noted_between=(since, since + timedelta(seconds=1))
        )
        self.lock.commit("state:VA")

    def notes_age(self) -> float | None:
        return REGISTRY.get_sample_value("copy_checkpoint_notes_unsaved_age_seconds", {"base": self.checkpoints.base})


async def _checkpointed_world(volume: Path) -> _Checkpointed:
    world = _Checkpointed(volume)
    await world.started()
    await world.follower.checkpointed()
    assert world.seq() == 1, "the first warm of plain made one due"
    return world


async def test_an_epoch_from_the_notes_on_a_table_the_copy_loads_from_the_volume_makes_no_checkpoint_due(
    _volume: Path,
) -> None:
    """Flips if an epoch taken from the change notes on a table whose full is on the volume makes a checkpoint due or
    sets ``copy_checkpoint_notes_unsaved_age_seconds``: no checkpoint holds that table, and a start loads it from the
    volume and never needs its notes, so the old trigger wrote a checkpoint of the other tables for nothing and showed
    an age no checkpoint could end."""
    world = await _checkpointed_world(_volume)
    world.too_large(_RESET, datetime.now(UTC) - timedelta(minutes=3))  # past a quarter of the 10-minute fence

    assert await world.follower.catch_up()
    await world.follower.checkpointed()

    assert world.seq() == 1, "a notes epoch on a table loaded from the volume made a checkpoint due"
    assert world.checkpoints.noted_due_in() is None
    assert world.notes_age() == 0.0
    world.follower.stop()


async def test_with_both_kinds_of_table_only_an_epoch_from_the_notes_on_a_checkpointed_table_makes_one_due(
    _volume: Path,
) -> None:
    """Flips if the older notes epoch of a table loaded from the volume sets when a checkpoint is due, or if one of a
    table the checkpoint holds stops making it due: the trigger and its gauge follow the checkpointed table alone."""
    world = await _checkpointed_world(_volume)
    world.too_large(_RESET, datetime.now(UTC) - timedelta(minutes=3))
    world.too_large(_PLAIN, datetime.now(UTC) - timedelta(seconds=60))

    assert await world.follower.catch_up()
    await world.follower.checkpointed()

    assert world.seq() == 1, "the volume table's older epoch made the checkpoint due"
    age = world.notes_age()
    assert age is not None and 60 <= age < 120, "plain's epoch alone waits: a minute old, not yet due"

    world.too_large(_PLAIN, datetime.now(UTC) - timedelta(minutes=3))
    assert await world.follower.catch_up()
    await world.follower.checkpointed()

    assert world.seq() == 2, "plain's epoch a quarter fence old makes the checkpoint due"
    assert world.notes_age() == 0.0
    world.follower.stop()
