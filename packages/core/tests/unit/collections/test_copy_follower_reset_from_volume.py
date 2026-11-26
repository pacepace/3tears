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
import threading
import time
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
from threetears.core.cache.duckdb import DuckDBBackend, StagedChange  # noqa: E402
from threetears.core.cache.duckdb_binds import run_bound  # noqa: E402
from threetears.core.collections import CollectionRegistry  # noqa: E402
import threetears.core.collections.copy_follower as copy_follower  # noqa: E402
from threetears.core.collections.change_notes import NotedKeysPage  # noqa: E402
from threetears.core.collections.copy_checkpoint import Checkpoint, CopyCheckpoints  # noqa: E402
from packages.core.tests.copy_workers_support import COPY_WORKERS  # noqa: E402
from threetears.core.collections.copy_key_work import INLINE_KEYS  # noqa: E402
from threetears.core.collections.copy_follower import (  # noqa: E402
    CopyFollower,
    FollowedTable,
    PartialStamp,
    PreparesStaged,
    is_reload_stage,
)
from threetears.core.testing.published_copy import PublishedCopy, SharedCopy, copy_rows  # noqa: E402
from threetears.core.collections.copy_queries import CopyQueries  # noqa: E402
from threetears.core.collections.delta_copy_beats import CopyBeats  # noqa: E402
from threetears.core.collections.delta_files import TableDelta, write_delta_file  # noqa: E402
from threetears.core.coordination.lease import KVLease  # noqa: E402
from threetears.core.testing.calibrate import budget_problem  # noqa: E402
from threetears.core.testing.kv import FakeNatsClient  # noqa: E402
import threetears.core.collections.volume_fulls as volume_fulls  # noqa: E402
from threetears.core.collections.epoch_change import (  # noqa: E402
    EVERY_STATE_NAMED,
    RESET_REASON,
    ChangedRow,
    EpochChange,
    EpochWalk,
    walk_by_count,
)
from threetears.core.collections.schema_backed import BIGINT_TYPE, STRING_TYPE, Column, rail_arrow_types  # noqa: E402
from threetears.core.collections.volume_fulls import (  # noqa: E402
    FullPart,
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


def _upsert(db: Any, table: str, rows: Sequence[tuple[str, int, int]], key: Sequence[str]) -> None:
    """``rows`` (state, geo_id, votes) upserted into ``table`` in one statement, the last row of a key winning."""
    if not rows:
        return
    put = pa.table(
        {
            "state": pa.array([r[0] for r in rows], pa.string()),
            "geo_id": pa.array([r[1] for r in rows], pa.int64()),
            "votes": pa.array([r[2] for r in rows], pa.int64()),
            "at__": pa.array(range(len(rows)), pa.int64()),
        }
    )
    db.register("_put", put)
    try:
        db.execute(
            f"INSERT OR REPLACE INTO {table} (state, geo_id, votes) SELECT state, geo_id, votes FROM _put "  # noqa: S608
            f"QUALIFY row_number() OVER (PARTITION BY {', '.join(key)} ORDER BY at__ DESC) = 1"
        )
    finally:
        db.unregister("_put")


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
        # one statement over the rows as an Arrow table, the last of a key winning as row-by-row upserts left it: a
        # statement a row binds each value from Python (a failed ``import pandas`` a value, under testament a stat of
        # every path), which held the loop past the loop-stall guard's limit (threetears.core.testing.loop_stall)
        _upsert(self.db, table, rows, ("state", "geo_id"))

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

    def unfiled(self, table: str, prefixes: list[dict[str, str]]) -> str:
        """an older writer's record of a load's key prefixes, naming no delta file."""
        token = self._next(table)
        self.records[(table, token)] = EpochChange(table=table, token=token, whole_table=True, prefixes=prefixes)
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
    def __init__(
        self,
        volume: Path,
        *,
        beats: CopyBeats | None = None,
        fulls: VolumeFulls | None = None,
        recreate: Any = None,
        shared: bool = False,
    ) -> None:
        """``shared``: the copy is the copy's worker backend (``copy_stages.SharedCopyStages``: each change applied in
        its apply worker, a reload staged in its reload worker), not the in-process DuckDB published as shared Arrow."""
        self.volume = volume
        self.l3, self.log, self.lock = _L3(), _Log(), _Scopes()
        tables = [_followed(_RESET), _followed(_OTHER), _followed(_PLAIN)]
        self.shared: SharedCopy | None = None
        self.backend: Any
        self.published: Any
        if shared:
            self.shared = SharedCopy(tables)
            self.backend = self.published = self.shared.copy
            queries = self.shared.queries
        else:
            self.backend = DuckDBBackend()
            _OPENED.append(self.backend.reset)
            self.backend.initialize(_metadata())
            self.published = PublishedCopy(self.backend)
            queries = self.published.queries
        self.registry = CollectionRegistry()
        for name in (_RESET, _OTHER, _PLAIN):
            self.registry.follow_generation(name)
        self.follower = CopyFollower(
            self.registry,
            tables,
            self.published,
            self.l3,
            settled=self.lock.settled,
            generation=self.log.current,
            changes=self.log,
            page_size=50,
            stamp_table="scope_epochs",
            fulls=fulls or VolumeFulls(volume, [_followed(_RESET), _followed(_OTHER)], poll_seconds=0.01),
            beats=beats,
            recreate=recreate,
            file_poll_seconds=0.01 if recreate is not None else None,
            drift_workers=COPY_WORKERS,
            queries=queries,
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
        if self.shared is not None:
            return copy_rows(self.shared.store, table, _NAMES)
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


def _too_big(
    world: _World, table: str, rows: list[tuple[str, int, int]], gone: list[tuple[str, int]], scopes: Any
) -> str:
    """a commit too big to list (decision 21): L3 changed, its record keys unknown naming its delta file, which is not
    on the volume (a fold that deletes took it, or it was lost), and the scopes its writer named; its epoch."""
    world.l3.put(table, rows)
    for key in gone:
        world.l3.db.execute(f"DELETE FROM {table} WHERE state = ? AND geo_id = ?", list(key))  # noqa: S608
    world.log.count[table] += 1
    token = f"{world.log.incarnation[table]}:{world.log.count[table]}"
    world.log.records[(table, token)] = EpochChange(
        table=table,
        token=token,
        keys_unknown=True,
        noted_between=(datetime.now(UTC), datetime.now(UTC)),
        delta_file=f"{table}/gone{world.log.count[table]}.arrow",
        scopes=scopes,
    )
    world.lock.commit(*{f"state:{row[0]}" for row in rows}, *{f"state:{key[0]}" for key in gone})
    return token


def _parts_read(monkeypatch: pytest.MonkeyPatch) -> list[str | None]:
    """the state of every full's part a reload or start reads off the volume, in order."""
    read: list[str | None] = []
    real = volume_fulls.read_full_part

    def reading(root: Path, part: FullPart) -> Any:
        read.append(part.scope)
        return real(root, part)

    monkeypatch.setattr(volume_fulls, "read_full_part", reading)
    return read


async def _covered(table: str, file: str, epoch: str) -> str:
    """the writer's answer for a file a fold that deletes took: the current full covers it (decision 54)."""
    return "covered"


async def _held_past_a_gone_file(
    world: _World, scopes: Any, read: list[str | None]
) -> tuple[str, list[tuple[str, int, int]]]:
    """the copy started; a commit too big to list changes VA and IL keeps, its file gone; then, while the table waits,
    a VA write the full will hold, an IL write and a VA write it will not. The writer's full at the first VA write:
    VA as L3 held it then, IL as at the start (its later write is not in it), every part at that version. Returns the
    lost commit's epoch and the full's VA rows. ``read`` (:func:`_parts_read`) is emptied once the copy started."""
    await world.started()
    read.clear()
    epoch = _too_big(world, _RESET, [("VA", 0, 300), ("VA", 1, 301), ("VA", 9, 309)], [("VA", 3)], scopes)
    await world.follower.catch_up()
    assert _RESET in world.follower.status().waiting_for_full
    world.write(_RESET, [("VA", 2, 302)])
    assert await world.follower.catch_up()
    covered_at = f"inc:{world.log.count[_RESET]}"
    va = [row for row in world.l3.held(_RESET) if row[0] == "VA"]
    il = [row for row in world.l3.held(_RESET) if row[0] == "IL"]
    world.write(_RESET, [("IL", 2, 402)])
    world.write(_RESET, [("VA", 1, 501)])
    assert await world.follower.catch_up()
    parts = tuple(
        write_full_part(world.volume, _RESET, f"state:{state}", _part_rows(rows), covered_at)
        for state, rows in (("IL", il), ("VA", va))
    )
    publish_full_manifest(
        world.volume,
        term=2,
        tables={
            _RESET: FullTable(version=covered_at, parts=parts),
            _OTHER: _entry(world.volume, _OTHER, world.l3.held(_OTHER), f"inc:{world.log.count[_OTHER]}"),
        },
    )
    return epoch, va


@pytest.mark.parametrize("shared", [False, True], ids=["duckdb", "worker_backend"])
async def test_a_gone_covered_file_reloads_only_its_commits_states_from_the_full_and_reads_no_l3(
    _volume: Path, monkeypatch: pytest.MonkeyPatch, shared: bool
) -> None:
    """Flips if a copy finding a covered delta file gone (the writer answers ``covered``: a fold that deletes took it,
    decision 54) reloads the whole table from the full, reads L3, or stops serving: its commit's record named every
    state it wrote (VA, marked complete), so only VA's part is loaded (decision 54 (c)). VA's rows the copy held and
    the full does not leave; the VA change heard while it waited that the part holds is not applied over it, while
    the later VA change and the IL change apply from their files, ending equal to L3 with no file still waited for."""
    world = _World(_volume, recreate=_covered, shared=shared)
    read = _parts_read(monkeypatch)
    await _held_past_a_gone_file(world, ["state:VA", EVERY_STATE_NAMED], read)

    await world.reloaded(_RESET)
    assert await world.follower.catch_up()

    assert read == ["state:VA"], "only the commit's state's part is read off the volume"
    assert world.same()
    assert world.l3.statements[_RESET] == 0
    status = world.follower.status()
    assert _RESET not in status.waiting_for_file and _RESET not in status.waiting_for_full
    assert status.full_reach == {}
    world.follower.stop()


async def test_a_lost_file_of_a_commit_too_big_to_list_reloads_only_its_named_states_from_the_recovered_full(
    _volume: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Flips if a copy whose commit too big to list lost its file (the writer recovers it, decision 28) reloads the
    whole table from the writer's full: the record named VA complete, so VA's part alone is loaded."""
    world = _World(_volume)
    read = _parts_read(monkeypatch)
    await _held_past_a_gone_file(world, ["state:VA", EVERY_STATE_NAMED], read)

    await world.reloaded(_RESET)
    assert await world.follower.catch_up()

    assert read == ["state:VA"]
    assert world.same()
    assert world.l3.statements[_RESET] == 0
    assert _RESET not in world.follower.status().waiting_for_file
    world.follower.stop()


async def test_a_gone_files_record_whose_states_are_not_known_complete_reloads_the_whole_table(
    _volume: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Flips if a copy reloads only the states a record names when the record does not mark them every state its
    commit wrote: such a record (an older writer's, a table declaring no state scope) may name a strict subset, and
    the copy would keep the old rows of a state it missed. Every part is loaded, and the copy ends equal to L3."""
    world = _World(_volume, recreate=_covered)
    read = _parts_read(monkeypatch)
    await _held_past_a_gone_file(world, ["state:VA"], read)

    await world.reloaded(_RESET)
    assert await world.follower.catch_up()

    assert sorted(read, key=str) == ["state:IL", "state:VA"]
    assert world.same()
    assert world.l3.statements[_RESET] == 0
    world.follower.stop()


async def test_a_failed_swap_of_some_states_leaves_the_copy_as_it_was_and_drops_both_its_stages(
    _volume: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Flips if a reload of some states whose swap fails changes the copy, or leaves either of its stages behind: the
    keys it staged (the copy's keys of the states, which leave) as well as its rows (once only the rows were dropped).
    Looked for again, the reload then applies."""
    world = _World(_volume, recreate=_covered)
    await _held_past_a_gone_file(world, ["state:VA", EVERY_STATE_NAMED], [])
    held = world.copy(_RESET)
    real_apply, real_drop = world.backend.apply_staged, world.backend.drop_stage
    failed: list[Any] = []
    dropped: list[str] = []
    release = threading.Event()

    def apply(changes: Any, **kwargs: Any) -> int:
        if any(is_reload_stage(change.rows) for change in changes):
            if not failed:
                failed.append(list(changes))
                raise RuntimeError("the swap failed")
            release.wait(5)
        return int(real_apply(changes, **kwargs))

    def drop(*stages: str) -> None:
        if failed and not dropped:
            # the first drop after the failed swap: its own, before the reload is looked for again
            dropped.extend(stages)
        real_drop(*stages)

    monkeypatch.setattr(world.backend, "apply_staged", apply)
    monkeypatch.setattr(world.backend, "drop_stage", drop)
    for _ in range(500):
        if failed and dropped:
            break
        await asyncio.sleep(0.01)
    try:
        assert failed, "the reload never swapped"
        change = failed[0][0]
        assert change.keys is not None and not change.whole, "a reload of some states stages the keys that leave"
        assert {change.keys, change.rows} <= set(dropped), "both of the failed swap's stages are dropped"
        assert world.copy(_RESET) == held
        assert _RESET in world.follower.status().waiting_for_full
    finally:
        release.set()
    await world.reloaded(_RESET)
    assert await world.follower.catch_up()
    assert world.same()
    world.follower.stop()


async def test_a_table_whose_full_is_not_split_by_state_is_reloaded_whole(
    _volume: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Flips if a copy whose wait names a covered file's states complete fails, or loads nothing, when the full holds
    the table as one part of no state (StatesNotSplitError): it reloads the table whole, as before, and ends equal to
    L3."""
    world = _World(_volume, recreate=_covered)
    await world.started()
    _too_big(world, _RESET, [("VA", 0, 300)], [], ["state:VA", EVERY_STATE_NAMED])
    await world.follower.catch_up()
    version = f"inc:{world.log.count[_RESET]}"
    publish_full_manifest(
        _volume,
        term=2,
        tables={
            _RESET: FullTable(
                version=version,
                parts=(write_full_part(_volume, _RESET, None, _part_rows(world.l3.held(_RESET)), version),),
            ),
            _OTHER: _entry(_volume, _OTHER, world.l3.held(_OTHER), f"inc:{world.log.count[_OTHER]}"),
        },
    )
    read = _parts_read(monkeypatch)

    await world.reloaded(_RESET)
    assert await world.follower.catch_up()

    assert read == [None]
    assert world.same()
    assert _RESET not in world.follower.status().waiting_for_file
    world.follower.stop()


class _BeatSeenFulls(VolumeFulls):
    """the volume's fulls, reading the copies' beats as the copy stages its first: what the fold sees while it loads."""

    def __init__(self, volume: Path, reader: CopyBeats) -> None:
        super().__init__(volume, [_followed(_RESET), _followed(_OTHER)], poll_seconds=0.01)
        self.reader = reader
        self.seen: list[str] = []

    async def stage(self, *args: Any, **kwargs: Any) -> Any:
        if not self.seen:
            self.seen.append((await self.reader.positions()).judge(_OTHER, "inc:1"))
        return await super().stage(*args, **kwargs)


async def test_a_copy_with_a_beat_says_starting_while_it_loads_then_beats_its_marks_and_deletes_it_when_it_ends(
    _volume: Path,
) -> None:
    """Flips if a copy given a beat (a writer whose fold deletes its delta files, decision 54) loads anything before it
    registers ``starting`` (the fold would delete files the loading copy needs), does not beat the marks it holds once
    started and again after a change it applies, or leaves its beat behind when its run ends (the fold would keep
    every file until the recovery's stale time)."""
    nats = FakeNatsClient()
    reader = CopyBeats(KVLease(nats, pod_id="reports"))
    fulls = _BeatSeenFulls(_volume, reader)
    world = _World(_volume, beats=CopyBeats(KVLease(nats, pod_id="reports"), copy_id="copy-a"), fulls=fulls)

    async def beat_of(table: str, epoch: str) -> None:
        for _ in range(500):
            beats = (await reader.positions()).beats
            if beats and not beats[0].starting and beats[0].epochs.get(table) == epoch:
                return
            await asyncio.sleep(0.01)
        raise AssertionError(f"no beat of {table} at {epoch}: {(await reader.positions()).beats}")

    await world.started()
    assert fulls.seen == ["starting"], "the copy loaded its full before its starting beat was put"
    await beat_of(_OTHER, "inc:1")
    run = asyncio.create_task(world.follower.run())
    world.write(_OTHER, [("VA", 0, 77)])
    world.follower.poke()
    await beat_of(_OTHER, f"inc:{world.log.count[_OTHER]}")
    run.cancel()
    await asyncio.gather(run, return_exceptions=True)
    assert (await reader.positions()).beats == (), "a copy's run ended and its beat is still there"


_AFTER_RESET = [("VA", i, 100 + i) for i in range(3)] + [("TX", 1, 500)]


@pytest.mark.parametrize("shared", [False, True], ids=["duckdb", "worker_backend"])
async def test_a_running_copy_hearing_a_reset_of_an_opted_in_table_reads_no_l3_and_reloads_it_from_the_full(
    _volume: Path, shared: bool
) -> None:
    """Flips if a running copy reads an opted-in table from L3 when it hears the table reset (the old ``_on_reset``
    read it whole from L3: decisions 12 and 14 say the writer rebuilds it once and copies wait); if it takes the full
    from before the reset; if it stops following the other tables while it waits; or if, once the writer publishes the
    reset table's full, it does not reload that table alone from it and apply the later deltas from their files,
    ending equal to L3. On the worker backend the reload stages in its reload worker and publishes through its apply
    lane (``SharedCopyStages.prepare_staged``)."""
    world = _World(_volume, shared=shared)
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
    if world.shared is not None:
        # the reload staged in the reload worker and published through the apply lane, once
        assert world.shared.copy.reloads.status().reloads == 1


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


class _SlowFull:
    """the backend's staging of ``results``' rows held on a worker thread until released: a big table's full staging.
    The loop is free meanwhile (the hold is on the staging thread), so a pass may run while it stages."""

    def __init__(self, world: _World, monkeypatch: pytest.MonkeyPatch) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        stage_rows = world.backend.stage_rows

        def staged(stage: str, table: str, rows: Any, key: Any, **kwargs: Any) -> int:
            if table == _RESET and rows.num_rows:
                self.entered.set()
                self.release.wait(timeout=30)
            return int(stage_rows(stage, table, rows, key, **kwargs))

        monkeypatch.setattr(world.backend, "stage_rows", staged)

    async def staging(self) -> None:
        for _ in range(1000):
            if self.entered.is_set():
                return
            await asyncio.sleep(0.01)
        raise AssertionError("the reset table's full never began to stage")


async def _waiting_for_full(world: _World) -> list[tuple[Any, ...]]:
    """``results`` reset and waiting for its full; the copy it serves meanwhile."""
    old = world.copy(_RESET)
    world.l3.reset(_RESET, _AFTER_RESET)
    world.log.reset(_RESET)
    world.lock.commit("state:VA", "state:IL", "state:TX")
    await world.follower.catch_up()
    assert _RESET in world.follower.status().waiting_for_full
    return old


def _publish_reset_full(world: _World, term: int) -> None:
    reset_at = f"inc:{world.log.count[_RESET]}"
    publish_full_manifest(
        world.volume,
        term=term,
        tables={
            _RESET: _entry(world.volume, _RESET, world.l3.held(_RESET), reset_at),
            _OTHER: _entry(world.volume, _OTHER, world.l3.held(_OTHER), f"inc:{world.log.count[_OTHER]}"),
        },
    )


async def test_another_tables_change_applies_while_a_reset_table_stages_its_full(
    _volume: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Flips if a reload holds the pass lock while it stages its table's full (decision 52): the old reload staged
    and applied under the lock, so every other table's ready change waited for the whole reload (a local burst
    replay: commit to publish a median 9.4 s, up to 45 s, behind one). Now the full stages outside the lock: while it
    stages, another table's change is applied and published within the bound, and the reload then swaps in alone."""
    world = _World(_volume)
    await world.started()
    old = await _waiting_for_full(world)
    slow = _SlowFull(world, monkeypatch)
    published: list[Any] = []
    world.follower.on_change(published.append)
    _publish_reset_full(world, term=2)
    try:
        await slow.staging()
        world.write(_OTHER, [("VA", 0, 4242), ("IL", 7, 77)])
        began = time.monotonic()
        applied = await asyncio.wait_for(world.follower.catch_up(), timeout=5.0)
        took = time.monotonic() - began
        assert applied, "the other table's pass did not apply while the reset table's full staged"
        assert took < 2.0, f"the other table's change waited {took:.2f} s for the reload"
        assert world.same(_OTHER), "the other table's change was not applied while the full staged"
        assert published, "the other table's change was not published while the full staged"
        assert world.copy(_RESET) == old, "the reset table changed before its full swapped in"
        assert _RESET in world.follower.status().waiting_for_full
    finally:
        slow.release.set()
    await world.reloaded(_RESET)
    assert await world.follower.catch_up()
    assert world.same()
    assert world.l3.statements[_RESET] == 0


@pytest.mark.parametrize("shared", [False, True], ids=["duckdb", "worker_backend"])
async def test_a_change_to_the_reset_table_heard_while_its_full_stages_is_applied_after_the_swap(
    _volume: Path, monkeypatch: pytest.MonkeyPatch, shared: bool
) -> None:
    """Flips if a change to the reloading table heard while its full stages is lost, applied to the table the reset
    replaced, or waits with every other table for the reload (decision 52). While the full stages a pass runs (within
    the bound) and applies the other table's change, never the reset table's: its change stays held, in order, and
    once the full swaps in it is applied over it, none lost."""
    world = _World(_volume, shared=shared)
    await world.started()
    old = await _waiting_for_full(world)
    slow = _SlowFull(world, monkeypatch)
    _publish_reset_full(world, term=2)
    try:
        await slow.staging()
        world.write(_RESET, [("VA", 1, 9191), ("IL", 3, 33)])
        world.write(_OTHER, [("VA", 2, 5151)])
        world.write(_RESET, [("VA", 1, 9292)])
        assert await asyncio.wait_for(world.follower.catch_up(), timeout=5.0)
        assert world.same(_OTHER)
        assert world.copy(_RESET) == old, "a change heard while the full staged was applied to the replaced table"
    finally:
        slow.release.set()
    await world.reloaded(_RESET)
    assert await world.follower.catch_up()
    assert world.same(), "a change heard while the full staged was lost or applied out of order"
    assert ("VA", 1, 9292) in world.copy(_RESET)
    assert ("IL", 3, 33) in world.copy(_RESET)
    assert world.l3.statements[_RESET] == 0
    if world.shared is not None:
        assert world.shared.copy.reloads.status().reloads == 1


async def test_a_backend_that_prepares_a_reload_does_it_outside_the_pass_lock_on_the_reload_s_own_stage(
    _volume: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Flips if a backend's heavy work for a reload (``PreparesStaged.prepare_staged``: the pod's rewrite of the
    table's parts in a worker process) runs under the pass lock, or is handed anything but the reload's own stage
    (decision 52): while it prepares, another table's change applies; then the reload swaps in."""
    world = _World(_volume)
    await world.started()
    old = await _waiting_for_full(world)
    entered, release = threading.Event(), threading.Event()
    prepared: list[list[str | None]] = []

    async def prepare_staged(changes: Any) -> None:
        prepared.append([change.rows for change in changes])
        entered.set()
        await asyncio.to_thread(release.wait, 30)

    monkeypatch.setattr(world.published, "prepare_staged", prepare_staged, raising=False)
    assert isinstance(world.published, PreparesStaged)
    _publish_reset_full(world, term=2)
    try:
        for _ in range(1000):
            if entered.is_set():
                break
            await asyncio.sleep(0.01)
        assert entered.is_set(), "the backend was never asked to prepare the reload"
        assert len(prepared) == 1 and len(prepared[0]) == 1, "one reload, one stage"
        (stage,) = prepared[0]
        assert stage is not None and is_reload_stage(stage) and _RESET in stage
        world.write(_OTHER, [("VA", 3, 3131)])
        assert await asyncio.wait_for(world.follower.catch_up(), timeout=5.0)
        assert world.same(_OTHER)
        assert world.copy(_RESET) == old
    finally:
        release.set()
    await world.reloaded(_RESET)
    assert await world.follower.catch_up()
    assert world.same()


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
        left = int(unstage(stages, table, key, keys))
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
        count = int(stage_rows(stage, table, rows, key))
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
            if getattr(call, "__name__", "") == "_plan_ipc" and not told:
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
    looked_up = CopyQueries.lookup
    told: list[str] = []

    async def reading(self: CopyQueries, *args: Any, **kwargs: Any) -> Any:
        # the pass's first read of the copy (a lookup, in a query worker) is the one naming the cleared prefix's keys:
        # the reset heard on the loop while it reads
        if not told:
            told.append(_reset_now(world, _AFTER_RESET))
        return await looked_up(self, *args, **kwargs)

    monkeypatch.setattr(CopyQueries, "lookup", reading)
    await world.follower.catch_up()
    monkeypatch.setattr(CopyQueries, "lookup", looked_up)
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
        self.published = published = PublishedCopy(self.backend)
        self.follower = CopyFollower(
            self.registry,
            [_followed(_RESET), _followed(_OTHER), _followed(_PLAIN)],
            published,
            self.l3,
            settled=self.lock.settled,
            generation=self.log.current,
            changes=self.log,
            page_size=50,
            stamp_table="scope_epochs",
            checkpoints=self.checkpoints,
            fulls=VolumeFulls(volume, [_followed(_RESET), _followed(_OTHER)], poll_seconds=0.01),
            drift_workers=COPY_WORKERS,
            queries=published.queries,
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


async def _checkpointed_world(volume: Path) -> _Checkpointed:
    world = _Checkpointed(volume)
    await world.started()
    await world.follower.checkpointed()
    assert world.seq() == 1, "the first warm of plain made one due"
    return world


async def test_an_epoch_from_the_notes_on_a_table_the_copy_loads_from_the_volume_makes_no_checkpoint_due(
    _volume: Path,
) -> None:
    """Flips if an epoch taken from a keys-unknown bracket on a table whose full is on the volume makes a checkpoint
    due: no checkpoint holds that table, and a start loads it from the volume."""
    world = await _checkpointed_world(_volume)
    world.too_large(_RESET, datetime.now(UTC) - timedelta(minutes=3))  # past a quarter of the 10-minute fence

    assert await world.follower.catch_up()
    await world.follower.checkpointed()

    assert world.seq() == 1, "a notes epoch on a table loaded from the volume made a checkpoint due"
    world.follower.stop()


async def test_no_epoch_from_a_keys_unknown_bracket_makes_a_checkpoint_due_of_either_kind_of_table(
    _volume: Path,
) -> None:
    """Flips if an epoch taken from a keys-unknown bracket makes a checkpoint due by its age (the change-note ring's
    quarter-fence rule, until 0.95): the bracket is paged from the table's own versions, which cleanup keeps while the
    change log holds a record from then, so a start from an older checkpoint pages it again (append-only L3 chunk 7)."""
    world = await _checkpointed_world(_volume)
    world.too_large(_RESET, datetime.now(UTC) - timedelta(minutes=3))
    world.too_large(_PLAIN, datetime.now(UTC) - timedelta(minutes=3))

    assert await world.follower.catch_up()
    await world.follower.checkpointed()

    assert world.seq() == 1, "an epoch from a bracket three minutes old made a checkpoint due"
    world.follower.stop()


async def test_a_reset_tables_reload_and_its_next_large_pass_judge_their_keys_in_the_worker_process(
    _volume: Path,
) -> None:
    """Flips if a reload's keys heard while the table waited are judged against the full's version (``_past_floor``),
    or the next large pass of the reloaded table plans its epochs, merges its files, cuts its stage or finds each
    state's newest epoch, in the serving process: on a thread they took 27 ms, 7.8 ms, 98 ms, 6.3 ms and 51 ms of its
    CPU at 100,000 keys, and 396 ms, 77 ms, 64 ms and 649 ms at 1,000,000 (the merge 210 ms at 200,000; 2026-11-25).
    The copy still ends equal to L3, reading none of it from L3."""
    world = _World(_volume)
    await world.started()
    world.l3.reset(_RESET, _AFTER_RESET)
    reset_at = world.log.reset(_RESET)
    world.lock.commit("state:VA", "state:IL", "state:TX")
    await world.follower.catch_up()
    assert _RESET in world.follower.status().waiting_for_full
    count = INLINE_KEYS
    # heard while it waits: past the full, so applied from their files once it is reloaded
    await asyncio.to_thread(world.write, _RESET, [("VA", 1_000 + i, i) for i in range(count)])
    assert await world.follower.catch_up()
    publish_full_manifest(
        _volume,
        term=2,
        tables={
            _RESET: _entry(_volume, _RESET, _AFTER_RESET, reset_at),
            _OTHER: _entry(_volume, _OTHER, world.l3.held(_OTHER), f"inc:{world.log.count[_OTHER]}"),
        },
    )
    COPY_WORKERS.jobs.clear()
    await world.reloaded(_RESET)
    assert await world.follower.catch_up()
    assert await asyncio.to_thread(world.same)
    assert "past_floor_job" in COPY_WORKERS.jobs, f"judged in the serving process: {COPY_WORKERS.jobs}"
    COPY_WORKERS.jobs.clear()
    await asyncio.to_thread(world.write, _RESET, [("IL", 2_000 + i, i) for i in range(count)])
    assert await world.follower.catch_up()
    assert await asyncio.to_thread(world.same)
    assert {"epoch_plan_job", "merged_job", "staged_scopes_job", "newest_by_scope_job"} <= set(COPY_WORKERS.jobs), (
        f"judged in the serving process: {COPY_WORKERS.jobs}"
    )
    assert world.l3.statements[_RESET] == 0


async def test_a_reload_of_some_states_judges_the_keys_heard_meanwhile_in_the_worker_process(_volume: Path) -> None:
    """Flips if a reload of some states (decision 54 (c)) judges the keys heard while the table waited against each
    state's part in the serving process: on a thread it took 77 ms of its CPU at 100,000 keys, 878 ms at 1,000,000 and
    3.0 ms at 2,000 with 25 states (a pass over the keys a state, 2026-11-25), so it runs in the worker at any size.
    Only the commit's state's part is loaded, and the copy ends equal to L3."""
    world = _World(_volume, recreate=_covered)
    await _held_past_a_gone_file(world, ["state:VA", EVERY_STATE_NAMED], [])
    COPY_WORKERS.jobs.clear()
    await world.reloaded(_RESET)
    assert await world.follower.catch_up()
    assert world.same()
    assert "past_states_job" in COPY_WORKERS.jobs, f"judged in the serving process: {COPY_WORKERS.jobs}"
    world.follower.stop()


async def test_the_scopes_a_waiting_tables_many_keys_heard_name_are_found_in_the_worker_process(
    _volume: Path,
) -> None:
    """Flips if the scopes a table waiting for a delta file holds back, read off its keys heard meanwhile, are found on
    the event loop past the bound: there they took 8.6-9.3 ms at 100,000 keys heard and 41-91 ms at 1,000,000
    (2026-11-25). The label still keeps each such scope held, the one the lost file named and the one heard since."""
    world = _World(_volume)
    await world.started()
    world.write(_RESET, [("VA", 0, 900)])
    lost = world.log.records[(_RESET, f"inc:{world.log.count[_RESET]}")].delta_file
    assert lost is not None
    (_volume / lost).unlink()
    await world.follower.catch_up()
    assert lost in world.follower.status().waiting_for_file[_RESET]
    await asyncio.to_thread(world.write, _RESET, [("TX", i, i) for i in range(INLINE_KEYS)])
    COPY_WORKERS.jobs.clear()
    await world.follower.catch_up()
    assert "heard_scopes_job" in COPY_WORKERS.jobs, f"found in the serving process: {COPY_WORKERS.jobs}"
    stamp = world.follower.status().stamp
    assert isinstance(stamp, PartialStamp), f"labelled {stamp!r} while a table waits"
    assert {"state:VA", "state:TX"} <= stamp.held


# --- every whole reload counted by why, a recovery said at WARNING (Pace, 2026-11-27) --------------------------------

#: every why ``collection_whole_reloads_total`` counts
_RELOAD_WHYS = (
    "cold_start",
    "reset",
    "schema_change",
    "lost_log",
    "lost_file",
    "lost_writes",
    "held_past_cap",
    "partition_refused",
)


def _reloads(table: str = _RESET) -> dict[str, float]:
    """``collection_whole_reloads_total`` of ``table``, by why."""
    return {
        why: REGISTRY.get_sample_value("collection_whole_reloads_total", {"table": table, "why": why}) or 0.0
        for why in _RELOAD_WHYS
    }


def _counted(before: dict[str, float], table: str = _RESET) -> dict[str, float]:
    """what ``collection_whole_reloads_total`` of ``table`` counted since ``before``, by why (none left out)."""
    return {why: n - before[why] for why, n in _reloads(table).items() if n != before[why]}


def _said_at_warning(caplog: pytest.LogCaptureFixture, why: str) -> list[str]:
    """the tables each WARNING line of a whole reload for ``why`` names."""
    return [
        record.extra_data["table"]  # type: ignore[attr-defined]
        for record in caplog.records
        if record.levelname == "WARNING"
        and getattr(record, "extra_data", {}).get("metric") == "collection_whole_reloads_total"
        and record.extra_data.get("why") == why  # type: ignore[attr-defined]
    ]


async def test_a_copys_start_from_the_volume_counts_each_opted_in_table_as_a_cold_start_reload(_volume: Path) -> None:
    """Flips if a copy's start from the writer's full on the volume does not count each opted-in table it loads under
    ``collection_whole_reloads_total{why="cold_start"}``, counts one that is not opted in (read from L3 instead), or
    counts it under any other why."""
    before = {name: _reloads(name) for name in (_RESET, _OTHER, _PLAIN)}
    world = _World(_volume)
    await world.started()
    assert _counted(before[_RESET], _RESET) == {"cold_start": 1}
    assert _counted(before[_OTHER], _OTHER) == {"cold_start": 1}
    assert _counted(before[_PLAIN], _PLAIN) == {}
    world.follower.stop()


async def test_a_reset_and_a_schema_change_of_an_opted_in_table_are_each_counted_once_by_their_own_why(
    _volume: Path,
) -> None:
    """Flips if a reset or a schema change of an opted-in table (reloaded from the writer's full, never L3) is not
    counted under its own why, is counted as a recovery, or a second reset while the first waits counts a second
    reload (one full answers both)."""
    world = _World(_volume)
    await world.started()
    before = _reloads()
    world.l3.reset(_RESET, _AFTER_RESET)
    world.log.reset(_RESET)
    world.lock.commit("state:VA", "state:IL", "state:TX")
    await world.follower.catch_up()
    world.log.reset(_RESET)
    world.lock.commit("state:VA")
    await world.follower.catch_up()
    assert _counted(before) == {"reset": 1}

    other = _reloads(_OTHER)
    world.follower.schema_changed(_OTHER)
    assert await world.follower.catch_up()
    assert _OTHER in world.follower.status().waiting_for_full
    assert _counted(other, _OTHER) == {"schema_change": 1}
    world.follower.stop()


async def test_a_lost_epoch_log_is_counted_as_a_lost_log_reload_and_said_at_warning(
    _volume: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Flips if a copy that can no longer walk an opted-in table's epoch log (replaced under a new incarnation, or a
    record gone) waits for the writer's recovered full without counting it under ``why="lost_log"`` or saying so at
    WARNING with the table: a recovery reload is never silent, and a replay fails on one."""
    world = _World(_volume)
    await world.started()
    before, other = _reloads(), _reloads(_OTHER)
    world.l3.put(_RESET, [("IL", 2, 222)])
    world.log.lose(_RESET, "new")
    with caplog.at_level("WARNING"):
        await world.follower.catch_up()
    assert _RESET in world.follower.status().waiting_for_full
    assert _counted(before) == {"lost_log": 1}
    assert _said_at_warning(caplog, "lost_log") == [_RESET]

    world.write(_OTHER, [("VA", 1, 111)])
    world.write(_OTHER, [("VA", 1, 112)])
    del world.log.records[(_OTHER, f"inc:{world.log.count[_OTHER]}")]
    await world.follower.catch_up()
    assert _OTHER in world.follower.status().waiting_for_full
    assert _counted(other, _OTHER) == {"lost_log": 1}
    world.follower.stop()


@pytest.mark.parametrize("answer", [None, "covered"], ids=["lost", "covered"])
async def test_a_lost_delta_file_is_counted_once_as_a_lost_file_reload_and_said_at_warning(
    _volume: Path, caplog: pytest.LogCaptureFixture, answer: str | None
) -> None:
    """Flips if a copy whose delta file of a commit too big to list never comes (lost, or swept by a fold because the
    full covers it: the writer answers ``covered``) waits for the full without counting it under ``why="lost_file"``
    exactly once or saying so at WARNING with the table."""

    async def answered(table: str, file: str, epoch: str) -> str:
        return "covered"

    world = _World(_volume, recreate=answered if answer == "covered" else None)
    await world.started()
    before = _reloads()
    with caplog.at_level("WARNING"):
        _too_big(world, _RESET, [("VA", 0, 300)], [], ["state:VA", EVERY_STATE_NAMED])
        await world.follower.catch_up()
        await asyncio.sleep(0.05)  # several looks at the file, and the writer's answer
    assert _RESET in world.follower.status().waiting_for_full
    assert _counted(before) == {"lost_file": 1}
    assert _said_at_warning(caplog, "lost_file") == [_RESET]
    world.follower.stop()


async def test_a_table_holding_past_its_cap_while_a_file_waits_is_counted_as_a_held_past_cap_reload(
    _volume: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Flips if a table that stops holding changes past its cap while waiting for a delta file, and waits for the
    writer's next full instead, is not counted under ``why="held_past_cap"``."""
    monkeypatch.setattr(copy_follower, "HELD_WHILE_WAITING_CAP", 2)
    world = _World(_volume)
    await world.started()
    before = _reloads()
    world.write(_RESET, [("VA", 0, 900)])
    lost = world.log.records[(_RESET, f"inc:{world.log.count[_RESET]}")].delta_file
    assert lost is not None
    (_volume / lost).unlink()
    await world.follower.catch_up()
    for geo in range(1, 4):
        world.write(_RESET, [("VA", geo, 900 + geo)])
        await world.follower.catch_up()
    assert "held_past_cap" in world.follower.status().waiting_for_full[_RESET]
    assert _counted(before) == {"held_past_cap": 1}
    world.follower.stop()


async def test_a_refused_prefix_record_naming_no_file_is_counted_as_a_partition_refused_reload(_volume: Path) -> None:
    """Flips if a copy refusing an opted-in table's prefix record naming no delta file (an older writer's), waiting
    for the writer's full instead, does not count it under ``why="partition_refused"``."""
    world = _World(_volume)
    await world.started()
    before = _reloads()
    world.log.unfiled(_RESET, [{"state": "VA"}])
    world.lock.commit("state:VA")
    await world.follower.catch_up()
    assert _RESET in world.follower.status().waiting_for_full
    assert _counted(before) == {"partition_refused": 1}
    world.follower.stop()


# --- the reload from the volume: reached only while a table waits, within its budget (Pace 2026-11-27) --------------


async def test_a_reload_from_the_volume_is_reached_only_while_a_table_waits_for_its_full(
    _volume: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """the reload from the volume (CopyFollower._reload_when_published's looked, calling ``_reload_from_volume``: a
    reset, a schema change, or a recovery from a lost epoch log or delta file) is reached only while a table waits for
    the writer's full, never on a normal pass, however many changes apply from their files; and its whole apply
    (DuckDBBackend.apply_staged, emptying the copy's table under kind recovery) is the reload's alone. Flips if a pass
    reloads a table that waits for nothing, a wait is not answered by a reload, or a whole change of kind recovery is
    applied by anything but the reload."""
    world = _World(_volume)
    await world.started()
    # each whole change the copy's apply_staged is handed from here on, as its table and kind: a reload's whole apply
    # is the one of kind recovery, the front door the reload ends at
    applied: list[tuple[str, str | None]] = []
    real_apply = world.backend.apply_staged

    def apply_staged(changes: Sequence[StagedChange], *, mark: int | None = None) -> Any:
        applied.extend((change.table, change.whole_kind) for change in changes if change.whole)
        return real_apply(changes, mark=mark)

    monkeypatch.setattr(world.backend, "apply_staged", apply_staged)
    for geo in range(5):
        world.write(_RESET, [("VA", geo, 500 + geo)])
        world.write(_OTHER, [("IL", geo, 600 + geo)])
        assert await world.follower.catch_up()
    await asyncio.sleep(0.05)  # several polls of the volume's own pace
    assert not world.follower.status().waiting_for_full, "a normal pass left a table waiting for a full"
    assert applied == [], "a normal pass reloaded a table from the volume, or applied anything whole"
    world.l3.reset(_RESET, _AFTER_RESET)
    reset_at = world.log.reset(_RESET)
    world.lock.commit("state:VA", "state:IL", "state:TX")
    await world.follower.catch_up()
    publish_full_manifest(
        _volume,
        term=2,
        tables={
            _RESET: _entry(_volume, _RESET, _AFTER_RESET, reset_at),
            _OTHER: _entry(_volume, _OTHER, world.l3.held(_OTHER), f"inc:{world.log.count[_OTHER]}"),
        },
    )
    await world.reloaded(_RESET)
    assert applied == [(_RESET, "recovery")], applied  # the reload's whole apply, and nothing else whole
    world.follower.stop()


async def test_a_reload_from_the_volume_keeps_its_budget(_volume: Path) -> None:
    """the reload from the volume (CopyFollower._reload_when_published's looked) at size: a reset table's reload
    of 100,000 rows from the writer's full on the volume (_reload_from_volume: its parts read and staged, the swap
    under the pass lock, the publish) within 2x of staging, applying and publishing the same full straight into a copy
    of its own that holds the table already (VolumeFulls.stage and apply_staged, the work a reload must do), plus the wait's own fixed cost (the volume's poll
    and the status look), measured on its own as a one-row reload."""
    states = [f"S{n:02d}" for n in range(50)]
    big = [(state, geo, geo) for state in states for geo in range(2000)]
    world = _World(_volume)
    await world.started()
    term = 1

    async def reload_of(rows: list[tuple[str, int, int]]) -> float:
        nonlocal term
        term += 1
        reset_at = world.log.reset(_RESET)
        world.lock.commit("state:VA")
        await world.follower.catch_up()
        # the writer's full written off the loop: building 100,000 rows' parts from Python is the test's, not the copy's
        tables = {
            _RESET: await asyncio.to_thread(_entry, _volume, _RESET, rows, reset_at),
            _OTHER: _entry(_volume, _OTHER, world.l3.held(_OTHER), f"inc:{world.log.count[_OTHER]}"),
        }
        began = time.perf_counter()
        publish_full_manifest(_volume, term=term, tables=tables)
        await world.reloaded(_RESET)
        return time.perf_counter() - began

    async def bare() -> float:
        backend = DuckDBBackend()
        _OPENED.append(backend.reset)
        backend.initialize(_metadata())
        copy = PublishedCopy(backend)
        fulls = VolumeFulls(_volume, [_followed(_RESET)], poll_seconds=0.01)
        # it holds the table already, as the reloading copy does: the timed apply replaces its rows, never fills
        filled = await fulls.stage(copy)
        assert not isinstance(filled, volume_fulls.FullAbsent), filled
        await copy.apply_staged(filled.changes, mark=1)
        began = time.perf_counter()
        found = await fulls.stage(copy)
        assert not isinstance(found, volume_fulls.FullAbsent), found
        await copy.apply_staged(found.changes, mark=2)  # its apply, and the publish of the state it holds
        return time.perf_counter() - began

    fixed = min([await reload_of([("VA", 0, 1)]) for _ in range(3)])
    took, baseline = [], []
    for _ in range(3):
        took.append(await reload_of(big))
        baseline.append(await bare())
    assert len(await asyncio.to_thread(world.copy, _RESET)) == len(big)
    problem = budget_problem(
        f"a reset table's reload of {len(big)} rows from the volume", min(took), min(baseline), plus=fixed
    )
    world.follower.stop()
    assert problem is None, problem
