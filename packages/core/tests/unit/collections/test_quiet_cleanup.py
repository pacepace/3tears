"""quiet-time cleanup of versioned tables (``design-append-only-l3.md`` decision 5, build plan chunk 6).

Cleanup removes a key's versions superseded below the restore window's floor, a "gone" marker below it with every older
version of its key, and a run marker below it after its run's rows; it never touches a version at or above the floor,
which a restore could read. It runs only while nothing loads, and stops before its next batch when a load starts. Every
statement it sends is keyed.

The statements run on an in-process DuckDB (the keyed helpers' one spelling for PostgreSQL, YugabyteDB and DuckDB), so
what is removed is the database's answer, not a double's. The live check (``tests/integration/
test_quiet_cleanup_plans_live.py``) proves the plans on YugabyteDB.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

import duckdb
import pytest

from threetears.core.backends.broker_generation import CLEANUP_DELETE_FIELD, CLEANUP_FLOOR_FIELD, stamp_fields
from threetears.core.collections.cleanup_progress import CLEANUP_PROGRESS_TABLE
from threetears.core.collections.quiet_cleanup import LoadGate, QuietCleanup, log_window_floor
from threetears.core.collections.schema_backed import INT_TYPE, STRING_TYPE, Column, TableSchema
from threetears.core.collections.scope_epochs import WHOLE, EpochSnapshot, ScopeEpochs
from threetears.core.ddl.row_epoch import cleanup_floor, epoch_wall_ms, row_versions
from threetears.core.ddl.run_markers import RUN_MARKERS_TABLE
from threetears.core.keyed import ThreadedDuckDB
from threetears.core.testing.l1_tables import TEST_L1_TABLE
from threetears.enforcement.no_whole_reads.schema import SchemaCatalog, collect_schema
from threetears.enforcement.no_whole_reads.sql import Reader, judge_sql

#: a results table as reports declares one: its scope (the state) is not in its key
_RESULTS = TableSchema(
    name="cl_results",
    primary_key=("office_key", "candidate"),
    columns=[
        Column("office_key", STRING_TYPE),
        Column("candidate", STRING_TYPE),
        Column("join_state_code", STRING_TYPE),
        Column("votes", INT_TYPE, nullable=True),
    ],
    row_epoch=row_versions("join_state_code"),
    l1_eviction=TEST_L1_TABLE,
)

#: an LSF table: its key leads with the scope, then the run (a whole run deleted is one run marker)
_PROJECTIONS = TableSchema(
    name="cl_projections",
    primary_key=("join_state_code", "run_id", "geo_id"),
    columns=[
        Column("join_state_code", STRING_TYPE),
        Column("run_id", INT_TYPE),
        Column("geo_id", STRING_TYPE),
        Column("value", INT_TYPE, nullable=True),
    ],
    row_epoch=row_versions("join_state_code"),
    l1_eviction=TEST_L1_TABLE,
)

_MODULE = "packages/core/src/threetears/core/collections/quiet_cleanup.py"

#: the tables as a migration declares them, for the no-whole-reads judge: keys, and the purposed (scope, epoch, key)
#: indexes cleanup pages by
_DDL = f"""
CREATE TABLE cl_results (office_key TEXT, candidate TEXT, join_state_code TEXT, votes BIGINT, epoch BIGINT NOT NULL,
    gone BOOLEAN NOT NULL DEFAULT false, PRIMARY KEY (office_key, candidate, epoch));
-- purpose: {_MODULE}::QuietCleanup._candidates
CREATE INDEX cl_results_scope_epoch ON cl_results (join_state_code, epoch, office_key, candidate);
CREATE TABLE cl_projections (join_state_code TEXT, run_id BIGINT, geo_id TEXT, value BIGINT, epoch BIGINT NOT NULL,
    gone BOOLEAN NOT NULL DEFAULT false, PRIMARY KEY (join_state_code, run_id, geo_id, epoch));
-- purpose: {_MODULE}::QuietCleanup._candidates
CREATE INDEX cl_projections_scope_epoch ON cl_projections (join_state_code, epoch, run_id, geo_id);
CREATE TABLE {RUN_MARKERS_TABLE} (table_name TEXT, scope TEXT, prefix TEXT, epoch BIGINT,
    PRIMARY KEY (table_name, scope, prefix, epoch));
CREATE TABLE {CLEANUP_PROGRESS_TABLE} (table_name TEXT, scope TEXT, cleaned_to BIGINT, PRIMARY KEY (table_name, scope));
"""

#: an epoch a test writes at: its wall part far from 0, so a floor below it is a real one
_BASE = 1_000_000 << 15


def _at(offset: int) -> int:
    return _BASE + offset


class _Epochs:
    """the scope epochs cleanup listens to: a snapshot the test sets, and its change listeners."""

    def __init__(self, scopes: tuple[str, ...]) -> None:
        self.snap = EpochSnapshot(
            version=1, writing=None, epochs={WHOLE: 1, **{f"state:{s}": 1 for s in scopes}}, previous={}
        )
        self._listeners: list[Callable[[], None]] = []
        self.looks = 0

    def on_change(self, listener: Callable[[], None]) -> Callable[[], None]:
        self._listeners.append(listener)
        return lambda: self._listeners.remove(listener)

    async def snapshot(self) -> EpochSnapshot:
        self.looks += 1
        return self.snap

    def change(self, **fields: Any) -> None:
        """a write began, touched or committed: the snapshot moves and every listener hears it."""
        self.snap = replace(self.snap, **fields)
        for listener in list(self._listeners):
            listener()


class _Spy(ThreadedDuckDB):
    """the DuckDB cleanup runs on, recording every statement it is sent; ``after`` runs after each one."""

    def __init__(self, conn: Any) -> None:
        super().__init__(conn)
        self.statements: list[str] = []
        #: what each statement's request would say on the rail (stamp_fields, read as the proxy reads it)
        self.fields: list[dict[str, Any]] = []
        self.after: Callable[[str], None] | None = None

    async def answers_from_yugabyte(self) -> bool:
        return False

    def _seen(self, query: str) -> None:
        self.statements.append(query)
        self.fields.append(stamp_fields())
        if self.after is not None:
            self.after(query)

    async def fetch(self, query: str, /, *params: Any) -> list[dict[str, Any]]:
        answered = await super().fetch(query, *params)
        self._seen(query)
        return answered

    async def fetch_arrow(self, query: str, /, *params: Any, types: Any, size: int | None = None) -> Any:
        answered = await super().fetch_arrow(query, *params, types=types, size=size)
        self._seen(query)
        return answered

    async def execute(self, query: str, /, *params: Any) -> str:
        answered = await super().execute(query, *params)
        self._seen(query)
        return answered


@dataclass
class _World:
    conn: Any
    spy: _Spy
    epochs: _Epochs
    gate: LoadGate

    def cleanup(self, floor: int | None, *, batch: int = 500) -> QuietCleanup:
        async def window() -> int | None:
            return floor

        return QuietCleanup(self.spy, (_RESULTS, _PROJECTIONS), self.gate, window, scope_prefix="state:", batch=batch)

    def rows(self, table: str) -> set[tuple[Any, ...]]:
        columns = "office_key, candidate" if table == "cl_results" else "run_id, geo_id"
        return {tuple(row) for row in self.conn.execute(f"SELECT {columns}, epoch, gone FROM {table}").fetchall()}

    def markers(self) -> set[tuple[Any, ...]]:
        return {tuple(row) for row in self.conn.execute(f"SELECT prefix, epoch FROM {RUN_MARKERS_TABLE}").fetchall()}


@pytest.fixture
def world() -> _World:
    conn = duckdb.connect()
    for statement in _DDL.split(";"):
        if statement.strip():
            conn.execute("\n".join(line for line in statement.splitlines() if not line.startswith("--")))
    epochs = _Epochs(("NC", "VA"))
    gate = LoadGate(cast(ScopeEpochs, epochs))
    return _World(conn, _Spy(conn), epochs, gate)


def _result(conn: Any, key: tuple[str, str], state: str, epoch: int, *, gone: bool = False) -> None:
    conn.execute(
        "INSERT INTO cl_results VALUES (?, ?, ?, ?, ?, ?)",
        [key[0], key[1], state, None if gone else 1, epoch, gone],
    )


def _projection(conn: Any, run: int, geo: str, epoch: int) -> None:
    conn.execute("INSERT INTO cl_projections VALUES ('NC', ?, ?, 1, ?, false)", [run, geo, epoch])


# -- in any order: no key comes back, and everything removable goes -----------------------------------------------


@pytest.mark.parametrize(
    "floors",
    [(5, 15, 25, 35, 45), (45,), (25, 45), (15, 45), (35, 45), (5, 45)],
    ids=["oldest-first", "all-at-once", "middle-first", "early", "late", "nothing-then-all"],
)
async def test_whatever_order_versions_go_in_no_key_comes_back_and_all_go(
    world: _World, floors: tuple[int, ...]
) -> None:
    """a key whose newest version is a marker never shows an older live version, a key's newest live version never
    goes, and once the floor passes everything, each key holds its newest live version alone, or nothing."""
    _result(world.conn, ("o1", "a"), "NC", _at(10))
    _result(world.conn, ("o1", "a"), "NC", _at(20))
    _result(world.conn, ("o1", "a"), "NC", _at(30), gone=True)
    _result(world.conn, ("o2", "a"), "VA", _at(10))
    _result(world.conn, ("o2", "a"), "VA", _at(20), gone=True)
    _result(world.conn, ("o2", "a"), "VA", _at(40))
    for floor in floors:
        await world.cleanup(_at(floor), batch=1).clean()
        held = world.rows("cl_results")
        o1 = sorted(row for row in held if row[0] == "o1")
        # o1's newest is its marker: if any of it is left, the marker is too, so no older version reads as live
        assert not o1 or o1[-1] == ("o1", "a", _at(30), True)
        # o2's newest live version never goes
        assert ("o2", "a", _at(40), False) in held
    assert world.rows("cl_results") == {("o2", "a", _at(40), False)}


# -- the window: never a version a restore could read ----------------------------------------------------------------


@dataclass(frozen=True)
class _State:
    first_seq: int
    last_seq: int


@dataclass(frozen=True)
class _Record:
    s_min: int | None


class _Log:
    def __init__(self, state: _State | None, oldest: _Record | None) -> None:
        self._state, self._oldest = state, oldest
        self.asked: list[int] = []

    async def log_state(self) -> _State | None:
        return self._state

    async def epoch_at(self, seq: int) -> _Record | None:
        self.asked.append(seq)
        return self._oldest


async def test_the_floor_lies_below_every_restore_from_a_mark_the_log_still_holds() -> None:
    s_min = (1_700_000_000_000 << 15) | 7
    log = _Log(_State(first_seq=40, last_seq=90), _Record(s_min))
    floor = await log_window_floor(log)
    assert log.asked == [40]
    assert floor == cleanup_floor(s_min)
    # a restore from a mark taken at or after the oldest record's commit reads from (T - 10 min) << 15: the floor is
    # below it even for a writer clock S ahead of the mark's
    margin_ms = 10 * 60 * 1000
    earliest_mark_ms = epoch_wall_ms(s_min) - 5 * 1000
    assert floor is not None and floor <= (earliest_mark_ms - margin_ms) << 15


@pytest.mark.parametrize(
    "log",
    [
        _Log(None, None),
        _Log(_State(first_seq=1, last_seq=0), None),
        _Log(_State(first_seq=3, last_seq=9), _Record(None)),
        _Log(_State(first_seq=3, last_seq=9), None),
    ],
    ids=["no-log", "empty-log", "unstamped-oldest", "oldest-gone"],
)
async def test_an_unknown_window_removes_nothing(log: _Log, world: _World) -> None:
    _result(world.conn, ("o1", "a"), "NC", _at(1))
    _result(world.conn, ("o1", "a"), "NC", _at(2))
    before = world.rows("cl_results")
    assert await log_window_floor(log) is None
    swept = await world.cleanup(None).clean()
    assert (swept.removed, swept.floor) == (0, None)
    assert world.rows("cl_results") == before


async def test_cleanup_never_touches_a_version_at_or_above_the_floor(world: _World) -> None:
    floor = _at(100)
    # a key superseded below the floor, then again above it; a marker above it; a run marked gone above it
    _result(world.conn, ("o1", "a"), "NC", _at(10))
    _result(world.conn, ("o1", "a"), "NC", _at(100))
    _result(world.conn, ("o1", "a"), "NC", _at(150))
    _result(world.conn, ("o2", "a"), "VA", _at(20))
    _result(world.conn, ("o2", "a"), "VA", _at(120), gone=True)
    _projection(world.conn, 7, "g1", _at(30))
    world.conn.execute(f"INSERT INTO {RUN_MARKERS_TABLE} VALUES ('cl_projections', 'NC', '{'NC'}\x1f7', ?)", [_at(130)])
    restorable = {row for row in world.rows("cl_results") if row[2] >= floor}
    swept = await world.cleanup(floor).clean()
    assert swept.finished
    held = world.rows("cl_results")
    assert restorable <= held
    # below the floor, only what a version at or above it supersedes would go, and that waits for the floor to pass it
    assert ("o1", "a", _at(10), False) in held
    assert ("o2", "a", _at(20), False) in held
    assert world.rows("cl_projections") == {(7, "g1", _at(30), False)}
    assert world.markers() == {("NC\x1f7", _at(130))}


async def test_a_superseded_version_below_the_floor_goes_and_the_newest_stays(world: _World) -> None:
    for offset in (10, 20, 30):
        _result(world.conn, ("o1", "a"), "NC", _at(offset))
    swept = await world.cleanup(_at(100)).clean()
    assert swept.removed == 2
    assert world.rows("cl_results") == {("o1", "a", _at(30), False)}


async def test_a_key_is_removed_fully_after_its_marker_ages_out(world: _World) -> None:
    _result(world.conn, ("o1", "a"), "NC", _at(10))
    _result(world.conn, ("o1", "a"), "NC", _at(20))
    _result(world.conn, ("o1", "a"), "NC", _at(30), gone=True)
    _result(world.conn, ("o9", "a"), "NC", _at(10))
    # the marker still inside the window: only the version its newer one supersedes goes
    await world.cleanup(_at(25)).clean()
    assert world.rows("cl_results") == {
        ("o1", "a", _at(20), False),
        ("o1", "a", _at(30), True),
        ("o9", "a", _at(10), False),
    }
    # the marker aged out: the key goes whole, marker and every version
    await world.cleanup(_at(31)).clean()
    assert world.rows("cl_results") == {("o9", "a", _at(10), False)}


async def test_a_gone_run_goes_row_by_row_and_then_its_marker(world: _World) -> None:
    for geo in ("g1", "g2", "g3"):
        _projection(world.conn, 7, geo, _at(10))
    # the run written again after it was marked gone: that version is live
    _projection(world.conn, 7, "g1", _at(60))
    world.conn.execute(f"INSERT INTO {RUN_MARKERS_TABLE} VALUES ('cl_projections', 'NC', 'NC\x1f7', ?)", [_at(50)])
    swept = await world.cleanup(_at(100), batch=2).clean()
    assert swept.finished
    assert world.rows("cl_projections") == {(7, "g1", _at(60), False)}
    assert world.markers() == set()


async def test_a_reset_marker_below_the_floor_is_removed_and_one_above_it_kept(world: _World) -> None:
    """Flips if a table's reset marker (scope chr(30), empty prefix) below F is never removed (cleanup read real scope
    values alone), one at or above F (a restore may still read it) is removed, a run marker is taken for one, or a round
    whose floor has not moved reads the markers again."""
    reset = f"INSERT INTO {RUN_MARKERS_TABLE} VALUES (?, chr(30), '', ?)"
    world.conn.execute(reset, ["cl_results", _at(10)])
    world.conn.execute(reset, ["cl_results", _at(150)])
    world.conn.execute(reset, ["cl_projections", _at(20)])
    # a run marker above the floor: not a reset's, and not yet past F
    world.conn.execute(f"INSERT INTO {RUN_MARKERS_TABLE} VALUES ('cl_projections', 'NC', 'NC\x1f7', ?)", [_at(130)])
    swept = await world.cleanup(_at(100)).clean()
    assert swept.finished and swept.removed == 2
    held = {
        tuple(row) for row in world.conn.execute(f"SELECT table_name, scope, epoch FROM {RUN_MARKERS_TABLE}").fetchall()
    }
    assert held == {("cl_results", "\x1e", _at(150)), ("cl_projections", "NC", _at(130))}
    world.spy.statements.clear()
    again = await world.cleanup(_at(100)).clean()
    assert again.finished and again.removed == 0
    assert all(CLEANUP_PROGRESS_TABLE in query for query in world.spy.statements), world.spy.statements


# -- only while nothing loads -----------------------------------------------------------------------------------------


async def test_cleanup_does_nothing_while_a_load_is_on(world: _World) -> None:
    _result(world.conn, ("o1", "a"), "NC", _at(10))
    _result(world.conn, ("o1", "a"), "NC", _at(20))
    world.epochs.change(writing=2)
    swept = await world.cleanup(_at(100)).clean()
    assert swept.paused and swept.removed == 0
    assert world.spy.statements == []
    # a scope pending (touched, not committed) is a load too; one a commit withheld is not
    world.epochs.change(writing=None, pending=frozenset({"state:NC"}))
    assert (await world.cleanup(_at(100)).clean()).paused
    world.epochs.change(pending=frozenset({"state:NC"}), withheld=frozenset({"state:NC"}))
    assert (await world.cleanup(_at(100)).clean()).finished


async def test_cleanup_pauses_within_one_batch_when_a_load_starts_and_resumes_where_it_stopped(world: _World) -> None:
    for index in range(6):
        key = (f"o{index}", "a")
        _result(world.conn, key, "NC", _at(10 + index))
        _result(world.conn, key, "NC", _at(50 + index))
    cleanup = world.cleanup(_at(100), batch=2)

    def begin_after_first_delete(query: str) -> None:
        if query.lstrip().upper().startswith("DELETE"):
            world.spy.after = None
            world.epochs.change(writing=2)

    world.spy.after = begin_after_first_delete
    swept = await cleanup.clean()
    assert swept.paused and not swept.finished
    # the batch in flight finished (its delete was its last statement), and nothing was sent after it
    assert world.spy.statements[-1].lstrip().upper().startswith("DELETE")
    # the first delete is the batch visiting o0 and o1's newer versions: their older two go
    assert swept.removed == 2
    sent = len(world.spy.statements)
    # still loading: a look, and no statement
    assert (await cleanup.clean()).paused
    assert len(world.spy.statements) == sent
    # the load committed: cleanup resumes past the last version it visited, and finishes
    world.epochs.change(writing=None)
    resumed = await cleanup.clean()
    assert resumed.finished
    assert world.rows("cl_results") == {(f"o{index}", "a", _at(50 + index), False) for index in range(6)}
    assert swept.removed + resumed.removed == 6


async def test_a_round_whose_floor_has_not_moved_reads_no_data(world: _World) -> None:
    _result(world.conn, ("o1", "a"), "NC", _at(10))
    _result(world.conn, ("o1", "a"), "NC", _at(20))
    await world.cleanup(_at(100)).clean()
    world.spy.statements.clear()
    world.epochs.change(version=2)
    again = await world.cleanup(_at(100)).clean()
    assert again.finished and again.removed == 0
    assert all(CLEANUP_PROGRESS_TABLE in query for query in world.spy.statements), world.spy.statements


async def test_a_load_starting_is_heard_with_no_read(world: _World) -> None:
    await world.gate.look()
    looks = world.epochs.looks
    assert not world.gate.stirred
    world.epochs.change(writing=3)
    assert world.gate.stirred
    assert world.epochs.looks == looks


# -- keyed statements only --------------------------------------------------------------------------------------------


@pytest.fixture
def catalog(tmp_path: Path) -> SchemaCatalog:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "v001.sql").write_text(_DDL, encoding="utf-8")
    return collect_schema(tmp_path, (tmp_path / "src",))


async def test_every_statement_cleanup_sends_is_keyed(world: _World, catalog: SchemaCatalog) -> None:
    for index in range(5):
        _result(world.conn, (f"o{index}", "a"), "NC", _at(10 + index))
        _result(world.conn, (f"o{index}", "a"), "NC", _at(30 + index), gone=index % 2 == 0)
    for geo in ("g1", "g2", "g3"):
        _projection(world.conn, 7, geo, _at(10))
    world.conn.execute(f"INSERT INTO {RUN_MARKERS_TABLE} VALUES ('cl_projections', 'NC', 'NC\x1f7', ?)", [_at(50)])
    swept = await world.cleanup(_at(100), batch=2).clean()
    assert swept.finished and swept.removed
    names = frozenset(
        f"{_MODULE}::QuietCleanup.{name}" for name in ("_candidates", "_clear", "_clear_runs", "_cleaned_to", "_record")
    )
    reader = Reader(catalog, names)
    found = {query: judge_sql(query, reader) for query in world.spy.statements}
    # the judge is no rubber stamp: a delete by the epoch alone, or a filter off the key, is a whole read
    assert judge_sql("DELETE FROM cl_results WHERE epoch < $1", reader)
    assert judge_sql("SELECT * FROM cl_results WHERE votes = $1", reader)
    assert not {query: hits for query, hits in found.items() if hits}
    assert not any("TRUNCATE" in query.upper() for query in world.spy.statements)
    # the scope's versions are read through its (scope, epoch, key) range, a page at a time
    assert any('t."join_state_code" = ' in query and "LIMIT" in query.upper() for query in world.spy.statements)


# -- the broker can tell cleanup's deletes, and bound them ------------------------------------------------------------


async def test_every_delete_of_a_versioned_table_says_it_is_cleanups_with_its_floor(world: _World) -> None:
    """Flips if a delete of a versioned table goes out unmarked (the broker refuses every other delete of one), names
    a floor other than the one it was worked out against, or a point delete carries no epoch bound of its own; and if
    any other statement claims to be cleanup's."""
    floor = _at(100)
    for index in range(3):
        _result(world.conn, (f"o{index}", "a"), "NC", _at(10 + index))
        _result(world.conn, (f"o{index}", "a"), "NC", _at(30 + index))
    for geo in ("g1", "g2"):
        _projection(world.conn, 7, geo, _at(10))
    world.conn.execute(f"INSERT INTO {RUN_MARKERS_TABLE} VALUES ('cl_projections', 'NC', 'NC\x1f7', ?)", [_at(50)])
    swept = await world.cleanup(floor, batch=2).clean()
    assert swept.finished and swept.removed
    sent = list(zip(world.spy.statements, world.spy.fields, strict=True))
    versioned = [
        (query, fields)
        for query, fields in sent
        if "DELETE FROM" in query.upper()
        and any(f'"{table}"' in query.upper().split("WHERE")[0].lower() for table in ("cl_results", "cl_projections"))
    ]
    point = [query for query, _ in versioned if "LIMIT" not in query.upper()]
    runs = [query for query, _ in versioned if "LIMIT" in query.upper()]
    # superseded versions by their whole stored key, and a run's rows by its prefix
    assert point and runs, sent
    for query, fields in versioned:
        assert fields.get(CLEANUP_DELETE_FIELD) is True and fields.get(CLEANUP_FLOOR_FIELD) == floor, query
    for query in point:
        # the point delete bounds itself below F in its own WHERE, past the keys it names: no key can reach F
        assert re.search(r'\) AND \(?"epoch" < \$\d+\)?$', query.rstrip()), query
    others = [fields for query, fields in sent if (query, fields) not in versioned]
    assert others and not any(CLEANUP_DELETE_FIELD in fields for fields in others)
