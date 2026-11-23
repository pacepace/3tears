"""a copy follower applies a table that opts in to delta files from the file each epoch's record names, never L3.

Election night failed with L3 on the hot path: every pod copy read each change's rows back from L3
(``design-pod-delta-files.md``). The registry, the follower and the delta files are the real ones; L3 is a DuckDB that
counts every statement it answers, kept equal to what the writes would leave in L3, so "the copy equals L3" and "no
statement reached L3" are both checked. The contract this pins:

- an opted-in table's change (an update, an insert, a delete) is applied from its file with no L3 statement;
- an upsert that left a column out keeps the copy's value of it and the row's ``date_created`` (decision 19);
- a delete marker deletes the key;
- a record naming a file the volume does not hold reads that change from L3 by key, counted by why;
- a table that does not opt in reads L3 as before, counted as no fallback;
- a key a pass leaves out (its scope moved under the read) is staged again from its files by the next pass, every
  epoch of it in order, with no L3 read.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable, Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from sqlalchemy import BigInteger, Column as SaColumn, DateTime, MetaData, String, Table
from threetears.core.testing.l1_tables import L1_EVICTION_INFO, TEST_L1_TABLE

pa = pytest.importorskip("pyarrow")
duckdb = pytest.importorskip("duckdb")

from prometheus_client import REGISTRY  # noqa: E402
from threetears.core.cache.duckdb import DuckDBBackend  # noqa: E402
from threetears.core.cache.duckdb_binds import run_bound  # noqa: E402
from threetears.core.collections import CollectionRegistry  # noqa: E402
from threetears.core.collections.complete_copy import Unsettled  # noqa: E402
from threetears.core.collections.copy_follower import CopyFollower, FollowedTable, PartialStamp  # noqa: E402
from threetears.core.collections.delta_files import TableDelta, write_delta_file  # noqa: E402
from threetears.core.collections.epoch_change import ChangedRow, EpochChange, EpochWalk, walk_by_count  # noqa: E402
from threetears.core.collections.schema_backed import (  # noqa: E402
    BIGINT_TYPE,
    DATETIMETZ_TYPE,
    STRING_TYPE,
    Column,
    rail_arrow_types,
)

_TABLE = "results"
_KEY = ("state", "geo_id")
_COLUMNS = (
    Column(name="state", column_type=STRING_TYPE),
    Column(name="geo_id", column_type=BIGINT_TYPE),
    Column(name="votes", column_type=BIGINT_TYPE),
    Column(name="note", column_type=STRING_TYPE),
    Column(name="date_created", column_type=DATETIMETZ_TYPE),
)
_NAMES = tuple(column.name for column in _COLUMNS)
_TYPES = rail_arrow_types(_COLUMNS)

#: when the warm's rows were first inserted: the date_created an upsert of them keeps
_BORN = datetime(2026, 11, 3, 21, 45, tzinfo=UTC)
#: the date_created a later write carries for a row it inserts, and leaves out for a row L3 held
_LATER = datetime(2026, 11, 4, 3, 0, tzinfo=UTC)

_OPENED: list[Callable[[], None]] = []


@pytest.fixture(autouse=True)
def _close_what_the_test_opened() -> Iterator[None]:
    """close every DuckDB the test opened (each runs a worker thread a core; test_copy_follower has the account)."""
    yield
    while _OPENED:
        _OPENED.pop()()


@pytest.fixture(autouse=True)
def _volume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """the shared volume the delta files are saved on and read off."""
    monkeypatch.setenv("THREETEARS_DELTA_FILES_DIR", str(tmp_path))
    return tmp_path


def _metadata() -> MetaData:
    metadata = MetaData()
    Table(
        _TABLE,
        metadata,
        SaColumn("state", String(8), primary_key=True),
        SaColumn("geo_id", BigInteger, primary_key=True),
        SaColumn("votes", BigInteger),
        SaColumn("note", String(32)),
        SaColumn("date_created", DateTime(timezone=True)),
        info={L1_EVICTION_INFO: TEST_L1_TABLE},
    )
    return metadata


def _dict_rows(result: Any) -> list[dict[str, Any]]:
    names = [d[0] for d in result.description]
    return [dict(zip(names, row, strict=True)) for row in result.fetchall()]


class _L3:
    """L3 as a DuckDB answering ``fetch``; counts every statement that reads the table."""

    def __init__(self) -> None:
        self.db = duckdb.connect(":memory:")
        _OPENED.append(self.db.close)
        self.db.execute("SET TimeZone = 'UTC'")
        self.db.execute(
            f"CREATE TABLE {_TABLE} (state VARCHAR, geo_id BIGINT, votes BIGINT, note VARCHAR, "
            "date_created TIMESTAMPTZ, PRIMARY KEY (state, geo_id))"
        )
        self.statements = 0

    def insert(self, rows: Sequence[dict[str, Any]]) -> None:
        for row in rows:
            self.db.execute(
                f"INSERT OR REPLACE INTO {_TABLE} VALUES (?, ?, ?, ?, ?)",  # noqa: S608
                [row[name] for name in _NAMES],
            )

    def update(self, key: tuple[str, int], **values: Any) -> None:
        # an upsert of a row L3 held: only the columns its write set
        sets = ", ".join(f"{name} = ?" for name in values)
        self.db.execute(
            f"UPDATE {_TABLE} SET {sets} WHERE state = ? AND geo_id = ?",  # noqa: S608
            [*values.values(), *key],
        )

    def delete(self, key: tuple[str, int]) -> None:
        self.db.execute(f"DELETE FROM {_TABLE} WHERE state = ? AND geo_id = ?", list(key))  # noqa: S608

    def held(self) -> list[tuple[Any, ...]]:
        return sorted(self.db.execute(f"SELECT {', '.join(_NAMES)} FROM {_TABLE}").fetchall())  # noqa: S608

    async def fetch(self, sql: str, *params: Any) -> list[dict[str, Any]]:
        await asyncio.sleep(0)
        rows = run_bound(self.db.cursor(), sql, params, _dict_rows)
        if not sql.startswith("SELECT version()"):
            self.statements += 1
        return rows


class _Log:
    """the table's write generation and its change records, each naming the commit's delta file when it saved one."""

    def __init__(self) -> None:
        self.count = 0
        self.records: dict[str, EpochChange] = {}

    async def current(self, table: str) -> str:
        return f"inc:{self.count}"

    def commit(self, keys: Sequence[tuple[str, int]], delta_file: str | None = None) -> str:
        self.count += 1
        token = f"inc:{self.count}"
        self.records[token] = EpochChange(
            table=_TABLE, token=token, rows=[ChangedRow(ids=[s, str(g)]) for s, g in keys], delta_file=delta_file
        )
        return token

    async def change(self, table_name: str, token: str) -> EpochChange | None:
        return self.records.get(token)

    async def changes_after(self, table_name: str, after: str, upto: str) -> EpochWalk:
        return await walk_by_count(self, table_name, after, upto)


class _Scopes:
    """the writer's record as ScopeEpochs answers it: each scope's epoch, and the scopes a write in progress holds."""

    def __init__(self) -> None:
        self.version = 1
        self.epochs: dict[str, int] = {"state:VA": 1, "state:IL": 1}
        self.held: frozenset[str] | None = None
        self.on_read: Any = None
        self.reads = 0

    async def settled(self) -> Any:
        self.reads += 1
        if self.on_read is not None and self.on_read[0] == self.reads:
            self.on_read[1]()
        if self.held is None:
            return SimpleNamespace(version=self.version, epochs=dict(self.epochs), pending=frozenset())
        return Unsettled("a long write", epochs=dict(self.epochs), pending=self.held, version=self.version)

    def commit(self, *scopes: str) -> None:
        self.version += 1
        for scope in scopes:
            self.epochs[scope] = self.version


def _world(*, delta_files: bool = True) -> tuple[_L3, _Log, _Scopes, DuckDBBackend, CopyFollower]:
    l3, log, lock = _L3(), _Log(), _Scopes()
    backend = DuckDBBackend()
    _OPENED.append(backend.reset)
    backend.initialize(_metadata())
    registry = CollectionRegistry()
    registry.follow_generation(_TABLE)
    follower = CopyFollower(
        registry,
        [FollowedTable(name=_TABLE, key=_KEY, columns=_COLUMNS, scope=("state", "state:"), delta_files=delta_files)],
        backend,
        l3,
        settled=lock.settled,
        generation=log.current,
        changes=log,
        page_size=50,
        stamp_table="scope_epochs",
    )
    follower.register()
    return l3, log, lock, backend, follower


def _row(state: str, geo_id: int, votes: int, note: str | None = "first", born: datetime = _BORN) -> dict[str, Any]:
    return {"state": state, "geo_id": geo_id, "votes": votes, "note": note, "date_created": born}


def _copy(backend: DuckDBBackend) -> list[tuple[Any, ...]]:
    with backend.read_snapshot() as cursor:
        return sorted(cursor.execute(f"SELECT {', '.join(_NAMES)} FROM {_TABLE}").fetchall())  # noqa: S608


def _instant(value: Any) -> datetime | None:
    """an instant in UTC, as the copy keeps it (its ISO text) or L3 answers it (a datetime)."""
    if value is None:
        return None
    return (datetime.fromisoformat(value) if isinstance(value, str) else value).astimezone(UTC)


def _instants(rows: Iterable[tuple[Any, ...]]) -> list[tuple[Any, ...]]:
    """rows with their date_created as an instant: the copy keeps it as text, L3 answers a datetime."""
    return [(*row[:-1], _instant(row[-1])) for row in rows]


def _same(backend: DuckDBBackend, l3: _L3) -> bool:
    return _instants(_copy(backend)) == _instants(l3.held())


def _save(volume: Path, write: str, *parts: tuple[str, Any, Any]) -> str:
    """save a commit's delta file, as the writer does before its commit; its name, as the record names it."""
    delta = TableDelta(name=_TABLE, types=_TYPES, key=_KEY)
    for kind, part, written in parts:
        if kind == "rows":
            delta.rows(part, written)
        else:
            delta.deletes(part)
    return write_delta_file(volume, delta, write)


def _fallbacks(why: str) -> float:
    return (
        REGISTRY.get_sample_value("copy_follower_delta_file_fallback_keys_total", {"table": _TABLE, "why": why}) or 0.0
    )


async def _warmed(world: tuple[_L3, _Log, _Scopes, DuckDBBackend, CopyFollower]) -> None:
    l3, log, _lock, backend, follower = world
    rows = [_row("VA", i, 10 + i) for i in range(5)] + [_row("IL", i, 20 + i) for i in range(5)]
    l3.insert(rows)
    log.commit([(row["state"], row["geo_id"]) for row in rows])
    await follower.warm()
    assert await follower.catch_up()
    assert _same(backend, l3)
    l3.statements = 0


async def test_an_opted_in_tables_change_is_applied_from_its_file_with_no_l3_statement(_volume: Path) -> None:
    """Flips if a copy of an opted-in table reads a change's rows from L3: election night failed because every pod
    copy did, so L3's read load grew with the pods (design-pod-delta-files.md, decision 3)."""
    world = _world()
    l3, log, lock, backend, follower = world
    await _warmed(world)
    every = list(_NAMES)
    updated, inserted = _row("VA", 1, 99, "second"), _row("VA", 500, 7, "new", _LATER)
    name = _save(_volume, "w1", ("rows", [updated, inserted], every), ("deletes", [("VA", 4)], ()))
    l3.update(("VA", 1), votes=99, note="second")
    l3.insert([inserted])
    l3.delete(("VA", 4))
    log.commit([("VA", 1), ("VA", 500), ("VA", 4)], delta_file=name)
    lock.commit("state:VA")
    assert await follower.catch_up()
    assert _same(backend, l3)
    assert l3.statements == 0
    assert follower.stamp.epochs == lock.epochs


async def test_an_upsert_that_left_a_column_out_keeps_the_copys_value_and_date_created(_volume: Path) -> None:
    """Flips if a column the write left out, or the row's date_created, is taken from the file (null there, as an
    insert would leave it) instead of from the row the copy holds: L3's upsert keeps both (decision 19)."""
    world = _world()
    l3, log, lock, backend, follower = world
    await _warmed(world)
    # the write set votes alone; the file shows note and date_created as a fresh insert would, null and now
    name = _save(_volume, "w1", ("rows", [_row("VA", 2, 77, None, _LATER)], ["state", "geo_id", "votes"]))
    l3.update(("VA", 2), votes=77)
    log.commit([("VA", 2)], delta_file=name)
    lock.commit("state:VA")
    assert await follower.catch_up()
    assert _same(backend, l3)
    with backend.read_snapshot() as cursor:
        votes, note, born = cursor.execute(
            f"SELECT votes, note, date_created FROM {_TABLE} WHERE state = 'VA' AND geo_id = 2"  # noqa: S608
        ).fetchone()
    assert (votes, note, _instant(born)) == (77, "first", _BORN)
    assert l3.statements == 0


async def test_a_delete_marker_deletes_the_key(_volume: Path) -> None:
    """Flips if a file's delete marker is staged as a row (a row of nulls but its key) or skipped: the copy must lose
    the key, as L3 did."""
    world = _world()
    l3, log, lock, backend, follower = world
    await _warmed(world)
    name = _save(_volume, "w1", ("deletes", [("IL", 3)], ()))
    l3.delete(("IL", 3))
    log.commit([("IL", 3)], delta_file=name)
    lock.commit("state:IL")
    assert await follower.catch_up()
    assert ("IL", 3) not in [row[:2] for row in _copy(backend)]
    assert _same(backend, l3)
    assert l3.statements == 0


async def test_a_record_naming_a_file_the_volume_lost_reads_that_change_from_l3_by_key_counted() -> None:
    """Flips if a lost volume stalls the copy, or its fallback goes uncounted: the record's keys are read from L3 by
    key, as before delta files, and copy_follower_delta_file_fallback_keys_total{why="file_missing"} says so."""
    world = _world()
    l3, log, lock, backend, follower = world
    await _warmed(world)
    before = _fallbacks("file_missing")
    l3.update(("VA", 1), votes=61)
    l3.update(("VA", 3), votes=63)
    log.commit([("VA", 1), ("VA", 3)], delta_file=f"{_TABLE}/lost.arrow")
    lock.commit("state:VA")
    assert await follower.catch_up()
    assert _same(backend, l3)
    assert l3.statements == 1  # both keys, in one keyed read
    assert _fallbacks("file_missing") - before == 2


async def test_a_table_that_does_not_opt_in_reads_l3_as_before(_volume: Path) -> None:
    """Flips if a table that does not opt in is applied from a file, or counted as falling back: L3 stays the default
    everywhere (decision 5), and its keyed read is that table's normal path."""
    world = _world(delta_files=False)
    l3, log, lock, backend, follower = world
    await _warmed(world)
    before = {why: _fallbacks(why) for why in ("file_missing", "no_file_named", "no_epoch")}
    l3.update(("VA", 1), votes=88)
    log.commit([("VA", 1)])
    lock.commit("state:VA")
    assert await follower.catch_up()
    assert _same(backend, l3)
    assert l3.statements == 1
    assert {why: _fallbacks(why) for why in before} == before


async def test_a_key_left_out_of_a_change_is_staged_again_from_its_files_next_pass_with_no_l3_read(
    _volume: Path,
) -> None:
    """Flips if a key a pass leaves out (its scope moved while the pass read) is read from L3 next pass, or if its
    epochs are applied out of order or one is lost: the next pass stages it again from the copy's row and each of its
    epochs' files in order. Here the second epoch sets the note alone, so applying only the newest, or the two in the
    wrong order, leaves a value L3 does not hold."""
    world = _world()
    l3, log, lock, backend, follower = world
    await _warmed(world)
    first = _save(
        _volume,
        "w1",
        ("rows", [_row("VA", 1, 40, None, _LATER), _row("IL", 2, 50, None, _LATER)], ["state", "geo_id", "votes"]),
    )
    l3.update(("VA", 1), votes=40)
    l3.update(("IL", 2), votes=50)
    log.commit([("VA", 1), ("IL", 2)], delta_file=first)
    lock.commit("state:VA", "state:IL")
    settled_va = [row for row in _copy(backend) if row[0] == "VA"]

    def moves() -> None:
        # between the pass's two reads of the stamp, another load commits VA: its own file, setting the note alone
        second = _save(_volume, "w2", ("rows", [_row("VA", 1, 0, "late", _LATER)], ["state", "geo_id", "note"]))
        l3.update(("VA", 1), note="late")
        log.commit([("VA", 1)], delta_file=second)
        lock.commit("state:VA")

    lock.on_read = (lock.reads + 2, moves)
    assert await follower.catch_up()
    stamp = follower.stamp
    assert isinstance(stamp, PartialStamp) and "state:VA" in stamp.held
    assert [row for row in _copy(backend) if row[0] == "VA"] == settled_va
    assert _instants(row for row in _copy(backend) if row[0] == "IL") == _instants(
        row for row in l3.held() if row[0] == "IL"
    )
    lock.on_read = None
    assert await follower.catch_up()
    assert _same(backend, l3)
    with backend.read_snapshot() as cursor:
        assert cursor.execute(
            f"SELECT votes, note FROM {_TABLE} WHERE state = 'VA' AND geo_id = 1"  # noqa: S608
        ).fetchone() == (40, "late")
    assert l3.statements == 0
    assert follower.stamp.epochs == lock.epochs
