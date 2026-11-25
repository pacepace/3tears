"""a copy follower applies a table that opts in to delta files from the file each epoch's record names, never L3.

Election night failed with L3 on the hot path: every pod copy read each change's rows back from L3
(``design-pod-delta-files.md``). The registry, the follower and the delta files are the real ones; L3 is a DuckDB that
counts every statement it answers, kept equal to what the writes would leave in L3, so "the copy equals L3" and "no
statement reached L3" are both checked. The contract this pins:

- an opted-in table's change (an update, an insert, a delete) is applied from its file with no L3 statement;
- an upsert that left a column out keeps the copy's value of it and the row's ``date_created`` (decision 19);
- a delete marker deletes the key;
- a record naming a file the volume does not hold reads nothing from L3: the table waits for the file, its changes
  held, until the writer saves it again (decision 29; the end-to-end round with the writer is in
  ``test_delta_recreate.py``);
- a table that does not opt in reads L3 as before, counted as no fallback;
- a key a pass leaves out (its scope moved under the read) is staged again from its files by the next pass, every
  epoch of it in order, with no L3 read;
- a commit too big to list its rows (keys unknown) whose record names its file is applied whole from the file, with
  no L3 statement and no read of the change notes; one whose record names none is named from the notes as before;
- a commit of several records, each naming its one file, is applied once;
- a load from files of an opted-in table (``load_files`` with its ``LoadDelta``) saves each request's file before the
  request and names it in the request's record; a copy applies it with no L3 statement;
- a load that replaces a key prefix clears the copy's rows under it first (a row no longer in the export leaves), and
  a key-prefix delete (``delete_prefixes``) deletes the copy's rows under its prefixes, both with no L3 statement
  (decision 23);
- a load on a direct pool that fails after committing some rows names its file unconfirmed in the record it advances
  (the file may hold rows L3 never took): copies wait with no L3 read, and are current once the writer confirms the
  file from L3 by key; a pod's failed request writes no record of its own (its broker owns it), a cancel with no
  answer none at all; a record naming no file of an opted-in table is refused by the copy, never read from L3 (ruling C
  of 2026-11-23).
"""

from __future__ import annotations

import asyncio
import logging
import threading
import tracemalloc
from collections.abc import Callable, Iterable, Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator, Mapping
from sqlalchemy import BigInteger, Column as SaColumn, DateTime, MetaData, String, Table
from threetears.core.testing.l1_tables import L1_EVICTION_INFO, TEST_L1_TABLE
from threetears.core.testing.loop_lateness import turn_holds

pa = pytest.importorskip("pyarrow")
duckdb = pytest.importorskip("duckdb")

from prometheus_client import REGISTRY  # noqa: E402
from threetears.core.cache.duckdb import DuckDBBackend  # noqa: E402
from threetears.core.cache.duckdb_binds import run_bound  # noqa: E402
from threetears.core.collections import CollectionRegistry  # noqa: E402
from threetears.core.collections.complete_copy import Unsettled  # noqa: E402
import threetears.core.arrow_keys as arrow_keys  # noqa: E402
import threetears.core.collections.copy_follower as copy_follower  # noqa: E402
import threetears.core.collections.copy_key_work as copy_key_work  # noqa: E402
import threetears.core.collections.delta_files as delta_files  # noqa: E402
from threetears.core.collections.change_notes import NotedKeysPage  # noqa: E402
from threetears.core.collections.copy_follower import CopyFollower, FollowedTable, PartialStamp  # noqa: E402
from threetears.core.collections.delta_files import TableDelta, read_delta_file_keys, write_delta_file  # noqa: E402
from threetears.core.collections.delta_recreate import DeltaRecreator  # noqa: E402
from threetears.core.collections.epoch_change import (  # noqa: E402
    EVERY_STATE_NAMED,
    ChangedRow,
    EpochChange,
    EpochWalk,
    walk_by_count,
)
from threetears.core.backends.broker_generation import (  # noqa: E402
    GENERATIONS_REPLY_FIELD,
    named_delta_files,
    note_load_answer,
    record_reply_generations,
)
from threetears.core.collections.base import NO_L2  # noqa: E402
from threetears.core.backends.protocol import DurableStore, FileLoad, LoadFile, LoadResult  # noqa: E402
from threetears.core.collections import WRITE_GENERATION  # noqa: E402
from threetears.core.collections.caller_transaction import CallerTransaction  # noqa: E402
from threetears.core.collections.delta_files import LoadDelta, read_volume_file  # noqa: E402
from threetears.core.testing.delta_files import read_delta_file  # noqa: E402
from threetears.core.collections.schema_backed import (  # noqa: E402
    BIGINT_TYPE,
    DATETIMETZ_TYPE,
    STRING_TYPE,
    Column,
    TableSchema,
    collection_for_schema,
    rail_arrow_types,
)
from threetears.core.testing.delta_copies import NO_COPIES  # noqa: E402
from threetears.core.collections.delta_fold import DeltaFolds, FoldedTable  # noqa: E402
from threetears.core.collections.volume_fulls import (  # noqa: E402
    FullTable,
    VolumeFulls,
    current_full_manifest,
    publish_full_manifest,
)
from threetears.core.collections.registry import WantedFile  # noqa: E402
from threetears.core.config import DefaultCoreConfig  # noqa: E402
from packages.core.tests.copy_workers_support import COPY_WORKERS as _WORKERS  # noqa: E402
from threetears.core.process_workers import ProcessWorkers  # noqa: E402
from threetears.core.entities.base import BaseEntity  # noqa: E402
from threetears.core.testing.l1_tables import TEST_L1_TABLE as _TEST_L1_TABLE  # noqa: E402

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
    _WORKERS.jobs.clear()


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
        # one statement over the rows as an Arrow table, the last of a key winning as row-by-row upserts left it: a
        # statement a row binds each value from Python (a failed ``import pandas`` a value, under testament a stat of
        # every path), past the loop-stall guard's limit (threetears.core.testing.loop_stall)
        if not rows:
            return
        put = pa.Table.from_pylist(
            [{**{name: row[name] for name in _NAMES}, "at__": at} for at, row in enumerate(rows)],
            schema=pa.schema(
                [
                    ("state", pa.string()),
                    ("geo_id", pa.int64()),
                    ("votes", pa.int64()),
                    ("note", pa.string()),
                    ("date_created", pa.timestamp("us", tz="UTC")),
                    ("at__", pa.int64()),
                ]
            ),
        )
        self.db.register("_put", put)
        try:
            self.db.execute(
                f"INSERT OR REPLACE INTO {_TABLE} SELECT {', '.join(_NAMES)} FROM _put "  # noqa: S608
                "QUALIFY row_number() OVER (PARTITION BY state, geo_id ORDER BY at__ DESC) = 1"
            )
        finally:
            self.db.unregister("_put")

    def update(self, key: tuple[str, int], **values: Any) -> None:
        # an upsert of a row L3 held: only the columns its write set
        sets = ", ".join(f"{name} = ?" for name in values)
        self.db.execute(
            f"UPDATE {_TABLE} SET {sets} WHERE state = ? AND geo_id = ?",  # noqa: S608
            [*values.values(), *key],
        )

    def delete(self, key: tuple[str, int]) -> None:
        self.db.execute(f"DELETE FROM {_TABLE} WHERE state = ? AND geo_id = ?", list(key))  # noqa: S608

    def delete_state(self, state: str) -> int:
        held = self.db.execute(f"SELECT count(*) FROM {_TABLE} WHERE state = ?", [state]).fetchone()  # noqa: S608
        self.db.execute(f"DELETE FROM {_TABLE} WHERE state = ?", [state])  # noqa: S608
        return int(held[0]) if held is not None else 0

    def held(self) -> list[tuple[Any, ...]]:
        return sorted(self.db.execute(f"SELECT {', '.join(_NAMES)} FROM {_TABLE}").fetchall())  # noqa: S608

    async def fetch(self, sql: str, *params: Any) -> list[dict[str, Any]]:
        # on a thread, as the real rail's reply is read off the loop: a whole read's 200,000 rows turned into dicts
        # held it 0.59 s (the loop-stall guard, 2026-11-25)
        rows = await asyncio.to_thread(run_bound, self.db.cursor(), sql, params, _dict_rows)
        if not sql.startswith("SELECT version()"):
            self.statements += 1
        return rows


#: the most rows one epoch record lists, as the broker's record cap (256 KB) lets one list them: a commit of more is
#: recorded keys unknown, its keys in its delta file (decision 21). Why the fake keeps the cap: a record listing
#: 200,000 rows holds 400,000 Python objects (a row and its ids each), which the broker never records, and which made
#: every full collection of the loop-turn tests walk them (2026-11-23: 211-354 ms turns, decision 35)
_RECORD_ROWS = 5_000


class _Log:
    """the table's write generation and its change records, each naming the commit's delta file when it saved one."""

    def __init__(self) -> None:
        self.count = 0
        self.records: dict[str, EpochChange] = {}
        #: run once at the next read of the generation, before it answers: a commit landing between a pass's reads
        self.on_current: Callable[[], None] | None = None

    async def current(self, table: str) -> str:
        if self.on_current is not None:
            hook, self.on_current = self.on_current, None
            hook()
        return f"inc:{self.count}"

    def commit(self, keys: Sequence[tuple[str, int]], delta_file: str | None = None) -> str:
        if len(keys) > _RECORD_ROWS:
            # past the record cap the broker records the commit keys unknown, naming its file (decision 21)
            return self.too_large(delta_file)
        self.count += 1
        token = f"inc:{self.count}"
        self.records[token] = EpochChange(
            table=_TABLE, token=token, rows=[ChangedRow(ids=[s, str(g)]) for s, g in keys], delta_file=delta_file
        )
        return token

    def too_large(self, delta_file: str | None, scopes: Sequence[str] | None = None) -> str:
        """a commit the broker recorded keys unknown, with its notes bracket, naming its file when it saved one, and
        the states its commit touched when it names them (decision 31)."""
        self.count += 1
        token = f"inc:{self.count}"
        self.records[token] = EpochChange(
            table=_TABLE,
            token=token,
            keys_unknown=True,
            noted_between=(_LATER, _LATER),
            delta_file=delta_file,
            scopes=list(scopes) if scopes is not None else None,
        )
        return token

    async def advance(self, table_name: str) -> str:
        self.count += 1
        return f"inc:{self.count}"

    async def advance_with_change(
        self,
        table_name: str,
        rows: Sequence[ChangedRow] | None,
        *,
        keys_unknown: bool = False,
        prefixes: Sequence[Mapping[str, str]] | None = None,
        delta_file: str | None = None,
        unconfirmed_delta_file: str | None = None,
        unconfirmed_writer: str | None = None,
    ) -> str:
        """the writer's own advance (a load, a key-prefix delete), recorded as its broker would: its reach and its file,
        a failed load request's named unconfirmed, a load's intent beside its loading writer's beat key."""
        token = await self.advance(table_name)
        self.records[token] = EpochChange(
            table=table_name,
            token=token,
            whole_table=rows is None and not keys_unknown,
            keys_unknown=keys_unknown,
            rows=list(rows or []),
            prefixes=[dict(prefix) for prefix in prefixes] if prefixes else None,
            delta_file=delta_file if unconfirmed_delta_file is None else unconfirmed_delta_file,
            delta_file_unconfirmed=unconfirmed_delta_file is not None,
            unconfirmed_writer=unconfirmed_writer,
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


class _Notes:
    """the change notes as the broker's l3.noted_keys door answers them: one page of the keys given; counts each ask."""

    def __init__(self, keys: Sequence[tuple[str, ...]] = ()) -> None:
        self.keys = tuple(keys)
        self.asked = 0

    async def noted_keys(self, table: str, **_: Any) -> NotedKeysPage:
        self.asked += 1
        return NotedKeysPage(keys=self.keys, whole_table=False, next=None)


def _world(
    *,
    delta_files: bool = True,
    notes: _Notes | None = None,
    recreate: Any = None,
    key: tuple[str, ...] = _KEY,
    registry: CollectionRegistry | None = None,
    fulls: Any = None,
    page_size: int = 50,
    workers: ProcessWorkers | None = None,
) -> tuple[_L3, _Log, _Scopes, DuckDBBackend, CopyFollower]:
    l3, log, lock = _L3(), _Log(), _Scopes()
    backend = DuckDBBackend()
    _OPENED.append(backend.reset)
    backend.initialize(_metadata())
    registry = CollectionRegistry() if registry is None else registry
    registry.follow_generation(_TABLE)
    registry.set_noted_keys_reader(notes)
    follower = CopyFollower(
        registry,
        [FollowedTable(name=_TABLE, key=key, columns=_COLUMNS, scope=("state", "state:"), delta_files=delta_files)],
        backend,
        l3,
        settled=lock.settled,
        generation=log.current,
        changes=log,
        page_size=page_size,
        stamp_table="scope_epochs",
        recreate=recreate,
        fulls=fulls,
        drift_workers=_WORKERS if workers is None else workers,
    )
    follower.register()
    return l3, log, lock, backend, follower


class _HereWorkers(ProcessWorkers):
    """the copy's worker, but every job run here, on the caller's thread, each named: a test then sees inside a job
    (a file read, a merge) what it cannot see in another process."""

    def __init__(self) -> None:
        super().__init__(workers=1, name="test copy follower, here")
        self.jobs: list[str] = []

    async def run(self, job: Callable[..., Any], *args: Any, inline: bool = False) -> Any:
        self.jobs.append(job.__name__)
        return await super().run(job, *args, inline=True)


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


async def _same_now(backend: DuckDBBackend, l3: _L3) -> bool:
    """:func:`_same`, on a thread: both tables read into Python rows, 200,000 of them a large test's, held the loop
    0.56 s (the loop-stall guard, 2026-11-25)."""
    return await asyncio.to_thread(_same, backend, l3)


def _save_arrow(volume: Path, write: str, rows: Any, written: Sequence[str]) -> str:
    """save a commit's delta file of rows already Arrow, as a bulk writer does: no Python row a key."""
    delta = TableDelta(name=_TABLE, types=_TYPES, key=_KEY)
    delta.arrow(rows, written)
    return write_delta_file(volume, delta, write)


def _votes_plus_one(count: int) -> Any:
    """``count`` rows of VA, each key's votes one past its geo_id, written with votes alone (an upsert of a held row)."""
    return pa.table(
        {
            "state": pa.array(["VA"] * count, pa.string()),
            "geo_id": pa.array(range(count), pa.int64()),
            "votes": pa.array(range(1, count + 1), pa.int64()),
            # left out of the write: a held row keeps its own (decision 19)
            "note": pa.nulls(count, pa.string()),
            "date_created": pa.nulls(count, pa.string()),
        }
    )


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
    assert await _same_now(backend, l3)
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
    assert await _same_now(backend, l3)
    assert l3.statements == 0
    assert follower.stamp.epochs == lock.epochs


async def test_a_change_committed_between_a_pass_s_stamp_and_its_generation_read_is_read_from_l3_once(
    _volume: Path,
) -> None:
    """Flips if a change committed while a pass reads (here: before its generation read answers) is read, left out when
    the stamp is seen to have moved, and read again next pass: two L3 reads of one changed row (reports'
    test_one_changed_row_reaches_the_other_replica_by_its_key_alone, 3 runs in 13, 2026-11-25). The pass reads each
    table's generation before its stamp, and a key heard past that generation waits for the next pass: one read."""
    world = _world(delta_files=False)
    l3, log, lock, backend, follower = world
    await _warmed(world)

    def commits() -> None:
        l3.update(("VA", 1), votes=77)
        log.commit([("VA", 1)])
        lock.commit("state:VA")

    log.on_current = commits
    for _ in range(3):
        await follower.catch_up()
    assert await _same_now(backend, l3)
    assert l3.statements == 1, f"the changed row was read from L3 {l3.statements} times"
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
    assert await _same_now(backend, l3)
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
    assert await _same_now(backend, l3)
    assert l3.statements == 0


async def test_a_record_naming_a_file_the_volume_lost_reads_nothing_from_l3_and_waits_for_the_file(
    _volume: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Flips if a copy reads a lost file's keys from L3 (every pod at once: the load election night could not carry), or
    applies anything of the table while its file is missing, or never applies the held change once the file is back.
    The copy reads nothing from L3, says which file it waits for, and applies the change once the writer saves it again
    (decision 29, which amends decision 3)."""
    monkeypatch.setattr(copy_follower, "FILE_POLL_SECONDS", 0.01, raising=False)
    world = _world()
    l3, log, lock, backend, follower = world
    await _warmed(world)
    held = _copy(backend)
    updated = _row("VA", 1, 61, "second")
    l3.update(("VA", 1), votes=61, note="second")
    lost = f"{_TABLE}/lost.arrow"
    log.commit([("VA", 1)], delta_file=lost)
    lock.commit("state:VA")

    await follower.catch_up()

    assert l3.statements == 0, "a copy never reads a lost file's change from L3"
    assert _copy(backend) == held, "nothing of the table is applied while its file is missing"
    assert lost in follower.status().waiting_for_file[_TABLE]

    # the writer saves the file again under its name (test_delta_recreate.py runs the writer's side)
    assert _save(_volume, "lost", ("rows", [updated], list(_NAMES))) == lost
    await asyncio.wait_for(_until(lambda: not follower.status().waiting_for_file), timeout=5)
    assert await follower.catch_up()

    assert await _same_now(backend, l3)
    assert l3.statements == 0


async def _until(done: Callable[[], bool]) -> None:
    while not done():
        await asyncio.sleep(0.005)


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
    assert await _same_now(backend, l3)
    assert l3.statements == 1
    assert {why: _fallbacks(why) for why in before} == before


async def test_a_copy_keyed_by_a_timestamp_reads_its_own_rows_of_a_change_by_the_instant(_volume: Path) -> None:
    """Flips if the copy's own read of a change's rows by key (CopyFollower._copy_rows_of_keys, run for a table whose
    scope is in its rows only) compares the copy's timestamp text with the key's timestamp uncast: DuckDB's IN refuses
    VARCHAR against TIMESTAMPTZ (a Binder Error), so every pass failed and the copy never applied the change. Reports'
    result_snapshots_by_id is keyed by source_timestamp; its copy on a second replica stood current at the right epochs
    without ever answering (test_refresh_live::test_a_first_load_on_one_replica_reaches_the_others_copy, 2026-11-24).
    The key names the instant in another text form than the copy keeps (a space, ``+00``): it matches all the same."""
    world = _world(delta_files=False, key=("geo_id", "date_created"))
    l3, log, lock, backend, follower = world
    rows = [_row("VA", i, 10 + i) for i in range(5)] + [_row("IL", 10 + i, 20 + i) for i in range(5)]
    l3.insert(rows)
    log.commit([(str(row["geo_id"]), row["date_created"].isoformat()) for row in rows])
    await follower.warm()
    assert await follower.catch_up()
    assert await _same_now(backend, l3)
    l3.update(("VA", 1), votes=88)
    log.commit([("1", "2026-11-03 21:45:00+00")])  # type: ignore[list-item]
    lock.commit("state:VA")
    assert await follower.catch_up()
    assert await _same_now(backend, l3)
    assert follower.status().failures == 0


async def test_a_copy_keyed_by_a_timestamp_merges_a_files_upsert_onto_the_row_it_holds(_volume: Path) -> None:
    """Flips if the copy's read of the rows it holds of a file's keys (the row an upsert that left columns out is
    merged onto, decision 19) compares the copy's timestamp text with the key's timestamp uncast: DuckDB's IN refuses
    VARCHAR against TIMESTAMPTZ (a Binder Error), so the pass fails and the change is never applied. Applied, the copy
    keeps the columns the write left out."""
    key = ("geo_id", "date_created")
    world = _world(key=key)
    l3, log, lock, backend, follower = world
    rows = [_row("VA", i, 10 + i) for i in range(3)]
    l3.insert(rows)
    log.commit([(str(row["geo_id"]), row["date_created"].isoformat()) for row in rows])
    await follower.warm()
    assert await follower.catch_up()
    assert await _same_now(backend, l3)
    l3.statements = 0
    # the write set votes alone: the copy keeps the row's state and note
    delta = TableDelta(name=_TABLE, types=_TYPES, key=key)
    delta.rows([{"state": None, "geo_id": 1, "votes": 77, "note": None, "date_created": _BORN}], key + ("votes",))
    name = write_delta_file(_volume, delta, "w1")
    l3.update(("VA", 1), votes=77)
    # the key as the file names it (TableDelta's text of an instant)
    log.commit([("1", "2026-11-03T21:45:00.000000+00:00")], delta_file=name)  # type: ignore[list-item]
    lock.commit("state:VA")
    assert await follower.catch_up()
    assert await _same_now(backend, l3)
    assert l3.statements == 0


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
    assert await _same_now(backend, l3)
    with backend.read_snapshot() as cursor:
        assert cursor.execute(
            f"SELECT votes, note FROM {_TABLE} WHERE state = 'VA' AND geo_id = 1"  # noqa: S608
        ).fetchone() == (40, "late")
    assert l3.statements == 0
    assert follower.stamp.epochs == lock.epochs


async def test_a_commit_too_big_to_list_is_applied_whole_from_its_file_reading_neither_l3_nor_the_notes(
    _volume: Path,
) -> None:
    """Flips if a keys-unknown commit whose record names its file is named from the change notes, or any of its rows
    read from L3: the notes expire, which left a restarted copy behind on 2026-11-23, and L3 is what election night
    could not carry. Every row and delete marker of the file is applied, merged onto the copy's rows (decision 21)."""
    notes = _Notes()
    world = _world(notes=notes)
    l3, log, lock, backend, follower = world
    await _warmed(world)
    updated, inserted = _row("VA", 1, 99, None, _LATER), _row("IL", 700, 7, "new", _LATER)
    name = _save(
        _volume,
        "big",
        ("rows", [updated], ["state", "geo_id", "votes"]),
        ("rows", [inserted], list(_NAMES)),
        ("deletes", [("VA", 4), ("IL", 0)], ()),
    )
    l3.update(("VA", 1), votes=99)
    l3.insert([inserted])
    l3.delete(("VA", 4))
    l3.delete(("IL", 0))
    log.too_large(name)
    lock.commit("state:VA", "state:IL")

    assert await follower.catch_up()

    assert await _same_now(backend, l3)
    assert (l3.statements, notes.asked) == (0, 0)
    assert follower.status().behind == {}
    assert follower.stamp.epochs == lock.epochs


async def test_a_lost_file_of_a_commit_too_big_to_list_naming_its_states_reads_neither_l3_nor_the_notes(
    _volume: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Flips if a copy, finding the file of a keys-unknown commit whose record names its states lost, reads those
    states (or the notes' keys) from L3 itself: only the writer recovers such a commit (decision 31), once; every copy
    reads nothing, holds the table, and waits for the writer."""
    monkeypatch.setattr(copy_follower, "FILE_POLL_SECONDS", 0.01, raising=False)
    notes = _Notes([("VA", "1")])
    world = _world(notes=notes)
    l3, log, lock, backend, follower = world
    await _warmed(world)
    held = _copy(backend)
    l3.update(("VA", 1), votes=71)
    lost = f"{_TABLE}/lost-big.arrow"
    log.too_large(lost, scopes=["state:VA"])
    lock.commit("state:VA")

    await follower.catch_up()

    assert (l3.statements, notes.asked) == (0, 0), "a copy reads neither L3 nor the notes for a lost file"
    assert _copy(backend) == held
    assert lost in follower.status().waiting_for_file[_TABLE]


def _told_wanted() -> tuple[CollectionRegistry, list[WantedFile]]:
    """a registry, and every delta file it tells the copies of the table it could not apply, in order: told to a cache
    of its own that waits for files beside the copy's."""
    registry = CollectionRegistry()
    told: list[WantedFile] = []
    registry.register_derived_cache(
        _TABLE,
        on_row=lambda _message: None,
        on_table_dropped=lambda: None,
        on_partition=lambda _prefix: None,
        on_file_wanted=told.append,
    )
    return registry, told


async def test_a_lost_file_of_a_commit_too_big_to_list_reaches_the_copy_with_the_states_its_record_names(
    _volume: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Flips if the states a keys-unknown record names its commit touched (decision 31) are dropped on the way to the
    copy's wait for its lost file: a copy whose file the current full covers could then load only the whole table, not
    the touched states' parts (decision 54 (c)). The registry tells them with the file, and the wait holds them."""
    monkeypatch.setattr(copy_follower, "FILE_POLL_SECONDS", 0.01, raising=False)
    registry, told = _told_wanted()
    world = _world(notes=_Notes(), registry=registry)
    l3, log, lock, backend, follower = world
    await _warmed(world)
    lost = f"{_TABLE}/lost-big.arrow"
    log.too_large(lost, scopes=["state:IL", "state:VA", EVERY_STATE_NAMED])
    lock.commit("state:VA", "state:IL")

    await follower.catch_up()

    assert [(file.file, file.keys_unknown, file.scopes) for file in told] == [(lost, True, ("state:IL", "state:VA"))]
    assert "its commit touched state:IL, state:VA" in follower.status().waiting_for_file[_TABLE]
    assert l3.statements == 0
    follower.stop()


async def test_a_lost_files_states_not_marked_complete_reach_the_copy_as_none(
    _volume: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Flips if a record naming its states without the mark that they are every state its commit wrote
    (EVERY_STATE_NAMED, read off its file's rows) is told to the copy with them: such a record (an older writer's, a
    table declaring no state scope) may name a strict subset, and a copy reloading only those states from the full
    would keep the old rows of the rest (decision 54 (c)). It is told with none, so the copy reloads the whole table."""
    monkeypatch.setattr(copy_follower, "FILE_POLL_SECONDS", 0.01, raising=False)
    registry, told = _told_wanted()
    world = _world(notes=_Notes(), registry=registry)
    l3, log, lock, backend, follower = world
    await _warmed(world)
    lost = f"{_TABLE}/lost-unmarked.arrow"
    log.too_large(lost, scopes=["state:VA"])
    lock.commit("state:VA", "state:IL")

    await follower.catch_up()

    assert [(file.file, file.keys_unknown, file.scopes) for file in told] == [(lost, True, None)]
    assert "touched" not in follower.status().waiting_for_file[_TABLE]
    follower.stop()


async def test_a_keyed_records_lost_file_holds_its_keys_and_a_loads_names_no_states(
    _volume: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Flips if a keyed record's lost file stops holding the keys heard at its epoch (what the file brings once it
    reads), or a record that lists its rows or prefixes is told with states: only a commit too big to list names them
    (decision 31). A load's lost file is told with its prefixes' reach and no states."""
    monkeypatch.setattr(copy_follower, "FILE_POLL_SECONDS", 0.01, raising=False)
    registry, told = _told_wanted()
    world = _world(registry=registry)
    l3, log, lock, backend, follower = world
    await _warmed(world)
    keyed = f"{_TABLE}/lost.arrow"
    log.commit([("VA", 1)], delta_file=keyed)
    lock.commit("state:VA")
    await follower.catch_up()
    status = follower.status()
    assert keyed in status.waiting_for_file[_TABLE] and "touched" not in status.waiting_for_file[_TABLE]
    assert status.pending[_TABLE] == 1, "the key heard at the epoch is held for the file"
    assert told == [], "a keyed record's file is found missing by the copy's pass, not told by the registry"
    follower.stop()

    registry, told = _told_wanted()
    world = _world(registry=registry)
    l3, log, lock, backend, follower = world
    await _warmed(world)
    loaded = f"{_TABLE}/lost-load.arrow"
    await log.advance_with_change(_TABLE, None, prefixes=[{"state": "VA"}], delta_file=loaded)
    lock.commit("state:VA")
    await follower.catch_up()
    assert [(file.file, file.keys_unknown, file.scopes) for file in told] == [(loaded, False, None)]
    assert "touched" not in follower.status().waiting_for_file[_TABLE]
    assert l3.statements == 0
    follower.stop()


async def test_lost_files_of_commits_too_big_to_list_carry_their_states_to_the_full_the_table_waits_for(
    _volume: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Flips if a lost file's states are dropped once the table waits for a full holding its change (decision 28's
    recovery, or ``covered``: a fold that deletes took the file, decision 54): that full must know which states' parts
    it reaches (decision 54 (c)). A second lost file's states join the first's; one whose record names none makes the
    full the whole table's."""
    monkeypatch.setattr(copy_follower, "FILE_POLL_SECONDS", 0.01, raising=False)
    asked: list[str] = []

    async def covered(table: str, file: str, epoch: str) -> str:
        asked.append(file)
        return "covered"

    followed = [FollowedTable(name=_TABLE, key=_KEY, columns=_COLUMNS, scope=("state", "state:"), delta_files=True)]
    world = _world(notes=_Notes(), recreate=covered, fulls=VolumeFulls(_volume, followed, poll_seconds=0.01))
    l3, log, lock, backend, follower = world
    await _warmed(world)
    try:
        log.too_large(f"{_TABLE}/a.arrow", scopes=["state:VA", EVERY_STATE_NAMED])
        lock.commit("state:VA")
        await follower.catch_up()
        assert follower.status().full_reach == {_TABLE: ("state:VA",)}
        await asyncio.wait_for(_until(lambda: f"{_TABLE}/a.arrow" in asked), timeout=5)
        await asyncio.wait_for(
            _until(lambda: "file_covered" in follower.status().waiting_for_full.get(_TABLE, "")), timeout=5
        )
        assert follower.status().full_reach == {_TABLE: ("state:VA",)}, "the writer's covered answer keeps them"

        log.too_large(f"{_TABLE}/b.arrow", scopes=["state:IL", EVERY_STATE_NAMED])
        lock.commit("state:IL")
        await follower.catch_up()
        assert follower.status().full_reach == {_TABLE: ("state:IL", "state:VA")}

        log.too_large(f"{_TABLE}/c.arrow")
        lock.commit("state:IL")
        await follower.catch_up()
        assert _TABLE in follower.status().waiting_for_full
        assert follower.status().full_reach == {}, "a change of unknown states needs the whole table's full"
        assert l3.statements == 0
    finally:
        follower.stop()


async def test_a_commit_of_several_records_naming_one_file_is_applied_once(
    _volume: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Flips if a key heard at two records of one commit (each record of a commit names its one file) has the file
    read, or its changes applied, once per record: the commit would be replayed over itself, a pass's work doubled.
    The file is read once, and each change of it applied once."""
    here = _HereWorkers()
    world = _world(workers=here)
    l3, log, lock, backend, follower = world
    await _warmed(world)
    name = _save(
        _volume,
        "runs",
        ("rows", [_row("VA", 1, 51, None, _LATER), _row("VA", 2, 52, None, _LATER)], ["state", "geo_id", "votes"]),
        ("rows", [_row("VA", 1, 0, "again", _LATER)], ["state", "geo_id", "note"]),
    )
    l3.update(("VA", 1), votes=51, note="again")
    l3.update(("VA", 2), votes=52)
    # the commit's rows in two runs, each its own record naming the one file; VA 1 is in both
    log.commit([("VA", 1)], delta_file=name)
    log.commit([("VA", 1), ("VA", 2)], delta_file=name)
    lock.commit("state:VA")
    reads: list[Path] = []
    merges: list[Any] = []
    read = delta_files.read_delta_table

    def counted_read(path: Path, **checks: Any) -> Any:
        reads.append(path)
        return read(path, **checks)

    class CountedRows(delta_files.HeldDeltaRows):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            merges.append(self)

    # the merge's own module: it runs in the copy's worker, here on the caller's thread so its reads are seen
    monkeypatch.setattr(copy_key_work, "read_delta_table", counted_read)
    monkeypatch.setattr(copy_key_work, "HeldDeltaRows", CountedRows)

    assert await follower.catch_up()
    assert "merged_job" in here.jobs

    assert await _same_now(backend, l3)
    assert l3.statements == 0
    assert len(reads) == 1
    assert sum(merge.applied for merge in merges) == 3, "VA 1's two changes and VA 2's one, each once"


#: the rows of the bounded-memory pass below: a tenth of the audit's 200,000, enough that a Python object a row stands
#: well clear of everything else a stage allocates
_BIG_ROWS = 20_000

#: the most Python memory (tracemalloc, numpy's arrays counted, Arrow's buffers not) a pass applying a 20,000-row
#: file's change may allocate at its peak, per row: the whole pass, the keys' hearing with the stage. Measured
#: 2026-11-23 (macOS arm64, Python 3.14), the stage alone: the old apply (the file's rows as dicts, its changes grouped
#: by key, the copy's rows as dicts, each key's row merged into a dict) 1,447 bytes a row; the Arrow apply 561 bytes a
#: row, all of it the stage's bookkeeping a key (the staged set, each key's scopes and epochs). The whole pass with
#: every key Arrow (decision 35): 210 bytes a row, numpy's position arrays. The bound sits between: a Python object a
#: key kept anywhere in the pass trips it.
_STAGE_BYTES_PER_ROW = 400


async def test_a_large_files_pass_builds_no_python_row_and_applies_it_exactly(_volume: Path) -> None:
    """Flips if a copy applies a delta file by turning its rows into Python objects (the old to_pylist, a dict a key:
    1,447 bytes of Python a row here; 450 MB at 200,000 rows of 15 text columns), or keeps a Python object a key heard
    or staged (a tuple and a set a key: 561 bytes a row; 210 now, the pass's numpy positions), or if the applied
    rows differ from L3's: half the rows updated with every column set, half with votes alone (note and date_created
    kept), every hundredth key deleted."""

    world = _world()
    l3, log, lock, backend, follower = world
    born = pa.table({"state": ["VA"] * _BIG_ROWS, "geo_id": list(range(_BIG_ROWS))})
    l3.db.register("born", born)
    l3.db.execute(f"INSERT INTO {_TABLE} SELECT state, geo_id, geo_id, 'first', ? FROM born", [_BORN])  # noqa: S608
    l3.db.unregister("born")
    log.commit([("VA", i) for i in range(_BIG_ROWS)])
    await follower.warm()
    assert await follower.catch_up()
    l3.statements = 0
    half = _BIG_ROWS // 2
    full = [_row("VA", i, i + 1, "second", _LATER) for i in range(half)]
    votes = [_row("VA", i, i + 2, None, _LATER) for i in range(half, _BIG_ROWS)]
    gone = [("VA", i) for i in range(0, _BIG_ROWS, 100)]
    name = _save(
        _volume,
        "big",
        ("rows", full, list(_NAMES)),
        ("rows", votes, ["state", "geo_id", "votes"]),
        ("deletes", gone, ()),
    )
    l3.db.execute(
        f"UPDATE {_TABLE} SET votes = geo_id + 1, note = 'second', date_created = ? WHERE geo_id < ?", [_LATER, half]
    )  # noqa: S608
    l3.db.execute(f"UPDATE {_TABLE} SET votes = geo_id + 2 WHERE geo_id >= ?", [half])  # noqa: S608
    l3.db.execute(f"DELETE FROM {_TABLE} WHERE geo_id % 100 = 0")  # noqa: S608
    log.commit([("VA", i) for i in range(_BIG_ROWS)], delta_file=name)
    lock.commit("state:VA")
    del full, votes, gone
    # the whole pass: the keys' hearing (one Arrow batch from the registry) with the stage
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        assert await follower.catch_up()
        peaks = [tracemalloc.get_traced_memory()[1]]
    finally:
        tracemalloc.stop()
    peak = max(peaks)
    assert peak / _BIG_ROWS < _STAGE_BYTES_PER_ROW, f"the pass allocated {peak / _BIG_ROWS:.0f} bytes of Python a row"
    assert l3.statements == 0
    assert await _same_now(backend, l3)


#: the audit's real size for a loop bound: a 200,000-row commit applied from its file
_LOOP_ROWS = 200_000


def _born(l3: Any, log: Any, keys: Sequence[tuple[Any, ...]]) -> None:
    """L3's ``_LOOP_ROWS`` rows of VA and the commit naming ``keys``: a test's own setup, run on a thread (on the loop
    its Python rows and record held one turn past the loop-stall guard's limit on a loaded pool host, 2026-11-25)."""
    born = pa.table({"state": ["VA"] * _LOOP_ROWS, "geo_id": list(range(_LOOP_ROWS))})
    l3.db.register("born", born)
    l3.db.execute(f"INSERT INTO {_TABLE} SELECT state, geo_id, geo_id, 'first', ? FROM born", [_BORN])  # noqa: S608
    l3.db.unregister("born")
    log.commit(keys)


#: the longest one turn of the loop may run (CPU time, turn_holds) while a pass applies a 200,000-row commit from its
#: file, the collector on (decision 35: a few ms, under the SDK watchdog's 50 ms stall alarm). Measured 2026-11-23
#: (macOS arm64, Python 3.14), the stage alone: with the file's changes grouped and each key merged on the loop, 125.6
#: ms; with the merge on a thread but a Python tuple and set a key heard (and the fake log's 400,000 record objects),
#: 211-375 ms turns, every one a full collection walking them. The whole pass, the keys' hearing with it: a row message,
#: a tuple and a set a key, 135 ms; every key Arrow (registry, follower and fixture), 0.9 ms
_LOOP_TURN_SECONDS = 0.010


async def test_a_large_files_pass_holds_no_turn_of_the_loop_long(_volume: Path) -> None:
    """Flips if a pass applying a 200,000-row commit from its file holds one turn of the event loop past 10 ms, the
    collector on (decision 35): the keys heard a Python object each (a row message a key from the registry, a tuple and
    a set of epochs a key in the follower, each walked by every full collection), or the file's changes grouped and
    each key merged on the loop. The whole pass is timed, the keys' hearing with it: the registry hands the file's keys
    as one Arrow batch, the follower keeps them so, and the merge runs on a thread."""
    world = _world()
    l3, log, lock, backend, follower = world
    await asyncio.to_thread(_born, l3, log, [("VA", i) for i in range(_LOOP_ROWS)])
    await follower.warm()
    assert await follower.catch_up()

    def changed() -> None:
        name = _save_arrow(_volume, "loop", _votes_plus_one(_LOOP_ROWS), ["state", "geo_id", "votes"])
        l3.db.execute(f"UPDATE {_TABLE} SET votes = geo_id + 1")  # noqa: S608
        log.commit([("VA", i) for i in range(_LOOP_ROWS)], delta_file=name)

    await asyncio.to_thread(changed)
    lock.commit("state:VA")
    with turn_holds() as holds:
        assert await follower.catch_up()
    longest = max(seconds for seconds, _ in holds)
    assert longest < _LOOP_TURN_SECONDS, f"a turn held the loop {longest * 1000:.1f} ms during the pass"
    assert await _same_now(backend, l3)


async def test_a_large_pass_of_a_table_whose_scope_is_in_its_rows_stages_its_keys_off_the_loop(
    _volume: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Flips if a pass staging a table whose scope is in its rows alone (reports' snapshots and results, keyed by the
    race) works out its keys' scopes on the event loop: the copy's held rows beside the rows read, made distinct and
    matched to the page by Arrow kernels on the loop, held it 12.2 s in one page (``distinct_keys`` under
    ``_stage_keyed``), and the keys heard 24.3 s more (``_heard_batch`` under ``_drained``), in the local burst replay of
    2026-11-25. CPU belongs in a worker process: each runs in the copy's, and no turn holds the loop past 10 ms."""
    key = ("geo_id", "date_created")
    world = _world(key=key, page_size=_LOOP_ROWS)
    l3, log, lock, backend, follower = world
    instant = "2026-11-03T21:45:00.000000+00:00"  # the key's instant as a delta file names it
    await asyncio.to_thread(_born, l3, log, [(str(i), instant) for i in range(_LOOP_ROWS)])
    await follower.warm()
    assert await follower.catch_up()

    def changed() -> None:
        delta = TableDelta(name=_TABLE, types=_TYPES, key=key)
        delta.arrow(
            pa.table(
                {
                    "state": pa.nulls(_LOOP_ROWS, pa.string()),
                    "geo_id": pa.array(range(_LOOP_ROWS), pa.int64()),
                    "votes": pa.array(range(1, _LOOP_ROWS + 1), pa.int64()),
                    "note": pa.nulls(_LOOP_ROWS, pa.string()),
                    "date_created": pa.repeat(pa.scalar(_BORN, _TYPES["date_created"]), _LOOP_ROWS),
                }
            ),
            [*key, "votes"],
        )
        name = write_delta_file(_volume, delta, "scoped")
        l3.db.execute(f"UPDATE {_TABLE} SET votes = geo_id + 1")  # noqa: S608
        log.commit([(str(i), instant) for i in range(_LOOP_ROWS)], delta_file=name)  # type: ignore[misc]

    await asyncio.to_thread(changed)
    lock.commit("state:VA")
    loop_thread = threading.get_ident()
    on_loop: list[str] = []
    # the public kernels the key work runs through, wherever it runs: distinct_keys (made distinct, matched) and
    # key_text_table (the keys heard, spelled as text). Seen here only when called in this process, on the loop
    distinct, texts = arrow_keys.distinct_keys, delta_files.key_text_table

    def watched_distinct(rows: Any, columns: Sequence[str]) -> Any:
        if threading.get_ident() == loop_thread:
            on_loop.append(f"distinct_keys of {rows.num_rows} rows")
        return distinct(rows, columns)

    def watched_texts(rows: Any, key: Sequence[str], instants: Any = frozenset()) -> Any:
        if threading.get_ident() == loop_thread:
            on_loop.append(f"key_text_table of {rows.num_rows} keys")
        return texts(rows, key, instants)

    monkeypatch.setattr(copy_key_work, "distinct_keys", watched_distinct)
    monkeypatch.setattr(copy_key_work, "key_text_table", watched_texts)
    l3.statements = 0
    with turn_holds() as holds:
        assert await follower.catch_up()
    assert on_loop == [], f"the pass ran its keys' Arrow work on the event loop: {on_loop}"
    assert {"heard_job", "held_scopes_job", "page_scopes_job"} <= set(_WORKERS.jobs), (
        f"the key work did not run in the copy's worker process: {_WORKERS.jobs}"
    )
    longest = max(seconds for seconds, _ in holds)
    assert longest < _LOOP_TURN_SECONDS, f"a turn held the loop {longest * 1000:.1f} ms during the pass"
    assert await _same_now(backend, l3)
    assert l3.statements == 0


#: the longest one turn of the loop may run (CPU time) while a table waiting for a 200,000-key commit too big to list
#: hears its file's keys, the collector on (decision 35). Measured 2026-11-23 (macOS arm64, Python 3.14): the file read
#: whole and every key heard in one turn, 255 ms; the reach read on a thread and the keys heard in runs of 2,000, 32 ms
#: (222 ms on the pool), all of it one collection of the collector the keys' new tuples set off; the keys kept as the
#: file's one Arrow batch, 0.8-0.9 ms at 200,000 keys
_WAIT_TURN_SECONDS = 0.010


async def test_a_waited_for_commit_too_big_to_list_hears_its_keys_in_short_turns(
    _volume: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Flips if a table waiting for a commit too big to list hears its file's keys in one turn of the loop once it
    reads (the audit's 337,000 keys, the copy answering nothing else meanwhile), or reads the file's rows on the loop:
    the reach is read on a thread and the keys spelled in the copy's worker process. The change then applies whole, with
    no L3 statement."""
    monkeypatch.setattr(copy_follower, "FILE_POLL_SECONDS", 0.01, raising=False)
    world = _world()
    l3, log, lock, backend, follower = world
    await asyncio.to_thread(_born, l3, log, [("VA", i) for i in range(_LOOP_ROWS)])
    await follower.warm()
    assert await follower.catch_up()
    l3.statements = 0
    lost = f"{_TABLE}/big.arrow"
    log.too_large(lost)
    lock.commit("state:VA")
    await follower.catch_up()
    assert lost in follower.status().waiting_for_file[_TABLE]
    l3.db.execute(f"UPDATE {_TABLE} SET votes = geo_id + 1")  # noqa: S608
    staging = _volume / "staging"
    staging.mkdir()
    saved = await asyncio.to_thread(
        _save_arrow, staging, "big", _votes_plus_one(_LOOP_ROWS), ["state", "geo_id", "votes"]
    )
    assert saved == lost
    (_volume / lost).parent.mkdir(exist_ok=True)
    # the file's keys spelled in this process are seen here (key_text_table, under _heard_batch): on the loop they held
    # one turn 11.1 ms (past the bar, the pool host brooks-14th-mbp, 2026-11-25); they belong in the worker process
    loop_thread = threading.get_ident()
    on_loop: list[int] = []
    texts = delta_files.key_text_table

    def watched_texts(rows: Any, key: Sequence[str], instants: Any = ()) -> Any:
        if threading.get_ident() == loop_thread:
            on_loop.append(rows.num_rows)
        return texts(rows, key, instants)

    monkeypatch.setattr(copy_key_work, "key_text_table", watched_texts)
    with turn_holds() as holds:
        # the writer's file lands whole under its name, as its rename lands it
        (staging / lost).replace(_volume / lost)
        await asyncio.wait_for(_until(lambda: not follower.status().waiting_for_file), timeout=60)
    assert on_loop == [], f"the file's keys were spelled on the event loop: {on_loop}"
    assert "heard_job" in _WORKERS.jobs, (
        f"the file's keys were not spelled in the copy's worker process: {_WORKERS.jobs}"
    )
    longest = max(seconds for seconds, _ in holds)
    assert longest < _WAIT_TURN_SECONDS, f"a turn held the loop {longest * 1000:.1f} ms hearing the file's keys"
    assert await follower.catch_up()
    assert await _same_now(backend, l3)
    assert l3.statements == 0


async def test_a_commit_too_big_to_list_whose_record_names_no_file_is_named_from_the_notes() -> None:
    """Flips if a keys-unknown record that names no file (a writer before delta files) stops being named from the
    change notes: its keys come from them, with no epoch, and are read from L3 by key, counted ``no_epoch``."""
    notes = _Notes([("VA", "1"), ("VA", "3")])
    world = _world(notes=notes)
    l3, log, lock, backend, follower = world
    await _warmed(world)
    before = _fallbacks("no_epoch")
    l3.update(("VA", 1), votes=31)
    l3.update(("VA", 3), votes=33)
    log.too_large(None)
    lock.commit("state:VA")

    assert await follower.catch_up()

    assert await _same_now(backend, l3)
    assert notes.asked == 1
    assert l3.statements == 1
    assert _fallbacks("no_epoch") - before == 2
    assert follower.status().behind == {}


# --- the writer's side: a load from files and a key-prefix delete of the opted-in table -------------------------------

_WRITER_SCHEMA = TableSchema(
    name=_TABLE,
    primary_key=_KEY,
    columns=list(_COLUMNS),
    l1_eviction=_TEST_L1_TABLE,
    delta_files=True,
)


class _Result(BaseEntity):
    primary_key_field = "geo_id"


class _Results(collection_for_schema(_WRITER_SCHEMA, entity_class=_Result)):  # type: ignore[misc]
    write_generation = WRITE_GENERATION


class _Store(DurableStore):
    """L3 as the writer reaches it: a load from files (the hub's broker copies each file's rows itself) and a key-prefix
    delete on a caller's transaction, both into the DuckDB the copy's keyed reads would read. It notes, at each load
    and each commit, which delta files the volume already held."""

    def __init__(self, l3: _L3, volume: Path) -> None:
        self.l3 = l3
        self.volume = volume
        #: each file's rows, by url, as the export wrote them (no constant column)
        self.contents: dict[str, list[dict[str, Any]]] = {}
        #: per load request, the file it named and whether the volume held it when the request was sent
        self.loads: list[tuple[str | None, bool]] = []
        #: per commit of a caller's transaction, the delta files the volume held as it committed
        self.commits: list[set[str]] = []

    async def fetch_one(self, table: str, pk: Mapping[str, Any], *, conn: Any = None) -> dict[str, Any] | None:
        raise AssertionError("the writer reads no row")

    async def upsert(self, table: str, row: Any, **kwargs: Any) -> int:
        raise AssertionError("the writer upserts no row here")

    async def delete(self, table: str, pk: Any, *, conn: Any = None) -> None:
        raise AssertionError("the writer deletes no row by key here")

    async def scan(self, table: str, filters: Any = None) -> list[dict[str, Any]]:
        raise AssertionError("the writer scans nothing")

    async def load_files(
        self,
        table: str,
        files: Sequence[LoadFile],
        *,
        columns: Sequence[str],
        constants: Mapping[str, Any],
        rows_per_transaction: int,
        timeout_ms: int,
        key_prefixes: Sequence[Mapping[str, Any]] | None = None,
    ) -> LoadResult:
        named = named_delta_files().get(table)
        self.loads.append((named, named is not None and (self.volume / named).is_file()))
        loaded = []
        for index, file in enumerate(files):
            self.l3.insert([{**row, **constants} for row in self.contents[file.url]])
            loaded.append(FileLoad(index=index, rows_loaded=file.rows))
        return LoadResult(row_count=sum(file.rows for file in files), files=tuple(loaded))

    async def delete_prefixes(self, table: str, prefixes: Sequence[Mapping[str, Any]], *, conn: Any) -> int:
        return sum(self.l3.delete_state(str(prefix["state"])) for prefix in prefixes)

    @asynccontextmanager
    async def _transaction(self) -> AsyncIterator[None]:
        yield
        self.commits.append({str(path.relative_to(self.volume)) for path in self.volume.rglob("*.arrow")})

    def transaction(self, **options: Any) -> Any:
        return self._transaction()

    @asynccontextmanager
    async def _acquired(self) -> AsyncIterator[Any]:
        yield self

    def acquire(self) -> Any:
        return self._acquired()


def _writer(l3: _L3, log: _Log, volume: Path) -> tuple[_Results, _Store]:
    """the writing pod: its own registry, advancing through the epoch log the copy reads."""
    registry = CollectionRegistry()
    registry.set_generation_source(log)
    results = _Results(registry, DefaultCoreConfig(), None)
    store = _Store(l3, volume)
    results.l3_pool = store
    return results, store


def _export(store: _Store, url: str, rows: Sequence[dict[str, Any]]) -> tuple[LoadFile, Any]:
    """one export file of a state's rows (no state column: the load's constant), and the rows the pod read from it."""
    exported = [{name: row[name] for name in _NAMES if name != "state"} for row in rows]
    store.contents[url] = exported
    return LoadFile(url=url, rows=len(exported)), pa.Table.from_pylist(exported)


async def test_a_load_from_files_saves_its_file_before_the_load_and_the_copy_applies_it_with_no_l3_statement(
    _volume: Path,
) -> None:
    """Flips if a load of an opted-in table sends a request before that request's delta file is on the volume, names no
    file in its record, or a copy reads the load's rows back from L3 (decisions 11 and 17). Two files in two requests:
    each request's record names the file of its own rows."""
    world = _world()
    l3, log, lock, backend, follower = world
    await _warmed(world)
    results, store = _writer(l3, log, _volume)
    first, first_rows = _export(store, "https://bucket/va-0.parquet", [_row("VA", 100, 1, "new", _LATER)])
    second, second_rows = _export(
        store, "https://bucket/va-1.parquet", [_row("VA", 101, 2, "new", _LATER), _row("VA", 102, 3, None, _LATER)]
    )
    loaded = await results.load_files(
        [first, second],
        columns=[name for name in _NAMES if name != "state"],
        constants={"state": "VA"},
        connections=2,
        delta=LoadDelta(rows=[first_rows, second_rows]),
    )
    assert loaded == 3
    assert len(store.loads) == 2 and all(saved for _, saved in store.loads), "each file saved before its request"
    named = {name for name, _ in store.loads}
    recorded = {record.delta_file for record in list(log.records.values())[-2:]}
    assert recorded == named and None not in named
    assert sorted(len(read_delta_file(_volume / str(name)).changes) for name in named) == [1, 2]
    lock.commit("state:VA")

    assert await follower.catch_up()

    assert await _same_now(backend, l3)
    assert l3.statements == 0
    assert follower.stamp.epochs == lock.epochs


class _BrokerStore(_Store):
    """L3 as a pod reaches it: its broker loads the request, advances the table, records the epoch naming the
    request's file and its prefixes, and names the epoch in its reply (``record_reply_generations``)."""

    def __init__(self, l3: _L3, volume: Path, log: _Log) -> None:
        super().__init__(l3, volume)
        self.log = log

    async def load_files(self, table: str, files: Sequence[LoadFile], **kwargs: Any) -> LoadResult:
        result = await super().load_files(table, files, **kwargs)
        prefixes = [{column: str(value) for column, value in prefix.items()} for prefix in kwargs["key_prefixes"]]
        token = await self.log.advance_with_change(
            table, None, prefixes=prefixes, delta_file=named_delta_files().get(table)
        )
        record_reply_generations({GENERATIONS_REPLY_FIELD: {table: token}}, ends_write=True)
        return result


def _refused_prefixes() -> float:
    return REGISTRY.get_sample_value("copy_follower_partition_refused_total", {"table": _TABLE}) or 0.0


async def test_a_pods_own_load_reaches_its_own_copy_from_the_file_its_broker_named(_volume: Path) -> None:
    """Flips if a writing pod's own copy is told its own load's prefixes with no file: a collection that advances no
    generation of its own (a pod's table built without L2, its broker advancing it) dropped the prefixes here, and the
    pod's copy of a table that opts in refused them (and with fulls, waited for the next full) on every load it made
    (2026-11-24, reports' LSF loads). The epoch the broker's reply names is applied here from the file, accounted as
    this pod's own: no refusal, no L3 statement, the copy equal to L3 once the pass runs."""
    registry = CollectionRegistry()
    world = _world(registry=registry)
    l3, log, lock, backend, follower = world
    await _warmed(world)
    results = _Results(registry, DefaultCoreConfig(), NO_L2)
    store = _BrokerStore(l3, _volume, log)
    results.l3_pool = store
    file, rows = _export(store, "https://bucket/va-0.parquet", [_row("VA", 100, 1, "new", _LATER)])
    refused = _refused_prefixes()

    await results.load_files(
        [file],
        columns=[name for name in _NAMES if name != "state"],
        constants={"state": "VA"},
        delta=LoadDelta(rows=[rows]),
    )
    lock.commit("state:VA")
    assert await follower.catch_up()

    assert _refused_prefixes() == refused, "the pod's own copy refused its own load's prefixes"
    assert await _same_now(backend, l3)
    assert l3.statements == 0


async def test_a_load_that_replaces_a_prefix_clears_the_copys_rows_under_it_with_no_l3_statement(
    _volume: Path,
) -> None:
    """Flips if a row no longer in the export stays in the copy after a load that replaces its prefix, or the copy
    reads L3 for it (decision 23): L3 deleted the prefix's rows before the load, so the copy's must go too, cleared
    from the delete's file; a row the export holds again is the export's."""
    world = _world()
    l3, log, lock, backend, follower = world
    await _warmed(world)
    results, store = _writer(l3, log, _volume)
    file, rows = _export(
        store, "https://bucket/va.parquet", [_row("VA", 1, 70, "again", _LATER), _row("VA", 9, 90, "new", _LATER)]
    )
    await results.load_files(
        [file],
        columns=[name for name in _NAMES if name != "state"],
        constants={"state": "VA"},
        delta=LoadDelta(rows=[rows], replaces=[{"state": "VA"}]),
    )
    assert sorted(row[:2] for row in l3.held() if row[0] == "VA") == [("VA", 1), ("VA", 9)]
    (delete_files,) = store.commits
    assert delete_files, "the delete's file was on the volume before its commit"
    lock.commit("state:VA")

    assert await follower.catch_up()

    assert sorted(row[:2] for row in _copy(backend) if row[0] == "VA") == [("VA", 1), ("VA", 9)]
    assert await _same_now(backend, l3)
    assert l3.statements == 0


async def test_a_key_prefix_delete_deletes_the_copys_rows_under_it_with_no_l3_statement(_volume: Path) -> None:
    """Flips if ``delete_prefixes`` of an opted-in table is refused, commits before its file is saved, or leaves the
    copy holding the prefix's rows, or reads L3 to find them (decision 23): the file names the prefix and no row."""
    world = _world()
    l3, log, lock, backend, follower = world
    await _warmed(world)
    results, store = _writer(l3, log, _volume)
    async with store.acquire() as conn, CallerTransaction(conn) as transaction:
        await results.delete_prefixes([{"state": "IL"}], conn=conn)
    (name,) = transaction.delta_files.values()
    assert store.commits == [{name}], "the file was saved before the commit"
    deleted = read_delta_file(_volume / name)
    assert (deleted.changes, deleted.prefixes) == ([], ({"state": "IL"},))
    assert list(log.records.values())[-1].delta_file == name
    lock.commit("state:IL")

    assert await follower.catch_up()

    assert [row for row in _copy(backend) if row[0] == "IL"] == []
    assert await _same_now(backend, l3)
    assert l3.statements == 0


async def test_a_key_prefix_delete_of_many_rows_reaches_its_keys_in_the_worker_process(
    _volume: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Flips if the keys a cleared prefix reaches (the copy's rows under it beside every key heard or staged, matched,
    made distinct and spelled) are worked out in the serving process past the bound: the match alone took a thread
    20 ms at 1,000,000 keys heard (2026-11-25). The copy still loses every row under the prefix, reading no L3."""
    count = copy_key_work.INLINE_KEYS * 2
    world = _world()
    l3, log, lock, backend, follower = world

    def born() -> None:
        rows = [_row("IL", i, i) for i in range(count)] + [_row("VA", i, i) for i in range(5)]
        l3.insert(rows)
        log.commit([(row["state"], row["geo_id"]) for row in rows])

    await asyncio.to_thread(born)
    await follower.warm()
    assert await follower.catch_up()
    l3.statements = 0
    here: list[int] = []
    reach = copy_key_work.cleared_reach_of

    def watched(key: Any, instants: Any, held: Sequence[Any], known: Any, *rest: Any) -> Any:
        here.append(sum(rows.num_rows for rows in held) + known.num_rows)
        return reach(key, instants, held, known, *rest)

    monkeypatch.setattr(copy_follower, "cleared_reach_of", watched)
    results, store = _writer(l3, log, _volume)
    async with store.acquire() as conn, CallerTransaction(conn):
        await results.delete_prefixes([{"state": "IL"}], conn=conn)
    lock.commit("state:IL")
    assert await follower.catch_up()
    assert "cleared_reach_job" in _WORKERS.jobs, f"the prefix's keys were found in the serving process: {_WORKERS.jobs}"
    assert not [keys for keys in here if keys >= copy_key_work.INLINE_KEYS], f"found here: {here}"
    assert [row for row in await asyncio.to_thread(_copy, backend) if row[0] == "IL"] == []
    assert await _same_now(backend, l3)
    assert l3.statements == 0


class _FailingAfterAFileStore(_Store):
    """a direct pool's load that commits its first file's rows and then fails, as the store's ``COPY`` of a later file
    does: the request's rows are partly in L3."""

    async def load_files(self, table: str, files: Sequence[LoadFile], **kwargs: Any) -> LoadResult:
        named = named_delta_files().get(table)
        self.loads.append((named, named is not None and (self.volume / named).is_file()))
        self.l3.insert([{**row, **kwargs["constants"]} for row in self.contents[files[0].url]])
        raise RuntimeError("the second file's copy failed; the first file's rows stay committed")


async def test_a_direct_pool_load_failing_after_some_rows_names_its_delta_file_unconfirmed(_volume: Path) -> None:
    """Flips if a load from files on a direct pool that fails after committing some of its rows names its request's
    delta file as a whole load's (the file holds rows L3 never took, the failed file's, which a copy applying it would
    hold), or names none (every copy would read the load's prefixes back from L3). Its record names the file
    unconfirmed, on the load's prefixes (ruling C of 2026-11-23); the hub's broker keeps the same rule for a pod's
    load."""
    world = _world()
    l3, log, _lock, _backend, _follower = world
    await _warmed(world)
    results, _ = _writer(l3, log, _volume)
    store = _FailingAfterAFileStore(l3, _volume)
    results.l3_pool = store
    first, first_rows = _export(store, "https://bucket/va-0.parquet", [_row("VA", 100, 1, "new", _LATER)])
    second, second_rows = _export(store, "https://bucket/va-1.parquet", [_row("VA", 101, 2, "new", _LATER)])
    before = log.count
    with pytest.raises(RuntimeError, match="second file"):
        await results.load_files(
            [first, second],
            columns=[name for name in _NAMES if name != "state"],
            constants={"state": "VA"},
            delta=LoadDelta(rows=[first_rows, second_rows]),
        )
    (sent,) = store.loads
    assert sent[0] is not None and sent[1], "the request's file was saved and named before the load"
    advanced = [log.records[f"inc:{count}"] for count in range(before + 1, log.count + 1)]
    assert len(advanced) == 1, "the failed load advanced the table once"
    (record,) = advanced
    assert (record.delta_file, record.delta_file_unconfirmed) == (sent[0], True), "named, and unconfirmed"
    assert record.prefixes == [{"state": "VA"}]


class _Told:
    """what a load hands its caller for each request that committed: the files and the generations, as a commit's
    CallerTransaction names them (delta_files, committed_generations)."""

    def __init__(self) -> None:
        self.commits: list[tuple[dict[str, str], dict[str, str]]] = []

    def __call__(self, files: Mapping[str, str], generations: Mapping[str, str]) -> None:
        self.commits.append((dict(files), dict(generations)))


def _token_of(log: _Log, file: str) -> str:
    (token,) = [token for token, record in log.records.items() if record.delta_file == file]
    return token


async def test_a_load_from_files_hands_each_committed_request_its_file_and_generation_and_a_fold_fed_them_holds_its_rows(
    _volume: Path,
) -> None:
    """Flips if a load of an opted-in table does not hand its caller each committed request's delta file and the
    generation its advance wrote, in the shapes a CallerTransaction names them: the writer's fold never hears of the
    load (nearly every LSF write), folds a state past it, and publishes a full missing its rows. Fed what the load
    handed, the fold's full holds the loaded rows."""
    world = _world()
    l3, log, _lock, _backend, _follower = world
    await _warmed(world)
    results, store = _writer(l3, log, _volume)
    before = f"inc:{log.count}"
    first, first_rows = _export(store, "https://bucket/va-0.parquet", [_row("VA", 100, 1, "new", _LATER)])
    second, second_rows = _export(
        store, "https://bucket/va-1.parquet", [_row("VA", 101, 2, "new", _LATER), _row("VA", 102, 3, None, _LATER)]
    )
    told = _Told()
    await results.load_files(
        [first, second],
        columns=[name for name in _NAMES if name != "state"],
        constants={"state": "VA"},
        connections=2,
        delta=LoadDelta(rows=[first_rows, second_rows]),
        on_commit=told,
    )
    named = sorted(str(name) for name, _ in store.loads)
    assert sorted(files[_TABLE] for files, _ in told.commits) == named
    for files, generations in told.commits:
        assert set(files) == set(generations) == {_TABLE}
        assert generations[_TABLE] == _token_of(log, files[_TABLE])
    workers = ProcessWorkers(workers=1, name="test load folds")
    try:
        publish_full_manifest(_volume, term=1, tables={_TABLE: FullTable(version=before)})
        folds = DeltaFolds(
            _volume,
            [FoldedTable.of(_WRITER_SCHEMA, scope=("state", "state:"))],
            retention="delete",
            copies=NO_COPIES,
            workers=workers,
        )
        for files, generations in told.commits:
            folds.committed(files, generations, version=2, in_flight_from=None)
        await folds.wait()
        await folds.close()
    finally:
        await workers.close()
    manifest = current_full_manifest(_volume)
    assert manifest is not None
    (part,) = [part for part in manifest.tables[_TABLE].parts if part.scope == "state:VA"]
    folded = read_volume_file(_volume / part.file)
    assert sorted(folded.column("geo_id").to_pylist()) == [100, 101, 102]


async def test_a_copy_lagging_its_writer_applies_every_delta_file_a_published_full_covers_when_the_fold_keeps(
    _volume: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Flips if the writer's fold deletes a delta file once its full covers it while the fold keeps its delta files
    (``retention="keep"``, design-pod-delta-files decision 51): a copy lagging its writer (a median 7.7 s on the
    2026-11-24 replay) then finds its next epoch's file gone, waits on it as lost, and the table is reloaded whole (536
    files, 91 reloads of 339k rows). Kept, every file is still on the volume after the publish and the lagging copy
    applies each from its file: no L3 statement, no file waited for, no fallback."""
    monkeypatch.setattr(copy_follower, "FILE_POLL_SECONDS", 0.01, raising=False)
    world = _world()
    l3, log, lock, backend, follower = world
    await _warmed(world)
    publish_full_manifest(_volume, term=1, tables={_TABLE: FullTable(version=f"inc:{log.count}")})
    every = list(_NAMES)
    tokens: dict[str, str] = {}
    for write, (geo_id, votes) in enumerate(((1, 71), (2, 72), (3, 73)), start=1):
        changed = _row("VA", geo_id, votes, "kept")
        file = _save(_volume, f"keep{write}", ("rows", [changed], every))
        l3.update(("VA", geo_id), votes=votes, note="kept")
        tokens[file] = log.commit([("VA", geo_id)], delta_file=file)
        lock.commit("state:VA")
    missing = _fallbacks("file_missing")
    workers = ProcessWorkers(workers=1, name="test keep folds")
    try:
        folds = DeltaFolds(
            _volume, [FoldedTable.of(_WRITER_SCHEMA, scope=("state", "state:"))], retention="keep", workers=workers
        )
        # the writer folds and publishes before its copy has applied any of the three: the copy lags
        for file, token in tokens.items():
            folds.committed({_TABLE: file}, {_TABLE: token}, version=2, in_flight_from=None)
            await folds.wait()
        await folds.close()
    finally:
        await workers.close()
    manifest = current_full_manifest(_volume)
    assert manifest is not None and manifest.tables[_TABLE].version == list(tokens.values())[-1], (
        "the published full covers every file"
    )
    assert all((_volume / file).is_file() for file in tokens), "a delta file the full covers is kept"

    assert await follower.catch_up()

    assert not follower.status().waiting_for_file, "the lagging copy found every file it applies"
    assert await _same_now(backend, l3)
    assert l3.statements == 0
    assert _fallbacks("file_missing") == missing


async def test_a_load_that_replaces_a_prefix_hands_the_delete_s_commit_first(_volume: Path) -> None:
    """Flips if the delete of the prefixes a load replaces, a commit of its own with its own delta file, is not handed
    to the caller before the load's: the fold would never clear the prefix's old rows from its part."""
    world = _world()
    l3, log, _lock, _backend, _follower = world
    await _warmed(world)
    results, store = _writer(l3, log, _volume)
    file, rows = _export(store, "https://bucket/va.parquet", [_row("VA", 9, 90, "new", _LATER)])
    told = _Told()
    await results.load_files(
        [file],
        columns=[name for name in _NAMES if name != "state"],
        constants={"state": "VA"},
        delta=LoadDelta(rows=[rows], replaces=[{"state": "VA"}]),
        on_commit=told,
    )
    (load_file,) = [str(name) for name, _ in store.loads]
    (deleted, _), (loaded, _) = told.commits
    assert read_delta_file(_volume / deleted[_TABLE]).prefixes == ({"state": "VA"},)
    assert loaded[_TABLE] == load_file
    assert [generations[_TABLE] for _, generations in told.commits] == [
        _token_of(log, deleted[_TABLE]),
        _token_of(log, load_file),
    ]


async def test_a_load_request_that_fails_hands_its_caller_no_file(_volume: Path) -> None:
    """Flips if a request that failed part way is handed to the caller: its file holds rows L3 never took, and its
    record names no file, so the fold must never hear of it."""
    world = _world()
    l3, log, _lock, _backend, _follower = world
    await _warmed(world)
    results, _ = _writer(l3, log, _volume)
    store = _FailingAfterAFileStore(l3, _volume)
    results.l3_pool = store
    first, first_rows = _export(store, "https://bucket/va-0.parquet", [_row("VA", 100, 1, "new", _LATER)])
    told = _Told()
    with pytest.raises(RuntimeError, match="second file"):
        await results.load_files(
            [first],
            columns=[name for name in _NAMES if name != "state"],
            constants={"state": "VA"},
            delta=LoadDelta(rows=[first_rows]),
            on_commit=told,
        )
    assert told.commits == []


# --- a failed load request's file, named unconfirmed (ruling C of 2026-11-23) ------------------------------------------


class _Confirmed:
    """the writer's fold as the recreator tells it each confirmed file (``DeltaFolds.confirmed``)."""

    def __init__(self) -> None:
        self.told: list[tuple[str, str, str]] = []

    def confirmed(self, table: str, *, file: str, epoch: str) -> None:
        self.told.append((table, file, epoch))


def _writer_l3(l3: _L3) -> _L3:
    """the writer's own L3: the same rows, its statements counted apart from the copy's."""
    writer = _L3.__new__(_L3)
    writer.db = l3.db
    writer.statements = 0
    return writer


async def test_a_load_failing_after_a_commit_leaves_copies_waiting_with_no_l3_read_then_current_once_the_writer_confirms(
    _volume: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Flips if a copy applies a failed load request's own file (rows L3 never took: the second file's), reads the
    load's prefixes or keys back from L3 itself (every pod at once, election night's load), or stays behind once the
    writer has confirmed the file. The request's first transaction commits, then it fails: the record names the file
    unconfirmed; the copy waits and asks; the writer reads the request's keys (from the file) from L3 by key, once,
    writes the file again holding exactly what L3 holds, marked confirmed, and tells its fold; the copy is then current,
    having read nothing from L3."""
    monkeypatch.setattr(copy_follower, "FILE_POLL_SECONDS", 0.01, raising=False)
    folds = _Confirmed()
    recreator: list[DeltaRecreator] = []

    async def recreate(table: str, file: str, epoch: str) -> str:
        return await recreator[0].recreate(table, file, epoch)

    world = _world(recreate=recreate)
    l3, log, lock, backend, follower = world
    writer_l3 = _writer_l3(l3)
    recreator.append(
        DeltaRecreator(
            [FollowedTable(name=_TABLE, key=_KEY, columns=_COLUMNS, scope=("state", "state:"), delta_files=True)],
            writer_l3,
            log,
            folds=folds,
            page_size=50,
        )
    )
    await _warmed(world)
    held = _copy(backend)
    results, _ = _writer(l3, log, _volume)
    store = _FailingAfterAFileStore(l3, _volume)
    results.l3_pool = store
    first, first_rows = _export(store, "https://bucket/va-0.parquet", [_row("VA", 100, 1, "new", _LATER)])
    second, second_rows = _export(store, "https://bucket/va-1.parquet", [_row("VA", 101, 2, "new", _LATER)])
    with pytest.raises(RuntimeError, match="second file"):
        await results.load_files(
            [first, second],
            columns=[name for name in _NAMES if name != "state"],
            constants={"state": "VA"},
            delta=LoadDelta(rows=[first_rows, second_rows]),
        )
    ((file, _),) = store.loads
    assert file is not None
    assert not read_delta_file_keys(_volume / file).confirmed, "the request's own file, holding both files' rows"
    lock.commit("state:VA")

    await follower.catch_up()

    assert l3.statements == 0, "a copy never reads a failed load request's rows from L3"
    assert _copy(backend) == held, "nothing of the request's file is applied before the writer confirms it"
    await asyncio.wait_for(_until(lambda: not follower.status().waiting_for_file), timeout=5)
    assert await follower.catch_up()

    assert await _same_now(backend, l3), "the copy holds what L3 holds: the first file's row, not the second's"
    assert ("VA", 100) in [row[:2] for row in _copy(backend)] and ("VA", 101) not in [row[:2] for row in _copy(backend)]
    assert l3.statements == 0
    assert writer_l3.statements == 1, "the writer read the request's two keys from L3 by key, once"
    confirmed = read_delta_file(_volume / file)
    assert read_delta_file_keys(_volume / file).confirmed
    assert sorted((change.row["geo_id"], change.deleted) for change in confirmed.changes) == [(100, False), (101, True)]
    assert folds.told == [(_TABLE, file, _token_of(log, file))], "the fold hears the rows that committed"


async def test_a_record_whose_unconfirmed_file_the_writer_has_confirmed_is_applied_with_no_l3_read_and_no_ask(
    _volume: Path,
) -> None:
    """Flips if a copy meeting a failed load request's record after the writer confirmed its file (a restart, a copy
    behind) waits or asks again, or reads L3: the confirmed file is applied as any load's."""
    asked: list[str] = []

    async def recreate(table: str, file: str, epoch: str) -> str:
        asked.append(file)
        return "failed"

    world = _world(recreate=recreate)
    l3, log, lock, backend, follower = world
    await _warmed(world)
    row = _row("VA", 100, 1, "new", _LATER)
    l3.insert([row])
    delta = TableDelta(name=_TABLE, types=_TYPES, key=_KEY)
    delta.rows([row], list(_NAMES))
    name = write_delta_file(_volume, delta, "confirmed", confirmed=True)
    await log.advance_with_change(_TABLE, None, prefixes=[{"state": "VA"}], unconfirmed_delta_file=name)
    lock.commit("state:VA")

    assert await follower.catch_up()

    assert await _same_now(backend, l3)
    assert l3.statements == 0
    assert asked == [] and not follower.status().waiting_for_file


class _PodSource:
    """a pod's generation source: its broker records each change, so it records none; counts every advance asked."""

    def __init__(self) -> None:
        self.advances = 0

    async def current(self, table_name: str) -> str:
        return f"inc:{self.advances}"

    async def advance(self, table_name: str) -> str:
        self.advances += 1
        return f"inc:{self.advances}"


class _UnansweredStore(_Store):
    """a pod's load request the broker answers as told (the rows it committed, then a failure), or never (cancelled)."""

    def __init__(self, l3: _L3, volume: Path, answer: int | None) -> None:
        super().__init__(l3, volume)
        self.answer = answer

    async def load_files(self, table: str, files: Sequence[LoadFile], **kwargs: Any) -> LoadResult:
        named = named_delta_files().get(table)
        self.loads.append((named, named is not None and (self.volume / named).is_file()))
        if self.answer is None:
            raise asyncio.CancelledError
        note_load_answer(self.answer)
        raise RuntimeError("the broker answered the load failed")


@pytest.mark.parametrize(("answer", "kept"), [(None, False), (0, False), (7, True)])
async def test_a_pods_failed_load_request_writes_no_record_of_its_own(
    _volume: Path, answer: int | None, kept: bool
) -> None:
    """Flips if a pod writes its own record for a failed load request of an opted-in table: one naming no file sent
    every copy to L3 for the load's prefixes, even for a cancel that committed nothing. The broker owns the record (it
    names the file unconfirmed when rows committed, so the file is kept for the writer to confirm; none when none did,
    so it is an orphan); a cancel with no answer writes none either (ruling C of 2026-11-23)."""
    registry = CollectionRegistry()
    source = _PodSource()
    registry.set_generation_source(source)
    results = _Results(registry, DefaultCoreConfig(), None)
    l3 = _L3()
    store = _UnansweredStore(l3, _volume, answer)
    results.l3_pool = store
    file, rows = _export(store, "https://bucket/va-0.parquet", [_row("VA", 100, 1, "new", _LATER)])
    raised: type[BaseException] = asyncio.CancelledError if answer is None else RuntimeError
    with pytest.raises(raised):
        await results.load_files(
            [file],
            columns=[name for name in _NAMES if name != "state"],
            constants={"state": "VA"},
            delta=LoadDelta(rows=[rows]),
        )
    ((name, saved),) = store.loads
    assert name is not None and saved
    assert source.advances == 0, "the pod advanced the table itself: a record of its own"
    assert (name in delta_files.OWN_DELTA_SAVES.orphans) is not kept
    assert name not in delta_files.OWN_DELTA_SAVES.in_flight
    delta_files.OWN_DELTA_SAVES.swept([name])


def _refused() -> float:
    return REGISTRY.get_sample_value("copy_follower_partition_refused_total", {"table": _TABLE}) or 0.0


async def test_a_copy_of_an_opted_in_table_refuses_a_prefix_record_naming_no_file_and_reads_no_l3(
    _volume: Path,
) -> None:
    """Flips if a record of an opted-in table's load naming no file (an older writer's or broker's) sends the copy to
    L3 for the load's prefix: every copy reading it at once is election night's load. The copy refuses it, counted
    and logged, and reads nothing from L3 (ruling C of 2026-11-23, step 5)."""
    world = _world()
    l3, log, lock, backend, follower = world
    await _warmed(world)
    held = _copy(backend)
    before = _refused()
    l3.insert([_row("VA", 100, 1, "new", _LATER)])
    await log.advance_with_change(_TABLE, None, prefixes=[{"state": "VA"}])
    lock.commit("state:VA")

    await follower.catch_up()

    assert l3.statements == 0, "the copy read the prefix back from L3"
    assert _copy(backend) == held
    assert _refused() - before == 1


async def test_a_fold_told_a_confirmed_file_holds_its_rows(_volume: Path) -> None:
    """Flips if the writer's fold never hears a failed load request's rows that committed once the writer confirmed
    its file: a load hands its caller only the requests that landed whole, so every full published after would lack
    them."""
    workers = ProcessWorkers(workers=1, name="test confirmed folds")
    try:
        publish_full_manifest(_volume, term=1, tables={_TABLE: FullTable(version="inc:0")})
        folds = DeltaFolds(
            _volume,
            [FoldedTable.of(_WRITER_SCHEMA, scope=("state", "state:"))],
            retention="delete",
            copies=NO_COPIES,
            workers=workers,
        )
        # the writer's own commit first, as under the lease: the fold's term is the writer's
        earlier = _save(_volume, "earlier", ("rows", [_row("VA", 99, 9, "new", _LATER)], list(_NAMES)))
        folds.committed({_TABLE: earlier}, {_TABLE: "inc:1"}, version=2, in_flight_from=None)
        delta = TableDelta(name=_TABLE, types=_TYPES, key=_KEY)
        delta.rows([_row("VA", 100, 1, "new", _LATER)], list(_NAMES))
        name = write_delta_file(_volume, delta, "confirmed", confirmed=True)
        folds.confirmed(_TABLE, file=name, epoch="inc:2")
        await folds.wait()
        await folds.close()
    finally:
        await workers.close()
    manifest = current_full_manifest(_volume)
    assert manifest is not None
    (part,) = [part for part in manifest.tables[_TABLE].parts if part.scope == "state:VA"]
    assert sorted(read_volume_file(_volume / part.file).column("geo_id").to_pylist()) == [99, 100]


# --- a load's intent record, written before its first commit (Pace's ruling of 2026-11-23) ------------------------------

_HUB = "write-ledger.aibots.hub-0195f3"


class _Beats:
    """the writer beats as ``lost_writes.WriterBeats`` reads them: each kept key, and every key asked after."""

    def __init__(self, *kept: str) -> None:
        self.kept = frozenset(kept)
        self.asked: list[str] = []

    async def gone(self, key: str) -> bool:
        self.asked.append(key)
        return key not in self.kept


def _intent_world(
    beats: _Beats | None, answers: list[str], folds: _Confirmed | None = None
) -> tuple[tuple[_L3, _Log, _Scopes, DuckDBBackend, CopyFollower], _L3, list[DeltaRecreator]]:
    """a copy whose writer is the recreator below, reading the hub's beats and walking the log for a superseding
    record; every answer the writer gave the copy kept in ``answers``."""
    recreator: list[DeltaRecreator] = []

    async def recreate(table: str, file: str, epoch: str) -> str:
        answer = await recreator[0].recreate(table, file, epoch)
        answers.append(answer)
        return answer

    world = _world(recreate=recreate)
    l3, log, _lock, _backend, _follower = world
    writer_l3 = _writer_l3(l3)
    recreator.append(
        DeltaRecreator(
            [FollowedTable(name=_TABLE, key=_KEY, columns=_COLUMNS, scope=("state", "state:"), delta_files=True)],
            writer_l3,
            log,
            folds=folds,
            page_size=50,
            writers=beats,
            generations=log,
        )
    )
    return world, writer_l3, recreator


def _load_file(volume: Path, write: str, rows: Sequence[dict[str, Any]]) -> str:
    """the pod's file for a load request of VA (an upsert): every row the request sends."""
    delta = TableDelta(name=_TABLE, types=_TYPES, key=_KEY)
    delta.rows(rows, list(_NAMES))
    return write_delta_file(volume, delta, write)


async def test_a_hub_crash_after_one_commit_leaves_the_intent_copies_wait_with_no_l3_read_and_the_writer_confirms_once_it_is_gone(
    _volume: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Flips if a hub that crashed after its first 50k commit leaves rows no record names, if a copy applies the
    request's own file (rows L3 never took) or reads L3 while the hub may still be loading, if the writer confirms the
    file while the hub's beat is kept (a hub still copying), or if it never confirms it once the beat is gone. The
    intent record names the file unconfirmed beside the hub's beat key; the copy waits (``file_unconfirmed``) and asks;
    the writer answers ``pending`` while the beat is kept, then, the beat deleted (the platform's recovery declared the
    hub dead), reads the request's keys from L3 by key once and writes the file confirmed; the copy is then current."""
    monkeypatch.setattr(copy_follower, "FILE_POLL_SECONDS", 0.01, raising=False)
    beats, answers, folds = _Beats(_HUB), list[str](), _Confirmed()
    world, writer_l3, _ = _intent_world(beats, answers, folds)
    l3, log, lock, backend, follower = world
    await _warmed(world)
    held = _copy(backend)
    sent = [_row("VA", 100, 1, "new", _LATER), _row("VA", 101, 2, "new", _LATER)]
    file = _load_file(_volume, "intent", sent)
    intent = await log.advance_with_change(
        _TABLE, None, prefixes=[{"state": "VA"}], unconfirmed_delta_file=file, unconfirmed_writer=_HUB
    )
    # the first transaction committed (its leading rows), then the hub died: no record of its own follows
    l3.insert([sent[0]])
    lock.commit("state:VA")

    await follower.catch_up()
    await asyncio.wait_for(_until(lambda: answers.count("pending") >= 3), timeout=5)

    assert "file_unconfirmed" in follower.status().waiting_for_file[_TABLE], "its own why, not key_not_in_file"
    assert _copy(backend) == held, "nothing of the request's file is applied while the hub may still be loading"
    assert l3.statements == 0 and writer_l3.statements == 0, "no copy and no writer reads L3 while the hub is alive"
    assert not read_delta_file_keys(_volume / file).confirmed, "never confirmed over a hub still copying"
    assert set(answers) == {"pending"} and set(beats.asked) == {_HUB}

    beats.kept = frozenset()
    await asyncio.wait_for(_until(lambda: not follower.status().waiting_for_file), timeout=5)
    assert await follower.catch_up()

    assert await _same_now(backend, l3), "the copy holds what L3 holds: the committed row, not the one L3 never took"
    assert l3.statements == 0
    assert writer_l3.statements == 1, "the writer read the request's keys from L3 by key, once"
    assert read_delta_file_keys(_volume / file).confirmed
    assert answers[-1] == "recreated" and folds.told == [(_TABLE, file, intent)]


async def test_a_normal_load_writes_the_intent_then_its_confirmed_record_and_is_applied_once_from_the_later_record(
    _volume: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Flips if the hub's confirmed record does not supersede its intent: the copy waiting for the hub's death (for
    ever, the hub being alive), the writer reading L3 or rewriting the hub's own file, or the file applied twice (once
    for the intent, once for the confirmed record). The copy ends the intent's wait with nothing heard
    (``superseded``) and applies the load from the confirmed record alone."""
    monkeypatch.setattr(copy_follower, "FILE_POLL_SECONDS", 0.01, raising=False)
    beats, answers, folds = _Beats(_HUB), list[str](), _Confirmed()
    world, writer_l3, _ = _intent_world(beats, answers, folds)
    l3, log, lock, backend, follower = world
    await _warmed(world)
    # the two places a load's file is heard by a copy, read from what each logs: a waited-for file read (its wait
    # ends "the file reads"), and the record the registry applies from its file
    caplog.set_level(logging.INFO, logger="threetears.core.collections")
    sent = [_row("VA", 100, 1, "new", _LATER), _row("VA", 101, 2, "new", _LATER)]
    file = _load_file(_volume, "whole", sent)
    intent = await log.advance_with_change(
        _TABLE, None, prefixes=[{"state": "VA"}], unconfirmed_delta_file=file, unconfirmed_writer=_HUB
    )
    await follower.catch_up()
    await asyncio.wait_for(_until(lambda: "pending" in answers), timeout=5)
    l3.insert(sent)
    confirmed = await log.advance_with_change(_TABLE, None, prefixes=[{"state": "VA"}], delta_file=file)
    lock.commit("state:VA")

    await asyncio.wait_for(_until(lambda: not follower.status().waiting_for_file), timeout=5)
    assert await follower.catch_up()

    assert await _same_now(backend, l3)
    assert l3.statements == 0 and writer_l3.statements == 0
    assert answers[-1] == "superseded"
    said = [(record.getMessage(), getattr(record, "extra_data", {})) for record in caplog.records]
    applied = [
        data["epoch"] for message, data in said if message.startswith("a load's or a clear's epoch applied from")
    ]
    heard = [data["epoch"] for message, data in said if data.get("why") == "the file reads"]
    assert applied == [confirmed] and heard == [], "the file is applied once, at the confirmed record"
    assert not read_delta_file_keys(_volume / file).confirmed, "the hub's own file, left as it is"
    assert folds.told == [] and intent != confirmed


async def test_a_load_the_hub_answered_failed_after_its_intent_is_confirmed_at_the_hubs_own_record_while_it_lives(
    _volume: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Flips if the hub's own record of a failed load (unconfirmed, naming no writer) does not supersede its intent:
    the copy then waits for a live hub's death, or the swallowed wait never applies the rows that committed. The wait
    moves to the hub's record, which the writer confirms from L3 at once."""
    monkeypatch.setattr(copy_follower, "FILE_POLL_SECONDS", 0.01, raising=False)
    beats, answers, folds = _Beats(_HUB), list[str](), _Confirmed()
    world, writer_l3, _ = _intent_world(beats, answers, folds)
    l3, log, lock, backend, follower = world
    await _warmed(world)
    sent = [_row("VA", 100, 1, "new", _LATER), _row("VA", 101, 2, "new", _LATER)]
    file = _load_file(_volume, "failed", sent)
    await log.advance_with_change(
        _TABLE, None, prefixes=[{"state": "VA"}], unconfirmed_delta_file=file, unconfirmed_writer=_HUB
    )
    l3.insert([sent[0]])
    stopped = await log.advance_with_change(_TABLE, None, prefixes=[{"state": "VA"}], unconfirmed_delta_file=file)
    lock.commit("state:VA")

    await follower.catch_up()
    await asyncio.wait_for(_until(lambda: not follower.status().waiting_for_file), timeout=5)
    assert await follower.catch_up()

    assert await _same_now(backend, l3)
    assert l3.statements == 0 and writer_l3.statements == 1
    assert folds.told == [(_TABLE, file, stopped)], "confirmed once, at the hub's own record"
    assert "pending" not in answers, "a hub still alive is never waited on once its own record names the file"


async def test_a_writer_given_no_beats_never_confirms_an_intent_the_hub_has_not_superseded(
    _volume: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Flips if a writer that reads no beats confirms an intent record over a hub that may still be loading."""
    monkeypatch.setattr(copy_follower, "FILE_POLL_SECONDS", 0.01, raising=False)
    answers: list[str] = []
    world, writer_l3, _ = _intent_world(None, answers)
    l3, log, lock, _backend, follower = world
    await _warmed(world)
    file = _load_file(_volume, "nobeats", [_row("VA", 100, 1, "new", _LATER)])
    await log.advance_with_change(
        _TABLE, None, prefixes=[{"state": "VA"}], unconfirmed_delta_file=file, unconfirmed_writer=_HUB
    )
    lock.commit("state:VA")
    await follower.catch_up()
    await asyncio.wait_for(_until(lambda: answers.count("pending") >= 2), timeout=5)
    assert writer_l3.statements == 0 and not read_delta_file_keys(_volume / file).confirmed
    follower.stop()
