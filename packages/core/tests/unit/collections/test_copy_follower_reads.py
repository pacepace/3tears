"""a follower over a copy published as shared Arrow reads its copy in query workers, never on a cursor here.

Chunk 3T-7 of ``design-copy-worker-backend.md``: the follower's own reads of its copy move off the serving process's
DuckDB cursor (``copy_reads``). Its checkpoint's state is pinned by ``CopyFollower.current`` (a version leased with its
mark, no cursor) and its parts are read and encoded in a query worker; a drift check's rows of a state are read and
fingerprinted in a query worker; a pass's keyed reads are ``CopyQueries.lookup``. The registry, the follower, the
delta files, the fulls and the query pool are the real ones; the copy is the in-process DuckDB that publishes each
applied state to a shared Arrow store with its mark, as a pod's backend does (``_PublishedCopy``). The contract this
pins:

- a checkpoint write and a drift check open no DuckDB connection and call nothing of DuckDB on any thread of the
  serving process (every state's read in each of the three ways a state reaches the copy's rows: its key's prefix, its
  part's keys, its scope column);
- the checkpoint they write loads back as the copy's rows, and the drift check finds every state equal;
- a pass's keyed reads of the copy (a delta file merged onto the rows the copy holds) answer as before, from a worker.
"""

from __future__ import annotations

import asyncio
import sys
import threading
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from contextlib import AbstractContextManager
from pathlib import Path
from types import FrameType, SimpleNamespace
from typing import Any

import pytest
from sqlalchemy import BigInteger, Column as SaColumn, MetaData, String, Table
from threetears.core.testing.l1_tables import L1_EVICTION_INFO, TEST_L1_TABLE

pa = pytest.importorskip("pyarrow")
duckdb = pytest.importorskip("duckdb")
ipc = pytest.importorskip("pyarrow.ipc")

from threetears.core.cache.duckdb import DuckDBBackend, StagedChange  # noqa: E402
from threetears.core.cache.duckdb_binds import run_bound  # noqa: E402
from threetears.core.collections import CollectionRegistry  # noqa: E402
from threetears.core.collections.copy_checkpoint import CopyCheckpoints  # noqa: E402
from threetears.core.collections.copy_follower import CopyFollower, FollowedTable  # noqa: E402
from threetears.core.collections.copy_queries import CopyQueries  # noqa: E402
from threetears.core.collections.delta_files import TableDelta, write_delta_file  # noqa: E402
from threetears.core.collections.epoch_change import ChangedRow, EpochChange, EpochWalk, walk_by_count  # noqa: E402
from threetears.core.collections.schema_backed import BIGINT_TYPE, STRING_TYPE, Column, rail_arrow_types  # noqa: E402
from threetears.core.collections.shared_copy import MARK_TABLE  # noqa: E402
from threetears.core.collections.volume_fulls import (  # noqa: E402
    FullTable,
    VolumeFulls,
    publish_full_manifest,
    write_full_part,
)
from threetears.core.process_workers import ProcessWorkers  # noqa: E402
from threetears.core.shared_arrow import SharedArrowStore  # noqa: E402

_RESULTS = "results"
_TALLIES = "tallies"
_OPTED = (_RESULTS, _TALLIES)
#: a table with no scope and no delta files, read from L3: its checkpoint part is patched by key
_RACES = "races"
_RACE_COLUMNS = (Column(name="race_id", column_type=BIGINT_TYPE), Column(name="name", column_type=STRING_TYPE))
#: the key with its state leading (a state read by that prefix of the key), with it second (read by the keys its
#: part holds), and without it (its state a row column: read by that column)
_KEYS = {"state_led": ("state", "geo_id"), "state_not_led": ("geo_id", "state"), "state_not_in_key": ("geo_id",)}
_COLUMNS = (
    Column(name="state", column_type=STRING_TYPE),
    Column(name="geo_id", column_type=BIGINT_TYPE),
    Column(name="votes", column_type=BIGINT_TYPE),
)
_NAMES = tuple(column.name for column in _COLUMNS)
_TYPES = rail_arrow_types(_COLUMNS)
#: each geo_id of one state only, so a key without the state names one row
_ROWS = [("VA", i, 10 + i) for i in range(4)] + [("IL", 10 + i, 20 + i) for i in range(4)]

_OPENED: list[Callable[[], None]] = []


@pytest.fixture(autouse=True)
def _close_what_the_test_opened() -> Iterator[None]:
    yield
    while _OPENED:
        _OPENED.pop()()


@pytest.fixture(autouse=True)
def _volume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """the shared volume the delta files and the fulls are saved on and read off."""
    monkeypatch.setenv("THREETEARS_DELTA_FILES_DIR", str(tmp_path / "volume"))
    (tmp_path / "volume").mkdir()
    return tmp_path / "volume"


@pytest.fixture
async def workers() -> AsyncIterator[ProcessWorkers]:
    made = ProcessWorkers(workers=1, name="test copy follower reads")
    try:
        yield made
    finally:
        await made.close()


@pytest.fixture
def store(tmp_path: Path) -> SharedArrowStore:
    return SharedArrowStore(tmp_path / "shm", max_bytes=1 << 30)


@pytest.fixture
async def queries(store: SharedArrowStore) -> AsyncIterator[CopyQueries]:
    pool = CopyQueries(store, workers=1)
    try:
        yield pool
    finally:
        await pool.close()


@pytest.fixture(params=sorted(_KEYS))
def key(request: pytest.FixtureRequest) -> tuple[str, ...]:
    return _KEYS[request.param]


class _DuckDBCalls:
    """every call into DuckDB this process makes (a connection opened, a cursor taken, a statement run, a table
    registered), on any thread: a profile hook on each, as ``test_copy_queries`` watches ``duckdb.connect``."""

    def __init__(self) -> None:
        self.seen: list[tuple[str, int]] = []
        self._lock = threading.Lock()

    def __call__(self, frame: FrameType, event: str, arg: Any) -> None:
        if event != "c_call" or getattr(arg, "__module__", None) != "_duckdb":
            return
        with self._lock:
            self.seen.append((getattr(arg, "__name__", repr(arg)), threading.get_ident()))

    def __enter__(self) -> _DuckDBCalls:
        sys.setprofile(self)
        threading.setprofile_all_threads(self)
        return self

    def __exit__(self, *exc: object) -> None:
        threading.setprofile_all_threads(None)
        sys.setprofile(None)


class _PublishedCopy:
    """the copy as a pod's backend keeps it: each change applied (here in the in-process DuckDB) and the applied state
    published to a shared Arrow store as a version with its mark, which the follower's reads then read."""

    def __init__(self, backend: DuckDBBackend, store: SharedArrowStore, tables: Sequence[str]) -> None:
        self.backend = backend
        self._store = store
        self._tables = tuple(tables)
        self._published = 0

    def stage_rows(self, stage: str, table: str, rows: Any, key: Sequence[str]) -> int:
        return int(self.backend.stage_rows(stage, table, rows, key))

    def stage_keys(
        self, stage: str, table: str, key: Sequence[str], keys: Any, *, rows_stage: str | None = None
    ) -> int:
        return int(self.backend.stage_keys(stage, table, key, keys, rows_stage=rows_stage))

    def unstage_keys(self, stages: Sequence[str], table: str, key: Sequence[str], keys: Any) -> int:
        return int(self.backend.unstage_keys(stages, table, key, keys))

    def drop_stage(self, *stages: str) -> None:
        self.backend.drop_stage(*stages)

    def staged_stages(self) -> list[str]:
        return list(self.backend.staged_stages())

    def apply_staged(self, changes: Sequence[StagedChange], *, mark: int | None = None) -> int:
        rows = int(self.backend.apply_staged(changes, mark=mark))
        with self.backend.read_snapshot() as cursor:
            tables = {
                name: cursor.execute(f'SELECT * FROM "{name}"').fetch_arrow_table()  # noqa: S608
                for name in self._tables
            }
        tables[MARK_TABLE] = pa.table({"mark": pa.array([mark], pa.int64())})
        self._store.publish(f"c-{self._published}", tables)
        self._published += 1
        return rows

    def read_snapshot(self) -> AbstractContextManager[Any]:
        return self.backend.read_snapshot()

    def staged_mark(self, cursor: Any) -> int | None:
        return self.backend.staged_mark(cursor)


def _followed(name: str, key: tuple[str, ...]) -> FollowedTable:
    return FollowedTable(name=name, key=key, columns=_COLUMNS, scope=("state", "state:"), delta_files=True)


def _races() -> FollowedTable:
    return FollowedTable(name=_RACES, key=("race_id",), columns=_RACE_COLUMNS)


def _metadata(key: tuple[str, ...]) -> MetaData:
    metadata = MetaData()
    Table(
        _RACES,
        metadata,
        SaColumn("race_id", BigInteger, primary_key=True),
        SaColumn("name", String(32)),
        info={L1_EVICTION_INFO: TEST_L1_TABLE},
    )
    kinds: dict[str, Any] = {"state": String(8), "geo_id": BigInteger}
    for name in _OPTED:
        Table(
            name,
            metadata,
            *(SaColumn(column, kinds[column], primary_key=True) for column in key),
            *(SaColumn(column, kinds[column]) for column in ("state",) if column not in key),
            SaColumn("votes", BigInteger),
            info={L1_EVICTION_INFO: TEST_L1_TABLE},
        )
    return metadata


def _dict_rows(result: Any) -> list[dict[str, Any]]:
    names = [d[0] for d in result.description]
    return [dict(zip(names, row, strict=True)) for row in result.fetchall()]


class _L3:
    """L3 as a DuckDB answering ``fetch``; counts every statement it answers."""

    def __init__(self, key: tuple[str, ...]) -> None:
        self.key = key
        self.db = duckdb.connect(":memory:")
        _OPENED.append(self.db.close)
        for name in _OPTED:
            self.db.execute(
                f"CREATE TABLE {name} (state VARCHAR, geo_id BIGINT, votes BIGINT, PRIMARY KEY ({', '.join(key)}))"
            )
        self.db.execute(f"CREATE TABLE {_RACES} (race_id BIGINT PRIMARY KEY, name VARCHAR)")
        self.statements = 0

    def race(self, race_id: int, name: str) -> None:
        self.db.execute(f"INSERT OR REPLACE INTO {_RACES} VALUES (?, ?)", [race_id, name])  # noqa: S608

    def put(self, table: str, rows: list[tuple[str, int, int]]) -> None:
        put = pa.table(
            {
                "state": pa.array([r[0] for r in rows], pa.string()),
                "geo_id": pa.array([r[1] for r in rows], pa.int64()),
                "votes": pa.array([r[2] for r in rows], pa.int64()),
            }
        )
        self.db.register("_put", put)
        try:
            self.db.execute(f"INSERT OR REPLACE INTO {table} SELECT state, geo_id, votes FROM _put")  # noqa: S608
        finally:
            self.db.unregister("_put")

    def held(self, table: str) -> list[tuple[Any, ...]]:
        return sorted(self.db.execute(f"SELECT {', '.join(_NAMES)} FROM {table}").fetchall())  # noqa: S608

    async def fetch(self, sql: str, *params: Any) -> list[dict[str, Any]]:
        await asyncio.sleep(0)
        self.statements += 1
        return run_bound(self.db.cursor(), sql, params, _dict_rows)


class _Log:
    """each table's write generation and its change records."""

    def __init__(self) -> None:
        self.count: dict[str, int] = dict.fromkeys((*_OPTED, _RACES), 1)
        self.records: dict[tuple[str, str], EpochChange] = {}

    async def current(self, table: str) -> str:
        return f"inc:{self.count[table]}"

    def commit(self, table: str, keys: list[tuple[str, ...]], delta_file: str | None) -> str:
        self.count[table] += 1
        token = f"inc:{self.count[table]}"
        self.records[(table, token)] = EpochChange(
            table=table, token=token, rows=[ChangedRow(ids=list(ids)) for ids in keys], delta_file=delta_file
        )
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


class _Objects:
    """an Object Store in memory."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    async def require_room(self, objects: Sequence[tuple[str, int]]) -> int:
        return sum(size for _, size in objects)

    async def put(self, name: str, data: bytes) -> None:
        self.objects[name] = data

    async def get(self, name: str) -> bytes:
        return self.objects[name]

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


def _part_rows(rows: list[tuple[str, int, int]]) -> Any:
    return pa.table(
        {
            "state": pa.array([r[0] for r in rows], pa.string()),
            "geo_id": pa.array([r[1] for r in rows], pa.int64()),
            "votes": pa.array([r[2] for r in rows], pa.int64()),
        }
    )


class _World:
    def __init__(
        self,
        volume: Path,
        workers: ProcessWorkers,
        store: SharedArrowStore,
        queries: CopyQueries,
        key: tuple[str, ...],
    ) -> None:
        self.volume = volume
        self.key = key
        self.l3, self.log, self.lock = _L3(key), _Log(), _Scopes()
        self.backend = DuckDBBackend()
        _OPENED.append(self.backend.reset)
        self.backend.initialize(_metadata(key))
        self.copy = _PublishedCopy(self.backend, store, (*_OPTED, _RACES))
        self.registry = CollectionRegistry()
        for name in (*_OPTED, _RACES):
            self.registry.follow_generation(name)
        opted = [_followed(name, key) for name in _OPTED]
        tables = [*opted, _races()]
        self.objects, self.pointers = _Objects(), _Pointers()
        # the checkpoint is written by the test through its front door (CopyCheckpoints.write of CopyFollower.current),
        # of every table, as a follower's own checkpoints write theirs
        self.checkpoints = CopyCheckpoints(
            name="copy",
            tables=tables,
            backend=self.backend,
            store=self.objects,
            pointers=self.pointers,
            retire=self.objects.retire,
            owner="a",
            encode_stamp=lambda stamp: {"version": stamp.version, "epochs": dict(stamp.epochs)},
        )
        self.follower = CopyFollower(
            self.registry,
            tables,
            self.copy,
            self.l3,
            settled=self.lock.settled,
            generation=self.log.current,
            changes=self.log,
            page_size=50,
            stamp_table="scope_epochs",
            fulls=VolumeFulls(volume, opted, poll_seconds=0.01),
            drift_check_seconds=0,
            drift_workers=workers,
            queries=queries,
        )
        self.writes = 0

    async def started(self) -> None:
        """L3 and the writer's first full at inc:1; the copy starts from it."""
        for name in _OPTED:
            self.l3.put(name, _ROWS)
        for race_id, name in ((1, "gov"), (2, "sen"), (3, "house")):
            self.l3.race(race_id, name)
        publish_full_manifest(
            self.volume,
            term=1,
            tables={
                name: FullTable(
                    version="inc:1",
                    parts=tuple(
                        write_full_part(
                            self.volume,
                            name,
                            f"state:{state}",
                            _part_rows([r for r in _ROWS if r[0] == state]),
                            "inc:1",
                        )
                        for state in ("IL", "VA")
                    ),
                )
                for name in _OPTED
            },
        )
        await self.follower.start()
        assert await self.follower.catch_up()
        self.l3.statements = 0

    def write(self, table: str, rows: list[tuple[str, int, int]]) -> None:
        """a commit: L3 changed and its delta file saved before the record names it."""
        self.l3.put(table, rows)
        self.writes += 1
        delta = TableDelta(name=table, types=_TYPES, key=self.key)
        delta.rows([dict(zip(_NAMES, row, strict=True)) for row in rows], list(_NAMES))
        delta_file = write_delta_file(self.volume, delta, f"w{self.writes}")
        named = [dict(zip(_NAMES, row, strict=True)) for row in rows]
        self.log.commit(table, [tuple(str(row[column]) for column in self.key) for row in named], delta_file)
        self.lock.commit(*{f"state:{r[0]}" for r in rows})

    def race(self, race_id: int, name: str) -> None:
        """a commit to the table read from L3: its record names the key, no file."""
        self.l3.race(race_id, name)
        self.log.commit(_RACES, [(str(race_id),)], None)
        self.lock.commit()

    def held(self, table: str) -> list[tuple[Any, ...]]:
        columns = "race_id, name" if table == _RACES else ", ".join(_NAMES)
        with self.backend.read_snapshot() as cursor:
            return sorted(cursor.execute(f"SELECT {columns} FROM {table}").fetchall())  # noqa: S608


def _checkpointed_rows(objects: _Objects, written: Any, table: str) -> list[tuple[Any, ...]]:
    """every row the checkpoint's parts of ``table`` hold."""
    columns = ("race_id", "name") if table == _RACES else _NAMES
    rows: list[tuple[Any, ...]] = []
    for name, _scope, _rows in written.tables[table].parts:
        part = ipc.open_stream(objects.objects[name]).read_all()
        rows.extend(zip(*(part.column(column).to_pylist() for column in columns), strict=True))
    return sorted(rows)


async def test_a_checkpoint_write_and_a_drift_check_open_no_duckdb_in_the_serving_process(
    _volume: Path, workers: ProcessWorkers, store: SharedArrowStore, queries: CopyQueries, key: tuple[str, ...]
) -> None:
    """Flips if a checkpoint write or a drift check reads the copy on a DuckDB connection or cursor of this process
    (the old follower's ``reading_marked`` and ``copy_rows_of_state`` read on the backend's snapshot cursor, and a
    shared copy's snapshot held an engine slot of the serving process for each): any call into DuckDB on any thread
    here while they run. Also flips if what they read is wrong: the checkpoint's parts must hold the copy's rows of
    every table, and every state must check equal, in each of the three ways a state reaches the copy's rows."""
    world = _World(_volume, workers, store, queries, key)
    await world.started()
    tables = (*_OPTED, _RACES)
    # a worker of each pool started before the spy: forking them is not what is watched
    async with world.follower.current() as (state, _stamp, _marks):
        assert state.mark is not None
    calls = _DuckDBCalls()
    # the first checkpoint reads every table whole (its cold start), in a query worker
    with calls:
        first = await world.checkpoints.write(world.follower.current)
    assert first is not None, "the first checkpoint was not written"
    for name in tables:
        assert _checkpointed_rows(world.objects, first, name) == world.held(name)
    # a change from a delta file (the pass's keyed read of the rows the copy holds, its merge's base, ran in a query
    # worker) and one read from L3 by key; applied outside the watch: the in-process apply here is the test's backend
    world.write(_RESULTS, [("VA", 1, 111)])
    world.race(2, "senate")
    assert await world.follower.catch_up()
    assert world.held(_RESULTS) == world.l3.held(_RESULTS)
    world.checkpoints.applied(
        2, {_RESULTS: frozenset({"state:VA"}), _RACES: frozenset()}, {_RACES: pa.table({"race_id": ["2"]})}
    )
    world.l3.statements = 0
    # the second rewrites VA's part by its scope and patches races' one part by its key, each in a query worker; the
    # drift check reads and fingerprints each state in one
    with calls:
        second = await world.checkpoints.write(world.follower.current)
        outcomes = await world.follower.check_drift()

    assert calls.seen == [], f"the serving process called into DuckDB: {sorted({name for name, _ in calls.seen})}"
    assert second is not None, "the second checkpoint was not written"
    for name in tables:
        assert _checkpointed_rows(world.objects, second, name) == world.held(name)
    assert (2, "senate") in _checkpointed_rows(world.objects, second, _RACES)
    # results' VA is held at inc:2, past its part's inc:1: skipped, never read as drift; every other state equal
    assert outcomes == {
        (_RESULTS, "state:IL"): "equal",
        (_RESULTS, "state:VA"): "skipped",
        (_TALLIES, "state:IL"): "equal",
        (_TALLIES, "state:VA"): "equal",
    }
    assert world.l3.statements == 0
