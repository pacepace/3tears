"""Integration: a copy follower's few whole reads cost what a bare read of the same rows costs, at real size.

The whole reads the no-whole-reads ledger keeps, each held within 2x its bare operation timed on the same database
(:func:`threetears.core.testing.calibrate.budget_problem`):

- ``CopyFollower._stage_whole`` (a cold start, a reset, a schema change): reading a table from L3 a keyset page at a
  time and staging it beside the copy, against the bare keyset read of the same pages and their copy-in to DuckDB;
- ``delta_fold._cold_start_pages`` (the writer's rebuild of a full from L3: no full on the volume, a reset, a schema
  change): the table read a keyset page at a time and written as a part a state, against the bare keyset read of the
  same pages and their parts;
- ``CopyCheckpoints.stage`` (a start from a checkpoint): every part read back from the Object Store and staged, against
  the bare decode and copy-in of the same parts into DuckDB;
- ``copy_checkpoint._batches`` (a checkpoint write at a table's cold start): the table read from the copy and stored
  as parts, against the bare read of the same rows and their encoding.

And the one keyed patch beside them: ``copy_checkpoint._patched_part`` (a checkpoint write of a table with no scope:
its one part's untouched rows kept through ``keyed.anti``, the touched keys' rows read from the copy by key), against
the bare work of the same patch (the touched keys' keyed read, the part's untouched rows by DuckDB's own ANTI JOIN).

Timed runs belong under measure-lock. Against the session ``db_container`` (Postgres); set
``THREETEARS_TEST_POSTGRES_URL`` to a YugabyteDB URL to run them there. ``THREETEARS_FOLLOW_SPEED_ROWS`` sets the rows
(default 1,000,000: a long table of an election night's copy).
"""

from __future__ import annotations

import asyncio
import os
import uuid
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import asyncpg
import pytest
from sqlalchemy import BigInteger, Column as SaColumn, MetaData, String, Table
from threetears.core.testing.l1_tables import L1_EVICTION_INFO, TEST_L1_TABLE

duckdb = pytest.importorskip("duckdb")
pa = pytest.importorskip("pyarrow")
ipc = pytest.importorskip("pyarrow.ipc")
pc = pytest.importorskip("pyarrow.compute")

from threetears.core.cache.duckdb import STAGE_PREFIX, DuckDBBackend, StagedChange  # noqa: E402
from threetears.core.cache.duckdb_binds import run_bound  # noqa: E402
from threetears.core.collections import CollectionRegistry  # noqa: E402
from threetears.core.collections.complete_copy import DEFAULT_PAGE_SIZE, rail_arrow_types  # noqa: E402
from threetears.core.collections.copy_checkpoint import CopyCheckpoints  # noqa: E402
from threetears.core.collections.copy_follower import CopyFollower, FollowedTable  # noqa: E402
from threetears.core.testing.published_copy import PublishedCopy, close_published_copies  # noqa: E402
from threetears.core.collections.copy_reads import PinnedCopy, QueriedReads  # noqa: E402
from threetears.core.collections.delta_fold import DeltaFolds, FoldedTable  # noqa: E402
from threetears.core.testing.delta_copies import NO_COPIES  # noqa: E402
from threetears.core.collections.epoch_change import EpochChange, EpochWalk, walk_by_count  # noqa: E402
from threetears.core.collections.schema_backed import BIGINT_TYPE, STRING_TYPE, Column, TableSchema  # noqa: E402
from threetears.core.collections.volume_fulls import write_full_part  # noqa: E402
from threetears.core.keyset import keyset_arrow_pages  # noqa: E402
from threetears.core.sql_fragments import unnest_key_join  # noqa: E402
from threetears.core.testing.calibrate import best_seconds, budget_problem  # noqa: E402

pytestmark = pytest.mark.integration

_TABLE = "votes"
_KEY = ("state", "geo_id")
_COLUMNS = (
    Column(name="state", column_type=STRING_TYPE),
    Column(name="geo_id", column_type=BIGINT_TYPE),
    Column(name="votes", column_type=BIGINT_TYPE),
)
_ROWS = int(os.environ.get("THREETEARS_FOLLOW_SPEED_ROWS", "1000000"))
#: the states the rows spread over: a checkpoint's scopes
_STATES = 50
_TABLE_DEF = FollowedTable(name=_TABLE, key=_KEY, columns=_COLUMNS, scope=("state", "state:"))


@pytest.fixture(autouse=True)
async def _published_copies_closed() -> AsyncIterator[None]:
    """every copy a test published (``testing.published_copy``) closed after it: its query workers and its store."""
    yield
    await close_published_copies()


def _metadata() -> MetaData:
    metadata = MetaData()
    Table(
        _TABLE,
        metadata,
        SaColumn("state", String(8), primary_key=True),
        SaColumn("geo_id", BigInteger, primary_key=True),
        SaColumn("votes", BigInteger),
        info={L1_EVICTION_INFO: TEST_L1_TABLE},
    )
    return metadata


def _rows() -> Any:
    per = _ROWS // _STATES
    return pa.table(
        {
            "state": pa.array([f"S{s:02d}" for s in range(_STATES) for _ in range(per)], pa.string()),
            "geo_id": pa.array([g for _ in range(_STATES) for g in range(per)], pa.int64()),
            "votes": pa.array([1] * (per * _STATES), pa.int64()),
        }
    )


@dataclass
class _Db:
    pool: asyncpg.Pool


@pytest.fixture
async def db(db_container: str) -> AsyncIterator[_Db]:
    """the table at real size, in its own schema."""
    schema = f"follow_whole_{uuid.uuid4().hex[:8]}"
    conn = await asyncpg.connect(db_container, command_timeout=1800)
    try:
        await conn.execute(f"CREATE SCHEMA {schema}")
        await conn.execute(
            f"CREATE TABLE {schema}.{_TABLE} (state TEXT, geo_id BIGINT, votes BIGINT, PRIMARY KEY (state, geo_id))"
        )
        await conn.execute(
            f"INSERT INTO {schema}.{_TABLE} SELECT 'S' || lpad((g % $2)::text, 2, '0'), g, 1 "  # noqa: S608
            "FROM generate_series(1, $1) g",
            _ROWS,
            _STATES,
        )
        await conn.execute(f"ANALYZE {schema}.{_TABLE}")
        pool = await asyncpg.create_pool(
            db_container, min_size=1, max_size=4, command_timeout=1800, server_settings={"search_path": schema}
        )
        try:
            yield _Db(pool=pool)
        finally:
            await pool.close()
    finally:
        await conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        await conn.close()


class _Log:
    def __init__(self) -> None:
        self.records: dict[str, EpochChange] = {}

    async def current(self, table: str) -> str:
        return "inc:0"

    async def change(self, table_name: str, token: str) -> EpochChange | None:
        return self.records.get(token)

    async def changes_after(self, table_name: str, after: str, upto: str) -> EpochWalk:
        return await walk_by_count(self, table_name, after, upto)


class _Store:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    async def put(self, name: str, data: bytes) -> None:
        self.objects[name] = data

    async def get(self, name: str) -> bytes:
        return self.objects[name]

    async def require_room(self, objects: Sequence[tuple[str, int]]) -> int:
        """no bound: the room a checkpoint asks for is always there."""
        return sum(size for _, size in objects)

    async def list_names(self, *, prefix: str = "") -> list[str]:
        return [n for n in self.objects if n.startswith(prefix)]

    async def retire(self, names: list[str]) -> None:
        for name in names:
            self.objects.pop(name, None)


class _Pointers:
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


def _backend() -> DuckDBBackend:
    backend = DuckDBBackend()
    backend.initialize(_metadata())
    return backend


def _filled() -> DuckDBBackend:
    """a copy holding the rows, applied with mark 1 (a reader's stamp names it). Synchronous, a Python pass a row and the
    in-process apply: a test runs it on a thread (asyncio.to_thread), never on the loop."""
    backend = _backend()
    stage = f"{STAGE_PREFIX}{_TABLE}__rows"
    backend.stage_rows(stage, _TABLE, _rows(), _KEY)
    backend.apply_staged(
        [StagedChange(table=_TABLE, key=_KEY, rows=stage, whole=True, whole_kind="cold_start")], mark=1
    )
    return backend


def _checkpoints(backend: DuckDBBackend, store: _Store, pointers: _Pointers) -> CopyCheckpoints:
    return CopyCheckpoints(
        name="speed",
        tables=[_TABLE_DEF],
        backend=backend,
        store=store,
        pointers=pointers,
        retire=store.retire,
        owner="a",
    )


def _reader(backend: DuckDBBackend) -> Any:
    published = PublishedCopy(backend)

    @asynccontextmanager
    async def read() -> AsyncIterator[tuple[PinnedCopy, Any, Any]]:
        # the copy's state now, published and pinned as its follower pins it (a leased version, read in a query worker)
        await asyncio.to_thread(published.publish)
        state = await QueriedReads(published.queries).current()
        try:
            yield state, 1, {_TABLE: "inc:0"}
        finally:
            await state.close()

    return read


@pytest.mark.loop_timing
async def test_stage_whole_speed_at_real_size(db: _Db) -> None:
    """a cold start, end to end (CopyFollower.warm and the catch-up that applies it): the whole read of the table from
    L3 declared in CopyFollower._stage_cold_start (and its twins _stage_schema_change and _stage_legacy_whole_record,
    the same read), which CopyFollower._stage_whole stages, and the one apply (DuckDBBackend.apply_staged, its declared
    empty of the table and the rows' copy-in), within 2x a bare keyset
    read of the same pages and their copy-in to DuckDB (the work a cold start must do: the rows read, and held)."""
    types = rail_arrow_types(_COLUMNS)
    selected = ", ".join(types)

    def copy_in(con: Any, page: Any, first: bool) -> None:
        con.register("page", page)
        try:
            con.execute("CREATE TABLE plain AS SELECT * FROM page" if first else "INSERT INTO plain SELECT * FROM page")
        finally:
            con.unregister("page")

    async def bare() -> None:
        con = duckdb.connect(":memory:")
        try:
            first = True
            # the key's lead (state, TEXT) is hashed on YugabyteDB, as the follower's declaration says: the bare read
            # walks it the way the follower does, or every page of the baseline would read the whole table there
            async for page in keyset_arrow_pages(
                db.pool,
                f"SELECT {selected} FROM {_TABLE}",
                _KEY,
                types=types,
                page_size=DEFAULT_PAGE_SIZE,
                lead_array="text[]",
            ):
                await asyncio.to_thread(copy_in, con, page, first)
                first = False
        finally:
            con.close()

    async def cold_start() -> None:
        registry = CollectionRegistry()
        registry.follow_generation(_TABLE)
        log = _Log()
        published = PublishedCopy(_backend())
        follower = CopyFollower(
            registry,
            [_TABLE_DEF],
            published,
            db.pool,
            settled=lambda: asyncio.sleep(0, 1),
            generation=log.current,
            changes=log,
            stamp_table="scope_epochs",
            queries=published.queries,
        )
        follower.register()
        await follower.warm()
        assert await follower.catch_up()

    baseline = await best_seconds(bare)
    took = await best_seconds(cold_start)
    print(
        f"cold start of {_ROWS} rows {took:.2f} s, bare keyset read and copy-in {baseline:.2f} s, {took / baseline:.2f}x"
    )
    problem = budget_problem(f"a cold start (_stage_whole) of {_ROWS} rows", took, baseline)
    assert problem is None, problem


class _Generations:
    async def current(self, table_name: str) -> str:
        return "inc:1"

    async def advance(self, table_name: str) -> str | None:
        raise AssertionError("a rebuild advances nothing")


@pytest.mark.loop_timing
async def test_rebuild_speed_at_real_size(db: _Db, tmp_path: Path) -> None:
    """the writer's rebuild of a table's full from L3 (DeltaFolds.start): the whole read declared in
    DeltaFolds._fresh_fulls and delta_fold._cold_start_pages (and its twin _schema_change_pages, the same read), its
    rows written beside the full, then split into a part a state and published in a worker process, within 2x a bare keyset read of the same pages
    and the same rows written as a part a state (the work a rebuild must do: the rows read, and held on the volume)."""
    types = rail_arrow_types(_COLUMNS)
    selected = ", ".join(types)
    schema = TableSchema(
        name=_TABLE, primary_key=_KEY, columns=list(_COLUMNS), l1_eviction=TEST_L1_TABLE, delta_files=True
    )
    runs = iter(range(1_000_000))

    def write_parts(root: Path, rows: Any) -> None:
        states = rows.column("state")
        for value in pc.unique(states).to_pylist():
            write_full_part(root, _TABLE, f"state:{value}", rows.filter(pc.equal(states, value)), "inc:1")

    async def bare() -> None:
        pages = [
            page
            async for page in keyset_arrow_pages(
                db.pool,
                f"SELECT {selected} FROM {_TABLE}",
                _KEY,
                types=types,
                page_size=DEFAULT_PAGE_SIZE,
                lead_array="text[]",
            )
        ]
        await asyncio.to_thread(write_parts, tmp_path / f"bare-{next(runs)}", pa.concat_tables(pages))

    async def rebuild() -> None:
        folds = DeltaFolds(
            tmp_path / f"rebuild-{next(runs)}",
            [FoldedTable.of(schema, scope=("state", "state:"))],
            retention="delete",
            copies=NO_COPIES,
        )
        try:
            assert await folds.start(db.pool, _Generations(), changes=_Log(), version=1) == (_TABLE,)
        finally:
            await folds.close()

    baseline = await best_seconds(bare)
    took = await best_seconds(rebuild)
    print(f"rebuild of {_ROWS} rows {took:.2f} s, bare keyset read and parts {baseline:.2f} s, {took / baseline:.2f}x")
    problem = budget_problem(f"a rebuild (_cold_start_pages) of {_ROWS} rows", took, baseline)
    assert problem is None, problem


@pytest.mark.loop_timing
async def test_checkpoint_stage_speed_at_real_size() -> None:
    """CopyCheckpoints.stage (a start's load of every table from L2) within 2x the bare decode and copy-in of the same
    parts into DuckDB."""
    source = await asyncio.to_thread(_filled)
    store, pointers = _Store(), _Pointers()
    assert await _checkpoints(source, store, pointers).write(_reader(source)) is not None
    parts = [store.objects[name] for name in sorted(store.objects)]

    def copy_in() -> None:
        con = duckdb.connect(":memory:")
        for index, data in enumerate(parts):
            part = ipc.open_stream(data).read_all()
            con.register("part", part)
            if index == 0:
                con.execute("CREATE TABLE plain AS SELECT * FROM part")
            else:
                con.execute("INSERT INTO plain SELECT * FROM part")
            con.unregister("part")
        con.close()

    async def bare() -> None:
        await asyncio.to_thread(copy_in)

    async def stage() -> None:
        target = _backend()
        staged = await _checkpoints(target, store, pointers).stage()
        assert staged.rows == _ROWS

    baseline = await best_seconds(bare)
    took = await best_seconds(stage)
    print(f"CopyCheckpoints.stage of {_ROWS} rows {took:.2f} s, bare copy-in {baseline:.2f} s, {took / baseline:.2f}x")
    problem = budget_problem(f"CopyCheckpoints.stage of {_ROWS} rows", took, baseline)
    assert problem is None, problem


@pytest.mark.loop_timing
async def test_checkpoint_whole_write_speed_at_real_size() -> None:
    """a checkpoint write at a table's cold start, whose whole read of the copy is copy_checkpoint._batches_cold_start
    (the same statement _batches_schema_change and _batches_rolling_upgrade run), within 2x the bare read of the same
    rows and their encoding as parts."""
    source = await asyncio.to_thread(_filled)

    def read_and_encode() -> None:
        with source.read_snapshot() as cursor:
            reader = cursor.execute(f"SELECT * FROM {_TABLE}").fetch_record_batch(250_000)  # noqa: S608
            for batch in reader:
                sink = pa.BufferOutputStream()
                with ipc.new_stream(sink, batch.schema, options=ipc.IpcWriteOptions(compression="zstd")) as writer:
                    writer.write_batch(batch)

    async def bare() -> None:
        await asyncio.to_thread(read_and_encode)

    async def write() -> None:
        written = await _checkpoints(source, _Store(), _Pointers()).write(_reader(source))
        assert written is not None and written.rows == _ROWS

    baseline = await best_seconds(bare)
    took = await best_seconds(write)
    print(
        f"cold-start write of {_ROWS} rows {took:.2f} s, bare read and encode {baseline:.2f} s, {took / baseline:.2f}x"
    )
    problem = budget_problem(f"a cold-start checkpoint write (_batches_cold_start) of {_ROWS} rows", took, baseline)
    assert problem is None, problem


#: a table with no scope at its largest: one part (copy_checkpoint's default part_rows)
_PART_ROWS = 250_000
#: the keys one checkpoint write touched in it: a tenth of its rows, half of them changed and half new
_PART_TOUCHED = 25_000


def _patch_rows() -> tuple[list[tuple[str, int]], list[tuple[str, int]], Any, Any]:
    """the patch test's part before and after its change, with the keys changed in place and the keys added."""
    per = _PART_ROWS // _STATES
    states = [f"S{s:02d}" for s in range(_STATES) for _ in range(per)]
    geos = [g for _ in range(_STATES) for g in range(per)]
    # the copy after the change: half the touched keys changed in place, half added past the part's rows
    changed = _PART_TOUCHED // 2
    step = _PART_ROWS // changed
    touched = [(states[i * step], geos[i * step]) for i in range(changed)]
    added = [(f"S{(i % _STATES):02d}", per + i) for i in range(_PART_TOUCHED - changed)]
    changed_at = set(touched)
    before = pa.table(
        {
            "state": pa.array(states, pa.string()),
            "geo_id": pa.array(geos, pa.int64()),
            "votes": pa.array([1] * len(states), pa.int64()),
        }
    )
    after = pa.table(
        {
            "state": pa.array(states + [s for s, _ in added], pa.string()),
            "geo_id": pa.array(geos + [g for _, g in added], pa.int64()),
            "votes": pa.array(
                [2 if key in changed_at else 1 for key in zip(states, geos, strict=True)] + [2] * len(added),
                pa.int64(),
            ),
        }
    )
    return touched, added, before, after


@pytest.mark.loop_timing
async def test_checkpoint_patch_speed_at_real_size() -> None:
    """a checkpoint write of a table with no scope (CopyCheckpoints.write patching its one part,
    copy_checkpoint._patched_part: the part's untouched rows kept by keyed.anti over the part and the staged touched
    keys, the touched keys' rows read from the copy by key) of a 250,000-row part with 25,000 keys touched, within 2x
    the bare work of the same patch: the part decoded, the touched keys' keyed read of the copy, the part's untouched
    rows by DuckDB's own ANTI JOIN to the keys, put together and encoded as a part."""
    table = FollowedTable(name=_TABLE, key=_KEY, columns=_COLUMNS)
    # the setup's rows are built and applied on a thread, never on the loop: building them is a Python pass a row and
    # their apply is the in-process DuckDB's (a follower's apply is its apply worker's, copy_stages.SharedCopyStages),
    # either one a loop turn past threetears.core.testing.loop_stall's limit at this size
    touched, added, before, after = await asyncio.to_thread(_patch_rows)
    backend = await asyncio.to_thread(_backend)
    stage = f"{STAGE_PREFIX}{_TABLE}__rows"
    stamp = [1]

    def put(rows: Any) -> None:
        backend.stage_rows(stage, _TABLE, rows, _KEY)
        backend.apply_staged(
            [StagedChange(table=_TABLE, key=_KEY, rows=stage, whole=True, whole_kind="cold_start")], mark=stamp[0]
        )

    published = PublishedCopy(backend)

    @asynccontextmanager
    async def read() -> AsyncIterator[tuple[PinnedCopy, Any, Any]]:
        await asyncio.to_thread(published.publish)
        state = await QueriedReads(published.queries).current()
        try:
            yield state, stamp[0], {_TABLE: f"inc:{stamp[0]}"}
        finally:
            await state.close()

    await asyncio.to_thread(put, before)
    store, pointers = _Store(), _Pointers()
    checkpoints = CopyCheckpoints(
        name="patch",
        tables=[table],
        backend=backend,
        store=store,
        pointers=pointers,
        retire=store.retire,
        owner="a",
        part_rows=_PART_ROWS,
    )
    first = await checkpoints.write(read)
    assert first is not None
    [(name, _, _)] = first.tables[_TABLE].parts
    part = store.objects[name]
    stamp[0] = 2
    await asyncio.to_thread(put, after)
    keys = [(s, str(g)) for s, g in touched + added]
    bound = [[s for s, _ in keys], [g for _, g in keys]]
    # the touched keys as the follower hands them to the checkpoint: one Arrow table of the key columns as text
    touched_keys = pa.table({column: pa.array(values, pa.string()) for column, values in zip(_KEY, bound, strict=True)})
    pk_unnest, pk_join = unnest_key_join(_KEY, [table.key_cast(column) for column in _KEY])

    def bare_patch() -> None:
        old = ipc.open_stream(part).read_all()
        with backend.read_snapshot() as cursor:
            # the keys bound through Arrow, as the patch binds them (run_bound: a list bound as Python values is
            # probed value by value, 100x the read itself at 25,000 keys)
            fresh = run_bound(
                cursor,
                f"SELECT t.* FROM {_TABLE} AS t JOIN (SELECT {pk_unnest}) AS k ON {pk_join}",  # noqa: S608
                bound,
                lambda result: result.fetch_arrow_table(),
            )
            # the untouched rows by DuckDB's own ANTI JOIN to the unnested keys: the filter a patch cannot skip, in
            # its barest form (a plain read back of the part, 6 ms here, leaves it out; the anti join is ~30 ms of
            # either side's ~60 ms)
            cursor.register("part", old)
            try:
                held = run_bound(
                    cursor,
                    f"SELECT t.* FROM part AS t ANTI JOIN (SELECT {pk_unnest}) AS k ON {pk_join}",  # noqa: S608
                    bound,
                    lambda result: result.fetch_arrow_table(),
                )
            finally:
                cursor.unregister("part")
        rows = pa.concat_tables([held, fresh.cast(held.schema)])
        sink = pa.BufferOutputStream()
        with ipc.new_stream(sink, rows.schema, options=ipc.IpcWriteOptions(compression="zstd")) as writer:
            writer.write_table(rows)

    async def bare() -> None:
        await asyncio.to_thread(bare_patch)

    async def patched() -> None:
        # each run patches the newest part by the same keys: the same rows read and written every time
        stamp[0] += 1
        checkpoints.applied(len(keys), {_TABLE: frozenset()}, {_TABLE: touched_keys})
        written = await checkpoints.write(read)
        assert written is not None and written.rows == _PART_ROWS + len(added)

    baseline = await best_seconds(bare)
    took = await best_seconds(patched)
    print(
        f"patched part of {_PART_ROWS} rows, {_PART_TOUCHED} keys touched {took:.3f} s, bare patch "
        f"{baseline:.3f} s, {took / baseline:.2f}x"
    )
    problem = budget_problem(f"a checkpoint write patching a {_PART_ROWS}-row part", took, baseline)
    assert problem is None, problem
