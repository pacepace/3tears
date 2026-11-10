"""Integration: a copy follower's few whole reads cost what a bare read of the same rows costs, at real size.

The whole reads the no-whole-reads ledger keeps, each held within 2x its bare operation timed on the same database
(:func:`threetears.core.testing.calibrate.budget_problem`):

- ``CopyFollower._stage_whole`` (a cold start, a reset, a schema change): reading a table from L3 a keyset page at a
  time and staging it beside the copy, against the bare keyset read of the same pages and their copy-in to DuckDB;
- ``CopyCheckpoints.stage`` (a start from a checkpoint): every part read back from the Object Store and staged, against
  the bare decode and copy-in of the same parts into DuckDB;
- ``copy_checkpoint._batches`` (a checkpoint write at a table's cold start): the table read from the copy and stored
  as parts, against the bare read of the same rows and their encoding.

Timed runs belong under measure-lock. Against the session ``db_container`` (Postgres); set
``THREETEARS_TEST_POSTGRES_URL`` to a YugabyteDB URL to run them there. ``THREETEARS_FOLLOW_SPEED_ROWS`` sets the rows
(default 1,000,000: a long table of an election night's copy).
"""

from __future__ import annotations

import asyncio
import os
import uuid
from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import asyncpg
import pytest
from sqlalchemy import BigInteger, Column as SaColumn, MetaData, String, Table

duckdb = pytest.importorskip("duckdb")
pa = pytest.importorskip("pyarrow")
ipc = pytest.importorskip("pyarrow.ipc")

from threetears.core.cache.duckdb import STAGE_PREFIX, DuckDBBackend, StagedChange  # noqa: E402
from threetears.core.collections import CollectionRegistry  # noqa: E402
from threetears.core.collections.complete_copy import DEFAULT_PAGE_SIZE, rail_arrow_types  # noqa: E402
from threetears.core.collections.copy_checkpoint import CopyCheckpoints  # noqa: E402
from threetears.core.collections.copy_follower import CopyFollower, FollowedTable  # noqa: E402
from threetears.core.collections.epoch_change import EpochChange, EpochWalk, walk_by_count  # noqa: E402
from threetears.core.collections.schema_backed import BIGINT_TYPE, STRING_TYPE, Column  # noqa: E402
from threetears.core.keyset import keyset_arrow_pages  # noqa: E402
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


def _metadata() -> MetaData:
    metadata = MetaData()
    Table(
        _TABLE,
        metadata,
        SaColumn("state", String(8), primary_key=True),
        SaColumn("geo_id", BigInteger, primary_key=True),
        SaColumn("votes", BigInteger),
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
    """a copy holding the rows, applied with mark 1 (a reader's stamp names it)."""
    backend = _backend()
    stage = f"{STAGE_PREFIX}{_TABLE}__rows"
    backend.stage_rows(stage, _TABLE, _rows(), _KEY)
    backend.apply_staged([StagedChange(table=_TABLE, key=_KEY, rows=stage, whole=True)], mark=1)
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
    @contextmanager
    def read() -> Iterator[Any]:
        with backend.read_snapshot() as cursor:
            yield cursor, 1, {_TABLE: "inc:0"}

    return read


async def test_stage_whole_speed_at_real_size(db: _Db) -> None:
    """a cold start, end to end (CopyFollower.warm and the catch-up that applies it): CopyFollower._stage_whole's
    whole read of the table from L3, which CopyFollower._stage stages, and the one apply, within 2x a bare keyset
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
            async for page in keyset_arrow_pages(
                db.pool, f"SELECT {selected} FROM {_TABLE}", _KEY, types=types, page_size=DEFAULT_PAGE_SIZE
            ):
                await asyncio.to_thread(copy_in, con, page, first)
                first = False
        finally:
            con.close()

    async def cold_start() -> None:
        registry = CollectionRegistry()
        registry.follow_generation(_TABLE)
        log = _Log()
        follower = CopyFollower(
            registry,
            [_TABLE_DEF],
            _backend(),
            db.pool,
            settled=lambda: asyncio.sleep(0, 1),
            generation=log.current,
            changes=log,
            stamp_table="scope_epochs",
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


async def test_checkpoint_stage_speed_at_real_size() -> None:
    """CopyCheckpoints.stage (a start's load of every table from L2) within 2x the bare decode and copy-in of the same
    parts into DuckDB."""
    source = _filled()
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


async def test_checkpoint_whole_write_speed_at_real_size() -> None:
    """a checkpoint write at a table's cold start, whose whole read of the copy is copy_checkpoint._batches, within 2x
    the bare read of the same rows and their encoding as parts."""
    source = _filled()

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
    problem = budget_problem(f"a cold-start checkpoint write (_batches) of {_ROWS} rows", took, baseline)
    assert problem is None, problem
