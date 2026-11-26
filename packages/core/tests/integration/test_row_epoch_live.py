"""Integration: every row a write leaves carries its write's epoch, on a real database (``design-append-only-l3.md``,
chunk 1).

What the unit tests cannot prove: that the DDL a versioned table declares makes the primary key it says (the key, then
the epoch; on YugabyteDB the first key column hashed and ``epoch DESC`` last); that
each write path the store owns -- an upsert, a copy, an Arrow write without the hub's door, one row, a load from files
-- lands rows stamped with an epoch of the write's version; that two writes of one key in one version land two versions,
the later newer; and that a write with no epoch is refused, by 3tears before it is sent and by the database for a
statement that bypasses 3tears.

Against the session ``db_container``: Postgres in the ``integration`` suite, YugabyteDB in ``integration_yugabyte``.
"""

from __future__ import annotations

import io
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import asyncpg
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from threetears.core.backends.protocol import LoadFile
from threetears.core.backends.sql import SqlL3Backend
from threetears.core.collections.asyncpg_init import init_connection
from threetears.core.collections.caller_transaction import CallerTransaction
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.collections.schema_backed import (
    DATETIMETZ_TYPE,
    INT_TYPE,
    STRING_TYPE,
    Column,
    TableSchema,
    collection_for_schema,
)
from threetears.core.collections.scope_epochs import ScopeEpochs, scope_epochs_collection
from threetears.core.config import DefaultCoreConfig
from threetears.core.ddl.row_epoch import (
    EPOCH_COLUMN,
    ROW_STAMPED,
    UnstampedWriteError,
    next_epoch,
    row_versions,
)
from threetears.core.entities.base import BaseEntity
from threetears.core.testing.l1_tables import TEST_L1_TABLE
from threetears.core.testing.plans import on_yugabyte

pytestmark = pytest.mark.integration

_EPOCHS = "row_epoch_scope_epochs"
_WHEN = datetime(2026, 11, 3, 20, tzinfo=UTC)

_RESULTS = TableSchema(
    name="row_epoch_results",
    primary_key=("office_key", "candidate"),
    columns=[
        Column("office_key", STRING_TYPE),
        Column("candidate", STRING_TYPE),
        Column("join_state_code", STRING_TYPE),
        Column("votes", INT_TYPE, nullable=True),
        Column("date_created", DATETIMETZ_TYPE, immutable=True),
        Column("date_updated", DATETIMETZ_TYPE),
    ],
    row_epoch=row_versions("join_state_code"),
    l1_eviction=TEST_L1_TABLE,
)

_RUNS = TableSchema(
    name="row_epoch_runs",
    primary_key="run_id",
    columns=[Column("run_id", STRING_TYPE), Column("state", STRING_TYPE, nullable=True)],
    row_epoch=ROW_STAMPED,
    l1_eviction=TEST_L1_TABLE,
)


class _Result(BaseEntity):
    primary_key_field = "candidate"


class _Run(BaseEntity):
    primary_key_field = "run_id"


@dataclass
class _Held:
    pool: asyncpg.Pool
    yugabyte: bool
    epochs: ScopeEpochs
    results: Any
    runs: Any


@pytest.fixture
async def held(db_container: str) -> AsyncIterator[_Held]:
    schema = f"row_epoch_{uuid.uuid4().hex[:8]}"
    admin = await asyncpg.connect(db_container)
    try:
        await admin.execute(f"CREATE SCHEMA {schema}")
    finally:
        await admin.close()
    pool = await asyncpg.create_pool(
        db_container, min_size=1, max_size=4, init=init_connection, server_settings={"search_path": schema}
    )
    assert pool is not None
    try:
        async with pool.acquire() as conn:
            yugabyte = await on_yugabyte(conn)
        epochs_class = scope_epochs_collection(_EPOCHS)
        for table in (epochs_class.schema, _RESULTS, _RUNS):
            for statement in table.create_statements(yugabyte=yugabyte):
                await pool.execute(statement)
        registry = CollectionRegistry()
        registry.configure(l3_pool=SqlL3Backend(pool))
        config = DefaultCoreConfig()
        yield _Held(
            pool=pool,
            yugabyte=yugabyte,
            epochs=ScopeEpochs(epochs_class(registry, config, None)),
            results=collection_for_schema(_RESULTS, entity_class=_Result)(registry, config, None),
            runs=collection_for_schema(_RUNS, entity_class=_Run)(registry, config, None),
        )
    finally:
        await pool.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        await pool.close()


async def _in_transaction(pool: asyncpg.Pool, step: Callable[[Any], Awaitable[Any]]) -> Any:
    async with pool.acquire() as conn, CallerTransaction(conn):
        return await step(conn)


async def _begin(held: _Held) -> int:
    version: int = await _in_transaction(held.pool, lambda conn: held.epochs.begin(conn=conn))
    return version


def _rows(candidates: list[str], votes: int = 1) -> list[dict[str, Any]]:
    return [
        {"office_key": "sn_NC1", "candidate": candidate, "join_state_code": "NC", "votes": votes}
        for candidate in candidates
    ]


async def _stored(held: _Held) -> list[asyncpg.Record]:
    rows: list[asyncpg.Record] = await held.pool.fetch(
        f"SELECT candidate, votes, {EPOCH_COLUMN} FROM {_RESULTS.name} ORDER BY candidate, {EPOCH_COLUMN}"  # noqa: S608
    )
    return rows


# -- the DDL ---------------------------------------------------------------------------------------------------------


async def test_a_versioned_table_is_stored_by_its_key_then_its_epoch(held: _Held) -> None:
    primary = await held.pool.fetchrow(
        "SELECT i.indexrelid, pg_get_indexdef(i.indexrelid) AS definition, "
        "ARRAY(SELECT a.attname::text FROM unnest(i.indkey) WITH ORDINALITY AS k(attnum, place) "
        "JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = k.attnum ORDER BY k.place) AS columns "
        "FROM pg_index i WHERE i.indrelid = $1::regclass AND i.indisprimary",
        _RESULTS.name,
    )
    assert primary is not None
    assert primary["columns"] == ["office_key", "candidate", EPOCH_COLUMN]
    if held.yugabyte:
        # the first key column hashed as before, the epoch the last range column, newest first
        assert "(office_key HASH, candidate ASC, epoch DESC)" in primary["definition"]
    else:
        assert "(office_key, candidate, epoch)" in primary["definition"]
    for table in (_RESULTS.name, _RUNS.name):
        column = await held.pool.fetchrow(
            "SELECT data_type, is_nullable, column_default FROM information_schema.columns "
            "WHERE table_schema = current_schema() AND table_name = $1 AND column_name = $2",
            table,
            EPOCH_COLUMN,
        )
        assert column is not None
        assert (column["data_type"], column["is_nullable"], column["column_default"]) == ("bigint", "NO", None)
    runs_key = await held.pool.fetchval(
        "SELECT pg_get_indexdef(indexrelid) FROM pg_index WHERE indrelid = $1::regclass AND indisprimary", _RUNS.name
    )
    # a stamped table keeps its key alone
    assert runs_key.rstrip().endswith("(run_id HASH)" if held.yugabyte else "(run_id)")


# -- every write path stamps ------------------------------------------------------------------------------------------


def _parquet(tmp_path: Path, name: str, candidates: list[str]) -> LoadFile:
    table = pa.table(
        {"candidate": candidates, "join_state_code": ["NC"] * len(candidates), "votes": [7] * len(candidates)},
        schema=pa.schema([("candidate", pa.string()), ("join_state_code", pa.string()), ("votes", pa.int64())]),
    )
    sink = io.BytesIO()
    pq.write_table(table, sink)
    (tmp_path / name).write_bytes(sink.getvalue())
    return LoadFile(url=(tmp_path / name).as_uri(), rows=len(candidates))


async def test_every_write_path_stamps_each_row_with_an_epoch_of_its_writes_version(
    held: _Held, tmp_path: Path
) -> None:
    version = await _begin(held)
    begun = next_epoch()

    async def write(conn: Any) -> None:
        await held.epochs.touch(version, ["state:NC"], conn=conn)
        await held.results.save_rows(_rows(["a"]), conn=conn)
        await held.results.save_rows(_rows(["b"]), conn=conn, copy=True)
        # a direct connection has no Arrow write door: the rows are written as save_rows writes them
        await held.results.save_arrow(pa.Table.from_pylist(_rows(["c"])), conn=conn)
        await held.results.save_to_store({**_rows(["d"])[0], "date_created": _WHEN, "date_updated": _WHEN}, conn=conn)
        await held.runs.save_rows([{"run_id": "r1", "state": "NC"}], conn=conn)
        await held.runs.save_to_store({"run_id": "r2", "state": "VA"}, conn=conn)

    await _in_transaction(held.pool, write)
    loaded = await held.results.load_files(
        [_parquet(tmp_path, "e.parquet", ["e", "f"])],
        columns=["candidate", "join_state_code", "votes"],
        constants={"office_key": "sn_NC1", "date_created": _WHEN, "date_updated": _WHEN},
        write_version=version,
    )
    assert loaded == 2

    stored = await _stored(held)
    assert [row["candidate"] for row in stored] == ["a", "b", "c", "d", "e", "f"]
    assert all(row[EPOCH_COLUMN] > begun for row in stored), "each row carries a stamp of this write's clock"
    epochs = [row[EPOCH_COLUMN] for row in stored]
    # each statement drew its own, in the order they ran; the load's rows share one
    assert epochs[:5] == sorted(epochs[:5]) and len(set(epochs[:5])) == 5
    assert epochs[4] == epochs[5]
    runs = await held.pool.fetch(f"SELECT run_id, {EPOCH_COLUMN} FROM {_RUNS.name} ORDER BY run_id")  # noqa: S608
    assert [row["run_id"] for row in runs] == ["r1", "r2"]
    assert all(row[EPOCH_COLUMN] > begun for row in runs)


async def test_two_writes_of_one_key_in_one_version_land_two_versions_the_later_newer(held: _Held) -> None:
    version = await _begin(held)
    begun = next_epoch()

    def write(votes: int) -> Callable[[Any], Awaitable[None]]:
        async def step(conn: Any) -> None:
            await held.epochs.touch(version, ["state:NC"], conn=conn)
            await held.results.save_rows(_rows(["a"], votes), conn=conn)

        return step

    # two page transactions of one version, as a page written again after a failed copy
    await _in_transaction(held.pool, write(1))
    await _in_transaction(held.pool, write(2))

    async def twice(conn: Any) -> None:
        await held.epochs.touch(version, ["state:NC"], conn=conn)
        await held.results.save_rows(_rows(["b"], 1), conn=conn)
        await held.results.save_rows(_rows(["b"], 2), conn=conn)

    # and two statements of one transaction
    await _in_transaction(held.pool, twice)

    stored = await _stored(held)
    for candidate in ("a", "b"):
        versions = [row for row in stored if row["candidate"] == candidate]
        assert [row["votes"] for row in versions] == [1, 2], "the later write did not sort newer"
        assert all(row[EPOCH_COLUMN] > begun for row in versions)
        newest = await held.pool.fetchval(
            f"SELECT votes FROM {_RESULTS.name} WHERE office_key = 'sn_NC1' AND candidate = $1 "  # noqa: S608
            f"ORDER BY {EPOCH_COLUMN} DESC LIMIT 1",
            candidate,
        )
        assert newest == 2


# -- a write with no epoch is refused ---------------------------------------------------------------------------------


async def test_a_write_with_no_epoch_is_refused_and_writes_nothing(held: _Held, tmp_path: Path) -> None:
    # a transaction that took no write's turn has no version to stamp with
    with pytest.raises(UnstampedWriteError, match="took no write's turn"):
        await _in_transaction(held.pool, lambda conn: held.results.save_rows(_rows(["a"]), conn=conn))
    with pytest.raises(UnstampedWriteError, match="took no write's turn"):
        await _in_transaction(held.pool, lambda conn: held.runs.save_rows([{"run_id": "r1"}], conn=conn))
    # one row on no transaction at all
    with pytest.raises(UnstampedWriteError, match="joins no transaction"):
        await held.runs.save_to_store({"run_id": "r1"})
    # a load naming no version
    with pytest.raises(UnstampedWriteError, match="write_version"):
        await held.results.load_files(
            [_parquet(tmp_path, "g.parquet", ["g"])],
            columns=["candidate", "join_state_code", "votes"],
            constants={"office_key": "sn_NC1", "date_created": _WHEN, "date_updated": _WHEN},
        )
    # a statement that bypasses 3tears: refused by the database
    with pytest.raises(asyncpg.NotNullViolationError):
        await held.pool.execute(
            f"INSERT INTO {_RESULTS.name} (office_key, candidate, join_state_code, date_created, date_updated) "  # noqa: S608
            "VALUES ('sn_NC1', 'x', 'NC', now(), now())"
        )
    with pytest.raises(asyncpg.NotNullViolationError):
        await held.pool.execute(f"INSERT INTO {_RUNS.name} (run_id) VALUES ('x')")  # noqa: S608
    assert await held.pool.fetchval(f"SELECT count(*) FROM {_RESULTS.name}") == 0  # noqa: S608
    assert await held.pool.fetchval(f"SELECT count(*) FROM {_RUNS.name}") == 0  # noqa: S608
