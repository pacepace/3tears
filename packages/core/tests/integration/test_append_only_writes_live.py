"""Integration: a versioned table's writes append, on a real database (``design-append-only-l3.md``, chunk 2).

What the unit tests cannot prove: that a rewrite of keys a versioned table holds lands a second version of each, both
kept, at a copy's cost now the change-note row trigger is gone (on YugabyteDB an upsert into a table with a row trigger
reads and flushes a row at a time: the scratch measurement was 1.58 ms a row against 0.09 for a copy); that a delete
appends a "gone" marker at a newer epoch and removes nothing, the stored table taking a marker that leaves out its
required columns while still refusing a live version without them; that a whole run deleted is exactly one run marker, its
prefix as text; and that a stamped table still upserts, one row a key.

Against the session ``db_container``: Postgres in the ``integration`` suite, YugabyteDB in ``integration_yugabyte``. The
speed ratio is asserted on YugabyteDB alone (``yugabyte_only``), where the cause it removes lives; both engines log
their numbers.
"""

from __future__ import annotations

import functools
import logging
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import asyncpg
import pytest

from threetears.core.backends.sql import SqlL3Backend
from threetears.core.collections.asyncpg_init import init_connection
from threetears.core.collections.caller_transaction import CallerTransaction
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.collections.run_markers import RUN_MARKERS_SCHEMA
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
from threetears.core.ddl.row_epoch import EPOCH_COLUMN, GONE_COLUMN, ROW_STAMPED, row_versions
from threetears.core.ddl.run_markers import RUN_MARKERS_TABLE
from threetears.core.entities.base import BaseEntity
from threetears.core.testing.l1_tables import TEST_L1_TABLE
from threetears.core.testing.plans import on_yugabyte

pytestmark = pytest.mark.integration

_log = logging.getLogger(__name__)

_EPOCHS = "append_only_scope_epochs"
_WHEN = datetime(2026, 11, 3, 20, tzinfo=UTC)
#: the rewrite the design measured: 5,000 keys a table already holds
_KEYS = 5_000


def _results(name: str, *, versioned: bool) -> TableSchema:
    """a results table as reports declares one: its scope (the state) not in its key; versioned, or today's shape."""
    return TableSchema(
        name=name,
        primary_key=("office_key", "candidate"),
        columns=[
            Column("office_key", STRING_TYPE),
            Column("candidate", STRING_TYPE),
            Column("join_state_code", STRING_TYPE),
            Column("votes", INT_TYPE),
            Column("date_created", DATETIMETZ_TYPE, immutable=True),
            Column("date_updated", DATETIMETZ_TYPE),
        ],
        row_epoch=row_versions("join_state_code", epochs=_EPOCHS, scope_prefix="state:") if versioned else None,
        l1_eviction=TEST_L1_TABLE,
    )


_VERSIONED = _results("aw_versioned", versioned=True)
#: today's table: the same columns, its key alone, upserted
_TODAY = _results("aw_today", versioned=False)

#: an LSF table: its key leads with the scope, then the run
_PROJECTIONS = TableSchema(
    name="aw_projections",
    primary_key=("join_state_code", "run_id", "geo_id"),
    columns=[
        Column("join_state_code", STRING_TYPE),
        Column("run_id", INT_TYPE),
        Column("geo_id", STRING_TYPE),
        Column("value", INT_TYPE),
    ],
    row_epoch=row_versions("join_state_code", epochs=_EPOCHS, scope_prefix="state:"),
    l1_eviction=TEST_L1_TABLE,
)

_RUNS = TableSchema(
    name="aw_runs",
    primary_key="run_id",
    columns=[Column("run_id", STRING_TYPE), Column("state", STRING_TYPE, nullable=True)],
    row_epoch=ROW_STAMPED,
    l1_eviction=TEST_L1_TABLE,
)


class _Result(BaseEntity):
    primary_key_field = "candidate"


class _Projection(BaseEntity):
    primary_key_field = "geo_id"


class _Run(BaseEntity):
    primary_key_field = "run_id"


@dataclass
class _Held:
    pool: asyncpg.Pool
    schema: str
    yugabyte: bool
    epochs: ScopeEpochs
    versioned: Any
    today: Any
    projections: Any
    runs: Any


async def _ddl(pool: asyncpg.Pool, statement: str) -> None:
    await pool.execute(statement)


@pytest.fixture
async def held(db_container: str) -> AsyncIterator[_Held]:
    schema = f"append_only_{uuid.uuid4().hex[:8]}"
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
        for table in (epochs_class.schema, _VERSIONED, _TODAY, _PROJECTIONS, _RUNS, RUN_MARKERS_SCHEMA):
            for statement in table.create_statements(yugabyte=yugabyte):
                await _ddl(pool, statement)
        registry = CollectionRegistry()
        registry.configure(l3_pool=SqlL3Backend(pool))
        config = DefaultCoreConfig()
        yield _Held(
            pool=pool,
            schema=schema,
            yugabyte=yugabyte,
            epochs=ScopeEpochs(epochs_class(registry, config, None)),
            versioned=collection_for_schema(_VERSIONED, entity_class=_Result)(registry, config, None),
            today=collection_for_schema(_TODAY, entity_class=_Result)(registry, config, None),
            projections=collection_for_schema(_PROJECTIONS, entity_class=_Projection)(registry, config, None),
            runs=collection_for_schema(_RUNS, entity_class=_Run)(registry, config, None),
        )
    finally:
        await pool.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        await pool.close()


async def _begun(held: _Held) -> int:
    """a fresh write's version, begun in a transaction of its own."""
    async with held.pool.acquire() as conn, CallerTransaction(conn):
        return await held.epochs.begin(conn=conn)
    raise AssertionError("the begin's transaction swallowed what it raised")


async def _in_version(held: _Held, step: Callable[[Any], Awaitable[Any]], *, version: int | None = None) -> Any:
    """``step`` on a transaction that took ``version``'s turn (a fresh version when None)."""
    if version is None:
        version = await _begun(held)
    async with held.pool.acquire() as conn, CallerTransaction(conn):
        await held.epochs.touch(version, ["state:NC", "state:PA"], conn=conn)
        return await step(conn)


def _rows(count: int, votes: int) -> list[dict[str, Any]]:
    return [
        {
            "office_key": f"sn_NC{n % 13}",
            "candidate": f"c{n:05d}",
            "join_state_code": "NC",
            "votes": votes,
            "date_created": _WHEN,
            "date_updated": _WHEN,
        }
        for n in range(count)
    ]


async def _count(held: _Held, table: str, where: str = "true") -> int:
    found: int = await held.pool.fetchval(f"SELECT count(*) FROM {table} WHERE {where}")  # noqa: S608 -- test names
    return found


async def _timed(held: _Held, step: Callable[[Any], Awaitable[Any]]) -> float:
    """``step``'s milliseconds a row of :data:`_KEYS`, on a fresh version's transaction, its COMMIT included."""
    started = time.perf_counter()
    await _in_version(held, step)
    return (time.perf_counter() - started) * 1000 / _KEYS


_STAGE = '"AwStage"'
_APPENDED = ("office_key", "candidate", "join_state_code", "votes", "date_created", "date_updated", EPOCH_COLUMN)


async def _raw_append(conn: Any, votes: int, *, staged: bool) -> float:
    """an append of :data:`_KEYS` held keys' new versions as the hub's broker could run it, its statements' ms a row:
    a COPY into the table, or (``staged``) a COPY into a temp stage then one ``INSERT ... SELECT`` taking the rows out
    of it, no conflict clause. The stage is made before the clock starts, as the broker keeps one on its connection."""
    transaction = CallerTransaction.enclosing(conn)
    assert transaction is not None
    epoch = transaction.row_epoch(_VERSIONED.name)
    records = [(*(row[name] for name in _APPENDED[:-1]), epoch) for row in _rows(_KEYS, votes)]
    columns = ", ".join(_APPENDED)
    if staged:
        await conn.execute(
            f"CREATE TEMP TABLE IF NOT EXISTS {_STAGE} (office_key text, candidate text, join_state_code text, "
            "votes integer, date_created timestamptz, date_updated timestamptz, epoch bigint)"
        )
    started = time.perf_counter()
    if staged:
        await conn.copy_records_to_table(_STAGE.strip('"'), records=records, columns=list(_APPENDED))
        await conn.execute(
            f"WITH taken AS (DELETE FROM {_STAGE} RETURNING {columns}) "  # noqa: S608 -- test names
            f"INSERT INTO {_VERSIONED.name} ({columns}) SELECT {columns} FROM taken"
        )
    else:
        await conn.copy_records_to_table(_VERSIONED.name, records=records, columns=list(_APPENDED))
    return (time.perf_counter() - started) * 1000 / _KEYS


async def _saved(collection: Any, rows: list[dict[str, Any]], conn: Any, *, copy: bool = False) -> Any:
    """``collection.save_rows(rows)`` on ``conn``: a step for :func:`_in_version` and :func:`_timed`."""
    return await collection.save_rows(rows, conn=conn, copy=copy)


async def _appended_raw(votes: int, staged: bool, conn: Any) -> float:
    """:func:`_raw_append` as a step for :func:`_in_version`."""
    return await _raw_append(conn, votes, staged=staged)


async def _rewrite_times(held: _Held) -> dict[str, float]:
    """the ms a row of a rewrite of :data:`_KEYS` held keys: today's upsert, and the versioned table's append."""
    for collection in (held.today, held.versioned):
        await _in_version(held, functools.partial(_saved, collection, _rows(_KEYS, 1), copy=True))
    times: dict[str, float] = {}
    for votes, trigger in ((2, "first round"), (3, "second round")):
        times[f"upsert {trigger}"] = await _timed(held, functools.partial(_saved, held.today, _rows(_KEYS, votes)))
        times[f"append {trigger}"] = await _timed(held, functools.partial(_saved, held.versioned, _rows(_KEYS, votes)))
        # the hub's two forms of an append, their statements alone: a COPY straight into the table (what the broker
        # runs) against a COPY into a stage then INSERT ... SELECT from it with no conflict clause
        for staged in (False, True):
            form = "staged" if staged else "copied"
            times[f"append {form} {trigger}"] = await _in_version(
                held, functools.partial(_appended_raw, votes + 10 * (1 + staged), staged)
            )
    engine = "YugabyteDB" if held.yugabyte else "Postgres"
    _log.warning("rewrite of %d held keys, ms a row on %s: %s", _KEYS, engine, times)
    print(f"rewrite of {_KEYS} held keys, ms a row on {engine}: {times}")  # noqa: T201 -- the measurement's record
    return times


# -- a write appends ---------------------------------------------------------------------------------------------------


async def test_a_rewrite_of_held_keys_appends_a_version_of_each_and_keeps_both(held: _Held) -> None:
    await _rewrite_times(held)
    # seven writes of each key: the seed, then two rounds of three appends; every version kept, today's one row a key
    assert await _count(held, _VERSIONED.name) == 7 * _KEYS
    assert await _count(held, _TODAY.name) == _KEYS
    versions = await held.pool.fetch(
        f"SELECT votes, {EPOCH_COLUMN} FROM {_VERSIONED.name} WHERE office_key = 'sn_NC0' AND candidate = 'c00000' "  # noqa: S608
        f"ORDER BY {EPOCH_COLUMN}"
    )
    assert [row["votes"] for row in versions] == [1, 2, 12, 22, 3, 13, 23], "a later write did not sort newer"
    assert await _count(held, _VERSIONED.name, f"{GONE_COLUMN}") == 0


@pytest.mark.yugabyte_only
async def test_a_rewrite_of_held_keys_appends_at_a_copys_speed_not_an_upserts(held: _Held) -> None:
    times = await _rewrite_times(held)
    shown = "; ".join(f"{name} {ms:.3f}" for name, ms in sorted(times.items()))
    # measured on the pool's YugabyteDB, 2026-11-25, ms a row (5,000 held keys, 7 columns, a direct connection):
    #   with the change-note row trigger: upsert 0.447, append 0.226 (a COPY 0.226, staged + INSERT ... SELECT 0.222)
    #   without it:                       upsert 0.030, append 0.019 (a COPY 0.005, staged 0.006)
    # the ring's trigger retired in chunk 7, so both rounds run without it: the append at a copy's cost, never above the
    # upsert's. Chunk 1's upsert of a versioned table ran at the upsert's cost (1.0)
    assert times["append first round"] <= times["upsert first round"], shown
    assert times["append second round"] <= times["upsert second round"], shown


# -- a delete appends a gone marker -----------------------------------------------------------------------------------


async def test_a_delete_appends_a_gone_marker_at_a_newer_epoch_and_removes_nothing(held: _Held) -> None:
    await _in_version(held, lambda conn: held.versioned.save_rows(_rows(3, 1), conn=conn))
    live = await held.pool.fetchval(f"SELECT max({EPOCH_COLUMN}) FROM {_VERSIONED.name}")  # noqa: S608
    keys = [("sn_NC0", "c00000"), ("sn_NC1", "c00001")]
    deleted = await _in_version(held, lambda conn: held.versioned.delete_rows(keys, conn=conn, scope="NC"))
    assert deleted == 2
    assert await _count(held, _VERSIONED.name) == 5
    markers = await held.pool.fetch(
        f"SELECT office_key, candidate, join_state_code, votes, date_created, {EPOCH_COLUMN} "  # noqa: S608
        f"FROM {_VERSIONED.name} WHERE {GONE_COLUMN} ORDER BY candidate"
    )
    assert [(row["office_key"], row["candidate"]) for row in markers] == keys
    # the marker carries the key, the scope and the epoch alone, newer than the version it marks gone
    assert all(row["join_state_code"] == "NC" for row in markers)
    assert all(row["votes"] is None and row["date_created"] is None for row in markers)
    assert all(row[EPOCH_COLUMN] > live for row in markers)


async def test_the_stored_table_still_refuses_a_live_version_without_its_required_columns(held: _Held) -> None:
    with pytest.raises(asyncpg.CheckViolationError):
        await held.pool.execute(
            f"INSERT INTO {_VERSIONED.name} (office_key, candidate, join_state_code, {EPOCH_COLUMN}) "  # noqa: S608
            "VALUES ('sn_NC1', 'x', 'NC', 1)"
        )
    # a marker leaves them out
    await held.pool.execute(
        f"INSERT INTO {_VERSIONED.name} (office_key, candidate, join_state_code, {EPOCH_COLUMN}, {GONE_COLUMN}) "  # noqa: S608
        "VALUES ('sn_NC1', 'x', 'NC', 1, true)"
    )


# -- a whole run deleted is one run marker ----------------------------------------------------------------------------


async def test_a_whole_run_deleted_writes_exactly_one_run_marker_and_deletes_nothing(held: _Held) -> None:
    rows = [
        {"join_state_code": "PA", "run_id": run, "geo_id": f"g{n:03d}", "value": n}
        for run in (7, 8)
        for n in range(100)
    ]
    await _in_version(held, lambda conn: held.projections.save_rows(rows, conn=conn, copy=True))
    marked = await _in_version(
        held, lambda conn: held.projections.delete_prefixes([{"join_state_code": "PA", "run_id": 7}], conn=conn)
    )
    assert marked == 1
    assert await _count(held, _PROJECTIONS.name) == 200
    [marker] = await held.pool.fetch(f"SELECT table_name, scope, prefix, {EPOCH_COLUMN} FROM {RUN_MARKERS_TABLE}")  # noqa: S608
    assert (marker["table_name"], marker["scope"], marker["prefix"]) == (
        _PROJECTIONS.name,
        "PA",
        "PA\x1f7",
    )
    newest = await held.pool.fetchval(f"SELECT max({EPOCH_COLUMN}) FROM {_PROJECTIONS.name}")  # noqa: S608
    assert marker[EPOCH_COLUMN] > newest


# -- a stamped table still upserts ------------------------------------------------------------------------------------


async def test_a_stamped_table_still_upserts_one_row_a_key_carrying_its_last_writes_epoch(held: _Held) -> None:
    await _in_version(held, lambda conn: held.runs.save_rows([{"run_id": "r1", "state": "NC"}], conn=conn))
    first = await held.pool.fetchval(f"SELECT {EPOCH_COLUMN} FROM {_RUNS.name}")  # noqa: S608
    await _in_version(held, lambda conn: held.runs.save_rows([{"run_id": "r1", "state": "PA"}], conn=conn))
    [row] = await held.pool.fetch(f"SELECT state, {EPOCH_COLUMN} FROM {_RUNS.name}")  # noqa: S608
    assert row["state"] == "PA"
    assert row[EPOCH_COLUMN] > first
