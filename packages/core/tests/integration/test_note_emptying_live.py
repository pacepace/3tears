"""Integration: emptying a note table walks its key, so a night's notes are deleted in about the database's time.

The keeper empties the ring's idle note table in batches. Each batch used to pick its notes from the table's
start, stepping over every tombstone the batches before it left: emptying ``N`` notes read about
``N²/2/batch`` of them. The batches now walk the key in its own order (YugabyteDB: bands of the key's hash).
This seeds an idle table at night's size and empties it: every note gone, each counted once, and on
YugabyteDB within twice the database's own time to delete that many keyed rows in one statement.

Against the session ``db_container``; set ``THREETEARS_TEST_POSTGRES_URL`` to a YugabyteDB URL to run it there.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import asyncpg
import pytest

from threetears.core.collections.change_note_keeper import ChangeNoteKeeper
from threetears.core.ddl.change_notes import note_table
from threetears.core.testing.calibrate import budget_problem, plain_keyed_delete_seconds
from threetears.core.testing.note_ring import install_note_ring
from threetears.core.testing.plans import on_yugabyte

pytestmark = pytest.mark.integration

#: the idle table's notes: a busy stretch of the night
_NOTES = 200_000

#: the ring's idle table, after the current one (0)
_IDLE = 1


@dataclass
class _Ring:
    pool: asyncpg.Pool
    schema: str


class _NoLog:
    """the keeper's change log, never read by an emptying."""

    #: names no database on its records: the keeper reads this at construction (EpochGenerationSource.database)
    database = None

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"an emptying reads no change log ({name})")


class _Logged:
    """the keeper's pool, every statement it sends recorded in order with its seconds (the query logger)."""

    def __init__(self, pool: asyncpg.Pool) -> None:
        self._pool = pool
        self.sent: list[str] = []
        self.took: list[float] = []

    async def fetch(self, query: str, *args: Any) -> list[Any]:
        self.sent.append(query)
        began = time.perf_counter()
        try:
            return await self._pool.fetch(query, *args)
        finally:
            self.took.append(time.perf_counter() - began)

    async def execute(self, query: str, *args: Any) -> str:
        self.sent.append(query)
        began = time.perf_counter()
        try:
            return await self._pool.execute(query, *args)
        finally:
            self.took.append(time.perf_counter() - began)

    def acquire(self) -> Any:
        raise AssertionError("an emptying shares no session")


def _is_vacuum(query: str) -> bool:
    return query.lstrip().upper().startswith("VACUUM")


@pytest.fixture
async def ring(db_container: str) -> AsyncIterator[_Ring]:
    """a ring whose idle table holds a night's notes."""
    schema = f"empty_{uuid.uuid4().hex[:8]}"
    pool = await asyncpg.create_pool(db_container, min_size=1, max_size=2, command_timeout=1200)
    assert pool is not None
    try:
        await pool.execute(f"CREATE SCHEMA {schema}")
        await install_note_ring(pool, schema)
        await pool.execute(
            f"INSERT INTO {schema}.{note_table(_IDLE)} (table_name, keys, key_text) "
            f"SELECT 'table_' || (g % 2000), ARRAY[g::text], g::text FROM generate_series(1, {_NOTES}) g"
        )
        yield _Ring(pool=pool, schema=schema)
    finally:
        await pool.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        await pool.close()


async def test_an_idle_table_is_emptied_once_each_note_within_twice_a_keyed_delete(ring: _Ring) -> None:
    """Flips if an emptying leaves a note, counts one twice, or (on YugabyteDB) rereads its own tombstones past
    twice the time of one keyed delete of as many rows (``_delete_band`` on YugabyteDB, ``_delete_past`` elsewhere: the
    slot's whole emptying, declared ``ring_empty``)."""
    keeper = ChangeNoteKeeper(
        ring.pool,
        notes_schema=ring.schema,
        log=_NoLog(),
        empty_pause=timedelta(0),
        empty_budget=timedelta(hours=1),
    )
    began = time.perf_counter()
    empty, deleted = await keeper.empty_slot(_IDLE)
    took = time.perf_counter() - began

    assert (empty, deleted) == (True, _NOTES)
    assert await ring.pool.fetchval(f"SELECT count(*) FROM {ring.schema}.{note_table(_IDLE)}") == 0
    async with ring.pool.acquire() as conn:
        if not await on_yugabyte(conn):
            return
        await conn.execute(f"SET search_path = {ring.schema}")
        baseline = await plain_keyed_delete_seconds(conn, _NOTES, runs=1)
    problem = budget_problem(f"emptying {_NOTES:,} notes", took, baseline)
    assert problem is None, problem


async def test_an_emptied_table_is_vacuumed_before_it_counts_as_empty_on_postgres_alone(ring: _Ring) -> None:
    """Flips if, on Postgres, an emptying reports the table empty (what the keeper marks ``emptied`` on, the switch's
    guard) before its VACUUM has run, or leaves the table's pages behind; or if a VACUUM is ever sent to YugabyteDB,
    where it does nothing (Pace 2026-11-15; R1's reused-slot measurement, at ``_vacuum_emptied``)."""
    logged = _Logged(ring.pool)
    keeper = ChangeNoteKeeper(
        logged,
        notes_schema=ring.schema,
        log=_NoLog(),
        empty_pause=timedelta(0),
        empty_budget=timedelta(hours=1),
    )
    assert await keeper.empty_slot(_IDLE) == (True, _NOTES)
    vacuums = [query for query in logged.sent if _is_vacuum(query)]
    async with ring.pool.acquire() as conn:
        yugabyte = await on_yugabyte(conn)
    if yugabyte:
        assert vacuums == [], logged.sent
        return
    assert len(vacuums) == 1 and logged.sent[-1] == vacuums[0], logged.sent
    # every note dead and vacuumed away: the vacuum gives the table's empty pages back
    pages = await ring.pool.fetchval(
        "SELECT pg_relation_size(c.oid) FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
        "WHERE n.nspname = $1 AND c.relname = $2",
        ring.schema,
        note_table(_IDLE),
    )
    assert pages == 0, pages


#: a slot's notes at its switch: the 64 MiB log's restore budget (75 %) over two, over 64 bytes a note
_FULL_SLOT = 393_216


async def test_an_emptied_full_slot_s_vacuum_costs_within_twice_a_bare_vacuum(db_container: str) -> None:
    """Flips if the VACUUM the keeper sends after emptying a full slot on Postgres (``_vacuum_emptied``, Pace 2026-11-15)
    costs over twice a bare VACUUM of a table of the slot's own shape holding as many deleted rows (a FULL, an ANALYZE
    or a second pass would). The cost at a full slot is recorded at ``_vacuum_emptied``. Nothing to time on YugabyteDB,
    where no VACUUM is sent."""
    schema = f"vac_{uuid.uuid4().hex[:8]}"
    pool = await asyncpg.create_pool(db_container, min_size=1, max_size=2, command_timeout=1200)
    assert pool is not None
    try:
        async with pool.acquire() as conn:
            if await on_yugabyte(conn):
                return
        await pool.execute(f"CREATE SCHEMA {schema}")
        await install_note_ring(pool, schema)
        slot = f"{schema}.{note_table(_IDLE)}"
        fill = (
            "INSERT INTO {} (table_name, keys, key_text) "
            f"SELECT 'table_' || (g % 2000), ARRAY[g::text], g::text FROM generate_series(1, {_FULL_SLOT}) g"
        )
        await pool.execute(fill.format(slot))
        logged = _Logged(pool)
        keeper = ChangeNoteKeeper(
            logged, notes_schema=schema, log=_NoLog(), empty_pause=timedelta(0), empty_budget=timedelta(hours=1)
        )
        assert await keeper.empty_slot(_IDLE) == (True, _FULL_SLOT)
        took = sum(seconds for query, seconds in zip(logged.sent, logged.took, strict=True) if _is_vacuum(query))
        # the bare operation: the slot's own shape (its indexes), as many rows deleted in one statement, vacuumed
        plain = f"{schema}.plain_slot"
        await pool.execute(f"CREATE TABLE {plain} (LIKE {slot} INCLUDING ALL)")
        await pool.execute(fill.format(plain))
        await pool.execute(f"DELETE FROM {plain}")
        began = time.perf_counter()
        await pool.execute(f"VACUUM {plain}")
        baseline = time.perf_counter() - began
        print(
            f"vacuum of an emptied full slot ({_FULL_SLOT:,} notes): {took * 1000:.1f} ms, bare {baseline * 1000:.1f} ms"
        )
        problem = budget_problem(f"the vacuum of {_FULL_SLOT:,} emptied notes", took, baseline)
        assert problem is None, problem
    finally:
        await pool.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        await pool.close()
