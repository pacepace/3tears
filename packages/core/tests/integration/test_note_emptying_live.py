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

from threetears.core.collections.change_notes import ChangeNoteKeeper, install_change_notes
from threetears.core.ddl.change_notes import note_table
from threetears.core.testing.calibrate import budget_problem, plain_keyed_delete_seconds
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

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"an emptying reads no change log ({name})")


@pytest.fixture
async def ring(db_container: str) -> AsyncIterator[_Ring]:
    """a ring whose idle table holds a night's notes."""
    schema = f"empty_{uuid.uuid4().hex[:8]}"
    pool = await asyncpg.create_pool(db_container, min_size=1, max_size=2, command_timeout=1200)
    assert pool is not None
    try:
        await pool.execute(f"CREATE SCHEMA {schema}")
        await install_change_notes(pool, schema)
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
    twice the time of one keyed delete of as many rows."""
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
