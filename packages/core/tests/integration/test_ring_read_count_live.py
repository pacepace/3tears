"""Integration: a switch's count of the notes written without a cached slot reads those notes and no others.

``ChangeNoteKeeper.count_ring_reads`` counts, per table, the notes of the table the ring just switched away from
whose statement read the ring row (``WHERE ring_read GROUP BY 1``). It binds none of the partial ``_ring_read``
index's columns: the index's predicate is all that bounds it, so the no-whole-reads rule asks for proof at real size
that it stays bounded (coordinator ruling, 3tears #80). This seeds the switched-away table with a night's notes, a
few of them ring reads, and times the count against the same count on a table holding only those few notes: within
twice it, so the read grows with the ring reads, never with the table.

Against the session ``db_container``; set ``THREETEARS_TEST_POSTGRES_URL`` to a YugabyteDB URL to run it there.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import asyncpg
import pytest

from threetears.core.collections.change_notes import ChangeNoteKeeper
from threetears.core.ddl.change_notes import note_table
from threetears.core.testing.calibrate import best_seconds_interleaved, budget_problem

pytestmark = pytest.mark.integration

#: the switched-away table's notes: a busy night
_NOTES = 1_000_000
_TABLES = 2_000
#: of them, written without a cached slot
_RING_READS = 200
#: the table the ring just switched away from
_SWITCHED = 1


@dataclass
class _Rings:
    pool: asyncpg.Pool
    full: str
    few: str


class _NoLog:
    """the keeper's change log, never read by the count."""

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"the ring-read count reads no change log ({name})")


async def _seed(pool: asyncpg.Pool, schema: str, notes: int) -> None:
    """a ring in ``schema`` whose switched-away table holds ``notes`` notes, the first :data:`_RING_READS` ring reads."""
    await pool.execute(f"CREATE SCHEMA {schema}")
    await ChangeNoteKeeper(pool, notes_schema=schema, log=_NoLog()).install()
    table = f"{schema}.{note_table(_SWITCHED)}"
    await pool.execute(
        f"INSERT INTO {table} (table_name, keys, key_text, ring_read) "  # noqa: S608 - test identifiers
        f"SELECT 'table_' || (g % {_TABLES}), ARRAY[g::text], g::text, "
        f"CASE WHEN g <= {_RING_READS} THEN TRUE END FROM generate_series(1, {notes}) g"
    )
    await pool.execute(f"ANALYZE {table}")


@pytest.fixture
async def rings(db_container: str) -> AsyncIterator[_Rings]:
    """two rings: one whose switched-away table holds a night's notes, one holding only its ring reads."""
    suffix = uuid.uuid4().hex[:8]
    full, few = f"ringcount_{suffix}", f"ringfew_{suffix}"
    pool = await asyncpg.create_pool(db_container, min_size=1, max_size=2, command_timeout=1200)
    assert pool is not None
    try:
        await _seed(pool, full, _NOTES)
        await _seed(pool, few, _RING_READS)
        yield _Rings(pool=pool, full=full, few=few)
    finally:
        for schema in (full, few):
            await pool.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        await pool.close()


async def test_a_switchs_ring_read_count_reads_only_the_ring_reads(rings: _Rings) -> None:
    """Flips if ``count_ring_reads`` over a night's notes takes more than twice the same count over a table holding
    only its ring reads: the partial index's predicate then no longer bounds the read."""
    full = ChangeNoteKeeper(rings.pool, notes_schema=rings.full, log=_NoLog())
    few = ChangeNoteKeeper(rings.pool, notes_schema=rings.few, log=_NoLog())
    baseline, took = await best_seconds_interleaved(
        lambda: few.count_ring_reads(_SWITCHED), lambda: full.count_ring_reads(_SWITCHED)
    )
    problem = budget_problem(f"counting {_RING_READS} ring reads among {_NOTES:,} notes", took, baseline)
    assert problem is None, problem
