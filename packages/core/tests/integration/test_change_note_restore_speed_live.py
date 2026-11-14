"""Integration: putting a lost change log back reads its coverage within twice the bare reads of it.

A restore reads every note of the coverage (the current and the previous ring tables): ``_count_since`` counts
them, ``_estimate`` sizes their records in the database, and ``_StreamedChanges._stream`` reads the whole-table notes
and then the distinct keyed ones (declared ``whole_read(kind="cold_start")``, Pace 2026-11-14; the proof that only a
restore reaches them is ``tests/unit/collections/test_change_note_restore_reads.py``). This seeds the current table
with a night's notes, loses the log, and times the keeper's pass that restores it against the bare reads a restore
needs on the same database: one count, one sized distinct read, and one read of every note.

Against the session ``db_container``; set ``THREETEARS_TEST_POSTGRES_URL`` to a YugabyteDB URL to run it there.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterable, AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Any

import asyncpg
import pytest
from threetears.nats.change_log import ChangeLogState

from threetears.core.collections.change_notes import ChangeNoteKeeper
from threetears.core.ddl.change_notes import RING_TABLE, note_table
from threetears.core.testing.calibrate import best_seconds_interleaved, budget_problem
from threetears.core.testing.note_ring import install_note_ring

pytestmark = pytest.mark.integration

_NOTES = 200_000
_TABLES = 2_000
#: every this many tables, one was loaded whole
_WHOLE_EVERY = 40
_LOST = "lost-log"
_RESTORED = "restored-log"


@dataclass
class _Ring:
    pool: asyncpg.Pool
    schema: str


class _LostLog:
    """a change log that is gone: no state until a restore declares the new one, which takes every change."""

    def __init__(self) -> None:
        self.declared = False
        self.appended = 0

    async def log_state(self) -> ChangeLogState | None:
        if not self.declared:
            return None
        return ChangeLogState(first_seq=0, last_seq=0, bytes=0, max_bytes=1 << 40, incarnation=_RESTORED)

    async def begin_restore(self, lost: str) -> str | None:
        assert lost == _LOST
        self.declared = True
        return _RESTORED

    async def finish_restore(
        self,
        changes: Sequence[Any] | AsyncIterable[Any],
        *,
        incarnation: str,
        lost: str,
        covered_from: int | None,
    ) -> int:
        assert (incarnation, lost) == (_RESTORED, _LOST)
        assert isinstance(changes, AsyncIterable)
        async for _change in changes:
            self.appended += 1
        return self.appended


@pytest.fixture
async def ring(db_container: str) -> AsyncIterator[_Ring]:
    """a ring whose current table holds a night's keyed notes and a few whole-table ones."""
    schema = f"restore_{uuid.uuid4().hex[:8]}"
    pool = await asyncpg.create_pool(db_container, min_size=1, max_size=2, command_timeout=1200)
    assert pool is not None
    try:
        await pool.execute(f"CREATE SCHEMA {schema}")
        await install_note_ring(pool, schema)
        notes = f"{schema}.{note_table(0)}"
        await pool.execute(
            f"INSERT INTO {notes} (table_name, keys, key_text) "  # noqa: S608 - test identifiers
            f"SELECT 'table_' || (g % {_TABLES}), ARRAY[g::text], g::text FROM generate_series(1, {_NOTES}) g"
        )
        await pool.execute(
            f"INSERT INTO {notes} (table_name) "  # noqa: S608 - test identifiers
            f"SELECT 'table_' || g FROM generate_series(0, {_TABLES - 1}, {_WHOLE_EVERY}) g"
        )
        yield _Ring(pool=pool, schema=schema)
    finally:
        await pool.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        await pool.close()


async def test_a_restore_reads_its_coverage_within_twice_the_bare_reads(ring: _Ring) -> None:
    """Flips if the keeper's pass putting a lost log back (``_count_since``, ``_estimate`` and ``_stream`` over the
    coverage) takes more than twice the bare count, sized distinct read and read of every note, or puts back other
    than one change per table's whole notes and per table's keyed ones."""
    notes = f"{ring.schema}.{note_table(0)}"
    ring_row = f"{ring.schema}.{RING_TABLE}"
    appended: list[int] = []

    async def bare() -> None:
        await ring.pool.fetchval(f"SELECT count(*) FROM {notes}")  # noqa: S608 - test identifiers
        await ring.pool.fetch(
            f"SELECT count(*), sum(length(table_name) + coalesce(length(key_text), 0)) "  # noqa: S608 - test identifiers
            f"FROM (SELECT DISTINCT table_name, key_text, columns FROM {notes}) d"
        )
        await ring.pool.fetch(f"SELECT table_name, tenant, key_text, keys, columns FROM {notes}")  # noqa: S608

    async def restore() -> None:
        await ring.pool.execute(f"UPDATE {ring_row} SET log_incarnation = $1", _LOST)  # noqa: S608 - test identifiers
        log = _LostLog()
        restored = (await ChangeNoteKeeper(ring.pool, notes_schema=ring.schema, log=log).run_pass()).restored
        assert restored == _RESTORED
        appended.append(log.appended)

    baseline, took = await best_seconds_interleaved(bare, restore, runs=3)

    # one change per table loaded whole, and one per other table (its keyed notes, fewer than a chunk)
    assert set(appended) == {_TABLES}, appended
    problem = budget_problem(f"restoring a log from {_NOTES:,} notes", took, baseline)
    assert problem is None, problem
