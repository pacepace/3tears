"""Integration: putting a lost change log back reads its coverage within twice the bare reads of it.

A restore reads every note of the coverage (the current and the previous ring tables): ``_count_since`` counts
them and bounds their records' size, ``_estimate`` sizes the distinct rows in the database only when that bound does
not fit the log, and ``_StreamedChanges._stream`` reads the whole-table notes and then the keyed ones
(``_StreamedChanges._folded``: a ``COPY`` folded in the restore's worker process), folded to the distinct rows
(declared ``whole_read(kind="cold_start")``, Pace 2026-11-14; the proof that only a restore reaches them is ``tests/unit/collections/test_change_note_restore_reads.py``). This seeds the current table
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

from threetears.core.collections.change_notes import RESTORE_BUDGET_RATIO, ChangeNoteKeeper
from threetears.core.collections.epoch_change import EpochChange
from threetears.core.ddl.change_notes import RING_TABLE, note_table
from threetears.core.testing.calibrate import best_seconds_interleaved, budget_problem
from threetears.core.testing.loop_lateness import turn_holds
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

    def __init__(self, max_bytes: int = 1 << 40) -> None:
        self.declared = False
        self.appended = 0
        self.max_bytes = max_bytes
        #: the changes each finish_restore call took before it ended (a refused one ends early, by raising)
        self.calls: list[int] = []

    async def log_state(self) -> ChangeLogState | None:
        if not self.declared:
            return None
        return ChangeLogState(first_seq=0, last_seq=0, bytes=0, max_bytes=self.max_bytes, incarnation=_RESTORED)

    async def epoch_at(self, seq: int) -> EpochChange | None:
        raise AssertionError(f"a restore reads no record of the lost log ({seq})")

    async def start_log(self) -> str:
        raise AssertionError("a restore declares its log through begin_restore")

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
        self.calls.append(0)
        async for _change in changes:
            self.appended += 1
            self.calls[-1] += 1
        return self.appended


async def _seeded(db_container: str, *, rows: int, max_size: int = 2) -> AsyncIterator[_Ring]:
    """a ring whose current table holds ``_NOTES`` keyed notes naming ``rows`` distinct rows (each noted
    ``_NOTES / rows`` times) and a few whole-table ones, on a pool of ``max_size`` connections."""
    schema = f"restore_{uuid.uuid4().hex[:8]}"
    pool = await asyncpg.create_pool(db_container, min_size=1, max_size=max_size, command_timeout=1200)
    assert pool is not None
    try:
        await pool.execute(f"CREATE SCHEMA {schema}")
        await install_note_ring(pool, schema)
        notes = f"{schema}.{note_table(0)}"
        await pool.execute(
            f"INSERT INTO {notes} (table_name, keys, key_text) "  # noqa: S608 - test identifiers
            f"SELECT 'table_' || (g % {_TABLES}), ARRAY[(g % {rows})::text], (g % {rows})::text "
            f"FROM generate_series(1, {_NOTES}) g"
        )
        await pool.execute(
            f"INSERT INTO {notes} (table_name) "  # noqa: S608 - test identifiers
            f"SELECT 'table_' || g FROM generate_series(0, {_TABLES - 1}, {_WHOLE_EVERY}) g"
        )
        yield _Ring(pool=pool, schema=schema)
    finally:
        await pool.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        await pool.close()


@pytest.fixture
async def ring(db_container: str) -> AsyncIterator[_Ring]:
    """a ring whose current table holds a night's keyed notes, each row noted once, and a few whole-table ones."""
    async for seeded in _seeded(db_container, rows=_NOTES):
        yield seeded


@pytest.fixture
async def ring_twice(db_container: str) -> AsyncIterator[_Ring]:
    """as ``ring``, each keyed row noted twice: the count's bound (every note sized) is about twice the distinct
    rows' estimate, so a log between them makes the restore size the distinct rows."""
    async for seeded in _seeded(db_container, rows=_NOTES // 2):
        yield seeded


@pytest.fixture
async def ring_twice_one_connection(db_container: str) -> AsyncIterator[_Ring]:
    """as ``ring_twice``, on a pool of one connection: the estimate cannot run beside the stream."""
    async for seeded in _seeded(db_container, rows=_NOTES // 2, max_size=1):
        yield seeded


async def _sizes(ring: _Ring) -> tuple[int, int]:
    """the restore's two sizes of the coverage: the distinct rows' estimate and the count's bound (every note), each
    its rows' table and key text plus 64 bytes a row and 256 a table, as the keeper sizes a restored record."""
    notes = f"{ring.schema}.{note_table(0)}"
    sized = "sum(length(table_name) + coalesce(length(key_text), 0)) + 64 * count(*)"
    distinct = await ring.pool.fetchval(
        f"SELECT {sized} FROM (SELECT DISTINCT table_name, key_text FROM {notes}) d"  # noqa: S608 - test identifiers
    )
    every = await ring.pool.fetchval(f"SELECT {sized} FROM {notes}")  # noqa: S608 - test identifiers
    return int(distinct) + 256 * _TABLES, int(every) + 256 * _TABLES


async def test_a_restore_reads_its_coverage_within_twice_the_bare_reads(ring: _Ring) -> None:
    """Flips if the keeper's pass putting a lost log back (``_count_since``, ``_estimate`` and ``_stream`` with its
    keyed notes' ``_folded`` over the coverage) takes more than twice the bare count, sized distinct read and read of every note, or puts back other
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


async def _bare(ring: _Ring) -> None:
    """the bare reads a restore needs: one count, one sized distinct read, one read of every note."""
    notes = f"{ring.schema}.{note_table(0)}"
    await ring.pool.fetchval(f"SELECT count(*) FROM {notes}")  # noqa: S608 - test identifiers
    await ring.pool.fetch(
        f"SELECT count(*), sum(length(table_name) + coalesce(length(key_text), 0)) "  # noqa: S608 - test identifiers
        f"FROM (SELECT DISTINCT table_name, key_text, columns FROM {notes}) d"
    )
    await ring.pool.fetch(f"SELECT table_name, tenant, key_text, keys, columns FROM {notes}")  # noqa: S608


async def _restore_into(ring: _Ring, log: _LostLog) -> str | None:
    """lose the ring's log and run the keeper's pass that restores it into ``log``; the restored incarnation."""
    ring_row = f"{ring.schema}.{RING_TABLE}"
    await ring.pool.execute(f"UPDATE {ring_row} SET log_incarnation = $1", _LOST)  # noqa: S608 - test identifiers
    return (await ChangeNoteKeeper(ring.pool, notes_schema=ring.schema, log=log).run_pass()).restored


async def test_a_restore_whose_bound_does_not_fit_reads_within_twice_the_bare_reads(ring_twice: _Ring) -> None:
    """Flips if a restore whose coverage fits the log only once its distinct rows are sized (each row noted twice, so
    the count's bound passes the budget and the estimate does not) takes more than twice the bare reads, or puts
    back other than the whole coverage: one change per table's whole notes and per table's keyed ones."""
    estimate, bound = await _sizes(ring_twice)
    budget = (estimate + bound) / 2
    assert estimate < budget < bound
    max_bytes = int(budget / RESTORE_BUDGET_RATIO)
    calls: list[list[int]] = []

    async def bare() -> None:
        await _bare(ring_twice)

    async def restore() -> None:
        log = _LostLog(max_bytes=max_bytes)
        assert await _restore_into(ring_twice, log) == _RESTORED
        calls.append(log.calls)

    baseline, took = await best_seconds_interleaved(bare, restore, runs=3)
    # the whole coverage, in one finish_restore: one change per table loaded whole, and one per other table
    assert {tuple(each) for each in calls} == {(_TABLES,)}, calls
    problem = budget_problem(f"restoring a log from {_NOTES:,} notes, its bound over the budget", took, baseline)
    assert problem is None, problem


async def _appends_nothing(ring: _Ring) -> None:
    """a restore whose whole coverage does not fit, with no mark to cut it at: refused, nothing appended."""
    estimate, _bound = await _sizes(ring)
    log = _LostLog(max_bytes=int(estimate * 0.9 / RESTORE_BUDGET_RATIO))
    assert await _restore_into(ring, log) is None
    assert log.appended == 0
    assert log.calls in ([], [0]), log.calls


async def _restored_whole(ring: _Ring) -> None:
    """a restore whose bound does not fit but whose distinct rows do: the whole coverage, in one finish_restore."""
    estimate, bound = await _sizes(ring)
    log = _LostLog(max_bytes=int((estimate + bound) / 2 / RESTORE_BUDGET_RATIO))
    assert await _restore_into(ring, log) == _RESTORED
    assert log.calls == [_TABLES]


async def test_a_coverage_that_does_not_fit_appends_nothing(ring_twice: _Ring) -> None:
    """Flips if a restore whose whole coverage does not fit (and has no mark to cut it at) lets any change reach the
    new log before it is refused, the estimate running beside the stream."""
    await _appends_nothing(ring_twice)


async def test_a_coverage_that_does_not_fit_appends_nothing_on_one_connection(
    ring_twice_one_connection: _Ring,
) -> None:
    """Flips if, on a pool of one connection (no estimate beside the stream: it sizes first), a restore whose whole
    coverage does not fit deadlocks, fails, or appends anything."""
    await _appends_nothing(ring_twice_one_connection)


async def test_a_coverage_that_fits_only_by_its_estimate_is_restored_whole(ring_twice: _Ring) -> None:
    """Flips if a restore whose bound does not fit but whose distinct rows do puts back other than the whole
    coverage, the estimate running beside the stream."""
    await _restored_whole(ring_twice)


async def test_a_coverage_that_fits_only_by_its_estimate_is_restored_whole_on_one_connection(
    ring_twice_one_connection: _Ring,
) -> None:
    """Flips if, on a pool of one connection (sized first), a restore whose distinct rows fit puts back other than
    the whole coverage, or deadlocks."""
    await _restored_whole(ring_twice_one_connection)


#: the most one turn of the loop may hold it during a restore, in CPU: test_arrow_copy's bound for a turn
_TURN_BUDGET = 0.001


@pytest.mark.loop_timing
async def test_a_restore_holds_the_loop_under_1_ms_a_turn(ring: _Ring) -> None:
    """Flips if any one turn of the event loop during a restore of the 200,000-note ring holds it 1 ms of CPU or more:
    the notes are read, folded and written into records off the loop (threetears.core.change_note_fold), and the loop
    only moves bytes. Read as rows and folded on the loop it held 29.1 ms at worst on PG 16 and 22.8 ms on YugabyteDB
    (p99 7.8 and 8.5 ms; L1, 2026-11-15). The first restore of a process spawns the worker (on a thread), so one runs
    before the one measured."""
    assert await _restore_into(ring, _LostLog()) == _RESTORED
    log = _LostLog()
    with turn_holds() as holds:
        assert await _restore_into(ring, log) == _RESTORED
    assert log.appended == _TABLES
    cpu = sorted(held for held, _ in holds)
    p99 = cpu[int(len(cpu) * 0.99)]
    summary = (
        f"{len(cpu)} turns, the longest {cpu[-1] * 1000:.2f} ms, p99 {p99 * 1000:.3f} ms; "
        f"the longest five {sorted(holds)[-5:]}"
    )
    assert cpu[-1] < _TURN_BUDGET, summary
