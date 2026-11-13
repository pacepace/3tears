"""Integration: the no-whole-reads rule's runtime half fails a load path that reads a whole table, on a real plan.

:class:`~threetears.core.testing.plans.WholeScanGuard` wraps the connection a load-path test hands its code. These
seed a table past the guard's threshold and read it two ways through the real pager, ``read_l3_rows``: by one
state (keyed: it passes) and with no filter (every page keyed by its cursor, the table read whole: it fails), and
a count of the whole table (a sequential scan: it fails). Against the session ``db_container``.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator

import asyncpg
import pytest

from threetears.core.collections.complete_copy import read_l3_rows
from threetears.core.testing.plans import WholeScanGuard

pytestmark = pytest.mark.integration

_STATES = 10
_PER_STATE = 300


@pytest.fixture
async def seeded(db_container: str) -> AsyncIterator[asyncpg.Connection]:
    """3,000 rows in a schema of their own, keyed and indexed by state, with fresh statistics."""
    schema = f"wholescan_{uuid.uuid4().hex[:8]}"
    conn = await asyncpg.connect(db_container)
    try:
        await conn.execute(f"CREATE SCHEMA {schema}")
        await conn.execute(f"SET search_path = {schema}")
        await conn.execute("CREATE TABLE results (state TEXT NOT NULL, id INT NOT NULL, PRIMARY KEY (state, id))")
        await conn.executemany(
            "INSERT INTO results (state, id) VALUES ($1, $2)",
            [(f"S{s:02d}", i) for s in range(_STATES) for i in range(_PER_STATE)],
        )
        await conn.execute("ANALYZE results")
        yield conn
    finally:
        await conn.execute(f"DROP SCHEMA {schema} CASCADE")
        await conn.close()


async def test_a_keyed_read_passes_the_guard(seeded: asyncpg.Connection) -> None:
    """Flips if a read by one state's key is taken for a whole read."""
    async with WholeScanGuard(seeded, threshold_rows=1000) as guarded:
        rows = await read_l3_rows(guarded, "results", ("state", "id"), ("state", "id"), where={"state": "S03"})
    assert len(rows) == _PER_STATE


async def test_a_whole_read_through_the_pager_fails_the_guard(seeded: asyncpg.Connection) -> None:
    """Flips if a pager walking the whole table passes because each of its pages is keyed by its cursor."""
    guarded = WholeScanGuard(seeded, threshold_rows=1000)
    rows = await read_l3_rows(guarded, "results", ("state", "id"), ("state", "id"), page_size=500)
    assert len(rows) == _STATES * _PER_STATE
    with pytest.raises(AssertionError, match="whole read"):
        guarded.check()


async def test_a_whole_count_fails_the_guard(seeded: asyncpg.Connection) -> None:
    """Flips if a count over every row (a verification scan) passes the guard."""
    guarded = WholeScanGuard(seeded, threshold_rows=1000)
    assert await guarded.fetchval("SELECT count(*) FROM results") == _STATES * _PER_STATE
    assert any("results (3000 rows" in problem for problem in guarded.problems)


async def test_a_partial_index_walked_whole_through_a_bitmap_fails_the_guard(seeded: asyncpg.Connection) -> None:
    """Flips if a bitmap heap scan over a bitmap index scan with no index condition passes: every entry of a partial
    index read, the filter on another column applied after (hub: credit rates by model over the open-rates index)."""
    await seeded.execute("ALTER TABLE results ADD COLUMN open BOOLEAN NOT NULL DEFAULT true, ADD COLUMN model INT")
    await seeded.execute("UPDATE results SET model = id % 50")
    await seeded.execute("CREATE INDEX results_open_by_id ON results (id) WHERE open")
    await seeded.execute("ANALYZE results")
    await seeded.execute("SET enable_seqscan = off")
    try:
        guarded = WholeScanGuard(seeded, threshold_rows=1000)
        plan = await seeded.fetchval("EXPLAIN (FORMAT JSON) SELECT * FROM results WHERE open AND model = ANY($1)", [7])
        rows = await guarded.fetch("SELECT * FROM results WHERE open AND model = ANY($1)", [7])
    finally:
        await seeded.execute("RESET enable_seqscan")
    assert len(rows) == _STATES * _PER_STATE // 50
    if "Bitmap Index Scan" not in plan or "Index Cond" in plan:
        pytest.skip(f"the planner did not walk the partial index by a bitmap here: {plan}")
    assert any("bitmap index scan with no index condition of" in problem for problem in guarded.problems), (
        guarded.problems
    )
