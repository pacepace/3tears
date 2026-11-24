"""Integration: a platform pool plans every run of a keyed read for its own keys, never one generic plan from the sixth.

PostgreSQL and YugabyteDB switch a prepared statement (asyncpg's statement cache) to a generic plan after five runs
(``plan_cache_mode = auto``) when its estimate is no worse than the custom plans' average. A keyed read's key arrays are
unknown to a generic plan (planned for any keys), so a composite keyed read collapsed from the sixth run on
(design-pod-delta-files.md, decision 39). A platform pool
(:func:`~threetears.core.utils.pg_pool_kwargs.create_pool_with_startup_timeout`) starts its sessions with
``plan_cache_mode = force_custom_plan``.

A table at real size in a schema of their own: 300,000 rows keyed ``(state, geo, id)`` (the reports pod's results).
Measured 2026-11-23 on the testament pool, runs 1..12 of one statement on one session, ``auto`` against
``force_custom_plan``: 50,000 keys over the table, PostgreSQL 16 ~88 ms a run to ~417 ms from the sixth (4.7x), ~88 ms
throughout with custom plans; 6,000 keys of one state 8.5 -> 23.8 ms (2.8x). YugabyteDB 2026.1 planned the same
statements generically from the sixth run (``generic_plans`` 7) at about the custom plans' time for these shapes; the
Arrow-mode composite binding's bounds collapsed there (~15 -> ~430 ms).
"""

from __future__ import annotations

import statistics
import time
import uuid
from collections.abc import AsyncIterator

import asyncpg
import pytest

from threetears.core.keyed import KeyBatch, read_keyed
from threetears.core.utils.pg_pool_kwargs import create_pool_with_startup_timeout

pytestmark = pytest.mark.integration

_STATES, _GEOS, _IDS = 50, 200, 30
_CASTS = ("text", "text", "bigint")
#: runs of the read on one session: the first five are custom plans under ``auto`` either way; the sixth on switched
_RUNS = 12
#: the median run from the sixth on, at most this many times the first five's median (``auto`` on PostgreSQL: 4.7x)
_FACTOR = 2.0


def _spread(count: int) -> KeyBatch:
    """``count`` composite keys spread evenly over the whole table."""
    step = _STATES * _GEOS * _IDS // count
    return KeyBatch(
        ("state", "geo", "id"),
        [
            (f"S{(n // (_GEOS * _IDS)) + 1:02d}", f"{((n // _IDS) % _GEOS) + 1:03d}", (n % _IDS) + 1)
            for n in range(0, step * count, step)
        ],
        _CASTS,
    )


#: a tight-bounded read: one state, 20 geographies, every id (600 keys, each column's distinct values naming exactly
#: them, so each column is bound to its distinct values beside the keys)
_TIGHT = KeyBatch(("state", "geo", "id"), [("S07", f"{g:03d}", i) for g in range(41, 61) for i in range(1, 31)], _CASTS)


@pytest.fixture
async def pool(db_container: str) -> AsyncIterator[asyncpg.Pool]:
    """a one-connection platform pool on the table at real size, with fresh statistics, in a schema of its own."""
    schema = f"kcustom_{uuid.uuid4().hex[:8]}"
    held = await asyncpg.connect(db_container)
    try:
        await held.execute(f"CREATE SCHEMA {schema}")
        await held.execute(f"SET search_path = {schema}")
        await held.execute(
            "CREATE TABLE kc (state TEXT NOT NULL, geo TEXT NOT NULL, id BIGINT NOT NULL, v BIGINT, "
            "PRIMARY KEY (state, geo, id))"
        )
        await held.execute(
            f"""INSERT INTO kc SELECT 'S' || lpad(s::text, 2, '0'), lpad(g::text, 3, '0'), i, s * g * i
            FROM generate_series(1, {_STATES}) s, generate_series(1, {_GEOS}) g, generate_series(1, {_IDS}) i"""
        )
        await held.execute("ANALYZE kc")
        platform = await create_pool_with_startup_timeout(
            db_container,
            pool_name="keyed_custom_plans",
            min_size=1,
            max_size=1,
            server_settings={"search_path": schema},
        )
        try:
            yield platform
        finally:
            await platform.close()
    finally:
        await held.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        await held.close()


async def _runs(conn: asyncpg.Connection, keys: KeyBatch) -> list[float]:
    """each of :data:`_RUNS` runs of the read of ``keys`` (one statement a run), in seconds; each answers every key."""
    seconds: list[float] = []
    for _ in range(_RUNS):
        started = time.perf_counter()
        rows = await read_keyed(conn, "kc", keys=keys, columns=["v"], batch=len(keys.values), records=True)
        seconds.append(time.perf_counter() - started)
        assert len(rows) == len(keys.values)
    return seconds


async def test_a_platform_session_never_plans_a_keyed_read_generically(pool: asyncpg.Pool) -> None:
    """Flips if a platform pool's session runs without ``plan_cache_mode = force_custom_plan``, or plans any of 12 runs
    of the same tight-bounded composite keyed read generically (``pg_prepared_statements.generic_plans``; ``auto``
    planned runs 6..12 generically on PostgreSQL and YugabyteDB)."""
    async with pool.acquire() as conn:
        assert await conn.fetchval("SHOW plan_cache_mode") == "force_custom_plan"
        await _runs(conn, _TIGHT)
        plans = await conn.fetch(
            "SELECT statement, generic_plans, custom_plans FROM pg_prepared_statements "
            "WHERE statement LIKE '%kc%' AND statement NOT LIKE '%pg_prepared_statements%'"
        )
    assert len(plans) == 1, [plan["statement"][:120] for plan in plans]
    assert (plans[0]["generic_plans"], plans[0]["custom_plans"]) == (0, _RUNS)


@pytest.mark.loop_timing
async def test_a_keyed_read_stays_as_fast_from_its_sixth_run(pool: asyncpg.Pool) -> None:
    """Flips if the median of runs 6..12 of a 50,000-key composite read (where ``auto`` switched to a generic plan)
    takes more than :data:`_FACTOR` times the first five runs' median, on a platform pool's session."""
    async with pool.acquire() as conn:
        seconds = await _runs(conn, _spread(50_000))
    shown = ", ".join(f"{run * 1000:.1f}" for run in seconds)
    assert statistics.median(seconds[5:]) <= _FACTOR * statistics.median(seconds[:5]), f"runs (ms): {shown}"
