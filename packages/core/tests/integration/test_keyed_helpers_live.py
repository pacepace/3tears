"""Integration: each keyed-read helper reads about its own rows at real size, and keeps its calibrated budget.

A table shaped like the reports pod's results (300,000 rows: 50 states, 200 geographies a state, 30 rows each, keyed
``(state, geo, id)``, a stamp index for change records and expiry), seeded in a schema of its own. Each helper's
statements are planned for real (:class:`~threetears.core.testing.plans.PlanRecorder`) and judged by the shared
plan check (rows scanned within 1.1x the rows answered; a count's within its bound), then timed against the bare
operation on a plain table holding only the rows it reads (:mod:`threetears.core.testing.calibrate`), held to 2x.

Against the session ``db_container`` (Postgres); set ``THREETEARS_TEST_POSTGRES_URL`` to a YugabyteDB URL to run
them there, the platform's database.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import asyncpg
import pytest

from threetears.core.keyed import (
    AnyOf,
    KeyBatch,
    Sort,
    count_at_most,
    counts_by,
    delete_before,
    holds_any,
    keyset_page,
    read_keyed,
    read_past_marks,
)
from threetears.core.testing.calibrate import best_seconds, budget_problem, plain_copies
from threetears.core.testing.plans import PlanRecorder, on_yugabyte, plan_problems

pytestmark = pytest.mark.integration

_TABLE = "results"
_STATES = 50
_GEOS = 200
_IDS = 30
_START = datetime(2026, 11, 3, tzinfo=UTC)


@dataclass
class _Seeded:
    conn: asyncpg.Connection
    yugabyte: bool


@pytest.fixture
async def seeded(db_container: str) -> AsyncIterator[_Seeded]:
    """the results table at real size, with its stamp index and fresh statistics, in a schema of its own."""
    schema = f"keyed_{uuid.uuid4().hex[:8]}"
    conn = await asyncpg.connect(db_container)
    try:
        await conn.execute(f"CREATE SCHEMA {schema}")
        await conn.execute(f"SET search_path = {schema}")
        await conn.execute(
            f"CREATE TABLE {_TABLE} (state TEXT NOT NULL, geo TEXT NOT NULL, id BIGINT NOT NULL, votes BIGINT, "
            "stamp TIMESTAMPTZ NOT NULL, pad TEXT, PRIMARY KEY (state, geo, id))"
        )
        await conn.execute(f"CREATE INDEX results_by_stamp ON {_TABLE} (stamp ASC)")
        await conn.execute(
            f"""INSERT INTO {_TABLE}
            SELECT 'S' || lpad(s::text, 2, '0'), lpad(g::text, 3, '0'), i, s * g * i,
                   $1::timestamptz + make_interval(secs => (s * {_GEOS} * {_IDS} + g * {_IDS} + i)), repeat('x', 40)
            FROM generate_series(1, {_STATES}) s, generate_series(1, {_GEOS}) g, generate_series(1, {_IDS}) i""",
            _START,
        )
        await conn.execute(f"ANALYZE {_TABLE}")
        yield _Seeded(conn=conn, yugabyte=await on_yugabyte(conn))
    finally:
        await conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        await conn.close()


async def _baseline(conn: asyncpg.Connection, rows_sql: str, *args: Any) -> float:
    """the bare read of the same rows: one statement over a plain table holding only them."""
    async with plain_copies(conn, {"rows": (rows_sql, args)}) as plain:
        return await best_seconds(lambda: conn.fetch(f"SELECT * FROM {plain['rows']}"))  # noqa: S608


def _plans_ok(recorder: PlanRecorder, seeded: _Seeded, *, scan_ratio: float | None = 1.1) -> None:
    # Postgres plans a small batch its own way (a bitmap scan, a sort); the rows-scanned ratio is the platform's
    # (YugabyteDB) contract, and every database's plan is held to no looped sequential scan
    problems = plan_problems(recorder.plans, scan_ratio=scan_ratio if seeded.yugabyte else None)
    assert not problems, problems


def _keys(count: int) -> list[tuple[str, str, int]]:
    return [(f"S{1 + n % _STATES:02d}", f"{1 + (n * 7) % _GEOS:03d}", 1 + n % _IDS) for n in range(count)]


class TestReadKeyed:
    async def test_a_batch_of_keys_reads_its_rows_within_budget(self, seeded: _Seeded) -> None:
        """Flips if a keyed batch read scans past its keys' rows, or is over 2x the bare read of them."""
        keys = list(dict.fromkeys(_keys(2000)))
        batch = KeyBatch(("state", "geo", "id"), keys, ("text", "text", "bigint"))
        recorder = PlanRecorder(seeded.conn)
        rows = await read_keyed(recorder, _TABLE, keys=batch, columns=["state", "geo", "id", "votes"])
        assert sorted((r["state"], r["geo"], r["id"]) for r in rows) == sorted(keys)
        _plans_ok(recorder, seeded)
        baseline = await _baseline(
            seeded.conn,
            f"SELECT state, geo, id, votes FROM {_TABLE} WHERE (state, geo, id) IN "
            "(SELECT * FROM unnest($1::text[], $2::text[], $3::bigint[]))",
            [k[0] for k in keys],
            [k[1] for k in keys],
            [k[2] for k in keys],
        )
        took = await best_seconds(
            lambda: read_keyed(seeded.conn, _TABLE, keys=batch, columns=["state", "geo", "id", "votes"])
        )
        problem = budget_problem(f"read_keyed of {len(keys)} keys", took, baseline)
        assert problem is None, problem

    async def test_a_batch_of_prefixes_reads_their_rows(self, seeded: _Seeded) -> None:
        """Flips if a prefix batch (state, geo) reads past the prefixes' rows."""
        prefixes = [("S07", f"{g:03d}") for g in range(1, 41)]
        recorder = PlanRecorder(seeded.conn)
        rows = await read_keyed(
            recorder, _TABLE, keys=KeyBatch(("state", "geo"), prefixes), columns=["state", "geo", "id"]
        )
        assert len(rows) == len(prefixes) * _IDS
        _plans_ok(recorder, seeded)


class TestKeysetPage:
    async def test_a_page_reads_about_its_rows_within_budget(self, seeded: _Seeded) -> None:
        """Flips if a page reads the prefix whole (a sort of every row) or is over 2x the bare page read."""
        order = [Sort("geo"), Sort("id", "bigint")]
        first = await keyset_page(
            seeded.conn, _TABLE, columns=["geo", "id", "votes"], by={"state": "S10"}, order=order, limit=200
        )
        recorder = PlanRecorder(seeded.conn)
        page = await keyset_page(
            recorder,
            _TABLE,
            columns=["geo", "id", "votes"],
            by={"state": "S10"},
            order=order,
            limit=200,
            cursor=first.next_cursor,
        )
        expected = await seeded.conn.fetch(
            f"SELECT geo, id FROM {_TABLE} WHERE state = 'S10' ORDER BY geo, id OFFSET 200 LIMIT 200"
        )
        assert [(r["geo"], r["id"]) for r in page.rows] == [(r["geo"], r["id"]) for r in expected]
        _plans_ok(recorder, seeded)
        last = first.rows[-1]
        baseline = await _baseline(
            seeded.conn,
            f"SELECT geo, id, votes FROM {_TABLE} WHERE state = $1 AND (geo, id) > ($2, $3) ORDER BY geo, id LIMIT 201",
            "S10",
            last["geo"],
            last["id"],
        )
        took = await best_seconds(
            lambda: keyset_page(
                seeded.conn,
                _TABLE,
                columns=["geo", "id", "votes"],
                by={"state": "S10"},
                order=order,
                limit=200,
                cursor=first.next_cursor,
            )
        )
        problem = budget_problem("keyset_page of 200 rows", took, baseline)
        assert problem is None, problem


class TestCounts:
    async def test_counts_by_counts_only_its_keys_rows(self, seeded: _Seeded) -> None:
        """Flips if a keyed count reads rows no key names, or is over 2x the bare read of the rows it counts."""
        geos = [f"{g:03d}" for g in range(1, 101)]
        recorder = PlanRecorder(seeded.conn)
        counted = await counts_by(recorder, _TABLE, column="geo", values=AnyOf(geos), by={"state": "S20"})
        assert counted == dict.fromkeys(geos, _IDS)
        assert all(plan.rows_scanned <= 1.1 * len(geos) * _IDS for plan in recorder.plans), recorder.plans
        baseline = await _baseline(
            seeded.conn, f"SELECT state, geo FROM {_TABLE} WHERE state = $1 AND geo = ANY($2::text[])", "S20", geos
        )
        took = await best_seconds(
            lambda: counts_by(seeded.conn, _TABLE, column="geo", values=AnyOf(geos), by={"state": "S20"})
        )
        problem = budget_problem("counts_by of 100 geographies", took, baseline)
        assert problem is None, problem

    async def test_bounded_counts_and_probes_read_no_further_than_their_bound(self, seeded: _Seeded) -> None:
        """Flips if a bounded count or a probe reads every row its key names."""
        recorder = PlanRecorder(seeded.conn)
        bounded = await count_at_most(recorder, _TABLE, by={"state": "S30"}, at_most=100)
        assert (bounded.count, bounded.more) == (100, True)
        assert await holds_any(recorder, _TABLE, by={"state": "S30"})
        assert not await holds_any(recorder, _TABLE, by={"state": "none"})
        assert recorder.plans[0].rows_scanned <= 102, recorder.plans[0]
        assert all(plan.rows_scanned <= 2 for plan in recorder.plans[1:]), recorder.plans[1:]


class TestMarks:
    async def test_a_read_past_a_mark_reads_only_what_moved_within_budget(self, seeded: _Seeded) -> None:
        """Flips if a change-record read scans past the rows that moved, or is over 2x the bare read of them."""
        newest = await seeded.conn.fetchval(f"SELECT stamp FROM {_TABLE} ORDER BY stamp DESC LIMIT 1")
        mark = newest - timedelta(seconds=1500)
        recorder = PlanRecorder(seeded.conn)
        read = await read_past_marks(recorder, _TABLE, columns=["state", "geo", "id", "stamp"], marks={"stamp": mark})
        assert len(read.rows) == 1500
        assert read.marks == {"stamp": newest}
        _plans_ok(recorder, seeded)
        baseline = await _baseline(seeded.conn, f"SELECT state, geo, id, stamp FROM {_TABLE} WHERE stamp > $1", mark)
        took = await best_seconds(
            lambda: read_past_marks(seeded.conn, _TABLE, columns=["state", "geo", "id", "stamp"], marks={"stamp": mark})
        )
        problem = budget_problem("read_past_marks of 1,500 moved rows", took, baseline)
        assert problem is None, problem


class TestDeleteBefore:
    async def test_a_sweep_deletes_page_by_page_within_budget(self, seeded: _Seeded) -> None:
        """Flips if a sweep's pages read past their own rows, or the sweep is over 2x one bare delete of them."""
        oldest = await seeded.conn.fetchval(f"SELECT stamp FROM {_TABLE} ORDER BY stamp LIMIT 1")
        before = oldest + timedelta(seconds=3000)
        baseline_rows = f"SELECT state, geo, id, stamp FROM {_TABLE} WHERE stamp < $1"
        async with plain_copies(seeded.conn, {"rows": (baseline_rows, (before,))}) as plain:
            began = time.perf_counter()
            await seeded.conn.execute(f"DELETE FROM {plain['rows']} WHERE stamp < $1", before)  # noqa: S608
            baseline = time.perf_counter() - began
        began = time.perf_counter()
        swept = await delete_before(
            seeded.conn, _TABLE, key=["state", "geo", "id"], column="stamp", before=before, page=500
        )
        took = time.perf_counter() - began
        assert swept.deleted == 3000
        assert swept.statements == 7
        assert not await holds_any(seeded.conn, _TABLE, by={"state": "S01", "geo": "001"})
        problem = budget_problem("delete_before of 3,000 rows in pages of 500", took, baseline)
        assert problem is None, problem
