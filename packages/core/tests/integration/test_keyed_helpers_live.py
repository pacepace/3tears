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

import asyncio
import sqlite3
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, cast

import asyncpg
import pytest

from threetears.core.keyed import (
    AnyOf,
    KeyBatch,
    Sort,
    count_at_most,
    counts_by,
    ThreadedSQLite,
    counts_by_relations,
    delete_before,
    delete_oldest,
    holds_any,
    keyset_page,
    read_joined,
    read_keyed,
    read_past_marks,
    update_before,
)
from threetears.core.testing.calibrate import best_seconds, budget_problem, plain_copies
from threetears.core.testing.plans import (
    PlanRecorder,
    StatementPlan,
    explain_write,
    on_yugabyte,
    plan_problems,
    plan_uses_index,
)

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


class _Settling:
    """a collection's pool and bypassing write, the keys it is told of kept: what a paged sweep settles."""

    def __init__(self, conn: asyncpg.Connection) -> None:
        self.required_l3_pool = conn
        self.settled: list[Any] = []

    @asynccontextmanager
    async def bypassing_write(self, *entity_ids: Any, conn: Any = None) -> AsyncIterator[Any]:
        settled = self.settled

        class _Write:
            def touches(self, *keys: Any) -> None:
                settled.extend(keys)

        yield _Write()


async def _baseline(conn: asyncpg.Connection, rows_sql: str, *args: Any) -> float:
    """the bare read of the same rows: one statement over a plain table holding only them."""
    async with plain_copies(conn, {"rows": (rows_sql, args)}) as plain:
        return await best_seconds(lambda: conn.fetch(f"SELECT * FROM {plain['rows']}"))  # noqa: S608


class _WritePlans:
    """a connection whose every write is planned (EXPLAIN, never run: :func:`explain_write`) before it runs."""

    def __init__(self, conn: asyncpg.Connection) -> None:
        self._conn = conn
        self.plans: list[StatementPlan] = []

    async def fetch(self, sql: str, *args: Any) -> Any:
        self.plans.append(await explain_write(self._conn, sql, *args))
        return await self._conn.fetch(sql, *args)


_KEY = ["state", "geo", "id"]


def _node_types(plan: Any) -> list[str]:
    node = plan.plan if isinstance(plan, StatementPlan) else plan
    return [str(node.get("Node Type", "")), *(t for child in node.get("Plans", ()) for t in _node_types(child))]


def _batched_on_yugabyte(plan: StatementPlan, seeded: _Seeded) -> None:
    """on YugabyteDB the keys are joined by a batched nested loop: one storage request a batch, never one a key (the
    hint every keyed join carries; this fails if the hint is ignored)."""
    if seeded.yugabyte:
        # YugabyteDB's JSON plan names the node "YB Batched Nested Loop" (its text plan adds " Join")
        assert any(node.startswith("YB Batched Nested Loop") for node in _node_types(plan)), plan.plan


async def _rolled_back(conn: asyncpg.Connection, run: Any) -> float:
    """the best seconds of ``run()`` inside a transaction rolled back after each run: a write timed on the real table,
    which each run finds as it was."""

    async def once() -> None:
        transaction = conn.transaction()
        await transaction.start()
        try:
            await run()
        finally:
            await transaction.rollback()

    return await best_seconds(once)


def _budget(what: str, took: float, baseline: float) -> str | None:
    """the budget's verdict, its numbers printed (``-s``) for the run's record."""
    print(f"BUDGET {what} | baseline {baseline * 1000:.2f} ms | took {took * 1000:.2f} ms | {took / baseline:.2f}x")
    return budget_problem(what, took, baseline)


def _plans_ok(recorder: PlanRecorder, seeded: _Seeded, *, scan_ratio: float | None = 1.1) -> None:
    # Postgres plans a small batch its own way (a bitmap scan, a sort); the rows-scanned ratio is the platform's
    # (YugabyteDB) contract, and every database's plan is held to no looped sequential scan
    problems = plan_problems(recorder.plans, scan_ratio=scan_ratio if seeded.yugabyte else None)
    assert not problems, problems


def _keys(count: int) -> list[tuple[str, str, int]]:
    return [(f"S{1 + n % _STATES:02d}", f"{1 + (n * 7) % _GEOS:03d}", 1 + n % _IDS) for n in range(count)]


class TestReadKeyed:
    async def test_a_batch_of_keys_reads_its_rows_within_budget(self, seeded: _Seeded) -> None:
        """Flips if a keyed batch read scans past its keys' rows, or is over 2x the bare read of them.

        The bare is the best hand-written statement doing the same work on the same table: the same 600 keys looked
        up on the real 300k-row table by its key (IN-unnest of natively typed arrays), returning the same rows. Never
        a scan of a small copy holding only the answer: that skips the lookups the helper must make (it measured
        0.40 ms against 2.30 ms for the keyed lookup, and read 4.6x where the helper is 1.5x the lookup)."""
        keys = list(dict.fromkeys(_keys(2000)))
        batch = KeyBatch(("state", "geo", "id"), keys, ("text", "text", "bigint"))
        recorder = PlanRecorder(seeded.conn)
        rows = await read_keyed(recorder, _TABLE, keys=batch, columns=["state", "geo", "id", "votes"])
        assert sorted((r["state"], r["geo"], r["id"]) for r in rows) == sorted(keys)
        _plans_ok(recorder, seeded)
        # the keys found by the key's index, about one row read a key, on PostgreSQL and YugabyteDB alike
        (plan,) = recorder.plans
        assert plan_uses_index(plan, "results_pkey"), plan.plan
        assert plan.rows_scanned <= 1.1 * len(keys), plan
        _batched_on_yugabyte(plan, seeded)
        bare_sql = (
            f"SELECT state, geo, id, votes FROM {_TABLE} WHERE (state, geo, id) IN "
            "(SELECT * FROM unnest($1::text[], $2::text[], $3::bigint[]))"
        )
        typed = ([k[0] for k in keys], [k[1] for k in keys], [k[2] for k in keys])
        baseline = await best_seconds(lambda: seeded.conn.fetch(bare_sql, *typed))
        took = await best_seconds(
            lambda: read_keyed(seeded.conn, _TABLE, keys=batch, columns=["state", "geo", "id", "votes"])
        )
        problem = _budget(f"read_keyed of {len(keys)} keys", took, baseline)
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
        problem = _budget("keyset_page of 200 rows", took, baseline)
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
        problem = _budget("counts_by of 100 geographies", took, baseline)
        assert problem is None, problem

    async def test_bounded_counts_and_probes_read_no_further_than_their_bound(self, seeded: _Seeded) -> None:
        """Flips if a bounded count or a probe reads past its bound: every row its key names, or (a scan stopped at the
        LIMIT) every row before them. The hard case: the key's rows clustered at the end of the heap."""
        await seeded.conn.execute(f"CREATE TABLE clustered AS SELECT * FROM {_TABLE} ORDER BY state")
        await seeded.conn.execute("ALTER TABLE clustered ADD PRIMARY KEY (state, geo, id)")
        await seeded.conn.execute("ANALYZE clustered")
        for table, state in ((_TABLE, "S30"), ("clustered", "S50")):
            recorder = PlanRecorder(seeded.conn)
            bounded = await count_at_most(recorder, table, by={"state": state}, at_most=100, order=_KEY)
            assert (bounded.count, bounded.more) == (100, True)
            assert await holds_any(recorder, table, by={"state": state}, order=_KEY)
            assert not await holds_any(recorder, table, by={"state": "none"}, order=_KEY)
            assert recorder.plans[0].rows_scanned <= 102, (table, recorder.plans[0])
            assert all(plan.rows_scanned <= 2 for plan in recorder.plans[1:]), (table, recorder.plans[1:])


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
        problem = _budget("read_past_marks of 1,500 moved rows", took, baseline)
        assert problem is None, problem


class TestDeleteBefore:
    async def test_a_sweep_deletes_page_by_page_within_budget(self, seeded: _Seeded) -> None:
        """Flips if a sweep sends other than ceil(rows / page) + 1 statements, a page's plan reads its rows other than
        by the stamp index and the key (batched on YugabyteDB), or a page is over 2x the bare page delete.

        The bare is the best hand-written statement doing the same work on the same table: it finds the page (the 500
        oldest by stamp, ORDER BY ... LIMIT) and deletes it RETURNING the keys, as the helper must (it does not know
        the page's end stamp, and the keys settle every cache). Never a range delete to a stamp the test looked up
        first: that skips finding the page and returning its keys (0.89 ms against 2.05 ms)."""
        oldest = await seeded.conn.fetchval(f"SELECT stamp FROM {_TABLE} ORDER BY stamp LIMIT 1")
        before = oldest + timedelta(seconds=3000)
        planned = _WritePlans(seeded.conn)
        await _rolled_back(
            seeded.conn,
            lambda: delete_before(
                _TABLE, collection=None, conn=planned, key=_KEY, column="stamp", before=before, page=500, most_pages=1
            ),
        )
        plan = planned.plans[0]
        assert plan_uses_index(plan, "results_by_stamp") and plan_uses_index(plan, "results_pkey"), plan.plan
        _batched_on_yugabyte(plan, seeded)
        bare_page = (
            f"DELETE FROM {_TABLE} AS t WHERE (t.state, t.geo, t.id) IN (SELECT state, geo, id FROM {_TABLE} "
            "WHERE stamp < $1 ORDER BY stamp LIMIT 500) RETURNING t.state, t.geo, t.id"
        )
        baseline = await _rolled_back(seeded.conn, lambda: seeded.conn.fetch(bare_page, before))
        took = await _rolled_back(
            seeded.conn,
            lambda: delete_before(
                _TABLE,
                collection=None,
                conn=seeded.conn,
                key=_KEY,
                column="stamp",
                before=before,
                page=500,
                most_pages=1,
            ),
        )
        swept = await delete_before(
            _TABLE, collection=cast(Any, _Settling(seeded.conn)), key=_KEY, column="stamp", before=before, page=500
        )
        assert swept.deleted == 3000
        assert swept.statements == 3000 // 500 + 1
        assert not await holds_any(seeded.conn, _TABLE, by={"state": "S01", "geo": "001"}, order=_KEY)
        problem = _budget("a delete_before page of 500 rows", took, baseline)
        assert problem is None, problem


class TestUpdateBefore:
    async def test_a_reap_updates_page_by_page_within_budget(self, seeded: _Seeded) -> None:
        """Flips if a reap sends other than ceil(rows / page) + 1 statements, a page's plan reads its rows other than
        by the stamp index and the key (batched on YugabyteDB), or a page is over 2x one bare update of the same 500
        rows by the same predicate on the same table."""
        oldest = await seeded.conn.fetchval(f"SELECT stamp FROM {_TABLE} ORDER BY stamp LIMIT 1")
        before = oldest + timedelta(seconds=3000)
        moved = before + timedelta(days=365)
        page_end = await seeded.conn.fetchval(f"SELECT stamp FROM {_TABLE} ORDER BY stamp OFFSET 500 LIMIT 1")

        def page(conn: Any) -> Any:
            return update_before(
                _TABLE,
                collection=None,
                conn=conn,
                key=_KEY,
                column="stamp",
                before=before,
                set_to={"stamp": moved},
                page=500,
                most_pages=1,
            )

        planned = _WritePlans(seeded.conn)
        await _rolled_back(seeded.conn, lambda: page(planned))
        plan = planned.plans[0]
        assert plan_uses_index(plan, "results_by_stamp") and plan_uses_index(plan, "results_pkey"), plan.plan
        _batched_on_yugabyte(plan, seeded)
        baseline = await _rolled_back(
            seeded.conn,
            lambda: seeded.conn.execute(f"UPDATE {_TABLE} SET stamp = $2 WHERE stamp < $1", page_end, moved),
        )
        took = await _rolled_back(seeded.conn, lambda: page(seeded.conn))
        swept = await update_before(
            _TABLE,
            collection=cast(Any, _Settling(seeded.conn)),
            key=_KEY,
            column="stamp",
            before=before,
            set_to={"stamp": moved},
            page=500,
        )
        assert swept.deleted == 3000
        assert swept.statements == 3000 // 500 + 1
        assert await seeded.conn.fetchval(f"SELECT count(*) FROM {_TABLE} WHERE stamp < $1", before) == 0
        problem = _budget("an update_before page of 500 rows", took, baseline)
        assert problem is None, problem


class TestReadJoined:
    async def test_a_page_s_joined_rows_are_read_by_its_values_within_budget(self, seeded: _Seeded) -> None:
        """Flips if a joined read scans past the rows the page's values name, or is over 2x the bare read of them."""
        page = [{"state": "S11", "geo": f"{g:03d}"} for g in range(1, 201)]
        recorder = PlanRecorder(seeded.conn)
        joined = await read_joined(
            recorder, page, _TABLE, on={"state": "state", "geo": "geo"}, columns=["state", "geo", "id", "votes"]
        )
        assert sum(len(rows) for rows in joined.values()) == len(page) * _IDS
        _plans_ok(recorder, seeded)
        (plan,) = recorder.plans
        assert plan_uses_index(plan, "results_pkey"), plan.plan
        assert plan.rows_scanned <= 1.1 * len(page) * _IDS, plan
        _batched_on_yugabyte(plan, seeded)
        baseline = await _baseline(
            seeded.conn,
            f"SELECT state, geo, id, votes FROM {_TABLE} WHERE state = $1 AND geo = ANY($2::text[])",
            "S11",
            [row["geo"] for row in page],
        )
        took = await best_seconds(
            lambda: read_joined(
                seeded.conn, page, _TABLE, on={"state": "state", "geo": "geo"}, columns=["state", "geo", "id", "votes"]
            )
        )
        problem = _budget(f"read_joined of {len(page) * _IDS} rows", took, baseline)
        assert problem is None, problem


class TestCountsByRelations:
    async def test_two_relations_count_only_their_keys_rows_within_budget(self, seeded: _Seeded) -> None:
        """Flips if a count over several relations reads rows no key names, or is over 2x the bare read of them."""
        # a second relation as large as the first, its rows clustered by state (the key's rows together at one end of
        # the heap): a count reads its keys' rows by the key on each, however the rows lie
        await seeded.conn.execute(f"CREATE TABLE results_b AS SELECT * FROM {_TABLE} ORDER BY state")
        await seeded.conn.execute("ALTER TABLE results_b ADD PRIMARY KEY (state, geo, id)")
        await seeded.conn.execute("ANALYZE results_b")
        geos = [f"{g:03d}" for g in range(5, 200, 20)]
        tables = {"a": _TABLE, "b": "results_b"}
        recorder = PlanRecorder(seeded.conn)
        counted = await counts_by_relations(recorder, tables, column="geo", values=AnyOf(geos), by={"state": "S50"})
        assert counted == {"a": dict.fromkeys(geos, _IDS), "b": dict.fromkeys(geos, _IDS)}
        assert all(plan.rows_scanned <= 1.1 * 2 * len(geos) * _IDS for plan in recorder.plans), recorder.plans
        baseline = await _baseline(
            seeded.conn,
            f"SELECT state, geo FROM {_TABLE} WHERE state = $1 AND geo = ANY($2::text[]) "  # noqa: S608
            "UNION ALL SELECT state, geo FROM results_b WHERE state = $1 AND geo = ANY($2::text[])",
            "S50",
            geos,
        )
        took = await best_seconds(
            lambda: counts_by_relations(seeded.conn, tables, column="geo", values=AnyOf(geos), by={"state": "S50"})
        )
        problem = _budget(f"counts_by_relations of {len(geos)} geographies in 2 relations", took, baseline)
        assert problem is None, problem


class TestDeleteOldest:
    async def test_a_cache_trim_deletes_its_oldest_rows_within_budget(self) -> None:
        """Flips if a cache trim reads past the rows it deletes (a sort of the table) or is over 2x the bare trim.

        The bare is the best hand-written statement doing the same work on the same table through the same thread
        hop: the same 3,000 oldest rows by rowid, deleted through asyncio.to_thread, since IO in a thread is the rule on
        both sides. Never a range delete to a rowid the test already knows, run in the loop: that skips finding the
        oldest rows and the hop (0.17 ms against 0.54 ms). Each side is the best of several runs, each rolled back,
        so every run deletes from the same 300,000 rows."""
        local = sqlite3.connect(":memory:", check_same_thread=False, isolation_level=None)
        try:
            local.execute("CREATE TABLE cached (state TEXT, geo TEXT, id INTEGER, pad TEXT)")
            local.execute("BEGIN")
            local.executemany(
                "INSERT INTO cached VALUES (?, ?, ?, ?)",
                ((f"S{n % _STATES:02d}", f"{n % _GEOS:03d}", n, "x" * 40) for n in range(_STATES * _GEOS * _IDS)),
            )
            local.execute("COMMIT")
            plan = " ".join(
                str(row[-1])
                for row in local.execute(
                    "EXPLAIN QUERY PLAN DELETE FROM cached WHERE rowid IN "
                    "(SELECT rowid FROM cached ORDER BY rowid LIMIT 3000)"
                )
            )
            assert "USE TEMP B-TREE" not in plan, plan
            bare = "DELETE FROM cached WHERE rowid IN (SELECT rowid FROM cached ORDER BY rowid LIMIT 3000)"

            async def rolled_back(run: Any) -> float:
                async def once() -> None:
                    local.execute("BEGIN")
                    try:
                        await run()
                    finally:
                        local.execute("ROLLBACK")

                return await best_seconds(once)

            baseline = await rolled_back(lambda: asyncio.to_thread(local.execute, bare))
            took = await rolled_back(lambda: delete_oldest(ThreadedSQLite(local), "cached", count=3000))
            deleted = await delete_oldest(ThreadedSQLite(local), "cached", count=3000)
            assert deleted == 3000
            assert local.execute("SELECT min(rowid) FROM cached").fetchone()[0] == 3001
        finally:
            local.close()
        problem = _budget("delete_oldest of 3,000 rows of 300,000", took, baseline)
        assert problem is None, problem


_RATES = "credit_rates"
_MODELS = 300
_RATES_A_MODEL = 10


@pytest.fixture
async def rates(db_container: str) -> AsyncIterator[_Seeded]:
    """a rates table found by its model through a declared index (the gateway's credit rates), 3,000 rows."""
    schema = f"keyed_rates_{uuid.uuid4().hex[:8]}"
    conn = await asyncpg.connect(db_container)
    try:
        await conn.execute(f"CREATE SCHEMA {schema}")
        await conn.execute(f"SET search_path = {schema}")
        await conn.execute(
            f"CREATE TABLE {_RATES} (id UUID PRIMARY KEY, model_id UUID NOT NULL, kind TEXT NOT NULL, "
            "effective_from TIMESTAMPTZ NOT NULL)"
        )
        await conn.execute(f"CREATE INDEX credit_rates_by_model ON {_RATES} (model_id ASC, kind ASC)")
        await conn.execute(
            f"""INSERT INTO {_RATES}
            SELECT md5('r' || m || '-' || r)::uuid, md5('m' || m)::uuid, 'k' || r, now()
            FROM generate_series(1, {_MODELS}) m, generate_series(1, {_RATES_A_MODEL}) r"""
        )
        await conn.execute(f"ANALYZE {_RATES}")
        # the join the hub's statistics led Postgres to (a hash join of the keys, the table scanned): a keyed read must
        # be an index condition whatever join is picked, never only while the planner happens to nest a loop
        await conn.execute("SET enable_nestloop = off")
        yield _Seeded(conn=conn, yugabyte=await on_yugabyte(conn))
    finally:
        await conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        await conn.close()


class TestAKeyedReadIsAnIndexCondition:
    async def test_a_batch_of_one_key_column_reads_its_index_by_the_keys(self, rates: _Seeded) -> None:
        """Flips if the keys reach the index through a per-element cast in a join (an index scan with no index
        condition: every row of the index read for three keys), as the first read_keyed did."""
        models = [
            str(row["model_id"])
            for row in await rates.conn.fetch(f"SELECT DISTINCT model_id FROM {_RATES} ORDER BY model_id LIMIT 3")
        ]
        recorder = PlanRecorder(rates.conn)
        rows = await read_keyed(
            recorder, _RATES, keys=KeyBatch(("model_id",), [(m,) for m in models], ("uuid",)), columns=["id", "kind"]
        )
        assert len(rows) == len(models) * _RATES_A_MODEL
        (plan,) = recorder.plans
        assert plan_uses_index(plan, "credit_rates_by_model"), plan.plan
        assert plan.rows_scanned <= 1.1 * len(models) * _RATES_A_MODEL, plan
