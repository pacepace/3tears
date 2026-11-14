"""Integration: each keyed-read helper reads about its own rows at real size, and keeps its calibrated budget.

A table shaped like the reports pod's results (300,000 rows: 50 states, 200 geographies a state, 30 rows each, keyed
``(state, geo, id)``, a stamp index for change records and expiry), seeded in a schema of its own. Each helper's
statements are planned for real (:class:`~threetears.core.testing.plans.PlanRecorder`) and judged by the shared
plan check (rows scanned within 1.1x the rows answered; a count's within its bound), then timed against the bare:
the best hand-written statement doing the same work on the same table, natively bound, best of N
(:mod:`threetears.core.testing.calibrate`), held to 2x. Each test's docstring names its bare and why.

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
    Derived,
    Increment,
    apply_stage,
    delete_keyed,
    delete_oldest,
    holds_any,
    keyset_cursor,
    keyset_page,
    keyset_rows,
    read_joined,
    read_keyed,
    read_past_marks,
    update_before,
    update_from,
    update_keyed,
)
from threetears.core.testing.calibrate import (
    DEFAULT_BUDGET_FACTOR,
    best_seconds,
    best_seconds_interleaved,
    budget_problem,
)
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


async def _bare(conn: asyncpg.Connection, sql: str, *args: Any) -> float:
    """the bare: the best of N runs of the best hand-written statement doing the helper's work, on the same real table,
    its parameters bound natively. Never a scan of a plain copy holding only the answer: that skips the lookups, the
    seek and the sort the helper must make, and times a different statement."""
    return await best_seconds(lambda: conn.fetch(sql, *args))


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


def _budget(what: str, took: float, baseline: float, *, factor: float = DEFAULT_BUDGET_FACTOR) -> str | None:
    """the budget's verdict, its numbers printed (``-s``) for the run's record."""
    print(f"BUDGET {what} | baseline {baseline * 1000:.2f} ms | took {took * 1000:.2f} ms | {took / baseline:.2f}x")
    return budget_problem(what, took, baseline, factor=factor)


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
        """Flips if a page reads the prefix whole (a sort of every row), PostgreSQL pages past the cursor other than
        by the key's index range in order (the row comparison; the expansion planned a bitmap scan and a sort, 2.3 ms
        against 0.28 ms for 1,001 rows), YugabyteDB by a row comparison, or a page is over 2x the bare page read.

        The bare is the best hand-written statement doing the same work on the same table: the same page past the
        same last row, seeking the key's index by the row-value comparison ``(geo, id) > ($2, $3)`` with natively
        typed binds, LIMIT 201. Never a scan of a plain copy holding only the page's rows: that skips the seek
        (it measured 0.39 ms on YugabyteDB, against the helper's 0.79 ms)."""
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
        # the page's own statement (a reader that is no asyncpg connection is asked its database once, beside it)
        (plan,) = [recorded for recorded in recorder.plans if " LIMIT " in recorded.sql]
        if seeded.yugabyte:
            # a hashed bucket is sought by the leading bound; a row comparison there reads it from its start
            assert ', t."id") >' not in plan.sql, plan.sql
        else:
            # PostgreSQL seeks the key's index by the row comparison: an index range in the page's order, never a
            # bitmap scan of every row past the lead and a sort of them
            # the pinned state is bound by equality beside the comparison of the sort's columns, never inside it:
            # one range of the key's index, sought by both
            assert 't."state" = $1 AND (t."geo", t."id") > ($2::text, $3::text::bigint)' in plan.sql, plan.sql
            assert plan_uses_index(plan, "results_pkey"), plan.plan
            nodes = set(_node_types(plan))
            assert not nodes & {"Sort", "Bitmap Heap Scan", "Bitmap Index Scan", "Incremental Sort"}, nodes
        last = first.rows[-1]
        baseline = await _bare(
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

    async def test_a_pinned_column_leads_the_page_s_range_of_its_index(self, seeded: _Seeded) -> None:
        """Flips if a page whose by= pins the column leading an index (kind, of a 50/50 split) reads by an index of
        the sort's columns alone and filters (PostgreSQL picked the unique id index: 43 rows read for 21), rather
        than one range of the composite index, reading at most a page and one row."""
        await seeded.conn.execute(
            "CREATE TABLE listed (kind TEXT NOT NULL, name TEXT NOT NULL, id BIGINT NOT NULL, PRIMARY KEY (id))"
        )
        await seeded.conn.execute("CREATE INDEX listed_by_kind_name ON listed (kind ASC, name ASC, id ASC)")
        await seeded.conn.execute(
            "INSERT INTO listed SELECT CASE WHEN i % 2 = 0 THEN 'a' ELSE 'b' END, 'n' || lpad((i % 997)::text, 4, '0'), "
            "i FROM generate_series(1, 20000) i"
        )
        await seeded.conn.execute("ANALYZE listed")
        order = [Sort("name"), Sort("id", "bigint")]
        first = await keyset_page(
            seeded.conn, "listed", columns=["name", "id"], by={"kind": "a"}, order=order, limit=20
        )
        recorder = PlanRecorder(seeded.conn)
        page = await keyset_page(
            recorder,
            "listed",
            columns=["name", "id"],
            by={"kind": "a"},
            order=order,
            limit=20,
            cursor=first.next_cursor,
        )
        expected = await seeded.conn.fetch(
            "SELECT name, id FROM listed WHERE kind = 'a' ORDER BY name, id OFFSET 20 LIMIT 20"
        )
        assert [(r["name"], r["id"]) for r in page.rows] == [(r["name"], r["id"]) for r in expected]
        (plan,) = [recorded for recorded in recorder.plans if " LIMIT " in recorded.sql]
        assert plan_uses_index(plan, "listed_by_kind_name"), plan.plan
        # PostgreSQL: one range, a page and one row. YugabyteDB seeks by the leading bound and filters the
        # expansion in storage (threetears.core.keyset): it may read the cursor's name group again, never more
        group = await seeded.conn.fetchval(
            "SELECT max(n) FROM (SELECT count(*) AS n FROM listed WHERE kind = 'a' GROUP BY name) AS g"
        )
        assert plan.rows_scanned <= 21 + (group if seeded.yugabyte else 0), (plan, group)

    async def test_a_near_constant_pin_beside_a_unique_index_of_the_next_column_reads_about_a_page(
        self, seeded: _Seeded
    ) -> None:
        """Flips if a cursor page pinning a near-constant column (kind: 4,032 of 4,040 rows one value) reads a unique
        index of the sort's next column from its start, filtering (the hub's datasource page: 2,041 rows read for
        21), rather than either index sought by the comparison: at most two pages and one row read.

        The hub's own table shape: varchar kind and name, a uuid key, a UNIQUE index of name beside the listing's
        (kind, name, id) range index, and the cursor deep in the listing."""
        await seeded.conn.execute(
            "CREATE TABLE sources (kind VARCHAR(32) NOT NULL, name VARCHAR(255) NOT NULL, id UUID NOT NULL, "
            "pad TEXT, PRIMARY KEY (id))"
        )
        await seeded.conn.execute("CREATE INDEX sources_by_kind_name ON sources (kind ASC, name ASC, id ASC)")
        await seeded.conn.execute("CREATE UNIQUE INDEX sources_by_name ON sources (name ASC)")
        await seeded.conn.execute(
            "INSERT INTO sources SELECT CASE WHEN i % 500 = 0 THEN 'postgres' ELSE 'redshift' END, "
            "'n' || lpad(i::text, 6, '0'), md5('s' || i)::uuid, repeat('x', 200) FROM generate_series(1, 4040) i"
        )
        await seeded.conn.execute("ANALYZE sources")
        order = [Sort("name"), Sort("id", "uuid")]
        limit = 20
        last = await seeded.conn.fetchrow(
            "SELECT name, id FROM sources WHERE kind = 'redshift' ORDER BY name, id OFFSET 2019 LIMIT 1"
        )
        assert last is not None
        cursor = keyset_cursor(order, {"name": last["name"], "id": last["id"]})
        recorder = PlanRecorder(seeded.conn)
        page = await keyset_page(
            recorder,
            "sources",
            columns=["name", "id"],
            by={"kind": "redshift"},
            order=order,
            limit=limit,
            cursor=cursor,
        )
        expected = await seeded.conn.fetch(
            "SELECT name, id FROM sources WHERE kind = 'redshift' ORDER BY name, id OFFSET 2020 LIMIT 20"
        )
        assert [(r["name"], r["id"]) for r in page.rows] == [(r["name"], r["id"]) for r in expected]
        (plan,) = [recorded for recorded in recorder.plans if " LIMIT " in recorded.sql]
        assert plan.rows_scanned <= 2 * limit + 1, plan

    async def test_a_pin_outside_the_index_does_not_make_the_page_sort(self, seeded: _Seeded) -> None:
        """Flips if a page pinning a column its index does not hold (a soft-delete flag beside the key's index,
        the hub's dataset_definitions latest version) names that pin in its ORDER BY: PostgreSQL folds a boolean
        pin to ``NOT is_deleted``, which is no equality it can drop from the order, so it read the dataset's rows
        and sorted them rather than walking the key's index backward. A pin is a constant on every row the page
        reads: it orders nothing."""
        await seeded.conn.execute(
            "CREATE TABLE definitions (dataset_id TEXT NOT NULL, version INTEGER NOT NULL, "
            "is_deleted BOOLEAN NOT NULL, body TEXT, PRIMARY KEY (dataset_id, version))"
        )
        await seeded.conn.execute(
            "INSERT INTO definitions SELECT 'd' || (i % 500), i / 500, i % 7 = 0, repeat('x', 50) "
            "FROM generate_series(1, 50000) i"
        )
        await seeded.conn.execute("ANALYZE definitions")
        recorder = PlanRecorder(seeded.conn)
        page = await keyset_page(
            recorder,
            "definitions",
            columns=["dataset_id", "version"],
            by={"dataset_id": "d7", "is_deleted": False},
            order=[Sort("version", "integer", descending=True)],
            limit=1,
        )
        expected = await seeded.conn.fetchval(
            "SELECT max(version) FROM definitions WHERE dataset_id = 'd7' AND NOT is_deleted"
        )
        assert [row["version"] for row in page.rows] == [expected]
        (plan,) = [recorded for recorded in recorder.plans if " LIMIT " in recorded.sql]
        assert plan_uses_index(plan, "definitions_pkey"), plan.plan
        nodes = set(_node_types(plan))
        assert not nodes & {"Sort", "Incremental Sort", "Bitmap Heap Scan"}, nodes
        assert '"is_deleted" ASC' not in plan.sql, plan.sql


class TestCounts:
    async def test_counts_by_counts_only_its_keys_rows(self, seeded: _Seeded) -> None:
        """Flips if a keyed count reads rows no key names, or is over 2x the bare count of them.

        The bare is the best hand-written statement doing the same work on the same table: the same GROUP BY count of
        the state's 100 geographies by the key (``geo = ANY`` a natively typed array). Never a read of a plain copy
        holding only the counted rows: that returns rows where the count returns 100, and skips the key's seek."""
        geos = [f"{g:03d}" for g in range(1, 101)]
        recorder = PlanRecorder(seeded.conn)
        counted = await counts_by(recorder, _TABLE, column="geo", values=AnyOf(geos), by={"state": "S20"})
        assert counted == dict.fromkeys(geos, _IDS)
        assert all(plan.rows_scanned <= 1.1 * len(geos) * _IDS for plan in recorder.plans), recorder.plans
        baseline = await _bare(
            seeded.conn,
            f"SELECT geo, count(*) FROM {_TABLE} WHERE state = $1 AND geo = ANY($2::text[]) GROUP BY geo",
            "S20",
            geos,
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
        """Flips if a change-record read scans past the rows that moved, or is over 2x the bare read of them.

        The bare is the best hand-written statement doing the same work on the same table: the rows past the mark,
        found by the stamp index (``stamp > $1``). Never a scan of a plain copy holding only the moved rows: that skips
        the index range (it measured 2.93 ms on YugabyteDB, against the helper's 6.91 ms, 2.36x)."""
        newest = await seeded.conn.fetchval(f"SELECT stamp FROM {_TABLE} ORDER BY stamp DESC LIMIT 1")
        mark = newest - timedelta(seconds=1500)
        recorder = PlanRecorder(seeded.conn)
        read = await read_past_marks(recorder, _TABLE, columns=["state", "geo", "id", "stamp"], marks={"stamp": mark})
        assert len(read.rows) == 1500
        assert read.marks == {"stamp": newest}
        _plans_ok(recorder, seeded)
        baseline = await _bare(seeded.conn, f"SELECT state, geo, id, stamp FROM {_TABLE} WHERE stamp > $1", mark)
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


class _Executes:
    """a connection whose every write is recorded, then run."""

    def __init__(self, conn: asyncpg.Connection) -> None:
        self._conn = conn
        self.asked: list[tuple[str, tuple[Any, ...]]] = []

    async def execute(self, sql: str, *args: Any) -> str:
        self.asked.append((sql, args))
        return str(await self._conn.execute(sql, *args))

    async def fetch(self, sql: str, *args: Any) -> Any:
        self.asked.append((sql, args))
        return await self._conn.fetch(sql, *args)


class TestKeyedWrites:
    async def test_an_update_and_a_delete_by_key_write_their_rows_within_budget(self, seeded: _Seeded) -> None:
        """Flips if an update or a delete by a (state, geo) prefix plans other than through the key's index, writes
        other than its 30 rows, or is over 2x the bare statement.

        The bare is the same write hand-written on the same table: ``UPDATE results SET votes = $3 WHERE state = $1
        AND geo = $2`` (and the ``DELETE``), natively bound; the helper writes exactly that statement, quoted."""
        by = {"state": "S11", "geo": "017"}
        recorder = _Executes(seeded.conn)
        transaction = seeded.conn.transaction()
        await transaction.start()
        try:
            updated = await update_keyed(recorder, _TABLE, values={"votes": -1}, by=by)
            deleted = await delete_keyed(recorder, _TABLE, by=by, returning=["id"])
        finally:
            await transaction.rollback()
        assert (updated.count, deleted.count, len(deleted.rows)) == (_IDS, _IDS, _IDS)
        for sql, args in recorder.asked:
            plan = await explain_write(seeded.conn, sql, *args)
            assert plan_uses_index(plan, "results_pkey"), plan.plan
            assert plan.rows_scanned <= 1.1 * _IDS, plan
        bare_update = await _rolled_back(
            seeded.conn,
            lambda: seeded.conn.execute(
                f"UPDATE {_TABLE} SET votes = $3 WHERE state = $1 AND geo = $2", "S11", "017", -1
            ),
        )
        took_update = await _rolled_back(
            seeded.conn, lambda: update_keyed(seeded.conn, _TABLE, values={"votes": -1}, by=by)
        )
        bare_delete = await _rolled_back(
            seeded.conn,
            lambda: seeded.conn.fetch(f"DELETE FROM {_TABLE} WHERE state = $1 AND geo = $2 RETURNING id", "S11", "017"),
        )
        took_delete = await _rolled_back(
            seeded.conn, lambda: delete_keyed(seeded.conn, _TABLE, by=by, returning=["id"])
        )
        problems = [
            _budget("update_keyed of a key's 30 rows", took_update, bare_update),
            _budget("delete_keyed of a key's 30 rows", took_delete, bare_delete),
        ]
        assert problems == [None, None], problems

    async def test_a_drained_listing_reads_page_by_page_within_budget(self, seeded: _Seeded) -> None:
        """Flips if keyset_rows loses or doubles a row of a state's 6,000, a page's plan reads past its rows, or the
        drain is over 2x the bare.

        The bare is the same drain hand-written: a keyset loop of 1,000-row pages, the same key predicate and ORDER BY
        (``SELECT ... WHERE state = $1 [AND (geo, id) > ($2, $3)] ORDER BY geo, id LIMIT 1000``) until a short page,
        natively bound; the two run turn about, best of N each, so a busy machine's load lands on both."""
        order = [Sort("geo"), Sort("id", cast="bigint")]
        columns = ["state", "geo", "id", "votes"]
        recorder = PlanRecorder(seeded.conn)
        rows = await keyset_rows(recorder, _TABLE, columns=columns, by={"state": "S23"}, order=order, page=1000)
        bare_sql = f"SELECT state, geo, id, votes FROM {_TABLE} WHERE state = $1 ORDER BY geo, id"
        assert [(r["geo"], r["id"]) for r in rows] == [
            (r["geo"], r["id"]) for r in await seeded.conn.fetch(bare_sql, "S23")
        ]
        assert len(rows) == _GEOS * _IDS
        _plans_ok(recorder, seeded)
        first_sql = f"{bare_sql} LIMIT 1000"
        next_sql = (
            f"SELECT state, geo, id, votes FROM {_TABLE} WHERE state = $1 AND (geo, id) > ($2, $3) "
            "ORDER BY geo, id LIMIT 1000"
        )

        async def bare_drain() -> list[Any]:
            drained = list(await seeded.conn.fetch(first_sql, "S23"))
            page = drained
            while len(page) == 1000:
                page = list(await seeded.conn.fetch(next_sql, "S23", page[-1]["geo"], page[-1]["id"]))
                drained += page
            return drained

        assert len(await bare_drain()) == len(rows)
        baseline, took = await best_seconds_interleaved(
            bare_drain,
            lambda: keyset_rows(seeded.conn, _TABLE, columns=columns, by={"state": "S23"}, order=order, page=1000),
        )
        problem = _budget("keyset_rows of a state's 6,000 rows, 1,000 a page", took, baseline)
        assert problem is None, problem


class TestKeyedWritesByBatchAndStage:
    async def test_a_key_batch_s_update_delete_and_source_update_within_budget(self, seeded: _Seeded) -> None:
        """Flips if a write by 600 keys, an update from a keyed source, or a stage applied by its key plans other
        than through the key's index, writes other than its rows, or is over 2x its bare.

        Each bare is the same write hand-written on the same table, natively bound: ``UPDATE ... WHERE (state,
        geo, id) IN (SELECT * FROM unnest($1::text[], $2::text[], $3::bigint[]))`` (and the ``DELETE``); ``UPDATE
        ... FROM (unnest ...) AS s WHERE <key> = s.<key>``; the stage's ``DELETE ... WHERE EXISTS`` plus
        ``INSERT ... SELECT``."""
        keys = list(dict.fromkeys(_keys(600)))
        batch = KeyBatch(("state", "geo", "id"), keys, ("text", "text", "bigint"))
        typed = ([k[0] for k in keys], [k[1] for k in keys], [k[2] for k in keys])
        named = "(state, geo, id) IN (SELECT * FROM unnest($1::text[], $2::text[], $3::bigint[]))"
        recorder = _Executes(seeded.conn)
        transaction = seeded.conn.transaction()
        await transaction.start()
        try:
            updated = await update_keyed(recorder, _TABLE, values={"votes": Increment(1)}, keys=batch)
            deleted = await delete_keyed(recorder, _TABLE, keys=batch)
        finally:
            await transaction.rollback()
        assert (updated.count, deleted.count) == (len(keys), len(keys))
        for sql, args in recorder.asked:
            plan = await explain_write(seeded.conn, sql, *args)
            assert plan_uses_index(plan, "results_pkey"), plan.plan
            assert plan.rows_scanned <= 1.1 * len(keys), plan
        source = Derived(
            f"SELECT state, geo, id, -votes AS votes FROM {_TABLE} WHERE state = 'S05' AND geo IN ('001', '002', '003')"
        )
        await seeded.conn.execute(
            f"CREATE TEMP TABLE stage AS SELECT state, geo, id, votes + 1 AS votes, stamp, pad FROM {_TABLE} "
            "WHERE state = 'S06' AND geo IN ('001', '002', '003')"
        )
        cols = ["state", "geo", "id", "votes", "stamp", "pad"]
        timings = {
            "update_keyed of 600 keys": (
                lambda: seeded.conn.execute(f"UPDATE {_TABLE} SET votes = votes + 1 WHERE {named}", *typed),
                lambda: update_keyed(seeded.conn, _TABLE, values={"votes": Increment(1)}, keys=batch),
            ),
            "delete_keyed of 600 keys": (
                lambda: seeded.conn.execute(f"DELETE FROM {_TABLE} WHERE {named}", *typed),
                lambda: delete_keyed(seeded.conn, _TABLE, keys=batch),
            ),
            "update_from a keyed source of 90 rows": (
                lambda: seeded.conn.execute(
                    f"UPDATE {_TABLE} SET votes = s.votes FROM ({source.sql}) AS s WHERE {_TABLE}.state = s.state "
                    f"AND {_TABLE}.geo = s.geo AND {_TABLE}.id = s.id"
                ),
                lambda: update_from(seeded.conn, _TABLE, key=["state", "geo", "id"], source=source, columns=["votes"]),
            ),
            "apply_stage replace of 90 rows": (
                lambda: _replace_bare(seeded.conn),
                lambda: apply_stage(seeded.conn, _TABLE, "stage", key=["state", "geo", "id"], columns=cols),
            ),
        }
        problems = []
        for what, (bare, helper) in timings.items():
            baseline = await _rolled_back(seeded.conn, bare)
            took = await _rolled_back(seeded.conn, helper)
            problems.append(_budget(what, took, baseline))
        assert problems == [None] * len(timings), problems

    async def test_every_value_of_a_prefix_is_counted_a_page_at_a_time_within_budget(self, seeded: _Seeded) -> None:
        """Flips if counts_by with no values counts other than every geo of a state, reads past its pages' rows,
        or is over 2x the same paged count hand-written: ``GROUP BY geo ORDER BY geo LIMIT 100`` pages past the last
        geo, then the NULL group's count, through the same index; the two run turn about, best of N each."""
        recorder = PlanRecorder(seeded.conn)
        counted = await counts_by(recorder, _TABLE, column="geo", by={"state": "S31"}, batch=100)
        assert counted == {f"{g:03d}": _IDS for g in range(1, _GEOS + 1)}
        # a count reads its groups' rows (30 a geo), so no rows-scanned ratio: no looped sequential scan
        _plans_ok(recorder, seeded, scan_ratio=None)
        first = (
            f"SELECT geo, count(*) AS held FROM {_TABLE} WHERE state = $1 AND geo IS NOT NULL "
            "GROUP BY geo ORDER BY geo LIMIT 100"
        )
        after = (
            f"SELECT geo, count(*) AS held FROM {_TABLE} WHERE state = $1 AND geo IS NOT NULL AND geo > $2 "
            "GROUP BY geo ORDER BY geo LIMIT 100"
        )
        nulls = f"SELECT count(*) AS held FROM {_TABLE} WHERE state = $1 AND geo IS NULL"

        async def bare_paged() -> dict[Any, int]:
            page = list(await seeded.conn.fetch(first, "S31"))
            groups = {row["geo"]: int(row["held"]) for row in page}
            while len(page) == 100:
                page = list(await seeded.conn.fetch(after, "S31", page[-1]["geo"]))
                groups.update({row["geo"]: int(row["held"]) for row in page})
            await seeded.conn.fetch(nulls, "S31")
            return groups

        assert await bare_paged() == counted
        baseline, took = await best_seconds_interleaved(
            bare_paged, lambda: counts_by(seeded.conn, _TABLE, column="geo", by={"state": "S31"}, batch=100)
        )
        problem = _budget("counts_by every geo of a state, 100 a page", took, baseline)
        assert problem is None, problem


async def _replace_bare(conn: asyncpg.Connection) -> None:
    await conn.execute(
        f"DELETE FROM {_TABLE} WHERE EXISTS (SELECT 1 FROM stage s WHERE s.state = {_TABLE}.state "
        f"AND s.geo = {_TABLE}.geo AND s.id = {_TABLE}.id)"
    )
    await conn.execute(f"INSERT INTO {_TABLE} SELECT * FROM stage")


class TestReadJoined:
    async def test_a_page_s_joined_rows_are_read_by_its_values_within_budget(self, seeded: _Seeded) -> None:
        """Flips if a joined read scans past the rows the page's values name, or is over 2x the bare read of them.

        The bare is the best hand-written statement doing the same work on the same table: the rows of the page's 200
        (state, geo) values by the key, IN-unnest of natively typed arrays with each column bound to its distinct
        values (the keys alone plan on PostgreSQL as a hash join over the whole table: 18 ms against 6 ms). Never a
        read of a plain copy holding only the answer: that skips the 200 lookups."""
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
        baseline = await _bare(
            seeded.conn,
            f"SELECT state, geo, id, votes FROM {_TABLE} WHERE (state, geo) IN "
            "(SELECT * FROM unnest($1::text[], $2::text[])) AND state = ANY($3::text[]) AND geo = ANY($4::text[])",
            [row["state"] for row in page],
            [row["geo"] for row in page],
            list(dict.fromkeys(row["state"] for row in page)),
            list(dict.fromkeys(row["geo"] for row in page)),
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
        """Flips if a count over several relations reads rows no key names, or is over 2x the bare count of them.

        The bare is the best hand-written statement doing the same work on the same tables: one UNION ALL of each
        relation's GROUP BY count by the key. Never a read of a plain copy holding only the counted rows."""
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
        baseline = await _bare(
            seeded.conn,
            f"SELECT 0 AS relation, geo, count(*) FROM {_TABLE} "  # noqa: S608
            "WHERE state = $1 AND geo = ANY($2::text[]) GROUP BY geo "
            "UNION ALL SELECT 1, geo, count(*) FROM results_b WHERE state = $1 AND geo = ANY($2::text[]) GROUP BY geo",
            "S50",
            geos,
        )
        took = await best_seconds(
            lambda: counts_by_relations(seeded.conn, tables, column="geo", values=AnyOf(geos), by={"state": "S50"})
        )
        problem = _budget(f"counts_by_relations of {len(geos)} geographies in 2 relations", took, baseline)
        assert problem is None, problem

    async def test_a_relations_own_pins_count_the_same_rows_in_the_one_statement(self, seeded: _Seeded) -> None:
        """Flips if ``pins=`` changes what a relation counts when every row holds a pinned value, narrows another
        relation, or a narrower pin miscounts, on PostgreSQL (the per-key seek) and YugabyteDB (``= ANY``) alike."""
        await seeded.conn.execute(f"CREATE TABLE results_p AS SELECT *, id % 3 AS slot FROM {_TABLE}")
        await seeded.conn.execute("ALTER TABLE results_p ADD PRIMARY KEY (state, geo, id)")
        geos = [f"{g:03d}" for g in range(5, 200, 20)]
        tables = {"a": _TABLE, "p": "results_p"}
        try:
            whole = await counts_by_relations(
                seeded.conn, tables, column="geo", values=AnyOf(geos, "text"), by={"state": "S50"}
            )
            pinned = await counts_by_relations(
                seeded.conn,
                tables,
                column="geo",
                values=AnyOf(geos, "text"),
                by={"state": "S50"},
                pins={"p": {"slot": AnyOf([0, 1, 2], "int")}},
            )
            narrowed = await counts_by_relations(
                seeded.conn,
                tables,
                column="geo",
                values=AnyOf(geos, "text"),
                by={"state": "S50"},
                pins={"p": {"slot": 0}},
            )
        finally:
            await seeded.conn.execute("DROP TABLE results_p")
        assert pinned == whole == {"a": dict.fromkeys(geos, _IDS), "p": dict.fromkeys(geos, _IDS)}
        assert narrowed == {"a": dict.fromkeys(geos, _IDS), "p": dict.fromkeys(geos, _IDS // 3)}


class TestDeleteOldest:
    async def test_a_cache_trim_deletes_its_oldest_rows_within_budget(self) -> None:
        """Flips if a cache trim reads past the rows it deletes (a sort of the table) or is over 2x the bare trim.

        The bare is the best hand-written statement doing the same work on the same table through the same thread
        hop: the 3,000 oldest rows deleted as one rowid range up to the 3,000th rowid (found by one seek, OFFSET 2999),
        through asyncio.to_thread, since IO in a thread is the rule on both sides. Never a range delete to a rowid the
        test already knows, run in the loop: that skips finding the 3,000th row and the hop (0.17 ms against 0.54 ms
        for the helper's first form, a list of rowids probed). Each side is the best of several runs, each rolled
        back, so every run deletes from the same 300,000 rows."""
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
                    "EXPLAIN QUERY PLAN DELETE FROM cached WHERE rowid <= COALESCE((SELECT rowid FROM cached "
                    "ORDER BY rowid LIMIT 1 OFFSET 2999), (SELECT max(rowid) FROM cached))"
                )
            )
            assert "USE TEMP B-TREE" not in plan, plan
            bare = (
                "DELETE FROM cached WHERE rowid <= COALESCE((SELECT rowid FROM cached ORDER BY rowid LIMIT 1 "
                "OFFSET 2999), (SELECT max(rowid) FROM cached))"
            )

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


_GRANTS = "grants"
_GRANT_ROWS = 500_000
_GRANT_KEYS = 2_000
#: runs each side of a 1.1x budget is timed over, turn about: a tenth is inside one run's noise on a busy machine
_RECORD_RUNS = 40


@pytest.fixture
async def grants(db_container: str) -> AsyncIterator[_Seeded]:
    """a grants table keyed by a uuid (the hub's grant reads), 500,000 rows laid in key order as a time-ordered uuid
    (uuid7) table's are, with fresh statistics. Laid in random order, PostgreSQL reads 2,000 keys of it by a scan,
    the hand-written read and the helper's alike."""
    schema = f"keyed_grants_{uuid.uuid4().hex[:8]}"
    conn = await asyncpg.connect(db_container)
    try:
        await conn.execute(f"CREATE SCHEMA {schema}")
        await conn.execute(f"SET search_path = {schema}")
        await conn.execute(
            f"CREATE TABLE {_GRANTS} (id UUID PRIMARY KEY, principal_id UUID NOT NULL, grant_kind TEXT NOT NULL, "
            "scope TEXT, date_created TIMESTAMPTZ NOT NULL)"
        )
        await conn.execute(
            f"""INSERT INTO {_GRANTS}
            SELECT md5('g' || i)::uuid, md5('p' || i)::uuid, 'k' || (i % 7), 'scope/' || i, $1::timestamptz
            FROM generate_series(1, {_GRANT_ROWS}) i ORDER BY 1""",
            _START,
        )
        await conn.execute(f"ANALYZE {_GRANTS}")
        yield _Seeded(conn=conn, yugabyte=await on_yugabyte(conn))
    finally:
        await conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        await conn.close()


class TestReadKeyedRecords:
    async def test_uuid_keys_read_as_records_within_a_tenth_of_the_hand_written_any(self, grants: _Seeded) -> None:
        """Flips if read_keyed(records=True) of 2,000 uuid keys on a direct connection is over 1.1x the bare, or
        answers other rows.

        The bare is the hand-written statement doing the same work on the same table: ``id = ANY($1::uuid[])`` for
        the same 2,000 keys, natively bound, answering the same asyncpg Records. The helper's statement is that
        statement (an Index Cond on the key), so what is left is its own Python: the batch, the statement's text
        (rendered once a shape) and no copy of a row. Timed turn about, best of 40."""
        ids = [
            row["id"]
            for row in await grants.conn.fetch(f"SELECT id FROM {_GRANTS} ORDER BY md5(id::text) LIMIT {_GRANT_KEYS}")
        ]
        columns = ["id", "principal_id", "grant_kind", "scope", "date_created"]
        keys = KeyBatch(("id",), [(key,) for key in ids], ("uuid",))
        hand = f"SELECT {', '.join(columns)} FROM {_GRANTS} WHERE id = ANY($1::uuid[])"
        recorder = PlanRecorder(grants.conn)
        await read_keyed(recorder, _GRANTS, keys=keys, columns=columns, records=True)
        (plan,) = recorder.plans
        assert plan_uses_index(plan, f"{_GRANTS}_pkey"), plan.plan
        rows = await read_keyed(grants.conn, _GRANTS, keys=keys, columns=columns, records=True)
        assert isinstance(rows[0], asyncpg.Record)
        assert sorted(tuple(row) for row in rows) == sorted(tuple(row) for row in await grants.conn.fetch(hand, ids))
        baseline, took = await best_seconds_interleaved(
            lambda: grants.conn.fetch(hand, ids),
            lambda: read_keyed(grants.conn, _GRANTS, keys=keys, columns=columns, records=True),
            runs=_RECORD_RUNS,
        )
        problem = _budget(f"read_keyed(records=True) of {len(ids)} uuid keys", took, baseline, factor=1.1)
        assert problem is None, problem
