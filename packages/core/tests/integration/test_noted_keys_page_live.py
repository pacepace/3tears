"""Integration: a commit's keys, too many to name in its epoch record, are read back from its change notes in pages.

A commit naming more rows than sixteen change records hold is recorded keys unknown, with the database clock's bracket
of its notes (``EpochChange.noted_between``). The broker answers a copy follower's ask for those keys with
:func:`noted_keys_page`: the table's notes of the asker's tenant in the bracket, a page at a time, continued by the last
``noted_at`` read. These seed a ring with a night of other tables' notes, the table's notes before the bracket, another
tenant's notes of the same table inside it, and the commit's own 20,000, then check that:

- the pages name exactly the commit's keys (the other tenant's never), whatever the page's bound;
- every page is an index range scan of the table's notes in the bracket (no sort, no scan of other tables' notes):
  rows scanned within 1.1x the rows returned, over every page together;
- reading every page takes within 2x the database's time to read the same notes from a plain table in one statement.

Against the session ``db_container`` (Postgres); set ``THREETEARS_TEST_POSTGRES_URL`` to a YugabyteDB URL to run them
there.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import asyncpg
import pytest

from threetears.core.collections.change_notes import NotedKeysCursor, NotedKeysPage, noted_keys_page
from threetears.core.ddl.change_notes import NOTE_RING_SIZE, RING_TABLE, note_table
from threetears.core.testing.calibrate import best_seconds, budget_problem, plain_copies
from threetears.core.testing.note_ring import install_note_ring
from threetears.core.testing.plans import PlanRecorder, plan_problems

pytestmark = pytest.mark.integration

#: other tables' notes through the night, and how many tables
_NOISE = 300_000
_TABLES = 1_000

#: the table, the tenant asking, and the commit's rows
_TABLE = "lsf_runs"
_TENANT = "agent_reports"
_COMMIT = 20_000

#: the most rows the pages scan for each row they answer
_SCAN_RATIO = 1.1

#: runs the read and its baseline are each timed over, the best kept
_RUNS = 5


@dataclass
class _Ring:
    conn: asyncpg.Connection
    schema: str
    bracket: tuple[datetime, datetime]


@pytest.fixture
async def ring(db_container: str) -> AsyncIterator[_Ring]:
    """a ring whose current table holds a night of notes and one large commit's, its bracket around them."""
    schema = f"notes_{uuid.uuid4().hex[:8]}"
    conn = await asyncpg.connect(db_container, command_timeout=1200)
    try:
        await conn.execute(f"CREATE SCHEMA {schema}")
        await install_note_ring(conn, schema)
        notes = f"{schema}.{note_table(0)}"
        night = datetime.now(UTC) - timedelta(hours=6)
        insert = f"INSERT INTO {notes} (table_name, keys, key_text, tenant, noted_at) "  # noqa: S608
        await conn.execute(
            insert + f"SELECT 'table_' || (g % {_TABLES}), ARRAY[g::text], g::text, $2, "
            "$1::timestamptz + make_interval(secs => g * 0.05) FROM generate_series(1, $3) g",
            night,
            _TENANT,
            _NOISE,
        )
        # the table's notes before the bracket: earlier commits, already applied
        await conn.execute(
            insert + "SELECT $1, ARRAY['old-' || g], 'old-' || g, $2, $3::timestamptz + make_interval(secs => g) "
            "FROM generate_series(1, 2000) g",
            _TABLE,
            _TENANT,
            night,
        )
        since = datetime.now(UTC) - timedelta(minutes=1)
        until = since + timedelta(seconds=2)
        # the commit: a note a row, each on the clock as its row was written; some at one instant together
        await conn.execute(
            insert + "SELECT $1, ARRAY[g::text, 'TX'], g::text || chr(31) || 'TX', $2, "
            "$3::timestamptz + make_interval(secs => (g / 7) * 0.0005) FROM generate_series(1, $4) g",
            _TABLE,
            _TENANT,
            since,
            _COMMIT,
        )
        # another schema's table of the same name, written in the same instant: never this tenant's keys
        await conn.execute(
            insert + "SELECT $1, ARRAY['theirs-' || g], 'theirs-' || g, 'agent_other', "
            "$2::timestamptz + make_interval(secs => g * 0.01) FROM generate_series(1, 100) g",
            _TABLE,
            since,
        )
        for slot in range(NOTE_RING_SIZE):
            await conn.execute(f"ANALYZE {schema}.{note_table(slot)}")
        yield _Ring(conn=conn, schema=schema, bracket=(since, until))
    finally:
        await conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        await conn.close()


async def _every_page(
    conn: Any, ring: _Ring, *, limit: int, max_bytes: int, every_tenant: bool = False
) -> tuple[set[tuple[str, ...]], int]:
    keys: set[tuple[str, ...]] = set()
    cursor: NotedKeysCursor | None = None
    pages = 0
    while True:
        page = await noted_keys_page(
            conn,
            notes_schema=ring.schema,
            table=_TABLE,
            tenant_token=_TENANT,
            noted_between=ring.bracket,
            cursor=cursor,
            limit=limit,
            max_bytes=max_bytes,
            every_tenant=every_tenant,
        )
        assert isinstance(page, NotedKeysPage), page
        assert not page.whole_table
        keys.update(page.keys)
        pages += 1
        if page.next is None:
            return keys, pages
        cursor = page.next


def _scans(plan: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    found = [plan]
    for child in plan.get("Plans", ()):
        found.extend(_scans(child))
    return found


class TestACommitsKeysAreReadBackFromItsNotes:
    @pytest.mark.parametrize(("limit", "max_bytes"), [(1000, 1 << 20), (5000, 16 << 10), (50_000, 4 << 20)])
    async def test_the_pages_name_exactly_the_commits_keys(self, ring: _Ring, limit: int, max_bytes: int) -> None:
        """Flips if a page boundary skips or splits an instant's notes, or another tenant's keys are named."""
        keys, pages = await _every_page(ring.conn, ring, limit=limit, max_bytes=max_bytes)
        assert keys == {(str(g), "TX") for g in range(1, _COMMIT + 1)}
        assert pages >= _COMMIT // limit

    async def test_every_tenant_names_the_other_tenants_keys_too(self, ring: _Ring) -> None:
        """Flips if an operator's read of every tenant's notes misses another tenant's, or a tenant's reader sees
        them."""
        mine, _ = await _every_page(ring.conn, ring, limit=5000, max_bytes=4 << 20)
        every, _ = await _every_page(ring.conn, ring, limit=5000, max_bytes=4 << 20, every_tenant=True)
        theirs = {(f"theirs-{g}",) for g in range(1, 101)}
        assert not mine & theirs
        assert every == mine | theirs

    async def test_the_index_order_a_page_is_read_in_is_never_left_on_its_connection(self, ring: _Ring) -> None:
        """Flips if the planner settings a page's transaction takes outlive it on the (pooled) connection."""
        await _every_page(ring.conn, ring, limit=1000, max_bytes=1 << 20)
        for setting in ("enable_bitmapscan", "enable_seqscan", "enable_sort"):
            assert await ring.conn.fetchval(f"SHOW {setting}") == "on", setting

    async def test_a_page_the_ring_no_longer_covers_leaves_no_planner_setting_on_its_connection(
        self, ring: _Ring
    ) -> None:
        """Flips if a page that answers before reading any note (the ring switched past the bracket) leaves the index
        order it told the planner on the (pooled) connection."""
        await ring.conn.execute(
            f"UPDATE {ring.schema}.{RING_TABLE} SET switch_no = 2, previous_switched_at = now() WHERE id = 1"  # noqa: S608
        )
        page = await noted_keys_page(
            ring.conn,
            notes_schema=ring.schema,
            table=_TABLE,
            tenant_token=_TENANT,
            noted_between=ring.bracket,
            cursor=None,
            limit=1000,
            max_bytes=1 << 20,
        )
        assert page == "not_covered"
        for setting in ("enable_bitmapscan", "enable_seqscan", "enable_sort"):
            assert await ring.conn.fetchval(f"SHOW {setting}") == "on", setting

    async def test_a_commit_noted_across_a_ring_switch_is_read_from_both_tables_within_each_pages_bound(
        self, ring: _Ring
    ) -> None:
        """Flips if a page over the previous and the current table holds more than its limit, or skips the notes
        at the switch."""
        since, _ = ring.bracket
        switched_at = since + timedelta(seconds=0.4)
        moved = f"table_name = '{_TABLE}' AND noted_at < '{switched_at.isoformat()}'"
        previous = f"{ring.schema}.{note_table(2)}"
        current = f"{ring.schema}.{note_table(0)}"
        await ring.conn.execute(f"INSERT INTO {previous} SELECT * FROM {current} WHERE {moved}")  # noqa: S608
        await ring.conn.execute(f"DELETE FROM {current} WHERE {moved}")  # noqa: S608
        limit = 1000
        keys: set[tuple[str, ...]] = set()
        cursor: NotedKeysCursor | None = None
        slots: set[int] = set()
        while True:
            page = await noted_keys_page(
                ring.conn,
                notes_schema=ring.schema,
                table=_TABLE,
                tenant_token=_TENANT,
                noted_between=ring.bracket,
                cursor=cursor,
                limit=limit,
                max_bytes=1 << 20,
            )
            assert isinstance(page, NotedKeysPage), page
            assert len(page.keys) <= limit, "one page's notes over both tables stay within its limit"
            assert len(page.columns) == len(page.keys)
            keys.update(page.keys)
            if page.next is None:
                break
            slots.add(page.next.slot)
            cursor = page.next
        assert keys == {(str(g), "TX") for g in range(1, _COMMIT + 1)}
        assert slots == {0, 2}, "the pages went on through the previous table, then the current one"

    async def test_every_page_is_a_range_scan_of_the_tables_notes_in_the_bracket(self, ring: _Ring) -> None:
        """Flips if a page sorts, scans other tables' notes, or offsets past the notes it read before."""
        checked = PlanRecorder(ring.conn)
        keys, _ = await _every_page(checked, ring, limit=1000, max_bytes=1 << 20)
        assert len(keys) == _COMMIT
        problems = plan_problems(checked.plans, scan_ratio=_SCAN_RATIO, together=True)
        assert not problems, problems
        note_reads = [plan for plan in checked.plans if "noted_at" in plan.sql and "ORDER BY" in plan.sql]
        assert note_reads
        for statement in note_reads:
            nodes = _scans(statement.plan)
            assert not [node for node in nodes if node["Node Type"] in {"Sort", "Incremental Sort"}], statement.sql
            scans = [node for node in nodes if "Scan" in node["Node Type"]]
            # each note table is read through its own (table_name, noted_at) index, the previous one too
            assert scans and all(node.get("Index Name") == f"{node.get('Relation Name')}_table" for node in scans), [
                (node["Node Type"], node.get("Index Name")) for node in scans
            ]
            # both ends of the bracket bound the range the index reads, not a filter over what it read
            for node in scans:
                bounds = str(node.get("Index Cond", ""))
                assert "noted_at >" in bounds and "noted_at <=" in bounds, (node["Node Type"], bounds)

    async def test_reading_every_page_keeps_within_twice_a_bare_read_of_the_same_notes(self, ring: _Ring) -> None:
        """Flips if the paged read costs over twice one plain read of the notes it must read."""
        since, until = ring.bracket
        sources = {
            "notes": (
                f"SELECT noted_at, keys FROM {ring.schema}.{note_table(0)} "  # noqa: S608
                "WHERE table_name = $1 AND tenant = $2 AND noted_at BETWEEN $3 AND $4",
                (_TABLE, _TENANT, since, until),
            )
        }
        async with plain_copies(ring.conn, sources) as plain:
            baseline = await best_seconds(
                lambda: ring.conn.fetch(f"SELECT noted_at, keys FROM {plain['notes']}"), runs=_RUNS
            )  # noqa: S608
        took = await best_seconds(lambda: _every_page(ring.conn, ring, limit=5000, max_bytes=4 << 20), runs=_RUNS)
        print(f"paged {took * 1000:.1f} ms, bare {baseline * 1000:.1f} ms, {took / baseline:.2f}x")
        problem = budget_problem("noted_keys_page, every page", took, baseline)
        assert problem is None, problem
