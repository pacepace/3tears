"""Integration: the change-note ring's two joined reads, now through the keyed helpers, answer what they answered.

``noted_rows_since`` read the ring row and its notes in one hand-written statement (the ring row ``LEFT JOIN
LATERAL`` the distinct notes of the current and previous tables); it is now ``keyed.read_row_beside`` over the ring
row and a step of one keyed leg a slot. ``noted_keys_page`` read the ring row ``CROSS JOIN`` a table-less row of the
planner settings as they were, turning them off in the same select list; it is now ``keyed.read_row(beside=)``.
Each new read is checked here against the statement it replaced, kept below as written (the 4cddfe029 tree), over
randomized rings: the slot current, the switch count, when the last two switches were, the fence and the wait, notes
of the table and of others in every slot at times around the reach, tenants and whole-table notes. Mutation-checked
on Postgres (2026-11-15): a slot's one-time condition inverted (``>`` for ``<``), the coverage condition left out of
a slot's leg, and the settings read after they are turned off each turn it red.

Against the session ``db_container`` (Postgres); set ``THREETEARS_TEST_POSTGRES_URL`` to a YugabyteDB URL to run them
there.
"""

from __future__ import annotations

import json
import random
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import asyncpg
import pytest

from threetears.core.collections.change_notes import NotedRows, noted_keys_page, noted_rows_since, whole_noted_change
from threetears.core.ddl.change_notes import NOTE_RING_SIZE, RING_TABLE, note_table
from threetears.core.sql_fragments import quote_identifier
from threetears.core.testing.note_ring import install_note_ring

pytestmark = pytest.mark.integration

#: the table asked about, and another whose notes share the slots
_ASKED = "results_asked"
_OTHER = "results_other"

#: randomized rings per run
_RINGS = 120


#: the ring row's columns both noted_rows_since's statement and the old one answer
_RING_ANSWERED = ("switch_no", "previous_switched_at")


def _old_noted_rows_sql(owner: str) -> str:
    """``_noted_rows_sql`` as it was (change_notes.py at 4cddfe029), the reference."""
    branches = " UNION ALL ".join(
        f"SELECT keys, columns, key_text, tenant FROM {owner}.{quote_identifier(note_table(index))} "  # noqa: S608
        f"WHERE (r.current_ring = {index} OR ((r.current_ring - 1 + {NOTE_RING_SIZE}) % {NOTE_RING_SIZE} = {index} "
        "AND r.reach < r.writes_previous_until)) "
        "AND table_name = $1 AND noted_at >= r.reach"
        for index in range(NOTE_RING_SIZE)
    )
    return (
        "SELECT r.switch_no, r.previous_switched_at, r.reach, n.noted, n.key_text, n.keys, n.columns, n.tenant "
        f"FROM (SELECT current_ring, switch_no, previous_switched_at, "
        "$2::timestamptz - make_interval(secs => wait_seconds / 2) AS reach, "
        "switched_at + make_interval(secs => GREATEST(fence_seconds, wait_seconds)) AS writes_previous_until "
        f"FROM {owner}.{quote_identifier(RING_TABLE)} WHERE id = 1) r "
        "LEFT JOIN LATERAL (SELECT TRUE AS noted, key_text, keys, columns, tenant "
        f"FROM (SELECT DISTINCT key_text, keys, columns, tenant FROM ({branches}) u) d LIMIT $3) n "
        "ON r.switch_no = 0 OR r.reach >= r.previous_switched_at"
    )


def _old_page_ring_sql(owner: str) -> str:
    """``_noted_keys_page``'s ring row read as it was (change_notes.py at 4cddfe029), the reference."""
    return (
        "SELECT was.bitmapscan, was.seqscan, was.sort, set_config('enable_bitmapscan', 'off', false), "  # noqa: S608
        "set_config('enable_seqscan', 'off', false), set_config('enable_sort', 'off', false), current_ring, switch_no, "
        "previous_switched_at, wait_seconds, "
        "switched_at + make_interval(secs => GREATEST(fence_seconds, wait_seconds)) AS writes_previous_until "
        "FROM (SELECT current_setting('enable_bitmapscan') AS bitmapscan, current_setting('enable_seqscan') AS seqscan, "
        "current_setting('enable_sort') AS sort OFFSET 0) was "
        f"CROSS JOIN {owner}.{quote_identifier(RING_TABLE)} WHERE id = 1"
    )


@dataclass
class _Ring:
    conn: asyncpg.Connection
    schema: str


@pytest.fixture
async def ring(db_container: str) -> AsyncIterator[_Ring]:
    schema = f"ringeq_{uuid.uuid4().hex[:8]}"
    conn = await asyncpg.connect(db_container, command_timeout=600)
    try:
        await conn.execute(f"CREATE SCHEMA {schema}")
        await install_note_ring(conn, schema)
        yield _Ring(conn=conn, schema=schema)
    finally:
        await conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        await conn.close()


async def _seed(ring: _Ring, rng: random.Random, now: datetime) -> datetime:
    """one randomized ring and its notes; the ask's ``since``."""
    fence = rng.choice((30.0, 60.0, 600.0))
    wait = fence + rng.choice((0.0, 15.5, 120.0))
    switch_no = rng.choice((0, 0, 1, 7))
    switched_at = now - timedelta(seconds=rng.uniform(0, 1800))
    previous = None if switch_no == 0 else switched_at - timedelta(seconds=rng.uniform(0, 3600))
    await ring.conn.execute(
        f"UPDATE {ring.schema}.{RING_TABLE} SET current_ring = $1, switch_no = $2, switched_at = $3, "  # noqa: S608
        "previous_switched_at = $4, fence_seconds = $5, wait_seconds = $6 WHERE id = 1",
        rng.randrange(NOTE_RING_SIZE),
        switch_no,
        switched_at,
        previous,
        fence,
        wait,
    )
    for index in range(NOTE_RING_SIZE):
        notes = f"{ring.schema}.{note_table(index)}"
        await ring.conn.execute(f"DELETE FROM {notes}")  # noqa: S608
        rows: list[tuple[str, list[str] | None, str | None, str | None, str | None, datetime]] = []
        for _ in range(rng.randrange(0, 25)):
            table = rng.choice((_ASKED, _ASKED, _OTHER))
            tenant = rng.choice((None, "a", "b"))
            at = now - timedelta(seconds=rng.uniform(0, 3600))
            if rng.random() < 0.1:
                reach = rng.choice((None, json.dumps({"reason": "reset"}), json.dumps({"prefixes": [{"s": "PA"}]})))
                rows.append((table, None, None, reach, tenant, at))
            else:
                key = f"k{rng.randrange(12)}"
                columns = json.dumps({"c": "v"}) if rng.random() < 0.2 else None
                rows.append((table, [key], key, columns, tenant, at))
        if rows:
            await ring.conn.executemany(
                f"INSERT INTO {notes} (table_name, keys, key_text, columns, tenant, noted_at) "  # noqa: S608
                "VALUES ($1, $2, $3, $4::jsonb, $5, $6)",
                rows,
            )
    return now - timedelta(seconds=rng.uniform(0, 1800))


def _notes(rows: list[Any], at: int) -> list[str]:
    """each note row's (key_text, keys, columns, tenant), from its place ``at`` on, as comparable text."""
    return sorted(json.dumps([row[at + 1], row[at + 2], row[at + 3], row[at + 4]]) for row in rows if row[at])


def _decided(found: list[Any], covered: bool, cap: int) -> str:
    """what noted_rows_since decides from a statement's rows: unavailable, not covered, no notes, too many, or the
    notes it names."""
    if not found:
        return "notes_unavailable"
    if not covered:
        return "not_covered"
    held = [row for row in found if row["noted"]]
    if not held:
        return "no_notes"
    return "too_many_rows" if len(held) > cap else "notes"


class _Kept:
    """the connection, keeping each answer it gives."""

    def __init__(self, conn: asyncpg.Connection) -> None:
        self.conn = conn
        self.answers: list[list[Any]] = []

    async def fetch(self, query: str, *args: Any) -> list[Any]:
        found = list(await self.conn.fetch(query, *args))
        self.answers.append(found)
        return found


async def test_noted_rows_since_answers_what_the_hand_written_statement_answered(ring: _Ring) -> None:
    """Flips if noted_rows_since (read_row_beside over the ring row and one keyed leg a slot) reads other notes than
    the old one statement, or decides otherwise (unavailable, not covered, no notes, too many), over randomized
    rings; each decision is reached. Past the cap it names every whole-table note the old statement holds uncapped."""
    rng = random.Random(20261115)
    owner = quote_identifier(ring.schema)
    reached: set[str] = set()
    for case in range(_RINGS):
        now = await ring.conn.fetchval("SELECT now()")
        since = await _seed(ring, rng, now)
        for cap in (1_000, rng.randrange(1, 4)):
            old = await ring.conn.fetch(_old_noted_rows_sql(owner), _ASKED, since, cap + 1)
            kept = _Kept(ring.conn)
            answer = await noted_rows_since(kept, notes_schema=ring.schema, table=_ASKED, since=since, cap=cap)
            new = kept.answers[0]
            assert bool(old) == bool(new), case
            if not old:
                continue
            ring_row = old[0]
            covered = int(ring_row["switch_no"]) == 0 or ring_row["reach"] >= ring_row["previous_switched_at"]
            # the ring's columns come back on one row alone (read_row_beside(first=)), the same values the old
            # statement repeated on each
            carrying = [row for row in new if row["ring_row"] is not False]
            assert len(carrying) == 1, (case, cap, len(carrying))
            assert all(row["switch_no"] is None for row in new if row["ring_row"] is False), case
            assert [carrying[0][name] for name in _RING_ANSWERED] == [ring_row[name] for name in _RING_ANSWERED], case
            reach = since - timedelta(seconds=float(carrying[0]["wait_seconds"]) / 2)
            assert reach == ring_row["reach"], (case, reach, ring_row["reach"])
            decided = _decided(old, covered, cap)
            reached.add(decided)
            too_many = isinstance(answer, NotedRows) and answer.too_many_rows
            said = "too_many_rows" if too_many else answer if isinstance(answer, str) else "notes"
            assert said == decided, (case, cap, answer, decided)
            if too_many:
                # past the cap the whole-table notes are read alone, after the statement (a look a slot, then those in
                # reach by their keys; RC2, 2026-11-16: the answer was "too_many_rows" and named none of them, so a
                # truncate in the window was never recorded): every one the old statement reads with no cap, each
                # tenant's folded into one change
                assert isinstance(answer, NotedRows) and not answer.rows
                assert len(kept.answers) >= 2, case
                every = await ring.conn.fetch(_old_noted_rows_sql(owner), _ASKED, since, 1_000_000)
                reaches: dict[str | None, list[Any]] = {}
                for row in every:
                    if row["noted"] and row["key_text"] is None:
                        reaches.setdefault(row["tenant"], []).append(row["columns"])
                expected = [whole_noted_change(_ASKED, tenant, held) for tenant, held in reaches.items()]
                assert sorted(map(repr, answer.whole)) == sorted(map(repr, expected)), (case, cap)
                assert answer.whole_table == bool(expected), case
                if expected:
                    reached.add("too_many_rows_with_whole")
            else:
                assert len(kept.answers) == 1, case
            if cap == 1_000:
                # the same notes, each once
                assert _notes([row[:6] + row[7:] for row in new], 5) == _notes(old, 3), (case, cap)
            else:
                # past the cap which notes come is the database's, as it was: the same count
                assert len([r for r in new if r["noted"]]) == len([r for r in old if r["noted"]]), (case, cap)
    assert reached >= {"not_covered", "no_notes", "too_many_rows", "too_many_rows_with_whole", "notes"}, reached


async def test_a_noted_keys_pages_ring_read_answers_what_the_hand_written_statement_answered(ring: _Ring) -> None:
    """Flips if the noted keys page's ring row (read_row beside the planner settings as they were) reads other ring
    values than the old CROSS JOIN, or reads the settings after it turned them off rather than before, or leaves
    them on its connection, over randomized rings: every page from the bracket's start reads the same keys."""
    rng = random.Random(20261116)
    owner = quote_identifier(ring.schema)
    for case in range(_RINGS // 4):
        now = await ring.conn.fetchval("SELECT now()")
        await _seed(ring, rng, now)
        # the settings as a pooled connection may hold them: one off already, the others on
        await ring.conn.execute("SET enable_sort = off")
        old = await ring.conn.fetchrow(_old_page_ring_sql(owner))
        await ring.conn.execute("RESET enable_bitmapscan; RESET enable_seqscan; SET enable_sort = off")
        assert old is not None and (old["bitmapscan"], old["seqscan"], old["sort"]) == ("on", "on", "off")
        bracket = (now - timedelta(seconds=rng.uniform(0, 3600)), now)
        pages = []
        cursor = None
        while True:
            page = await noted_keys_page(
                ring.conn,
                notes_schema=ring.schema,
                table=_ASKED,
                tenant_token=None,
                noted_between=bracket,
                cursor=cursor,
                limit=rng.randrange(1, 6),
                max_bytes=1_000_000,
                every_tenant=True,
            )
            pages.append(page)
            if isinstance(page, str) or page.next is None:
                break
            cursor = page.next
        # the settings it found are the ones it leaves: enable_sort still off, the others on
        held = await ring.conn.fetchrow(
            "SELECT current_setting('enable_bitmapscan') AS b, current_setting('enable_seqscan') AS s, "
            "current_setting('enable_sort') AS o"
        )
        assert tuple(held) == ("on", "on", "off"), (case, tuple(held))
        await ring.conn.execute("RESET enable_sort")
        # the ring the page read: the old statement's values, the same decision on coverage
        reach = bracket[0] - timedelta(seconds=float(old["wait_seconds"]) / 2)
        covered = int(old["switch_no"]) == 0 or reach >= old["previous_switched_at"]
        assert (pages[0] != "not_covered") == covered, (case, pages[0], dict(old))
        if covered:
            await ring.conn.execute("RESET ALL")
            named = sorted(key for page in pages if not isinstance(page, str) for keyed in page.keys for key in keyed)
            current = int(old["current_ring"])
            previous = (current - 1 + NOTE_RING_SIZE) % NOTE_RING_SIZE
            slots = {current} if reach >= old["writes_previous_until"] else {current, previous}
            want: list[str] = []
            for index in sorted(slots):
                rows = await ring.conn.fetch(
                    f"SELECT key_text FROM {owner}.{quote_identifier(note_table(index))} "  # noqa: S608
                    "WHERE table_name = $1 AND noted_at > $2 AND noted_at <= $3 AND key_text IS NOT NULL",
                    _ASKED,
                    bracket[0] - timedelta(microseconds=1),
                    bracket[1],
                )
                want.extend(row["key_text"] for row in rows)
            want.sort()
            assert named == want, (case, named, want)
