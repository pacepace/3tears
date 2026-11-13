"""Integration: a keyset read of a table whose key's lead YugabyteDB hashes reads each page from its lead's bucket.

YugabyteDB hashes a key's first column. A page past a cursor ordered by the whole key cannot seek a hashed column,
so before the walk each page read every row of the table and sorted it (2026.1: a 200,000-row table, every page
scanned 200,000 rows). A read whose declaration leaves its lead hashed walks the lead's values on YugabyteDB
(:mod:`threetears.core.keyset`, "A hashed lead"): each page then scans about the rows it answers, judged with the
shared plan check (:mod:`threetears.core.testing.plans`). Postgres pages by the cursor, so there the reads are checked
for their rows only.

Against the session ``db_container`` (Postgres); set ``THREETEARS_TEST_POSTGRES_URL`` to a YugabyteDB URL to run them
there, where the problem was measured.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import asyncpg
import pytest

from threetears.core.backends.sql import SqlL3Backend
from threetears.core.collections.schema_backed import BIGINT_TYPE, STRING_TYPE, Column, TableSchema
from threetears.core.keyset import keyset_pages
from threetears.core.testing.plans import PlanRecorder, StatementPlan, on_yugabyte, plan_problems

pytestmark = pytest.mark.integration

#: a few large lead values: each value's rows span several pages
_WIDE = TableSchema(
    name="wide",
    primary_key=("state", "geo_id"),
    columns=[Column("state", STRING_TYPE), Column("geo_id", BIGINT_TYPE), Column("votes", BIGINT_TYPE)],
)
_WIDE_STATES = 50
_WIDE_PER_STATE = 3_000

#: many small lead values: a page holds many of them
_NARROW = TableSchema(
    name="narrow",
    primary_key=("race", "candidate"),
    columns=[Column("race", STRING_TYPE), Column("candidate", BIGINT_TYPE), Column("votes", BIGINT_TYPE)],
)
_NARROW_RACES = 3_000
_NARROW_PER_RACE = 2

#: the rows a page keeps (the rail's)
_PAGE = 999

#: the most rows a read scans for each row it answers
_SCAN_RATIO = 1.1


@dataclass
class _Seeded:
    conn: asyncpg.Connection
    yugabyte: bool


@pytest.fixture
async def seeded(db_container: str) -> AsyncIterator[_Seeded]:
    """both tables, in a schema of their own, with fresh statistics."""
    schema = f"hashed_lead_{uuid.uuid4().hex[:8]}"
    conn = await asyncpg.connect(db_container, command_timeout=600)
    try:
        await conn.execute(f"CREATE SCHEMA {schema}")
        await conn.execute(f"SET search_path = {schema}")
        await conn.execute("CREATE TABLE wide (state TEXT, geo_id BIGINT, votes BIGINT, PRIMARY KEY (state, geo_id))")
        await conn.execute(
            "INSERT INTO wide SELECT 'S' || lpad((g % $1)::text, 2, '0'), g, 1 FROM generate_series(1, $2) g",
            _WIDE_STATES,
            _WIDE_STATES * _WIDE_PER_STATE,
        )
        await conn.execute(
            "CREATE TABLE narrow (race TEXT, candidate BIGINT, votes BIGINT, PRIMARY KEY (race, candidate))"
        )
        await conn.execute(
            "INSERT INTO narrow SELECT 'r' || lpad(r::text, 5, '0'), c, r * c "
            "FROM generate_series(1, $1) r, generate_series(1, $2) c",
            _NARROW_RACES,
            _NARROW_PER_RACE,
        )
        await conn.execute("ANALYZE wide")
        await conn.execute("ANALYZE narrow")
        yield _Seeded(conn=conn, yugabyte=await on_yugabyte(conn))
    finally:
        await conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        await conn.close()


async def _ordered(conn: asyncpg.Connection, schema: TableSchema) -> list[tuple[Any, ...]]:
    """every key of the table, in the database's key order."""
    lead, rest = schema.pk_columns
    rows = await conn.fetch(f"SELECT {lead}, {rest} FROM {schema.name} ORDER BY {lead}, {rest}")  # noqa: S608
    return [tuple(row) for row in rows]


def _keys(rows: list[dict[str, Any]], schema: TableSchema) -> list[tuple[Any, ...]]:
    return [tuple(row[column] for column in schema.pk_columns) for row in rows]


def _reads(plans: list[StatementPlan]) -> list[StatementPlan]:
    """the statements that read the table (not the read's question of which database it is on)."""
    return [plan for plan in plans if "version()" not in plan.sql]


def _bounded(plans: list[StatementPlan], values: int) -> list[str]:
    """each statement that read past its own rows: a page of rows scanning more than two pages' rows (it read past
    its bucket), or a read of the lead's values scanning more than a row a value (it read the rows, not the bucket
    heads)."""
    return [
        f"scanned {plan.rows_scanned} rows for {plan.rows_returned}: {' '.join(plan.sql.split())[:200]}"
        for plan in plans
        if plan.rows_scanned > (values if plan.sql.startswith("SELECT DISTINCT") else 2 * (_PAGE + 1))
    ]


class TestAWholeReadOfAHashedLead:
    async def test_a_scan_of_a_table_of_large_lead_values_scans_about_its_own_rows(self, seeded: _Seeded) -> None:
        """Flips if a whole read pages a hashed lead by the cursor alone: every page reads the whole table."""
        checked = PlanRecorder(seeded.conn)
        backend = SqlL3Backend(checked)
        backend.register_schema(_WIDE.name, _WIDE)

        rows = await backend.scan(_WIDE.name)

        assert _keys(rows, _WIDE) == await _ordered(seeded.conn, _WIDE)
        if seeded.yugabyte:
            reads = _reads(checked.plans)
            assert not _bounded(reads, _WIDE_STATES), _bounded(reads, _WIDE_STATES)
            problems = plan_problems(reads, scan_ratio=_SCAN_RATIO, together=True)
            assert not problems, problems

    async def test_a_scan_of_a_table_of_many_small_lead_values_takes_a_statement_a_page(self, seeded: _Seeded) -> None:
        """Flips if small lead values are read one statement each, or a page re-reads the table."""
        checked = PlanRecorder(seeded.conn)
        backend = SqlL3Backend(checked)
        backend.register_schema(_NARROW.name, _NARROW)

        rows = await backend.scan(_NARROW.name)

        assert _keys(rows, _NARROW) == await _ordered(seeded.conn, _NARROW)
        if seeded.yugabyte:
            reads = _reads(checked.plans)
            assert not _bounded(reads, _NARROW_RACES), _bounded(reads, _NARROW_RACES)
            # the lead's values (three pages of them), then the rows a page or so a statement, ramping up from one
            # value a statement: never one statement a value
            assert len(reads) < _NARROW_RACES // 20, len(reads)
            problems = plan_problems(reads, scan_ratio=2.0, together=True)
            assert not problems, problems

    @pytest.mark.parametrize("schema", [_WIDE, _NARROW], ids=["large-values", "small-values"])
    async def test_a_read_past_a_saved_key_reads_exactly_the_rows_after_it(
        self, seeded: _Seeded, schema: TableSchema
    ) -> None:
        """Flips if a walk resumed past a key inside a lead value skips or repeats a row."""
        ordered = await _ordered(seeded.conn, schema)
        middle = len(ordered) // 2 + 7
        lead, rest = schema.pk_columns
        read: list[tuple[Any, ...]] = []
        async for page in keyset_pages(
            seeded.conn,
            f"SELECT {lead}, {rest}, votes FROM {schema.name}",  # noqa: S608 - trusted identifiers
            schema.pk_columns,
            page_size=_PAGE,
            after=ordered[middle],
            lead_array="text[]",
        ):
            read.extend(_keys(page, schema))

        assert read == ordered[middle + 1 :]

    async def test_a_read_of_no_rows_is_one_empty_page(self, seeded: _Seeded) -> None:
        """Flips if a walk of an empty table hands over no page, or a page with rows."""
        await seeded.conn.execute("CREATE TABLE empty (state TEXT, geo_id BIGINT, PRIMARY KEY (state, geo_id))")
        pages = [
            page
            async for page in keyset_pages(
                seeded.conn,
                "SELECT state, geo_id FROM empty",
                ("state", "geo_id"),
                page_size=_PAGE,
                lead_array="text[]",
            )
        ]

        assert pages == [[]]


#: the lead values a set filter pins in the set-pinned reads: a fifth of the wide table's states, a fifth of the
#: narrow table's races, given out of order and with one repeated and one holding no rows
_PINNED_STATES = [f"S{n:02d}" for n in range(0, _WIDE_STATES, 5)][::-1] + ["S05", "S99"]
_PINNED_RACES = [f"r{n:05d}" for n in range(1, _NARROW_RACES + 1, 5)][::-1] + ["r99999"]


def _sorts(node: Mapping[str, Any]) -> Iterator[int]:
    """the rows each sort in a plan sorted."""
    if "Sort" in str(node.get("Node Type", "")):
        yield int(node.get("Actual Rows", 0)) * int(node.get("Actual Loops", 1))
    for child in node.get("Plans", ()):
        yield from _sorts(child)


async def _pinned_read(
    conn: Any, schema: TableSchema, values: list[str], *, after: tuple[Any, ...] | None = None
) -> list[tuple[Any, ...]]:
    """the keys a read of ``schema`` with its lead pinned to ``values`` answers, page after page."""
    lead, rest = schema.pk_columns
    read: list[tuple[Any, ...]] = []
    async for page in keyset_pages(
        conn,
        f"SELECT {lead}, {rest}, votes FROM {schema.name}",  # noqa: S608 - trusted identifiers
        schema.pk_columns,
        where_any={lead: ("text[]", values)},
        page_size=_PAGE,
        after=after,
        lead_array="text[]",
    ):
        read.extend(_keys(page, schema))
    return read


async def _pinned_ordered(conn: asyncpg.Connection, schema: TableSchema, values: list[str]) -> list[tuple[Any, ...]]:
    """the keys the set filter names, in the database's key order, read in one statement."""
    lead, rest = schema.pk_columns
    rows = await conn.fetch(
        f"SELECT {lead}, {rest} FROM {schema.name} WHERE {lead} = ANY($1::text[]) ORDER BY {lead}, {rest}",  # noqa: S608
        values,
    )
    return [tuple(row) for row in rows]


class TestAReadOfALeadASetFilterPins:
    """hub #53: a read whose ``where_any`` pins the key's hashed lead to a set walks the set's values on YugabyteDB.
    Paged by the cursor it read every row of the set on every page and sorted them across values (33,528 rows in three
    pages: 33,528 scanned a page, two sorts on disk). On Postgres the lead is ordered and the read pages by the
    cursor; the same plan checks hold there."""

    @pytest.mark.parametrize(
        ("schema", "values"), [(_WIDE, _PINNED_STATES), (_NARROW, _PINNED_RACES)], ids=["large-values", "small-values"]
    )
    async def test_each_page_reads_its_own_rows_and_sorts_no_more_than_a_page(
        self, seeded: _Seeded, schema: TableSchema, values: list[str]
    ) -> None:
        """Flips if a page re-reads the set's rows past its cursor, or sorts more than a page of them: the read of
        the set's rows, in key order, scans about the rows it answers."""
        checked = PlanRecorder(seeded.conn)

        read = await _pinned_read(checked, schema, values)

        assert read == await _pinned_ordered(seeded.conn, schema, values)
        reads = _reads(checked.plans)
        sorted_past_a_page = [
            f"sorted {rows} rows: {' '.join(plan.sql.split())[:160]}"
            for plan in reads
            for rows in _sorts(plan.plan)
            if rows > _PAGE + 1
        ]
        assert not sorted_past_a_page, sorted_past_a_page
        returned = sum(plan.rows_returned for plan in reads)
        scanned = sum(plan.rows_scanned for plan in reads)
        detail = "\n".join(f"  {p.rows_scanned} / {p.rows_returned}: {' '.join(p.sql.split())[:220]}" for p in reads)
        assert scanned <= returned + _PAGE + 1, f"scanned {scanned} rows to answer {returned}\n{detail}"
        if schema is _NARROW:
            # small values go together: never a statement a value
            assert len(reads) < len(values) // 10, len(reads)

    @pytest.mark.parametrize(
        ("schema", "values"), [(_WIDE, _PINNED_STATES), (_NARROW, _PINNED_RACES)], ids=["large-values", "small-values"]
    )
    async def test_a_pinned_read_past_a_saved_key_reads_exactly_the_rows_after_it(
        self, seeded: _Seeded, schema: TableSchema, values: list[str]
    ) -> None:
        """Flips if a set-pinned read resumed past a key inside a value skips or repeats a row."""
        ordered = await _pinned_ordered(seeded.conn, schema, values)
        middle = len(ordered) // 2 + 7

        assert await _pinned_read(seeded.conn, schema, values, after=ordered[middle]) == ordered[middle + 1 :]
