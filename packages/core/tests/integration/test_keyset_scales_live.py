"""Integration: a filtered keyset read touches about its own rows, at real size, and keeps its budget.

A keyset read pages ``ORDER BY`` the columns it is given. Ordered by the primary key while the filter is
found through another index, every page re-reads and re-sorts every row the filter leaves (YugabyteDB,
a state of 33,222 candidate rows: 34 pages, about 1.13M rows scanned, 10.1 s). These read a table seeded
like the reports pod's candidates (about 357,000 rows across 50 states, 33,528 in the largest), filtered
by state and by 200 of its geographies. Every read answers exactly the rows a plain query names, once each.
On YugabyteDB, the platform's database, each read's real plans are judged with the shared plan check
(:mod:`threetears.core.testing.plans`): rows scanned within 1.1x the rows returned over the read. Postgres
plans these pages its own way (it takes the state and the geography for independent and sorts), so there
the reads are checked for correctness only.

Against the session ``db_container`` (Postgres); set ``THREETEARS_TEST_POSTGRES_URL`` to a YugabyteDB URL
to run them there, where the problem was measured.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import asyncpg
import pytest

from threetears.core.backends.sql import SqlL3Backend
from threetears.core.collections.complete_copy import read_l3_rows
from threetears.core.collections.schema_backed import BIGINT_TYPE, STRING_TYPE, Column, Index, TableSchema
from threetears.core.testing.plans import PlanRecorder, on_yugabyte, plan_problems

pytestmark = pytest.mark.integration

_TABLE = "results"
_STATE = "join_state_code"
_GEO = "join_geo_id"

#: shaped like the pod's live candidates table: keyed on the race, found by the state
_SCHEMA = TableSchema(
    name=_TABLE,
    primary_key=("office_key", "candidate", _GEO),
    columns=[
        Column("office_key", STRING_TYPE),
        Column("candidate", STRING_TYPE),
        Column(_GEO, STRING_TYPE),
        Column(_STATE, STRING_TYPE),
        Column("votes", BIGINT_TYPE, nullable=True),
        Column("pad", STRING_TYPE, nullable=True),
    ],
    # led by the state, narrowed by the geography, the key its tail
    indexes=(Index("results_by_state", _STATE, _GEO, "office_key", "candidate"),),
)

#: the largest state's geographies; every other state has 50
_BIG_GEOGRAPHIES = 254
_STATES = 50
_OFFICES = 33
_CANDIDATES = 4

#: the most rows a read scans for each row it answers
_SCAN_RATIO = 1.1


@dataclass
class _Seeded:
    conn: asyncpg.Connection
    columns: list[str]
    yugabyte: bool


@pytest.fixture
async def seeded(db_container: str) -> AsyncIterator[_Seeded]:
    """the table, seeded at real size in a schema of its own, with its declared index and fresh statistics."""
    schema = f"keyset_{uuid.uuid4().hex[:8]}"
    conn = await asyncpg.connect(db_container)
    try:
        await conn.execute(f"CREATE SCHEMA {schema}")
        await conn.execute(f"SET search_path = {schema}")
        await conn.execute(
            f"CREATE TABLE {_TABLE} (office_key TEXT NOT NULL, candidate TEXT NOT NULL, {_GEO} TEXT NOT NULL, "
            f"{_STATE} TEXT NOT NULL, votes BIGINT, pad TEXT, PRIMARY KEY (office_key, candidate, {_GEO}))"
        )
        for index in _SCHEMA.indexes:
            await conn.execute(f"CREATE INDEX {index.name} ON {_TABLE} ({', '.join(index.columns)})")
        await conn.execute(
            f"""INSERT INTO {_TABLE}
            SELECT 'o' || o, 'c' || c, st.s || '-' || lpad(g::text, 3, '0'), st.s, o * c * g, repeat('x', 40)
            FROM (SELECT CASE WHEN n = 0 THEN 'TX' ELSE 'S' || lpad(n::text, 2, '0') END AS s,
                         CASE WHEN n = 0 THEN {_BIG_GEOGRAPHIES} ELSE 50 END AS geos
                  FROM generate_series(0, {_STATES - 1}) n) st,
                 generate_series(1, {_OFFICES}) o, generate_series(1, {_CANDIDATES}) c,
                 LATERAL generate_series(1, st.geos) g"""
        )
        await conn.execute(f"ANALYZE {_TABLE}")
        yield _Seeded(conn=conn, columns=[column.name for column in _SCHEMA.columns], yugabyte=await on_yugabyte(conn))
    finally:
        await conn.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        await conn.close()


def _keys(rows: list[dict[str, Any]]) -> list[tuple[Any, ...]]:
    return [tuple(row[column] for column in _SCHEMA.pk_columns) for row in rows]


async def _expected(conn: asyncpg.Connection, where_sql: str, *args: Any) -> set[tuple[Any, ...]]:
    rows = await conn.fetch(f"SELECT office_key, candidate, {_GEO} FROM {_TABLE} WHERE {where_sql}", *args)
    return {tuple(row) for row in rows}


def _geographies() -> list[str]:
    return [f"TX-{number:03d}" for number in range(1, 201)]


class TestAStatesReadScalesWithItsRows:
    async def test_a_scan_of_the_largest_state_scans_about_its_own_rows(self, seeded: _Seeded) -> None:
        """Flips if a filtered scan pages in an order no index answers: its pages re-read the whole state."""
        checked = PlanRecorder(seeded.conn)
        backend = SqlL3Backend(checked)
        backend.register_schema(_TABLE, _SCHEMA)

        rows = await backend.scan(_TABLE, {_STATE: "TX"})

        keys = _keys(rows)
        assert len(keys) == len(set(keys)), "a row was read twice"
        assert set(keys) == await _expected(seeded.conn, f"{_STATE} = 'TX'")
        if seeded.yugabyte:
            problems = plan_problems(checked.plans, scan_ratio=_SCAN_RATIO, together=True)
            assert not problems, problems

    async def test_a_read_of_the_largest_state_scans_about_its_own_rows(self, seeded: _Seeded) -> None:
        """Flips if read_l3_rows, given the schema's keyset order, re-reads the state each page."""
        order = _SCHEMA.keyset_order({_STATE})
        checked = PlanRecorder(seeded.conn)

        rows = await read_l3_rows(checked, _TABLE, seeded.columns, order, where={_STATE: "TX"})

        keys = _keys(rows)
        assert len(keys) == len(set(keys)), "a row was read twice"
        assert set(keys) == await _expected(seeded.conn, f"{_STATE} = 'TX'")
        if seeded.yugabyte:
            problems = plan_problems(checked.plans, scan_ratio=_SCAN_RATIO, together=True)
            assert not problems, problems

    async def test_a_read_of_200_geographies_scans_about_its_own_rows(self, seeded: _Seeded) -> None:
        """Flips if a read of many geographies at once re-reads its rows each page."""
        order = _SCHEMA.keyset_order({_STATE})
        geographies = _geographies()
        where_any = {_GEO: ("text[]", geographies)}
        checked = PlanRecorder(seeded.conn)

        rows = await read_l3_rows(checked, _TABLE, seeded.columns, order, where={_STATE: "TX"}, where_any=where_any)

        keys = _keys(rows)
        assert len(keys) == len(set(keys)), "a row was read twice"
        assert set(keys) == await _expected(seeded.conn, f"{_STATE} = 'TX' AND {_GEO} = ANY($1::text[])", geographies)
        if seeded.yugabyte:
            problems = plan_problems(checked.plans, scan_ratio=_SCAN_RATIO, together=True)
            assert not problems, problems

    async def test_the_plan_check_names_a_read_paged_by_the_primary_key(self, seeded: _Seeded) -> None:
        """Flips if the plan check stops seeing the audit's shape: a state's rows paged by a key no index orders."""
        checked = PlanRecorder(seeded.conn)

        rows = await read_l3_rows(checked, _TABLE, seeded.columns, _SCHEMA.pk_columns, where={_STATE: "TX"})

        assert len(rows) == len(await _expected(seeded.conn, f"{_STATE} = 'TX'"))
        problems = plan_problems(checked.plans, scan_ratio=_SCAN_RATIO, together=True)
        assert problems and "scanned" in problems[0], problems
        # on either database: a state's every row, re-read each of its 34 pages
