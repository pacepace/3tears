"""Integration: a page resumed past a cursor of a sort led by a NULLable column reads a bounded range of its index.

The one-condition form (``lead > $v OR lead IS NULL OR (lead = $v AND id > $w)``) gives the lead no bound an index can
start from, so a resumed page read the ``(tenant, label, id)`` index from the tenant's start: 1,872 rows scanned for a
page of 21 on the hub's api_keys. Split at the NULLs (:func:`threetears.interfaces.keyed.page.page_statement`), each
leg is an ``Index Cond`` range of the same index and a page scans at most ``2 * (limit + 1) + 1`` rows. The rows are
the old form's, at every mark position (a value, the last value, the first NULL, a NULL, the last row), both
directions.

On the session's ``db_container`` (PostgreSQL), or YugabyteDB through ``THREETEARS_TEST_POSTGRES_URL``; a schema of
its own.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator, Iterator, Mapping
from typing import Any

import asyncpg
import pytest

from threetears.core.keyed import Sort, keyset_page
from threetears.core.testing.plans import PlanRecorder
from threetears.interfaces.keyed import keyset_cursor

pytestmark = pytest.mark.integration

#: the tenant paged, its rows, and every fifth row's label NULL
_TENANT = 1
_ROWS = 5000
_LIMIT = 20


@pytest.fixture
async def conn(db_container: str) -> AsyncIterator[asyncpg.Connection]:
    """a table of labelled rows by tenant, every fifth label NULL, its ``(tenant, label, id)`` index analyzed."""
    schema = f"nullable_{uuid.uuid4().hex[:10]}"
    connection = await asyncpg.connect(db_container)
    try:
        await connection.execute(f'CREATE SCHEMA "{schema}"')
        await connection.execute(f'SET search_path TO "{schema}"')
        await connection.execute(
            "CREATE TABLE nl (tenant INT NOT NULL, id BIGINT NOT NULL, label TEXT, PRIMARY KEY (tenant, id))"
        )
        await connection.execute("CREATE INDEX nl_label ON nl (tenant, label, id)")
        await connection.execute(
            "INSERT INTO nl SELECT t, i, CASE WHEN i % 5 = 0 THEN NULL ELSE 'l' || lpad((i % 997)::text, 4, '0') END "
            "FROM generate_series(1, 4) t, generate_series(1, $1) i",
            _ROWS,
        )
        await connection.execute("ANALYZE nl")
        yield connection
    finally:
        await connection.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await connection.close()


def _order(descending: bool) -> list[Sort]:
    return [Sort("label", descending=descending, nullable=True), Sort("id", "bigint", descending)]


def _old_statement(descending: bool, label: str | None, row_id: int) -> tuple[str, list[Any]]:
    """the page past ``(label, row_id)`` as the one-condition form wrote it before the split: the lead unbounded."""
    params: list[Any] = [_TENANT]
    mark: str | None = None
    if label is not None:
        params.append(label)
        mark = f"${len(params)}::text"
    params.append(str(row_id))
    id_mark = f"${len(params)}::text::bigint"
    column = 't."label"'
    if mark is None:
        after = f"{column} IS NOT NULL" if descending else "FALSE"
        same = f"{column} IS NULL"
    else:
        after = f"{column} < {mark}" if descending else f"({column} > {mark} OR {column} IS NULL)"
        same = f"{column} = {mark}"
    beyond = f't."id" {"<" if descending else ">"} {id_mark}'
    direction = "DESC NULLS FIRST" if descending else "ASC NULLS LAST"
    tie = "DESC" if descending else "ASC"
    sql = (
        f'SELECT t."label", t."id" FROM "nl" AS t WHERE t."tenant" = $1 AND ({after} OR ({same} AND ({beyond}))) '
        f'ORDER BY t."tenant" ASC, {column} {direction}, t."id" {tie} LIMIT {_LIMIT + 1}'
    )
    return sql, params


async def _every_row(conn: asyncpg.Connection, descending: bool) -> list[dict[str, Any]]:
    direction = "DESC NULLS FIRST" if descending else "ASC NULLS LAST"
    tie = "DESC" if descending else "ASC"
    rows = await conn.fetch(
        f"SELECT label, id FROM nl WHERE tenant = $1 ORDER BY label {direction}, id {tie}",  # noqa: S608
        _TENANT,
    )
    return [dict(row) for row in rows]


def _marks(every: list[dict[str, Any]]) -> list[int]:
    """the mark positions: a value, the last value, the first NULL, a NULL, the last row, and every 701st."""
    nulls = [index for index, row in enumerate(every) if row["label"] is None]
    values = [index for index, row in enumerate(every) if row["label"] is not None]
    picked = {values[len(values) // 2], values[-1], values[0], nulls[0], nulls[len(nulls) // 2], len(every) - 1}
    picked.update(range(0, len(every), 701))
    return sorted(picked)


def _nodes(node: Mapping[str, Any]) -> Iterator[Mapping[str, Any]]:
    yield node
    for child in node.get("Plans", ()):
        yield from _nodes(child)


@pytest.mark.parametrize("descending", [False, True], ids=["ascending", "descending"])
async def test_a_resumed_page_reads_the_old_forms_rows_from_a_bounded_range(
    conn: asyncpg.Connection, descending: bool
) -> None:
    """Flips if a page resumed past any mark (a value, the last value, the first NULL, a NULL, the last row) answers
    other rows than the one-condition form did, or reads its index from anywhere but a bounded range: every scan an
    Index Cond range of (tenant, label, id), no table read whole, and at most 2 * (limit + 1) + 1 rows scanned (or,
    within a few pages of the sort's end, no more than the rows past the mark and a page of its ties: there the planner
    may read that short range and sort it)."""
    order = _order(descending)
    every = await _every_row(conn, descending)
    assert any(row["label"] is None for row in every) and any(row["label"] is not None for row in every)
    for index in _marks(every):
        row = every[index]
        recorder = PlanRecorder(conn)
        page = await keyset_page(
            recorder,
            "nl",
            columns=["label", "id"],
            by={"tenant": _TENANT},
            order=order,
            limit=_LIMIT,
            cursor=keyset_cursor(order, row),
        )
        old_sql, old_params = _old_statement(descending, row["label"], row["id"])
        old = [dict(found) for found in await conn.fetch(old_sql, *old_params)][:_LIMIT]
        assert page.rows == old == every[index + 1 : index + 1 + _LIMIT], (index, row)
        (plan,) = recorder.plans
        past = len(every) - index - 1
        bound = 2 * (_LIMIT + 1) + 1 if past > 5 * (_LIMIT + 1) else past + _LIMIT + 1
        assert plan.rows_scanned <= bound, (index, row, plan.rows_scanned, bound, plan.sql)
        nodes = list(_nodes(plan.plan))
        assert not [node for node in nodes if node["Node Type"] == "Seq Scan"], (index, json.dumps(plan.plan)[:3000])
        # each read of the index (an index scan, or a bitmap's index scan) is a range of the lead, never its start
        reads = [node for node in nodes if "Index Name" in node]
        assert reads, json.dumps(plan.plan)[:3000]
        for read in reads:
            assert read["Index Name"] == "nl_label", (index, row, json.dumps(plan.plan)[:3000])
            assert "label" in str(read.get("Index Cond", "")), (index, row, json.dumps(read)[:2000])
