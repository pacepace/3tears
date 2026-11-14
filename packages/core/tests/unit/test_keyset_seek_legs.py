"""YugabyteDB's resumed page (``page_statement(..., seek_legs=True)``) answers the rows the one-condition form did.

On YugabyteDB a page past a cursor is a ``UNION ALL`` of legs, each the cursor's tied prefix and one range (``c1 >= $1
AND c1 <= $1 AND c2 < $2``, then ``c1 < $1``), so no condition is left to filter after the index read and the page's
``LIMIT`` reaches storage. Run on an in-process DuckDB here, against the expansion it replaces, at every cursor
position of a table with ties, NULLs in a later sort column and mixed directions; the plan it reads on YugabyteDB is
the conversations listing's plan test (``packages/conversations/tests/integration/test_index_plans.py``).
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

import duckdb
import pytest

from threetears.interfaces.keyed.page import Sort, next_past, page_statement

#: the page size
_LIMIT = 4


@pytest.fixture
def conn() -> Iterator[duckdb.DuckDBPyConnection]:
    """rows of two tenants: a timestamp with many ties, a NULLable label, a unique id."""
    db = duckdb.connect()
    db.execute("CREATE TABLE r (tenant INTEGER, stamp TIMESTAMPTZ, label VARCHAR, id INTEGER)")
    db.execute(
        "INSERT INTO r SELECT i % 2, TIMESTAMPTZ '2026-01-01 00:00:00+00' + to_minutes(i % 7), "
        "CASE WHEN i % 3 = 0 THEN NULL ELSE 'l' || (i % 4) END, i FROM range(60) t(i)"
    )
    try:
        yield db
    finally:
        db.close()


_ORDERS: dict[str, Sequence[Sort]] = {
    "stamp desc, id desc": (
        Sort("stamp", cast="timestamptz", descending=True),
        Sort("id", cast="integer", descending=True),
    ),
    "stamp asc, id asc": (Sort("stamp", cast="timestamptz"), Sort("id", cast="integer")),
    "stamp desc, label asc nullable, id asc": (
        Sort("stamp", cast="timestamptz", descending=True),
        Sort("label", cast="text", nullable=True),
        Sort("id", cast="integer"),
    ),
    "label asc nullable, id asc": (Sort("label", cast="text", nullable=True), Sort("id", cast="integer")),
    "label desc nullable, id desc": (
        Sort("label", cast="text", descending=True, nullable=True),
        Sort("id", cast="integer", descending=True),
    ),
    "stamp asc, label desc nullable, id desc": (
        Sort("stamp", cast="timestamptz"),
        Sort("label", cast="text", descending=True, nullable=True),
        Sort("id", cast="integer", descending=True),
    ),
}


def _page(
    conn: duckdb.DuckDBPyConnection, order: Sequence[Sort], past: list[Any] | None, legs: bool
) -> list[tuple[Any, ...]]:
    sql, params = page_statement(
        "r", ["stamp", "label", "id"], {"tenant": 1}, order, _LIMIT + 1, past, False, (), (), None, seek_legs=legs
    )
    return [tuple(row) for row in conn.execute(sql, params).fetchall()]


@pytest.mark.parametrize("name", list(_ORDERS))
def test_every_resumed_page_answers_the_expansion_s_rows(conn: duckdb.DuckDBPyConnection, name: str) -> None:
    """Flips if a page past any row, by legs, answers other rows (or another order) than the expansion."""
    order = _ORDERS[name]
    sql, params = page_statement("r", ["stamp", "label", "id"], {"tenant": 1}, order, 1000, None, False, (), (), None)
    every: list[tuple[Any, ...]] = [tuple(row) for row in conn.execute(sql, params).fetchall()]
    assert len(every) == 30
    for row in every:
        past = next_past(order, dict(zip(("stamp", "label", "id"), row, strict=True)))
        legs, expansion = _page(conn, order, past, True), _page(conn, order, past, False)
        assert legs == expansion, (row, legs, expansion)
        start = every.index(row) + 1
        assert legs == every[start : start + _LIMIT + 1], row


def test_the_legs_are_ranges_with_nothing_left_to_filter() -> None:
    """Flips if a leg carries an ``OR`` (a condition YugabyteDB filters after the index read, which keeps the page's
    ``LIMIT`` from storage) or the page stops being one leg a sort column."""
    order = _ORDERS["stamp desc, id desc"]
    sql, params = page_statement(
        "r",
        ["stamp", "id"],
        {"tenant": 1},
        order,
        9,
        ["2026-01-01 00:03:00+00", "7"],
        False,
        (),
        (),
        None,
        seek_legs=True,
    )
    assert " OR " not in sql, sql
    assert sql.count("UNION ALL") == 1, sql
    assert (
        't."stamp" >= $2::text::timestamptz AND t."stamp" <= $2::text::timestamptz AND t."id" < $3::text::integer'
    ) in sql, sql
    assert 't."stamp" < $2::text::timestamptz' in sql, sql
    assert params == [1, "2026-01-01 00:03:00+00", "7"]


def test_each_leg_asks_only_what_the_legs_before_it_left_short() -> None:
    """Flips if a later leg asks a whole page again (a page past a cursor among many ties of a NULLable lead read
    ties + a page a leg on YugabyteDB, 44 rows for a page of 21 against the bound 43), or the legs stop being read in
    the sort's order."""
    order = _ORDERS["label asc nullable, id asc"]
    sql, _ = page_statement(
        "r", ["label", "id"], {"tenant": 1}, order, 9, ["l2", "7"], False, (), (), None, seek_legs=True
    )
    assert sql.startswith("WITH l0 AS ("), sql
    assert (
        't."label" >= $2::text AND t."label" <= $2::text AND t."id" > $3::text::integer' in sql.split("l1 AS (")[0]
    ), sql
    assert "LIMIT 9 - (SELECT count(*) FROM l0))" in sql, sql
    assert "LIMIT 9 - (SELECT count(*) FROM l0) - (SELECT count(*) FROM l1))" in sql, sql
    assert 't."label" IS NULL' in sql.split("l2 AS (")[1], sql


def test_a_tie_of_the_cursor_is_a_closed_range_not_an_equality() -> None:
    """Flips if a leg states the cursor's tie on an earlier sort column as an equality. ``status = $2`` makes the
    column a constant to the planner, which drops it from the leg's order, so an index that does not hold it (the
    primary key ``(agent_id, conversation_id)``) answers the order too and YugabyteDB walked it, filtering the status:
    27-65 rows read for a page of 9 (test_conv_agent_status, 3tears #86). A closed range (``status >= $2 AND status <=
    $2``) keeps the column in the order, so only an index ordered by the sort serves the leg, every condition its
    range."""
    order = _ORDERS["stamp desc, id desc"]
    sql, _ = page_statement(
        "r",
        ["stamp", "id"],
        {"tenant": 1},
        order,
        9,
        ["2026-01-01 00:03:00+00", "7"],
        False,
        (),
        (),
        None,
        seek_legs=True,
    )
    assert 't."stamp" = $' not in sql, sql
    assert 't."stamp" >= $2::text::timestamptz AND t."stamp" <= $2::text::timestamptz' in sql, sql
