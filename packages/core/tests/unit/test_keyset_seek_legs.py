"""YugabyteDB's resumed page (``page_statement(..., seek_legs=True)``) answers the rows the one-condition form did.

On YugabyteDB a page past a cursor is a ``UNION ALL`` of legs, each an equality prefix of the cursor and one range
(``c1 = $1 AND c2 < $2``, then ``c1 < $1``), so no condition is left to filter after the index read and the page's
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
    "stamp asc, label desc nullable, id desc": (
        Sort("stamp", cast="timestamptz"),
        Sort("label", cast="text", descending=True, nullable=True),
        Sort("id", cast="integer", descending=True),
    ),
}


def _page(conn: duckdb.DuckDBPyConnection, order: Sequence[Sort], past: list[Any] | None, legs: bool) -> list[tuple]:
    sql, params = page_statement(
        "r", ["stamp", "label", "id"], {"tenant": 1}, order, _LIMIT + 1, past, False, (), (), None, seek_legs=legs
    )
    return [tuple(row) for row in conn.execute(sql, params).fetchall()]


@pytest.mark.parametrize("name", list(_ORDERS))
def test_every_resumed_page_answers_the_expansion_s_rows(conn: duckdb.DuckDBPyConnection, name: str) -> None:
    """Flips if a page past any row, by legs, answers other rows (or another order) than the expansion."""
    order = _ORDERS[name]
    sql, params = page_statement("r", ["stamp", "label", "id"], {"tenant": 1}, order, 1000, None, False, (), (), None)
    every = [tuple(row) for row in conn.execute(sql, params).fetchall()]
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
    assert 't."stamp" = $2::text::timestamptz AND t."id" < $3::text::integer' in sql, sql
    assert 't."stamp" < $2::text::timestamptz' in sql, sql
    assert params == [1, "2026-01-01 00:03:00+00", "7"]
