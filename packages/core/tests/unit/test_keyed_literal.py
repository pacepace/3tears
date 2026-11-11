"""``keyed.literal_select``: a keyed read written out with literals (an export's ``SELECT``) reads what the bound read
reads."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal

import duckdb
import pytest

from threetears.core.keyed import AnyOf, Fragment, Selected, literal_select


@pytest.fixture
def db() -> duckdb.DuckDBPyConnection:
    """runs of a long relation: run ids 1 .. 4, two families each, one row with no geography."""
    conn = duckdb.connect()
    conn.execute("CREATE TABLE parts (run_id BIGINT, family VARCHAR, geo VARCHAR, value DOUBLE)")
    conn.execute(
        "INSERT INTO parts VALUES (1, 'race', 'a', 1.5), (1, 'county_group', 'b', 2), (2, 'race', NULL, 3), "
        "(3, 'race', 'c', 4), (4, 'o''hare', 'd', 5)"
    )
    return conn


def test_a_literal_select_reads_what_the_bound_read_reads(db: duckdb.DuckDBPyConnection) -> None:
    """Flips if the written-out statement reads other rows than the same keyed read bound: a set as IN, a value as =,
    a filter's parameters as literals, an empty set as no row, a quote in a value doubled."""
    written = literal_select(
        "parts",
        columns=["run_id", "family", Selected("upper(t.geo)", "geo")],
        by={"run_id": AnyOf([1, 2, 4], "bigint"), "family": AnyOf(["race", "o'hare"])},
        where=[Fragment("t.geo IS NOT NULL AND t.value > $1", (1,))],
    )
    assert "$" not in written and 't."run_id" IN (1, 2, 4)' in written
    bound = db.execute(
        "SELECT run_id, family, upper(geo) AS geo FROM parts AS t WHERE run_id IN (1, 2, 4) "
        "AND family IN ('race', 'o''hare') AND geo IS NOT NULL AND value > 1"
    ).fetchall()
    assert sorted(db.execute(written).fetchall()) == sorted(bound) == [(1, "race", "A"), (4, "o'hare", "D")]
    empty = literal_select("parts", columns=["run_id"], by={"run_id": AnyOf([], "bigint")})
    assert db.execute(empty).fetchall() == []
    one = literal_select("parts", columns=["run_id"], by={"run_id": 3, "geo": None})
    assert db.execute(one).fetchall() == [] and 't."geo" IS NULL' in one


def test_a_literal_select_refuses_a_whole_read_and_a_value_with_no_literal() -> None:
    """Flips if a statement pinning nothing is written (a whole read), or a value is written that no literal holds."""
    with pytest.raises(ValueError, match="whole read"):
        literal_select("parts", columns=["run_id"], by={})
    with pytest.raises(ValueError, match="no SQL literal"):
        literal_select("parts", columns=["run_id"], by={"run_id": object()})
    with pytest.raises(ValueError, match="finite"):
        literal_select("parts", columns=["value"], by={"value": float("nan")})
    values = (True, 7, Decimal("1.50"), 2.5, date(2026, 11, 3), "it's", datetime(2026, 11, 3, 21, 0))  # noqa: DTZ001
    written = [literal_select("p", columns=["v"], by={"v": value}).rsplit(" = ", 1)[1] for value in values]
    assert written == ["TRUE", "7", "1.50", "2.5", "'2026-11-03'", "'it''s'", "'2026-11-03 21:00:00'"]
