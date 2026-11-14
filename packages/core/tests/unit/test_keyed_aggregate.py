"""keyed aggregates (:func:`aggregate_keyed`): counts and sums of the rows a key prefix, a set of key values and a
range name, grouped or not, a bounded batch of values a statement and the batches merged; on an in-process DuckDB."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timedelta
from typing import Any

import duckdb
import pytest

from threetears.core.keyed import AnyOf, Count, Fragment, InlineDuckDB, Max, Min, Range, Selected, Sum, aggregate_keyed

_START = datetime(2026, 11, 1)


@pytest.fixture
def db() -> Iterator[duckdb.DuckDBPyConnection]:
    conn = duckdb.connect()
    conn.execute(
        "CREATE TABLE events (customer_id VARCHAR, created TIMESTAMP, agent_id VARCHAR, kind VARCHAR, "
        "input_tokens BIGINT, output_tokens BIGINT)"
    )
    conn.execute(
        "INSERT INTO events SELECT 'c' || (i % 5), TIMESTAMP '2026-11-01' + to_minutes(i * 7), 'a' || (i % 3), "
        "CASE WHEN i % 4 = 0 THEN 'tool.call' ELSE 'turn' END, i, i % 11 FROM range(600) r(i)"
    )
    yield conn
    conn.close()


class _Recording:
    """an inline DuckDB that keeps every statement it is sent."""

    binds_native_arrays = False

    def __init__(self, db: duckdb.DuckDBPyConnection) -> None:
        self._db = InlineDuckDB(db)
        self.statements: list[str] = []

    async def fetch(self, query: str, *params: Any) -> list[dict[str, Any]]:
        self.statements.append(query)
        return await self._db.fetch(query, *params)


def _plain(db: duckdb.DuckDBPyConnection, sql: str, *params: Any) -> list[tuple[Any, ...]]:
    return db.execute(sql, list(params)).fetchall()


_LOW, _HIGH = _START + timedelta(hours=10), _START + timedelta(hours=40)


async def test_counts_and_sums_by_group_are_the_hand_written_aggregate_s(db: duckdb.DuckDBPyConnection) -> None:
    """Flips if a grouped count or sum differs from the hand-written GROUP BY over the same pins, values and range,
    or a batch of values is lost or counted twice when merged."""
    customers = ["c0", "c1", "c3", "c4", "c9"]
    reader = _Recording(db)
    rows = await aggregate_keyed(
        reader,
        "events",
        aggregates={"requests": Count(), "tokens": Sum('t."input_tokens" + t."output_tokens"')},
        by={"kind": "turn", "customer_id": AnyOf(customers)},
        ranges=[Range("created", _LOW, _HIGH)],
        group=["agent_id"],
        batch=2,
    )
    assert len(reader.statements) == 3 and all("GROUP BY" in statement for statement in reader.statements)
    expected = _plain(
        db,
        "SELECT agent_id, count(*), sum(input_tokens + output_tokens) FROM events WHERE kind = 'turn' "
        "AND customer_id IN ('c0', 'c1', 'c3', 'c4', 'c9') AND created >= ? AND created < ? GROUP BY agent_id",
        _LOW,
        _HIGH,
    )
    assert sorted((row["agent_id"], row["requests"], row["tokens"]) for row in rows) == sorted(expected)


async def test_an_ungrouped_aggregate_is_one_row_even_of_no_rows_and_of_no_values(
    db: duckdb.DuckDBPyConnection,
) -> None:
    """Flips if an ungrouped aggregate answers other than one row, a count of nothing is not zero (a sum of nothing
    is None, as SQL's), or an empty set of values sends a statement."""
    reader = _Recording(db)
    (whole,) = await aggregate_keyed(
        reader, "events", aggregates={"n": Count(), "s": Sum('t."input_tokens"')}, by={"customer_id": "c2"}
    )
    assert (whole["n"], whole["s"]) == _plain(
        db, "SELECT count(*), sum(input_tokens) FROM events WHERE customer_id = 'c2'"
    )[0]
    (none,) = await aggregate_keyed(
        reader, "events", aggregates={"n": Count(), "s": Sum('t."input_tokens"')}, by={"customer_id": "nobody"}
    )
    assert none == {"n": 0, "s": None}
    sent = len(reader.statements)
    (empty,) = await aggregate_keyed(
        reader, "events", aggregates={"n": Count(), "s": Sum('t."input_tokens"')}, by={"customer_id": AnyOf([])}
    )
    assert empty == {"n": 0, "s": None} and len(reader.statements) == sent


async def test_largest_and_smallest_merge_across_batches_as_the_hand_written_max_and_min(
    db: duckdb.DuckDBPyConnection,
) -> None:
    """Flips if a MAX or MIN differs from the hand-written aggregate over the same pins, grouped or not, a batch's
    answer replaces a larger (smaller) one when merged, or a MAX of no row is other than None, as SQL's."""
    customers = ["c0", "c1", "c3", "c4", "c9"]
    reader = _Recording(db)
    aggregates = {"newest": Max('t."created"'), "fewest": Min('t."input_tokens"'), "n": Count('t."agent_id"')}
    grouped = await aggregate_keyed(
        reader, "events", aggregates=aggregates, by={"customer_id": AnyOf(customers)}, group=["kind"], batch=2
    )
    assert len(reader.statements) == 3
    expected = _plain(
        db,
        "SELECT kind, max(created), min(input_tokens), count(agent_id) FROM events "
        "WHERE customer_id IN ('c0', 'c1', 'c3', 'c4', 'c9') GROUP BY kind",
    )
    assert sorted((row["kind"], row["newest"], row["fewest"], row["n"]) for row in grouped) == sorted(expected)
    (whole,) = await aggregate_keyed(
        reader, "events", aggregates=aggregates, by={"customer_id": AnyOf(customers)}, batch=2
    )
    assert (whole["newest"], whole["fewest"], whole["n"]) == _plain(
        db,
        "SELECT max(created), min(input_tokens), count(agent_id) FROM events "
        "WHERE customer_id IN ('c0', 'c1', 'c3', 'c4', 'c9')",
    )[0]
    (none,) = await aggregate_keyed(reader, "events", aggregates=aggregates, by={"customer_id": "nobody"})
    assert none == {"newest": None, "fewest": None, "n": 0}


async def test_two_ranges_on_one_column_are_or_ed_and_an_inclusive_edge_kept(db: duckdb.DuckDBPyConnection) -> None:
    """Flips if two ranges are AND-ed (no row), an inclusive upper edge drops its row, or a range named by its upper
    edge alone reads past it."""
    edge = _START + timedelta(minutes=7 * 30)
    (counted,) = await aggregate_keyed(
        InlineDuckDB(db),
        "events",
        aggregates={"n": Count()},
        by={"kind": "turn"},
        ranges=[
            Range("created", _START, _START + timedelta(hours=1)),
            Range("created", _LOW, edge, high_inclusive=True),
        ],
    )
    expected = _plain(
        db,
        "SELECT count(*) FROM events WHERE kind = 'turn' AND ((created >= ? AND created < ?) OR "
        "(created >= ? AND created <= ?))",
        _START,
        _START + timedelta(hours=1),
        _LOW,
        edge,
    )
    assert counted["n"] == expected[0][0]
    (upto,) = await aggregate_keyed(
        InlineDuckDB(db), "events", aggregates={"n": Count()}, by={"kind": "turn"}, ranges=[Range("created", high=_LOW)]
    )
    assert upto["n"] == _plain(db, "SELECT count(*) FROM events WHERE kind = 'turn' AND created < ?", _LOW)[0][0]


async def test_a_group_expression_buckets_and_a_where_narrows(db: duckdb.DuckDBPyConnection) -> None:
    """Flips if a group named as an expression (a time bucket) is not grouped by, or a where fragment is not
    renumbered past the pins and the range."""
    rows = await aggregate_keyed(
        InlineDuckDB(db),
        "events",
        aggregates={"n": Count()},
        by={"customer_id": AnyOf(["c1", "c2"])},
        ranges=[Range("created", _LOW, _HIGH)],
        where=[Fragment('t."agent_id" = $1', ("a1",))],
        group=[Selected("date_trunc('hour', t.\"created\")", "bucket")],
        batch=1,
    )
    expected = _plain(
        db,
        "SELECT date_trunc('hour', created) AS b, count(*) FROM events WHERE customer_id IN ('c1', 'c2') "
        "AND created >= ? AND created < ? AND agent_id = 'a1' GROUP BY b",
        _LOW,
        _HIGH,
    )
    assert sorted((row["bucket"], row["n"]) for row in rows) == sorted(expected)


async def test_groups_with_no_aggregate_are_the_distinct_values_across_batches(db: duckdb.DuckDBPyConnection) -> None:
    """Flips if a keyed DISTINCT (groups, no aggregate) answers a value twice when two batches both hold it."""
    rows = await aggregate_keyed(
        InlineDuckDB(db),
        "events",
        aggregates={},
        by={"customer_id": AnyOf(["c0", "c1", "c2", "c3"])},
        ranges=[Range("created", _LOW, _HIGH)],
        group=["agent_id"],
        batch=1,
    )
    assert sorted(row["agent_id"] for row in rows) == ["a0", "a1", "a2"]


async def test_an_aggregate_names_its_rows_and_its_ranges_one_column() -> None:
    """Flips if an aggregate of no prefix and no range (the whole table) is sent, ranges on two columns are taken,
    a range with no edge, or nothing to answer (no aggregate, no group)."""
    db = duckdb.connect()
    db.execute("CREATE TABLE t (a VARCHAR, b TIMESTAMP, c TIMESTAMP)")
    reader = InlineDuckDB(db)
    with pytest.raises(ValueError, match="whole"):
        await aggregate_keyed(reader, "t", aggregates={"n": Count()})
    with pytest.raises(ValueError, match="one column"):
        await aggregate_keyed(reader, "t", aggregates={"n": Count()}, ranges=[Range("b", _START), Range("c", _START)])
    with pytest.raises(ValueError, match="edge"):
        Range("b")
    with pytest.raises(ValueError, match="answers"):
        await aggregate_keyed(reader, "t", aggregates={}, by={"a": "x"})
    with pytest.raises(ValueError, match="batch"):
        await aggregate_keyed(reader, "t", aggregates={"n": Count()}, by={"a": "x"}, batch=0)
    db.close()
