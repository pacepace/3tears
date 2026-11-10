"""The runtime half of the no-whole-reads rule: a guarded connection fails a load-path test on any whole read."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from threetears.core.testing.plans import WholeScanGuard


def _scan(node: str, relation: str = "results", **extra: object) -> dict[str, Any]:
    return {"Node Type": node, "Relation Name": relation, "Schema": "public", **extra}


class _Conn:
    """a connection answering ``EXPLAIN`` with a given plan per statement, counts by table, and rows as asked."""

    def __init__(self, plans: Mapping[str, Mapping[str, Any]], sizes: Mapping[str, int], rows: int = 0) -> None:
        self._plans = plans
        self._sizes = sizes
        self._rows = rows
        self.ran: list[str] = []

    async def fetchval(self, sql: str, *args: Any) -> Any:
        if sql.startswith("EXPLAIN"):
            statement = sql.removeprefix("EXPLAIN (VERBOSE, FORMAT JSON) ")
            return json.dumps([{"Plan": self._plans[statement]}])
        if sql.startswith("SELECT count(*) FROM"):
            return self._sizes[sql.split(".")[-1].strip('"')]
        self.ran.append(sql)
        return 1

    async def fetch(self, sql: str, *args: Any) -> list[dict[str, Any]]:
        self.ran.append(sql)
        return [{"n": i} for i in range(self._rows)]

    async def execute(self, sql: str, *args: Any) -> str:
        self.ran.append(sql)
        return "OK"

    async def executemany(self, sql: str, args: Sequence[Sequence[Any]]) -> None:
        self.ran.append(sql)


class TestGuard:
    async def test_a_whole_scan_of_a_large_table_fails_and_of_a_small_one_does_not(self) -> None:
        """Flips if the guard stops reading plans, or flags a lookup table under the threshold."""
        conn = _Conn(
            {"SELECT * FROM results": _scan("Seq Scan"), "SELECT * FROM modes": _scan("Seq Scan", "modes")},
            {"results": 5000, "modes": 3},
        )
        guard = WholeScanGuard(conn, threshold_rows=1000)
        await guard.fetch("SELECT * FROM results")
        await guard.fetch("SELECT * FROM modes")
        assert conn.ran == ["SELECT * FROM results", "SELECT * FROM modes"]
        assert len(guard.problems) == 1
        assert "public.results (5000 rows" in guard.problems[0]
        with pytest.raises(AssertionError, match="1 whole read"):
            guard.check()

    async def test_an_index_walked_end_to_end_fails_and_a_filtered_seq_scan_does_too(self) -> None:
        """Flips if a filtered sequential scan, or an index scan with no index condition, stops counting as whole."""
        conn = _Conn(
            {
                "SELECT * FROM a WHERE x IS NULL": _scan("Seq Scan", "a", Filter="(x IS NULL)"),
                "SELECT * FROM b ORDER BY k": _scan("Index Scan", "b"),
            },
            {"a": 2000, "b": 2000},
        )
        guard = WholeScanGuard(conn, threshold_rows=1000)
        await guard.fetch("SELECT * FROM a WHERE x IS NULL")
        await guard.fetch("SELECT * FROM b ORDER BY k")
        assert [problem.split(" of ")[0] for problem in guard.problems] == [
            "sequential scan with a filter",
            "index scan with no index condition",
        ]

    async def test_a_keyed_read_passes(self) -> None:
        """Flips if an index lookup by key is taken for a whole read."""
        sql = "SELECT * FROM results WHERE state = $1"
        conn = _Conn({sql: _scan("Index Scan", **{"Index Cond": "(state = $1)"})}, {"results": 5000}, rows=40)
        async with WholeScanGuard(conn, threshold_rows=1000) as guard:
            await guard.fetch(sql, "TX")
        assert guard.problems == []

    async def test_keyed_pages_adding_up_to_most_of_a_table_fail(self) -> None:
        """Flips if a keyset pager walking a whole table (each page keyed) passes because no page alone is whole."""
        sql = "SELECT * FROM results WHERE k > $1 ORDER BY k LIMIT 1001"
        conn = _Conn({sql: _scan("Index Scan", **{"Index Cond": "(k > $1)"})}, {"results": 3000}, rows=1001)
        guard = WholeScanGuard(conn, threshold_rows=1000)
        for page in range(3):
            await guard.fetch(sql, page)
        assert any("read whole, a page at a time" in problem for problem in guard.problems)

    async def test_a_truncate_fails_and_a_write_is_planned_not_run_twice(self) -> None:
        """Flips if a TRUNCATE passes, or a write's plan is taken by running it."""
        update = "UPDATE stamps SET v = $1"
        conn = _Conn({update: _scan("Seq Scan", "stamps")}, {"stamps": 2000})
        guard = WholeScanGuard(conn, threshold_rows=1000)
        await guard.execute("TRUNCATE results")
        await guard.execute(update, 1)
        assert conn.ran == ["TRUNCATE results", update]
        assert [problem.split(" ", 1)[0] for problem in guard.problems] == ["TRUNCATE", "sequential"]
