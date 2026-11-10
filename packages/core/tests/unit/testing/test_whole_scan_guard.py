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


class _Analysing(_Conn):
    """a connection that also answers ``EXPLAIN (ANALYZE ...)`` with a given analysed plan, inside a transaction it
    rolls back, and says which database it is."""

    def __init__(
        self,
        plans: Mapping[str, Mapping[str, Any]],
        analysed: Mapping[str, Mapping[str, Any]],
        sizes: Mapping[str, int],
    ) -> None:
        super().__init__(plans, sizes)
        self._analysed = analysed
        self.analysed: list[str] = []
        self.rolled_back = 0

    async def fetchval(self, sql: str, *args: Any) -> Any:
        if sql == "SELECT version()":
            return "PostgreSQL 16.4"
        if sql.startswith("EXPLAIN (ANALYZE, FORMAT JSON) "):
            statement = sql.removeprefix("EXPLAIN (ANALYZE, FORMAT JSON) ")
            self.analysed.append(statement)
            return json.dumps([{"Plan": self._analysed[statement], "Execution Time": 0.1}])
        return await super().fetchval(sql, *args)

    def transaction(self) -> Any:
        conn = self

        class _Transaction:
            async def start(self) -> None:
                return None

            async def rollback(self) -> None:
                conn.rolled_back += 1

        return _Transaction()


def _limit(child: Mapping[str, Any], **actual: object) -> dict[str, Any]:
    return {"Node Type": "Limit", "Plans": [child], **actual}


class TestAWalkALimitStops:
    _PAGE = "SELECT * FROM shares ORDER BY share_id LIMIT 51"

    async def test_an_index_walk_its_limit_stopped_is_a_page(self) -> None:
        """Flips if a keyset listing's first page (an index walked in order, no index condition, stopped by its
        Limit after the rows it answered) fails as a whole read, or is judged without its actual rows read, or the
        analysis is not rolled back, or the statement runs other than once."""
        walk = _scan("Index Scan", "shares", **{"Index Name": "shares_pkey"})
        conn = _Analysing(
            {self._PAGE: _limit(walk)},
            {self._PAGE: _limit({**walk, "Actual Rows": 51, "Actual Loops": 1}, **{"Actual Rows": 51})},
            {"shares": 5000},
        )
        guard = WholeScanGuard(conn, threshold_rows=1000)
        await guard.fetch(self._PAGE)
        assert guard.problems == []
        assert conn.analysed == [self._PAGE]
        assert conn.rolled_back == 1
        assert conn.ran == [self._PAGE]

    async def test_an_index_walk_its_filter_drove_past_its_limit_stays_whole(self) -> None:
        """Flips if an index walk under a Limit whose filter made it read far past the rows answered (the page's rows
        rare in the index's order) passes as a page."""
        filtered = "SELECT * FROM shares WHERE kind = $1 ORDER BY share_id LIMIT 51"
        walk = _scan("Index Scan", "shares", Filter="(kind = $1)", **{"Index Name": "shares_pkey"})
        analysed = {**walk, "Actual Rows": 51, "Actual Loops": 1, "Rows Removed by Filter": 4200}
        conn = _Analysing(
            {filtered: _limit(walk)}, {filtered: _limit(analysed, **{"Actual Rows": 51})}, {"shares": 5000}
        )
        guard = WholeScanGuard(conn, threshold_rows=1000)
        await guard.fetch(filtered, "rare")
        assert [problem.split(" of ")[0] for problem in guard.problems] == ["index scan with no index condition"]

    async def test_a_write_s_walk_under_a_limit_is_never_run_to_judge_it(self) -> None:
        """Flips if a write whose index walk lies under a Limit is EXPLAIN ANALYZE-d (it would run twice) or passes
        unjudged."""
        trim = "DELETE FROM shares WHERE share_id IN (SELECT share_id FROM shares ORDER BY share_id LIMIT 10)"
        walk = _scan("Index Only Scan", "shares", **{"Index Name": "shares_pkey"})
        conn = _Analysing({trim: {"Node Type": "ModifyTable", "Plans": [_limit(walk)]}}, {}, {"shares": 5000})
        guard = WholeScanGuard(conn, threshold_rows=1000)
        await guard.execute(trim)
        assert conn.analysed == []
        assert [problem.split(" of ")[0] for problem in guard.problems] == ["index only scan with no index condition"]
