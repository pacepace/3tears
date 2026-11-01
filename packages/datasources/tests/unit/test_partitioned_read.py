"""Reading a relation by parts: each part fingerprinted or read whole, several at once, never more than asked.

The fake answers after a pause and counts how many requests are in flight at once, so a helper that
awaited the parts one after another (in flight never above one) or launched them all together (in
flight equal to the number of parts) fails here, and so does one that returned the parts out of order.
"""

from __future__ import annotations

import asyncio
import hashlib
from typing import Any
from uuid import uuid7

import pytest

from threetears.datasources.partitioned_read import (
    DEFAULT_PART_CONCURRENCY,
    fingerprint_parts,
    read_partitions,
    read_parts,
)
from threetears.datasources.query_client import (
    DatasourceQueryError,
    DatasourceQueryResult,
    IncompleteReadError,
    RelationFingerprintResult,
    keyset_condition,
)

_STATES = [f"S{index:02d}" for index in range(12)]


def _rows() -> list[dict[str, Any]]:
    return [{"state": state, "race": f"{state}-r{n}", "votes": n * 10} for state in _STATES for n in range(3)]


class _Warehouse:
    """a relation partitioned by ``state``; every request waits, so requests overlap if sent together."""

    def __init__(self, rows: list[dict[str, Any]], *, fail_state: str | None = None) -> None:
        self.rows = rows
        self.fail_state = fail_state
        self.in_flight = 0
        self.most_in_flight = 0
        self.asked: list[dict[str, str]] = []
        self.finished = 0

    def forwarded_identity_token(self) -> str:
        return "fake-identity-token"

    def _enter(self) -> None:
        self.in_flight += 1
        self.most_in_flight = max(self.most_in_flight, self.in_flight)

    def _leave(self) -> None:
        self.in_flight -= 1
        self.finished += 1

    def _part(self, where: dict[str, str]) -> list[dict[str, Any]]:
        return [r for r in self.rows if all(str(r[c]) == v for c, v in where.items())]

    async def relation_fingerprint(
        self, datasource_name: str, *, relation: str, key: Any, where: Any = None, **_: Any
    ) -> RelationFingerprintResult:
        filters = dict(where or {})
        self.asked.append(filters)
        self._enter()
        try:
            await asyncio.sleep(0.01)
            if filters.get("state") == self.fail_state:
                raise DatasourceQueryError("QUERY_EXECUTION_ERROR", "the warehouse refused the statement")
            kept = self._part(filters)
            payload = "\x1e".join("\x1f".join(str(row[c]) for c in key) for row in kept)
            return RelationFingerprintResult(row_count=len(kept), digest=hashlib.sha256(payload.encode()).hexdigest())
        finally:
            self._leave()

    async def query(
        self, datasource_name: str, query: str, *, params: list[Any] | None = None, **_: Any
    ) -> DatasourceQueryResult:
        bound = list(params or [])
        self._enter()
        try:
            await asyncio.sleep(0.01)
            kept = self._part({"state": bound[0]})
            if len(bound) > 1:
                kept = [r for r in kept if (r["race"],) > (bound[-1],)]
            kept = sorted(kept, key=lambda r: r["race"])
            limit = int(query.rsplit("LIMIT", 1)[1])
            served = kept[:limit]
            return DatasourceQueryResult(rows=served, row_count=len(served), truncated=False, correlation_id=uuid7())
        finally:
            self._leave()


async def test_each_part_is_fingerprinted_over_the_columns_named_in_the_order_given() -> None:
    warehouse = _Warehouse(_rows())
    parts = [{"state": state} for state in _STATES]

    results = await fingerprint_parts(
        warehouse,  # type: ignore[arg-type]
        "warehouse",
        relation="s.results",
        columns=["state", "race", "votes"],
        parts=parts,
        concurrency=4,
    )

    assert [r.row_count for r in results] == [3] * len(_STATES)
    assert len({r.digest for r in results}) == len(_STATES)
    assert sorted(map(str, warehouse.asked)) == sorted(map(str, parts))


async def test_parts_run_several_at_once_and_never_more_than_asked() -> None:
    warehouse = _Warehouse(_rows())

    await fingerprint_parts(
        warehouse,  # type: ignore[arg-type]
        "warehouse",
        relation="s.results",
        columns=["race"],
        parts=[{"state": state} for state in _STATES],
        concurrency=4,
    )

    assert warehouse.most_in_flight == 4


async def test_a_value_change_moves_only_its_parts_fingerprint() -> None:
    rows = _rows()
    warehouse = _Warehouse(rows)
    parts = [{"state": state} for state in _STATES]
    columns = ["state", "race", "votes"]
    before = await fingerprint_parts(warehouse, "w", relation="s.r", columns=columns, parts=parts)  # type: ignore[arg-type]

    rows[4]["votes"] += 1  # a row of the second state
    after = await fingerprint_parts(warehouse, "w", relation="s.r", columns=columns, parts=parts)  # type: ignore[arg-type]

    moved = [part["state"] for part, old, new in zip(parts, before, after, strict=True) if old != new]
    assert moved == [_STATES[1]]


async def test_each_part_is_read_whole_and_returned_in_the_order_given() -> None:
    warehouse = _Warehouse(_rows())
    wanted = list(reversed(_STATES[:6]))

    parts = await read_parts(
        warehouse,  # type: ignore[arg-type]
        "warehouse",
        columns=["state", "race", "votes"],
        relation="s.results",
        key=["race"],
        parts=[{"state": state} for state in wanted],
        concurrency=3,
        page_size=2,
    )

    assert [{row["state"] for row in part} for part in parts] == [{state} for state in wanted]
    assert all(len(part) == 3 for part in parts)
    assert warehouse.most_in_flight == 3


async def test_a_failing_part_fails_the_read_and_stops_the_parts_not_yet_finished() -> None:
    warehouse = _Warehouse(_rows(), fail_state=_STATES[0])

    with pytest.raises(DatasourceQueryError, match="refused"):
        await fingerprint_parts(
            warehouse,  # type: ignore[arg-type]
            "warehouse",
            relation="s.results",
            columns=["race"],
            parts=[{"state": state} for state in _STATES],
            concurrency=2,
        )

    await asyncio.sleep(0.05)
    assert warehouse.in_flight == 0
    assert warehouse.finished < len(_STATES), "the parts after the failure all ran anyway"


async def test_an_incomplete_part_is_raised_as_itself() -> None:
    rows = _rows()
    warehouse = _Warehouse(rows)
    original = warehouse.relation_fingerprint
    calls = 0

    async def drops_a_row_between_readings(*args: Any, **kwargs: Any) -> RelationFingerprintResult:
        nonlocal calls
        calls += 1
        result = await original(*args, **kwargs)
        if calls == 1:
            rows.pop(0)  # the part changed after its first reading
        return result

    warehouse.relation_fingerprint = drops_a_row_between_readings  # type: ignore[method-assign]

    with pytest.raises(IncompleteReadError):
        await read_parts(
            warehouse,  # type: ignore[arg-type]
            "warehouse",
            columns=["state", "race"],
            relation="s.results",
            key=["race"],
            parts=[{"state": _STATES[0]}],
        )


def test_the_default_concurrency_is_the_hubs_own_bound() -> None:
    # the hub answers a datasource with at most this many open warehouse connections; asking for
    # more only queues at the hub
    assert DEFAULT_PART_CONCURRENCY == 5


async def test_a_concurrency_below_one_is_refused() -> None:
    with pytest.raises(ValueError, match="concurrency"):
        await fingerprint_parts(
            _Warehouse(_rows()),  # type: ignore[arg-type]
            "w",
            relation="s.r",
            columns=["race"],
            parts=[{"state": "S00"}],
            concurrency=0,
        )


class TestOneKeysetBuilder:
    """``read_all``'s next page and a partitioned read's page bounds come from the one keyset builder."""

    def test_after_a_key_is_strictly_after_it_numbered_from_the_first_placeholder(self) -> None:
        assert keyset_condition(["a", "b"], [1, 2], earlier=">", last=">", first=3) == (
            "(a > $3) OR (a = $4 AND b > $5)",
            [1, 1, 2],
        )

    def test_a_page_bound_is_at_or_above_its_start_and_below_the_next(self) -> None:
        assert keyset_condition(["a", "b"], ["x", 5], earlier=">", last=">=") == (
            "(a > $1) OR (a = $2 AND b >= $3)",
            ["x", "x", 5],
        )
        assert keyset_condition(["a"], ["y"], earlier="<", last="<", first=4) == ("(a < $4)", ["y"])

    async def test_a_partitioned_reads_pages_bind_every_bound_in_placeholder_order(self) -> None:
        rows = [{"state": "VA", "race": race, "n": n} for race, n in (("r1", 1), ("r1", 2), ("r2", 1))]
        pages: list[tuple[str, list[Any]]] = []

        # parity-exempt: the read rail as a partitioned read uses it, answering page bounds from real rows
        class _Rail:
            def forwarded_identity_token(self) -> str:
                return "fake-identity-token"

            async def relation_fingerprint_groups(
                self, *_: Any, **__: Any
            ) -> dict[str | None, RelationFingerprintResult]:
                return {"VA": RelationFingerprintResult(row_count=3, digest="d")}

            async def query(self, datasource: str, sql: str, params: list[Any] | None = None, **_: Any) -> Any:
                bound = list(params or [])
                if "ROW_NUMBER()" in sql:
                    starts = [{"part__": "VA", "race": "r1", "n": 1}, {"part__": "VA", "race": "r2", "n": 1}]
                    return DatasourceQueryResult(rows=starts, row_count=2, truncated=False, correlation_id=uuid7())
                pages.append((sql, bound))
                # (race, n) >= its start, and < the next start when the SQL names one
                lower = (bound[2], bound[3])
                upper = (bound[5], bound[6]) if len(bound) > 4 else None
                found = [
                    r for r in rows if lower <= (r["race"], r["n"]) and (upper is None or (r["race"], r["n"]) < upper)
                ]
                return DatasourceQueryResult(rows=found, row_count=len(found), truncated=False, correlation_id=uuid7())

        read = [
            part
            async for part in read_partitions(
                _Rail(),  # type: ignore[arg-type]
                "w",
                columns=["state", "race", "n"],
                relation="s.r",
                key=["race", "n"],
                partition_by="state",
                page_size=2,
            )
        ]

        assert read == [("VA", rows)]
        first, second = sorted(pages, key=lambda page: len(page[1]), reverse=True)
        assert (
            "WHERE state = $1 AND ((race > $2) OR (race = $3 AND n >= $4)) AND ((race < $5) OR (race = $6 AND n < $7))"
            in first[0]
        )
        assert first[1] == ["VA", "r1", "r1", 1, "r2", "r2", 1]
        assert "WHERE state = $1 AND ((race > $2) OR (race = $3 AND n >= $4)) ORDER BY" in second[0]
        assert second[1] == ["VA", "r2", "r2", 1]


# parity-exempt: the read rail as a partitioned read uses it, over rows a test rewrites between pages
class _SettlingRail:
    """the three asks a partitioned read makes, answered from ``rows``; ``after_page`` runs after each page read."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.after_page: Any = None
        self.page_starts_asked = 0
        self.refuse_pages = False

    def forwarded_identity_token(self) -> str:
        return "fake-identity-token"

    def part_rows(self, state: str) -> list[dict[str, Any]]:
        return sorted((r for r in self.rows if r["state"] == state), key=lambda r: r["race"])

    async def relation_fingerprint_groups(
        self, datasource_name: str, *, relation: str, key: Any, group_by: str, where_in: Any = None, **_: Any
    ) -> dict[str | None, RelationFingerprintResult]:
        wanted = set(where_in[group_by]) if where_in else {r["state"] for r in self.rows}
        found: dict[str | None, RelationFingerprintResult] = {}
        for state in sorted(wanted):
            kept = self.part_rows(state)
            if kept:
                payload = "\x1e".join("\x1f".join(str(row[c]) for c in key) for row in kept)
                found[state] = RelationFingerprintResult(
                    row_count=len(kept), digest=hashlib.sha256(payload.encode()).hexdigest()
                )
        return found

    async def query(
        self, datasource_name: str, sql: str, *, params: list[Any] | None = None, **_: Any
    ) -> DatasourceQueryResult:
        bound = list(params or [])
        if "ROW_NUMBER()" in sql:
            self.page_starts_asked += 1
            size = int(sql.split("MOD(row__ - 1, ")[1].split(")")[0])
            starts = [
                {"part__": state, "race": row["race"]}
                for state in bound
                for index, row in enumerate(self.part_rows(state))
                if index % size == 0
            ]
            return DatasourceQueryResult(rows=starts, row_count=len(starts), truncated=False, correlation_id=uuid7())
        if self.refuse_pages:
            raise DatasourceQueryError("QUERY_EXECUTION_ERROR", "the warehouse refused the statement")
        # race >= its page's start, and < the next page's start when the SQL names one
        lower, upper = bound[1], (bound[2] if len(bound) > 2 else None)
        found = [r for r in self.part_rows(bound[0]) if lower <= r["race"] and (upper is None or r["race"] < upper)]
        found = found[: int(sql.rsplit("LIMIT", 1)[1])]
        if self.after_page is not None:
            self.after_page()
        return DatasourceQueryResult(rows=found, row_count=len(found), truncated=False, correlation_id=uuid7())


async def _read_all_partitions(rail: _SettlingRail, read: list[tuple[str, list[dict[str, Any]]]]) -> None:
    async for part in read_partitions(
        rail,  # type: ignore[arg-type]
        "w",
        columns=["state", "race", "votes"],
        relation="s.r",
        key=["race"],
        partition_by="state",
        page_size=2,
    ):
        read.append(part)


class TestAPartChangedWhileItWasRead:
    """a batch whose part a source rewrote mid-read is read again, a bounded number of times, and only then."""

    async def test_a_part_rewritten_once_mid_read_is_read_again_and_comes_back_whole(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        rows = _rows()
        rail = _SettlingRail(rows)

        def one_upload() -> None:
            # an upload lands while the first read is under way: a value moves and a row is added
            rows[0]["votes"] += 1
            rows.append({"state": _STATES[0], "race": f"{_STATES[0]}-r9", "votes": 90})
            rail.after_page = None

        rail.after_page = one_upload
        read: list[tuple[str, list[dict[str, Any]]]] = []
        with caplog.at_level("WARNING"):
            await _read_all_partitions(rail, read)

        assert dict(read) == {state: rail.part_rows(state) for state in _STATES}
        assert len(dict(read)[_STATES[0]]) == 4
        assert rail.page_starts_asked == 2
        retries = [r for r in caplog.records if r.levelname == "WARNING"]
        assert [getattr(r, "extra_data", {}).get("attempt") for r in retries] == [1]
        assert getattr(retries[0], "extra_data", {}).get("part") == _STATES[0]
        assert not [r for r in caplog.records if r.levelname == "ERROR"]

    async def test_a_part_that_never_settles_fails_the_read_as_before_and_hands_back_none_of_its_batch(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        rows = _rows()
        rail = _SettlingRail(rows)

        def keeps_writing() -> None:
            rows[0]["votes"] += 1

        rail.after_page = keeps_writing
        read: list[tuple[str, list[dict[str, Any]]]] = []
        with caplog.at_level("WARNING"), pytest.raises(IncompleteReadError, match="changed while it was read"):
            await _read_all_partitions(rail, read)

        assert read == [], "a part of the batch that never settled was handed back"
        assert rail.page_starts_asked == 3
        assert [getattr(r, "extra_data", {}).get("attempt") for r in caplog.records if r.levelname == "WARNING"] == [
            1,
            2,
        ]
        errors = [r for r in caplog.records if r.levelname == "ERROR"]
        assert len(errors) == 1
        assert getattr(errors[0], "extra_data", {}).get("attempts") == 3

    async def test_a_refused_statement_is_not_read_again(self, caplog: pytest.LogCaptureFixture) -> None:
        rail = _SettlingRail(_rows())
        rail.refuse_pages = True

        with caplog.at_level("WARNING"), pytest.raises(DatasourceQueryError, match="refused"):
            await _read_all_partitions(rail, [])

        assert rail.page_starts_asked == 1
        assert not [r for r in caplog.records if r.levelname in {"WARNING", "ERROR"}]

    async def test_a_part_short_of_its_count_with_its_fingerprint_unmoved_is_not_read_again(self) -> None:
        rail = _SettlingRail(_rows())
        original = rail.query

        async def drops_a_row(datasource_name: str, sql: str, **kwargs: Any) -> DatasourceQueryResult:
            result = await original(datasource_name, sql, **kwargs)
            if "ROW_NUMBER()" not in sql and result.rows and result.rows[0]["race"] == f"{_STATES[0]}-r0":
                served = result.rows[1:]
                return DatasourceQueryResult(
                    rows=served, row_count=len(served), truncated=False, correlation_id=uuid7()
                )
            return result

        rail.query = drops_a_row  # type: ignore[method-assign]

        with pytest.raises(IncompleteReadError, match="counted 3 rows and the read returned 2"):
            await _read_all_partitions(rail, [])

        assert rail.page_starts_asked == 1
