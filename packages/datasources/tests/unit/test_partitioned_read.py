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

from threetears.datasources.partitioned_read import DEFAULT_PART_CONCURRENCY, fingerprint_parts, read_parts
from threetears.datasources.query_client import (
    DatasourceQueryError,
    DatasourceQueryResult,
    IncompleteReadError,
    RelationFingerprintResult,
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
