"""a bounded look at an external table: the count stops at its cap and says so, the sample at its limit, whatever
the author's predicate; both run as written on a real engine (DuckDB, standing in for a warehouse)."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import duckdb
import pytest

from threetears.datasources.bounded_look import (
    SAMPLE_LIMIT_MAX,
    BoundedCount,
    count_matching_at_most,
    sample_matching,
)

from .helpers.fake_driver import FakeDriver


class _DuckDriver(FakeDriver):
    """a driver answering from one DuckDB connection, keeping every statement sent."""

    def __init__(self, db: duckdb.DuckDBPyConnection) -> None:
        super().__init__()
        self._db = db
        self.sent: list[str] = []

    async def fetch(self, sql: str, *params: Any, timeout_seconds: int | None = None) -> list[dict[str, Any]]:
        self.sent.append(sql)
        cursor = self._db.execute(sql, list(params))
        names = [column[0] for column in cursor.description]
        return [dict(zip(names, row, strict=True)) for row in cursor.fetchall()]


@pytest.fixture
def driver() -> Iterator[_DuckDriver]:
    db = duckdb.connect()
    db.execute("CREATE SCHEMA wh")
    db.execute('CREATE TABLE wh."Geo" AS SELECT i AS id, i % 2 AS odd FROM range(2500) r(i)')
    yield _DuckDriver(db)
    db.close()


async def test_the_count_stops_at_its_cap_and_says_more(driver: _DuckDriver) -> None:
    """Flips if the count reads past its cap (no LIMIT inside), misses ``more`` at the cap, or miscounts under it."""
    assert await count_matching_at_most(driver, "wh", "Geo", None, at_most=1000) == BoundedCount(1000, True)
    assert await count_matching_at_most(driver, "wh", "Geo", "odd = 1 AND id < 100", at_most=1000) == BoundedCount(
        50, False
    )
    assert await count_matching_at_most(driver, "wh", "Geo", "  ", at_most=50) == BoundedCount(50, True)
    assert all(" LIMIT " in sql and '"wh"."Geo"' in sql for sql in driver.sent), driver.sent


async def test_the_sample_reads_at_most_its_limit(driver: _DuckDriver) -> None:
    """Flips if the sample returns more than asked, ignores the predicate, or takes a limit past its maximum."""
    rows = await sample_matching(driver, "wh", "Geo", "odd = 0", limit=3)
    assert len(rows) == 3 and all(row["odd"] == 0 for row in rows)
    assert driver.sent[-1].endswith("LIMIT 3")
    with pytest.raises(ValueError, match="at most"):
        await sample_matching(driver, "wh", "Geo", None, limit=SAMPLE_LIMIT_MAX + 1)


@pytest.mark.parametrize("bad", [0, -1, True, 1.5])
async def test_a_cap_that_is_no_positive_integer_is_refused(driver: _DuckDriver, bad: Any) -> None:
    """Flips if a cap or limit other than a positive integer reaches the statement text."""
    with pytest.raises(ValueError, match="positive integer"):
        await count_matching_at_most(driver, "wh", "Geo", None, at_most=bad)
    with pytest.raises(ValueError, match="positive integer"):
        await sample_matching(driver, "wh", "Geo", None, limit=bad)
    assert driver.sent == []
