"""On PostgreSQL L3 a keyed count of a set of values seeks each value apart; elsewhere it stays one ``= ANY``.

PostgreSQL 16 planned ``geo = ANY($2)`` beside the pinned state as a near tie between the index condition holding the
set and one holding the state alone (the set rechecked on every row of it), and took the second on 15 of 40
``ANALYZE`` samples: the state's 6,000 rows read to count the 3,000 of 100 geographies (the live
``test_counts_by_counts_only_its_keys_rows``). A seek a value (``unnest`` and a ``LATERAL`` count) reads only the
values' rows on every sample; YugabyteDB, which hands ``= ANY`` to storage in one request, keeps it (a seek a value
there took 6.9 ms against 1.0 ms), as does a warehouse (Redshift has no ``LATERAL``).
"""

from __future__ import annotations

from typing import Any

from threetears.core.keyed import AnyOf, Dialect, counts_by


class _Reader:
    """a reader that records each statement and answers its server's version, and the counts it is given."""

    def __init__(self, version: str, rows: list[dict[str, Any]]) -> None:
        self.version = version
        self.rows = rows
        self.asked: list[tuple[str, tuple[Any, ...]]] = []
        self.probed = 0

    async def fetch(self, query: str, *params: Any) -> list[dict[str, Any]]:
        if query.startswith("SELECT version()"):
            self.probed += 1
            return [{"v": self.version}]
        self.asked.append((query, params))
        return self.rows


class _Warehouse(_Reader):
    sql_dialect = Dialect.REDSHIFT


_ROWS = [{"relation": 0, "counted": "001", "held": 30}, {"relation": 0, "counted": "002", "held": 30}]


async def test_postgres_seeks_each_value() -> None:
    """Flips if a count of a set on PostgreSQL is not one seek a value (the set unnested, each value's count a lateral
    read pinned by the prefix and the value), or answers other counts than the rows say."""
    reader = _Reader("PostgreSQL 16.4", _ROWS)
    counted = await counts_by(reader, "results", column="geo", values=AnyOf(["001", "002", "001"]), by={"state": "S20"})
    assert counted == {"001": 30, "002": 30}
    ((sql, params),) = reader.asked
    assert sql == (
        "SELECT 0 AS relation, counted_keys.counted_key AS counted, c.held FROM unnest($2::text[]) AS "
        'counted_keys(counted_key) CROSS JOIN LATERAL (SELECT count(*) AS held FROM "results" AS r WHERE '
        'r."state" = $1 AND r."geo" = counted_keys.counted_key) AS c WHERE c.held > 0'
    ), sql
    assert params == ("S20", ["001", "002"])


async def test_yugabyte_and_a_warehouse_keep_one_set() -> None:
    """Flips if YugabyteDB or a warehouse is sent anything but the one ``= ANY`` count."""
    for reader in (_Reader("PostgreSQL 15.12-YB-2026.1.0.0-b0", _ROWS), _Warehouse("Redshift", _ROWS)):
        await counts_by(reader, "results", column="geo", values=AnyOf(["001", "002"]), by={"state": "S20"})
        ((sql, _params),) = reader.asked
        assert "LATERAL" not in sql and '"geo" = ANY(' in sql, sql


async def test_a_count_with_no_pinned_prefix_asks_nothing_and_stays_one_set() -> None:
    """Flips if a count with no pinned prefix (the set the only index condition: no near tie to break) asks the server
    its version or is sent other than the one ``= ANY`` count."""
    reader = _Reader("PostgreSQL 16.4", _ROWS)
    await counts_by(reader, "results", column="geo", values=AnyOf(["001", "002"]))
    ((sql, _params),) = reader.asked
    assert "LATERAL" not in sql and '"geo" = ANY(' in sql, sql
    assert reader.probed == 0
