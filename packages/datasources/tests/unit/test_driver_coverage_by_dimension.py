"""unit tests for :meth:`Driver.column_value_coverage_by_dimension`.

the grouped sibling of :meth:`Driver.column_value_coverage`: the same decidable
question (how many rows, in how many is the column non-null and non-zero) but
GROUPED BY a designated dimension column, so the caller can decide a column is
unloaded for SOME dimension values while loaded for others -- the partial-coverage
case the whole-table probe cannot see (e.g. ``agg_24_pres_total_votes`` is loaded
for 49 states but every New York county row is a placeholder zero, so a ranking
with ``WHERE col > 0`` silently omits New York and the whole-table probe never
flags it).

CONCRETE on the ABC (portable ``GROUP BY`` SQL routed through :meth:`Driver.fetch`),
so every backend inherits it and ``__abstractmethods__`` is unchanged.
"""

from __future__ import annotations

from typing import Any

import pytest

from threetears.datasources.drivers.base import ColumnRow, Driver, TableRow


class _RecordingDriver(Driver):
    """real :class:`Driver` subclass returning seeded grouped rows; records the SQL.

    ``fetch`` returns the seeded list of aggregate rows (one per dimension value)
    and captures ``last_sql``; every other abstract method is a trivial stub so the
    ABC instantiates.
    """

    def __init__(self, agg_rows: list[dict[str, Any]] | None = None) -> None:
        self._agg_rows = agg_rows or []
        self.last_sql: str | None = None
        self.fetch_calls = 0

    async def fetch(self, sql: str, *params: Any) -> list[dict[str, Any]]:
        self.last_sql = sql
        self.fetch_calls += 1
        return [dict(row) for row in self._agg_rows]

    async def execute(self, sql: str, *params: Any) -> None:
        return None

    async def list_tables(self, schemas: list[str]) -> list[TableRow]:
        return []

    async def list_columns(self, schemas: list[str]) -> list[ColumnRow]:
        return []

    async def table_hashes(self, schemas: list[str]) -> dict[tuple[str, str], str]:
        return {}

    async def test_connection(self) -> None:
        return None

    async def close(self) -> None:
        return None


class TestNotAbstract:
    """the grouped probe is concrete: it does not enter the abstract-method set."""

    def test_not_in_abstractmethods(self) -> None:
        assert "column_value_coverage_by_dimension" not in Driver.__abstractmethods__


class TestGroupedParsing:
    """the probe parses the grouped rows into per-column, per-dimension counts."""

    @pytest.mark.asyncio
    async def test_partial_coverage_per_dimension(self) -> None:
        # agg_24_pres_total_votes: loaded for CA (731 of 1000), all-zero for NY
        # (0 of 62) -- the partial-region case.
        driver = _RecordingDriver(
            [
                {"dim_value": "CA", "total_rows": 1000, "nz_0": 731},
                {"dim_value": "NY", "total_rows": 62, "nz_0": 0},
            ]
        )
        result = await driver.column_value_coverage_by_dimension(
            "reporting_prod",
            "report_geofacts_past_elections",
            "state_code",
            ["agg_24_pres_total_votes"],
        )
        col = result["agg_24_pres_total_votes"]
        assert col["CA"]["total_rows"] == 1000
        assert col["CA"]["nonzero_count"] == 731
        assert col["NY"]["total_rows"] == 62
        assert col["NY"]["nonzero_count"] == 0

    @pytest.mark.asyncio
    async def test_multiple_columns_mapped_by_index_per_dimension(self) -> None:
        driver = _RecordingDriver(
            [
                {"dim_value": "CA", "total_rows": 50, "nz_0": 40, "nz_1": 0},
                {"dim_value": "NY", "total_rows": 30, "nz_0": 0, "nz_1": 0},
            ]
        )
        result = await driver.column_value_coverage_by_dimension("s", "t", "state_code", ["votes", "eip"])
        assert result["votes"]["CA"]["nonzero_count"] == 40
        assert result["votes"]["NY"]["nonzero_count"] == 0
        assert result["eip"]["CA"]["nonzero_count"] == 0
        assert result["eip"]["NY"]["nonzero_count"] == 0
        assert set(result) == {"votes", "eip"}

    @pytest.mark.asyncio
    async def test_null_dimension_value_maps_to_empty_string(self) -> None:
        # a NULL group (rows with no dimension value) keys on '' rather than
        # crashing on None.
        driver = _RecordingDriver([{"dim_value": None, "total_rows": 5, "nz_0": 0}])
        result = await driver.column_value_coverage_by_dimension("s", "t", "state_code", ["c"])
        assert result["c"][""]["total_rows"] == 5

    @pytest.mark.asyncio
    async def test_every_requested_column_has_an_entry(self) -> None:
        # even with no rows returned, each requested column maps to an (empty)
        # per-dimension dict so callers can index result[col] safely.
        driver = _RecordingDriver([])
        result = await driver.column_value_coverage_by_dimension("s", "t", "state_code", ["a", "b"])
        assert result == {"a": {}, "b": {}}


class TestGroupedSql:
    """the built SQL is a single grouped scan with quoted identifiers."""

    @pytest.mark.asyncio
    async def test_sql_groups_by_quoted_dimension(self) -> None:
        driver = _RecordingDriver([{"dim_value": "CA", "total_rows": 1, "nz_0": 1}])
        await driver.column_value_coverage_by_dimension(
            "reporting_prod", "report_geofacts_past_elections", "state_code", ["agg_24_pres_total_votes"]
        )
        sql = driver.last_sql or ""
        assert "COUNT(*)" in sql
        assert '"reporting_prod"."report_geofacts_past_elections"' in sql
        assert '"agg_24_pres_total_votes"' in sql
        assert "NULLIF" in sql
        assert '"state_code" AS dim_value' in sql
        assert 'GROUP BY "state_code"' in sql

    @pytest.mark.asyncio
    async def test_identifier_with_quote_is_escaped(self) -> None:
        driver = _RecordingDriver([{"dim_value": "x", "total_rows": 1, "nz_0": 0}])
        await driver.column_value_coverage_by_dimension("s", 'we"ird', 'di"m', ['c"x'])
        sql = driver.last_sql or ""
        assert '"we""ird"' in sql
        assert '"di""m"' in sql
        assert '"c""x"' in sql


class TestGroupedEmpty:
    """no columns OR no dimension column -> no query, empty result."""

    @pytest.mark.asyncio
    async def test_empty_columns_skips_query(self) -> None:
        driver = _RecordingDriver([{"dim_value": "CA", "total_rows": 1}])
        result = await driver.column_value_coverage_by_dimension("s", "t", "state_code", [])
        assert result == {}
        assert driver.fetch_calls == 0
        assert driver.last_sql is None

    @pytest.mark.asyncio
    async def test_empty_dimension_column_skips_query(self) -> None:
        driver = _RecordingDriver([{"dim_value": "CA", "total_rows": 1, "nz_0": 1}])
        result = await driver.column_value_coverage_by_dimension("s", "t", "", ["c"])
        assert result == {}
        assert driver.fetch_calls == 0
        assert driver.last_sql is None
