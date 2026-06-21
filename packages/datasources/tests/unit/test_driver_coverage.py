"""unit tests for :meth:`Driver.column_value_coverage` (the honesty coverage probe).

the probe answers one decidable question per numeric column FROM A WAREHOUSE
SCAN: how many rows does the table have, and in how many of them is the column a
non-null, non-zero value. a column with rows but ZERO non-zero values is an
unloaded / dead column whose ``0`` must never be read as a measured zero (the
Alaska ``agg_dummy_eip`` incident -- 0 across all 181M rows for 2024, real values
in prior cycles). the driver returns the raw counts; the loaded-vs-unloaded
VERDICT is the hub reconcile's call, not the driver's.

the method is CONCRETE on the ABC (portable ``COUNT(*)`` / ``COUNT(NULLIF(col,0))``
SQL routed through :meth:`Driver.fetch`), so every backend inherits it and the
``__abstractmethods__`` contract is unchanged.
"""

from __future__ import annotations

from typing import Any

import pytest

from threetears.datasources.drivers.base import ColumnRow, Driver, TableRow


class _RecordingDriver(Driver):
    """real :class:`Driver` subclass that records the SQL and returns a seeded row.

    only ``fetch`` carries behaviour (it captures ``last_sql`` and returns the
    seeded aggregate row); every other abstract method is a trivial stub so the
    ABC instantiates.
    """

    def __init__(self, agg_row: dict[str, Any] | None = None) -> None:
        self._agg_row = agg_row or {}
        self.last_sql: str | None = None
        self.fetch_calls = 0

    async def fetch(self, sql: str, *params: Any) -> list[dict[str, Any]]:
        self.last_sql = sql
        self.fetch_calls += 1
        return [dict(self._agg_row)]

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


class TestColumnValueCoverageNotAbstract:
    """the probe is concrete: it does not enter the abstract-method set."""

    def test_not_in_abstractmethods(self) -> None:
        assert "column_value_coverage" not in Driver.__abstractmethods__


class TestColumnValueCoverageParsing:
    """the probe parses the single aggregate row into per-column counts."""

    @pytest.mark.asyncio
    async def test_entirely_zero_column_reports_zero_nonzero(self) -> None:
        # AK agg_dummy_eip: 556_559 rows, none non-zero -> unloaded signal.
        driver = _RecordingDriver({"total_rows": 556_559, "nz_0": 0})
        result = await driver.column_value_coverage(
            "reporting_prod",
            "early_vote_current_full",
            ["agg_dummy_eip"],
        )
        assert result["agg_dummy_eip"]["total_rows"] == 556_559
        assert result["agg_dummy_eip"]["nonzero_count"] == 0

    @pytest.mark.asyncio
    async def test_loaded_column_reports_nonzero_count(self) -> None:
        driver = _RecordingDriver({"total_rows": 1000, "nz_0": 731})
        result = await driver.column_value_coverage("s", "t", ["voted"])
        assert result["voted"]["total_rows"] == 1000
        assert result["voted"]["nonzero_count"] == 731

    @pytest.mark.asyncio
    async def test_multiple_columns_mapped_by_index(self) -> None:
        driver = _RecordingDriver({"total_rows": 50, "nz_0": 0, "nz_1": 12, "nz_2": 0})
        result = await driver.column_value_coverage("s", "t", ["eip", "early_vote", "vbm"])
        assert result["eip"]["nonzero_count"] == 0
        assert result["early_vote"]["nonzero_count"] == 12
        assert result["vbm"]["nonzero_count"] == 0
        assert set(result) == {"eip", "early_vote", "vbm"}

    @pytest.mark.asyncio
    async def test_zero_row_table_reports_zero_total(self) -> None:
        # an empty table is not an unloaded column -- total_rows 0 lets the
        # caller distinguish "no rows at all" from "rows but all-zero".
        driver = _RecordingDriver({"total_rows": 0, "nz_0": 0})
        result = await driver.column_value_coverage("s", "t", ["c"])
        assert result["c"]["total_rows"] == 0
        assert result["c"]["nonzero_count"] == 0


class TestColumnValueCoverageSql:
    """the built SQL is a single scan with quoted identifiers."""

    @pytest.mark.asyncio
    async def test_sql_scans_quoted_schema_table(self) -> None:
        driver = _RecordingDriver({"total_rows": 1, "nz_0": 1})
        await driver.column_value_coverage("reporting_prod", "early_vote_current_full", ["agg_dummy_eip"])
        sql = driver.last_sql or ""
        assert "COUNT(*)" in sql
        assert '"reporting_prod"."early_vote_current_full"' in sql
        assert '"agg_dummy_eip"' in sql
        assert "NULLIF" in sql

    @pytest.mark.asyncio
    async def test_identifier_with_quote_is_escaped(self) -> None:
        # a column name carrying a double-quote is escaped by doubling, never
        # interpolated raw -- the identifiers come from the catalog but the
        # quoting discipline holds regardless.
        driver = _RecordingDriver({"total_rows": 1, "nz_0": 0})
        await driver.column_value_coverage("s", 'we"ird', ['c"x'])
        sql = driver.last_sql or ""
        assert '"we""ird"' in sql
        assert '"c""x"' in sql


class TestColumnValueCoverageEmpty:
    """no numeric columns -> no query, empty result."""

    @pytest.mark.asyncio
    async def test_empty_columns_skips_query(self) -> None:
        driver = _RecordingDriver({"total_rows": 1})
        result = await driver.column_value_coverage("s", "t", [])
        assert result == {}
        assert driver.fetch_calls == 0
        assert driver.last_sql is None
