"""``warehouse_arrow_types`` against a real PostgreSQL, the warehouse's integration stand-in: a table's and a view's
columns read from the catalogue the dialect names (``information_schema.columns``), in ordinal order, each as its JSON
form on the rail. A client here answers the hub's ask through the asyncpg driver itself.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any
from urllib.parse import urlsplit
from uuid import uuid7

import asyncpg
import pyarrow as pa
import pytest

from threetears.core.keyed import Dialect
from threetears.datasources.arrow_rows import warehouse_arrow_types
from threetears.datasources.config import PostgresConnectionConfig
from threetears.datasources.drivers.asyncpg_driver import AsyncpgDriver
from threetears.datasources.entities import DataSourceType
from threetears.datasources.query_client import DatasourceQueryResult

pytestmark = pytest.mark.integration

_SCHEMA = "ds_it_arrow_types"

#: each column's Arrow type as the rail's JSON reply carries it, in the table's order
_TYPES = {
    "race": pa.string(),
    "votes": pa.int64(),
    "margin": pa.string(),
    "share": pa.float64(),
    "called": pa.bool_(),
    "called_at": pa.string(),
    "polls_close": pa.string(),
}


class _Client:
    """the hub's query ask, answered by the driver."""

    def __init__(self, driver: AsyncpgDriver) -> None:
        self._driver = driver
        self.queries: list[str] = []

    async def query(self, datasource_name: str, sql: str, params: Sequence[Any] = ()) -> DatasourceQueryResult:
        self.queries.append(sql)
        rows = await self._driver.fetch(sql, *params)
        return DatasourceQueryResult(rows=rows, row_count=len(rows), truncated=False, correlation_id=uuid7())


@pytest.fixture
async def warehouse(db_container: str, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[_Client]:
    parts = urlsplit(db_container)
    host, port = parts.hostname or "localhost", parts.port or 5432
    database, user = (parts.path or "/postgres").lstrip("/"), parts.username or "postgres"
    conn = await asyncpg.connect(host=host, port=port, database=database, user=user, password=parts.password or "")
    await conn.execute(f"DROP SCHEMA IF EXISTS {_SCHEMA} CASCADE")
    await conn.execute(f"CREATE SCHEMA {_SCHEMA}")
    await conn.execute(
        f"CREATE TABLE {_SCHEMA}.results_20261119 (race varchar(20), votes integer, margin numeric(10,2), "
        "share double precision, called boolean, called_at timestamptz, polls_close timetz, payload jsonb)"
    )
    await conn.execute(f"CREATE VIEW {_SCHEMA}.results AS SELECT * FROM {_SCHEMA}.results_20261119")
    monkeypatch.setenv("ARROW_TYPES_TEST_PW", parts.password or "")
    driver = AsyncpgDriver(
        PostgresConnectionConfig(
            datasource_type=DataSourceType.POSTGRES,
            host=host,
            port=port,
            database=database,
            username=user,
            password_ref="env://ARROW_TYPES_TEST_PW",
            pool_min_size=1,
            pool_max_size=2,
            command_timeout_seconds=10,
            allowed_schemas=[],
        )
    )
    try:
        yield _Client(driver)
    finally:
        await driver.close()
        await conn.execute(f"DROP SCHEMA IF EXISTS {_SCHEMA} CASCADE")
        await conn.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("relation", ["results_20261119", "results"])
async def test_a_postgres_relation_s_columns_are_read_from_information_schema(
    warehouse: _Client, relation: str
) -> None:
    """Flips if a PostgreSQL table's or view's columns are not read from ``information_schema.columns`` (PostgreSQL
    holds no ``SVV_COLUMNS``), in ordinal order, each as its JSON form; a ``jsonb`` column, of no one form, left out."""
    every = await warehouse_arrow_types(
        warehouse,  # type: ignore[arg-type]
        "w",
        f"{_SCHEMA}.{relation}",
        None,
        dialect=Dialect.POSTGRES,
    )
    named = await warehouse_arrow_types(
        warehouse,  # type: ignore[arg-type]
        "w",
        f"{_SCHEMA}.{relation}",
        ["polls_close", "race"],
        dialect=Dialect.POSTGRES,
    )

    assert every == _TYPES
    assert list(every) == list(_TYPES)
    assert named == {"polls_close": pa.string(), "race": pa.string()}
    assert all("information_schema.columns" in query for query in warehouse.queries)
