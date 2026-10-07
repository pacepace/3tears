"""What a caller running many queries at once must know of a driver: how many it may run, and whose pool it draws on.

The hub answers datasource queries side by side. A driver that bounds its own warehouse connections
(Redshift, a Postgres pool it owns) needs nothing more; one whose logins are not guarded against a
refused credential (Snowflake, BigQuery) must be asked one query at a time, or a wrong credential
fails a burst of logins at once; one that borrows the host's own pool (agent_internal) must be
bounded with every other driver borrowing it, or it starves the host.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

from threetears.datasources.config import (
    BigQueryConnectionConfig,
    BorrowedPoolConnectionConfig,
    PostgresConnectionConfig,
    RedshiftConnectionConfig,
    SnowflakeConnectionConfig,
)
from threetears.datasources.drivers.asyncpg_driver import AsyncpgDriver
from threetears.datasources.drivers.base import Driver
from threetears.datasources.drivers.bigquery_driver import BigQueryDriver
from threetears.datasources.drivers.redshift_driver import RedshiftDriver
from threetears.datasources.drivers.snowflake_driver import SnowflakeDriver
from threetears.datasources.entities import DataSourceType


def _postgres() -> PostgresConnectionConfig:
    return PostgresConnectionConfig(
        datasource_type=DataSourceType.POSTGRES, host="localhost", database="x", password_ref="env://ABSENT_PW"
    )


def test_a_driver_that_caps_its_own_connections_states_that_cap_and_borrows_nothing() -> None:
    """a caller gating every datasource gates this one at the cap it already keeps, so a query waits at
    the caller's gate (which refuses busy, with a deadline) and never on the driver's own semaphore."""
    postgres = PostgresConnectionConfig(
        datasource_type=DataSourceType.POSTGRES,
        host="localhost",
        database="x",
        password_ref="env://ABSENT_PW",
        pool_max_size=7,
    )
    redshift = RedshiftConnectionConfig(
        datasource_type=DataSourceType.REDSHIFT,
        host="rs.example.com",
        database="analytics",
        username="u",
        password_ref="env://ABSENT_PW",
        executor_max_workers=3,
        connection_cache_size=3,
    )
    drivers: list[Any] = [AsyncpgDriver(postgres), RedshiftDriver(redshift)]
    assert [(d.concurrent_queries, d.borrowed_pool) for d in drivers] == [(7, None), (3, None)]


def test_a_driver_whose_logins_are_not_guarded_takes_one_query_at_a_time() -> None:
    snowflake = SnowflakeDriver(
        SnowflakeConnectionConfig(
            datasource_type=DataSourceType.SNOWFLAKE, account="a", warehouse="w", user="u", password_ref="env://X"
        )
    )
    bigquery = BigQueryDriver(
        BigQueryConnectionConfig(
            datasource_type=DataSourceType.BIGQUERY, project_id="p", credentials_json_ref="env://X"
        )
    )
    assert (snowflake.concurrent_queries, bigquery.concurrent_queries) == (1, 1)


def test_a_driver_borrowing_the_hosts_pool_names_it() -> None:
    pool = MagicMock(name="hub-l3-pool")
    driver = AsyncpgDriver(
        BorrowedPoolConnectionConfig(datasource_type=DataSourceType.AGENT_INTERNAL, schema_name="agent_abc"),
        external_pool=pool,
    )
    assert driver.borrowed_pool is pool
    # bounded with every other borrower of the pool, not on its own
    assert driver.concurrent_queries is None


async def test_a_driver_that_cannot_stop_early_still_answers_no_more_than_asked() -> None:
    """the default for a driver with no early stop: it reads what it reads, and keeps the bound."""

    class _ReadsEverything(RedshiftDriver):
        async def fetch(self, sql: str, *params: Any, timeout_seconds: int | None = None) -> list[dict[str, Any]]:
            return [{"n": index} for index in range(10)]

    driver = Driver.fetch_at_most  # the base implementation, reached through a subclass that keeps it
    reader = _ReadsEverything(
        RedshiftConnectionConfig(
            datasource_type=DataSourceType.REDSHIFT, host="h", database="d", username="u", password_ref="env://X"
        )
    )
    assert await driver(reader, "SELECT n FROM t", max_rows=4) == [{"n": index} for index in range(4)]
