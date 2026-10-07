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
from threetears.datasources.drivers.bigquery_driver import BigQueryDriver
from threetears.datasources.drivers.redshift_driver import RedshiftDriver
from threetears.datasources.drivers.snowflake_driver import SnowflakeDriver
from threetears.datasources.entities import DataSourceType


def _postgres() -> PostgresConnectionConfig:
    return PostgresConnectionConfig(
        datasource_type=DataSourceType.POSTGRES, host="localhost", database="x", password_ref="env://ABSENT_PW"
    )


def test_a_driver_that_bounds_its_own_connections_asks_no_bound_and_borrows_nothing() -> None:
    drivers: list[Any] = [
        AsyncpgDriver(_postgres()),
        RedshiftDriver(
            RedshiftConnectionConfig(
                datasource_type=DataSourceType.REDSHIFT,
                host="rs.example.com",
                database="analytics",
                username="u",
                password_ref="env://ABSENT_PW",
            )
        ),
    ]
    assert [(d.concurrent_queries, d.borrowed_pool) for d in drivers] == [(None, None), (None, None)]


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
