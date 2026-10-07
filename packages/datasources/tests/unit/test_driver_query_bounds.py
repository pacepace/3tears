"""What a caller running many queries at once must know of a driver: how many it may run, and whose pool it draws on.

The hub gates every datasource's calls. A driver that caps its own warehouse connections (Redshift,
a Postgres pool it owns) states that cap, so the caller's gate (which refuses busy, with a deadline)
is where a burst waits, never the driver's own semaphore; one whose logins are not guarded against a
refused credential (Snowflake, BigQuery) states 1, or a wrong credential fails a burst of logins at
once; one that borrows the host's own pool (agent_internal) states none and names the pool, so it is
bounded with every other driver borrowing it.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

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
        executor_max_workers=6,
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
    # bounded with every other borrower of the pool; its own answer fails closed
    assert driver.concurrent_queries == 1


async def test_a_driver_that_cannot_stop_early_still_answers_no_more_than_asked() -> None:
    """the default for a driver with no early stop: it reads what it reads, and keeps the bound."""

    class _ReadsEverything(SnowflakeDriver):
        async def fetch(self, sql: str, *params: Any, timeout_seconds: int | None = None) -> list[dict[str, Any]]:
            return [{"n": index} for index in range(10)]

    reader = _ReadsEverything(
        SnowflakeConnectionConfig(
            datasource_type=DataSourceType.SNOWFLAKE, account="a", warehouse="w", user="u", password_ref="env://X"
        )
    )
    assert await reader.fetch_at_most("SELECT n FROM t", max_rows=4) == [{"n": index} for index in range(4)]
    with pytest.raises(ValueError, match="max_rows"):
        await reader.fetch_at_most("SELECT n FROM t", max_rows=0)


def test_a_driver_lent_a_pool_states_no_cap_of_its_own_whatever_its_config_says() -> None:
    """the pool lent to it decides, not the pool size its config would have opened."""
    pool = MagicMock(name="lent-pool")
    driver = AsyncpgDriver(
        PostgresConnectionConfig(
            datasource_type=DataSourceType.POSTGRES,
            host="localhost",
            database="x",
            password_ref="env://ABSENT_PW",
            pool_max_size=9,
        ),
        external_pool=pool,
    )
    assert (driver.concurrent_queries, driver.borrowed_pool) == (1, pool)


def test_a_driver_that_says_nothing_runs_one_query_at_a_time() -> None:
    """fail closed: a driver that states no cap of its own is asked one query at a time."""
    snowflake = SnowflakeDriver(
        SnowflakeConnectionConfig(
            datasource_type=DataSourceType.SNOWFLAKE, account="a", warehouse="w", user="u", password_ref="env://X"
        )
    )
    assert Driver.concurrent_queries.fget(snowflake) == 1  # type: ignore[attr-defined]


def test_a_config_for_a_lent_pool_with_no_pool_is_refused() -> None:
    with pytest.raises(ValueError, match="lent"):
        AsyncpgDriver(
            BorrowedPoolConnectionConfig(datasource_type=DataSourceType.AGENT_INTERNAL, schema_name="agent_a")
        )
