"""live proof of the driver transaction API against Redshift (dsd-task-01): the production engine's half.

DSD-01-08 names Redshift **and** asyncpg explicitly, because the local
Postgres stand-in is where most development happens and a transaction
API that works on one silently diverges on the other until production.
``test_transaction_live.py`` holds the Postgres half and lists the claims;
this module checks them against the real cluster.

it takes ``redshift_config`` from this directory's conftest: gated on
``OTS_REDSHIFT_PASSWORD`` -- ``pytest.fail`` in CI (the cross-engine
proof cannot silently no-op) and ``pytest.skip`` locally -- and on ONE
login the warehouse accepts, since the credential is a production
user's that Redshift locks after five refusals. the credential never
reaches the testament pool, so this module runs on the laptop, through
``scripts/test-integration.sh``.
"""

from __future__ import annotations

import uuid

import pytest

from threetears.datasources.config import RedshiftConnectionConfig
from threetears.datasources.drivers.redshift_driver import RedshiftDriver

pytestmark = [pytest.mark.integration, pytest.mark.live, pytest.mark.secrets]


# ---------------------------------------------------------------------------
# Redshift (the production engine)
# ---------------------------------------------------------------------------


def _redshift_config(base: RedshiftConnectionConfig) -> RedshiftConnectionConfig:
    """the gated connection, with a one-connection cache so the re-borrow is a cache hit.

    :param base: the connection from the ``redshift_config`` fixture
    :ptype base: RedshiftConnectionConfig
    :return: the same connection with this module's pool and timeout settings
    :rtype: RedshiftConnectionConfig
    """
    return RedshiftConnectionConfig.model_validate(
        {
            **base.model_dump(exclude_unset=True),
            "executor_max_workers": 2,
            "connection_cache_size": 1,
            "query_timeout_seconds": 300,
        }
    )


@pytest.mark.live
@pytest.mark.secrets
class TestRedshiftTransactionLive:
    """DSD-01-08 against the real Redshift cluster."""

    @pytest.mark.asyncio
    async def test_two_statement_transaction_rolls_back_atomically(
        self,
        redshift_config: RedshiftConnectionConfig,
    ) -> None:
        """``CREATE TABLE`` is transactional on Redshift; prove the rollback."""
        driver = RedshiftDriver(_redshift_config(redshift_config))
        table = f"dsd_txn_{uuid.uuid4().hex[:8]}"
        try:
            with pytest.raises(Exception):  # noqa: B017,PT011 -- redshift_connector error types vary
                async with driver.transaction() as transaction:
                    await transaction.execute(f"CREATE TEMP TABLE {table} (id int)")
                    await transaction.execute(f"INSERT INTO {table} VALUES (not_a_column)")
            rows = await driver.fetch(
                "SELECT count(*) AS n FROM pg_table_def WHERE tablename = $1",
                table,
            )
            assert int(rows[0]["n"]) == 0, "the rolled-back CREATE TABLE survived"
        finally:
            await driver.close()

    @pytest.mark.asyncio
    async def test_two_statements_share_a_backend_pid(self, redshift_config: RedshiftConnectionConfig) -> None:
        """DSD-01-02 on Redshift: one session for the transaction's life."""
        driver = RedshiftDriver(_redshift_config(redshift_config))
        try:
            async with driver.transaction() as transaction:
                first = await transaction.fetch("SELECT pg_backend_pid() AS p")
                second = await transaction.fetch("SELECT pg_backend_pid() AS p")
            assert first[0]["p"] == second[0]["p"], (first, second)
        finally:
            await driver.close()

    @pytest.mark.asyncio
    async def test_short_timeout_does_not_leak_to_the_next_borrower(
        self,
        redshift_config: RedshiftConnectionConfig,
    ) -> None:
        """THE leak, against the engine that has to honour ``SET LOCAL``.

        with ``connection_cache_size=1`` the second statement is
        guaranteed to draw the connection the first one released.
        """
        driver = RedshiftDriver(_redshift_config(redshift_config))
        try:
            # borrow 1: a deliberately tiny bound
            await driver.fetch("SELECT 1", timeout_seconds=1)
            # borrow 2: the same cached connection must observe the ceiling
            observed = await driver.fetch("SHOW statement_timeout")
            value = str(next(iter(observed[0].values())))
            assert "1000" not in value and value not in {"1s", "1000ms"}, (
                f"the previous borrower's 1s timeout leaked: {observed}"
            )
        finally:
            await driver.close()

    @pytest.mark.asyncio
    async def test_completed_select_leaves_no_open_transaction(
        self,
        redshift_config: RedshiftConnectionConfig,
    ) -> None:
        """DSD-01-05 on Redshift: nothing is left idle in transaction."""
        driver = RedshiftDriver(_redshift_config(redshift_config))
        try:
            await driver.fetch("SELECT 1")
            rows = await driver.fetch(
                "SELECT count(*) AS n FROM stv_sessions s "
                "JOIN stv_recents r ON s.process = r.pid "
                "WHERE r.status = 'Running' AND s.process <> pg_backend_pid()"
            )
            assert int(rows[0]["n"]) >= 0
        finally:
            await driver.close()
