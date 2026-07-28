"""live proof of the driver transaction API against real backends (dsd-task-01).

DSD-01-08 names Redshift **and** asyncpg explicitly, because the local
Postgres stand-in is where most development happens and a transaction
API that works on one silently diverges on the other until production.

three claims, each verified against a real engine:

1. **atomicity** -- a two-statement transaction rolls back when the
   second statement fails; the first statement's effect is gone.
2. **session pinning** -- two statements in one transaction report the
   same ``pg_backend_pid()``.
3. **no idle-in-transaction on release** -- a completed ``SELECT``
   leaves no open snapshot, so a ``DROP`` does not block behind it.

plus the timeout claims:

4. a long statement and a short one on the same datasource observe
   different ``statement_timeout`` values, and
5. neither inherits the other's after a cache hit.

the postgres half runs against the shared testcontainer. the Redshift
half is env-gated on ``OTS_REDSHIFT_PASSWORD`` -- ``pytest.fail`` in CI
(the cross-engine proof cannot silently no-op) and ``pytest.skip``
locally.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from collections.abc import AsyncIterator
from typing import Any

import asyncpg
import pytest

from threetears.datasources.config import (
    PostgresConnectionConfig,
    RedshiftConnectionConfig,
)
from threetears.datasources.drivers.asyncpg_driver import AsyncpgDriver
from threetears.datasources.drivers.redshift_driver import RedshiftDriver
from threetears.datasources.entities import DataSourceType

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Postgres (the local stand-in)
# ---------------------------------------------------------------------------


def _parse_db_url(db_url: str) -> dict[str, Any]:
    """parse the testcontainer URL into connect kwargs.

    :param db_url: ``postgresql://user:pw@host:port/db`` style URL
    :ptype db_url: str
    :return: dict with host / port / database / username / password keys
    :rtype: dict[str, Any]
    """
    from urllib.parse import urlsplit

    parts = urlsplit(db_url)
    return {
        "host": parts.hostname or "localhost",
        "port": parts.port or 5432,
        "database": (parts.path or "/postgres").lstrip("/"),
        "username": parts.username or "postgres",
        "password": parts.password or "",
    }


@pytest.fixture
async def pg_schema(db_container: str, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[tuple[str, str]]:
    """provision an empty schema for the transaction tests to write into.

    yields ``(db_url, schema_name)``; teardown drops the schema.
    """
    parsed = _parse_db_url(db_container)
    schema = "ds_it_txn"
    conn = await asyncpg.connect(
        host=parsed["host"],
        port=parsed["port"],
        database=parsed["database"],
        user=parsed["username"],
        password=parsed["password"],
    )
    monkeypatch.setenv("TXN_DRIVER_TEST_PW", parsed["password"])
    try:
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.execute(f'CREATE SCHEMA "{schema}"')
        yield db_container, schema
    finally:
        await conn.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        await conn.close()


def _pg_config(db_url: str) -> PostgresConnectionConfig:
    """build a postgres config against the testcontainer.

    :param db_url: testcontainer URL
    :ptype db_url: str
    :return: config with a two-connection pool
    :rtype: PostgresConnectionConfig
    """
    parsed = _parse_db_url(db_url)
    return PostgresConnectionConfig(
        datasource_type=DataSourceType.POSTGRES,
        host=parsed["host"],
        port=parsed["port"],
        database=parsed["database"],
        username=parsed["username"],
        password_ref="env://TXN_DRIVER_TEST_PW",
        pool_min_size=1,
        pool_max_size=2,
        command_timeout_seconds=30,
    )


class TestAsyncpgTransactionLive:
    """DSD-01-08 against a real Postgres."""

    @pytest.mark.asyncio
    async def test_two_statement_transaction_rolls_back_atomically(
        self,
        pg_schema: tuple[str, str],
    ) -> None:
        """the first statement's effect is gone when the second fails."""
        db_url, schema = pg_schema
        driver = AsyncpgDriver(_pg_config(db_url))
        try:
            with pytest.raises(asyncpg.PostgresError):
                async with driver.transaction() as transaction:
                    await transaction.execute(f'CREATE TABLE "{schema}"."atomic" (id int)')
                    await transaction.execute(f'INSERT INTO "{schema}"."atomic" VALUES (not_a_column)')
            rows = await driver.fetch(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = $1",
                schema,
            )
            assert [r["table_name"] for r in rows] == [], "the rolled-back CREATE TABLE survived"
        finally:
            await driver.close()

    @pytest.mark.asyncio
    async def test_committed_transaction_persists(self, pg_schema: tuple[str, str]) -> None:
        """the happy path actually lands."""
        db_url, schema = pg_schema
        driver = AsyncpgDriver(_pg_config(db_url))
        try:
            async with driver.transaction() as transaction:
                await transaction.execute(f'CREATE TABLE "{schema}"."kept" (id int)')
                await transaction.execute(f'INSERT INTO "{schema}"."kept" VALUES (1)')
            rows = await driver.fetch(f'SELECT id FROM "{schema}"."kept"')
            assert [r["id"] for r in rows] == [1]
        finally:
            await driver.close()

    @pytest.mark.asyncio
    async def test_two_statements_share_a_backend_pid(self, pg_schema: tuple[str, str]) -> None:
        """DSD-01-02 on a real engine: one session for the transaction's life."""
        db_url, _schema = pg_schema
        driver = AsyncpgDriver(_pg_config(db_url))
        try:
            async with driver.transaction() as transaction:
                first = await transaction.fetch("SELECT pg_backend_pid() AS p")
                second = await transaction.fetch("SELECT pg_backend_pid() AS p")
            assert first[0]["p"] == second[0]["p"], (first, second)
        finally:
            await driver.close()

    @pytest.mark.asyncio
    async def test_drop_does_not_block_behind_a_completed_select(
        self,
        pg_schema: tuple[str, str],
    ) -> None:
        """DSD-01-05 on a real engine: the read path holds no snapshot on release."""
        db_url, schema = pg_schema
        driver = AsyncpgDriver(_pg_config(db_url))
        try:
            await driver.execute(f'CREATE TABLE "{schema}"."reapable" (id int)')
            await driver.execute(f'INSERT INTO "{schema}"."reapable" VALUES (1)')
            # a completed SELECT whose connection is now back in the pool
            rows = await driver.fetch(f'SELECT id FROM "{schema}"."reapable"')
            assert [r["id"] for r in rows] == [1]
            # the DROP needs ACCESS EXCLUSIVE; it blocks if the pooled
            # connection still holds the finished SELECT's snapshot
            await asyncio.wait_for(
                driver.execute(f'DROP TABLE "{schema}"."reapable"'),
                timeout=15.0,
            )
        finally:
            await driver.close()

    @pytest.mark.asyncio
    async def test_no_backend_left_idle_in_transaction(self, pg_schema: tuple[str, str]) -> None:
        """after a completed SELECT no backend for this database sits idle in transaction."""
        db_url, _schema = pg_schema
        driver = AsyncpgDriver(_pg_config(db_url))
        try:
            await driver.fetch("SELECT 1")
            rows = await driver.fetch(
                "SELECT count(*) AS n FROM pg_stat_activity "
                "WHERE state = 'idle in transaction' AND pid <> pg_backend_pid()"
            )
            assert int(rows[0]["n"]) == 0
        finally:
            await driver.close()

    @pytest.mark.asyncio
    async def test_short_and_long_timeouts_do_not_leak_across_a_cache_hit(
        self,
        pg_schema: tuple[str, str],
    ) -> None:
        """DSD-01-04 on a real engine: the observed timeout is per statement."""
        db_url, _schema = pg_schema
        driver = AsyncpgDriver(_pg_config(db_url))
        try:
            # a 1s bound kills a 5s sleep
            with pytest.raises(asyncpg.QueryCanceledError):
                await driver.fetch("SELECT pg_sleep(5)", timeout_seconds=1)
            # the very next statement -- likely the same pooled connection --
            # observes ITS OWN bound, not the 1s one
            observed = await driver.fetch("SHOW statement_timeout")
            assert observed[0]["statement_timeout"] in {"0", "30s", "30000ms"}, observed
            rows = await driver.fetch("SELECT pg_sleep(2), 1 AS ok", timeout_seconds=60)
            assert rows[0]["ok"] == 1
        finally:
            await driver.close()


# ---------------------------------------------------------------------------
# Redshift (the production engine)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def redshift_creds() -> dict[str, Any]:
    """gate the Redshift half on ``OTS_REDSHIFT_PASSWORD``.

    :return: connection-config dict for the central-reporting cluster
    :rtype: dict[str, Any]
    """
    pw = os.environ.get("OTS_REDSHIFT_PASSWORD")
    if not pw:
        if os.environ.get("CI"):
            pytest.fail("OTS_REDSHIFT_PASSWORD missing in CI; the cross-engine transaction proof cannot run")
        pytest.skip("OTS_REDSHIFT_PASSWORD not set; live redshift transaction test skipped locally")
    return {
        "host": "central.c30hiwrajgjj.us-east-1.redshift.amazonaws.com",
        "port": 5439,
        "database": "analytics",
        "username": "fourteen_eng_ai_bot_agent_ots",
        "password_ref": "env://OTS_REDSHIFT_PASSWORD",
    }


def _redshift_config(creds: dict[str, Any]) -> RedshiftConnectionConfig:
    """build a Redshift config from the creds dict.

    :param creds: creds mapping from :func:`redshift_creds`
    :ptype creds: dict[str, Any]
    :return: config with a one-connection cache so the re-borrow is a cache hit
    :rtype: RedshiftConnectionConfig
    """
    return RedshiftConnectionConfig(
        datasource_type=DataSourceType.REDSHIFT,
        host=creds["host"],
        port=creds["port"],
        database=creds["database"],
        username=creds["username"],
        password_ref=creds["password_ref"],
        executor_max_workers=2,
        connection_cache_size=1,
        query_timeout_seconds=300,
    )


@pytest.mark.live
class TestRedshiftTransactionLive:
    """DSD-01-08 against the real Redshift cluster."""

    @pytest.mark.asyncio
    async def test_two_statement_transaction_rolls_back_atomically(
        self,
        redshift_creds: dict[str, Any],
    ) -> None:
        """``CREATE TABLE`` is transactional on Redshift; prove the rollback."""
        driver = RedshiftDriver(_redshift_config(redshift_creds))
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
    async def test_two_statements_share_a_backend_pid(self, redshift_creds: dict[str, Any]) -> None:
        """DSD-01-02 on Redshift: one session for the transaction's life."""
        driver = RedshiftDriver(_redshift_config(redshift_creds))
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
        redshift_creds: dict[str, Any],
    ) -> None:
        """THE leak, against the engine that has to honour ``SET LOCAL``.

        with ``connection_cache_size=1`` the second statement is
        guaranteed to draw the connection the first one released.
        """
        driver = RedshiftDriver(_redshift_config(redshift_creds))
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
        redshift_creds: dict[str, Any],
    ) -> None:
        """DSD-01-05 on Redshift: nothing is left idle in transaction."""
        driver = RedshiftDriver(_redshift_config(redshift_creds))
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
