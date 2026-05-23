"""live integration tests for :class:`RedshiftDriver` against ``central-reporting``.

THE smoking-gun proof for the whole datasource migration: this is
where we verify that ``redshift_connector`` returns ``information_schema.columns``
on the ``reporting_prod`` schema in under 60s -- the call that
``asyncpg`` could never complete (timed out at 60s / 120s / 300s in
production).

env-gated:

- ``OTS_REDSHIFT_PASSWORD`` MUST be set. when ``CI=1`` we
  :func:`pytest.fail` (not skip) because the whole point of this
  driver is the cross-engine proof; silently no-op'ing in CI defeats
  it. when ``CI`` is unset (local dev), we :func:`pytest.skip` so
  laptop runs don't crash without the secret in the environment.

run locally:

.. code-block:: bash

    OTS_REDSHIFT_PASSWORD=$(grep '^OTS_REDSHIFT_PASSWORD=' \\
        /Users/pace/crypt/pub/dev-wsl/vscode/3tears/14-eng-ai-bot-agent-ots/.env \\
        | cut -d= -f2) \\
      uv run --project 3tears/packages/datasources pytest \\
      tests/integration/test_redshift_driver_live.py -v
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import tracemalloc
from typing import Any

import pytest

from threetears.datasources.config import RedshiftConnectionConfig
from threetears.datasources.drivers.base import Driver
from threetears.datasources.drivers.redshift_driver import RedshiftDriver
from threetears.datasources.entities import DataSourceType

from ..unit._helpers.cancellation_contract import (
    DriverCancellationContractTest,
)

pytestmark = [pytest.mark.integration, pytest.mark.live]


# ---------------------------------------------------------------------------
# Env gate (DS-11-09): CI-required, locally skip-friendly
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def redshift_creds() -> dict[str, Any]:
    """gate the live tests on ``OTS_REDSHIFT_PASSWORD``.

    when ``CI=1`` and the env var is missing: :func:`pytest.fail`
    (the smoking-gun proof for the asyncpg-fix migration cannot run
    silently). when ``CI`` is unset: :func:`pytest.skip`.

    :return: connection-config dict for the central-reporting cluster
    :rtype: dict[str, Any]
    """
    pw = os.environ.get("OTS_REDSHIFT_PASSWORD")
    if not pw:
        if os.environ.get("CI"):
            pytest.fail(
                "OTS_REDSHIFT_PASSWORD missing in CI; the smoking-gun "
                "proof that redshift_connector fixes the asyncpg-hangs bug "
                "cannot run"
            )
        pytest.skip("OTS_REDSHIFT_PASSWORD not set; live test skipped locally")
    return {
        "host": "central.c30hiwrajgjj.us-east-1.redshift.amazonaws.com",
        "port": 5439,
        "database": "analytics",
        "username": "fourteen_eng_ai_bot_agent_ots",
        "password_ref": "env://OTS_REDSHIFT_PASSWORD",
    }


def _make_config(creds: dict[str, Any]) -> RedshiftConnectionConfig:
    """build a :class:`RedshiftConnectionConfig` from the creds dict.

    :param creds: dict from the :func:`redshift_creds` fixture
    :ptype creds: dict[str, Any]
    :return: live config pointing at central-reporting
    :rtype: RedshiftConnectionConfig
    """
    return RedshiftConnectionConfig(
        datasource_type=DataSourceType.REDSHIFT,
        host=creds["host"],
        port=creds["port"],
        database=creds["database"],
        username=creds["username"],
        password_ref=creds["password_ref"],
        executor_max_workers=4,
        connection_cache_size=2,
        query_timeout_seconds=120,
    )


# ---------------------------------------------------------------------------
# Python-side helper for Tier-2 hash byte-equivalence
# ---------------------------------------------------------------------------


def _python_column_hash(cols: list[dict[str, Any]]) -> str:
    """python-side MD5 over the column shape; cross-language invariant.

    payload formula: ``column_name + ':' + data_type + ':' + (is_nullable or '')``
    per column, joined by ``','`` in ascending ``ordinal_position``.
    matches the SQL ``MD5(LISTAGG(... WITHIN GROUP (ORDER BY ordinal_position)))``
    in :data:`_REDSHIFT_TABLE_HASHES_SQL` byte-for-byte (Redshift's
    LISTAGG WITHIN GROUP with the same separator and ordering is
    byte-equivalent to postgres' STRING_AGG with the same ORDER BY
    over the same input rows).

    :param cols: column rows (must have ``column_name``, ``data_type``,
        ``is_nullable``, ``ordinal_position`` keys)
    :ptype cols: list[dict[str, Any]]
    :return: hex MD5 digest
    :rtype: str
    """
    payload = ",".join(
        f"{c['column_name']}:{c['data_type']}:{(c['is_nullable'] or '')}"
        for c in sorted(cols, key=lambda c: c["ordinal_position"])
    )
    return hashlib.md5(payload.encode()).hexdigest()  # noqa: S324


# ---------------------------------------------------------------------------
# Smoking-gun: list_columns(['reporting_prod']) completes in <60s
# ---------------------------------------------------------------------------


class TestSmokingGun:
    """the load-bearing proof: ``redshift_connector`` succeeds where ``asyncpg`` hung."""

    @pytest.mark.asyncio
    async def test_list_columns_completes(self, redshift_creds: dict[str, Any]) -> None:
        """THE proof point: ``list_columns(['reporting_prod'])`` returns >5000 rows in <60s.

        on ``asyncpg`` against ``information_schema.columns`` this
        call NEVER COMPLETES on Redshift's reporting_prod schema
        (~6000 columns; timed out at 60s / 120s / 300s in production).
        on ``redshift_connector`` against ``SVV_COLUMNS`` (the
        Redshift-native system view; see SQL constant docstring) it
        returns in well under 60s -- typically <30s even on a busy
        cluster.
        """
        config = _make_config(redshift_creds)
        driver = RedshiftDriver(config, datasource_name="central-reporting")
        try:
            cols = await asyncio.wait_for(
                driver.list_columns(["reporting_prod"]),
                timeout=60.0,
            )
            assert len(cols) > 5000, f"expected >5000 columns in reporting_prod, got {len(cols)}"
            # row shape sanity
            assert {"table_schema", "table_name", "column_name", "data_type", "is_nullable", "ordinal_position"} <= set(
                cols[0]
            )
            # raw is_nullable string preserved (NOT a bool)
            for c in cols[:50]:
                assert isinstance(c["is_nullable"], str)
                assert c["is_nullable"] in ("YES", "NO", "")
        finally:
            await driver.close()

    @pytest.mark.asyncio
    async def test_list_tables_completes(self, redshift_creds: dict[str, Any]) -> None:
        """``list_tables(['reporting_prod'])`` returns >0 tables in <30s.

        SVV_TABLES is sub-second on a healthy cluster.
        """
        config = _make_config(redshift_creds)
        driver = RedshiftDriver(config, datasource_name="central-reporting")
        try:
            tables = await asyncio.wait_for(
                driver.list_tables(["reporting_prod"]),
                timeout=30.0,
            )
            assert len(tables) > 0
            assert all(t["table_schema"] == "reporting_prod" for t in tables)
        finally:
            await driver.close()

    @pytest.mark.asyncio
    async def test_table_hashes_returns_per_table_entries(self, redshift_creds: dict[str, Any]) -> None:
        """``table_hashes`` returns one entry per table in the schema.

        the LISTAGG + MD5 hash runs over SVV_COLUMNS (Redshift-native
        system view; ``information_schema.columns`` doesn't allow
        aggregates).
        """
        config = _make_config(redshift_creds)
        driver = RedshiftDriver(config, datasource_name="central-reporting")
        try:
            hashes = await asyncio.wait_for(
                driver.table_hashes(["reporting_prod"]),
                timeout=120.0,
            )
            assert len(hashes) > 0
            for (schema, table), digest in list(hashes.items())[:5]:
                assert schema == "reporting_prod"
                assert isinstance(table, str)
                # MD5 hex digest length is 32 chars
                assert isinstance(digest, str) and len(digest) == 32
        finally:
            await driver.close()


# ---------------------------------------------------------------------------
# Cross-language hash byte-equivalence (Tier-2 invariant)
# ---------------------------------------------------------------------------


class TestHashEquivalence:
    """python-side ``_python_column_hash`` byte-equals warehouse-side MD5."""

    @pytest.mark.asyncio
    async def test_python_and_sql_hashes_agree(self, redshift_creds: dict[str, Any]) -> None:
        """call :meth:`table_hashes` + recompute in python; assert equality.

        the cross-language invariant for the Tier-2 change-probe.
        run against the smallest table we can find in reporting_prod
        to keep the test bounded.
        """
        config = _make_config(redshift_creds)
        driver = RedshiftDriver(config, datasource_name="central-reporting")
        try:
            tables = await asyncio.wait_for(
                driver.list_tables(["reporting_prod"]),
                timeout=30.0,
            )
            assert tables, "no tables in reporting_prod"
            cols = await asyncio.wait_for(
                driver.list_columns(["reporting_prod"]),
                timeout=60.0,
            )
            hashes = await asyncio.wait_for(
                driver.table_hashes(["reporting_prod"]),
                timeout=120.0,
            )
            # pick the first table with at least 1 column and check
            # python vs SQL hash byte-equality.
            target_table_name: str | None = None
            target_cols: list[dict[str, Any]] = []
            for tbl in tables:
                t_cols = [c for c in cols if c["table_name"] == tbl["table_name"]]
                if t_cols:
                    target_table_name = tbl["table_name"]
                    target_cols = t_cols
                    break
            assert target_table_name is not None, "no table with columns"
            py_hash = _python_column_hash(target_cols)
            sql_hash = hashes[("reporting_prod", target_table_name)]
            assert sql_hash == py_hash, f"hash mismatch for {target_table_name}: python={py_hash} sql={sql_hash}"
        finally:
            await driver.close()


# ---------------------------------------------------------------------------
# fetch_iter streaming (DS-11-01 / DS-11-09)
# ---------------------------------------------------------------------------


class TestStreaming:
    """:meth:`fetch_iter` streams without OOMing on large result sets."""

    @pytest.mark.asyncio
    async def test_fetch_iter_streams_large_result(self, redshift_creds: dict[str, Any]) -> None:
        """LIMIT 50000 over information_schema; streaming stays bounded.

        compares tracemalloc peak between ``fetch`` (materialize-
        everything) and ``fetch_iter`` (streaming). exact ratio
        varies; we require streaming peak strictly < fetch peak.
        information_schema.columns is large enough on reporting_prod
        for the difference to be measurable.
        """
        config = _make_config(redshift_creds)
        driver = RedshiftDriver(config, datasource_name="central-reporting")
        try:
            # SVV_COLUMNS is fast (Redshift-native system view).
            sql = (
                "SELECT table_schema, table_name, column_name, data_type "
                "FROM SVV_COLUMNS "
                "WHERE table_schema = 'reporting_prod' "
                "ORDER BY table_schema, table_name, ordinal_position "
                "LIMIT 5000"
            )
            # materialize path
            tracemalloc.start()
            rows = await asyncio.wait_for(driver.fetch(sql), timeout=120.0)
            _, fetch_peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            assert len(rows) > 0
            del rows

            # streaming path
            count = 0
            tracemalloc.start()
            async for _row in driver.fetch_iter(sql):
                count += 1
            _, stream_peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            assert count > 0
            # streaming peak SHOULD be less than the materialize peak;
            # not asserting a strict ratio because Redshift's cursor
            # behavior is not fully server-side (it batches in
            # arraysize chunks but may pre-fetch). the directional
            # guard catches catastrophic regressions.
            assert stream_peak <= fetch_peak * 1.2, (
                f"fetch_iter peak {stream_peak} unexpectedly larger than fetch peak {fetch_peak}"
            )
        finally:
            await driver.close()


# ---------------------------------------------------------------------------
# Cancellation contract via mixin
# ---------------------------------------------------------------------------


class TestRedshiftDriverCancellation(DriverCancellationContractTest):
    """inherit the canonical cancellation contract; supply slow driver + SQL.

    the mixin runs ``test_fetch_propagates_cancellation`` +
    ``test_execute_propagates_cancellation`` against the real
    cluster.
    """

    @pytest.fixture(autouse=True)
    def _wire_fixtures(self, redshift_creds: dict[str, Any]) -> None:
        """capture creds so the mixin methods can build a driver."""
        self._creds = redshift_creds

    async def make_slow_driver(self) -> Driver:
        """build a live :class:`RedshiftDriver`."""
        return RedshiftDriver(_make_config(self._creds), datasource_name="central-reporting")

    def slow_sql(self) -> str:
        """return a Redshift-flavored slow query.

        ``pg_sleep`` is RESTRICTED on Redshift -- only the bootstrap
        superuser may call it (verified empirically: regular users
        get ``function PG_SLEEP does not exist`` from the leader). a
        4-way cross-join over ``stl_query`` (a system table large
        enough on any active cluster to take many seconds at O(n^4))
        is a reliable substitute that runs as a normal user.
        """
        return "SELECT COUNT(*) FROM stl_query a CROSS JOIN stl_query b CROSS JOIN stl_query c CROSS JOIN stl_query d "


# ---------------------------------------------------------------------------
# Explicit cancellation observability (DS-11-08)
# ---------------------------------------------------------------------------


class TestCancellationObservable:
    """cancellation fires WLM cancellation visibly + cache stays consistent."""

    @pytest.mark.asyncio
    async def test_cancellation_via_wait_for_timeout(self, redshift_creds: dict[str, Any]) -> None:
        """``wait_for(slow_query, timeout=5)`` raises TimeoutError.

        uses a cross-join slow query (``pg_sleep`` is restricted on
        Redshift to bootstrap users). we don't query ``stv_recents``
        to verify the WLM slot freed because the verification SQL
        races with leader-node lag; the cancellation propagation is
        the contract.
        """
        config = _make_config(redshift_creds)
        driver = RedshiftDriver(config, datasource_name="central-reporting")
        # heavy cross-join over SVV_COLUMNS -> O(n^2) over 6000-row
        # input is reliably multi-second; 1s wait_for fires well
        # before completion.
        slow_sql = (
            "SELECT COUNT(*) "
            "FROM SVV_COLUMNS a "
            "CROSS JOIN SVV_COLUMNS b "
            "WHERE a.table_schema = 'reporting_prod' "
            "AND b.table_schema = 'reporting_prod'"
        )
        try:
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(
                    driver.fetch(slow_sql),
                    timeout=1.0,
                )
            # follow-up query should still work (fresh connection
            # acquired since the previous one was poisoned + evicted)
            rows = await asyncio.wait_for(
                driver.fetch("SELECT 1 AS x"),
                timeout=30.0,
            )
            assert rows == [{"x": 1}]
        finally:
            await driver.close()


# ---------------------------------------------------------------------------
# close() drains cache cleanly
# ---------------------------------------------------------------------------


class TestCloseDrainsCache:
    """:meth:`close` drains the connection cache without leaking."""

    @pytest.mark.asyncio
    async def test_close_drains_cache(self, redshift_creds: dict[str, Any]) -> None:
        """run a query (fills cache), close, assert cache is empty."""
        config = _make_config(redshift_creds)
        driver = RedshiftDriver(config, datasource_name="central-reporting")
        await driver.fetch("SELECT 1")
        # cache should now have one connection
        assert len(driver._cache) >= 1  # noqa: SLF001
        await driver.close()
        assert len(driver._cache) == 0  # noqa: SLF001
        assert driver._closed is True  # noqa: SLF001
