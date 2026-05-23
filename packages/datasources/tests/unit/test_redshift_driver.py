"""unit tests for :class:`RedshiftDriver` against a mocked ``redshift_connector``.

scope: driver construction + close-concurrency + cache routing +
SQL-constant + secret-resolution + cancellation-callback code paths
that don't need a real Redshift cluster.

live cancellation against a real cluster is covered by
``tests/integration/test_redshift_driver_live.py``. these unit tests
mock the connector module so cancellation paths can be exercised
deterministically.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from threetears.datasources.config import RedshiftConnectionConfig
from threetears.datasources.drivers.redshift_driver import (
    DriverConnectError,
    RedshiftDriver,
    _CANCEL_TIMEOUT_SECONDS,
    _PING_SQL,
    _REDSHIFT_COLUMNS_SQL_TEMPLATE,
    _REDSHIFT_TABLES_SQL_TEMPLATE,
    _REDSHIFT_TABLE_HASHES_SQL_TEMPLATE,
)
from threetears.datasources.entities import DataSourceType


def _is_set_stmt_timeout(sql: str) -> bool:
    """test helper: classify a SQL string as the SET statement_timeout call.

    Redshift rejects bind parameters in ``SET``, so the driver inlines
    the integer milliseconds value into the SQL text. tests filter
    on the canonical prefix rather than an exact-match constant.

    :param sql: SQL text from a ``cursor.execute`` call
    :ptype sql: str
    :return: True iff this is the SET statement_timeout call
    :rtype: bool
    """
    return sql.startswith("SET statement_timeout")


# ---------------------------------------------------------------------------
# Mock builder
# ---------------------------------------------------------------------------


def _build_mock_connection(
    *,
    fetchall_rows: list[tuple[Any, ...]] | None = None,
    description: list[tuple[str, Any]] | None = None,
    fetchone_row: tuple[Any, ...] | None = None,
    fetchmany_chunks: list[list[tuple[Any, ...]]] | None = None,
) -> MagicMock:
    """build a MagicMock that behaves like a ``redshift_connector.Connection``.

    cursor + connection cycles work as expected; ``execute`` / ``fetchall``
    / ``fetchmany`` / ``fetchone`` return the configured sequences.

    :param fetchall_rows: rows ``cursor.fetchall()`` returns
    :ptype fetchall_rows: list[tuple] | None
    :param description: ``cursor.description`` value (list of column
        descriptors; only the first element of each is used)
    :ptype description: list[tuple] | None
    :param fetchone_row: row ``cursor.fetchone()`` returns
    :ptype fetchone_row: tuple | None
    :param fetchmany_chunks: successive return values for fetchmany;
        the final element should be ``[]`` to terminate the loop
    :ptype fetchmany_chunks: list[list[tuple]] | None
    :return: connection mock with cursor/close/commit wired
    :rtype: MagicMock
    """
    conn = MagicMock(name="MockRedshiftConn")
    cursor = MagicMock(name="MockRedshiftCursor")
    cursor.description = description or []
    cursor.fetchall = MagicMock(return_value=fetchall_rows or [])
    cursor.fetchone = MagicMock(return_value=fetchone_row)
    if fetchmany_chunks is not None:
        cursor.fetchmany = MagicMock(side_effect=fetchmany_chunks)
    else:
        cursor.fetchmany = MagicMock(return_value=[])
    cursor.execute = MagicMock(return_value=None)
    cursor.close = MagicMock(return_value=None)
    conn.cursor = MagicMock(return_value=cursor)
    conn.commit = MagicMock(return_value=None)
    conn.close = MagicMock(return_value=None)
    # surface the cursor on the conn mock for assertions.
    conn._cursor = cursor  # noqa: SLF001 - test surface only
    return conn


@pytest.fixture
def redshift_config() -> RedshiftConnectionConfig:
    """default :class:`RedshiftConnectionConfig` for the unit tests.

    no ``password_env`` -- the connect call gets ``password=None``,
    which is fine for the mocked path.
    """
    return RedshiftConnectionConfig(
        datasource_type=DataSourceType.REDSHIFT,
        host="rs.example.com",
        port=5439,
        database="analytics",
        username="rs_user",
        password_env=None,
        executor_max_workers=2,
        connection_cache_size=2,
        query_timeout_seconds=60,
    )


# ---------------------------------------------------------------------------
# Construction + lifecycle
# ---------------------------------------------------------------------------


class TestConstruction:
    """``__init__`` stores config + bridge + cache; no I/O."""

    def test_init_does_not_open_connection(
        self, redshift_config: RedshiftConnectionConfig
    ) -> None:
        """constructing the driver does NOT open a redshift connection."""
        with patch(
            "threetears.datasources.drivers.redshift_driver.redshift_connector.connect"
        ) as connect_mock:
            driver = RedshiftDriver(redshift_config)
            assert driver._config is redshift_config  # noqa: SLF001
            assert driver._closed is False  # noqa: SLF001
            assert len(driver._cache) == 0  # noqa: SLF001
            connect_mock.assert_not_called()

    def test_init_bridge_sized_from_config(
        self, redshift_config: RedshiftConnectionConfig
    ) -> None:
        """bridge's executor max_workers matches ``executor_max_workers``."""
        driver = RedshiftDriver(redshift_config)
        # public surface on AsyncSyncBridge
        assert driver._bridge.max_workers == 2  # noqa: SLF001

    def test_init_datasource_name_default_is_unknown(
        self, redshift_config: RedshiftConnectionConfig
    ) -> None:
        """omitting ``datasource_name`` defaults to ``"unknown"``."""
        driver = RedshiftDriver(redshift_config)
        assert driver._datasource_name == "unknown"  # noqa: SLF001

    def test_init_datasource_name_captured(
        self, redshift_config: RedshiftConnectionConfig
    ) -> None:
        """passing ``datasource_name`` is stored for metric tagging."""
        driver = RedshiftDriver(redshift_config, datasource_name="ots")
        assert driver._datasource_name == "ots"  # noqa: SLF001

    def test_init_registers_finalize(
        self, redshift_config: RedshiftConnectionConfig
    ) -> None:
        """:func:`weakref.finalize` is registered for pod-crash mitigation."""
        driver = RedshiftDriver(redshift_config)
        # the finalize is alive until detach() / GC.
        assert driver._finalize.alive  # noqa: SLF001


class TestClose:
    """close() concurrency contract per DS-09-12 / DS-11-10."""

    @pytest.mark.asyncio
    async def test_close_idempotent(
        self, redshift_config: RedshiftConnectionConfig
    ) -> None:
        """second :meth:`close` is a no-op."""
        driver = RedshiftDriver(redshift_config)
        await driver.close()
        assert driver._closed is True  # noqa: SLF001
        # second call: no-op
        await driver.close()

    @pytest.mark.asyncio
    async def test_close_drains_cache(
        self, redshift_config: RedshiftConnectionConfig
    ) -> None:
        """every cached connection has ``close`` called on close()."""
        driver = RedshiftDriver(redshift_config)
        # inject two mock connections into the cache
        c1 = _build_mock_connection()
        c2 = _build_mock_connection()
        driver._cache.append(c1)  # noqa: SLF001
        driver._cache.append(c2)  # noqa: SLF001
        await driver.close()
        c1.close.assert_called()
        c2.close.assert_called()
        assert len(driver._cache) == 0  # noqa: SLF001

    @pytest.mark.asyncio
    async def test_methods_reject_after_close(
        self, redshift_config: RedshiftConnectionConfig
    ) -> None:
        """every public method raises :class:`RuntimeError` post-close."""
        driver = RedshiftDriver(redshift_config)
        await driver.close()
        with pytest.raises(RuntimeError, match="closed"):
            await driver.fetch("SELECT 1")
        with pytest.raises(RuntimeError, match="closed"):
            await driver.execute("SELECT 1")
        with pytest.raises(RuntimeError, match="closed"):
            await driver.list_tables(["s"])
        with pytest.raises(RuntimeError, match="closed"):
            await driver.list_columns(["s"])
        with pytest.raises(RuntimeError, match="closed"):
            await driver.table_hashes(["s"])
        with pytest.raises(RuntimeError, match="closed"):
            await driver.test_connection()

    @pytest.mark.asyncio
    async def test_fetch_iter_rejects_after_close(
        self, redshift_config: RedshiftConnectionConfig
    ) -> None:
        """:meth:`fetch_iter` (async generator) also raises post-close."""
        driver = RedshiftDriver(redshift_config)
        await driver.close()
        with pytest.raises(RuntimeError, match="closed"):
            async for _row in driver.fetch_iter("SELECT 1"):
                pass  # pragma: no cover

    @pytest.mark.asyncio
    async def test_close_detaches_finalize(
        self, redshift_config: RedshiftConnectionConfig
    ) -> None:
        """post-close, the weakref finalize is detached."""
        driver = RedshiftDriver(redshift_config)
        await driver.close()
        # detached finalize is no longer alive
        assert not driver._finalize.alive  # noqa: SLF001


# ---------------------------------------------------------------------------
# Connection caching
# ---------------------------------------------------------------------------


class TestConnectionCaching:
    """cache hit/miss + release routing."""

    @pytest.mark.asyncio
    async def test_first_fetch_misses_then_caches(
        self, redshift_config: RedshiftConnectionConfig
    ) -> None:
        """initial fetch: opens a fresh connection + caches it on release."""
        conn = _build_mock_connection(
            fetchall_rows=[(1, "alpha")],
            description=[("a", None), ("b", None)],
        )
        with patch(
            "threetears.datasources.drivers.redshift_driver.redshift_connector.connect",
            return_value=conn,
        ) as connect_mock:
            driver = RedshiftDriver(redshift_config)
            rows = await driver.fetch("SELECT a, b FROM t")
            assert rows == [{"a": 1, "b": "alpha"}]
            # connect called once on miss
            connect_mock.assert_called_once()
            # connection released back to cache
            assert len(driver._cache) == 1  # noqa: SLF001

    @pytest.mark.asyncio
    async def test_second_fetch_hits_cache(
        self, redshift_config: RedshiftConnectionConfig
    ) -> None:
        """second fetch reuses the cached connection (no second connect)."""
        conn = _build_mock_connection(
            fetchall_rows=[],
            description=[],
        )
        with patch(
            "threetears.datasources.drivers.redshift_driver.redshift_connector.connect",
            return_value=conn,
        ) as connect_mock:
            driver = RedshiftDriver(redshift_config)
            await driver.fetch("SELECT 1")
            await driver.fetch("SELECT 2")
            # connect called exactly once across both fetches
            connect_mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_set_statement_timeout_applied_on_open(
        self, redshift_config: RedshiftConnectionConfig
    ) -> None:
        """fresh connection has ``SET statement_timeout`` issued once."""
        conn = _build_mock_connection(
            fetchall_rows=[], description=[]
        )
        with patch(
            "threetears.datasources.drivers.redshift_driver.redshift_connector.connect",
            return_value=conn,
        ):
            driver = RedshiftDriver(redshift_config)
            await driver.fetch("SELECT 1")
            # the cursor saw at least one execute call with the
            # statement_timeout SQL (plus the fetch's own execute).
            calls = conn._cursor.execute.call_args_list  # noqa: SLF001
            stmt_timeout_calls = [
                c for c in calls if c.args and _is_set_stmt_timeout(c.args[0])
            ]
            assert len(stmt_timeout_calls) == 1
            # ms value is inlined into the SQL (Redshift rejects
            # parameter binding in SET statements). assert the value
            # is present in the SQL text.
            expected_ms = redshift_config.query_timeout_seconds * 1000
            assert str(expected_ms) in stmt_timeout_calls[0].args[0]


# ---------------------------------------------------------------------------
# Query routing
# ---------------------------------------------------------------------------


class TestQueryRouting:
    """fetch / execute / list_* route through the mocked connection."""

    @pytest.mark.asyncio
    async def test_fetch_translates_placeholders_dollar_to_percent_s(
        self, redshift_config: RedshiftConnectionConfig
    ) -> None:
        """``$1, $2`` becomes ``%s, %s`` for DB-API."""
        conn = _build_mock_connection(
            fetchall_rows=[],
            description=[],
        )
        with patch(
            "threetears.datasources.drivers.redshift_driver.redshift_connector.connect",
            return_value=conn,
        ):
            driver = RedshiftDriver(redshift_config)
            await driver.fetch("SELECT $1, $2", 1, "x")
            # find the non-statement_timeout execute call
            calls = [
                c for c in conn._cursor.execute.call_args_list  # noqa: SLF001
                if c.args and not _is_set_stmt_timeout(c.args[0])
            ]
            assert len(calls) == 1
            assert calls[0].args[0] == "SELECT %s, %s"
            assert calls[0].args[1] == (1, "x")

    @pytest.mark.asyncio
    async def test_list_tables_uses_tables_sql(
        self, redshift_config: RedshiftConnectionConfig
    ) -> None:
        """:meth:`list_tables` issues the canonical tables SQL."""
        conn = _build_mock_connection(
            fetchall_rows=[("s1", "t1")],
            description=[("table_schema", None), ("table_name", None)],
        )
        with patch(
            "threetears.datasources.drivers.redshift_driver.redshift_connector.connect",
            return_value=conn,
        ):
            driver = RedshiftDriver(redshift_config)
            rows = await driver.list_tables(["s1"])
            assert rows == [{"table_schema": "s1", "table_name": "t1"}]
            calls = [
                c for c in conn._cursor.execute.call_args_list  # noqa: SLF001
                if c.args and not _is_set_stmt_timeout(c.args[0])
            ]
            # SQL is built from template + IN-clause placeholder for one schema
            expected_sql = _REDSHIFT_TABLES_SQL_TEMPLATE.format(placeholders="%s")
            assert calls[0].args[0] == expected_sql
            assert calls[0].args[1] == ("s1",)

    @pytest.mark.asyncio
    async def test_list_columns_preserves_raw_is_nullable(
        self, redshift_config: RedshiftConnectionConfig
    ) -> None:
        """``is_nullable`` is the raw warehouse string, NOT a bool."""
        conn = _build_mock_connection(
            fetchall_rows=[
                ("s1", "t1", "c1", "INT4", "NO", 1),
                ("s1", "t1", "c2", "VARCHAR", "YES", 2),
            ],
            description=[
                ("table_schema", None),
                ("table_name", None),
                ("column_name", None),
                ("data_type", None),
                ("is_nullable", None),
                ("ordinal_position", None),
            ],
        )
        with patch(
            "threetears.datasources.drivers.redshift_driver.redshift_connector.connect",
            return_value=conn,
        ):
            driver = RedshiftDriver(redshift_config)
            rows = await driver.list_columns(["s1"])
            assert rows[0]["is_nullable"] == "NO"
            assert rows[1]["is_nullable"] == "YES"
            assert isinstance(rows[0]["is_nullable"], str)
            calls = [
                c for c in conn._cursor.execute.call_args_list  # noqa: SLF001
                if c.args and not _is_set_stmt_timeout(c.args[0])
            ]
            expected_sql = _REDSHIFT_COLUMNS_SQL_TEMPLATE.format(placeholders="%s")
            assert calls[0].args[0] == expected_sql

    @pytest.mark.asyncio
    async def test_table_hashes_returns_dict_keyed_by_schema_table(
        self, redshift_config: RedshiftConnectionConfig
    ) -> None:
        """:meth:`table_hashes` keys on ``(schema, table)``."""
        conn = _build_mock_connection(
            fetchall_rows=[("s1", "t1", "abc123")],
            description=[
                ("table_schema", None),
                ("table_name", None),
                ("column_hash", None),
            ],
        )
        with patch(
            "threetears.datasources.drivers.redshift_driver.redshift_connector.connect",
            return_value=conn,
        ):
            driver = RedshiftDriver(redshift_config)
            hashes = await driver.table_hashes(["s1"])
            assert hashes == {("s1", "t1"): "abc123"}
            calls = [
                c for c in conn._cursor.execute.call_args_list  # noqa: SLF001
                if c.args and not _is_set_stmt_timeout(c.args[0])
            ]
            expected_sql = _REDSHIFT_TABLE_HASHES_SQL_TEMPLATE.format(placeholders="%s")
            assert calls[0].args[0] == expected_sql

    @pytest.mark.asyncio
    async def test_execute_commits(
        self, redshift_config: RedshiftConnectionConfig
    ) -> None:
        """:meth:`execute` issues ``conn.commit()`` so DDL/DML lands."""
        conn = _build_mock_connection()
        with patch(
            "threetears.datasources.drivers.redshift_driver.redshift_connector.connect",
            return_value=conn,
        ):
            driver = RedshiftDriver(redshift_config)
            await driver.execute("INSERT INTO t VALUES ($1)", 1)
            conn.commit.assert_called()


# ---------------------------------------------------------------------------
# test_connection sanitization
# ---------------------------------------------------------------------------


class TestTestConnection:
    """:meth:`test_connection` issues PING + sanitizes failures."""

    @pytest.mark.asyncio
    async def test_test_connection_happy_path(
        self, redshift_config: RedshiftConnectionConfig
    ) -> None:
        """successful round-trip returns None silently."""
        conn = _build_mock_connection(
            fetchone_row=(1,),
            description=[("?column?", None)],
        )
        with patch(
            "threetears.datasources.drivers.redshift_driver.redshift_connector.connect",
            return_value=conn,
        ):
            driver = RedshiftDriver(redshift_config)
            await driver.test_connection()
            # the PING SQL was issued
            calls = [
                c for c in conn._cursor.execute.call_args_list  # noqa: SLF001
                if c.args and not _is_set_stmt_timeout(c.args[0])
            ]
            assert any(c.args[0] == _PING_SQL for c in calls)

    @pytest.mark.asyncio
    async def test_test_connection_sanitizes_connect_failure(
        self, redshift_config: RedshiftConnectionConfig
    ) -> None:
        """``redshift_connector.connect`` failure wraps in :class:`DriverConnectError`."""
        with patch(
            "threetears.datasources.drivers.redshift_driver.redshift_connector.connect",
            side_effect=RuntimeError("kapow"),
        ):
            driver = RedshiftDriver(redshift_config)
            with pytest.raises(DriverConnectError) as exc_info:
                await driver.test_connection()
            # ``from None`` clears __cause__
            assert exc_info.value.__cause__ is None
            assert "rs.example.com" in str(exc_info.value)
            assert "kapow" not in str(exc_info.value)


# ---------------------------------------------------------------------------
# fetch_iter streaming
# ---------------------------------------------------------------------------


class TestFetchIter:
    """:meth:`fetch_iter` streams via DB-API ``fetchmany``."""

    @pytest.mark.asyncio
    async def test_fetch_iter_yields_chunks_correctly(
        self, redshift_config: RedshiftConnectionConfig
    ) -> None:
        """multiple fetchmany chunks are concatenated correctly."""
        conn = _build_mock_connection(
            description=[("a", None)],
            fetchmany_chunks=[
                [(1,), (2,)],
                [(3,)],
                [],  # end of stream
            ],
        )
        with patch(
            "threetears.datasources.drivers.redshift_driver.redshift_connector.connect",
            return_value=conn,
        ):
            driver = RedshiftDriver(redshift_config)
            rows: list[dict[str, Any]] = []
            async for row in driver.fetch_iter("SELECT a FROM t"):
                rows.append(row)
            assert rows == [{"a": 1}, {"a": 2}, {"a": 3}]

    @pytest.mark.asyncio
    async def test_fetch_iter_sets_arraysize(
        self, redshift_config: RedshiftConnectionConfig
    ) -> None:
        """cursor.arraysize is set before iteration (server-side fetchmany)."""
        conn = _build_mock_connection(
            description=[("a", None)],
            fetchmany_chunks=[[]],
        )
        with patch(
            "threetears.datasources.drivers.redshift_driver.redshift_connector.connect",
            return_value=conn,
        ):
            driver = RedshiftDriver(redshift_config)
            async for _row in driver.fetch_iter("SELECT a FROM t"):
                pass  # pragma: no cover
            # arraysize attribute was set on the cursor (>=1000 per
            # module constant; we don't import to avoid coupling, just
            # check it's a positive int)
            assert isinstance(conn._cursor.arraysize, int)  # noqa: SLF001
            assert conn._cursor.arraysize > 0  # noqa: SLF001


# ---------------------------------------------------------------------------
# Cancellation contract (DS-11-08)
# ---------------------------------------------------------------------------


class TestCancellation:
    """cancellation routes through wait_for + conn.close + observability."""

    @pytest.mark.asyncio
    async def test_cancellation_closes_connection_and_evicts(
        self, redshift_config: RedshiftConnectionConfig
    ) -> None:
        """cancellation closes the connection + evicts from cache."""
        conn = _build_mock_connection()

        # make the cursor.execute block forever to simulate a slow query
        execute_blocked = asyncio.Event()

        def _blocking_execute(*args: Any, **kwargs: Any) -> None:
            if args and _is_set_stmt_timeout(args[0]):
                return
            # signal that the slow path is engaged, then block.
            execute_blocked.set()
            # in the real thread this blocks indefinitely; mock by
            # spinning until the connection is closed (cancel cb).
            import time
            for _ in range(100):
                if conn.close.called:
                    return
                time.sleep(0.05)
            # safety bail
            return

        conn._cursor.execute.side_effect = _blocking_execute  # noqa: SLF001

        with patch(
            "threetears.datasources.drivers.redshift_driver.redshift_connector.connect",
            return_value=conn,
        ):
            driver = RedshiftDriver(redshift_config)
            task = asyncio.create_task(driver.fetch("SELECT pg_sleep(60)"))
            # let the task progress to the blocking execute
            await asyncio.wait_for(execute_blocked.wait(), timeout=2.0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            # the connection was closed via the cancel path
            conn.close.assert_called()
            # cache should not contain the poisoned connection
            assert conn not in driver._cache  # noqa: SLF001
            await driver.close()

    @pytest.mark.asyncio
    async def test_cancellation_callback_failure_observable(
        self, redshift_config: RedshiftConnectionConfig,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """if conn.close hangs past the timeout, WARNING is logged + cancellation.failed.

        we make conn.close raise after a delay simulating a hung
        terminate; the wait_for guard should fire and log a WARNING.
        """
        conn = _build_mock_connection()

        # make execute block so cancel actually has something to cancel
        execute_blocked = asyncio.Event()

        def _blocking_execute(*args: Any, **kwargs: Any) -> None:
            if args and _is_set_stmt_timeout(args[0]):
                return
            execute_blocked.set()
            import time
            for _ in range(200):
                if close_started.is_set():
                    return
                time.sleep(0.05)

        close_started = asyncio.Event()
        loop = asyncio.get_event_loop()

        def _hanging_close() -> None:
            # signal that close started, then hang
            loop.call_soon_threadsafe(close_started.set)
            import time
            time.sleep(_CANCEL_TIMEOUT_SECONDS + 2)
            # release after the wait_for has already fired

        conn._cursor.execute.side_effect = _blocking_execute  # noqa: SLF001
        conn.close.side_effect = _hanging_close

        with patch(
            "threetears.datasources.drivers.redshift_driver.redshift_connector.connect",
            return_value=conn,
        ):
            driver = RedshiftDriver(redshift_config)
            task = asyncio.create_task(driver.fetch("SELECT slow"))
            await asyncio.wait_for(execute_blocked.wait(), timeout=2.0)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            # the WARNING log carries the "cancel ... failed" marker.
            warnings = [r for r in caplog.records if r.levelname == "WARNING"]
            assert any("cancel" in r.getMessage().lower() for r in warnings)
            await driver.close()
