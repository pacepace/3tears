"""unit tests for :class:`AsyncpgDriver` against a mocked ``asyncpg.Pool``.

scope: driver construction + close-concurrency + pool routing +
SQL-constant + secret-resolution code paths that don't need a real
backend.

cancellation + streaming behaviour are integration-tested against a
testcontainer (see ``tests/integration/test_asyncpg_driver_live.py``)
because the cancellation contract requires a real backend round-trip
to verify cancel propagation in earnest.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import asyncpg
import pytest

from threetears.datasources.config import (
    BorrowedPoolConnectionConfig,
    PostgresConnectionConfig,
    YugabyteConnectionConfig,
)
from threetears.datasources.drivers import (
    DriverAuthError,
    DriverConnectError,
    DriverMissingCredentialError,
)
from threetears.datasources.drivers.asyncpg_driver import BORROWED_POOL_ACQUIRE_TIMEOUT_SECONDS, AsyncpgDriver
from threetears.datasources.drivers.errors import DriverPoolBusyError
from threetears.datasources.entities import DataSourceType


def _single_catalog_query(pool: MagicMock, schemas: list[str]) -> str:
    """the one catalog query the driver issued, after checking the allow-list was bound as ``$1``.

    the statement text is the driver's own; what the unit tier pins is that exactly one
    statement ran, that it filters on ``table_schema = ANY($1)`` and that the caller's
    allow-list is the bound value. the statement's result against a real engine is
    ``tests/integration/test_asyncpg_driver_live.py``'s.

    :param pool: the mocked pool the driver owns
    :ptype pool: MagicMock
    :param schemas: the allow-list the caller passed
    :ptype schemas: list[str]
    :return: the SQL text issued
    :rtype: str
    """
    pool.recorded_conn.fetch.assert_awaited_once()
    sql, bound = pool.recorded_conn.fetch.await_args.args
    assert bound == schemas
    assert "table_schema = ANY($1)" in sql
    result: str = sql
    return result


# ---------------------------------------------------------------------------
# Mocked-pool builder
# ---------------------------------------------------------------------------


def _build_mock_pool(
    *,
    fetch_records: list[dict[str, Any]] | None = None,
    fetchval_value: Any = 1,
) -> MagicMock:
    """build a MagicMock that behaves like an ``asyncpg.Pool``.

    pool.acquire() returns an async context manager yielding a
    Connection mock with fetch / execute / fetchval coroutines wired up.

    the connection is spec'd against the REAL :class:`asyncpg.Connection`
    (dsd-task-02), so an attribute the class does not carry raises
    :class:`AttributeError` here as it does in production. this builder
    used to assign ``conn.cancel`` -- a method asyncpg has never had --
    and that assignment is what concealed a driver cancellation path
    which cancelled nothing.

    :param fetch_records: rows ``conn.fetch`` should resolve to
    :ptype fetch_records: list[dict[str, Any]] | None
    :param fetchval_value: scalar ``conn.fetchval`` should resolve to
    :ptype fetchval_value: Any
    :return: pool mock with acquire/close wired
    :rtype: MagicMock
    """
    records = fetch_records or []

    pool = MagicMock(name="MockPool")

    # connection mock: every method we route through is async
    conn = MagicMock(spec=asyncpg.Connection, name="MockConn")
    conn.fetch = AsyncMock(return_value=records)
    conn.execute = AsyncMock(return_value=None)
    conn.fetchval = AsyncMock(return_value=fetchval_value)

    # async-context-manager shape for ``pool.acquire()``
    pool.acquire_options = []

    @asynccontextmanager
    async def _acquire(**options: Any) -> Any:
        pool.acquire_options.append(options)
        yield conn

    pool.acquire = _acquire
    pool.close = AsyncMock(return_value=None)
    pool.is_closing = MagicMock(return_value=False)

    # surface the connection mock so tests can assert against it
    pool.recorded_conn = conn
    return pool


def _driver_owning(
    pool: MagicMock,
    config: PostgresConnectionConfig | YugabyteConnectionConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncpgDriver:
    """build a driver that will create ``pool`` as its OWN pool on first use.

    the owned-pool path is what a postgres / yugabyte datasource takes in
    production: ``asyncpg.create_pool`` runs lazily on the first query.
    patching it to hand back ``pool`` keeps that path intact.

    :param pool: the mocked pool the driver's first query creates
    :ptype pool: MagicMock
    :param config: an owned-pool config
    :ptype config: PostgresConnectionConfig | YugabyteConnectionConfig
    :param monkeypatch: pytest monkeypatch fixture
    :ptype monkeypatch: pytest.MonkeyPatch
    :return: a driver that has not yet created its pool
    :rtype: AsyncpgDriver
    """
    monkeypatch.setattr(
        "threetears.datasources.drivers.asyncpg_driver.asyncpg.create_pool",
        AsyncMock(return_value=pool),
    )
    return AsyncpgDriver(config)


def _unresolvable_password_config(monkeypatch: pytest.MonkeyPatch) -> PostgresConnectionConfig:
    """an owned-pool config whose password reference resolves to nothing.

    the driver refuses it by name before any login, which is how a test
    reads the ``datasource_name`` a driver carries.

    :param monkeypatch: pytest monkeypatch fixture
    :ptype monkeypatch: pytest.MonkeyPatch
    :return: the config
    :rtype: PostgresConnectionConfig
    """
    monkeypatch.delenv("ABSENT_ASYNCPG_DRIVER_PW", raising=False)
    return PostgresConnectionConfig(
        datasource_type=DataSourceType.POSTGRES,
        host="localhost",
        database="x",
        password_ref="env://ABSENT_ASYNCPG_DRIVER_PW",
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def postgres_config() -> PostgresConnectionConfig:
    """default :class:`PostgresConnectionConfig` for the unit tests.

    no ``password_ref`` so the lazy pool creation path doesn't try to
    resolve a credential reference.
    """
    return PostgresConnectionConfig(
        datasource_type=DataSourceType.POSTGRES,
        host="localhost",
        database="x",
    )


@pytest.fixture
def yugabyte_config() -> YugabyteConnectionConfig:
    """default :class:`YugabyteConnectionConfig`."""
    return YugabyteConnectionConfig(
        datasource_type=DataSourceType.YUGABYTE,
        host="localhost",
        database="x",
    )


@pytest.fixture
def agent_internal_config() -> BorrowedPoolConnectionConfig:
    """default :class:`BorrowedPoolConnectionConfig`."""
    return BorrowedPoolConnectionConfig(
        datasource_type=DataSourceType.AGENT_INTERNAL,
        schema_name="agent_abc123",
    )


# ---------------------------------------------------------------------------
# Construction + lifecycle
# ---------------------------------------------------------------------------


class TestConstruction:
    """``__init__`` stores config + external_pool correctly; no I/O."""

    @pytest.mark.asyncio
    async def test_init_postgres_no_external_pool(
        self, postgres_config: PostgresConnectionConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """constructing a postgres driver does NOT open a pool eagerly; the first query opens one it owns."""
        pool = _build_mock_pool()
        create_pool = AsyncMock(return_value=pool)
        monkeypatch.setattr("threetears.datasources.drivers.asyncpg_driver.asyncpg.create_pool", create_pool)
        driver = AsyncpgDriver(postgres_config)
        create_pool.assert_not_awaited()
        # not closed at construction: the first query creates the pool from the config.
        await driver.fetch("SELECT 1")
        create_pool.assert_awaited_once()
        assert create_pool.await_args.kwargs["host"] == postgres_config.host
        assert create_pool.await_args.kwargs["database"] == postgres_config.database
        # the driver owns that pool, so its close closes it.
        await driver.close()
        pool.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_init_agent_internal_with_external_pool(
        self, agent_internal_config: BorrowedPoolConnectionConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """agent-internal driver borrows the passed-in pool: queries run on it, none is created, close leaves it."""
        create_pool = AsyncMock()
        monkeypatch.setattr("threetears.datasources.drivers.asyncpg_driver.asyncpg.create_pool", create_pool)
        external = _build_mock_pool(fetch_records=[{"x": 1}])
        driver = AsyncpgDriver(agent_internal_config, external_pool=external)
        assert await driver.fetch("SELECT 1") == [{"x": 1}]
        create_pool.assert_not_awaited()
        await driver.close()
        external.close.assert_not_called()

    @pytest.mark.asyncio
    async def test_a_borrowed_pool_is_waited_on_for_a_bounded_time(
        self, agent_internal_config: BorrowedPoolConnectionConfig
    ) -> None:
        """the host's pool may be busy with its own work; a query does not wait on it past its caller."""
        external = _build_mock_pool(fetch_records=[{"x": 1}])
        driver = AsyncpgDriver(agent_internal_config, external_pool=external)

        await driver.fetch("SELECT 1")

        assert external.acquire_options == [{"timeout": BORROWED_POOL_ACQUIRE_TIMEOUT_SECONDS}]
        assert BORROWED_POOL_ACQUIRE_TIMEOUT_SECONDS <= 30

    @pytest.mark.asyncio
    async def test_init_datasource_name_default_is_unknown(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """omitting ``datasource_name`` defaults to ``"unknown"``.

        the OTel metric label still tags emissions; ``"unknown"`` is
        the documented sentinel for callers who don't have the name
        in scope. a refusal names the datasource the same way, which
        is how this reads it.
        """
        with pytest.raises(DriverMissingCredentialError, match="datasource 'unknown'"):
            await AsyncpgDriver(_unresolvable_password_config(monkeypatch)).fetch("SELECT 1")

    @pytest.mark.asyncio
    async def test_init_datasource_name_captured(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """a passed ``datasource_name`` is the name the driver reports itself under."""
        driver = AsyncpgDriver(_unresolvable_password_config(monkeypatch), datasource_name="warehouse")
        with pytest.raises(DriverMissingCredentialError, match="datasource 'warehouse'"):
            await driver.fetch("SELECT 1")


class TestClose:
    """close() concurrency contract per DS-09-12 / DS-10-07."""

    @pytest.mark.asyncio
    async def test_close_idempotent(self, postgres_config: PostgresConnectionConfig) -> None:
        """second :meth:`close` call is a no-op (does NOT raise)."""
        driver = AsyncpgDriver(postgres_config)
        # no pool created yet, close should still work
        await driver.close()
        with pytest.raises(RuntimeError, match="closed"):
            await driver.fetch("SELECT 1")
        # second call: no-op
        await driver.close()

    @pytest.mark.asyncio
    async def test_close_owned_pool_calls_pool_close(
        self, postgres_config: PostgresConnectionConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """owned-pool path: :meth:`close` awaits ``pool.close()``."""
        pool = _build_mock_pool()
        driver = _driver_owning(pool, postgres_config, monkeypatch)
        # the owned pool exists once a query has created it.
        await driver.fetch("SELECT 1")
        await driver.close()
        pool.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_close_borrowed_pool_does_not_call_pool_close(
        self, agent_internal_config: BorrowedPoolConnectionConfig
    ) -> None:
        """borrowed-pool path: :meth:`close` MUST NOT close the pool."""
        pool = _build_mock_pool()
        driver = AsyncpgDriver(agent_internal_config, external_pool=pool)
        await driver.close()
        pool.close.assert_not_called()

    @pytest.mark.asyncio
    async def test_methods_reject_after_close(self, postgres_config: PostgresConnectionConfig) -> None:
        """every public method raises :class:`RuntimeError` post-close."""
        driver = AsyncpgDriver(postgres_config)
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
    async def test_fetch_iter_rejects_after_close(self, postgres_config: PostgresConnectionConfig) -> None:
        """:meth:`fetch_iter` (async generator) also raises post-close."""
        driver = AsyncpgDriver(postgres_config)
        await driver.close()
        with pytest.raises(RuntimeError, match="closed"):
            async for _row in driver.fetch_iter("SELECT 1"):
                pass  # pragma: no cover -- the for loop body never runs


# ---------------------------------------------------------------------------
# Query routing
# ---------------------------------------------------------------------------


class TestQueryRouting:
    """fetch/execute route through the mocked pool's acquired connection."""

    @pytest.mark.asyncio
    async def test_fetch_returns_dicts(
        self, postgres_config: PostgresConnectionConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """:meth:`fetch` returns the records as dicts."""
        pool = _build_mock_pool(fetch_records=[{"a": 1, "b": "x"}])
        driver = _driver_owning(pool, postgres_config, monkeypatch)
        rows = await driver.fetch("SELECT $1, $2", 1, "x")
        assert rows == [{"a": 1, "b": "x"}]
        # the connection mock's fetch should have been awaited with the
        # SQL unchanged ($N placeholders are asyncpg-native).
        pool.recorded_conn.fetch.assert_awaited_once_with("SELECT $1, $2", 1, "x")

    @pytest.mark.asyncio
    async def test_execute_routes_through_conn_execute(
        self, postgres_config: PostgresConnectionConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """:meth:`execute` calls ``conn.execute`` once."""
        pool = _build_mock_pool()
        driver = _driver_owning(pool, postgres_config, monkeypatch)
        await driver.execute("INSERT INTO t VALUES ($1)", 42)
        pool.recorded_conn.execute.assert_awaited_once_with("INSERT INTO t VALUES ($1)", 42)


class TestIntrospectionRouting:
    """list_tables / list_columns / table_hashes use the right SQL constants."""

    @pytest.mark.asyncio
    async def test_list_tables_uses_tables_sql(
        self, postgres_config: PostgresConnectionConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """:meth:`list_tables` reads base tables from ``information_schema`` with the allow-list bound."""
        pool = _build_mock_pool(fetch_records=[{"table_schema": "s1", "table_name": "t1", "selectable": True}])
        driver = _driver_owning(pool, postgres_config, monkeypatch)
        rows = await driver.list_tables(["s1"])
        assert rows == [{"table_schema": "s1", "table_name": "t1"}]
        sql = _single_catalog_query(pool, ["s1"])
        assert "FROM information_schema.tables" in sql
        assert "table_type = 'BASE TABLE'" in sql

    @pytest.mark.asyncio
    async def test_list_columns_uses_columns_sql_and_preserves_is_nullable(
        self, postgres_config: PostgresConnectionConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """:meth:`list_columns` preserves raw ``is_nullable`` (not bool)."""
        pool = _build_mock_pool(
            fetch_records=[
                {
                    "table_schema": "s1",
                    "table_name": "t1",
                    "column_name": "c1",
                    "data_type": "integer",
                    "is_nullable": "NO",
                    "ordinal_position": 1,
                    "selectable": True,
                }
            ]
        )
        driver = _driver_owning(pool, postgres_config, monkeypatch)
        rows = await driver.list_columns(["s1"])
        assert rows[0]["is_nullable"] == "NO"  # raw string, NOT bool
        assert isinstance(rows[0]["is_nullable"], str)
        sql = _single_catalog_query(pool, ["s1"])
        assert "FROM information_schema.columns" in sql
        assert "is_nullable" in sql

    @pytest.mark.asyncio
    async def test_table_hashes_returns_dict_keyed_by_schema_table(
        self, postgres_config: PostgresConnectionConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """:meth:`table_hashes` returns ``{(schema, table): digest}``."""
        pool = _build_mock_pool(
            fetch_records=[
                {
                    "table_schema": "s1",
                    "table_name": "t1",
                    "column_hash": "abc123",
                    "selectable": True,
                }
            ]
        )
        driver = _driver_owning(pool, postgres_config, monkeypatch)
        hashes = await driver.table_hashes(["s1"])
        assert hashes == {("s1", "t1"): "abc123"}
        sql = _single_catalog_query(pool, ["s1"])
        assert "FROM information_schema.columns" in sql
        assert "AS column_hash" in sql


#: the privilege test every catalog query applies: the datasource's own user can SELECT the relation,
#: asked by the relation's OID -- a name would be resolved against the CURRENT catalog, and a table
#: dropped or renamed while the catalog query runs would fail the whole query
_SELECTABLE = "has_table_privilege(current_user, rel.oid, 'SELECT')"

_LOGGER = "threetears.datasources.drivers.base"


def _column(table: str, selectable: bool | None, column: str = "c1") -> dict[str, Any]:
    """one ``list_columns`` record as the catalog query returns it, its privilege answer included."""
    return {
        "table_schema": "s1",
        "table_name": table,
        "column_name": column,
        "data_type": "integer",
        "is_nullable": "NO",
        "ordinal_position": 1,
        "selectable": selectable,
    }


def _left_out(caplog: pytest.LogCaptureFixture) -> list[str]:
    """the messages saying the catalog left relations out for want of a grant."""
    return [r.getMessage() for r in caplog.records if r.name == _LOGGER and "cannot SELECT" in r.getMessage()]


class TestIntrospectionCatalogsOnlySelectableTables:
    """a table the datasource user cannot SELECT is never catalogued, hashed or listed.

    the warehouse users are least-privilege by design: ``allowed_schemas`` names a whole
    schema, the grants name a few of its tables. a catalog that listed the rest sent every
    later read of them (the coverage probe, the schema tool) into ``42501`` permission denied.
    the live proof against a real engine is ``test_asyncpg_driver_live.py``'s.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize("method", ["list_tables", "list_columns", "table_hashes"])
    async def test_every_catalog_query_asks_the_select_grant_by_oid(
        self, postgres_config: PostgresConnectionConfig, monkeypatch: pytest.MonkeyPatch, method: str
    ) -> None:
        pool = _build_mock_pool(fetch_records=[])
        driver = _driver_owning(pool, postgres_config, monkeypatch)
        await getattr(driver, method)(["s1"])
        sql = _single_catalog_query(pool, ["s1"])
        assert _SELECTABLE in sql
        # never by name: a name is resolved against the current catalog, not the query's snapshot
        assert "quote_ident" not in sql

    @pytest.mark.asyncio
    async def test_list_tables_keeps_only_the_selectable_and_says_what_it_left_out(
        self,
        postgres_config: PostgresConnectionConfig,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        records = [
            {"table_schema": "s1", "table_name": "granted", "selectable": True},
            # a relation dropped while the query ran: its OID answers NULL, and it is simply gone
            {"table_schema": "s1", "table_name": "vanished", "selectable": None},
        ] + [{"table_schema": "s1", "table_name": f"ungranted_{i}", "selectable": False} for i in range(7)]
        pool = _build_mock_pool(fetch_records=records)
        driver = _driver_owning(pool, postgres_config, monkeypatch)
        with caplog.at_level("INFO", logger=_LOGGER):
            rows = await driver.list_tables(["s1"])
        assert rows == [{"table_schema": "s1", "table_name": "granted"}]
        (message,) = _left_out(caplog)
        assert "7 relation" in message
        assert "list_tables" in message
        # at most five examples, and never the vanished one
        assert message.count("s1.ungranted_") == 5
        assert "vanished" not in message

    @pytest.mark.asyncio
    async def test_list_columns_counts_relations_not_columns(
        self,
        postgres_config: PostgresConnectionConfig,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        records = [
            _column("granted", True),
            _column("ungranted", False, "a"),
            _column("ungranted", False, "b"),
            _column("vanished", None),
        ]
        pool = _build_mock_pool(fetch_records=records)
        driver = _driver_owning(pool, postgres_config, monkeypatch)
        with caplog.at_level("INFO", logger=_LOGGER):
            rows = await driver.list_columns(["s1"])
        assert [(r["table_name"], r["column_name"]) for r in rows] == [("granted", "c1")]
        assert "selectable" not in rows[0]
        (message,) = _left_out(caplog)
        assert "1 relation" in message
        assert "s1.ungranted" in message

    @pytest.mark.asyncio
    async def test_table_hashes_keeps_only_the_selectable(
        self,
        postgres_config: PostgresConnectionConfig,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        records = [
            {"table_schema": "s1", "table_name": "granted", "column_hash": "h1", "selectable": True},
            {"table_schema": "s1", "table_name": "ungranted", "column_hash": "h2", "selectable": False},
            {"table_schema": "s1", "table_name": "vanished", "column_hash": "h3", "selectable": None},
        ]
        pool = _build_mock_pool(fetch_records=records)
        driver = _driver_owning(pool, postgres_config, monkeypatch)
        with caplog.at_level("INFO", logger=_LOGGER):
            hashes = await driver.table_hashes(["s1"])
        assert hashes == {("s1", "granted"): "h1"}
        (message,) = _left_out(caplog)
        assert "table_hashes" in message

    @pytest.mark.asyncio
    async def test_nothing_left_out_logs_nothing(
        self,
        postgres_config: PostgresConnectionConfig,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        pool = _build_mock_pool(fetch_records=[{"table_schema": "s1", "table_name": "t1", "selectable": True}])
        driver = _driver_owning(pool, postgres_config, monkeypatch)
        with caplog.at_level("INFO", logger=_LOGGER):
            await driver.list_tables(["s1"])
        assert _left_out(caplog) == []


# ---------------------------------------------------------------------------
# test_connection sanitization (DS-10-06)
# ---------------------------------------------------------------------------


class TestTestConnection:
    """:meth:`test_connection` issues ``SELECT 1`` and sanitizes failures."""

    @pytest.mark.asyncio
    async def test_test_connection_happy_path(
        self, postgres_config: PostgresConnectionConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """successful round-trip returns None silently."""
        pool = _build_mock_pool(fetchval_value=1)
        driver = _driver_owning(pool, postgres_config, monkeypatch)
        # should not raise
        await driver.test_connection()
        pool.recorded_conn.fetchval.assert_awaited_once_with("SELECT 1")

    @pytest.mark.asyncio
    async def test_test_connection_sanitizes_failure(
        self, postgres_config: PostgresConnectionConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """backend failure surfaces as :class:`DriverConnectError`, no chain."""
        pool = _build_mock_pool()
        # seed a failure
        pool.recorded_conn.fetchval.side_effect = RuntimeError("kapow")
        driver = _driver_owning(pool, postgres_config, monkeypatch)
        with pytest.raises(DriverConnectError) as exc_info:
            await driver.test_connection()
        # ``from None`` MUST break the cause chain so the original
        # exception isn't reachable via ``__cause__``.
        assert exc_info.value.__cause__ is None
        # message carries host/port/db identity
        assert "localhost" in str(exc_info.value)
        assert "/x" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Borrowed-pool semantics (DS-10-03)
# ---------------------------------------------------------------------------


class TestBorrowedPool:
    """AGENT_INTERNAL config branch uses the external pool, doesn't close it."""

    @pytest.mark.asyncio
    async def test_external_pool_used_for_fetch(self, agent_internal_config: BorrowedPoolConnectionConfig) -> None:
        """fetch routes through the borrowed pool's acquired connection."""
        pool = _build_mock_pool(fetch_records=[{"x": 1}])
        driver = AsyncpgDriver(agent_internal_config, external_pool=pool)
        rows = await driver.fetch("SELECT 1")
        assert rows == [{"x": 1}]

    @pytest.mark.asyncio
    async def test_owns_pool_false_for_borrowed(
        self, agent_internal_config: BorrowedPoolConnectionConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """the borrowed path never creates a pool of its own, even after use."""
        create_pool = AsyncMock()
        monkeypatch.setattr("threetears.datasources.drivers.asyncpg_driver.asyncpg.create_pool", create_pool)
        pool = _build_mock_pool()
        driver = AsyncpgDriver(agent_internal_config, external_pool=pool)
        await driver.fetch("SELECT 1")
        await driver.execute("SELECT 1")
        await driver.close()
        create_pool.assert_not_awaited()
        pool.close.assert_not_called()

    @pytest.mark.asyncio
    async def test_close_does_not_close_borrowed_pool(
        self, agent_internal_config: BorrowedPoolConnectionConfig
    ) -> None:
        """borrowed pool is NOT closed by the driver's :meth:`close`."""
        pool = _build_mock_pool()
        driver = AsyncpgDriver(agent_internal_config, external_pool=pool)
        await driver.close()
        pool.close.assert_not_called()


# ---------------------------------------------------------------------------
# Placeholder translation surface (DS-10-04)
# ---------------------------------------------------------------------------


class TestPlaceholderPassthrough:
    """asyncpg sees $N-style placeholders unchanged (no-op translation)."""

    @pytest.mark.asyncio
    async def test_dollar_n_placeholder_passed_through_unchanged(
        self, postgres_config: PostgresConnectionConfig, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``$1, $2`` SQL is forwarded to ``conn.fetch`` verbatim."""
        pool = _build_mock_pool(fetch_records=[])
        driver = _driver_owning(pool, postgres_config, monkeypatch)
        await driver.fetch("SELECT $1, $2, $10")
        # the helper is a no-op for asyncpg style; the SQL passed to
        # the connection MUST match the input verbatim.
        pool.recorded_conn.fetch.assert_awaited_once_with("SELECT $1, $2, $10")


# ---------------------------------------------------------------------------
# Pool creation path (DS-10-02 / DS-10-10)
# ---------------------------------------------------------------------------


class TestPoolCreation:
    """``_create_owned_pool`` builds the pool with config-sourced sizing."""

    @pytest.mark.asyncio
    async def test_create_pool_uses_config_sizing(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """min/max/command_timeout come from the config, not literals."""
        # use a non-default config to detect literal-leaks: if any of
        # these end up as a Constant in the driver, the assertions
        # below will fail.
        cfg = PostgresConnectionConfig(
            datasource_type=DataSourceType.POSTGRES,
            host="db.example.com",
            port=5444,
            database="warehouse",
            username="ots",
            password_ref=None,
            pool_min_size=3,
            pool_max_size=11,
            command_timeout_seconds=42,
        )

        fake_pool = _build_mock_pool()
        create_pool_mock = AsyncMock(return_value=fake_pool)
        monkeypatch.setattr(
            "threetears.datasources.drivers.asyncpg_driver.asyncpg.create_pool",
            create_pool_mock,
        )

        driver = AsyncpgDriver(cfg)
        # trigger lazy pool creation via a fetch
        await driver.fetch("SELECT 1")

        # one call, with kwargs sourced from the config
        create_pool_mock.assert_awaited_once()
        await_args = create_pool_mock.await_args
        assert await_args is not None
        kwargs = await_args.kwargs
        assert kwargs["host"] == "db.example.com"
        assert kwargs["port"] == 5444
        assert kwargs["database"] == "warehouse"
        assert kwargs["user"] == "ots"
        assert kwargs["password"] is None  # password_ref=None
        assert kwargs["min_size"] == 3
        assert kwargs["max_size"] == 11
        assert kwargs["command_timeout"] == 42
        # carries the platform-default inactive lifetime
        assert "max_inactive_connection_lifetime" in kwargs

    @pytest.mark.asyncio
    async def test_create_pool_resolves_secret_str_to_plain_value(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """:class:`SecretStr` is unwrapped at the LAST moment for asyncpg."""
        monkeypatch.setenv("MY_PG_PW", "horse-battery-staple")
        cfg = PostgresConnectionConfig(
            datasource_type=DataSourceType.POSTGRES,
            host="h",
            database="x",
            username="u",
            password_ref="env://MY_PG_PW",
        )
        fake_pool = _build_mock_pool()
        create_pool_mock = AsyncMock(return_value=fake_pool)
        monkeypatch.setattr(
            "threetears.datasources.drivers.asyncpg_driver.asyncpg.create_pool",
            create_pool_mock,
        )
        driver = AsyncpgDriver(cfg)
        await driver.fetch("SELECT 1")
        create_pool_mock.assert_awaited_once()
        await_args = create_pool_mock.await_args
        assert await_args is not None
        kwargs = await_args.kwargs
        # the resolved password reaches asyncpg as a plain string
        assert kwargs["password"] == "horse-battery-staple"

    @pytest.mark.asyncio
    async def test_create_pool_failure_sanitized(
        self,
        monkeypatch: pytest.MonkeyPatch,
        postgres_config: PostgresConnectionConfig,
    ) -> None:
        """``create_pool`` failure wraps in :class:`DriverConnectError`."""
        create_pool_mock = AsyncMock(side_effect=RuntimeError("kapow"))
        monkeypatch.setattr(
            "threetears.datasources.drivers.asyncpg_driver.asyncpg.create_pool",
            create_pool_mock,
        )
        driver = AsyncpgDriver(postgres_config)
        with pytest.raises(DriverConnectError) as exc_info:
            await driver.fetch("SELECT 1")
        # ``from None`` breaks the cause chain
        assert exc_info.value.__cause__ is None
        # message carries identity but no backend internals
        assert "localhost" in str(exc_info.value)
        assert "kapow" not in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_a_refused_login_is_an_auth_error_carrying_the_servers_reason(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """the server's SQLSTATE and message survive the ``from None`` that drops the chain.

        Without them a refused login reads exactly like an unreachable host, and a caller
        that retries it is the thing that locks the account.
        """
        monkeypatch.setenv("MY_PG_PW", "horse-battery-staple")
        cfg = PostgresConnectionConfig(
            datasource_type=DataSourceType.POSTGRES,
            host="h",
            database="x",
            username="u",
            password_ref="env://MY_PG_PW",
        )
        monkeypatch.setattr(
            "threetears.datasources.drivers.asyncpg_driver.asyncpg.create_pool",
            AsyncMock(
                side_effect=asyncpg.exceptions.InvalidPasswordError('password authentication failed for user "u"')
            ),
        )
        driver = AsyncpgDriver(cfg)

        with pytest.raises(DriverAuthError) as exc_info:
            await driver.fetch("SELECT 1")

        assert exc_info.value.__cause__ is None
        assert exc_info.value.sqlstate == "28P01"
        assert 'password authentication failed for user "u"' in str(exc_info.value)
        assert "horse-battery-staple" not in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_a_reference_that_resolves_to_nothing_is_refused_before_any_connect(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """a named credential that is absent is a missing credential, not a network failure."""
        monkeypatch.delenv("MISSING_PG_PW", raising=False)
        cfg = PostgresConnectionConfig(
            datasource_type=DataSourceType.POSTGRES,
            host="h",
            database="x",
            username="u",
            password_ref="env://MISSING_PG_PW",
        )
        create_pool_mock = AsyncMock()
        monkeypatch.setattr(
            "threetears.datasources.drivers.asyncpg_driver.asyncpg.create_pool",
            create_pool_mock,
        )
        driver = AsyncpgDriver(cfg, datasource_name="reporting")

        with pytest.raises(DriverMissingCredentialError) as exc_info:
            await driver.fetch("SELECT 1")

        create_pool_mock.assert_not_awaited()
        assert "reporting" in str(exc_info.value)
        assert exc_info.value.__cause__ is None

    @pytest.mark.asyncio
    async def test_test_connection_keeps_the_auth_type(
        self,
        monkeypatch: pytest.MonkeyPatch,
        postgres_config: PostgresConnectionConfig,
    ) -> None:
        """the connection probe must not re-wrap an auth refusal into a plain connect error.

        That re-wrap dropped both the type a caller stops retrying on and the server's reason.
        """
        monkeypatch.setattr(
            "threetears.datasources.drivers.asyncpg_driver.asyncpg.create_pool",
            AsyncMock(
                side_effect=asyncpg.exceptions.InvalidPasswordError('password authentication failed for user "u"')
            ),
        )
        driver = AsyncpgDriver(postgres_config)

        with pytest.raises(DriverAuthError) as exc_info:
            await driver.test_connection()

        assert exc_info.value.sqlstate == "28P01"


class TestServerSettingsSearchPath:
    """``_create_owned_pool`` wires ``allowed_schemas`` -> startup ``search_path``.

    asyncpg's pool RESETs every released connection (``DISCARD ALL``)
    so a session-level ``SET search_path`` would not survive between
    acquires. instead we pass the value through ``server_settings``,
    which asyncpg sends in the pgwire STARTUP packet -- making it
    the connection's documented "session default", which RESET ALL /
    DISCARD ALL preserve. these tests pin the kwarg the driver hands
    to :func:`asyncpg.create_pool`.
    """

    @pytest.mark.asyncio
    async def test_server_settings_carries_search_path_when_allowed_schemas_non_empty(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """non-empty ``allowed_schemas`` -> ``server_settings['search_path']`` is set."""
        cfg = PostgresConnectionConfig(
            datasource_type=DataSourceType.POSTGRES,
            host="h",
            database="x",
            allowed_schemas=["reporting_prod", "audit"],
        )
        fake_pool = _build_mock_pool()
        create_pool_mock = AsyncMock(return_value=fake_pool)
        monkeypatch.setattr(
            "threetears.datasources.drivers.asyncpg_driver.asyncpg.create_pool",
            create_pool_mock,
        )
        driver = AsyncpgDriver(cfg)
        await driver.fetch("SELECT 1")
        await_args = create_pool_mock.await_args
        assert await_args is not None
        server_settings = await_args.kwargs.get("server_settings")
        assert server_settings == {"search_path": '"reporting_prod", "audit"'}

    @pytest.mark.asyncio
    async def test_no_server_settings_when_allowed_schemas_empty(
        self,
        monkeypatch: pytest.MonkeyPatch,
        postgres_config: PostgresConnectionConfig,
    ) -> None:
        """empty ``allowed_schemas`` -> ``server_settings`` is NOT passed.

        empty is the explicit signal "leave the backend default in
        place"; we must not send an empty server_settings dict (which
        would still hit the startup-parameter code path and be a
        latent gotcha).
        """
        # default fixture has allowed_schemas=[]
        assert postgres_config.allowed_schemas == []
        fake_pool = _build_mock_pool()
        create_pool_mock = AsyncMock(return_value=fake_pool)
        monkeypatch.setattr(
            "threetears.datasources.drivers.asyncpg_driver.asyncpg.create_pool",
            create_pool_mock,
        )
        driver = AsyncpgDriver(postgres_config)
        await driver.fetch("SELECT 1")
        await_args = create_pool_mock.await_args
        assert await_args is not None
        # server_settings must be absent
        assert "server_settings" not in await_args.kwargs

    @pytest.mark.asyncio
    async def test_server_settings_quotes_schema_names_safely(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """adversarial schema names are identifier-quoted via the shared helper."""
        cfg = PostgresConnectionConfig(
            datasource_type=DataSourceType.POSTGRES,
            host="h",
            database="x",
            allowed_schemas=['my"schema'],
        )
        fake_pool = _build_mock_pool()
        create_pool_mock = AsyncMock(return_value=fake_pool)
        monkeypatch.setattr(
            "threetears.datasources.drivers.asyncpg_driver.asyncpg.create_pool",
            create_pool_mock,
        )
        driver = AsyncpgDriver(cfg)
        await driver.fetch("SELECT 1")
        server_settings = create_pool_mock.await_args.kwargs["server_settings"]
        assert server_settings == {"search_path": '"my""schema"'}

    @pytest.mark.asyncio
    async def test_server_settings_applied_for_yugabyte_config(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """the same code path applies to :class:`YugabyteConnectionConfig`."""
        cfg = YugabyteConnectionConfig(
            datasource_type=DataSourceType.YUGABYTE,
            host="h",
            database="x",
            allowed_schemas=["app"],
        )
        fake_pool = _build_mock_pool()
        create_pool_mock = AsyncMock(return_value=fake_pool)
        monkeypatch.setattr(
            "threetears.datasources.drivers.asyncpg_driver.asyncpg.create_pool",
            create_pool_mock,
        )
        driver = AsyncpgDriver(cfg)
        await driver.fetch("SELECT 1")
        server_settings = create_pool_mock.await_args.kwargs["server_settings"]
        assert server_settings == {"search_path": '"app"'}


# ---------------------------------------------------------------------------
# Borrowed-pool search_path scoping
# ---------------------------------------------------------------------------


class TestABorrowedConnectionIsScopedToItsSchema:
    """an ``agent_internal`` driver shares the Hub's L3 pool and must scope itself.

    A driver that opens its own pool sends ``search_path`` in the pgwire STARTUP
    packet, which is why it survives the ``RESET ALL`` asyncpg issues on release.
    A BORROWED pool never sends a startup packet, so before this the
    ``agent_internal`` driver inherited the Hub's own ``search_path`` and had no
    scoping at all.

    Observed on cobalt-dev: an unqualified ``SELECT count(*) FROM users`` against
    an ``agent_internal`` datasource raised ``UndefinedTableError`` while the
    identical query fully qualified returned rows -- a datasource that did not
    route to the schema its own name advertises. The Hub-side docstring asserted
    a "per-query SET search_path" that existed nowhere in this driver;
    ``schema_name`` was read in one place, to build a display string.
    """

    @pytest.mark.asyncio
    async def test_fetch_sets_search_path_on_the_borrowed_connection(
        self,
        agent_internal_config: BorrowedPoolConnectionConfig,
    ) -> None:
        """
        :return: nothing
        :rtype: None
        """
        fake_pool = _build_mock_pool()
        driver = AsyncpgDriver(agent_internal_config, external_pool=fake_pool)

        await driver.fetch("SELECT 1")

        conn = fake_pool.recorded_conn
        executed = [call.args[0] for call in conn.execute.await_args_list]
        assert 'SET search_path TO "agent_abc123"' in executed

    @pytest.mark.asyncio
    async def test_the_schema_name_is_identifier_quoted(
        self,
    ) -> None:
        """The name reaches SQL as an identifier, not as interpolated text.

        ``schema_name`` is operator-controlled rather than caller-controlled, so
        this is defence in depth rather than the primary boundary -- but a
        driver that interpolates a name into DDL-adjacent SQL should quote it,
        and ``build_search_path_value`` already does.

        :return: nothing
        :rtype: None
        """
        cfg = BorrowedPoolConnectionConfig(
            datasource_type=DataSourceType.AGENT_INTERNAL,
            schema_name='weird"name',
        )
        fake_pool = _build_mock_pool()
        driver = AsyncpgDriver(cfg, external_pool=fake_pool)

        await driver.fetch("SELECT 1")

        conn = fake_pool.recorded_conn
        executed = [call.args[0] for call in conn.execute.await_args_list]
        assert any('"weird""name"' in statement for statement in executed)

    @pytest.mark.asyncio
    async def test_an_owned_pool_is_not_re_scoped_every_acquire(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The startup packet already did it; repeating it is a wasted round trip.

        :return: nothing
        :rtype: None
        """
        cfg = PostgresConnectionConfig(
            datasource_type=DataSourceType.POSTGRES,
            host="h",
            database="x",
            allowed_schemas=["app"],
        )
        fake_pool = _build_mock_pool()
        monkeypatch.setattr(
            "threetears.datasources.drivers.asyncpg_driver.asyncpg.create_pool",
            AsyncMock(return_value=fake_pool),
        )
        driver = AsyncpgDriver(cfg)

        await driver.fetch("SELECT 1")

        conn = fake_pool.recorded_conn
        executed = [call.args[0] for call in conn.execute.await_args_list]
        assert not any("search_path" in statement for statement in executed)


class TestFetchAtMost:
    """a read that stops after ``max_rows`` rows: the rest are never fetched from the server."""

    @pytest.mark.asyncio
    async def test_rows_are_read_through_a_cursor_that_fetches_no_more_than_asked(
        self, agent_internal_config: BorrowedPoolConnectionConfig
    ) -> None:
        external = _build_mock_pool()
        conn = external.recorded_conn
        cursor = MagicMock(name="MockCursor")
        cursor.fetch = AsyncMock(return_value=[{"n": 0}, {"n": 1}, {"n": 2}])
        opened: list[tuple[Any, ...]] = []

        async def open_cursor(*args: Any) -> Any:
            opened.append(args)
            return cursor

        conn.cursor = MagicMock(side_effect=lambda *args: open_cursor(*args))

        @asynccontextmanager
        async def transaction() -> Any:
            yield None

        conn.transaction = MagicMock(side_effect=transaction)
        driver = AsyncpgDriver(agent_internal_config, external_pool=external)

        rows = await driver.fetch_at_most("SELECT n FROM big WHERE a = $1", 7, max_rows=3)

        assert rows == [{"n": 0}, {"n": 1}, {"n": 2}]
        assert opened == [("SELECT n FROM big WHERE a = $1", 7)]
        cursor.fetch.assert_awaited_once_with(3)
        conn.fetch.assert_not_called()

    @pytest.mark.asyncio
    async def test_a_bound_below_one_is_refused(self, agent_internal_config: BorrowedPoolConnectionConfig) -> None:
        driver = AsyncpgDriver(agent_internal_config, external_pool=_build_mock_pool())
        with pytest.raises(ValueError, match="max_rows"):
            await driver.fetch_at_most("SELECT 1", max_rows=0)


class TestABorrowedPoolThatHasNoConnectionToSpare:
    @pytest.mark.asyncio
    async def test_a_timed_out_wait_for_a_connection_says_the_pool_is_busy_not_that_a_statement_timed_out(
        self, agent_internal_config: BorrowedPoolConnectionConfig
    ) -> None:
        external = _build_mock_pool()

        @asynccontextmanager
        async def exhausted(**options: Any) -> Any:
            raise TimeoutError
            yield  # pragma: no cover - never reached

        external.acquire = exhausted
        driver = AsyncpgDriver(agent_internal_config, external_pool=external)

        with pytest.raises(DriverPoolBusyError):
            await driver.fetch("SELECT 1")
        with pytest.raises(DriverPoolBusyError):
            await driver.fetch_at_most("SELECT 1", max_rows=2)

    @pytest.mark.asyncio
    async def test_every_way_into_a_borrowed_pool_says_busy_when_it_has_no_connection(
        self, agent_internal_config: BorrowedPoolConnectionConfig
    ) -> None:
        """a transaction and a streamed read take their connection by other routes than fetch; each must say busy."""

        class _Exhausted:
            """a pool's acquire that runs out of time whether it is awaited or entered."""

            def __await__(self) -> Any:
                raise TimeoutError
                yield  # pragma: no cover - never reached

            async def __aenter__(self) -> Any:
                raise TimeoutError

            async def __aexit__(self, *exc_info: Any) -> bool:
                return False

        external = _build_mock_pool()
        external.acquire = lambda **options: _Exhausted()
        driver = AsyncpgDriver(agent_internal_config, external_pool=external)

        with pytest.raises(DriverPoolBusyError):
            await driver.begin()
        with pytest.raises(DriverPoolBusyError):
            async for _row in driver.fetch_iter("SELECT 1"):
                pass  # pragma: no cover - the acquire fails before any row
