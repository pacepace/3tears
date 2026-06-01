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

import pytest

from threetears.datasources.config import (
    AgentInternalConnectionConfig,
    PostgresConnectionConfig,
    YugabyteConnectionConfig,
)
from threetears.datasources.drivers.asyncpg_driver import (
    AsyncpgDriver,
    DriverConnectError,
    _POSTGRES_COLUMNS_SQL,
    _POSTGRES_TABLE_HASHES_SQL,
    _POSTGRES_TABLES_SQL,
)
from threetears.datasources.entities import DataSourceType


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
    Connection mock with fetch/execute/fetchval/cancel/cursor/transaction
    coroutines wired up.

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
    conn = MagicMock(name="MockConn")
    conn.fetch = AsyncMock(return_value=records)
    conn.execute = AsyncMock(return_value=None)
    conn.fetchval = AsyncMock(return_value=fetchval_value)
    conn.cancel = MagicMock(return_value=None)
    conn.terminate = MagicMock(return_value=None)

    # async-context-manager shape for ``pool.acquire()``
    @asynccontextmanager
    async def _acquire() -> Any:
        yield conn

    pool.acquire = _acquire
    pool.close = AsyncMock(return_value=None)
    pool.is_closing = MagicMock(return_value=False)

    # surface the connection mock so tests can assert against it
    pool._conn = conn  # noqa: SLF001 - test surface only
    return pool


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
def agent_internal_config() -> AgentInternalConnectionConfig:
    """default :class:`AgentInternalConnectionConfig`."""
    return AgentInternalConnectionConfig(
        datasource_type=DataSourceType.AGENT_INTERNAL,
        schema_name="agent_abc123",
    )


# ---------------------------------------------------------------------------
# Construction + lifecycle
# ---------------------------------------------------------------------------


class TestConstruction:
    """``__init__`` stores config + external_pool correctly; no I/O."""

    def test_init_postgres_no_external_pool(self, postgres_config: PostgresConnectionConfig) -> None:
        """constructing a postgres driver does NOT open a pool eagerly."""
        driver = AsyncpgDriver(postgres_config)
        assert driver._config is postgres_config  # noqa: SLF001
        assert driver._pool is None  # noqa: SLF001
        assert driver._owns_pool is True  # noqa: SLF001
        assert driver._closed is False  # noqa: SLF001

    def test_init_agent_internal_with_external_pool(self, agent_internal_config: AgentInternalConnectionConfig) -> None:
        """agent-internal driver borrows the passed-in pool."""
        external = _build_mock_pool()
        driver = AsyncpgDriver(agent_internal_config, external_pool=external)
        assert driver._pool is external  # noqa: SLF001
        assert driver._owns_pool is False  # noqa: SLF001

    def test_init_datasource_name_default_is_unknown(self, postgres_config: PostgresConnectionConfig) -> None:
        """omitting ``datasource_name`` defaults to ``"unknown"``.

        the OTel metric label still tags emissions; ``"unknown"`` is
        the documented sentinel for callers who don't have the name
        in scope.
        """
        driver = AsyncpgDriver(postgres_config)
        assert driver._datasource_name == "unknown"  # noqa: SLF001

    def test_init_datasource_name_captured(self, postgres_config: PostgresConnectionConfig) -> None:
        """passing ``datasource_name`` stores it for metric tagging."""
        driver = AsyncpgDriver(postgres_config, datasource_name="warehouse")
        assert driver._datasource_name == "warehouse"  # noqa: SLF001


class TestClose:
    """close() concurrency contract per DS-09-12 / DS-10-07."""

    @pytest.mark.asyncio
    async def test_close_idempotent(self, postgres_config: PostgresConnectionConfig) -> None:
        """second :meth:`close` call is a no-op (does NOT raise)."""
        driver = AsyncpgDriver(postgres_config)
        # no pool created yet, close should still work
        await driver.close()
        assert driver._closed is True  # noqa: SLF001
        # second call: no-op
        await driver.close()

    @pytest.mark.asyncio
    async def test_close_owned_pool_calls_pool_close(self, postgres_config: PostgresConnectionConfig) -> None:
        """owned-pool path: :meth:`close` awaits ``pool.close()``."""
        pool = _build_mock_pool()
        driver = AsyncpgDriver(postgres_config)
        driver._pool = pool  # noqa: SLF001 - inject mock
        await driver.close()
        pool.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_close_borrowed_pool_does_not_call_pool_close(
        self, agent_internal_config: AgentInternalConnectionConfig
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
    async def test_fetch_returns_dicts(self, postgres_config: PostgresConnectionConfig) -> None:
        """:meth:`fetch` returns the records as dicts."""
        pool = _build_mock_pool(fetch_records=[{"a": 1, "b": "x"}])
        driver = AsyncpgDriver(postgres_config)
        driver._pool = pool  # noqa: SLF001
        rows = await driver.fetch("SELECT $1, $2", 1, "x")
        assert rows == [{"a": 1, "b": "x"}]
        # the connection mock's fetch should have been awaited with the
        # SQL unchanged ($N placeholders are asyncpg-native).
        pool._conn.fetch.assert_awaited_once_with(  # noqa: SLF001
            "SELECT $1, $2", 1, "x"
        )

    @pytest.mark.asyncio
    async def test_execute_routes_through_conn_execute(self, postgres_config: PostgresConnectionConfig) -> None:
        """:meth:`execute` calls ``conn.execute`` once."""
        pool = _build_mock_pool()
        driver = AsyncpgDriver(postgres_config)
        driver._pool = pool  # noqa: SLF001
        await driver.execute("INSERT INTO t VALUES ($1)", 42)
        pool._conn.execute.assert_awaited_once_with(  # noqa: SLF001
            "INSERT INTO t VALUES ($1)", 42
        )


class TestIntrospectionRouting:
    """list_tables / list_columns / table_hashes use the right SQL constants."""

    @pytest.mark.asyncio
    async def test_list_tables_uses_tables_sql(self, postgres_config: PostgresConnectionConfig) -> None:
        """:meth:`list_tables` calls fetch with :data:`_POSTGRES_TABLES_SQL`."""
        pool = _build_mock_pool(fetch_records=[{"table_schema": "s1", "table_name": "t1"}])
        driver = AsyncpgDriver(postgres_config)
        driver._pool = pool  # noqa: SLF001
        rows = await driver.list_tables(["s1"])
        assert rows == [{"table_schema": "s1", "table_name": "t1"}]
        pool._conn.fetch.assert_awaited_once_with(  # noqa: SLF001
            _POSTGRES_TABLES_SQL, ["s1"]
        )

    @pytest.mark.asyncio
    async def test_list_columns_uses_columns_sql_and_preserves_is_nullable(
        self, postgres_config: PostgresConnectionConfig
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
                }
            ]
        )
        driver = AsyncpgDriver(postgres_config)
        driver._pool = pool  # noqa: SLF001
        rows = await driver.list_columns(["s1"])
        assert rows[0]["is_nullable"] == "NO"  # raw string, NOT bool
        assert isinstance(rows[0]["is_nullable"], str)
        pool._conn.fetch.assert_awaited_once_with(  # noqa: SLF001
            _POSTGRES_COLUMNS_SQL, ["s1"]
        )

    @pytest.mark.asyncio
    async def test_table_hashes_returns_dict_keyed_by_schema_table(
        self, postgres_config: PostgresConnectionConfig
    ) -> None:
        """:meth:`table_hashes` returns ``{(schema, table): digest}``."""
        pool = _build_mock_pool(
            fetch_records=[
                {
                    "table_schema": "s1",
                    "table_name": "t1",
                    "column_hash": "abc123",
                }
            ]
        )
        driver = AsyncpgDriver(postgres_config)
        driver._pool = pool  # noqa: SLF001
        hashes = await driver.table_hashes(["s1"])
        assert hashes == {("s1", "t1"): "abc123"}
        pool._conn.fetch.assert_awaited_once_with(  # noqa: SLF001
            _POSTGRES_TABLE_HASHES_SQL, ["s1"]
        )


# ---------------------------------------------------------------------------
# test_connection sanitization (DS-10-06)
# ---------------------------------------------------------------------------


class TestTestConnection:
    """:meth:`test_connection` issues ``SELECT 1`` and sanitizes failures."""

    @pytest.mark.asyncio
    async def test_test_connection_happy_path(self, postgres_config: PostgresConnectionConfig) -> None:
        """successful round-trip returns None silently."""
        pool = _build_mock_pool(fetchval_value=1)
        driver = AsyncpgDriver(postgres_config)
        driver._pool = pool  # noqa: SLF001
        # should not raise
        await driver.test_connection()
        pool._conn.fetchval.assert_awaited_once_with("SELECT 1")  # noqa: SLF001

    @pytest.mark.asyncio
    async def test_test_connection_sanitizes_failure(self, postgres_config: PostgresConnectionConfig) -> None:
        """backend failure surfaces as :class:`DriverConnectError`, no chain."""
        pool = _build_mock_pool()
        # seed a failure
        pool._conn.fetchval.side_effect = RuntimeError(  # noqa: SLF001
            "kapow"
        )
        driver = AsyncpgDriver(postgres_config)
        driver._pool = pool  # noqa: SLF001
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
    async def test_external_pool_used_for_fetch(self, agent_internal_config: AgentInternalConnectionConfig) -> None:
        """fetch routes through the borrowed pool's acquired connection."""
        pool = _build_mock_pool(fetch_records=[{"x": 1}])
        driver = AsyncpgDriver(agent_internal_config, external_pool=pool)
        rows = await driver.fetch("SELECT 1")
        assert rows == [{"x": 1}]

    @pytest.mark.asyncio
    async def test_owns_pool_false_for_borrowed(self, agent_internal_config: AgentInternalConnectionConfig) -> None:
        """``_owns_pool`` flag is False for the borrowed path."""
        pool = _build_mock_pool()
        driver = AsyncpgDriver(agent_internal_config, external_pool=pool)
        assert driver._owns_pool is False  # noqa: SLF001

    @pytest.mark.asyncio
    async def test_close_does_not_close_borrowed_pool(
        self, agent_internal_config: AgentInternalConnectionConfig
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
        self, postgres_config: PostgresConnectionConfig
    ) -> None:
        """``$1, $2`` SQL is forwarded to ``conn.fetch`` verbatim."""
        pool = _build_mock_pool(fetch_records=[])
        driver = AsyncpgDriver(postgres_config)
        driver._pool = pool  # noqa: SLF001
        await driver.fetch("SELECT $1, $2, $10")
        # the helper is a no-op for asyncpg style; the SQL passed to
        # the connection MUST match the input verbatim.
        pool._conn.fetch.assert_awaited_once_with(  # noqa: SLF001
            "SELECT $1, $2, $10"
        )


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
