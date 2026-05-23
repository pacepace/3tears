"""tests for the :class:`SnowflakeDriver` stub (``datasource-task-12``).

verifies the ABC contract holds even for the not-yet-implemented
driver: subclass relationship, every abstract method overridden,
every method raises :class:`NotImplementedError` with an actionable
message. minimal but load-bearing -- the stub-driver contract is
what makes the ABC's shape claim ("supports stateful-pooled DB-API
backends as well as stateless-HTTPS ones") concrete today rather
than speculative.
"""

from __future__ import annotations

import pytest

from threetears.datasources.config import SnowflakeConnectionConfig
from threetears.datasources.drivers.base import Driver
from threetears.datasources.drivers.snowflake_driver import SnowflakeDriver
from threetears.datasources.entities import DataSourceType


@pytest.fixture
def snowflake_config() -> SnowflakeConnectionConfig:
    """minimal valid SnowflakeConnectionConfig for stub-construction tests."""
    return SnowflakeConnectionConfig(
        datasource_type=DataSourceType.SNOWFLAKE,
        account="acct-12345",
        warehouse="WH_TEST",
        user="testuser",
        password_env="TEST_SF_PW",
    )


class TestStubContract:
    """ABC subclass + abstract-methods coverage."""

    def test_is_driver_subclass(
        self, snowflake_config: SnowflakeConnectionConfig
    ) -> None:
        """the stub is a real :class:`Driver` subclass."""
        driver = SnowflakeDriver(snowflake_config)
        assert isinstance(driver, Driver)

    def test_abstractmethods_empty(self) -> None:
        """every abstract method is overridden (stub or not)."""
        assert SnowflakeDriver.__abstractmethods__ == frozenset()

    def test_init_validates_config(
        self, snowflake_config: SnowflakeConnectionConfig
    ) -> None:
        """``__init__`` stores the config without backend I/O."""
        driver = SnowflakeDriver(snowflake_config)
        assert driver._config is snowflake_config  # noqa: SLF001

    def test_init_datasource_name_default_is_unknown(
        self, snowflake_config: SnowflakeConnectionConfig
    ) -> None:
        """default ``datasource_name`` matches the asyncpg / redshift contract."""
        driver = SnowflakeDriver(snowflake_config)
        assert driver._datasource_name == "unknown"  # noqa: SLF001

    def test_init_datasource_name_captured(
        self, snowflake_config: SnowflakeConnectionConfig
    ) -> None:
        """passing ``datasource_name`` is stored for the future metric path."""
        driver = SnowflakeDriver(snowflake_config, datasource_name="sf-prod")
        assert driver._datasource_name == "sf-prod"  # noqa: SLF001


class TestStubMethodsRaiseNotImplemented:
    """every public method raises :class:`NotImplementedError` with an actionable message.

    the message MUST name the method + point at the roadmap doc so a
    caller hitting the stub gets a clear "this is intentional; here's
    where to look" signal instead of an opaque crash.
    """

    @pytest.fixture
    def driver(
        self, snowflake_config: SnowflakeConnectionConfig
    ) -> SnowflakeDriver:
        return SnowflakeDriver(snowflake_config)

    @pytest.mark.asyncio
    async def test_fetch_raises(self, driver: SnowflakeDriver) -> None:
        with pytest.raises(NotImplementedError, match="SnowflakeDriver.fetch"):
            await driver.fetch("SELECT 1")

    @pytest.mark.asyncio
    async def test_execute_raises(self, driver: SnowflakeDriver) -> None:
        with pytest.raises(NotImplementedError, match="SnowflakeDriver.execute"):
            await driver.execute("CREATE TABLE x (i INT)")

    @pytest.mark.asyncio
    async def test_list_tables_raises(
        self, driver: SnowflakeDriver
    ) -> None:
        with pytest.raises(
            NotImplementedError, match="SnowflakeDriver.list_tables"
        ):
            await driver.list_tables(["public"])

    @pytest.mark.asyncio
    async def test_list_columns_raises(
        self, driver: SnowflakeDriver
    ) -> None:
        with pytest.raises(
            NotImplementedError, match="SnowflakeDriver.list_columns"
        ):
            await driver.list_columns(["public"])

    @pytest.mark.asyncio
    async def test_table_hashes_raises(
        self, driver: SnowflakeDriver
    ) -> None:
        with pytest.raises(
            NotImplementedError, match="SnowflakeDriver.table_hashes"
        ):
            await driver.table_hashes(["public"])

    @pytest.mark.asyncio
    async def test_test_connection_raises(
        self, driver: SnowflakeDriver
    ) -> None:
        with pytest.raises(
            NotImplementedError, match="SnowflakeDriver.test_connection"
        ):
            await driver.test_connection()

    @pytest.mark.asyncio
    async def test_close_raises(self, driver: SnowflakeDriver) -> None:
        with pytest.raises(NotImplementedError, match="SnowflakeDriver.close"):
            await driver.close()

    @pytest.mark.asyncio
    async def test_error_messages_reference_roadmap(
        self, driver: SnowflakeDriver
    ) -> None:
        """every stub error names the doc / docstring as the next step.

        catches "I'll fix this stub later" silent partial
        implementations that ship with an empty message.
        """
        with pytest.raises(NotImplementedError) as exc_info:
            await driver.fetch("SELECT 1")
        msg = str(exc_info.value)
        assert "module docstring" in msg or "datasource-task-12" in msg


class TestStubDoesNotImportBackend:
    """importing the stub module MUST NOT pull ``snowflake.connector`` in.

    the stub doesn't need the backend lib -- when the impl lands, the
    backend import goes in then. shipping the stub with an eager
    backend import would break the lazy-import contract verified by
    :mod:`tests.unit.test_lazy_imports`.
    """

    def test_snowflake_connector_not_loaded_after_stub_import(self) -> None:
        import sys
        # the stub may already be imported by other tests; we just
        # verify the backend lib isn't in sys.modules as a side effect
        # of importing the stub module itself.
        import threetears.datasources.drivers.snowflake_driver  # noqa: F401
        assert "snowflake.connector" not in sys.modules
