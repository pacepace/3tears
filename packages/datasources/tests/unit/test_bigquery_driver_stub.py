"""tests for the :class:`BigQueryDriver` stub (``datasource-task-12``).

paired with :mod:`tests.unit.test_snowflake_driver_stub`; the two
stubs together prove the :class:`Driver` ABC fits both stateful-
pooled (Snowflake) and stateless-HTTPS (BigQuery) backends. minimal
but load-bearing -- the stub-driver contract is what makes the
ABC's shape-claim concrete today rather than speculative.
"""

from __future__ import annotations

import pytest

from threetears.datasources.config import BigQueryConnectionConfig
from threetears.datasources.drivers.base import Driver
from threetears.datasources.drivers.bigquery_driver import BigQueryDriver
from threetears.datasources.entities import DataSourceType


@pytest.fixture
def bigquery_config() -> BigQueryConnectionConfig:
    """minimal valid BigQueryConnectionConfig for stub-construction tests."""
    return BigQueryConnectionConfig(
        datasource_type=DataSourceType.BIGQUERY,
        project_id="my-test-project",
        credentials_json_ref="env://TEST_GCP_SA_JSON",
    )


class TestStubContract:
    """ABC subclass + abstract-methods coverage."""

    def test_is_driver_subclass(self, bigquery_config: BigQueryConnectionConfig) -> None:
        """the stub is a real :class:`Driver` subclass."""
        driver = BigQueryDriver(bigquery_config)
        assert isinstance(driver, Driver)

    def test_abstractmethods_empty(self) -> None:
        """every abstract method is overridden (stub or not)."""
        assert BigQueryDriver.__abstractmethods__ == frozenset()

    def test_init_validates_config(self, bigquery_config: BigQueryConnectionConfig) -> None:
        """``__init__`` stores the config without backend I/O."""
        driver = BigQueryDriver(bigquery_config)
        assert driver._config is bigquery_config  # noqa: SLF001

    def test_init_datasource_name_default_is_unknown(self, bigquery_config: BigQueryConnectionConfig) -> None:
        """default ``datasource_name`` matches the asyncpg / redshift contract."""
        driver = BigQueryDriver(bigquery_config)
        assert driver._datasource_name == "unknown"  # noqa: SLF001

    def test_init_datasource_name_captured(self, bigquery_config: BigQueryConnectionConfig) -> None:
        """passing ``datasource_name`` is stored for the future metric path."""
        driver = BigQueryDriver(bigquery_config, datasource_name="bq-marts")
        assert driver._datasource_name == "bq-marts"  # noqa: SLF001


class TestStubMethodsRaiseNotImplemented:
    """every public method raises :class:`NotImplementedError` with an actionable message."""

    @pytest.fixture
    def driver(self, bigquery_config: BigQueryConnectionConfig) -> BigQueryDriver:
        return BigQueryDriver(bigquery_config)

    @pytest.mark.asyncio
    async def test_fetch_raises(self, driver: BigQueryDriver) -> None:
        with pytest.raises(NotImplementedError, match="BigQueryDriver.fetch"):
            await driver.fetch("SELECT 1")

    @pytest.mark.asyncio
    async def test_execute_raises(self, driver: BigQueryDriver) -> None:
        with pytest.raises(NotImplementedError, match="BigQueryDriver.execute"):
            await driver.execute("CREATE TABLE x (i INT)")

    @pytest.mark.asyncio
    async def test_list_tables_raises(self, driver: BigQueryDriver) -> None:
        with pytest.raises(NotImplementedError, match="BigQueryDriver.list_tables"):
            await driver.list_tables(["my_dataset"])

    @pytest.mark.asyncio
    async def test_list_columns_raises(self, driver: BigQueryDriver) -> None:
        with pytest.raises(NotImplementedError, match="BigQueryDriver.list_columns"):
            await driver.list_columns(["my_dataset"])

    @pytest.mark.asyncio
    async def test_table_hashes_raises(self, driver: BigQueryDriver) -> None:
        with pytest.raises(NotImplementedError, match="BigQueryDriver.table_hashes"):
            await driver.table_hashes(["my_dataset"])

    @pytest.mark.asyncio
    async def test_test_connection_raises(self, driver: BigQueryDriver) -> None:
        with pytest.raises(NotImplementedError, match="BigQueryDriver.test_connection"):
            await driver.test_connection()

    @pytest.mark.asyncio
    async def test_close_raises(self, driver: BigQueryDriver) -> None:
        with pytest.raises(NotImplementedError, match="BigQueryDriver.close"):
            await driver.close()

    @pytest.mark.asyncio
    async def test_error_messages_reference_roadmap(self, driver: BigQueryDriver) -> None:
        """every stub error names the doc / docstring as the next step."""
        with pytest.raises(NotImplementedError) as exc_info:
            await driver.fetch("SELECT 1")
        msg = str(exc_info.value)
        assert "module docstring" in msg or "datasource-task-12" in msg


class TestStubDoesNotImportBackend:
    """importing the stub module MUST NOT pull ``google.cloud.bigquery`` in."""

    def test_google_cloud_bigquery_not_loaded_after_stub_import(self) -> None:
        import sys
        import threetears.datasources.drivers.bigquery_driver  # noqa: F401

        assert "google.cloud.bigquery" not in sys.modules
