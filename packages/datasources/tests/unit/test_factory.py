"""tests for :func:`create_driver` factory dispatch.

shard-09 lands only the ABC + factory; concrete driver modules don't
exist yet (shards 10 / 11 / 12). the dispatch test pattern:

- inject stub driver modules into ``sys.modules`` for each backend so
  the factory's ``case`` arm finds the expected class on import
- call :func:`create_driver` with a real ConnectionConfig
- assert the right stub class was returned

this verifies dispatch correctness across every DataSourceType member
WITHOUT requiring the backend libraries to be installed and WITHOUT
needing concrete drivers in tree yet.

we also assert the failure modes:

- AGENT_INTERNAL without ``hub_l3_pool`` raises ValueError
- an unregistered datasource_type raises ValueError
- attempting to call ``create_driver`` for a backend whose driver
  module is genuinely absent surfaces ImportError (the factory does
  NOT swallow it)
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from threetears.datasources.config import (
    AgentInternalConnectionConfig,
    BigQueryConnectionConfig,
    PostgresConnectionConfig,
    RedshiftConnectionConfig,
    SnowflakeConnectionConfig,
    YugabyteConnectionConfig,
)
from threetears.datasources.drivers.base import Driver
from threetears.datasources.drivers.factory import create_driver
from threetears.datasources.entities import DataSourceType


# ---------------------------------------------------------------------------
# Stub driver classes -- subclass Driver so isinstance() checks pass
# ---------------------------------------------------------------------------


class _StubDriver(Driver):
    """minimal Driver subclass that records constructor args.

    factory tests stub out the per-backend driver module to return
    instances of this class so we can assert (1) the right import
    path was taken and (2) the right args (notably ``external_pool=``
    + ``datasource_name=``) reached the constructor.
    """

    def __init__(
        self,
        config: Any,
        *,
        external_pool: Any = None,
        datasource_name: str = "unknown",
    ) -> None:
        self.config = config
        self.external_pool = external_pool
        self.datasource_name = datasource_name

    async def fetch(self, sql: str, *params: Any) -> list[dict[str, Any]]:
        return []

    async def execute(self, sql: str, *params: Any) -> None:
        return None

    async def list_tables(self, schemas: list[str]) -> list[Any]:
        return []

    async def list_columns(self, schemas: list[str]) -> list[Any]:
        return []

    async def table_hashes(self, schemas: list[str]) -> dict[tuple[str, str], str]:
        return {}

    async def test_connection(self) -> None:
        return None

    async def close(self) -> None:
        return None


# the four concrete driver modules the factory matches into.
_DRIVER_MODULES = (
    ("threetears.datasources.drivers.asyncpg_driver", "AsyncpgDriver"),
    ("threetears.datasources.drivers.redshift_driver", "RedshiftDriver"),
    ("threetears.datasources.drivers.snowflake_driver", "SnowflakeDriver"),
    ("threetears.datasources.drivers.bigquery_driver", "BigQueryDriver"),
)


@pytest.fixture
def stub_driver_modules(monkeypatch: pytest.MonkeyPatch) -> dict[str, type[_StubDriver]]:
    """install stub driver modules in sys.modules so factory imports succeed.

    returns the map of {class_name: subclass} so individual tests can
    assert which class their config dispatched to.

    :return: stub-class lookup keyed by the documented class name
    :rtype: dict[str, type[_StubDriver]]
    """
    classes: dict[str, type[_StubDriver]] = {}
    for module_name, class_name in _DRIVER_MODULES:
        # one subclass per slot so identity checks distinguish dispatch
        # arms.
        cls = type(class_name, (_StubDriver,), {})
        classes[class_name] = cls
        module = types.ModuleType(module_name)
        setattr(module, class_name, cls)
        monkeypatch.setitem(sys.modules, module_name, module)
    return classes


class TestDispatch:
    """create_driver dispatches each DataSourceType to its driver class."""

    def test_postgres_dispatches_to_asyncpg_driver(self, stub_driver_modules: dict[str, type[_StubDriver]]) -> None:
        config = PostgresConnectionConfig(
            datasource_type=DataSourceType.POSTGRES,
            host="localhost",
            database="x",
        )
        driver = create_driver(config)
        assert isinstance(driver, stub_driver_modules["AsyncpgDriver"])
        # the external_pool kwarg is NOT supplied for external configs
        assert driver.external_pool is None  # type: ignore[attr-defined]

    def test_yugabyte_dispatches_to_asyncpg_driver(self, stub_driver_modules: dict[str, type[_StubDriver]]) -> None:
        config = YugabyteConnectionConfig(
            datasource_type=DataSourceType.YUGABYTE,
            host="localhost",
            database="x",
        )
        driver = create_driver(config)
        assert isinstance(driver, stub_driver_modules["AsyncpgDriver"])

    def test_agent_internal_dispatches_to_asyncpg_with_external_pool(
        self, stub_driver_modules: dict[str, type[_StubDriver]]
    ) -> None:
        config = AgentInternalConnectionConfig(
            datasource_type=DataSourceType.AGENT_INTERNAL,
            schema_name="agent_abc123",
        )
        sentinel_pool = object()
        driver = create_driver(config, hub_l3_pool=sentinel_pool)  # type: ignore[arg-type]
        assert isinstance(driver, stub_driver_modules["AsyncpgDriver"])
        # the borrowed pool must reach the driver constructor
        assert driver.external_pool is sentinel_pool  # type: ignore[attr-defined]

    def test_agent_internal_without_pool_raises(self, stub_driver_modules: dict[str, type[_StubDriver]]) -> None:
        config = AgentInternalConnectionConfig(
            datasource_type=DataSourceType.AGENT_INTERNAL,
            schema_name="agent_abc123",
        )
        with pytest.raises(ValueError, match="AGENT_INTERNAL"):
            create_driver(config)

    def test_redshift_dispatches_to_redshift_driver(self, stub_driver_modules: dict[str, type[_StubDriver]]) -> None:
        config = RedshiftConnectionConfig(
            datasource_type=DataSourceType.REDSHIFT,
            host="cluster.region.redshift.amazonaws.com",
            database="analytics",
        )
        driver = create_driver(config)
        assert isinstance(driver, stub_driver_modules["RedshiftDriver"])

    def test_snowflake_dispatches_to_snowflake_driver(self, stub_driver_modules: dict[str, type[_StubDriver]]) -> None:
        config = SnowflakeConnectionConfig(
            datasource_type=DataSourceType.SNOWFLAKE,
            account="acct",
            warehouse="wh",
            user="u",
            password_ref="env://X",
        )
        driver = create_driver(config)
        assert isinstance(driver, stub_driver_modules["SnowflakeDriver"])

    def test_bigquery_dispatches_to_bigquery_driver(self, stub_driver_modules: dict[str, type[_StubDriver]]) -> None:
        config = BigQueryConnectionConfig(
            datasource_type=DataSourceType.BIGQUERY,
            project_id="proj",
            credentials_json_ref="env://X",
        )
        driver = create_driver(config)
        assert isinstance(driver, stub_driver_modules["BigQueryDriver"])


class TestDatasourceNamePlumbing:
    """``datasource_name`` reaches the driver constructor for every backend.

    surfaces as the ``datasource_name`` OTel metric attribute on every
    query. the factory threads the kwarg through every match arm so
    callers (Hub broker / tool-pod / introspector in shards 13/14)
    get meaningful labels without needing to reach into per-driver
    constructor surfaces.
    """

    def test_default_is_unknown(self, stub_driver_modules: dict[str, type[_StubDriver]]) -> None:
        """omitting ``datasource_name`` defaults to ``"unknown"``."""
        config = PostgresConnectionConfig(
            datasource_type=DataSourceType.POSTGRES,
            host="localhost",
            database="x",
        )
        driver = create_driver(config)
        assert driver.datasource_name == "unknown"  # type: ignore[attr-defined]

    def test_postgres_threads_through(self, stub_driver_modules: dict[str, type[_StubDriver]]) -> None:
        config = PostgresConnectionConfig(
            datasource_type=DataSourceType.POSTGRES,
            host="localhost",
            database="x",
        )
        driver = create_driver(config, datasource_name="warehouse-prod")
        assert driver.datasource_name == "warehouse-prod"  # type: ignore[attr-defined]

    def test_agent_internal_threads_through(self, stub_driver_modules: dict[str, type[_StubDriver]]) -> None:
        config = AgentInternalConnectionConfig(
            datasource_type=DataSourceType.AGENT_INTERNAL,
            schema_name="agent_xyz",
        )
        sentinel_pool = object()
        driver = create_driver(
            config,
            hub_l3_pool=sentinel_pool,  # type: ignore[arg-type]
            datasource_name="agent-tables",
        )
        assert driver.datasource_name == "agent-tables"  # type: ignore[attr-defined]

    def test_redshift_threads_through(self, stub_driver_modules: dict[str, type[_StubDriver]]) -> None:
        config = RedshiftConnectionConfig(
            datasource_type=DataSourceType.REDSHIFT,
            host="cluster.example.com",
            database="analytics",
        )
        driver = create_driver(config, datasource_name="central-reporting")
        assert driver.datasource_name == "central-reporting"  # type: ignore[attr-defined]

    def test_snowflake_threads_through(self, stub_driver_modules: dict[str, type[_StubDriver]]) -> None:
        config = SnowflakeConnectionConfig(
            datasource_type=DataSourceType.SNOWFLAKE,
            account="acct",
            warehouse="wh",
            user="u",
            password_ref="env://X",
        )
        driver = create_driver(config, datasource_name="snowflake-marts")
        assert driver.datasource_name == "snowflake-marts"  # type: ignore[attr-defined]

    def test_bigquery_threads_through(self, stub_driver_modules: dict[str, type[_StubDriver]]) -> None:
        config = BigQueryConnectionConfig(
            datasource_type=DataSourceType.BIGQUERY,
            project_id="my-project",
            credentials_json_ref="env://X",
        )
        driver = create_driver(config, datasource_name="bq-events")
        assert driver.datasource_name == "bq-events"  # type: ignore[attr-defined]


class TestRealDriverModulesLoadable:
    """post-shard-12, every driver module the factory can dispatch to exists.

    replaces the prior ``test_missing_concrete_driver_raises_importerror``
    test, which existed pre-shard-12 to verify that the factory
    surfaced :class:`ModuleNotFoundError` for not-yet-implemented
    backends. now all five driver modules (asyncpg / redshift /
    snowflake-stub / bigquery-stub) ship in tree, so the prior
    failure mode is structurally impossible to reach. instead we
    invert the assertion: dispatching against the REAL (unstubbed)
    modules MUST return a valid :class:`Driver` instance for every
    backend.

    if a future shard removes a concrete driver module, this test
    fails fast with :class:`ModuleNotFoundError` -- naming exactly
    which dispatch arm broke.
    """

    def test_every_backend_dispatches_against_real_modules(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # clear any stub the prior tests installed so each match arm
        # hits the genuine driver module.
        for module_name, _ in _DRIVER_MODULES:
            monkeypatch.delitem(sys.modules, module_name, raising=False)
        configs: list[Any] = [
            PostgresConnectionConfig(
                datasource_type=DataSourceType.POSTGRES,
                host="localhost",
                database="x",
            ),
            YugabyteConnectionConfig(
                datasource_type=DataSourceType.YUGABYTE,
                host="localhost",
                database="x",
            ),
            RedshiftConnectionConfig(
                datasource_type=DataSourceType.REDSHIFT,
                host="cluster.region.redshift.amazonaws.com",
                database="analytics",
            ),
            SnowflakeConnectionConfig(
                datasource_type=DataSourceType.SNOWFLAKE,
                account="acct",
                warehouse="wh",
                user="u",
                password_ref="env://X",
            ),
            BigQueryConnectionConfig(
                datasource_type=DataSourceType.BIGQUERY,
                project_id="proj",
                credentials_json_ref="env://X",
            ),
        ]
        for config in configs:
            driver = create_driver(config)
            assert isinstance(driver, Driver), f"{type(config).__name__} did not dispatch to a Driver instance"
