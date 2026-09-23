"""driver factory: dispatch a ConnectionConfig to its concrete driver.

:func:`create_driver` is the only public way callers construct a
:class:`Driver`. direct instantiation of concrete driver classes is
discouraged -- the factory enforces:

- **lazy import of backend libraries** (DS-09-08, DS-09-09).
  ``asyncpg``, ``redshift_connector``, ``snowflake.connector`` and
  ``google.cloud.bigquery`` are imported inside the ``match`` arm
  that dispatches to the driver using them. a Hub that never queries
  Snowflake doesn't pay the snowflake-connector-python import cost.
- **discriminated dispatch on ``config.datasource_type``** -- pydantic
  has already routed the incoming config to the right ConnectionConfig
  member; the factory just selects the matching driver class.
- **AGENT_INTERNAL pool-borrow plumbing** -- the agent-internal driver
  variant doesn't open its own connection. it borrows Hub's L3
  asyncpg pool via the ``hub_l3_pool=`` kwarg. the factory routes the
  pool into the AsyncpgDriver's ``external_pool=`` constructor arg;
  callers that don't pass ``hub_l3_pool`` get a clear ValueError.

concrete driver implementations land in shards 10 / 11 / 12; the
factory's dispatch table is the contract those shards slot into.
until those shards land, calling :func:`create_driver` for a given
backend imports the concrete-driver module which will raise
:class:`ImportError`. tests stub the import to verify dispatch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from threetears.observe import get_logger

from threetears.datasources.config import ConnectionConfig
from threetears.datasources.drivers.base import Driver
from threetears.datasources.drivers.connect_guard import ConnectGuard
from threetears.datasources.entities import DataSourceType

if TYPE_CHECKING:
    # type-only import: keeps asyncpg out of the runtime module graph
    # so the lazy-import contract holds.
    import asyncpg

__all__ = ["create_driver"]

log = get_logger(__name__)


def create_driver(
    config: ConnectionConfig,
    *,
    hub_l3_pool: "asyncpg.Pool[Any] | None" = None,
    datasource_name: str = "unknown",
    connect_guard: ConnectGuard | None = None,
) -> Driver:
    """dispatch to the concrete :class:`Driver` for ``config``.

    every backend lib is imported inside its matching ``case`` arm
    (DS-09-08). importing this module does NOT pull any backend lib
    into ``sys.modules`` -- the lazy-import audit verifies the
    contract across all three package roots (DS-09-09).

    :param config: per-driver connection config (discriminated on
        ``datasource_type``); pydantic has already routed the incoming
        dict to the matching ConnectionConfig member
    :ptype config: ConnectionConfig
    :param hub_l3_pool: Hub's L3 asyncpg pool, ONLY consumed by the
        AGENT_INTERNAL branch (passed to ``AsyncpgDriver`` as
        ``external_pool=``). external driver branches ignore the
        kwarg. omitting it raises :class:`ValueError` for the
        AGENT_INTERNAL case
    :ptype hub_l3_pool: asyncpg.Pool | None
    :param datasource_name: human-readable name of the datasource the
        constructed driver serves; surfaces on every OTel metric
        emitted by :func:`_observed` as the ``datasource_name``
        attribute. defaults to ``"unknown"`` so callers that don't
        know the name (or don't care about per-datasource metrics)
        can omit. the Hub broker / tool-pod / introspector (shards
        13/14) thread the name from :attr:`DatasourceConfig.name`
    :ptype datasource_name: str
    :param connect_guard: the datasource's guard against logging in with a
        credential the warehouse already refused, or ``None`` for an unguarded
        driver -- which is what an explicit connection probe is. honoured by
        the drivers whose logins classify a refusal (Postgres, Yugabyte,
        Redshift); the agent-internal variant logs into no warehouse, and a
        guard handed to a driver that cannot honour it is logged, not dropped
        silently
    :ptype connect_guard: ConnectGuard | None
    :return: live :class:`Driver` instance ready for use; no async
        ``initialize()`` step is required -- drivers that need async
        warm-up do it lazily on first ``fetch``
    :rtype: Driver
    :raises ValueError: if ``config.datasource_type`` has no
        registered driver, OR if the AGENT_INTERNAL case is dispatched
        without ``hub_l3_pool``
    :raises ImportError: if the backend library for the selected
        driver is not installed (e.g. the redshift extras key wasn't
        installed but a RedshiftConnectionConfig is passed)
    """
    driver: Driver
    match config.datasource_type:
        case DataSourceType.POSTGRES | DataSourceType.YUGABYTE:
            from threetears.datasources.drivers.asyncpg_driver import (
                AsyncpgDriver,
            )

            driver = AsyncpgDriver(config, datasource_name=datasource_name, connect_guard=connect_guard)
        case DataSourceType.AGENT_INTERNAL:
            if hub_l3_pool is None:
                raise ValueError(
                    "AGENT_INTERNAL driver requires hub_l3_pool; the agent-internal variant borrows Hub's L3 pool"
                )
            from threetears.datasources.drivers.asyncpg_driver import (
                AsyncpgDriver,
            )

            driver = AsyncpgDriver(
                config,
                external_pool=hub_l3_pool,
                datasource_name=datasource_name,
            )
        case DataSourceType.REDSHIFT:
            from threetears.datasources.drivers.redshift_driver import (
                RedshiftDriver,
            )

            driver = RedshiftDriver(config, datasource_name=datasource_name, connect_guard=connect_guard)
        case DataSourceType.SNOWFLAKE:
            from threetears.datasources.drivers.snowflake_driver import (
                SnowflakeDriver,
            )

            _log_unhonoured_guard(connect_guard, config.datasource_type, datasource_name)
            driver = SnowflakeDriver(config, datasource_name=datasource_name)
        case DataSourceType.BIGQUERY:
            from threetears.datasources.drivers.bigquery_driver import (
                BigQueryDriver,
            )

            _log_unhonoured_guard(connect_guard, config.datasource_type, datasource_name)
            driver = BigQueryDriver(config, datasource_name=datasource_name)
        case _:
            raise ValueError(f"no driver registered for datasource_type={config.datasource_type!r}")
    return driver


def _log_unhonoured_guard(
    connect_guard: ConnectGuard | None,
    datasource_type: DataSourceType,
    datasource_name: str,
) -> None:
    """say so when a connect guard reaches a driver that cannot honour it.

    these drivers do not yet classify a refused login as :class:`DriverAuthError`, so a
    guard would never learn of a refusal; the operator reading the log should know this
    datasource is not protected, rather than assume it is.

    :param connect_guard: the guard the caller supplied, or ``None``
    :ptype connect_guard: ConnectGuard | None
    :param datasource_type: the backend
    :ptype datasource_type: DataSourceType
    :param datasource_name: the datasource
    :ptype datasource_name: str
    :return: nothing
    :rtype: None
    """
    if connect_guard is not None:
        log.warning(
            "connect guard not honoured: this backend does not yet classify a refused login, so a wrong "
            "credential is retried on every connect",
            extra={"extra_data": {"datasource_type": f"{datasource_type}", "datasource_name": datasource_name}},
        )
