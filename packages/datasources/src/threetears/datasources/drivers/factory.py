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

from threetears.datasources.config import ConnectionConfig
from threetears.datasources.drivers.base import Driver
from threetears.datasources.entities import DataSourceType

if TYPE_CHECKING:
    # type-only import: keeps asyncpg out of the runtime module graph
    # so the lazy-import contract holds.
    import asyncpg

__all__ = ["create_driver"]


def create_driver(
    config: ConnectionConfig,
    *,
    hub_l3_pool: "asyncpg.Pool[Any] | None" = None,
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

            driver = AsyncpgDriver(config)
        case DataSourceType.AGENT_INTERNAL:
            if hub_l3_pool is None:
                raise ValueError(
                    "AGENT_INTERNAL driver requires hub_l3_pool; "
                    "the agent-internal variant borrows Hub's L3 pool"
                )
            from threetears.datasources.drivers.asyncpg_driver import (
                AsyncpgDriver,
            )

            driver = AsyncpgDriver(config, external_pool=hub_l3_pool)
        case DataSourceType.REDSHIFT:
            from threetears.datasources.drivers.redshift_driver import (
                RedshiftDriver,
            )

            driver = RedshiftDriver(config)
        case DataSourceType.SNOWFLAKE:
            from threetears.datasources.drivers.snowflake_driver import (
                SnowflakeDriver,
            )

            driver = SnowflakeDriver(config)
        case DataSourceType.BIGQUERY:
            from threetears.datasources.drivers.bigquery_driver import (
                BigQueryDriver,
            )

            driver = BigQueryDriver(config)
        case _:
            raise ValueError(
                f"no driver registered for datasource_type={config.datasource_type!r}"
            )
    return driver
