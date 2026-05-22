"""datasource entities, collections, namespace helpers, and agent.yaml config model.

this package is the single source of truth for datasource primitives
across every 3tears consumer (Hub, agent pods, future products). the
``Driver`` abstraction + per-backend driver implementations land in
``threetears.datasources.drivers`` (datasource-task-09 through 12);
drivers are accessed via the
:func:`threetears.datasources.drivers.create_driver` factory and are
NOT re-exported from this top-level ``__all__`` -- the package root
stays free of backend-library imports (asyncpg, redshift_connector,
etc.) so a consumer that never uses a given driver doesn't pay the
import cost.

public surface (per shard DS-07-10):

- entities -- :class:`DataSourceEntity`, :class:`DataSourceTableEntity`,
  :class:`DataSourceColumnEntity`, :class:`DataSourceRelationEntity`,
  :class:`TableTemplateEntity` + lifecycle enums (:class:`DataSourceType`,
  :class:`DataSourceAccessMode`, :class:`DataSourceStatus`)
- collections -- :class:`DataSourceCollection`,
  :class:`DataSourceTableCollection`, :class:`DataSourceColumnCollection`,
  :class:`DataSourceRelationCollection`, :class:`TableTemplateCollection`
- namespace helpers -- :data:`DATASOURCE_NAMESPACE_TYPE`,
  :func:`datasource_namespace_id`, :func:`datasource_namespace_name`
- agent.yaml-facing config -- :class:`DatasourceConfig`

driver implementations live in the ``drivers`` subpackage and are
imported lazily via the factory:

    from threetears.datasources.drivers import create_driver, Driver
"""

from __future__ import annotations

# Version is derived from package metadata so pyproject.toml stays the
# single source of truth -- a release that bumps the version in
# pyproject without updating ``__init__.py`` can't drift the runtime
# ``__version__``.
from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _version

try:
    __version__ = _version("3tears-datasources")
except _PackageNotFoundError:  # pragma: no cover - dev fallback
    __version__ = "unknown"

from threetears.datasources.collections import (
    DataSourceCollection,
    DataSourceColumnCollection,
    DataSourceRelationCollection,
    DataSourceTableCollection,
    TableTemplateCollection,
)
from threetears.datasources.config import (
    AgentInternalConnectionConfig,
    BigQueryConnectionConfig,
    ConnectionConfig,
    DatasourceConfig,
    PostgresConnectionConfig,
    RedshiftConnectionConfig,
    SnowflakeConnectionConfig,
    YugabyteConnectionConfig,
)
from threetears.datasources.entities import (
    DataSourceAccessMode,
    DataSourceColumnEntity,
    DataSourceEntity,
    DataSourceRelationEntity,
    DataSourceStatus,
    DataSourceTableEntity,
    DataSourceType,
    TableTemplateEntity,
)
from threetears.datasources.namespace import (
    DATASOURCE_NAMESPACE_TYPE,
    datasource_namespace_id,
    datasource_namespace_name,
)

__all__ = [
    "DATASOURCE_NAMESPACE_TYPE",
    "AgentInternalConnectionConfig",
    "BigQueryConnectionConfig",
    "ConnectionConfig",
    "DataSourceAccessMode",
    "DataSourceCollection",
    "DataSourceColumnCollection",
    "DataSourceColumnEntity",
    "DataSourceEntity",
    "DataSourceRelationCollection",
    "DataSourceRelationEntity",
    "DataSourceStatus",
    "DataSourceTableCollection",
    "DataSourceTableEntity",
    "DataSourceType",
    "DatasourceConfig",
    "PostgresConnectionConfig",
    "RedshiftConnectionConfig",
    "SnowflakeConnectionConfig",
    "TableTemplateCollection",
    "TableTemplateEntity",
    "YugabyteConnectionConfig",
    "datasource_namespace_id",
    "datasource_namespace_name",
]
