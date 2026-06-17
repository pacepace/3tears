"""datasource entity definitions for the registry, schema-metadata, and table-template tables.

each entity subclasses :class:`threetears.core.entities.base.BaseEntity`
and matches the shape of the corresponding ``platform.*`` table:

- :class:`DataSourceEntity` -- the external datasource registration
  row (``platform.datasources``). composite PK ``(customer_id, id)``;
  customer-scoped per namespace-task-01 phase 4.5.
- :class:`DataSourceTableEntity` -- discovered table row
  (``platform.datasource_tables``). flat PK ``id``.
- :class:`DataSourceColumnEntity` -- discovered column row
  (``platform.datasource_columns``). flat PK ``id``; natural unique
  key ``(datasource_id, schema_name, table_name, column_name)``.
- :class:`DataSourceRelationEntity` -- cross-table join metadata
  (``platform.datasource_relations``). flat PK ``id``.
- :class:`TableTemplateEntity` -- reusable table-shape definition
  (``platform.table_templates``). composite PK ``(customer_id, id)``
  matching :class:`DataSourceEntity`.

plus the lifecycle enums:

- :class:`DataSourceType` -- ``redshift`` / ``snowflake`` / ``bigquery``
  / ``postgres`` (or ``yugabyte`` once shard 08 adds it) / ``agent_internal``
- :class:`DataSourceAccessMode` -- ``read`` / ``write`` / ``readwrite``
- :class:`DataSourceStatus` -- ``active`` / ``disabled``
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from threetears.core.entities.base import BaseEntity

__all__ = [
    "DataSourceAccessMode",
    "DataSourceColumnEntity",
    "DataSourceEntity",
    "DataSourceRelationEntity",
    "DataSourceSchemaDigestEntity",
    "DataSourceStatus",
    "DataSourceTableEntity",
    "DataSourceType",
    "TableTemplateEntity",
]


class DataSourceType(StrEnum):
    """supported data source types.

    external types map to a connection driver against the named backend;
    ``AGENT_INTERNAL`` (data-task-01) is the agent-created-table variant
    where the broker routes queries through the L3 broker bound to the
    agent's ``agent_<hex>`` schema instead of opening an external
    connection. agent_internal rows carry ``owner_agent_id`` (the
    creating agent) and ``schema_name`` (the agent's hub-managed
    schema); external rows carry neither. the database CHECK constraint
    ``datasources_agent_internal_shape_ck`` (v056) enforces the
    bidirectional invariant.

    :cvar REDSHIFT: Amazon Redshift (postgres-compatible)
    :cvar SNOWFLAKE: Snowflake Data Cloud
    :cvar BIGQUERY: Google BigQuery
    :cvar POSTGRES: standard PostgreSQL
    :cvar YUGABYTE: YugabyteDB (postgres-compatible; same asyncpg driver
        as POSTGRES, separate enum value so operators can see which
        backend they're talking to)
    :cvar AGENT_INTERNAL: agent-created table variant; owner_agent_id +
        schema_name carry the routing target
    """

    REDSHIFT = "redshift"
    SNOWFLAKE = "snowflake"
    BIGQUERY = "bigquery"
    POSTGRES = "postgres"
    YUGABYTE = "yugabyte"
    AGENT_INTERNAL = "agent_internal"


class DataSourceAccessMode(StrEnum):
    """data source access mode controlling which query tools are registered.

    :cvar READ: read-only (SELECT queries via DataSourceReadTool)
    :cvar WRITE: write-only (INSERT/UPDATE/DELETE via DataSourceWriteTool)
    :cvar READWRITE: full access (all query tools registered)
    """

    READ = "read"
    WRITE = "write"
    READWRITE = "readwrite"


class DataSourceStatus(StrEnum):
    """data source lifecycle status values.

    :cvar ACTIVE: data source is available for queries
    :cvar DISABLED: data source is intentionally disabled
    """

    ACTIVE = "active"
    DISABLED = "disabled"


class DataSourceEntity(BaseEntity):
    """data source entity representing a registered external data source.

    extends :class:`BaseEntity` with data-source-specific field access.
    all field data lives in L1 cache, accessed via the parent collection
    proxy. fields match ``platform.datasources``.

    flat primary key ``id`` post-knowledge-task-08: the v016 migration
    rebuilt the table PK on ``id`` alone (dropping the v001 composite
    ``(customer_id, id)`` partition PK) so a platform-shared datasource
    can carry ``customer_id = NULL`` (KNW-76). ``customer_id`` is now a
    plain nullable column, no longer the partition / addressing key —
    every lookup resolves by the global ``id`` (backed by the
    ``datasources_id_unique`` index), which a NULL customer_id never
    blocks.

    :param data: initial field data dictionary carrying ``id``
    :ptype data: dict[str, Any]
    :param is_new: whether entity is newly created
    :ptype is_new: bool
    :param collection: parent collection reference
    :ptype collection: Any
    """

    primary_key_field: str = "id"


class DataSourceTableEntity(BaseEntity):
    """data source table entity representing a discovered database table.

    extends :class:`BaseEntity` with table-specific field access. all
    field data lives in L1 cache, accessed via the parent collection
    proxy. fields match ``platform.datasource_tables``.

    :param data: initial field data dictionary
    :ptype data: dict[str, Any]
    :param is_new: whether entity is newly created
    :ptype is_new: bool
    :param collection: parent collection reference
    :ptype collection: Any
    """

    primary_key_field: str = "id"


class DataSourceColumnEntity(BaseEntity):
    """data source column entity representing a discovered database column.

    extends :class:`BaseEntity` with column-specific field access. all
    field data lives in L1 cache, accessed via the parent collection
    proxy. fields match ``platform.datasource_columns``.

    :param data: initial field data dictionary
    :ptype data: dict[str, Any]
    :param is_new: whether entity is newly created
    :ptype is_new: bool
    :param collection: parent collection reference
    :ptype collection: Any
    """

    primary_key_field: str = "id"


class DataSourceSchemaDigestEntity(BaseEntity):
    """materialized documented-schema digest for one datasource (schema-priming).

    extends :class:`BaseEntity`. one row per datasource in
    ``platform.datasource_schema_digests``, holding the pre-derived
    structured projection of the datasource's DOCUMENTED tables and
    columns (the hub materializes it from
    ``platform.datasource_tables`` + ``platform.datasource_columns``,
    keeping only rows whose ``description`` is non-null). agent pods read
    it BY PRIMARY KEY so the lookup is served from the hot L1 cache — the
    whole reason the digest is materialized rather than re-derived per
    turn (a list-by-datasource read cache-bypasses to L3).

    the primary key IS ``datasource_id`` (one row per datasource), so the
    agent addresses the digest by the datasource it already holds. fields
    match ``platform.datasource_schema_digests``: ``datasource_id`` /
    ``customer_id`` / ``tables`` (JSONB structured projection) /
    ``source_fingerprint`` (a content hash of the documented rows the
    digest was built from, for idempotent reconcile + observability) /
    ``date_created`` / ``date_updated``.

    the stored projection is STRUCTURED, not rendered markdown — token
    budgeting + block rendering are the consumer's (agent's) presentation
    concern (schema-priming-task-01b), kept out of the platform artifact.

    :param data: initial field data dictionary carrying ``datasource_id``
    :ptype data: dict[str, Any]
    :param is_new: whether entity is newly created
    :ptype is_new: bool
    :param collection: parent collection reference
    :ptype collection: Any
    """

    primary_key_field: str = "datasource_id"


class DataSourceRelationEntity(BaseEntity):
    """data source relation entity representing cross-table join metadata.

    extends :class:`BaseEntity` with relation-specific field access. all
    field data lives in L1 cache, accessed via the parent collection
    proxy. fields match ``platform.datasource_relations``.

    :param data: initial field data dictionary
    :ptype data: dict[str, Any]
    :param is_new: whether entity is newly created
    :ptype is_new: bool
    :param collection: parent collection reference
    :ptype collection: Any
    """

    primary_key_field: str = "id"


class TableTemplateEntity(BaseEntity):
    """reusable table-shape definition scoped to one customer.

    extends :class:`BaseEntity` with composite-PK ``(customer_id, id)``
    addressing so the hub's L1/L2/L3 lookup paths route correctly
    through the shared cache primitives. fields match
    ``platform.table_templates``: id, customer_id, name, description,
    caveats, date_created, date_updated.

    customer-scoping is a hard invariant: every template belongs to
    exactly one customer, the unique index on ``(customer_id, name)``
    keeps slug collisions inside a customer's namespace, and the
    ``customer_id`` FK to ``platform.customers`` cascades on customer
    delete.

    :param data: initial field data dictionary; must carry both
        ``customer_id`` and ``id``
    :ptype data: dict[str, Any]
    :param is_new: whether entity is newly created
    :ptype is_new: bool
    :param collection: parent collection reference
    :ptype collection: Any
    """

    primary_key_field: str = "customer_id"

    def __init__(
        self,
        data: dict[str, Any],
        is_new: bool = True,
        collection: Any = None,
    ) -> None:
        """initialize entity with composite-pk ``_id`` tuple.

        :param data: row dict carrying both ``customer_id`` and ``id``
        :ptype data: dict[str, Any]
        :param is_new: whether entity is unsaved
        :ptype is_new: bool
        :param collection: owning collection reference
        :ptype collection: Any
        :return: nothing
        :rtype: None
        """
        super().__init__(data, is_new=is_new, collection=collection)
        object.__setattr__(self, "_row_id", data["id"])
        object.__setattr__(self, "_id", (data["customer_id"], data["id"]))

    @property
    def id(self) -> Any:
        """return the scalar template UUID.

        :return: template UUID
        :rtype: Any
        """
        return self._row_id
