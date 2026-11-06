"""three-tier collections for datasource registry, schema metadata, and table templates.

merged from Hub's ``3tears/hub/datasources/collections.py``,
``schema_collections.py``, and ``template_collections.py`` per
``datasource-task-07``. class definitions are byte-identical to the Hub
originals -- this shard is pure relocation, not refactor.

collections in this module:

- :class:`CapabilitySourceCollection` -- ``SchemaBackedCollection`` for
  the ``datasources`` registry, generalized (Fork-1,
  gu-task-08) to hold database datasources, API imports, and MCP
  imports discriminated by ``kind``. flat PK ``id`` with a
  ``find_by_id`` helper that uses the v054 ``UNIQUE (id)`` constraint.
- :class:`DataSourceTableCollection` -- ``DeclaredTableCollection`` for the
  ``datasource_tables`` row set.
- :class:`DataSourceColumnCollection` -- ``DeclaredTableCollection`` for
  ``datasource_columns``. natural-key upsert on
  ``(datasource_id, schema_name, table_name, column_name)``.
- :class:`DataSourceSchemaDigestCollection` -- ``DeclaredTableCollection``
  for ``datasource_schema_digests``, keyed by ``datasource_id``.
- :class:`DataSourceRelationCollection` -- ``DeclaredTableCollection`` for
  ``datasource_relations``.
- :class:`TableTemplateCollection` -- ``DeclaredTableCollection`` for
  ``table_templates``. PK ``id`` (hub v007).

the collections that write their own SQL each declare their table as a
``TableSchema``, as the hub's migrations leave it, so a host that gives its
registry an L1 gets each one's L1 table generated and created for it
(:class:`~threetears.core.collections.declared_table.DeclaredTableCollection`).

their upserts are the one :meth:`DeclaredTableCollection.write_row` builds from
the schema, and write only the columns a row carries: a column the row does not
name is never reset to NULL or a default. their L2 codecs rehydrate each column
by the type the schema declares; no collection here keeps a field-type map or a
column list of its own.

per-table column variants (``TableTemplateColumnCollection``) stay in
Hub for now because they have no cross-consumer demand yet; lift later
if a second 3tears consumer needs them.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, ClassVar
from uuid import UUID

from threetears.core.collections import WRITE_GENERATION, DeleteCascade
from threetears.core.collections.declared_table import DeclaredTableCollection
from threetears.core.collections.schema_backed import (
    BIGINT_TYPE,
    BOOL_TYPE,
    DATETIMETZ_TYPE,
    INT_TYPE,
    JSONB_TYPE,
    STRING_TYPE,
    UUID_TYPE,
    Column,
    SchemaBackedCollection,
    TableSchema,
)
from threetears.core.security.secret_refs import validate_ref
from threetears.observe import get_logger

from threetears.datasources.entities import (
    CapabilitySourceEntity,
    CapabilitySourceKind,
    DataSourceColumnEntity,
    DataSourceRelationEntity,
    DataSourceSchemaDigestEntity,
    DataSourceStatus,
    DataSourceTableEntity,
    TableTemplateEntity,
)

log = get_logger(__name__)


__all__ = [
    "CapabilitySourceCollection",
    "DataSourceColumnCollection",
    "DataSourceRelationCollection",
    "DataSourceSchemaDigestCollection",
    "DataSourceTableCollection",
    "TableTemplateCollection",
]

# gu-task-08 GU-08-03: config storage is KIND-CONDITIONAL. rows of these
# kinds address their config via a secret_refs ``scheme://locator``
# reference, validated at the write boundary (:func:`validate_ref`) and
# resolved at use time. a ``datasource``-kind row is DELIBERATELY absent
# here: its ``connection_config`` is an encrypted JSON blob, NOT a
# ``scheme://locator``, so validating it as a ref would reject every
# existing datasource write.
_CONFIG_REF_KINDS: frozenset[str] = frozenset(
    {CapabilitySourceKind.API_IMPORT.value, CapabilitySourceKind.MCP_IMPORT.value},
)


#: ``datasource_tables`` as hub v001, v006, v010 and v033 leave it
_DATASOURCE_TABLES_SCHEMA = TableSchema(
    name="datasource_tables",
    primary_key="id",
    columns=[
        Column("id", UUID_TYPE),
        Column("datasource_id", UUID_TYPE),
        Column("schema_name", STRING_TYPE),
        Column("table_name", STRING_TYPE),
        Column("description", STRING_TYPE, nullable=True),
        Column("row_count_approx", BIGINT_TYPE, nullable=True),
        Column("caveats", STRING_TYPE, nullable=True),
        Column("template_id", UUID_TYPE, nullable=True),
        Column("caveats_replaces_definition", BOOL_TYPE, server_default="false"),
        Column("date_introspected", DATETIMETZ_TYPE, nullable=True),
        Column("date_described", DATETIMETZ_TYPE, nullable=True),
        Column("date_created", DATETIMETZ_TYPE, immutable=True),
        Column("date_updated", DATETIMETZ_TYPE),
        Column("column_hash", STRING_TYPE, nullable=True),
        Column("coverage_dimension", STRING_TYPE, nullable=True),
    ],
    write_generation=WRITE_GENERATION,
)

#: ``datasource_columns`` as hub v001 and v006 leave it
_DATASOURCE_COLUMNS_SCHEMA = TableSchema(
    name="datasource_columns",
    primary_key="id",
    columns=[
        Column("id", UUID_TYPE),
        Column("datasource_id", UUID_TYPE),
        Column("schema_name", STRING_TYPE),
        Column("table_name", STRING_TYPE),
        Column("column_name", STRING_TYPE),
        Column("data_type", STRING_TYPE, nullable=True),
        Column("is_nullable", BOOL_TYPE, nullable=True),
        Column("ordinal_position", INT_TYPE, nullable=True),
        Column("description", STRING_TYPE, nullable=True),
        Column("valid_range", STRING_TYPE, nullable=True),
        Column("caveats", STRING_TYPE, nullable=True),
        Column("tags", JSONB_TYPE, nullable=True, server_default="'[]'::jsonb"),
        Column("caveats_replaces_definition", BOOL_TYPE, server_default="false"),
        Column("date_introspected", DATETIMETZ_TYPE, nullable=True),
        Column("date_described", DATETIMETZ_TYPE, nullable=True),
        Column("date_created", DATETIMETZ_TYPE, immutable=True),
        Column("date_updated", DATETIMETZ_TYPE),
    ],
)

#: ``datasource_schema_digests`` as hub v029 leaves it: keyed by ``datasource_id``, no ``id`` column
_DATASOURCE_SCHEMA_DIGESTS_SCHEMA = TableSchema(
    name="datasource_schema_digests",
    primary_key="datasource_id",
    columns=[
        Column("datasource_id", UUID_TYPE),
        Column("customer_id", UUID_TYPE, nullable=True),
        Column("tables", JSONB_TYPE, server_default="'[]'::jsonb"),
        Column("source_fingerprint", STRING_TYPE, nullable=True),
        Column("date_created", DATETIMETZ_TYPE),
        Column("date_updated", DATETIMETZ_TYPE),
    ],
)

#: ``datasource_relations`` as hub v001, v055 and v056 leave it
_DATASOURCE_RELATIONS_SCHEMA = TableSchema(
    name="datasource_relations",
    primary_key="id",
    columns=[
        Column("id", UUID_TYPE),
        Column("customer_id", UUID_TYPE, nullable=True),
        Column("name", STRING_TYPE),
        Column("description", STRING_TYPE, nullable=True),
        Column("datasource_ids", JSONB_TYPE, server_default="'[]'::jsonb"),
        Column("join_paths", JSONB_TYPE, server_default="'[]'::jsonb"),
        Column("edges", JSONB_TYPE, server_default="'[]'::jsonb"),
        Column("aggregation_notes", STRING_TYPE, nullable=True),
        Column("caveats", STRING_TYPE, nullable=True),
        Column("date_created", DATETIMETZ_TYPE, immutable=True),
        Column("date_updated", DATETIMETZ_TYPE),
    ],
)

#: ``table_templates`` as hub v006 and v007 leave it
_TABLE_TEMPLATES_SCHEMA = TableSchema(
    name="table_templates",
    primary_key="id",
    columns=[
        Column("id", UUID_TYPE),
        Column("customer_id", UUID_TYPE, nullable=True),
        Column("name", STRING_TYPE),
        Column("description", STRING_TYPE, nullable=True),
        Column("caveats", STRING_TYPE, nullable=True),
        Column("visibility", STRING_TYPE, server_default="'private'"),
        Column("origin_template_id", UUID_TYPE, nullable=True),
        Column("date_created", DATETIMETZ_TYPE, immutable=True),
        Column("date_updated", DATETIMETZ_TYPE),
    ],
)


class CapabilitySourceCollection(SchemaBackedCollection[CapabilitySourceEntity]):
    """three-tier collection for capability-source entities.

    generalized from the former datasource-only collection (Fork-1,
    gu-task-08): the SAME ``datasources`` registry holds
    database datasources, external API imports, and MCP imports,
    discriminated by the ``kind`` column
    (:class:`CapabilitySourceKind`). the table stays; the SHAPE widens.

    provides CRUD operations with L1 -> L2 -> L3 caching. capability
    sources are hard-deleted (no soft-delete pattern).

    ``connection_config`` is KIND-CONDITIONAL (GU-08-03): a
    ``datasource``-kind row keeps its existing encrypted-JSON-blob shape;
    an ``api_import`` / ``mcp_import``-kind row stores a
    :mod:`threetears.core.security.secret_refs` ``scheme://locator``
    reference. both slot into the SAME ``STRING_TYPE`` passthrough so the
    write path forwards the string unchanged; only the write-boundary
    VALIDATION branches on ``kind`` (see :meth:`save_to_store`). never
    plaintext for either kind.

    ``allowed_schemas`` is stored as a JSONB array and scopes the source
    for every kind (a datasource scopes schemas; an import scopes
    operations / paths). ``ingress_agent_id`` (GU-08-05) is the per-source
    ingress-agent principal for the external-API call flow, ``NULL`` for
    pure internal datasources.

    CRUD comes from the declarative :class:`TableSchema`; no domain
    queries live on the subclass beyond the by-id / status / origin-link
    helpers (every other callsite resolves by primary key or filters in
    admin endpoints via the L3 pool with cache-bypass rationales).
    """

    #: switched on (epoch-task-06): its writes advance the table's write generation, so a cache
    #: derived from it (a visibility scan) is evicted when a broadcast is missed, not timed out
    # so its schema declares ``write_generation=WRITE_GENERATION``, which this class reads (one declaration)

    primary_key_column: str = "id"
    schema = TableSchema(
        name="datasources",
        write_generation=WRITE_GENERATION,
        primary_key="id",
        columns=[
            Column("id", UUID_TYPE),
            Column("name", STRING_TYPE),
            # customer_id is nullable + a plain column post-knowledge-
            # task-08 (KNW-76): a platform-shared source (visibility
            # != 'private') carries customer_id NULL. v016 rebuilt the
            # table PK on ``id`` alone (dropping the v001 composite
            # partition PK) so the addressing key is the global ``id``,
            # backed by datasources_id_unique; a NULL customer_id never
            # blocks resolution.
            Column("customer_id", UUID_TYPE, nullable=True),
            # gu-task-08 GU-08-02: the capability-source discriminator.
            # existing rows are 'datasource'; 'api_import' / 'mcp_import'
            # widen the registry. distinct from datasource_type (the
            # driver axis, applies only to kind='datasource').
            Column("kind", STRING_TYPE, server_default="'datasource'"),
            # datasource_type is the DRIVER axis (redshift / snowflake /
            # ...); applies only to kind='datasource'. nullable so an
            # api_import / mcp_import row carries no driver. immutable:
            # a row's backend identity is fixed once materialized.
            Column("datasource_type", STRING_TYPE, immutable=True, nullable=True),
            # connection_config is nullable post-v056 (agent_internal
            # rows carry no external connection config because the broker
            # routes via the L3 broker bound to schema_name) and
            # KIND-CONDITIONAL post-gu-task-08: a datasource-kind row
            # holds an encrypted JSON blob; an import-kind row holds a
            # secret_refs scheme://locator reference. same STRING_TYPE
            # passthrough; validation branches on kind in save_to_store.
            Column("connection_config", STRING_TYPE, nullable=True),
            # allowed_schemas scopes the source for every kind (schemas
            # for a datasource; operations / paths for an api_import).
            Column("allowed_schemas", JSONB_TYPE, server_default="'[]'::jsonb"),
            Column("access_mode", STRING_TYPE, server_default="'read'"),
            Column("status", STRING_TYPE),
            # gu-task-08 GU-08-05: the per-source ingress-agent principal
            # id — the agent identity RBAC grants on this source's tool
            # namespaces for the external-API call flow. NULL for pure
            # internal datasources (no external ingress).
            Column("ingress_agent_id", UUID_TYPE, nullable=True),
            # owner_agent_id: NULL for external datasources, set for
            # agent_internal rows. v056 CHECK
            # datasources_agent_internal_shape_ck enforces the
            # bidirectional invariant with datasource_type.
            Column("owner_agent_id", UUID_TYPE, immutable=True, nullable=True),
            # schema_name: NULL for external datasources, set to
            # ``agent_<hex>`` for agent_internal rows. immutable: the
            # routing target is part of the row's identity once
            # materialized.
            Column("schema_name", STRING_TYPE, immutable=True, nullable=True),
            # knowledge-task-08 (KNW-76/77): cross-customer sharing. a
            # platform-shared datasource carries visibility 'public' /
            # 'restricted' + customer_id NULL; a customer datasource links
            # to its canonical platform-shared form via
            # origin_datasource_id (the table-LEVEL origin link the merge
            # retrieval gathers across: datasource_id IN (D, P)).
            Column("visibility", STRING_TYPE, server_default="'private'"),
            Column("origin_datasource_id", UUID_TYPE, nullable=True),
            # knowledge-quarantine foundation: a datasource "requires
            # knowledge" purely because someone authored knowledge anchored
            # to it. the hub knowledge-write handlers AUTO-STAMP this flag
            # true (through this Collection's write path — set field +
            # save_entity) the first step of every entry write, BEFORE the
            # entry persists, so intent (the requirement) is independent of
            # load state (the entries). NEVER hand-set. NOT NULL with a FALSE
            # server-default so the ADD COLUMN lands false on every existing
            # row (hub migration v046). mutable: the stamp is an UPDATE of an
            # existing row, so the column stays out of the immutable set.
            Column(
                "knowledge_required",
                BOOL_TYPE,
                nullable=False,
                server_default="false",
            ),
            # maturity flag (hub migration v076): when true the hub read path
            # refuses a query that references a table with no description in
            # ``datasource_tables``. set by ``PATCH /admin/v1/datasources/{id}``
            # through this Collection's write path, so it MUST be declared here
            # or the schema-driven upsert drops it and the flag never persists.
            # declared as v076 adds it: boolean, NOT NULL, FALSE server-default
            # (an omitted value keeps the stored one). mutable: flipping it is an
            # UPDATE of an existing row.
            Column(
                "require_documented_tables",
                BOOL_TYPE,
                nullable=False,
                server_default="false",
            ),
            Column("date_created", DATETIMETZ_TYPE, immutable=True),
            Column("date_updated", DATETIMETZ_TYPE),
        ],
        cas_column="date_updated",
    )

    @property
    def table_name(self) -> str:
        """return database table name.

        :return: table name string
        :rtype: str
        """
        return "datasources"

    @property
    def entity_class(self) -> type[CapabilitySourceEntity]:
        """return entity class for this collection.

        :return: CapabilitySourceEntity class
        :rtype: type[CapabilitySourceEntity]
        """
        return CapabilitySourceEntity

    async def save_to_store(
        self,
        data: dict[str, Any],
        original_timestamp: datetime | None = None,
        *,
        conn: Any = None,
    ) -> int:
        """persist one capability-source row, validating config by ``kind``.

        the L3 write boundary for the registry (gu-task-08 GU-08-03).
        config-storage validation is KIND-CONDITIONAL: an ``api_import``
        / ``mcp_import`` row MUST carry a valid
        :mod:`threetears.core.security.secret_refs` ``scheme://locator``
        reference in ``connection_config`` (rejected via
        :func:`validate_ref` when malformed), while a ``datasource`` row's
        ``connection_config`` is an encrypted JSON blob that is passed
        through UNVALIDATED — validating it as a ref would reject every
        existing datasource write. after the kind-conditional check the
        row is forwarded to the schema-driven upsert unchanged.

        :param data: row payload keyed by column name
        :ptype data: dict[str, Any]
        :param original_timestamp: pre-mutation CAS fence value
        :ptype original_timestamp: datetime | None
        :param conn: optional connection that overrides ``l3_pool`` for
            this single write so it commits with the caller's transaction
        :ptype conn: Any
        :return: rows affected reported by the backend
        :rtype: int
        :raises SecretResolutionError: when an ``api_import`` /
            ``mcp_import`` row's ``connection_config`` is not a valid
            ``scheme://locator`` reference
        """
        self._validate_config_for_kind(data)
        return await super().save_to_store(data, original_timestamp, conn=conn)

    def _validate_config_for_kind(self, data: dict[str, Any]) -> None:
        """enforce the KIND-CONDITIONAL config-ref rule at the write boundary.

        for ``api_import`` / ``mcp_import`` rows a non-null
        ``connection_config`` MUST be a valid ``scheme://locator``
        reference. ``datasource``-kind rows are deliberately exempt (their
        config is an encrypted JSON blob, not a ref). a ``None`` config is
        allowed for either kind (the column is nullable).

        :param data: row payload keyed by column name
        :ptype data: dict[str, Any]
        :return: nothing
        :rtype: None
        :raises SecretResolutionError: when the reference is malformed for
            an import kind
        """
        kind = data.get("kind")
        config = data.get("connection_config")
        if kind in _CONFIG_REF_KINDS and config is not None:
            validate_ref(config)

    async def iter_active_ids(self) -> list[UUID]:
        """list every ACTIVE source's primary-key ``id``.

        consumed by background-task sweeps (e.g. the Hub-owned
        introspect scheduler in ``datasource-task-04``) that need
        to walk the full row set without per-customer scoping.
        callers MUST follow up with per-id :meth:`find_by_id` calls
        when they need the full entity -- this helper deliberately
        returns ONLY ids so the L3 round-trip stays small and the
        per-row decode happens through the canonical
        ``find_by_id`` path (which also writes the L1 cache).

        status filter: only rows with ``status = 'active'`` are
        returned. ``DataSourceStatus.DISABLED`` rows are skipped
        because the operator intentionally disabled them and
        probing them every sweep wastes warehouse round-trips +
        emits spurious audit failure rows. audit-pass-3 CRITICAL-1.

        cross-customer by design: the scheduler is one-process,
        not per-customer, and the v054 ``UNIQUE (id)`` constraint
        makes per-id resolution unambiguous without the partition
        column. callers in per-customer contexts MUST NOT use this
        helper -- use the admin-endpoint keyset-paginated list
        with proper partition filtering instead.

        :return: list of ``id`` values for every ACTIVE row,
            ordered by ``(date_created, id)`` so the sweep order
            is deterministic across restarts
        :rtype: list[UUID]
        """
        if self.l3_pool is None:
            return []
        # cache-bypass: scheduler sweep needs every active row; no
        # Collection surface exists for cross-partition list-all by
        # design.
        # partition-bypass: cross-customer sweep is documented above.
        rows = await self.l3_pool.fetch(
            """
            SELECT id FROM datasources
            WHERE status = $1
            ORDER BY date_created, id
            """,
            DataSourceStatus.ACTIVE.value,
        )
        result = [row["id"] for row in rows]
        return result

    async def find_by_id(
        self,
        datasource_id: UUID,
    ) -> CapabilitySourceEntity | None:
        """resolve a capability source by ``id`` alone via the v054 ``UNIQUE (id)``.

        the admin endpoints (GET / DELETE / connection-config update)
        and the agent-side tool flow take ``{datasource_id}`` in the
        URL but not the partition column ``customer_id``. uniqueness
        is preserved by the ``UNIQUE (id)`` constraint added by hub
        migration v054.

        :param datasource_id: capability-source UUID
        :ptype datasource_id: UUID
        :return: capability-source entity or ``None`` when no row exists
        :rtype: CapabilitySourceEntity | None
        """
        read_ticket = self.scan_ticket()  # before the first await
        result: CapabilitySourceEntity | None = None
        if self.l3_pool is not None:
            row = await self.l3_pool.fetchrow(
                "SELECT * FROM datasources WHERE id = $1",
                datasource_id,
            )
            if row is not None:
                data = self._coerce_row(dict(row))
                self.write_to_cache_sync(data, read_since=read_ticket)
                result = self.entity_class(data, is_new=False, collection=self)
        return result

    async def resolve_origin_datasource_id(
        self,
        datasource_id: UUID,
    ) -> UUID | None:
        """return a datasource's ``origin_datasource_id`` (its shared link).

        knowledge-task-08 (KNW-77): a customer datasource D links to its
        canonical platform-shared datasource P via ``origin_datasource_id``;
        turn-time retrieval gathers knowledge across ``datasource_id IN
        (D, P)``. this is the single read both the hub effective-view
        serving query and the SDK retrieval call use to resolve P from D
        without threading module-level state — a fresh per-call lookup
        over the (rbac-read / hub) pool.

        :param datasource_id: the customer datasource D to resolve P for
        :ptype datasource_id: UUID
        :return: the linked platform-shared datasource id P, or ``None``
            when D carries no origin link
        :rtype: UUID | None
        """
        origin: UUID | None = None
        if self.l3_pool is not None:
            # cache-bypass: one-column projection by id; the by-pk
            # Collection get would decode the full row + write the L1
            # cache, but this is a hot per-turn lookup that only needs
            # the single origin column.
            row = await self.l3_pool.fetchrow(
                "SELECT origin_datasource_id FROM datasources WHERE id = $1",
                datasource_id,
            )
            if row is not None:
                origin = row["origin_datasource_id"]
        return origin


class DataSourceTableCollection(DeclaredTableCollection[DataSourceTableEntity]):
    """three-tier collection for data source table entities.

    provides CRUD operations with L1 -> L2 -> L3 caching.
    data source tables are hard-deleted (no soft-delete pattern).
    """

    schema: ClassVar[TableSchema] = _DATASOURCE_TABLES_SCHEMA

    #: ``coverage_dimension`` (v033) is written on insert and never on update. the per-region
    #: coverage designation of an existing table is set by the hub's admin table PATCH and moved by
    #: the data-upgrade rename carry, both direct writes. the one writer that saves an existing row
    #: through this collection is the schema introspector, which saves back the row it prefetched
    #: at the start of a pass; writing its copy of the designation would undo a PATCH that landed
    #: during the pass. a write carrying it reads the row back.
    insert_only_columns: ClassVar[tuple[str, ...]] = ("coverage_dimension",)

    #: switched on (epoch-task-06): its writes advance the table's write generation, so a cache
    #: derived from it (a visibility scan) is evicted when a broadcast is missed, not timed out
    write_generation = WRITE_GENERATION

    @property
    def entity_class(self) -> type[DataSourceTableEntity]:
        """return entity class for this collection.

        :return: DataSourceTableEntity class
        :rtype: type[DataSourceTableEntity]
        """
        return DataSourceTableEntity

    async def fetch_from_store(self, entity_id: Any) -> dict[str, Any] | None:
        """fetch the row from L3 by primary key, on the collection's pool.

        :param entity_id: the primary key
        :ptype entity_id: Any
        :return: row dict on hit, ``None`` on miss or with no L3 pool
        :rtype: dict[str, Any] | None
        """
        if self.l3_pool is None:
            return None
        return await self._select_row(self.l3_pool, entity_id)

    async def fetch_from_store_in_transaction(self, entity_id: Any, *, conn: Any) -> dict[str, Any] | None:
        """fetch the row by primary key on a caller's open transaction, with :meth:`fetch_from_store`'s SELECT.

        :param entity_id: the primary key
        :ptype entity_id: Any
        :param conn: the caller's connection, with its transaction open
        :ptype conn: Any
        :return: row dict on hit, ``None`` on miss
        :rtype: dict[str, Any] | None
        """
        return await self._select_row(conn, entity_id)

    async def _select_row(self, target: Any, entity_id: Any) -> dict[str, Any] | None:
        """the row by primary key, read on ``target``: the collection's pool, or a caller's connection.

        :param target: the pool or connection the SELECT runs on
        :ptype target: Any
        :param entity_id: data source table UUID
        :ptype entity_id: Any
        :return: table data dictionary or None if not found
        :rtype: dict[str, Any] | None
        """
        row = await target.fetchrow(
            "SELECT * FROM datasource_tables WHERE id = $1",
            entity_id,
        )
        result: dict[str, Any] | None = None
        if row is not None:
            data = dict(row)
            result = data
        return result

    async def save_to_store(
        self,
        data: dict[str, Any],
        original_timestamp: datetime | None = None,
        *,
        conn: Any = None,
    ) -> int:
        """upsert data source table record to L3, writing only the columns ``data`` carries.

        a column ``data`` does not carry is left to the table: its default on
        an insert, its stored value on an update. ``caveats_replaces_definition``
        given as ``None`` counts as not carried, so the ``NOT NULL DEFAULT
        FALSE`` column keeps its additive-concat default. ``column_hash`` given
        as ``None`` IS carried: it is the "force re-introspect" sentinel.
        ``coverage_dimension`` is written on insert and never on update (see
        :attr:`insert_only_columns`).

        :param data: entity field data to persist
        :ptype data: dict[str, Any]
        :param original_timestamp: unused; this table has no CAS fence
        :ptype original_timestamp: datetime | None
        :param conn: the caller's connection the write joins, or ``None`` for the pool
        :ptype conn: Any
        :return: number of rows affected
        :rtype: int
        :raises RowCannotBeInsertedError: when a row short of a ``NOT NULL`` column finds no
            row to update (see :meth:`DeclaredTableCollection.write_row`)
        """
        return await self.write_row(data, conn=conn)

    async def delete_from_store(self, entity_id: Any) -> None:
        """hard-delete data source table from L3.

        :param entity_id: data source table UUID
        :ptype entity_id: Any
        :return: nothing
        :rtype: None
        """
        if self.l3_pool is None:
            return
        await self.l3_pool.execute(
            "DELETE FROM datasource_tables WHERE id = $1",
            entity_id,
        )

    async def get_by_natural_key(
        self,
        datasource_id: UUID,
        schema_name: str,
        table_name: str,
    ) -> DataSourceTableEntity | None:
        """resolve a table row by its ``(datasource_id, schema, table)`` natural key.

        the introspector's "insert vs update" decision keys on the
        natural unique constraint, not on the row's id. cache-bypass
        by design: the natural-key lookup is only used in
        introspection workflows (Hub-orchestrator concern, not the
        hot read path); a future optimization can add a natural-key
        secondary index in L1 if measurements justify it.

        :param datasource_id: owning datasource UUID
        :ptype datasource_id: UUID
        :param schema_name: ``information_schema.tables.table_schema``
        :ptype schema_name: str
        :param table_name: ``information_schema.tables.table_name``
        :ptype table_name: str
        :return: the entity if a row exists, ``None`` otherwise
        :rtype: DataSourceTableEntity | None
        """
        result: DataSourceTableEntity | None = None
        if self.l3_pool is not None:
            row = await self.l3_pool.fetchrow(
                """
                SELECT * FROM datasource_tables
                WHERE datasource_id = $1
                AND schema_name = $2
                AND table_name = $3
                """,
                datasource_id,
                schema_name,
                table_name,
            )
            if row is not None:
                data = dict(row)
                result = self.entity_class(data, is_new=False, collection=self)
        return result


class DataSourceColumnCollection(DeclaredTableCollection[DataSourceColumnEntity]):
    """three-tier collection for data source column entities.

    provides CRUD operations with L1 -> L2 -> L3 caching.
    data source columns are hard-deleted (no soft-delete pattern).
    tags is stored as JSONB array in L3.
    """

    schema: ClassVar[TableSchema] = _DATASOURCE_COLUMNS_SCHEMA

    #: re-introspection upserts on the natural key, so a column keeps its row and its id
    conflict_columns: ClassVar[tuple[str, ...]] = ("datasource_id", "schema_name", "table_name", "column_name")

    @property
    def entity_class(self) -> type[DataSourceColumnEntity]:
        """return entity class for this collection.

        :return: DataSourceColumnEntity class
        :rtype: type[DataSourceColumnEntity]
        """
        return DataSourceColumnEntity

    async def fetch_from_store(self, entity_id: Any) -> dict[str, Any] | None:
        """fetch the row from L3 by primary key, on the collection's pool.

        :param entity_id: the primary key
        :ptype entity_id: Any
        :return: row dict on hit, ``None`` on miss or with no L3 pool
        :rtype: dict[str, Any] | None
        """
        if self.l3_pool is None:
            return None
        return await self._select_row(self.l3_pool, entity_id)

    async def fetch_from_store_in_transaction(self, entity_id: Any, *, conn: Any) -> dict[str, Any] | None:
        """fetch the row by primary key on a caller's open transaction, with :meth:`fetch_from_store`'s SELECT.

        :param entity_id: the primary key
        :ptype entity_id: Any
        :param conn: the caller's connection, with its transaction open
        :ptype conn: Any
        :return: row dict on hit, ``None`` on miss
        :rtype: dict[str, Any] | None
        """
        return await self._select_row(conn, entity_id)

    async def _select_row(self, target: Any, entity_id: Any) -> dict[str, Any] | None:
        """the row by primary key, read on ``target``: the collection's pool, or a caller's connection.

        :param target: the pool or connection the SELECT runs on
        :ptype target: Any
        :param entity_id: data source column UUID
        :ptype entity_id: Any
        :return: column data dictionary or None if not found
        :rtype: dict[str, Any] | None
        """
        row = await target.fetchrow(
            "SELECT * FROM datasource_columns WHERE id = $1",
            entity_id,
        )
        result: dict[str, Any] | None = None
        if row is not None:
            # ``tags`` comes back as a python list via the jsonb codec / proxy
            # NATS-JSON decode -- no per-collection json.loads (collections-task-04).
            result = dict(row)
        return result

    async def save_to_store(
        self,
        data: dict[str, Any],
        original_timestamp: datetime | None = None,
        *,
        conn: Any = None,
    ) -> int:
        """upsert data source column record to L3 on its natural key, writing only the columns ``data`` carries.

        the conflict target is ``(datasource_id, schema_name, table_name, column_name)``
        (:attr:`conflict_columns`), so re-introspection updates the stored column. the
        statement returns the id of the row it wrote. two introspections racing on one
        new column each send a fresh id; the second lands on the row the first stored,
        under the first's id. that row is evicted from every tier, and the second's own
        id, which holds no row, is cached nowhere
        (:meth:`~threetears.core.collections.declared_table.DeclaredTableCollection.settle_stored_key`).
        ``caveats_replaces_definition`` given as ``None`` counts as not carried, so the
        ``NOT NULL DEFAULT FALSE`` column keeps its additive-concat default.

        :param data: entity field data to persist
        :ptype data: dict[str, Any]
        :param original_timestamp: unused; this table has no CAS fence
        :ptype original_timestamp: datetime | None
        :param conn: the caller's connection the write joins, or ``None`` for the pool
        :ptype conn: Any
        :return: number of rows affected
        :rtype: int
        :raises RowCannotBeInsertedError: when a row short of a ``NOT NULL`` column finds no
            row to update (see :meth:`DeclaredTableCollection.write_row`)
        """
        return await self.write_row(data, conn=conn)

    async def delete_from_store(self, entity_id: Any) -> None:
        """hard-delete data source column from L3.

        :param entity_id: data source column UUID
        :ptype entity_id: Any
        :return: nothing
        :rtype: None
        """
        if self.l3_pool is None:
            return
        await self.l3_pool.execute(
            "DELETE FROM datasource_columns WHERE id = $1",
            entity_id,
        )

    async def get_by_natural_key(
        self,
        datasource_id: UUID,
        schema_name: str,
        table_name: str,
        column_name: str,
    ) -> DataSourceColumnEntity | None:
        """resolve a column row by its natural key.

        the natural unique constraint is
        ``(datasource_id, schema_name, table_name, column_name)``.
        the introspector uses this to decide insert vs update during
        per-table re-introspect.

        :param datasource_id: owning datasource UUID
        :ptype datasource_id: UUID
        :param schema_name: schema name
        :ptype schema_name: str
        :param table_name: table name
        :ptype table_name: str
        :param column_name: column name
        :ptype column_name: str
        :return: the entity if a row exists, ``None`` otherwise
        :rtype: DataSourceColumnEntity | None
        """
        result: DataSourceColumnEntity | None = None
        if self.l3_pool is not None:
            row = await self.l3_pool.fetchrow(
                """
                SELECT * FROM datasource_columns
                WHERE datasource_id = $1
                AND schema_name = $2
                AND table_name = $3
                AND column_name = $4
                """,
                datasource_id,
                schema_name,
                table_name,
                column_name,
            )
            if row is not None:
                # ``tags`` is a python list via the jsonb codec / proxy decode.
                result = self.entity_class(dict(row), is_new=False, collection=self)
        return result


class DataSourceSchemaDigestCollection(
    DeclaredTableCollection[DataSourceSchemaDigestEntity],
):
    """three-tier collection for the materialized documented-schema digest.

    one row per datasource, addressed BY PRIMARY KEY ``datasource_id`` so
    the agent-side read (schema-priming-task-01b) is a by-pk hot-L1
    lookup, with L2/L3 fallback for a cold pod and cross-pod invalidation
    when the hub re-materializes. the hub is the only writer (the
    materializer reuses the existing documented-schema computation); agent
    pods bind this SAME class over the ``system.platform.rbac`` proxy pool
    and read only.

    the ``tables`` projection is stored as JSONB. digest rows are
    hard-deleted (no soft-delete) — a datasource removal drops its digest.
    """

    # the L1/L2 key is the schema's primary key, ``datasource_id``: this table has NO ``id``
    # column, and keying by the ``"id"`` default would break every by-pk read and invalidation.
    schema: ClassVar[TableSchema] = _DATASOURCE_SCHEMA_DIGESTS_SCHEMA

    @property
    def entity_class(self) -> type[DataSourceSchemaDigestEntity]:
        """return entity class for this collection.

        :return: DataSourceSchemaDigestEntity class
        :rtype: type[DataSourceSchemaDigestEntity]
        """
        return DataSourceSchemaDigestEntity

    async def fetch_from_store(self, entity_id: Any) -> dict[str, Any] | None:
        """fetch the row from L3 by primary key, on the collection's pool.

        :param entity_id: the primary key
        :ptype entity_id: Any
        :return: row dict on hit, ``None`` on miss or with no L3 pool
        :rtype: dict[str, Any] | None
        """
        if self.l3_pool is None:
            return None
        return await self._select_row(self.l3_pool, entity_id)

    async def fetch_from_store_in_transaction(self, entity_id: Any, *, conn: Any) -> dict[str, Any] | None:
        """fetch the row by primary key on a caller's open transaction, with :meth:`fetch_from_store`'s SELECT.

        :param entity_id: the primary key
        :ptype entity_id: Any
        :param conn: the caller's connection, with its transaction open
        :ptype conn: Any
        :return: row dict on hit, ``None`` on miss
        :rtype: dict[str, Any] | None
        """
        return await self._select_row(conn, entity_id)

    async def _select_row(self, target: Any, entity_id: Any) -> dict[str, Any] | None:
        """the row by primary key, read on ``target``: the collection's pool, or a caller's connection.

        :param target: the pool or connection the SELECT runs on
        :ptype target: Any
        :param entity_id: datasource UUID (the digest primary key)
        :ptype entity_id: Any
        :return: digest data dictionary or None if not found
        :rtype: dict[str, Any] | None
        """
        row = await target.fetchrow(
            "SELECT * FROM datasource_schema_digests WHERE datasource_id = $1",
            entity_id,
        )
        result: dict[str, Any] | None = None
        if row is not None:
            # ``tables`` comes back already decoded to a python list: the hub
            # l3 pool's jsonb codec decodes the direct read, and the agent's
            # NatsProxyL3Backend read decodes the NATS-JSON array. NO manual
            # json.loads -- collections-task-04 removed the per-collection
            # decode that mirrored the (now deleted) write-side json.dumps.
            result = dict(row)
        return result

    async def save_to_store(
        self,
        data: dict[str, Any],
        original_timestamp: datetime | None = None,
        *,
        conn: Any = None,
    ) -> int:
        """upsert the digest row to L3 keyed on ``datasource_id``, writing only the columns ``data`` carries.

        ``date_created`` is rewritten with the rest: the digest re-materializes through a
        fresh ``create()``, so it tracks the last write, and L3 agrees with the value every
        tier caches. ``tables`` is JSONB bound natively (the jsonb codec applies the single
        encode).

        :param data: entity field data to persist
        :ptype data: dict[str, Any]
        :param original_timestamp: unused; this table has no CAS fence
        :ptype original_timestamp: datetime | None
        :param conn: the caller's connection the write joins, or ``None`` for the pool
        :ptype conn: Any
        :return: number of rows affected
        :rtype: int
        """
        return await self.write_row(data, conn=conn)

    async def delete_from_store(self, entity_id: Any) -> None:
        """hard-delete the digest row from L3.

        :param entity_id: datasource UUID (the digest primary key)
        :ptype entity_id: Any
        :return: nothing
        :rtype: None
        """
        if self.l3_pool is None:
            return
        await self.l3_pool.execute(
            "DELETE FROM datasource_schema_digests WHERE datasource_id = $1",
            entity_id,
        )


class DataSourceRelationCollection(DeclaredTableCollection[DataSourceRelationEntity]):
    """three-tier collection for data source relation entities.

    provides CRUD operations with L1 -> L2 -> L3 caching.
    data source relations are hard-deleted (no soft-delete pattern).
    datasource_ids and join_paths are stored as JSONB arrays in L3.
    """

    schema: ClassVar[TableSchema] = _DATASOURCE_RELATIONS_SCHEMA

    @property
    def entity_class(self) -> type[DataSourceRelationEntity]:
        """return entity class for this collection.

        :return: DataSourceRelationEntity class
        :rtype: type[DataSourceRelationEntity]
        """
        return DataSourceRelationEntity

    async def fetch_from_store(self, entity_id: Any) -> dict[str, Any] | None:
        """fetch the row from L3 by primary key, on the collection's pool.

        :param entity_id: the primary key
        :ptype entity_id: Any
        :return: row dict on hit, ``None`` on miss or with no L3 pool
        :rtype: dict[str, Any] | None
        """
        if self.l3_pool is None:
            return None
        return await self._select_row(self.l3_pool, entity_id)

    async def fetch_from_store_in_transaction(self, entity_id: Any, *, conn: Any) -> dict[str, Any] | None:
        """fetch the row by primary key on a caller's open transaction, with :meth:`fetch_from_store`'s SELECT.

        :param entity_id: the primary key
        :ptype entity_id: Any
        :param conn: the caller's connection, with its transaction open
        :ptype conn: Any
        :return: row dict on hit, ``None`` on miss
        :rtype: dict[str, Any] | None
        """
        return await self._select_row(conn, entity_id)

    async def _select_row(self, target: Any, entity_id: Any) -> dict[str, Any] | None:
        """the row by primary key, read on ``target``: the collection's pool, or a caller's connection.

        :param target: the pool or connection the SELECT runs on
        :ptype target: Any
        :param entity_id: data source relation UUID
        :ptype entity_id: Any
        :return: relation data dictionary or None if not found
        :rtype: dict[str, Any] | None
        """
        row = await target.fetchrow(
            "SELECT * FROM datasource_relations WHERE id = $1",
            entity_id,
        )
        result: dict[str, Any] | None = None
        if row is not None:
            # ``datasource_ids`` / ``join_paths`` come back as python lists via
            # the jsonb codec / proxy decode -- no per-collection json.loads
            # (collections-task-04).
            result = dict(row)
        return result

    async def save_to_store(
        self,
        data: dict[str, Any],
        original_timestamp: datetime | None = None,
        *,
        conn: Any = None,
    ) -> int:
        """upsert data source relation record to L3, writing only the columns ``data`` carries.

        ``customer_id`` (hub v056) is the relation's scope: NULL is
        platform-shared, a value is that customer's. it is written when
        ``data`` carries it, so a row read and saved back keeps its scope,
        and a row that does not carry it never resets a stored scope to
        platform-shared. ``datasource_ids``, ``join_paths`` and ``edges`` are
        JSONB bound natively (the jsonb codec applies the single encode,
        collections-task-04); ``None`` for any of them counts as not carried,
        since each is ``NOT NULL DEFAULT '[]'``.

        :param data: entity field data to persist
        :ptype data: dict[str, Any]
        :param original_timestamp: unused; this table has no CAS fence
        :ptype original_timestamp: datetime | None
        :param conn: the caller's connection the write joins, or ``None`` for the pool
        :ptype conn: Any
        :return: number of rows affected
        :rtype: int
        :raises RowCannotBeInsertedError: when a row short of a ``NOT NULL`` column finds no
            row to update (see :meth:`DeclaredTableCollection.write_row`)
        """
        return await self.write_row(data, conn=conn)

    async def delete_from_store(self, entity_id: Any) -> None:
        """hard-delete data source relation from L3.

        :param entity_id: data source relation UUID
        :ptype entity_id: Any
        :return: nothing
        :rtype: None
        """
        if self.l3_pool is None:
            return
        await self.l3_pool.execute(
            "DELETE FROM datasource_relations WHERE id = $1",
            entity_id,
        )


class TableTemplateCollection(DeclaredTableCollection[TableTemplateEntity]):
    """three-tier collection for table-template entities.

    provides CRUD with L1 -> L2 -> L3 caching for template definition
    rows, addressed by ``id`` alone: hub v007 rebuilt the primary key on
    ``id`` so a platform-owned template (``visibility`` ``public`` or
    ``restricted``) can carry ``customer_id`` NULL. the unique index on
    ``(customer_id, name)`` keeps slug collisions inside a customer's
    namespace.

    ``visibility`` is required on every write and never defaulted: the
    column's ``'private'`` default is only valid with a customer, and a
    save that silently wrote it over a public template would hide that
    template from every other customer. ``origin_template_id`` points a
    promoted platform copy back at the customer template it came from.

    templates are hard-deleted; the FK from
    ``datasource_tables.template_id`` is ``ON DELETE SET NULL`` so
    deleting a template never destroys instance metadata, and the FK
    from ``table_template_columns.template_id`` is ``ON DELETE
    CASCADE`` so the per-template column list goes with it.
    """

    schema: ClassVar[TableSchema] = _TABLE_TEMPLATES_SCHEMA

    # datasource_tables.template_id is ON DELETE SET NULL: a bound table's cached row would keep
    # naming the deleted template, so every delete announces the tables it unbinds
    delete_cascades: ClassVar[tuple[DeleteCascade, ...]] = (
        DeleteCascade("datasource_tables", "template_id", "SET NULL"),
    )

    @property
    def entity_class(self) -> type[TableTemplateEntity]:
        """return entity class for this collection.

        :return: TableTemplateEntity class
        :rtype: type[TableTemplateEntity]
        """
        return TableTemplateEntity

    async def fetch_from_store(self, entity_id: Any) -> dict[str, Any] | None:
        """fetch the row from L3 by primary key, on the collection's pool.

        :param entity_id: the primary key
        :ptype entity_id: Any
        :return: row dict on hit, ``None`` on miss or with no L3 pool
        :rtype: dict[str, Any] | None
        """
        if self.l3_pool is None:
            return None
        return await self._select_row(self.l3_pool, entity_id)

    async def fetch_from_store_in_transaction(self, entity_id: Any, *, conn: Any) -> dict[str, Any] | None:
        """fetch the row by primary key on a caller's open transaction, with :meth:`fetch_from_store`'s SELECT.

        :param entity_id: the primary key
        :ptype entity_id: Any
        :param conn: the caller's connection, with its transaction open
        :ptype conn: Any
        :return: row dict on hit, ``None`` on miss
        :rtype: dict[str, Any] | None
        """
        return await self._select_row(conn, entity_id)

    async def _select_row(self, target: Any, entity_id: Any) -> dict[str, Any] | None:
        """the row by primary key, read on ``target``: the collection's pool, or a caller's connection.

        :param target: the pool or connection the SELECT runs on
        :ptype target: Any
        :param entity_id: template UUID
        :ptype entity_id: Any
        :return: template data dictionary or None if not found
        :rtype: dict[str, Any] | None
        """
        row = await target.fetchrow(
            "SELECT * FROM table_templates WHERE id = $1",
            entity_id,
        )
        result: dict[str, Any] | None = None
        if row is not None:
            result = dict(row)
        return result

    async def save_to_store(
        self,
        data: dict[str, Any],
        original_timestamp: datetime | None = None,
        *,
        conn: Any = None,
    ) -> int:
        """upsert table-template row to L3 on its primary key, writing only the columns ``data`` carries.

        the conflict target is ``id``, the primary key since hub v007. a
        re-save with the same id keeps the row's identity; the unique index
        on ``(customer_id, name)`` still refuses a slug collision inside a
        customer's namespace.

        :param data: entity field data to persist
        :ptype data: dict[str, Any]
        :param original_timestamp: unused; this table has no CAS fence
        :ptype original_timestamp: datetime | None
        :param conn: the caller's connection the write joins, or ``None`` for the pool
        :ptype conn: Any
        :return: number of rows affected
        :rtype: int
        :raises ValueError: when ``data`` carries no ``visibility``
        :raises RowCannotBeInsertedError: when a row short of a ``NOT NULL`` column finds no
            row to update (see :meth:`DeclaredTableCollection.write_row`)
        """
        if data.get("visibility") is None:
            raise ValueError(
                f"table_templates: template {data.get('id')} carries no visibility. every write names it "
                f"('private' with a customer_id, 'public' or 'restricted' without one); the column default "
                f"is never assumed, since writing 'private' over a platform template hides it"
            )
        return await self.write_row(data, conn=conn)

    async def delete_from_store(self, entity_id: Any) -> None:
        """hard-delete template row from L3.

        :param entity_id: template UUID
        :ptype entity_id: Any
        :return: nothing
        :rtype: None
        """
        if self.l3_pool is None:
            return
        await self.l3_pool.execute(
            "DELETE FROM table_templates WHERE id = $1",
            entity_id,
        )
