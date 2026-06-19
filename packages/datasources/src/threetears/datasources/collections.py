"""three-tier collections for datasource registry, schema metadata, and table templates.

merged from Hub's ``aibots/hub/datasources/collections.py``,
``schema_collections.py``, and ``template_collections.py`` per
``datasource-task-07``. class definitions are byte-identical to the Hub
originals -- this shard is pure relocation, not refactor.

collections in this module:

- :class:`DataSourceCollection` -- ``SchemaBackedCollection`` for the
  ``platform.datasources`` registry. composite PK
  ``(customer_id, id)`` with a ``find_by_id`` helper that uses the
  v054 ``UNIQUE (id)`` constraint for partition-exempt lookups.
- :class:`DataSourceTableCollection` -- ``BaseCollection`` for the
  ``platform.datasource_tables`` row set.
- :class:`DataSourceColumnCollection` -- ``BaseCollection`` for
  ``platform.datasource_columns``. natural-key upsert on
  ``(datasource_id, schema_name, table_name, column_name)``.
- :class:`DataSourceRelationCollection` -- ``BaseCollection`` for
  ``platform.datasource_relations``.
- :class:`TableTemplateCollection` -- ``BaseCollection`` for
  ``platform.table_templates``. composite PK ``(customer_id, id)``.

per-table column variants (``TableTemplateColumnCollection``) stay in
Hub for now because they have no cross-consumer demand yet; lift later
if a second 3tears consumer needs them.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any
from uuid import UUID

from threetears.core.collections.base import BaseCollection
from threetears.core.collections.schema_backed import (
    DATETIMETZ_TYPE,
    JSONB_TYPE,
    STRING_TYPE,
    UUID_TYPE,
    Column,
    SchemaBackedCollection,
    TableSchema,
    encode_jsonb,
)
from threetears.core.serialization import deserialize_from_json, serialize_to_json
from threetears.observe import get_logger

from threetears.datasources.entities import (
    DataSourceColumnEntity,
    DataSourceEntity,
    DataSourceRelationEntity,
    DataSourceSchemaDigestEntity,
    DataSourceStatus,
    DataSourceTableEntity,
    TableTemplateEntity,
)

log = get_logger(__name__)


__all__ = [
    "DataSourceCollection",
    "DataSourceColumnCollection",
    "DataSourceRelationCollection",
    "DataSourceSchemaDigestCollection",
    "DataSourceTableCollection",
    "TableTemplateCollection",
]


_TABLE_FIELD_TYPES: dict[str, Any] = {
    "id": UUID,
    "datasource_id": UUID,
    "schema_name": str,
    "table_name": str,
    "description": str,
    "row_count_approx": int,
    "caveats": str,
    # template-task-01: nullable FK into platform.table_templates;
    # most tables stay unbound (template_id IS NULL). when bound,
    # the read-time merge in template-task-02 overlays the
    # template's column docs onto this row's instance docs.
    "template_id": UUID,
    # template-task-01: when True, the merged caveat returns the
    # instance caveat alone (full override) instead of concatenating
    # the template caveat. default FALSE keeps the additive concat.
    "caveats_replaces_definition": bool,
    # datasource-task-02: per-table column-shape MD5 (the Tier-2
    # change-probe digest). NULL = "force re-introspect" sentinel.
    # the warehouse-side SQL in AsyncpgDriver / RedshiftDriver's
    # ``table_hashes`` byte-equal to ``compute_column_hash`` over
    # the same column set.
    "column_hash": str,
    "date_introspected": datetime,
    "date_described": datetime,
    "date_created": datetime,
    "date_updated": datetime,
}

_SCHEMA_DIGEST_FIELD_TYPES: dict[str, Any] = {
    "datasource_id": UUID,
    "customer_id": UUID,
    # structured documented projection:
    # [{schema, table, description, columns: [{name, type, description}]}]
    "tables": list,
    "source_fingerprint": str,
    "date_created": datetime,
    "date_updated": datetime,
}

_COLUMN_FIELD_TYPES: dict[str, Any] = {
    "id": UUID,
    "datasource_id": UUID,
    "schema_name": str,
    "table_name": str,
    "column_name": str,
    "data_type": str,
    "is_nullable": bool,
    "ordinal_position": int,
    "description": str,
    "valid_range": str,
    "caveats": str,
    "tags": list,
    # template-task-01: per-column override of the additive caveat
    # concat rule. mirrors the table-level flag for the rare case
    # where the instance docs replace rather than augment the
    # template's column-level caveats.
    "caveats_replaces_definition": bool,
    "date_introspected": datetime,
    "date_described": datetime,
    "date_created": datetime,
    "date_updated": datetime,
}

_RELATION_FIELD_TYPES: dict[str, Any] = {
    "id": UUID,
    "name": str,
    "description": str,
    "datasource_ids": list,
    "join_paths": list,
    "aggregation_notes": str,
    "caveats": str,
    "date_created": datetime,
    "date_updated": datetime,
}

_TEMPLATE_FIELD_TYPES: dict[str, Any] = {
    "id": UUID,
    "customer_id": UUID,
    "name": str,
    "description": str,
    "caveats": str,
    "date_created": datetime,
    "date_updated": datetime,
}


class DataSourceCollection(SchemaBackedCollection[DataSourceEntity]):
    """three-tier collection for data source entities.

    provides CRUD operations with L1 -> L2 -> L3 caching. data sources
    are hard-deleted (no soft-delete pattern). ``connection_config`` is
    stored as encrypted JSON string in L3 (kept as ``STRING_TYPE`` so
    the SchemaBackedCollection write path passes the ciphertext through
    unchanged); ``allowed_schemas`` is stored as a JSONB array. CRUD
    comes from the declarative :class:`TableSchema`; no domain queries
    live on the subclass today (every callsite resolves by primary key
    or filters in admin endpoints via the L3 pool with cache-bypass
    rationales).
    """

    primary_key_column: str = "id"
    schema = TableSchema(
        name="datasources",
        primary_key="id",
        columns=[
            Column("id", UUID_TYPE),
            Column("name", STRING_TYPE),
            # customer_id is nullable + a plain column post-knowledge-
            # task-08 (KNW-76): a platform-shared datasource (visibility
            # != 'private') carries customer_id NULL. v016 rebuilt the
            # table PK on ``id`` alone (dropping the v001 composite
            # partition PK) so the addressing key is the global ``id``,
            # backed by datasources_id_unique; a NULL customer_id never
            # blocks resolution.
            Column("customer_id", UUID_TYPE, nullable=True),
            Column("datasource_type", STRING_TYPE, immutable=True),
            # connection_config is nullable post-v056: agent_internal
            # rows carry no external connection config because the
            # broker routes via the L3 broker bound to schema_name.
            Column("connection_config", STRING_TYPE, nullable=True),
            Column("allowed_schemas", JSONB_TYPE, nullable=True),
            Column("access_mode", STRING_TYPE),
            Column("status", STRING_TYPE),
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
            Column("visibility", STRING_TYPE),
            Column("origin_datasource_id", UUID_TYPE, nullable=True),
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
    def entity_class(self) -> type[DataSourceEntity]:
        """return entity class for this collection.

        :return: DataSourceEntity class
        :rtype: type[DataSourceEntity]
        """
        return DataSourceEntity

    async def iter_active_ids(self) -> list[UUID]:
        """list every ACTIVE datasource's primary-key ``id``.

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
    ) -> DataSourceEntity | None:
        """resolve a datasource by ``id`` alone via the v054 ``UNIQUE (id)``.

        the admin endpoints (GET / DELETE / connection-config update)
        and the agent-side tool flow take ``{datasource_id}`` in the
        URL but not the partition column ``customer_id``. uniqueness
        is preserved by the ``UNIQUE (id)`` constraint added by hub
        migration v054.

        :param datasource_id: data source UUID
        :ptype datasource_id: UUID
        :return: datasource entity or ``None`` when no row exists
        :rtype: DataSourceEntity | None
        """
        result: DataSourceEntity | None = None
        if self.l3_pool is not None:
            row = await self.l3_pool.fetchrow(
                "SELECT * FROM datasources WHERE id = $1",
                datasource_id,
            )
            if row is not None:
                data = self._coerce_row(dict(row))
                self.write_to_cache_sync(data)
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


class DataSourceTableCollection(BaseCollection[DataSourceTableEntity]):
    """three-tier collection for data source table entities.

    provides CRUD operations with L1 -> L2 -> L3 caching.
    data source tables are hard-deleted (no soft-delete pattern).
    """

    @property
    def table_name(self) -> str:
        """return database table name.

        :return: table name string
        :rtype: str
        """
        return "datasource_tables"

    @property
    def entity_class(self) -> type[DataSourceTableEntity]:
        """return entity class for this collection.

        :return: DataSourceTableEntity class
        :rtype: type[DataSourceTableEntity]
        """
        return DataSourceTableEntity

    def serialize(self, data: dict[str, Any]) -> bytes:
        """serialize entity data to JSON bytes for L2 cache.

        :param data: entity field data
        :ptype data: dict[str, Any]
        :return: JSON-encoded bytes
        :rtype: bytes
        """
        return serialize_to_json(data)

    def deserialize(self, data: bytes) -> dict[str, Any]:
        """deserialize JSON bytes from L2 cache to entity data.

        :param data: JSON-encoded bytes
        :ptype data: bytes
        :return: entity field data dictionary
        :rtype: dict[str, Any]
        """
        return deserialize_from_json(data, _TABLE_FIELD_TYPES)

    async def fetch_from_postgres(self, entity_id: Any) -> dict[str, Any] | None:
        """fetch data source table record from L3 by primary key.

        :param entity_id: data source table UUID
        :ptype entity_id: Any
        :return: table data dictionary or None if not found
        :rtype: dict[str, Any] | None
        """
        if self.l3_pool is None:
            return None
        row = await self.l3_pool.fetchrow(
            "SELECT * FROM datasource_tables WHERE id = $1",
            entity_id,
        )
        result: dict[str, Any] | None = None
        if row is not None:
            data = dict(row)
            result = data
        return result

    async def save_to_postgres(
        self,
        data: dict[str, Any],
        original_timestamp: datetime | None = None,
        *,
        conn: Any = None,
    ) -> int:
        """upsert data source table record to L3 with optimistic concurrency.

        :param data: entity field data to persist
        :ptype data: dict[str, Any]
        :param original_timestamp: original date_updated for concurrency check
        :ptype original_timestamp: datetime | None
        :return: number of rows affected
        :rtype: int
        """
        if self.l3_pool is None:
            return 0

        result = await self.l3_pool.execute(
            """
            INSERT INTO datasource_tables (
                id, datasource_id, schema_name, table_name, description,
                row_count_approx, caveats, template_id,
                caveats_replaces_definition, column_hash,
                date_introspected, date_described,
                date_created, date_updated
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14
            )
            ON CONFLICT (id) DO UPDATE SET
                datasource_id = EXCLUDED.datasource_id,
                schema_name = EXCLUDED.schema_name,
                table_name = EXCLUDED.table_name,
                description = EXCLUDED.description,
                row_count_approx = EXCLUDED.row_count_approx,
                caveats = EXCLUDED.caveats,
                template_id = EXCLUDED.template_id,
                caveats_replaces_definition = EXCLUDED.caveats_replaces_definition,
                column_hash = EXCLUDED.column_hash,
                date_introspected = EXCLUDED.date_introspected,
                date_described = EXCLUDED.date_described,
                date_updated = EXCLUDED.date_updated
            """,
            data.get("id"),
            data.get("datasource_id"),
            data.get("schema_name"),
            data.get("table_name"),
            data.get("description"),
            data.get("row_count_approx"),
            data.get("caveats"),
            data.get("template_id"),
            # template-task-01: explicit FALSE default at the write
            # boundary so legacy callers that don't set the flag get
            # the additive-concat semantics rather than NULL (which
            # the column rejects via NOT NULL).
            data.get("caveats_replaces_definition") or False,
            # datasource-task-02: column_hash is nullable. None is
            # the "force re-introspect" sentinel; the introspector
            # writes the digest after computing it over the column set.
            data.get("column_hash"),
            data.get("date_introspected"),
            data.get("date_described"),
            data.get("date_created"),
            data.get("date_updated"),
        )
        return int(result.split()[-1])

    async def delete_from_postgres(self, entity_id: Any) -> None:
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


class DataSourceColumnCollection(BaseCollection[DataSourceColumnEntity]):
    """three-tier collection for data source column entities.

    provides CRUD operations with L1 -> L2 -> L3 caching.
    data source columns are hard-deleted (no soft-delete pattern).
    tags is stored as JSONB array in L3.
    """

    @property
    def table_name(self) -> str:
        """return database table name.

        :return: table name string
        :rtype: str
        """
        return "datasource_columns"

    @property
    def entity_class(self) -> type[DataSourceColumnEntity]:
        """return entity class for this collection.

        :return: DataSourceColumnEntity class
        :rtype: type[DataSourceColumnEntity]
        """
        return DataSourceColumnEntity

    def serialize(self, data: dict[str, Any]) -> bytes:
        """serialize entity data to JSON bytes for L2 cache.

        :param data: entity field data
        :ptype data: dict[str, Any]
        :return: JSON-encoded bytes
        :rtype: bytes
        """
        return serialize_to_json(data)

    def deserialize(self, data: bytes) -> dict[str, Any]:
        """deserialize JSON bytes from L2 cache to entity data.

        :param data: JSON-encoded bytes
        :ptype data: bytes
        :return: entity field data dictionary
        :rtype: dict[str, Any]
        """
        return deserialize_from_json(data, _COLUMN_FIELD_TYPES)

    async def fetch_from_postgres(self, entity_id: Any) -> dict[str, Any] | None:
        """fetch data source column record from L3 by primary key.

        :param entity_id: data source column UUID
        :ptype entity_id: Any
        :return: column data dictionary or None if not found
        :rtype: dict[str, Any] | None
        """
        if self.l3_pool is None:
            return None
        row = await self.l3_pool.fetchrow(
            "SELECT * FROM datasource_columns WHERE id = $1",
            entity_id,
        )
        result: dict[str, Any] | None = None
        if row is not None:
            # ``tags`` comes back as a python list via the jsonb codec / proxy
            # NATS-JSON decode -- no per-collection json.loads (collections-task-04).
            result = dict(row)
        return result

    async def save_to_postgres(
        self,
        data: dict[str, Any],
        original_timestamp: datetime | None = None,
        *,
        conn: Any = None,
    ) -> int:
        """upsert data source column record to L3 with natural key conflict resolution.

        uses ON CONFLICT on (datasource_id, schema_name, table_name, column_name)
        natural key for upsert, allowing re-introspection to update existing columns.

        :param data: entity field data to persist
        :ptype data: dict[str, Any]
        :param original_timestamp: original date_updated for concurrency check
        :ptype original_timestamp: datetime | None
        :return: number of rows affected
        :rtype: int
        """
        if self.l3_pool is None:
            return 0

        result = await self.l3_pool.execute(
            """
            INSERT INTO datasource_columns (
                id, datasource_id, schema_name, table_name, column_name,
                data_type, is_nullable, ordinal_position, description,
                valid_range, caveats, tags, caveats_replaces_definition,
                date_introspected, date_described,
                date_created, date_updated
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11,
                -- $12 is bound NATIVELY (a python list via encode_jsonb); the
                -- jsonb codec applies the single json.dumps (collections-task-04).
                $12, $13, $14, $15, $16, $17
            )
            ON CONFLICT (datasource_id, schema_name, table_name, column_name) DO UPDATE SET
                data_type = EXCLUDED.data_type,
                is_nullable = EXCLUDED.is_nullable,
                ordinal_position = EXCLUDED.ordinal_position,
                description = EXCLUDED.description,
                valid_range = EXCLUDED.valid_range,
                caveats = EXCLUDED.caveats,
                tags = EXCLUDED.tags,
                caveats_replaces_definition = EXCLUDED.caveats_replaces_definition,
                date_introspected = EXCLUDED.date_introspected,
                date_described = EXCLUDED.date_described,
                date_updated = EXCLUDED.date_updated
            """,
            data.get("id"),
            data.get("datasource_id"),
            data.get("schema_name"),
            data.get("table_name"),
            data.get("column_name"),
            data.get("data_type"),
            data.get("is_nullable"),
            data.get("ordinal_position"),
            data.get("description"),
            data.get("valid_range"),
            data.get("caveats"),
            encode_jsonb(data.get("tags")),
            # template-task-01: NOT NULL column with FALSE default;
            # explicit fallback ensures legacy callers that omit the
            # field get the additive-concat semantics.
            data.get("caveats_replaces_definition") or False,
            data.get("date_introspected"),
            data.get("date_described"),
            data.get("date_created"),
            data.get("date_updated"),
        )
        return int(result.split()[-1])

    async def delete_from_postgres(self, entity_id: Any) -> None:
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
    BaseCollection[DataSourceSchemaDigestEntity],
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

    # the L1/L2 key is the SEPARATE ``primary_key_column`` attribute, NOT
    # the entity's ``primary_key_field``; it defaults to ``"id"`` on
    # BaseCollection. this table has NO ``id`` column (PK is
    # ``datasource_id``), so the default would emit ``WHERE id = ?`` /
    # ``ON CONFLICT (id)`` against the agent SQLite mirror + the hub L1
    # upsert and break every by-pk read + invalidation. it MUST name
    # ``datasource_id`` to match the entity PK + the v029 DDL.
    primary_key_column: str = "datasource_id"

    @property
    def table_name(self) -> str:
        """return database table name.

        :return: table name string
        :rtype: str
        """
        return "datasource_schema_digests"

    @property
    def entity_class(self) -> type[DataSourceSchemaDigestEntity]:
        """return entity class for this collection.

        :return: DataSourceSchemaDigestEntity class
        :rtype: type[DataSourceSchemaDigestEntity]
        """
        return DataSourceSchemaDigestEntity

    def serialize(self, data: dict[str, Any]) -> bytes:
        """serialize entity data to JSON bytes for L2 cache.

        :param data: entity field data
        :ptype data: dict[str, Any]
        :return: JSON-encoded bytes
        :rtype: bytes
        """
        return serialize_to_json(data)

    def deserialize(self, data: bytes) -> dict[str, Any]:
        """deserialize JSON bytes from L2 cache to entity data.

        :param data: JSON-encoded bytes
        :ptype data: bytes
        :return: entity field data dictionary
        :rtype: dict[str, Any]
        """
        return deserialize_from_json(data, _SCHEMA_DIGEST_FIELD_TYPES)

    async def fetch_from_postgres(self, entity_id: Any) -> dict[str, Any] | None:
        """fetch the digest row from L3 by primary key (``datasource_id``).

        :param entity_id: datasource UUID (the digest primary key)
        :ptype entity_id: Any
        :return: digest data dictionary or None if not found
        :rtype: dict[str, Any] | None
        """
        if self.l3_pool is None:
            return None
        row = await self.l3_pool.fetchrow(
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

    async def save_to_postgres(
        self,
        data: dict[str, Any],
        original_timestamp: datetime | None = None,
        *,
        conn: Any = None,
    ) -> int:
        """upsert the digest row to L3 keyed on ``datasource_id``.

        :param data: entity field data to persist
        :ptype data: dict[str, Any]
        :param original_timestamp: original date_updated for concurrency check
        :ptype original_timestamp: datetime | None
        :return: number of rows affected
        :rtype: int
        """
        if self.l3_pool is None:
            return 0

        result = await self.l3_pool.execute(
            """
            INSERT INTO datasource_schema_digests (
                datasource_id, customer_id, tables, source_fingerprint,
                date_created, date_updated
            ) VALUES (
                -- $3 is bound NATIVELY (a python list via encode_jsonb), NOT a
                -- pre-json.dumps'd string with a ::text::jsonb cast. the platform
                -- registers a text-format jsonb codec (threetears.core.collections.
                -- init_connection) on the hub l3 pool (the only pool that touches
                -- this table, shared by the broker), whose encoder is json.dumps.
                -- binding the native list lets the codec apply the SINGLE encode --
                -- collections-task-04 removed the per-collection json.dumps + cast
                -- that double-encoded the cell into a JSON STRING scalar.
                $1, $2, $3, $4, $5, $6
            )
            ON CONFLICT (datasource_id) DO UPDATE SET
                customer_id = EXCLUDED.customer_id,
                tables = EXCLUDED.tables,
                source_fingerprint = EXCLUDED.source_fingerprint,
                -- include date_created so L3 agrees with the L1/L2 value
                -- the collection stamps on every (is_new) re-materialize;
                -- omitting it diverges the tiers (the digest re-materializes
                -- via a fresh create(), so date_created tracks last-write).
                date_created = EXCLUDED.date_created,
                date_updated = EXCLUDED.date_updated
            """,
            data.get("datasource_id"),
            data.get("customer_id"),
            encode_jsonb(data.get("tables")),
            data.get("source_fingerprint"),
            data.get("date_created"),
            data.get("date_updated"),
        )
        return int(result.split()[-1])

    async def delete_from_postgres(self, entity_id: Any) -> None:
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


class DataSourceRelationCollection(BaseCollection[DataSourceRelationEntity]):
    """three-tier collection for data source relation entities.

    provides CRUD operations with L1 -> L2 -> L3 caching.
    data source relations are hard-deleted (no soft-delete pattern).
    datasource_ids and join_paths are stored as JSONB arrays in L3.
    """

    @property
    def table_name(self) -> str:
        """return database table name.

        :return: table name string
        :rtype: str
        """
        return "datasource_relations"

    @property
    def entity_class(self) -> type[DataSourceRelationEntity]:
        """return entity class for this collection.

        :return: DataSourceRelationEntity class
        :rtype: type[DataSourceRelationEntity]
        """
        return DataSourceRelationEntity

    def serialize(self, data: dict[str, Any]) -> bytes:
        """serialize entity data to JSON bytes for L2 cache.

        :param data: entity field data
        :ptype data: dict[str, Any]
        :return: JSON-encoded bytes
        :rtype: bytes
        """
        return serialize_to_json(data)

    def deserialize(self, data: bytes) -> dict[str, Any]:
        """deserialize JSON bytes from L2 cache to entity data.

        :param data: JSON-encoded bytes
        :ptype data: bytes
        :return: entity field data dictionary
        :rtype: dict[str, Any]
        """
        return deserialize_from_json(data, _RELATION_FIELD_TYPES)

    async def fetch_from_postgres(self, entity_id: Any) -> dict[str, Any] | None:
        """fetch data source relation record from L3 by primary key.

        :param entity_id: data source relation UUID
        :ptype entity_id: Any
        :return: relation data dictionary or None if not found
        :rtype: dict[str, Any] | None
        """
        if self.l3_pool is None:
            return None
        row = await self.l3_pool.fetchrow(
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

    async def save_to_postgres(
        self,
        data: dict[str, Any],
        original_timestamp: datetime | None = None,
        *,
        conn: Any = None,
    ) -> int:
        """upsert data source relation record to L3 with optimistic concurrency.

        :param data: entity field data to persist
        :ptype data: dict[str, Any]
        :param original_timestamp: original date_updated for concurrency check
        :ptype original_timestamp: datetime | None
        :return: number of rows affected
        :rtype: int
        """
        if self.l3_pool is None:
            return 0

        result = await self.l3_pool.execute(
            """
            INSERT INTO datasource_relations (
                id, name, description, datasource_ids, join_paths,
                aggregation_notes, caveats, date_created, date_updated
            ) VALUES (
                -- $4 / $5 are bound NATIVELY (python lists via encode_jsonb); the
                -- jsonb codec applies the single json.dumps (collections-task-04).
                $1, $2, $3, $4, $5, $6, $7, $8, $9
            )
            ON CONFLICT (id) DO UPDATE SET
                name = EXCLUDED.name,
                description = EXCLUDED.description,
                datasource_ids = EXCLUDED.datasource_ids,
                join_paths = EXCLUDED.join_paths,
                aggregation_notes = EXCLUDED.aggregation_notes,
                caveats = EXCLUDED.caveats,
                date_updated = EXCLUDED.date_updated
            """,
            data.get("id"),
            data.get("name"),
            data.get("description"),
            encode_jsonb(data.get("datasource_ids")),
            encode_jsonb(data.get("join_paths")),
            data.get("aggregation_notes"),
            data.get("caveats"),
            data.get("date_created"),
            data.get("date_updated"),
        )
        return int(result.split()[-1])

    async def delete_from_postgres(self, entity_id: Any) -> None:
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


class TableTemplateCollection(BaseCollection[TableTemplateEntity]):
    """three-tier collection for table-template entities.

    provides CRUD with L1 -> L2 -> L3 caching for the customer-scoped
    template definition rows. composite PK ``(customer_id, id)`` is
    enforced via the framework's normalize_pk lookup, so callers
    address rows by the full tuple. natural-key conflict resolution
    on ``(customer_id, name)`` keeps slug collisions inside a
    customer's namespace.

    templates are hard-deleted; the FK from
    ``datasource_tables.template_id`` is ``ON DELETE SET NULL`` so
    deleting a template never destroys instance metadata, and the FK
    from ``table_template_columns.template_id`` is ``ON DELETE
    CASCADE`` so the per-template column list goes with it.
    """

    @property
    def table_name(self) -> str:
        """return database table name.

        :return: table name string
        :rtype: str
        """
        return "table_templates"

    @property
    def entity_class(self) -> type[TableTemplateEntity]:
        """return entity class for this collection.

        :return: TableTemplateEntity class
        :rtype: type[TableTemplateEntity]
        """
        return TableTemplateEntity

    def serialize(self, data: dict[str, Any]) -> bytes:
        """serialize entity data to JSON bytes for L2 cache.

        :param data: entity field data
        :ptype data: dict[str, Any]
        :return: JSON-encoded bytes
        :rtype: bytes
        """
        return serialize_to_json(data)

    def deserialize(self, data: bytes) -> dict[str, Any]:
        """deserialize JSON bytes from L2 cache to entity data.

        :param data: JSON-encoded bytes
        :ptype data: bytes
        :return: entity field data dictionary
        :rtype: dict[str, Any]
        """
        return deserialize_from_json(data, _TEMPLATE_FIELD_TYPES)

    async def fetch_from_postgres(self, entity_id: Any) -> dict[str, Any] | None:
        """fetch table-template row from L3 by composite primary key.

        :param entity_id: ``(customer_id, id)`` tuple
        :ptype entity_id: Any
        :return: template data dictionary or None if not found
        :rtype: dict[str, Any] | None
        """
        if self.l3_pool is None:
            return None
        customer_id, template_id = entity_id
        row = await self.l3_pool.fetchrow(
            "SELECT * FROM table_templates WHERE customer_id = $1 AND id = $2",
            customer_id,
            template_id,
        )
        result: dict[str, Any] | None = None
        if row is not None:
            result = dict(row)
        return result

    async def save_to_postgres(
        self,
        data: dict[str, Any],
        original_timestamp: datetime | None = None,
        *,
        conn: Any = None,
    ) -> int:
        """upsert table-template row to L3.

        natural-key conflict on ``(customer_id, name)`` is enforced by
        the unique index added in v006; the upsert routes on the
        primary-key conflict so a re-save with the same id keeps the
        row's identity stable.

        :param data: entity field data to persist
        :ptype data: dict[str, Any]
        :param original_timestamp: original date_updated for
            optimistic concurrency check (unused today; pattern
            mirror of DataSourceTableCollection)
        :ptype original_timestamp: datetime | None
        :return: number of rows affected
        :rtype: int
        """
        if self.l3_pool is None:
            return 0

        result = await self.l3_pool.execute(
            """
            INSERT INTO table_templates (
                id, customer_id, name, description, caveats,
                date_created, date_updated
            ) VALUES (
                $1, $2, $3, $4, $5, $6, $7
            )
            ON CONFLICT (customer_id, id) DO UPDATE SET
                name = EXCLUDED.name,
                description = EXCLUDED.description,
                caveats = EXCLUDED.caveats,
                date_updated = EXCLUDED.date_updated
            """,
            data.get("id"),
            data.get("customer_id"),
            data.get("name"),
            data.get("description"),
            data.get("caveats"),
            data.get("date_created"),
            data.get("date_updated"),
        )
        return int(result.split()[-1])

    async def delete_from_postgres(self, entity_id: Any) -> None:
        """hard-delete template row from L3.

        :param entity_id: ``(customer_id, id)`` tuple
        :ptype entity_id: Any
        :return: nothing
        :rtype: None
        """
        if self.l3_pool is None:
            return
        customer_id, template_id = entity_id
        await self.l3_pool.execute(
            "DELETE FROM table_templates WHERE customer_id = $1 AND id = $2",
            customer_id,
            template_id,
        )
