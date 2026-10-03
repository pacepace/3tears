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
- :class:`DataSourceTableCollection` -- ``BaseCollection`` for the
  ``datasource_tables`` row set.
- :class:`DataSourceColumnCollection` -- ``BaseCollection`` for
  ``datasource_columns``. natural-key upsert on
  ``(datasource_id, schema_name, table_name, column_name)``.
- :class:`DataSourceRelationCollection` -- ``BaseCollection`` for
  ``datasource_relations``.
- :class:`TableTemplateCollection` -- ``BaseCollection`` for
  ``table_templates``. PK ``id`` (hub v007).

the hand-written upserts (tables, relations, templates) are built from one
:class:`_UpsertShape` each, and write only the columns a row carries: a column
the row does not name is never reset to NULL or a default.

per-table column variants (``TableTemplateColumnCollection``) stay in
Hub for now because they have no cross-consumer demand yet; lift later
if a second 3tears consumer needs them.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import UUID

from threetears.core.backends import parse_rowcount
from threetears.core.collections.base import BaseCollection
from threetears.core.collections.schema_backed import (
    BOOL_TYPE,
    DATETIMETZ_TYPE,
    JSONB_TYPE,
    STRING_TYPE,
    UUID_TYPE,
    Column,
    SchemaBackedCollection,
    TableSchema,
    encode_jsonb,
)
from threetears.core.security.secret_refs import validate_ref
from threetears.core.serialization import deserialize_from_json, serialize_to_json
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


_TABLE_FIELD_TYPES: dict[str, Any] = {
    "id": UUID,
    "datasource_id": UUID,
    "schema_name": str,
    "table_name": str,
    "description": str,
    "row_count_approx": int,
    "caveats": str,
    # template-task-01: nullable FK into table_templates;
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
    # hub v033: the column per-region value coverage is broken down by;
    # NULL = whole-table coverage only.
    "coverage_dimension": str,
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
    # hub v056: NULL = platform-shared, set = owned by that customer.
    "customer_id": UUID,
    "name": str,
    "description": str,
    "datasource_ids": list,
    "join_paths": list,
    # hub v056: named traversal graphs for a non-chain relation.
    "edges": list,
    "aggregation_notes": str,
    "caveats": str,
    "date_created": datetime,
    "date_updated": datetime,
}

_TEMPLATE_FIELD_TYPES: dict[str, Any] = {
    "id": UUID,
    # hub v007: NULL on a platform-owned (public / restricted) template.
    "customer_id": UUID,
    "name": str,
    "description": str,
    "caveats": str,
    # hub v007: 'private' | 'public' | 'restricted'.
    "visibility": str,
    # hub v007: the customer template a platform copy was promoted from.
    "origin_template_id": UUID,
    "date_created": datetime,
    "date_updated": datetime,
}


@dataclass(frozen=True)
class _UpsertShape:
    """the columns a hand-written upsert may name, and how it treats each.

    one rule governs every statement built from a shape: **a column the row does
    not carry is never written.** an INSERT leaves it to the column's default
    (``NULL``, or the table's server default); an update of an existing row keeps
    the stored value. a row carries a column when it names it, except that
    ``None`` for a ``NOT NULL`` column with a server default is not a value the
    column can hold, so it counts as not carried too.

    :cvar table: the table written
    :cvar columns: every column of the table this collection owns, in the
        order a statement names them
    :cvar conflict: the upsert's conflict target
    a row carrying every ``NOT NULL`` column without a default is written as
    an upsert. a row that does not is only an update of an existing row --
    Postgres checks ``NOT NULL`` on the row an INSERT proposes before any
    conflict is resolved, so it could not be inserted -- and is written as a
    plain ``UPDATE``; on a key no row holds it affects 0 rows, which
    :meth:`BaseCollection.save_entity` reports.

    :cvar table: the table written
    :cvar columns: every column of the table this collection owns, in the
        order a statement names them
    :cvar conflict: the upsert's conflict target
    :cvar required: the ``NOT NULL`` columns with no default, which an
        insert must carry
    :cvar jsonb: columns bound through :func:`encode_jsonb`
    :cvar defaulted: ``NOT NULL`` columns with a server default
    :cvar insert_only: columns an INSERT writes and an update never does,
        besides the conflict target and ``date_created``
    """

    table: str
    columns: tuple[str, ...]
    conflict: tuple[str, ...]
    required: frozenset[str]
    jsonb: frozenset[str] = frozenset()
    defaulted: frozenset[str] = frozenset()
    insert_only: frozenset[str] = frozenset()

    def carried(self, data: dict[str, Any]) -> tuple[str, ...]:
        """the columns ``data`` carries, in declared order.

        :param data: the row as the write sends it
        :ptype data: dict[str, Any]
        :return: the carried column names
        :rtype: tuple[str, ...]
        """
        return tuple(
            column
            for column in self.columns
            if column in data and not (column in self.defaulted and data[column] is None)
        )

    def decided_by_store(self, data: dict[str, Any]) -> tuple[str, ...]:
        """the columns whose stored value a write of ``data`` does not determine.

        a column ``data`` does not carry is filled by the table's default on an
        insert and kept on an update; an insert-only column ``data`` does carry
        is kept on an update. either way the row sent is not known to be the row
        stored, and :meth:`BaseCollection.save_entity` reads it back.

        :param data: the row as the write sends it
        :ptype data: dict[str, Any]
        :return: the column names, in declared order
        :rtype: tuple[str, ...]
        """
        carried = set(self.carried(data))
        return tuple(column for column in self.columns if column not in carried or column in self.insert_only)

    def statement(self, data: dict[str, Any]) -> tuple[str, list[Any]]:
        """the statement writing exactly the columns ``data`` carries, with its parameters.

        an upsert when ``data`` carries every :attr:`required` column, a plain
        ``UPDATE`` of the existing row otherwise.

        :param data: the row as the write sends it
        :ptype data: dict[str, Any]
        :return: the SQL and its positional parameters
        :rtype: tuple[str, list[Any]]
        :raises KeyError: when ``data`` does not carry every conflict column,
            or is an update that carries no column to update
        """
        carried = self.carried(data)
        missing = [column for column in self.conflict if column not in carried]
        if missing:
            raise KeyError(
                f"{self.table}: a write must carry its conflict key {list(self.conflict)!r}, missing {missing!r}"
            )
        kept = {*self.conflict, *self.insert_only, "date_created"}
        updated = [column for column in carried if column not in kept]
        if self.required <= set(carried):
            placeholders = ", ".join(f"${index}" for index in range(1, len(carried) + 1))
            on_conflict = (
                "DO UPDATE SET " + ", ".join(f"{column} = EXCLUDED.{column}" for column in updated)
                if updated
                else "DO NOTHING"
            )
            sql = (
                f"INSERT INTO {self.table} ({', '.join(carried)}) VALUES ({placeholders}) "  # noqa: S608
                f"ON CONFLICT ({', '.join(self.conflict)}) {on_conflict}"
            )
            bound = list(carried)
        else:
            if not updated:
                raise KeyError(f"{self.table}: a write of an existing row must carry a column to update")
            assignments = ", ".join(f"{column} = ${index}" for index, column in enumerate(updated, start=1))
            keys = " AND ".join(
                f"{column} = ${index}" for index, column in enumerate(self.conflict, start=len(updated) + 1)
            )
            sql = f"UPDATE {self.table} SET {assignments} WHERE {keys}"  # noqa: S608
            bound = [*updated, *self.conflict]
        params = [encode_jsonb(data[column]) if column in self.jsonb else data[column] for column in bound]
        return sql, params


#: ``datasource_tables`` as hub v001, v006, v010 and v033 leave it.
#:
#: ``coverage_dimension`` (v033) is insert-only. the per-region coverage
#: designation of an existing table is set by the hub's admin table PATCH and
#: moved by the data-upgrade rename carry, both direct writes. the one writer
#: that saves an existing row through this collection is the schema
#: introspector, which saves back the row it prefetched at the start of a pass;
#: writing its copy of the designation would undo a PATCH that landed during the
#: pass.
_TABLE_SHAPE = _UpsertShape(
    table="datasource_tables",
    columns=(
        "id",
        "datasource_id",
        "schema_name",
        "table_name",
        "description",
        "row_count_approx",
        "caveats",
        "template_id",
        "caveats_replaces_definition",
        "column_hash",
        "coverage_dimension",
        "date_introspected",
        "date_described",
        "date_created",
        "date_updated",
    ),
    conflict=("id",),
    required=frozenset({"id", "datasource_id", "schema_name", "table_name", "date_created", "date_updated"}),
    defaulted=frozenset({"caveats_replaces_definition"}),
    insert_only=frozenset({"coverage_dimension"}),
)

#: ``datasource_relations`` as hub v001 and v056 leave it.
_RELATION_SHAPE = _UpsertShape(
    table="datasource_relations",
    columns=(
        "id",
        "customer_id",
        "name",
        "description",
        "datasource_ids",
        "join_paths",
        "edges",
        "aggregation_notes",
        "caveats",
        "date_created",
        "date_updated",
    ),
    conflict=("id",),
    required=frozenset({"id", "name", "date_created", "date_updated"}),
    jsonb=frozenset({"datasource_ids", "join_paths", "edges"}),
    defaulted=frozenset({"datasource_ids", "join_paths", "edges"}),
)

#: ``table_templates`` as hub v001, v006 and v007 leave it: the primary key is
#: ``id`` alone (v007 dropped the ``(customer_id, id)`` composite key so a
#: platform-owned template can carry ``customer_id`` NULL).
_TEMPLATE_SHAPE = _UpsertShape(
    table="table_templates",
    columns=(
        "id",
        "customer_id",
        "name",
        "description",
        "caveats",
        "visibility",
        "origin_template_id",
        "date_created",
        "date_updated",
    ),
    conflict=("id",),
    required=frozenset({"id", "name", "visibility", "date_created", "date_updated"}),
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

    primary_key_column: str = "id"
    schema = TableSchema(
        name="datasources",
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
            Column("kind", STRING_TYPE),
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
            Column("allowed_schemas", JSONB_TYPE, nullable=True),
            Column("access_mode", STRING_TYPE),
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
            Column("visibility", STRING_TYPE),
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
        result: CapabilitySourceEntity | None = None
        if self.l3_pool is not None:
            row = await self.l3_pool.fetchrow(
                "SELECT * FROM datasources WHERE id = $1",
                datasource_id,
            )
            if row is not None:
                data = self._coerce_row(dict(row))
                self.write_to_cache_sync(data, from_lower_tier=True)
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

    async def fetch_from_store(self, entity_id: Any) -> dict[str, Any] | None:
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
        :data:`_TABLE_SHAPE`).

        :param data: entity field data to persist
        :ptype data: dict[str, Any]
        :param original_timestamp: unused; this table has no CAS fence
        :ptype original_timestamp: datetime | None
        :param conn: unused; accepted for the extension-point signature
        :ptype conn: Any
        :return: number of rows affected
        :rtype: int
        """
        if self.l3_pool is None:
            return 0
        sql, params = _TABLE_SHAPE.statement(data)
        result = await self.l3_pool.execute(sql, *params)
        return parse_rowcount(result)

    def columns_decided_by_store(self, data: dict[str, Any]) -> tuple[str, ...]:
        """the columns whose stored value writing ``data`` leaves to the database.

        see :meth:`BaseCollection.columns_decided_by_store`. every column
        ``data`` does not carry, and ``coverage_dimension`` when it does, since
        an update keeps the stored designation.

        :param data: the row as the write sends it, stamped
        :ptype data: dict[str, Any]
        :return: the column names, in declared order
        :rtype: tuple[str, ...]
        """
        return _TABLE_SHAPE.decided_by_store(data)

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

    async def fetch_from_store(self, entity_id: Any) -> dict[str, Any] | None:
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

    async def save_to_store(
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
        return parse_rowcount(result)

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

    async def fetch_from_store(self, entity_id: Any) -> dict[str, Any] | None:
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

    async def save_to_store(
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
        return parse_rowcount(result)

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

    async def fetch_from_store(self, entity_id: Any) -> dict[str, Any] | None:
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
        :param conn: unused; accepted for the extension-point signature
        :ptype conn: Any
        :return: number of rows affected
        :rtype: int
        """
        if self.l3_pool is None:
            return 0
        sql, params = _RELATION_SHAPE.statement(data)
        result = await self.l3_pool.execute(sql, *params)
        return parse_rowcount(result)

    def columns_decided_by_store(self, data: dict[str, Any]) -> tuple[str, ...]:
        """the columns whose stored value writing ``data`` leaves to the database.

        see :meth:`BaseCollection.columns_decided_by_store`: every column
        ``data`` does not carry.

        :param data: the row as the write sends it, stamped
        :ptype data: dict[str, Any]
        :return: the column names, in declared order
        :rtype: tuple[str, ...]
        """
        return _RELATION_SHAPE.decided_by_store(data)

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


class TableTemplateCollection(BaseCollection[TableTemplateEntity]):
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

    async def fetch_from_store(self, entity_id: Any) -> dict[str, Any] | None:
        """fetch table-template row from L3 by primary key.

        :param entity_id: template UUID
        :ptype entity_id: Any
        :return: template data dictionary or None if not found
        :rtype: dict[str, Any] | None
        """
        if self.l3_pool is None:
            return None
        row = await self.l3_pool.fetchrow(
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
        :param conn: unused; accepted for the extension-point signature
        :ptype conn: Any
        :return: number of rows affected
        :rtype: int
        :raises ValueError: when ``data`` carries no ``visibility``
        """
        if data.get("visibility") is None:
            raise ValueError(
                f"table_templates: template {data.get('id')} carries no visibility. every write names it "
                f"('private' with a customer_id, 'public' or 'restricted' without one); the column default "
                f"is never assumed, since writing 'private' over a platform template hides it"
            )
        if self.l3_pool is None:
            return 0
        sql, params = _TEMPLATE_SHAPE.statement(data)
        result = await self.l3_pool.execute(sql, *params)
        return parse_rowcount(result)

    def columns_decided_by_store(self, data: dict[str, Any]) -> tuple[str, ...]:
        """the columns whose stored value writing ``data`` leaves to the database.

        see :meth:`BaseCollection.columns_decided_by_store`: every column
        ``data`` does not carry.

        :param data: the row as the write sends it, stamped
        :ptype data: dict[str, Any]
        :return: the column names, in declared order
        :rtype: tuple[str, ...]
        """
        return _TEMPLATE_SHAPE.decided_by_store(data)

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
