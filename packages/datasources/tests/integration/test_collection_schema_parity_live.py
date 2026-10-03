"""live parity between every datasource Collection and the hub table it serves.

The drift this pins was found 2026-10-03. Three Collections in this package wrote
SQL naming fewer columns than the hub's tables carry:

- ``DataSourceTableCollection`` never named ``coverage_dimension`` (hub v033);
- ``DataSourceRelationCollection`` never named ``customer_id`` or ``edges`` (hub v056);
- ``TableTemplateCollection`` never named ``visibility`` or ``origin_template_id``,
  and kept upserting ``ON CONFLICT (customer_id, id)`` after hub v007 rebuilt the
  primary key on ``id`` alone. Postgres refuses that conflict target outright.

Nothing failed, because nothing compared a Collection with the table it writes.
This module does that, for every Collection in the package. Each table below is
built the way the hub's migrations leave it. ``information_schema`` names its
columns, and each case must account for every one of them: a column the
Collection owns is written with a known value and must read back as that value.
A column the hub owns and writes directly must come through a Collection save
untouched. When the hub adds a column, add it to the DDL here. The case's
column-accounting test then fails until the Collection owns the column or the
column is listed as hub-owned, with a reason.

The DDL carries the hub's primary keys, unique indexes, defaults and CHECK
constraints. Foreign keys to tables outside this module (``customers``) are left
out; the self-reference on ``table_templates.origin_template_id`` is kept.

Uses the session-scoped ``db_container`` fixture; a checkout without docker skips cleanly.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import asyncpg
import pytest
from threetears.core.collections import init_connection
from threetears.core.collections.base import BaseCollection
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig

from threetears.datasources.collections import (
    CapabilitySourceCollection,
    DataSourceColumnCollection,
    DataSourceRelationCollection,
    DataSourceSchemaDigestCollection,
    DataSourceTableCollection,
    TableTemplateCollection,
)

pytestmark = pytest.mark.integration

# ``platform.datasources`` after hub v001, v016, v037, v042, v046, v053, v076.
_DATASOURCES_DDL = """
CREATE TABLE datasources (
    id uuid NOT NULL PRIMARY KEY,
    name character varying(255) NOT NULL,
    datasource_type character varying(50),
    connection_config text,
    allowed_schemas jsonb DEFAULT '[]'::jsonb NOT NULL,
    status character varying(20) NOT NULL,
    date_created timestamp with time zone NOT NULL,
    date_updated timestamp with time zone NOT NULL,
    access_mode character varying(20) DEFAULT 'read'::character varying NOT NULL,
    customer_id uuid,
    owner_agent_id uuid,
    schema_name character varying(255),
    visibility text DEFAULT 'private'::text NOT NULL,
    origin_datasource_id uuid,
    kind character varying(50) NOT NULL DEFAULT 'datasource',
    ingress_agent_id uuid,
    spec text,
    face_api boolean NOT NULL DEFAULT FALSE,
    face_mcp boolean NOT NULL DEFAULT FALSE,
    face_platform_tool boolean NOT NULL DEFAULT TRUE,
    knowledge_required boolean NOT NULL DEFAULT FALSE,
    geo jsonb,
    require_documented_tables boolean DEFAULT FALSE NOT NULL
)
"""

# ``platform.datasource_tables`` after hub v001, v006, v010, v033.
_DATASOURCE_TABLES_DDL = """
CREATE TABLE datasource_tables (
    id uuid NOT NULL PRIMARY KEY,
    datasource_id uuid NOT NULL,
    schema_name character varying(255) NOT NULL,
    table_name character varying(255) NOT NULL,
    description text,
    row_count_approx bigint,
    caveats text,
    template_id uuid,
    caveats_replaces_definition boolean DEFAULT FALSE NOT NULL,
    date_introspected timestamp with time zone,
    date_described timestamp with time zone,
    date_created timestamp with time zone NOT NULL,
    date_updated timestamp with time zone NOT NULL,
    column_hash text,
    coverage_dimension text,
    CONSTRAINT datasource_tables_datasource_id_schema_name_table_name_key
        UNIQUE (datasource_id, schema_name, table_name)
)
"""

# ``platform.datasource_columns`` after hub v001, v006.
_DATASOURCE_COLUMNS_DDL = """
CREATE TABLE datasource_columns (
    id uuid NOT NULL PRIMARY KEY,
    datasource_id uuid NOT NULL,
    schema_name character varying(255) NOT NULL,
    table_name character varying(255) NOT NULL,
    column_name character varying(255) NOT NULL,
    data_type character varying(255),
    is_nullable boolean,
    ordinal_position integer,
    description text,
    valid_range text,
    caveats text,
    tags jsonb DEFAULT '[]'::jsonb,
    caveats_replaces_definition boolean DEFAULT FALSE NOT NULL,
    date_introspected timestamp with time zone,
    date_described timestamp with time zone,
    date_created timestamp with time zone NOT NULL,
    date_updated timestamp with time zone NOT NULL,
    CONSTRAINT datasource_columns_natural_key
        UNIQUE (datasource_id, schema_name, table_name, column_name)
)
"""

# ``platform.datasource_relations`` after hub v001, v056.
_DATASOURCE_RELATIONS_DDL = """
CREATE TABLE datasource_relations (
    id uuid NOT NULL PRIMARY KEY,
    customer_id uuid,
    name character varying(255) NOT NULL,
    description text,
    datasource_ids jsonb DEFAULT '[]'::jsonb NOT NULL,
    join_paths jsonb DEFAULT '[]'::jsonb NOT NULL,
    edges jsonb DEFAULT '[]'::jsonb NOT NULL,
    aggregation_notes text,
    caveats text,
    date_created timestamp with time zone NOT NULL,
    date_updated timestamp with time zone NOT NULL
)
"""

_DATASOURCE_RELATIONS_INDEXES = (
    "CREATE UNIQUE INDEX datasource_relations_customer_id_name_key ON datasource_relations (customer_id, name)",
    "CREATE UNIQUE INDEX datasource_relations_platform_name_key ON datasource_relations (name) "
    "WHERE customer_id IS NULL",
)

# ``platform.table_templates`` after hub v001, v006, v007: the primary key is ``id`` alone.
_TABLE_TEMPLATES_DDL = """
CREATE TABLE table_templates (
    id uuid NOT NULL PRIMARY KEY,
    customer_id uuid,
    name character varying(255) NOT NULL,
    description text,
    caveats text,
    visibility text DEFAULT 'private' NOT NULL,
    origin_template_id uuid REFERENCES table_templates(id) ON DELETE SET NULL,
    date_created timestamp with time zone NOT NULL,
    date_updated timestamp with time zone NOT NULL,
    CONSTRAINT table_templates_visibility_check
        CHECK (visibility IN ('private', 'public', 'restricted')),
    CONSTRAINT table_templates_visibility_customer_check
        CHECK ((visibility = 'private' AND customer_id IS NOT NULL)
            OR (visibility IN ('public', 'restricted') AND customer_id IS NULL))
)
"""

_TABLE_TEMPLATES_INDEXES = (
    "CREATE UNIQUE INDEX table_templates_customer_id_name_key ON table_templates (customer_id, name)",
)

# ``platform.datasource_schema_digests`` after hub v029.
_DATASOURCE_SCHEMA_DIGESTS_DDL = """
CREATE TABLE datasource_schema_digests (
    datasource_id uuid NOT NULL PRIMARY KEY,
    customer_id uuid,
    tables jsonb DEFAULT '[]'::jsonb NOT NULL,
    source_fingerprint text,
    date_created timestamp with time zone NOT NULL,
    date_updated timestamp with time zone NOT NULL
)
"""

_ALL_DDL: tuple[str, ...] = (
    _DATASOURCES_DDL,
    _DATASOURCE_TABLES_DDL,
    _DATASOURCE_COLUMNS_DDL,
    _DATASOURCE_RELATIONS_DDL,
    *_DATASOURCE_RELATIONS_INDEXES,
    _TABLE_TEMPLATES_DDL,
    *_TABLE_TEMPLATES_INDEXES,
    _DATASOURCE_SCHEMA_DIGESTS_DDL,
)

#: columns a save stamps itself, so a case cannot choose their value
_STAMPED: frozenset[str] = frozenset({"date_created", "date_updated"})


def _when(days: int) -> datetime:
    """a fixed, distinct, aware instant.

    :param days: offset from a fixed base, in days
    :ptype days: int
    :return: the instant
    :rtype: datetime
    """
    return datetime(2026, 1, 1, tzinfo=UTC) + timedelta(days=days)


def _capability_source_row() -> dict[str, Any]:
    """every column :class:`CapabilitySourceCollection` owns, platform-shared shape.

    :return: the row
    :rtype: dict[str, Any]
    """
    return {
        "id": uuid.uuid4(),
        "name": f"ds-{uuid.uuid4().hex[:6]}",
        "customer_id": None,
        "kind": "datasource",
        "datasource_type": "postgres",
        "connection_config": "encrypted-blob",
        "allowed_schemas": ["public", "geo"],
        "access_mode": "read_write",
        "status": "active",
        "ingress_agent_id": uuid.uuid4(),
        "owner_agent_id": uuid.uuid4(),
        "schema_name": "agent_0123",
        "visibility": "restricted",
        "origin_datasource_id": uuid.uuid4(),
        "knowledge_required": True,
        "require_documented_tables": True,
    }


def _capability_source_change(row: dict[str, Any]) -> dict[str, Any]:
    """a different value for every mutable column of a capability-source row.

    :param row: the stored row
    :ptype row: dict[str, Any]
    :return: the changed fields
    :rtype: dict[str, Any]
    """
    return {
        "name": f"{row['name']}-renamed",
        "customer_id": uuid.uuid4(),
        "kind": "datasource",
        "connection_config": "other-blob",
        "allowed_schemas": ["public"],
        "access_mode": "read",
        "status": "disabled",
        "ingress_agent_id": uuid.uuid4(),
        "visibility": "private",
        "origin_datasource_id": uuid.uuid4(),
        "knowledge_required": False,
        "require_documented_tables": False,
    }


def _table_row() -> dict[str, Any]:
    """every column :class:`DataSourceTableCollection` owns.

    :return: the row
    :rtype: dict[str, Any]
    """
    return {
        "id": uuid.uuid4(),
        "datasource_id": uuid.uuid4(),
        "schema_name": "public",
        "table_name": f"t_{uuid.uuid4().hex[:6]}",
        "description": "one row per county",
        "row_count_approx": 3_143,
        "caveats": "2024 counties only",
        "template_id": uuid.uuid4(),
        "caveats_replaces_definition": True,
        "column_hash": "0123456789abcdef0123456789abcdef",
        "coverage_dimension": "state_code",
        "date_introspected": _when(1),
        "date_described": _when(2),
    }


def _table_change(row: dict[str, Any]) -> dict[str, Any]:
    """a different value for every column an update of a table row writes.

    ``coverage_dimension`` is absent on purpose: it is written on insert only
    (see :class:`DataSourceTableCollection`), and its own test pins that.

    :param row: the stored row
    :ptype row: dict[str, Any]
    :return: the changed fields
    :rtype: dict[str, Any]
    """
    return {
        "datasource_id": uuid.uuid4(),
        "schema_name": "reporting",
        "table_name": f"{row['table_name']}_v2",
        "description": "one row per county, 2028",
        "row_count_approx": 3_144,
        "caveats": "2028 counties",
        "template_id": uuid.uuid4(),
        "caveats_replaces_definition": False,
        "column_hash": "fedcba9876543210fedcba9876543210",
        "date_introspected": _when(3),
        "date_described": _when(4),
    }


def _column_row() -> dict[str, Any]:
    """every column :class:`DataSourceColumnCollection` owns.

    :return: the row
    :rtype: dict[str, Any]
    """
    return {
        "id": uuid.uuid4(),
        "datasource_id": uuid.uuid4(),
        "schema_name": "public",
        "table_name": "counties",
        "column_name": f"c_{uuid.uuid4().hex[:6]}",
        "data_type": "integer",
        "is_nullable": True,
        "ordinal_position": 7,
        "description": "total votes",
        "valid_range": ">= 0",
        "caveats": "zero means unloaded in NY",
        "tags": ["votes", "pres"],
        "caveats_replaces_definition": True,
        "date_introspected": _when(1),
        "date_described": _when(2),
    }


def _column_change(row: dict[str, Any]) -> dict[str, Any]:
    """a different value for every column an update of a column row writes.

    the natural key and ``id`` are what the upsert conflicts on, so they stay.

    :param row: the stored row
    :ptype row: dict[str, Any]
    :return: the changed fields
    :rtype: dict[str, Any]
    """
    return {
        "data_type": "bigint",
        "is_nullable": False,
        "ordinal_position": 8,
        "description": "total votes, all parties",
        "valid_range": ">= 1",
        "caveats": "none",
        "tags": ["votes"],
        "caveats_replaces_definition": False,
        "date_introspected": _when(3),
        "date_described": _when(4),
    }


def _relation_row() -> dict[str, Any]:
    """every column :class:`DataSourceRelationCollection` owns.

    :return: the row
    :rtype: dict[str, Any]
    """
    return {
        "id": uuid.uuid4(),
        "customer_id": uuid.uuid4(),
        "name": f"household_{uuid.uuid4().hex[:6]}",
        "description": "people to households",
        "datasource_ids": [str(uuid.uuid4()), str(uuid.uuid4())],
        "join_paths": [{"left": "people.hh_id", "right": "households.id", "type": "inner"}],
        "edges": [{"name": "to_household", "steps": [{"from": "people", "to": "households"}]}],
        "aggregation_notes": "count distinct households",
        "caveats": "one household per address",
    }


def _relation_change(row: dict[str, Any]) -> dict[str, Any]:
    """a different value for every mutable column of a relation row.

    :param row: the stored row
    :ptype row: dict[str, Any]
    :return: the changed fields
    :rtype: dict[str, Any]
    """
    return {
        "customer_id": uuid.uuid4(),
        "name": f"{row['name']}_v2",
        "description": "people to households, deduplicated",
        "datasource_ids": [str(uuid.uuid4())],
        "join_paths": [],
        "edges": [{"name": "to_unit", "steps": [{"from": "people", "to": "units"}]}],
        "aggregation_notes": "count distinct units",
        "caveats": "units, not addresses",
    }


def _template_row() -> dict[str, Any]:
    """every column :class:`TableTemplateCollection` owns, platform-owned shape.

    :return: the row
    :rtype: dict[str, Any]
    """
    return {
        "id": uuid.uuid4(),
        "customer_id": None,
        "name": f"geofacts_{uuid.uuid4().hex[:6]}",
        "description": "county geofacts",
        "caveats": "2024 boundaries",
        "visibility": "public",
        "origin_template_id": None,
    }


def _template_change(row: dict[str, Any]) -> dict[str, Any]:
    """a different value for every mutable column of a template row.

    :param row: the stored row
    :ptype row: dict[str, Any]
    :return: the changed fields
    :rtype: dict[str, Any]
    """
    return {
        "customer_id": uuid.uuid4(),
        "name": f"{row['name']}_v2",
        "description": "county geofacts, 2028",
        "caveats": "2028 boundaries",
        "visibility": "private",
    }


def _digest_row() -> dict[str, Any]:
    """every column :class:`DataSourceSchemaDigestCollection` owns.

    :return: the row
    :rtype: dict[str, Any]
    """
    return {
        "datasource_id": uuid.uuid4(),
        "customer_id": uuid.uuid4(),
        "tables": [{"schema": "public", "table": "counties", "columns": []}],
        "source_fingerprint": "abc123",
    }


def _digest_change(row: dict[str, Any]) -> dict[str, Any]:
    """a different value for every mutable column of a digest row.

    :param row: the stored row
    :ptype row: dict[str, Any]
    :return: the changed fields
    :rtype: dict[str, Any]
    """
    return {
        "customer_id": uuid.uuid4(),
        "tables": [],
        "source_fingerprint": "def456",
    }


@dataclass(frozen=True)
class _Case:
    """one Collection, the table it serves, and how every column of that table is accounted for.

    :cvar collection: the Collection class
    :cvar table: the table it serves
    :cvar pk: the primary-key column
    :cvar row: every column the Collection owns, each with a value it must write
    :cvar change: a different value for every column an update of an existing row writes
    :cvar hub_owned: columns the hub writes directly, each with the value a test sets
        before a Collection save, which the save must leave as it found it
    """

    collection: type[BaseCollection[Any]]
    table: str
    pk: str
    row: Callable[[], dict[str, Any]]
    change: Callable[[dict[str, Any]], dict[str, Any]]
    hub_owned: dict[str, Any] = field(default_factory=dict)


# The hub writes these ``datasources`` columns itself and the Collection must
# never touch them: ``spec`` (v042) is the imported OpenAPI / MCP document the
# capability-source import routes store; ``face_api`` / ``face_mcp`` /
# ``face_platform_tool`` (v042) are the faces the hub's face endpoints toggle;
# ``geo`` (v053) is the geo-layer map the hub's geo enrichment writes. Each is
# written by raw SQL at the one hub site that owns it, and a Collection save of
# the same row (a status change, a knowledge stamp) must not reset them.
_CAPABILITY_SOURCE_HUB_OWNED: dict[str, Any] = {
    "spec": "openapi: 3.1.0",
    "face_api": True,
    "face_mcp": True,
    "face_platform_tool": False,
    "geo": {"layers": [{"name": "county"}]},
}

_CASES: tuple[_Case, ...] = (
    _Case(
        CapabilitySourceCollection,
        "datasources",
        "id",
        _capability_source_row,
        _capability_source_change,
        _CAPABILITY_SOURCE_HUB_OWNED,
    ),
    _Case(DataSourceTableCollection, "datasource_tables", "id", _table_row, _table_change),
    _Case(DataSourceColumnCollection, "datasource_columns", "id", _column_row, _column_change),
    _Case(DataSourceRelationCollection, "datasource_relations", "id", _relation_row, _relation_change),
    _Case(TableTemplateCollection, "table_templates", "id", _template_row, _template_change),
    _Case(
        DataSourceSchemaDigestCollection,
        "datasource_schema_digests",
        "datasource_id",
        _digest_row,
        _digest_change,
    ),
)


def test_every_collection_in_the_package_has_a_case() -> None:
    """a Collection added to the package without a parity case fails here."""
    import threetears.datasources.collections as module

    exported = {
        getattr(module, name)
        for name in module.__all__
        if isinstance(getattr(module, name), type) and issubclass(getattr(module, name), BaseCollection)
    }
    assert exported, "the module exports no Collection, so this check would pass vacuously"
    assert {case.collection for case in _CASES} == exported


@dataclass
class _Store:
    """the pool over a fresh schema holding every hub table, and replicas over it."""

    pool: asyncpg.Pool

    def replica(self, collection: type[BaseCollection[Any]]) -> Any:
        """a Collection over the shared L3 with no L1 and no L2, as the hub wires these.

        every read on a replica that did not write the row is served from L3,
        which is what makes it the read that proves what was persisted.

        :param collection: the Collection class
        :ptype collection: type[BaseCollection[Any]]
        :return: the Collection
        :rtype: Any
        """
        registry = CollectionRegistry()
        registry.configure(l3_pool=self.pool)
        return collection(
            registry=registry,
            config=DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables=""),
        )

    async def stored(self, case: _Case, key: Any) -> dict[str, Any]:
        """the row as Postgres holds it.

        :param case: the case
        :ptype case: _Case
        :param key: the primary-key value
        :ptype key: Any
        :return: the stored row
        :rtype: dict[str, Any]
        """
        row = await self.pool.fetchrow(f"SELECT * FROM {case.table} WHERE {case.pk} = $1", key)  # noqa: S608
        assert row is not None
        return dict(row)


@pytest.fixture
async def store(db_container: str) -> AsyncIterator[_Store]:
    """a fresh schema holding every hub table this package's Collections serve.

    :param db_container: the Postgres container's connection URL
    :ptype db_container: str
    :return: the store over the schema
    :rtype: AsyncIterator[_Store]
    """
    schema = f"dsparity_{uuid.uuid4().hex[:8]}"
    admin = await asyncpg.connect(db_container)
    try:
        await admin.execute(f"CREATE SCHEMA {schema}")
    finally:
        await admin.close()
    pool = await asyncpg.create_pool(
        db_container, min_size=1, max_size=4, server_settings={"search_path": schema}, init=init_connection
    )
    assert pool is not None
    try:
        for statement in _ALL_DDL:
            await pool.execute(statement)
        yield _Store(pool)
    finally:
        await pool.close()
        admin = await asyncpg.connect(db_container)
        try:
            await admin.execute(f"DROP SCHEMA {schema} CASCADE")
        finally:
            await admin.close()


def _stamped(row: dict[str, Any]) -> dict[str, Any]:
    """``row`` with the two timestamps every hub caller sets on a new row.

    :param row: the row
    :ptype row: dict[str, Any]
    :return: a copy carrying ``date_created`` and ``date_updated``
    :rtype: dict[str, Any]
    """
    now = datetime.now(UTC)
    return {**row, "date_created": now, "date_updated": now}


async def _create(store: _Store, case: _Case, row: dict[str, Any]) -> Any:
    """create ``row`` through a fresh replica, then set the hub-owned columns directly.

    :param store: the store
    :ptype store: _Store
    :param case: the case
    :ptype case: _Case
    :param row: the row to create
    :ptype row: dict[str, Any]
    :return: the primary-key value
    :rtype: Any
    """
    writer = store.replica(case.collection)
    await writer.save_entity(writer.create(_stamped(row)))
    key = row[case.pk]
    for column, value in case.hub_owned.items():
        await store.pool.execute(f"UPDATE {case.table} SET {column} = $2 WHERE {case.pk} = $1", key, value)  # noqa: S608
    return key


def _owned(stored: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    """the stored values of the columns ``row`` names.

    :param stored: the stored row
    :ptype stored: dict[str, Any]
    :param row: the row written
    :ptype row: dict[str, Any]
    :return: the stored values, keyed as ``row`` is
    :rtype: dict[str, Any]
    """
    return {column: stored[column] for column in row}


@pytest.mark.parametrize("case", _CASES, ids=lambda case: case.table)
async def test_every_table_column_is_owned_by_the_collection_or_listed_as_hub_owned(store: _Store, case: _Case) -> None:
    table_columns = {
        row["column_name"]
        for row in await store.pool.fetch(
            "SELECT column_name FROM information_schema.columns WHERE table_schema = current_schema() "
            "AND table_name = $1",
            case.table,
        )
    }
    assert table_columns, f"no columns found for {case.table}, so this check would pass vacuously"
    accounted = set(case.row()) | set(case.hub_owned) | _STAMPED
    assert accounted == table_columns


@pytest.mark.parametrize("case", _CASES, ids=lambda case: case.table)
async def test_a_created_row_persists_every_column_the_collection_owns(store: _Store, case: _Case) -> None:
    row = case.row()
    key = await _create(store, case, row)

    stored = await store.stored(case, key)
    assert _owned(stored, row) == row
    assert stored["date_created"] is not None and stored["date_updated"] is not None


@pytest.mark.parametrize("case", _CASES, ids=lambda case: case.table)
async def test_a_read_returns_every_column_typed_as_stored_from_l3_and_through_the_l2_codec(
    store: _Store, case: _Case
) -> None:
    row = case.row()
    key = await _create(store, case, row)
    stored = await store.stored(case, key)

    reader = store.replica(case.collection)
    entity = await reader.get(key)
    assert entity is not None
    read = entity.to_dict()
    owned = [*row, *sorted(_STAMPED)]
    assert {column: read[column] for column in owned} == {column: stored[column] for column in owned}
    assert reader.deserialize(reader.serialize(read)) == read


@pytest.mark.parametrize("case", _CASES, ids=lambda case: case.table)
async def test_an_update_persists_every_column_it_changes_and_leaves_hub_owned_columns_alone(
    store: _Store, case: _Case
) -> None:
    row = case.row()
    key = await _create(store, case, row)

    updater = store.replica(case.collection)
    loaded = await updater.get(key)
    assert loaded is not None
    change = case.change(row)
    for column, value in change.items():
        setattr(loaded, column, value)
    await updater.save_entity(loaded)

    stored = await store.stored(case, key)
    assert _owned(stored, change) == change
    assert _owned(stored, case.hub_owned) == case.hub_owned


# --- the never-SET-what-the-row-does-not-carry rule ---------------------------------------

_PARTIAL_UPDATES: tuple[tuple[_Case, str, Any], ...] = (
    (_CASES[1], "description", "rewritten"),
    (_CASES[3], "description", "rewritten"),
    (_CASES[4], "description", "rewritten"),
)


@pytest.mark.parametrize(
    ("case", "column", "value"), _PARTIAL_UPDATES, ids=[case.table for case, _, _ in _PARTIAL_UPDATES]
)
async def test_a_save_of_a_partial_row_writes_only_the_columns_it_carries(
    store: _Store, case: _Case, column: str, value: Any
) -> None:
    row = case.row()
    key = await _create(store, case, row)
    before = await store.stored(case, key)

    writer = store.replica(case.collection)
    partial = {case.pk: key, column: value, "date_updated": before["date_updated"]}
    if case.collection is TableTemplateCollection:
        # a template row is refused without its visibility; carry the stored one.
        partial["visibility"] = before["visibility"]
    entity = writer.entity_class(partial, is_new=False, collection=writer)
    await writer.save_entity(entity)

    after = await store.stored(case, key)
    assert after[column] == value
    untouched = {name for name in before if name not in partial}
    assert {name: after[name] for name in untouched} == {name: before[name] for name in untouched}
    # the handle holds the row as L3 holds it, not the partial row it sent.
    assert {name: entity.to_dict()[name] for name in after} == after


# --- datasource_tables.coverage_dimension: insert-only ------------------------------------


async def test_an_introspection_write_back_of_a_stale_table_row_keeps_the_coverage_dimension_set_since(
    store: _Store,
) -> None:
    """the introspector saves the row it prefetched; an admin PATCH may land in between."""
    case = _CASES[1]
    row = case.row()
    key = await _create(store, case, row)

    introspector = store.replica(DataSourceTableCollection)
    prefetched = introspector.entity_class(await store.stored(case, key), is_new=False, collection=introspector)
    await store.pool.execute("UPDATE datasource_tables SET coverage_dimension = 'county_fips' WHERE id = $1", key)

    prefetched.column_hash = "11111111111111111111111111111111"
    await introspector.save_entity(prefetched)

    stored = await store.stored(case, key)
    assert stored["coverage_dimension"] == "county_fips"
    assert stored["column_hash"] == "11111111111111111111111111111111"
    assert prefetched.coverage_dimension == "county_fips"


async def test_a_new_table_row_with_no_coverage_dimension_stores_null(store: _Store) -> None:
    case = _CASES[1]
    row = {key: value for key, value in case.row().items() if key != "coverage_dimension"}
    writer = store.replica(DataSourceTableCollection)
    entity = writer.create(_stamped(row))
    await writer.save_entity(entity)

    assert (await store.stored(case, row["id"]))["coverage_dimension"] is None
    assert entity.coverage_dimension is None


# --- datasource_relations ------------------------------------------------------------------


async def test_a_relation_created_without_edges_stores_and_holds_the_column_default(store: _Store) -> None:
    case = _CASES[3]
    row = {key: value for key, value in case.row().items() if key != "edges"}
    writer = store.replica(DataSourceRelationCollection)
    entity = writer.create(_stamped(row))
    await writer.save_entity(entity)

    assert (await store.stored(case, row["id"]))["edges"] == []
    assert entity.edges == []


async def test_a_relation_read_through_the_l2_codec_keeps_customer_id_a_uuid(store: _Store) -> None:
    case = _CASES[3]
    row = case.row()
    key = await _create(store, case, row)
    reader = store.replica(DataSourceRelationCollection)
    entity = await reader.get(key)
    assert entity is not None

    decoded = reader.deserialize(reader.serialize(entity.to_dict()))
    assert isinstance(decoded["customer_id"], uuid.UUID)
    assert decoded["customer_id"] == row["customer_id"]


# --- table_templates -------------------------------------------------------------------------


async def test_a_platform_template_is_read_and_updated_by_id_alone(store: _Store) -> None:
    case = _CASES[4]
    row = case.row()
    key = await _create(store, case, row)

    reader = store.replica(TableTemplateCollection)
    entity = await reader.get(key)
    assert entity is not None and entity.customer_id is None and entity.visibility == "public"

    entity.description = "republished"
    await reader.save_entity(entity)
    assert (await store.stored(case, key))["description"] == "republished"
    assert await store.pool.fetchval("SELECT count(*) FROM table_templates") == 1


async def test_a_template_promoted_from_a_customer_template_keeps_its_origin(store: _Store) -> None:
    case = _CASES[4]
    source = {**case.row(), "customer_id": uuid.uuid4(), "visibility": "private"}
    await _create(store, case, source)
    promoted = {**case.row(), "origin_template_id": source["id"]}
    key = await _create(store, case, promoted)

    stored = await store.stored(case, key)
    assert stored["origin_template_id"] == source["id"]
    assert stored["visibility"] == "public"


async def test_a_template_without_a_visibility_is_refused_and_nothing_is_written(store: _Store) -> None:
    case = _CASES[4]
    row = {key: value for key, value in case.row().items() if key != "visibility"}
    row["customer_id"] = uuid.uuid4()
    writer = store.replica(TableTemplateCollection)

    with pytest.raises(ValueError, match="visibility"):
        await writer.save_entity(writer.create(_stamped(row)))
    assert await store.pool.fetchval("SELECT count(*) FROM table_templates") == 0
