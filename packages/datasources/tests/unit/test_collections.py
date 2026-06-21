"""tests for threetears.datasources.collections.

mirrors the Hub-side coverage that existed pre-relocation:

- ``table_name`` + ``entity_class`` property identity for every
  collection class
- ``serialize`` / ``deserialize`` round-trip for the BaseCollection
  subclasses
- DataSourceCollection's ``find_by_id`` is exercised via integration
  in Hub's existing suite; this unit file only pins the static
  surface (the SchemaBackedCollection-flavored datasource collection
  doesn't expose its own serialize/deserialize -- the schema does
  that).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from threetears.datasources.collections import (
    DataSourceCollection,
    DataSourceColumnCollection,
    DataSourceRelationCollection,
    DataSourceSchemaDigestCollection,
    DataSourceTableCollection,
    TableTemplateCollection,
)
from threetears.datasources.entities import (
    DataSourceColumnEntity,
    DataSourceEntity,
    DataSourceRelationEntity,
    DataSourceSchemaDigestEntity,
    DataSourceTableEntity,
    TableTemplateEntity,
)


def _make_registry_and_config() -> tuple[MagicMock, MagicMock]:
    """build mocked registry and config for collection instantiation.

    :return: tuple of (registry, config) mocks
    :rtype: tuple[MagicMock, MagicMock]
    """
    registry = MagicMock()
    registry.get_l1_backend.return_value = None
    registry.get_l3_pool.return_value = None
    config = MagicMock()
    config.collection_flush = "ALWAYS"
    config.collection_flush_tables = ""
    return registry, config


# -- DataSourceCollection (SchemaBackedCollection) --


class TestDataSourceCollection:
    """the registry collection wires to ``platform.datasources``."""

    def test_table_name(self) -> None:
        registry, config = _make_registry_and_config()
        coll = DataSourceCollection(registry=registry, config=config)
        assert coll.table_name == "datasources"

    def test_entity_class(self) -> None:
        registry, config = _make_registry_and_config()
        coll = DataSourceCollection(registry=registry, config=config)
        assert coll.entity_class is DataSourceEntity

    def test_primary_key_column_is_flat_id(self) -> None:
        # knowledge-task-08: the datasource PK is the flat ``id`` (v016
        # rebuilt the composite ``(customer_id, id)`` partition PK on ``id``
        # alone so a platform-shared datasource can carry customer_id NULL).
        assert DataSourceCollection.primary_key_column == "id"

    @pytest.mark.asyncio
    async def test_iter_active_ids_filters_status_active(self) -> None:
        """audit-pass-3 CRITICAL-1: iter_active_ids must filter by
        ``status = 'active'`` so the scheduler does not probe DISABLED
        datasources every sweep.

        the assertion shape is "the SQL string includes
        ``WHERE status``" + "the value passed is the ACTIVE enum
        value." we don't run a real query; the mock pool records
        the args.
        """
        registry, config = _make_registry_and_config()
        mock_pool = MagicMock()
        mock_pool.fetch = AsyncMock(return_value=[])
        registry.get_l3_pool.return_value = mock_pool

        coll = DataSourceCollection(registry=registry, config=config)
        result = await coll.iter_active_ids()

        assert result == []
        mock_pool.fetch.assert_awaited_once()
        sql_arg, *bound_args = mock_pool.fetch.await_args.args
        assert "WHERE status" in sql_arg
        assert bound_args == ["active"]

    @pytest.mark.asyncio
    async def test_iter_active_ids_returns_only_active_rows(self) -> None:
        """when the L3 pool returns rows, iter_active_ids extracts
        their ids; DISABLED rows are filtered server-side by the
        ``WHERE`` clause (the test verifies the contract by checking
        the SQL + by trusting the pool's filter -- a real Postgres
        instance enforces it).
        """
        registry, config = _make_registry_and_config()
        mock_pool = MagicMock()
        ids = [uuid4() for _ in range(3)]
        mock_pool.fetch = AsyncMock(return_value=[{"id": ds_id} for ds_id in ids])
        registry.get_l3_pool.return_value = mock_pool

        coll = DataSourceCollection(registry=registry, config=config)
        result = await coll.iter_active_ids()

        assert result == ids


# -- DataSourceTableCollection --


class TestDataSourceTableCollection:
    """table collection round-trips JSON-encoded payloads."""

    def test_table_name(self) -> None:
        registry, config = _make_registry_and_config()
        coll = DataSourceTableCollection(registry=registry, config=config)
        assert coll.table_name == "datasource_tables"

    def test_entity_class(self) -> None:
        registry, config = _make_registry_and_config()
        coll = DataSourceTableCollection(registry=registry, config=config)
        assert coll.entity_class is DataSourceTableEntity

    def test_serialize_deserialize_roundtrip(self) -> None:
        registry, config = _make_registry_and_config()
        coll = DataSourceTableCollection(registry=registry, config=config)
        now = datetime.now(UTC)
        data: dict[str, Any] = {
            "id": uuid4(),
            "datasource_id": uuid4(),
            "schema_name": "public",
            "table_name": "users",
            "description": "user accounts",
            "row_count_approx": 10000,
            "caveats": None,
            "column_hash": "abc123def456",
            "date_introspected": now,
            "date_described": None,
            "date_created": now,
            "date_updated": now,
        }
        serialized = coll.serialize(data)
        assert isinstance(serialized, bytes)
        deserialized = coll.deserialize(serialized)
        assert deserialized["id"] == data["id"]
        assert deserialized["schema_name"] == "public"
        assert deserialized["table_name"] == "users"
        assert deserialized["row_count_approx"] == 10000
        # datasource-task-02: column_hash is now part of the
        # canonical table-row shape; round-trip must preserve it
        # (None is the documented "force re-introspect" sentinel,
        # asserted via the missing-key path below).
        assert deserialized["column_hash"] == "abc123def456"

    def test_column_hash_none_sentinel_roundtrips(self) -> None:
        """``None`` column_hash (the migration-backfill sentinel) round-trips cleanly.

        the migration v010 adds the column with no backfill, so every
        existing row reads NULL until the introspector populates it.
        the introspector's diff treats NULL as "force re-introspect";
        deserialization must NOT spuriously promote None to a string.
        """
        registry, config = _make_registry_and_config()
        coll = DataSourceTableCollection(registry=registry, config=config)
        now = datetime.now(UTC)
        data: dict[str, Any] = {
            "id": uuid4(),
            "datasource_id": uuid4(),
            "schema_name": "public",
            "table_name": "users",
            "column_hash": None,
            "date_introspected": now,
            "date_created": now,
            "date_updated": now,
        }
        serialized = coll.serialize(data)
        deserialized = coll.deserialize(serialized)
        assert deserialized.get("column_hash") is None


# -- DataSourceSchemaDigestCollection --


class TestDataSourceSchemaDigestCollection:
    """digest collection round-trips the structured projection by-pk."""

    def test_table_name(self) -> None:
        registry, config = _make_registry_and_config()
        coll = DataSourceSchemaDigestCollection(registry=registry, config=config)
        assert coll.table_name == "datasource_schema_digests"

    def test_entity_class(self) -> None:
        registry, config = _make_registry_and_config()
        coll = DataSourceSchemaDigestCollection(registry=registry, config=config)
        assert coll.entity_class is DataSourceSchemaDigestEntity

    def test_primary_key_is_datasource_id(self) -> None:
        # the digest is addressed BY datasource_id so the agent-side read
        # is a by-pk hot-L1 lookup (schema-priming-task-01b).
        assert DataSourceSchemaDigestEntity.primary_key_field == "datasource_id"

    def test_collection_l1_key_column_is_datasource_id(self) -> None:
        # the COLLECTION's primary_key_column (the L1/L2 key, SEPARATE from
        # the entity's primary_key_field) must also be datasource_id. the
        # BaseCollection default is "id"; this table has no id column, so an
        # inherited default would emit WHERE id=? against the L1 mirror and
        # break every by-pk read/write. regression guard for the inherited-
        # default bug.
        registry, config = _make_registry_and_config()
        coll = DataSourceSchemaDigestCollection(registry=registry, config=config)
        assert coll.primary_key_columns == ("datasource_id",)

    @pytest.mark.asyncio
    async def test_fetch_from_postgres_returns_codec_decoded_list(self) -> None:
        # collections-task-04 (Option B): the read trusts the jsonb codec (hub
        # l3 pool) / the proxy's NATS-JSON decode to hand back ``tables`` as a
        # python list -- no per-collection json.loads. the mock pool stands in
        # for the codec by returning a list, and fetch passes it through. the
        # real codec decode is proven in tests/integration/test_collections_jsonb_live.
        datasource_id = uuid4()
        now = datetime.now(UTC)
        row = {
            "datasource_id": datasource_id,
            "customer_id": uuid4(),
            "tables": [{"schema": "rp", "table": "geo", "description": "d", "columns": []}],
            "source_fingerprint": "fp",
            "date_created": now,
            "date_updated": now,
        }
        coll, _pool = _make_coll_with_pool(DataSourceSchemaDigestCollection, row)
        result = await coll.fetch_from_postgres(datasource_id)
        assert isinstance(result["tables"], list)
        assert result["tables"][0]["table"] == "geo"

    @pytest.mark.asyncio
    async def test_save_binds_string_tables_decoded_to_native_list(self) -> None:
        # a stray pre-encoded JSON string reaching save is normalized to a
        # native list by encode_jsonb (one decode) and bound NATIVELY ($3, no
        # ::text::jsonb cast). the registered codec then applies the single
        # json.dumps -- no per-collection json.dumps, the duplication that let
        # the double-encode bug ship.
        coll, pool = _make_coll_with_pool(DataSourceSchemaDigestCollection)
        pool.execute = AsyncMock(return_value="INSERT 0 1")
        tables_str = '[{"schema": "s", "table": "t", "description": "d", "columns": []}]'
        now = datetime.now(UTC)
        await coll.save_to_postgres(
            {
                "datasource_id": uuid4(),
                "customer_id": uuid4(),
                "tables": tables_str,
                "source_fingerprint": "fp",
                "date_created": now,
                "date_updated": now,
            },
        )
        # $3 (the tables param, 4th positional after the SQL) is the NATIVE
        # python list -- NOT a json string. the codec encodes it once.
        tables_param = pool.execute.await_args.args[3]
        assert tables_param == json.loads(tables_str)
        assert isinstance(tables_param, list)

    @pytest.mark.asyncio
    async def test_save_binds_list_tables_natively(self) -> None:
        # the normal path: a python list is bound natively, unchanged, for the
        # codec to encode exactly once.
        coll, pool = _make_coll_with_pool(DataSourceSchemaDigestCollection)
        pool.execute = AsyncMock(return_value="INSERT 0 1")
        tables = [{"schema": "s", "table": "t", "columns": []}]
        now = datetime.now(UTC)
        await coll.save_to_postgres(
            {
                "datasource_id": uuid4(),
                "customer_id": uuid4(),
                "tables": tables,
                "source_fingerprint": "fp",
                "date_created": now,
                "date_updated": now,
            },
        )
        tables_param = pool.execute.await_args.args[3]
        assert tables_param == tables
        assert isinstance(tables_param, list)

    def test_serialize_deserialize_roundtrip(self) -> None:
        registry, config = _make_registry_and_config()
        coll = DataSourceSchemaDigestCollection(registry=registry, config=config)
        now = datetime.now(UTC)
        datasource_id = uuid4()
        data: dict[str, Any] = {
            "datasource_id": datasource_id,
            "customer_id": uuid4(),
            "tables": [
                {
                    "schema": "reporting_prod",
                    "table": "report_geofacts_joined_data",
                    "description": "joined geo facts",
                    "columns": [
                        {
                            "name": "metric_name",
                            "type": "character varying",
                            "description": "the EAV metric label",
                        },
                    ],
                },
            ],
            "source_fingerprint": "deadbeef",
            "date_created": now,
            "date_updated": now,
        }
        serialized = coll.serialize(data)
        assert isinstance(serialized, bytes)
        deserialized = coll.deserialize(serialized)
        assert deserialized["datasource_id"] == datasource_id
        assert deserialized["source_fingerprint"] == "deadbeef"
        # the structured projection must survive the L2 round-trip intact
        assert deserialized["tables"][0]["table"] == "report_geofacts_joined_data"
        assert deserialized["tables"][0]["columns"][0]["name"] == "metric_name"

    def test_entity_id_is_datasource_id(self) -> None:
        # BaseEntity keys _id off primary_key_field, so the entity the
        # agent reads is addressed by its datasource_id.
        datasource_id = uuid4()
        entity = DataSourceSchemaDigestEntity(
            {
                "datasource_id": datasource_id,
                "customer_id": uuid4(),
                "tables": [],
                "source_fingerprint": "x",
            },
        )
        assert entity.id == datasource_id


# -- DataSourceColumnCollection --


class TestDataSourceColumnCollection:
    """column collection preserves tag list + is_nullable on round-trip."""

    def test_table_name(self) -> None:
        registry, config = _make_registry_and_config()
        coll = DataSourceColumnCollection(registry=registry, config=config)
        assert coll.table_name == "datasource_columns"

    def test_entity_class(self) -> None:
        registry, config = _make_registry_and_config()
        coll = DataSourceColumnCollection(registry=registry, config=config)
        assert coll.entity_class is DataSourceColumnEntity

    def test_serialize_deserialize_roundtrip(self) -> None:
        registry, config = _make_registry_and_config()
        coll = DataSourceColumnCollection(registry=registry, config=config)
        now = datetime.now(UTC)
        data: dict[str, Any] = {
            "id": uuid4(),
            "datasource_id": uuid4(),
            "schema_name": "public",
            "table_name": "users",
            "column_name": "email",
            "data_type": "varchar(255)",
            "is_nullable": False,
            "ordinal_position": 3,
            "description": "user email",
            "valid_range": None,
            "caveats": None,
            "tags": ["pii"],
            "date_introspected": now,
            "date_described": None,
            "date_created": now,
            "date_updated": now,
        }
        serialized = coll.serialize(data)
        deserialized = coll.deserialize(serialized)
        assert deserialized["id"] == data["id"]
        assert deserialized["column_name"] == "email"
        assert deserialized["tags"] == ["pii"]
        assert deserialized["is_nullable"] is False


# -- DataSourceRelationCollection --


class TestDataSourceRelationCollection:
    """relation collection preserves arrays in JSONB shape on round-trip."""

    def test_table_name(self) -> None:
        registry, config = _make_registry_and_config()
        coll = DataSourceRelationCollection(registry=registry, config=config)
        assert coll.table_name == "datasource_relations"

    def test_entity_class(self) -> None:
        registry, config = _make_registry_and_config()
        coll = DataSourceRelationCollection(registry=registry, config=config)
        assert coll.entity_class is DataSourceRelationEntity

    def test_serialize_deserialize_roundtrip(self) -> None:
        registry, config = _make_registry_and_config()
        coll = DataSourceRelationCollection(registry=registry, config=config)
        now = datetime.now(UTC)
        ds_id1 = str(uuid4())
        ds_id2 = str(uuid4())
        data: dict[str, Any] = {
            "id": uuid4(),
            "name": "users_orders",
            "description": "join path",
            "datasource_ids": [ds_id1, ds_id2],
            "join_paths": [{"left": "a.b", "right": "c.d", "type": "inner"}],
            "aggregation_notes": None,
            "caveats": None,
            "date_created": now,
            "date_updated": now,
        }
        serialized = coll.serialize(data)
        deserialized = coll.deserialize(serialized)
        assert deserialized["id"] == data["id"]
        assert deserialized["name"] == "users_orders"
        assert deserialized["datasource_ids"] == [ds_id1, ds_id2]
        assert len(deserialized["join_paths"]) == 1


# -- TableTemplateCollection --


class TestTableTemplateCollection:
    """template collection round-trips customer-scoped rows."""

    def test_table_name(self) -> None:
        registry, config = _make_registry_and_config()
        coll = TableTemplateCollection(registry=registry, config=config)
        assert coll.table_name == "table_templates"

    def test_entity_class(self) -> None:
        registry, config = _make_registry_and_config()
        coll = TableTemplateCollection(registry=registry, config=config)
        assert coll.entity_class is TableTemplateEntity

    def test_serialize_deserialize_roundtrip(self) -> None:
        registry, config = _make_registry_and_config()
        coll = TableTemplateCollection(registry=registry, config=config)
        now = datetime.now(UTC)
        data: dict[str, Any] = {
            "id": uuid4(),
            "customer_id": uuid4(),
            "name": "report_geofacts",
            "description": "geofacts report shape",
            "caveats": None,
            "date_created": now,
            "date_updated": now,
        }
        serialized = coll.serialize(data)
        deserialized = coll.deserialize(serialized)
        assert deserialized["id"] == data["id"]
        assert deserialized["name"] == "report_geofacts"
        assert deserialized["customer_id"] == data["customer_id"]


# ---------------------------------------------------------------------------
# get_by_natural_key (introspector lookup path)
# ---------------------------------------------------------------------------


def _make_coll_with_pool(cls: type[Any], row: dict[str, Any] | None = None) -> tuple[Any, AsyncMock]:
    """build a collection with a mocked L3 pool returning ``row`` from fetchrow.

    used by the natural-key tests below; isolates the collection
    instance from real L3 wiring so the assertion is purely about the
    SQL the method emits + how it maps the row dict back to the entity.

    :param cls: Collection class to instantiate
    :ptype cls: type
    :param row: row dict that ``pool.fetchrow`` returns; None means
        the lookup miss path
    :ptype row: dict[str, Any] | None
    :return: ``(collection, mock_pool)`` so tests can assert the SQL
    :rtype: tuple[Any, AsyncMock]
    """
    registry, config = _make_registry_and_config()
    coll = cls(registry=registry, config=config)
    pool = AsyncMock()
    pool.fetchrow = AsyncMock(return_value=row)
    coll.l3_pool = pool
    return coll, pool


class TestDataSourceTableGetByNaturalKey:
    """``(datasource_id, schema_name, table_name)`` lookup for the introspector."""

    @pytest.mark.asyncio
    async def test_returns_entity_when_row_exists(self) -> None:
        now = datetime.now(UTC)
        ds_id = uuid4()
        row = {
            "id": uuid4(),
            "datasource_id": ds_id,
            "schema_name": "reporting_prod",
            "table_name": "events",
            "date_introspected": now,
            "date_created": now,
            "date_updated": now,
        }
        coll, _pool = _make_coll_with_pool(DataSourceTableCollection, row=row)
        entity = await coll.get_by_natural_key(ds_id, "reporting_prod", "events")
        assert entity is not None
        assert isinstance(entity, DataSourceTableEntity)
        assert entity.id == row["id"]

    @pytest.mark.asyncio
    async def test_returns_none_when_row_missing(self) -> None:
        coll, _pool = _make_coll_with_pool(DataSourceTableCollection, row=None)
        entity = await coll.get_by_natural_key(uuid4(), "s", "t")
        assert entity is None

    @pytest.mark.asyncio
    async def test_passes_natural_key_to_pool(self) -> None:
        coll, pool = _make_coll_with_pool(DataSourceTableCollection, row=None)
        ds_id = uuid4()
        await coll.get_by_natural_key(ds_id, "schema_x", "table_y")
        pool.fetchrow.assert_called_once()
        args = pool.fetchrow.call_args.args
        # ds_id + schema + table are the three positional bind params
        assert args[1:] == (ds_id, "schema_x", "table_y")

    @pytest.mark.asyncio
    async def test_returns_none_when_l3_pool_is_none(self) -> None:
        """defensive null-check on l3_pool (matches the surrounding pattern)."""
        registry, config = _make_registry_and_config()
        coll = DataSourceTableCollection(registry=registry, config=config)
        coll.l3_pool = None
        entity = await coll.get_by_natural_key(uuid4(), "s", "t")
        assert entity is None


class TestDataSourceColumnGetByNaturalKey:
    """``(datasource_id, schema, table, column)`` lookup for the introspector."""

    @pytest.mark.asyncio
    async def test_returns_entity_when_row_exists(self) -> None:
        now = datetime.now(UTC)
        ds_id = uuid4()
        row = {
            "id": uuid4(),
            "datasource_id": ds_id,
            "schema_name": "reporting_prod",
            "table_name": "events",
            "column_name": "user_id",
            "data_type": "integer",
            "is_nullable": False,
            "ordinal_position": 1,
            "tags": [],
            "date_introspected": now,
            "date_created": now,
            "date_updated": now,
        }
        coll, _pool = _make_coll_with_pool(DataSourceColumnCollection, row=row)
        entity = await coll.get_by_natural_key(ds_id, "reporting_prod", "events", "user_id")
        assert entity is not None
        assert isinstance(entity, DataSourceColumnEntity)
        assert entity.id == row["id"]

    @pytest.mark.asyncio
    async def test_returns_none_when_row_missing(self) -> None:
        coll, _pool = _make_coll_with_pool(DataSourceColumnCollection, row=None)
        entity = await coll.get_by_natural_key(uuid4(), "s", "t", "c")
        assert entity is None

    @pytest.mark.asyncio
    async def test_returns_codec_decoded_jsonb_tags(self) -> None:
        """JSONB ``tags`` arrive already decoded to a list (codec / proxy)."""
        now = datetime.now(UTC)
        ds_id = uuid4()
        row = {
            "id": uuid4(),
            "datasource_id": ds_id,
            "schema_name": "s",
            "table_name": "t",
            "column_name": "c",
            "data_type": "varchar",
            "is_nullable": False,
            "ordinal_position": 1,
            # collections-task-04 (Option B): the jsonb codec / proxy NATS-JSON
            # decode hands ``tags`` back as a python list; the mock stands in for
            # the codec by returning a list, and the method passes it through.
            "tags": ["pii", "email"],
            "date_introspected": now,
            "date_created": now,
            "date_updated": now,
        }
        coll, _pool = _make_coll_with_pool(DataSourceColumnCollection, row=row)
        entity = await coll.get_by_natural_key(ds_id, "s", "t", "c")
        assert entity is not None
        assert entity.tags == ["pii", "email"]

    @pytest.mark.asyncio
    async def test_passes_natural_key_to_pool(self) -> None:
        coll, pool = _make_coll_with_pool(DataSourceColumnCollection, row=None)
        ds_id = uuid4()
        await coll.get_by_natural_key(ds_id, "s", "t", "col")
        pool.fetchrow.assert_called_once()
        args = pool.fetchrow.call_args.args
        assert args[1:] == (ds_id, "s", "t", "col")

    @pytest.mark.asyncio
    async def test_returns_none_when_l3_pool_is_none(self) -> None:
        registry, config = _make_registry_and_config()
        coll = DataSourceColumnCollection(registry=registry, config=config)
        coll.l3_pool = None
        entity = await coll.get_by_natural_key(uuid4(), "s", "t", "c")
        assert entity is None
