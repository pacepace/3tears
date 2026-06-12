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

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from threetears.datasources.collections import (
    DataSourceCollection,
    DataSourceColumnCollection,
    DataSourceRelationCollection,
    DataSourceTableCollection,
    TableTemplateCollection,
)
from threetears.datasources.entities import (
    DataSourceColumnEntity,
    DataSourceEntity,
    DataSourceRelationEntity,
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

    def test_primary_key_column_is_composite(self) -> None:
        assert DataSourceCollection.primary_key_column == ("customer_id", "id")

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
    async def test_parses_jsonb_tags_when_stored_as_string(self) -> None:
        """JSONB ``tags`` arrive as ``str`` from raw L3; method round-trips them."""
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
            "tags": '["pii", "email"]',  # JSONB-as-str from asyncpg
            "date_introspected": now,
            "date_created": now,
            "date_updated": now,
        }
        coll, _pool = _make_coll_with_pool(DataSourceColumnCollection, row=row)
        entity = await coll.get_by_natural_key(ds_id, "s", "t", "c")
        assert entity is not None
        # the method MUST json-decode the tags array so callers see a list
        # (matches the existing fetch_from_store behaviour)
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
