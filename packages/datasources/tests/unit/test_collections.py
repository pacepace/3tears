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
from unittest.mock import MagicMock
from uuid import uuid4

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
