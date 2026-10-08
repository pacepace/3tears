"""A collection class built from a table schema, for tables declared as data rather than as classes."""

from __future__ import annotations

from threetears.core.collections.registry import CollectionRegistry
from threetears.core.collections.schema_backed import (
    DATETIMETZ_TYPE,
    NUMERIC_TYPE,
    STRING_TYPE,
    Column,
    SchemaBackedCollection,
    TableSchema,
    collection_for_schema,
)
import pytest

from threetears.core.config import DefaultCoreConfig
from threetears.core.entities.base import BaseEntity


class _ResultRow(BaseEntity):
    """a result row: its own id is the geography, within the race its key leads with."""

    primary_key_field: str = "geo_id"


def _schema(name: str = "results") -> TableSchema:
    return TableSchema(
        name=name,
        primary_key=("office_key", "geo_id"),
        columns=[
            Column("office_key", STRING_TYPE),
            Column("geo_id", STRING_TYPE),
            Column("votes", NUMERIC_TYPE, nullable=True, precision=20, scale=4),
            Column("date_created", DATETIMETZ_TYPE, immutable=True),
            Column("date_updated", DATETIMETZ_TYPE),
        ],
    )


def test_the_class_carries_the_schema_its_table_and_key() -> None:
    schema = _schema()
    collection_class = collection_for_schema(schema, entity_class=_ResultRow)
    assert issubclass(collection_class, SchemaBackedCollection)
    assert collection_class.schema is schema
    assert collection_class.primary_key_column == ("office_key", "geo_id")
    assert "results" in collection_class.__name__


def test_an_instance_names_its_table_and_builds_entities() -> None:
    collection_class = collection_for_schema(_schema(), entity_class=_ResultRow)
    registry = CollectionRegistry()
    collection = collection_class(registry, DefaultCoreConfig(), None)
    assert collection.table_name == "results"
    entity = collection.create({"office_key": "sn_NC1", "geo_id": "37067", "votes": None})
    assert isinstance(entity, BaseEntity)
    assert entity.is_new


def test_the_entity_class_given_is_the_one_used() -> None:
    class ResultRow(BaseEntity):
        primary_key_field: str = "office_key"

    collection_class = collection_for_schema(_schema(), entity_class=ResultRow)
    registry = CollectionRegistry()
    collection = collection_class(registry, DefaultCoreConfig(), None)
    assert collection.entity_class is ResultRow


def test_two_schemas_give_two_classes() -> None:
    first = collection_for_schema(_schema("a"), entity_class=_ResultRow)
    second = collection_for_schema(_schema("b"), entity_class=_ResultRow)
    assert first is not second
    assert first.schema.name == "a"
    assert second.schema.name == "b"


def test_a_single_column_key_names_the_entity_s_key_whole() -> None:
    """A key written as one name keys the entity on that name, not on its first letter."""
    schema = TableSchema(
        name="loads",
        primary_key="source",
        columns=[
            Column("source", STRING_TYPE),
            Column("date_created", DATETIMETZ_TYPE, immutable=True),
            Column("date_updated", DATETIMETZ_TYPE),
        ],
    )
    collection = collection_for_schema(schema)(CollectionRegistry(), DefaultCoreConfig(), None)
    assert collection.entity_class.primary_key_field == "source"
    assert collection.create({"source": "geos"}).id == "geos"


def test_a_composite_key_with_no_entity_class_is_refused_naming_the_table() -> None:
    """On a composite key an entity's id names its bare row id, not the column the key leads with;
    which column that is only the caller knows, so it is never guessed."""
    with pytest.raises(ValueError, match="results.*composite key.*entity_class"):
        collection_for_schema(_schema())
