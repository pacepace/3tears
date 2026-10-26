"""the datasource collections that write their own SQL cache in any L1 a host gives them, exactly.

Two defects, both found when the hub handed its L1 to every collection its registry builds:

- these collections declared no table, so the first read through an L1 that had not been told
  about them failed (``no such table: datasource_columns``) and a host had to hand-copy each
  table's columns into its own L1 schema. Each now declares its table as a ``TableSchema``, and its
  L1 table is generated from it and created on whatever L1 it is given.
- ``DataSourceColumnCollection.save_to_store`` upserted on the column's natural key without
  returning the id it stored. Two introspections racing on one new column each send a fresh id;
  the second lands on the first's row, so every tier went on serving the first's row as it was,
  and cached the second's row under an id no row holds. The upsert now returns the stored id: the
  stored row is evicted and the sent id caches nothing.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

import pytest
from sqlalchemy import MetaData

from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections.base import BaseCollection
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.datasources.collections import (
    DataSourceColumnCollection,
    DataSourceRelationCollection,
    DataSourceSchemaDigestCollection,
    DataSourceTableCollection,
    TableTemplateCollection,
)

#: the columns ``DataSourceColumnCollection``'s upsert binds, in its parameter order
_NATURAL_KEY = ("datasource_id", "schema_name", "table_name", "column_name")


class _ColumnTable:
    """``datasource_columns`` in memory, upserting on its natural key as Postgres does.

    An asyncpg-pool-shaped double: the raw pool has no production protocol (``l3_pool`` is
    typed ``Any`` by design), so there is nothing to declare parity against.
    """

    def __init__(self) -> None:
        self.rows: dict[UUID, dict[str, Any]] = {}

    async def fetchrow(self, sql: str, *args: Any) -> dict[str, Any] | None:
        if sql.startswith("INSERT"):
            return self._upsert(sql, args)
        row = self.rows.get(args[0])
        return None if row is None else dict(row)

    def _upsert(self, sql: str, args: tuple[Any, ...]) -> dict[str, Any]:
        """the upsert on the natural key, its columns named by the statement, returning the stored id."""
        columns = [c.strip() for c in sql[sql.index("(") + 1 : sql.index(")")].split(",")]
        sent = dict(zip(columns, args, strict=True))
        natural = tuple(sent[c] for c in _NATURAL_KEY)
        held = next((r for r in self.rows.values() if tuple(r[c] for c in _NATURAL_KEY) == natural), None)
        if held is None:
            self.rows[sent["id"]] = sent
            return {"id": sent["id"]}
        held.update({k: v for k, v in sent.items() if k not in {"id", "date_created", *_NATURAL_KEY}})
        return {"id": held["id"]}

    async def execute(self, sql: str, *args: Any) -> str:
        return "INSERT 0 1"

    async def fetch(self, sql: str, *args: Any) -> list[dict[str, Any]]:
        return []


@pytest.fixture()
def l1() -> Iterator[SQLiteBackend]:
    """an L1 told about no table at all, as a host's L1 that never heard of these collections."""
    backend = SQLiteBackend(db_name=f"datasource_tables_{uuid.uuid4().hex[:8]}")
    backend.initialize(MetaData())
    yield backend
    backend.reset()


def _config() -> DefaultCoreConfig:
    return DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")


def _column(column_id: UUID, datasource_id: UUID, description: str) -> dict[str, Any]:
    now = datetime.now(UTC)
    return {
        "id": column_id,
        "datasource_id": datasource_id,
        "schema_name": "reporting",
        "table_name": "events",
        "column_name": "user_id",
        "data_type": "integer",
        "is_nullable": False,
        "ordinal_position": 1,
        "description": description,
        "valid_range": None,
        "caveats": None,
        "tags": [],
        "caveats_replaces_definition": False,
        "date_introspected": now,
        "date_described": None,
        "date_created": now,
        "date_updated": now,
    }


class TestEachCollectionCreatesItsL1Table:
    """a collection built on an L1 that has no table for it creates the table from its schema."""

    @pytest.mark.parametrize(
        "collection_class",
        [
            DataSourceColumnCollection,
            DataSourceRelationCollection,
            DataSourceSchemaDigestCollection,
            DataSourceTableCollection,
            TableTemplateCollection,
        ],
    )
    def test_the_table_is_created_with_every_declared_column(
        self, l1: SQLiteBackend, collection_class: type[BaseCollection[Any]]
    ) -> None:
        registry = CollectionRegistry()
        registry.configure(l1_backend=l1)
        collection = collection_class(registry=registry, config=_config(), nats_client=None)
        assert l1.has_table(collection.table_name)
        schema = collection_class.schema  # type: ignore[attr-defined]
        assert set(l1.column_types(collection.table_name)) == {c.name for c in schema.columns}
        assert collection.primary_key_columns == schema.pk_columns


class TestARacingColumnUpsertSettlesTheStoredRow:
    """an upsert that lands on a column stored under another id leaves no tier serving a stale row."""

    async def test_the_stored_row_is_evicted_and_the_sent_id_caches_nothing(self, l1: SQLiteBackend) -> None:
        table = _ColumnTable()
        registry = CollectionRegistry()
        registry.configure(l1_backend=l1, l3_pool=table)
        columns = DataSourceColumnCollection(registry=registry, config=_config(), nats_client=None)
        datasource_id, first_id, second_id = uuid4(), uuid4(), uuid4()

        await columns.save_entity(columns.create(_column(first_id, datasource_id, "first pass")))
        cached = await columns.get(first_id)
        assert cached is not None and cached.description == "first pass"

        # a second introspection that did not see the first's row sends its own fresh id
        await columns.save_entity(columns.create(_column(second_id, datasource_id, "second pass")))

        assert set(table.rows) == {first_id}
        assert columns.get_row_sync(first_id) is None, "the stored row's old copy is still cached"
        assert columns.get_row_sync(second_id) is None, "a row is cached under an id no row holds"
        reread = await columns.get(first_id)
        assert reread is not None and reread.description == "second pass"
        assert await columns.get(second_id) is None
