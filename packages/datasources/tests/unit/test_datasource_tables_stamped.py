"""``datasources``, ``datasource_tables`` and ``table_templates`` are ROW_STAMPED: declared so, stamped by the hub's
migration through this package's frozen tables, every write stamped, every delete a gone mark (a template's SET NULL
taken by 3tears), and a restore reads their 16 buckets.

Append-only L3 (decision 3): the change-note ring is retired, so each datasource table comes back from its own rows
after a lost epoch log. Each test flips on the code before the tables were stamped.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from threetears.core.backends.sql import SqlL3Backend
from threetears.core.collections.base import NO_L2
from threetears.core.ddl.row_epoch import next_epoch
from threetears.core.testing.stamped_tables import (
    MigrationStore,
    RecordingPool,
    assert_declared_stamped,
    assert_migration_stamps,
    assert_restore_reads_every_bucket,
    stamps_in,
)
from threetears.datasources.collections import (
    CapabilitySourceCollection,
    DataSourceTableCollection,
    TableTemplateCollection,
)
from threetears.datasources.stamp_tables import STAMPED_TABLES, datasource_tables_carry_their_epoch

_COLLECTIONS: tuple[type[Any], ...] = (CapabilitySourceCollection, DataSourceTableCollection, TableTemplateCollection)

_SCHEMAS = [collection.schema for collection in _COLLECTIONS]


def _collection(cls: type[Any], pool: RecordingPool, siblings: dict[str, Any] | None = None) -> Any:
    registry = MagicMock()
    registry.get_l1_backend.return_value = None
    registry.get_l3_pool.return_value = pool
    registry.register.return_value = None
    registry.generation_source = None
    registry.publish_invalidation = AsyncMock(return_value=None)
    registry.get_collection.side_effect = (siblings or {}).get
    config = MagicMock()
    config.collection_flush = "ALWAYS"
    config.collection_flush_tables = ""
    collection = cls(registry=registry, config=config, nats_client=NO_L2)
    if cls is CapabilitySourceCollection:
        store = SqlL3Backend(pool)
        store.register_schema(collection.schema.name, collection.schema)
        collection.l3_pool = store
    else:
        collection.l3_pool = pool
    return collection


@pytest.mark.parametrize("schema", _SCHEMAS, ids=lambda schema: schema.name)
def test_each_datasource_table_is_declared_stamped_with_its_restore_index_and_comment(schema: Any) -> None:
    assert_declared_stamped(schema)


async def test_the_hubs_migration_stamps_each_existing_table_its_backfill_keyed() -> None:
    store = MigrationStore({schema.name: [{"id": uuid4()}, {"id": uuid4()}] for schema in _SCHEMAS})
    await datasource_tables_carry_their_epoch(store)
    assert_migration_stamps(store, STAMPED_TABLES, _SCHEMAS)


async def test_a_declared_tables_upsert_is_stamped_and_rises() -> None:
    now = datetime.now(UTC)
    rows = {
        DataSourceTableCollection: {
            "id": uuid4(),
            "datasource_id": uuid4(),
            "schema_name": "public",
            "table_name": "t",
            "date_created": now,
            "date_updated": now,
        },
        TableTemplateCollection: {
            "id": uuid4(),
            "name": "t",
            "visibility": "private",
            "date_created": now,
            "date_updated": now,
        },
    }
    for cls, row in rows.items():
        pool = RecordingPool()
        collection = _collection(cls, pool)
        before = next_epoch()
        await collection.save_to_store(dict(row))
        [(sql, params)] = pool.sent
        name = collection.schema.name
        assert f"epoch = GREATEST({name}.epoch + 1, EXCLUDED.epoch)" in sql, sql
        assert len(stamps_in(params, before)) == 1


async def test_a_sources_save_is_stamped_and_rises() -> None:
    pool = RecordingPool()
    collection = _collection(CapabilitySourceCollection, pool)
    before = next_epoch()
    now = datetime.now(UTC)
    await collection.save_to_store(
        {
            "id": uuid4(),
            "name": "s",
            "status": "active",
            "datasource_type": "postgres",
            "date_created": now,
            "date_updated": now,
        }
    )
    [(sql, params)] = pool.sent
    assert "epoch = GREATEST(datasources.epoch + 1, EXCLUDED.epoch)" in sql, sql
    assert len(stamps_in(params, before)) == 1


@pytest.mark.parametrize("cls", _COLLECTIONS, ids=lambda cls: cls.schema.name)
async def test_a_delete_is_a_gone_mark_never_a_delete(cls: type[Any]) -> None:
    pool = RecordingPool()
    collection = _collection(cls, pool)
    before = next_epoch()
    await collection.delete_from_store(uuid4())
    assert not [sql for sql, _ in pool.sent if sql.upper().startswith("DELETE")], pool.sent
    [(sql, params)] = pool.sent
    assert sql.startswith(f'UPDATE "{collection.schema.name}" SET "gone" = $1'), sql
    assert 'GREATEST("epoch" + 1, $2::bigint)' in sql and 'NOT "gone"' in sql, sql
    assert params[0] is True and len(stamps_in(params, before)) == 1


async def test_a_template_delete_sets_its_bound_tables_template_null_itself_stamped() -> None:
    """the template is marked gone, never deleted, so the database's ON DELETE SET NULL never runs: 3tears sets each
    bound table's template_id NULL by key, its epoch risen."""
    template_id, bound = uuid4(), uuid4()

    def answer(sql: str, params: tuple[Any, ...]) -> list[dict[str, Any]]:
        if 'FROM "datasource_tables"' in sql and sql.startswith("SELECT"):
            return [{"id": bound}]
        if 'FROM "table_templates"' in sql:
            return [{"id": template_id, "customer_id": None}]
        return []

    pool = RecordingPool(answer)
    tables = _collection(DataSourceTableCollection, pool)
    tables.invalidate_cache_many = AsyncMock()
    templates = _collection(TableTemplateCollection, pool, {"datasource_tables": tables})
    before = next_epoch()
    await templates.delete(template_id)
    assert not [sql for sql, _ in pool.sent if sql.upper().startswith("DELETE")], pool.sent
    updates = [(sql, params) for sql, params in pool.sent if sql.startswith("UPDATE")]
    [(gone, gone_params)] = [(sql, params) for sql, params in updates if '"table_templates"' in sql]
    [(unbind, unbind_params)] = [(sql, params) for sql, params in updates if '"datasource_tables"' in sql]
    assert gone.startswith('UPDATE "table_templates" SET "gone" = $1'), gone
    assert unbind.startswith('UPDATE "datasource_tables" SET "template_id" = NULL, "epoch" = GREATEST('), unbind
    assert f"{bound}" in unbind_params[-1] and len(stamps_in(unbind_params, before)) == 1
    assert len(stamps_in(gone_params, before)) == 1
    tables.invalidate_cache_many.assert_awaited_once()


async def test_a_restore_reads_each_datasource_tables_16_buckets_since_x() -> None:
    await assert_restore_reads_every_bucket(_SCHEMAS)
