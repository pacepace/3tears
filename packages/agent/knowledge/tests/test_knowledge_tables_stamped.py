"""``concepts`` and ``playbook_entries`` are ROW_STAMPED: declared so, stamped by the hub's migration through this
package's frozen tables, every write stamped, every delete a gone mark (a shadow's origin link set NULL by 3tears),
every read live, and a restore reads their 16 buckets.

Append-only L3 (decision 3): the change-note ring is retired, so each knowledge table comes back from its own rows
after a lost epoch log, and ``refuse_tables_without_epoch`` refuses a hub migration run while a switched-on table
carries no epoch. Each test fails on the code before the tables were stamped.
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

from threetears.agent.knowledge.collections import ConceptCollection, PlaybookEntryCollection
from threetears.agent.knowledge.stamp_tables import STAMPED_TABLES, knowledge_tables_carry_their_epoch

#: in the frozen list's order
_COLLECTIONS: tuple[type[Any], ...] = (ConceptCollection, PlaybookEntryCollection)

_SCHEMAS = [collection.schema for collection in _COLLECTIONS]

#: each table's self-referencing shadow link (hub v012 / v013, ``ON DELETE SET NULL``)
_ORIGIN: dict[str, str] = {"concepts": "origin_concept_id", "playbook_entries": "origin_entry_id"}


def _collection(cls: type[Any], pool: RecordingPool) -> Any:
    """``cls`` over a store on ``pool``, its registry holding only itself (a shadow's link is its own table's)."""
    registry = MagicMock()
    registry.get_l1_backend.return_value = None
    registry.get_l3_pool.return_value = pool
    registry.register.return_value = None
    registry.generation_source = None
    registry.scan_cache = None
    registry.publish_invalidation = AsyncMock(return_value=None)
    config = MagicMock()
    config.collection_flush = "ALWAYS"
    config.collection_flush_tables = ""
    collection = cls(registry=registry, config=config, nats_client=NO_L2)
    store = SqlL3Backend(pool)
    store.register_schema(collection.schema.name, collection.schema)
    collection.l3_pool = store
    registry.get_collection.side_effect = {collection.schema.name: collection}.get
    return collection


def _row(cls: type[Any]) -> dict[str, Any]:
    """one row of ``cls``'s table, every required column named."""
    now = datetime.now(UTC)
    common = {"id": uuid4(), "datasource_id": uuid4(), "date_created": now, "date_updated": now}
    if cls is ConceptCollection:
        return {**common, "name": "n", "definition": "d"}
    return {**common, "playbook_id": uuid4(), "title": "t", "body": "b"}


@pytest.mark.parametrize("schema", _SCHEMAS, ids=lambda schema: schema.name)
def test_each_knowledge_table_is_declared_stamped_with_its_restore_index_and_comment(schema: Any) -> None:
    assert_declared_stamped(schema)


async def test_the_hubs_migration_stamps_each_existing_table_its_backfill_keyed() -> None:
    store = MigrationStore({schema.name: [{"id": uuid4()}, {"id": uuid4()}] for schema in _SCHEMAS})
    await knowledge_tables_carry_their_epoch(store)
    assert_migration_stamps(store, STAMPED_TABLES, _SCHEMAS)


@pytest.mark.parametrize("cls", _COLLECTIONS, ids=lambda cls: cls.schema.name)
async def test_a_save_is_stamped_and_rises(cls: type[Any]) -> None:
    pool = RecordingPool()
    collection = _collection(cls, pool)
    before = next_epoch()
    await collection.save_to_store(_row(cls))
    [(sql, params)] = pool.sent
    name = collection.schema.name
    assert f"epoch = GREATEST({name}.epoch + 1, EXCLUDED.epoch)" in sql, sql
    assert len(stamps_in(params, before)) == 1


@pytest.mark.parametrize("cls", _COLLECTIONS, ids=lambda cls: cls.schema.name)
async def test_a_delete_is_a_gone_mark_never_a_delete(cls: type[Any]) -> None:
    pool = RecordingPool()
    collection = _collection(cls, pool)
    before = next_epoch()
    await collection.delete_from_store(uuid4())
    assert not [sql for sql, _ in pool.sent if sql.upper().startswith("DELETE")], pool.sent
    [(sql, params)] = pool.sent
    assert sql.startswith(f'UPDATE "{collection.schema.name}" SET "gone" = '), sql
    assert "GREATEST" in sql and "NOT" in sql and '"gone"' in sql, sql
    assert True in params and len(stamps_in(params, before)) == 1


@pytest.mark.parametrize("cls", _COLLECTIONS, ids=lambda cls: cls.schema.name)
async def test_an_originals_delete_unlinks_its_shadows_itself_stamped(cls: type[Any]) -> None:
    """the original is marked gone, never deleted, so the database's ON DELETE SET NULL never runs: 3tears sets each
    shadow's origin link NULL by key, its epoch risen, and marks the original gone."""
    original, shadow = uuid4(), uuid4()
    origin = _ORIGIN[cls.schema.name]

    def answer(sql: str, params: tuple[Any, ...]) -> list[dict[str, Any]]:
        if sql.startswith("SELECT") and f'"{origin}"' in sql:
            return [{"id": shadow}]
        return []

    pool = RecordingPool(answer)
    collection = _collection(cls, pool)
    collection.invalidate_cache_many = AsyncMock()
    before = next_epoch()
    await collection.delete(original)
    assert not [sql for sql, _ in pool.sent if sql.upper().startswith("DELETE")], pool.sent
    updates = [(sql, params) for sql, params in pool.sent if sql.startswith("UPDATE")]
    [(unlink, unlink_params)] = [(sql, params) for sql, params in updates if f'"{origin}" = NULL' in sql]
    [(gone, gone_params)] = [(sql, params) for sql, params in updates if 'SET "gone" = ' in sql]
    assert unlink.startswith(f'UPDATE "{cls.schema.name}" SET "{origin}" = NULL, "epoch" = GREATEST('), unlink
    assert f"{shadow}" in f"{unlink_params}" and len(stamps_in(unlink_params, before)) == 1
    assert len(stamps_in(gone_params, before)) == 1


class _ScopedPool(RecordingPool):
    """the rbac proxy pool's shape: each read takes the conversation's customer scope."""

    async def fetch(self, sql: str, *params: Any, customer_scope: Any = None) -> list[dict[str, Any]]:
        assert customer_scope is not None
        return await super().fetch(sql, *params)


@pytest.mark.parametrize("cls", _COLLECTIONS, ids=lambda cls: cls.schema.name)
async def test_a_drafts_read_answers_live_rows_only(cls: type[Any]) -> None:
    pool = _ScopedPool()
    collection = _collection(cls, pool)
    collection.l3_pool = pool
    await collection.list_own_drafts(uuid4(), customer_scope=uuid4())
    [(sql, _)] = pool.sent
    assert 'NOT t."gone"' in sql, sql


async def test_a_restore_reads_each_knowledge_tables_16_buckets_since_x() -> None:
    await assert_restore_reads_every_bucket(_SCHEMAS)
