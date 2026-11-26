"""``mcp_tool_grants`` is ROW_STAMPED: declared so, stamped by mcp v003, a grant's write stamped, a revoke a gone mark,
the cold fill reading live grants alone, and a restore reads its 16 buckets.

Append-only L3 (decision 3): the change-note ring is retired, so the grants come back from their own rows after a lost
epoch log. Each test flips on the code before the table was stamped.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from threetears.core.backends.sql import SqlL3Backend
from threetears.core.collections.base import NO_L2
from threetears.core.data.migrations import MigrationRunner
from threetears.core.ddl.row_epoch import next_epoch
from threetears.core.testing.stamped_tables import (
    MigrationStore,
    RecordingPool,
    assert_declared_stamped,
    assert_migration_stamps,
    assert_restore_reads_every_bucket,
    stamps_in,
)
from threetears.mcp.migrations import register
from threetears.mcp.migrations.v003_grants_carry_their_epoch import STAMPED_TABLES, grants_carry_their_epoch
from threetears.mcp.rbac import McpToolGrantCollection

_SCHEMA = McpToolGrantCollection.schema


def _collection(pool: RecordingPool) -> McpToolGrantCollection:
    registry = MagicMock()
    registry.get_l1_backend.return_value = None
    registry.get_l3_pool.return_value = pool
    registry.generation_source = None
    registry.publish_invalidation = AsyncMock(return_value=None)
    config = MagicMock()
    config.collection_flush = "ALWAYS"
    config.collection_flush_tables = ""
    collection = McpToolGrantCollection(registry=registry, config=config, nats_client=NO_L2)
    store = SqlL3Backend(pool)
    store.register_schema(_SCHEMA.name, _SCHEMA)
    collection.l3_pool = store
    return collection


def test_the_grants_are_declared_stamped_with_their_restore_index_and_comment() -> None:
    assert_declared_stamped(_SCHEMA)


async def test_v003_stamps_the_existing_table_its_backfill_keyed_always_attached() -> None:
    assert register(MigrationRunner()).versions[3] is grants_carry_their_epoch
    store = MigrationStore({"mcp_tool_grants": [{"grant_id": uuid4()}, {"grant_id": uuid4()}]})
    await grants_carry_their_epoch(store)  # type: ignore[arg-type]
    assert_migration_stamps(store, STAMPED_TABLES, [_SCHEMA])


async def test_a_grant_is_written_stamped() -> None:
    pool = RecordingPool()
    collection = _collection(pool)
    before = next_epoch()
    await collection.add_grant(principal_type="user", principal_id=uuid4(), tool_name="t", permission="use")
    [(sql, params)] = [(sql, params) for sql, params in pool.sent if sql.startswith("INSERT")]
    assert "epoch = GREATEST(mcp_tool_grants.epoch + 1, EXCLUDED.epoch)" in sql, sql
    assert len(stamps_in(params, before)) == 1


async def test_a_revoke_is_a_gone_mark_never_a_delete() -> None:
    grant_id = uuid4()

    def answer(sql: str, params: tuple[Any, ...]) -> list[dict[str, Any]]:
        if sql.startswith("SELECT"):
            return [
                {
                    "grant_id": grant_id,
                    "principal_type": "user",
                    "principal_id": uuid4(),
                    "tool_name": "t",
                    "permission": "use",
                    "date_created": None,
                    "epoch": 1,
                    "gone": False,
                }
            ]
        return []

    pool = RecordingPool(answer)
    collection = _collection(pool)
    before = next_epoch()
    assert await collection.remove_grant(grant_id)
    assert not [sql for sql, _ in pool.sent if sql.upper().startswith("DELETE")], pool.sent
    [(sql, params)] = [(sql, params) for sql, params in pool.sent if sql.startswith("UPDATE")]
    assert sql.startswith('UPDATE "mcp_tool_grants" SET "gone" = $1'), sql
    assert params[0] is True and len(stamps_in(params, before)) == 1


async def test_the_cold_fill_reads_live_grants_alone() -> None:
    pool = RecordingPool()
    collection = _collection(pool)
    await collection.load_all_grants()
    [(sql, _)] = pool.sent
    assert sql.endswith("FROM mcp_tool_grants WHERE NOT gone"), sql


async def test_a_restore_reads_the_grants_16_buckets_since_x() -> None:
    await assert_restore_reads_every_bucket([_SCHEMA])
