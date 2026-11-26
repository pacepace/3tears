"""``coordination_revocations`` is ROW_STAMPED: declared so, stamped by coordination v007, every write stamped, a delete
and an expiry sweep each a gone mark, and a restore reads its 16 buckets.

Append-only L3 (decision 3): the change-note ring is retired, so the one switched-on coordination table comes back from
its own rows after a lost epoch log. Each test flips on the code before the table was stamped.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import MagicMock

import pytest

from threetears.core.backends.sql import SqlL3Backend
from threetears.core.collections.base import NO_L2
from threetears.core.config import DefaultCoreConfig
from threetears.core.coordination.migrations import register
from threetears.core.coordination.migrations.v007_revocations_carry_their_epoch import revocations_carry_their_epoch
from threetears.core.coordination.tables import CoordinationRevocationsCollection
from threetears.core.data.migrations import MigrationRunner
from threetears.core.data.migrations.stamp_tables import StampingTable
from threetears.core.ddl.row_epoch import next_epoch
from threetears.core.testing.stamped_tables import (
    MigrationStore,
    RecordingPool,
    assert_declared_stamped,
    assert_migration_stamps,
    assert_restore_reads_every_bucket,
    stamps_in,
)

#: the table coordination v007 stamps, its key as declared at v007 (frozen: a later change of its key is its own
#: migration)
STAMPED_TABLES = (
    StampingTable("coordination_revocations", (("purpose", "text"), ("key", "text")), tenancy=("schema", None)),
)

_SCHEMA = CoordinationRevocationsCollection.schema


def _collection(pool: RecordingPool) -> CoordinationRevocationsCollection:
    registry = MagicMock()
    registry.get_l1_backend.return_value = None
    registry.get_l3_pool.return_value = pool
    registry.kv_key_scope = "test"
    collection = CoordinationRevocationsCollection(
        registry, DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables=""), nats_client=NO_L2
    )
    store = SqlL3Backend(pool)
    store.register_schema(_SCHEMA.name, _SCHEMA)
    collection.l3_pool = store
    return collection


def test_the_revocation_list_is_declared_stamped_with_its_restore_index_and_comment() -> None:
    assert_declared_stamped(_SCHEMA)


async def test_v007_stamps_the_existing_table_its_backfill_keyed() -> None:
    assert register(MigrationRunner()).versions[7] is revocations_carry_their_epoch
    store = MigrationStore({"coordination_revocations": [{"purpose": "p", "key": "k1"}, {"purpose": "p", "key": "k2"}]})
    await revocations_carry_their_epoch(store)  # type: ignore[arg-type]
    assert_migration_stamps(store, STAMPED_TABLES, [_SCHEMA])


async def test_a_write_is_stamped_and_rises() -> None:
    pool = RecordingPool()
    collection = _collection(pool)
    before = next_epoch()
    now = datetime.now(UTC)
    await collection.save_to_store(
        {"purpose": "p", "key": "k", "revoked_at": now, "expires_at": now, "date_created": now, "date_updated": now}
    )
    [(sql, params)] = pool.sent
    assert "epoch = GREATEST(coordination_revocations.epoch + 1, EXCLUDED.epoch)" in sql, sql
    assert len(stamps_in(params, before)) == 1


async def test_a_delete_is_a_gone_mark_never_a_delete() -> None:
    pool = RecordingPool()
    collection = _collection(pool)
    before = next_epoch()
    await collection.delete_from_store(("p", "k"))
    [(sql, params)] = pool.sent
    assert sql.startswith('UPDATE "coordination_revocations" SET "gone" = $1'), sql
    assert 'GREATEST("epoch" + 1, $2::bigint)' in sql and 'NOT "gone"' in sql, sql
    assert params[0] is True and len(stamps_in(params, before)) == 1


async def test_an_expiry_sweep_marks_the_expired_rows_gone_never_deletes_them() -> None:
    expired = [{"purpose": "p", "key": "k1", "expires_at": None}, {"purpose": "p", "key": "k2", "expires_at": None}]

    def answer(sql: str, params: tuple[Any, ...]) -> list[dict[str, Any]]:
        return list(expired) if sql.startswith("SELECT") else []

    pool = RecordingPool(answer)
    collection = _collection(pool)
    before = next_epoch()
    assert await collection.sweep_expired(now=datetime.now(UTC) + timedelta(days=1)) == 2
    read, (mark, params) = pool.sent[0][0], pool.sent[-1]
    assert 'FROM "coordination_revocations" AS t' in read and 'NOT t."gone"' in read, read
    assert not [sql for sql, _ in pool.sent if sql.upper().startswith("DELETE")], pool.sent
    assert mark.startswith('UPDATE "coordination_revocations" SET "gone" = $1'), mark
    assert len(stamps_in(params, before)) == 1


async def test_a_restore_reads_its_16_buckets_since_x() -> None:
    await assert_restore_reads_every_bucket([_SCHEMA])


@pytest.mark.parametrize("yugabyte", [True, False])
def test_v001_renders_its_epoch_and_gone_flag_and_leaves_the_bucket_to_v007(yugabyte: bool) -> None:
    from threetears.core.coordination.tables import table_def_for

    table = table_def_for(_SCHEMA)
    names = {column.name for column in table.columns}
    assert {"epoch", "gone"} <= names and "epoch_bucket" not in names
    assert all("epoch_bucket" not in index.columns for index in table.indexes)
    del yugabyte
