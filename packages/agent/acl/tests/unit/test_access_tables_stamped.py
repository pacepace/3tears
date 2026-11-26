"""the five access tables are ROW_STAMPED: declared so, stamped by agent_acl v006, every write stamped, every delete a
gone mark (a group's memberships and grants, a namespace's grants, with it), and a restore reads their 16 buckets.

Append-only L3 (decision 3): the change-note ring is retired, so each access table comes back from its own rows after a
lost epoch log. A delete marks a row gone in place, so the database's ``ON DELETE CASCADE`` no longer runs: 3tears takes
each cascade itself, or a deleted group's members would keep its roles. Each test flips on the code before the tables
were stamped.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import pytest

from threetears.agent.acl.builtin_roles import ensure_platform_builtin_tool_user_role
from threetears.agent.acl.collections import (
    GroupCollection,
    GroupMemberCollection,
    NamespaceCollection,
    RoleAssignmentCollection,
    RoleCollection,
)
from threetears.agent.acl.migrations import register
from threetears.agent.acl.migrations.v006_access_tables_carry_their_epoch import (
    STAMPED_TABLES,
    access_tables_carry_their_epoch,
)
from threetears.agent.acl.query_visibility import caller_visible_customer_clause
from threetears.core.backends.sql import SqlL3Backend
from threetears.core.collections.base import NO_L2
from threetears.core.collections.schema_backed import (
    BOOL_TYPE,
    DATETIMETZ_TYPE,
    INT_TYPE,
    JSONB_TYPE,
    STRING_TYPE,
    UUID_TYPE,
)
from threetears.core.data.migrations import MigrationRunner
from threetears.core.ddl.row_epoch import EPOCH_COLUMN, next_epoch
from threetears.core.testing.stamped_tables import (
    MigrationStore,
    RecordingPool,
    assert_declared_stamped,
    assert_migration_stamps,
    assert_restore_reads_every_bucket,
    stamps_in,
)

#: the access collections, in v006's order
_COLLECTIONS: tuple[type[Any], ...] = (
    NamespaceCollection,
    GroupCollection,
    GroupMemberCollection,
    RoleCollection,
    RoleAssignmentCollection,
)

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
    store = SqlL3Backend(pool)
    store.register_schema(collection.schema.name, collection.schema)
    collection.l3_pool = store
    return collection


@pytest.mark.parametrize("schema", _SCHEMAS, ids=lambda schema: schema.name)
def test_each_access_table_is_declared_stamped_with_its_restore_index_and_comment(schema: Any) -> None:
    assert_declared_stamped(schema)


async def test_v006_stamps_each_existing_table_its_backfill_keyed() -> None:
    assert register(MigrationRunner()).versions[6] is access_tables_carry_their_epoch
    keys = {
        "namespaces": [{"row_scope": "customer", "namespace_id": uuid4()}],
        "groups": [{"row_scope": "customer", "group_id": uuid4()}],
        "group_members": [{"group_id": uuid4(), "id": uuid4()}],
        "roles": [{"role_id": uuid4()}],
        "role_assignments": [{"row_scope": "platform", "assignment_id": uuid4()}],
    }
    store = MigrationStore(keys)
    await access_tables_carry_their_epoch(store)  # type: ignore[arg-type]
    assert_migration_stamps(store, STAMPED_TABLES, _SCHEMAS)


@pytest.mark.parametrize("cls", _COLLECTIONS, ids=lambda cls: cls.schema.name)
async def test_a_save_of_each_access_table_is_stamped_and_rises(cls: type[Any]) -> None:
    pool = RecordingPool()
    collection = _collection(cls, pool)
    before = next_epoch()
    now = datetime.now(UTC)
    samples: dict[str, Any] = {
        UUID_TYPE: uuid4(),
        STRING_TYPE: "customer",
        DATETIMETZ_TYPE: now,
        JSONB_TYPE: {},
        BOOL_TYPE: False,
        INT_TYPE: 0,
    }
    row = {
        column.name: samples[column.column_type]
        for column in collection.schema.columns
        if column.name != EPOCH_COLUMN and (not column.nullable or column.name == "customer_id")
    }
    row["customer_id"] = uuid4()
    await collection.save_to_store(row)
    [(sql, params)] = pool.sent
    name = collection.schema.name
    assert f"{EPOCH_COLUMN} = GREATEST({name}.{EPOCH_COLUMN} + 1, EXCLUDED.{EPOCH_COLUMN})" in sql, sql
    assert len(stamps_in(params, before)) == 1


async def test_the_raw_inserts_carry_their_epoch() -> None:
    """the builtin role's insert, a namespace's ensure and a grant's ensure each carry a fresh epoch."""
    before = next_epoch()
    pool = RecordingPool()
    roles = MagicMock()
    roles.l3_pool = pool
    roles.invalidate_cache = AsyncMock()
    await ensure_platform_builtin_tool_user_role(roles)
    inserts = [(sql, params) for sql, params in pool.sent if sql.startswith("INSERT INTO roles")]
    [(sql, params)] = inserts
    assert "epoch" in sql.split(")", 1)[0] and len(stamps_in(params, before)) == 1, sql

    def namespace(sql: str, params: tuple[Any, ...]) -> list[dict[str, Any]]:
        return []

    pool = RecordingPool(namespace)
    namespaces = _collection(NamespaceCollection, pool)
    namespaces.invalidate_cache = AsyncMock()
    with pytest.raises(ValueError, match="not written"):
        # the double holds no row, so the ensure finds none written; its insert is what is asked of here
        await namespaces.ensure_namespace(
            namespace_id=uuid4(), name="n", namespace_type="tool", owner_agent_id=None, customer_id=None
        )
    [(sql, params)] = [(sql, params) for sql, params in pool.sent if sql.startswith("INSERT INTO namespaces")]
    assert "epoch" in sql.split(")", 1)[0] and len(stamps_in(params, before)) == 1, sql

    pool = RecordingPool(lambda sql, params: [{"assignment_id": uuid4()}] if sql.startswith("INSERT") else [])
    grants = _collection(RoleAssignmentCollection, pool)
    grants.invalidate_cache_many = AsyncMock()
    await grants.ensure_group_role_assignment(group_id=uuid4(), role_id=uuid4(), scope_type="all", scope_id=None)
    [(sql, params)] = [(sql, params) for sql, params in pool.sent if sql.startswith("INSERT INTO role_assignments")]
    assert "epoch" in sql.split(")", 1)[0] and len(stamps_in(params, before)) == 1, sql


async def test_a_revoked_grant_ensured_again_is_made_live_in_place() -> None:
    """a revoke marks its grant gone, and the gone row keeps the natural key: an ensure of the same grant revives it
    rather than failing on the natural-key index that absorbed its insert."""
    revived = uuid4()

    def answer(sql: str, params: tuple[Any, ...]) -> list[dict[str, Any]]:
        if sql.startswith('UPDATE "role_assignments"'):
            return [{"assignment_id": revived, "row_scope": "platform"}]
        return []

    pool = RecordingPool(answer)
    grants = _collection(RoleAssignmentCollection, pool)
    grants.invalidate_cache_many = AsyncMock()
    before = next_epoch()
    assignment_id, created = await grants.ensure_group_role_assignment(
        group_id=uuid4(), role_id=uuid4(), scope_type="all", scope_id=None
    )
    assert (assignment_id, created) == (revived, True)
    [(sql, params)] = [(sql, params) for sql, params in pool.sent if sql.startswith("UPDATE")]
    assert 'SET "gone" = $1' in sql and '("gone")' in sql and params[0] is False, sql
    assert len(stamps_in(params, before)) == 1


async def test_a_group_delete_marks_it_its_memberships_and_its_grants_gone() -> None:
    group_id, member, grant = uuid4(), uuid4(), uuid4()

    def answer(sql: str, params: tuple[Any, ...]) -> list[dict[str, Any]]:
        if 'FROM "group_members"' in sql:
            return [{"member_type": "user", "member_id": uuid4(), "id": member}]
        if 'FROM "role_assignments"' in sql:
            return [{"row_scope": "customer", "assignment_id": grant}]
        return []

    pool = RecordingPool(answer)
    members = _collection(GroupMemberCollection, pool)
    assignments = _collection(RoleAssignmentCollection, pool)
    members.invalidate_cache_many = AsyncMock()
    assignments.invalidate_cache_many = AsyncMock()
    groups = _collection(GroupCollection, pool, {"group_members": members, "role_assignments": assignments})
    before = next_epoch()
    await groups.delete(("customer", group_id))
    assert not [sql for sql, _ in pool.sent if sql.upper().startswith("DELETE")], pool.sent
    marks = {sql.split('"', 2)[1]: params for sql, params in pool.sent if sql.startswith("UPDATE")}
    assert set(marks) == {"groups", "group_members", "role_assignments"}, pool.sent
    for table, params in marks.items():
        assert params[0] is True and len(stamps_in(params, before)) == 1, (table, params)
    # each mark names its rows by key (bound as text: the double takes no native arrays)
    assert [f"{member}"] in marks["group_members"] and [f"{grant}"] in marks["role_assignments"]


async def test_a_namespace_delete_marks_it_and_its_grants_gone() -> None:
    namespace_id, grant = uuid4(), uuid4()

    def answer(sql: str, params: tuple[Any, ...]) -> list[dict[str, Any]]:
        if 'FROM "role_assignments"' in sql:
            return [{"row_scope": "customer", "assignment_id": grant, "group_id": uuid4()}]
        return []

    pool = RecordingPool(answer)
    assignments = _collection(RoleAssignmentCollection, pool)
    assignments.invalidate_cache_many = AsyncMock()
    namespaces = _collection(NamespaceCollection, pool, {"role_assignments": assignments})
    before = next_epoch()
    await namespaces.delete(("customer", namespace_id))
    assert not [sql for sql, _ in pool.sent if sql.upper().startswith("DELETE")], pool.sent
    marks = {sql.split('"', 2)[1]: params for sql, params in pool.sent if sql.startswith("UPDATE")}
    assert set(marks) == {"namespaces", "role_assignments"}, pool.sent
    for table, params in marks.items():
        assert params[0] is True and len(stamps_in(params, before)) == 1, (table, params)


def test_the_visibility_clause_reads_live_access_rows_alone() -> None:
    """a grant, role, membership, group or namespace a delete marked gone admits no row."""
    fragment, _ = caller_visible_customer_clause(
        user_id=UUID(int=1), customer_id_column="t.customer_id", scope_namespace_type="tool"
    )
    for alias in ("ra", "vis_role", "gm", "vis_group", "ns", "gm2"):
        assert f"NOT {alias}.gone" in fragment, alias


async def test_a_restore_reads_each_access_tables_16_buckets_since_x() -> None:
    await assert_restore_reads_every_bucket(_SCHEMAS)
