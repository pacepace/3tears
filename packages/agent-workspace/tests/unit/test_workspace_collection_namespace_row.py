"""unit tests for WorkspaceCollection._save_to_postgres paired namespace write.

workspace-task-19 Phase 5 (WS-ACL-03): new workspace inserts must
pair the agent-schema ``workspaces`` row with a matching
``platform.namespaces`` row under one transaction. these tests drive
the collection against a mock pool so we can assert both SQL
statements are issued without a live DB.

the default pool mock in :mod:`test_collections` lacks a usable
``acquire()`` so it exercises the fall-back direct-upsert path. a
richer mock here exposes an async context manager returning a
transaction-capable connection so we can verify the two-statement
transactional branch as well.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest

from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from sqlalchemy import Column, DateTime, Integer, MetaData, String, Table, Text

from threetears.agent.workspace.collections import WorkspaceCollection


def _workspaces_metadata() -> MetaData:
    """build SQLite metadata describing workspaces table."""
    metadata = MetaData()
    Table(
        "workspaces",
        metadata,
        Column("id", String(64), primary_key=True),
        Column("agent_id", String(64)),
        Column("name", String(255)),
        Column("description", Text),
        Column("template_name", String(255)),
        Column("created_by", String(64)),
        Column("current_version", Integer),
        Column("date_created", DateTime),
        Column("date_updated", DateTime),
        Column("date_deleted", DateTime),
    )
    return metadata


@pytest.fixture()
def workspaces_l1() -> SQLiteBackend:
    """build SQLite L1 backend with workspaces schema."""
    backend = SQLiteBackend(db_name=f"test_ws_ns_{uuid4().hex[:8]}")
    backend.initialize(_workspaces_metadata())
    yield backend
    backend.reset()


@pytest.fixture()
def config_always() -> DefaultCoreConfig:
    """return core config that flushes writes to L3 immediately."""
    return DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")


class _FakeTransaction:
    """async context manager yielding nothing, just like a real tx."""

    async def __aenter__(self) -> None:
        return None

    async def __aexit__(self, *exc: Any) -> None:
        return None


class _FakeConnection:
    """records SQL statements executed inside the transaction."""

    def __init__(self) -> None:
        self.statements: list[tuple[str, tuple[Any, ...]]] = []

    def transaction(self) -> _FakeTransaction:
        return _FakeTransaction()

    async def execute(self, query: str, *args: Any) -> str:
        self.statements.append((query, args))
        return "INSERT 0 1"


class _FakePoolCM:
    """async context manager returning a :class:`_FakeConnection`."""

    def __init__(self, conn: _FakeConnection) -> None:
        self._conn = conn

    async def __aenter__(self) -> _FakeConnection:
        return self._conn

    async def __aexit__(self, *exc: Any) -> None:
        return None


class _FakePool:
    """mock pool exposing a real async acquire() and bare execute()."""

    def __init__(self) -> None:
        self.conn = _FakeConnection()
        self.direct_statements: list[tuple[str, tuple[Any, ...]]] = []

    def acquire(self) -> _FakePoolCM:
        return _FakePoolCM(self.conn)

    async def execute(self, query: str, *args: Any) -> str:
        self.direct_statements.append((query, args))
        return "INSERT 0 1"

    async def fetchrow(self, query: str, *args: Any) -> None:
        return None


def _make_row(*, customer_id: UUID | None = None) -> dict[str, Any]:
    """build a workspace row dict carrying the customer_id hint."""
    now = datetime.now(UTC)
    return {
        "id": uuid4(),
        "agent_id": uuid4(),
        "name": "test",
        "description": None,
        "template_name": None,
        "created_by": uuid4(),
        "current_version": 0,
        "date_created": now,
        "date_updated": now,
        "date_deleted": None,
        "customer_id": customer_id,
        "schema_name": f"agent_{uuid4().hex}",
    }


@pytest.mark.asyncio
async def test_save_on_insert_writes_both_rows_under_transaction(
    workspaces_l1: SQLiteBackend,
    config_always: DefaultCoreConfig,
) -> None:
    """a new workspace insert emits both the workspace + namespace SQL."""
    customer_id = uuid4()
    pool = _FakePool()
    registry = CollectionRegistry()
    registry.configure(l1_backend=workspaces_l1)
    coll = WorkspaceCollection(registry, config_always, postgres_pool=pool)

    row = _make_row(customer_id=customer_id)
    await coll._save_to_postgres(row)

    statements = pool.conn.statements
    assert len(statements) == 2, f"expected 2 statements, got {statements}"
    ws_sql, ws_args = statements[0]
    ns_sql, ns_args = statements[1]
    assert "INSERT INTO workspaces" in ws_sql
    assert "INSERT INTO platform.namespaces" in ns_sql
    assert "'workspace'" in ns_sql  # namespace_type literal
    # namespace row shares the workspace's id as the primary key
    assert ns_args[0] == row["id"]
    # namespace name follows ``workspace.<id>`` convention
    assert ns_args[1] == f"workspace.{row['id']}"
    # owner_agent_id matches the workspace's agent_id
    assert ns_args[2] == row["agent_id"]
    # customer_id flows through
    assert ns_args[4] == customer_id


@pytest.mark.asyncio
async def test_save_on_update_does_not_touch_namespace_row(
    workspaces_l1: SQLiteBackend,
    config_always: DefaultCoreConfig,
) -> None:
    """an UPDATE path (original_timestamp set) only writes the workspace row."""
    pool = _FakePool()
    registry = CollectionRegistry()
    registry.configure(l1_backend=workspaces_l1)
    coll = WorkspaceCollection(registry, config_always, postgres_pool=pool)

    row = _make_row(customer_id=uuid4())
    now = datetime.now(UTC)
    await coll._save_to_postgres(row, original_timestamp=now)

    statements = pool.conn.statements
    assert len(statements) == 1
    assert "INSERT INTO workspaces" in statements[0][0]


@pytest.mark.asyncio
async def test_save_insert_without_customer_skips_namespace_row(
    workspaces_l1: SQLiteBackend,
    config_always: DefaultCoreConfig,
) -> None:
    """insert without customer_id only writes workspaces (partial upgrade path)."""
    pool = _FakePool()
    registry = CollectionRegistry()
    registry.configure(l1_backend=workspaces_l1)
    coll = WorkspaceCollection(registry, config_always, postgres_pool=pool)

    row = _make_row(customer_id=None)
    await coll._save_to_postgres(row)

    statements = pool.conn.statements
    assert len(statements) == 1
    assert "INSERT INTO workspaces" in statements[0][0]


@pytest.mark.asyncio
async def test_save_without_acquire_falls_back_to_direct_execute(
    workspaces_l1: SQLiteBackend,
    config_always: DefaultCoreConfig,
) -> None:
    """pools without acquire() still emit both statements back-to-back."""

    class _DirectPool:
        def __init__(self) -> None:
            self.statements: list[tuple[str, tuple[Any, ...]]] = []

        async def execute(self, query: str, *args: Any) -> str:
            self.statements.append((query, args))
            return "INSERT 0 1"

        async def fetchrow(self, query: str, *args: Any) -> None:
            return None

    pool = _DirectPool()
    registry = CollectionRegistry()
    registry.configure(l1_backend=workspaces_l1)
    coll = WorkspaceCollection(registry, config_always, postgres_pool=pool)

    row = _make_row(customer_id=uuid4())
    await coll._save_to_postgres(row)

    assert len(pool.statements) == 2
    assert "INSERT INTO workspaces" in pool.statements[0][0]
    assert "INSERT INTO platform.namespaces" in pool.statements[1][0]
