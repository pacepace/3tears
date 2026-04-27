"""unit tests for workspace collections covering serialize, save, delete, and invalidation."""

from __future__ import annotations

import base64
import json
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest
from sqlalchemy import CHAR, Column, DateTime, Integer, LargeBinary, MetaData, String, Table, Text

from pydantic import BaseModel

from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections.registry import CollectionRegistry
from threetears.nats import Subject, Subjects
from threetears.core.config import DefaultCoreConfig

from threetears.agent.workspace.collections import (
    WorkspaceCollection,
    WorkspaceFileCollection,
    WorkspaceFileVersionCollection,
)
from threetears.agent.workspace.entities import Workspace, WorkspaceFile, WorkspaceFileVersion


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


def _workspace_files_metadata() -> MetaData:
    """build SQLite metadata describing workspace_files table."""
    metadata = MetaData()
    Table(
        "workspace_files",
        metadata,
        Column("id", String(64), primary_key=True),
        Column("workspace_id", String(64)),
        Column("relative_path", String(512)),
        Column("content", LargeBinary),
        Column("sha256", CHAR(64)),
        Column("version", Integer),
        Column("date_updated", DateTime),
    )
    return metadata


def _workspace_file_versions_metadata() -> MetaData:
    """build SQLite metadata describing workspace_file_versions table."""
    metadata = MetaData()
    Table(
        "workspace_file_versions",
        metadata,
        Column("id", String(64), primary_key=True),
        Column("workspace_id", String(64)),
        Column("relative_path", String(512)),
        Column("version", Integer),
        Column("content", LargeBinary),
        Column("sha256", CHAR(64)),
        Column("action", String(32)),
        Column("label", String(255)),
        Column("actor_id", String(64)),
        Column("correlation_id", String(64)),
        Column("date_created", DateTime),
    )
    return metadata


def _make_workspace_row() -> dict[str, Any]:
    """return sample workspaces row dict."""
    return {
        "id": uuid4(),
        "agent_id": uuid4(),
        "name": "docs",
        "description": "docs workspace",
        "template_name": None,
        "created_by": uuid4(),
        "current_version": 0,
        "date_created": datetime.now(UTC),
        "date_updated": datetime.now(UTC),
        "date_deleted": None,
    }


def _make_workspace_file_row() -> dict[str, Any]:
    """return sample workspace_files row dict."""
    return {
        "id": uuid4(),
        "workspace_id": uuid4(),
        "relative_path": "README.md",
        "content": b"# hi\n\x00binary",
        "sha256": "a" * 64,
        "version": 1,
        "date_updated": datetime.now(UTC),
    }


def _make_workspace_file_version_row() -> dict[str, Any]:
    """return sample workspace_file_versions row dict."""
    return {
        "id": uuid4(),
        "workspace_id": uuid4(),
        "relative_path": "README.md",
        "version": 1,
        "content": b"# hi\n\x00binary",
        "sha256": "b" * 64,
        "action": "create",
        "label": None,
        "actor_id": uuid4(),
        "correlation_id": uuid4(),
        "date_created": datetime.now(UTC),
    }


class _FakeNatsBus:
    """in-process fake NATS client matching typed NatsClient surface.

    supports the two methods the cache-invalidation path uses:
    ``publish(subject=Subject, message=BaseModel)`` and
    ``subscribe_typed(subject=Subject, message_type=BaseModel, cb=...)``.
    each subscriber owns its own decoder, so multiple subscribers on the
    same subject with different message_types coexist.
    """

    def __init__(self) -> None:
        """initialize empty subscriber registry and KV store."""
        self._typed_subscribers: dict[str, list[tuple[type[BaseModel], Callable[[Any], Awaitable[None]]]]] = {}
        self._kv: dict[str, dict[str, bytes]] = {}

    def bucket_name(self, suffix: str) -> str:
        """
        returns canonical bucket name for suffix.

        :param suffix: logical bucket suffix
        :ptype suffix: str
        :return: bucket name
        :rtype: str
        """
        return f"test_{suffix}"

    async def publish(
        self,
        *,
        subject: Subject,
        message: BaseModel,
        reply_to: Subject | None = None,
    ) -> None:
        """
        deliver typed message synchronously to all subject subscribers.

        :param subject: target subject
        :ptype subject: Subject
        :param message: typed pydantic message to dispatch
        :ptype message: BaseModel
        :param reply_to: ignored, present for api parity
        :ptype reply_to: Subject | None
        """
        del reply_to
        payload = message.model_dump_json().encode("utf-8")
        for message_type, handler in self._typed_subscribers.get(subject.path, []):
            decoded = message_type.model_validate_json(payload)
            await handler(decoded)

    async def subscribe_typed(
        self,
        *,
        subject: Subject,
        cb: Callable[[Any], Awaitable[None]],
        message_type: type[BaseModel],
        queue: str | None = None,
        max_in_flight: int | None = None,
        deadletter_on_error: bool = True,
    ) -> None:
        """
        register typed handler for subject.

        :param subject: target subject
        :ptype subject: Subject
        :param cb: async callback receiving the decoded message
        :ptype cb: Callable[[Any], Awaitable[None]]
        :param message_type: pydantic class to decode incoming bytes into
        :ptype message_type: type[BaseModel]
        :param queue: ignored, present for api parity
        :ptype queue: str | None
        :param max_in_flight: ignored, present for api parity
        :ptype max_in_flight: int | None
        :param deadletter_on_error: ignored, present for api parity
        :ptype deadletter_on_error: bool
        """
        del queue, max_in_flight, deadletter_on_error
        self._typed_subscribers.setdefault(subject.path, []).append((message_type, cb))

    async def get(self, bucket: str, key: str) -> bytes | None:
        """fetch bytes for key from bucket or None."""
        return self._kv.get(bucket, {}).get(key)

    async def put(self, bucket: str, key: str, value: bytes) -> bool:
        """store bytes under key in bucket."""
        self._kv.setdefault(bucket, {})[key] = value
        return True

    async def delete(self, bucket: str, key: str) -> bool:
        """remove key from bucket."""
        self._kv.get(bucket, {}).pop(key, None)
        return True


def _make_pool_mock() -> AsyncMock:
    """
    returns AsyncMock with recorded execute/fetchrow/fetch calls.

    :return: mock pool recording SQL statements
    :rtype: AsyncMock
    """
    store: dict[str, dict[str, Any]] = {}
    executed: list[tuple[str, tuple[Any, ...]]] = []

    async def _fetchrow(query: str, *args: Any) -> dict[str, Any] | None:
        entity_id = args[0] if args else None
        return store.get(str(entity_id))

    async def _execute(query: str, *args: Any) -> str:
        executed.append((query, args))
        if "INSERT" in query.upper():
            entity_id = str(args[0])
            store[entity_id] = {"id": entity_id}
            return "INSERT 0 1"
        if "DELETE" in query.upper():
            entity_id = str(args[0])
            store.pop(entity_id, None)
            return "DELETE 1"
        return "0"

    async def _fetch(query: str, *args: Any) -> list[dict[str, Any]]:
        return list(store.values())

    pool = AsyncMock()
    pool.fetchrow = AsyncMock(side_effect=_fetchrow)
    pool.execute = AsyncMock(side_effect=_execute)
    pool.fetch = AsyncMock(side_effect=_fetch)
    pool.store = store
    pool.executed = executed
    return pool


@pytest.fixture()
def config_always() -> DefaultCoreConfig:
    """return core config that flushes writes to L3 immediately."""
    return DefaultCoreConfig(collection_flush="ALWAYS", collection_flush_tables="")


@pytest.fixture()
def workspaces_l1() -> SQLiteBackend:
    """build SQLite L1 backend with workspaces schema."""
    backend = SQLiteBackend(db_name=f"test_ws_{uuid4().hex[:8]}")
    backend.initialize(_workspaces_metadata())
    yield backend
    backend.reset()


@pytest.fixture()
def workspace_files_l1() -> SQLiteBackend:
    """build SQLite L1 backend with workspace_files schema."""
    backend = SQLiteBackend(db_name=f"test_wsf_{uuid4().hex[:8]}")
    backend.initialize(_workspace_files_metadata())
    yield backend
    backend.reset()


@pytest.fixture()
def workspace_file_versions_l1() -> SQLiteBackend:
    """build SQLite L1 backend with workspace_file_versions schema."""
    backend = SQLiteBackend(db_name=f"test_wsfv_{uuid4().hex[:8]}")
    backend.initialize(_workspace_file_versions_metadata())
    yield backend
    backend.reset()


class TestWorkspaceCollectionSerialization:
    """tests for WorkspaceCollection serialize/deserialize round-trip."""

    def test_serialize_deserialize_roundtrip(
        self,
        workspaces_l1: SQLiteBackend,
        config_always: DefaultCoreConfig,
    ) -> None:
        """row containing UUID and datetime values survives JSON round-trip."""
        registry = CollectionRegistry()
        registry.configure(l1_backend=workspaces_l1)
        pool = _make_pool_mock()
        coll = WorkspaceCollection(registry, config_always, postgres_pool=pool)
        row = _make_workspace_row()
        row["date_created"] = datetime(2026, 1, 1, tzinfo=UTC)
        row["date_updated"] = datetime(2026, 1, 2, tzinfo=UTC)
        payload = coll.serialize(row)
        assert isinstance(payload, bytes)
        restored = coll.deserialize(payload)
        assert restored["id"] == row["id"]
        assert restored["agent_id"] == row["agent_id"]
        assert restored["created_by"] == row["created_by"]
        assert restored["name"] == row["name"]
        assert restored["current_version"] == 0
        assert restored["template_name"] is None
        assert isinstance(restored["date_created"], datetime)


class TestWorkspaceCollectionSave:
    """tests for WorkspaceCollection save path."""

    async def test_save_issues_upsert(
        self,
        workspaces_l1: SQLiteBackend,
        config_always: DefaultCoreConfig,
    ) -> None:
        """save_to_postgres uses INSERT ... ON CONFLICT (id) DO UPDATE."""
        registry = CollectionRegistry()
        registry.configure(l1_backend=workspaces_l1)
        pool = _make_pool_mock()
        coll = WorkspaceCollection(registry, config_always, postgres_pool=pool)
        row = _make_workspace_row()
        rows_affected = await coll.save_to_postgres(row)
        assert rows_affected == 1
        issued_sql = pool.executed[0][0]
        assert "INSERT INTO workspaces" in issued_sql
        assert "ON CONFLICT (id) DO UPDATE" in issued_sql

    async def test_delete_issues_delete_by_id(
        self,
        workspaces_l1: SQLiteBackend,
        config_always: DefaultCoreConfig,
    ) -> None:
        """delete_from_postgres issues DELETE WHERE id = $1."""
        registry = CollectionRegistry()
        registry.configure(l1_backend=workspaces_l1)
        pool = _make_pool_mock()
        coll = WorkspaceCollection(registry, config_always, postgres_pool=pool)
        target_id = uuid4()
        await coll.delete_from_postgres(target_id)
        issued_sql = pool.executed[0][0]
        assert "DELETE FROM workspaces" in issued_sql
        assert "WHERE id = $1" in issued_sql


class TestWorkspaceFileCollectionSave:
    """tests for WorkspaceFileCollection save and delete paths."""

    async def test_save_issues_upsert(
        self,
        workspace_files_l1: SQLiteBackend,
        config_always: DefaultCoreConfig,
    ) -> None:
        """save_to_postgres uses INSERT ... ON CONFLICT (id) DO UPDATE for head state."""
        registry = CollectionRegistry()
        registry.configure(l1_backend=workspace_files_l1)
        pool = _make_pool_mock()
        coll = WorkspaceFileCollection(registry, config_always, postgres_pool=pool)
        row = _make_workspace_file_row()
        rows_affected = await coll.save_to_postgres(row)
        assert rows_affected == 1
        issued_sql = pool.executed[0][0]
        assert "INSERT INTO workspace_files" in issued_sql
        assert "ON CONFLICT (id) DO UPDATE" in issued_sql

    async def test_delete_issues_delete_by_id(
        self,
        workspace_files_l1: SQLiteBackend,
        config_always: DefaultCoreConfig,
    ) -> None:
        """delete_from_postgres issues DELETE WHERE id = $1."""
        registry = CollectionRegistry()
        registry.configure(l1_backend=workspace_files_l1)
        pool = _make_pool_mock()
        coll = WorkspaceFileCollection(registry, config_always, postgres_pool=pool)
        target_id = uuid4()
        await coll.delete_from_postgres(target_id)
        issued_sql = pool.executed[0][0]
        assert "DELETE FROM workspace_files" in issued_sql

    def test_bytes_content_roundtrip(
        self,
        workspace_files_l1: SQLiteBackend,
        config_always: DefaultCoreConfig,
    ) -> None:
        """BYTEA content survives L2 JSON serialize/deserialize via base64."""
        registry = CollectionRegistry()
        registry.configure(l1_backend=workspace_files_l1)
        pool = _make_pool_mock()
        coll = WorkspaceFileCollection(registry, config_always, postgres_pool=pool)
        row = _make_workspace_file_row()
        row["content"] = b"\x00\x01\x02\xff payload bytes"
        payload = coll.serialize(row)
        # confirm wire format base64-encoded the bytes
        parsed = json.loads(payload)
        assert isinstance(parsed["content"], str)
        assert base64.b64decode(parsed["content"]) == row["content"]
        restored = coll.deserialize(payload)
        assert restored["content"] == row["content"]
        assert isinstance(restored["content"], bytes)


class TestWorkspaceFileVersionCollectionSave:
    """tests for WorkspaceFileVersionCollection append-only journal save path."""

    async def test_save_is_insert_only_no_upsert(
        self,
        workspace_file_versions_l1: SQLiteBackend,
        config_always: DefaultCoreConfig,
    ) -> None:
        """journal save uses plain INSERT with no ON CONFLICT clause."""
        registry = CollectionRegistry()
        registry.configure(l1_backend=workspace_file_versions_l1)
        pool = _make_pool_mock()
        coll = WorkspaceFileVersionCollection(registry, config_always, postgres_pool=pool)
        row = _make_workspace_file_version_row()
        rows_affected = await coll.save_to_postgres(row)
        assert rows_affected == 1
        issued_sql = pool.executed[0][0]
        assert "INSERT INTO workspace_file_versions" in issued_sql
        assert "ON CONFLICT" not in issued_sql.upper()

    async def test_duplicate_triple_propagates_error(
        self,
        workspace_file_versions_l1: SQLiteBackend,
        config_always: DefaultCoreConfig,
    ) -> None:
        """
        duplicate (workspace_id, relative_path, version) insert propagates
        the UNIQUE-violation error from the pool; journal is append-only.
        """
        registry = CollectionRegistry()
        registry.configure(l1_backend=workspace_file_versions_l1)

        class _UniqueViolation(Exception):
            """stand-in for asyncpg.exceptions.UniqueViolationError."""

        pool = _make_pool_mock()

        async def _raise(query: str, *args: Any) -> str:
            raise _UniqueViolation("duplicate key value violates unique constraint")

        pool.execute = AsyncMock(side_effect=_raise)
        coll = WorkspaceFileVersionCollection(registry, config_always, postgres_pool=pool)
        row = _make_workspace_file_version_row()
        with pytest.raises(_UniqueViolation):
            await coll.save_to_postgres(row)


class TestCrossPodInvalidation:
    """tests for cross-pod L1 invalidation via the NATS pub/sub path."""

    async def test_invalidation_evicts_peer_l1(
        self,
        config_always: DefaultCoreConfig,
    ) -> None:
        """
        two WorkspaceCollection instances on separate L1 backends share a
        fake NATS bus; publishing invalidation through registry 1 evicts
        the entity from registry 2's L1 backend, matching the production
        registry.py:124-146 subscriber behavior.
        """
        pool_a = _make_pool_mock()
        pool_b = _make_pool_mock()
        bus = _FakeNatsBus()

        l1_a = SQLiteBackend(db_name=f"podA_{uuid4().hex[:8]}")
        l1_a.initialize(_workspaces_metadata())
        l1_b = SQLiteBackend(db_name=f"podB_{uuid4().hex[:8]}")
        l1_b.initialize(_workspaces_metadata())

        registry_a = CollectionRegistry()
        registry_a.configure(l1_backend=l1_a)
        coll_a = WorkspaceCollection(registry_a, config_always, postgres_pool=pool_a, nats_client=bus)

        registry_b = CollectionRegistry()
        registry_b.configure(l1_backend=l1_b)
        # construction registers the collection with registry_b so the
        # invalidation subscriber can resolve the table back to a collection
        WorkspaceCollection(registry_b, config_always, postgres_pool=pool_b, nats_client=bus)

        # pod B subscribes to invalidation signals like a real pod would
        await registry_b.start_invalidation_listener(bus)

        # seed pod B's L1 with a row so we can observe eviction
        row = _make_workspace_row()
        entity_id = str(row["id"])
        l1_row = {k: str(v) if isinstance(v, UUID) else v for k, v in row.items()}
        l1_b.upsert("workspaces", l1_row, "id")
        assert l1_b.select_by_id("workspaces", entity_id, "id") is not None

        # publish through pod A's registry; fake bus dispatches to pod B's subscriber
        await registry_a.publish_invalidation(bus, coll_a.table_name, row["id"])

        # pod B's L1 entry is now gone
        assert l1_b.select_by_id("workspaces", entity_id, "id") is None

        l1_a.reset()
        l1_b.reset()


class TestCollectionShapes:
    """tests for collection identity, table name, and entity class metadata."""

    def test_workspace_collection_identity(
        self,
        workspaces_l1: SQLiteBackend,
        config_always: DefaultCoreConfig,
    ) -> None:
        """workspace collection exposes correct table and entity class."""
        registry = CollectionRegistry()
        registry.configure(l1_backend=workspaces_l1)
        pool = _make_pool_mock()
        coll = WorkspaceCollection(registry, config_always, postgres_pool=pool)
        assert coll.table_name == "workspaces"
        assert coll.entity_class is Workspace
        assert coll.primary_key_column == "id"

    def test_workspace_file_collection_identity(
        self,
        workspace_files_l1: SQLiteBackend,
        config_always: DefaultCoreConfig,
    ) -> None:
        """workspace_file collection exposes correct table and entity class."""
        registry = CollectionRegistry()
        registry.configure(l1_backend=workspace_files_l1)
        pool = _make_pool_mock()
        coll = WorkspaceFileCollection(registry, config_always, postgres_pool=pool)
        assert coll.table_name == "workspace_files"
        assert coll.entity_class is WorkspaceFile
        assert coll.primary_key_column == "id"

    def test_workspace_file_version_collection_identity(
        self,
        workspace_file_versions_l1: SQLiteBackend,
        config_always: DefaultCoreConfig,
    ) -> None:
        """workspace_file_version collection exposes correct table and entity class."""
        registry = CollectionRegistry()
        registry.configure(l1_backend=workspace_file_versions_l1)
        pool = _make_pool_mock()
        coll = WorkspaceFileVersionCollection(registry, config_always, postgres_pool=pool)
        assert coll.table_name == "workspace_file_versions"
        assert coll.entity_class is WorkspaceFileVersion
        assert coll.primary_key_column == "id"


class TestInvalidationSubjectConstant:
    """sanity check that the cache-invalidation subject from registry is importable."""

    def test_subject_constant_is_importable(self) -> None:
        """confirm cache-invalidate subject resolves to a non-empty string."""
        subject = Subjects.cache_invalidate()
        assert isinstance(subject.path, str)
        assert subject.path


class TestWorkspaceCollectionFindByAgent:
    """tests for WorkspaceCollection.find_by_agent and find_by_agent_and_name."""

    @staticmethod
    def _make_query_pool(rows: list[dict[str, Any]]) -> AsyncMock:
        """
        returns an AsyncMock pool that serves fetch and fetchrow from rows.

        :param rows: rows to return; fetch returns the whole list,
            fetchrow returns the first row that matches all named args
            beyond the query
        :ptype rows: list[dict[str, Any]]
        :return: configured asyncpg-style mock pool
        :rtype: AsyncMock
        """
        executed: list[tuple[str, tuple[Any, ...]]] = []

        async def _fetch(query: str, *args: Any) -> list[dict[str, Any]]:
            executed.append((query, args))
            agent_id = args[0]
            return [r for r in rows if r["agent_id"] == agent_id]

        async def _fetchrow(query: str, *args: Any) -> dict[str, Any] | None:
            executed.append((query, args))
            agent_id = args[0]
            name = args[1]
            for r in rows:
                if r["agent_id"] == agent_id and r["name"] == name:
                    return r
            return None

        pool = AsyncMock()
        pool.fetch = AsyncMock(side_effect=_fetch)
        pool.fetchrow = AsyncMock(side_effect=_fetchrow)
        pool.executed = executed
        return pool

    async def test_find_by_agent_returns_entities_for_agent(
        self,
        workspaces_l1: SQLiteBackend,
        config_always: DefaultCoreConfig,
    ) -> None:
        """
        find_by_agent emits SELECT WHERE agent_id = $1 with date_deleted IS NULL
        ORDER BY date_updated DESC and returns hydrated entities for the agent.
        """
        registry = CollectionRegistry()
        registry.configure(l1_backend=workspaces_l1)

        agent_id = uuid4()
        other_agent = uuid4()
        row1 = _make_workspace_row()
        row1["agent_id"] = agent_id
        row1["name"] = "alpha"
        row2 = _make_workspace_row()
        row2["agent_id"] = agent_id
        row2["name"] = "beta"
        row_other = _make_workspace_row()
        row_other["agent_id"] = other_agent
        pool = self._make_query_pool([row1, row2, row_other])

        coll = WorkspaceCollection(registry, config_always, postgres_pool=pool)

        entities = await coll.find_by_agent(agent_id)

        assert len(entities) == 2
        assert all(isinstance(e, Workspace) for e in entities)
        assert {e.name for e in entities} == {"alpha", "beta"}
        for e in entities:
            assert e.agent_id == agent_id
        issued_sql = pool.executed[0][0]
        assert "SELECT * FROM workspaces" in issued_sql
        assert "WHERE agent_id = $1" in issued_sql
        assert "date_deleted IS NULL" in issued_sql
        assert "ORDER BY date_updated DESC" in issued_sql

    async def test_find_by_agent_default_filters_soft_deleted(
        self,
        workspaces_l1: SQLiteBackend,
        config_always: DefaultCoreConfig,
    ) -> None:
        """soft-deleted rows are excluded from the default find_by_agent result."""
        registry = CollectionRegistry()
        registry.configure(l1_backend=workspaces_l1)

        agent_id = uuid4()
        live = _make_workspace_row()
        live["agent_id"] = agent_id
        live["name"] = "live"
        deleted = _make_workspace_row()
        deleted["agent_id"] = agent_id
        deleted["name"] = "ghost"
        deleted["date_deleted"] = datetime(2026, 4, 16, 9, 0, 0, tzinfo=UTC)

        # custom pool that mirrors WHERE date_deleted IS NULL filter when present
        executed: list[tuple[str, tuple[Any, ...]]] = []

        async def _fetch(query: str, *args: Any) -> list[dict[str, Any]]:
            executed.append((query, args))
            agent = args[0]
            rows = [r for r in [live, deleted] if r["agent_id"] == agent]
            if "date_deleted IS NULL" in query:
                rows = [r for r in rows if r.get("date_deleted") is None]
            return rows

        pool = AsyncMock()
        pool.fetch = AsyncMock(side_effect=_fetch)
        pool.executed = executed
        coll = WorkspaceCollection(registry, config_always, postgres_pool=pool)

        live_only = await coll.find_by_agent(agent_id)
        assert {e.name for e in live_only} == {"live"}

        all_rows = await coll.find_by_agent(agent_id, include_deleted=True)
        assert {e.name for e in all_rows} == {"live", "ghost"}

        # confirm we issued two distinct SQL shapes
        assert "date_deleted IS NULL" in executed[0][0]
        assert "date_deleted IS NULL" not in executed[1][0]

    async def test_find_by_agent_returns_empty_list_when_no_rows(
        self,
        workspaces_l1: SQLiteBackend,
        config_always: DefaultCoreConfig,
    ) -> None:
        """find_by_agent returns [] when the agent owns no workspaces."""
        registry = CollectionRegistry()
        registry.configure(l1_backend=workspaces_l1)

        agent_id = uuid4()
        pool = self._make_query_pool([])
        coll = WorkspaceCollection(registry, config_always, postgres_pool=pool)

        entities = await coll.find_by_agent(agent_id)

        assert entities == []

    async def test_find_by_agent_and_name_returns_match(
        self,
        workspaces_l1: SQLiteBackend,
        config_always: DefaultCoreConfig,
    ) -> None:
        """
        find_by_agent_and_name issues SELECT WHERE agent_id AND name and
        returns a single hydrated Workspace entity on hit.
        """
        registry = CollectionRegistry()
        registry.configure(l1_backend=workspaces_l1)

        agent_id = uuid4()
        row = _make_workspace_row()
        row["agent_id"] = agent_id
        row["name"] = "main"
        pool = self._make_query_pool([row])
        coll = WorkspaceCollection(registry, config_always, postgres_pool=pool)

        found = await coll.find_by_agent_and_name(agent_id, "main")

        assert found is not None
        assert isinstance(found, Workspace)
        assert found.name == "main"
        assert found.agent_id == agent_id
        issued_sql = pool.executed[0][0]
        assert "SELECT * FROM workspaces" in issued_sql
        assert "WHERE agent_id = $1 AND name = $2" in issued_sql

    async def test_find_by_agent_and_name_returns_none_when_missing(
        self,
        workspaces_l1: SQLiteBackend,
        config_always: DefaultCoreConfig,
    ) -> None:
        """find_by_agent_and_name returns None when no row matches."""
        registry = CollectionRegistry()
        registry.configure(l1_backend=workspaces_l1)

        agent_id = uuid4()
        pool = self._make_query_pool([])
        coll = WorkspaceCollection(registry, config_always, postgres_pool=pool)

        found = await coll.find_by_agent_and_name(agent_id, "missing")

        assert found is None

    async def test_find_by_agent_and_name_returns_soft_deleted_row(
        self,
        workspaces_l1: SQLiteBackend,
        config_always: DefaultCoreConfig,
    ) -> None:
        """find_by_agent_and_name returns soft-deleted rows; callers decide policy."""
        registry = CollectionRegistry()
        registry.configure(l1_backend=workspaces_l1)

        agent_id = uuid4()
        row = _make_workspace_row()
        row["agent_id"] = agent_id
        row["name"] = "ghost"
        row["date_deleted"] = datetime(2026, 4, 16, 9, 0, 0, tzinfo=UTC)
        pool = self._make_query_pool([row])
        coll = WorkspaceCollection(registry, config_always, postgres_pool=pool)

        found = await coll.find_by_agent_and_name(agent_id, "ghost")

        assert found is not None
        assert found.date_deleted is not None
        # confirm SQL shape lacks the date_deleted filter
        issued_sql = pool.executed[0][0]
        assert "date_deleted" not in issued_sql


class TestWorkspaceCollectionSaveIncludesDateDeleted:
    """tests confirming UPSERT round-trips the date_deleted column."""

    async def test_save_includes_date_deleted_in_insert_and_update(
        self,
        workspaces_l1: SQLiteBackend,
        config_always: DefaultCoreConfig,
    ) -> None:
        """INSERT column list and ON CONFLICT SET clause both include date_deleted."""
        registry = CollectionRegistry()
        registry.configure(l1_backend=workspaces_l1)
        pool = _make_pool_mock()
        coll = WorkspaceCollection(registry, config_always, postgres_pool=pool)
        row = _make_workspace_row()
        row["date_deleted"] = datetime(2026, 4, 16, 12, 0, 0, tzinfo=UTC)
        await coll.save_to_postgres(row)
        issued_sql = pool.executed[0][0]
        assert "date_deleted" in issued_sql
        # ON CONFLICT clause also wires the column
        assert "date_deleted = EXCLUDED.date_deleted" in issued_sql
        # confirm the row's date_deleted value got bound as naive UTC
        # (SchemaBackedCollection converts aware datetimes at the WRITE
        # boundary per CLAUDE.md's "YugabyteDB WRITE: Convert aware ->
        # naive for TIMESTAMP columns" rule)
        bound_args = pool.executed[0][1]
        assert row["date_deleted"].replace(tzinfo=None) in bound_args


class TestWorkspaceFileCollectionFindByWorkspace:
    """tests for WorkspaceFileCollection.find_by_workspace."""

    async def test_find_by_workspace_returns_files_for_workspace(
        self,
        workspace_files_l1: SQLiteBackend,
        config_always: DefaultCoreConfig,
    ) -> None:
        """find_by_workspace emits SELECT WHERE workspace_id = $1 and hydrates entities."""
        registry = CollectionRegistry()
        registry.configure(l1_backend=workspace_files_l1)

        workspace_id = uuid4()
        other_workspace = uuid4()
        row1 = _make_workspace_file_row()
        row1["workspace_id"] = workspace_id
        row1["relative_path"] = "a.md"
        row2 = _make_workspace_file_row()
        row2["workspace_id"] = workspace_id
        row2["relative_path"] = "b.md"
        row_other = _make_workspace_file_row()
        row_other["workspace_id"] = other_workspace
        rows = [row1, row2, row_other]

        executed: list[tuple[str, tuple[Any, ...]]] = []

        async def _fetch(query: str, *args: Any) -> list[dict[str, Any]]:
            executed.append((query, args))
            wid = args[0]
            return [r for r in rows if r["workspace_id"] == wid]

        pool = AsyncMock()
        pool.fetch = AsyncMock(side_effect=_fetch)
        pool.executed = executed
        coll = WorkspaceFileCollection(registry, config_always, postgres_pool=pool)

        entities = await coll.find_by_workspace(workspace_id)

        assert len(entities) == 2
        assert all(isinstance(e, WorkspaceFile) for e in entities)
        assert {e.relative_path for e in entities} == {"a.md", "b.md"}
        issued_sql = executed[0][0]
        assert "SELECT * FROM workspace_files" in issued_sql
        assert "WHERE workspace_id = $1" in issued_sql

    async def test_find_by_workspace_empty(
        self,
        workspace_files_l1: SQLiteBackend,
        config_always: DefaultCoreConfig,
    ) -> None:
        """find_by_workspace returns [] when no rows match."""
        registry = CollectionRegistry()
        registry.configure(l1_backend=workspace_files_l1)

        async def _fetch(query: str, *args: Any) -> list[dict[str, Any]]:
            return []

        pool = AsyncMock()
        pool.fetch = AsyncMock(side_effect=_fetch)
        coll = WorkspaceFileCollection(registry, config_always, postgres_pool=pool)

        entities = await coll.find_by_workspace(uuid4())
        assert entities == []


class TestWorkspaceFileVersionCollectionHistoryQueries:
    """tests for WorkspaceFileVersionCollection history-shaped queries."""

    async def test_find_by_workspace_orders_newest_first_with_limit(
        self,
        workspace_file_versions_l1: SQLiteBackend,
        config_always: DefaultCoreConfig,
    ) -> None:
        """find_by_workspace issues ORDER BY date_created DESC LIMIT $2 and hydrates rows."""
        registry = CollectionRegistry()
        registry.configure(l1_backend=workspace_file_versions_l1)

        workspace_id = uuid4()
        rows = [
            {**_make_workspace_file_version_row(), "workspace_id": workspace_id, "relative_path": "a.md", "version": 1},
            {**_make_workspace_file_version_row(), "workspace_id": workspace_id, "relative_path": "b.md", "version": 1},
        ]
        executed: list[tuple[str, tuple[Any, ...]]] = []

        async def _fetch(query: str, *args: Any) -> list[dict[str, Any]]:
            executed.append((query, args))
            wid = args[0]
            limit = args[1]
            return [r for r in rows if r["workspace_id"] == wid][:limit]

        pool = AsyncMock()
        pool.fetch = AsyncMock(side_effect=_fetch)
        coll = WorkspaceFileVersionCollection(registry, config_always, postgres_pool=pool)

        entities = await coll.find_by_workspace(workspace_id, 10)
        assert len(entities) == 2
        assert all(isinstance(e, WorkspaceFileVersion) for e in entities)
        issued_sql = executed[0][0]
        assert "SELECT * FROM workspace_file_versions" in issued_sql
        assert "WHERE workspace_id = $1" in issued_sql
        assert "ORDER BY date_created DESC" in issued_sql
        assert "LIMIT $2" in issued_sql
        assert executed[0][1] == (workspace_id, 10)

    async def test_find_by_workspace_and_path_narrows_to_single_path(
        self,
        workspace_file_versions_l1: SQLiteBackend,
        config_always: DefaultCoreConfig,
    ) -> None:
        """find_by_workspace_and_path emits SQL with AND relative_path = $2 and hydrates rows."""
        registry = CollectionRegistry()
        registry.configure(l1_backend=workspace_file_versions_l1)

        workspace_id = uuid4()
        rows = [
            {**_make_workspace_file_version_row(), "workspace_id": workspace_id, "relative_path": "a.md", "version": 1},
            {**_make_workspace_file_version_row(), "workspace_id": workspace_id, "relative_path": "b.md", "version": 1},
        ]
        executed: list[tuple[str, tuple[Any, ...]]] = []

        async def _fetch(query: str, *args: Any) -> list[dict[str, Any]]:
            executed.append((query, args))
            wid, path, limit = args
            return [r for r in rows if r["workspace_id"] == wid and r["relative_path"] == path][:limit]

        pool = AsyncMock()
        pool.fetch = AsyncMock(side_effect=_fetch)
        coll = WorkspaceFileVersionCollection(registry, config_always, postgres_pool=pool)

        entities = await coll.find_by_workspace_and_path(workspace_id, "a.md", 25)
        assert len(entities) == 1
        assert entities[0].relative_path == "a.md"
        issued_sql = executed[0][0]
        assert "AND relative_path = $2" in issued_sql
        assert "ORDER BY date_created DESC" in issued_sql
        assert "LIMIT $3" in issued_sql
        assert executed[0][1] == (workspace_id, "a.md", 25)
