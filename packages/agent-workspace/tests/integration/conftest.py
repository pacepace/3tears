"""integration-test fixtures: fake NATS + in-memory DB pool for workspace tools.

REALISM / LIMITS
================

the workspace integration tests in this directory wire together the
REAL :class:`threetears.agent.workspace.tools.*` tool classes with:

- **fake NATS KV**: :class:`_fake_kv.FakeNatsClient` from the core unit
  tests. the CAS semantics match real nats-py :class:`KeyValue` -- same
  fake :class:`WorkspaceFileLease` uses in ``test_bind_lease_race.py``.
  :meth:`publish` is added here so audit envelopes are captured in-
  memory for cross-component assertions.

- **fake asyncpg pool**: :class:`_FakePool` below, a minimal in-memory
  PostgreSQL-shaped store that understands the specific SQL statements
  :func:`_write_file_atomic`, :func:`_capture_back`, :func:`materialize`,
  and the lifecycle tools issue. it is NOT a general-purpose SQL engine
  -- it pattern-matches on statement prefix and carries the three tables
  (``workspaces`` / ``workspace_files`` / ``workspace_file_versions``)
  as plain dicts keyed by the natural keys used in those statements.
  this is the same trade-off the unit-test fakes use in ``tests/unit/``;
  upgrading to real YugabyteDB testcontainers is tracked in the README.

- **real WorkspaceFileLease + real KVLease**: via the fake NATS KV. the
  cross-pod serialization test is exercising the real lease wrapper,
  which in turn exercises the real KVLease CAS semantics.

- **real WorkspaceSandbox** built from :class:`WorkspaceConfig`.

THE REAL-VS-FAKE TRADE-OFF IS CALLED OUT EXPLICITLY in every integration
test file's module docstring so readers know which layer is being
exercised for real and which is being simulated.

deferred to the aibots-repo integration suite
--------------------------------------------

- REAL NATS JetStream (testcontainers): the lease contract is already
  exercised against real NATS in upstream test suites; the CAS
  semantics of the fake are intentionally equivalent.
- REAL YugabyteDB (testcontainers) with the migration-built schema: the
  unit-test fake pool pattern-matches the SQL the production code runs;
  round-tripping through real YB adds confidence that nothing depends
  on behaviours the fake doesn't model. see tests/integration/README.md.
- Hub-side :class:`UnifiedAuditConsumer` landing rows in the real
  ``platform_audit.audit_events`` table (audit-task-01 Phase 3 retired
  the per-domain ``WorkspaceAuditConsumer``; the unified consumer owns
  the whole ``{ns}.audit.>`` subtree). that consumer lives in the
  aibots repo; here we assert the agent side publishes the canonical
  envelope and an in-process stub consumer lands it into a stub
  :class:`AuditEventCollection`.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, AsyncIterator
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4, uuid7

import pytest

from _fake_kv import FakeNatsClient as _FakeNatsKVClient  # type: ignore[import-not-found]

from threetears.agent.tools.call_scope import (
    ToolCallScope,
    enter_call_scope,
)
from threetears.agent.tools.context_envelope import CallContext


# ---------------------------------------------------------------------------
# in-memory asyncpg-shaped store
# ---------------------------------------------------------------------------


@dataclass
class _StoredFile:
    """head-state row shape."""

    id: UUID
    workspace_id: UUID
    relative_path: str
    content: bytes
    sha256: str
    version: int
    date_updated: datetime


@dataclass
class _StoredVersion:
    """journal row shape."""

    id: UUID
    workspace_id: UUID
    relative_path: str
    version: int
    content: bytes
    sha256: str
    action: str
    label: str | None
    actor_id: UUID
    correlation_id: UUID
    date_created: datetime


@dataclass
class _StoredWorkspace:
    """workspaces row shape."""

    id: UUID
    agent_id: UUID
    name: str
    description: str | None
    template_name: str | None
    created_by: UUID
    current_version: int
    date_created: datetime
    date_updated: datetime
    date_deleted: datetime | None
    # WS-ACL-10 audit identity: customer_id mirrors the stamped value
    # from platform.namespaces for integration tests that need the
    # audit envelope's five-UUID tuple.
    customer_id: UUID | None = None

    @property
    def owner_agent_id(self) -> UUID:
        """return owning agent UUID (alias of :attr:`agent_id`)."""
        return self.agent_id

    @property
    def namespace_name(self) -> str:
        """canonical workspace namespace name (WS-ACL-06)."""
        return f"workspace.{self.id}"


class _FakeStore:
    """
    process-local, dict-backed stand-in for the workspace schema tables.

    pattern-matches on SQL statement prefixes. every statement used by
    the production code paths exercised in these integration tests is
    handled explicitly; unrecognized statements raise
    :class:`NotImplementedError` so a drift in production SQL fails
    loudly rather than silently accepting no-op inserts.
    """

    def __init__(self) -> None:
        self.workspaces: dict[UUID, _StoredWorkspace] = {}
        self.files: dict[tuple[UUID, str], _StoredFile] = {}
        self.versions: list[_StoredVersion] = []
        self.statements_executed: list[tuple[str, tuple[Any, ...]]] = []

    async def execute(self, query: str, *args: Any) -> str:
        """
        route a SQL statement to its handler by prefix match.

        :param query: SQL text (may be multi-line)
        :ptype query: str
        :param args: bound parameters, 1-indexed by placeholder
        :ptype args: Any
        :return: asyncpg-style status tag (ignored by the helpers)
        :rtype: str
        """
        self.statements_executed.append((query, args))
        q = " ".join(query.split())  # collapse whitespace for prefix match
        if q.startswith("INSERT INTO workspaces"):
            self._handle_insert_workspace(args)
        elif q.startswith("INSERT INTO workspace_file_versions"):
            self._handle_insert_version(args)
        elif q.startswith("INSERT INTO workspace_files"):
            self._handle_upsert_file(args)
        elif q.startswith("UPDATE workspaces"):
            self._handle_update_workspace(args)
        elif q.startswith("DELETE FROM workspace_files"):
            self._handle_delete_file(args)
        else:
            raise NotImplementedError(f"_FakeStore does not recognize SQL: {q[:120]!r}")
        return "OK 0 1"

    async def fetchrow(self, query: str, *args: Any) -> dict[str, Any] | None:
        """
        route SELECT statements to the right lookup.

        :param query: SQL text
        :ptype query: str
        :param args: bound parameters
        :ptype args: Any
        :return: dict-shaped row or None (matches asyncpg.Record.__getitem__
            and dict conversion used by :func:`_resolve_ref`)
        :rtype: dict[str, Any] | None
        """
        q = " ".join(query.split())
        if q.startswith("SELECT content, sha256, version") and "FROM workspace_files" in q:
            workspace_id: UUID = args[0]
            relative_path: str = args[1]
            row = self.files.get((workspace_id, relative_path))
            if row is None:
                return None
            return {
                "content": row.content,
                "sha256": row.sha256,
                "version": row.version,
            }
        if "COALESCE(MAX(version)" in q and "FROM workspace_file_versions" in q:
            workspace_id_j: UUID = args[0]
            relative_path_j: str = args[1]
            matches_j = [
                v for v in self.versions if v.workspace_id == workspace_id_j and v.relative_path == relative_path_j
            ]
            max_version = max((v.version for v in matches_j), default=0)
            return {"max_version": max_version}
        if "FROM workspace_file_versions" in q:
            # simplified resolver for _resolve_ref variants exercised by
            # the rollback integration; the full set of SELECTs appears
            # in helpers.py and is covered by unit tests.
            return self._handle_select_version(q, args)
        raise NotImplementedError(f"_FakeStore does not recognize SELECT: {q[:120]!r}")

    def _handle_insert_workspace(self, args: tuple[Any, ...]) -> None:
        (
            ws_id,
            agent_id,
            name,
            description,
            template_name,
            created_by,
            current_version,
            date_created,
            date_updated,
        ) = args
        self.workspaces[ws_id] = _StoredWorkspace(
            id=ws_id,
            agent_id=agent_id,
            name=name,
            description=description,
            template_name=template_name,
            created_by=created_by,
            current_version=current_version,
            date_created=date_created,
            date_updated=date_updated,
            date_deleted=None,
        )

    def _handle_insert_version(self, args: tuple[Any, ...]) -> None:
        (
            version_id,
            workspace_id,
            relative_path,
            version,
            content,
            sha256,
            action,
            label,
            actor_id,
            correlation_id,
            date_created,
        ) = args
        self.versions.append(
            _StoredVersion(
                id=version_id,
                workspace_id=workspace_id,
                relative_path=relative_path,
                version=version,
                content=content,
                sha256=sha256,
                action=action,
                label=label,
                actor_id=actor_id,
                correlation_id=correlation_id,
                date_created=date_created,
            )
        )

    def _handle_upsert_file(self, args: tuple[Any, ...]) -> None:
        (
            file_id,
            workspace_id,
            relative_path,
            content,
            sha256,
            version,
            date_updated,
        ) = args
        existing = self.files.get((workspace_id, relative_path))
        if existing is not None:
            existing.content = content
            existing.sha256 = sha256
            existing.version = version
            existing.date_updated = date_updated
        else:
            self.files[(workspace_id, relative_path)] = _StoredFile(
                id=file_id,
                workspace_id=workspace_id,
                relative_path=relative_path,
                content=content,
                sha256=sha256,
                version=version,
                date_updated=date_updated,
            )

    def _handle_update_workspace(self, args: tuple[Any, ...]) -> None:
        new_version, date_updated, workspace_id, agent_id = args
        ws = self.workspaces.get(workspace_id)
        if ws is not None and ws.agent_id == agent_id:
            ws.current_version = max(ws.current_version, new_version)
            ws.date_updated = date_updated

    def _handle_delete_file(self, args: tuple[Any, ...]) -> None:
        workspace_id, relative_path = args
        self.files.pop((workspace_id, relative_path), None)

    def _handle_select_version(self, query: str, args: tuple[Any, ...]) -> dict[str, Any] | None:
        """best-effort emulation of the _resolve_ref SELECTs."""
        # only the "ORDER BY version DESC LIMIT 1" case is exercised
        # today (ref='head') plus exact-version lookup.
        if "ORDER BY version DESC LIMIT 1" in query and "action = 'checkpoint'" not in query:
            workspace_id, relative_path = args[0], args[1]
            matches = [v for v in self.versions if v.workspace_id == workspace_id and v.relative_path == relative_path]
            if not matches:
                return None
            latest = max(matches, key=lambda v: v.version)
            return _version_to_dict(latest)
        if re.search(r"AND version\s*=\s*\$3", query):
            workspace_id, relative_path, wanted = args[0], args[1], args[2]
            for v in self.versions:
                if v.workspace_id == workspace_id and v.relative_path == relative_path and v.version == wanted:
                    return _version_to_dict(v)
            return None
        if "action = 'checkpoint'" in query:
            workspace_id, relative_path, label = args[0], args[1], args[2]
            matches = [
                v
                for v in self.versions
                if v.workspace_id == workspace_id
                and v.relative_path == relative_path
                and v.action == "checkpoint"
                and v.label == label
            ]
            if not matches:
                return None
            latest = max(matches, key=lambda v: v.version)
            return _version_to_dict(latest)
        return None


def _version_to_dict(v: _StoredVersion) -> dict[str, Any]:
    """marshal a stored journal row into the dict shape callers expect."""
    return {
        "id": v.id,
        "workspace_id": v.workspace_id,
        "relative_path": v.relative_path,
        "version": v.version,
        "content": v.content,
        "sha256": v.sha256,
        "action": v.action,
        "label": v.label,
        "actor_id": v.actor_id,
        "correlation_id": v.correlation_id,
        "date_created": v.date_created,
    }


@dataclass
class _FakeTransaction:
    """context-manager placeholder; the fake store is single-process."""

    parent: _FakeConnection

    async def __aenter__(self) -> _FakeTransaction:
        self.parent.transaction_open = True
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.parent.transaction_open = False


@dataclass
class _FakeConnection:
    """asyncpg-shaped connection routed to the shared :class:`_FakeStore`."""

    store: _FakeStore
    transaction_open: bool = False

    def transaction(self, namespace: Any = None) -> _FakeTransaction:
        return _FakeTransaction(parent=self)

    async def execute(self, query: str, *args: Any) -> str:
        return await self.store.execute(query, *args)

    async def fetchrow(self, query: str, *args: Any) -> dict[str, Any] | None:
        return await self.store.fetchrow(query, *args)


@dataclass
class _FakeAcquireCM:
    conn: _FakeConnection

    async def __aenter__(self) -> _FakeConnection:
        return self.conn

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        return None


@dataclass
class _FakePool:
    """asyncpg-shaped pool sharing a single backing store across connections."""

    store: _FakeStore = field(default_factory=_FakeStore)

    def acquire(self) -> _FakeAcquireCM:
        return _FakeAcquireCM(conn=_FakeConnection(store=self.store))


# ---------------------------------------------------------------------------
# collection-shaped helpers over the fake store
# ---------------------------------------------------------------------------


@dataclass
class _StoreBackedFile:
    """
    entity-shaped mirror of :class:`_StoredFile` the tools' code paths
    read via attribute access.
    """

    relative_path: str
    content: bytes
    sha256: str
    version: int


class _StoreBackedWorkspaceCollection:
    """
    collection stand-in serving :class:`_StoredWorkspace` rows as the
    production :class:`Workspace` entity's attribute surface.
    """

    def __init__(self, store: _FakeStore) -> None:
        self._store = store

    async def find_by_agent_and_name(self, agent_id: UUID, name: str) -> _StoredWorkspace | None:
        for ws in self._store.workspaces.values():
            if ws.agent_id == agent_id and ws.name == name:
                return ws
        return None

    async def find_by_id_and_agent(self, workspace_id: UUID, agent_id: UUID) -> _StoredWorkspace | None:
        ws = self._store.workspaces.get(workspace_id)
        if ws is None or ws.agent_id != agent_id:
            return None
        return ws

    async def find_by_id(
        self, agent_id: UUID, workspace_id: UUID,
    ) -> _StoredWorkspace | None:
        ws = self._store.workspaces.get(workspace_id)
        if ws is None or ws.agent_id != agent_id:
            return None
        return ws


class _StoreBackedFileCollection:
    """collection stand-in for head-state file rows."""

    def __init__(self, store: _FakeStore) -> None:
        self._store = store

    async def find_by_workspace_and_relative_path(
        self, workspace_id: UUID, relative_path: str
    ) -> _StoreBackedFile | None:
        row = self._store.files.get((workspace_id, relative_path))
        if row is None:
            return None
        return _StoreBackedFile(
            relative_path=row.relative_path,
            content=row.content,
            sha256=row.sha256,
            version=row.version,
        )

    async def find_by_workspace(self, workspace_id: UUID) -> list[_StoreBackedFile]:
        return [
            _StoreBackedFile(
                relative_path=row.relative_path,
                content=row.content,
                sha256=row.sha256,
                version=row.version,
            )
            for (ws_id, _path), row in self._store.files.items()
            if ws_id == workspace_id
        ]


class _StoreBackedVersionCollection:
    """collection stand-in for journal rows (read-only helper surface)."""

    def __init__(self, store: _FakeStore) -> None:
        self._store = store

    async def find_by_workspace(self, workspace_id: UUID) -> list[_StoredVersion]:
        return [v for v in self._store.versions if v.workspace_id == workspace_id]


# ---------------------------------------------------------------------------
# fake NATS client with publish recording
# ---------------------------------------------------------------------------


class RecordingFakeNatsClient(_FakeNatsKVClient):  # type: ignore[misc]
    """
    fake NATS client with publish recording on top of the core fake KV.

    the core fake covers the JetStream + KV surface :class:`KVLease`
    depends on; this subclass adds :meth:`publish` so audit envelopes
    emitted by :func:`threetears.agent.audit.publish_audit` are
    captured for assertion. subscriptions are handled in-process:
    :meth:`subscribe` registers a coroutine that :meth:`publish`
    awaits for any matching subject prefix.
    """

    def __init__(self) -> None:
        super().__init__()
        self.published: list[tuple[str, bytes]] = []
        self._subscriptions: list[tuple[str, Any]] = []

    async def publish(self, subject: str, payload: bytes) -> None:
        """record publish, dispatch to any matching in-process subscriber.

        :param subject: NATS subject string
        :ptype subject: str
        :param payload: payload bytes
        :ptype payload: bytes
        :return: None
        :rtype: None
        """
        self.published.append((subject, payload))
        for prefix, handler in self._subscriptions:
            if _subject_matches(subject, prefix):
                await handler(_FakeMsg(subject=subject, data=payload))

    def register_subscription(self, subject_prefix: str, handler: Any) -> None:
        """
        in-process subscription used by the audit-consumer stub.

        :param subject_prefix: canonical prefix up to the first wildcard
        :ptype subject_prefix: str
        :param handler: async callable ``(msg) -> None``
        :ptype handler: Any
        :return: None
        :rtype: None
        """
        self._subscriptions.append((subject_prefix, handler))


def _subject_matches(subject: str, prefix: str) -> bool:
    """trivial prefix match; tests use ``foo.audit.workspace`` prefixes."""
    return subject.startswith(prefix)


@dataclass
class _FakeMsg:
    """minimal subset of nats-py ``Msg`` used by handler shims."""

    subject: str
    data: bytes


# ---------------------------------------------------------------------------
# fake context + pin collection
# ---------------------------------------------------------------------------


class _InMemoryContextCollection:
    """
    lightweight :class:`ContextItemCollection` stand-in for pin storage.

    the production :mod:`threetears.agent.workspace.pin` module writes
    through :attr:`context._collection`; this stand-in captures
    ``save_entity`` and ``delete`` without reaching any real backend.
    """

    def __init__(self) -> None:
        self.saved: list[Any] = []
        self.deleted: list[Any] = []

    class _EntityAdapter:
        """entity-shape shim the production code constructs around pin dicts."""

        def __init__(
            self,
            data: dict[str, Any],
            is_new: bool,
            collection: Any,
        ) -> None:
            self._data = data
            self._is_new = is_new
            self._collection = collection

    @property
    def entity_class(self) -> Any:
        return _InMemoryContextCollection._EntityAdapter

    async def save_entity(self, entity: Any) -> None:
        self.saved.append(entity)

    async def delete(self, entity_id: Any) -> None:
        self.deleted.append(entity_id)


class _FakeToolContextManager:
    """
    duck-typed stand-in for :class:`ToolContextManager`.

    the pin module calls through the public ``*_item_by_type_and_key``
    API; this stand-in implements those three methods over an in-memory
    list so tests do not need a real collection or L3.
    """

    def __init__(self, conversation_id: UUID) -> None:
        self.conversation_id = conversation_id
        self._items: list[dict[str, Any]] = []
        self._collection = _InMemoryContextCollection()

    async def get_item_by_type_and_key(
        self,
        context_type: str,
        key: str,
    ) -> dict[str, Any] | None:
        result: dict[str, Any] | None = None
        for item in self._items:
            if item["context_type"] == context_type and item["key"] == key:
                result = item
                break
        return result

    async def save_item_by_type_and_key(
        self,
        *,
        context_type: str,
        key: str,
        content: str,
        short_desc: str = "",
        long_desc: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        from datetime import UTC, datetime

        existing = await self.get_item_by_type_and_key(context_type, key)
        if existing is not None:
            self._items = [i for i in self._items if i is not existing]
            await self._collection.delete(existing["context_id"])
        now = datetime.now(UTC)
        context_id = uuid4()
        data: dict[str, Any] = {
            "context_id": context_id,
            "conversation_id": self.conversation_id,
            "context_type": context_type,
            "key": key,
            "short_desc": short_desc,
            "long_desc": long_desc,
            "content": content,
            "metadata": metadata or {},
            "date_accessed": now,
            "date_created": now,
            "date_updated": now,
        }
        await self._collection.save_entity(
            self._collection.entity_class(data, is_new=True, collection=self._collection)
        )
        self._items.append(data)
        return str(context_id)

    async def delete_item_by_type_and_key(
        self,
        context_type: str,
        key: str,
    ) -> bool:
        existing = await self.get_item_by_type_and_key(context_type, key)
        result: bool
        if existing is None:
            result = False
        else:
            self._items = [i for i in self._items if i is not existing]
            await self._collection.delete(existing["context_id"])
            result = True
        return result


# ---------------------------------------------------------------------------
# pytest fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_nats_client() -> RecordingFakeNatsClient:
    """fresh recording-capable fake NATS client per test.

    :return: empty fake NATS client with publish recording
    :rtype: RecordingFakeNatsClient
    """
    return RecordingFakeNatsClient()


@pytest.fixture
def fake_store() -> _FakeStore:
    """fresh in-memory store per test.

    :return: empty _FakeStore
    :rtype: _FakeStore
    """
    return _FakeStore()


@pytest.fixture
def fake_db_pool(fake_store: _FakeStore) -> _FakePool:
    """fresh pool sharing the current test's store.

    :param fake_store: session-less backing store
    :ptype fake_store: _FakeStore
    :return: fake pool routed to the store
    :rtype: _FakePool
    """
    return _FakePool(store=fake_store)


@pytest.fixture
def fake_workspace_collection(
    fake_store: _FakeStore,
) -> _StoreBackedWorkspaceCollection:
    """workspace collection routed to the shared store.

    :param fake_store: shared in-memory store
    :ptype fake_store: _FakeStore
    :return: collection wrapper over the store
    :rtype: _StoreBackedWorkspaceCollection
    """
    return _StoreBackedWorkspaceCollection(fake_store)


@pytest.fixture
def fake_file_collection(
    fake_store: _FakeStore,
) -> _StoreBackedFileCollection:
    """head-state file collection routed to the shared store.

    :param fake_store: shared in-memory store
    :ptype fake_store: _FakeStore
    :return: collection wrapper over the store
    :rtype: _StoreBackedFileCollection
    """
    return _StoreBackedFileCollection(fake_store)


@pytest.fixture
def fake_version_collection(
    fake_store: _FakeStore,
) -> _StoreBackedVersionCollection:
    """journal collection routed to the shared store.

    :param fake_store: shared in-memory store
    :ptype fake_store: _FakeStore
    :return: collection wrapper over the store
    :rtype: _StoreBackedVersionCollection
    """
    return _StoreBackedVersionCollection(fake_store)


@pytest.fixture
def fake_tool_context() -> _FakeToolContextManager:
    """fresh tool context with in-memory pin collection.

    :return: context manager with conversation id
    :rtype: _FakeToolContextManager
    """
    return _FakeToolContextManager(conversation_id=uuid4())


@dataclass
class WorkspaceFixture:
    """
    bag of wired pieces an integration test needs to drive the tool
    surface against the shared in-memory store.
    """

    agent_id: UUID
    workspace_id: UUID
    workspace_name: str
    store: _FakeStore
    pool: _FakePool
    nats: RecordingFakeNatsClient
    context: _FakeToolContextManager
    workspace_collection: _StoreBackedWorkspaceCollection
    file_collection: _StoreBackedFileCollection
    version_collection: _StoreBackedVersionCollection
    fixture_path: Any


def _seed_workspace(
    store: _FakeStore,
    agent_id: UUID,
    name: str,
    *,
    customer_id: UUID | None = None,
) -> UUID:
    """
    insert a live workspace row into the store; return its id.

    namespace-task-01 phase 7 plumbed the path-level rbac gate into
    every write-class tool; the gate hard-denies cross-customer
    access, so fixtures that drive those tools must stamp the
    workspace with the SAME ``customer_id`` the calling
    :class:`ToolCallScope` carries. callers pass the scope's
    customer_id explicitly; ``None`` falls back to an independent
    ``uuid7()`` for fixtures that deliberately test cross-customer
    denial.

    :param store: shared in-memory store
    :ptype store: _FakeStore
    :param agent_id: owning agent
    :ptype agent_id: UUID
    :param name: workspace name
    :ptype name: str
    :param customer_id: customer the workspace belongs to; aligns
        with the scope's ``customer_id`` for same-customer fixtures
    :ptype customer_id: UUID | None
    :return: new workspace id
    :rtype: UUID
    """
    ws_id = uuid7()
    now = datetime.now(UTC)
    ws_customer_id = customer_id if customer_id is not None else uuid7()
    store.workspaces[ws_id] = _StoredWorkspace(
        id=ws_id,
        agent_id=agent_id,
        name=name,
        description=None,
        template_name=None,
        created_by=agent_id,
        current_version=0,
        date_created=now,
        date_updated=now,
        date_deleted=None,
        customer_id=ws_customer_id,
    )
    return ws_id


def _seed_file(
    store: _FakeStore,
    workspace_id: UUID,
    relative_path: str,
    content: bytes,
    actor_id: UUID,
) -> None:
    """
    seed a file head row plus its initial create-journal row.

    :param store: shared store
    :ptype store: _FakeStore
    :param workspace_id: target workspace
    :ptype workspace_id: UUID
    :param relative_path: file path
    :ptype relative_path: str
    :param content: bytes to seed
    :ptype content: bytes
    :param actor_id: actor to stamp on the create row
    :ptype actor_id: UUID
    :return: None
    :rtype: None
    """
    sha = hashlib.sha256(content).hexdigest()
    now = datetime.now(UTC)
    store.files[(workspace_id, relative_path)] = _StoredFile(
        id=uuid7(),
        workspace_id=workspace_id,
        relative_path=relative_path,
        content=content,
        sha256=sha,
        version=1,
        date_updated=now,
    )
    store.versions.append(
        _StoredVersion(
            id=uuid7(),
            workspace_id=workspace_id,
            relative_path=relative_path,
            version=1,
            content=content,
            sha256=sha,
            action="create",
            label=None,
            actor_id=actor_id,
            correlation_id=uuid7(),
            date_created=now,
        )
    )


@pytest.fixture
def workspace_with_audience_fixture(
    fake_store: _FakeStore,
    fake_db_pool: _FakePool,
    fake_nats_client: RecordingFakeNatsClient,
    fake_workspace_collection: _StoreBackedWorkspaceCollection,
    fake_file_collection: _StoreBackedFileCollection,
    fake_version_collection: _StoreBackedVersionCollection,
    fake_tool_context: _FakeToolContextManager,
    integration_tool_scope_context: CallContext,
) -> WorkspaceFixture:
    """
    seed a workspace with the three audience_test YAML fixtures.

    inserts each fixture file as a version-1 head row and a matching
    create-action journal row so subsequent writes observe them as
    existing files (and fs_write goes down the ``update`` branch).
    the workspace's ``customer_id`` aligns with the autouse
    :class:`ToolCallScope`'s ``customer_id`` so the phase-7
    cross-customer write gate permits same-customer writes.

    :return: fixture bag
    :rtype: WorkspaceFixture
    """
    from pathlib import Path

    agent_id = uuid4()
    workspace_name = "audience_test"
    workspace_id = _seed_workspace(
        fake_store,
        agent_id,
        workspace_name,
        customer_id=integration_tool_scope_context.customer_id,
    )

    fixture_dir = Path(__file__).resolve().parent / "fixtures" / "audience_test"
    assert fixture_dir.is_dir(), f"fixture directory missing: {fixture_dir}"
    for name in (
        "audience_settings.yaml",
        "linkedin_audience_units.yaml",
        "standard_audience_units.yaml",
    ):
        path = fixture_dir / name
        assert path.is_file(), f"fixture file missing: {path}"
        _seed_file(
            fake_store,
            workspace_id,
            name,
            path.read_bytes(),
            agent_id,
        )

    return WorkspaceFixture(
        agent_id=agent_id,
        workspace_id=workspace_id,
        workspace_name=workspace_name,
        store=fake_store,
        pool=fake_db_pool,
        nats=fake_nats_client,
        context=fake_tool_context,
        workspace_collection=fake_workspace_collection,
        file_collection=fake_file_collection,
        version_collection=fake_version_collection,
        fixture_path=fixture_dir,
    )


# ---------------------------------------------------------------------------
# scope + acl_cache fixtures (post WS-ACL-05 hard-fail)
# ---------------------------------------------------------------------------


@pytest.fixture
def integration_tool_scope_context() -> CallContext:
    """test-scoped CallContext with non-None identity dims for integration tests.

    integration tests that exercise tool dispatch outside an explicit
    :func:`enter_call_scope` block rely on the :func:`integration_tool_call_scope`
    autouse fixture below; this builder lets a test override identity by
    overriding this fixture.

    :return: call context with all identity fields populated
    :rtype: CallContext
    """
    return CallContext(
        agent_id=uuid7(),
        customer_id=uuid7(),
        user_id=uuid7(),
        conversation_id=uuid7(),
        correlation_id=uuid7(),
    )


@pytest.fixture(autouse=True)
async def integration_tool_call_scope(
    integration_tool_scope_context: CallContext,
) -> AsyncIterator[ToolCallScope]:
    """install a default :class:`ToolCallScope` for every integration test.

    autouse so tests that drive tool dispatch directly do not have to
    opt in. tests that need bespoke identity (e.g. cross-customer
    matrix) wrap their own :func:`enter_call_scope` blocks; nested
    scopes shadow the autouse one for the duration of the inner block.

    :param integration_tool_scope_context: identity envelope for the scope
    :ptype integration_tool_scope_context: CallContext
    :return: async iterator yielding the installed scope
    :rtype: AsyncIterator[ToolCallScope]
    """
    scope = ToolCallScope(context=integration_tool_scope_context)
    async with enter_call_scope(scope):
        yield scope


@pytest.fixture
def permissive_acl_cache() -> MagicMock:
    """AclCache-shaped mock returning ``"write"`` on every access check.

    integration tests that don't explicitly exercise the RBAC grant
    decision pass this cache to tool constructors so the post-WS-ACL-05
    hard-fail in :func:`authorize_workspace` doesn't reject the
    construction. tests that DO exercise the cache (cross-customer,
    grant matrix) build their own cache with the desired behavior.

    :return: mock with an :class:`AsyncMock` ``check_access`` returning
        ``"write"``
    :rtype: MagicMock
    """
    cache = MagicMock()
    cache.check_access = AsyncMock(return_value="write")
    return cache


def _is_real_authorize_test(request: pytest.FixtureRequest) -> bool:
    """return True when the requesting test wants the real authorize path.

    the cross-agent integration test exercises the real ACL grant
    decision end-to-end against a live PostgreSQL container, so the
    autouse stubs in this conftest must not patch it out for that
    file.

    :param request: pytest fixture request from the autouse fixture
    :ptype request: pytest.FixtureRequest
    :return: True when this test runs against the real authorize path
    :rtype: bool
    """
    nodeid = getattr(request.node, "nodeid", "") or ""
    return "test_cross_agent_workspace" in nodeid


@pytest.fixture(autouse=True)
def integration_stub_authorize_workspace_access(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncMock | None:
    """no-op stub for :func:`authorize_workspace_access` in integration tests.

    integration tests use lightweight workspace entities that do not
    expose the full :class:`WorkspaceLike` protocol surface
    (``namespace_name`` / ``owner_agent_id`` / ``created_by_user_id``).
    the workspace-shape-dependent grant decision is exercised end-to-end
    in ``tests/integration/test_cross_agent_workspace.py``; here we mock
    the inner call so other integration tests focus on tool behavior.
    the outer :func:`authorize_workspace` helper still enforces both
    preconditions (scope installed, ``acl_cache`` injected).

    namespace-task-01 phase 7 added a sibling
    :func:`authorize_workspace_file_access` that runs the path-glob
    RBAC gate on every write-class tool; it hits
    ``evaluate_file_access`` which in turn reaches for
    ``acl_cache.membership_loader.load_for_user`` — not available on
    the lightweight permissive mock. stub the same way so integration
    tests focus on tool behavior; the real path is exercised in
    ``tests/integration/test_cross_agent_workspace.py``.

    skipped for ``test_cross_agent_workspace`` so its real-PostgreSQL
    authorize matrix runs unaltered.

    :param request: pytest fixture request used to opt out per file
    :ptype request: pytest.FixtureRequest
    :param monkeypatch: pytest monkeypatch fixture
    :ptype monkeypatch: pytest.MonkeyPatch
    :return: the installed mock, or ``None`` when the stub is skipped
    :rtype: AsyncMock | None
    """
    if _is_real_authorize_test(request):
        return None
    from threetears.agent.workspace import authorize as _authorize_module

    stub = AsyncMock(return_value=None)
    monkeypatch.setattr(_authorize_module, "authorize_workspace_access", stub)
    file_stub = AsyncMock(return_value=None)
    monkeypatch.setattr(
        _authorize_module, "authorize_workspace_file_access", file_stub,
    )
    return stub


@pytest.fixture(autouse=True)
def integration_stub_enrich_workspace_identity(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncMock | None:
    """no-op stub for :func:`enrich_workspace_identity` in integration tests.

    the production helper does a ``SELECT customer_id FROM
    platform.namespaces`` lookup; the integration fake pool does not
    serve that statement and the fake workspace entities do not expose
    a ``customer_id`` setter, so the real enrichment cannot run. patch
    it out for these tests; identity enrichment + cross-customer denial
    are exercised in ``tests/integration/test_cross_agent_workspace.py``.

    :param request: pytest fixture request used to opt out per file
    :ptype request: pytest.FixtureRequest
    :param monkeypatch: pytest monkeypatch fixture
    :ptype monkeypatch: pytest.MonkeyPatch
    :return: the installed mock, or ``None`` when the stub is skipped
    :rtype: AsyncMock | None
    """
    if _is_real_authorize_test(request):
        return None
    from threetears.agent.workspace.tools import helpers as _helpers_module

    async def _passthrough(workspace, db_pool):  # type: ignore[no-untyped-def]
        return workspace

    stub = AsyncMock(side_effect=_passthrough)
    monkeypatch.setattr(_helpers_module, "enrich_workspace_identity", stub)
    return stub
