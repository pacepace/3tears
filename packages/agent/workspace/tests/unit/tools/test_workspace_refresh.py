"""tests for ``threetears.workspace.refresh_from_disk`` -- WorkspaceRefreshTool.

coverage:

- happy path: files on disk missing from L3 are journaled + upserted.
- no-bind-root: sandbox raises :class:`KeyError` -> clean error-as-data.
- idempotency: repeated refresh on unchanged files is a no-op.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock
from uuid import UUID, uuid4, uuid7

import pytest

from threetears.agent.workspace.tools.workspace_refresh import WorkspaceRefreshTool
from _helpers.asyncpg_shims import FakeAsyncpgAcquireCM, FakeAsyncpgConnection, FakeAsyncpgPool, FakeAsyncpgTransaction
from _helpers.workspace_shims import (
    FakeWorkspaceCollection,
    FakeWorkspaceContext,
    FakeWorkspaceEntity,
    FakeWorkspaceFile,
    FakeWorkspaceFileCollection,
    FakeWorkspaceFileVersionCollection,
    FakeWorkspaceSandbox,
)


def _sha(content: bytes) -> str:
    """compute hex sha256 of raw bytes.

    :param content: bytes to digest
    :ptype content: bytes
    :return: hex digest
    :rtype: str
    """
    return hashlib.sha256(content).hexdigest()


@dataclass
class _FakeWorkspaceEntity(FakeWorkspaceEntity):
    """stand-in for :class:`Workspace`.

    :ivar id: workspace identifier
    :ivar name: workspace name
    :ivar agent_id: owning agent (partition column on workspaces)
    :ivar date_deleted: soft-delete timestamp, None when live
    """

    id: UUID
    name: str
    agent_id: UUID = field(default_factory=uuid4)
    date_deleted: Any = None

    @property
    def namespace_name(self) -> str:
        """canonical workspace namespace name (WS-ACL-06)."""
        return f"workspace.{self.id}"


class _FakeWorkspaceCollection(FakeWorkspaceCollection):
    """fake workspace collection serving _resolve_workspace."""

    def __init__(self, entities: list[_FakeWorkspaceEntity]) -> None:
        """capture entities.

        :param entities: workspace entities to serve
        :ptype entities: list[_FakeWorkspaceEntity]
        :return: None
        :rtype: None
        """
        self._entities = entities

    async def find_by_agent_and_name(
        self,
        agent_id: UUID,
        name: str,
    ) -> _FakeWorkspaceEntity | None:
        """locate by (agent_id, name).

        :param agent_id: owning agent (unused)
        :ptype agent_id: UUID
        :param name: workspace name
        :ptype name: str
        :return: matched entity or None
        :rtype: _FakeWorkspaceEntity | None
        """
        del agent_id
        result: _FakeWorkspaceEntity | None = None
        for e in self._entities:
            if e.name == name:
                result = e
                break
        return result

    async def find_by_id_and_agent(
        self,
        workspace_id: UUID,
        agent_id: UUID,
    ) -> _FakeWorkspaceEntity | None:
        """locate by (id, agent).

        :param workspace_id: target id
        :ptype workspace_id: UUID
        :param agent_id: owning agent (unused)
        :ptype agent_id: UUID
        :return: matched entity or None
        :rtype: _FakeWorkspaceEntity | None
        """
        del agent_id
        result: _FakeWorkspaceEntity | None = None
        for e in self._entities:
            if e.id == workspace_id:
                result = e
                break
        return result


@dataclass
class _FakeFileEntity(FakeWorkspaceFile):
    """stand-in for :class:`WorkspaceFile`.

    :ivar relative_path: path within workspace
    :ivar sha256: hex digest
    :ivar version: version number
    """

    relative_path: str
    sha256: str
    version: int = 1


class _FakeFileCollection(FakeWorkspaceFileCollection):
    """fake head-state file collection."""

    def __init__(self, files: list[_FakeFileEntity]) -> None:
        """capture seeded files.

        :param files: head-state entities to serve
        :ptype files: list[_FakeFileEntity]
        :return: None
        :rtype: None
        """
        self._files = files

    async def find_by_workspace(
        self,
        workspace_id: UUID,
    ) -> list[_FakeFileEntity]:
        """return seeded files (copy).

        :param workspace_id: unused
        :ptype workspace_id: UUID
        :return: copy of seeded files
        :rtype: list[_FakeFileEntity]
        """
        del workspace_id
        return list(self._files)


class _FakeVersionCollection(FakeWorkspaceFileVersionCollection):
    """placeholder; tool uses pool directly."""


class _FakeSandbox(FakeWorkspaceSandbox):
    """stand-in for :class:`WorkspaceSandbox` with optional root map."""

    def __init__(self, roots: dict[str, Path]) -> None:
        """capture the root mapping.

        :param roots: root name -> absolute path
        :ptype roots: dict[str, Path]
        :return: None
        :rtype: None
        """
        self._roots = roots

    def resolve_fs_path(self, path: str, root_name: str) -> Path:
        """resolve ``root / path``; raise :class:`KeyError` for unknown roots.

        :param path: relative path below the root
        :ptype path: str
        :param root_name: named root
        :ptype root_name: str
        :return: resolved absolute path
        :rtype: Path
        """
        return self._roots[root_name] / path


@dataclass
class _FakeTransaction(FakeAsyncpgTransaction):
    """fake asyncpg transaction context manager."""

    parent: _FakeConnection
    entered: bool = False
    exited: bool = False

    async def __aenter__(self) -> _FakeTransaction:
        """open the transaction.

        :return: self
        :rtype: _FakeTransaction
        """
        self.entered = True
        self.parent.transaction_open = True
        return self

    async def __aexit__(
        self,
        exc_type: Any,
        exc_val: Any,
        exc_tb: Any,
    ) -> None:
        """close the transaction.

        :param exc_type: exception type
        :ptype exc_type: Any
        :param exc_val: exception value
        :ptype exc_val: Any
        :param exc_tb: exception traceback
        :ptype exc_tb: Any
        :return: None
        :rtype: None
        """
        del exc_type, exc_val, exc_tb
        self.exited = True
        self.parent.transaction_open = False


@dataclass
class _FakeConnection(FakeAsyncpgConnection):
    """fake asyncpg connection recording execute + fetchrow."""

    executions: list[tuple[str, tuple[Any, ...]]] = field(default_factory=list)
    transactions: list[_FakeTransaction] = field(default_factory=list)
    transaction_open: bool = False
    journal_max_by_path: dict[str, int] = field(default_factory=dict)

    def transaction(self, namespace: Any = None) -> _FakeTransaction:
        """create a transaction wrapper.

        :return: transaction context manager
        :rtype: _FakeTransaction
        """
        tx = _FakeTransaction(parent=self)
        self.transactions.append(tx)
        return tx

    async def execute(self, query: str, *args: Any) -> str:
        """record the execute and maintain journal-max state.

        :param query: SQL text
        :ptype query: str
        :param args: bound parameters
        :ptype args: Any
        :return: asyncpg-style tag
        :rtype: str
        """
        self.executions.append((query, args))
        if "INSERT INTO workspace_file_versions" in query:
            rel = args[2]
            inserted = int(args[3])
            prior = self.journal_max_by_path.get(rel, 0)
            if inserted > prior:
                self.journal_max_by_path[rel] = inserted
        return "INSERT 0 1"

    async def fetchrow(
        self,
        query: str,
        *args: Any,
    ) -> dict[str, Any] | None:
        """resolve journal-max SELECT.

        :param query: SQL text
        :ptype query: str
        :param args: bound parameters
        :ptype args: Any
        :return: dict row or None
        :rtype: dict[str, Any] | None
        """
        result: dict[str, Any] | None = None
        if "COALESCE(MAX(version)" in query:
            rel = args[1]
            result = {"max_version": self.journal_max_by_path.get(rel, 0)}
        return result


@dataclass
class _FakeAcquireCM(FakeAsyncpgAcquireCM):
    """asyncpg-shaped acquire CM."""

    conn: _FakeConnection

    async def __aenter__(self) -> _FakeConnection:
        """return the wrapped connection.

        :return: fake connection
        :rtype: _FakeConnection
        """
        return self.conn

    async def __aexit__(
        self,
        exc_type: Any,
        exc_val: Any,
        exc_tb: Any,
    ) -> None:
        """no-op close.

        :param exc_type: exception type
        :ptype exc_type: Any
        :param exc_val: exception value
        :ptype exc_val: Any
        :param exc_tb: exception traceback
        :ptype exc_tb: Any
        :return: None
        :rtype: None
        """
        del exc_type, exc_val, exc_tb


@dataclass
class _FakePool(FakeAsyncpgPool):
    """asyncpg-shaped pool sharing a single connection."""

    conn: _FakeConnection = field(default_factory=_FakeConnection)

    def acquire(self) -> _FakeAcquireCM:
        """acquire the shared connection.

        :return: acquire context manager
        :rtype: _FakeAcquireCM
        """
        return _FakeAcquireCM(conn=self.conn)


class _FakeContext(FakeWorkspaceContext):
    """sentinel context manager; no methods invoked in these tests."""


@pytest.mark.asyncio
async def test_refresh_happy_path_imports_disk_files(
    tmp_path: Path,
    permissive_acl_cache: MagicMock,
) -> None:
    """files on disk not in L3 are imported as create at version 1.

    :param tmp_path: scratch root from pytest
    :ptype tmp_path: Path
    :return: None
    :rtype: None
    """
    agent_id = uuid4()
    workspace_id = uuid7()
    entity = _FakeWorkspaceEntity(id=workspace_id, name="ws_test")
    wcoll = _FakeWorkspaceCollection([entity])
    fcoll = _FakeFileCollection([])
    vcoll = _FakeVersionCollection()

    bind_root = tmp_path / "bind_root"
    disk_root = bind_root / entity.name
    disk_root.mkdir(parents=True)
    (disk_root / "alpha.yaml").write_bytes(b"alpha: 1\n")
    (disk_root / "beta.md").write_bytes(b"# hello\n")

    sandbox = _FakeSandbox({"bind": bind_root})
    pool = _FakePool()

    tool = WorkspaceRefreshTool(
        workspace_collection=wcoll,
        workspace_file_collection=fcoll,
        workspace_file_version_collection=vcoll,
        sandbox=sandbox,
        context_provider=lambda: _FakeContext(),
        agent_id=agent_id,
        db_pool=pool,
        acl_cache=permissive_acl_cache,
    )

    result = await tool.execute(workspace="ws_test")
    assert result.success is True
    assert result.error is None
    assert (result.metadata or {}).get("imported_count") == 2
    assert "2 files" in result.content

    journal = [e for e in pool.conn.executions if "INSERT INTO workspace_file_versions" in e[0]]
    rels = sorted(row[1][2] for row in journal)
    assert rels == ["alpha.yaml", "beta.md"]
    for row in journal:
        assert row[1][6] == "create"
        assert row[1][3] == 1


@pytest.mark.asyncio
async def test_refresh_no_bind_root_returns_clean_error(
    tmp_path: Path,
    permissive_acl_cache: MagicMock,
) -> None:
    """sandbox missing ``bind`` root yields ``success=False`` with useful message.

    :param tmp_path: scratch root from pytest (only for structural parity)
    :ptype tmp_path: Path
    :return: None
    :rtype: None
    """
    del tmp_path
    agent_id = uuid4()
    workspace_id = uuid7()
    entity = _FakeWorkspaceEntity(id=workspace_id, name="ws_test")
    wcoll = _FakeWorkspaceCollection([entity])
    fcoll = _FakeFileCollection([])
    vcoll = _FakeVersionCollection()
    sandbox = _FakeSandbox({})  # no bind root registered
    pool = _FakePool()

    tool = WorkspaceRefreshTool(
        workspace_collection=wcoll,
        workspace_file_collection=fcoll,
        workspace_file_version_collection=vcoll,
        sandbox=sandbox,
        context_provider=lambda: _FakeContext(),
        agent_id=agent_id,
        db_pool=pool,
        acl_cache=permissive_acl_cache,
    )

    result = await tool.execute(workspace="ws_test")
    assert result.success is False
    assert result.error is not None
    assert "bind_root" in result.error


@pytest.mark.asyncio
async def test_refresh_idempotent_on_unchanged_files(
    tmp_path: Path,
    permissive_acl_cache: MagicMock,
) -> None:
    """repeated refresh on unchanged files performs no writes.

    :param tmp_path: scratch root from pytest
    :ptype tmp_path: Path
    :return: None
    :rtype: None
    """
    agent_id = uuid4()
    workspace_id = uuid7()
    entity = _FakeWorkspaceEntity(id=workspace_id, name="ws_test")
    content = b"stable\n"
    head = _FakeFileEntity(
        relative_path="alpha.yaml",
        sha256=_sha(content),
        version=1,
    )
    wcoll = _FakeWorkspaceCollection([entity])
    fcoll = _FakeFileCollection([head])
    vcoll = _FakeVersionCollection()

    bind_root = tmp_path / "bind_root"
    disk_root = bind_root / entity.name
    disk_root.mkdir(parents=True)
    (disk_root / "alpha.yaml").write_bytes(content)

    sandbox = _FakeSandbox({"bind": bind_root})
    pool = _FakePool()

    tool = WorkspaceRefreshTool(
        workspace_collection=wcoll,
        workspace_file_collection=fcoll,
        workspace_file_version_collection=vcoll,
        sandbox=sandbox,
        context_provider=lambda: _FakeContext(),
        agent_id=agent_id,
        db_pool=pool,
        acl_cache=permissive_acl_cache,
    )

    result = await tool.execute(workspace="ws_test")
    assert result.success is True
    assert (result.metadata or {}).get("imported_count") == 0
    assert pool.conn.executions == []
    assert pool.conn.transactions == []
