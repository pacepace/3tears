"""tests confirming every write-class workspace tool publishes an audit
envelope on its success path and tolerates audit-publish failure.

the mainline tests for each tool live in their own ``test_*.py`` module
and run with ``namespace=None`` so the audit block is a deliberate no-op
there (keeping those tests focused on SQL / sandbox behavior). this
module supplies ``namespace="ns"`` + a recording NATS fake to prove the
injection fires with the correct subject, event_type, resource_type,
action, and details shape, and that a raising NATS client does NOT
break the tool's success return.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest

from threetears.agent.workspace.tools import workspace_create as workspace_create_module
from threetears.agent.workspace.tools import workspace_delete as workspace_delete_module
from threetears.agent.workspace.tools.doc_merge import DocMergeTool
from threetears.agent.workspace.tools.doc_set import DocSetTool
from threetears.agent.workspace.tools.fs_edit import FsEditTool
from threetears.agent.workspace.tools.fs_write import FsWriteTool
from threetears.agent.workspace.tools.workspace_create import WorkspaceCreateTool
from threetears.agent.workspace.tools.workspace_delete import WorkspaceDeleteTool
from threetears.agent.workspace.tools.workspace_reset import WorkspaceResetTool
from threetears.agent.workspace.tools.workspace_rollback import WorkspaceRollbackTool


# ---------------------------------------------------------------------------
# shared fakes
# ---------------------------------------------------------------------------


@dataclass
class _FakeNats:
    published: list[tuple[str, bytes]] = field(default_factory=list)
    raise_on_publish: BaseException | None = None

    async def publish(self, subject: str, payload: bytes) -> None:
        self.published.append((subject, payload))
        if self.raise_on_publish is not None:
            raise self.raise_on_publish


@dataclass
class _FakeWorkspaceEntity:
    id: UUID
    name: str
    template_name: str | None = None
    date_deleted: Any = None
    current_version: int = 0
    # WS-ACL-10 audit envelope inputs: workspace_audit_identity reads
    # these directly off the workspace to fill owner_agent_id and
    # customer_id on the envelope. fixture defaults to fresh UUIDs so
    # tests do not depend on the specific identity values.
    owner_agent_id: UUID = field(default_factory=uuid4)
    customer_id: UUID | None = field(default_factory=uuid4)

    @property
    def namespace_name(self) -> str:
        """canonical workspace namespace name (WS-ACL-06)."""
        return f"workspace.{self.id}"


class _FakeWorkspaceCollection:
    def __init__(self, entities: list[_FakeWorkspaceEntity]) -> None:
        self._entities = entities

    async def find_by_agent_and_name(self, agent_id: UUID, name: str) -> _FakeWorkspaceEntity | None:
        for e in self._entities:
            if e.name == name:
                return e
        return None

    async def find_by_id_and_agent(self, workspace_id: UUID, agent_id: UUID) -> _FakeWorkspaceEntity | None:
        for e in self._entities:
            if e.id == workspace_id:
                return e
        return None

    async def find_by_workspace(self, workspace_id: UUID) -> list[Any]:
        return []


@dataclass
class _FakeFileEntity:
    relative_path: str
    content: bytes
    sha256: str
    version: int = 1


class _FakeFileCollection:
    def __init__(self, files: list[_FakeFileEntity] | None = None) -> None:
        self._files = files or []

    async def find_by_workspace_and_relative_path(
        self, workspace_id: UUID, relative_path: str
    ) -> _FakeFileEntity | None:
        for f in self._files:
            if f.relative_path == relative_path:
                return f
        return None

    async def find_by_workspace(self, workspace_id: UUID) -> list[_FakeFileEntity]:
        return list(self._files)


class _FakeVersionCollection:
    pass


class _RecordingSandbox:
    def __init__(self, templates_root: Path | None = None) -> None:
        self._templates_root = templates_root

    def resolve_fs_path(self, path: str, root_name: str) -> Path:
        if self._templates_root is None:
            raise KeyError(root_name)
        return self._templates_root

    def enforce(self, action: str, target: str) -> None:
        return None


class _FakeContext:
    pass


@dataclass
class _FakeTransaction:
    parent: _FakeConnection
    entered: bool = False
    exited: bool = False

    async def __aenter__(self) -> _FakeTransaction:
        self.entered = True
        self.parent.transaction_open = True
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.exited = True
        self.parent.transaction_open = False


@dataclass
class _FakeConnection:
    executions: list[tuple[str, tuple[Any, ...], bool]] = field(default_factory=list)
    fetchrows: list[tuple[str, tuple[Any, ...], bool]] = field(default_factory=list)
    transactions: list[_FakeTransaction] = field(default_factory=list)
    transaction_open: bool = False
    head_row: dict[str, Any] | None = None
    journal_max_version: int = 0

    def transaction(self, namespace: Any = None) -> _FakeTransaction:
        tx = _FakeTransaction(parent=self)
        self.transactions.append(tx)
        return tx

    async def execute(self, query: str, *args: Any, namespace: Any = None) -> str:
        self.executions.append((query, args, self.transaction_open))
        return "INSERT 0 1"

    async def fetchrow(self, query: str, *args: Any, namespace: Any = None) -> dict[str, Any] | None:
        """dispatch by SQL shape: journal-max SELECT returns a row with
        ``max_version``; head SELECT (and any fallback) returns ``head_row``.
        """
        self.fetchrows.append((query, args, self.transaction_open))
        result: dict[str, Any] | None
        if "COALESCE(MAX(version)" in query:
            result = {"max_version": self.journal_max_version}
        else:
            result = self.head_row
        return result


@dataclass
class _FakeAcquireCM:
    conn: _FakeConnection

    async def __aenter__(self) -> _FakeConnection:
        return self.conn

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        return None


@dataclass
class _FakePool:
    conn: _FakeConnection = field(default_factory=_FakeConnection)

    def acquire(self) -> _FakeAcquireCM:
        return _FakeAcquireCM(conn=self.conn)


@dataclass
class _PinnedSnapshot:
    workspace_id: UUID
    workspace_name: str


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _only_envelope(nats: _FakeNats) -> dict[str, Any]:
    """extract single envelope; assert exactly one publish happened."""
    assert len(nats.published) == 1, nats.published
    return json.loads(nats.published[0][1].decode("utf-8"))


def _subject(nats: _FakeNats) -> str:
    return nats.published[0][0]


# ---------------------------------------------------------------------------
# fs_write
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fs_write_publishes_audit_on_success(
    permissive_acl_cache: MagicMock,
) -> None:
    """fs_write success path emits one workspace.fs_write audit event."""
    agent_id = uuid4()
    ws = _FakeWorkspaceEntity(id=uuid4(), name="ws")
    nats = _FakeNats()
    pool = _FakePool()
    tool = FsWriteTool(
        workspace_collection=_FakeWorkspaceCollection([ws]),  # type: ignore[arg-type]
        workspace_file_collection=_FakeFileCollection(),  # type: ignore[arg-type]
        workspace_file_version_collection=_FakeVersionCollection(),  # type: ignore[arg-type]
        sandbox=_RecordingSandbox(),  # type: ignore[arg-type]
        context_provider=lambda: _FakeContext(),
        agent_id=agent_id,
        db_pool=pool,
        nats_client=nats,
        namespace="ns",
        acl_cache=permissive_acl_cache,
    )
    result = await tool.execute(relative_path="a.md", content="hello", workspace="ws")
    assert result.success is True, result.error
    assert _subject(nats) == "ns.audit.workspace.write"
    envelope = _only_envelope(nats)
    assert envelope["event_type"] == "workspace.fs_write"
    assert envelope["actor_type"] == "agent"
    # WS-ACL-10: actor_user_id + calling_agent_id + owner_agent_id +
    # customer_id + namespace_id ride on every envelope.
    assert UUID(envelope["actor_user_id"])
    assert envelope["agent_id"] == str(agent_id)
    assert UUID(envelope["calling_agent_id"])
    assert envelope["owner_agent_id"] == str(ws.owner_agent_id)
    assert UUID(envelope["customer_id"])
    assert envelope["namespace_id"] == str(ws.id)
    assert envelope["resource_type"] == "workspace_file"
    assert envelope["resource_id"] == f"{ws.id}/a.md"
    assert envelope["action"] == "write"
    assert envelope["details"]["bytes_after"] == 5
    assert envelope["details"]["sha256_before"] is None
    assert envelope["details"]["sha256_after"] is not None
    assert envelope["details"]["version"] == 1


@pytest.mark.asyncio
async def test_fs_write_success_preserved_when_publish_raises(
    permissive_acl_cache: MagicMock,
) -> None:
    """raising NATS publish does not break the tool's success return."""
    ws = _FakeWorkspaceEntity(id=uuid4(), name="ws")
    nats = _FakeNats(raise_on_publish=RuntimeError("nats offline"))
    pool = _FakePool()
    tool = FsWriteTool(
        workspace_collection=_FakeWorkspaceCollection([ws]),  # type: ignore[arg-type]
        workspace_file_collection=_FakeFileCollection(),  # type: ignore[arg-type]
        workspace_file_version_collection=_FakeVersionCollection(),  # type: ignore[arg-type]
        sandbox=_RecordingSandbox(),  # type: ignore[arg-type]
        context_provider=lambda: _FakeContext(),
        agent_id=uuid4(),
        db_pool=pool,
        nats_client=nats,
        namespace="ns",
        acl_cache=permissive_acl_cache,
    )
    result = await tool.execute(relative_path="a.md", content="hello", workspace="ws")
    assert result.success is True, result.error


# ---------------------------------------------------------------------------
# fs_edit
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fs_edit_publishes_audit_with_occurrence_count(
    permissive_acl_cache: MagicMock,
) -> None:
    """fs_edit audit details include occurrences, bytes, and sha pair."""
    ws = _FakeWorkspaceEntity(id=uuid4(), name="ws")
    existing = _FakeFileEntity(relative_path="a.md", content=b"foo bar foo", sha256="b" * 64, version=1)
    head = {"content": existing.content, "sha256": existing.sha256, "version": 1}
    pool = _FakePool()
    pool.conn.head_row = head
    pool.conn.journal_max_version = 1
    nats = _FakeNats()
    tool = FsEditTool(
        workspace_collection=_FakeWorkspaceCollection([ws]),  # type: ignore[arg-type]
        workspace_file_collection=_FakeFileCollection([existing]),  # type: ignore[arg-type]
        workspace_file_version_collection=_FakeVersionCollection(),  # type: ignore[arg-type]
        sandbox=_RecordingSandbox(),  # type: ignore[arg-type]
        context_provider=lambda: _FakeContext(),
        agent_id=uuid4(),
        db_pool=pool,
        nats_client=nats,
        namespace="ns",
        acl_cache=permissive_acl_cache,
    )
    result = await tool.execute(relative_path="a.md", find="foo", replace="qux", workspace="ws")
    assert result.success is True, result.error
    assert _subject(nats) == "ns.audit.workspace.edit"
    envelope = _only_envelope(nats)
    assert envelope["event_type"] == "workspace.fs_edit"
    assert envelope["action"] == "edit"
    assert envelope["details"]["occurrences"] == 2
    assert envelope["details"]["bytes_before"] == 11
    assert envelope["details"]["sha256_before"] == "b" * 64


# ---------------------------------------------------------------------------
# doc_set
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_doc_set_publishes_audit_with_jsonpath_and_value(
    permissive_acl_cache: MagicMock,
) -> None:
    """doc_set audit details carry jsonpath + value + sha pair."""
    ws = _FakeWorkspaceEntity(id=uuid4(), name="ws")
    existing = _FakeFileEntity(
        relative_path="config.yaml",
        content=b"k: 1\n",
        sha256="e" * 64,
        version=3,
    )
    head = {"content": existing.content, "sha256": existing.sha256, "version": 3}
    pool = _FakePool()
    pool.conn.head_row = head
    pool.conn.journal_max_version = 3
    nats = _FakeNats()
    tool = DocSetTool(
        workspace_collection=_FakeWorkspaceCollection([ws]),  # type: ignore[arg-type]
        workspace_file_collection=_FakeFileCollection([existing]),  # type: ignore[arg-type]
        workspace_file_version_collection=_FakeVersionCollection(),  # type: ignore[arg-type]
        sandbox=_RecordingSandbox(),  # type: ignore[arg-type]
        context_provider=lambda: _FakeContext(),
        agent_id=uuid4(),
        db_pool=pool,
        nats_client=nats,
        namespace="ns",
        acl_cache=permissive_acl_cache,
    )
    result = await tool.execute(
        relative_path="config.yaml",
        jsonpath="$.k",
        value=42,
        workspace="ws",
    )
    assert result.success is True, result.error
    assert _subject(nats) == "ns.audit.workspace.set"
    envelope = _only_envelope(nats)
    assert envelope["event_type"] == "workspace.doc_set"
    assert envelope["action"] == "set"
    assert envelope["details"]["jsonpath"] == "$.k"
    assert envelope["details"]["value"] == 42
    assert envelope["details"]["sha256_before"] == "e" * 64


# ---------------------------------------------------------------------------
# doc_merge
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_doc_merge_publishes_audit_with_partial_keys(
    permissive_acl_cache: MagicMock,
) -> None:
    """doc_merge audit details list merged top-level keys."""
    ws = _FakeWorkspaceEntity(id=uuid4(), name="ws")
    existing = _FakeFileEntity(
        relative_path="config.yaml",
        content=b"a: 1\nb: 2\n",
        sha256="f" * 64,
        version=2,
    )
    head = {"content": existing.content, "sha256": existing.sha256, "version": 2}
    pool = _FakePool()
    pool.conn.head_row = head
    pool.conn.journal_max_version = 2
    nats = _FakeNats()
    tool = DocMergeTool(
        workspace_collection=_FakeWorkspaceCollection([ws]),  # type: ignore[arg-type]
        workspace_file_collection=_FakeFileCollection([existing]),  # type: ignore[arg-type]
        workspace_file_version_collection=_FakeVersionCollection(),  # type: ignore[arg-type]
        sandbox=_RecordingSandbox(),  # type: ignore[arg-type]
        context_provider=lambda: _FakeContext(),
        agent_id=uuid4(),
        db_pool=pool,
        nats_client=nats,
        namespace="ns",
        acl_cache=permissive_acl_cache,
    )
    result = await tool.execute(
        relative_path="config.yaml",
        partial={"c": 3, "d": 4},
        workspace="ws",
    )
    assert result.success is True, result.error
    assert _subject(nats) == "ns.audit.workspace.merge"
    envelope = _only_envelope(nats)
    assert envelope["action"] == "merge"
    assert sorted(envelope["details"]["partial_keys"]) == ["c", "d"]


# ---------------------------------------------------------------------------
# workspace.create
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_workspace_create_publishes_audit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """empty-create emits workspace.create audit with files_changed=0."""

    async def _noop_set_pin(*args: Any, **kwargs: Any) -> None:
        return None

    monkeypatch.setattr(workspace_create_module.pin, "set_pin", _noop_set_pin)
    nats = _FakeNats()
    pool = _FakePool()
    agent_id = uuid4()
    tool = WorkspaceCreateTool(
        workspace_collection=_FakeWorkspaceCollection([]),  # type: ignore[arg-type]
        workspace_file_collection=_FakeFileCollection(),  # type: ignore[arg-type]
        workspace_file_version_collection=_FakeVersionCollection(),  # type: ignore[arg-type]
        sandbox=_RecordingSandbox(),  # type: ignore[arg-type]
        context_provider=lambda: _FakeContext(),
        agent_id=agent_id,
        db_pool=pool,
        nats_client=nats,
        namespace="ns",
    )
    result = await tool.execute(name="new_ws")
    assert result.success is True, result.error
    assert _subject(nats) == "ns.audit.workspace.create"
    envelope = _only_envelope(nats)
    assert envelope["event_type"] == "workspace.create"
    assert envelope["resource_type"] == "workspace"
    assert envelope["action"] == "create"
    assert envelope["details"]["name"] == "new_ws"
    assert envelope["details"]["files_changed"] == 0
    assert envelope["details"]["template_name"] is None


# ---------------------------------------------------------------------------
# workspace.reset
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_workspace_reset_publishes_audit(
    tmp_path: Path,
    permissive_acl_cache: MagicMock,
) -> None:
    """reset emits workspace.reset with template_name and files_changed."""
    template_dir = tmp_path / "starter"
    template_dir.mkdir()
    (template_dir / "a.md").write_bytes(b"hello\n")
    ws_id = uuid4()
    ws = _FakeWorkspaceEntity(id=ws_id, name="seed", template_name="starter", current_version=1)
    nats = _FakeNats()
    pool = _FakePool()
    tool = WorkspaceResetTool(
        workspace_collection=_FakeWorkspaceCollection([ws]),  # type: ignore[arg-type]
        workspace_file_collection=_FakeFileCollection(),  # type: ignore[arg-type]
        workspace_file_version_collection=_FakeVersionCollection(),  # type: ignore[arg-type]
        sandbox=_RecordingSandbox(templates_root=template_dir),  # type: ignore[arg-type]
        context_provider=lambda: _FakeContext(),
        agent_id=uuid4(),
        db_pool=pool,
        nats_client=nats,
        namespace="ns",
        acl_cache=permissive_acl_cache,
    )
    result = await tool.execute(name="seed")
    assert result.success is True, result.error
    assert _subject(nats) == "ns.audit.workspace.reset"
    envelope = _only_envelope(nats)
    assert envelope["event_type"] == "workspace.reset"
    assert envelope["action"] == "reset"
    assert envelope["resource_type"] == "workspace"
    assert envelope["resource_id"] == str(ws_id)
    assert envelope["details"]["template_name"] == "starter"
    assert envelope["details"]["files_changed"] == 1


# ---------------------------------------------------------------------------
# workspace.delete
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_workspace_delete_publishes_audit(
    monkeypatch: pytest.MonkeyPatch,
    permissive_acl_cache: MagicMock,
) -> None:
    """soft-delete emits workspace.delete audit."""
    ws_id = uuid4()
    ws = _FakeWorkspaceEntity(id=ws_id, name="bye")

    async def _get_pin(_ctx: Any) -> Any:
        return None

    async def _clear_pin(_ctx: Any) -> None:
        return None

    monkeypatch.setattr(workspace_delete_module.pin, "get_pin", _get_pin)
    monkeypatch.setattr(workspace_delete_module.pin, "clear_pin", _clear_pin)

    nats = _FakeNats()
    pool = _FakePool()
    tool = WorkspaceDeleteTool(
        workspace_collection=_FakeWorkspaceCollection([ws]),  # type: ignore[arg-type]
        workspace_file_collection=_FakeFileCollection(),  # type: ignore[arg-type]
        workspace_file_version_collection=_FakeVersionCollection(),  # type: ignore[arg-type]
        sandbox=_RecordingSandbox(),  # type: ignore[arg-type]
        context_provider=lambda: _FakeContext(),
        agent_id=uuid4(),
        db_pool=pool,
        nats_client=nats,
        namespace="ns",
        acl_cache=permissive_acl_cache,
    )
    result = await tool.execute(name="bye")
    assert result.success is True, result.error
    assert _subject(nats) == "ns.audit.workspace.delete"
    envelope = _only_envelope(nats)
    assert envelope["event_type"] == "workspace.delete"
    assert envelope["action"] == "delete"
    assert envelope["resource_id"] == str(ws_id)
    assert envelope["details"]["name"] == "bye"


# ---------------------------------------------------------------------------
# workspace.rollback_to
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_workspace_rollback_publishes_single_audit_event(
    permissive_acl_cache: MagicMock,
) -> None:
    """rollback_to emits exactly one audit event with files_changed count."""
    ws_id = uuid4()
    ws = _FakeWorkspaceEntity(id=ws_id, name="ws", current_version=5)
    nats = _FakeNats()

    # pool where _resolve_ref sees no matching row -> rollback becomes
    # a no-op on file writes; the single summary audit event still fires
    pool = _FakePool()
    pool.conn.head_row = None  # _resolve_ref returns None -> skip
    tool = WorkspaceRollbackTool(
        workspace_collection=_FakeWorkspaceCollection([ws]),  # type: ignore[arg-type]
        workspace_file_collection=_FakeFileCollection(
            [
                _FakeFileEntity(
                    relative_path="a.md",
                    content=b"x",
                    sha256="aa" * 32,
                    version=1,
                )
            ]
        ),  # type: ignore[arg-type]
        workspace_file_version_collection=_FakeVersionCollection(),  # type: ignore[arg-type]
        sandbox=_RecordingSandbox(),  # type: ignore[arg-type]
        context_provider=lambda: _FakeContext(),
        agent_id=uuid4(),
        db_pool=pool,
        nats_client=nats,
        namespace="ns",
        acl_cache=permissive_acl_cache,
    )
    result = await tool.execute(ref="head", workspace="ws")
    assert result.success is True, result.error
    assert _subject(nats) == "ns.audit.workspace.rollback_to"
    envelope = _only_envelope(nats)
    assert envelope["event_type"] == "workspace.rollback_to"
    assert envelope["action"] == "rollback_to"
    assert envelope["resource_type"] == "workspace"
    assert envelope["resource_id"] == str(ws_id)
    assert envelope["details"]["ref"] == "head"
    assert envelope["details"]["files_changed"] == 0
