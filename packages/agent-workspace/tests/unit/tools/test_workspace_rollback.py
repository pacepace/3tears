"""tests for ``threetears.workspace.rollback_to`` -- WorkspaceRollbackTool."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest

from threetears.agent.tools.base_tool import MCPToolDefinition
from threetears.core.security import SandboxDenied

from threetears.agent.workspace.tools import (
    workspace_rollback as workspace_rollback_module,
)
from threetears.agent.workspace.tools.workspace_rollback import (
    WorkspaceRollbackTool,
)


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------


@dataclass
class _FakeWorkspaceEntity:
    id: UUID
    name: str
    date_deleted: Any = None


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


@dataclass
class _FakeFileEntity:
    relative_path: str
    content: bytes = b""
    sha256: str = "s" * 64
    version: int = 1


class _FakeFileCollection:
    def __init__(self, files: list[_FakeFileEntity]) -> None:
        self._files = files

    async def find_by_workspace(self, workspace_id: UUID) -> list[_FakeFileEntity]:
        return list(self._files)

    async def find_by_workspace_and_relative_path(
        self, workspace_id: UUID, relative_path: str
    ) -> _FakeFileEntity | None:
        for f in self._files:
            if f.relative_path == relative_path:
                return f
        return None


class _FakeVersionCollection:
    """placeholder -- rollback delegates writes to _write_file_atomic."""


class _RecordingSandbox:
    def __init__(self, deny_writes: list[str] | None = None) -> None:
        self._deny_writes = set(deny_writes or [])
        self.enforce_calls: list[tuple[str, str]] = []

    def enforce(self, action: str, target: str) -> None:
        self.enforce_calls.append((action, target))
        if action == "write" and target in self._deny_writes:
            raise SandboxDenied(action, target, "not in write globs")


@dataclass
class _FakeAcquireCM:
    conn: Any

    async def __aenter__(self) -> Any:
        return self.conn

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        return None


@dataclass
class _FakePool:
    """pool with a single long-lived conn that scripts _resolve_ref lookups.

    scripts match by (relative_path, selector); selector is the last
    positional arg _resolve_ref passes (int for numeric ref, label str for
    checkpoint label, "head" implied by two-arg call shape).
    """

    script: dict[tuple[str, Any], dict[str, Any] | None] = field(default_factory=dict)
    fetchrows: list[tuple[str, tuple[Any, ...]]] = field(default_factory=list)

    def acquire(self) -> _FakeAcquireCM:
        return _FakeAcquireCM(conn=self)

    async def fetchrow(self, query: str, *args: Any) -> dict[str, Any] | None:
        self.fetchrows.append((query, args))
        relative_path = args[1]
        if len(args) == 2:
            selector: Any = "head"
        else:
            selector = args[2]
        return self.script.get((relative_path, selector))


class _FakeContext:
    pass


def _build_tool(
    *,
    workspace_entities: list[_FakeWorkspaceEntity],
    files: list[_FakeFileEntity],
    acl_cache: Any,
    script: dict[tuple[str, Any], dict[str, Any] | None] | None = None,
    deny_writes: list[str] | None = None,
) -> tuple[
    WorkspaceRollbackTool,
    _FakePool,
    _RecordingSandbox,
    _FakeFileCollection,
]:
    workspaces = _FakeWorkspaceCollection(workspace_entities)
    file_coll = _FakeFileCollection(files)
    pool = _FakePool(script=dict(script or {}))
    sandbox = _RecordingSandbox(deny_writes=deny_writes)
    tool = WorkspaceRollbackTool(
        workspace_collection=workspaces,  # type: ignore[arg-type]
        workspace_file_collection=file_coll,  # type: ignore[arg-type]
        workspace_file_version_collection=_FakeVersionCollection(),  # type: ignore[arg-type]
        sandbox=sandbox,  # type: ignore[arg-type]
        context_provider=lambda: _FakeContext(),
        agent_id=uuid4(),
        db_pool=pool,
        acl_cache=acl_cache,
    )
    return tool, pool, sandbox, file_coll


def _install_atomic_recorder(
    monkeypatch: pytest.MonkeyPatch,
) -> list[dict[str, Any]]:
    """replace _write_file_atomic in rollback module with a recorder."""
    calls: list[dict[str, Any]] = []

    async def _recorder(**kwargs: Any) -> tuple[int, str]:
        calls.append(kwargs)
        return 99, "r" * 64

    monkeypatch.setattr(workspace_rollback_module, "_write_file_atomic", _recorder)
    return calls


# ---------------------------------------------------------------------------
# happy paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rollback_whole_workspace_reverts_each_file(
    monkeypatch: pytest.MonkeyPatch,
    permissive_acl_cache: MagicMock,
) -> None:
    """ref=1 across all head files -> _write_file_atomic called per file with action=revert."""
    ws = _FakeWorkspaceEntity(id=uuid4(), name="ws")
    files = [
        _FakeFileEntity(relative_path="a.txt"),
        _FakeFileEntity(relative_path="b.md"),
    ]
    script = {
        ("a.txt", 1): {
            "content": b"old-a",
            "sha256": "a" * 64,
            "version": 1,
        },
        ("b.md", 1): {
            "content": b"old-b",
            "sha256": "b" * 64,
            "version": 1,
        },
    }
    recorded = _install_atomic_recorder(monkeypatch)
    tool, _pool, sandbox, _files = _build_tool(
        workspace_entities=[ws], files=files, script=script, acl_cache=permissive_acl_cache
    )
    result = await tool.execute(ref=1, workspace="ws")
    assert result.success is True, result.error
    assert "2 files" in result.content
    assert "ref 1" in result.content or "ref 1" in result.content

    # enforce called once per file BEFORE any atomic write
    assert sandbox.enforce_calls == [
        ("write", "a.txt"),
        ("write", "b.md"),
    ]
    # one atomic write call per file
    assert len(recorded) == 2
    paths_written = {call["relative_path"] for call in recorded}
    assert paths_written == {"a.txt", "b.md"}
    for call in recorded:
        assert call["action"] == "revert"
        assert call["expected_sha256"] is None
        assert call["workspace"] is ws


@pytest.mark.asyncio
async def test_rollback_single_file_narrows_set(
    monkeypatch: pytest.MonkeyPatch,
    permissive_acl_cache: MagicMock,
) -> None:
    """relative_path narrows rollback to exactly one file."""
    ws = _FakeWorkspaceEntity(id=uuid4(), name="ws")
    files = [
        _FakeFileEntity(relative_path="a.txt"),
        _FakeFileEntity(relative_path="b.md"),
    ]
    script = {
        ("b.md", 2): {
            "content": b"v2",
            "sha256": "x" * 64,
            "version": 2,
        },
    }
    recorded = _install_atomic_recorder(monkeypatch)
    tool, _pool, sandbox, _ = _build_tool(
        workspace_entities=[ws], files=files, script=script, acl_cache=permissive_acl_cache
    )
    result = await tool.execute(ref=2, relative_path="b.md", workspace="ws")
    assert result.success is True, result.error
    # enforce only called for the one path
    assert sandbox.enforce_calls == [("write", "b.md")]
    assert len(recorded) == 1
    assert recorded[0]["relative_path"] == "b.md"
    assert recorded[0]["action"] == "revert"


@pytest.mark.asyncio
async def test_rollback_skips_files_absent_at_ref(
    monkeypatch: pytest.MonkeyPatch,
    permissive_acl_cache: MagicMock,
) -> None:
    """files whose ref resolves to None are skipped; n_changed reflects it."""
    ws = _FakeWorkspaceEntity(id=uuid4(), name="ws")
    files = [
        _FakeFileEntity(relative_path="a.txt"),
        _FakeFileEntity(relative_path="new.md"),
    ]
    script = {
        ("a.txt", 1): {
            "content": b"old-a",
            "sha256": "a" * 64,
            "version": 1,
        },
        # new.md did not exist at version 1
        ("new.md", 1): None,
    }
    recorded = _install_atomic_recorder(monkeypatch)
    tool, _pool, sandbox, _ = _build_tool(
        workspace_entities=[ws], files=files, script=script, acl_cache=permissive_acl_cache
    )
    result = await tool.execute(ref=1, workspace="ws")
    assert result.success is True, result.error
    # n_changed is 1 -- only the path that had a target row
    assert "1 files" in result.content
    # but BOTH paths were enforced (sandbox-enforce precedes resolve)
    assert sandbox.enforce_calls == [
        ("write", "a.txt"),
        ("write", "new.md"),
    ]
    # only one atomic write actually issued
    assert len(recorded) == 1
    assert recorded[0]["relative_path"] == "a.txt"


# ---------------------------------------------------------------------------
# error paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rollback_sandbox_denied_aborts_before_any_write(
    monkeypatch: pytest.MonkeyPatch,
    permissive_acl_cache: MagicMock,
) -> None:
    """a single denied path aborts the entire rollback; zero atomic writes occur."""
    ws = _FakeWorkspaceEntity(id=uuid4(), name="ws")
    files = [
        _FakeFileEntity(relative_path="public.md"),
        _FakeFileEntity(relative_path="secret.env"),
    ]
    script = {
        ("public.md", 1): {
            "content": b"old",
            "sha256": "p" * 64,
            "version": 1,
        },
        ("secret.env", 1): {
            "content": b"old-env",
            "sha256": "e" * 64,
            "version": 1,
        },
    }
    recorded = _install_atomic_recorder(monkeypatch)
    tool, pool, sandbox, _ = _build_tool(
        workspace_entities=[ws],
        files=files,
        script=script,
        deny_writes=["secret.env"],
        acl_cache=permissive_acl_cache,
    )
    result = await tool.execute(ref=1, workspace="ws")
    assert result.success is False
    assert result.error is not None
    # rollback fails-wholesale: NO atomic writes happened
    assert recorded == []
    # no ref-resolve queries were issued (sandbox-pass sweep happens first)
    assert pool.fetchrows == []
    # sandbox.enforce was called at least for the denied path
    assert ("write", "secret.env") in sandbox.enforce_calls


@pytest.mark.asyncio
async def test_rollback_missing_ref_required_returns_error(
    permissive_acl_cache: MagicMock,
) -> None:
    """omitting ref returns clean error without mutations."""
    ws = _FakeWorkspaceEntity(id=uuid4(), name="ws")
    tool, _pool, _sandbox, _ = _build_tool(
        workspace_entities=[ws], files=[], acl_cache=permissive_acl_cache
    )
    result = await tool.execute(workspace="ws")
    assert result.success is False
    assert result.error is not None
    assert "ref" in result.error


@pytest.mark.asyncio
async def test_rollback_unknown_workspace_returns_clean_error(
    permissive_acl_cache: MagicMock,
) -> None:
    """unknown workspace name -> clean error."""
    tool, _pool, _sandbox, _ = _build_tool(
        workspace_entities=[], files=[], acl_cache=permissive_acl_cache
    )
    result = await tool.execute(ref=1, workspace="ghost")
    assert result.success is False
    assert result.error is not None
    assert "ghost" in result.error


# ---------------------------------------------------------------------------
# MCP surface
# ---------------------------------------------------------------------------


def test_rollback_mcp_name_is_exact_string(
    permissive_acl_cache: MagicMock,
) -> None:
    tool, _, _, _ = _build_tool(
        workspace_entities=[_FakeWorkspaceEntity(id=uuid4(), name="ws")],
        files=[],
        acl_cache=permissive_acl_cache,
    )
    assert tool.mcp_name() == "threetears.workspace.rollback_to"


def test_rollback_mcp_schema_shape(
    permissive_acl_cache: MagicMock,
) -> None:
    tool, _, _, _ = _build_tool(
        workspace_entities=[_FakeWorkspaceEntity(id=uuid4(), name="ws")],
        files=[],
        acl_cache=permissive_acl_cache,
    )
    defn = tool.mcp_schema()
    assert isinstance(defn, MCPToolDefinition)
    schema = defn.input_schema
    assert schema["required"] == ["ref"]
    assert schema["additionalProperties"] is False
    props = schema["properties"]
    assert set(props.keys()) == {"ref", "relative_path", "workspace"}
    assert props["ref"]["type"] == ["string", "integer"]
