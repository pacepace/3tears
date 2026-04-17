"""tests for ``threetears.workspace.diff`` -- WorkspaceDiffTool."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

import pytest

from threetears.agent.tools.base_tool import MCPToolDefinition
from threetears.core.security import SandboxDenied

from threetears.agent.workspace.tools.workspace_diff import WorkspaceDiffTool


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


class _RecordingSandbox:
    def __init__(self, deny_reads: list[str] | None = None) -> None:
        self._deny_reads = set(deny_reads or [])
        self.enforce_calls: list[tuple[str, str]] = []

    def enforce(self, action: str, target: str) -> None:
        self.enforce_calls.append((action, target))
        if action == "read" and target in self._deny_reads:
            raise SandboxDenied(action, target, "not in read globs")


class _FakeContext:
    pass


@dataclass
class _JournalRow:
    """stand-in for a raw asyncpg Record -- dict-lookup row."""

    data: dict[str, Any]

    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def keys(self) -> Any:
        return self.data.keys()

    def values(self) -> Any:
        return self.data.values()

    def items(self) -> Any:
        return self.data.items()


@dataclass
class _FakeConnection:
    """scripts fetchrow by (relative_path, ref_selector) tuple."""

    script: dict[tuple[str, Any], dict[str, Any] | None] = field(default_factory=dict)
    fetchrows: list[tuple[str, tuple[Any, ...]]] = field(default_factory=list)

    async def fetchrow(self, query: str, *args: Any) -> dict[str, Any] | None:
        self.fetchrows.append((query, args))
        # selector: last positional matches the ref shape
        # for head: args = (ws_id, path)
        # for int: args = (ws_id, path, int_version)
        # for label: args = (ws_id, path, label)
        relative_path = args[1]
        if len(args) == 2:
            selector: Any = "head"
        else:
            selector = args[2]
        return self.script.get((relative_path, selector))


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


def _journal_row(content: bytes, version: int = 1) -> dict[str, Any]:
    return {
        "id": uuid4(),
        "workspace_id": uuid4(),
        "relative_path": "x",
        "version": version,
        "content": content,
        "sha256": "s" * 64,
        "action": "update",
        "label": None,
        "actor_id": uuid4(),
        "correlation_id": uuid4(),
        "date_created": datetime.now(UTC),
    }


def _build_tool(
    *,
    workspace_entities: list[_FakeWorkspaceEntity],
    script: dict[tuple[str, Any], dict[str, Any] | None] | None = None,
    deny_reads: list[str] | None = None,
) -> tuple[WorkspaceDiffTool, _FakePool, _RecordingSandbox]:
    workspaces = _FakeWorkspaceCollection(workspace_entities)
    pool = _FakePool(conn=_FakeConnection(script=dict(script or {})))
    sandbox = _RecordingSandbox(deny_reads=deny_reads)
    tool = WorkspaceDiffTool(
        workspace_collection=workspaces,  # type: ignore[arg-type]
        workspace_file_collection=object(),  # type: ignore[arg-type]
        workspace_file_version_collection=object(),  # type: ignore[arg-type]
        sandbox=sandbox,  # type: ignore[arg-type]
        context_provider=lambda: _FakeContext(),
        agent_id=uuid4(),
        db_pool=pool,
    )
    return tool, pool, sandbox


# ---------------------------------------------------------------------------
# happy paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_diff_two_integer_versions_returns_unified_diff() -> None:
    """from_ref=1, to_ref=3 emit plain unified diff between their content."""
    ws = _FakeWorkspaceEntity(id=uuid4(), name="ws")
    script: dict[tuple[str, Any], dict[str, Any] | None] = {
        ("a.txt", 1): _journal_row(b"first\nsecond\nthird\n", version=1),
        ("a.txt", 3): _journal_row(b"first\nchanged\nthird\n", version=3),
    }
    tool, _pool, sandbox = _build_tool(workspace_entities=[ws], script=script)
    result = await tool.execute(relative_path="a.txt", from_ref=1, to_ref=3, workspace="ws")
    assert result.success is True, result.error
    assert "--- a.txt@1" in result.content
    assert "+++ a.txt@3" in result.content
    assert "-second" in result.content
    assert "+changed" in result.content
    assert sandbox.enforce_calls == [("read", "a.txt")]


@pytest.mark.asyncio
async def test_diff_checkpoint_label_resolves_via_label_selector() -> None:
    """string ref (non-digit, non-head) treated as checkpoint label."""
    ws = _FakeWorkspaceEntity(id=uuid4(), name="ws")
    script = {
        ("m.txt", "v1"): _journal_row(b"alpha\n"),
        ("m.txt", "head"): _journal_row(b"alpha-plus\n"),
    }
    tool, _pool, _ = _build_tool(workspace_entities=[ws], script=script)
    result = await tool.execute(relative_path="m.txt", from_ref="v1", to_ref="head", workspace="ws")
    assert result.success is True
    assert "--- m.txt@v1" in result.content
    assert "+++ m.txt@head" in result.content


# ---------------------------------------------------------------------------
# error paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_diff_binary_content_returns_clean_error() -> None:
    """non-UTF-8 content in either ref triggers clean error text."""
    ws = _FakeWorkspaceEntity(id=uuid4(), name="ws")
    script = {
        ("bin.dat", 1): _journal_row(b"\xff\xfe\x00\x01"),
        ("bin.dat", 2): _journal_row(b"ok text\n"),
    }
    tool, _pool, _ = _build_tool(workspace_entities=[ws], script=script)
    result = await tool.execute(relative_path="bin.dat", from_ref=1, to_ref=2, workspace="ws")
    assert result.success is False
    assert result.error is not None
    assert "diff only supported on text" in result.error


@pytest.mark.asyncio
async def test_diff_missing_from_ref_returns_clean_error() -> None:
    """from_ref resolves to no row -> clean error naming the ref."""
    ws = _FakeWorkspaceEntity(id=uuid4(), name="ws")
    script = {
        ("a.txt", 99): None,
        ("a.txt", "head"): _journal_row(b"x\n"),
    }
    tool, _pool, _ = _build_tool(workspace_entities=[ws], script=script)
    result = await tool.execute(relative_path="a.txt", from_ref=99, to_ref="head", workspace="ws")
    assert result.success is False
    assert result.error is not None
    assert "99" in result.error
    assert "a.txt" in result.error


@pytest.mark.asyncio
async def test_diff_missing_to_ref_returns_clean_error() -> None:
    """to_ref resolves to no row -> clean error naming the ref."""
    ws = _FakeWorkspaceEntity(id=uuid4(), name="ws")
    script = {
        ("a.txt", "head"): _journal_row(b"x\n"),
        ("a.txt", "never"): None,
    }
    tool, _pool, _ = _build_tool(workspace_entities=[ws], script=script)
    result = await tool.execute(relative_path="a.txt", from_ref="head", to_ref="never", workspace="ws")
    assert result.success is False
    assert result.error is not None
    assert "never" in result.error


@pytest.mark.asyncio
async def test_diff_sandbox_denied_returns_clean_error() -> None:
    """SandboxDenied surfaces as clean tool error; no ref queries issued."""
    ws = _FakeWorkspaceEntity(id=uuid4(), name="ws")
    tool, pool, sandbox = _build_tool(workspace_entities=[ws], script={}, deny_reads=["private.env"])
    result = await tool.execute(
        relative_path="private.env",
        from_ref=1,
        to_ref=2,
        workspace="ws",
    )
    assert result.success is False
    assert result.error is not None
    assert sandbox.enforce_calls == [("read", "private.env")]
    assert pool.conn.fetchrows == []


# ---------------------------------------------------------------------------
# MCP surface
# ---------------------------------------------------------------------------


def test_diff_mcp_name_is_exact_string() -> None:
    tool, _, _ = _build_tool(workspace_entities=[_FakeWorkspaceEntity(id=uuid4(), name="ws")])
    assert tool.mcp_name() == "threetears.workspace.diff"


def test_diff_mcp_schema_shape() -> None:
    tool, _, _ = _build_tool(workspace_entities=[_FakeWorkspaceEntity(id=uuid4(), name="ws")])
    defn = tool.mcp_schema()
    assert isinstance(defn, MCPToolDefinition)
    schema = defn.input_schema
    assert schema["required"] == ["relative_path", "from_ref", "to_ref"]
    assert schema["additionalProperties"] is False
    props = schema["properties"]
    assert set(props.keys()) == {
        "relative_path",
        "from_ref",
        "to_ref",
        "workspace",
    }
    assert props["from_ref"]["type"] == ["string", "integer"]
    assert props["to_ref"]["type"] == ["string", "integer"]
