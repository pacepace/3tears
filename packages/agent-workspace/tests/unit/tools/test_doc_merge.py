"""tests for ``threetears.workspace.doc_merge`` -- DocMergeTool."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest

from threetears.agent.tools.base_tool import MCPToolDefinition
from threetears.core.security import SandboxDenied

from threetears.agent.workspace.tools.doc_merge import DocMergeTool


# ---------------------------------------------------------------------------
# fixture path
# ---------------------------------------------------------------------------


_FIXTURE_PATH = Path(__file__).parent.parent / "handlers" / "fixtures" / "audience_settings.yaml"


# ---------------------------------------------------------------------------
# fakes (mirror of test_fs_write fakes)
# ---------------------------------------------------------------------------


@dataclass
class _FakeWorkspaceEntity:
    id: UUID
    name: str
    date_deleted: datetime | None = None
    agent_id: UUID = field(default_factory=uuid4)

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


@dataclass
class _FakeFileEntity:
    relative_path: str
    content: bytes
    sha256: str
    version: int
    date_updated: datetime = datetime.now(UTC)


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


class _FakeVersionCollection:
    pass


class _RecordingSandbox:
    def __init__(self, deny_writes: list[str] | None = None) -> None:
        self._deny_writes = set(deny_writes or [])
        self.syntax_calls: list[str] = []

    def validate_syntax(self, target: str) -> None:
        self.syntax_calls.append(target)
        if target in self._deny_writes:
            raise SandboxDenied("access", target, "syntactic deny (test fixture)")


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
    head_row: dict[str, Any] | None = None
    journal_max_version: int = 0
    executions: list[tuple[str, tuple[Any, ...], bool]] = field(default_factory=list)
    fetchrows: list[tuple[str, tuple[Any, ...], bool]] = field(default_factory=list)
    transactions: list[_FakeTransaction] = field(default_factory=list)
    transaction_open: bool = False
    captured_writes: list[bytes] = field(default_factory=list)

    def transaction(self, namespace: Any = None) -> _FakeTransaction:
        tx = _FakeTransaction(parent=self)
        self.transactions.append(tx)
        return tx

    async def execute(self, query: str, *args: Any) -> str:
        self.executions.append((query, args, self.transaction_open))
        if "workspace_file_versions" in query:
            self.captured_writes.append(args[4])
        return "INSERT 0 1"

    async def fetchrow(self, query: str, *args: Any) -> dict[str, Any] | None:
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


# ---------------------------------------------------------------------------
# builder
# ---------------------------------------------------------------------------


def _build_tool(
    *,
    acl_cache: Any,
    workspace_entities: list[_FakeWorkspaceEntity] | None = None,
    files: list[_FakeFileEntity] | None = None,
    head_row: dict[str, Any] | None = None,
    deny_writes: list[str] | None = None,
    agent_id: UUID | None = None,
) -> tuple[DocMergeTool, _FakePool, _RecordingSandbox, UUID]:
    agent_id = agent_id or uuid4()
    ws_entity = workspace_entities[0] if workspace_entities else _FakeWorkspaceEntity(id=uuid4(), name="ws")
    workspaces = _FakeWorkspaceCollection(workspace_entities or [ws_entity])
    file_coll = _FakeFileCollection(files)
    sandbox = _RecordingSandbox(deny_writes=deny_writes)
    pool = _FakePool()
    pool.conn.head_row = head_row
    if head_row is not None and "version" in head_row:
        pool.conn.journal_max_version = int(head_row["version"])
    tool = DocMergeTool(
        workspace_collection=workspaces,  # type: ignore[arg-type]
        workspace_file_collection=file_coll,  # type: ignore[arg-type]
        workspace_file_version_collection=_FakeVersionCollection(),  # type: ignore[arg-type]
        sandbox=sandbox,  # type: ignore[arg-type]
        context_provider=lambda: _FakeContext(),
        agent_id=agent_id,
        db_pool=pool,
        acl_cache=acl_cache,
    )
    return tool, pool, sandbox, agent_id


def _audience_yaml_bytes() -> bytes:
    return _FIXTURE_PATH.read_bytes()


# ---------------------------------------------------------------------------
# happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_doc_merge_adds_new_top_level_key_preserves_structure(
    permissive_acl_cache: MagicMock,
) -> None:
    """merge a new top-level key into audience YAML; existing keys intact."""
    ws = _FakeWorkspaceEntity(id=uuid4(), name="ws")
    initial = _audience_yaml_bytes()
    sha_initial = hashlib.sha256(initial).hexdigest()
    file_entity = _FakeFileEntity(
        relative_path="audience_settings.yaml",
        content=initial,
        sha256=sha_initial,
        version=1,
    )
    head = {"content": initial, "sha256": sha_initial, "version": 1}
    tool, pool, _sandbox, _ = _build_tool(
        workspace_entities=[ws], files=[file_entity], head_row=head, acl_cache=permissive_acl_cache
    )

    result = await tool.execute(
        relative_path="audience_settings.yaml",
        partial={"metadata": {"version": 1, "label": "v1"}},
        workspace="ws",
    )

    assert result.success is True, result.error
    new_text = pool.conn.captured_writes[0].decode("utf-8")
    # existing top-level key preserved
    assert "audience_units:" in new_text
    # existing inner content preserved
    assert "knowwho_all" in new_text
    # new key appears
    assert "metadata:" in new_text
    assert "label: v1" in new_text
    # content message reports 1 top-level key merged
    assert "merged 1 top-level keys" in result.content


@pytest.mark.asyncio
async def test_doc_merge_replaces_list_wholesale_per_handler_contract(
    permissive_acl_cache: MagicMock,
) -> None:
    """list-valued merge replaces wholesale per YamlHandler semantics.

    YamlHandler's merge substitutes lists (and scalars) wholesale rather
    than concatenating -- callers wanting surgical list edits use doc_set.
    """
    ws = _FakeWorkspaceEntity(id=uuid4(), name="ws")
    initial = _audience_yaml_bytes()
    sha_initial = hashlib.sha256(initial).hexdigest()
    file_entity = _FakeFileEntity(
        relative_path="audience_settings.yaml",
        content=initial,
        sha256=sha_initial,
        version=1,
    )
    head = {"content": initial, "sha256": sha_initial, "version": 1}
    tool, pool, _sandbox, _ = _build_tool(
        workspace_entities=[ws], files=[file_entity], head_row=head, acl_cache=permissive_acl_cache
    )

    result = await tool.execute(
        relative_path="audience_settings.yaml",
        partial={"audience_units": [{"audience_unit": "only_one", "vb_candidates": 1}]},
        workspace="ws",
    )

    assert result.success is True, result.error
    new_text = pool.conn.captured_writes[0].decode("utf-8")
    # only_one present
    assert "only_one" in new_text
    # pre-existing list entries gone (replaced wholesale)
    assert "knowwho_all" not in new_text
    assert "exec_logic" not in new_text
    assert "donors" not in new_text


@pytest.mark.asyncio
async def test_doc_merge_deep_merges_nested_mappings(
    permissive_acl_cache: MagicMock,
) -> None:
    """mapping-in-mapping merges recursively rather than replacing."""
    initial_yaml = ("settings:\n    existing: keep-me\n    nested:\n        a: 1\n").encode("utf-8")
    ws = _FakeWorkspaceEntity(id=uuid4(), name="ws")
    sha_initial = hashlib.sha256(initial_yaml).hexdigest()
    file_entity = _FakeFileEntity(
        relative_path="settings.yaml",
        content=initial_yaml,
        sha256=sha_initial,
        version=1,
    )
    head = {"content": initial_yaml, "sha256": sha_initial, "version": 1}
    tool, pool, _sandbox, _ = _build_tool(
        workspace_entities=[ws], files=[file_entity], head_row=head, acl_cache=permissive_acl_cache
    )

    result = await tool.execute(
        relative_path="settings.yaml",
        partial={"settings": {"nested": {"b": 2}}},
        workspace="ws",
    )

    assert result.success is True, result.error
    new_text = pool.conn.captured_writes[0].decode("utf-8")
    # pre-existing mapping keys preserved (deep merge, not replace)
    assert "existing: keep-me" in new_text
    assert "a: 1" in new_text
    # new inner key added
    assert "b: 2" in new_text


# ---------------------------------------------------------------------------
# non-dict partial
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_doc_merge_partial_not_a_dict_returns_clean_error(
    permissive_acl_cache: MagicMock,
) -> None:
    """non-dict partial rejected at the tool boundary with clean error."""
    ws = _FakeWorkspaceEntity(id=uuid4(), name="ws")
    initial = _audience_yaml_bytes()
    sha_initial = hashlib.sha256(initial).hexdigest()
    file_entity = _FakeFileEntity(
        relative_path="audience_settings.yaml",
        content=initial,
        sha256=sha_initial,
        version=1,
    )
    tool, pool, _sandbox, _ = _build_tool(
        workspace_entities=[ws], files=[file_entity], acl_cache=permissive_acl_cache
    )

    # tool called directly with a scalar; schema validator is upstream of
    # .execute in normal flows but direct call must still reject cleanly.
    result = await tool.execute(
        relative_path="audience_settings.yaml",
        partial="not a dict",
        workspace="ws",
    )

    assert result.success is False
    assert result.error is not None
    assert "partial" in result.error
    # no DB activity
    assert pool.conn.executions == []
    assert pool.conn.fetchrows == []


# ---------------------------------------------------------------------------
# sandbox gate ordering
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_doc_merge_sandbox_denied_returns_clean_error_no_writes(
    permissive_acl_cache: MagicMock,
) -> None:
    """SandboxDenied -> clean error; no DB reads or writes happened."""
    ws = _FakeWorkspaceEntity(id=uuid4(), name="ws")
    file_entity = _FakeFileEntity(
        relative_path="secret.yaml",
        content=b"k: v\n",
        sha256="a" * 64,
        version=1,
    )
    tool, pool, sandbox, _ = _build_tool(
        workspace_entities=[ws],
        files=[file_entity],
        deny_writes=["secret.yaml"],
        acl_cache=permissive_acl_cache,
    )

    result = await tool.execute(
        relative_path="secret.yaml",
        partial={"k": "new"},
        workspace="ws",
    )

    assert result.success is False
    assert result.error is not None
    assert "secret.yaml" in result.error
    assert sandbox.syntax_calls == ["secret.yaml"]
    assert pool.conn.executions == []
    assert pool.conn.fetchrows == []


# ---------------------------------------------------------------------------
# OCC failure
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_doc_merge_stale_expected_sha_returns_mismatch_error(
    permissive_acl_cache: MagicMock,
) -> None:
    """expected_sha256 != current sha -> clean error naming current sha."""
    ws = _FakeWorkspaceEntity(id=uuid4(), name="ws")
    initial = _audience_yaml_bytes()
    sha_current = hashlib.sha256(initial).hexdigest()
    file_entity = _FakeFileEntity(
        relative_path="audience_settings.yaml",
        content=initial,
        sha256=sha_current,
        version=1,
    )
    head = {"content": initial, "sha256": sha_current, "version": 1}
    tool, pool, _sandbox, _ = _build_tool(
        workspace_entities=[ws], files=[file_entity], head_row=head, acl_cache=permissive_acl_cache
    )

    result = await tool.execute(
        relative_path="audience_settings.yaml",
        partial={"meta": {"a": 1}},
        expected_sha256="d" * 64,
        workspace="ws",
    )

    assert result.success is False
    assert result.error is not None
    assert "sha256 mismatch" in result.error
    assert sha_current in result.error
    assert pool.conn.executions == []
    assert len(pool.conn.fetchrows) == 1


# ---------------------------------------------------------------------------
# unknown format
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_doc_merge_unknown_format_returns_clean_error(
    permissive_acl_cache: MagicMock,
) -> None:
    """file with unregistered suffix -> clean error; no DB activity."""
    ws = _FakeWorkspaceEntity(id=uuid4(), name="ws")
    file_entity = _FakeFileEntity(
        relative_path="notes.txt",
        content=b"plain",
        sha256="b" * 64,
        version=1,
    )
    tool, pool, _sandbox, _ = _build_tool(
        workspace_entities=[ws], files=[file_entity], acl_cache=permissive_acl_cache
    )

    result = await tool.execute(
        relative_path="notes.txt",
        partial={"x": 1},
        workspace="ws",
    )

    assert result.success is False
    assert result.error is not None
    assert "notes.txt" in result.error
    assert "fs_*" in result.error
    assert pool.conn.executions == []
    assert pool.conn.fetchrows == []


# ---------------------------------------------------------------------------
# missing file
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_doc_merge_missing_file_returns_clean_error(
    permissive_acl_cache: MagicMock,
) -> None:
    """no head row -> clean error naming file and workspace, no writes."""
    ws = _FakeWorkspaceEntity(id=uuid4(), name="ws")
    tool, pool, _sandbox, _ = _build_tool(
        workspace_entities=[ws], files=[], acl_cache=permissive_acl_cache
    )

    result = await tool.execute(
        relative_path="audience_settings.yaml",
        partial={"x": 1},
        workspace="ws",
    )

    assert result.success is False
    assert result.error is not None
    assert "audience_settings.yaml" in result.error
    assert "'ws'" in result.error
    assert pool.conn.executions == []


# ---------------------------------------------------------------------------
# single-transaction wrapping
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_doc_merge_writes_inside_single_transaction(
    permissive_acl_cache: MagicMock,
) -> None:
    """every execute + fetchrow happens inside one transaction."""
    ws = _FakeWorkspaceEntity(id=uuid4(), name="ws")
    initial = _audience_yaml_bytes()
    sha_initial = hashlib.sha256(initial).hexdigest()
    file_entity = _FakeFileEntity(
        relative_path="audience_settings.yaml",
        content=initial,
        sha256=sha_initial,
        version=1,
    )
    head = {"content": initial, "sha256": sha_initial, "version": 1}
    tool, pool, _sandbox, _ = _build_tool(
        workspace_entities=[ws], files=[file_entity], head_row=head, acl_cache=permissive_acl_cache
    )

    await tool.execute(
        relative_path="audience_settings.yaml",
        partial={"meta": {"a": 1}},
        workspace="ws",
    )

    assert len(pool.conn.transactions) == 1
    assert pool.conn.transactions[0].entered is True
    assert pool.conn.transactions[0].exited is True
    assert all(in_tx for _s, _a, in_tx in pool.conn.executions)
    assert all(in_tx for _s, _a, in_tx in pool.conn.fetchrows)


# ---------------------------------------------------------------------------
# MCP surface
# ---------------------------------------------------------------------------


def test_doc_merge_mcp_name_is_exact_string(
    permissive_acl_cache: MagicMock,
) -> None:
    tool, _, _, _ = _build_tool(acl_cache=permissive_acl_cache)
    assert tool.mcp_name() == "threetears.workspace.doc_merge"


def test_doc_merge_mcp_schema_requires_path_and_partial(
    permissive_acl_cache: MagicMock,
) -> None:
    tool, _, _, _ = _build_tool(acl_cache=permissive_acl_cache)
    defn = tool.mcp_schema()
    assert isinstance(defn, MCPToolDefinition)
    assert defn.input_schema["required"] == ["relative_path", "partial"]
    assert defn.input_schema["additionalProperties"] is False
    props = defn.input_schema["properties"]
    assert set(props.keys()) == {
        "relative_path",
        "partial",
        "expected_sha256",
        "workspace",
    }
    # partial is constrained to object type
    assert props["partial"]["type"] == "object"
