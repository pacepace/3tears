"""tests for ``threetears.workspace.doc_get`` -- DocGetTool."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from threetears.agent.tools.base_tool import MCPToolDefinition
from threetears.core.security import SandboxDenied

from threetears.agent.workspace.tools.doc_get import DocGetTool


# ---------------------------------------------------------------------------
# fixture path
# ---------------------------------------------------------------------------


_FIXTURE_PATH = Path(__file__).parent.parent / "handlers" / "fixtures" / "audience_settings.yaml"


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------


@dataclass
class _FakeWorkspaceEntity:
    id: UUID
    name: str
    date_deleted: datetime | None = None


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
        self.find_calls: list[tuple[UUID, str]] = []

    async def find_by_workspace_and_relative_path(
        self, workspace_id: UUID, relative_path: str
    ) -> _FakeFileEntity | None:
        self.find_calls.append((workspace_id, relative_path))
        for f in self._files:
            if f.relative_path == relative_path:
                return f
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


# ---------------------------------------------------------------------------
# builder
# ---------------------------------------------------------------------------


def _build_tool(
    *,
    workspace_entities: list[_FakeWorkspaceEntity] | None = None,
    files: list[_FakeFileEntity] | None = None,
    deny_reads: list[str] | None = None,
    agent_id: UUID | None = None,
) -> tuple[DocGetTool, _FakeFileCollection, _RecordingSandbox, UUID]:
    agent_id = agent_id or uuid4()
    workspaces = _FakeWorkspaceCollection(workspace_entities or [_FakeWorkspaceEntity(id=uuid4(), name="ws")])
    file_coll = _FakeFileCollection(files)
    sandbox = _RecordingSandbox(deny_reads=deny_reads)
    tool = DocGetTool(
        workspace_collection=workspaces,  # type: ignore[arg-type]
        workspace_file_collection=file_coll,  # type: ignore[arg-type]
        sandbox=sandbox,  # type: ignore[arg-type]
        context_provider=lambda: _FakeContext(),
        agent_id=agent_id,
    )
    return tool, file_coll, sandbox, agent_id


def _audience_yaml_bytes() -> bytes:
    return _FIXTURE_PATH.read_bytes()


# ---------------------------------------------------------------------------
# whole-document read
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_doc_get_whole_document_returns_dumped_yaml_with_structure() -> None:
    """no jsonpath -> whole document dumped; structural markers survive."""
    ws = _FakeWorkspaceEntity(id=uuid4(), name="ws")
    yaml_bytes = _audience_yaml_bytes()
    file_entity = _FakeFileEntity(
        relative_path="audience_settings.yaml",
        content=yaml_bytes,
        sha256="a" * 64,
        version=3,
    )
    tool, _files, _sandbox, _ = _build_tool(workspace_entities=[ws], files=[file_entity])

    result = await tool.execute(relative_path="audience_settings.yaml", workspace="ws")

    assert result.success is True, result.error
    # top-level key preserved
    assert "audience_units:" in result.content
    # at least one audience unit name preserved
    assert "knowwho_all" in result.content
    # key ordering preserved -- audience_unit appears before vb_candidates
    assert result.content.index("audience_unit:") < result.content.index("vb_candidates:")
    # metadata carries sha and format
    assert result.metadata is not None
    assert result.metadata["sha256"] == "a" * 64
    assert result.metadata["format"] == ".yaml"


# ---------------------------------------------------------------------------
# jsonpath sub-read
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_doc_get_jsonpath_scalar_returns_serialized_scalar() -> None:
    """jsonpath to a scalar returns the JSON-serialized scalar."""
    ws = _FakeWorkspaceEntity(id=uuid4(), name="ws")
    yaml_bytes = _audience_yaml_bytes()
    file_entity = _FakeFileEntity(
        relative_path="audience_settings.yaml",
        content=yaml_bytes,
        sha256="b" * 64,
        version=1,
    )
    tool, _files, _sandbox, _ = _build_tool(workspace_entities=[ws], files=[file_entity])

    result = await tool.execute(
        relative_path="audience_settings.yaml",
        jsonpath="$.audience_units[0].vb_candidates",
        workspace="ws",
    )

    assert result.success is True, result.error
    # scalar -- fixture has vb_candidates: 10 on the first unit
    assert result.content == "10"


@pytest.mark.asyncio
async def test_doc_get_jsonpath_not_found_returns_clean_error() -> None:
    """jsonpath with no matches yields a clean error naming the expression."""
    ws = _FakeWorkspaceEntity(id=uuid4(), name="ws")
    file_entity = _FakeFileEntity(
        relative_path="audience_settings.yaml",
        content=_audience_yaml_bytes(),
        sha256="c" * 64,
        version=1,
    )
    tool, _files, _sandbox, _ = _build_tool(workspace_entities=[ws], files=[file_entity])

    result = await tool.execute(
        relative_path="audience_settings.yaml",
        jsonpath="$.does_not_exist.nope",
        workspace="ws",
    )

    assert result.success is False
    assert result.error is not None
    assert "no matches" in result.error
    assert "does_not_exist" in result.error


# ---------------------------------------------------------------------------
# unknown format
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_doc_get_unknown_format_returns_clean_error() -> None:
    """file whose suffix has no registered handler -> clean error with suffix."""
    ws = _FakeWorkspaceEntity(id=uuid4(), name="ws")
    file_entity = _FakeFileEntity(
        relative_path="notes.txt",
        content=b"plain text",
        sha256="d" * 64,
        version=1,
    )
    tool, _files, _sandbox, _ = _build_tool(workspace_entities=[ws], files=[file_entity])

    result = await tool.execute(relative_path="notes.txt", workspace="ws")

    assert result.success is False
    assert result.error is not None
    assert ".txt" in result.error
    assert "fs_*" in result.error


# ---------------------------------------------------------------------------
# binary file
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_doc_get_binary_file_with_known_suffix_returns_clean_error() -> None:
    """YAML-suffixed file with non-UTF-8 content reports binary cleanly."""
    ws = _FakeWorkspaceEntity(id=uuid4(), name="ws")
    # .yaml suffix so handler dispatch succeeds; bytes are not valid UTF-8
    file_entity = _FakeFileEntity(
        relative_path="rogue.yaml",
        content=b"\xff\xfe\x00\x01",
        sha256="e" * 64,
        version=1,
    )
    tool, _files, _sandbox, _ = _build_tool(workspace_entities=[ws], files=[file_entity])

    result = await tool.execute(relative_path="rogue.yaml", workspace="ws")

    assert result.success is False
    assert result.error is not None
    assert result.error == "doc_get requires text file; got binary"


# ---------------------------------------------------------------------------
# sandbox gate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_doc_get_sandbox_denied_returns_clean_error_no_fetch() -> None:
    """SandboxDenied -> clean error; no fetch attempted (gate-then-act)."""
    ws = _FakeWorkspaceEntity(id=uuid4(), name="ws")
    file_entity = _FakeFileEntity(
        relative_path="secret.yaml",
        content=b"secret: value\n",
        sha256="f" * 64,
        version=1,
    )
    tool, files, sandbox, _ = _build_tool(
        workspace_entities=[ws],
        files=[file_entity],
        deny_reads=["secret.yaml"],
    )

    result = await tool.execute(relative_path="secret.yaml", workspace="ws")

    assert result.success is False
    assert result.error is not None
    assert "secret.yaml" in result.error
    assert sandbox.enforce_calls == [("read", "secret.yaml")]
    # no file lookup happened -- sandbox blocked first
    assert files.find_calls == []


# ---------------------------------------------------------------------------
# missing file
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_doc_get_missing_file_returns_clean_error() -> None:
    """head row absent -> clean error naming file and workspace."""
    ws = _FakeWorkspaceEntity(id=uuid4(), name="ws")
    tool, _files, _sandbox, _ = _build_tool(workspace_entities=[ws], files=[])

    result = await tool.execute(relative_path="missing.yaml", workspace="ws")

    assert result.success is False
    assert result.error is not None
    assert "missing.yaml" in result.error
    assert "'ws'" in result.error


# ---------------------------------------------------------------------------
# MCP surface
# ---------------------------------------------------------------------------


def test_doc_get_mcp_name_is_exact_string() -> None:
    tool, _, _, _ = _build_tool()
    assert tool.mcp_name() == "threetears.workspace.doc_get"


def test_doc_get_mcp_schema_requires_relative_path() -> None:
    tool, _, _, _ = _build_tool()
    defn = tool.mcp_schema()
    assert isinstance(defn, MCPToolDefinition)
    assert defn.input_schema["required"] == ["relative_path"]
    assert defn.input_schema["additionalProperties"] is False
    props = defn.input_schema["properties"]
    assert set(props.keys()) == {"relative_path", "jsonpath", "workspace"}
