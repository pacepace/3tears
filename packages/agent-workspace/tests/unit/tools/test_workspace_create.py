"""tests for ``threetears.workspace.create`` -- WorkspaceCreateTool."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import asyncpg
import pytest

from threetears.agent.tools.base_tool import MCPToolDefinition

from threetears.agent.workspace.tools import workspace_create as workspace_create_module
from threetears.agent.workspace.tools.workspace_create import WorkspaceCreateTool


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------


@dataclass
class _FakeWorkspaceEntity:
    """minimal stand-in for :class:`Workspace` for source-fork lookups."""

    id: UUID
    name: str
    template_name: str | None = None
    date_deleted: Any = None


@dataclass
class _FakeFileEntity:
    """minimal stand-in for :class:`WorkspaceFile`."""

    relative_path: str
    content: bytes
    sha256: str


class _FakeWorkspaceCollection:
    """records lookup calls and serves entities by (agent_id, name)."""

    def __init__(self, entities: list[_FakeWorkspaceEntity] | None = None) -> None:
        self._entities = entities or []
        self.find_by_agent_and_name_calls: list[tuple[UUID, str]] = []

    async def find_by_agent_and_name(self, agent_id: UUID, name: str) -> _FakeWorkspaceEntity | None:
        self.find_by_agent_and_name_calls.append((agent_id, name))
        for e in self._entities:
            if e.name == name:
                return e
        return None


class _FakeFileCollection:
    """records find_by_workspace calls and serves a fixed file list."""

    def __init__(self, files: list[_FakeFileEntity] | None = None) -> None:
        self._files = files or []
        self.find_by_workspace_calls: list[UUID] = []

    async def find_by_workspace(self, workspace_id: UUID) -> list[_FakeFileEntity]:
        self.find_by_workspace_calls.append(workspace_id)
        return list(self._files)


class _FakeVersionCollection:
    """placeholder; tool wires INSERTs through db_pool, not this collection."""


class _RecordingSandbox:
    """captures resolve_fs_path and validate_syntax calls.

    serves a fixed *resolved* templates root for one template name; the
    test passes the on-disk path the directory walk should target, the
    fake records the (path, root_name) call, and returns that exact
    directory regardless of the requested template name. this matches
    how :class:`WorkspaceSandbox.resolve_fs_path` resolves a templates
    sub-path under the registered ``templates`` root in production.

    namespace-task-01 phase 7: the glob-driven ``enforce`` surface is
    retired; the stand-in records syntactic validation calls only.
    """

    def __init__(self, templates_root: Path) -> None:
        self._templates_root = templates_root
        self.resolve_calls: list[tuple[str, str]] = []
        self.syntax_calls: list[str] = []

    def resolve_fs_path(self, path: str, root_name: str) -> Path:
        self.resolve_calls.append((path, root_name))
        return self._templates_root

    def validate_syntax(self, target: str) -> None:
        self.syntax_calls.append(target)


class _NoTemplateSandbox:
    """sandbox that has no templates root; resolve raises KeyError."""

    def resolve_fs_path(self, path: str, root_name: str) -> Path:
        raise KeyError(root_name)

    def validate_syntax(self, target: str) -> None:
        return None


@dataclass
class _FakeTransaction:
    """async-context-manager recorder for conn.transaction()."""

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
    """records execute calls and whether each occurred inside a transaction."""

    executions: list[tuple[str, tuple[Any, ...], bool]] = field(default_factory=list)
    transaction_calls: int = 0
    transactions: list[_FakeTransaction] = field(default_factory=list)
    transaction_open: bool = False
    raise_on: type[BaseException] | None = None

    def transaction(self, namespace: Any = None) -> _FakeTransaction:
        self.transaction_calls += 1
        tx = _FakeTransaction(parent=self)
        self.transactions.append(tx)
        return tx

    async def execute(self, query: str, *args: Any) -> str:
        self.executions.append((query, args, self.transaction_open))
        if self.raise_on is not None:
            cls = self.raise_on
            self.raise_on = None
            raise cls("duplicate key value violates unique constraint")
        return "INSERT 0 1"


@dataclass
class _FakeAcquireCM:
    """async-context-manager wrapper around a single connection."""

    conn: _FakeConnection

    async def __aenter__(self) -> _FakeConnection:
        return self.conn

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        return None


@dataclass
class _FakePool:
    """records acquired connections and dispatches to a single in-test conn."""

    conn: _FakeConnection = field(default_factory=_FakeConnection)
    acquire_calls: int = 0

    def acquire(self) -> _FakeAcquireCM:
        self.acquire_calls += 1
        return _FakeAcquireCM(conn=self.conn)


class _RecordingPin:
    """records pin.set_pin calls."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def set_pin(
        self,
        context: Any,
        workspace_id: UUID,
        workspace_name: str,
        pinned_by_actor_id: UUID,
    ) -> None:
        self.calls.append(
            {
                "context": context,
                "workspace_id": workspace_id,
                "workspace_name": workspace_name,
                "pinned_by_actor_id": pinned_by_actor_id,
            }
        )


class _FakeContext:
    """sentinel context returned by the context_provider closure."""


class _RecordingNatsClient:
    """records every ``publish`` call made by :class:`WorkspaceCreateTool`.

    the tool publishes one ``WorkspaceCreateEvent`` on
    ``{ns}.workspaces.create`` per successful create (post-emit
    re-materialization wave); this fake captures the subject + the
    typed Pydantic message so tests can assert the published shape.
    mirrors the canonical
    :meth:`threetears.nats.NatsClient.publish` signature
    (``subject=`` + ``message=``).
    """

    def __init__(self) -> None:
        self.published: list[tuple[Any, Any]] = []

    async def publish(
        self, *, subject: Any, message: Any, reply_to: Any | None = None,
    ) -> None:
        """record the publish call and return without dispatching."""
        del reply_to
        self.published.append((subject, message))


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def patch_set_pin(monkeypatch: pytest.MonkeyPatch) -> _RecordingPin:
    """replace pin.set_pin with a recorder."""
    rec = _RecordingPin()
    monkeypatch.setattr(workspace_create_module.pin, "set_pin", rec.set_pin)
    return rec


def _build_tool(
    *,
    workspace_collection: Any = None,
    workspace_file_collection: Any = None,
    workspace_file_version_collection: Any = None,
    sandbox: Any = None,
    context_provider: Any = None,
    agent_id: UUID | None = None,
    db_pool: Any = None,
    nats_client: Any = None,
    namespace: str = "aibots",
    customer_id: UUID | None = None,
) -> WorkspaceCreateTool:
    """assemble a tool with sensible default fakes for every dep."""
    return WorkspaceCreateTool(
        workspace_collection=workspace_collection or _FakeWorkspaceCollection(),
        workspace_file_collection=workspace_file_collection or _FakeFileCollection(),
        workspace_file_version_collection=workspace_file_version_collection or _FakeVersionCollection(),
        sandbox=sandbox or _NoTemplateSandbox(),
        context_provider=context_provider or (lambda: _FakeContext()),
        agent_id=agent_id or uuid4(),
        db_pool=db_pool or _FakePool(),
        nats_client=nats_client or _RecordingNatsClient(),
        namespace=namespace,
        customer_id=customer_id,
    )


# ---------------------------------------------------------------------------
# happy-path tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_empty_workspace_inserts_row_pins_and_logs_zero_files(
    patch_set_pin: _RecordingPin,
) -> None:
    """create with no source emits one INSERT into workspaces and pins."""
    agent_id = uuid4()
    pool = _FakePool()
    fake_ctx = _FakeContext()
    tool = _build_tool(
        agent_id=agent_id,
        context_provider=lambda: fake_ctx,
        db_pool=pool,
    )

    result = await tool.execute(name="empty")

    assert result.success is True
    assert result.error is None
    assert "created workspace 'empty'" in result.content

    # exactly one acquire and one transaction wrap
    assert pool.acquire_calls == 1
    assert pool.conn.transaction_calls == 1
    assert pool.conn.transactions[0].entered is True
    assert pool.conn.transactions[0].exited is True

    # only the workspaces INSERT happened
    inserts_into_workspaces = [e for e in pool.conn.executions if "workspaces" in e[0]]
    assert len(inserts_into_workspaces) == 1
    workspaces_sql, workspaces_args, in_tx = inserts_into_workspaces[0]
    assert "INSERT INTO workspaces" in workspaces_sql
    assert in_tx is True
    # current_version=0 for an empty workspace; located at column 7 per insert order
    # (id, agent_id, name, description, template_name, created_by, current_version, ...)
    assert workspaces_args[6] == 0
    # template_name is NULL
    assert workspaces_args[4] is None

    # no file or version inserts
    assert not [e for e in pool.conn.executions if "workspace_files" in e[0]]
    assert not [e for e in pool.conn.executions if "workspace_file_versions" in e[0]]

    # pin set
    assert len(patch_set_pin.calls) == 1
    pin_call = patch_set_pin.calls[0]
    assert pin_call["context"] is fake_ctx
    assert pin_call["workspace_name"] == "empty"
    assert pin_call["pinned_by_actor_id"] == agent_id


@pytest.mark.asyncio
async def test_create_from_template_walks_directory_and_inserts_each_file(
    tmp_path: Path,
    patch_set_pin: _RecordingPin,
) -> None:
    """from_template reads via rglob; INSERTs file head + journal per file; pins."""
    template_dir = tmp_path / "starter"
    template_dir.mkdir()
    (template_dir / "README.md").write_bytes(b"# starter\n")
    (template_dir / "src").mkdir()
    (template_dir / "src" / "main.py").write_bytes(b"print('hi')\n")
    (template_dir / "config.yaml").write_bytes(b"key: value\n")

    sandbox = _RecordingSandbox(templates_root=template_dir)
    pool = _FakePool()
    agent_id = uuid4()
    tool = _build_tool(
        sandbox=sandbox,
        agent_id=agent_id,
        db_pool=pool,
    )

    result = await tool.execute(name="seed", from_template="starter")

    assert result.success is True, result.error
    assert "created workspace 'seed'" in result.content

    # sandbox observed: one resolve and one validate_syntax per file (3 files)
    assert sandbox.resolve_calls == [("starter", "templates")]
    syntax_targets = sorted(sandbox.syntax_calls)
    assert syntax_targets == sorted(["README.md", "config.yaml", "src/main.py"])

    # workspace insert: template_name="starter", current_version=1
    workspaces_inserts = [e for e in pool.conn.executions if "INSERT INTO workspaces" in e[0]]
    assert len(workspaces_inserts) == 1
    _sql, args, _intx = workspaces_inserts[0]
    assert args[4] == "starter"
    # current_version is set to 1 either via the initial INSERT or a later UPDATE
    workspace_id = args[0]

    # 3 file head inserts + 3 journal inserts; each in transaction
    file_inserts = [e for e in pool.conn.executions if "INSERT INTO workspace_files" in e[0]]
    journal_inserts = [e for e in pool.conn.executions if "INSERT INTO workspace_file_versions" in e[0]]
    assert len(file_inserts) == 3
    assert len(journal_inserts) == 3
    for _sql, _args, in_tx in file_inserts + journal_inserts:
        assert in_tx is True

    # journal action="create" for every file, version=1
    journal_actions = {row[1][6] for row in journal_inserts}  # action col
    assert journal_actions == {"create"}
    journal_versions = {row[1][3] for row in journal_inserts}  # version col
    assert journal_versions == {1}
    # workspace_id consistent across journal rows
    assert {row[1][1] for row in journal_inserts} == {workspace_id}
    # relative paths captured as the on-disk POSIX layout
    journal_rel_paths = sorted(row[1][2] for row in journal_inserts)
    assert journal_rel_paths == sorted(["README.md", "config.yaml", "src/main.py"])

    # content sha256 for one of the files matches actual hash
    expected_sha = hashlib.sha256(b"# starter\n").hexdigest()
    readme_journal = next(row for row in journal_inserts if row[1][2] == "README.md")
    assert readme_journal[1][5] == expected_sha
    # contents bound as bytes
    assert readme_journal[1][4] == b"# starter\n"

    # pin recorded
    assert len(patch_set_pin.calls) == 1
    assert patch_set_pin.calls[0]["workspace_name"] == "seed"


@pytest.mark.asyncio
async def test_create_from_workspace_copies_files_and_carries_template_name(
    patch_set_pin: _RecordingPin,
) -> None:
    """from_workspace queries source files and copies their bytes/sha into new rows."""
    source_id = uuid4()
    source = _FakeWorkspaceEntity(id=source_id, name="orig", template_name="starter")
    src_files = [
        _FakeFileEntity(relative_path="a.md", content=b"AAA", sha256="a" * 64),
        _FakeFileEntity(relative_path="b.md", content=b"BBB", sha256="b" * 64),
    ]
    workspaces = _FakeWorkspaceCollection([source])
    files_coll = _FakeFileCollection(src_files)
    pool = _FakePool()
    tool = _build_tool(
        workspace_collection=workspaces,
        workspace_file_collection=files_coll,
        db_pool=pool,
    )

    result = await tool.execute(name="fork", from_workspace="orig")

    assert result.success is True, result.error
    # source lookup happened
    assert files_coll.find_by_workspace_calls == [source_id]

    workspaces_inserts = [e for e in pool.conn.executions if "INSERT INTO workspaces" in e[0]]
    args = workspaces_inserts[0][1]
    # template_name carried over from source
    assert args[4] == "starter"

    # 2 file head + 2 journal rows
    assert len([e for e in pool.conn.executions if "INSERT INTO workspace_files" in e[0]]) == 2
    journal = [e for e in pool.conn.executions if "INSERT INTO workspace_file_versions" in e[0]]
    assert len(journal) == 2
    # journal copied content bytes
    contents = sorted(row[1][4] for row in journal)
    assert contents == sorted([b"AAA", b"BBB"])


@pytest.mark.asyncio
async def test_create_pins_with_fresh_workspace_id(patch_set_pin: _RecordingPin) -> None:
    """pin's workspace_id matches the id passed to the workspace INSERT."""
    pool = _FakePool()
    tool = _build_tool(db_pool=pool)
    await tool.execute(name="x")
    workspaces_args = [e[1] for e in pool.conn.executions if "INSERT INTO workspaces" in e[0]][0]
    workspace_id = workspaces_args[0]
    assert patch_set_pin.calls[0]["workspace_id"] == workspace_id


# ---------------------------------------------------------------------------
# error / validation paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_with_both_sources_returns_clean_error(
    patch_set_pin: _RecordingPin,
) -> None:
    """specifying both from_template and from_workspace yields error-as-data."""
    pool = _FakePool()
    tool = _build_tool(db_pool=pool)
    result = await tool.execute(name="x", from_template="t", from_workspace="w")
    assert result.success is False
    assert result.error is not None
    assert "mutually exclusive" in result.error or "both" in result.error.lower()
    # nothing inserted, no pin
    assert pool.conn.executions == []
    assert patch_set_pin.calls == []


@pytest.mark.asyncio
async def test_create_duplicate_name_returns_clean_error(
    patch_set_pin: _RecordingPin,
) -> None:
    """asyncpg.UniqueViolationError on workspace insert surfaces as ToolResult."""
    pool = _FakePool()
    pool.conn.raise_on = asyncpg.exceptions.UniqueViolationError
    tool = _build_tool(db_pool=pool)
    result = await tool.execute(name="dup")
    assert result.success is False
    assert result.error is not None
    assert "'dup'" in result.error
    assert "already exists" in result.error
    # pin not set on failure
    assert patch_set_pin.calls == []


@pytest.mark.asyncio
async def test_create_from_workspace_unknown_source_returns_error(
    patch_set_pin: _RecordingPin,
) -> None:
    """from_workspace for missing source yields clean error and no inserts."""
    pool = _FakePool()
    tool = _build_tool(
        workspace_collection=_FakeWorkspaceCollection([]),
        db_pool=pool,
    )
    result = await tool.execute(name="x", from_workspace="missing")
    assert result.success is False
    assert result.error is not None
    assert "missing" in result.error
    assert pool.conn.executions == []
    assert patch_set_pin.calls == []


@pytest.mark.asyncio
async def test_create_traps_unexpected_exceptions_as_data(
    patch_set_pin: _RecordingPin,
) -> None:
    """unexpected runtime errors surface as ToolResult(success=False)."""

    class _Boom:
        async def find_by_agent_and_name(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("db blew up")

    pool = _FakePool()
    tool = _build_tool(workspace_collection=_Boom(), db_pool=pool)
    result = await tool.execute(name="x", from_workspace="anything")
    assert result.success is False
    assert result.error is not None
    assert "db blew up" in result.error
    assert patch_set_pin.calls == []


# ---------------------------------------------------------------------------
# MCP surface
# ---------------------------------------------------------------------------


def test_mcp_name_is_exact_string() -> None:
    tool = _build_tool()
    assert tool.mcp_name() == "threetears.workspace.create"


def test_mcp_version_is_semver_string() -> None:
    tool = _build_tool()
    assert tool.mcp_version() == "1.0"


def test_mcp_schema_declares_required_name_and_optional_sources() -> None:
    tool = _build_tool()
    definition = tool.mcp_schema()
    assert isinstance(definition, MCPToolDefinition)
    assert definition.name == "threetears.workspace.create"
    schema = definition.input_schema
    assert schema["type"] == "object"
    props = schema["properties"]
    assert "name" in props
    assert "description" in props
    assert "from_template" in props
    assert "from_workspace" in props
    assert schema["required"] == ["name"]
    assert schema["additionalProperties"] is False
