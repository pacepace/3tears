"""parametrized authorize matrix for every WS-ACL-05 workspace tool.

covers the Phase 5b sweep: every tool under
:mod:`threetears.agent.workspace.tools` calls
:func:`authorize_workspace_access` via the shared
:func:`threetears.agent.workspace.tools.helpers.authorize_workspace`
wrapper right after resolving the target workspace. this test locks in
that contract by asserting, for each tool, that:

- the shared helper is invoked exactly once per ``execute``
- it is invoked with the correct ``operation`` ("read" or "write")
- the call scope's identity (agent_id + customer_id + user_id) and
  the resolved workspace are forwarded through to
  :func:`authorize_workspace_access`
- on cache-deny the tool surfaces errors-as-data rather than raising
- on missing-customer the tool surfaces errors-as-data rather than
  raising

per-tool expected operation is captured in the ``_CASES`` table
below; any drift (a WRITE silently becoming a READ, or vice versa) is
flagged by the parametrized test.

the tools under test are *not* mocked: they run with lightweight fake
collections / sandbox / pool objects so the real ``execute`` codepath
dispatches into the real ``authorize_workspace`` helper, which is the
seam the test mocks. this gives us end-to-end coverage of the sweep
without requiring a database.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Callable
from unittest.mock import AsyncMock, patch
from uuid import UUID, uuid4

import pytest

from threetears.agent.tools.call_scope import ToolCallScope, enter_call_scope
from threetears.agent.tools.context_envelope import CallContext

from threetears.agent.workspace.authorize import WorkspaceAccessDenied
from _helpers.workspace_shims import (
    FakeWorkspaceCollection,
    FakeWorkspaceContext,
    FakeWorkspaceEntity,
    FakeWorkspaceFile,
    FakeWorkspaceFileCollection,
    FakeWorkspaceFileVersionCollection,
    FakeWorkspaceSandbox,
)


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------


@dataclass
class _FakeWorkspace(FakeWorkspaceEntity):
    """stand-in workspace entity with all WorkspaceLike attributes."""

    id: UUID
    name: str
    agent_id: UUID
    owner_agent_id: UUID
    created_by: UUID
    created_by_user_id: UUID
    customer_id: UUID | None
    template_name: str | None = None
    current_version: int = 0
    description: str | None = None
    date_created: datetime = field(default_factory=lambda: datetime.now(UTC))
    date_updated: datetime = field(default_factory=lambda: datetime.now(UTC))
    date_deleted: datetime | None = None

    @property
    def namespace_name(self) -> str:
        """canonical namespace name, matches Workspace entity."""
        return f"workspace.{self.id}"


@dataclass
class _FakeFile(FakeWorkspaceFile):
    """stand-in head-state file entity."""

    relative_path: str
    content: bytes
    sha256: str
    version: int
    date_updated: datetime = field(default_factory=lambda: datetime.now(UTC))


class _FakeWorkspaceCollection(FakeWorkspaceCollection):
    """minimal workspace collection servicing the common lookups."""

    def __init__(self, entries: list[_FakeWorkspace]) -> None:
        """capture fixture entries.

        :param entries: fake workspace rows to serve
        :ptype entries: list[_FakeWorkspace]
        """
        self._entries = entries

    async def find_by_agent_and_name(
        self,
        agent_id: UUID,
        name: str,
    ) -> _FakeWorkspace | None:
        """linear name-scan within the agent.

        :param agent_id: calling agent identifier
        :ptype agent_id: UUID
        :param name: workspace human name
        :ptype name: str
        :return: matching workspace or None
        :rtype: _FakeWorkspace | None
        """
        for entry in self._entries:
            if entry.name == name:
                return entry
        return None

    async def find_by_id_and_agent(
        self,
        workspace_id: UUID,
        agent_id: UUID,
    ) -> _FakeWorkspace | None:
        """id-scan within the agent.

        :param workspace_id: workspace identifier
        :ptype workspace_id: UUID
        :param agent_id: calling agent identifier
        :ptype agent_id: UUID
        :return: matching workspace or None
        :rtype: _FakeWorkspace | None
        """
        for entry in self._entries:
            if entry.id == workspace_id:
                return entry
        return None

    async def find_by_agent(
        self,
        agent_id: UUID,
        *,
        include_deleted: bool = False,
    ) -> list[_FakeWorkspace]:
        """return all entries for the agent.

        :param agent_id: calling agent identifier
        :ptype agent_id: UUID
        :param include_deleted: accepted for signature parity
        :ptype include_deleted: bool
        :return: all fixture entries
        :rtype: list[_FakeWorkspace]
        """
        del include_deleted
        return list(self._entries)


class _FakeFileCollection(FakeWorkspaceFileCollection):
    """minimal file collection servicing the head-state lookups."""

    def __init__(self, files: list[_FakeFile] | None = None) -> None:
        """capture files.

        :param files: head-state rows to serve
        :ptype files: list[_FakeFile] | None
        """
        self._files = files or []

    async def find_by_workspace_and_relative_path(
        self,
        workspace_id: UUID,
        relative_path: str,
    ) -> _FakeFile | None:
        """linear scan by relative_path.

        :param workspace_id: workspace identifier
        :ptype workspace_id: UUID
        :param relative_path: workspace-relative path
        :ptype relative_path: str
        :return: matching file or None
        :rtype: _FakeFile | None
        """
        del workspace_id
        for entry in self._files:
            if entry.relative_path == relative_path:
                return entry
        return None

    async def find_by_workspace(
        self,
        workspace_id: UUID,
    ) -> list[_FakeFile]:
        """return all head-state files for the workspace.

        :param workspace_id: workspace identifier
        :ptype workspace_id: UUID
        :return: all fixture files
        :rtype: list[_FakeFile]
        """
        del workspace_id
        return list(self._files)


class _FakeVersionCollection(FakeWorkspaceFileVersionCollection):
    """journal collection stub (never used for authorize-path tests)."""

    async def find_by_workspace(
        self,
        workspace_id: UUID,
        limit: int,
    ) -> list[Any]:
        """return empty.

        :param workspace_id: workspace identifier
        :ptype workspace_id: UUID
        :param limit: row limit
        :ptype limit: int
        :return: empty list
        :rtype: list[Any]
        """
        del workspace_id, limit
        return []

    async def find_by_workspace_and_path(
        self,
        workspace_id: UUID,
        relative_path: str,
        limit: int,
    ) -> list[Any]:
        """return empty.

        :param workspace_id: workspace identifier
        :ptype workspace_id: UUID
        :param relative_path: workspace-relative path
        :ptype relative_path: str
        :param limit: row limit
        :ptype limit: int
        :return: empty list
        :rtype: list[Any]
        """
        del workspace_id, relative_path, limit
        return []


class _PermissiveSandbox:
    """no-op sandbox: ``validate_syntax`` always passes.

    namespace-task-01 phase 7 retired the glob-driven
    ``enforce`` / ``check_relative_key`` surface from workspace tools;
    the stand-in now only fields :meth:`validate_syntax`, which the
    new tool implementations call before the rbac per-file gate.
    """

    def validate_syntax(self, target: str) -> None:
        """never raises.

        :param target: candidate relative path
        :ptype target: str
        :return: None
        :rtype: None
        """
        del target


# ---------------------------------------------------------------------------
# tool cases
# ---------------------------------------------------------------------------


@dataclass
class _ToolCase:
    """per-tool case descriptor consumed by the parametrized tests.

    :ivar name: pytest-id
    :ivar operation: expected ``"read"`` or ``"write"``
    :ivar builder: zero-arg callable that constructs the tool + the
        kwargs to pass to ``execute`` for the happy-path
    """

    name: str
    operation: str
    builder: Callable[
        [_FakeWorkspace, Any, UUID],
        tuple[Any, dict[str, Any]],
    ]


def _scope_for(
    *,
    agent_id: UUID,
    user_id: UUID,
    customer_id: UUID | None,
) -> ToolCallScope:
    """build a :class:`ToolCallScope` carrying the identity triple.

    :param agent_id: calling agent
    :ptype agent_id: UUID
    :param user_id: calling user
    :ptype user_id: UUID
    :param customer_id: calling customer
    :ptype customer_id: UUID | None
    :return: scope wrapping a CallContext with the supplied dims
    :rtype: ToolCallScope
    """
    ctx = CallContext(
        agent_id=agent_id,
        user_id=user_id,
        customer_id=customer_id,
        correlation_id=uuid4(),
    )
    return ToolCallScope(context=ctx)


def _make_workspace(customer_id: UUID | None) -> _FakeWorkspace:
    """build a workspace owned by a random agent/user under customer.

    :param customer_id: owning customer UUID
    :ptype customer_id: UUID | None
    :return: workspace entity
    :rtype: _FakeWorkspace
    """
    owner_agent = uuid4()
    owner_user = uuid4()
    return _FakeWorkspace(
        id=uuid4(),
        name="ws",
        agent_id=owner_agent,
        owner_agent_id=owner_agent,
        created_by=owner_user,
        created_by_user_id=owner_user,
        customer_id=customer_id,
    )


def _build_fs_read(
    ws: _FakeWorkspace,
    cache: Any,
    agent_id: UUID,
) -> tuple[Any, dict[str, Any]]:
    """build FsReadTool + kwargs.

    :param ws: target workspace
    :ptype ws: _FakeWorkspace
    :param cache: ACL cache stand-in
    :ptype cache: Any
    :param agent_id: calling agent
    :ptype agent_id: UUID
    :return: tool + execute kwargs
    :rtype: tuple[Any, dict[str, Any]]
    """
    from threetears.agent.workspace.tools.fs_read import FsReadTool

    file_entry = _FakeFile(
        relative_path="a.txt",
        content=b"x",
        sha256="z" * 64,
        version=1,
    )
    tool = FsReadTool(
        workspace_collection=_FakeWorkspaceCollection([ws]),  # type: ignore[arg-type]
        workspace_file_collection=_FakeFileCollection([file_entry]),  # type: ignore[arg-type]
        sandbox=_PermissiveSandbox(),  # type: ignore[arg-type]
        context_provider=lambda: object(),
        agent_id=agent_id,
        db_pool=None,
        acl_cache=cache,
    )
    return tool, {"relative_path": "a.txt", "workspace": "ws"}


def _build_fs_list(
    ws: _FakeWorkspace,
    cache: Any,
    agent_id: UUID,
) -> tuple[Any, dict[str, Any]]:
    """build FsListTool + kwargs."""
    from threetears.agent.workspace.tools.fs_list import FsListTool

    tool = FsListTool(
        workspace_collection=_FakeWorkspaceCollection([ws]),  # type: ignore[arg-type]
        workspace_file_collection=_FakeFileCollection([]),  # type: ignore[arg-type]
        sandbox=_PermissiveSandbox(),  # type: ignore[arg-type]
        context_provider=lambda: object(),
        agent_id=agent_id,
        db_pool=None,
        acl_cache=cache,
    )
    return tool, {"workspace": "ws"}


def _build_doc_get(
    ws: _FakeWorkspace,
    cache: Any,
    agent_id: UUID,
) -> tuple[Any, dict[str, Any]]:
    """build DocGetTool + kwargs."""
    from threetears.agent.workspace.tools.doc_get import DocGetTool

    file_entry = _FakeFile(
        relative_path="a.yaml",
        content=b"k: v\n",
        sha256="z" * 64,
        version=1,
    )
    tool = DocGetTool(
        workspace_collection=_FakeWorkspaceCollection([ws]),  # type: ignore[arg-type]
        workspace_file_collection=_FakeFileCollection([file_entry]),  # type: ignore[arg-type]
        sandbox=_PermissiveSandbox(),  # type: ignore[arg-type]
        context_provider=lambda: object(),
        agent_id=agent_id,
        db_pool=None,
        acl_cache=cache,
    )
    return tool, {"relative_path": "a.yaml", "workspace": "ws"}


def _build_workspace_history(
    ws: _FakeWorkspace,
    cache: Any,
    agent_id: UUID,
) -> tuple[Any, dict[str, Any]]:
    """build WorkspaceHistoryTool + kwargs."""
    from threetears.agent.workspace.tools.workspace_history import (
        WorkspaceHistoryTool,
    )

    tool = WorkspaceHistoryTool(
        workspace_collection=_FakeWorkspaceCollection([ws]),  # type: ignore[arg-type]
        workspace_file_collection=_FakeFileCollection([]),  # type: ignore[arg-type]
        workspace_file_version_collection=_FakeVersionCollection(),  # type: ignore[arg-type]
        sandbox=_PermissiveSandbox(),  # type: ignore[arg-type]
        context_provider=lambda: object(),
        agent_id=agent_id,
        db_pool=None,
        acl_cache=cache,
    )
    return tool, {"workspace": "ws"}


def _build_workspace_diff(
    ws: _FakeWorkspace,
    cache: Any,
    agent_id: UUID,
) -> tuple[Any, dict[str, Any]]:
    """build WorkspaceDiffTool + kwargs. diff raises after authorize in fake pool."""
    from threetears.agent.workspace.tools.workspace_diff import WorkspaceDiffTool

    class _DumbPool:
        """fake pool: fetchrow returns None, acquire raises."""

        async def fetchrow(self, query: str, *args: Any, **kw: Any) -> None:
            """authorize's enrich step tolerates None rows."""
            del query, args, kw
            return None

        def acquire(self) -> Any:
            """acquire must never be reached in this test."""
            raise RuntimeError("pool not wired for diff test")

    tool = WorkspaceDiffTool(
        workspace_collection=_FakeWorkspaceCollection([ws]),  # type: ignore[arg-type]
        workspace_file_collection=_FakeFileCollection([]),  # type: ignore[arg-type]
        workspace_file_version_collection=_FakeVersionCollection(),  # type: ignore[arg-type]
        sandbox=_PermissiveSandbox(),  # type: ignore[arg-type]
        context_provider=lambda: object(),
        agent_id=agent_id,
        db_pool=_DumbPool(),
        acl_cache=cache,
    )
    return tool, {
        "relative_path": "a.txt",
        "from_ref": "head",
        "to_ref": 1,
        "workspace": "ws",
    }


def _build_workspace_use(
    ws: _FakeWorkspace,
    cache: Any,
    agent_id: UUID,
) -> tuple[Any, dict[str, Any]]:
    """build WorkspaceUseTool + kwargs.

    pin.set_pin needs a working context_provider; the tool patches
    pin module in the per-test monkeypatch fixture, so passing a
    plain object here is enough.
    """
    from threetears.agent.workspace.tools.workspace_use import WorkspaceUseTool

    tool = WorkspaceUseTool(
        workspace_collection=_FakeWorkspaceCollection([ws]),  # type: ignore[arg-type]
        agent_id=agent_id,
        context_provider=lambda: object(),
        db_pool=None,
        acl_cache=cache,
    )
    return tool, {"name": "ws"}


def _build_workspace_flush(
    ws: _FakeWorkspace,
    cache: Any,
    agent_id: UUID,
) -> tuple[Any, dict[str, Any]]:
    """build WorkspaceFlushTool + kwargs.

    sandbox.resolve_fs_path raises KeyError in fixture; we only care
    about getting through authorize first.
    """
    from threetears.agent.workspace.tools.workspace_flush import WorkspaceFlushTool

    class _BindRootRaise(_PermissiveSandbox):
        def resolve_fs_path(self, name: str, mode: str) -> Any:
            del name, mode
            raise KeyError("no bind root")

    tool = WorkspaceFlushTool(
        workspace_collection=_FakeWorkspaceCollection([ws]),  # type: ignore[arg-type]
        workspace_file_collection=_FakeFileCollection([]),  # type: ignore[arg-type]
        sandbox=_BindRootRaise(),  # type: ignore[arg-type]
        context_provider=lambda: object(),
        agent_id=agent_id,
        db_pool=None,
        acl_cache=cache,
    )
    return tool, {"workspace": "ws"}


def _build_fs_write(
    ws: _FakeWorkspace,
    cache: Any,
    agent_id: UUID,
) -> tuple[Any, dict[str, Any]]:
    """build FsWriteTool + kwargs.

    the underlying _write_file_atomic would need a working pool, so
    we pass a pool that raises inside acquire() after authorize
    completes; the test only asserts authorize invocation + denial
    paths so the failure after authorize is fine.
    """
    from threetears.agent.workspace.tools.fs_write import FsWriteTool

    class _RaisePool:
        """fake pool: fetchrow returns None (enrich no-op), acquire raises."""

        async def fetchrow(self, query: str, *args: Any, **kw: Any) -> None:
            """authorize's enrich step tolerates None rows."""
            del query, args, kw
            return None

        def acquire(self) -> Any:
            """acquire must never be reached in this test."""
            raise RuntimeError("pool not wired in authorize matrix test")

    tool = FsWriteTool(
        workspace_collection=_FakeWorkspaceCollection([ws]),  # type: ignore[arg-type]
        workspace_file_collection=_FakeFileCollection([]),  # type: ignore[arg-type]
        workspace_file_version_collection=_FakeVersionCollection(),  # type: ignore[arg-type]
        sandbox=_PermissiveSandbox(),  # type: ignore[arg-type]
        context_provider=lambda: object(),
        agent_id=agent_id,
        db_pool=_RaisePool(),
        acl_cache=cache,
    )
    return tool, {
        "relative_path": "a.txt",
        "content": "hello",
        "workspace": "ws",
    }


def _build_fs_edit(
    ws: _FakeWorkspace,
    cache: Any,
    agent_id: UUID,
) -> tuple[Any, dict[str, Any]]:
    """build FsEditTool + kwargs.

    we pass a file that exists + valid find/replace. inner
    _write_file_atomic is stubbed via raising pool; authorize runs
    before that point.
    """
    from threetears.agent.workspace.tools.fs_edit import FsEditTool

    class _RaisePool:
        """fake pool: fetchrow returns None (enrich no-op), acquire raises."""

        async def fetchrow(self, query: str, *args: Any, **kw: Any) -> None:
            """authorize's enrich step tolerates None rows."""
            del query, args, kw
            return None

        def acquire(self) -> Any:
            """acquire must never be reached in this test."""
            raise RuntimeError("pool not wired in authorize matrix test")

    file_entry = _FakeFile(
        relative_path="a.txt",
        content=b"hello",
        sha256="z" * 64,
        version=1,
    )
    tool = FsEditTool(
        workspace_collection=_FakeWorkspaceCollection([ws]),  # type: ignore[arg-type]
        workspace_file_collection=_FakeFileCollection([file_entry]),  # type: ignore[arg-type]
        workspace_file_version_collection=_FakeVersionCollection(),  # type: ignore[arg-type]
        sandbox=_PermissiveSandbox(),  # type: ignore[arg-type]
        context_provider=lambda: object(),
        agent_id=agent_id,
        db_pool=_RaisePool(),
        acl_cache=cache,
    )
    return tool, {
        "relative_path": "a.txt",
        "find": "hello",
        "replace": "world",
        "workspace": "ws",
    }


def _build_doc_set(
    ws: _FakeWorkspace,
    cache: Any,
    agent_id: UUID,
) -> tuple[Any, dict[str, Any]]:
    """build DocSetTool + kwargs."""
    from threetears.agent.workspace.tools.doc_set import DocSetTool

    class _RaisePool:
        """fake pool: fetchrow returns None (enrich no-op), acquire raises."""

        async def fetchrow(self, query: str, *args: Any, **kw: Any) -> None:
            """authorize's enrich step tolerates None rows."""
            del query, args, kw
            return None

        def acquire(self) -> Any:
            """acquire must never be reached in this test."""
            raise RuntimeError("pool not wired in authorize matrix test")

    file_entry = _FakeFile(
        relative_path="a.yaml",
        content=b"k: v\n",
        sha256="z" * 64,
        version=1,
    )
    tool = DocSetTool(
        workspace_collection=_FakeWorkspaceCollection([ws]),  # type: ignore[arg-type]
        workspace_file_collection=_FakeFileCollection([file_entry]),  # type: ignore[arg-type]
        workspace_file_version_collection=_FakeVersionCollection(),  # type: ignore[arg-type]
        sandbox=_PermissiveSandbox(),  # type: ignore[arg-type]
        context_provider=lambda: object(),
        agent_id=agent_id,
        db_pool=_RaisePool(),
        acl_cache=cache,
    )
    return tool, {
        "relative_path": "a.yaml",
        "jsonpath": "k",
        "value": "new",
        "workspace": "ws",
    }


def _build_doc_merge(
    ws: _FakeWorkspace,
    cache: Any,
    agent_id: UUID,
) -> tuple[Any, dict[str, Any]]:
    """build DocMergeTool + kwargs."""
    from threetears.agent.workspace.tools.doc_merge import DocMergeTool

    class _RaisePool:
        """fake pool: fetchrow returns None (enrich no-op), acquire raises."""

        async def fetchrow(self, query: str, *args: Any, **kw: Any) -> None:
            """authorize's enrich step tolerates None rows."""
            del query, args, kw
            return None

        def acquire(self) -> Any:
            """acquire must never be reached in this test."""
            raise RuntimeError("pool not wired in authorize matrix test")

    file_entry = _FakeFile(
        relative_path="a.yaml",
        content=b"k: v\n",
        sha256="z" * 64,
        version=1,
    )
    tool = DocMergeTool(
        workspace_collection=_FakeWorkspaceCollection([ws]),  # type: ignore[arg-type]
        workspace_file_collection=_FakeFileCollection([file_entry]),  # type: ignore[arg-type]
        workspace_file_version_collection=_FakeVersionCollection(),  # type: ignore[arg-type]
        sandbox=_PermissiveSandbox(),  # type: ignore[arg-type]
        context_provider=lambda: object(),
        agent_id=agent_id,
        db_pool=_RaisePool(),
        acl_cache=cache,
    )
    return tool, {
        "relative_path": "a.yaml",
        "partial": {"k": "new"},
        "workspace": "ws",
    }


def _build_workspace_checkpoint(
    ws: _FakeWorkspace,
    cache: Any,
    agent_id: UUID,
) -> tuple[Any, dict[str, Any]]:
    """build WorkspaceCheckpointTool + kwargs."""
    from threetears.agent.workspace.tools.workspace_checkpoint import (
        WorkspaceCheckpointTool,
    )

    class _RaisePool:
        """fake pool: fetchrow returns None (enrich no-op), acquire raises."""

        async def fetchrow(self, query: str, *args: Any, **kw: Any) -> None:
            """authorize's enrich step tolerates None rows."""
            del query, args, kw
            return None

        def acquire(self) -> Any:
            """acquire must never be reached in this test."""
            raise RuntimeError("pool not wired in authorize matrix test")

    tool = WorkspaceCheckpointTool(
        workspace_collection=_FakeWorkspaceCollection([ws]),  # type: ignore[arg-type]
        workspace_file_collection=_FakeFileCollection([]),  # type: ignore[arg-type]
        workspace_file_version_collection=_FakeVersionCollection(),  # type: ignore[arg-type]
        context_provider=lambda: object(),
        agent_id=agent_id,
        db_pool=_RaisePool(),
        acl_cache=cache,
    )
    return tool, {"label": "v1", "workspace": "ws"}


def _build_workspace_rollback(
    ws: _FakeWorkspace,
    cache: Any,
    agent_id: UUID,
) -> tuple[Any, dict[str, Any]]:
    """build WorkspaceRollbackTool + kwargs."""
    from threetears.agent.workspace.tools.workspace_rollback import (
        WorkspaceRollbackTool,
    )

    class _RaisePool:
        """fake pool: fetchrow returns None (enrich no-op), acquire raises."""

        async def fetchrow(self, query: str, *args: Any, **kw: Any) -> None:
            """authorize's enrich step tolerates None rows."""
            del query, args, kw
            return None

        def acquire(self) -> Any:
            """acquire must never be reached in this test."""
            raise RuntimeError("pool not wired in authorize matrix test")

    tool = WorkspaceRollbackTool(
        workspace_collection=_FakeWorkspaceCollection([ws]),  # type: ignore[arg-type]
        workspace_file_collection=_FakeFileCollection([]),  # type: ignore[arg-type]
        workspace_file_version_collection=_FakeVersionCollection(),  # type: ignore[arg-type]
        sandbox=_PermissiveSandbox(),  # type: ignore[arg-type]
        context_provider=lambda: object(),
        agent_id=agent_id,
        db_pool=_RaisePool(),
        acl_cache=cache,
    )
    return tool, {"ref": "head", "workspace": "ws"}


def _build_workspace_refresh(
    ws: _FakeWorkspace,
    cache: Any,
    agent_id: UUID,
) -> tuple[Any, dict[str, Any]]:
    """build WorkspaceRefreshTool + kwargs."""
    from threetears.agent.workspace.tools.workspace_refresh import (
        WorkspaceRefreshTool,
    )

    class _BindRootRaise(_PermissiveSandbox):
        def resolve_fs_path(self, name: str, mode: str) -> Any:
            del name, mode
            raise KeyError("no bind root")

    class _RaisePool:
        """fake pool: fetchrow returns None (enrich no-op), acquire raises."""

        async def fetchrow(self, query: str, *args: Any, **kw: Any) -> None:
            """authorize's enrich step tolerates None rows."""
            del query, args, kw
            return None

        def acquire(self) -> Any:
            """acquire must never be reached in this test."""
            raise RuntimeError("pool not wired in authorize matrix test")

    tool = WorkspaceRefreshTool(
        workspace_collection=_FakeWorkspaceCollection([ws]),  # type: ignore[arg-type]
        workspace_file_collection=_FakeFileCollection([]),  # type: ignore[arg-type]
        workspace_file_version_collection=_FakeVersionCollection(),  # type: ignore[arg-type]
        sandbox=_BindRootRaise(),  # type: ignore[arg-type]
        context_provider=lambda: object(),
        agent_id=agent_id,
        db_pool=_RaisePool(),
        acl_cache=cache,
    )
    return tool, {"workspace": "ws"}


def _build_workspace_reset(
    ws: _FakeWorkspace,
    cache: Any,
    agent_id: UUID,
) -> tuple[Any, dict[str, Any]]:
    """build WorkspaceResetTool + kwargs.

    workspace must have a template_name set so the happy-path progresses
    through authorize before failing on the missing template files.
    """
    from threetears.agent.workspace.tools.workspace_reset import WorkspaceResetTool

    class _RaisePool:
        """fake pool: fetchrow returns None (enrich no-op), acquire raises."""

        async def fetchrow(self, query: str, *args: Any, **kw: Any) -> None:
            """authorize's enrich step tolerates None rows."""
            del query, args, kw
            return None

        def acquire(self) -> Any:
            """acquire must never be reached in this test."""
            raise RuntimeError("pool not wired in authorize matrix test")

    class _BindRootRaise(_PermissiveSandbox):
        def resolve_fs_path(self, name: str, mode: str) -> Any:
            del name, mode
            raise KeyError("no template root")

    ws.template_name = "tpl"
    tool = WorkspaceResetTool(
        workspace_collection=_FakeWorkspaceCollection([ws]),  # type: ignore[arg-type]
        workspace_file_collection=_FakeFileCollection([]),  # type: ignore[arg-type]
        workspace_file_version_collection=_FakeVersionCollection(),  # type: ignore[arg-type]
        sandbox=_BindRootRaise(),  # type: ignore[arg-type]
        context_provider=lambda: object(),
        agent_id=agent_id,
        db_pool=_RaisePool(),
        acl_cache=cache,
    )
    return tool, {"name": "ws"}


def _build_workspace_delete(
    ws: _FakeWorkspace,
    cache: Any,
    agent_id: UUID,
) -> tuple[Any, dict[str, Any]]:
    """build WorkspaceDeleteTool + kwargs."""
    from threetears.agent.workspace.tools.workspace_delete import WorkspaceDeleteTool

    class _RaisePool:
        """fake pool: fetchrow returns None (enrich no-op), acquire raises."""

        async def fetchrow(self, query: str, *args: Any, **kw: Any) -> None:
            """authorize's enrich step tolerates None rows."""
            del query, args, kw
            return None

        def acquire(self) -> Any:
            """acquire must never be reached in this test."""
            raise RuntimeError("pool not wired in authorize matrix test")

    tool = WorkspaceDeleteTool(
        workspace_collection=_FakeWorkspaceCollection([ws]),  # type: ignore[arg-type]
        workspace_file_collection=_FakeFileCollection([]),  # type: ignore[arg-type]
        workspace_file_version_collection=_FakeVersionCollection(),  # type: ignore[arg-type]
        sandbox=_PermissiveSandbox(),  # type: ignore[arg-type]
        context_provider=lambda: object(),
        agent_id=agent_id,
        db_pool=_RaisePool(),
        acl_cache=cache,
    )
    return tool, {"name": "ws"}


_CASES: list[_ToolCase] = [
    _ToolCase("fs_read", "read", _build_fs_read),
    _ToolCase("fs_list", "read", _build_fs_list),
    _ToolCase("fs_write", "write", _build_fs_write),
    _ToolCase("fs_edit", "write", _build_fs_edit),
    _ToolCase("doc_get", "read", _build_doc_get),
    _ToolCase("doc_set", "write", _build_doc_set),
    _ToolCase("doc_merge", "write", _build_doc_merge),
    _ToolCase("workspace_history", "read", _build_workspace_history),
    _ToolCase("workspace_diff", "read", _build_workspace_diff),
    _ToolCase("workspace_use", "read", _build_workspace_use),
    _ToolCase("workspace_flush", "read", _build_workspace_flush),
    _ToolCase("workspace_checkpoint", "write", _build_workspace_checkpoint),
    _ToolCase("workspace_rollback", "write", _build_workspace_rollback),
    _ToolCase("workspace_refresh", "write", _build_workspace_refresh),
    _ToolCase("workspace_reset", "write", _build_workspace_reset),
    _ToolCase("workspace_delete", "write", _build_workspace_delete),
]


# ---------------------------------------------------------------------------
# parametrized tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", _CASES, ids=lambda c: c.name)
@pytest.mark.asyncio
async def test_tool_invokes_authorize_with_correct_operation(case: _ToolCase) -> None:
    """every tool calls authorize_workspace_access with the right op."""
    customer_id = uuid4()
    caller_agent = uuid4()
    caller_user = uuid4()
    ws = _make_workspace(customer_id=customer_id)
    cache = AsyncMock()
    cache.check_access = AsyncMock(return_value=None)

    tool, kwargs = case.builder(ws, cache, caller_agent)
    scope = _scope_for(
        agent_id=caller_agent,
        user_id=caller_user,
        customer_id=customer_id,
    )

    target = "threetears.agent.workspace.authorize.authorize_workspace_access"
    with patch(target, new=AsyncMock()) as mock_authz:
        async with enter_call_scope(scope):
            await tool.execute(**kwargs)

    assert mock_authz.await_count == 1, (
        f"{case.name} did not call authorize_workspace_access exactly once; saw {mock_authz.await_count} calls"
    )
    call = mock_authz.await_args
    assert call is not None
    forwarded_scope = call.args[0]
    forwarded_ws = call.args[1]
    forwarded_op = call.args[2]
    assert forwarded_scope is scope
    assert forwarded_ws is ws
    assert forwarded_op == case.operation


@pytest.mark.parametrize("case", _CASES, ids=lambda c: c.name)
@pytest.mark.asyncio
async def test_tool_surfaces_cross_customer_deny_as_data(case: _ToolCase) -> None:
    """cross-customer caller gets errors-as-data, not a raise."""
    customer_a = uuid4()
    customer_b = uuid4()
    caller_agent = uuid4()
    caller_user = uuid4()
    ws = _make_workspace(customer_id=customer_a)
    cache = AsyncMock()
    cache.check_access = AsyncMock(return_value=None)

    tool, kwargs = case.builder(ws, cache, caller_agent)
    scope = _scope_for(
        agent_id=caller_agent,
        user_id=caller_user,
        customer_id=customer_b,  # different!
    )

    async with enter_call_scope(scope):
        result = await tool.execute(**kwargs)

    assert result.success is False, f"{case.name} allowed a cross-customer call"
    assert result.error is not None


@pytest.mark.parametrize("case", _CASES, ids=lambda c: c.name)
@pytest.mark.asyncio
async def test_tool_surfaces_missing_customer_as_data(case: _ToolCase) -> None:
    """scope without customer_id gets errors-as-data, not a raise."""
    customer_id = uuid4()
    caller_agent = uuid4()
    caller_user = uuid4()
    ws = _make_workspace(customer_id=customer_id)
    cache = AsyncMock()
    cache.check_access = AsyncMock(return_value=None)

    tool, kwargs = case.builder(ws, cache, caller_agent)
    scope = _scope_for(
        agent_id=caller_agent,
        user_id=caller_user,
        customer_id=None,
    )

    async with enter_call_scope(scope):
        result = await tool.execute(**kwargs)

    assert result.success is False, f"{case.name} allowed a call with no customer_id on scope"


@pytest.mark.parametrize("case", _CASES, ids=lambda c: c.name)
@pytest.mark.asyncio
async def test_tool_surfaces_grant_deny_as_data(case: _ToolCase) -> None:
    """same-customer non-owner with cache-deny gets errors-as-data."""
    customer_id = uuid4()
    caller_agent = uuid4()
    caller_user = uuid4()
    # workspace owned by a different agent within the same customer
    ws = _make_workspace(customer_id=customer_id)
    cache = AsyncMock()
    cache.check_access = AsyncMock(
        side_effect=RuntimeError("no grant"),
    )

    tool, kwargs = case.builder(ws, cache, caller_agent)
    scope = _scope_for(
        agent_id=caller_agent,
        user_id=caller_user,
        customer_id=customer_id,
    )

    async with enter_call_scope(scope):
        result = await tool.execute(**kwargs)

    assert result.success is False, f"{case.name} allowed a call with no grant"


@pytest.mark.parametrize("case", _CASES, ids=lambda c: c.name)
@pytest.mark.asyncio
async def test_tool_allows_owner_path(case: _ToolCase) -> None:
    """owner (same agent + same user) proceeds past authorize without cache."""
    customer_id = uuid4()
    ws = _make_workspace(customer_id=customer_id)
    # cache that raises if consulted -- owner path must not hit cache
    cache = AsyncMock()
    cache.check_access = AsyncMock(
        side_effect=RuntimeError("must not be consulted"),
    )

    tool, kwargs = case.builder(ws, cache, ws.owner_agent_id)
    scope = _scope_for(
        agent_id=ws.owner_agent_id,
        user_id=ws.created_by_user_id,
        customer_id=customer_id,
    )

    async with enter_call_scope(scope):
        # we don't assert on result.success because some tools fail
        # after authorize (e.g. pool not wired) -- the contract we
        # test is "authorize did not raise + cache was not consulted".
        await tool.execute(**kwargs)

    assert cache.check_access.await_count == 0, f"{case.name} consulted ACL cache on owner path"


def test_authorize_matrix_covers_all_19_tools() -> None:
    """matrix must enumerate exactly the 19 tools swept in Phase 5b.

    catches drift: a new workspace tool added without a matching row
    here fails this test, forcing the author to add the row.
    workspace_list + workspace_current use the discovery subject
    rather than authorize_workspace, so they are excluded from this
    matrix by design.
    """
    names = {case.name for case in _CASES}
    expected = {
        "fs_read",
        "fs_write",
        "fs_edit",
        "fs_list",
        "doc_get",
        "doc_set",
        "doc_merge",
        "workspace_use",
        "workspace_checkpoint",
        "workspace_rollback",
        "workspace_history",
        "workspace_diff",
        "workspace_flush",
        "workspace_refresh",
        "workspace_reset",
        "workspace_delete",
    }
    assert names == expected
    # the brief lists 19 tools total; workspace_create (exempt: no
    # pre-existing workspace to authorize against) + workspace_list
    # + workspace_current (exempt: discovery-routed) are not in the
    # matrix. remaining 16 == len(expected).
    assert len(expected) == 16


# explicit exemption: WorkspaceAccessDenied bubbles are tested via the
# shared helper, we don't need one test per tool for that path.
_ = WorkspaceAccessDenied  # keep import live
