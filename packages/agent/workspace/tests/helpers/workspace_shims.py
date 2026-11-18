"""workspace entity + collection parity bases for the test tree.

each workspace test that exercises code reading workspace state used
to declare its own ``_FakeWorkspace`` / ``_FakeFile`` /
``_FakeWorkspaceCollection`` / ``_FakeFileCollection`` /
``_FakeVersionCollection`` / ``_FakeContext`` / ``_FakeSandbox``
inline. the per-test variation lived in the data the test seeded
into the fake; the surface (the methods the code under test calls)
was uniform across files.

the entities are the real ones: :func:`make_workspace`, :func:`make_workspace_file` and
:func:`make_workspace_file_version` build a production entity holding the row a test names,
with no collection behind it. the collection and sandbox bases ARE the production classes,
built without their wiring (no registry, pool or cache), so a test subclass goes wherever the
production class is taken and overrides only the methods its code under test calls; mypy
checks each override against production. the lease bases stay markers: the production lease
surface is a partial subset the tests canonicalize.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from threetears.agent.workspace.collections import (
    WorkspaceCollection,
    WorkspaceFileCollection,
    WorkspaceFileVersionCollection,
)
from threetears.agent.workspace.entities import Workspace, WorkspaceFile, WorkspaceFileVersion
from threetears.agent.workspace.pin import PinContext
from threetears.agent.workspace.sandbox import WorkspaceSandbox

__all__ = [
    "FakeWorkspaceContext",
    "FakeWorkspaceFileCollection",
    "FakeWorkspaceCollection",
    "FakeWorkspaceFileLease",
    "FakeWorkspaceFileLeaseHandle",
    "FakeWorkspaceFileVersionCollection",
    "FakeWorkspaceSandbox",
    "make_workspace",
    "make_workspace_file",
    "make_workspace_file_version",
]

#: the instant a test entity is dated at when the test does not say
_EPOCH = datetime(2026, 1, 1, tzinfo=UTC)


def make_workspace(
    *,
    id: UUID | None = None,
    name: str = "ws",
    agent_id: UUID | None = None,
    created_by: UUID | None = None,
    current_version: int = 0,
    description: str | None = None,
    template_name: str | None = None,
    date_created: datetime = _EPOCH,
    date_updated: datetime = _EPOCH,
    date_deleted: datetime | None = None,
    customer_id: UUID | None = None,
) -> Workspace:
    """a real :class:`Workspace` entity holding the row a test names, with no collection behind it.

    an entity built with no collection keeps its row in memory (see ``unit/test_entities.py``), so
    the code under test reads it exactly as it reads a row a collection loaded.

    :param id: workspace id; a fresh one when omitted
    :ptype id: UUID | None
    :param name: workspace name
    :ptype name: str
    :param agent_id: owning agent id; a fresh one when omitted
    :ptype agent_id: UUID | None
    :param created_by: creating user id; a fresh one when omitted
    :ptype created_by: UUID | None
    :param current_version: workspace version counter
    :ptype current_version: int
    :param description: optional description
    :ptype description: str | None
    :param template_name: template the workspace was created from
    :ptype template_name: str | None
    :param date_created: creation instant
    :ptype date_created: datetime
    :param date_updated: last-update instant
    :ptype date_updated: datetime
    :param date_deleted: soft-delete instant, None while live
    :ptype date_deleted: datetime | None
    :param customer_id: resolved owning customer, None when unresolved
    :ptype customer_id: UUID | None
    :return: the entity
    :rtype: Workspace
    """
    return Workspace(
        {
            "workspace_id": id if id is not None else uuid4(),
            "agent_id": agent_id if agent_id is not None else uuid4(),
            "name": name,
            "description": description,
            "template_name": template_name,
            "created_by": created_by if created_by is not None else uuid4(),
            "current_version": current_version,
            "date_created": date_created,
            "date_updated": date_updated,
            "date_deleted": date_deleted,
            "customer_id": customer_id,
        }
    )


def make_workspace_file(
    *,
    relative_path: str,
    content: bytes = b"",
    sha256: str | None = None,
    version: int = 1,
    workspace_id: UUID | None = None,
    id: UUID | None = None,
    date_updated: datetime = _EPOCH,
) -> WorkspaceFile:
    """a real :class:`WorkspaceFile` head row with no collection behind it.

    :param relative_path: the file's workspace-relative path
    :ptype relative_path: str
    :param content: file bytes
    :ptype content: bytes
    :param sha256: content digest; the digest of ``content`` when omitted
    :ptype sha256: str | None
    :param version: file version
    :ptype version: int
    :param workspace_id: owning workspace; a fresh id when omitted
    :ptype workspace_id: UUID | None
    :param id: file id; a fresh one when omitted
    :ptype id: UUID | None
    :param date_updated: last-update instant
    :ptype date_updated: datetime
    :return: the entity
    :rtype: WorkspaceFile
    """
    return WorkspaceFile(
        {
            "file_id": id if id is not None else uuid4(),
            "workspace_id": workspace_id if workspace_id is not None else uuid4(),
            "relative_path": relative_path,
            "content": content,
            "sha256": sha256 if sha256 is not None else hashlib.sha256(content).hexdigest(),
            "version": version,
            "date_updated": date_updated,
        }
    )


def make_workspace_file_version(
    *,
    relative_path: str,
    version: int,
    content: bytes = b"",
    sha256: str | None = None,
    action: str = "update",
    label: str | None = None,
    actor_id: UUID | None = None,
    correlation_id: UUID | None = None,
    workspace_id: UUID | None = None,
    id: UUID | None = None,
    date_created: datetime = _EPOCH,
) -> WorkspaceFileVersion:
    """a real :class:`WorkspaceFileVersion` journal row with no collection behind it.

    :param relative_path: the file's workspace-relative path
    :ptype relative_path: str
    :param version: the version this row records
    :ptype version: int
    :param content: file bytes at that version
    :ptype content: bytes
    :param sha256: content digest; the digest of ``content`` when omitted
    :ptype sha256: str | None
    :param action: journal action (``create`` / ``update`` / ``delete`` / ...)
    :ptype action: str
    :param label: checkpoint label, if any
    :ptype label: str | None
    :param actor_id: acting principal; a fresh id when omitted
    :ptype actor_id: UUID | None
    :param correlation_id: the write's correlation id; a fresh one when omitted
    :ptype correlation_id: UUID | None
    :param workspace_id: owning workspace; a fresh id when omitted
    :ptype workspace_id: UUID | None
    :param id: version-row id; a fresh one when omitted
    :ptype id: UUID | None
    :param date_created: when the row was journaled
    :ptype date_created: datetime
    :return: the entity
    :rtype: WorkspaceFileVersion
    """
    return WorkspaceFileVersion(
        {
            "version_id": id if id is not None else uuid4(),
            "workspace_id": workspace_id if workspace_id is not None else uuid4(),
            "relative_path": relative_path,
            "version": version,
            "content": content,
            "sha256": sha256 if sha256 is not None else hashlib.sha256(content).hexdigest(),
            "action": action,
            "label": label,
            "actor_id": actor_id if actor_id is not None else uuid4(),
            "correlation_id": correlation_id if correlation_id is not None else uuid4(),
            "date_created": date_created,
        }
    )


# parity-exempt: parity base for workspace test subclasses; the production WorkspaceFileLease surface is a partial subset the bases canonicalize
class FakeWorkspaceFileLease:
    """parity base for the production :class:`WorkspaceFileLease`.

    workspace tests use ``_FakeLease`` to model the lease coordinator
    that bind/capture acquire before mutating workspace files.
    subclasses provide ``acquire`` returning a
    :class:`FakeWorkspaceFileLeaseHandle` per their test's
    counter / race-detection setup.
    """


# parity-exempt: parity base for workspace test subclasses; the production LeaseHandle async-CM surface is dunder-only with no public parity to assert
class FakeWorkspaceFileLeaseHandle:
    """parity base for the production :class:`LeaseHandle`.

    async-context-manager handle that workspace tests use to model
    the per-call lease that bind / capture hold during their
    critical sections. subclasses customise ``__aenter__`` /
    ``__aexit__`` for race-detection counters.
    """


class FakeWorkspaceCollection(WorkspaceCollection):
    """the production :class:`WorkspaceCollection` with no registry, pool or cache behind it.

    a test subclass overrides the lookups its code under test calls (``find_by_id`` /
    ``find_by_agent_and_name`` / ``find_by_id_and_agent`` / ``find_by_agent``) and serves real
    :class:`Workspace` entities (:func:`make_workspace`); being the production class, it goes
    wherever the production collection is taken.
    """

    def __init__(self) -> None:
        """build the fake without the production wiring: no registry, config, pool or L1 table."""


class FakeWorkspaceFileCollection(WorkspaceFileCollection):
    """the production :class:`WorkspaceFileCollection` with no registry, pool or cache behind it.

    a test subclass overrides ``find_by_workspace`` / ``find_by_workspace_and_relative_path`` and
    the writes its code under test makes, serving real :class:`WorkspaceFile` entities
    (:func:`make_workspace_file`).
    """

    def __init__(self) -> None:
        """build the fake without the production wiring: no registry, config, pool or L1 table."""


class FakeWorkspaceFileVersionCollection(WorkspaceFileVersionCollection):
    """the production :class:`WorkspaceFileVersionCollection` with no registry, pool or cache behind it.

    most tests pass the version collection through bind/capture without reading it, so many
    subclasses are body-empty; those that read the journal override its lookups.
    """

    def __init__(self) -> None:
        """build the fake without the production wiring: no registry, config, pool or L1 table."""


class FakeWorkspaceContext(PinContext):
    """an in-memory :class:`~threetears.agent.workspace.pin.PinContext`: the conversation context a pin lives in.

    what workspace tools and the pin functions take of the conversation's ``ToolContextManager``.
    items are held by ``(context_type, key)`` in :attr:`items`; a test that patches the pin functions
    passes one as a context nothing reads.
    """

    def __init__(self) -> None:
        self.items: dict[tuple[str, str], dict[str, Any]] = {}

    async def get_item_by_type_and_key(self, context_type: str, key: str) -> dict[str, Any] | None:
        item = self.items.get((context_type, key))
        return dict(item) if item is not None else None

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
        item_id = str(uuid4())
        self.items[(context_type, key)] = {
            "context_id": item_id,
            "context_type": context_type,
            "key": key,
            "content": content,
            "short_desc": short_desc,
            "long_desc": long_desc,
            "metadata": metadata,
        }
        return item_id

    async def delete_item_by_type_and_key(self, context_type: str, key: str) -> bool:
        return self.items.pop((context_type, key), None) is not None


class FakeWorkspaceSandbox(WorkspaceSandbox):
    """the production :class:`WorkspaceSandbox` with no filesystem roots.

    workspace tests use ``_FakeSandbox`` to short-circuit the path resolution the production
    sandbox does against the bind root: subclasses override ``resolve_fs_path`` and the path
    validators individual tests exercise.
    """

    def __init__(self) -> None:
        """build the sandbox with no roots and empty allow-lists, as ``from_config`` does for an empty config."""
        super().__init__(fs_roots={}, allow_read=[], allow_write=[])
