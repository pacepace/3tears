"""workspace entity + collection parity bases for the test tree.

each workspace test that exercises code reading workspace state used
to declare its own ``_FakeWorkspace`` / ``_FakeFile`` /
``_FakeWorkspaceCollection`` / ``_FakeFileCollection`` /
``_FakeVersionCollection`` / ``_FakeContext`` / ``_FakeSandbox``
inline. the per-test variation lived in the data the test seeded
into the fake; the surface (the methods the code under test calls)
was uniform across files.

these classes serve as the canonical PARITY BASES every per-test
shell subclasses. each base is a thin marker; subclass declarations
satisfy the fake-protocol-parity walker without forcing each test's
custom seeding logic into a shared module. the production class each
shell parallels is named in the docstring.

# parity-with: production-class deliberately not declared
----------------------------------------------------------

production collections (``WorkspaceCollection``,
``WorkspaceFileCollection``, etc.) carry many methods workspace
tests never exercise (``list_by_X``, ``count_X``, etc). the parity
walker compares the fake to these subset bases rather than to the
full production class so each test's subclass declaration honestly
reflects the methods that test exercises.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from threetears.agent.workspace.pin import PinContext

__all__ = [
    "FakeWorkspaceContext",
    "FakeWorkspaceEntity",
    "FakeWorkspaceFile",
    "FakeWorkspaceFileCollection",
    "FakeWorkspaceCollection",
    "FakeWorkspaceFileLease",
    "FakeWorkspaceFileLeaseHandle",
    "FakeWorkspaceFileVersionCollection",
    "FakeWorkspaceSandbox",
]


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


# parity-exempt: parity base for workspace test subclasses; the production Workspace entity has many fields tests do not read, so the base is attribute-shape only
class FakeWorkspaceEntity:
    """parity base for the production :class:`Workspace` entity.

    workspace tests construct ``_FakeWorkspace`` / ``_FakeWorkspaceEntity``
    instances exposing only the attributes the code under test reads
    (``id`` / ``name`` / ``agent_id`` / ``date_deleted`` / etc.). the
    base is a marker; subclasses declare the attribute set per their
    test's needs.
    """


# parity-exempt: parity base for workspace test subclasses; the production WorkspaceFile entity has many fields tests do not read, so the base is attribute-shape only
class FakeWorkspaceFile:
    """parity base for the production :class:`WorkspaceFile` entity.

    workspace tests use ``_FakeFile`` / ``_FakeFileEntity`` to model
    head-state file rows in unit tests. subclasses provide the
    attribute surface (``relative_path`` / ``content`` / ``sha256`` /
    ``version``).
    """


# parity-exempt: parity base for workspace test subclasses; the production WorkspaceCollection has many lookup methods tests never exercise so subset parity is intentional
class FakeWorkspaceCollection:
    """parity base for the production :class:`WorkspaceCollection`.

    subclasses provide ``find_by_id`` / ``find_by_agent_and_name`` /
    ``find_by_id_and_agent`` / ``find_by_agent`` per their test's
    needs. the production collection has more lookup methods workspace
    tests never call.
    """


# parity-exempt: parity base for workspace test subclasses; the production WorkspaceFileCollection has many lookup methods tests never exercise so subset parity is intentional
class FakeWorkspaceFileCollection:
    """parity base for the production :class:`WorkspaceFileCollection`.

    subclasses provide ``find_by_workspace`` /
    ``find_by_workspace_and_relative_path`` and the various lookup
    methods individual tests need.
    """


# parity-exempt: parity base for workspace test subclasses; the production WorkspaceFileVersionCollection has many lookup methods tests never exercise so subset parity is intentional
class FakeWorkspaceFileVersionCollection:
    """parity base for the production :class:`WorkspaceFileVersionCollection`.

    most tests don't actually call methods on the version collection
    (it's passed through bind/capture without being read), so many
    subclasses are body-empty markers. subclasses that DO exercise
    the version collection provide the journal-row lookup methods
    inline.
    """


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


# parity-exempt: parity base for workspace test subclasses; production WorkspaceSandbox has path-resolution behaviour tests short-circuit to tmp_path
class FakeWorkspaceSandbox:
    """parity base for the production :class:`WorkspaceSandbox`.

    workspace tests use ``_FakeSandbox`` to short-circuit the path
    resolution that the production sandbox does against the bind
    root. subclasses provide ``resolve_fs_path`` and the path
    validators individual tests exercise.
    """
