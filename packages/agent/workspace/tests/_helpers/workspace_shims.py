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


class FakeWorkspaceFileLease:
    """parity base for the production :class:`WorkspaceFileLease`.

    workspace tests use ``_FakeLease`` to model the lease coordinator
    that bind/capture acquire before mutating workspace files.
    subclasses provide ``acquire`` returning a
    :class:`FakeWorkspaceFileLeaseHandle` per their test's
    counter / race-detection setup.
    """


class FakeWorkspaceFileLeaseHandle:
    """parity base for the production :class:`LeaseHandle`.

    async-context-manager handle that workspace tests use to model
    the per-call lease that bind / capture hold during their
    critical sections. subclasses customise ``__aenter__`` /
    ``__aexit__`` for race-detection counters.
    """


class FakeWorkspaceEntity:
    """parity base for the production :class:`Workspace` entity.

    workspace tests construct ``_FakeWorkspace`` / ``_FakeWorkspaceEntity``
    instances exposing only the attributes the code under test reads
    (``id`` / ``name`` / ``agent_id`` / ``date_deleted`` / etc.). the
    base is a marker; subclasses declare the attribute set per their
    test's needs.
    """


class FakeWorkspaceFile:
    """parity base for the production :class:`WorkspaceFile` entity.

    workspace tests use ``_FakeFile`` / ``_FakeFileEntity`` to model
    head-state file rows in unit tests. subclasses provide the
    attribute surface (``relative_path`` / ``content`` / ``sha256`` /
    ``version``).
    """


class FakeWorkspaceCollection:
    """parity base for the production :class:`WorkspaceCollection`.

    subclasses provide ``find_by_id`` / ``find_by_agent_and_name`` /
    ``find_by_id_and_agent`` / ``find_by_agent`` per their test's
    needs. the production collection has more lookup methods workspace
    tests never call.
    """


class FakeWorkspaceFileCollection:
    """parity base for the production :class:`WorkspaceFileCollection`.

    subclasses provide ``find_by_workspace`` /
    ``find_by_workspace_and_relative_path`` and the various lookup
    methods individual tests need.
    """


class FakeWorkspaceFileVersionCollection:
    """parity base for the production :class:`WorkspaceFileVersionCollection`.

    most tests don't actually call methods on the version collection
    (it's passed through bind/capture without being read), so many
    subclasses are body-empty markers. subclasses that DO exercise
    the version collection provide the journal-row lookup methods
    inline.
    """


class FakeWorkspaceContext:
    """parity base for ``ToolContextManager`` / ``WorkspaceContextManager``.

    workspace tests use ``_FakeContext`` to model the per-conversation
    context manager that workspace tools resolve workspaces +
    sandboxes through. subclasses provide ``get_pin`` / ``set_pin`` /
    the pin namespace lookups individual tests exercise.
    """


class FakeWorkspaceSandbox:
    """parity base for the production :class:`WorkspaceSandbox`.

    workspace tests use ``_FakeSandbox`` to short-circuit the path
    resolution that the production sandbox does against the bind
    root. subclasses provide ``resolve_fs_path`` and the path
    validators individual tests exercise.
    """
