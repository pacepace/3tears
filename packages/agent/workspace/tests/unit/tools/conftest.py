"""shared fixtures for the workspace-tools unit suite.

every workspace tool requires both an installed
:class:`~threetears.agent.tools.call_scope.ToolCallScope` and a live
:class:`~threetears.agent.acl.AclCache` (required ctor param on every
tool since three-tier-task-01 Phase E; no silent-bypass path remains).
the fixtures here satisfy both pre-conditions for every test in this
directory:

- :func:`tool_scope_context` builds a fully-populated
  :class:`~threetears.agent.tools.context_envelope.CallContext` with
  non-None identity dimensions; tests can override it to vary identity
  per-case.
- :func:`tool_call_scope` is autouse and pushes a
  :class:`ToolCallScope` carrying that context for the duration of each
  test, so the helper's ``current_scope() is None`` guard never fires
  inside this directory.
- :func:`permissive_acl_cache` returns a real :class:`AclCache` wired
  with in-memory :class:`MembershipLoader` + :class:`GrantLoader`
  stubs; tests pass it as ``acl_cache=`` when constructing tools.
- :func:`stub_authorize_workspace_access` is autouse and replaces the
  workspace-shape-dependent authorize entry point with an
  :class:`AsyncMock` for the duration of each test. unit tests under
  this directory exercise tool behavior with lightweight fake workspace
  entities that intentionally skip the
  :class:`~threetears.agent.workspace.authorize.WorkspaceLike` protocol
  surface (no ``namespace_name`` / ``owner_agent_id`` /
  ``created_by_user_id`` / ``customer_id``); that surface is exercised
  end-to-end in ``tests/unit/test_workspace_tool_authorization.py``.

tests that need to assert deny behavior should not use
``permissive_acl_cache`` -- they construct their own cache wiring and
pass it explicitly.
"""

from __future__ import annotations

from typing import AsyncIterator
from unittest.mock import AsyncMock
from uuid import UUID, uuid7

import pytest

from threetears.agent.acl import (
    AclCache,
    GroupMembership,
    Namespace,
    Role,
    RoleAssignment,
)
from threetears.agent.tools.call_scope import (
    ToolCallScope,
    enter_call_scope,
)
from threetears.agent.tools.context_envelope import CallContext


class _EmptyMembershipLoader:
    """membership loader stub yielding no memberships for any actor."""

    async def load_for_user(
        self,
        user_id: UUID,
    ) -> tuple[GroupMembership, ...]:
        """
        return empty tuple for every user id.

        :param user_id: user UUID (ignored)
        :ptype user_id: UUID
        :return: empty tuple
        :rtype: tuple[GroupMembership, ...]
        """
        del user_id
        return ()

    async def load_for_agent(
        self,
        agent_id: UUID,
    ) -> tuple[GroupMembership, ...]:
        """
        return empty tuple for every agent id.

        :param agent_id: agent UUID (ignored)
        :ptype agent_id: UUID
        :return: empty tuple
        :rtype: tuple[GroupMembership, ...]
        """
        del agent_id
        return ()


class _EmptyGrantLoader:
    """grant loader stub yielding no assignments / roles / groups."""

    async def load_assignments_for_groups(
        self,
        group_ids: tuple[UUID, ...],
        namespace: Namespace,
    ) -> tuple[RoleAssignment, ...]:
        """
        return empty tuple for every group set.

        :param group_ids: group UUIDs (ignored)
        :ptype group_ids: tuple[UUID, ...]
        :param namespace: namespace (ignored)
        :ptype namespace: Namespace
        :return: empty tuple
        :rtype: tuple[RoleAssignment, ...]
        """
        del group_ids
        del namespace
        return ()

    async def load_roles(
        self,
        role_ids: tuple[UUID, ...],
    ) -> dict[UUID, Role]:
        """
        return empty mapping for every role set.

        :param role_ids: role UUIDs (ignored)
        :ptype role_ids: tuple[UUID, ...]
        :return: empty mapping
        :rtype: dict[UUID, Role]
        """
        del role_ids
        return {}

    async def load_groups(
        self,
        group_ids: tuple[UUID, ...],
    ) -> dict[UUID, object]:
        """
        return empty mapping for every group set.

        :param group_ids: group UUIDs (ignored)
        :ptype group_ids: tuple[UUID, ...]
        :return: empty mapping
        :rtype: dict[UUID, object]
        """
        del group_ids
        return {}


@pytest.fixture
def tool_scope_context() -> CallContext:
    """test-scoped CallContext populated with non-None identity dims.

    every dimension is a fresh :func:`uuid.uuid7` so identity-routing
    paths exercised by the tools see realistic, distinct values. tests
    that need a specific identity (e.g. matching a workspace's
    customer_id) should override this fixture in their own scope.

    :return: call context with all identity fields populated
    :rtype: CallContext
    """
    return CallContext(
        agent_id=uuid7(),
        customer_id=uuid7(),
        user_id=uuid7(),
        conversation_id=uuid7(),
        correlation_id=uuid7(),
    )


@pytest.fixture(autouse=True)
async def tool_call_scope(
    tool_scope_context: CallContext,
) -> AsyncIterator[ToolCallScope]:
    """install a :class:`ToolCallScope` for the duration of every tool test.

    autouse so individual tests do not need to opt in. the scope wraps
    the test body inside :func:`enter_call_scope`; tests that need a
    different identity override :func:`tool_scope_context`. tests in
    other directories (e.g. ``tests/unit/test_workspace_tool_authorization.py``)
    are unaffected because this conftest only applies to this directory.

    :param tool_scope_context: identity envelope for the scope
    :ptype tool_scope_context: CallContext
    :return: async iterator yielding the installed scope
    :rtype: AsyncIterator[ToolCallScope]
    """
    scope = ToolCallScope(context=tool_scope_context)
    async with enter_call_scope(scope):
        yield scope


@pytest.fixture
def permissive_acl_cache() -> AclCache:
    """real :class:`AclCache` wired with empty loader stubs.

    three-tier-task-01 Phase E: every workspace tool takes an
    :class:`AclCache` as a required constructor param. the tests in
    this directory stub :func:`authorize_workspace_access` via the
    autouse fixture, so the loaders are never actually consulted —
    but we wire real loader stubs anyway so the cache + loader surface
    is structurally honest and a future test that unstubs the helper
    will not break on protocol drift.

    tools constructed with this cache pass through the authorize gate
    unconditionally (the gate itself is stubbed). used by the bulk of
    the tool tests that exercise functionality unrelated to RBAC.
    authorization-matrix tests build their own cache wiring to assert
    deny paths.

    :return: live cache instance
    :rtype: AclCache
    """
    return AclCache(
        membership_loader=_EmptyMembershipLoader(),
        grant_loader=_EmptyGrantLoader(),
        ttl_seconds=60,
    )


@pytest.fixture(autouse=True)
def stub_authorize_workspace_access(
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncMock:
    """replace :func:`authorize_workspace_access` with an :class:`AsyncMock`.

    the unit tests in this directory wire fake workspace entities that
    do not expose the full :class:`WorkspaceLike` protocol surface
    (``namespace_name`` / ``owner_agent_id`` / ``created_by_user_id`` /
    ``customer_id``). the workspace-shape-dependent grant decision is
    exercised end-to-end in ``tests/unit/test_workspace_tool_authorization.py``;
    here we mock the inner call so unit tests focus on tool behavior.

    the outer :func:`authorize_workspace` helper still enforces both
    preconditions (scope installed, ``acl_cache`` injected) and still
    runs identity enrichment when a ``db_pool`` is supplied -- only
    the actual ACL grant decision is short-circuited.

    :param monkeypatch: pytest monkeypatch fixture
    :ptype monkeypatch: pytest.MonkeyPatch
    :return: the installed mock so individual tests can assert on calls
    :rtype: AsyncMock
    """
    from threetears.agent.workspace import authorize as _authorize_module

    stub = AsyncMock(return_value=None)
    monkeypatch.setattr(
        _authorize_module,
        "authorize_workspace_access",
        stub,
    )
    return stub


@pytest.fixture(autouse=True)
def stub_authorize_workspace_file_access(
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncMock:
    """replace :func:`authorize_workspace_file_access` with an :class:`AsyncMock`.

    namespace-task-01 phase 7 adds a per-file rbac gate on every
    read / write enforcement site. the unit tests here wire fake
    workspace entities that skip the full :class:`WorkspaceLike`
    surface; the workspace-shape path-glob decision is exercised end
    to end in ``tests/integration/`` against real rbac data. here we
    mock the helper so tool unit tests focus on tool behavior.

    the outer :func:`authorize_workspace_file` wrapper still enforces
    both preconditions (scope installed, ``acl_cache`` injected) --
    only the underlying evaluator + glob match is short-circuited.

    :param monkeypatch: pytest monkeypatch fixture
    :ptype monkeypatch: pytest.MonkeyPatch
    :return: the installed mock so individual tests can assert on calls
    :rtype: AsyncMock
    """
    from threetears.agent.workspace import authorize as _authorize_module

    stub = AsyncMock(return_value=None)
    monkeypatch.setattr(
        _authorize_module,
        "authorize_workspace_file_access",
        stub,
    )
    return stub


@pytest.fixture(autouse=True)
def stub_enrich_workspace_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncMock:
    """replace :func:`enrich_workspace_identity` with a no-op :class:`AsyncMock`.

    the production helper does a ``SELECT customer_id FROM
    platform.namespaces`` against the supplied pool to stamp
    ``workspace.customer_id`` before authorize. unit-tests in this
    directory wire bespoke fake pools (no ``fetchrow``) and fake
    workspace entities (no ``customer_id`` setter), so the real
    enrichment cannot run. patching it out keeps the unit suite focused
    on tool behavior; identity enrichment is exercised via the
    cross-customer cases in
    ``tests/unit/test_workspace_tool_authorization.py``.

    :param monkeypatch: pytest monkeypatch fixture
    :ptype monkeypatch: pytest.MonkeyPatch
    :return: the installed mock so individual tests can assert on calls
    :rtype: AsyncMock
    """
    from threetears.agent.workspace.tools import helpers as _helpers_module

    async def _passthrough(workspace, db_pool):  # type: ignore[no-untyped-def]
        return workspace

    stub = AsyncMock(side_effect=_passthrough)
    monkeypatch.setattr(
        _helpers_module,
        "enrich_workspace_identity",
        stub,
    )
    return stub
