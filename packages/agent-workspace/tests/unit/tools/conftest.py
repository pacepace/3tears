"""shared fixtures for the workspace-tools unit suite.

every workspace tool now requires both an installed
:class:`~threetears.agent.tools.call_scope.ToolCallScope` and a non-None
``acl_cache`` to clear the WS-ACL-05 hard-fail in
:func:`threetears.agent.workspace.tools.helpers.authorize_workspace`. the
fixtures here satisfy both pre-conditions for every test in this
directory:

- :func:`tool_scope_context` builds a fully-populated
  :class:`~threetears.agent.tools.context_envelope.CallContext` with
  non-None identity dimensions; tests can override it to vary identity
  per-case.
- :func:`tool_call_scope` is autouse and pushes a
  :class:`ToolCallScope` carrying that context for the duration of each
  test, so the helper's ``current_scope() is None`` guard never fires
  inside this directory.
- :func:`permissive_acl_cache` returns an :class:`AclCache`-shaped mock
  whose ``check_access`` always grants ``"write"`` access; tests pass it
  as ``acl_cache=`` when constructing tools so the helper's
  ``acl_cache is None`` guard never fires either.
- :func:`stub_authorize_workspace_access` is autouse and replaces the
  workspace-shape-dependent authorize entry point with an
  :class:`AsyncMock` for the duration of each test. unit tests under
  this directory exercise tool behavior with lightweight fake workspace
  entities that intentionally skip the
  :class:`~threetears.agent.workspace.authorize.WorkspaceLike` protocol
  surface (no ``namespace_name`` / ``owner_agent_id`` /
  ``created_by_user_id`` / ``customer_id``); that surface is exercised
  end-to-end in ``tests/unit/test_workspace_tool_authorization.py``.
  the helper's two preconditions (scope installed, ``acl_cache``
  injected) still raise on miss, so tests cannot accidentally drop
  either wire.

tests that need to assert deny behavior should not use
``permissive_acl_cache`` -- they construct their own cache mock and pass
it explicitly.
"""

from __future__ import annotations

from typing import AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import pytest
from uuid import uuid7

from threetears.agent.tools.call_scope import (
    ToolCallScope,
    enter_call_scope,
)
from threetears.agent.tools.context_envelope import CallContext


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
def permissive_acl_cache() -> MagicMock:
    """AclCacheLike-shaped mock that permits every access.

    rbac-task-01 Phase 3: the :class:`AclCacheLike` protocol surfaces
    ``membership_loader`` + ``grant_loader`` instead of the retired
    ``check_access`` method. the helper in
    :mod:`threetears.agent.workspace.authorize` reads both attributes
    when building its :class:`EvaluationContext`. since the tests in
    this directory stub :func:`authorize_workspace_access` via the
    autouse fixture, the loaders are never actually consulted — but we
    still populate them with :class:`AsyncMock` attributes so the
    ``AclCacheLike`` protocol surface stays correct and a future test
    that unstubs the helper will not break on missing attrs.

    tools constructed with this cache pass through the WS-ACL-05
    authorize gate unconditionally. used by the bulk of the tool tests
    that exercise functionality unrelated to RBAC. authorization-matrix
    tests build their own cache mocks to assert deny paths.

    :return: mock exposing ``membership_loader`` + ``grant_loader`` attrs
    :rtype: MagicMock
    """
    cache = MagicMock()
    cache.membership_loader = AsyncMock()
    cache.grant_loader = AsyncMock()
    return cache


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

