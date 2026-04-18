"""unit tests for :func:`authorize_workspace_access`.

exercises every branch of the Phase-3 access-control helper:

- missing ``customer_id`` on the scope -> raise
- cross-customer access -> raise, cache never consulted
- owner path (same agent + same user) -> allow, cache never consulted
- same-customer granted -> allow, cache confirms the grant
- same-customer no-grant -> raise, cache surfaces deny
- user-scoped grant takes precedence over agent-wide grant (delegated
  to cache.check_access, which this test verifies by inspecting the
  kwargs the helper passed through)

the AclCache is represented by an :class:`~unittest.mock.AsyncMock` so
no live DB or NATS is required.
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import AsyncMock
from uuid import UUID, uuid4 as uuid7

import pytest

from threetears.agent.tools.call_scope import ToolCallScope
from threetears.agent.tools.context_envelope import CallContext
from threetears.agent.workspace.authorize import (
    WorkspaceAccessDenied,
    authorize_workspace_access,
)


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------


@dataclass
class _FakeWorkspace:
    """structural stand-in for :class:`Workspace` with the fields the helper reads."""

    id: UUID
    customer_id: UUID
    owner_agent_id: UUID
    created_by_user_id: UUID
    namespace_name: str


def _make_workspace(
    *,
    customer_id: UUID,
    owner_agent_id: UUID,
    created_by_user_id: UUID,
    namespace_name: str = "ws_abc",
) -> _FakeWorkspace:
    """build a fake workspace record.

    :param customer_id: owning customer UUID
    :ptype customer_id: UUID
    :param owner_agent_id: owning agent UUID
    :ptype owner_agent_id: UUID
    :param created_by_user_id: creating-user UUID
    :ptype created_by_user_id: UUID
    :param namespace_name: namespace name for cache keying
    :ptype namespace_name: str
    :return: fake workspace record
    :rtype: _FakeWorkspace
    """
    return _FakeWorkspace(
        id=uuid7(),
        customer_id=customer_id,
        owner_agent_id=owner_agent_id,
        created_by_user_id=created_by_user_id,
        namespace_name=namespace_name,
    )


def _make_scope(
    *,
    agent_id: UUID | None = None,
    user_id: UUID | None = None,
    customer_id: UUID | None = None,
) -> ToolCallScope:
    """build a :class:`ToolCallScope` carrying only the identity dims.

    :param agent_id: calling agent UUID
    :ptype agent_id: UUID | None
    :param user_id: calling user UUID
    :ptype user_id: UUID | None
    :param customer_id: owning customer UUID
    :ptype customer_id: UUID | None
    :return: scope with populated context
    :rtype: ToolCallScope
    """
    ctx = CallContext(
        agent_id=agent_id,
        user_id=user_id,
        customer_id=customer_id,
    )
    return ToolCallScope(context=ctx)


def _make_cache() -> AsyncMock:
    """build an :class:`AsyncMock` shaped like :class:`AclCache`.

    default behavior: ``check_access`` returns ``None`` (allow). tests
    override ``side_effect`` on the mock to simulate denies.

    :return: AsyncMock wired with a ``check_access`` method
    :rtype: AsyncMock
    """
    cache = AsyncMock()
    cache.check_access = AsyncMock(return_value=None)
    return cache


# ---------------------------------------------------------------------------
# guard clauses
# ---------------------------------------------------------------------------


class TestMissingCustomerId:
    """guard: ``scope.context.customer_id is None`` -> raise."""

    @pytest.mark.asyncio
    async def test_missing_customer_raises(self) -> None:
        """scope without customer_id is rejected before any cache call."""
        workspace = _make_workspace(
            customer_id=uuid7(),
            owner_agent_id=uuid7(),
            created_by_user_id=uuid7(),
        )
        scope = _make_scope(agent_id=uuid7(), user_id=uuid7(), customer_id=None)
        cache = _make_cache()
        with pytest.raises(WorkspaceAccessDenied, match="missing customer_id"):
            await authorize_workspace_access(
                scope, workspace, "read", acl_cache=cache,
            )
        cache.check_access.assert_not_called()


class TestCrossCustomerDenied:
    """guard: ``workspace.customer_id != scope.customer_id`` -> raise."""

    @pytest.mark.asyncio
    async def test_cross_customer_raises(self) -> None:
        """workspace owned by customer A is not accessible to caller from B."""
        customer_a = uuid7()
        customer_b = uuid7()
        workspace = _make_workspace(
            customer_id=customer_a,
            owner_agent_id=uuid7(),
            created_by_user_id=uuid7(),
        )
        scope = _make_scope(
            agent_id=uuid7(), user_id=uuid7(), customer_id=customer_b,
        )
        cache = _make_cache()
        with pytest.raises(WorkspaceAccessDenied, match="cross-customer"):
            await authorize_workspace_access(
                scope, workspace, "read", acl_cache=cache,
            )
        cache.check_access.assert_not_called()


# ---------------------------------------------------------------------------
# owner short-circuit
# ---------------------------------------------------------------------------


class TestOwnerAllow:
    """owner path (same agent + same user) allows without cache lookup."""

    @pytest.mark.asyncio
    async def test_owner_same_customer_agent_user_allows(self) -> None:
        """owning agent + creating user bypasses the grant check."""
        customer_id = uuid7()
        agent_id = uuid7()
        user_id = uuid7()
        workspace = _make_workspace(
            customer_id=customer_id,
            owner_agent_id=agent_id,
            created_by_user_id=user_id,
        )
        scope = _make_scope(
            agent_id=agent_id, user_id=user_id, customer_id=customer_id,
        )
        cache = _make_cache()
        # should not raise
        await authorize_workspace_access(
            scope, workspace, "write", acl_cache=cache,
        )
        cache.check_access.assert_not_called()

    @pytest.mark.asyncio
    async def test_owner_agent_different_user_falls_to_cache(self) -> None:
        """same agent but different user does not short-circuit."""
        customer_id = uuid7()
        agent_id = uuid7()
        creator_user_id = uuid7()
        asking_user_id = uuid7()
        workspace = _make_workspace(
            customer_id=customer_id,
            owner_agent_id=agent_id,
            created_by_user_id=creator_user_id,
        )
        scope = _make_scope(
            agent_id=agent_id,
            user_id=asking_user_id,
            customer_id=customer_id,
        )
        cache = _make_cache()  # allow
        await authorize_workspace_access(
            scope, workspace, "read", acl_cache=cache,
        )
        cache.check_access.assert_called_once()


# ---------------------------------------------------------------------------
# same-customer grant delegation
# ---------------------------------------------------------------------------


class TestSameCustomerGrantedAllow:
    """same-customer non-owner call succeeds when cache allows."""

    @pytest.mark.asyncio
    async def test_granted_read_succeeds(self) -> None:
        """cache returning None lets the helper return None."""
        customer_id = uuid7()
        workspace = _make_workspace(
            customer_id=customer_id,
            owner_agent_id=uuid7(),
            created_by_user_id=uuid7(),
        )
        caller_agent = uuid7()
        caller_user = uuid7()
        scope = _make_scope(
            agent_id=caller_agent,
            user_id=caller_user,
            customer_id=customer_id,
        )
        cache = _make_cache()
        await authorize_workspace_access(
            scope, workspace, "read", acl_cache=cache,
        )
        # verify the helper forwarded the scope identity to the cache
        cache.check_access.assert_awaited_once_with(
            agent_id=caller_agent,
            namespace_name=workspace.namespace_name,
            operation="select",
            user_id=caller_user,
        )

    @pytest.mark.asyncio
    async def test_granted_write_maps_to_upsert(self) -> None:
        """operation='write' forwards as 'upsert' at the cache boundary."""
        customer_id = uuid7()
        workspace = _make_workspace(
            customer_id=customer_id,
            owner_agent_id=uuid7(),
            created_by_user_id=uuid7(),
        )
        scope = _make_scope(
            agent_id=uuid7(), user_id=uuid7(), customer_id=customer_id,
        )
        cache = _make_cache()
        await authorize_workspace_access(
            scope, workspace, "write", acl_cache=cache,
        )
        call_kwargs = cache.check_access.await_args.kwargs
        assert call_kwargs["operation"] == "upsert"


class TestSameCustomerNoGrantDeny:
    """same-customer non-owner with no grant is denied."""

    @pytest.mark.asyncio
    async def test_cache_deny_surfaces_as_workspace_access_denied(self) -> None:
        """a cache-raised deny becomes a WorkspaceAccessDenied."""
        customer_id = uuid7()
        workspace = _make_workspace(
            customer_id=customer_id,
            owner_agent_id=uuid7(),
            created_by_user_id=uuid7(),
        )
        scope = _make_scope(
            agent_id=uuid7(), user_id=uuid7(), customer_id=customer_id,
        )
        cache = _make_cache()
        cache.check_access.side_effect = RuntimeError("agent has no access")
        with pytest.raises(WorkspaceAccessDenied, match="grant check failed"):
            await authorize_workspace_access(
                scope, workspace, "read", acl_cache=cache,
            )


# ---------------------------------------------------------------------------
# user-scoped grant preferred
# ---------------------------------------------------------------------------


class TestUserScopedGrantPreference:
    """helper forwards user_id, letting cache resolve user-scoped first."""

    @pytest.mark.asyncio
    async def test_helper_forwards_user_id_to_cache(self) -> None:
        """the cache receives the scope's user_id verbatim."""
        customer_id = uuid7()
        caller_user = uuid7()
        workspace = _make_workspace(
            customer_id=customer_id,
            owner_agent_id=uuid7(),
            created_by_user_id=uuid7(),
        )
        scope = _make_scope(
            agent_id=uuid7(),
            user_id=caller_user,
            customer_id=customer_id,
        )
        cache = _make_cache()
        await authorize_workspace_access(
            scope, workspace, "read", acl_cache=cache,
        )
        kwargs = cache.check_access.await_args.kwargs
        assert kwargs["user_id"] == caller_user

    @pytest.mark.asyncio
    async def test_owner_path_does_not_consult_cache_at_all(self) -> None:
        """owner short-circuit means user-scoped preference is moot."""
        customer_id = uuid7()
        agent_id = uuid7()
        user_id = uuid7()
        workspace = _make_workspace(
            customer_id=customer_id,
            owner_agent_id=agent_id,
            created_by_user_id=user_id,
        )
        scope = _make_scope(
            agent_id=agent_id, user_id=user_id, customer_id=customer_id,
        )
        cache = _make_cache()
        cache.check_access.side_effect = RuntimeError("must not be called")
        # succeeds despite cache set to raise -> owner short-circuit hit
        await authorize_workspace_access(
            scope, workspace, "read", acl_cache=cache,
        )


# ---------------------------------------------------------------------------
# unknown operation
# ---------------------------------------------------------------------------


class TestUnknownOperation:
    """unknown operation strings raise :class:`WorkspaceAccessDenied`."""

    @pytest.mark.asyncio
    async def test_unknown_operation_raises(self) -> None:
        """operation outside the documented set is a programming error surfaced as deny."""
        customer_id = uuid7()
        workspace = _make_workspace(
            customer_id=customer_id,
            owner_agent_id=uuid7(),
            created_by_user_id=uuid7(),
        )
        scope = _make_scope(
            agent_id=uuid7(), user_id=uuid7(), customer_id=customer_id,
        )
        cache = _make_cache()
        with pytest.raises(WorkspaceAccessDenied, match="unknown workspace operation"):
            await authorize_workspace_access(
                scope, workspace, "delete",  # type: ignore[arg-type]
                acl_cache=cache,
            )
