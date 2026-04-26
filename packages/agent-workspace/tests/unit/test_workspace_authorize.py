"""unit tests for :func:`authorize_workspace_access`.

rbac-task-01 Phase 3 rewired the helper to delegate to the unified
evaluator in :mod:`threetears.agent.acl`. three-tier-task-01 Phase E
made :class:`AclCache` the required type at every tool / authorize
call site. the current contract:

- ``scope.context.customer_id is None`` -> raise immediately.
- ``workspace.customer_id != scope.customer_id`` -> raise, no evaluator
  trip (cross-customer short-circuits).
- otherwise build an :class:`EvaluationContext` and call
  :func:`evaluate_decision` with the loaders the caller's
  :class:`AclCache` exposes.
- evaluator returning ``False`` surfaces as :class:`WorkspaceAccessDenied`.
- unknown operation strings raise before any evaluator trip.

the owner short-circuit is no longer implemented in the helper; the
evaluator's ``_resolve_side`` handles owner-match inside the agent
side. the helper therefore always goes through the evaluator unless
a guard clause fires.

tests build a real :class:`AclCache` wired with empty loader stubs
and patch :func:`evaluate_decision` at the authorize module's import
site to drive allow/deny outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock
from uuid import UUID, uuid4 as uuid7

import pytest

from threetears.agent.acl import (
    AclCache,
    GroupMembership,
    Namespace,
    Role,
    RoleAssignment,
)
from threetears.agent.tools.call_scope import ToolCallScope
from threetears.agent.tools.context_envelope import CallContext
from threetears.agent.workspace import authorize as _authorize_module
from threetears.agent.workspace.authorize import (
    WorkspaceAccessDenied,
    authorize_workspace_access,
)


class _EmptyMembershipLoader:
    """membership loader stub yielding no memberships for any actor."""

    async def load_for_user(
        self, user_id: UUID,
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
        self, agent_id: UUID,
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
        self, role_ids: tuple[UUID, ...],
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
        self, group_ids: tuple[UUID, ...],
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
    :param namespace_name: namespace name for logging
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


def _make_cache() -> AclCache:
    """build a real :class:`AclCache` wired with empty loader stubs.

    concrete loader methods are never exercised here because the
    tests patch :func:`evaluate_decision` directly at the authorize
    module's import site; the cache still has to be a real
    :class:`AclCache` because the ctor signature on the helper +
    tools is the concrete type (no Protocol duck-type).

    :return: live cache wired with empty loaders
    :rtype: AclCache
    """
    return AclCache(
        membership_loader=_EmptyMembershipLoader(),
        grant_loader=_EmptyGrantLoader(),
        ttl_seconds=60,
    )


def _patch_evaluate_decision(
    monkeypatch: pytest.MonkeyPatch,
    *,
    returning: bool | None = None,
    raising: BaseException | None = None,
) -> AsyncMock:
    """patch :func:`evaluate_decision` at the authorize import site.

    tests control the evaluator's outcome by substituting an
    :class:`AsyncMock`; the helper calls the import from within the
    authorize module, so the patch target is
    ``threetears.agent.workspace.authorize.evaluate_decision``.

    :param monkeypatch: pytest monkeypatch fixture
    :ptype monkeypatch: pytest.MonkeyPatch
    :param returning: value the stub returns when called (None -> no
        return_value set; ``raising`` takes precedence)
    :ptype returning: bool | None
    :param raising: exception the stub raises when called
    :ptype raising: BaseException | None
    :return: the installed mock so the test can assert on its calls
    :rtype: AsyncMock
    """
    stub = AsyncMock()
    if raising is not None:
        stub.side_effect = raising
    elif returning is not None:
        stub.return_value = returning
    monkeypatch.setattr(_authorize_module, "evaluate_decision", stub)
    return stub


# ---------------------------------------------------------------------------
# guard clauses
# ---------------------------------------------------------------------------


class TestMissingCustomerId:
    """guard: ``scope.context.customer_id is None`` -> raise."""

    @pytest.mark.asyncio
    async def test_missing_customer_raises(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """scope without customer_id is rejected before any evaluator trip."""
        workspace = _make_workspace(
            customer_id=uuid7(),
            owner_agent_id=uuid7(),
            created_by_user_id=uuid7(),
        )
        scope = _make_scope(agent_id=uuid7(), user_id=uuid7(), customer_id=None)
        cache = _make_cache()
        stub = _patch_evaluate_decision(monkeypatch, returning=True)
        with pytest.raises(WorkspaceAccessDenied, match="missing customer_id"):
            await authorize_workspace_access(
                scope, workspace, "read", acl_cache=cache,
            )
        stub.assert_not_awaited()


class TestCrossCustomerDenied:
    """guard: ``workspace.customer_id != scope.customer_id`` -> raise."""

    @pytest.mark.asyncio
    async def test_cross_customer_raises(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
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
        stub = _patch_evaluate_decision(monkeypatch, returning=True)
        with pytest.raises(WorkspaceAccessDenied, match="cross-customer"):
            await authorize_workspace_access(
                scope, workspace, "read", acl_cache=cache,
            )
        stub.assert_not_awaited()


# ---------------------------------------------------------------------------
# evaluator delegation
# ---------------------------------------------------------------------------


class TestEvaluatorDelegation:
    """same-customer call routes to the unified evaluator."""

    @pytest.mark.asyncio
    async def test_allow_returns_none(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """evaluator returning True lets the helper return None."""
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
        stub = _patch_evaluate_decision(monkeypatch, returning=True)

        await authorize_workspace_access(
            scope, workspace, "read", acl_cache=cache,
        )
        stub.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_deny_raises_workspace_access_denied(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """evaluator returning False surfaces as :class:`WorkspaceAccessDenied`."""
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
        _patch_evaluate_decision(monkeypatch, returning=False)

        with pytest.raises(WorkspaceAccessDenied, match="evaluator denied"):
            await authorize_workspace_access(
                scope, workspace, "read", acl_cache=cache,
            )

    @pytest.mark.asyncio
    async def test_helper_builds_context_from_scope_and_workspace(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """helper forwards scope identity + workspace ns into the evaluator."""
        customer_id = uuid7()
        caller_agent = uuid7()
        caller_user = uuid7()
        owner_agent = uuid7()
        workspace = _make_workspace(
            customer_id=customer_id,
            owner_agent_id=owner_agent,
            created_by_user_id=uuid7(),
        )
        scope = _make_scope(
            agent_id=caller_agent, user_id=caller_user,
            customer_id=customer_id,
        )
        cache = _make_cache()
        captured: dict[str, Any] = {}

        async def fake_eval(ctx: Any, **kwargs: Any) -> bool:
            captured["namespace_id"] = ctx.namespace.id
            captured["namespace_customer_id"] = ctx.namespace.customer_id
            captured["namespace_type"] = ctx.namespace.namespace_type
            captured["namespace_owner_agent_id"] = ctx.namespace.owner_agent_id
            captured["action"] = ctx.action
            captured["user_id"] = ctx.user_id
            captured["agent_id"] = ctx.agent_id
            captured["cache"] = kwargs["cache"]
            return True

        monkeypatch.setattr(_authorize_module, "evaluate_decision", fake_eval)

        await authorize_workspace_access(
            scope, workspace, "read", acl_cache=cache,
        )

        assert captured["namespace_id"] == workspace.id
        assert captured["namespace_customer_id"] == customer_id
        assert captured["namespace_type"] == "workspace"
        assert captured["namespace_owner_agent_id"] == owner_agent
        assert captured["action"] == "read"
        assert captured["user_id"] == caller_user
        assert captured["agent_id"] == caller_agent
        assert captured["cache"] is cache

    @pytest.mark.asyncio
    async def test_write_forwards_write_action_verbatim(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``operation='write'`` lands on the evaluator context as ``'write'``."""
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
        captured_action: dict[str, str] = {}

        async def fake_eval(ctx: Any, **kwargs: Any) -> bool:
            captured_action["action"] = ctx.action
            return True

        monkeypatch.setattr(_authorize_module, "evaluate_decision", fake_eval)

        await authorize_workspace_access(
            scope, workspace, "write", acl_cache=cache,
        )
        assert captured_action["action"] == "write"


# ---------------------------------------------------------------------------
# unknown operation
# ---------------------------------------------------------------------------


class TestUnknownOperation:
    """unknown operation strings raise :class:`WorkspaceAccessDenied`."""

    @pytest.mark.asyncio
    async def test_unknown_operation_raises(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """operation outside the documented set raises before any evaluator trip."""
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
        stub = _patch_evaluate_decision(monkeypatch, returning=True)
        with pytest.raises(
            WorkspaceAccessDenied, match="unknown workspace operation",
        ):
            await authorize_workspace_access(
                scope, workspace, "delete",  # type: ignore[arg-type]
                acl_cache=cache,
            )
        stub.assert_not_awaited()
