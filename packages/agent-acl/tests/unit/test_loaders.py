"""unit tests for the canonical Collection-backed loader implementations.

these tests exercise the contract between the canonical
:class:`CollectionMembershipLoader` /
:class:`CollectionGrantLoader` and the unified evaluator. the
loaders are thin adapters over the canonical Collections; the tests
mock the Collections and assert the loader shapes / namespace
filtering match the loader Protocols exactly.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid7

import pytest
from threetears.agent.acl import (
    CollectionGrantLoader,
    CollectionMembershipLoader,
    Group,
    GroupMembership,
    MemberType,
    Namespace,
    Role,
    RoleAssignment,
    ScopeType,
)


# ---------------------------------------------------------------------------
# CollectionMembershipLoader
# ---------------------------------------------------------------------------


class TestCollectionMembershipLoader:
    """delegating loader returns Protocol-shaped tuple result."""

    @pytest.mark.asyncio
    async def test_load_for_user_freezes_to_tuple(self) -> None:
        user_id = uuid7()
        memb = GroupMembership(
            group_id=uuid7(),
            member_type=MemberType.USER,
            member_id=user_id,
            customer_id=uuid7(),
        )
        coll = MagicMock()
        coll.load_for_user = AsyncMock(return_value=[memb])
        loader = CollectionMembershipLoader(collection=coll)
        result = await loader.load_for_user(user_id)
        assert isinstance(result, tuple)
        assert result == (memb,)

    @pytest.mark.asyncio
    async def test_load_for_agent_freezes_to_tuple(self) -> None:
        agent_id = uuid7()
        memb = GroupMembership(
            group_id=uuid7(),
            member_type=MemberType.AGENT,
            member_id=agent_id,
            customer_id=uuid7(),
        )
        coll = MagicMock()
        coll.load_for_agent = AsyncMock(return_value=[memb])
        loader = CollectionMembershipLoader(collection=coll)
        result = await loader.load_for_agent(agent_id)
        assert isinstance(result, tuple)
        assert result == (memb,)


# ---------------------------------------------------------------------------
# CollectionGrantLoader
# ---------------------------------------------------------------------------


def _ns(*, namespace_type: str = "workspace") -> Namespace:
    """build a :class:`Namespace` for tests."""
    return Namespace(
        id=uuid7(),
        customer_id=uuid7(),
        namespace_type=namespace_type,
        owner_agent_id=uuid7(),
    )


class TestCollectionGrantLoaderAssignmentsForGroups:
    """``load_assignments_for_groups`` filters via :meth:`covers`."""

    @pytest.mark.asyncio
    async def test_empty_input_returns_empty_tuple(self) -> None:
        loader = CollectionGrantLoader(
            assignment_collection=MagicMock(),
            role_collection=MagicMock(),
            group_collection=MagicMock(),
        )
        result = await loader.load_assignments_for_groups(
            (), namespace=_ns(),
        )
        assert result == ()

    @pytest.mark.asyncio
    async def test_filters_via_covers_predicate(self) -> None:
        """assignments whose scope does not cover are dropped."""
        ns = _ns()
        matching = RoleAssignment(
            id=uuid7(),
            role_id=uuid7(),
            group_id=uuid7(),
            scope_type=ScopeType.NAMESPACE,
            scope_namespace_id=ns.id,
            scope_namespace_type=None,
            scope_customer_id=None,
        )
        non_matching = RoleAssignment(
            id=uuid7(),
            role_id=uuid7(),
            group_id=uuid7(),
            scope_type=ScopeType.NAMESPACE,
            scope_namespace_id=uuid7(),
            scope_namespace_type=None,
            scope_customer_id=None,
        )
        assignment_coll = MagicMock()
        assignment_coll.load_for_groups = AsyncMock(
            return_value=[matching, non_matching],
        )
        loader = CollectionGrantLoader(
            assignment_collection=assignment_coll,
            role_collection=MagicMock(),
            group_collection=MagicMock(),
        )
        result = await loader.load_assignments_for_groups(
            (matching.group_id,), namespace=ns,
        )
        assert isinstance(result, tuple)
        assert result == (matching,)


class TestCollectionGrantLoaderRoles:
    """``load_roles`` shapes Collection result into mapping."""

    @pytest.mark.asyncio
    async def test_returns_mapping_keyed_by_role_id(self) -> None:
        role_a = Role(
            id=uuid7(),
            name="A",
            permissions={"*": frozenset({"read"})},
            is_built_in=True,
        )
        role_b = Role(
            id=uuid7(),
            name="B",
            permissions={"*": frozenset({"write"})},
            is_built_in=False,
        )
        role_coll = MagicMock()
        role_coll.get_many = AsyncMock(return_value=[role_a, role_b])
        loader = CollectionGrantLoader(
            assignment_collection=MagicMock(),
            role_collection=role_coll,
            group_collection=MagicMock(),
        )
        result = await loader.load_roles((role_a.id, role_b.id))
        assert result == {role_a.id: role_a, role_b.id: role_b}


class TestCollectionGrantLoaderGroups:
    """``load_groups`` maps entity result into :class:`Group` dataclass."""

    @pytest.mark.asyncio
    async def test_returns_mapping_with_group_dataclass(self) -> None:
        gid = uuid7()
        cid = uuid7()
        # mock entity exposes id / name / customer_id attrs
        entity: Any = MagicMock()
        entity.id = gid
        entity.name = "Engineering"
        entity.customer_id = cid
        group_coll = MagicMock()
        group_coll.get_many = AsyncMock(return_value=[entity])
        loader = CollectionGrantLoader(
            assignment_collection=MagicMock(),
            role_collection=MagicMock(),
            group_collection=group_coll,
        )
        result = await loader.load_groups((gid,))
        assert gid in result
        g = result[gid]
        assert isinstance(g, Group)
        assert g.name == "Engineering"
        assert g.customer_id == cid
