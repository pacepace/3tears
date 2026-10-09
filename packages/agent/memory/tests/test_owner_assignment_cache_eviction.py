"""the grant ``ensure_memory_owner_assignment`` writes is visible to the very next authorization.

every request that reaches the ensure has just been authorized, so the
caller's memberships -- and its owner group's contribution on this
namespace -- are already in the :class:`AclCache`, saying "no grant". these
tests run on a cache that keeps what it holds, as production's does: a cache
that kept nothing would hide the stale-cache denial completely, which is how
it reached a live deployment as a 500 on a user's second chat turn.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID, uuid7

import pytest
from threetears.agent.acl import AclCache, Role
from threetears.agent.memory.authorize import (
    ACTION_MEMORY_READ,
    ACTION_MEMORY_WRITE,
    MEMORY_NAMESPACE_TYPE,
    MEMORY_OWNER_ROLE_NAME,
    MemoryAccessDenied,
    MemoryAuthorizerDependencies,
    authorize_memory_access,
    ensure_memory_owner_assignment,
)

from .rbac_rows import RbacRows


def _wiring() -> tuple[RbacRows, MemoryAuthorizerDependencies]:
    """build rows seeded with the builtin owner role, and a bundle on a cache that keeps what it holds.

    :return: the rows and the bundle
    :rtype: tuple[RbacRows, MemoryAuthorizerDependencies]
    """
    rows = RbacRows(
        roles=[
            Role(
                id=uuid7(),
                name=MEMORY_OWNER_ROLE_NAME,
                permissions={"memory": frozenset({ACTION_MEMORY_READ, ACTION_MEMORY_WRITE})},
                is_built_in=True,
            )
        ]
    )
    deps = MemoryAuthorizerDependencies(
        acl_cache=AclCache(membership_loader=rows, grant_loader=rows),
        **rows.collections(),
    )
    return rows, deps


async def _read(deps: MemoryAuthorizerDependencies, *, agent_id: UUID, customer_id: UUID, user_id: UUID) -> Any:
    """authorize a user-initiated memory read.

    :param deps: authorizer bundle
    :ptype deps: MemoryAuthorizerDependencies
    :param agent_id: agent owning the memory namespace
    :ptype agent_id: UUID
    :param customer_id: customer owning the memory namespace
    :ptype customer_id: UUID
    :param user_id: calling user
    :ptype user_id: UUID
    :return: resolved namespace
    :rtype: Any
    """
    return await authorize_memory_access(
        action=ACTION_MEMORY_READ,
        agent_id=agent_id,
        customer_id=customer_id,
        caller_user_id=user_id,
        caller_agent_id=None,
        deps=deps,
    )


async def test_a_first_grant_is_honoured_by_the_next_request_on_the_same_cache() -> None:
    """deny (caching the user's empty memberships), ensure, allow."""
    rows, deps = _wiring()
    agent_id, customer_id, user_id = uuid7(), uuid7(), uuid7()

    with pytest.raises(MemoryAccessDenied):
        await _read(deps, agent_id=agent_id, customer_id=customer_id, user_id=user_id)

    namespace = rows.namespace(namespace_type=MEMORY_NAMESPACE_TYPE, agent_id=agent_id, customer_id=customer_id)
    await ensure_memory_owner_assignment(user_id=user_id, namespace=namespace, deps=deps)

    resolved = await _read(deps, agent_id=agent_id, customer_id=customer_id, user_id=user_id)
    assert resolved.id == namespace.id


async def test_a_grant_on_a_second_agent_is_honoured_when_the_membership_already_existed() -> None:
    """the owner group is per (customer, user) and spans agents, so its membership is not new here.

    what the evaluation cached is the GROUP's empty contribution on the second
    namespace; only evicting the group's assignment entries lets the new
    assignment through.
    """
    rows, deps = _wiring()
    first_agent, second_agent, customer_id, user_id = uuid7(), uuid7(), uuid7(), uuid7()
    first = rows.namespace(namespace_type=MEMORY_NAMESPACE_TYPE, agent_id=first_agent, customer_id=customer_id)
    await ensure_memory_owner_assignment(user_id=user_id, namespace=first, deps=deps)
    await _read(deps, agent_id=first_agent, customer_id=customer_id, user_id=user_id)

    with pytest.raises(MemoryAccessDenied):
        await _read(deps, agent_id=second_agent, customer_id=customer_id, user_id=user_id)

    second = rows.namespace(namespace_type=MEMORY_NAMESPACE_TYPE, agent_id=second_agent, customer_id=customer_id)
    await ensure_memory_owner_assignment(user_id=user_id, namespace=second, deps=deps)

    resolved = await _read(deps, agent_id=second_agent, customer_id=customer_id, user_id=user_id)
    assert resolved.id == second.id


def test_the_bundle_takes_no_invalidation_publisher() -> None:
    """the acl invalidation subjects are retired: other processes hear the rows' broadcasts and generations."""
    rows, _deps = _wiring()
    with pytest.raises(TypeError):
        MemoryAuthorizerDependencies(  # type: ignore[call-arg]
            acl_cache=AclCache(membership_loader=rows, grant_loader=rows),
            invalidation_publisher=object(),
            **rows.collections(),
        )
