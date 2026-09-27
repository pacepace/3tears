"""the grant ``ensure_memory_owner_assignment`` writes is visible to the very next authorization.

every request that reaches the ensure has just been authorized, so the
caller's memberships -- and its owner group's contribution on this
namespace -- are already in the :class:`AclCache`, saying "no grant". these
tests run on the cache's DEFAULT ttl, which is what production runs on: a
ttl of zero expires every entry on read and hides the stale-cache denial
completely, which is how it reached a live deployment as a 500 on a user's
second chat turn.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID, uuid7

import pytest
from pydantic import BaseModel
from threetears.agent.acl import AclCache, AssignmentInvalidatePayload, MembershipInvalidatePayload, Role
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

from ._rbac_rows import RbacRows


class _RecordingPublisher:
    """records every invalidation broadcast.

    not a ``Fake<Name>``; it implements the one method of
    :class:`~threetears.agent.acl.invalidation_bus.AclInvalidationPublisher`.
    """

    def __init__(self) -> None:
        """start with nothing sent.

        :return: nothing
        :rtype: None
        """
        self.sent: list[BaseModel] = []

    async def publish(self, *, subject: Any, message: BaseModel) -> None:
        """record the message.

        :param subject: invalidation subject (unused)
        :ptype subject: Any
        :param message: payload
        :ptype message: BaseModel
        :return: nothing
        :rtype: None
        """
        _ = subject
        self.sent.append(message)


@pytest.fixture(autouse=True)
def _namespace(monkeypatch: pytest.MonkeyPatch) -> None:
    """give the invalidation subjects a namespace.

    :param monkeypatch: pytest monkeypatch
    :ptype monkeypatch: pytest.MonkeyPatch
    :return: nothing
    :rtype: None
    """
    monkeypatch.setenv("THREETEARS_NATS_SUBJECT_NAMESPACE", "t")


def _wiring(publisher: _RecordingPublisher | None = None) -> tuple[RbacRows, MemoryAuthorizerDependencies]:
    """build rows seeded with the builtin owner role, and a bundle on a default-ttl cache.

    :param publisher: invalidation publisher for the bundle, or ``None``
    :ptype publisher: _RecordingPublisher | None
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
        invalidation_publisher=publisher,
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


async def test_what_the_ensure_wrote_is_broadcast_and_a_repeat_ensure_broadcasts_nothing() -> None:
    """other pods hold the same stale entries; a no-op ensure runs on every write and must stay silent."""
    publisher = _RecordingPublisher()
    rows, deps = _wiring(publisher)
    agent_id, customer_id, user_id = uuid7(), uuid7(), uuid7()
    namespace = rows.namespace(namespace_type=MEMORY_NAMESPACE_TYPE, agent_id=agent_id, customer_id=customer_id)

    await ensure_memory_owner_assignment(user_id=user_id, namespace=namespace, deps=deps)

    group_id = rows.assignments[0].group_id
    assert publisher.sent == [
        MembershipInvalidatePayload(actor_type="user", actor_id=user_id),
        AssignmentInvalidatePayload(group_id=group_id),
    ]

    await ensure_memory_owner_assignment(user_id=user_id, namespace=namespace, deps=deps)

    assert len(publisher.sent) == 2
