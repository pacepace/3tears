"""the grant ``ensure_conversation_owner_assignment`` writes is visible to the very next authorization.

the conversation twin of the memory-owner eviction tests: every message
send that reaches the ensure has just been authorized, so the caller's
memberships and its owner group's contribution on this namespace are
already cached, saying "no grant". run on the cache's DEFAULT ttl -- a ttl
of zero expires every entry on read and hides the stale-cache denial.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID, uuid7

import pytest
from pydantic import BaseModel
from threetears.agent.acl import AclCache, AssignmentInvalidatePayload, MembershipInvalidatePayload, Role
from threetears.conversations.authorize import (
    ACTION_CONVERSATION_READ,
    ACTION_CONVERSATION_WRITE,
    CONVERSATION_NAMESPACE_TYPE,
    CONVERSATION_OWNER_ROLE_NAME,
    ConversationAccessDenied,
    ConversationAuthorizerDependencies,
    authorize_conversation_access,
    ensure_conversation_owner_assignment,
)

from .rbac_rows import RbacRows


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


def _wiring(publisher: _RecordingPublisher | None = None) -> tuple[RbacRows, ConversationAuthorizerDependencies]:
    """build rows seeded with the builtin owner role, and a bundle on a default-ttl cache.

    :param publisher: invalidation publisher for the bundle, or ``None``
    :ptype publisher: _RecordingPublisher | None
    :return: the rows and the bundle
    :rtype: tuple[RbacRows, ConversationAuthorizerDependencies]
    """
    rows = RbacRows(
        roles=[
            Role(
                id=uuid7(),
                name=CONVERSATION_OWNER_ROLE_NAME,
                permissions={"conversation": frozenset({ACTION_CONVERSATION_READ, ACTION_CONVERSATION_WRITE})},
                is_built_in=True,
            )
        ]
    )
    deps = ConversationAuthorizerDependencies(
        acl_cache=AclCache(membership_loader=rows, grant_loader=rows),
        invalidation_publisher=publisher,
        **rows.collections(),
    )
    return rows, deps


async def _read(deps: ConversationAuthorizerDependencies, *, agent_id: UUID, customer_id: UUID, user_id: UUID) -> Any:
    """authorize a user-initiated conversation read.

    :param deps: authorizer bundle
    :ptype deps: ConversationAuthorizerDependencies
    :param agent_id: agent owning the conversation namespace
    :ptype agent_id: UUID
    :param customer_id: customer owning the conversation namespace
    :ptype customer_id: UUID
    :param user_id: calling user
    :ptype user_id: UUID
    :return: resolved namespace
    :rtype: Any
    """
    return await authorize_conversation_access(
        action=ACTION_CONVERSATION_READ,
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

    with pytest.raises(ConversationAccessDenied):
        await _read(deps, agent_id=agent_id, customer_id=customer_id, user_id=user_id)

    namespace = rows.namespace(namespace_type=CONVERSATION_NAMESPACE_TYPE, agent_id=agent_id, customer_id=customer_id)
    await ensure_conversation_owner_assignment(user_id=user_id, namespace=namespace, deps=deps)

    resolved = await _read(deps, agent_id=agent_id, customer_id=customer_id, user_id=user_id)
    assert resolved.id == namespace.id


async def test_a_grant_on_a_second_agent_is_honoured_when_the_membership_already_existed() -> None:
    """the owner group spans agents, so only evicting its assignment entries lets the new grant through."""
    rows, deps = _wiring()
    first_agent, second_agent, customer_id, user_id = uuid7(), uuid7(), uuid7(), uuid7()
    first = rows.namespace(namespace_type=CONVERSATION_NAMESPACE_TYPE, agent_id=first_agent, customer_id=customer_id)
    await ensure_conversation_owner_assignment(user_id=user_id, namespace=first, deps=deps)
    await _read(deps, agent_id=first_agent, customer_id=customer_id, user_id=user_id)

    with pytest.raises(ConversationAccessDenied):
        await _read(deps, agent_id=second_agent, customer_id=customer_id, user_id=user_id)

    second = rows.namespace(namespace_type=CONVERSATION_NAMESPACE_TYPE, agent_id=second_agent, customer_id=customer_id)
    await ensure_conversation_owner_assignment(user_id=user_id, namespace=second, deps=deps)

    resolved = await _read(deps, agent_id=second_agent, customer_id=customer_id, user_id=user_id)
    assert resolved.id == second.id


async def test_what_the_ensure_wrote_is_broadcast_and_a_repeat_ensure_broadcasts_nothing() -> None:
    """other pods hold the same stale entries; a no-op ensure runs on every send and must stay silent."""
    publisher = _RecordingPublisher()
    rows, deps = _wiring(publisher)
    agent_id, customer_id, user_id = uuid7(), uuid7(), uuid7()
    namespace = rows.namespace(namespace_type=CONVERSATION_NAMESPACE_TYPE, agent_id=agent_id, customer_id=customer_id)

    await ensure_conversation_owner_assignment(user_id=user_id, namespace=namespace, deps=deps)

    group_id = rows.assignments[0].group_id
    assert publisher.sent == [
        MembershipInvalidatePayload(actor_type="user", actor_id=user_id),
        AssignmentInvalidatePayload(group_id=group_id),
    ]

    await ensure_conversation_owner_assignment(user_id=user_id, namespace=namespace, deps=deps)

    assert len(publisher.sent) == 2
