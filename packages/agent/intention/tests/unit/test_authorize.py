"""unit tests for the intention authorize helper (owner-only RBAC, v0.15.0).

Intention authz is the evaluator's agent-owner short-circuit: the owning
agent's own calls are allowed grant-free, and any non-owner agent is
denied (there is no user-grant path in v1). These tests build the
authorizer bundle from empty in-memory ACL loaders -- the owner path
never consults them, and the deny path proves an ungranted non-owner is
refused.
"""

from __future__ import annotations

from typing import Any
from uuid import NAMESPACE_DNS, UUID, uuid4, uuid5

import pytest

from threetears.agent.acl import AclCache

from threetears.agent.intention.authorize import (
    ACTION_INTENTION_READ,
    ACTION_INTENTION_WRITE,
    INTENTION_NAMESPACE_TYPE,
    IntentionAccessDenied,
    IntentionAuthorizerDependencies,
    authorize_intention_access,
    intention_namespace_name,
)


class _EmptyMembershipLoader:
    """membership loader returning no memberships for anyone."""

    async def load_for_user(self, user_id: UUID) -> tuple[Any, ...]:
        _ = user_id
        return ()

    async def load_for_agent(self, agent_id: UUID) -> tuple[Any, ...]:
        _ = agent_id
        return ()

    async def load_for_group(self, group_id: UUID) -> tuple[Any, ...]:
        _ = group_id
        return ()


class _EmptyGrantLoader:
    """grant loader returning no assignments / roles / groups."""

    async def load_assignments_for_groups(self, group_ids: tuple[UUID, ...], namespace: Any) -> tuple[Any, ...]:
        _ = group_ids, namespace
        return ()

    async def load_roles(self, role_ids: tuple[UUID, ...]) -> dict[UUID, Any]:
        _ = role_ids
        return {}

    async def load_groups(self, group_ids: tuple[UUID, ...]) -> dict[UUID, Any]:
        _ = group_ids
        return {}


def _deps() -> IntentionAuthorizerDependencies:
    return IntentionAuthorizerDependencies(
        acl_cache=AclCache(
            membership_loader=_EmptyMembershipLoader(),
            grant_loader=_EmptyGrantLoader(),
        ),
    )


class TestIntentionNamespaceName:
    def test_shape(self) -> None:
        agent_id = UUID("019470a8-b5c3-7def-8123-456789abcdef")
        customer_id = UUID("11112222-3333-4444-5555-666677778888")
        assert intention_namespace_name(agent_id, customer_id) == (
            "intentions.019470a8b5c37def8123456789abcdef.11112222333344445555666677778888"
        )

    def test_two_agents_minted_in_one_millisecond_get_distinct_names(self) -> None:
        """a subtree grant is judged by this name; a name shared by two agents is a grant shared by both.

        a uuid7 leads with its timestamp, so a name cut from its head is the same for every
        agent minted in the same moment.
        """
        agent_a = UUID("019470a8-b5c3-7def-8123-456789abcdef")
        agent_b = UUID("019470a8-b5c3-7a01-9fed-cba987654321")
        customer_id = UUID("019470a8-b5c4-7000-8000-000000000001")
        assert agent_a.int >> 80 == agent_b.int >> 80, "the fixture must share the uuid7 timestamp"
        assert intention_namespace_name(agent_a, customer_id) != intention_namespace_name(agent_b, customer_id)

    def test_namespace_type_constant(self) -> None:
        assert INTENTION_NAMESPACE_TYPE == "intention"


class TestAuthorizeIntentionAccess:
    async def test_owner_read_allowed_grant_free(self) -> None:
        """the owning agent reads its own intentions without any grant."""
        agent_id = uuid4()
        customer_id = uuid4()
        # no raise == allowed
        await authorize_intention_access(
            action=ACTION_INTENTION_READ,
            agent_id=agent_id,
            customer_id=customer_id,
            caller_agent_id=agent_id,
            deps=_deps(),
        )

    async def test_owner_write_allowed_grant_free(self) -> None:
        """the owning agent writes its own intentions without any grant."""
        agent_id = uuid4()
        customer_id = uuid4()
        await authorize_intention_access(
            action=ACTION_INTENTION_WRITE,
            agent_id=agent_id,
            customer_id=customer_id,
            caller_agent_id=agent_id,
            deps=_deps(),
        )

    async def test_non_owner_agent_denied(self) -> None:
        """a different agent (not the namespace owner) has no grant -> denied."""
        agent_id = uuid4()
        other_agent_id = uuid4()
        customer_id = uuid4()
        with pytest.raises(IntentionAccessDenied, match="evaluator denied"):
            await authorize_intention_access(
                action=ACTION_INTENTION_READ,
                agent_id=agent_id,
                customer_id=customer_id,
                caller_agent_id=other_agent_id,
                deps=_deps(),
            )

    def test_namespace_id_deterministic(self) -> None:
        """the descriptor id is a stable uuid5 of the (agent, customer) pair."""
        from threetears.agent.intention.authorize import _intention_namespace_id

        agent_id = uuid4()
        customer_id = uuid4()
        assert _intention_namespace_id(agent_id, customer_id) == uuid5(
            NAMESPACE_DNS,
            f"threetears.namespaces.intention.{agent_id.hex}.{customer_id.hex}",
        )
