"""canonical Collection-backed implementations of the rbac loader Protocols.

every 3tears app feeding the unified evaluator wires these two
classes against the canonical rbac Collections (in
:mod:`threetears.agent.acl.collections`). the Collections speak to
whichever L3 pool is bound — direct asyncpg in a hub-style deployment,
:class:`threetears.core.backends.nats_proxy.NatsProxyL3Backend` in
each agent pod — so the SAME loader class works on both sides; the
pool choice is a registry-construction concern, not a loader concern.

splitting along ``Membership`` vs ``Grant`` lines mirrors the
:class:`threetears.agent.acl.AclCache` cache-layer split: membership
is keyed by actor identity, grants are keyed by ``(group, namespace)``
(or ``(group, namespace_type, customer)`` for the type_customer
scope). the evaluator visits the loaders in two distinct phases for
the same reason.
"""

from __future__ import annotations

from uuid import UUID

from threetears.observe import get_logger

from threetears.agent.acl.collections import (
    GroupCollection,
    GroupMemberCollection,
    RoleAssignmentCollection,
    RoleCollection,
)
from threetears.agent.acl.types import (
    Group,
    GroupMembership,
    Namespace,
    Role,
    RoleAssignment,
)

log = get_logger(__name__)

__all__ = [
    "CollectionGrantLoader",
    "CollectionMembershipLoader",
]


class CollectionMembershipLoader:
    """:class:`MembershipLoader` over :class:`GroupMemberCollection`.

    resolves a single actor (user or agent) to the tuple of
    :class:`GroupMembership` rows that name it as a member. delegates
    the SQL + dataclass construction to
    :meth:`GroupMemberCollection.load_for_user` /
    :meth:`~GroupMemberCollection.load_for_agent` so the persistence
    shape lives in exactly one place.

    method signatures match the
    :class:`threetears.agent.acl.MembershipLoader` Protocol exactly so
    :func:`evaluate_decision` / :func:`evaluate_with_trail` accept
    the loader unchanged. returns ``tuple[GroupMembership, ...]`` (not
    ``list``) because the Protocol specifies a tuple; the Collection
    returns a list for general use and the loader freezes it here.

    :param collection: three-tier Collection fronting ``group_members``
    :ptype collection: GroupMemberCollection
    """

    def __init__(self, collection: GroupMemberCollection) -> None:
        """capture the Collection reference.

        :param collection: three-tier Collection fronting
            ``group_members``
        :ptype collection: GroupMemberCollection
        :return: nothing
        :rtype: None
        """
        self._collection = collection

    async def load_for_user(
        self, user_id: UUID,
    ) -> tuple[GroupMembership, ...]:
        """return every membership row naming ``user_id`` as a user member.

        :param user_id: user UUID to resolve
        :ptype user_id: UUID
        :return: tuple of memberships (possibly empty)
        :rtype: tuple[GroupMembership, ...]
        """
        memberships = await self._collection.load_for_user(user_id)
        return tuple(memberships)

    async def load_for_agent(
        self, agent_id: UUID,
    ) -> tuple[GroupMembership, ...]:
        """return every membership row naming ``agent_id`` as an agent member.

        :param agent_id: agent UUID to resolve
        :ptype agent_id: UUID
        :return: tuple of memberships (possibly empty)
        :rtype: tuple[GroupMembership, ...]
        """
        memberships = await self._collection.load_for_agent(agent_id)
        return tuple(memberships)


class CollectionGrantLoader:
    """:class:`GrantLoader` over the rbac Collections.

    implements the three loader methods on the
    :class:`threetears.agent.acl.GrantLoader` Protocol:
    :meth:`load_assignments_for_groups`, :meth:`load_roles`, and
    :meth:`load_groups`. each delegates to the corresponding method
    on :class:`RoleAssignmentCollection`, :class:`RoleCollection`, or
    :class:`GroupCollection`.

    :param assignment_collection: three-tier Collection fronting
        ``role_assignments``
    :ptype assignment_collection: RoleAssignmentCollection
    :param role_collection: three-tier Collection fronting ``roles``
    :ptype role_collection: RoleCollection
    :param group_collection: three-tier Collection fronting ``groups``
    :ptype group_collection: GroupCollection
    """

    def __init__(
        self,
        assignment_collection: RoleAssignmentCollection,
        role_collection: RoleCollection,
        group_collection: GroupCollection,
    ) -> None:
        """capture the three Collection references.

        :param assignment_collection: three-tier Collection fronting
            ``role_assignments``
        :ptype assignment_collection: RoleAssignmentCollection
        :param role_collection: three-tier Collection fronting
            ``roles``
        :ptype role_collection: RoleCollection
        :param group_collection: three-tier Collection fronting
            ``groups``
        :ptype group_collection: GroupCollection
        :return: nothing
        :rtype: None
        """
        self._assignment_collection = assignment_collection
        self._role_collection = role_collection
        self._group_collection = group_collection

    async def load_assignments_for_groups(
        self,
        group_ids: tuple[UUID, ...],
        namespace: Namespace,
    ) -> tuple[RoleAssignment, ...]:
        """return assignments held by ``group_ids`` covering ``namespace``.

        delegates the bulk fetch to
        :meth:`RoleAssignmentCollection.load_for_groups` (which over-
        returns every assignment the groups hold, without a namespace
        filter) and applies the namespace-coverage filter here via
        :meth:`RoleAssignment.covers`. matches the Protocol shape: the
        evaluator itself re-checks coverage but loaders are permitted
        to pre-filter, and the pre-filter trims the wire size between
        loader and evaluator.

        :param group_ids: tuple of group UUIDs to inspect
        :ptype group_ids: tuple[UUID, ...]
        :param namespace: namespace under evaluation
        :ptype namespace: Namespace
        :return: tuple of assignments whose scope covers ``namespace``
        :rtype: tuple[RoleAssignment, ...]
        """
        result: tuple[RoleAssignment, ...] = ()
        if group_ids:
            assignments = await self._assignment_collection.load_for_groups(
                group_ids,
            )
            result = tuple(
                assignment
                for assignment in assignments
                if assignment.covers(namespace)
            )
        return result

    async def load_roles(
        self, role_ids: tuple[UUID, ...],
    ) -> dict[UUID, Role]:
        """resolve ``role_ids`` to :class:`Role` rows.

        delegates to :meth:`RoleCollection.get_many`, which emits
        :class:`Role` instances with the JSONB ``permissions`` column
        already coerced into the ``dict[str, frozenset[str]]`` shape
        the evaluator expects.

        :param role_ids: tuple of role UUIDs to resolve
        :ptype role_ids: tuple[UUID, ...]
        :return: mapping role_id -> Role for ids that exist
        :rtype: dict[UUID, Role]
        """
        roles = await self._role_collection.get_many(role_ids)
        return {role.id: role for role in roles}

    async def load_groups(
        self, group_ids: tuple[UUID, ...],
    ) -> dict[UUID, object]:
        """resolve ``group_ids`` to :class:`Group` rows.

        delegates to :meth:`GroupCollection.get_many` and maps the
        returned :class:`GroupEntity` instances into the
        :class:`Group` dataclass the evaluator's trail mode consumes
        (only ``id`` / ``name`` / ``customer_id`` are read). the
        Protocol's ``dict[UUID, object]`` return type admits this
        shape.

        :param group_ids: tuple of group UUIDs to resolve
        :ptype group_ids: tuple[UUID, ...]
        :return: mapping group_id -> :class:`Group`
        :rtype: dict[UUID, object]
        """
        entities = await self._group_collection.get_many(group_ids)
        return {
            entity.id: Group(
                id=entity.id,
                name=entity.name,
                customer_id=entity.customer_id,
            )
            for entity in entities
        }
