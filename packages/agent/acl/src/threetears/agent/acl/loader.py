"""i/o protocols the evaluator and cache call into.

the evaluator is pure python — it never opens a database connection
and never publishes a NATS message. all of its inputs come from one
of two narrow protocols:

- :class:`MembershipLoader` — "what groups does this actor belong to?"
- :class:`GrantLoader` — "what assignments does this group hold that
  could possibly cover this namespace?" plus the
  ``role_id -> Role`` lookup that turns each assignment into the
  action set it would contribute.

the broker (in ``3tears.hub.broker``) implements these protocols
against its postgres pool. each agent pod implements them against
its NATS-proxied L3 client. both implementations land in the
respective callers; this package owns only the shape.

splitting along ``Membership`` vs ``Grant`` lines mirrors the cache
layering: membership is keyed by actor identity, grants are keyed by
``(group, namespace)`` (or ``(group, namespace_type, customer)`` for
the type_customer scope). the evaluator visits the loaders in two
distinct phases for the same reason.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable
from uuid import UUID

from threetears.agent.acl.types import (
    GroupMembership,
    Namespace,
    Role,
    RoleAssignment,
)

__all__ = [
    "GrantLoader",
    "MembershipLoader",
]


@runtime_checkable
class MembershipLoader(Protocol):
    """fetch the group memberships an actor holds.

    implementations resolve a single actor (one of user / agent) to
    the set of :class:`GroupMembership` rows that name it as a
    member. the result is unsorted; the evaluator does its own
    deterministic ordering when building trails.

    the evaluator calls :meth:`load_for_user` for the user side of an
    intersection and :meth:`load_for_agent` for the agent side; the
    two sides never cross-pollinate even when a group has mixed
    membership.
    """

    async def load_for_user(self, user_id: UUID) -> tuple[GroupMembership, ...]:
        """return every membership row the user belongs to.

        :param user_id: user UUID to resolve
        :ptype user_id: UUID
        :return: tuple of memberships (empty if the user is in no
            groups; deterministic ordering is the caller's
            responsibility)
        :rtype: tuple[GroupMembership, ...]
        """

    async def load_for_agent(self, agent_id: UUID) -> tuple[GroupMembership, ...]:
        """return every membership row the agent belongs to.

        :param agent_id: agent UUID to resolve
        :ptype agent_id: UUID
        :return: tuple of memberships
        :rtype: tuple[GroupMembership, ...]
        """


@runtime_checkable
class GrantLoader(Protocol):
    """fetch the assignments + roles + groups the evaluator needs.

    after :class:`MembershipLoader` has produced the actor's group
    set, the evaluator asks this loader for every assignment those
    groups hold, restricted to assignments that could conceivably
    cover ``namespace``. concrete implementations are free to filter
    aggressively (only assignments whose scope matches the namespace,
    its type+customer, or "all") since the evaluator re-checks
    coverage anyway.

    the evaluator separately fetches :class:`Role` rows and
    :class:`Group` rows by id so the trail mode can name them
    legibly.
    """

    async def load_assignments_for_groups(
        self,
        group_ids: tuple[UUID, ...],
        namespace: Namespace,
    ) -> tuple[RoleAssignment, ...]:
        """return assignments held by ``group_ids`` covering ``namespace``.

        loaders may over-return (e.g. every assignment for the groups
        with no scope filter); the evaluator re-checks
        :meth:`RoleAssignment.covers`. loaders may NOT under-return —
        if an assignment exists in the database that the evaluator
        would have considered, the loader must surface it or the
        evaluator's decision is wrong.

        :param group_ids: tuple of group UUIDs to inspect
        :ptype group_ids: tuple[UUID, ...]
        :param namespace: namespace under evaluation; loaders may use
            the namespace's customer, type, and id to scope the query
        :ptype namespace: Namespace
        :return: tuple of role assignments
        :rtype: tuple[RoleAssignment, ...]
        """

    async def load_roles(self, role_ids: tuple[UUID, ...]) -> dict[UUID, Role]:
        """resolve a tuple of role ids to their :class:`Role` rows.

        the evaluator collects ``role_id`` values from the assignments
        it just fetched, dedupes them, and asks this method for the
        full role rows. the result is a mapping keyed by role id; ids
        without a row in the database are absent from the mapping
        (the evaluator skips assignments whose role does not resolve
        rather than failing the whole call).

        :param role_ids: tuple of role UUIDs to resolve
        :ptype role_ids: tuple[UUID, ...]
        :return: mapping role_id -> Role for ids that exist
        :rtype: dict[UUID, Role]
        """

    async def load_groups(self, group_ids: tuple[UUID, ...]) -> dict[UUID, object]:
        """resolve a tuple of group ids to their :class:`Group` rows.

        same shape as :meth:`load_roles`. used by trail-mode
        evaluation to attach the human-readable group name and
        customer scope to each :class:`Trail`.

        the return type is ``dict[UUID, object]`` so concrete loaders
        can return their own ORM rows; the evaluator only reads
        ``id``, ``name``, and ``customer_id`` attributes through the
        :class:`Group` shape.

        :param group_ids: tuple of group UUIDs to resolve
        :ptype group_ids: tuple[UUID, ...]
        :return: mapping group_id -> object exposing the
            :class:`Group` shape
        :rtype: dict[UUID, object]
        """
