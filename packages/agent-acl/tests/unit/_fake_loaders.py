"""in-memory fake :class:`MembershipLoader` + :class:`GrantLoader`.

every evaluator test uses these instead of mocks. the fakes hold
the full dataset in two dictionaries and answer queries against
that dataset directly. construction takes lists of memberships,
groups, roles, and assignments; the resulting object satisfies
both loader Protocols.

design notes:

- the fakes dedupe nothing on insert; the evaluator is responsible
  for deduping ids across sides. tests that want to verify the
  evaluator's dedupe pass duplicate rows in.
- :meth:`FakeGrantLoader.load_assignments_for_groups` returns
  every assignment held by any of the requested groups, regardless
  of whether the assignment's scope actually covers the namespace.
  the evaluator re-checks via :meth:`RoleAssignment.covers`; the
  test fixture deliberately over-returns to exercise that re-check
  branch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID

from threetears.agent.acl.cache import AclCache
from threetears.agent.acl.types import (
    Group,
    GroupMembership,
    MemberType,
    Namespace,
    Role,
    RoleAssignment,
)

__all__ = ["FakeStore", "make_cache"]


def make_cache(store: "FakeStore", ttl_seconds: int = 60) -> AclCache:
    """build an :class:`AclCache` whose loaders are backed by ``store``.

    convenience used by every evaluator unit test that needs a
    cache-shaped argument. ``store`` itself satisfies both loader
    Protocols, so the cache constructs cleanly with a single handle.

    :param store: in-memory fake fixture
    :ptype store: FakeStore
    :param ttl_seconds: cache ttl; defaults to sixty seconds
    :ptype ttl_seconds: int
    :return: cache wired against ``store``
    :rtype: AclCache
    """
    return AclCache(
        membership_loader=store,
        grant_loader=store,
        ttl_seconds=ttl_seconds,
    )


@dataclass
class FakeStore:
    """in-memory store satisfying both loader Protocols.

    construct empty and append rows via the ``add_*`` methods, or
    supply rows directly through the dataclass fields.

    :ivar memberships: list of :class:`GroupMembership` rows
    :ivar groups: dict of group_id -> :class:`Group`
    :ivar roles: dict of role_id -> :class:`Role`
    :ivar assignments: list of :class:`RoleAssignment` rows
    """

    memberships: list[GroupMembership] = field(default_factory=list)
    groups: dict[UUID, Group] = field(default_factory=dict)
    roles: dict[UUID, Role] = field(default_factory=dict)
    assignments: list[RoleAssignment] = field(default_factory=list)

    # ---------- builder helpers -------------------------------------

    def add_group(self, group: Group) -> Group:
        """register a :class:`Group` in the fake store.

        :param group: group to add
        :ptype group: Group
        :return: the same group (for chaining)
        :rtype: Group
        """
        self.groups[group.id] = group
        return group

    def add_role(self, role: Role) -> Role:
        """register a :class:`Role` in the fake store.

        :param role: role to add
        :ptype role: Role
        :return: the same role
        :rtype: Role
        """
        self.roles[role.id] = role
        return role

    def add_membership(self, membership: GroupMembership) -> GroupMembership:
        """register a :class:`GroupMembership` in the fake store.

        :param membership: membership to add
        :ptype membership: GroupMembership
        :return: the same membership
        :rtype: GroupMembership
        """
        self.memberships.append(membership)
        return membership

    def add_assignment(self, assignment: RoleAssignment) -> RoleAssignment:
        """register a :class:`RoleAssignment` in the fake store.

        :param assignment: assignment to add
        :ptype assignment: RoleAssignment
        :return: the same assignment
        :rtype: RoleAssignment
        """
        self.assignments.append(assignment)
        return assignment

    # ---------- MembershipLoader ------------------------------------

    async def load_for_user(self, user_id: UUID) -> tuple[GroupMembership, ...]:
        """return every membership row whose member is the user.

        :param user_id: user UUID to resolve
        :ptype user_id: UUID
        :return: tuple of memberships
        :rtype: tuple[GroupMembership, ...]
        """
        result = tuple(
            m for m in self.memberships
            if m.member_type == MemberType.USER and m.member_id == user_id
        )
        return result

    async def load_for_agent(self, agent_id: UUID) -> tuple[GroupMembership, ...]:
        """return every membership row whose member is the agent.

        :param agent_id: agent UUID to resolve
        :ptype agent_id: UUID
        :return: tuple of memberships
        :rtype: tuple[GroupMembership, ...]
        """
        result = tuple(
            m for m in self.memberships
            if m.member_type == MemberType.AGENT and m.member_id == agent_id
        )
        return result

    # ---------- GrantLoader -----------------------------------------

    async def load_assignments_for_groups(
        self,
        group_ids: tuple[UUID, ...],
        namespace: Namespace,
    ) -> tuple[RoleAssignment, ...]:
        """return every assignment held by any group in ``group_ids``.

        deliberately over-returns (no scope filtering) so the
        evaluator's re-check branch via :meth:`RoleAssignment.covers`
        is exercised.

        :param group_ids: tuple of group UUIDs to inspect
        :ptype group_ids: tuple[UUID, ...]
        :param namespace: namespace under evaluation (unused; the
            fake intentionally returns everything to exercise the
            evaluator's coverage re-check)
        :ptype namespace: Namespace
        :return: tuple of assignments
        :rtype: tuple[RoleAssignment, ...]
        """
        ids = set(group_ids)
        result = tuple(a for a in self.assignments if a.group_id in ids)
        _ = namespace  # intentionally unused; evaluator re-checks scope
        return result

    async def load_roles(self, role_ids: tuple[UUID, ...]) -> dict[UUID, Role]:
        """resolve role ids to :class:`Role` rows.

        :param role_ids: tuple of role UUIDs
        :ptype role_ids: tuple[UUID, ...]
        :return: mapping role_id -> Role for ids that exist
        :rtype: dict[UUID, Role]
        """
        result = {rid: self.roles[rid] for rid in role_ids if rid in self.roles}
        return result

    async def load_groups(
        self, group_ids: tuple[UUID, ...],
    ) -> dict[UUID, object]:
        """resolve group ids to :class:`Group` rows.

        :param group_ids: tuple of group UUIDs
        :ptype group_ids: tuple[UUID, ...]
        :return: mapping group_id -> Group for ids that exist
        :rtype: dict[UUID, object]
        """
        result: dict[UUID, object] = {
            gid: self.groups[gid] for gid in group_ids if gid in self.groups
        }
        return result
