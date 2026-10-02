"""one in-memory rbac table set that the ensure writes INTO and the evaluator reads FROM.

a test of "the grant an ensure just wrote is visible to the next
authorization" needs the write and the read to meet in the same rows --
otherwise the test can only assert that some method was called, which is
exactly how a cache that answers the read from a stale entry went unseen.
:class:`RbacRows` is that meeting point: it answers the two loader
protocols the :class:`~threetears.agent.acl.AclCache` misses into, and it
hands out collection stand-ins whose writes land in the same lists.

none of the stand-ins is a ``Fake<Name>``: each serves only the methods the
ensure path calls, so a parity marker would claim a surface comparison
that is not run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import UUID, uuid4

from threetears.agent.acl import (
    Group,
    GroupMembership,
    MemberType,
    Namespace,
    Role,
    RoleAssignment,
    ScopeType,
)

__all__ = ["NamespaceRef", "RbacRows"]


@dataclass(frozen=True)
class NamespaceRef:
    """the five fields the authorizer reads off a resolved namespace.

    :ivar id: namespace UUID
    :ivar customer_id: owning customer UUID
    :ivar namespace_type: namespace type discriminator
    :ivar owner_agent_id: owning agent UUID
    :ivar owner_namespace: owning agent's namespace name
    """

    id: UUID
    customer_id: UUID
    namespace_type: str
    owner_agent_id: UUID
    owner_namespace: str


class _Row:
    """entity stand-in carrying the dict an ensure built it from.

    :ivar data: column values
    """

    def __init__(self, data: dict[str, Any], *, is_new: bool, collection: Any) -> None:
        """store the column values.

        :param data: column values
        :ptype data: dict[str, Any]
        :param is_new: whether the row is new (unused)
        :ptype is_new: bool
        :param collection: owning collection (unused)
        :ptype collection: Any
        :return: nothing
        :rtype: None
        """
        _ = is_new, collection
        self.data = data


@dataclass
class RbacRows:
    """the rbac rows, both loaders, and collection stand-ins writing into them.

    :ivar roles: role rows
    :ivar groups: group rows keyed by id
    :ivar memberships: membership rows
    :ivar assignments: assignment rows
    :ivar namespaces: namespace rows keyed by ``(type, owner_agent_id, customer_id)``
    """

    roles: list[Role] = field(default_factory=list)
    groups: dict[UUID, Group] = field(default_factory=dict)
    memberships: list[GroupMembership] = field(default_factory=list)
    assignments: list[RoleAssignment] = field(default_factory=list)
    namespaces: dict[tuple[str, UUID, UUID], NamespaceRef] = field(default_factory=dict)

    # ---------- MembershipLoader ------------------------------------

    async def load_for_user(self, user_id: UUID) -> tuple[GroupMembership, ...]:
        """return the user's membership rows.

        :param user_id: user UUID
        :ptype user_id: UUID
        :return: memberships naming the user
        :rtype: tuple[GroupMembership, ...]
        """
        return tuple(m for m in self.memberships if m.member_type == MemberType.USER and m.member_id == user_id)

    async def load_for_agent(self, agent_id: UUID) -> tuple[GroupMembership, ...]:
        """return the agent's membership rows.

        :param agent_id: agent UUID
        :ptype agent_id: UUID
        :return: memberships naming the agent
        :rtype: tuple[GroupMembership, ...]
        """
        return tuple(m for m in self.memberships if m.member_type == MemberType.AGENT and m.member_id == agent_id)

    async def load_for_group(self, group_id: UUID) -> tuple[GroupMembership, ...]:
        """return a group's parent-group rows -- none; these groups are flat.

        :param group_id: child group UUID
        :ptype group_id: UUID
        :return: empty tuple
        :rtype: tuple[GroupMembership, ...]
        """
        _ = group_id
        return ()

    # ---------- GrantLoader -----------------------------------------

    async def load_assignments_for_groups(
        self,
        group_ids: tuple[UUID, ...],
        namespace: Namespace,
    ) -> tuple[RoleAssignment, ...]:
        """return every assignment held by the groups; the evaluator re-checks scope.

        :param group_ids: group UUIDs
        :ptype group_ids: tuple[UUID, ...]
        :param namespace: namespace under evaluation (unused)
        :ptype namespace: Namespace
        :return: assignments
        :rtype: tuple[RoleAssignment, ...]
        """
        _ = namespace
        return tuple(a for a in self.assignments if a.group_id in set(group_ids))

    async def load_roles(self, role_ids: tuple[UUID, ...]) -> dict[UUID, Role]:
        """resolve role rows.

        :param role_ids: role UUIDs
        :ptype role_ids: tuple[UUID, ...]
        :return: role rows by id
        :rtype: dict[UUID, Role]
        """
        return {r.id: r for r in self.roles if r.id in set(role_ids)}

    async def load_groups(self, group_ids: tuple[UUID, ...]) -> dict[UUID, object]:
        """resolve group rows.

        :param group_ids: group UUIDs
        :ptype group_ids: tuple[UUID, ...]
        :return: group rows by id
        :rtype: dict[UUID, object]
        """
        return {gid: self.groups[gid] for gid in group_ids if gid in self.groups}

    # ---------- collection stand-ins --------------------------------

    def namespace(self, *, namespace_type: str, agent_id: UUID, customer_id: UUID) -> NamespaceRef:
        """materialize (or return) the namespace row for an (agent, customer) pair.

        :param namespace_type: namespace type discriminator
        :ptype namespace_type: str
        :param agent_id: owning agent UUID
        :ptype agent_id: UUID
        :param customer_id: owning customer UUID
        :ptype customer_id: UUID
        :return: namespace row
        :rtype: NamespaceRef
        """
        key = (namespace_type, agent_id, customer_id)
        if key not in self.namespaces:
            self.namespaces[key] = NamespaceRef(
                id=uuid4(),
                customer_id=customer_id,
                namespace_type=namespace_type,
                owner_agent_id=agent_id,
                owner_namespace=f"agents.{agent_id.hex}",
            )
        return self.namespaces[key]

    def collections(self) -> dict[str, Any]:
        """the five collection stand-ins, keyed by the bundle's keyword names.

        :return: keyword arguments for an authorizer dependency bundle
        :rtype: dict[str, Any]
        """
        return {
            "namespace_collection": _NamespaceRows(self),
            "group_collection": _GroupRows(self),
            "group_member_collection": _MemberRows(self),
            "role_collection": _RoleRows(self),
            "role_assignment_collection": _AssignmentRows(self),
        }


class _NamespaceRows:
    """namespace lookups against :attr:`RbacRows.namespaces`."""

    def __init__(self, rows: RbacRows) -> None:
        """bind the shared rows.

        :param rows: shared rbac rows
        :ptype rows: RbacRows
        :return: nothing
        :rtype: None
        """
        self._rows = rows

    async def get_by_owner_and_customer(
        self,
        *,
        namespace_type: str,
        owner_agent_id: UUID,
        customer_id: UUID,
    ) -> NamespaceRef:
        """return the namespace row, creating it on first ask.

        :param namespace_type: namespace type discriminator
        :ptype namespace_type: str
        :param owner_agent_id: owning agent UUID
        :ptype owner_agent_id: UUID
        :param customer_id: owning customer UUID
        :ptype customer_id: UUID
        :return: namespace row
        :rtype: NamespaceRef
        """
        return self._rows.namespace(namespace_type=namespace_type, agent_id=owner_agent_id, customer_id=customer_id)


class _RoleRows:
    """builtin-role lookups against :attr:`RbacRows.roles`."""

    def __init__(self, rows: RbacRows) -> None:
        """bind the shared rows.

        :param rows: shared rbac rows
        :ptype rows: RbacRows
        :return: nothing
        :rtype: None
        """
        self._rows = rows

    async def list_builtin(self) -> list[Role]:
        """return every builtin role.

        :return: builtin roles
        :rtype: list[Role]
        """
        return [r for r in self._rows.roles if r.is_built_in]


class _GroupRows:
    """``groups`` reads and writes against :attr:`RbacRows.groups`."""

    entity_class = _Row

    def __init__(self, rows: RbacRows) -> None:
        """bind the shared rows.

        :param rows: shared rbac rows
        :ptype rows: RbacRows
        :return: nothing
        :rtype: None
        """
        self._rows = rows

    async def get(self, pk: tuple[str, UUID]) -> Group | None:
        """read a group by its composite key.

        :param pk: ``(row_scope, group_id)``
        :ptype pk: tuple[str, UUID]
        :return: group row or ``None``
        :rtype: Group | None
        """
        return self._rows.groups.get(pk[1])

    async def save_entity(self, entity: _Row) -> None:
        """write a group row.

        :param entity: row built by the ensure
        :ptype entity: _Row
        :return: nothing
        :rtype: None
        """
        data = entity.data
        self._rows.groups[data["group_id"]] = Group(
            id=data["group_id"],
            name=data["name"],
            customer_id=data["customer_id"],
        )


class _MemberRows:
    """``group_members`` reads and writes against :attr:`RbacRows.memberships`."""

    entity_class = _Row

    def __init__(self, rows: RbacRows) -> None:
        """bind the shared rows.

        :param rows: shared rbac rows
        :ptype rows: RbacRows
        :return: nothing
        :rtype: None
        """
        self._rows = rows
        self._ids: dict[tuple[UUID, UUID], GroupMembership] = {}

    async def get(self, pk: tuple[UUID, UUID]) -> GroupMembership | None:
        """read a membership by its composite key.

        :param pk: ``(group_id, id)``
        :ptype pk: tuple[UUID, UUID]
        :return: membership row or ``None``
        :rtype: GroupMembership | None
        """
        return self._ids.get(pk)

    async def save_entity(self, entity: _Row) -> None:
        """write a membership row.

        :param entity: row built by the ensure
        :ptype entity: _Row
        :return: nothing
        :rtype: None
        """
        data = entity.data
        membership = GroupMembership(
            group_id=data["group_id"],
            member_type=MemberType(data["member_type"]),
            member_id=data["member_id"],
            customer_id=data["customer_id"],
        )
        self._ids[(data["group_id"], data["id"])] = membership
        self._rows.memberships.append(membership)


class _AssignmentRows:
    """``role_assignments`` ensure against :attr:`RbacRows.assignments`."""

    def __init__(self, rows: RbacRows) -> None:
        """bind the shared rows.

        :param rows: shared rbac rows
        :ptype rows: RbacRows
        :return: nothing
        :rtype: None
        """
        self._rows = rows

    async def ensure_group_role_assignment(
        self,
        *,
        group_id: UUID,
        role_id: UUID,
        scope_type: str,
        scope_id: UUID | None,
        managed_by: str = "manual",
    ) -> tuple[UUID, bool]:
        """insert the namespace-scoped assignment unless the tuple already exists.

        :param group_id: group UUID
        :ptype group_id: UUID
        :param role_id: role UUID
        :ptype role_id: UUID
        :param scope_type: scope discriminator; only ``"namespace"`` here
        :ptype scope_type: str
        :param scope_id: namespace UUID
        :ptype scope_id: UUID | None
        :param managed_by: provenance marker (unused)
        :ptype managed_by: str
        :return: ``(assignment_id, created)``
        :rtype: tuple[UUID, bool]
        """
        _ = managed_by
        assert scope_type == "namespace"
        for existing in self._rows.assignments:
            if (existing.group_id, existing.role_id, existing.scope_namespace_id) == (group_id, role_id, scope_id):
                return existing.id, False
        assignment = RoleAssignment(
            id=uuid4(),
            role_id=role_id,
            group_id=group_id,
            scope_type=ScopeType.NAMESPACE,
            scope_namespace_id=scope_id,
            scope_namespace_type=None,
            scope_customer_id=None,
        )
        self._rows.assignments.append(assignment)
        return assignment.id, True
