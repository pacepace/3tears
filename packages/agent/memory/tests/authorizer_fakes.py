"""the rbac collections a :class:`MemoryAuthorizerDependencies` bundle takes, with nothing behind them.

each fake IS the production collection (a subclass), built without a registry, config, pool or
L1 cache, so mypy checks every override against production and the bundle takes it where the
real one goes. the rows they serve are real entities (:func:`make_namespace`, :func:`make_role`)
holding the row a test names.

an entity the code under test builds through ``entity_class(..., collection=fake)`` keeps its
row in memory: :meth:`write_to_cache_sync` reports no L1, which is the entity's own fallback.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

from threetears.agent.acl import (
    GroupCollection,
    GroupMemberCollection,
    NamespaceCollection,
    NamespaceEntity,
    Role,
    RoleAssignmentCollection,
    RoleCollection,
    RoleEntity,
)
from threetears.core.collections.base import ScanReadTicket
from threetears.core.entities.base import BaseEntity


def make_namespace(
    *,
    id: UUID,
    customer_id: UUID,
    owner_agent_id: UUID | None = None,
    namespace_type: str = "memory",
    name: str | None = None,
) -> NamespaceEntity:
    """a real :class:`NamespaceEntity` holding the row a test names, with no collection behind it.

    :param id: namespace UUID
    :ptype id: UUID
    :param customer_id: owning customer UUID
    :ptype customer_id: UUID
    :param owner_agent_id: owning agent UUID
    :ptype owner_agent_id: UUID | None
    :param namespace_type: namespace type discriminator
    :ptype namespace_type: str
    :param name: namespace name; derived from the id when omitted
    :ptype name: str | None
    :return: the entity
    :rtype: NamespaceEntity
    """
    return NamespaceEntity(
        {
            "namespace_id": id,
            "name": name if name is not None else f"{namespace_type}:{id.hex}",
            "namespace_type": namespace_type,
            "owner_agent_id": owner_agent_id,
            "customer_id": customer_id,
        }
    )


def make_role(role: Role) -> RoleEntity:
    """the ``roles`` row :meth:`RoleCollection.find_platform_by_name` answers for ``role``.

    :param role: the role the row describes
    :ptype role: Role
    :return: the entity
    :rtype: RoleEntity
    """
    return RoleEntity(
        {
            "role_id": role.id,
            "name": role.name,
            "permissions": {resource: sorted(actions) for resource, actions in role.permissions.items()},
            "is_builtin": role.is_built_in,
        }
    )


class FakeNamespaceCollection(NamespaceCollection):
    """the production :class:`NamespaceCollection` answering one stable namespace per owner.

    every :meth:`get_by_owner_and_customer` call returns the same namespace per
    ``(namespace_type, owner_agent_id, customer_id)``, made on first ask, so concurrent calls
    within one test see stable ids.
    """

    def __init__(self) -> None:
        """build the fake without the production wiring: no registry, config, pool or L1 table."""
        self.by_owner: dict[tuple[str, UUID, UUID], NamespaceEntity] = {}

    def write_to_cache_sync(
        self,
        data: dict[str, Any],
        primary_key: str | tuple[str, ...] | None = None,
        *,
        read_since: ScanReadTicket | None = None,
    ) -> bool:
        """no L1: an entity built against this collection keeps its row itself."""
        _ = data, primary_key, read_since
        return False

    async def get_by_owner_and_customer(
        self,
        *,
        namespace_type: str,
        owner_agent_id: UUID | None,
        customer_id: UUID | None,
    ) -> NamespaceEntity | None:
        """the namespace for this owner, made on first ask.

        :param namespace_type: namespace type filter
        :ptype namespace_type: str
        :param owner_agent_id: owning agent UUID
        :ptype owner_agent_id: UUID | None
        :param customer_id: owning customer UUID
        :ptype customer_id: UUID | None
        :return: the namespace; None when either owner half is missing
        :rtype: NamespaceEntity | None
        """
        if owner_agent_id is None or customer_id is None:
            return None
        key = (namespace_type, owner_agent_id, customer_id)
        entity = self.by_owner.get(key)
        if entity is None:
            entity = make_namespace(
                id=uuid4(),
                namespace_type=namespace_type,
                owner_agent_id=owner_agent_id,
                customer_id=customer_id,
            )
            self.by_owner[key] = entity
        return entity

    async def save_entity(self, entity: BaseEntity, *, conn: Any = None) -> None:
        """no-op: namespaces are made through :meth:`get_by_owner_and_customer`."""
        _ = entity, conn

    async def get(self, entity_id: Any) -> NamespaceEntity | None:
        """a namespace already made, by id.

        :param entity_id: namespace UUID
        :ptype entity_id: Any
        :return: the namespace, or None when none was made with that id
        :rtype: NamespaceEntity | None
        """
        return next((entity for entity in self.by_owner.values() if entity.id == entity_id), None)


class FakeGroupCollection(GroupCollection):
    """the production :class:`GroupCollection` holding no groups; saves are dropped."""

    def __init__(self) -> None:
        """build the fake without the production wiring: no registry, config, pool or L1 table."""

    def write_to_cache_sync(
        self,
        data: dict[str, Any],
        primary_key: str | tuple[str, ...] | None = None,
        *,
        read_since: ScanReadTicket | None = None,
    ) -> bool:
        """no L1: an entity built against this collection keeps its row itself."""
        _ = data, primary_key, read_since
        return False

    async def get(self, entity_id: Any) -> None:
        """every lookup misses, so the ensure path takes the create branch."""
        _ = entity_id

    async def save_entity(self, entity: BaseEntity, *, conn: Any = None) -> None:
        """no-op save."""
        _ = entity, conn


class FakeGroupMemberCollection(GroupMemberCollection):
    """the production :class:`GroupMemberCollection` holding no memberships; saves are dropped."""

    def __init__(self) -> None:
        """build the fake without the production wiring: no registry, config, pool or L1 table."""

    def write_to_cache_sync(
        self,
        data: dict[str, Any],
        primary_key: str | tuple[str, ...] | None = None,
        *,
        read_since: ScanReadTicket | None = None,
    ) -> bool:
        """no L1: an entity built against this collection keeps its row itself."""
        _ = data, primary_key, read_since
        return False

    async def get(self, entity_id: Any) -> None:
        """every lookup misses, so the ensure path takes the create branch."""
        _ = entity_id

    async def save_entity(self, entity: BaseEntity, *, conn: Any = None) -> None:
        """no-op save."""
        _ = entity, conn


class FakeRoleCollection(RoleCollection):
    """the production :class:`RoleCollection` holding the platform roles a test names (none by default)."""

    def __init__(self, *roles: Role) -> None:
        """build the fake without the production wiring, serving ``roles`` by name.

        :param roles: the platform roles :meth:`find_platform_by_name` finds
        :ptype roles: Role
        """
        self.roles = roles

    async def find_platform_by_name(self, name: str) -> RoleEntity | None:
        """the platform role named ``name``, as :meth:`RoleCollection.find_platform_by_name` answers it.

        :param name: the role's name
        :ptype name: str
        :return: the role's row; None when none is named so
        :rtype: RoleEntity | None
        """
        return next((make_role(role) for role in self.roles if role.name == name), None)

    async def get(self, entity_id: Any) -> RoleEntity | None:
        """a role by id.

        :param entity_id: role UUID
        :ptype entity_id: Any
        :return: the role's row, or None
        :rtype: RoleEntity | None
        """
        return next((make_role(role) for role in self.roles if role.id == entity_id), None)


class FakeRoleAssignmentCollection(RoleAssignmentCollection):
    """the production :class:`RoleAssignmentCollection` whose ensure writes nothing."""

    def __init__(self) -> None:
        """build the fake without the production wiring: no registry, config, pool or L1 table."""

    async def ensure_group_role_assignment(
        self,
        *,
        group_id: UUID,
        role_id: UUID,
        scope_type: str,
        scope_id: UUID | None,
        managed_by: str = "manual",
    ) -> tuple[UUID, bool]:
        """a fresh assignment id and ``False``: nothing was inserted.

        :param group_id: group UUID (unused)
        :ptype group_id: UUID
        :param role_id: role UUID (unused)
        :ptype role_id: UUID
        :param scope_type: scope discriminator (unused)
        :ptype scope_type: str
        :param scope_id: scope UUID (unused)
        :ptype scope_id: UUID | None
        :param managed_by: who manages the assignment (unused)
        :ptype managed_by: str
        :return: ``(assignment_id, created)``, as production returns it
        :rtype: tuple[UUID, bool]
        """
        _ = group_id, role_id, scope_type, scope_id, managed_by
        return uuid4(), False
