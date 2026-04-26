"""shared test fixtures for agent-memory.

the ``permissive_memory_authorizer`` fixture constructs a
:class:`MemoryAuthorizerDependencies` bundle that authorizes every
evaluate call and materializes a deterministic namespace for every
``(agent_id, customer_id)`` a caller asks about. three-tier-task-01
phase D retired the bespoke resolver / ensurer callables; the fixture
now passes in in-memory Collection stand-ins with the method surface
:func:`authorize_memory_access` + :func:`ensure_memory_owner_assignment`
rely on. tests that DO exercise rbac construct their own bundle with
realistic loaders + real Collections.

using this fixture makes the test-only bypass explicit at every
call site — there is no silent ``authorizer=None`` shim in the
production API, so tests declare "permissive" in the same line
they pass the authorizer.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

import pytest

from threetears.agent.acl import (
    Group,
    GroupMembership,
    MemberType,
    Role,
    RoleAssignment,
    ScopeType,
)
from threetears.agent.memory.authorize import (
    ACTION_MEMORY_EXTRACT,
    ACTION_MEMORY_READ,
    ACTION_MEMORY_WRITE,
    MemoryAuthorizerDependencies,
)


class _StubNamespaceEntity:
    """duck-typed stand-in for :class:`NamespaceEntity`.

    carries the four attributes :func:`authorize_memory_access` reads
    (``id`` / ``namespace_type`` / ``owner_agent_id`` / ``customer_id``)
    without depending on the real :class:`BaseEntity` proxy
    machinery. the fixture constructs one per
    ``(agent_id, customer_id)`` triple; the stub's identity is the
    synthetic namespace id the permissive grant is scoped against.

    :ivar id: namespace UUID
    :ivar namespace_type: always ``"memory"`` here
    :ivar owner_agent_id: owning agent UUID
    :ivar customer_id: owning customer UUID
    """

    __slots__ = ("id", "namespace_type", "owner_agent_id", "customer_id")

    def __init__(
        self,
        *,
        id: UUID,
        namespace_type: str,
        owner_agent_id: UUID,
        customer_id: UUID,
    ) -> None:
        """initialize a stub namespace entity.

        :param id: namespace UUID
        :ptype id: UUID
        :param namespace_type: namespace type discriminator
        :ptype namespace_type: str
        :param owner_agent_id: owning agent UUID
        :ptype owner_agent_id: UUID
        :param customer_id: owning customer UUID
        :ptype customer_id: UUID
        """
        self.id = id
        self.namespace_type = namespace_type
        self.owner_agent_id = owner_agent_id
        self.customer_id = customer_id


class _PermissiveNamespaceCollection:
    """duck-typed :class:`NamespaceCollection` fake for tests.

    every :meth:`get_by_owner_and_customer` call returns the same
    synthetic namespace instance per ``(owner_agent_id, customer_id)``
    tuple so concurrent calls within one test see stable ids.
    :meth:`save_entity` is a no-op because the fixture's resolver
    path always materializes through :meth:`get_by_owner_and_customer`
    on first call.
    """

    def __init__(self) -> None:
        """initialize the fixture store."""
        self._by_triple: dict[tuple[str, UUID, UUID], _StubNamespaceEntity] = {}
        self.entity_class = _StubNamespaceEntity

    async def get_by_owner_and_customer(
        self,
        *,
        namespace_type: str,
        owner_agent_id: UUID | None,
        customer_id: UUID | None,
    ) -> _StubNamespaceEntity | None:
        """return (and cache) a synthetic namespace per triple.

        :param namespace_type: namespace type filter
        :ptype namespace_type: str
        :param owner_agent_id: owning agent UUID
        :ptype owner_agent_id: UUID | None
        :param customer_id: owning customer UUID
        :ptype customer_id: UUID | None
        :return: stub namespace entity
        :rtype: _StubNamespaceEntity | None
        """
        if owner_agent_id is None or customer_id is None:
            return None
        key = (namespace_type, owner_agent_id, customer_id)
        entity = self._by_triple.get(key)
        if entity is None:
            entity = _StubNamespaceEntity(
                id=uuid4(),
                namespace_type=namespace_type,
                owner_agent_id=owner_agent_id,
                customer_id=customer_id,
            )
            self._by_triple[key] = entity
        return entity

    async def save_entity(self, entity: Any) -> None:
        """no-op save — fixture's resolver already cached the entity.

        :param entity: entity handed to :meth:`save_entity` (unused)
        :ptype entity: Any
        :return: nothing
        :rtype: None
        """
        _ = entity
        return None

    async def get(self, entity_id: UUID) -> _StubNamespaceEntity | None:
        """return a stored stub by id, or ``None`` if not cached.

        :param entity_id: namespace UUID
        :ptype entity_id: UUID
        :return: stored stub or ``None``
        :rtype: _StubNamespaceEntity | None
        """
        for entity in self._by_triple.values():
            if entity.id == entity_id:
                return entity
        return None


class _PermissiveMembershipLoader:
    """MembershipLoader that returns a single synthetic group membership.

    the membership binds every caller (user or agent) to a common
    permissive group; combined with
    :class:`_PermissiveGrantLoader` the caller is authorized for
    every memory action on every namespace.
    """

    def __init__(self, group_id: UUID) -> None:
        """store the shared group UUID every actor is a member of.

        :param group_id: UUID of the synthetic permissive group
        :ptype group_id: UUID
        """
        self._group_id = group_id

    async def load_for_user(
        self, user_id: UUID,
    ) -> tuple[GroupMembership, ...]:
        """return one membership naming the user as member of the group.

        :param user_id: user UUID
        :ptype user_id: UUID
        :return: tuple with one :class:`GroupMembership`
        :rtype: tuple[GroupMembership, ...]
        """
        return (
            GroupMembership(
                group_id=self._group_id,
                member_type=MemberType.USER,
                member_id=user_id,
                customer_id=None,
            ),
        )

    async def load_for_agent(
        self, agent_id: UUID,
    ) -> tuple[GroupMembership, ...]:
        """return one membership naming the agent as member of the group.

        :param agent_id: agent UUID
        :ptype agent_id: UUID
        :return: tuple with one :class:`GroupMembership`
        :rtype: tuple[GroupMembership, ...]
        """
        return (
            GroupMembership(
                group_id=self._group_id,
                member_type=MemberType.AGENT,
                member_id=agent_id,
                customer_id=None,
            ),
        )


class _PermissiveGrantLoader:
    """GrantLoader that surfaces one all-actions role for every namespace.

    the single synthetic role carries every canonical memory action,
    scoped to ``ScopeType.ALL`` so the evaluator admits the
    assignment against any memory namespace.
    """

    def __init__(self, group_id: UUID, role_id: UUID) -> None:
        """store the shared group + role UUIDs reused across every load.

        :param group_id: UUID of the synthetic permissive group
        :ptype group_id: UUID
        :param role_id: UUID of the synthetic permissive role
        :ptype role_id: UUID
        """
        self._group_id = group_id
        self._role_id = role_id
        self._role = Role(
            id=role_id,
            name="PermissiveTestRole",
            permissions={
                "memory": frozenset(
                    {
                        ACTION_MEMORY_READ,
                        ACTION_MEMORY_WRITE,
                        ACTION_MEMORY_EXTRACT,
                    },
                ),
            },
            is_built_in=True,
        )

    async def load_assignments_for_groups(
        self,
        group_ids: tuple[UUID, ...],
        namespace: object,
    ) -> tuple[RoleAssignment, ...]:
        """return one all-scope assignment per known group id.

        :param group_ids: group UUIDs to inspect
        :ptype group_ids: tuple[UUID, ...]
        :param namespace: namespace under evaluation (unused)
        :ptype namespace: object
        :return: tuple of role assignments
        :rtype: tuple[RoleAssignment, ...]
        """
        assignments: list[RoleAssignment] = []
        for group_id in group_ids:
            if group_id != self._group_id:
                continue
            assignments.append(
                RoleAssignment(
                    id=uuid4(),
                    role_id=self._role_id,
                    group_id=group_id,
                    scope_type=ScopeType.ALL,
                    scope_namespace_id=None,
                    scope_namespace_type=None,
                    scope_customer_id=None,
                ),
            )
        return tuple(assignments)

    async def load_roles(
        self, role_ids: tuple[UUID, ...],
    ) -> dict[UUID, Role]:
        """return the synthetic role for every requested role id it owns.

        :param role_ids: role UUIDs to resolve
        :ptype role_ids: tuple[UUID, ...]
        :return: mapping role_id -> :class:`Role`
        :rtype: dict[UUID, Role]
        """
        return {rid: self._role for rid in role_ids if rid == self._role_id}

    async def load_groups(
        self, group_ids: tuple[UUID, ...],
    ) -> dict[UUID, object]:
        """resolve every known group id to a synthetic platform-scoped group.

        the evaluator requires every assignment's group to resolve
        through this method or the assignment is skipped (see
        :func:`threetears.agent.acl.evaluator._walk_assignments`).
        return a platform-scoped group for the shared synthetic id so
        the permissive assignment actually contributes actions.

        :param group_ids: group UUIDs to resolve
        :ptype group_ids: tuple[UUID, ...]
        :return: mapping group_id -> :class:`Group`
        :rtype: dict[UUID, object]
        """
        result: dict[UUID, object] = {}
        for gid in group_ids:
            if gid == self._group_id:
                result[gid] = Group(
                    id=gid,
                    name="PermissiveTestGroup",
                    customer_id=None,
                )
        return result


class _NoopRoleCollection:
    """duck-typed :class:`RoleCollection` returning no builtin roles.

    permissive-authorizer tests never exercise the ensure path, so
    :meth:`list_builtin` returns an empty tuple — the ensure helper
    short-circuits on the missing ``MemoryOwner`` role.
    """

    async def list_builtin(self) -> tuple[Any, ...]:
        """return an empty tuple (no builtin roles wired in tests).

        :return: empty tuple
        :rtype: tuple[Any, ...]
        """
        return ()

    async def get(self, entity_id: UUID) -> None:
        """permissive fixture never caches roles.

        :param entity_id: role UUID (unused)
        :ptype entity_id: UUID
        :return: ``None``
        :rtype: None
        """
        _ = entity_id
        return None


class _NoopGroupCollection:
    """duck-typed :class:`GroupCollection` whose every method is a no-op."""

    class _StubEntity:
        """minimal stub exposing the field attributes the ensure path reads."""

        def __init__(self, data: dict[str, Any], *, is_new: bool, collection: Any) -> None:
            """store the field data on the stub instance.

            :param data: field data dict
            :ptype data: dict[str, Any]
            :param is_new: whether entity is newly created (unused)
            :ptype is_new: bool
            :param collection: parent collection (unused)
            :ptype collection: Any
            """
            _ = is_new
            _ = collection
            self._data = dict(data)
            for key, value in data.items():
                setattr(self, key, value)

    entity_class = _StubEntity

    async def get(self, entity_id: UUID) -> Any:
        """every lookup misses so the ensure path takes the create branch.

        :param entity_id: group UUID
        :ptype entity_id: UUID
        :return: ``None``
        :rtype: Any
        """
        _ = entity_id
        return None

    async def save_entity(self, entity: Any) -> None:
        """no-op save.

        :param entity: entity (unused)
        :ptype entity: Any
        :return: nothing
        :rtype: None
        """
        _ = entity
        return None


class _NoopGroupMemberCollection:
    """duck-typed :class:`GroupMemberCollection` whose every method is a no-op."""

    class _StubEntity:
        """minimal stub exposing the field attributes the ensure path reads."""

        def __init__(self, data: dict[str, Any], *, is_new: bool, collection: Any) -> None:
            """store the field data on the stub instance.

            :param data: field data dict
            :ptype data: dict[str, Any]
            :param is_new: whether entity is newly created (unused)
            :ptype is_new: bool
            :param collection: parent collection (unused)
            :ptype collection: Any
            """
            _ = is_new
            _ = collection
            self._data = dict(data)
            for key, value in data.items():
                setattr(self, key, value)

    entity_class = _StubEntity

    async def get(self, entity_id: UUID) -> Any:
        """every lookup misses so the ensure path takes the create branch.

        :param entity_id: membership UUID
        :ptype entity_id: UUID
        :return: ``None``
        :rtype: Any
        """
        _ = entity_id
        return None

    async def save_entity(self, entity: Any) -> None:
        """no-op save.

        :param entity: entity (unused)
        :ptype entity: Any
        :return: nothing
        :rtype: None
        """
        _ = entity
        return None


class _NoopRoleAssignmentCollection:
    """duck-typed :class:`RoleAssignmentCollection` whose ensure is a no-op."""

    async def ensure_group_role_assignment(
        self,
        *,
        group_id: UUID,
        role_id: UUID,
        scope_type: str,
        scope_id: UUID | None,
    ) -> UUID:
        """return a synthetic assignment id without persisting anything.

        :param group_id: group UUID (unused)
        :ptype group_id: UUID
        :param role_id: role UUID (unused)
        :ptype role_id: UUID
        :param scope_type: scope discriminator (unused)
        :ptype scope_type: str
        :param scope_id: scope UUID (unused)
        :ptype scope_id: UUID | None
        :return: synthetic assignment UUID
        :rtype: UUID
        """
        _ = group_id, role_id, scope_type, scope_id
        return uuid4()


@pytest.fixture
def permissive_memory_authorizer() -> MemoryAuthorizerDependencies:
    """TEST-ONLY permissive :class:`MemoryAuthorizerDependencies` bundle.

    every evaluator call allows; every
    :meth:`NamespaceCollection.get_by_owner_and_customer` call
    returns a synthetic stub namespace; every ensure-path Collection
    method is a no-op. intended for unit tests that focus on memory
    plumbing (SQL shape, embeddings, dedup, ...) and not on rbac
    behaviour. tests that exercise rbac construct their own bundle
    with production loaders + Collections.

    three-tier-task-01 phase D retired the bespoke resolver /
    ensurer callables the fixture previously supplied; the returned
    bundle carries permissive Collection stand-ins whose method
    surface matches what the production Collections expose.

    :return: permissive authorizer dependency bundle
    :rtype: MemoryAuthorizerDependencies
    """
    from threetears.agent.acl import AclCache
    shared_group_id = uuid4()
    shared_role_id = uuid4()
    _ = datetime.now(UTC)  # touch the import so linters don't flag it
    membership_loader = _PermissiveMembershipLoader(shared_group_id)
    grant_loader = _PermissiveGrantLoader(shared_group_id, shared_role_id)
    return MemoryAuthorizerDependencies(
        acl_cache=AclCache(
            membership_loader=membership_loader,
            grant_loader=grant_loader,
        ),
        membership_loader=membership_loader,
        grant_loader=grant_loader,
        namespace_collection=_PermissiveNamespaceCollection(),
        group_collection=_NoopGroupCollection(),
        group_member_collection=_NoopGroupMemberCollection(),
        role_collection=_NoopRoleCollection(),
        role_assignment_collection=_NoopRoleAssignmentCollection(),
    )
