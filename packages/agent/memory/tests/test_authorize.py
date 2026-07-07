"""tests for the memory authorize helper (namespace-task-01 phase 3).

three-tier-task-01 phase D retired the bespoke resolver / ensurer
callables and the parallel namespace-row value object. tests build
the authorizer bundle directly from in-memory Collection stand-ins
and the real ACL loaders; the bundle's
``namespace_collection.get_by_owner_and_customer(...)`` +
``save_entity(...)`` methods are the create-if-absent path
:func:`authorize_memory_access` consumes.
"""

from __future__ import annotations

from typing import Any
from uuid import NAMESPACE_DNS, UUID, uuid4, uuid5

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
    MEMORY_OWNER_ROLE_NAME,
    MemoryAccessDenied,
    MemoryAuthorizerDependencies,
    authorize_memory_access,
    memory_namespace_name,
)


class _StubNamespaceEntity:
    """duck-typed :class:`NamespaceEntity` with the four fields the evaluator reads.

    supports two construction shapes:

    1. kwarg-only: :class:`_StubNamespaceEntity(id=..., ...)` — used
       by tests building fixtures directly.
    2. the production ``entity_class(data, is_new=..., collection=...)``
       shape :func:`_resolve_or_create_memory_namespace` invokes after
       a miss.

    stores only the four fields the evaluator reads; any extra
    ``data`` keys passed through the positional shape are retained on
    ``self._data`` for debugging but not surfaced as attributes
    beyond the canonical four.
    """

    def __init__(
        self,
        data: dict[str, Any] | None = None,
        *,
        id: UUID | None = None,
        namespace_type: str | None = None,
        owner_agent_id: UUID | None = None,
        customer_id: UUID | None = None,
        is_new: bool = False,
        collection: Any = None,
    ) -> None:
        """initialize a stub namespace entity.

        :param data: field data dict (production construction shape)
        :ptype data: dict[str, Any] | None
        :param id: namespace UUID
        :ptype id: UUID | None
        :param namespace_type: namespace type discriminator
        :ptype namespace_type: str | None
        :param owner_agent_id: owning agent UUID
        :ptype owner_agent_id: UUID | None
        :param customer_id: owning customer UUID
        :ptype customer_id: UUID | None
        :param is_new: whether entity is newly created (unused)
        :ptype is_new: bool
        :param collection: parent collection (unused)
        :ptype collection: Any
        """
        _ = is_new, collection
        self._data = dict(data) if data else {}
        if data is not None:
            # v0.8.0 shard 04.6: namespaces PK renamed to namespace_id;
            # production code seeds the dict with that key.
            self.id = data["namespace_id"]
            self.namespace_type = data["namespace_type"]
            self.owner_agent_id = data["owner_agent_id"]
            self.customer_id = data["customer_id"]
        else:
            assert id is not None
            assert namespace_type is not None
            assert owner_agent_id is not None
            assert customer_id is not None
            self.id = id
            self.namespace_type = namespace_type
            self.owner_agent_id = owner_agent_id
            self.customer_id = customer_id


class _NamespaceCollectionFake:
    """duck-typed :class:`NamespaceCollection` keyed on the memory triple.

    :ivar resolved_entity: public setter for the entity the fake
        returns from :meth:`get_by_owner_and_customer`; tests flip
        this after :meth:`save_entity` to simulate the Collection
        re-read returning the freshly-saved row
    :ivar save_calls: list of entities passed through :meth:`save_entity`
    """

    def __init__(self, entity: _StubNamespaceEntity | None) -> None:
        """store a preconfigured stub (or ``None`` to exercise the miss path).

        :param entity: stub namespace entity or ``None``
        :ptype entity: _StubNamespaceEntity | None
        """
        self.resolved_entity = entity
        self.save_calls: list[Any] = []
        self.entity_class = _StubNamespaceEntity

    async def get_by_owner_and_customer(
        self,
        *,
        namespace_type: str,
        owner_agent_id: UUID | None,
        customer_id: UUID | None,
    ) -> _StubNamespaceEntity | None:
        """return the stored stub (may be ``None``).

        :param namespace_type: namespace type (unused)
        :ptype namespace_type: str
        :param owner_agent_id: owning agent UUID (unused)
        :ptype owner_agent_id: UUID | None
        :param customer_id: owning customer UUID (unused)
        :ptype customer_id: UUID | None
        :return: preconfigured stub or ``None``
        :rtype: _StubNamespaceEntity | None
        """
        _ = namespace_type, owner_agent_id, customer_id
        return self.resolved_entity

    async def save_entity(self, entity: Any) -> None:
        """record the save call; tests assert against :attr:`save_calls`.

        :param entity: entity passed in
        :ptype entity: Any
        :return: nothing
        :rtype: None
        """
        self.save_calls.append(entity)
        return None


class _NamespaceCollectionRaisingFake(_NamespaceCollectionFake):
    """variant of :class:`_NamespaceCollectionFake` whose ``save_entity`` raises.

    used to exercise the create-failed denial path in
    :func:`authorize_memory_access`.
    """

    async def save_entity(self, entity: Any) -> None:
        """raise to simulate a Collection save failure.

        :param entity: entity passed in (unused)
        :ptype entity: Any
        :return: never returns
        :rtype: None
        :raises RuntimeError: always
        """
        _ = entity
        raise RuntimeError("simulated save failure")


class _NamespaceCollectionUnavailableFake:
    """namespace collection that raises on every access.

    simulates the agent's sandboxed L3 backend, whose search_path is the
    per-agent schema (``agent_<hex>``) with no ``namespaces`` table -- any
    read or create there fails ``relation "namespaces" does not exist``.
    the owner path must never touch it.
    """

    def __init__(self) -> None:
        """initialize with the stub entity_class + empty save log.

        :return: nothing
        :rtype: None
        """
        self.entity_class = _StubNamespaceEntity
        self.save_calls: list[Any] = []

    async def get_by_owner_and_customer(
        self,
        *,
        namespace_type: str,
        owner_agent_id: UUID | None,
        customer_id: UUID | None,
    ) -> _StubNamespaceEntity | None:
        """raise as the sandboxed agent L3 would on a namespaces read.

        :param namespace_type: namespace type (unused)
        :ptype namespace_type: str
        :param owner_agent_id: owning agent UUID (unused)
        :ptype owner_agent_id: UUID | None
        :param customer_id: owning customer UUID (unused)
        :ptype customer_id: UUID | None
        :return: never returns
        :rtype: _StubNamespaceEntity | None
        :raises RuntimeError: always
        """
        _ = namespace_type, owner_agent_id, customer_id
        raise RuntimeError('relation "namespaces" does not exist')

    async def save_entity(self, entity: Any) -> None:
        """raise as the sandboxed agent L3 would on a namespaces write.

        :param entity: entity passed in (unused)
        :ptype entity: Any
        :return: never returns
        :rtype: None
        :raises RuntimeError: always
        """
        _ = entity
        raise RuntimeError('relation "namespaces" does not exist')


def _stub_ns(
    *,
    agent_id: UUID,
    customer_id: UUID,
) -> _StubNamespaceEntity:
    return _StubNamespaceEntity(
        id=uuid4(),
        namespace_type="memory",
        owner_agent_id=agent_id,
        customer_id=customer_id,
    )


def _owner_role() -> Role:
    return Role(
        id=uuid4(),
        name=MEMORY_OWNER_ROLE_NAME,
        permissions={
            "memory": frozenset(
                {ACTION_MEMORY_READ, ACTION_MEMORY_WRITE, ACTION_MEMORY_EXTRACT},
            ),
        },
        is_built_in=True,
    )


def _reader_role() -> Role:
    return Role(
        id=uuid4(),
        name="MemoryReader",
        permissions={"memory": frozenset({ACTION_MEMORY_READ})},
        is_built_in=True,
    )


def _build_deps(
    *,
    namespace_collection: Any,
    memberships_for_user: tuple[GroupMembership, ...] = (),
    memberships_for_agent: tuple[GroupMembership, ...] = (),
    assignments: tuple[RoleAssignment, ...] = (),
    roles: dict[UUID, Role] | None = None,
    groups: dict[UUID, Group] | None = None,
) -> MemoryAuthorizerDependencies:
    """build a :class:`MemoryAuthorizerDependencies` bundle with ACL mocks.

    :param namespace_collection: fake namespace collection
    :ptype namespace_collection: Any
    :param memberships_for_user: user-side memberships (keyed by caller)
    :ptype memberships_for_user: tuple[GroupMembership, ...]
    :param memberships_for_agent: agent-side memberships
    :ptype memberships_for_agent: tuple[GroupMembership, ...]
    :param assignments: assignments returned by loader
    :ptype assignments: tuple[RoleAssignment, ...]
    :param roles: role fixture keyed on role id
    :ptype roles: dict[UUID, Role] | None
    :param groups: group fixture keyed on group id
    :ptype groups: dict[UUID, Group] | None
    :return: populated bundle
    :rtype: MemoryAuthorizerDependencies
    """

    class _MembershipLoader:
        """in-memory membership loader."""

        async def load_for_user(
            self,
            user_id: UUID,
        ) -> tuple[GroupMembership, ...]:
            """return configured user memberships.

            :param user_id: user UUID (unused)
            :ptype user_id: UUID
            :return: configured tuple
            :rtype: tuple[GroupMembership, ...]
            """
            _ = user_id
            return memberships_for_user

        async def load_for_agent(
            self,
            agent_id: UUID,
        ) -> tuple[GroupMembership, ...]:
            """return configured agent memberships.

            :param agent_id: agent UUID (unused)
            :ptype agent_id: UUID
            :return: configured tuple
            :rtype: tuple[GroupMembership, ...]
            """
            _ = agent_id
            return memberships_for_agent

    class _GrantLoader:
        """in-memory grant loader keyed on group UUID."""

        async def load_assignments_for_groups(
            self,
            group_ids: tuple[UUID, ...],
            namespace: Any,
        ) -> tuple[RoleAssignment, ...]:
            """return all configured assignments (evaluator filters by coverage).

            :param group_ids: candidate group UUIDs (unused)
            :ptype group_ids: tuple[UUID, ...]
            :param namespace: namespace under evaluation (unused)
            :ptype namespace: Any
            :return: assignments
            :rtype: tuple[RoleAssignment, ...]
            """
            _ = group_ids, namespace
            return assignments

        async def load_roles(
            self,
            role_ids: tuple[UUID, ...],
        ) -> dict[UUID, Role]:
            """return role subset.

            :param role_ids: requested role UUIDs
            :ptype role_ids: tuple[UUID, ...]
            :return: role mapping subset
            :rtype: dict[UUID, Role]
            """
            return {rid: (roles or {})[rid] for rid in role_ids if rid in (roles or {})}

        async def load_groups(
            self,
            group_ids: tuple[UUID, ...],
        ) -> dict[UUID, Any]:
            """return group subset.

            :param group_ids: requested group UUIDs
            :ptype group_ids: tuple[UUID, ...]
            :return: group mapping subset
            :rtype: dict[UUID, Any]
            """
            return {gid: (groups or {})[gid] for gid in group_ids if gid in (groups or {})}

    from threetears.agent.acl import AclCache

    membership_loader = _MembershipLoader()
    grant_loader = _GrantLoader()
    return MemoryAuthorizerDependencies(
        acl_cache=AclCache(
            membership_loader=membership_loader,
            grant_loader=grant_loader,
        ),
        namespace_collection=namespace_collection,
        group_collection=object(),
        group_member_collection=object(),
        role_collection=object(),
        role_assignment_collection=object(),
    )


class TestMemoryNamespaceName:
    def test_shape(self) -> None:
        agent_id = UUID("019470a8-b5c3-7def-8123-456789abcdef")
        customer_id = UUID("11112222-3333-4444-5555-666677778888")
        name = memory_namespace_name(agent_id, customer_id)
        assert name == "memories.019470a8.11112222"


class TestAuthorizeMemoryAccess:
    async def test_namespace_materialized_on_miss_user_path(self) -> None:
        """user (non-owner) path creates the missing namespace before evaluating.

        materialization runs in ``_resolve_or_create_memory_namespace``
        before ``authorize_on_entity``; with no grant the call then denies,
        but ``save_entity`` was invoked. the owner path (deterministic, no
        Collection) is covered by :meth:`test_owner_shortcut_allows_agent_without_grant`.
        """
        agent_id = uuid4()
        customer_id = uuid4()
        ns_created = _stub_ns(agent_id=agent_id, customer_id=customer_id)
        # start with no namespace; after save_entity we flip the fake to
        # return the resolved stub so the re-read returns cleanly.
        namespace_collection = _NamespaceCollectionFake(None)

        async def _save(entity: Any) -> None:
            """flip the fake's stored entity after save and record the call.

            :param entity: entity being saved
            :ptype entity: Any
            :return: nothing
            :rtype: None
            """
            namespace_collection.resolved_entity = ns_created
            namespace_collection.save_calls.append(entity)

        namespace_collection.save_entity = _save  # type: ignore[assignment]

        deps = _build_deps(namespace_collection=namespace_collection)

        # non-owner caller (caller_agent_id None) with no grant: the row is
        # materialized, then the evaluator denies.
        with pytest.raises(MemoryAccessDenied, match="evaluator denied"):
            await authorize_memory_access(
                action=ACTION_MEMORY_READ,
                agent_id=agent_id,
                customer_id=customer_id,
                caller_user_id=uuid4(),
                caller_agent_id=None,
                deps=deps,
            )
        assert len(namespace_collection.save_calls) == 1

    async def test_namespace_create_failure_denies(self) -> None:
        """when ``save_entity`` raises, the helper denies cleanly."""
        agent_id = uuid4()
        customer_id = uuid4()
        namespace_collection = _NamespaceCollectionRaisingFake(None)
        deps = _build_deps(namespace_collection=namespace_collection)
        with pytest.raises(MemoryAccessDenied, match="could not be created"):
            await authorize_memory_access(
                action=ACTION_MEMORY_READ,
                agent_id=agent_id,
                customer_id=customer_id,
                caller_user_id=uuid4(),
                caller_agent_id=None,
                deps=deps,
            )

    async def test_owner_shortcut_allows_agent_without_grant(self) -> None:
        """owner path allows without a grant AND never touches the collection.

        reproduces the agent-internal retrieval/extraction fix: the owning
        agent resolves its memory namespace deterministically, so a
        collection whose every access raises ``relation "namespaces" does
        not exist`` (the sandboxed agent L3) does not break authorization.
        """
        agent_id = uuid4()
        customer_id = uuid4()
        namespace_collection = _NamespaceCollectionUnavailableFake()
        deps = _build_deps(namespace_collection=namespace_collection)

        result = await authorize_memory_access(
            action=ACTION_MEMORY_WRITE,
            agent_id=agent_id,
            customer_id=customer_id,
            caller_user_id=None,
            caller_agent_id=agent_id,
            deps=deps,
        )
        assert result.owner_agent_id == agent_id
        assert result.customer_id == customer_id
        assert result.namespace_type == "memory"
        assert result.id == uuid5(
            NAMESPACE_DNS,
            f"threetears.namespaces.memory.{agent_id.hex}.{customer_id.hex}",
        )
        assert namespace_collection.save_calls == []

    async def test_owner_branch_bypasses_collection_even_with_user_present(self) -> None:
        """retrieval shape (user + owning agent) skips the collection read.

        ``retrieve_memories`` / extraction pass ``caller_user_id=user`` AND
        ``caller_agent_id=agent`` (the owning agent). the owner BRANCH fires
        on ``caller_agent_id == agent_id`` regardless of the user, so the
        deterministic descriptor is used and the sandboxed-L3 collection --
        which raises ``relation "namespaces" does not exist`` on any access
        -- is never touched. the call still DENIES here (user ∩ agent
        intersection: an ungranted user caps the owner-implicit wildcard to
        empty), surfacing as a clean ``evaluator denied`` rather than the
        namespaces DB error -- so the guard is that the raised type is
        :class:`MemoryAccessDenied` from the evaluator, NOT a RuntimeError
        from a collection read.
        """
        agent_id = uuid4()
        customer_id = uuid4()
        namespace_collection = _NamespaceCollectionUnavailableFake()
        deps = _build_deps(namespace_collection=namespace_collection)

        with pytest.raises(MemoryAccessDenied, match="evaluator denied"):
            await authorize_memory_access(
                action=ACTION_MEMORY_READ,
                agent_id=agent_id,
                customer_id=customer_id,
                caller_user_id=uuid4(),
                caller_agent_id=agent_id,
                deps=deps,
            )
        # the collection was never consulted (no relation error): the owner
        # branch resolved the namespace deterministically.
        assert namespace_collection.save_calls == []

    async def test_user_without_grant_denied(self) -> None:
        agent_id = uuid4()
        customer_id = uuid4()
        ns = _stub_ns(agent_id=agent_id, customer_id=customer_id)
        deps = _build_deps(namespace_collection=_NamespaceCollectionFake(ns))
        with pytest.raises(MemoryAccessDenied, match="evaluator denied"):
            await authorize_memory_access(
                action=ACTION_MEMORY_READ,
                agent_id=agent_id,
                customer_id=customer_id,
                caller_user_id=uuid4(),
                caller_agent_id=None,
                deps=deps,
            )

    async def test_user_with_reader_grant_allowed(self) -> None:
        agent_id = uuid4()
        customer_id = uuid4()
        user_id = uuid4()
        group_id = uuid4()
        role = _reader_role()
        ns = _stub_ns(agent_id=agent_id, customer_id=customer_id)
        assignment = RoleAssignment(
            id=uuid4(),
            role_id=role.id,
            group_id=group_id,
            scope_type=ScopeType.NAMESPACE,
            scope_namespace_id=ns.id,
            scope_namespace_type=None,
            scope_customer_id=None,
        )
        membership = GroupMembership(
            group_id=group_id,
            member_type=MemberType.USER,
            member_id=user_id,
            customer_id=customer_id,
        )
        group = Group(id=group_id, name="memory-owner:x", customer_id=customer_id)
        deps = _build_deps(
            namespace_collection=_NamespaceCollectionFake(ns),
            memberships_for_user=(membership,),
            assignments=(assignment,),
            roles={role.id: role},
            groups={group_id: group},
        )
        result = await authorize_memory_access(
            action=ACTION_MEMORY_READ,
            agent_id=agent_id,
            customer_id=customer_id,
            caller_user_id=user_id,
            caller_agent_id=None,
            deps=deps,
        )
        assert result is ns

    async def test_user_with_reader_grant_cannot_write(self) -> None:
        agent_id = uuid4()
        customer_id = uuid4()
        user_id = uuid4()
        group_id = uuid4()
        role = _reader_role()
        ns = _stub_ns(agent_id=agent_id, customer_id=customer_id)
        assignment = RoleAssignment(
            id=uuid4(),
            role_id=role.id,
            group_id=group_id,
            scope_type=ScopeType.NAMESPACE,
            scope_namespace_id=ns.id,
            scope_namespace_type=None,
            scope_customer_id=None,
        )
        membership = GroupMembership(
            group_id=group_id,
            member_type=MemberType.USER,
            member_id=user_id,
            customer_id=customer_id,
        )
        group = Group(id=group_id, name="x", customer_id=customer_id)
        deps = _build_deps(
            namespace_collection=_NamespaceCollectionFake(ns),
            memberships_for_user=(membership,),
            assignments=(assignment,),
            roles={role.id: role},
            groups={group_id: group},
        )
        with pytest.raises(MemoryAccessDenied):
            await authorize_memory_access(
                action=ACTION_MEMORY_WRITE,
                agent_id=agent_id,
                customer_id=customer_id,
                caller_user_id=user_id,
                caller_agent_id=None,
                deps=deps,
            )

    async def test_owner_role_grants_extract(self) -> None:
        agent_id = uuid4()
        customer_id = uuid4()
        user_id = uuid4()
        group_id = uuid4()
        role = _owner_role()
        ns = _stub_ns(agent_id=agent_id, customer_id=customer_id)
        assignment = RoleAssignment(
            id=uuid4(),
            role_id=role.id,
            group_id=group_id,
            scope_type=ScopeType.NAMESPACE,
            scope_namespace_id=ns.id,
            scope_namespace_type=None,
            scope_customer_id=None,
        )
        membership = GroupMembership(
            group_id=group_id,
            member_type=MemberType.USER,
            member_id=user_id,
            customer_id=customer_id,
        )
        group = Group(id=group_id, name="x", customer_id=customer_id)
        deps = _build_deps(
            namespace_collection=_NamespaceCollectionFake(ns),
            memberships_for_user=(membership,),
            assignments=(assignment,),
            roles={role.id: role},
            groups={group_id: group},
        )
        result = await authorize_memory_access(
            action=ACTION_MEMORY_EXTRACT,
            agent_id=agent_id,
            customer_id=customer_id,
            caller_user_id=user_id,
            caller_agent_id=None,
            deps=deps,
        )
        assert result is ns
