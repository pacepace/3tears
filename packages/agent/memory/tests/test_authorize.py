"""tests for the memory authorize helper (namespace-task-01 phase 3).

three-tier-task-01 phase D retired the bespoke resolver / ensurer
callables and the parallel namespace-row value object. tests build
the authorizer bundle directly from in-memory Collection stand-ins
and the real ACL loaders.

the create half then left this process entirely: the non-owner path
READS through ``namespace_collection.get_by_owner_and_customer(...)``
and, on a miss, asks the HUB through the bundle's
``namespace_provisioner``. every test that used to assert on
``save_entity`` now asserts the opposite -- that nothing in this
package writes ``namespaces`` -- and the provisioner fake records what
was asked for so a refusal can be told from a request never made.
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
    memory_namespace_id,
    memory_namespace_name,
    memory_namespace_schema_name,
)
from threetears.agent.memory.namespace_client import (
    MemoryNamespaceRef,
    MemoryNamespaceUnavailableError,
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
        name: str | None = None,
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
        :param name: the name the row STORES, which is what a subtree grant is judged against
        :ptype name: str | None
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
            self.name = data.get("name")
        else:
            assert id is not None
            assert namespace_type is not None
            assert owner_agent_id is not None
            assert customer_id is not None
            self.id = id
            self.namespace_type = namespace_type
            self.owner_agent_id = owner_agent_id
            self.customer_id = customer_id
            self.name = name


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


class _ProvisionerFake:
    """hub-backed provisioner stand-in recording every ensure it was asked for.

    :ivar calls: ``(agent_id, customer_id)`` per :meth:`ensure` call, in order
    :ivar ref: reference returned on success
    :ivar failure: when set, raised instead of returning
    """

    def __init__(
        self,
        *,
        ref: MemoryNamespaceRef | None = None,
        failure: Exception | None = None,
    ) -> None:
        """store the configured outcome.

        :param ref: reference to return from :meth:`ensure`
        :ptype ref: MemoryNamespaceRef | None
        :param failure: exception to raise instead of returning
        :ptype failure: Exception | None
        :return: nothing
        :rtype: None
        """
        self.calls: list[tuple[UUID, UUID]] = []
        self.ref = ref
        self.failure = failure

    async def ensure(self, *, agent_id: UUID, customer_id: UUID) -> MemoryNamespaceRef:
        """record the request and return (or raise) the configured outcome.

        :param agent_id: owning agent UUID
        :ptype agent_id: UUID
        :param customer_id: owning customer UUID
        :ptype customer_id: UUID
        :return: configured reference
        :rtype: MemoryNamespaceRef
        :raises Exception: the configured failure, when one is set
        """
        self.calls.append((agent_id, customer_id))
        if self.failure is not None:
            raise self.failure
        assert self.ref is not None
        return self.ref


def _hub_ref(*, agent_id: UUID, customer_id: UUID) -> MemoryNamespaceRef:
    """build the reference a hub ensure would hand back.

    :param agent_id: owning agent UUID
    :ptype agent_id: UUID
    :param customer_id: owning customer UUID
    :ptype customer_id: UUID
    :return: namespace reference
    :rtype: MemoryNamespaceRef
    """
    return MemoryNamespaceRef(
        id=uuid4(),
        customer_id=customer_id,
        owner_agent_id=agent_id,
        namespace_type="memory",
        owner_namespace=f"agents.{agent_id}",
    )


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
    namespace_provisioner: Any = None,
    memberships_for_user: tuple[GroupMembership, ...] = (),
    memberships_for_agent: tuple[GroupMembership, ...] = (),
    assignments: tuple[RoleAssignment, ...] = (),
    roles: dict[UUID, Role] | None = None,
    groups: dict[UUID, Group] | None = None,
) -> MemoryAuthorizerDependencies:
    """build a :class:`MemoryAuthorizerDependencies` bundle with ACL mocks.

    :param namespace_collection: fake namespace collection
    :ptype namespace_collection: Any
    :param namespace_provisioner: fake hub-backed provisioner, or ``None`` to
        build a bundle that has nobody to ask
    :ptype namespace_provisioner: Any
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

        async def load_for_group(self, group_id: UUID) -> tuple[GroupMembership, ...]:
            """return parent-group memberships -- none; flat groups here.

            :param group_id: child group UUID
            :ptype group_id: UUID
            :return: empty tuple
            :rtype: tuple[GroupMembership, ...]
            """
            return ()

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
        namespace_provisioner=namespace_provisioner,
        group_collection=object(),
        group_member_collection=object(),
        role_collection=object(),
        role_assignment_collection=object(),
    )


#: two agent ids a uuid7 generator could mint in ONE millisecond: the leading 48 bits (the
#: timestamp) are identical, only the random tail differs. this is what a cluster apply
#: creating several agents at once produces.
_AGENT_A = UUID("019470a8-b5c3-7def-8123-456789abcdef")
_AGENT_B = UUID("019470a8-b5c3-7a01-9fed-cba987654321")
_CUSTOMER = UUID("019470a8-b5c4-7000-8000-000000000001")


class TestMemoryNamespaceName:
    def test_shape(self) -> None:
        """the name spells both ids in full; the schema name is the row id's hex."""
        agent_id = UUID("019470a8-b5c3-7def-8123-456789abcdef")
        customer_id = UUID("11112222-3333-4444-5555-666677778888")
        assert memory_namespace_name(agent_id, customer_id) == (
            "memories.019470a8b5c37def8123456789abcdef.11112222333344445555666677778888"
        )
        assert memory_namespace_schema_name(agent_id, customer_id) == (
            f"memory__{memory_namespace_id(agent_id, customer_id).hex}"
        )

    def test_two_agents_minted_in_one_millisecond_get_distinct_names(self) -> None:
        """``platform.namespaces`` is UNIQUE on name and on schema_name, so a shared value is a failed create.

        a uuid7 leads with its timestamp; a name cut from that head is the same for every
        agent minted in the same moment, and the second agent's memory namespace could never
        be created.
        """
        assert _AGENT_A.int >> 80 == _AGENT_B.int >> 80, "the fixture must share the uuid7 timestamp"
        assert _AGENT_A != _AGENT_B
        assert memory_namespace_name(_AGENT_A, _CUSTOMER) != memory_namespace_name(_AGENT_B, _CUSTOMER)
        assert memory_namespace_schema_name(_AGENT_A, _CUSTOMER) != memory_namespace_schema_name(_AGENT_B, _CUSTOMER)

    def test_one_agent_in_two_customers_minted_together_gets_distinct_names(self) -> None:
        """the customer side of the pair is a uuid7 too, and collides the same way."""
        other_customer = UUID("019470a8-b5c4-7fff-bfff-fffffffffffe")
        assert _CUSTOMER.int >> 80 == other_customer.int >> 80
        assert memory_namespace_name(_AGENT_A, _CUSTOMER) != memory_namespace_name(_AGENT_A, other_customer)
        assert memory_namespace_schema_name(_AGENT_A, _CUSTOMER) != memory_namespace_schema_name(
            _AGENT_A, other_customer
        )

    def test_both_fit_the_columns_and_the_schema_name_fits_an_identifier(self) -> None:
        """``name`` is varchar(255), ``schema_name`` varchar(100) and named like a Postgres schema (63)."""
        assert len(memory_namespace_name(_AGENT_A, _CUSTOMER)) <= 255
        assert len(memory_namespace_schema_name(_AGENT_A, _CUSTOMER)) <= 63


class TestAuthorizeMemoryAccess:
    async def test_missing_namespace_is_asked_of_the_hub_never_written_here(self) -> None:
        """the non-owner path materializes through the PROVISIONER, not a save.

        this is the whole point of the change: an agent process that could
        insert into ``platform.namespaces`` chooses what the control plane says
        about itself. with no grant the call still denies afterwards, but the
        namespace was resolved by asking the hub and the collection saw no
        write at all.
        """
        agent_id = uuid4()
        customer_id = uuid4()
        namespace_collection = _NamespaceCollectionFake(None)
        provisioner = _ProvisionerFake(ref=_hub_ref(agent_id=agent_id, customer_id=customer_id))
        deps = _build_deps(
            namespace_collection=namespace_collection,
            namespace_provisioner=provisioner,
        )

        with pytest.raises(MemoryAccessDenied, match="evaluator denied"):
            await authorize_memory_access(
                action=ACTION_MEMORY_READ,
                agent_id=agent_id,
                customer_id=customer_id,
                caller_user_id=uuid4(),
                caller_agent_id=None,
                deps=deps,
            )
        assert provisioner.calls == [(agent_id, customer_id)]
        assert namespace_collection.save_calls == []

    async def test_granted_user_is_allowed_on_the_hub_provisioned_namespace(self) -> None:
        """the admitted twin: a granted user passes, against the hub's own ref.

        without this a helper that denied every non-owner call would satisfy
        every other case in this class. it also pins that the reference the hub
        returned is what the evaluator judged and what the caller gets back.
        """
        agent_id = uuid4()
        customer_id = uuid4()
        user_id = uuid4()
        group_id = uuid4()
        role = _reader_role()
        ref = _hub_ref(agent_id=agent_id, customer_id=customer_id)
        assignment = RoleAssignment(
            id=uuid4(),
            group_id=group_id,
            role_id=role.id,
            scope_type=ScopeType.NAMESPACE,
            scope_namespace_id=ref.id,
            scope_namespace_type=None,
            scope_customer_id=None,
        )
        namespace_collection = _NamespaceCollectionFake(None)
        provisioner = _ProvisionerFake(ref=ref)
        deps = _build_deps(
            namespace_collection=namespace_collection,
            namespace_provisioner=provisioner,
            memberships_for_user=(
                GroupMembership(
                    group_id=group_id,
                    member_type=MemberType.USER,
                    member_id=user_id,
                    customer_id=customer_id,
                ),
            ),
            assignments=(assignment,),
            roles={role.id: role},
            groups={group_id: Group(id=group_id, name="readers", customer_id=customer_id)},
        )

        result = await authorize_memory_access(
            action=ACTION_MEMORY_READ,
            agent_id=agent_id,
            customer_id=customer_id,
            caller_user_id=user_id,
            caller_agent_id=None,
            deps=deps,
        )
        assert result is ref
        assert provisioner.calls == [(agent_id, customer_id)]
        assert namespace_collection.save_calls == []

    async def test_existing_row_is_not_re_asked_of_the_hub(self) -> None:
        """a resolved row short-circuits: the hub is asked only on a MISS."""
        agent_id = uuid4()
        customer_id = uuid4()
        ns = _stub_ns(agent_id=agent_id, customer_id=customer_id)
        namespace_collection = _NamespaceCollectionFake(ns)
        provisioner = _ProvisionerFake(ref=_hub_ref(agent_id=agent_id, customer_id=customer_id))
        deps = _build_deps(
            namespace_collection=namespace_collection,
            namespace_provisioner=provisioner,
        )

        with pytest.raises(MemoryAccessDenied, match="evaluator denied"):
            await authorize_memory_access(
                action=ACTION_MEMORY_READ,
                agent_id=agent_id,
                customer_id=customer_id,
                caller_user_id=uuid4(),
                caller_agent_id=None,
                deps=deps,
            )
        assert provisioner.calls == []

    async def test_namespace_create_failure_denies(self) -> None:
        """a hub that cannot answer denies cleanly rather than proceeding."""
        agent_id = uuid4()
        customer_id = uuid4()
        namespace_collection = _NamespaceCollectionFake(None)
        provisioner = _ProvisionerFake(
            failure=MemoryNamespaceUnavailableError("hub refused: IDENTITY_UNVERIFIED"),
        )
        deps = _build_deps(
            namespace_collection=namespace_collection,
            namespace_provisioner=provisioner,
        )
        with pytest.raises(MemoryAccessDenied, match="could not be created"):
            await authorize_memory_access(
                action=ACTION_MEMORY_READ,
                agent_id=agent_id,
                customer_id=customer_id,
                caller_user_id=uuid4(),
                caller_agent_id=None,
                deps=deps,
            )
        assert namespace_collection.save_calls == []

    async def test_missing_provisioner_denies_and_writes_nothing(self) -> None:
        """with nobody to ask, an absent namespace is a DENIAL, not a local write.

        the failure mode this rules out is the tempting one: falling back to
        building the row in-process, which is exactly the writer being retired.
        """
        agent_id = uuid4()
        customer_id = uuid4()
        namespace_collection = _NamespaceCollectionFake(None)
        deps = _build_deps(namespace_collection=namespace_collection, namespace_provisioner=None)

        with pytest.raises(MemoryAccessDenied, match="no namespace provisioner is wired"):
            await authorize_memory_access(
                action=ACTION_MEMORY_READ,
                agent_id=agent_id,
                customer_id=customer_id,
                caller_user_id=uuid4(),
                caller_agent_id=None,
                deps=deps,
            )
        assert namespace_collection.save_calls == []

    async def test_owner_path_never_asks_the_hub(self) -> None:
        """the owner short-circuit resolves in-process and sends no request.

        the owner path was already collection-free; it must also stay
        transport-free, or every agent-internal extraction would take a NATS
        round trip it does not need.
        """
        agent_id = uuid4()
        customer_id = uuid4()
        provisioner = _ProvisionerFake(ref=_hub_ref(agent_id=agent_id, customer_id=customer_id))
        deps = _build_deps(
            namespace_collection=_NamespaceCollectionUnavailableFake(),
            namespace_provisioner=provisioner,
        )

        await authorize_memory_access(
            action=ACTION_MEMORY_WRITE,
            agent_id=agent_id,
            customer_id=customer_id,
            caller_user_id=None,
            caller_agent_id=agent_id,
            deps=deps,
        )
        assert provisioner.calls == []

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


def _subtree_reader(*, root: str, user_id: UUID, customer_id: UUID) -> dict[str, Any]:
    """the acl fixture for one user holding a reader grant on the subtree rooted at ``root``.

    :param root: subtree root, a namespace name
    :ptype root: str
    :param user_id: the user holding the grant
    :ptype user_id: UUID
    :param customer_id: the user's customer
    :ptype customer_id: UUID
    :return: keyword arguments for :func:`_build_deps`
    :rtype: dict[str, Any]
    """
    group_id = uuid4()
    role = _reader_role()
    return {
        "memberships_for_user": (
            GroupMembership(group_id=group_id, member_type=MemberType.USER, member_id=user_id, customer_id=customer_id),
        ),
        "assignments": (
            RoleAssignment(
                id=uuid4(),
                role_id=role.id,
                group_id=group_id,
                scope_type=ScopeType.SUBTREE,
                scope_namespace_id=None,
                scope_namespace_type=None,
                scope_customer_id=None,
                scope_namespace_name=root,
            ),
        ),
        "roles": {role.id: role},
        "groups": {group_id: Group(id=group_id, name="subtree-readers", customer_id=customer_id)},
    }


class TestAnExistingRowIsJudgedByTheNameItStores:
    """rows written before the name rule changed keep their name, and are judged by it.

    the name is a unique column, not a key anything recomputes to find a row: the row is
    resolved by ``(type, owner agent, customer)`` and its grants address it by id. the
    one thing that reads the NAME is a subtree grant, so the evaluator must be handed the
    name the row actually carries -- recomputing it would judge a pre-existing row by a
    name it does not have.
    """

    #: the name an existing row carries: the eight-character rule that preceded this one.
    _LEGACY_NAME = "memories.019470a8.019470a8"

    async def test_a_legacy_named_row_is_resolved_and_its_subtree_grant_still_holds(self) -> None:
        user_id = uuid4()
        ns = _StubNamespaceEntity(
            id=memory_namespace_id(_AGENT_A, _CUSTOMER),
            namespace_type="memory",
            owner_agent_id=_AGENT_A,
            customer_id=_CUSTOMER,
            name=self._LEGACY_NAME,
        )
        namespace_collection = _NamespaceCollectionFake(ns)
        provisioner = _ProvisionerFake(ref=_hub_ref(agent_id=_AGENT_A, customer_id=_CUSTOMER))
        deps = _build_deps(
            namespace_collection=namespace_collection,
            namespace_provisioner=provisioner,
            **_subtree_reader(root=self._LEGACY_NAME, user_id=user_id, customer_id=_CUSTOMER),
        )

        result = await authorize_memory_access(
            action=ACTION_MEMORY_READ,
            agent_id=_AGENT_A,
            customer_id=_CUSTOMER,
            caller_user_id=user_id,
            caller_agent_id=None,
            deps=deps,
        )

        assert result is ns
        assert provisioner.calls == [], "an existing row is resolved, never re-provisioned"
        assert namespace_collection.save_calls == []

    async def test_the_hub_provisioned_row_is_judged_by_the_name_the_hub_wrote(self) -> None:
        user_id = uuid4()
        hub_name = "memories.written-by-the-hub"
        ref = MemoryNamespaceRef(
            id=uuid4(),
            customer_id=_CUSTOMER,
            owner_agent_id=_AGENT_A,
            namespace_type="memory",
            owner_namespace=f"agents.{_AGENT_A}",
            name=hub_name,
        )
        deps = _build_deps(
            namespace_collection=_NamespaceCollectionFake(None),
            namespace_provisioner=_ProvisionerFake(ref=ref),
            **_subtree_reader(root=hub_name, user_id=user_id, customer_id=_CUSTOMER),
        )

        result = await authorize_memory_access(
            action=ACTION_MEMORY_READ,
            agent_id=_AGENT_A,
            customer_id=_CUSTOMER,
            caller_user_id=user_id,
            caller_agent_id=None,
            deps=deps,
        )

        assert result is ref

    async def test_the_owner_path_names_the_namespace_by_the_current_rule(self) -> None:
        """the owner path reads no row, so the only name it can carry is the one the rule derives."""
        deps = _build_deps(namespace_collection=_NamespaceCollectionUnavailableFake())

        result = await authorize_memory_access(
            action=ACTION_MEMORY_WRITE,
            agent_id=_AGENT_A,
            customer_id=_CUSTOMER,
            caller_user_id=None,
            caller_agent_id=_AGENT_A,
            deps=deps,
        )

        assert result.name == memory_namespace_name(_AGENT_A, _CUSTOMER)
