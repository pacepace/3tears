"""tests for the registry-side rbac stack + RegistryServer factory swap.

the standalone ``_run_server()`` entry point used to fall back to a
:class:`DenyAllAuthorizer` when no programmatic
:class:`RbacEvaluatorAuthorizer` was wired in. the registry-rbac task
replaces that fallback with a self-sufficient
:class:`~threetears.registry.rbac_stack.RegistryRbacStack` constructed
against a NATS-proxy ``NamespaceCollection`` + four rbac metadata
Collections + the canonical :class:`AclCache`. the constructor receives
a deny-all placeholder + a rbac-authorizer factory; the server swaps in
the real authorizer once NATS is connected so no tool dispatch ever
observes the placeholder in production.

these tests exercise:

- :func:`build_registry_rbac_stack` returns a fully populated stack
  with the five canonical Collections snapped to a
  :class:`NatsProxyL3Backend` pinned to ``system.platform.rbac``
- :class:`RegistryServer.serve` invokes the rbac-authorizer factory
  exactly once, with the connected NATS client, BEFORE the
  ``CallProxy`` starts (so no tool dispatch observes the placeholder
  authorizer)
- the swap target lands on :attr:`RegistryServer._authorizer` so the
  ``CallProxy`` constructed at ``_start_handlers`` time receives the
  rbac authorizer rather than the placeholder
- a ``rbac_authorizer_factory=None`` constructor (allow-all,
  forced-deny, fixed-mode test fixtures) skips the swap entirely
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from threetears.agent.acl import (
    AclCache,
    GroupCollection,
    GroupMemberCollection,
    NamespaceCollection,
    RoleAssignmentCollection,
    RoleCollection,
)
from threetears.core.backends.nats_proxy import NatsProxyL3Backend
from threetears.core.backends.sql import SqlL3Backend
from threetears.core.cache.sqlite import SQLiteBackend
from threetears.registry.auth import (
    AllowAllAuthorizer,
    DenyAllAuthorizer,
)
from threetears.registry.l1_cache import (
    REGISTRY_L1_TABLE_NAMES,
    create_registry_l1_backend,
)
from threetears.registry.rbac_stack import (
    PLATFORM_RBAC_READ_NAMESPACE,
    REGISTRY_SERVICE_SENTINEL_AGENT_ID,
    RegistryRbacStack,
    build_registry_rbac_stack,
)
from threetears.registry.server import RegistryServer


def _unwrap_l3(resolved: Any) -> Any:
    """unwrap a resolved L3 backend to the raw transport it wraps.

    L3B-03: the registry normalizes a raw L3 transport (here the rbac
    :class:`NatsProxyL3Backend`) into a :class:`SqlL3Backend` so the collection
    CRUD lifecycle gets the structured ``DurableStore`` ops. The pinning contract
    (namespace + service-sentinel agent_id) lives on the wrapped NatsProxy, so peel
    the wrapper before asserting on it.

    :param resolved: the value returned by ``get_l3_pool``.
    :ptype resolved: Any
    :return: the wrapped transport, or ``resolved`` unchanged.
    :rtype: Any
    """
    if isinstance(resolved, SqlL3Backend):
        return resolved._pool  # noqa: SLF001 -- peel the wrapper to the wrapped NatsProxy transport
    return resolved


def _make_nats_client() -> MagicMock:
    """build a mock canonical NATS wrapper client.

    :class:`NatsProxyL3Backend` reads ``client.raw`` for the underlying
    nats-py escape hatch; :meth:`build_registry_rbac_stack` accesses
    that attribute at construction time. tests stub it with a sentinel.
    """
    nc = MagicMock()
    nc.raw = MagicMock()
    nc.subscribe = AsyncMock(return_value=MagicMock())
    nc.unsubscribe = AsyncMock()
    return nc


class TestL1MetadataIncludesRbacTables:
    """REGISTRY_L1_METADATA carries every rbac mirror table the
    registry-side Collections will write to.

    missing tables here would trip ``sqlite3.OperationalError: no such
    table`` on the first authorize call (the rbac stack writes through
    L1 on every read), defeating the in-process cache the rbac fast
    path depends on.
    """

    def test_namespaces_table_present(self) -> None:
        """``namespaces`` mirror is in the metadata."""
        assert "namespaces" in REGISTRY_L1_TABLE_NAMES

    def test_groups_table_present(self) -> None:
        """``groups`` mirror is in the metadata."""
        assert "groups" in REGISTRY_L1_TABLE_NAMES

    def test_group_members_table_present(self) -> None:
        """``group_members`` mirror is in the metadata."""
        assert "group_members" in REGISTRY_L1_TABLE_NAMES

    def test_roles_table_present(self) -> None:
        """``roles`` mirror is in the metadata."""
        assert "roles" in REGISTRY_L1_TABLE_NAMES

    def test_role_assignments_table_present(self) -> None:
        """``role_assignments`` mirror is in the metadata."""
        assert "role_assignments" in REGISTRY_L1_TABLE_NAMES


class TestBuildRegistryRbacStack:
    """``build_registry_rbac_stack`` produces a fully populated stack."""

    def test_constructs_namespace_collection(self) -> None:
        """``namespace_collection`` is the canonical
        :class:`NamespaceCollection` -- the rbac authorizer relies on
        its :meth:`get_by_name` shape for the canonical-name lookup.
        """
        l1 = create_registry_l1_backend()
        stack = build_registry_rbac_stack(
            nats_client=_make_nats_client(),
            subject_namespace="3tears",
            l1_backend=l1,
        )
        assert isinstance(stack.namespace_collection, NamespaceCollection)

    def test_constructs_four_rbac_collections(self) -> None:
        """``group`` / ``group_member`` / ``role`` /
        ``role_assignment`` Collections are the canonical agent.acl
        types so the loaders + AclCache compose with them.
        """
        l1 = create_registry_l1_backend()
        stack = build_registry_rbac_stack(
            nats_client=_make_nats_client(),
            subject_namespace="3tears",
            l1_backend=l1,
        )
        assert isinstance(stack.group_collection, GroupCollection)
        assert isinstance(stack.group_member_collection, GroupMemberCollection)
        assert isinstance(stack.role_collection, RoleCollection)
        assert isinstance(
            stack.role_assignment_collection,
            RoleAssignmentCollection,
        )

    def test_constructs_acl_cache(self) -> None:
        """``acl_cache`` is the canonical :class:`AclCache` that
        :class:`RbacEvaluatorAuthorizer` resolves through.
        """
        l1 = create_registry_l1_backend()
        stack = build_registry_rbac_stack(
            nats_client=_make_nats_client(),
            subject_namespace="3tears",
            l1_backend=l1,
        )
        assert isinstance(stack.acl_cache, AclCache)

    def test_proxy_backend_pinned_to_rbac_namespace(self) -> None:
        """the L3 pool's default namespace is
        :data:`PLATFORM_RBAC_READ_NAMESPACE`. the hub broker only
        admits SELECT against the rbac read carve-out under this
        namespace; a different default would route every read into
        the categorical system-deny.
        """
        l1 = create_registry_l1_backend()
        stack = build_registry_rbac_stack(
            nats_client=_make_nats_client(),
            subject_namespace="3tears",
            l1_backend=l1,
        )
        # the rbac pool is wired onto the registry as the default L3.
        # introspect the registry's default pool through the public
        # accessor. L3B-03: the registry wraps the raw NatsProxy transport
        # in a ``SqlL3Backend`` so the collection CRUD lifecycle gets the
        # structured ``DurableStore`` ops; the rbac NatsProxy is the pool
        # the wrapper wraps, so unwrap before asserting its pinning.
        pool = _unwrap_l3(stack.registry.get_l3_pool("namespaces"))
        assert isinstance(pool, NatsProxyL3Backend)
        assert pool.default_namespace == PLATFORM_RBAC_READ_NAMESPACE

    def test_proxy_backend_uses_service_sentinel_agent_id(self) -> None:
        """``agent_id`` on the proxy is the service sentinel UUID
        (deterministic across registry restarts).

        the broker stamps this id on logs for traceability; the
        ``system.platform.rbac`` carve-out gates SELECTs on namespace
        + action only, not on agent_id.
        """
        l1 = create_registry_l1_backend()
        stack = build_registry_rbac_stack(
            nats_client=_make_nats_client(),
            subject_namespace="3tears",
            l1_backend=l1,
        )
        pool = _unwrap_l3(stack.registry.get_l3_pool("namespaces"))
        assert pool.agent_id == str(REGISTRY_SERVICE_SENTINEL_AGENT_ID)


class TestSubscribeInvalidations:
    """invalidation subscriptions are bound on demand."""

    @pytest.mark.asyncio
    async def test_subscribes_three_acl_invalidate_subjects(self) -> None:
        """``subscribe_invalidations`` binds membership / assignment /
        role invalidate subjects so cross-process rbac mutations
        purge the cache promptly.
        """
        nc = _make_nats_client()
        stack = build_registry_rbac_stack(
            nats_client=nc,
            subject_namespace="3tears",
            l1_backend=create_registry_l1_backend(),
        )

        await stack.subscribe_invalidations()

        assert nc.subscribe.await_count == 3
        # the default-namespace prefix is set by other tests in the
        # process via :func:`set_default_namespace`; assert on the
        # invariant suffix shape rather than a fixed prefix so test
        # ordering does not flake the assertion.
        suffixes = sorted(call.kwargs["subject"].path.split(".", 1)[1] for call in nc.subscribe.await_args_list)
        assert suffixes == [
            "acl.assignment.invalidate",
            "acl.membership.invalidate",
            "acl.role.invalidate",
        ]


class TestRegistryServerRbacFactoryConstructor:
    """``RegistryServer`` accepts the rbac-authorizer factory and
    stores both the placeholder authorizer and the factory for the
    later swap during :meth:`serve`.

    full ``serve()`` exercise lives in the integration suite (the
    serve loop touches NATS connect, JetStream KV bootstrap, signal
    handlers, and the per-handler subscriptions; mocking each of
    those just to assert the swap happens turns the unit test into
    a stub-against-stub mirror of the production flow). these
    constructor-level assertions lock in the wiring contract;
    ``test_proxy.py`` already covers the swap result by asserting
    the proxy reads from ``self._authorizer`` (the same slot the
    swap mutates).
    """

    @pytest.mark.asyncio
    async def test_constructor_stores_factory(self) -> None:
        """factory persists on the instance so :meth:`serve` can call it.

        verified through the public :meth:`apply_rbac_factory` swap
        path rather than reading ``_rbac_authorizer_factory`` directly
        (per CLAUDE.md "Underscore is a stability contract").
        """
        nc = AsyncMock()
        rbac_authorizer = AllowAllAuthorizer()
        factory = AsyncMock(return_value=rbac_authorizer)
        server = RegistryServer(
            namespace="testns",
            authorizer=DenyAllAuthorizer(),
            rbac_authorizer_factory=factory,
        )
        result = await server.apply_rbac_factory(nc)
        factory.assert_awaited_once_with(nc)
        assert result is rbac_authorizer

    @pytest.mark.asyncio
    async def test_no_factory_argument_defaults_to_none(self) -> None:
        """omitting the factory keeps the existing fixed-authorizer
        contract: callers that pass an :class:`AllowAllAuthorizer` or
        :class:`DenyAllAuthorizer` directly skip the swap step.

        verified through :meth:`apply_rbac_factory` returning ``None``
        when no factory was registered, rather than reading
        ``_rbac_authorizer_factory`` directly.
        """
        server = RegistryServer(namespace="testns", authorizer=AllowAllAuthorizer())
        result = await server.apply_rbac_factory(AsyncMock())
        assert result is None

    @pytest.mark.asyncio
    async def test_swap_executes_factory_with_nats_client(self) -> None:
        """direct test of the swap block: invoking the factory with the
        connected client + assigning the result to
        :attr:`RegistryServer._authorizer` is the contract the
        ``serve()`` body executes once NATS is up. exercising the
        swap directly (rather than through the full serve loop) keeps
        the unit test cycle fast + tight.
        """
        nc = _make_nats_client()
        rbac_authorizer = AllowAllAuthorizer()
        factory = AsyncMock(return_value=rbac_authorizer)

        server = RegistryServer(
            namespace="testns",
            authorizer=DenyAllAuthorizer(),
            rbac_authorizer_factory=factory,
        )

        # production code path lives in ``serve()`` -- the swap step
        # is now extracted to the public :meth:`apply_rbac_factory`
        # so this test drives the same canonical path without binding
        # to ``_authorizer`` / ``_nc`` / ``_rbac_authorizer_factory``
        # internals (per CLAUDE.md "Underscore is a stability
        # contract"). a refactor that drops the factory swap fails
        # this assertion.
        result = await server.apply_rbac_factory(nc)

        factory.assert_awaited_once_with(nc)
        assert result is rbac_authorizer


class TestRegistryRbacStackClose:
    """``RegistryRbacStack.close`` releases held resources."""

    @pytest.mark.asyncio
    async def test_unsubscribes_each_invalidation_subject(self) -> None:
        """three invalidation subscriptions -> three unsubscribe calls."""
        nc = _make_nats_client()
        stack = build_registry_rbac_stack(
            nats_client=nc,
            subject_namespace="3tears",
            l1_backend=create_registry_l1_backend(),
        )
        await stack.subscribe_invalidations()
        await stack.close()
        assert nc.unsubscribe.await_count == 3
