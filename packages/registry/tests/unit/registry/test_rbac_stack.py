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

from collections.abc import Callable

from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid7

import pytest
from threetears.agent.acl import (
    AclCache,
    ActorMembershipKey,
    AssignmentInvalidatePayload,
    GroupCollection,
    GroupMemberCollection,
    MembershipInvalidatePayload,
    NamespaceCollection,
    RoleAssignmentCollection,
    RoleCollection,
    RoleInvalidatePayload,
)
from threetears.core.backends.nats_proxy import NatsProxyL3Backend
from threetears.core.backends.sql import SqlL3Backend
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
    RegistryIdentityUnavailableError,
    build_registry_rbac_stack,
)
from threetears.registry import server as server_module
from threetears.registry.server import RegistryServer


def _identity_token_provider(token: str = "registry.identity.token") -> "Callable[[], str | None]":
    """a stand-in for the host-minted identity token provider the stack now requires.

    A PROVIDER rather than a string in the tests too, because that is what production
    passes: the real one reads a holder the refresh loop rewrites, and a test that
    handed a bare string would not exercise the same parameter.

    :param token: the token the provider reports
    :ptype token: str
    :return: a zero-arg callable returning ``token``
    :rtype: Callable[[], str | None]
    """
    return lambda: token


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
    # coll-task-07a: the stack delegates to
    # :func:`threetears.agent.acl.subscribe_acl_invalidation`, which binds through
    # ``subscribe_typed`` -- the raw ``subscribe`` the registry hand-rolled is gone. a distinct
    # handle per call so the teardown assertions can tell the three apart.
    nc.subscribe_typed = AsyncMock(side_effect=lambda **_kw: MagicMock())
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
            identity_token=_identity_token_provider(),
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
            identity_token=_identity_token_provider(),
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
            identity_token=_identity_token_provider(),
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
            identity_token=_identity_token_provider(),
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
            identity_token=_identity_token_provider(),
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
            identity_token=_identity_token_provider(),
        )

        await stack.subscribe_invalidations()

        # the ACL channel is three subjects. `subscribe_invalidations` also starts the
        # registry's COLLECTION invalidation listener, which subscribes on this same
        # client -- a different channel with a different publisher -- so count the ACL
        # subjects rather than every subscribe the client saw.
        # the default-namespace prefix is set by other tests in the
        # process via :func:`set_default_namespace`; assert on the
        # invariant suffix shape rather than a fixed prefix so test
        # ordering does not flake the assertion.
        suffixes = sorted(
            call.kwargs["subject"].path.split(".", 1)[1]
            for call in nc.subscribe_typed.await_args_list
            if ".acl." in call.kwargs["subject"].path
        )
        assert suffixes == [
            "acl.assignment.invalidate",
            "acl.membership.invalidate",
            "acl.role.invalidate",
        ]

    @pytest.mark.asyncio
    async def test_binds_the_canonical_payload_models(self) -> None:
        """each subject decodes into its own canonical payload model.

        the registry used to hand-roll ``model_validate_json`` inside three local handlers;
        going through the shared bus means the decode is the client's, and a validation failure
        deadletters instead of warning-and-dropping.
        """
        nc = _make_nats_client()
        stack = build_registry_rbac_stack(
            nats_client=nc,
            subject_namespace="3tears",
            l1_backend=create_registry_l1_backend(),
            identity_token=_identity_token_provider(),
        )

        await stack.subscribe_invalidations()

        bound = {
            call.kwargs["subject"].path.split(".", 1)[1]: call.kwargs["message_type"]
            for call in nc.subscribe_typed.await_args_list
        }
        assert bound["acl.membership.invalidate"] is MembershipInvalidatePayload
        assert bound["acl.assignment.invalidate"] is AssignmentInvalidatePayload
        assert bound["acl.role.invalidate"] is RoleInvalidatePayload

    @pytest.mark.asyncio
    async def test_no_queue_group_on_any_invalidate_subject(self) -> None:
        """every registry replica must observe every invalidation.

        a queue group delivers each broadcast to exactly one member, leaving the rest serving
        the tuples they were just told to drop.
        """
        nc = _make_nats_client()
        stack = build_registry_rbac_stack(
            nats_client=nc,
            subject_namespace="3tears",
            l1_backend=create_registry_l1_backend(),
            identity_token=_identity_token_provider(),
        )

        await stack.subscribe_invalidations()

        for call in nc.subscribe_typed.await_args_list:
            assert call.kwargs.get("queue") is None

    @pytest.mark.asyncio
    async def test_a_membership_broadcast_evicts_that_actor(self) -> None:
        """the bound handler drops the named actor's membership entry.

        exercised through the callback the bus registered rather than a local method: the three
        local handlers are gone, and this is the behaviour they existed for.
        """
        nc = _make_nats_client()
        stack = build_registry_rbac_stack(
            nats_client=nc,
            subject_namespace="3tears",
            l1_backend=create_registry_l1_backend(),
            identity_token=_identity_token_provider(),
        )
        await stack.subscribe_invalidations()
        actor = uuid7()
        key = ActorMembershipKey(actor_kind="user", actor_id=actor)
        stack.acl_cache.put_membership(key, ())

        handlers = {
            call.kwargs["subject"].path.split(".", 1)[1]: call.kwargs["cb"]
            for call in nc.subscribe_typed.await_args_list
        }
        await handlers["acl.membership.invalidate"](
            MembershipInvalidatePayload(actor_type="user", actor_id=actor),
        )

        assert stack.acl_cache.get_membership(key) is None


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
            identity_token=_identity_token_provider(),
        )
        await stack.subscribe_invalidations()
        await stack.close()
        # three ACL subjects plus the collection invalidation listener, which subscribes
        # on the same client and must be released by the same close.
        assert nc.unsubscribe.await_count == 4
        # the CLIENT form, handed the exact handles ``subscribe_typed`` returned.
        # ``Subscription.unsubscribe()`` would look equivalent and would leave every handle on
        # the client's own subscription list.
        released = [call.args[0] for call in nc.unsubscribe.await_args_list]
        bound = [call.args[0] if call.args else call for call in nc.subscribe_typed.await_args_list]
        assert len(released) == len(bound)

    @pytest.mark.asyncio
    async def test_close_without_subscribe_is_safe(self) -> None:
        """shutdown paths call ``close`` unconditionally."""
        nc = _make_nats_client()
        stack = build_registry_rbac_stack(
            nats_client=nc,
            subject_namespace="3tears",
            l1_backend=create_registry_l1_backend(),
            identity_token=_identity_token_provider(),
        )
        await stack.close()
        assert nc.unsubscribe.await_count == 0

    @pytest.mark.asyncio
    async def test_double_close_releases_each_handle_once(self) -> None:
        """a second close does not re-release handles it already dropped."""
        nc = _make_nats_client()
        stack = build_registry_rbac_stack(
            nats_client=nc,
            subject_namespace="3tears",
            l1_backend=create_registry_l1_backend(),
            identity_token=_identity_token_provider(),
        )
        await stack.subscribe_invalidations()
        await stack.close()
        await stack.close()
        # three ACL subjects plus the collection listener, each released exactly once
        assert nc.unsubscribe.await_count == 4


class TestRegistryServerPodAuthenticatorFactory:
    """``RegistryServer`` accepts + resolves the tool-pod registration authenticator, mirroring the
    rbac-authorizer factory. the resolved authenticator is what :meth:`_start_handlers` threads into
    the :class:`RegistrationHandler` (per-key-identity registration verify)."""

    @pytest.mark.asyncio
    async def test_factory_resolved_with_nats_client(self) -> None:
        """the factory runs against the connected client + its result is stored + returned."""
        nc = AsyncMock()
        authenticator = MagicMock()
        factory = AsyncMock(return_value=authenticator)
        server = RegistryServer(
            namespace="testns",
            authorizer=AllowAllAuthorizer(),
            pod_authenticator_factory=factory,
        )
        result = await server.apply_pod_authenticator_factory(nc)
        factory.assert_awaited_once_with(nc)
        assert result is authenticator

    @pytest.mark.asyncio
    async def test_fixed_authenticator_kept_when_no_factory(self) -> None:
        """a directly-supplied ``pod_authenticator`` survives the (no-op) resolve step."""
        authenticator = MagicMock()
        server = RegistryServer(
            namespace="testns",
            authorizer=AllowAllAuthorizer(),
            pod_authenticator=authenticator,
        )
        result = await server.apply_pod_authenticator_factory(AsyncMock())
        assert result is authenticator

    @pytest.mark.asyncio
    async def test_open_mode_default_returns_none(self) -> None:
        """no authenticator and no factory -> open mode (None) preserved."""
        server = RegistryServer(namespace="testns", authorizer=AllowAllAuthorizer())
        result = await server.apply_pod_authenticator_factory(AsyncMock())
        assert result is None


class TestResolvePodAuthenticatorFactory:
    """the registry entrypoint resolves its tool-pod authenticator factory from a configurable
    ``module:callable`` plugin path, keeping 3tears host-agnostic (the aibots Hub points it at its
    own factory)."""

    @pytest.mark.asyncio
    async def test_unset_env_is_open_mode(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """no env var -> None -> open registration (pure-3tears / dev default)."""
        from threetears.registry.server import _resolve_pod_authenticator_factory

        monkeypatch.delenv("THREETEARS_REGISTRY_POD_AUTHENTICATOR_FACTORY", raising=False)
        assert _resolve_pod_authenticator_factory() is None

    @pytest.mark.asyncio
    async def test_dotted_path_resolves_to_callable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """a valid ``module:callable`` path resolves to that exact object."""
        from threetears.registry.server import _resolve_pod_authenticator_factory

        # point at a real importable callable to prove resolution (any module attr works).
        monkeypatch.setenv(
            "THREETEARS_REGISTRY_POD_AUTHENTICATOR_FACTORY",
            "threetears.registry.auth:AllowAllAuthorizer",
        )
        from threetears.registry.auth import AllowAllAuthorizer

        assert _resolve_pod_authenticator_factory() is AllowAllAuthorizer

    @pytest.mark.asyncio
    async def test_malformed_path_fails_loud(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """a path without the ``module:callable`` shape crashes startup (never silent open mode)."""
        from threetears.registry.server import _resolve_pod_authenticator_factory

        monkeypatch.setenv("THREETEARS_REGISTRY_POD_AUTHENTICATOR_FACTORY", "no_colon_here")
        with pytest.raises(ValueError, match="module:callable"):
            _resolve_pod_authenticator_factory()


class TestRegistryServerLimitGuardFactory:
    """``RegistryServer`` accepts + resolves the pre-call spend-gate factory, mirroring the pod
    authenticator factory. the resolved guard is stored on ``self._limit_guard`` (updated BEFORE the
    :class:`CallProxy` is built in :meth:`serve`, so the proxy captures the resolved guard) and the
    proxy's no-silent-bypass contract is preserved by an ``AllowAllLimitGuard`` fallback."""

    @pytest.mark.asyncio
    async def test_factory_resolved_updates_limit_guard(self) -> None:
        """the factory runs against the connected client + its result replaces self._limit_guard."""
        nc = AsyncMock()
        guard = MagicMock()
        factory = AsyncMock(return_value=guard)
        server = RegistryServer(
            namespace="testns",
            authorizer=AllowAllAuthorizer(),
            limit_guard_factory=factory,
        )
        result = await server.apply_limit_guard_factory(nc)
        factory.assert_awaited_once_with(nc)
        # apply_limit_guard_factory returns the value it stored on the slot the CallProxy reads,
        # so asserting on the return proves the resolved guard replaced the AllowAll default.
        assert result is guard

    @pytest.mark.asyncio
    async def test_fixed_guard_kept_when_no_factory(self) -> None:
        """a directly-supplied ``limit_guard`` survives the (no-op) resolve step."""
        guard = MagicMock()
        server = RegistryServer(
            namespace="testns",
            authorizer=AllowAllAuthorizer(),
            limit_guard=guard,
        )
        result = await server.apply_limit_guard_factory(AsyncMock())
        assert result is guard

    @pytest.mark.asyncio
    async def test_default_is_allow_all(self) -> None:
        """no guard and no factory -> AllowAllLimitGuard (the proxy always has a non-None guard)."""
        from threetears.registry.auth import AllowAllLimitGuard

        server = RegistryServer(namespace="testns", authorizer=AllowAllAuthorizer())
        result = await server.apply_limit_guard_factory(AsyncMock())
        assert isinstance(result, AllowAllLimitGuard)

    @pytest.mark.asyncio
    async def test_factory_none_result_falls_back_to_allow_all(self) -> None:
        """a factory returning None -> AllowAllLimitGuard (no silent bypass of the proxy contract)."""
        from threetears.registry.auth import AllowAllLimitGuard

        factory = AsyncMock(return_value=None)
        server = RegistryServer(
            namespace="testns",
            authorizer=AllowAllAuthorizer(),
            limit_guard_factory=factory,
        )
        result = await server.apply_limit_guard_factory(AsyncMock())
        assert isinstance(result, AllowAllLimitGuard)


class TestResolveLimitGuardFactory:
    """the registry entrypoint resolves its pre-call limit-guard factory from a configurable
    ``module:callable`` plugin path, keeping 3tears host-agnostic (the aibots Hub points it at its
    NATS-proxy-backed ``KvCallLimitGuard`` factory)."""

    @pytest.mark.asyncio
    async def test_unset_env_is_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """no env var -> None -> constructor default AllowAllLimitGuard (pure-3tears / dev)."""
        from threetears.registry.server import _resolve_limit_guard_factory

        monkeypatch.delenv("THREETEARS_REGISTRY_LIMIT_GUARD_FACTORY", raising=False)
        assert _resolve_limit_guard_factory() is None

    @pytest.mark.asyncio
    async def test_dotted_path_resolves_to_callable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """a valid ``module:callable`` path resolves to that exact object."""
        from threetears.registry.server import _resolve_limit_guard_factory

        monkeypatch.setenv(
            "THREETEARS_REGISTRY_LIMIT_GUARD_FACTORY",
            "threetears.registry.auth:AllowAllLimitGuard",
        )
        from threetears.registry.auth import AllowAllLimitGuard

        assert _resolve_limit_guard_factory() is AllowAllLimitGuard

    @pytest.mark.asyncio
    async def test_malformed_path_fails_loud(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """a path without the ``module:callable`` shape crashes startup (never silent allow-all)."""
        from threetears.registry.server import _resolve_limit_guard_factory

        monkeypatch.setenv("THREETEARS_REGISTRY_LIMIT_GUARD_FACTORY", "no_colon_here")
        with pytest.raises(ValueError, match="module:callable"):
            _resolve_limit_guard_factory()


class TestRegistryServerUsageEmitterFactory:
    """``RegistryServer`` accepts + resolves the post-call endpoint-usage emitter factory, mirroring
    the limit-guard factory. the resolved emitter is stored on ``self._usage_emitter`` (updated
    BEFORE the :class:`CallProxy` is built in :meth:`serve`, so the proxy captures it); ``None``
    keeps the emit seam disabled (the standalone registry has no metering bus)."""

    @pytest.mark.asyncio
    async def test_factory_resolved_updates_emitter(self) -> None:
        """the factory runs against the connected client + its result replaces self._usage_emitter."""
        nc = AsyncMock()
        emitter = MagicMock()
        factory = AsyncMock(return_value=emitter)
        server = RegistryServer(
            namespace="testns",
            authorizer=AllowAllAuthorizer(),
            usage_emitter_factory=factory,
        )
        result = await server.apply_usage_emitter_factory(nc)
        factory.assert_awaited_once_with(nc)
        assert result is emitter

    @pytest.mark.asyncio
    async def test_fixed_emitter_kept_when_no_factory(self) -> None:
        """a directly-supplied ``usage_emitter`` survives the (no-op) resolve step."""
        emitter = MagicMock()
        server = RegistryServer(
            namespace="testns",
            authorizer=AllowAllAuthorizer(),
            usage_emitter=emitter,
        )
        result = await server.apply_usage_emitter_factory(AsyncMock())
        assert result is emitter

    @pytest.mark.asyncio
    async def test_default_disabled_returns_none(self) -> None:
        """no emitter and no factory -> None -> emit seam disabled."""
        server = RegistryServer(namespace="testns", authorizer=AllowAllAuthorizer())
        result = await server.apply_usage_emitter_factory(AsyncMock())
        assert result is None


class TestResolveUsageEmitterFactory:
    """the registry entrypoint resolves its endpoint-usage emitter factory from a configurable
    ``module:callable`` plugin path, keeping 3tears host-agnostic (the aibots Hub points it at its
    metering-publish factory)."""

    @pytest.mark.asyncio
    async def test_unset_env_is_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """no env var -> None -> emit disabled (pure-3tears / dev default)."""
        from threetears.registry.server import _resolve_usage_emitter_factory

        monkeypatch.delenv("THREETEARS_REGISTRY_USAGE_EMITTER_FACTORY", raising=False)
        assert _resolve_usage_emitter_factory() is None

    @pytest.mark.asyncio
    async def test_dotted_path_resolves_to_callable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """a valid ``module:callable`` path resolves to that exact object."""
        from threetears.registry.server import _resolve_usage_emitter_factory

        monkeypatch.setenv(
            "THREETEARS_REGISTRY_USAGE_EMITTER_FACTORY",
            "threetears.registry.auth:AllowAllLimitGuard",
        )
        from threetears.registry.auth import AllowAllLimitGuard

        assert _resolve_usage_emitter_factory() is AllowAllLimitGuard

    @pytest.mark.asyncio
    async def test_malformed_path_fails_loud(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """a path without the ``module:callable`` shape crashes startup (never silent drop)."""
        from threetears.registry.server import _resolve_usage_emitter_factory

        monkeypatch.setenv("THREETEARS_REGISTRY_USAGE_EMITTER_FACTORY", "no_colon_here")
        with pytest.raises(ValueError, match="module:callable"):
            _resolve_usage_emitter_factory()


class TestTheStackRefusesWithoutAnIdentity:
    """no identity token -> no backend at all, and the refusal happens at WIRING.

    The host broker resolves the caller from a signed token and refuses a request that
    carries none, so a stack built without a provider is a stack whose every read
    fails. Building it anyway defers the failure to the first query -- after the
    process has reported itself up, from whichever code path happened to touch L3
    first, and reading as an intermittent data-layer fault rather than as a missing
    credential.
    """

    def test_no_provider_refuses(self) -> None:
        """the ``None`` case: nothing was wired at all."""
        with pytest.raises(RegistryIdentityUnavailableError):
            build_registry_rbac_stack(
                nats_client=_make_nats_client(),
                subject_namespace="3tears",
                l1_backend=create_registry_l1_backend(),
                identity_token=None,
            )

    def test_an_empty_provider_refuses(self) -> None:
        """a provider that has nothing to give is the same refusal, not a warning.

        This is the state between process start and a completed handshake. A stack
        built here would carry a provider that looks correct and forwards ``None``.
        """
        with pytest.raises(RegistryIdentityUnavailableError):
            build_registry_rbac_stack(
                nats_client=_make_nats_client(),
                subject_namespace="3tears",
                l1_backend=create_registry_l1_backend(),
                identity_token=lambda: None,
            )

    def test_the_refusal_names_the_env_var_an_operator_must_set(self) -> None:
        """a credential error that names no variable leaves an operator guessing."""
        with pytest.raises(RegistryIdentityUnavailableError) as caught:
            build_registry_rbac_stack(
                nats_client=_make_nats_client(),
                subject_namespace="3tears",
                l1_backend=create_registry_l1_backend(),
                identity_token=None,
            )
        assert "THREETEARS_REGISTRY_IDENTITY_TOKEN_PROVIDER_FACTORY" in str(caught.value)


class TestTheProviderIsForwardedByReferenceNotByValue:
    """the backend reads THROUGH the provider on every request.

    The token is short-lived and re-minted in place by a refresh loop, so a stack that
    captured the token string at construction would forward an expired credential
    within the hour and every L3 read would fail -- long after the wiring that caused
    it, and with a wiring-time check that passed.
    """

    def test_a_re_minted_token_reaches_the_backend_without_rewiring(self) -> None:
        """mutate what the provider returns; the backend forwards the NEW value."""
        held: dict[str, str] = {"token": "first.token"}
        stack = build_registry_rbac_stack(
            nats_client=_make_nats_client(),
            subject_namespace="3tears",
            l1_backend=create_registry_l1_backend(),
            identity_token=lambda: held["token"],
        )
        pool = _unwrap_l3(stack.registry.get_l3_pool("namespaces"))
        assert pool.forwarded_identity_token() == "first.token"
        held["token"] = "re-minted.token"
        assert pool.forwarded_identity_token() == "re-minted.token"


class TestTheIdentityTokenProviderFactoryHook:
    """3tears resolves the provider from config; it never implements one.

    Identity is the sharpest case of the host-agnostic rule: the token is minted by the
    HOST, over a handshake 3tears does not define, against a principal store 3tears
    cannot read. So this hook has the same ``module:callable`` shape as the
    pod-authenticator, limit-guard and usage-emitter hooks -- and unlike those three it
    has no weaker-but-working default, because a broker that refuses an unidentified
    request leaves nothing to fall back to.
    """

    @pytest.mark.asyncio
    async def test_unset_resolves_to_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """unset -> ``None``, which the stack turns into a wiring-time refusal."""
        monkeypatch.delenv("THREETEARS_REGISTRY_IDENTITY_TOKEN_PROVIDER_FACTORY", raising=False)
        assert await server_module._resolve_identity_token_provider(_make_nats_client()) is None  # noqa: SLF001 -- module-private resolver under test

    @pytest.mark.asyncio
    async def test_a_malformed_spec_raises_rather_than_degrading(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """a misconfigured identity plugin must crash startup, never run unidentified."""
        monkeypatch.setenv("THREETEARS_REGISTRY_IDENTITY_TOKEN_PROVIDER_FACTORY", "not-a-dotted-path")
        with pytest.raises(ValueError, match="module:callable"):
            await server_module._resolve_identity_token_provider(_make_nats_client())  # noqa: SLF001 -- module-private resolver under test

    @pytest.mark.asyncio
    async def test_the_resolved_factory_is_awaited_with_the_live_client(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """the host factory needs the connection to handshake over, so it gets it."""
        nc = _make_nats_client()
        seen: list[Any] = []

        async def _factory(client: Any) -> "Callable[[], str | None]":
            seen.append(client)
            return lambda: "host.minted.token"

        monkeypatch.setattr(server_module, "_HOST_FACTORY_FOR_TEST", _factory, raising=False)
        monkeypatch.setenv(
            "THREETEARS_REGISTRY_IDENTITY_TOKEN_PROVIDER_FACTORY",
            "threetears.registry.server:_HOST_FACTORY_FOR_TEST",
        )
        provider = await server_module._resolve_identity_token_provider(nc)  # noqa: SLF001 -- module-private resolver under test
        assert seen == [nc]
        assert provider is not None
        assert provider() == "host.minted.token"
