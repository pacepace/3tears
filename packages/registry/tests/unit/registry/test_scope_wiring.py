"""both registries in the registry-server process carry the L2 key scope (coll-task-06a).

``coll-task-03`` made every L2 key ``{scope}.{table}.{body}`` so the shared ``{ns}-collections``
bucket can be granted per principal instead of wholesale. This process is the awkward case the
scope work has to survive, for three reasons:

- **it runs TWO ``CollectionRegistry`` instances**, and an earlier draft of the wiring saw only
  the first. :func:`build_heartbeat_collection_registry` configures ``l2_client=`` directly;
  :func:`build_registry_rbac_stack` configures ``l1_backend`` + ``l3_pool`` and then hands each of
  five ACL collections its own ``nats_client=``, which WINS over the registry default. So the
  second registry is L2-live while ``configure()``'s unscoped-client refusal never fires for it.
- **both resolve to one principal.** One process, one NATS identity, one scope -- and replicas of
  the registry share it on purpose, because the scope is the sharing boundary rather than the
  connection.
- **``HeartbeatCollection`` has no L3.** L2 is its source of truth, so a scope that is merely
  present rather than correct does not cost a cache miss; it loses pod liveness records.

The bucket open is asserted here too: it must BIND rather than declare, and it must raise rather
than degrade.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from threetears.core.collections.base import BaseCollection
from threetears.core.config import DefaultCoreConfig
from threetears.nats import Principal, kv_key_scope_for
from threetears.nats.errors import KvConfigMismatch, KvError
from threetears.registry import server as server_module
from threetears.registry.auth import AllowAllAuthorizer
from threetears.registry.l1_cache import create_registry_l1_backend
from threetears.registry.rbac_stack import build_registry_rbac_stack
from threetears.registry.server import (
    COLLECTIONS_BUCKET_SUFFIX,
    RegistryServer,
    build_heartbeat_collection_registry,
)


def _nats_client() -> MagicMock:
    """a stand-in for the canonical wrapper client both builders take.

    :return: mock client exposing the ``raw`` escape hatch the proxy backend reads
    :rtype: MagicMock
    """
    client = MagicMock()
    client.raw = MagicMock()
    client.subscribe = AsyncMock(return_value=MagicMock())
    client.unsubscribe = AsyncMock()
    return client


class TestTheHeartbeatRegistryIsScoped:
    """the registry that owns ``HeartbeatCollection``."""

    def test_scope_is_the_registry_principal(self) -> None:
        """the scope comes from ``kv_key_scope_for``, never from a hand-built name."""
        registry, _ = build_heartbeat_collection_registry(
            nats_client=_nats_client(),
            l1_backend=create_registry_l1_backend(),
            core_config=DefaultCoreConfig(),
        )
        assert registry.kv_key_scope == kv_key_scope_for(Principal.REGISTRY)

    def test_keys_carry_the_scope_as_their_leading_token(self) -> None:
        """the property the grant depends on, asserted on a real composed key."""
        registry, heartbeat = build_heartbeat_collection_registry(
            nats_client=_nats_client(),
            l1_backend=create_registry_l1_backend(),
            core_config=DefaultCoreConfig(),
        )
        key = heartbeat.l2_key("pod-1")
        assert key.split(".")[0] == registry.kv_key_scope

    def test_it_binds_rather_than_declares_the_shared_bucket(self) -> None:
        """the registry holds ``STREAM.INFO`` on the collections stream and no ``CREATE``."""
        registry, _ = build_heartbeat_collection_registry(
            nats_client=_nats_client(),
            l1_backend=create_registry_l1_backend(),
            core_config=DefaultCoreConfig(),
        )
        assert registry.l2_create_if_missing is False


class TestTheRbacRegistryIsScoped:
    """the SECOND registry in the same process, L2-live through its collections."""

    def test_scope_is_the_registry_principal(self) -> None:
        """``configure()`` never sees a client here, so nothing else would demand a scope."""
        stack = build_registry_rbac_stack(
            nats_client=_nats_client(),
            subject_namespace="3tears",
            l1_backend=create_registry_l1_backend(),
        )
        assert stack.registry.kv_key_scope == kv_key_scope_for(Principal.REGISTRY)

    def test_its_collections_compose_scoped_keys(self) -> None:
        """the ACL collections hold their own NATS client; their keys must still be scoped."""
        stack = build_registry_rbac_stack(
            nats_client=_nats_client(),
            subject_namespace="3tears",
            l1_backend=create_registry_l1_backend(),
        )
        key = stack.group_collection.l2_key(("customer-1", "g1"))
        assert key.split(".")[0] == kv_key_scope_for(Principal.REGISTRY)

    def test_it_binds_rather_than_declares_the_shared_bucket(self) -> None:
        """both registries in one process must agree on bucket ownership."""
        stack = build_registry_rbac_stack(
            nats_client=_nats_client(),
            subject_namespace="3tears",
            l1_backend=create_registry_l1_backend(),
        )
        assert stack.registry.l2_create_if_missing is False


class TestBothRegistriesAgree:
    """two scopes in one process would split the cache and half the grant."""

    def test_the_two_registries_share_one_scope(self) -> None:
        """one process, one NATS identity, one key scope."""
        heartbeat_registry, _ = build_heartbeat_collection_registry(
            nats_client=_nats_client(),
            l1_backend=create_registry_l1_backend(),
            core_config=DefaultCoreConfig(),
        )
        stack = build_registry_rbac_stack(
            nats_client=_nats_client(),
            subject_namespace="3tears",
            l1_backend=create_registry_l1_backend(),
        )
        assert heartbeat_registry.kv_key_scope == stack.registry.kv_key_scope


class TestTheCollectionsBucketIsOpenedEagerly:
    """KVC-05: a mismatch must raise at wiring, not on the first cache access under load."""

    @pytest.mark.asyncio
    async def test_it_binds_the_collections_bucket(self) -> None:
        """the bucket name is the collection base's, and the open is a bind."""
        nc = MagicMock()
        nc.ensure_kv_bucket = AsyncMock(return_value=MagicMock())
        server = RegistryServer(namespace="testns", authorizer=AllowAllAuthorizer())

        await server.open_collections_bucket(nc)

        nc.ensure_kv_bucket.assert_awaited_once_with(
            name=BaseCollection.L2_BUCKET_SUFFIX,
            create_if_missing=False,
        )
        assert COLLECTIONS_BUCKET_SUFFIX == BaseCollection.L2_BUCKET_SUFFIX

    @pytest.mark.asyncio
    async def test_a_config_mismatch_propagates_on_the_first_attempt(self) -> None:
        """the raise is the point: a swallowed mismatch runs the fleet with L2 unenforceable.

        And it is not retried. Config drift does not heal on its own, so retrying it would only
        delay the report -- which is why ``KvConfigMismatch`` is deliberately not a ``KvError``
        subclass.
        """
        nc = MagicMock()
        nc.ensure_kv_bucket = AsyncMock(side_effect=KvConfigMismatch("allow_direct differs"))
        server = RegistryServer(namespace="testns", authorizer=AllowAllAuthorizer())

        with pytest.raises(KvConfigMismatch):
            await server.open_collections_bucket(nc)

        assert nc.ensure_kv_bucket.await_count == 1

    @pytest.mark.asyncio
    async def test_a_bind_failure_is_retried_then_succeeds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """the cold-cluster race: the hub declares this bucket and nothing sequences us behind it.

        Compose runs this service at ``restart: on-failure:5``, a budget a fast crash-loop burns
        through in seconds -- so a bind that fails because the declaring identity has not run yet
        must be waited out in-process rather than by dying.
        """
        monkeypatch.setattr(server_module.asyncio, "sleep", AsyncMock())
        nc = MagicMock()
        nc.ensure_kv_bucket = AsyncMock(side_effect=[KvError("bucket not found"), MagicMock()])
        server = RegistryServer(namespace="testns", authorizer=AllowAllAuthorizer())

        await server.open_collections_bucket(nc)

        assert nc.ensure_kv_bucket.await_count == 2

    @pytest.mark.asyncio
    async def test_an_exhausted_budget_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """a bounded retry that ended in silence would be the swallow this call exists to avoid."""
        monkeypatch.setattr(server_module.asyncio, "sleep", AsyncMock())
        nc = MagicMock()
        nc.ensure_kv_bucket = AsyncMock(side_effect=KvError("bucket not found"))
        server = RegistryServer(namespace="testns", authorizer=AllowAllAuthorizer())

        with pytest.raises(KvError, match="could not be bound"):
            await server.open_collections_bucket(nc)

        assert nc.ensure_kv_bucket.await_count > 1


class TestShutdownReleasesWhatStartupSubscribed:
    """every subscription `startup` makes must be released by `shutdown`.

    `startup` calls `start_invalidation_listener`, and `shutdown` tore down six other
    components while skipping it -- so the listener outlived the server that owned it.
    Nothing asserted the pairing, which is why the gap survived the shard that was
    supposed to close it. The registry server is also the in-repo model three consumer
    repos copy, so an unpaired lifecycle here propagates.
    """

    @pytest.mark.asyncio
    async def test_shutdown_stops_the_invalidation_listener(self) -> None:
        """the collection registry's listener is stopped, not left running.

        :return: nothing
        :rtype: None
        """
        server = RegistryServer(namespace="testns", authorizer=AllowAllAuthorizer())
        collection_registry = MagicMock()
        collection_registry.stop_invalidation_listener = AsyncMock()
        server._collection_registry = collection_registry  # noqa: SLF001 - drives the teardown under test

        await server.shutdown()

        collection_registry.stop_invalidation_listener.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_shutdown_runs_the_caller_supplied_teardown(self) -> None:
        """the `on_shutdown` seam is what gives `RegistryRbacStack.close()` a caller.

        The rbac factory builds a stack, subscribes its invalidations, and returns only the
        authorizer -- so without this hook the stack is unreachable from the server and its
        `close()` had zero production callers anywhere.

        :return: nothing
        :rtype: None
        """
        on_shutdown = AsyncMock()
        server = RegistryServer(
            namespace="testns",
            authorizer=AllowAllAuthorizer(),
            on_shutdown=on_shutdown,
        )

        await server.shutdown()

        on_shutdown.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_shutdown_is_safe_when_startup_never_ran(self) -> None:
        """a server torn down before it started must not raise.

        :return: nothing
        :rtype: None
        """
        server = RegistryServer(namespace="testns", authorizer=AllowAllAuthorizer())

        await server.shutdown()


class TestTheTeardownSeamIsBoundInEveryAuthorizerMode:
    """`_run_server` builds the teardown closure whatever mode it resolves.

    The list the closure drains is filled only by the rbac factory, so binding it inside
    that arm left it unbound under `THREETEARS_REGISTRY_ALLOW_ALL_TOOLS=true` and
    `THREETEARS_REGISTRY_FORCE_DENY_ALL=true` -- and the closure is passed
    unconditionally, so `shutdown()` raised `NameError` before the connection drained.
    Forced-deny is the production kill-switch, so that is the panic button failing to
    stop cleanly. The seam's own unit test could not catch it: it injects an `AsyncMock`
    and never reaches `_run_server`.
    """

    @pytest.mark.parametrize(
        "mode_env",
        ["THREETEARS_REGISTRY_ALLOW_ALL_TOOLS", "THREETEARS_REGISTRY_FORCE_DENY_ALL"],
    )
    def test_the_shutdown_hook_is_callable_in_a_non_rbac_mode(
        self, mode_env: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """the teardown resolves and runs, draining nothing, rather than raising.

        :param mode_env: the env var selecting a non-rbac authorizer mode
        :ptype mode_env: str
        :param monkeypatch: pytest monkeypatch fixture
        :ptype monkeypatch: pytest.MonkeyPatch
        :return: nothing
        :rtype: None
        """
        monkeypatch.setenv(mode_env, "true")
        captured: dict[str, object] = {}

        class _CapturingServer:
            def __init__(self, **kwargs: object) -> None:
                captured.update(kwargs)

            async def serve(self) -> None:
                return None

        monkeypatch.setattr(server_module, "RegistryServer", _CapturingServer)
        # `asyncio.run` is deliberately NOT patched: patching it on `server_module.asyncio`
        # patches the one shared module object, so the assertion below would hand its
        # coroutine to the stub and never execute it -- the test would pass against the
        # NameError it exists to catch. `_CapturingServer.serve` is already a no-op.

        server_module._run_server()  # noqa: SLF001 - the module entry point under test

        on_shutdown = captured["on_shutdown"]
        assert callable(on_shutdown)
        # the bug was a NameError raised the moment this ran
        asyncio.run(on_shutdown())  # type: ignore[operator]
