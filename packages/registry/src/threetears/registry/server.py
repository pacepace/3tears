"""registry server entry point.

connects to NATS, initializes catalog from KV, starts all
handlers (registration, heartbeat, discovery, call proxy),
and waits for shutdown signal. all handlers subscribe with
queue group for horizontal scaling.

usage: python -m threetears.registry.server
"""

from __future__ import annotations

import asyncio
import os
import signal
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from threetears.core.cache.sqlite import SQLiteBackend
from threetears.core.collections.bucket import (
    COLLECTIONS_BUCKET_SUFFIX as _COLLECTIONS_BUCKET_SUFFIX,
)
from threetears.core.collections.bucket import (
    bind_collections_bucket,
)
from threetears.core.collections.registry import CollectionRegistry
from threetears.core.coordination.replay_guard import ReplayGuard
from threetears.core.config import DefaultCoreConfig
from threetears.core.security import (
    CachedHubJwksProvider,
    ProxyAssertionSigner,
    resolve_secret,
)
from threetears.nats import (
    RESULT_RETENTION_SECONDS,
    RESULT_STREAM_SUFFIX,
    NatsClient,
    Principal,
    Subjects,
    kv_key_scope_for,
)
from threetears.observe import HealthCheck, HealthServer, HealthTier, InflightRequestsGauge, get_logger
from threetears.observe.resilience import retry_with_backoff
from threetears.registry.catalog import ToolCatalog
from threetears.registry.discovery import DiscoveryHandler
from threetears.registry.health import HeartbeatSubscriber
from threetears.registry.heartbeat_collection import HeartbeatCollection
from threetears.registry.l1_cache import create_registry_l1_backend
from threetears.registry.proxy import CallProxy
from threetears.registry.auth import (
    AgentToolAuthorizer,
    AllowAllLimitGuard,
    EndpointUsageEmitter,
    LimitGuard,
    ToolPodAuthenticator,
)
from threetears.registry.config import (
    get_jwks_request_timeout,
    get_proxy_assertion_signing_key_ref,
)
from threetears.registry.registration import RegistrationHandler

if TYPE_CHECKING:
    from threetears.registry.rbac_stack import RegistryRbacStack

__all__ = [
    "COLLECTIONS_BUCKET_SUFFIX",
    "RegistryServer",
    "build_heartbeat_collection_registry",
    "nats_connect",
]

_logger = get_logger(__name__)

# a pop nonce must be remembered at least as long as a proof stays valid: the iat freshness
# window is +/- the pop leeway, so a captured proof is acceptable across twice that span.
_POP_NONCE_TTL_SECONDS = 120

#: the shared L2 bucket every ``BaseCollection`` in every process opens. re-exported from
#: :mod:`threetears.core.collections.bucket`, which takes it from the collection base rather than
#: spelling it again: the name is a wire fact shared by the hub's canonical declaration, this
#: process's eager bind, and the minted grant, and a second literal would be a second source of
#: truth for it.
COLLECTIONS_BUCKET_SUFFIX: str = _COLLECTIONS_BUCKET_SUFFIX


def build_heartbeat_collection_registry(
    *,
    nats_client: NatsClient,
    l1_backend: SQLiteBackend,
    core_config: DefaultCoreConfig,
) -> tuple[CollectionRegistry, HeartbeatCollection]:
    """build the registry server's own L2-live collection registry and its heartbeat store.

    THE FIRST OF TWO REGISTRIES in this process; :func:`~threetears.registry.rbac_stack.build_registry_rbac_stack`
    builds the second. Both resolve to the same principal and therefore to the same key scope --
    one process, one NATS identity, one scope -- and replicas of the registry share it
    deliberately, because the scope is the SHARING boundary rather than the connection.

    ``HeartbeatCollection`` has **no L3**, so this process's L2 is a SOURCE OF TRUTH and not a
    cache: a key that no longer resolves is a pod liveness record that no longer exists. That is
    also why the scope has to be right rather than merely present -- a wrong one does not miss,
    it loses.

    Extracted from :meth:`RegistryServer._start_handlers` so the scope decision is drivable by a
    test without standing up the whole serve loop, mirroring the reason
    :meth:`RegistryServer.apply_rbac_factory` is a method rather than an inline block.

    :param nats_client: connected canonical NATS wrapper client, used as the L2 tier and as the
        heartbeat collection's own client
    :ptype nats_client: NatsClient
    :param l1_backend: the process-local SQLite L1 tier
    :ptype l1_backend: SQLiteBackend
    :param core_config: collection flush configuration
    :ptype core_config: DefaultCoreConfig
    :return: the configured registry and the heartbeat collection snapped onto it
    :rtype: tuple[CollectionRegistry, HeartbeatCollection]
    :raises L2ScopeNotConfiguredError: never from here -- the scope is supplied in the same call
        that supplies the client, which is what the raise exists to demand
    """
    collection_registry = CollectionRegistry()
    # ``l2_create_if_missing=False``: this process BINDS the shared collections bucket, it never
    # declares it. The hub is the declaring identity, and the registry's own NATS grant carries
    # ``$JS.API.STREAM.INFO.KV_{ns}-collections`` with no ``CREATE`` -- so leaving the default
    # ``True`` would issue a create the broker never answers (a permissions refusal arrives as a
    # deadline, not as an error) before falling through to the bind that was always going to
    # succeed. Binding also RESTORES the config check: the declare path reconciles or falls
    # through silently, while the bind path compares ``allow_direct`` and refuses a bucket this
    # process must not run against.
    collection_registry.configure(
        l1_backend=l1_backend,
        l2_client=nats_client,
        kv_key_scope=kv_key_scope_for(Principal.REGISTRY),
        l2_create_if_missing=False,
    )
    heartbeat_collection = HeartbeatCollection(
        collection_registry,
        core_config,
        nats_client=nats_client,
    )
    return collection_registry, heartbeat_collection


# ---------------------------------------------------------------------------
# NATS connection helper (patched in tests)
# ---------------------------------------------------------------------------


async def nats_connect(
    url: str,
    *,
    namespace: str = "3tears",
    user: str | None = None,
    password: str | None = None,
) -> NatsClient:
    """connect to NATS server via the canonical :class:`NatsClient` wrapper.

    delegates to :meth:`NatsClient.connect` which handles dual-phase
    reconnect, rate-limited error logging, and namespace binding on
    :class:`Subjects`. tests patch this symbol to swap a fake
    transport into :class:`RegistryServer.serve`.

    under enforce-only connection auth (v0.13.9) the registry presents its OWN static
    user/password (NATS ``authorization.users``) so the server applies the registry user's
    coarse subject permissions; the enforcing bus has no ``no_auth_user``, so ``user``/``password``
    are REQUIRED there. ``None`` leaves credential auth off for tests + a non-enforcing bus.

    :param url: NATS server URL
    :ptype url: str
    :param namespace: NATS subject namespace prefix bound on the wrapper
    :ptype namespace: str
    :param user: NATS static username (config-mode ``authorization.users``); ``None`` -> no creds
    :ptype user: str | None
    :param password: NATS static password paired with ``user``; ``None`` -> no creds
    :ptype password: str | None
    :return: connected canonical wrapper client
    :rtype: NatsClient
    """
    return await NatsClient.connect(
        nats_url=url,
        nats_subject_namespace=namespace,
        client_name="registry",
        user=user,
        password=password,
    )


# ---------------------------------------------------------------------------
# RegistryServer
# ---------------------------------------------------------------------------


class RegistryServer:
    """registry server managing all handler lifecycles.

    connects to NATS, initializes tool catalog from KV store,
    starts registration handler, heartbeat monitor, discovery
    handler, and call proxy. handles graceful shutdown on
    SIGINT/SIGTERM signals.
    """

    def __init__(
        self,
        authorizer: AgentToolAuthorizer,
        nats_url: str | None = None,
        namespace: str | None = None,
        heartbeat_check_interval: float | None = None,
        heartbeat_timeout: float | None = None,
        call_timeout: float | None = None,
        kv_bucket: str = "tool_catalog",
        rbac_authorizer_factory: ("Callable[[NatsClient], Awaitable[AgentToolAuthorizer]] | None") = None,
        pod_authenticator: ToolPodAuthenticator | None = None,
        pod_authenticator_factory: ("Callable[[NatsClient], Awaitable[ToolPodAuthenticator | None]] | None") = None,
        health_port: int | None = None,
        limit_guard: "LimitGuard | None" = None,
        limit_guard_factory: ("Callable[[NatsClient], Awaitable[LimitGuard | None]] | None") = None,
        usage_emitter: "EndpointUsageEmitter | None" = None,
        usage_emitter_factory: ("Callable[[NatsClient], Awaitable[EndpointUsageEmitter | None]] | None") = None,
        on_shutdown: "Callable[[], Awaitable[None]] | None" = None,
    ) -> None:
        """initialize registry server.

        :param nats_url: NATS server URL (defaults to THREETEARS_NATS_URL env var)
        :ptype nats_url: str | None
        :param namespace: NATS subject namespace prefix (defaults to THREETEARS_NATS_SUBJECT_NAMESPACE env var)
        :ptype namespace: str | None
        :param heartbeat_check_interval: seconds between heartbeat check sweeps.
            sourced from THREETEARS_REGISTRY_HEARTBEAT_CHECK_INTERVAL env var if not provided.
        :ptype heartbeat_check_interval: float | None
        :param heartbeat_timeout: seconds after which pod is considered dead.
            sourced from THREETEARS_REGISTRY_HEARTBEAT_TIMEOUT env var if not provided.
        :ptype heartbeat_timeout: float | None
        :param call_timeout: timeout in seconds for forwarded tool calls.
            sourced from THREETEARS_REGISTRY_CALL_TIMEOUT env var if not provided.
        :ptype call_timeout: float | None
        :param kv_bucket: NATS KV bucket name for catalog persistence
        :ptype kv_bucket: str
        :param authorizer: initial tool access authorizer; REQUIRED.
            for the standalone production path the caller passes a
            :class:`DenyAllAuthorizer` placeholder + a
            ``rbac_authorizer_factory``; the placeholder denies any
            tool dispatch that races the rbac wiring (a millisecond
            cold-start window) and the server swaps in the factory's
            return value once NATS is connected. fixed-mode callers
            (allow-all dev sandboxes, force-deny kill switches) pass
            the deterministic stub directly with
            ``rbac_authorizer_factory=None`` and the swap step is
            skipped.
        :ptype authorizer: AgentToolAuthorizer
        :param rbac_authorizer_factory: optional async factory taking
            the connected :class:`NatsClient` and returning the
            production authorizer. invoked from :meth:`serve` after
            the NATS connection is up + before the catalog handlers
            register their subscriptions, so by the time tool calls
            arrive the authorizer slot already holds the rbac
            implementation. ``None`` keeps the constructor-supplied
            ``authorizer`` for the whole serve loop.
        :ptype rbac_authorizer_factory: Callable[[NatsClient], Awaitable[AgentToolAuthorizer]] | None
        :param pod_authenticator: tool-pod REGISTRATION authenticator, threaded into the
            :class:`~threetears.registry.registration.RegistrationHandler`. verifies each pod's
            self-minted identity JWT (carried on the manifest) against the pod's stored key and
            returns its allowed namespaces. ``None`` (with ``pod_authenticator_factory`` also
            ``None``) leaves registration in OPEN mode -- every manifest admitted unverified, the
            pure-3tears / dev default. host applications with a pod identity store (the aibots Hub)
            pass this or a factory to CLOSE open mode.
        :ptype pod_authenticator: ToolPodAuthenticator | None
        :param pod_authenticator_factory: optional async factory taking the connected
            :class:`NatsClient` and returning the pod authenticator (or ``None`` to keep open mode).
            invoked from :meth:`serve` after NATS connects + before the handlers register, mirroring
            ``rbac_authorizer_factory``, so a factory whose authenticator needs a live connection (a
            NATS-proxy-backed tool_pods read) can build against it. takes precedence over
            ``pod_authenticator`` when both are set. ``None`` keeps the constructor-supplied
            ``pod_authenticator`` for the whole serve loop.
        :ptype pod_authenticator_factory: Callable[[NatsClient], Awaitable[ToolPodAuthenticator | None]] | None
        :param health_port: port the readiness HealthServer binds to;
            defaults to THREETEARS_REGISTRY_HEALTH_PORT env var,
            falling back to 8000. each container in the platform's
            docker stack owns its own port namespace so 8000 is fine
            there; honcho-driven dev runs every Procfile entry in the
            host namespace and the hub uvicorn already binds host:8000,
            so honcho callers MUST set THREETEARS_REGISTRY_HEALTH_PORT
            to a distinct port.
        :ptype health_port: int | None
        :param limit_guard: pre-call spend gate threaded into the
            :class:`~threetears.registry.proxy.CallProxy`. ``None`` (the
            pure-3tears / dev default) wires :class:`AllowAllLimitGuard`
            -- the standalone registry has no counter backend, so it
            does not enforce spend limits. host applications with a
            counter store (the aibots Hub) inject the concrete
            ``KvCallLimitGuard`` (gu-task-15a) here to CLOSE the gate,
            mirroring how ``pod_authenticator`` closes open registration
            mode. the proxy itself REQUIRES a guard (no silent bypass);
            this slot only chooses WHICH guard the standalone server
            wires.
        :ptype limit_guard: LimitGuard | None
        :param limit_guard_factory: optional async factory taking the connected
            :class:`NatsClient` and returning the pre-call spend gate (or ``None`` to keep the
            constructor-supplied ``limit_guard``). invoked from :meth:`serve` after NATS connects
            + before the handlers register (alongside ``pod_authenticator_factory``), so a guard
            whose counter reads run over the live connection (the aibots Hub's
            NATS-proxy-backed ``KvCallLimitGuard``) can build against it. takes precedence over
            ``limit_guard`` when both are set. ``None`` keeps the constructor-supplied
            ``limit_guard`` (``AllowAllLimitGuard`` by default) for the whole serve loop.
        :ptype limit_guard_factory: Callable[[NatsClient], Awaitable[LimitGuard | None]] | None
        :param usage_emitter: post-call endpoint-usage emit seam threaded into the
            :class:`~threetears.registry.proxy.CallProxy`. ``None`` (the pure-3tears / dev default)
            means no per-call usage event is published -- the standalone registry has no usage
            pipeline. host applications with a metering bus (the aibots Hub) inject a concrete emitter
            (or a factory) to publish the ``EndpointUsageEvent`` gu-task-16 consumes.
        :ptype usage_emitter: EndpointUsageEmitter | None
        :param usage_emitter_factory: optional async factory taking the connected
            :class:`NatsClient` and returning the usage emitter (or ``None`` for no emit). invoked
            from :meth:`serve` after NATS connects (alongside ``limit_guard_factory``) so an emitter
            whose publish path needs the live connection can build against it. takes precedence over
            ``usage_emitter`` when both are set. ``None`` keeps the constructor-supplied ``usage_emitter``.
        :ptype usage_emitter_factory: Callable[[NatsClient], Awaitable[EndpointUsageEmitter | None]] | None
        """
        from threetears.registry.config import get_call_timeout, get_heartbeat_check_interval, get_heartbeat_timeout

        self._nats_url = nats_url or os.environ.get(
            "THREETEARS_NATS_URL",
            "nats://localhost:4222",
        )
        self._namespace = namespace or os.environ.get(
            "THREETEARS_NATS_SUBJECT_NAMESPACE",
            "3tears",
        )
        # enforce-only connection auth (v0.13.9): the registry connects as its OWN static NATS user
        # (the enforcing dev bus has no ``no_auth_user``). registry is a 3tears-package consumer, so
        # the creds come from the THREETEARS_NATS_ env namespace. unset -> anonymous (tests + a
        # non-enforcing bus); on the enforcing bus the compose / Procfile sets these.
        self._nats_user = os.environ.get("THREETEARS_NATS_USER") or None
        self._nats_password = os.environ.get("THREETEARS_NATS_PASSWORD") or None
        self._heartbeat_check_interval = (
            heartbeat_check_interval if heartbeat_check_interval is not None else get_heartbeat_check_interval()
        )
        self._heartbeat_timeout = heartbeat_timeout if heartbeat_timeout is not None else get_heartbeat_timeout()
        self._call_timeout = call_timeout if call_timeout is not None else get_call_timeout()
        self._kv_bucket = kv_bucket
        self._authorizer = authorizer
        self._limit_guard: LimitGuard = limit_guard if limit_guard is not None else AllowAllLimitGuard()
        self._rbac_authorizer_factory = rbac_authorizer_factory
        self._pod_authenticator = pod_authenticator
        self._pod_authenticator_factory = pod_authenticator_factory
        self._limit_guard_factory = limit_guard_factory
        self._usage_emitter: EndpointUsageEmitter | None = usage_emitter
        self._usage_emitter_factory = usage_emitter_factory
        if health_port is not None:
            self._health_port = health_port
        else:
            env_port = os.environ.get("THREETEARS_REGISTRY_HEALTH_PORT")
            self._health_port = int(env_port) if env_port else 8000
        self._nc: "NatsClient | None" = None
        self._catalog = ToolCatalog()
        self._collection_registry: CollectionRegistry | None = None
        self._heartbeat_collection: HeartbeatCollection | None = None
        self._registration_handler: RegistrationHandler | None = None
        self._heartbeat_subscriber: HeartbeatSubscriber | None = None
        self._discovery_handler: DiscoveryHandler | None = None
        self._call_proxy: CallProxy | None = None
        self._jwks_provider: CachedHubJwksProvider | None = None
        # a caller-supplied teardown, awaited LAST before the connection closes. It exists
        # because a factory can build something whose lifecycle the server never sees:
        # `_rbac_factory` builds a `RegistryRbacStack`, subscribes its invalidations, and
        # returns only the authorizer -- so `RegistryRbacStack.close()` had no production
        # caller anywhere and its subscriptions outlived the server that made them.
        self._on_shutdown = on_shutdown
        self._health_server: HealthServer | None = None
        self._inflight_gauge: InflightRequestsGauge | None = None
        self._shutdown_event = asyncio.Event()

    async def open_collections_bucket(self, nc: "NatsClient") -> None:
        """bind the shared L2 collections bucket before any collection is wired against it.

        ``coll-task-04a`` KVC-05. Two properties, neither of which survives lazy resolution:

        - **it fails at wiring time, not in a request path.** ``BaseCollection`` resolves the
          bucket in ``_ensure_kv``, on the first read -- so a bucket running a configuration this
          process refuses (``allow_direct`` unset, which puts every KV read back on the
          body-carried form no key-scoped grant can constrain) would surface as a
          ``KvConfigMismatch`` raised under load rather than at startup.
        - **it pins the handle every later opener shares.** ``NatsClient.kv_bucket`` caches by
          full bucket name, so the first open in a process decides the configuration for the
          life of that process. Opening here, before either registry is configured, is what
          makes the two registries' view of the bucket the same one.

        The BIND-not-declare choice, and the two failure classes it distinguishes, belong to
        :func:`threetears.core.collections.bucket.bind_collections_bucket` -- promoted there by
        ``coll-task-07c`` so a tool pod's bootstrap does not become a third hand-written copy of a
        security-relevant startup step. This method stays as the registry's named entry point:
        ``serve()`` calls it, and tests drive it without standing up the serve loop.

        :param nc: connected canonical NATS wrapper client
        :ptype nc: NatsClient
        :return: nothing
        :rtype: None
        :raises KvError: the bucket could not be bound within the attempt budget -- it does not
            exist (nothing has declared it) or this principal is not granted it
        :raises KvConfigMismatch: the live bucket carries a configuration this process refuses
        """
        # component=: several processes bind this same bucket and the interesting failure is
        # an ORDERING one -- which of them reached it before the declaring identity. A bind
        # log that does not say who is speaking cannot answer that.
        await bind_collections_bucket(nc, component="registry")

    async def apply_rbac_factory(
        self,
        nc: "NatsClient",
    ) -> "AgentToolAuthorizer | None":
        """swap the placeholder authorizer for the live rbac one.

        the constructor receives a ``DenyAllAuthorizer`` placeholder
        while the rbac stack waits for a connected NATS client to back
        its proxy collections + invalidation subscriptions; the
        ``rbac_authorizer_factory`` builds the stack against the live
        connection and returns the live
        :class:`RbacEvaluatorAuthorizer`. extracted from
        :meth:`serve` so tests can exercise the swap without binding
        to private state, and so any other caller assembling the
        server out-of-band (e.g. an integration harness that
        constructs its own NATS connection) can drive the same swap
        through one canonical entry point.

        no-op when the factory is None (fixed-mode authorizers like
        ``AllowAllAuthorizer`` skip the swap because they have no
        live-binding requirement).

        :param nc: connected NATS client the factory needs to back its
            proxy collections + invalidation subscriptions
        :ptype nc: NatsClient
        :return: the new authorizer (also stored on self) when a
            factory was set, ``None`` when the placeholder was kept
        :rtype: AgentToolAuthorizer | None
        """
        result: "AgentToolAuthorizer | None" = None
        if self._rbac_authorizer_factory is not None:
            result = await self._rbac_authorizer_factory(nc)
            self._authorizer = result
        return result

    async def apply_pod_authenticator_factory(
        self,
        nc: "NatsClient",
    ) -> "ToolPodAuthenticator | None":
        """build the tool-pod registration authenticator from the factory (if one was set).

        mirrors :meth:`apply_rbac_factory`: the constructor may receive a ``pod_authenticator``
        directly (fixed authenticator) OR a ``pod_authenticator_factory`` that needs the live NATS
        connection to back its pod-identity read (e.g. a NATS-proxy-backed tool_pods collection).
        this resolves the factory once NATS is up and stores the result on ``self`` so
        :meth:`_start_handlers` threads it into the :class:`RegistrationHandler`. extracted from
        :meth:`serve` so tests can drive the swap without binding to private state.

        no-op when no factory is set (the constructor-supplied ``pod_authenticator`` -- possibly
        ``None`` for open mode -- is kept).

        :param nc: connected NATS client the factory needs to back its pod-identity read
        :ptype nc: NatsClient
        :return: the resolved authenticator (also stored on self) when a factory was set, else the
            existing ``pod_authenticator``
        :rtype: ToolPodAuthenticator | None
        """
        if self._pod_authenticator_factory is not None:
            self._pod_authenticator = await self._pod_authenticator_factory(nc)
        return self._pod_authenticator

    async def apply_limit_guard_factory(
        self,
        nc: "NatsClient",
    ) -> "LimitGuard":
        """build the pre-call spend gate from the factory (if one was set).

        mirrors :meth:`apply_pod_authenticator_factory`: the constructor may receive a
        ``limit_guard`` directly OR a ``limit_guard_factory`` that needs the live NATS connection
        to back its counter reads (the aibots Hub's NATS-proxy-backed ``KvCallLimitGuard``). this
        resolves the factory once NATS is up and stores the result on ``self._limit_guard`` so the
        :class:`~threetears.registry.proxy.CallProxy` built later in :meth:`serve` threads in the
        resolved guard. a factory returning ``None`` falls back to :class:`AllowAllLimitGuard` so
        the proxy's no-silent-bypass contract is never violated.

        no-op when no factory is set (the constructor-supplied ``limit_guard`` --
        ``AllowAllLimitGuard`` by default -- is kept).

        :param nc: connected NATS client the factory needs to back its counter reads
        :ptype nc: NatsClient
        :return: the resolved guard (also stored on self) when a factory was set, else the existing
            ``limit_guard``
        :rtype: LimitGuard
        """
        if self._limit_guard_factory is not None:
            self._limit_guard = await self._limit_guard_factory(nc) or AllowAllLimitGuard()
        return self._limit_guard

    async def apply_usage_emitter_factory(
        self,
        nc: "NatsClient",
    ) -> "EndpointUsageEmitter | None":
        """build the post-call endpoint-usage emitter from the factory (if one was set).

        mirrors :meth:`apply_limit_guard_factory`: resolves the factory once NATS is up and stores
        the result on ``self._usage_emitter`` so the :class:`~threetears.registry.proxy.CallProxy`
        built later in :meth:`serve` threads in the resolved emitter. a factory returning ``None``
        (or no factory) leaves the emit seam disabled -- the pure-3tears / dev default.

        :param nc: connected NATS client the factory needs to back its publish path
        :ptype nc: NatsClient
        :return: the resolved emitter (also stored on self), or ``None`` when no factory / no emitter
        :rtype: EndpointUsageEmitter | None
        """
        if self._usage_emitter_factory is not None:
            self._usage_emitter = await self._usage_emitter_factory(nc)
        return self._usage_emitter

    async def serve(self) -> None:
        """start registry server and wait for shutdown signal.

        connects to NATS, loads catalog from KV, starts all
        handlers, installs signal handlers, and blocks until
        shutdown is requested.
        """
        self._nc = await nats_connect(
            self._nats_url,
            namespace=self._namespace,
            user=self._nats_user,
            password=self._nats_password,
        )
        _logger.info(
            "connected to NATS",
            extra={"extra_data": {"nats_url": self._nats_url}},
        )

        # open the shared collections bucket EAGERLY, before anything configures an L2 client
        # against it. ``coll-task-04a`` KVC-05: a mismatch resolved lazily lands in a request
        # path under load, and ``NatsClient.kv_bucket`` caches by full bucket name, so whichever
        # call opens the bucket first hands its handle to every collection after it. Both
        # registries in this process are wired downstream of this line.
        await self.open_collections_bucket(self._nc)

        # swap in the production rbac authorizer now that NATS is up.
        # extracted to a method so tests can drive the same code path
        # without binding to private state.
        await self.apply_rbac_factory(self._nc)

        # resolve the tool-pod registration authenticator (per-key identity) now that NATS is up,
        # so a factory whose authenticator reads pod identity over the live connection can build
        # against it before the registration handler starts. no-op / open mode when unset.
        await self.apply_pod_authenticator_factory(self._nc)

        # resolve the pre-call spend gate now that NATS is up, so a factory whose counter reads run
        # over the live connection (the Hub's NATS-proxy-backed KvCallLimitGuard) builds against it
        # BEFORE the CallProxy is constructed and captures self._limit_guard. no-op / AllowAll when
        # unset.
        await self.apply_limit_guard_factory(self._nc)

        # resolve the post-call endpoint-usage emitter now that NATS is up, so a host emitter whose
        # publish path needs the live connection builds against it BEFORE the CallProxy captures
        # self._usage_emitter. no-op (emit disabled) when unset.
        await self.apply_usage_emitter_factory(self._nc)

        # the catalog KV bootstrap predates the wrapper's
        # :meth:`NatsClient.kv_bucket` cache; we still go through the
        # raw JetStream context here because the catalog persists JSON
        # blobs keyed by tool full_name and is loaded with a custom
        # iterator (``ToolCatalog.load_from_kv``) that expects a raw
        # nats-py KeyValue handle. migrating the catalog persistence
        # to NatsKvBucket is tracked as follow-up work.
        js = self._nc.jetstream_context()

        authorizer = self._authorizer
        if authorizer is not None and hasattr(authorizer, "initialize"):

            async def _initialize_authorizer() -> None:
                """initialize authorizer with JetStream context."""
                await authorizer.initialize(js, self._namespace)

            await retry_with_backoff(
                _initialize_authorizer,
                "registry.authorizer_initialize",
            )

        async def _ensure_kv_and_load_catalog() -> None:
            """ensure KV bucket exists and load catalog from it."""
            nonlocal js
            try:
                kv = await js.key_value(bucket=self._kv_bucket)
            except Exception:
                kv = await js.create_key_value(bucket=self._kv_bucket)
                _logger.info(
                    "created KV bucket",
                    extra={"extra_data": {"bucket": self._kv_bucket}},
                )
            await self._catalog.load_from_kv(kv)
            _logger.info(
                "catalog loaded from KV",
                extra={"extra_data": {"bucket": self._kv_bucket}},
            )

        await retry_with_backoff(
            _ensure_kv_and_load_catalog,
            "registry.kv_catalog_load",
        )

        await retry_with_backoff(
            self._ensure_result_stream,
            "registry.result_stream_declare",
        )

        await self._start_handlers()

        self._install_signal_handlers()

        _logger.info(
            "registry server started",
            extra={"extra_data": {"namespace": self._namespace}},
        )

        await self._shutdown_event.wait()

    async def _ensure_result_stream(self) -> None:
        """declare the stream that carries every answer too long to ride the reply inbox.

        Declared HERE, and before the handlers start, because the registry is the one process on both
        sides of that path: it consumes each pod's result and publishes each agent's reply. It is also
        the only participant guaranteed to be running before any call can route at all, so a pod can
        never find itself publishing into a stream that does not exist.

        Both families live on one stream. A subject belongs to exactly one stream on a NATS account,
        the two have identical retention needs, and one name means one JetStream control-plane grant
        per principal rather than two.

        ``max_msgs_per_subject=1`` matters as much as the age bound: each subject is minted for one
        call and answered once, so a retried delivery must replace its predecessor rather than leave a
        stale first answer for the waiter to collect.

        :return: nothing
        :rtype: None
        :raises RuntimeError: when invoked before NATS is connected
        """
        if self._nc is None:
            raise RuntimeError("_ensure_result_stream invoked before NATS connected")
        stream = await self._nc.ensure_jetstream_stream(
            name=RESULT_STREAM_SUFFIX,
            subjects=[
                str(Subjects.tools_result_wildcard()),
                str(Subjects.tools_reply_wildcard()),
            ],
            max_age_seconds=RESULT_RETENTION_SECONDS,
            max_msgs_per_subject=1,
        )
        _logger.info(
            "durable tool-result stream ensured",
            extra={"extra_data": {"stream": stream, "retention_seconds": RESULT_RETENTION_SECONDS}},
        )

    async def _start_handlers(self) -> None:
        """initialize and start all registry handlers.

        starts registration handler, heartbeat monitor,
        discovery handler, and call proxy. requires ``serve`` to have
        connected NATS first.

        :raises RuntimeError: when invoked before NATS is connected
        """
        if self._nc is None:
            raise RuntimeError("_start_handlers invoked before NATS connected")
        nc = self._nc
        registration_handler = RegistrationHandler(
            self._catalog,
            namespace=self._namespace,
            authenticator=self._pod_authenticator,
        )
        self._registration_handler = registration_handler
        await retry_with_backoff(
            lambda: registration_handler.start(nc),
            "registry.registration_handler.start",
        )

        # wire the heartbeat collection + subscriber. L1 is a
        # per-process SQLite tier; L2 is the shared NATS connection
        # that also carries the cross-pod invalidation subject.
        l1_backend = create_registry_l1_backend()
        core_config = DefaultCoreConfig(
            collection_flush="ALWAYS",
            collection_flush_tables="",
        )
        collection_registry, heartbeat_collection = build_heartbeat_collection_registry(
            nats_client=nc,
            l1_backend=l1_backend,
            core_config=core_config,
        )
        self._collection_registry = collection_registry
        self._heartbeat_collection = heartbeat_collection
        # log the RESOLVED scope, not just that one was supplied. A wrong-but-present scope
        # reads nothing and writes where no grant names, and its timeout signature is
        # identical to a missing grant -- so without this an operator is told to obtain a
        # grant they already hold.
        _logger.info(
            "registry L2 wired",
            extra={
                "extra_data": {
                    "kv_key_scope": kv_key_scope_for(Principal.REGISTRY),
                    "principal": Principal.REGISTRY.value,
                }
            },
        )
        await collection_registry.start_invalidation_listener(nc)

        heartbeat_subscriber = HeartbeatSubscriber(
            self._catalog,
            heartbeat_collection,
            namespace=self._namespace,
            check_interval=self._heartbeat_check_interval,
            timeout=self._heartbeat_timeout,
        )
        self._heartbeat_subscriber = heartbeat_subscriber
        await retry_with_backoff(
            lambda: heartbeat_subscriber.start(nc),
            "registry.heartbeat_subscriber.start",
        )

        discovery_handler = DiscoveryHandler(
            self._catalog,
            namespace=self._namespace,
        )
        self._discovery_handler = discovery_handler
        await retry_with_backoff(
            lambda: discovery_handler.start(nc),
            "registry.discovery_handler.start",
        )

        # a real JWKS provider that fetches the Hub's published identity-token keys (request/reply)
        # and caches them, so the proxy verifies tokens against live Hub keys before RBAC. fetch is
        # best-effort at start (fail-closed empty cache until the first success); a refresh loop
        # keeps it current through key rotation.
        jwks_provider = CachedHubJwksProvider(nc, request_timeout_seconds=get_jwks_request_timeout())
        await jwks_provider.start()
        self._jwks_provider = jwks_provider

        # the proxy's assertion-signing key (shared secret_ref with the Hub, which publishes its
        # PUBLIC key in the JWKS) is REQUIRED under enforce-only: a registry that cannot sign a
        # proxy->pod assertion would forward unsigned calls every pod is bound to reject. an absent
        # or malformed key must fail startup loudly (the exception propagates out of serve) rather
        # than boot the registry into a silently-broken state.
        proxy_signer = ProxyAssertionSigner.from_secret(resolve_secret(get_proxy_assertion_signing_key_ref()))

        # the pop replay guard records each per-call proof nonce for single-use enforcement;
        # always provisioned under enforce-only so a captured pop can never be replayed verbatim
        # for the same call body within the iat freshness window.
        pop_replay_guard = ReplayGuard(nc, bucket_name="pop_nonces", ttl_seconds=_POP_NONCE_TTL_SECONDS)

        # one in-flight-requests gauge for this registry replica: the CallProxy
        # brackets every dispatch through it, and the HealthServer serves it on
        # ``/metrics`` so KEDA's prometheus scaler can autoscale replicas on
        # aggregate in-flight tool-call load (the queue-group RPC path is not
        # JetStream, so there is no stream backlog for the nats scaler to read).
        inflight_gauge = InflightRequestsGauge("threetears_registry_inflight_requests")
        self._inflight_gauge = inflight_gauge

        call_proxy = CallProxy(
            self._catalog,
            namespace=self._namespace,
            timeout=self._call_timeout,
            authorizer=self._authorizer,
            limit_guard=self._limit_guard,
            usage_emitter=self._usage_emitter,
            inflight_gauge=inflight_gauge,
            jwks_provider=jwks_provider,
            # reactive self-heal on a Hub re-key: a kid-not-in-cache miss triggers ONE immediate,
            # debounced + rate-limited refresh + re-verify, so a valid token signed under a freshly-
            # rotated Hub key heals on the first failed-but-valid call rather than after the steady tick.
            jwks_refresh=jwks_provider.refresh_now,
            proxy_signer=proxy_signer,
            pop_replay_guard=pop_replay_guard,
        )
        self._call_proxy = call_proxy
        await retry_with_backoff(
            lambda: call_proxy.start(nc),
            "registry.call_proxy.start",
        )

        # canonical health endpoints -- consumed by docker compose + k8s probes +
        # the consumer's devx preflight. port 8000 matches the inherited upstream
        # hub Dockerfile HEALTHCHECK so the same probe works whether the container
        # runs as the hub, the registry, or any other consumer of that base.
        health_server = HealthServer(
            port=self._health_port,
            service_name="registry",
            metrics_provider=inflight_gauge.render,
            checks=[
                # key liveness on REAL NATS health (is_closed / is_healthy), NOT
                # is_connected -- the latter is a stale-socket flag that stays True
                # through a terminal close or a persistent auth / overflow wedge, so
                # a probe on it reports healthy forever with a dead registry (the
                # silent-zombie bug). a transient reconnect keeps this True so
                # forever-reconnect never flaps liveness into a restart loop.
                HealthCheck(
                    name="nats",
                    probe=lambda: self._nc is not None and not self._nc.is_closed and self._nc.is_healthy,
                    tier=HealthTier.LIVE,
                ),
                # a replica whose intake or routing subscription is unbound cannot
                # learn about tool pods or route a call, so it leaves rotation. NOT
                # liveness: the usual cause is a NATS outage, which a restart does
                # not fix and which self-heals on reconnect.
                HealthCheck(
                    name="registration_subscribed",
                    probe=lambda: (
                        self._registration_handler is not None and self._registration_handler.subscription_active
                    ),
                    tier=HealthTier.READY,
                ),
                HealthCheck(
                    name="call_proxy_subscribed",
                    probe=lambda: self._call_proxy is not None and self._call_proxy.subscription_active,
                    tier=HealthTier.READY,
                ),
                # readiness gate: report NOT-READY until the Hub JWKS cache has had its first
                # successful fetch. before it warms, the proxy verifies every identity token against
                # an EMPTY keyset and rejects fail-closed (TOOL_IDENTITY_UNVERIFIED), so a k8s
                # readiness probe that flipped ready too early would route calls the proxy is
                # guaranteed to fail. gating on is_warmed keeps the registry out of rotation until it
                # can actually verify a token.
                HealthCheck(
                    name="jwks_warmed",
                    probe=lambda: self._jwks_provider is not None and self._jwks_provider.is_warmed,
                    tier=HealthTier.READY,
                ),
            ],
        )
        await health_server.start()
        self._health_server = health_server

    def _install_signal_handlers(self) -> None:
        """install SIGINT and SIGTERM handlers for graceful shutdown."""
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self._request_shutdown)

    def _request_shutdown(self) -> None:
        """signal shutdown request from signal handler."""
        _logger.info("shutdown signal received")
        asyncio.ensure_future(self.shutdown())

    async def shutdown(self) -> None:
        """gracefully shut down registry server.

        stops all handlers, drains NATS subscriptions, and
        closes NATS connection.
        """
        _logger.info("shutting down registry server")

        if self._health_server is not None:
            await self._health_server.stop()
        if self._call_proxy is not None:
            await self._call_proxy.stop()
        if self._jwks_provider is not None:
            await self._jwks_provider.stop()
        if self._discovery_handler is not None:
            await self._discovery_handler.stop()
        if self._heartbeat_subscriber is not None:
            await self._heartbeat_subscriber.stop()
        if self._registration_handler is not None:
            await self._registration_handler.stop()

        # BEFORE the connection closes, and not optional. `startup` calls
        # `start_invalidation_listener`, and every other component it starts is stopped
        # above -- this one was skipped, so the subscription outlived the server that owned
        # it. `stop_invalidation_listener` is a no-op when no listener is live, so this is
        # safe on a partial startup, which is why it is unconditional rather than guarded.
        if self._collection_registry is not None:
            await self._collection_registry.stop_invalidation_listener()
        # whatever a factory built and the server never saw -- today the rbac stack's own
        # invalidation subscriptions.
        if self._on_shutdown is not None:
            await self._on_shutdown()

        if self._nc is not None:
            await self._nc.shutdown()

        self._shutdown_event.set()
        _logger.info("registry server stopped")


def _run_server() -> None:
    """create and run registry server in asyncio event loop.

    authorization mode resolution:

    1. ``THREETEARS_REGISTRY_ALLOW_ALL_TOOLS=true`` -> the
       :class:`~threetears.registry.auth.AllowAllAuthorizer`. test
       fixtures and dev sandboxes that intentionally bypass rbac.
    2. otherwise -> the production
       :class:`~threetears.registry.rbac_authorizer.RbacEvaluatorAuthorizer`
       wired against a self-contained
       :class:`~threetears.registry.rbac_stack.RegistryRbacStack`
       (NATS-proxy backed Collections + ACL cache + invalidation
       subscribers). the standalone registry no longer falls back
       to ``DenyAllAuthorizer`` -- the previous fallback existed
       only because the rbac wiring required hub-side loaders we
       could not construct from the standalone entrypoint. now the
       Collections snap a NATS-proxy L3 backend pinned to
       :data:`PLATFORM_RBAC_READ_NAMESPACE` and read through the
       hub broker's read-only carve-out, so the registry can
       authorize tool calls in any deployment that has a reachable
       hub broker (i.e. every real deployment).

    note: the rbac-stack construction is synchronous; the
    invalidation subscriptions are bound asynchronously after the
    registry's NATS connection comes up (see
    :meth:`RegistryServer._start_handlers` -- the rbac stack rides
    the same client). returning ``DenyAllAuthorizer`` only happens
    when the operator explicitly opts in via
    ``THREETEARS_REGISTRY_FORCE_DENY_ALL=true``, which exists
    purely as a panic-button kill switch for misconfigured prod
    deployments.
    """
    from threetears.observe import configure_logging

    configure_logging(level="INFO")

    allow_all = (
        os.environ.get(
            "THREETEARS_REGISTRY_ALLOW_ALL_TOOLS",
            "",
        ).lower()
        == "true"
    )
    force_deny = (
        os.environ.get(
            "THREETEARS_REGISTRY_FORCE_DENY_ALL",
            "",
        ).lower()
        == "true"
    )

    authorizer: AgentToolAuthorizer
    rbac_authorizer_factory: "Callable[[NatsClient], Awaitable[AgentToolAuthorizer]] | None" = None

    if allow_all:
        from threetears.registry.auth import AllowAllAuthorizer

        authorizer = AllowAllAuthorizer()
        _logger.warning(
            "registry running in allow-all mode (THREETEARS_REGISTRY_ALLOW_ALL_TOOLS=true)",
            extra={"extra_data": {"mode": "allow_all"}},
        )
    elif force_deny:
        from threetears.registry.auth import DenyAllAuthorizer

        authorizer = DenyAllAuthorizer()
        _logger.warning(
            "registry running in forced deny-all mode "
            "(THREETEARS_REGISTRY_FORCE_DENY_ALL=true). every tool "
            "dispatch will be denied -- intentional kill-switch.",
            extra={"extra_data": {"mode": "deny_all_forced"}},
        )
    else:
        # default production path: deny-all placeholder *until* the
        # NATS connection comes up + the rbac stack is constructed
        # against it. the placeholder denies any dispatch that races
        # the wiring (a cold-start window of milliseconds); once the
        # rbac stack is live the server swaps in the real authorizer.
        from threetears.registry.auth import DenyAllAuthorizer

        authorizer = DenyAllAuthorizer()

        # populated by the factory below, drained by the server's `on_shutdown`. A list
        # rather than a scalar because the factory runs on every (re)connect.
        built_rbac_stacks: list["RegistryRbacStack"] = []

        async def _rbac_factory(nc: NatsClient) -> AgentToolAuthorizer:
            from threetears.registry.l1_cache import (
                create_registry_l1_backend,
            )
            from threetears.registry.rbac_authorizer import (
                RbacEvaluatorAuthorizer,
            )
            from threetears.registry.rbac_stack import (
                build_registry_rbac_stack,
            )

            namespace = os.environ.get(
                "THREETEARS_NATS_SUBJECT_NAMESPACE",
                "3tears",
            )
            l1_backend = create_registry_l1_backend()
            stack = build_registry_rbac_stack(
                nats_client=nc,
                subject_namespace=namespace,
                l1_backend=l1_backend,
            )
            await stack.subscribe_invalidations()
            # hand the stack to the server's teardown. Without this the subscriptions this
            # call just made outlive the server: the factory returns only the authorizer, so
            # `RegistryRbacStack.close()` had no production caller in this package at all.
            built_rbac_stacks.append(stack)
            _logger.info(
                "registry running with RbacEvaluatorAuthorizer (rbac stack wired against system.platform.rbac proxy)",
                extra={"extra_data": {"mode": "rbac"}},
            )
            return RbacEvaluatorAuthorizer(
                acl_cache=stack.acl_cache,
                namespace_collection=stack.namespace_collection,
            )

        rbac_authorizer_factory = _rbac_factory

    async def _close_rbac_stacks() -> None:
        """release every rbac stack the factory built.

        ``close`` is idempotent and empties its own handle list, so a stack closed twice
        releases nothing the second time.

        :return: nothing
        :rtype: None
        """
        for stack in built_rbac_stacks:
            await stack.close()

    server = RegistryServer(
        authorizer=authorizer,
        rbac_authorizer_factory=rbac_authorizer_factory,
        on_shutdown=_close_rbac_stacks,
        pod_authenticator_factory=_resolve_pod_authenticator_factory(),
        limit_guard_factory=_resolve_limit_guard_factory(),
        usage_emitter_factory=_resolve_usage_emitter_factory(),
    )
    asyncio.run(server.serve())


def _resolve_pod_authenticator_factory() -> "Callable[[NatsClient], Awaitable[ToolPodAuthenticator | None]] | None":
    """resolve the tool-pod REGISTRATION authenticator factory from a configurable plugin path.

    3tears stays host-agnostic: the standalone registry cannot know how a given deployment stores
    its tool-pod identities, so the authenticator factory is supplied out-of-band via
    ``THREETEARS_REGISTRY_POD_AUTHENTICATOR_FACTORY``, a ``module:callable`` dotted path the operator
    points at a host-provided factory (e.g. the aibots Hub's
    ``aibots.hub.tools.registry_auth:pod_authenticator_factory``, which verifies each pod's
    self-minted identity JWT against the pod's stored public key). the referenced object is the async
    factory itself -- ``Callable[[NatsClient], Awaitable[ToolPodAuthenticator | None]]`` -- invoked by
    :meth:`RegistryServer.serve` once NATS is up. UNSET -> ``None`` -> OPEN registration mode (the
    pure-3tears / dev default; nothing to verify against without a host identity store).

    :return: the resolved factory, or ``None`` when the env var is unset
    :rtype: Callable[[NatsClient], Awaitable[ToolPodAuthenticator | None]] | None
    :raises ValueError: when the env var is set but not a ``module:callable`` dotted path
    :raises ImportError / AttributeError: when the path does not resolve (fail loud -- a
        misconfigured authenticator plugin must crash startup, never silently drop to open mode)
    """
    import importlib

    spec = os.environ.get("THREETEARS_REGISTRY_POD_AUTHENTICATOR_FACTORY", "").strip()
    result: "Callable[[NatsClient], Awaitable[ToolPodAuthenticator | None]] | None" = None
    if spec:
        module_name, sep, attr = spec.partition(":")
        if not module_name or not sep or not attr:
            raise ValueError(
                f"THREETEARS_REGISTRY_POD_AUTHENTICATOR_FACTORY must be a 'module:callable' dotted path; got {spec!r}"
            )
        module = importlib.import_module(module_name)
        result = getattr(module, attr)
        _logger.info(
            "registry tool-pod registration authenticator wired (per-key identity verify enabled)",
            extra={"extra_data": {"factory": spec}},
        )
    return result


def _resolve_limit_guard_factory() -> "Callable[[NatsClient], Awaitable[LimitGuard | None]] | None":
    """resolve the pre-call spend-gate factory from a configurable plugin path.

    3tears stays host-agnostic: the standalone registry has no counter backend, so the limit-guard
    factory is supplied out-of-band via ``THREETEARS_REGISTRY_LIMIT_GUARD_FACTORY``, a
    ``module:callable`` dotted path the operator points at a host-provided factory (the aibots Hub's
    ``aibots.hub.tools.registry_auth:limit_guard_factory``, which builds a NATS-proxy-backed
    ``KvCallLimitGuard`` over the usage counters). the referenced object is the async factory itself
    -- ``Callable[[NatsClient], Awaitable[LimitGuard | None]]`` -- invoked by
    :meth:`RegistryServer.serve` once NATS is up. UNSET -> ``None`` -> the constructor default
    ``AllowAllLimitGuard`` (the pure-3tears / dev default; no counter store to enforce against).

    :return: the resolved factory, or ``None`` when the env var is unset
    :rtype: Callable[[NatsClient], Awaitable[LimitGuard | None]] | None
    :raises ValueError: when the env var is set but not a ``module:callable`` dotted path
    :raises ImportError / AttributeError: when the path does not resolve (fail loud -- a
        misconfigured limit-guard plugin must crash startup, never silently drop to allow-all)
    """
    import importlib

    spec = os.environ.get("THREETEARS_REGISTRY_LIMIT_GUARD_FACTORY", "").strip()
    result: "Callable[[NatsClient], Awaitable[LimitGuard | None]] | None" = None
    if spec:
        module_name, sep, attr = spec.partition(":")
        if not module_name or not sep or not attr:
            raise ValueError(
                f"THREETEARS_REGISTRY_LIMIT_GUARD_FACTORY must be a 'module:callable' dotted path; got {spec!r}"
            )
        module = importlib.import_module(module_name)
        result = getattr(module, attr)
        _logger.info(
            "registry pre-call limit guard wired (spend enforcement enabled)",
            extra={"extra_data": {"factory": spec}},
        )
    return result


def _resolve_usage_emitter_factory() -> "Callable[[NatsClient], Awaitable[EndpointUsageEmitter | None]] | None":
    """resolve the post-call endpoint-usage emitter factory from a configurable plugin path.

    3tears stays host-agnostic: the standalone registry has no metering bus, so the usage emitter is
    supplied out-of-band via ``THREETEARS_REGISTRY_USAGE_EMITTER_FACTORY``, a ``module:callable``
    dotted path the operator points at a host-provided factory (the aibots Hub's factory that
    publishes an ``EndpointUsageEvent`` on the metering subject gu-task-16 consumes). the referenced
    object is the async factory itself -- ``Callable[[NatsClient], Awaitable[EndpointUsageEmitter |
    None]]`` -- invoked by :meth:`RegistryServer.serve` once NATS is up. UNSET -> ``None`` -> no
    per-call usage emit (the pure-3tears / dev default; no metering pipeline to publish to).

    :return: the resolved factory, or ``None`` when the env var is unset
    :rtype: Callable[[NatsClient], Awaitable[EndpointUsageEmitter | None]] | None
    :raises ValueError: when the env var is set but not a ``module:callable`` dotted path
    :raises ImportError / AttributeError: when the path does not resolve (fail loud -- a
        misconfigured emitter plugin must crash startup, never silently drop usage events)
    """
    import importlib

    spec = os.environ.get("THREETEARS_REGISTRY_USAGE_EMITTER_FACTORY", "").strip()
    result: "Callable[[NatsClient], Awaitable[EndpointUsageEmitter | None]] | None" = None
    if spec:
        module_name, sep, attr = spec.partition(":")
        if not module_name or not sep or not attr:
            raise ValueError(
                f"THREETEARS_REGISTRY_USAGE_EMITTER_FACTORY must be a 'module:callable' dotted path; got {spec!r}"
            )
        module = importlib.import_module(module_name)
        result = getattr(module, attr)
        _logger.info(
            "registry endpoint-usage emitter wired (per-call usage events enabled)",
            extra={"extra_data": {"factory": spec}},
        )
    return result
