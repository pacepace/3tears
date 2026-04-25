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

from threetears.core.collections.registry import CollectionRegistry
from threetears.core.config import DefaultCoreConfig
from threetears.nats import NatsClient
from threetears.observe import get_logger
from threetears.observe.resilience import retry_with_backoff
from threetears.registry.catalog import ToolCatalog
from threetears.registry.discovery import DiscoveryHandler
from threetears.registry.health import HeartbeatSubscriber
from threetears.registry.heartbeat_collection import HeartbeatCollection
from threetears.registry.l1_cache import create_registry_l1_backend
from threetears.registry.proxy import CallProxy
from threetears.registry.auth import AgentToolAuthorizer
from threetears.registry.registration import RegistrationHandler

__all__ = [
    "RegistryServer",
    "nats_connect",
]

_logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# NATS connection helper (patched in tests)
# ---------------------------------------------------------------------------


async def nats_connect(url: str, *, namespace: str = "aibots") -> NatsClient:
    """connect to NATS server via the canonical :class:`NatsClient` wrapper.

    delegates to :meth:`NatsClient.connect` which handles dual-phase
    reconnect, rate-limited error logging, and namespace binding on
    :class:`Subjects`. tests patch this symbol to swap a fake
    transport into :class:`RegistryServer.serve`.

    :param url: NATS server URL
    :ptype url: str
    :param namespace: NATS subject namespace prefix bound on the wrapper
    :ptype namespace: str
    :return: connected canonical wrapper client
    :rtype: NatsClient
    """
    return await NatsClient.connect(
        nats_url=url,
        nats_subject_namespace=namespace,
        client_name="registry",
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
    ) -> None:
        """initialize registry server.

        :param nats_url: NATS server URL (defaults to THREETEARS_NATS_URL env var)
        :ptype nats_url: str | None
        :param namespace: NATS subject namespace prefix (defaults to FOURTEENAIBOTS_NATS_SUBJECT_NAMESPACE env var)
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
        :param authorizer: tool access authorizer for agent call
            verification; REQUIRED. callers wire the production
            :class:`RbacEvaluatorAuthorizer` (hub path) or one of the
            deterministic stubs (:class:`AllowAllAuthorizer`,
            :class:`DenyAllAuthorizer`) for dev/test paths; no
            silent-bypass default remains
        :ptype authorizer: AgentToolAuthorizer
        """
        from threetears.registry.config import get_call_timeout, get_heartbeat_check_interval, get_heartbeat_timeout

        self._nats_url = nats_url or os.environ.get(
            "THREETEARS_NATS_URL",
            "nats://localhost:4222",
        )
        self._namespace = namespace or os.environ.get(
            "FOURTEENAIBOTS_NATS_SUBJECT_NAMESPACE",
            "aibots",
        )
        self._heartbeat_check_interval = (
            heartbeat_check_interval if heartbeat_check_interval is not None else get_heartbeat_check_interval()
        )
        self._heartbeat_timeout = heartbeat_timeout if heartbeat_timeout is not None else get_heartbeat_timeout()
        self._call_timeout = call_timeout if call_timeout is not None else get_call_timeout()
        self._kv_bucket = kv_bucket
        self._authorizer = authorizer
        self._nc: "NatsClient | None" = None
        self._catalog = ToolCatalog()
        self._collection_registry: CollectionRegistry | None = None
        self._heartbeat_collection: HeartbeatCollection | None = None
        self._registration_handler: RegistrationHandler | None = None
        self._heartbeat_subscriber: HeartbeatSubscriber | None = None
        self._discovery_handler: DiscoveryHandler | None = None
        self._call_proxy: CallProxy | None = None
        self._shutdown_event = asyncio.Event()

    async def serve(self) -> None:
        """start registry server and wait for shutdown signal.

        connects to NATS, loads catalog from KV, starts all
        handlers, installs signal handlers, and blocks until
        shutdown is requested.
        """
        self._nc = await nats_connect(self._nats_url, namespace=self._namespace)
        _logger.info(
            "connected to NATS",
            extra={"extra_data": {"nats_url": self._nats_url}},
        )

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

        await self._start_handlers()

        self._install_signal_handlers()

        _logger.info(
            "registry server started",
            extra={"extra_data": {"namespace": self._namespace}},
        )

        await self._shutdown_event.wait()

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
        collection_registry = CollectionRegistry()
        collection_registry.configure(l1_backend=l1_backend, l2_client=nc)
        core_config = DefaultCoreConfig(
            collection_flush="ALWAYS",
            collection_flush_tables="",
        )
        heartbeat_collection = HeartbeatCollection(
            collection_registry,
            core_config,
            nats_client=nc,
        )
        self._collection_registry = collection_registry
        self._heartbeat_collection = heartbeat_collection
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

        call_proxy = CallProxy(
            self._catalog,
            namespace=self._namespace,
            timeout=self._call_timeout,
            authorizer=self._authorizer,
        )
        self._call_proxy = call_proxy
        await retry_with_backoff(
            lambda: call_proxy.start(nc),
            "registry.call_proxy.start",
        )

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

        if self._call_proxy is not None:
            await self._call_proxy.stop()
        if self._discovery_handler is not None:
            await self._discovery_handler.stop()
        if self._heartbeat_subscriber is not None:
            await self._heartbeat_subscriber.stop()
        if self._registration_handler is not None:
            await self._registration_handler.stop()

        if self._nc is not None:
            await self._nc.shutdown()

        self._shutdown_event.set()
        _logger.info("registry server stopped")


def _run_server() -> None:
    """create and run registry server in asyncio event loop.

    reads FOURTEENAIBOTS_REGISTRY_ALLOW_ALL_TOOLS environment variable
    to determine authorization mode. when set to "true", all tool calls
    are permitted (development mode). otherwise the registry starts
    with :class:`~threetears.registry.auth.DenyAllAuthorizer` as a
    hard-deny placeholder — production deployments wire the real
    :class:`~threetears.registry.rbac_authorizer.RbacEvaluatorAuthorizer`
    programmatically (see the hub's registry startup in
    :mod:`aibots.hub.app`) because it requires loaders that depend on
    the hub's DB pool. running this module directly without
    ``ALLOW_ALL_TOOLS`` will refuse every dispatch — intentional so
    a mis-wired deployment surfaces as a hard failure rather than
    silent allow-all.
    """
    from threetears.observe import configure_logging

    configure_logging(level="INFO")

    allow_all = (
        os.environ.get(
            "FOURTEENAIBOTS_REGISTRY_ALLOW_ALL_TOOLS",
            "",
        ).lower()
        == "true"
    )

    if allow_all:
        from threetears.registry.auth import AllowAllAuthorizer

        authorizer: AgentToolAuthorizer = AllowAllAuthorizer()
        _logger.warning(
            "registry running in allow-all mode (FOURTEENAIBOTS_REGISTRY_ALLOW_ALL_TOOLS=true)",
            extra={"extra_data": {"mode": "allow_all"}},
        )
    else:
        from threetears.registry.auth import DenyAllAuthorizer

        authorizer = DenyAllAuthorizer()
        _logger.warning(
            "registry standalone entry-point: no RbacEvaluatorAuthorizer wired, "
            "running in deny-all mode. production deployments construct the "
            "authorizer programmatically with hub-side loaders.",
            extra={"extra_data": {"mode": "deny_all_placeholder"}},
        )

    server = RegistryServer(authorizer=authorizer)
    asyncio.run(server.serve())
