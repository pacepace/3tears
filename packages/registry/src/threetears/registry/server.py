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

from nats.aio.client import Client as NatsClient

from threetears.observe import get_logger
from threetears.observe.resilience import retry_with_backoff
from threetears.registry.catalog import ToolCatalog
from threetears.registry.discovery import DiscoveryHandler
from threetears.registry.health import HeartbeatMonitor
from threetears.registry.proxy import CallProxy
from threetears.registry.auth import AgentToolAuthorizer
from threetears.registry.registration import RegistrationHandler

_logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# NATS connection helper (patched in tests)
# ---------------------------------------------------------------------------


async def nats_connect(url: str) -> NatsClient:
    """connect to NATS server at given URL with reconnection support.

    configures infinite reconnect attempts with 2-second wait between
    attempts so registry survives NATS infrastructure restarts.

    :param url: NATS server URL
    :ptype url: str
    :return: connected NATS client
    :rtype: NatsClient
    """
    nc = NatsClient()
    await nc.connect(
        url,
        max_reconnect_attempts=-1,
        reconnect_time_wait=2,
        reconnected_cb=_on_reconnected,
        disconnected_cb=_on_disconnected,
        error_cb=_on_error,
    )
    return nc


async def _on_reconnected() -> None:
    """log NATS reconnection event.

    :return: nothing
    :rtype: None
    """
    _logger.info("NATS reconnected")


async def _on_disconnected() -> None:
    """log NATS disconnection event.

    :return: nothing
    :rtype: None
    """
    _logger.warning("NATS disconnected, attempting reconnect")


async def _on_error(exc: Exception) -> None:
    """log NATS client error.

    :param exc: exception from NATS client
    :ptype exc: Exception
    :return: nothing
    :rtype: None
    """
    _logger.error("NATS error: %s", exc)


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
        nats_url: str | None = None,
        namespace: str | None = None,
        heartbeat_check_interval: float | None = None,
        heartbeat_timeout: float | None = None,
        call_timeout: float | None = None,
        kv_bucket: str = "tool_catalog",
        authorizer: AgentToolAuthorizer | None = None,
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
        :param authorizer: tool access authorizer for agent call verification (defaults to None, resolved by _run_server)
        :ptype authorizer: AgentToolAuthorizer | None
        """
        from threetears.registry.config import get_call_timeout, get_heartbeat_check_interval, get_heartbeat_timeout

        self._nats_url = nats_url or os.environ.get(
            "THREETEARS_NATS_URL", "nats://localhost:4222",
        )
        self._namespace = namespace or os.environ.get(
            "FOURTEENAIBOTS_NATS_SUBJECT_NAMESPACE", "aibots",
        )
        self._heartbeat_check_interval = heartbeat_check_interval if heartbeat_check_interval is not None else get_heartbeat_check_interval()
        self._heartbeat_timeout = heartbeat_timeout if heartbeat_timeout is not None else get_heartbeat_timeout()
        self._call_timeout = call_timeout if call_timeout is not None else get_call_timeout()
        self._kv_bucket = kv_bucket
        self._authorizer = authorizer
        self._nc: NatsClient | None = None
        self._catalog = ToolCatalog()
        self._registration_handler: RegistrationHandler | None = None
        self._heartbeat_monitor: HeartbeatMonitor | None = None
        self._discovery_handler: DiscoveryHandler | None = None
        self._call_proxy: CallProxy | None = None
        self._shutdown_event = asyncio.Event()

    async def serve(self) -> None:
        """start registry server and wait for shutdown signal.

        connects to NATS, loads catalog from KV, starts all
        handlers, installs signal handlers, and blocks until
        shutdown is requested.
        """
        self._nc = await nats_connect(self._nats_url)
        _logger.info(
            "connected to NATS",
            extra={"extra_data": {"nats_url": self._nats_url}},
        )

        js = self._nc.jetstream()

        if hasattr(self._authorizer, "initialize"):
            async def _initialize_authorizer() -> None:
                """initialize authorizer with JetStream context."""
                await self._authorizer.initialize(js, self._namespace)

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
        discovery handler, and call proxy.
        """
        self._registration_handler = RegistrationHandler(
            self._catalog, namespace=self._namespace,
        )
        await retry_with_backoff(
            lambda: self._registration_handler.start(self._nc),
            "registry.registration_handler.start",
        )

        self._heartbeat_monitor = HeartbeatMonitor(
            self._catalog,
            namespace=self._namespace,
            check_interval=self._heartbeat_check_interval,
            timeout=self._heartbeat_timeout,
        )
        await retry_with_backoff(
            lambda: self._heartbeat_monitor.start(self._nc),
            "registry.heartbeat_monitor.start",
        )

        self._discovery_handler = DiscoveryHandler(
            self._catalog, namespace=self._namespace,
        )
        await retry_with_backoff(
            lambda: self._discovery_handler.start(self._nc),
            "registry.discovery_handler.start",
        )

        self._call_proxy = CallProxy(
            self._catalog,
            namespace=self._namespace,
            timeout=self._call_timeout,
            authorizer=self._authorizer,
        )
        await retry_with_backoff(
            lambda: self._call_proxy.start(self._nc),
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
        if self._heartbeat_monitor is not None:
            await self._heartbeat_monitor.stop()
        if self._registration_handler is not None:
            await self._registration_handler.stop()

        if self._nc is not None:
            await self._nc.drain()
            await self._nc.close()

        self._shutdown_event.set()
        _logger.info("registry server stopped")


def _run_server() -> None:
    """create and run registry server in asyncio event loop.

    reads FOURTEENAIBOTS_REGISTRY_ALLOW_ALL_TOOLS environment variable
    to determine authorization mode. when set to "true", all tool calls
    are permitted (development mode). otherwise, default-deny is enforced.
    host applications should pass a custom AgentToolAuthorizer for
    production use.
    """
    from threetears.observe import configure_logging

    configure_logging(level="INFO")

    allow_all = os.environ.get(
        "FOURTEENAIBOTS_REGISTRY_ALLOW_ALL_TOOLS", "",
    ).lower() == "true"

    if allow_all:
        from threetears.registry.auth import AllowAllAuthorizer

        authorizer: AgentToolAuthorizer = AllowAllAuthorizer()
        _logger.warning(
            "registry running in allow-all mode (FOURTEENAIBOTS_REGISTRY_ALLOW_ALL_TOOLS=true)",
            extra={"extra_data": {"mode": "allow_all"}},
        )
    else:
        from threetears.registry.auth import KvAgentToolAuthorizer

        authorizer = KvAgentToolAuthorizer()
        _logger.info(
            "registry running with KV-based tool authorization",
            extra={"extra_data": {"mode": "kv_authorizer"}},
        )

    server = RegistryServer(authorizer=authorizer)
    asyncio.run(server.serve())
