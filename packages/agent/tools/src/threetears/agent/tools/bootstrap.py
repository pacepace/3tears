"""``ToolServerBootstrap`` -- shared lifecycle for tool-pod entrypoints.

every tool-pod ``main`` function used to repeat the same scaffolding:
configure logging, instantiate ``ToolServer``, register tools, install
SIGTERM/SIGINT handlers that schedule ``server.shutdown()`` via
``spawn_background``, await ``server.serve()``, log start/stop. three
copies (admin's ``serve.py``, agent SDK's
``devx/schema/tool_server_entry.py``, and ``threetears.agent.tools.serve``)
drifted on small details: signal-handler implementation, log message
format, exception handling around ``serve()``.

this module owns the canonical lifecycle. host applications subclass
``ToolServerBootstrap`` to add lifecycle (e.g. opening / closing a hub
client around ``serve()``); the base class provides everything that is
not host-specific.
"""

from __future__ import annotations

import asyncio
import os
import signal
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from threetears.observe import (
    HealthCheck,
    HealthServer,
    configure_logging,
    get_logger,
    spawn_background,
)

if TYPE_CHECKING:
    from threetears.agent.tools.server import ToolServer

__all__ = ["ToolServerBootstrap"]

log = get_logger(__name__)


class ToolServerBootstrap:
    """canonical tool-pod lifecycle: logging, signals, serve loop.

    subclasses customize the build-and-serve pipeline by overriding
    ``build_server``, ``register_tools``, and ``run_serve``. the
    base ``run`` is the one-line entrypoint every tool-pod ``main()``
    calls.

    typical use::

        class MyBootstrap(ToolServerBootstrap):
            def __init__(self, ...): ...
            async def build_server(self) -> ToolServer: ...
            async def register_tools(self, server: ToolServer) -> None: ...

        def main() -> None:
            MyBootstrap(...).run()

    :param service_name: short identifier used in log messages and
        signal-handler task names
    :ptype service_name: str
    :param log_level: stdlib logging level name passed to
        ``configure_logging``
    :ptype log_level: str
    """

    def __init__(
        self,
        service_name: str,
        *,
        log_level: str = "INFO",
        health_port: int | None = None,
    ) -> None:
        """initialize bootstrap with service identity and log level.

        :param service_name: short identifier for log messages
        :ptype service_name: str
        :param log_level: log level name (e.g. ``"INFO"``, ``"DEBUG"``)
        :ptype log_level: str
        :param health_port: port the readiness HealthServer binds to;
            defaults to THREETEARS_TOOL_SERVER_HEALTH_PORT env var,
            falling back to 8000. each container in the platform's
            docker stack owns its own port namespace so 8000 is fine
            there; honcho-driven dev runs every Procfile entry in the
            host namespace and the hub uvicorn already binds host:8000,
            so honcho callers MUST set THREETEARS_TOOL_SERVER_HEALTH_PORT
            to a distinct port. bind failures are swallowed at startup
            (the tool pod's primary job is NATS, not health probing),
            so a collision degrades to "no /healthz" rather than aborting.
        :ptype health_port: int | None
        """
        self._service_name = service_name
        self._log_level = log_level
        if health_port is not None:
            self._health_port = health_port
        else:
            env_port = os.environ.get("THREETEARS_TOOL_SERVER_HEALTH_PORT")
            self._health_port = int(env_port) if env_port else 8000

    def run(self) -> None:
        """entrypoint: configure logging then drive the async serve loop.

        this is the only method tool-pod ``main()`` functions need to
        call. subclasses override async hooks to plug in host-specific
        lifecycle (build dependency, register tools, await serve).

        :return: None
        :rtype: None
        """
        configure_logging(level=self._log_level)
        asyncio.run(self.run_async())

    async def run_async(self) -> None:
        """async driver: build server, register tools, install signals, serve.

        public async entrypoint suitable for tests and callers that
        already own the event loop. ``run`` is the sync convenience
        wrapper that drives this through ``asyncio.run``.

        also starts a canonical :class:`HealthServer` on port 8000
        so docker / k8s liveness probes can verify the pod is alive
        + connected to NATS without a custom per-pod health
        endpoint.

        :return: None
        :rtype: None
        """
        server = await self.build_server()
        await self.register_tools(server)
        self.install_signal_handlers(server)

        health_server = await self._start_health_server(server)

        log.info(
            f"{self._service_name} starting",
            extra={
                "extra_data": {
                    "service": self._service_name,
                    "tools_count": server.tools_count,
                }
            },
        )
        try:
            await self.run_serve(server)
        finally:
            if health_server is not None:
                try:
                    await health_server.stop()
                except Exception as exc:
                    log.warning(
                        "health server stop failed",
                        extra={"extra_data": {"error": str(exc)}},
                    )
            log.info(
                f"{self._service_name} stopped",
                extra={"extra_data": {"service": self._service_name}},
            )

    async def _start_health_server(self, server: "ToolServer") -> "HealthServer | None":
        """start the canonical /healthz listener on port 8000.

        port 8000 matches the inherited 3tears-hub Dockerfile
        HEALTHCHECK so docker / k8s liveness probes work for every
        consumer of that base image without per-service compose
        overrides. failures here are logged + swallowed -- the tool
        pod's primary job is NATS request/reply, not health probing,
        so a port-bind collision (multiple pods on the same docker
        network competing for 8000) must not abort startup.

        :param server: tool server whose state the checks read from
        :ptype server: ToolServer
        :return: started :class:`HealthServer`, or ``None`` if the
            listener failed to bind
        :rtype: HealthServer | None
        """
        health_server = HealthServer(
            port=self._health_port,
            service_name=self._service_name,
            checks=[
                HealthCheck(
                    name="nats",
                    probe=lambda: server.is_connected,
                ),
                HealthCheck(
                    name="tools_registered",
                    probe=lambda: server.tools_count > 0,
                ),
            ],
        )
        try:
            await health_server.start()
        except Exception as exc:
            log.warning(
                "health server failed to start; tool pod will run without /healthz",
                extra={
                    "extra_data": {
                        "service": self._service_name,
                        "error": str(exc),
                    }
                },
            )
            return None
        return health_server

    async def build_server(self) -> "ToolServer":
        """build the ``ToolServer`` instance.

        subclasses MUST override to construct the server with
        host-specific configuration (NATS URL, namespace, pod id,
        bootstrap token, namespace collection, etc.).

        :return: configured but unstarted ToolServer
        :rtype: ToolServer
        :raises NotImplementedError: when subclass does not override
        """
        raise NotImplementedError("subclasses must override build_server")

    async def register_tools(self, server: "ToolServer") -> None:
        """register all tools on the server before serving.

        subclasses MUST override to call ``server.register(tool)`` for
        each tool the pod exposes.

        :param server: tool server returned by ``build_server``
        :ptype server: ToolServer
        :return: None
        :rtype: None
        :raises NotImplementedError: when subclass does not override
        """
        raise NotImplementedError("subclasses must override register_tools")

    async def run_serve(self, server: "ToolServer") -> None:
        """await ``server.serve()`` until shutdown completes.

        default implementation calls ``server.serve()`` directly.
        subclasses with additional lifecycle (e.g. holding a hub
        client open across the serve loop) override this hook.

        :param server: tool server with tools registered and signals installed
        :ptype server: ToolServer
        :return: None
        :rtype: None
        """
        await server.serve()

    def install_signal_handlers(self, server: "ToolServer") -> None:
        """install SIGTERM and SIGINT handlers that schedule shutdown.

        each handler spawns ``server.shutdown()`` via ``spawn_background``
        so the coroutine outcome lands in the structured logger rather
        than the default loop's exception printer.

        :param server: tool server whose ``shutdown`` coroutine is scheduled
        :ptype server: ToolServer
        :return: None
        :rtype: None
        """
        loop = asyncio.get_running_loop()
        for sig, sig_label in ((signal.SIGTERM, "sigterm"), (signal.SIGINT, "sigint")):
            handler = self.make_signal_handler(server, sig_label)
            loop.add_signal_handler(sig, handler)

    def make_signal_handler(
        self,
        server: "ToolServer",
        sig_label: str,
    ) -> Callable[[], Awaitable[None] | None]:
        """build the signal-handler closure for ``sig_label``.

        :param server: tool server whose ``shutdown`` will be scheduled
        :ptype server: ToolServer
        :param sig_label: short label used in the spawned task name
        :ptype sig_label: str
        :return: signal-handler callable suitable for ``loop.add_signal_handler``
        :rtype: Callable[[], Awaitable[None] | None]
        """
        task_name = f"{self._service_name}-shutdown-{sig_label}"

        def _handler() -> None:
            spawn_background(server.shutdown(), name=task_name, logger=log)

        return _handler
