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
import signal
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from threetears.observe import configure_logging, get_logger, spawn_background

if TYPE_CHECKING:
    from threetears.agent.tools.server import ToolServer

__all__ = ["ToolServerBootstrap"]

log = get_logger(__name__)


class ToolServerBootstrap:
    """canonical tool-pod lifecycle: logging, signals, serve loop.

    subclasses customize the build-and-serve pipeline by overriding
    ``_build_server``, ``_register_tools``, and ``_run_serve``. the
    base ``run`` is the one-line entrypoint every tool-pod ``main()``
    calls.

    typical use::

        class MyBootstrap(ToolServerBootstrap):
            def __init__(self, ...): ...
            async def _build_server(self) -> ToolServer: ...
            async def _register_tools(self, server: ToolServer) -> None: ...

        def main() -> None:
            MyBootstrap(...).run()

    :param service_name: short identifier used in log messages and
        signal-handler task names
    :ptype service_name: str
    :param log_level: stdlib logging level name passed to
        ``configure_logging``
    :ptype log_level: str
    """

    def __init__(self, service_name: str, *, log_level: str = "INFO") -> None:
        """initialize bootstrap with service identity and log level.

        :param service_name: short identifier for log messages
        :ptype service_name: str
        :param log_level: log level name (e.g. ``"INFO"``, ``"DEBUG"``)
        :ptype log_level: str
        """
        self._service_name = service_name
        self._log_level = log_level

    def run(self) -> None:
        """entrypoint: configure logging then drive the async serve loop.

        this is the only method tool-pod ``main()`` functions need to
        call. subclasses override async hooks to plug in host-specific
        lifecycle (build dependency, register tools, await serve).

        :return: None
        :rtype: None
        """
        configure_logging(level=self._log_level)
        asyncio.run(self._run_async())

    async def _run_async(self) -> None:
        """async driver: build server, register tools, install signals, serve.

        :return: None
        :rtype: None
        """
        server = await self._build_server()
        await self._register_tools(server)
        self._install_signal_handlers(server)

        log.info(
            f"{self._service_name} starting",
            extra={"extra_data": {
                "service": self._service_name,
                "tools_count": server.tools_count,
            }},
        )
        try:
            await self._run_serve(server)
        finally:
            log.info(
                f"{self._service_name} stopped",
                extra={"extra_data": {"service": self._service_name}},
            )

    async def _build_server(self) -> "ToolServer":
        """build the ``ToolServer`` instance.

        subclasses MUST override to construct the server with
        host-specific configuration (NATS URL, namespace, pod id,
        bootstrap token, namespace collection, etc.).

        :return: configured but unstarted ToolServer
        :rtype: ToolServer
        :raises NotImplementedError: when subclass does not override
        """
        raise NotImplementedError("subclasses must override _build_server")

    async def _register_tools(self, server: "ToolServer") -> None:
        """register all tools on the server before serving.

        subclasses MUST override to call ``server.register(tool)`` for
        each tool the pod exposes.

        :param server: tool server returned by ``_build_server``
        :ptype server: ToolServer
        :return: None
        :rtype: None
        :raises NotImplementedError: when subclass does not override
        """
        raise NotImplementedError("subclasses must override _register_tools")

    async def _run_serve(self, server: "ToolServer") -> None:
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

    def _install_signal_handlers(self, server: "ToolServer") -> None:
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
            handler = self._make_signal_handler(server, sig_label)
            loop.add_signal_handler(sig, handler)

    def _make_signal_handler(
        self, server: "ToolServer", sig_label: str,
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
