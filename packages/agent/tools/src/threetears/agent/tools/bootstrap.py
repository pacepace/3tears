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

it also owns the *terminal* half of that lifecycle. a permanent config
fault and a transient crash both used to exit ``1``, and a supervisor
cannot tell them apart, so ``restart: unless-stopped`` restarted a pod
that could never start. :class:`ToolPodConfigError` + :data:`EX_CONFIG`
give the supervisor something to branch on, and give the observability
pipeline one record instead of a traceback per restart.
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
    HealthTier,
    configure_logging,
    get_logger,
    spawn_background,
)

if TYPE_CHECKING:
    from threetears.agent.tools.server import ToolServer

__all__ = ["EX_CONFIG", "ToolPodConfigError", "ToolServerBootstrap"]

log = get_logger(__name__)

#: exit status for a permanent, operator-fixable configuration fault: ``EX_CONFIG``
#: from BSD ``sysexits.h``.
#:
#: a supervisor can branch on a STATUS and on nothing else. it never reads the
#: message, so a config fault that exits ``1`` is, to docker's
#: ``restart: unless-stopped`` or a k8s ``restartPolicy``, the same event as a
#: transient crash, and gets the same answer: restart. bluelabsio/14-eng-ai-bot#235
#: is what that costs -- a tool pod started with a renamed identity env var
#: restarted 9,580 times over 8 days, never once reaching ``running``, writing
#: 278,110 byte-identical traceback lines. nothing alerted: a healthcheck cannot
#: fire on a container that never starts, and nothing declared ``depends_on`` it,
#: so the only signal was a ``RestartCount`` no one reads.
#:
#: 78 is the conventional "the configuration is wrong, do not try again" status.
#: being conventional is the point: systemd's ``RestartPreventExitStatus``, a k8s
#: exit-code alert, and an operator reading ``docker inspect`` all classify it with
#: no agreement needed between them and this module.
EX_CONFIG = 78


class ToolPodConfigError(ValueError):
    """a tool pod's configuration is permanently wrong; restarting cannot fix it.

    raise this from a ``build_server`` (or any startup-config validation) when a
    required environment variable is missing, renamed, or malformed.
    :meth:`ToolServerBootstrap.run` catches exactly this type, logs one structured
    ERROR naming ``variable``, and exits :data:`EX_CONFIG`.

    it subclasses :class:`ValueError` deliberately. the config-validation sites it
    replaces raised bare ``ValueError``, so every existing ``except ValueError``
    caller and test keeps working unchanged. the reason it is a distinct type
    rather than the bare ``ValueError`` the incident report proposed catching: a
    ``ValueError`` raised from deep inside a tool's own business logic during
    startup is a bug, not a config fault, and catching the base type would make it
    terminal -- turning a pod that a restart might have recovered into one that
    stays down. the same subclass-for-compatibility shape is already used by
    :class:`threetears.core.security.secret_refs.SecretResolutionError`.

    :param message: operator-facing description of what is wrong and how to fix it
    :ptype message: str
    :param variable: name of the environment variable at fault. required, because
        an operator reading one ERROR line needs the thing to go change, and a
        message that only says "config is invalid" sends them to the source
    :ptype variable: str
    """

    def __init__(self, message: str, *, variable: str) -> None:
        """record the offending variable alongside the message.

        :param message: operator-facing description of the fault
        :ptype message: str
        :param variable: environment variable name at fault
        :ptype variable: str
        :return: None
        :rtype: None
        """
        super().__init__(message)
        self.variable = variable


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

        a :class:`ToolPodConfigError` terminates here: one structured ERROR
        record, then :data:`EX_CONFIG`. every other failure propagates
        untouched, exits ``1``, and stays retryable -- a NATS server that is
        not up yet is exactly the case a supervisor restart exists for.

        the catch lives here and NOT in :meth:`run_async` because this is the
        method that owns the process exit status. a caller that drives
        ``run_async`` on its own loop owns its own failure policy and must be
        free to handle the error rather than have the library exit under it.

        :return: None
        :rtype: None
        :raises SystemExit: with :data:`EX_CONFIG` when startup config is
            permanently wrong
        """
        configure_logging(level=self._log_level)
        try:
            asyncio.run(self.run_async())
        except ToolPodConfigError as exc:
            # ONE record, through the repo logger, so this reaches the observability
            # pipeline as a queryable event rather than as container stderr nobody
            # tails. `from None` drops the traceback: 278,110 lines of it is what the
            # incident actually produced, and none of them said anything this one
            # record does not.
            log.error(
                f"{self._service_name} configuration is invalid; exiting without retry",
                extra={
                    "extra_data": {
                        "service": self._service_name,
                        "variable": exc.variable,
                        "error": str(exc),
                        "exit_code": EX_CONFIG,
                    }
                },
            )
            raise SystemExit(EX_CONFIG) from None

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
            # serve the pod's in-flight-requests gauge on /metrics so KEDA's
            # prometheus scaler can autoscale the tool-pod Deployment on
            # aggregate in-flight call load through the one HTTP listener the
            # pod already runs for /healthz.
            metrics_provider=server.render_metrics,
            checks=[
                HealthCheck(
                    # key liveness on REAL NATS health, not object-existence: a
                    # terminal close (user-JWT expiry) or a persistent auth /
                    # overflow wedge trips is_healthy so k8s recycles the pod. the
                    # old is_connected probe reported healthy forever with a dead
                    # connection (the silent-zombie bug). the in-process
                    # heartbeat-loop supervisor is the no-k8s net (host / docker).
                    name="nats",
                    probe=lambda: server.is_healthy,
                    tier=HealthTier.LIVE,
                ),
                # readiness, NOT liveness: a pod that has registered no tools cannot
                # serve a call, but restarting it does not conjure tools -- it just
                # replays the same startup. this check being tier-less is why the
                # tool pods shipped with no livenessProbe at all.
                HealthCheck(
                    name="tools_registered",
                    probe=lambda: server.tools_count > 0,
                    tier=HealthTier.READY,
                ),
                # readiness gate: report NOT-READY until the pod's Hub-JWKS cache has had its first
                # successful fetch. before it warms, the pod verifies every inbound identity token
                # against an EMPTY keyset and rejects fail-closed, so a k8s readiness probe that
                # flipped ready too early would route calls the pod is guaranteed to fail. gating on
                # jwks_warmed keeps the pod out of rotation until it can actually verify a token.
                HealthCheck(
                    name="jwks_warmed",
                    probe=lambda: server.jwks_warmed,
                    tier=HealthTier.READY,
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
