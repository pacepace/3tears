"""``HealthServer`` -- minimal asyncio HTTP server for /healthz endpoints.

every long-running service in the 3tears + 3tears stack exposes a
``HealthServer``. NATS-only services (3tears registry, 3tears
agent-tools serve, 3tears agent-router, datasource tool pods, agent
pods, admin tool server) need an HTTP liveness endpoint that
container orchestrators (docker compose, kubernetes) can probe; HTTP
services (hub, gateway) get the same surface so operators have one
canonical /healthz contract across the entire platform.

contract:

- ``GET /healthz`` -- plain-text body, ``200 OK`` when every check
  passes, ``503 Service Unavailable`` with the failing check name
  in the body otherwise. this is the shape docker / k8s liveness
  probes expect.
- ``GET /healthz?format=json`` (or ``Accept: application/json``) --
  structured :class:`HealthStatus` JSON with per-component
  ``healthy`` + ``detail``. operators and CLI tooling read this for
  "which subsystem is down" without grepping logs.
- ``GET /readyz`` -- same as ``/healthz`` (k8s readiness probe alias).
- ``GET /metrics`` -- prometheus text exposition, served only when a
  ``metrics_provider`` callable is wired at construction (returns
  ``(content_type, body)``). NATS-only RPC pods (registry, tool pods)
  have no HTTP framework of their own, so this route is how their
  in-flight-requests gauge becomes scrapable by KEDA's prometheus
  scaler. absent a provider the route returns ``404`` -- the health
  surface stays exactly as it was for services that expose metrics
  elsewhere (hub, gateway run their own /metrics).
- :meth:`HealthServer.get_status` -- in-process accessor returning
  the same :class:`HealthStatus` value the JSON endpoint would
  serialize. consumers wired into the same event loop (e.g. an
  in-process admin endpoint, an integration test) can call it
  directly without round-tripping HTTP.

design choices:

- standard library only (no aiohttp / fastapi dep) so the module is
  consumable from every 3tears package without a transitive install.
  ``asyncio.start_server`` + a hand-rolled HTTP/1.1 frame are
  sufficient -- the surface is fixed (two endpoints, two response
  formats) so a real HTTP framework is overkill.
- liveness checks are a list of ``Callable[[], bool]`` registered at
  construction (or via :meth:`register_check`); each runs
  synchronously per probe with no caching. checks must be cheap
  (cached flag, local connection state); slow / network-bound checks
  belong on a background poll the check reads from.
- the server runs in the same event loop as the service. no
  background thread, no synchronization seam, no startup race
  between the listener and whatever produces the check state.

usage::

    from threetears.observe.health import HealthCheck, HealthServer

    server = HealthServer(
        port=8000,
        service_name="registry",
        checks=[
            HealthCheck(name="nats", probe=lambda: nats_client.is_connected),
            HealthCheck(name="catalog", probe=lambda: catalog.is_ready),
        ],
    )
    await server.start()
    # ... service runs ...
    # in-process status read (no HTTP round-trip):
    status = server.get_status()
    assert status.healthy
    # ... shutdown ...
    await server.stop()
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict, dataclass, field
from typing import Callable

MetricsProvider = Callable[[], tuple[str, bytes]]

from threetears.observe.logging import get_logger

__all__ = ["ComponentStatus", "HealthCheck", "HealthServer", "HealthStatus"]


log = get_logger(__name__)


@dataclass(frozen=True)
class HealthCheck:
    """one liveness check the :class:`HealthServer` evaluates per probe.

    :param name: short identifier the failure response includes (so
        operators see which check tripped without inspecting logs)
    :ptype name: str
    :param probe: zero-arg callable returning ``True`` when the
        underlying state is healthy. expected to be cheap (reads a
        cached flag, polls a connection's local state) -- the
        ``HealthServer`` calls it on every probe with no caching
    :ptype probe: Callable[[], bool]
    """

    name: str
    probe: Callable[[], bool]


@dataclass(frozen=True)
class ComponentStatus:
    """structured status for one component.

    serializes onto the JSON :class:`HealthStatus` payload so
    operators reading ``GET /healthz?format=json`` see exactly which
    subsystem reports unhealthy and (when the probe surfaces detail)
    why.

    :param name: component identifier (matches the
        :class:`HealthCheck` name)
    :ptype name: str
    :param healthy: True when the probe returned ``True``
    :ptype healthy: bool
    :param detail: optional human-readable reason; populated only
        when the probe raised (the exception message lands here)
    :ptype detail: str | None
    """

    name: str
    healthy: bool
    detail: str | None = None


@dataclass(frozen=True)
class HealthStatus:
    """aggregate health status the JSON endpoint serializes.

    :param service: service name (e.g. ``"registry"``,
        ``"agent-router"``); included so a multi-service log scrape
        can attribute the status without parsing the URL
    :ptype service: str
    :param healthy: ``True`` iff every component is healthy
    :ptype healthy: bool
    :param components: per-component status; ordered as the checks
        were registered (so the failing-check short-circuit yields a
        partial list when any check fails -- a downstream check's
        absence from the list means "we never got that far")
    :ptype components: list[ComponentStatus]
    """

    service: str
    healthy: bool
    components: list[ComponentStatus] = field(default_factory=list)


class HealthServer:
    """minimal asyncio HTTP server serving ``GET /healthz`` and
    ``GET /readyz`` with both plain-text and JSON response formats.

    intentional limitations: routing is limited to ``/healthz``,
    ``/readyz`` and (when a ``metrics_provider`` is wired) ``/metrics``,
    only ``GET``, only the ``format=json`` query argument (also accepts
    ``Accept: application/json``), no chunked transfer, no keep-alive.
    the surface is exactly what docker / kubernetes liveness probes need
    plus the JSON shape operators want for drill-in plus the optional
    prometheus exposition KEDA's scaler scrapes.

    :param port: TCP port to bind on. matches the 3tears-hub
        Dockerfile's HEALTHCHECK port (8000) so the inherited
        check works without compose overrides
    :ptype port: int
    :param service_name: short identifier echoed onto the
        :class:`HealthStatus` JSON body (e.g. ``"registry"``,
        ``"agent-router"``)
    :ptype service_name: str
    :param checks: liveness checks evaluated on every probe.
        ``200`` iff every check returns ``True``; ``503`` with the
        name of the first failing check in the body otherwise.
        additional checks can be appended at runtime via
        :meth:`register_check`
    :ptype checks: list[HealthCheck]
    :param host: bind interface; default ``0.0.0.0`` so the
        container's external port mapping reaches the listener
    :ptype host: str
    :param metrics_provider: optional zero-arg callable returning
        ``(content_type, body)`` for the ``GET /metrics`` route. wired by
        NATS-only RPC pods (registry, tool pods) to expose their
        in-flight-requests gauge to KEDA's prometheus scaler through the
        one HTTP listener they already run for ``/healthz``. ``None``
        leaves ``/metrics`` returning ``404`` (the default for services
        that expose prometheus elsewhere)
    :ptype metrics_provider: Callable[[], tuple[str, bytes]] | None
    """

    def __init__(
        self,
        *,
        port: int,
        service_name: str,
        checks: list[HealthCheck] | None = None,
        host: str = "0.0.0.0",
        metrics_provider: MetricsProvider | None = None,
    ) -> None:
        """initialize health server with the supplied checks.

        :param port: TCP port to bind on
        :ptype port: int
        :param service_name: identifier echoed onto status JSON
        :ptype service_name: str
        :param checks: liveness check list (may be empty; add later
            via :meth:`register_check`)
        :ptype checks: list[HealthCheck] | None
        :param host: bind interface
        :ptype host: str
        :param metrics_provider: optional ``() -> (content_type, body)``
            callable served on ``GET /metrics``; ``None`` -> route 404s
        :ptype metrics_provider: Callable[[], tuple[str, bytes]] | None
        :return: nothing
        :rtype: None
        """
        self._port = port
        self._host = host
        self._service_name = service_name
        self._checks: list[HealthCheck] = list(checks) if checks else []
        self._metrics_provider = metrics_provider
        self._server: asyncio.base_events.Server | None = None

    @property
    def port(self) -> int:
        """return the port the server is configured to bind on."""
        return self._port

    @property
    def service_name(self) -> str:
        """return the service identifier echoed on status responses."""
        return self._service_name

    def register_check(self, check: HealthCheck) -> None:
        """append a check to the list evaluated on every probe.

        services that wire their state lazily (e.g. NATS connection
        comes up after :meth:`start`) can register the corresponding
        check after the listener is live -- the check applies to the
        next probe.

        :param check: liveness check to add
        :ptype check: HealthCheck
        :return: nothing
        :rtype: None
        """
        self._checks.append(check)

    def get_status(self) -> HealthStatus:
        """return the structured :class:`HealthStatus` value.

        evaluates every registered check synchronously and returns
        the aggregate. unlike the HTTP path, this does NOT
        short-circuit on the first failure -- in-process callers
        typically want the full picture rather than just the first
        broken thing.

        :return: aggregate :class:`HealthStatus`
        :rtype: HealthStatus
        """
        components: list[ComponentStatus] = []
        all_healthy = True
        for check in self._checks:
            try:
                ok = bool(check.probe())
                detail = None
            except Exception as exc:
                ok = False
                detail = str(exc)
            if not ok:
                all_healthy = False
            components.append(
                ComponentStatus(name=check.name, healthy=ok, detail=detail),
            )
        return HealthStatus(
            service=self._service_name,
            healthy=all_healthy,
            components=components,
        )

    async def start(self) -> None:
        """start the asyncio TCP listener.

        idempotent: re-calling on an already-started server is a
        no-op. the server runs as a background task on the current
        event loop; :meth:`stop` cancels it.

        :return: nothing
        :rtype: None
        """
        if self._server is not None:
            return
        self._server = await asyncio.start_server(
            self._handle_request,
            host=self._host,
            port=self._port,
        )
        log.info(
            "health server listening",
            extra={
                "extra_data": {
                    "host": self._host,
                    "port": self._port,
                    "checks": [c.name for c in self._checks],
                }
            },
        )

    async def stop(self) -> None:
        """drain pending connections and stop the listener.

        idempotent. callers should ``await`` this from the same
        event loop that called :meth:`start` so the listener's
        cleanup completes before the loop exits.

        :return: nothing
        :rtype: None
        """
        if self._server is None:
            return
        self._server.close()
        await self._server.wait_closed()
        self._server = None
        log.info("health server stopped", extra={"extra_data": {"port": self._port}})

    async def _handle_request(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """parse one HTTP/1.1 request line + dispatch to the right path.

        only ``GET /healthz`` and ``GET /readyz`` (and ``GET /metrics``
        when a ``metrics_provider`` is wired) are recognized; every
        other request returns ``404``. the connection closes
        after the response (no keep-alive). exceptions during
        probing are caught and surface as ``503`` so a misbehaving
        check cannot bring down the listener.

        response shape switches on the ``Accept`` header / the
        ``?format=json`` query argument: JSON for richer drill-in,
        plain text for the docker / k8s probe path.

        :param reader: stream reader for the inbound connection
        :ptype reader: asyncio.StreamReader
        :param writer: stream writer for the outbound response
        :ptype writer: asyncio.StreamWriter
        :return: nothing
        :rtype: None
        """
        accept_json = False
        try:
            request_line = await reader.readline()
            # parse headers so we can honour Accept; close without
            # consuming the body (we never read one anyway, GET-only).
            while True:
                header_line = await reader.readline()
                if header_line in (b"\r\n", b"", b"\n"):
                    break
                header_text = header_line.decode("ascii", errors="replace")
                if header_text.lower().startswith("accept:"):
                    if "application/json" in header_text.lower():
                        accept_json = True

            parts = request_line.decode("ascii", errors="replace").split()
            status = 400
            content_type = "text/plain; charset=utf-8"
            body: str | bytes = "bad request\n"
            if len(parts) >= 2 and parts[0] == "GET":
                raw_path = parts[1]
                path, _, query = raw_path.partition("?")
                if "format=json" in query:
                    accept_json = True
                if path in ("/healthz", "/readyz"):
                    if accept_json:
                        status, body = self._evaluate_checks_json()
                        content_type = "application/json; charset=utf-8"
                    else:
                        status, body = self._evaluate_checks_text()
                elif path == "/metrics" and self._metrics_provider is not None:
                    status = 200
                    content_type, body = self._metrics_provider()
                else:
                    status, body = (404, "not found\n")

            self._write_response(writer, status, body, content_type)
            await writer.drain()
        except Exception as exc:
            log.warning(
                "health server request handler failed",
                extra={"extra_data": {"error": str(exc)}},
            )
            try:
                self._write_response(
                    writer,
                    500,
                    "internal error\n",
                    "text/plain; charset=utf-8",
                )
                await writer.drain()
            except Exception:
                pass
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    def _evaluate_checks_text(self) -> tuple[int, str]:
        """plain-text response: ``(status, body)``.

        short-circuits on the first failing check to avoid running
        slow downstream probes when an upstream one already failed.
        check exceptions are treated as failures.

        :return: ``(200, "ok\\n")`` when every check passes;
            ``(503, "<name>: failed\\n")`` for the first failing check
        :rtype: tuple[int, str]
        """
        for check in self._checks:
            try:
                ok = bool(check.probe())
            except Exception as exc:
                log.warning(
                    "health check raised",
                    extra={
                        "extra_data": {"check": check.name, "error": str(exc)},
                    },
                )
                ok = False
            if not ok:
                return (503, f"{check.name}: failed\n")
        return (200, "ok\n")

    def _evaluate_checks_json(self) -> tuple[int, str]:
        """JSON response: ``(status, json_body)``.

        full :class:`HealthStatus` payload (does NOT short-circuit;
        operators want the full picture). status code is ``200`` iff
        every component is healthy, ``503`` otherwise.

        :return: status + serialized JSON body
        :rtype: tuple[int, str]
        """
        status_obj = self.get_status()
        status_code = 200 if status_obj.healthy else 503
        return (status_code, json.dumps(asdict(status_obj)) + "\n")

    @staticmethod
    def _write_response(
        writer: asyncio.StreamWriter,
        status: int,
        body: str | bytes,
        content_type: str,
    ) -> None:
        """write a minimal HTTP/1.1 response onto the stream writer.

        no keep-alive, ``Content-Length`` set so the client knows
        when the body ends without depending on a connection close.

        :param writer: stream writer for the outbound response
        :ptype writer: asyncio.StreamWriter
        :param status: HTTP status code
        :ptype status: int
        :param body: response body; ``str`` is UTF-8 encoded, ``bytes``
            (the prometheus exposition path) is written verbatim
        :ptype body: str | bytes
        :param content_type: the response ``Content-Type`` header
        :ptype content_type: str
        :return: nothing
        :rtype: None
        """
        reason = {
            200: "OK",
            400: "Bad Request",
            404: "Not Found",
            500: "Internal Server Error",
            503: "Service Unavailable",
        }.get(status, "OK")
        body_bytes = body.encode("utf-8") if isinstance(body, str) else body
        writer.write(
            (
                f"HTTP/1.1 {status} {reason}\r\n"
                f"Content-Type: {content_type}\r\n"
                f"Content-Length: {len(body_bytes)}\r\n"
                f"Connection: close\r\n"
                f"\r\n"
            ).encode("ascii")
        )
        writer.write(body_bytes)
