"""``HealthServer`` -- minimal asyncio HTTP server for /healthz endpoints.

every long-running service in the 3tears + 3tears stack exposes a
``HealthServer``. NATS-only services (3tears registry, 3tears
agent-tools serve, 3tears agent-router, datasource tool pods, agent
pods, admin tool server) need an HTTP health endpoint that container
orchestrators (docker compose, kubernetes) can probe; HTTP services
(hub, gateway) reuse the same check list + verdict through
:meth:`HealthServer.get_status` on their own routes, so the platform has
ONE definition of what "alive" and "ready" mean.

liveness vs readiness:

the two questions kubernetes asks are different, and answering them with
one check list is actively harmful. a liveness failure RESTARTS the pod;
a readiness failure only pulls it from rotation. so a check like "the
Hub JWKS cache has completed its first fetch" -- which a restart cannot
fix and which resolves on its own -- must never reach the liveness
verdict. when it did, the only safe deployment was to drop the
livenessProbe entirely, which is how ``registry`` and the tool pods ended
up with no restart-on-wedge net at all.

each :class:`HealthCheck` therefore declares a :class:`HealthTier`:

- ``LIVE`` -- "if this fails, the process is unrecoverable and a restart
  is the right response". terminal NATS close, persistent auth wedge.
- ``READY`` -- "if this fails, we cannot serve, but a restart would not
  help". a warming cache, a not-yet-registered tool set, a dependency
  that is temporarily unreachable.

liveness is CONTAINED IN readiness: every ``LIVE`` check is evaluated by
the readiness path too, because a terminally wedged pod must leave
rotation as well as be restarted -- otherwise it keeps absorbing traffic
for the seconds it takes k8s to notice. the converse does not hold, and
that asymmetry is the entire design. there is deliberately no "both"
tier: ``LIVE`` already means both.

contract:

- ``GET /healthz/live`` -- liveness. ``200`` when every ``LIVE`` check
  passes, ``503`` with the failing check name in the body otherwise.
- ``GET /healthz/ready`` -- readiness. ``200`` when every ``LIVE`` AND
  ``READY`` check passes.
- ``GET /healthz`` and ``GET /readyz`` -- both answer READINESS. this is
  exactly what they meant before the tier split (one check list, every
  check evaluated), and keeping it that way is deliberate: every compose
  healthcheck and k8s probe already pointing at them keeps its current
  behavior, so this release is additive on the wire. a caller that wants
  the liveness question must ask for it explicitly at ``/healthz/live``.
- any of the above with ``?format=json`` (or ``Accept: application/json``)
  -- structured :class:`HealthStatus` JSON with per-component ``healthy``
  + ``detail``, scoped to the requested tier. operators and CLI tooling
  read this for "which subsystem is down" without grepping logs; tests
  assert individual checks against it rather than a bare status code.
- ``GET /metrics`` -- prometheus text exposition, served only when a
  ``metrics_provider`` callable is wired at construction (returns
  ``(content_type, body)``). NATS-only RPC pods (registry, tool pods)
  have no HTTP framework of their own, so this route is how their
  in-flight-requests gauge becomes scrapable by KEDA's prometheus
  scaler. absent a provider the route returns ``404``.
- :meth:`HealthServer.get_status` -- in-process accessor returning the
  same :class:`HealthStatus` value the JSON endpoint would serialize,
  for a requested tier. consumers that already run an HTTP framework
  (hub, gateway) render their own routes off this rather than binding a
  second listener.

design choices:

- standard library only (no aiohttp / fastapi dep) so the module is
  consumable from every 3tears package without a transitive install.
  ``asyncio.start_server`` + a hand-rolled HTTP/1.1 frame are
  sufficient -- the surface is fixed so a real HTTP framework is overkill.
- probes may be sync (``() -> bool``) or async (``() -> Awaitable[bool]``).
  sync probes read cached state and run inline. async probes exist for
  the checks that must force a real round-trip: a cached
  ``is_connected`` flag reports connected long after a half-open socket's
  broker has gone away, so readiness that matters is decided by an actual
  ``ping()`` / ``SELECT 1``. an async probe MUST declare
  ``timeout_seconds`` -- an unbounded network probe would let a hung
  dependency wedge the health surface itself, which is the one failure
  mode a health surface may never have. the constructor rejects an async
  probe without one.
- a probe that returns False, raises, or times out is a FAILURE. a check
  that cannot answer cannot vouch for its tier.
- the server runs in the same event loop as the service. no background
  thread, no synchronization seam, no startup race between the listener
  and whatever produces the check state. a wedged event loop therefore
  fails the probe by timeout, catching a hang that a status code alone
  cannot.

usage::

    from threetears.observe.health import HealthCheck, HealthServer, HealthTier

    server = HealthServer(
        port=8000,
        service_name="registry",
        checks=[
            HealthCheck(
                name="nats",
                probe=lambda: nats_client.is_healthy,
                tier=HealthTier.LIVE,
            ),
            HealthCheck(
                name="jwks_warmed",
                probe=lambda: jwks_provider.is_warmed,
                tier=HealthTier.READY,
            ),
        ],
    )
    await server.start()
    # ... service runs ...
    # in-process status read (no HTTP round-trip):
    status = await server.get_status(HealthTier.READY)
    assert status.healthy
    # ... shutdown ...
    await server.stop()
"""

from __future__ import annotations

import asyncio
import inspect
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Awaitable, Callable, Union

MetricsProvider = Callable[[], tuple[str, bytes]]
HealthProbe = Union[Callable[[], bool], Callable[[], Awaitable[bool]]]

from threetears.observe.logging import get_logger

__all__ = [
    "ComponentStatus",
    "HealthCheck",
    "HealthServer",
    "HealthStatus",
    "HealthTier",
]


log = get_logger(__name__)


class HealthTier(str, Enum):
    """which orchestrator question a :class:`HealthCheck` answers.

    the tier decides what a failure COSTS. a ``LIVE`` failure restarts the
    pod; a ``READY`` failure only removes it from the service endpoints.
    putting a self-healing condition in ``LIVE`` turns a transient blip
    into a restart loop, which is why the tier is a required field with no
    default -- every check must state its own blast radius.
    """

    #: process is unrecoverable; a restart is the correct response. also
    #: evaluated by the readiness path (see module docstring: containment).
    LIVE = "live"
    #: cannot serve right now, but a restart would not help. evaluated by
    #: the readiness path only.
    READY = "ready"


@dataclass(frozen=True)
class HealthCheck:
    """one check the :class:`HealthServer` evaluates per probe.

    :param name: short identifier the failure response includes (so
        operators see which check tripped without inspecting logs)
    :ptype name: str
    :param probe: zero-arg callable returning ``True`` when the underlying
        state is healthy, either directly or as an awaitable. a sync probe
        is expected to be cheap (reads a cached flag, polls a connection's
        local state); an async probe may do a bounded round-trip and MUST
        declare ``timeout_seconds``
    :ptype probe: Callable[[], bool] | Callable[[], Awaitable[bool]]
    :param tier: whether a failure means "restart me" (``LIVE``) or
        "stop routing to me" (``READY``). required: a check with an
        implicit tier is how a readiness gate ends up restart-looping a pod
    :ptype tier: HealthTier
    :param timeout_seconds: bound on an async probe, after which the check
        is recorded as failed. required for async probes, ignored for sync
        ones
    :ptype timeout_seconds: float | None
    :raises ValueError: if an async probe is declared without
        ``timeout_seconds``, or ``timeout_seconds`` is not positive
    """

    name: str
    probe: HealthProbe
    tier: HealthTier
    timeout_seconds: float | None = None

    def __post_init__(self) -> None:
        """reject an unbounded async probe at construction.

        failing here rather than at probe time is deliberate: an unbounded
        network probe works perfectly until the day its dependency hangs,
        and then takes the health surface down with it. that is not a
        failure mode worth discovering in production.

        :return: nothing
        :rtype: None
        :raises ValueError: if an async probe has no positive
            ``timeout_seconds``
        """
        if self.timeout_seconds is not None and self.timeout_seconds <= 0:
            raise ValueError(
                f"health check {self.name!r}: timeout_seconds must be positive, got {self.timeout_seconds}",
            )
        if inspect.iscoroutinefunction(self.probe) and self.timeout_seconds is None:
            raise ValueError(
                f"health check {self.name!r}: an async probe must declare timeout_seconds "
                "so a hung dependency cannot wedge the health surface",
            )


@dataclass(frozen=True)
class ComponentStatus:
    """structured status for one component.

    serializes onto the JSON :class:`HealthStatus` payload so operators
    reading ``GET /healthz/ready?format=json`` see exactly which subsystem
    reports unhealthy and (when the probe surfaces detail) why.

    :param name: component identifier (matches the :class:`HealthCheck` name)
    :ptype name: str
    :param healthy: True when the probe returned ``True``
    :ptype healthy: bool
    :param detail: optional human-readable reason; populated when the probe
        raised (the exception message lands here) or timed out
    :ptype detail: str | None
    """

    name: str
    healthy: bool
    detail: str | None = None


@dataclass(frozen=True)
class HealthStatus:
    """aggregate health status the JSON endpoint serializes.

    :param service: service name (e.g. ``"registry"``, ``"agent-router"``);
        included so a multi-service log scrape can attribute the status
        without parsing the URL
    :ptype service: str
    :param tier: which question this status answers, so a captured payload
        is unambiguous about whether it describes liveness or readiness
    :ptype tier: HealthTier
    :param healthy: ``True`` iff every component in the tier is healthy
    :ptype healthy: bool
    :param components: per-component status; liveness checks first, then
        readiness ones, each in registration order (so the failing-check
        short-circuit yields a partial list when any check fails -- a
        downstream check's absence means "we never got that far")
    :ptype components: list[ComponentStatus]
    """

    service: str
    tier: HealthTier
    healthy: bool
    components: list[ComponentStatus] = field(default_factory=list)


class HealthServer:
    """minimal asyncio HTTP server serving the liveness / readiness contract.

    intentional limitations: routing is limited to ``/healthz/live``,
    ``/healthz/ready``, their ``/healthz`` + ``/readyz`` aliases and (when a
    ``metrics_provider`` is wired) ``/metrics``; only ``GET``, only the
    ``format=json`` query argument (also accepts ``Accept: application/json``),
    no chunked transfer, no keep-alive. the surface is exactly what docker /
    kubernetes probes need plus the JSON shape operators want for drill-in plus
    the optional prometheus exposition KEDA's scaler scrapes.

    :param port: TCP port to bind on. matches the 3tears-hub Dockerfile's
        HEALTHCHECK port (8000) so the inherited check works without compose
        overrides
    :ptype port: int
    :param service_name: short identifier echoed onto the
        :class:`HealthStatus` JSON body (e.g. ``"registry"``, ``"agent-router"``)
    :ptype service_name: str
    :param checks: checks evaluated on every probe, each declaring its
        :class:`HealthTier`. additional checks can be appended at runtime via
        :meth:`register_check`
    :ptype checks: list[HealthCheck] | None
    :param host: bind interface; default ``0.0.0.0`` so the container's
        external port mapping reaches the listener
    :ptype host: str
    :param metrics_provider: optional zero-arg callable returning
        ``(content_type, body)`` for the ``GET /metrics`` route. wired by
        NATS-only RPC pods (registry, tool pods) to expose their
        in-flight-requests gauge to KEDA's prometheus scaler through the one
        HTTP listener they already run. ``None`` leaves ``/metrics``
        returning ``404``
    :ptype metrics_provider: Callable[[], tuple[str, bytes]] | None
    """

    #: path -> tier the probe routes answer.
    #:
    #: ``/healthz`` and ``/readyz`` both answer READY, which is EXACTLY what they
    #: meant before the tier split (one check list, every check evaluated). that is
    #: deliberate: every compose healthcheck and k8s probe already pointing at them
    #: keeps its current semantics, so this release adds the two tiered routes
    #: without changing any existing consumer's behavior. ``/healthz`` naming an
    #: effectively-readiness endpoint is inherited, not new.
    _TIER_ROUTES: dict[str, HealthTier] = {
        "/healthz/live": HealthTier.LIVE,
        "/healthz/ready": HealthTier.READY,
        "/healthz": HealthTier.READY,
        "/readyz": HealthTier.READY,
    }

    def __init__(
        self,
        *,
        port: int,
        service_name: str,
        checks: list[HealthCheck] | None = None,
        host: str = "0.0.0.0",  # noqa: S104 -- kube/compose probes reach the pod by IP
        metrics_provider: MetricsProvider | None = None,
    ) -> None:
        """initialize health server with the supplied checks.

        :param port: TCP port to bind on
        :ptype port: int
        :param service_name: identifier echoed onto status JSON
        :ptype service_name: str
        :param checks: tiered check list (may be empty; add later via
            :meth:`register_check`)
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

        services that wire their state lazily (e.g. NATS connection comes up
        after :meth:`start`) can register the corresponding check after the
        listener is live -- the check applies to the next probe.

        :param check: check to add, carrying its own tier
        :ptype check: HealthCheck
        :return: nothing
        :rtype: None
        """
        self._checks.append(check)

    def checks_for_tier(self, tier: HealthTier) -> list[HealthCheck]:
        """return the checks a probe of ``tier`` evaluates, in evaluation order.

        liveness checks come first so a short-circuited readiness failure
        names the fundamental problem (a dead NATS data plane) rather than a
        downstream symptom (a tool set that never registered because of it).

        :param tier: which question is being asked
        :ptype tier: HealthTier
        :return: checks to evaluate, liveness first then readiness
        :rtype: list[HealthCheck]
        """
        live = [c for c in self._checks if c.tier is HealthTier.LIVE]
        if tier is HealthTier.LIVE:
            result = live
        else:
            result = live + [c for c in self._checks if c.tier is HealthTier.READY]
        return result

    async def get_status(self, tier: HealthTier) -> HealthStatus:
        """return the structured :class:`HealthStatus` value for ``tier``.

        evaluates every check in the tier and returns the aggregate. unlike
        the plain-text HTTP path, this does NOT short-circuit on the first
        failure -- callers typically want the full picture rather than just
        the first broken thing. this is also the seam consumers that already
        run an HTTP framework (hub, gateway) render their own routes from,
        so the platform keeps one definition of the verdict.

        :param tier: which question to answer
        :ptype tier: HealthTier
        :return: aggregate :class:`HealthStatus` scoped to ``tier``
        :rtype: HealthStatus
        """
        components: list[ComponentStatus] = []
        all_healthy = True
        for check in self.checks_for_tier(tier):
            ok, detail = await self._run_probe(check)
            if not ok:
                all_healthy = False
            components.append(
                ComponentStatus(name=check.name, healthy=ok, detail=detail),
            )
        return HealthStatus(
            service=self._service_name,
            tier=tier,
            healthy=all_healthy,
            components=components,
        )

    async def start(self) -> None:
        """start the asyncio TCP listener.

        idempotent: re-calling on an already-started server is a no-op. the
        server runs as a background task on the current event loop;
        :meth:`stop` cancels it.

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
                    "live_checks": [c.name for c in self._checks if c.tier is HealthTier.LIVE],
                    "ready_checks": [c.name for c in self._checks if c.tier is HealthTier.READY],
                }
            },
        )

    async def stop(self) -> None:
        """drain pending connections and stop the listener.

        idempotent. callers should ``await`` this from the same event loop
        that called :meth:`start` so the listener's cleanup completes before
        the loop exits.

        :return: nothing
        :rtype: None
        """
        if self._server is None:
            return
        self._server.close()
        await self._server.wait_closed()
        self._server = None
        log.info("health server stopped", extra={"extra_data": {"port": self._port}})

    async def _run_probe(self, check: HealthCheck) -> tuple[bool, str | None]:
        """evaluate one check, returning ``(healthy, detail)``.

        a probe that returns False, raises, or exceeds its declared timeout
        is a failure -- a check that cannot answer cannot vouch for its tier.
        the exception message or timeout notice lands in ``detail`` so the
        JSON payload explains WHY without a log dive.

        :param check: check to evaluate
        :ptype check: HealthCheck
        :return: ``(healthy, detail)``; ``detail`` is ``None`` on a clean pass
        :rtype: tuple[bool, str | None]
        """
        ok = False
        detail: str | None = None
        try:
            raw = check.probe()
            if inspect.isawaitable(raw):
                if check.timeout_seconds is None:
                    # close the coroutine so it is not left un-awaited (which would
                    # otherwise surface as a confusing RuntimeWarning far from here).
                    # the constructor rejects `async def` probes without a timeout;
                    # this catches the exotic shape it cannot see -- a sync callable
                    # that RETURNS a coroutine.
                    if inspect.iscoroutine(raw):
                        raw.close()
                    raise RuntimeError(
                        "probe returned an awaitable but declared no timeout_seconds",
                    )
                try:
                    ok = bool(await asyncio.wait_for(raw, check.timeout_seconds))
                except asyncio.TimeoutError:
                    detail = f"timed out after {check.timeout_seconds}s"
                    log.warning(
                        "health check timed out",
                        extra={
                            "extra_data": {
                                "check": check.name,
                                "tier": check.tier.value,
                                "timeout_seconds": check.timeout_seconds,
                            },
                        },
                    )
            else:
                ok = bool(raw)
        except Exception as exc:
            ok = False
            detail = str(exc)
            log.warning(
                "health check raised",
                extra={
                    "extra_data": {
                        "check": check.name,
                        "tier": check.tier.value,
                        "error": str(exc),
                    },
                },
            )
        return (ok, detail)

    async def _handle_request(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """parse one HTTP/1.1 request line + dispatch to the right path.

        only the probe routes (and ``GET /metrics`` when a
        ``metrics_provider`` is wired) are recognized; every other request
        returns ``404``. the connection closes after the response (no
        keep-alive). exceptions during probing are caught and surface as
        ``503`` so a misbehaving check cannot bring down the listener.

        response shape switches on the ``Accept`` header / the
        ``?format=json`` query argument: JSON for richer drill-in, plain text
        for the docker / k8s probe path.

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
            # parse headers so we can honour Accept; close without consuming
            # the body (we never read one anyway, GET-only).
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
                # tolerate a trailing slash so probe configs are not
                # load-bearing on a character.
                normalized = path.rstrip("/") or "/healthz"
                tier = self._TIER_ROUTES.get(normalized)
                if tier is not None:
                    if accept_json:
                        status, body = await self._evaluate_json(tier)
                        content_type = "application/json; charset=utf-8"
                    else:
                        status, body = await self._evaluate_text(tier)
                elif normalized == "/metrics" and self._metrics_provider is not None:
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

    async def _evaluate_text(self, tier: HealthTier) -> tuple[int, str]:
        """plain-text response for ``tier``: ``(status, body)``.

        short-circuits on the first failing check to avoid running slow
        downstream probes when an upstream one already failed.

        :param tier: which question to answer
        :ptype tier: HealthTier
        :return: ``(200, "ok\\n")`` when every check in the tier passes;
            ``(503, "<name>: failed\\n")`` for the first failing check
        :rtype: tuple[int, str]
        """
        result = (200, "ok\n")
        for check in self.checks_for_tier(tier):
            ok, _detail = await self._run_probe(check)
            if not ok:
                result = (503, f"{check.name}: failed\n")
                break
        return result

    async def _evaluate_json(self, tier: HealthTier) -> tuple[int, str]:
        """JSON response for ``tier``: ``(status, json_body)``.

        full :class:`HealthStatus` payload (does NOT short-circuit; operators
        want the full picture). status code is ``200`` iff every component in
        the tier is healthy, ``503`` otherwise.

        :param tier: which question to answer
        :ptype tier: HealthTier
        :return: status + serialized JSON body
        :rtype: tuple[int, str]
        """
        status_obj = await self.get_status(tier)
        status_code = 200 if status_obj.healthy else 503
        payload = asdict(status_obj)
        payload["tier"] = status_obj.tier.value
        return (status_code, json.dumps(payload) + "\n")

    @staticmethod
    def _write_response(
        writer: asyncio.StreamWriter,
        status: int,
        body: str | bytes,
        content_type: str,
    ) -> None:
        """write a minimal HTTP/1.1 response onto the stream writer.

        no keep-alive, ``Content-Length`` set so the client knows when the
        body ends without depending on a connection close.

        :param writer: stream writer for the outbound response
        :ptype writer: asyncio.StreamWriter
        :param status: HTTP status code
        :ptype status: int
        :param body: response body; ``str`` is UTF-8 encoded, ``bytes`` (the
            prometheus exposition path) is written verbatim
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
