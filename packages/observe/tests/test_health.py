"""tests for ``threetears.observe.health.HealthServer``.

cover the surface every consumer relies on:

- tier routing: a READY check failing takes ``/healthz/ready`` to 503 while
  ``/healthz/live`` stays 200. this is the whole point of the split -- a
  readiness gate (a warming cache, an unregistered tool set) must never
  restart a pod
- containment: a LIVE check failing fails BOTH endpoints. a terminally wedged
  process must leave rotation as well as be restarted, so liveness checks are
  evaluated by the readiness path too
- ``/healthz`` is the load-balancer default aliased to LIVE; ``/readyz`` to READY
- sync and async probes, with async probes bounded by a mandatory timeout
- a probe that raises, or times out, is a FAILURE (a check that cannot answer
  cannot vouch for the tier)
- unknown paths return 404 without touching the check list
- non-GET methods return 400 (the surface is intentionally narrow)
- ``start`` / ``stop`` are idempotent
"""

from __future__ import annotations

import asyncio
import json
import socket
from contextlib import closing

import pytest

from threetears.observe.health import (
    ComponentStatus,
    HealthCheck,
    HealthServer,
    HealthStatus,
    HealthTier,
)


def _free_port() -> int:
    """find a free localhost port for tests.

    binds + immediately closes a socket so we get a port the OS will
    leave alone for the moment we then bind the HealthServer to it.
    """
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("127.0.0.1", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


async def _http_get(host: str, port: int, path: str) -> tuple[int, str]:
    """tiny GET-only HTTP/1.1 client. returns (status, body)."""
    reader, writer = await asyncio.open_connection(host, port)
    writer.write(
        f"GET {path} HTTP/1.1\r\nHost: {host}\r\nConnection: close\r\n\r\n".encode(),
    )
    await writer.drain()
    raw = await reader.read()
    writer.close()
    try:
        await writer.wait_closed()
    except Exception:
        pass
    head, _, body = raw.partition(b"\r\n\r\n")
    status_line = head.split(b"\r\n", 1)[0].decode("ascii")
    status = int(status_line.split()[1])
    return status, body.decode("utf-8")


async def _http_request(
    host: str,
    port: int,
    method: str,
    path: str,
    extra_headers: str = "",
) -> tuple[int, str]:
    """tiny request helper supporting non-GET methods + extra headers."""
    reader, writer = await asyncio.open_connection(host, port)
    writer.write(
        (f"{method} {path} HTTP/1.1\r\nHost: {host}\r\nConnection: close\r\n{extra_headers}\r\n").encode(),
    )
    await writer.drain()
    raw = await reader.read()
    writer.close()
    try:
        await writer.wait_closed()
    except Exception:
        pass
    head, _, body = raw.partition(b"\r\n\r\n")
    status_line = head.split(b"\r\n", 1)[0].decode("ascii")
    status = int(status_line.split()[1])
    return status, body.decode("utf-8")


class _Server:
    """async context manager starting/stopping a HealthServer on a free port."""

    def __init__(self, *checks: HealthCheck, service_name: str = "test-service") -> None:
        self.port = _free_port()
        self.server = HealthServer(
            port=self.port,
            service_name=service_name,
            host="127.0.0.1",
            checks=list(checks),
        )

    async def __aenter__(self) -> "_Server":
        await self.server.start()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.server.stop()

    async def get(self, path: str) -> tuple[int, str]:
        """GET ``path`` against this server."""
        return await _http_get("127.0.0.1", self.port, path)


class TestTierRouting:
    """the live/ready split -- the reason this server exists."""

    @pytest.mark.asyncio
    async def test_failing_ready_check_does_not_fail_liveness(self) -> None:
        """a readiness gate must never restart the pod.

        this is the exact condition that forced ``registry`` and the tool pods to
        ship with no livenessProbe at all: ``jwks_warmed`` is a readiness gate, and
        under the old aliased contract it dragged ``/healthz`` down with it, so a
        cold JWKS cache would have restart-looped the pod.
        """
        async with _Server(
            HealthCheck(name="nats", probe=lambda: True, tier=HealthTier.LIVE),
            HealthCheck(name="jwks_warmed", probe=lambda: False, tier=HealthTier.READY),
        ) as s:
            live_status, live_body = await s.get("/healthz/live")
            ready_status, ready_body = await s.get("/healthz/ready")

        assert live_status == 200
        assert "ok" in live_body
        assert ready_status == 503
        assert "jwks_warmed" in ready_body

    @pytest.mark.asyncio
    async def test_failing_live_check_fails_both_tiers(self) -> None:
        """containment: a dead process is neither alive nor ready.

        liveness checks are evaluated by the readiness path too, so a pod whose
        NATS data plane is terminally wedged leaves rotation immediately rather
        than absorbing traffic for the seconds it takes k8s to restart it.
        """
        async with _Server(
            HealthCheck(name="nats", probe=lambda: False, tier=HealthTier.LIVE),
            HealthCheck(name="tools", probe=lambda: True, tier=HealthTier.READY),
        ) as s:
            live_status, live_body = await s.get("/healthz/live")
            ready_status, ready_body = await s.get("/healthz/ready")

        assert live_status == 503
        assert "nats" in live_body
        assert ready_status == 503
        assert "nats" in ready_body

    @pytest.mark.asyncio
    async def test_ready_check_not_evaluated_by_liveness(self) -> None:
        """the liveness path must not even RUN a readiness probe.

        stronger than asserting the status code: a readiness probe may be
        network-bound, and the liveness path is the one that must stay cheap and
        dependency-free -- otherwise a slow dependency delays a restart decision.
        """
        invoked: list[str] = []

        def _ready_probe() -> bool:
            invoked.append("ready")
            return True

        async with _Server(
            HealthCheck(name="nats", probe=lambda: True, tier=HealthTier.LIVE),
            HealthCheck(name="db", probe=_ready_probe, tier=HealthTier.READY),
        ) as s:
            await s.get("/healthz/live")
            assert invoked == []
            await s.get("/healthz/ready")

        assert invoked == ["ready"]

    @pytest.mark.asyncio
    async def test_live_checks_evaluated_first_on_readiness(self) -> None:
        """the readiness path evaluates liveness checks before its own.

        ordering matters for the short-circuited text path: an operator reading a
        503 body wants the fundamental failure named, not a downstream symptom.
        """
        async with _Server(
            HealthCheck(name="warmed", probe=lambda: False, tier=HealthTier.READY),
            HealthCheck(name="nats", probe=lambda: False, tier=HealthTier.LIVE),
        ) as s:
            status, body = await s.get("/healthz/ready")

        assert status == 503
        assert "nats" in body

    @pytest.mark.asyncio
    async def test_tier_with_no_checks_reports_healthy(self) -> None:
        """a tier with no registered checks is vacuously healthy.

        a service with only readiness checks still answers liveness 200 -- "the
        process is answering HTTP at all" is itself the liveness signal there.
        """
        async with _Server(
            HealthCheck(name="warmed", probe=lambda: False, tier=HealthTier.READY),
        ) as s:
            live_status, _ = await s.get("/healthz/live")
            ready_status, _ = await s.get("/healthz/ready")

        assert live_status == 200
        assert ready_status == 503


class TestPathAliases:
    """``/healthz`` and ``/readyz`` -> READY; only ``/healthz/live`` asks for liveness."""

    @pytest.mark.asyncio
    async def test_healthz_still_means_every_check(self) -> None:
        """``/healthz`` keeps its PRE-SPLIT meaning: every check, not just liveness.

        this is the compatibility guarantee that makes the release additive on the
        wire. every compose healthcheck and k8s probe already pointing at
        ``/healthz`` evaluates the same set it did before, so nothing has to be
        repointed and no ``depends_on: service_healthy`` gate releases earlier
        than it used to.
        """
        async with _Server(
            HealthCheck(name="nats", probe=lambda: True, tier=HealthTier.LIVE),
            HealthCheck(name="warmed", probe=lambda: False, tier=HealthTier.READY),
        ) as s:
            status, body = await s.get("/healthz")

        assert status == 503
        assert "warmed" in body

    @pytest.mark.asyncio
    async def test_healthz_and_readyz_agree(self) -> None:
        """the two legacy paths answer the same question, as they always did."""
        async with _Server(
            HealthCheck(name="nats", probe=lambda: True, tier=HealthTier.LIVE),
            HealthCheck(name="warmed", probe=lambda: False, tier=HealthTier.READY),
        ) as s:
            healthz, _ = await s.get("/healthz")
            readyz, _ = await s.get("/readyz")
            live, _ = await s.get("/healthz/live")

        assert healthz == readyz == 503
        # only the explicit liveness route reports the process as alive.
        assert live == 200

    @pytest.mark.asyncio
    async def test_readyz_aliases_readiness(self) -> None:
        """``/readyz`` reports READINESS."""
        async with _Server(
            HealthCheck(name="nats", probe=lambda: True, tier=HealthTier.LIVE),
            HealthCheck(name="warmed", probe=lambda: False, tier=HealthTier.READY),
        ) as s:
            status, body = await s.get("/readyz")

        assert status == 503
        assert "warmed" in body

    @pytest.mark.asyncio
    async def test_trailing_slash_variants_route_identically(self) -> None:
        """``/healthz/live/`` matches ``/healthz/live``."""
        async with _Server(
            HealthCheck(name="nats", probe=lambda: True, tier=HealthTier.LIVE),
        ) as s:
            bare, _ = await s.get("/healthz/live")
            slashed, _ = await s.get("/healthz/live/")

        assert bare == 200
        assert slashed == 200


class TestAsyncProbes:
    """async probes -- required so the hub's real-I/O readiness can consolidate."""

    @pytest.mark.asyncio
    async def test_async_probe_passing(self) -> None:
        """an async probe returning True passes its tier."""

        async def _ping() -> bool:
            await asyncio.sleep(0)
            return True

        async with _Server(
            HealthCheck(name="nats_ping", probe=_ping, tier=HealthTier.READY, timeout_seconds=1.0),
        ) as s:
            status, _ = await s.get("/healthz/ready")

        assert status == 200

    @pytest.mark.asyncio
    async def test_async_probe_failing(self) -> None:
        """an async probe returning False fails its tier."""

        async def _ping() -> bool:
            await asyncio.sleep(0)
            return False

        async with _Server(
            HealthCheck(name="nats_ping", probe=_ping, tier=HealthTier.READY, timeout_seconds=1.0),
        ) as s:
            status, body = await s.get("/healthz/ready")

        assert status == 503
        assert "nats_ping" in body

    @pytest.mark.asyncio
    async def test_async_probe_timeout_is_a_failure(self) -> None:
        """a probe that hangs must fail its tier, not wedge the listener.

        the whole risk of admitting network-bound probes is a hung dependency
        taking the health surface with it; the timeout is what makes that safe.
        """

        async def _hangs() -> bool:
            await asyncio.sleep(30)
            return True

        async with _Server(
            HealthCheck(name="db", probe=_hangs, tier=HealthTier.READY, timeout_seconds=0.05),
        ) as s:
            status, body = await s.get("/healthz/ready")
            # the listener is still serving after the timeout.
            live_status, _ = await s.get("/healthz/live")

        assert status == 503
        assert "db" in body
        assert live_status == 200

    @pytest.mark.asyncio
    async def test_async_probe_timeout_detail_in_json(self) -> None:
        """the JSON payload names the timeout so operators see WHY."""

        async def _hangs() -> bool:
            await asyncio.sleep(30)
            return True

        async with _Server(
            HealthCheck(name="db", probe=_hangs, tier=HealthTier.READY, timeout_seconds=0.05),
        ) as s:
            status, body = await s.get("/healthz/ready?format=json")

        assert status == 503
        payload = json.loads(body)
        component = next(c for c in payload["components"] if c["name"] == "db")
        assert component["healthy"] is False
        assert "timed out" in component["detail"]

    def test_async_probe_without_timeout_is_a_construction_error(self) -> None:
        """an unbounded async probe is rejected at construction, not at probe time.

        failing loudly here is the point: the alternative is a check that works
        fine until the day its dependency hangs, and then wedges the probe.
        """

        async def _ping() -> bool:
            return True

        with pytest.raises(ValueError, match="timeout_seconds"):
            HealthCheck(name="nats_ping", probe=_ping, tier=HealthTier.READY)

    def test_sync_probe_needs_no_timeout(self) -> None:
        """sync probes read cached state and are exempt from the timeout rule."""
        check = HealthCheck(name="flag", probe=lambda: True, tier=HealthTier.LIVE)
        assert check.timeout_seconds is None

    def test_non_positive_timeout_is_a_construction_error(self) -> None:
        """a zero / negative timeout would fail every probe instantly."""

        async def _ping() -> bool:
            return True

        with pytest.raises(ValueError, match="must be positive"):
            HealthCheck(name="p", probe=_ping, tier=HealthTier.READY, timeout_seconds=0)

    @pytest.mark.asyncio
    async def test_sync_callable_returning_a_coroutine_fails_the_check(self) -> None:
        """the shape the constructor cannot see is caught at probe time.

        ``inspect.iscoroutinefunction`` is False for a plain function that RETURNS
        a coroutine, so such a probe slips past construction. it must fail its
        check loudly rather than run unbounded.
        """

        async def _inner() -> bool:
            return True

        def _sneaky() -> bool:
            return _inner()  # type: ignore[return-value]

        async with _Server(
            HealthCheck(name="sneaky", probe=_sneaky, tier=HealthTier.READY),
        ) as s:
            status, body = await s.get("/healthz/ready?format=json")

        assert status == 503
        payload = json.loads(body)
        component = next(c for c in payload["components"] if c["name"] == "sneaky")
        assert component["healthy"] is False
        assert "timeout_seconds" in component["detail"]


class TestProbeFailureModes:
    """raising probes, short-circuiting."""

    @pytest.mark.asyncio
    async def test_raising_probe_is_a_failure(self) -> None:
        """a check that crashes cannot vouch for its tier."""

        def _boom() -> bool:
            raise RuntimeError("connection reset")

        async with _Server(
            HealthCheck(name="nats", probe=_boom, tier=HealthTier.LIVE),
        ) as s:
            status, body = await s.get("/healthz/live")

        assert status == 503
        assert "nats" in body

    @pytest.mark.asyncio
    async def test_raising_probe_detail_surfaces_in_json(self) -> None:
        """the exception message lands in the component detail."""

        def _boom() -> bool:
            raise RuntimeError("connection reset")

        async with _Server(
            HealthCheck(name="nats", probe=_boom, tier=HealthTier.LIVE),
        ) as s:
            status, body = await s.get("/healthz/live?format=json")

        assert status == 503
        payload = json.loads(body)
        component = next(c for c in payload["components"] if c["name"] == "nats")
        assert component["healthy"] is False
        assert "connection reset" in component["detail"]

    @pytest.mark.asyncio
    async def test_text_path_short_circuits_on_first_failure(self) -> None:
        """the plain-text path stops at the first failing check in the tier."""
        invoked: list[str] = []

        def _upstream() -> bool:
            invoked.append("upstream")
            return False

        def _downstream() -> bool:
            invoked.append("downstream")
            return True

        async with _Server(
            HealthCheck(name="upstream", probe=_upstream, tier=HealthTier.LIVE),
            HealthCheck(name="downstream", probe=_downstream, tier=HealthTier.LIVE),
        ) as s:
            status, _ = await s.get("/healthz/live")

        assert status == 503
        assert invoked == ["upstream"]


class TestJsonResponse:
    """JSON drill-in, scoped to the requested tier."""

    @pytest.mark.asyncio
    async def test_json_readiness_includes_live_checks(self) -> None:
        """``/healthz/ready?format=json`` reports liveness checks then readiness ones."""
        async with _Server(
            HealthCheck(name="nats", probe=lambda: True, tier=HealthTier.LIVE),
            HealthCheck(name="warmed", probe=lambda: True, tier=HealthTier.READY),
        ) as s:
            status, body = await s.get("/healthz/ready?format=json")

        assert status == 200
        payload = json.loads(body)
        assert payload["service"] == "test-service"
        assert payload["tier"] == "ready"
        assert payload["healthy"] is True
        assert [c["name"] for c in payload["components"]] == ["nats", "warmed"]

    @pytest.mark.asyncio
    async def test_json_liveness_excludes_ready_checks(self) -> None:
        """``/healthz/live?format=json`` reports liveness components only."""
        async with _Server(
            HealthCheck(name="nats", probe=lambda: True, tier=HealthTier.LIVE),
            HealthCheck(name="warmed", probe=lambda: True, tier=HealthTier.READY),
        ) as s:
            status, body = await s.get("/healthz/live?format=json")

        assert status == 200
        payload = json.loads(body)
        assert payload["tier"] == "live"
        assert [c["name"] for c in payload["components"]] == ["nats"]

    @pytest.mark.asyncio
    async def test_json_does_not_short_circuit(self) -> None:
        """operators want the full picture, so every component is reported."""
        async with _Server(
            HealthCheck(name="a", probe=lambda: False, tier=HealthTier.LIVE),
            HealthCheck(name="b", probe=lambda: False, tier=HealthTier.LIVE),
        ) as s:
            status, body = await s.get("/healthz/live?format=json")

        assert status == 503
        payload = json.loads(body)
        assert [c["name"] for c in payload["components"]] == ["a", "b"]
        assert all(c["healthy"] is False for c in payload["components"])

    @pytest.mark.asyncio
    async def test_accept_json_header_returns_json(self) -> None:
        """``Accept: application/json`` -> JSON without a query string."""
        port = _free_port()
        server = HealthServer(
            port=port,
            service_name="test-service",
            host="127.0.0.1",
            checks=[HealthCheck(name="nats", probe=lambda: True, tier=HealthTier.LIVE)],
        )
        await server.start()
        try:
            status, body = await _http_request(
                "127.0.0.1",
                port,
                "GET",
                "/healthz/live",
                extra_headers="Accept: application/json\r\n",
            )
        finally:
            await server.stop()
        assert status == 200
        assert json.loads(body)["healthy"] is True


class TestErrorPaths:
    """unknown paths / methods."""

    @pytest.mark.asyncio
    async def test_unknown_path_returns_404(self) -> None:
        """``GET /unknown`` -> ``404`` without touching the check list."""
        invoked: list[str] = []

        def _track() -> bool:
            invoked.append("ran")
            return True

        async with _Server(
            HealthCheck(name="x", probe=_track, tier=HealthTier.LIVE),
        ) as s:
            status, _ = await s.get("/some/random/path")

        assert status == 404
        assert invoked == []

    @pytest.mark.asyncio
    async def test_non_get_method_returns_400(self) -> None:
        """non-GET methods are not in the surface -> ``400``."""
        port = _free_port()
        server = HealthServer(
            port=port,
            service_name="test-service",
            host="127.0.0.1",
            checks=[HealthCheck(name="x", probe=lambda: True, tier=HealthTier.LIVE)],
        )
        await server.start()
        try:
            status, _ = await _http_request("127.0.0.1", port, "POST", "/healthz")
        finally:
            await server.stop()
        assert status == 400


class TestMetricsRoute:
    """optional prometheus exposition on the one listener the pod already runs."""

    @pytest.mark.asyncio
    async def test_metrics_served_when_provider_wired(self) -> None:
        """``/metrics`` returns the provider's body verbatim."""

        def _provider() -> tuple[str, bytes]:
            return ("text/plain; version=0.0.4", b"inflight 3\n")

        port = _free_port()
        server = HealthServer(
            port=port,
            service_name="test-service",
            host="127.0.0.1",
            checks=[HealthCheck(name="x", probe=lambda: True, tier=HealthTier.LIVE)],
            metrics_provider=_provider,
        )
        await server.start()
        try:
            status, body = await _http_get("127.0.0.1", port, "/metrics")
        finally:
            await server.stop()
        assert status == 200
        assert "inflight 3" in body

    @pytest.mark.asyncio
    async def test_metrics_404s_without_provider(self) -> None:
        """no provider -> the route stays absent."""
        async with _Server(
            HealthCheck(name="x", probe=lambda: True, tier=HealthTier.LIVE),
        ) as s:
            status, _ = await s.get("/metrics")

        assert status == 404


class TestStatusAccessor:
    """``get_status`` -- the in-process seam consumers with their own HTTP server use.

    the hub and the gateway already run an HTTP framework; binding a second
    listener for health would be worse, not DRYer. they share the check list and
    the verdict logic through this accessor and render it on their own routes.
    """

    @pytest.mark.asyncio
    async def test_get_status_returns_full_picture_for_tier(self) -> None:
        """``get_status`` evaluates every check in the tier without short-circuiting."""
        server = HealthServer(
            port=_free_port(),
            service_name="test-service",
            host="127.0.0.1",
            checks=[
                HealthCheck(name="a", probe=lambda: False, tier=HealthTier.READY),
                HealthCheck(name="b", probe=lambda: True, tier=HealthTier.READY),
                HealthCheck(name="c", probe=lambda: True, tier=HealthTier.LIVE),
            ],
        )
        status = await server.get_status(HealthTier.READY)

        assert isinstance(status, HealthStatus)
        assert status.healthy is False
        assert status.tier is HealthTier.READY
        # live checks first, then readiness ones, each in registration order.
        assert status.components == [
            ComponentStatus(name="c", healthy=True, detail=None),
            ComponentStatus(name="a", healthy=False, detail=None),
            ComponentStatus(name="b", healthy=True, detail=None),
        ]

    @pytest.mark.asyncio
    async def test_get_status_live_excludes_ready_checks(self) -> None:
        """the LIVE tier query returns liveness checks only."""
        server = HealthServer(
            port=_free_port(),
            service_name="test-service",
            host="127.0.0.1",
            checks=[
                HealthCheck(name="a", probe=lambda: True, tier=HealthTier.LIVE),
                HealthCheck(name="b", probe=lambda: False, tier=HealthTier.READY),
            ],
        )
        status = await server.get_status(HealthTier.LIVE)

        assert [c.name for c in status.components] == ["a"]
        assert status.healthy is True

    @pytest.mark.asyncio
    async def test_get_status_awaits_async_probes(self) -> None:
        """the in-process accessor runs async probes exactly as the HTTP path does."""

        async def _ping() -> bool:
            await asyncio.sleep(0)
            return False

        server = HealthServer(
            port=_free_port(),
            service_name="test-service",
            host="127.0.0.1",
            checks=[
                HealthCheck(name="db", probe=_ping, tier=HealthTier.READY, timeout_seconds=1.0),
            ],
        )
        status = await server.get_status(HealthTier.READY)

        assert status.healthy is False

    @pytest.mark.asyncio
    async def test_register_check_appends_to_evaluation_list(self) -> None:
        """``register_check`` adds a check the next probe sees."""
        server = HealthServer(
            port=_free_port(),
            service_name="test-service",
            host="127.0.0.1",
            checks=[HealthCheck(name="early", probe=lambda: True, tier=HealthTier.LIVE)],
        )
        server.register_check(HealthCheck(name="late", probe=lambda: False, tier=HealthTier.LIVE))
        status = await server.get_status(HealthTier.LIVE)

        assert status.healthy is False
        assert [c.name for c in status.components] == ["early", "late"]


class TestLifecycle:
    """start / stop idempotence."""

    @pytest.mark.asyncio
    async def test_double_start_is_a_noop(self) -> None:
        """re-calling ``start`` does not raise a bind error."""
        port = _free_port()
        server = HealthServer(
            port=port,
            service_name="test-service",
            host="127.0.0.1",
            checks=[HealthCheck(name="x", probe=lambda: True, tier=HealthTier.LIVE)],
        )
        await server.start()
        try:
            await server.start()
            status, _ = await _http_get("127.0.0.1", port, "/healthz/live")
        finally:
            await server.stop()
        assert status == 200

    @pytest.mark.asyncio
    async def test_double_stop_is_a_noop(self) -> None:
        """re-calling ``stop`` after a clean stop does nothing."""
        server = HealthServer(
            port=_free_port(),
            service_name="test-service",
            host="127.0.0.1",
            checks=[HealthCheck(name="x", probe=lambda: True, tier=HealthTier.LIVE)],
        )
        await server.start()
        await server.stop()
        await server.stop()
