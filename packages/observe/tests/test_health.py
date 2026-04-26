"""tests for ``threetears.observe.health.HealthServer``.

cover the surface every consumer will rely on:

- ``GET /healthz`` returns 200 when every check passes
- ``GET /healthz`` returns 503 with the failing check's name in the
  body when one check returns False
- ``GET /healthz`` returns 503 when a check raises (raising is treated
  as failure -- a check that crashes cannot be alive)
- ``GET /readyz`` is a synonym for ``/healthz`` so kubernetes
  readiness probes can use either path
- unknown paths return 404 without touching the check list
- non-GET methods return 400 (the surface is intentionally narrow)
- ``start`` is idempotent (re-call is a no-op)
- ``stop`` is idempotent (re-call after stop is a no-op)
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
    host: str, port: int, method: str, path: str,
    extra_headers: str = "",
) -> tuple[int, str]:
    """tiny request helper supporting non-GET methods + extra headers."""
    reader, writer = await asyncio.open_connection(host, port)
    writer.write(
        (
            f"{method} {path} HTTP/1.1\r\n"
            f"Host: {host}\r\n"
            f"Connection: close\r\n"
            f"{extra_headers}"
            f"\r\n"
        ).encode(),
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


class TestHealthServerHealthz:
    """``/healthz`` endpoint contract."""

    @pytest.mark.asyncio
    async def test_returns_200_when_every_check_passes(self) -> None:
        """all checks True -> ``200 OK``."""
        port = _free_port()
        server = HealthServer(
            port=port,
            service_name="test-service",
            host="127.0.0.1",
            checks=[
                HealthCheck(name="nats", probe=lambda: True),
                HealthCheck(name="catalog", probe=lambda: True),
            ],
        )
        await server.start()
        try:
            status, body = await _http_get("127.0.0.1", port, "/healthz")
        finally:
            await server.stop()
        assert status == 200
        assert "ok" in body

    @pytest.mark.asyncio
    async def test_returns_503_with_failing_check_name(self) -> None:
        """one check False -> ``503`` and body names the failing check.

        operators read the body to know which subsystem tripped
        without having to grep the service logs.
        """
        port = _free_port()
        server = HealthServer(
            port=port,
            service_name="test-service",
            host="127.0.0.1",
            checks=[
                HealthCheck(name="nats", probe=lambda: True),
                HealthCheck(name="catalog", probe=lambda: False),
            ],
        )
        await server.start()
        try:
            status, body = await _http_get("127.0.0.1", port, "/healthz")
        finally:
            await server.stop()
        assert status == 503
        assert "catalog" in body

    @pytest.mark.asyncio
    async def test_check_raising_is_treated_as_failure(self) -> None:
        """a probe that raises -> ``503`` (a crashing probe cannot be alive).

        belt-and-braces: a misbehaving probe must not bring down the
        listener. the failure surfaces via 503 + the crashed check's
        name in the body.
        """
        def _crash() -> bool:
            raise RuntimeError("simulated probe crash")

        port = _free_port()
        server = HealthServer(
            port=port,
            service_name="test-service",
            host="127.0.0.1",
            checks=[HealthCheck(name="boom", probe=_crash)],
        )
        await server.start()
        try:
            status, body = await _http_get("127.0.0.1", port, "/healthz")
        finally:
            await server.stop()
        assert status == 503
        assert "boom" in body

    @pytest.mark.asyncio
    async def test_short_circuits_on_first_failure(self) -> None:
        """check evaluation stops at the first failure -- subsequent
        checks are not invoked.

        this is observable through the body content: a slow downstream
        probe should not run when an upstream one already failed
        (otherwise the probe payload could time out under degraded
        conditions).
        """
        invoked: list[str] = []

        def _track(name: str, ok: bool):
            def _check() -> bool:
                invoked.append(name)
                return ok
            return _check

        port = _free_port()
        server = HealthServer(
            port=port,
            service_name="test-service",
            host="127.0.0.1",
            checks=[
                HealthCheck(name="upstream", probe=_track("upstream", False)),
                HealthCheck(name="downstream", probe=_track("downstream", True)),
            ],
        )
        await server.start()
        try:
            status, _body = await _http_get("127.0.0.1", port, "/healthz")
        finally:
            await server.stop()
        assert status == 503
        # downstream MUST NOT have been touched.
        assert invoked == ["upstream"]


class TestHealthServerReadyz:
    """``/readyz`` endpoint -- alias for ``/healthz`` so kubernetes
    readiness probes can use either path.
    """

    @pytest.mark.asyncio
    async def test_readyz_returns_200_on_pass(self) -> None:
        """``/readyz`` mirrors ``/healthz`` ``200`` on success."""
        port = _free_port()
        server = HealthServer(
            port=port,
            service_name="test-service",
            host="127.0.0.1",
            checks=[HealthCheck(name="nats", probe=lambda: True)],
        )
        await server.start()
        try:
            status, body = await _http_get("127.0.0.1", port, "/readyz")
        finally:
            await server.stop()
        assert status == 200
        assert "ok" in body


class TestHealthServerErrorPaths:
    """unknown paths / methods."""

    @pytest.mark.asyncio
    async def test_unknown_path_returns_404(self) -> None:
        """``GET /unknown`` -> ``404`` without touching the check list."""
        invoked: list[str] = []

        def _track() -> bool:
            invoked.append("ran")
            return True

        port = _free_port()
        server = HealthServer(
            port=port,
            service_name="test-service",
            host="127.0.0.1",
            checks=[HealthCheck(name="x", probe=_track)],
        )
        await server.start()
        try:
            status, _body = await _http_get("127.0.0.1", port, "/some/random/path")
        finally:
            await server.stop()
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
            checks=[HealthCheck(name="x", probe=lambda: True)],
        )
        await server.start()
        try:
            status, _body = await _http_request(
                "127.0.0.1", port, "POST", "/healthz",
            )
        finally:
            await server.stop()
        assert status == 400


class TestHealthServerJsonResponse:
    """JSON response path for richer drill-in."""

    @pytest.mark.asyncio
    async def test_format_query_returns_json_200_when_healthy(self) -> None:
        """``?format=json`` -> ``application/json`` body with full status."""
        port = _free_port()
        server = HealthServer(
            port=port,
            service_name="test-service",
            host="127.0.0.1",
            checks=[
                HealthCheck(name="nats", probe=lambda: True),
                HealthCheck(name="catalog", probe=lambda: True),
            ],
        )
        await server.start()
        try:
            status, body = await _http_get(
                "127.0.0.1", port, "/healthz?format=json",
            )
        finally:
            await server.stop()
        assert status == 200
        payload = json.loads(body)
        assert payload["service"] == "test-service"
        assert payload["healthy"] is True
        assert [c["name"] for c in payload["components"]] == ["nats", "catalog"]
        assert all(c["healthy"] for c in payload["components"])

    @pytest.mark.asyncio
    async def test_accept_json_header_returns_json(self) -> None:
        """``Accept: application/json`` -> JSON without query string."""
        port = _free_port()
        server = HealthServer(
            port=port,
            service_name="test-service",
            host="127.0.0.1",
            checks=[HealthCheck(name="nats", probe=lambda: True)],
        )
        await server.start()
        try:
            status, body = await _http_request(
                "127.0.0.1", port, "GET", "/healthz",
                extra_headers="Accept: application/json\r\n",
            )
        finally:
            await server.stop()
        assert status == 200
        payload = json.loads(body)
        assert payload["service"] == "test-service"
        assert payload["healthy"] is True

    @pytest.mark.asyncio
    async def test_json_response_lists_every_component_on_failure(self) -> None:
        """JSON path does NOT short-circuit; every component appears with
        its individual ``healthy`` flag.

        operators want the full picture from the JSON query (which
        thing is broken AND what's still alive). short-circuit
        behaviour stays in the plain-text path where docker probes
        only care about the first failure.
        """
        port = _free_port()
        server = HealthServer(
            port=port,
            service_name="test-service",
            host="127.0.0.1",
            checks=[
                HealthCheck(name="upstream", probe=lambda: False),
                HealthCheck(name="downstream", probe=lambda: True),
            ],
        )
        await server.start()
        try:
            status, body = await _http_get(
                "127.0.0.1", port, "/healthz?format=json",
            )
        finally:
            await server.stop()
        assert status == 503
        payload = json.loads(body)
        assert payload["healthy"] is False
        names = {c["name"]: c["healthy"] for c in payload["components"]}
        assert names == {"upstream": False, "downstream": True}

    @pytest.mark.asyncio
    async def test_json_response_carries_failure_detail(self) -> None:
        """a probe that raises -> the exception message lands in
        ``component.detail``.
        """
        def _crash() -> bool:
            raise RuntimeError("simulated probe crash")

        port = _free_port()
        server = HealthServer(
            port=port,
            service_name="test-service",
            host="127.0.0.1",
            checks=[HealthCheck(name="boom", probe=_crash)],
        )
        await server.start()
        try:
            status, body = await _http_get(
                "127.0.0.1", port, "/healthz?format=json",
            )
        finally:
            await server.stop()
        assert status == 503
        payload = json.loads(body)
        assert payload["components"][0]["healthy"] is False
        assert "simulated probe crash" in payload["components"][0]["detail"]


class TestHealthServerStatusAccessor:
    """``get_status()`` -- in-process status read."""

    def test_get_status_returns_health_status_dataclass(self) -> None:
        """``get_status()`` returns a :class:`HealthStatus` value
        callers can introspect without HTTP.
        """
        server = HealthServer(
            port=0,
            service_name="test-service",
            host="127.0.0.1",
            checks=[
                HealthCheck(name="nats", probe=lambda: True),
                HealthCheck(name="catalog", probe=lambda: False),
            ],
        )
        status = server.get_status()
        assert isinstance(status, HealthStatus)
        assert status.service == "test-service"
        assert status.healthy is False
        assert status.components == [
            ComponentStatus(name="nats", healthy=True),
            ComponentStatus(name="catalog", healthy=False),
        ]

    def test_register_check_appends_to_evaluation_list(self) -> None:
        """``register_check`` adds a check that the next probe sees.

        services that wire dependencies lazily can register the
        corresponding check after :meth:`start` -- the appended
        check counts on the next ``get_status`` evaluation.
        """
        server = HealthServer(
            port=0,
            service_name="test-service",
            host="127.0.0.1",
            checks=[HealthCheck(name="initial", probe=lambda: True)],
        )
        server.register_check(HealthCheck(name="late", probe=lambda: False))
        status = server.get_status()
        assert status.healthy is False
        assert [c.name for c in status.components] == ["initial", "late"]


class TestHealthServerLifecycle:
    """``start`` / ``stop`` idempotency."""

    @pytest.mark.asyncio
    async def test_start_is_idempotent(self) -> None:
        """re-calling ``start`` after start is a no-op (does not re-bind)."""
        port = _free_port()
        server = HealthServer(
            port=port,
            service_name="test-service",
            host="127.0.0.1",
            checks=[HealthCheck(name="x", probe=lambda: True)],
        )
        await server.start()
        # re-call must not raise even though the port is already bound.
        await server.start()
        await server.stop()

    @pytest.mark.asyncio
    async def test_stop_is_idempotent(self) -> None:
        """re-calling ``stop`` after stop is a no-op."""
        port = _free_port()
        server = HealthServer(
            port=port,
            service_name="test-service",
            host="127.0.0.1",
            checks=[HealthCheck(name="x", probe=lambda: True)],
        )
        await server.start()
        await server.stop()
        # re-call after stop must not raise.
        await server.stop()
