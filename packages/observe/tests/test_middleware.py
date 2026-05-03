"""Tests for threetears.observe.middleware."""

from __future__ import annotations

import ast
import inspect
from typing import Any

import pytest

from threetears.observe.logging import clear_context, get_context
from threetears.observe.middleware import (
    CorrelationMiddleware,
    OTelMiddleware,
    _CORRELATION_HEADER,
    _MAX_CORRELATION_LENGTH,
)


@pytest.fixture(autouse=True)
def _clean_context():
    """Reset context between tests so cross-test bleed cannot mask bugs."""
    clear_context()
    yield
    clear_context()


def _http_scope(headers: list[tuple[bytes, bytes]] | None = None) -> dict[str, Any]:
    """Build a minimal ASGI HTTP scope for tests."""
    return {
        "type": "http",
        "method": "GET",
        "path": "/api/v1/example",
        "headers": headers or [],
    }


def _websocket_scope(
    path: str = "/ws",
    headers: list[tuple[bytes, bytes]] | None = None,
) -> dict[str, Any]:
    """Build a minimal ASGI WebSocket scope for tests."""
    return {
        "type": "websocket",
        "path": path,
        "headers": headers or [],
    }


async def _noop_receive() -> dict[str, Any]:
    """Return a single empty HTTP request body message."""
    return {"type": "http.request", "body": b"", "more_body": False}


class TestCorrelationMiddlewareHeaderExtraction:
    """Header extraction and UUID7 fallback behavior."""

    @pytest.mark.asyncio
    async def test_uses_header_when_valid(self):
        sent: list[dict[str, Any]] = []
        observed_cid: list[str | None] = []

        async def app(scope: Any, receive: Any, send: Any) -> None:
            observed_cid.append(get_context().get("cid"))
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b""})

        async def send(message: dict[str, Any]) -> None:
            sent.append(message)

        middleware = CorrelationMiddleware(app)
        scope = _http_scope([(_CORRELATION_HEADER, b"abc-123-xyz")])
        await middleware(scope, _noop_receive, send)

        assert observed_cid == ["abc-123-xyz"]
        start = next(m for m in sent if m["type"] == "http.response.start")
        echo = dict(start["headers"]).get(_CORRELATION_HEADER)
        assert echo == b"abc-123-xyz"

    @pytest.mark.asyncio
    async def test_generates_uuid7_when_header_absent(self):
        observed_cid: list[str | None] = []

        async def app(scope: Any, receive: Any, send: Any) -> None:
            observed_cid.append(get_context().get("cid"))
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b""})

        async def send(_: dict[str, Any]) -> None:
            pass

        middleware = CorrelationMiddleware(app)
        await middleware(_http_scope(), _noop_receive, send)

        assert observed_cid[0] is not None
        assert len(observed_cid[0]) > 0

    @pytest.mark.asyncio
    async def test_generates_uuid7_when_header_empty(self):
        observed_cid: list[str | None] = []

        async def app(scope: Any, receive: Any, send: Any) -> None:
            observed_cid.append(get_context().get("cid"))
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b""})

        async def send(_: dict[str, Any]) -> None:
            pass

        middleware = CorrelationMiddleware(app)
        scope = _http_scope([(_CORRELATION_HEADER, b"")])
        await middleware(scope, _noop_receive, send)

        assert observed_cid[0] != ""
        assert observed_cid[0] is not None

    @pytest.mark.asyncio
    async def test_generates_uuid7_when_header_too_long(self):
        observed_cid: list[str | None] = []

        async def app(scope: Any, receive: Any, send: Any) -> None:
            observed_cid.append(get_context().get("cid"))
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b""})

        async def send(_: dict[str, Any]) -> None:
            pass

        long_value = b"x" * (_MAX_CORRELATION_LENGTH + 1)
        middleware = CorrelationMiddleware(app)
        scope = _http_scope([(_CORRELATION_HEADER, long_value)])
        await middleware(scope, _noop_receive, send)

        assert observed_cid[0] != long_value.decode("ascii")
        assert observed_cid[0] is not None
        assert len(observed_cid[0]) <= _MAX_CORRELATION_LENGTH

    @pytest.mark.asyncio
    async def test_generates_uuid7_when_header_non_ascii(self):
        observed_cid: list[str | None] = []

        async def app(scope: Any, receive: Any, send: Any) -> None:
            observed_cid.append(get_context().get("cid"))
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b""})

        async def send(_: dict[str, Any]) -> None:
            pass

        middleware = CorrelationMiddleware(app)
        scope = _http_scope([(_CORRELATION_HEADER, b"\xff\xfe\xc3")])
        await middleware(scope, _noop_receive, send)

        assert observed_cid[0] is not None

    @pytest.mark.asyncio
    async def test_header_lookup_is_case_insensitive(self):
        observed_cid: list[str | None] = []

        async def app(scope: Any, receive: Any, send: Any) -> None:
            observed_cid.append(get_context().get("cid"))
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b""})

        async def send(_: dict[str, Any]) -> None:
            pass

        middleware = CorrelationMiddleware(app)
        scope = _http_scope([(b"X-Correlation-ID", b"mixed-case-id")])
        await middleware(scope, _noop_receive, send)

        assert observed_cid[0] == "mixed-case-id"


class TestCorrelationMiddlewareContextLifecycle:
    """Context publish + clear semantics across requests."""

    @pytest.mark.asyncio
    async def test_clears_cid_on_exit(self):
        async def app(scope: Any, receive: Any, send: Any) -> None:
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b""})

        async def send(_: dict[str, Any]) -> None:
            pass

        middleware = CorrelationMiddleware(app)
        scope = _http_scope([(_CORRELATION_HEADER, b"abc")])
        await middleware(scope, _noop_receive, send)

        assert "cid" not in get_context()

    @pytest.mark.asyncio
    async def test_clears_cid_on_app_exception(self):
        async def app(scope: Any, receive: Any, send: Any) -> None:
            raise RuntimeError("boom")

        async def send(_: dict[str, Any]) -> None:
            pass

        middleware = CorrelationMiddleware(app)
        scope = _http_scope([(_CORRELATION_HEADER, b"abc")])
        with pytest.raises(RuntimeError, match="boom"):
            await middleware(scope, _noop_receive, send)

        assert "cid" not in get_context()

    @pytest.mark.asyncio
    async def test_preserves_other_context_keys_set_by_inner_app(self):
        from threetears.observe.logging import set_context

        async def app(scope: Any, receive: Any, send: Any) -> None:
            set_context(other_key="set-by-inner-app")
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b""})

        async def send(_: dict[str, Any]) -> None:
            pass

        middleware = CorrelationMiddleware(app)
        scope = _http_scope([(_CORRELATION_HEADER, b"abc")])
        await middleware(scope, _noop_receive, send)

        ctx = get_context()
        assert "cid" not in ctx
        assert ctx.get("other_key") == "set-by-inner-app"


class TestCorrelationMiddlewareScopeRouting:
    """HTTP, WebSocket, and lifespan scope routing."""

    @pytest.mark.asyncio
    async def test_websocket_publishes_context_but_does_not_inject_header(self):
        observed_cid: list[str | None] = []
        sent: list[dict[str, Any]] = []

        async def app(scope: Any, receive: Any, send: Any) -> None:
            observed_cid.append(get_context().get("cid"))
            await send({"type": "websocket.accept"})

        async def send(message: dict[str, Any]) -> None:
            sent.append(message)

        middleware = CorrelationMiddleware(app)
        scope = _websocket_scope(headers=[(_CORRELATION_HEADER, b"ws-id")])
        await middleware(scope, _noop_receive, send)

        assert observed_cid == ["ws-id"]
        accept = next(m for m in sent if m["type"] == "websocket.accept")
        assert _CORRELATION_HEADER not in dict(accept.get("headers", []))

    @pytest.mark.asyncio
    async def test_lifespan_passes_through_unchanged(self):
        called = False

        async def app(scope: Any, receive: Any, send: Any) -> None:
            nonlocal called
            called = True
            assert "cid" not in get_context()

        async def send(_: dict[str, Any]) -> None:
            pass

        middleware = CorrelationMiddleware(app)
        scope = {"type": "lifespan"}
        await middleware(scope, _noop_receive, send)

        assert called
        assert "cid" not in get_context()


class TestCorrelationMiddlewareIntegrationWithTraced:
    """The whole point: ``@traced`` spans pick up the cid automatically."""

    @pytest.mark.asyncio
    async def test_inner_traced_function_sees_cid_via_get_context(self):
        from threetears.observe import traced

        observed: list[str | None] = []

        @traced
        async def inner_function() -> None:
            observed.append(get_context().get("cid"))

        async def app(scope: Any, receive: Any, send: Any) -> None:
            await inner_function()
            await send({"type": "http.response.start", "status": 200, "headers": []})
            await send({"type": "http.response.body", "body": b""})

        async def send(_: dict[str, Any]) -> None:
            pass

        middleware = CorrelationMiddleware(app)
        scope = _http_scope([(_CORRELATION_HEADER, b"trace-cid")])
        await middleware(scope, _noop_receive, send)

        assert observed == ["trace-cid"]


class TestOTelMiddlewareScopeRouting:
    """Scope routing for the WebSocket tracer."""

    @pytest.mark.asyncio
    async def test_http_passes_through(self):
        called = False

        async def app(scope: Any, receive: Any, send: Any) -> None:
            nonlocal called
            called = True

        async def send(_: dict[str, Any]) -> None:
            pass

        middleware = OTelMiddleware(app)
        await middleware(_http_scope(), _noop_receive, send)
        assert called

    @pytest.mark.asyncio
    async def test_websocket_health_path_passes_through(self):
        called = False

        async def app(scope: Any, receive: Any, send: Any) -> None:
            nonlocal called
            called = True

        async def send(_: dict[str, Any]) -> None:
            pass

        middleware = OTelMiddleware(app)
        scope = _websocket_scope(path="/healthz")
        await middleware(scope, _noop_receive, send)
        assert called

    @pytest.mark.asyncio
    async def test_websocket_normal_path_runs_app(self):
        # We are not asserting span emission here (OTel may or may not be
        # installed in this test environment); we are asserting the
        # middleware does not break the request flow.
        called = False

        async def app(scope: Any, receive: Any, send: Any) -> None:
            nonlocal called
            called = True

        async def send(_: dict[str, Any]) -> None:
            pass

        middleware = OTelMiddleware(app)
        scope = _websocket_scope(path="/ws")
        await middleware(scope, _noop_receive, send)
        assert called

    @pytest.mark.asyncio
    async def test_websocket_app_exception_propagates(self):
        async def app(scope: Any, receive: Any, send: Any) -> None:
            raise RuntimeError("ws-boom")

        async def send(_: dict[str, Any]) -> None:
            pass

        middleware = OTelMiddleware(app)
        scope = _websocket_scope(path="/ws")
        with pytest.raises(RuntimeError, match="ws-boom"):
            await middleware(scope, _noop_receive, send)


class TestFrameworkAgnosticContract:
    """The middleware module must remain framework-agnostic.

    No imports from starlette, fastapi, asgiref, hypercorn, or other
    ASGI frameworks.  ASGI types are defined inline per the spec so any
    conforming framework can mount the middleware without forcing a
    transitive dependency on a specific framework.
    """

    def _assert_no_imports_from(self, banned_prefixes: tuple[str, ...]) -> None:
        from threetears.observe import middleware as mw_mod

        source_path = inspect.getfile(mw_mod)
        with open(source_path) as f:
            source = f.read()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    for banned in banned_prefixes:
                        assert not alias.name.startswith(banned), (
                            f"middleware.py imports {alias.name}; "
                            f"observe must remain framework-agnostic."
                        )
            elif isinstance(node, ast.ImportFrom):
                if node.module is not None:
                    for banned in banned_prefixes:
                        assert not node.module.startswith(banned), (
                            f"middleware.py imports from {node.module}; "
                            f"observe must remain framework-agnostic."
                        )

    def test_no_starlette_import(self) -> None:
        self._assert_no_imports_from(("starlette",))

    def test_no_fastapi_import(self) -> None:
        self._assert_no_imports_from(("fastapi",))

    def test_no_asgiref_import(self) -> None:
        self._assert_no_imports_from(("asgiref",))

    def test_no_hypercorn_import(self) -> None:
        self._assert_no_imports_from(("hypercorn",))
