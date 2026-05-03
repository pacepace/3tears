"""ASGI middleware for cross-cutting request context.

Provides:

- :class:`CorrelationMiddleware` — extracts (or generates) a request
  correlation ID, publishes it via :func:`set_context` so structured
  logs and ``@traced`` spans pick it up automatically, and echoes it
  back on the HTTP response.  Works for both HTTP and WebSocket scopes.

- :class:`OTelMiddleware` — creates a root SERVER span for WebSocket
  connections.  HTTP root spans are emitted by ``FastAPIInstrumentor``;
  this middleware fills the gap WebSocket leaves.  Degrades to a
  passthrough when OpenTelemetry is not installed.

Both classes are raw ASGI.  Starlette's ``BaseHTTPMiddleware`` has
known issues with streaming responses and exception propagation, and
cannot wrap WebSocket scopes; raw ASGI avoids both pitfalls.

Requires the ``asgi`` extra: ``pip install 3tears-observe[asgi]``
(adds ``uuid-utils`` for UUID7 generation).  The middleware is
deliberately framework-agnostic — it does not import from starlette,
fastapi, or any other ASGI framework.  It uses inline ASGI type
aliases per the framework spec so consumers using starlette, fastapi,
hypercorn, or any conforming framework can mount it without forcing
a transitive dep.  The enforcement test
``test_middleware_does_not_import_framework`` in the observe test
suite locks this convention.

The middleware is intentionally not re-exported from
:mod:`threetears.observe`; consumers wanting the middleware import
from :mod:`threetears.observe.middleware` directly so that pure-logger
consumers without ``uuid-utils`` installed are not affected.

Design decisions (locked):

- Header name is fixed at ``X-Correlation-ID`` (HTTP convention).
- Context key is fixed at ``cid``.  Cross-cutting standardisation
  matters more than per-app configurability; consumers needing a
  different convention write their own middleware.
- ``request.state.correlation_id`` is NOT set.  Consumers read the
  value via :func:`threetears.observe.get_context` (or a small
  consumer-side typed helper that wraps the lookup).  This is the
  whole point of ContextVar-backed propagation: no per-handler
  threading, no manual extraction.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, MutableMapping
from typing import Any

from uuid_utils import uuid7

from threetears.observe.logging import set_context

__all__ = ["ASGIApp", "CorrelationMiddleware", "Message", "OTelMiddleware", "Receive", "Scope", "Send"]


# ---------------------------------------------------------------------------
# Inline ASGI type aliases (per the ASGI spec).  We define these locally
# rather than importing from starlette / asgiref so the observe package
# keeps zero hard-dep on any ASGI framework.  Any conforming framework
# (starlette, fastapi, hypercorn, etc.) emits scopes / messages that
# match these structural types.
# ---------------------------------------------------------------------------

Scope = MutableMapping[str, Any]
Message = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]


_CORRELATION_HEADER = b"x-correlation-id"
_MAX_CORRELATION_LENGTH = 128
_HEALTH_PATHS = frozenset(
    {
        "/healthz",
        "/healthz/ready",
        "/healthz/live",
        "/metrics",
    },
)


class CorrelationMiddleware:
    """ASGI middleware that publishes a correlation ID into request context.

    Extracts the ``X-Correlation-ID`` header (case-insensitive) from the
    incoming request.  If the header is missing, empty, longer than 128
    characters, or non-ASCII, a fresh UUID7 is generated.  The value is
    published to the request-local logging context via
    :func:`set_context` so that every log line and traced span emitted
    during the request carries the same ``cid`` tag automatically.

    For HTTP requests, the correlation ID is echoed back on the response
    via the ``X-Correlation-ID`` response header.  WebSocket scopes only
    receive the context publish — WS has no HTTP response headers
    post-handshake, so the correlation tag flows via log lines and span
    attributes only.

    Non-HTTP, non-WebSocket scopes (e.g. ASGI ``lifespan``) pass through
    unchanged.

    The ``cid`` context key is removed from the request-local context
    on exit (success or failure), preserving any other context keys
    that downstream middleware or handlers may have set.
    """

    def __init__(self, app: ASGIApp) -> None:
        """Initialize middleware.

        :param app: next ASGI application in chain
        :ptype app: ASGIApp
        """
        self._app = app

    async def __call__(
        self, scope: Scope, receive: Receive, send: Send,
    ) -> None:
        """Process request through middleware.

        :param scope: ASGI connection scope
        :ptype scope: Scope
        :param receive: ASGI receive callable
        :ptype receive: Receive
        :param send: ASGI send callable
        :ptype send: Send
        :return: nothing
        :rtype: None
        """
        if scope["type"] not in ("http", "websocket"):
            await self._app(scope, receive, send)
            return

        correlation_id = self._resolve_correlation_id(scope)
        set_context(cid=correlation_id)

        try:
            if scope["type"] == "http":

                async def send_with_header(message: Message) -> None:
                    """Inject ``X-Correlation-ID`` on response start."""
                    if message["type"] == "http.response.start":
                        headers = list(message.get("headers", []))
                        headers.append(
                            (
                                _CORRELATION_HEADER,
                                correlation_id.encode("ascii"),
                            ),
                        )
                        message["headers"] = headers
                    await send(message)

                await self._app(scope, receive, send_with_header)
            else:
                await self._app(scope, receive, send)
        finally:
            set_context(cid=None)

    @staticmethod
    def _resolve_correlation_id(scope: Scope) -> str:
        """Return correlation ID from header or generate a new UUID7.

        :param scope: ASGI scope containing headers
        :ptype scope: Scope
        :return: correlation ID string
        :rtype: str
        """
        for raw_key, raw_value in scope.get("headers", ()):
            if raw_key.lower() == _CORRELATION_HEADER:
                try:
                    candidate = raw_value.decode("ascii")
                except UnicodeDecodeError:
                    return str(uuid7())
                if candidate and len(candidate) <= _MAX_CORRELATION_LENGTH:
                    return candidate
                return str(uuid7())
        return str(uuid7())


class OTelMiddleware:
    """ASGI middleware creating a root SERVER span for WebSocket scopes.

    HTTP root spans are emitted by ``FastAPIInstrumentor``; this
    middleware fills the corresponding gap for WebSocket connections so
    that traces for streaming endpoints have a SERVER-kind span as the
    root.  Health-check paths are excluded so probes do not generate
    noise spans.

    Degrades to a passthrough when OpenTelemetry is not installed (the
    ``opentelemetry`` import is guarded; missing the package means the
    middleware does no work but does not raise).

    Span attribute conventions follow :func:`threetears.observe.traced`:
    ``ctx.*`` attributes are auto-injected from
    :func:`threetears.observe.get_context` by ``@traced``-decorated
    functions called within the span.  This middleware itself only
    sets ``http.route``; consumer-specific tagging is the consumer's
    job and should happen via ``set_context`` before the span is
    consulted.
    """

    def __init__(self, app: ASGIApp) -> None:
        """Initialize middleware.

        :param app: next ASGI application in chain
        :ptype app: ASGIApp
        """
        self._app = app

    async def __call__(
        self, scope: Scope, receive: Receive, send: Send,
    ) -> None:
        """Wrap WebSocket scope in a root SERVER span; pass others through.

        :param scope: ASGI connection scope
        :ptype scope: Scope
        :param receive: ASGI receive callable
        :ptype receive: Receive
        :param send: ASGI send callable
        :ptype send: Send
        :return: nothing
        :rtype: None
        """
        if scope["type"] != "websocket":
            await self._app(scope, receive, send)
            return

        path = scope.get("path", "")
        if path in _HEALTH_PATHS:
            await self._app(scope, receive, send)
            return

        try:
            from opentelemetry import trace  # noqa: PLC0415
            from opentelemetry.trace import (  # noqa: PLC0415
                SpanKind,
                StatusCode,
            )
        except ImportError:
            await self._app(scope, receive, send)
            return

        tracer = trace.get_tracer("threetears.observe.middleware")
        with tracer.start_as_current_span(
            f"WS {path}", kind=SpanKind.SERVER,
        ) as span:
            span.set_attribute("http.route", path)
            try:
                await self._app(scope, receive, send)
            except Exception as exc:
                span.record_exception(exc)
                span.set_status(StatusCode.ERROR, str(exc))
                raise
            else:
                span.set_status(StatusCode.OK)
