"""MCP Streamable-HTTP transport for :class:`~threetears.mcp.server.McpServer`.

stdio is the v1 transport (``server.py``); this module is the v2 HTTP
transport, kept in a sibling module so ``server.py``'s v1 surface stays
untouched (line-cap discipline). the RBAC gate, identity-resolution
point, and error mapping all live on :class:`McpServer` and run through
its wired ``self._sdk_server``; this module only changes how bytes reach
that server object -- it runs the SAME SDK server over the official MCP
SDK's Streamable-HTTP transport instead of stdio.

per-request bearer identity: the Streamable-HTTP transport is stateless
(a fresh transport + server task per request) so each request re-resolves
the caller. a small ASGI middleware extracts the ``Authorization: Bearer``
token from the request scope into a request-scoped contextvar; the copy
that the SDK's per-request server task takes carries that value into
:meth:`McpServer._dispatch`, where
:class:`~threetears.mcp.auth.BearerTokenIdentityProvider` reads it back
via :func:`current_bearer_token`. stateless mode is what makes the
contextvar-per-request model correct -- a persistent (stateful) session
task would capture only the first request's token.

logging discipline: like every ``threetears.mcp`` module, this one never
writes to stdout / stderr; all logging goes through
:func:`threetears.observe.get_logger`.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator
from contextvars import ContextVar
from typing import Any

from mcp.server import Server as _SdkServer
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.types import Receive, Scope, Send
from threetears.observe import get_logger

__all__ = [
    "build_mcp_http_app",
    "current_bearer_token",
    "serve_mcp_http",
]

log = get_logger(__name__)


_BEARER_TOKEN_CONTEXT: ContextVar[str | None] = ContextVar(
    "threetears_mcp_bearer_token",
    default=None,
)
"""request-scoped bearer token populated by :class:`_BearerTokenAsgiEndpoint`.

private to this module: the transport layer is the only writer, and
:func:`current_bearer_token` is the public reader. exposing the raw
contextvar would let callers bind to an implementation detail.
"""


def current_bearer_token() -> str | None:
    """return the current request's bearer token (or ``None`` when absent).

    the canonical ``token_source`` for
    :class:`~threetears.mcp.auth.BearerTokenIdentityProvider` under this
    transport: the endpoint middleware sets the request-scoped
    contextvar this reads. resolves to ``None`` outside a request or
    when the request carried no bearer credential.

    :return: bearer token for the current request or ``None``
    :rtype: str | None
    """
    return _BEARER_TOKEN_CONTEXT.get()


def _extract_bearer_token(scope: Scope) -> str | None:
    """pull the bearer token out of an ASGI HTTP scope's ``Authorization`` header.

    accepts the ``Bearer <token>`` scheme case-insensitively; any other
    scheme (or a missing header) yields ``None`` so the downstream
    provider raises the documented ``RuntimeError`` and ``_dispatch``
    maps it to ``IDENTITY_UNAVAILABLE``.

    :param scope: ASGI connection scope
    :ptype scope: Scope
    :return: extracted bearer token or ``None``
    :rtype: str | None
    """
    token: str | None = None
    for raw_name, raw_value in scope.get("headers", []):
        if raw_name == b"authorization":
            value = raw_value.decode("latin-1").strip()
            scheme, _, credential = value.partition(" ")
            if scheme.lower() == "bearer" and credential.strip():
                token = credential.strip()
            break
    return token


class _BearerTokenAsgiEndpoint:
    """ASGI endpoint wrapping the SDK session manager with bearer extraction.

    an instance (not a plain function) so Starlette treats it as a raw
    ASGI app rather than a request/response handler. for every HTTP
    request it seeds the request-scoped bearer-token contextvar before
    delegating to :meth:`StreamableHTTPSessionManager.handle_request`,
    then resets it. under stateless mode the SDK spawns the per-request
    server task while the contextvar is set, so the token propagates
    into the dispatch that runs the RBAC gate.

    :param session_manager: SDK Streamable-HTTP session manager wrapping
        the wired MCP server
    :ptype session_manager: StreamableHTTPSessionManager
    """

    def __init__(self, session_manager: StreamableHTTPSessionManager) -> None:
        """capture the session manager.

        :param session_manager: SDK Streamable-HTTP session manager
        :ptype session_manager: StreamableHTTPSessionManager
        :return: nothing
        :rtype: None
        """
        self._session_manager = session_manager

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """seed the bearer contextvar for the request, then handle it.

        :param scope: ASGI connection scope
        :ptype scope: Scope
        :param receive: ASGI receive callable
        :ptype receive: Receive
        :param send: ASGI send callable
        :ptype send: Send
        :return: nothing
        :rtype: None
        """
        token = _extract_bearer_token(scope) if scope.get("type") == "http" else None
        reset = _BEARER_TOKEN_CONTEXT.set(token)
        try:
            await self._session_manager.handle_request(scope, receive, send)
        finally:
            _BEARER_TOKEN_CONTEXT.reset(reset)


def build_mcp_http_app(
    sdk_server: _SdkServer[Any, Any],
    *,
    path: str = "/mcp",
    stateless: bool = True,
    json_response: bool = True,
) -> Starlette:
    """build the Streamable-HTTP ASGI app that serves ``sdk_server``.

    the returned Starlette app mounts a single MCP endpoint at ``path``
    and runs the SDK's :class:`StreamableHTTPSessionManager` in its
    lifespan. the app runs the SAME wired SDK server object over HTTP;
    no handlers are re-registered and ``_dispatch`` is not re-implemented.
    consumers (e.g. the hub MCP-export server) mount or serve this app;
    :func:`serve_mcp_http` runs it standalone.

    :param sdk_server: the wired ``mcp.server.Server`` from an
        :class:`McpServer` (its ``list_tools`` / ``call_tool`` handlers
        are already registered)
    :ptype sdk_server: mcp.server.Server[Any, Any]
    :param path: URL path the MCP endpoint is served at
    :ptype path: str
    :param stateless: run the transport statelessly (fresh transport +
        server task per request); required for correct per-request
        bearer identity
    :ptype stateless: bool
    :param json_response: return plain JSON responses instead of SSE
        streams
    :ptype json_response: bool
    :return: Starlette ASGI app serving the MCP Streamable-HTTP transport
    :rtype: starlette.applications.Starlette
    """
    session_manager = StreamableHTTPSessionManager(
        app=sdk_server,
        stateless=stateless,
        json_response=json_response,
    )
    endpoint = _BearerTokenAsgiEndpoint(session_manager)

    @contextlib.asynccontextmanager
    async def _lifespan(_app: Starlette) -> AsyncIterator[None]:
        """run the SDK session manager for the app's lifetime."""
        async with session_manager.run():
            log.info(
                "MCP Streamable-HTTP transport started",
                extra={"extra_data": {"path": path, "stateless": stateless}},
            )
            yield

    return Starlette(
        routes=[Route(path, endpoint=endpoint)],
        lifespan=_lifespan,
    )


async def serve_mcp_http(
    app: Starlette,
    *,
    host: str = "127.0.0.1",
    port: int = 8080,
) -> None:
    """serve a Streamable-HTTP ASGI ``app`` over uvicorn until shutdown.

    thin standalone runner for the app produced by
    :func:`build_mcp_http_app`. consumers embedding the transport in a
    larger ASGI application mount the app directly instead of calling
    this.

    :param app: Starlette ASGI app from :func:`build_mcp_http_app`
    :ptype app: starlette.applications.Starlette
    :param host: interface to bind
    :ptype host: str
    :param port: TCP port to bind (``0`` selects an ephemeral port)
    :ptype port: int
    :return: nothing
    :rtype: None
    """
    import uvicorn

    config = uvicorn.Config(app, host=host, port=port, log_config=None)
    server = uvicorn.Server(config)
    await server.serve()
