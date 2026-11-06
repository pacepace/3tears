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

import uvicorn
from mcp.server import Server as _SdkServer
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import BaseRoute, Route
from starlette.types import Receive, Scope, Send
from threetears.observe import get_logger

__all__ = [
    "McpHttpTransport",
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
        # set by McpHttpTransport.lifespan while the session manager runs
        self.running = False

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """seed the bearer contextvar for the request, then handle it.

        a request that arrives while the session manager is not running is answered 503 naming
        the fix: the SDK would raise a bare "Task group is not initialized", which a host that
        mounted the endpoint and never entered its lifespan cannot trace back to the cause.

        :param scope: ASGI connection scope
        :ptype scope: Scope
        :param receive: ASGI receive callable
        :ptype receive: Receive
        :param send: ASGI send callable
        :ptype send: Send
        :return: nothing
        :rtype: None
        """
        if not self.running:
            log.error(
                "MCP Streamable-HTTP request before the transport's lifespan was entered",
                extra={"extra_data": {"path": scope.get("path")}},
            )
            response = JSONResponse(
                {
                    "error": "the MCP transport is not running: the host app enters "
                    "McpHttpTransport.lifespan() in its own lifespan",
                },
                status_code=503,
            )
            await response(scope, receive, send)
            return
        token = _extract_bearer_token(scope) if scope.get("type") == "http" else None
        reset = _BEARER_TOKEN_CONTEXT.set(token)
        try:
            await self._session_manager.handle_request(scope, receive, send)
        finally:
            _BEARER_TOKEN_CONTEXT.reset(reset)


class McpHttpTransport:
    """the Streamable-HTTP transport for one wired SDK server, for a host app to embed.

    a host that serves MCP inside its own Starlette app takes the endpoint (or :attr:`routes`)
    and enters :meth:`lifespan` in its own lifespan. that step is the one mounting alone misses:
    Starlette does not run a mounted sub-app's lifespan, so the SDK's session manager never starts
    and every request fails::

        transport = server.build_http_transport()

        @contextlib.asynccontextmanager
        async def lifespan(app: Starlette) -> AsyncIterator[None]:
            async with transport.lifespan():
                yield

        app = Starlette(routes=[*own_routes, *transport.routes], lifespan=lifespan)

    a request that arrives while the lifespan is not entered is answered 503 with a message
    naming it. one transport runs once: its lifespan can be entered a single time (the SDK's
    session manager is single-use), so a host that restarts builds a new transport.

    :param sdk_server: the wired ``mcp.server.Server`` from an :class:`McpServer`
    :ptype sdk_server: mcp.server.Server[Any]
    :param path: URL path :attr:`routes` serves the endpoint at
    :ptype path: str
    :param stateless: run the transport statelessly (fresh transport + server task per request);
        required for correct per-request bearer identity
    :ptype stateless: bool
    :param json_response: return plain JSON responses instead of SSE streams
    :ptype json_response: bool
    """

    def __init__(
        self,
        sdk_server: _SdkServer[Any],
        *,
        path: str = "/mcp",
        stateless: bool = True,
        json_response: bool = True,
    ) -> None:
        """build the session manager and the endpoint; nothing runs until :meth:`lifespan`.

        :param sdk_server: the wired SDK server
        :ptype sdk_server: mcp.server.Server[Any]
        :param path: URL path of the endpoint in :attr:`routes`
        :ptype path: str
        :param stateless: run the transport statelessly
        :ptype stateless: bool
        :param json_response: return plain JSON responses instead of SSE streams
        :ptype json_response: bool
        :return: nothing
        :rtype: None
        """
        self._path = path
        self._stateless = stateless
        self._session_manager = StreamableHTTPSessionManager(
            app=sdk_server,
            stateless=stateless,
            json_response=json_response,
        )
        self._endpoint = _BearerTokenAsgiEndpoint(self._session_manager)

    @property
    def endpoint(self) -> Any:
        """the raw ASGI app serving MCP requests, for a host's own ``Route`` or ``Mount``.

        typed ``Any`` so it is accepted as a Starlette ``Route`` endpoint (an ASGI callable, not a
        request handler).

        :return: the ASGI endpoint
        :rtype: Any
        """
        return self._endpoint

    @property
    def routes(self) -> list[BaseRoute]:
        """one route serving :attr:`endpoint` at ``path``, for a host's route list.

        :return: the route list
        :rtype: list[starlette.routing.BaseRoute]
        """
        return [Route(self._path, endpoint=self._endpoint)]

    @contextlib.asynccontextmanager
    async def lifespan(self) -> AsyncIterator[None]:
        """run the SDK's session manager while the block runs; the host enters it in its lifespan.

        :return: an async context manager
        :rtype: AsyncIterator[None]
        :raises RuntimeError: when entered a second time (the SDK's session manager runs once)
        """
        async with self._session_manager.run():
            self._endpoint.running = True
            log.info(
                "MCP Streamable-HTTP transport started",
                extra={"extra_data": {"path": self._path, "stateless": self._stateless}},
            )
            try:
                yield
            finally:
                self._endpoint.running = False


def build_mcp_http_app(
    sdk_server: _SdkServer[Any],
    *,
    path: str = "/mcp",
    stateless: bool = True,
    json_response: bool = True,
) -> Starlette:
    """build the Streamable-HTTP ASGI app that serves ``sdk_server``.

    the returned Starlette app serves a single MCP endpoint at ``path``
    and runs the SDK's :class:`StreamableHTTPSessionManager` in its
    lifespan. the app runs the SAME wired SDK server object over HTTP;
    no handlers are re-registered and ``_dispatch`` is not re-implemented.
    :func:`serve_mcp_http` runs it standalone. a host embedding MCP in
    its own app uses :class:`McpHttpTransport` instead, since a mounted
    app's lifespan does not run.

    :param sdk_server: the wired ``mcp.server.Server`` from an
        :class:`McpServer` (its ``list_tools`` / ``call_tool`` handlers
        are already registered)
    :ptype sdk_server: mcp.server.Server[Any]
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
    transport = McpHttpTransport(sdk_server, path=path, stateless=stateless, json_response=json_response)

    @contextlib.asynccontextmanager
    async def _lifespan(_app: Starlette) -> AsyncIterator[None]:
        """run the transport for the app's lifetime."""
        async with transport.lifespan():
            yield

    return Starlette(routes=transport.routes, lifespan=_lifespan)


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
    config = uvicorn.Config(app, host=host, port=port, log_config=None)
    server = uvicorn.Server(config)
    await server.serve()
