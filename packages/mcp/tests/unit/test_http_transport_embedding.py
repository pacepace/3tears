"""a host app serves MCP inside its own Starlette app through :class:`McpHttpTransport`.

the host owns the app and its lifespan: it adds the transport's routes (or mounts its endpoint)
and enters ``transport.lifespan()`` in its own lifespan, the step a mounted sub-app's lifespan
never gets. driven by a real MCP client through the host app, bearer identity included, so the
request-scoped token is shown to reach dispatch on the host's path as well as on
``build_http_app``'s.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import httpx
import mcp
import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import PlainTextResponse
from starlette.routing import BaseRoute, Mount, Route
from threetears.mcp import McpHttpTransport
from threetears.mcp.auth import BearerTokenIdentityProvider, Identity
from threetears.mcp.http_server import current_bearer_token
from threetears.mcp.server import McpServer
from threetears.mcp.tool import McpTool, ToolRegistry

_CALLER = uuid4()


async def _resolve(token: str) -> Identity:
    """the host's resolver: one known key, everything else unknown."""
    if token == "client-key":
        return Identity(principal_type="user", principal_id=_CALLER)
    raise RuntimeError("unknown client key")


async def _whoami() -> dict[str, Any]:
    """the ``whoami`` op: the token the request carried, as dispatch saw it."""
    return {"token": current_bearer_token()}


def _server() -> McpServer:
    """a server with bearer identity and an authorizer that allows every call (the host authorizes)."""
    registry = ToolRegistry()
    registry.register(
        McpTool(
            name="whoami",
            description="the caller's token",
            input_schema={"type": "object", "properties": {}},
            required_permission="host.whoami",
            handler=_whoami,
        ),
    )
    authorizer = MagicMock()
    authorizer.allows = AsyncMock(return_value=True)
    authorizer.start = AsyncMock()
    authorizer.stop = AsyncMock()
    return McpServer(
        name="host-test",
        identity_provider=BearerTokenIdentityProvider(resolver=_resolve, token_source=current_bearer_token),
        authorizer=authorizer,
        registry=registry,
    )


async def _health(_request: Request) -> PlainTextResponse:
    """one of the host's own routes."""
    return PlainTextResponse("ok")


def _host_app(transport: McpHttpTransport, mcp_routes: list[BaseRoute]) -> Starlette:
    """the host's app: its own route, the MCP routes it chose, and a lifespan that runs the transport."""

    @contextlib.asynccontextmanager
    async def _lifespan(_app: Starlette) -> AsyncIterator[None]:
        async with transport.lifespan():
            yield

    return Starlette(routes=[Route("/health", _health), *mcp_routes], lifespan=_lifespan)


def _http(app: Starlette) -> httpx.AsyncClient:
    """an in-process HTTP client for ``app`` presenting the known client key."""
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://host.test",
        headers={"Authorization": "Bearer client-key"},
    )


@asynccontextmanager
async def _session(app: Starlette, url: str) -> AsyncIterator[ClientSession]:
    """an initialized MCP client session at ``url`` through ``app``, its lifespan running."""
    async with app.router.lifespan_context(app), _http(app) as client:
        async with streamable_http_client(url, http_client=client) as streams:
            async with ClientSession(streams[0], streams[1]) as session:  # 1.x yields a third value
                await session.initialize()
                yield session


def _token_seen(result: Any) -> str:
    """the token the ``whoami`` op reported."""
    token: str = json.loads(result.content[0].text)["token"]
    return token


class TestAHostServesMcpOnItsOwnApp:
    @pytest.mark.asyncio
    async def test_the_transports_routes_in_the_hosts_route_list(self) -> None:
        server = _server()
        await server.start()
        transport = server.build_http_transport(path="/mcp")
        app = _host_app(transport, transport.routes)
        try:
            async with _session(app, "http://host.test/mcp") as session:
                listed = await session.list_tools()
                result = await session.call_tool("whoami", {})
            async with _http(app) as client:
                health = await client.get("/health")
        finally:
            await server.stop()
        assert [tool.name for tool in listed.tools] == ["whoami"]
        assert result.model_dump(by_alias=True)["isError"] is False
        assert _token_seen(result) == "client-key"
        assert health.text == "ok"

    @pytest.mark.asyncio
    async def test_the_endpoint_under_a_mount(self) -> None:
        # a Mount matches below its prefix, so the client addresses /mcp/ (POST /mcp would be
        # redirected); a host that wants exactly /mcp adds transport.routes instead
        server = _server()
        await server.start()
        transport = server.build_http_transport()
        app = _host_app(transport, [Mount("/mcp", app=transport.endpoint)])
        try:
            async with _session(app, "http://host.test/mcp/") as session:
                result = await session.call_tool("whoami", {})
        finally:
            await server.stop()
        assert _token_seen(result) == "client-key"


class TestTheLifespanTheHostMustEnter:
    @pytest.mark.asyncio
    async def test_a_request_before_the_lifespan_is_answered_503_naming_it(self) -> None:
        server = _server()
        await server.start()
        transport = server.build_http_transport()
        app = Starlette(routes=transport.routes)  # the host forgot the lifespan
        try:
            async with _http(app) as client:
                response = await client.post(
                    "/mcp",
                    json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
                    headers={"Accept": "application/json, text/event-stream"},
                )
        finally:
            await server.stop()
        assert response.status_code == 503
        assert "McpHttpTransport.lifespan()" in response.json()["error"]

    @pytest.mark.asyncio
    async def test_a_request_after_the_lifespan_exits_is_answered_503(self) -> None:
        server = _server()
        await server.start()
        transport = server.build_http_transport()
        app = Starlette(routes=transport.routes)
        try:
            async with transport.lifespan():
                pass
            async with _http(app) as client:
                response = await client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
        finally:
            await server.stop()
        assert response.status_code == 503

    @pytest.mark.asyncio
    async def test_a_transport_runs_once(self) -> None:
        server = _server()
        await server.start()
        transport = server.build_http_transport()
        try:
            async with transport.lifespan():
                pass
            with pytest.raises(RuntimeError, match="once"):
                async with transport.lifespan():
                    pass
        finally:
            await server.stop()

    def test_the_transport_is_start_gated(self) -> None:
        with pytest.raises(RuntimeError, match="start"):
            _server().build_http_transport()


@pytest.mark.skipif(
    not hasattr(mcp, "Client"), reason="protocol 2026-07-28 needs mcp 2.x; 1.x speaks the handshake only"
)
class TestProtocol20260728:
    """the locked SDK serves the 2026-07-28 single-exchange protocol through the same dispatch."""

    @pytest.mark.asyncio
    async def test_a_client_that_probes_server_discover_settles_on_2026_07_28(self) -> None:
        server = _server()
        await server.start()
        transport = server.build_http_transport()
        app = _host_app(transport, transport.routes)
        try:
            async with app.router.lifespan_context(app), _http(app) as http:
                async with mcp.Client(streamable_http_client("http://host.test/mcp", http_client=http)) as client:
                    version = client.protocol_version
                    result = await client.call_tool("whoami", {})
        finally:
            await server.stop()
        assert version == "2026-07-28"
        assert result.model_dump(by_alias=True)["isError"] is False
        assert _token_seen(result) == "client-key"
