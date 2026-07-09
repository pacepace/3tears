"""unit tests for the MCP Streamable-HTTP transport (``http_server`` + ``serve_http``).

drives a real :class:`~threetears.mcp.server.McpServer` over the SDK's
Streamable-HTTP transport in-process (the Starlette ASGI app via
``httpx.ASGITransport``, its lifespan run explicitly) and asserts:

1. transport smoke -- a real MCP client can ``initialize``,
   ``tools/list`` (sees the registered tool), and ``tools/call``
   (gets the tool result);
2. start-gating -- the HTTP entry point raises ``RuntimeError`` when
   :meth:`McpServer.start` was not called (parity with
   :meth:`McpServer.serve_stdio`);
3. RBAC still fires -- a denied permission returns a
   ``CallToolResult(isError=True)`` carrying ``PERMISSION_DENIED``,
   proving the shared ``_dispatch`` core is reused over HTTP.

the bearer token flows client -> ``Authorization`` header ->
transport middleware contextvar -> :class:`BearerTokenIdentityProvider`
-> ``_dispatch``: a real end-to-end identity resolution.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from uuid import uuid4

import httpx
import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from threetears.mcp.auth import (
    BearerTokenIdentityProvider,
    Identity,
    LocalGrantAuthorizer,
)
from threetears.mcp.http_server import current_bearer_token
from threetears.mcp.server import McpServer
from threetears.mcp.tool import McpTool, ToolRegistry

_ADMIN_ID = uuid4()
_USER_ID = uuid4()


async def _empty_grant_loader() -> list[dict[str, Any]]:
    """grant loader with no grants: admins short-circuit, everyone else denies."""
    return []


async def _resolve_token(token: str) -> Identity:
    """test resolver: ``admin-token`` -> admin identity, ``user-token`` -> plain user.

    :param token: bearer token from the request
    :ptype token: str
    :return: resolved identity
    :rtype: Identity
    :raises RuntimeError: for any unknown token
    """
    if token == "admin-token":
        return Identity(principal_type="user", principal_id=_ADMIN_ID, is_admin=True)
    if token == "user-token":
        return Identity(principal_type="user", principal_id=_USER_ID, is_admin=False)
    raise RuntimeError(f"unknown token {token!r}")


def _echo_registry() -> ToolRegistry:
    """registry holding a single ``echo`` tool."""
    registry = ToolRegistry()

    async def _echo(**kwargs: Any) -> str:
        return json.dumps(kwargs)

    registry.register(
        McpTool(
            name="echo",
            description="echo the arguments back as JSON",
            input_schema={
                "type": "object",
                "properties": {"value": {"type": "string"}},
            },
            required_permission="t.echo.call",
            handler=_echo,
        ),
    )
    return registry


def _build_server() -> McpServer:
    """construct an McpServer with bearer identity + local-grant authorizer."""
    return McpServer(
        name="http-test",
        identity_provider=BearerTokenIdentityProvider(
            resolver=_resolve_token,
            token_source=current_bearer_token,
        ),
        authorizer=LocalGrantAuthorizer(grant_loader=_empty_grant_loader),
        registry=_echo_registry(),
    )


@asynccontextmanager
async def _mcp_client(app: Any, *, token: str) -> AsyncIterator[ClientSession]:
    """open an MCP client session against ``app`` over an in-process ASGI transport.

    :param app: the Starlette ASGI app returned by ``build_http_app``
    :ptype app: Any
    :param token: bearer token to present in the ``Authorization`` header
    :ptype token: str
    :return: an initialized-capable MCP client session
    :rtype: AsyncIterator[ClientSession]
    """
    headers = {"Authorization": f"Bearer {token}"}

    def _factory(
        headers: dict[str, str] | None = None,
        timeout: httpx.Timeout | None = None,
        auth: httpx.Auth | None = None,
    ) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://mcp.test",
            headers=headers,
            timeout=timeout,
            follow_redirects=True,
        )

    async with app.router.lifespan_context(app):
        async with streamablehttp_client(
            "http://mcp.test/mcp",
            headers=headers,
            httpx_client_factory=_factory,
        ) as (read_stream, write_stream, _get_session_id):
            async with ClientSession(read_stream, write_stream) as session:
                yield session


class TestHttpTransportSmoke:
    """a real MCP client drives the server over Streamable-HTTP."""

    @pytest.mark.asyncio
    async def test_list_and_call_tool_over_http(self) -> None:
        """initialize + tools/list + tools/call succeed for an admin bearer token."""
        server = _build_server()
        await server.start()
        app = server.build_http_app()
        try:
            async with _mcp_client(app, token="admin-token") as session:
                await session.initialize()
                listed = await session.list_tools()
                assert "echo" in [tool.name for tool in listed.tools]
                result = await session.call_tool("echo", {"value": "hi"})
                assert result.isError is False
                text = result.content[0].text  # type: ignore[union-attr]
                assert json.loads(text) == {"value": "hi"}
        finally:
            await server.stop()


class TestHttpStartGating:
    """the HTTP entry points refuse to build/serve before ``start()``."""

    def test_build_http_app_without_start_raises(self) -> None:
        """building the ASGI app before start() is a clear error, not silent default-deny."""
        server = _build_server()
        with pytest.raises(RuntimeError, match="start"):
            server.build_http_app()

    @pytest.mark.asyncio
    async def test_serve_http_without_start_raises(self) -> None:
        """serving before start() raises RuntimeError (parity with serve_stdio)."""
        server = _build_server()
        with pytest.raises(RuntimeError, match="start"):
            await server.serve_http(host="127.0.0.1", port=0)


class TestHttpRbacStillFires:
    """the shared _dispatch RBAC gate runs over the HTTP transport too."""

    @pytest.mark.asyncio
    async def test_denied_permission_returns_permission_denied(self) -> None:
        """a non-admin bearer token with no grant gets PERMISSION_DENIED over HTTP."""
        server = _build_server()
        await server.start()
        app = server.build_http_app()
        try:
            async with _mcp_client(app, token="user-token") as session:
                await session.initialize()
                result = await session.call_tool("echo", {"value": "hi"})
                assert result.isError is True
                envelope = json.loads(result.content[0].text)  # type: ignore[union-attr]
                assert envelope["error"]["code"] == "PERMISSION_DENIED"
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_unknown_bearer_token_returns_identity_unavailable(self) -> None:
        """an unresolvable bearer token maps to IDENTITY_UNAVAILABLE via _dispatch."""
        server = _build_server()
        await server.start()
        app = server.build_http_app()
        try:
            async with _mcp_client(app, token="mystery-token") as session:
                await session.initialize()
                result = await session.call_tool("echo", {"value": "hi"})
                assert result.isError is True
                envelope = json.loads(result.content[0].text)  # type: ignore[union-attr]
                assert envelope["error"]["code"] == "IDENTITY_UNAVAILABLE"
        finally:
            await server.stop()
