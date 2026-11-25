"""fixtures for the mcp unit tests that need a real HTTP connection.

most wire tests serve the app in-process through ``httpx.ASGITransport``, which never tells the app
that a client went away. a test of what happens when a client gives up on a call -- the
cancellation a handler sees -- needs a real socket, and so does a test that a notification reaches
the client while the call still runs, since ``ASGITransport`` hands over a response only once it is
complete. :func:`loopback_mcp` serves a server's Streamable-HTTP app with uvicorn on an ephemeral
loopback port; :func:`loopback_app` serves any app, such as a host's embedding the transport.
"""

from __future__ import annotations

import socket
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Any

import anyio
import pytest
import uvicorn
from threetears.mcp.server import McpServer

LoopbackMcp = Callable[..., AbstractAsyncContextManager[str]]
"""``loopback_mcp(server, **build_http_app_options)``: serve ``server``, yielding the endpoint's URL."""

LoopbackApp = Callable[[Any], AbstractAsyncContextManager[str]]
"""``loopback_app(app)``: serve any ASGI app (a host's, embedding the transport), yielding its base URL."""


@asynccontextmanager
async def _serve_app(app: Any) -> AsyncIterator[str]:
    """serve the ASGI ``app`` with uvicorn on a loopback port for the block, yielding its base URL."""
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    web = uvicorn.Server(uvicorn.Config(app, log_config=None, lifespan="on", timeout_graceful_shutdown=2))
    try:
        async with anyio.create_task_group() as group:
            group.start_soon(web.serve, [listener])
            with anyio.fail_after(10):
                while not web.started:
                    await anyio.sleep(0.01)
            try:
                yield f"http://127.0.0.1:{port}"
            finally:
                web.should_exit = True
    finally:
        listener.close()


@asynccontextmanager
async def _serve(server: McpServer, **app_options: Any) -> AsyncIterator[str]:
    """start ``server``, serve its HTTP app on a loopback port, and stop both when the block ends."""
    await server.start()
    try:
        async with _serve_app(server.build_http_app(**app_options)) as base:
            yield f"{base}{app_options.get('path', '/mcp')}"
    finally:
        await server.stop()


@pytest.fixture
def loopback_mcp() -> LoopbackMcp:
    """serve an :class:`McpServer` over a real loopback connection for the length of a block.

    :return: ``serve(server, **build_http_app_options)``, an async context manager yielding the URL
    :rtype: LoopbackMcp
    """
    return _serve


@pytest.fixture
def loopback_app() -> LoopbackApp:
    """serve an ASGI app over a real loopback connection for the length of a block.

    :return: ``serve(app)``, an async context manager yielding the base URL
    :rtype: LoopbackApp
    """
    return _serve_app
