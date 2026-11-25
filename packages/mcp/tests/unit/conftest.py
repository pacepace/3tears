"""fixtures for the mcp unit tests that need a real HTTP connection.

most wire tests serve the app in-process through ``httpx.ASGITransport``, which never tells the app
that a client went away. a test of what happens when a client gives up on a call -- the
cancellation a handler sees -- needs a real socket, so :func:`loopback_mcp` serves a server's
Streamable-HTTP app with uvicorn on an ephemeral loopback port.
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


@asynccontextmanager
async def _serve(server: McpServer, **app_options: Any) -> AsyncIterator[str]:
    """start ``server``, serve its HTTP app on a loopback port, and stop both when the block ends."""
    await server.start()
    app = server.build_http_app(**app_options)
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
                yield f"http://127.0.0.1:{port}/mcp"
            finally:
                web.should_exit = True
    finally:
        listener.close()
        await server.stop()


@pytest.fixture
def loopback_mcp() -> LoopbackMcp:
    """serve an :class:`McpServer` over a real loopback connection for the length of a block.

    :return: ``serve(server, **build_http_app_options)``, an async context manager yielding the URL
    :rtype: LoopbackMcp
    """
    return _serve
