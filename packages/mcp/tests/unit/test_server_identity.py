"""a server's version and instructions reach the client on both protocol eras; left unset, nothing changes.

the SDK's own clients over a real loopback connection: ``ClientSession`` on the handshake-era protocol (2025-11-25),
where both ride the ``initialize`` result, and ``mcp.Client`` settling on 2026-07-28, where the instructions ride the
``server/discover`` result and the version each result's ``serverInfo`` stamp.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import httpx2
import mcp
import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from threetears.mcp import Identity, McpServer, ToolRegistry

from .conftest import LoopbackMcp


def _server(**identity: Any) -> McpServer:
    provider = MagicMock()
    provider.identify = AsyncMock(return_value=Identity(principal_type="user", principal_id=uuid4()))
    authorizer = MagicMock()
    authorizer.allows = AsyncMock(return_value=True)
    authorizer.start = AsyncMock()
    authorizer.stop = AsyncMock()
    return McpServer(
        name="identified", identity_provider=provider, authorizer=authorizer, registry=ToolRegistry(), **identity
    )


@contextlib.asynccontextmanager
async def _handshake(url: str) -> AsyncIterator[Any]:
    async with httpx2.AsyncClient() as http:
        async with streamable_http_client(url, http_client=http) as (read, write):
            async with ClientSession(read, write) as session:
                yield await session.initialize()


@contextlib.asynccontextmanager
async def _modern(url: str) -> AsyncIterator[mcp.Client]:
    async with httpx2.AsyncClient() as http:
        async with mcp.Client(streamable_http_client(url, http_client=http)) as client:
            assert client.protocol_version == "2026-07-28"
            yield client


_GUIDE = "Use session_send for long work; poll with session_status."


@pytest.mark.asyncio
@pytest.mark.parametrize("stateless", [True, False], ids=["stateless", "stateful"])
async def test_the_handshake_carries_the_version_and_the_instructions(
    loopback_mcp: LoopbackMcp, stateless: bool
) -> None:
    async with loopback_mcp(_server(version="2.0.0", instructions=_GUIDE), stateless=stateless) as url:
        async with _handshake(url) as initialized:
            assert (initialized.server_info.name, initialized.server_info.version) == ("identified", "2.0.0")
            assert initialized.instructions == _GUIDE


@pytest.mark.asyncio
async def test_a_2026_client_reads_the_instructions_from_discover_and_the_version_from_the_stamp(
    loopback_mcp: LoopbackMcp,
) -> None:
    async with loopback_mcp(_server(version="2.0.0", instructions=_GUIDE)) as url:
        async with _modern(url) as client:
            assert client.instructions == _GUIDE
            await client.list_tools()
            assert client.server_info is not None
            assert (client.server_info.name, client.server_info.version) == ("identified", "2.0.0")


@pytest.mark.asyncio
async def test_left_unset_the_version_is_empty_and_no_instructions_are_sent(loopback_mcp: LoopbackMcp) -> None:
    async with loopback_mcp(_server()) as url:
        async with _handshake(url) as initialized:
            assert initialized.server_info.version == ""
            assert initialized.instructions is None
        async with _modern(url) as client:
            assert client.instructions is None
