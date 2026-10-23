"""a server built on a tool provider serves the tool set as it stands at each request.

the provider is called on every ``tools/list`` and every ``tools/call``, so a registry that adds,
removes or reshapes a tool between two requests of one session is seen by the second. driven by a
real MCP client over the server's Streamable-HTTP app; ``scripts/test-mcp-v1.sh`` runs this file
on mcp 1.x, whose ``call_tool`` decorator caches the tools of an earlier ``tools/list``.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import httpx
import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.shared import exceptions as mcp_exceptions
from threetears.mcp.auth import Identity
from threetears.mcp.server import McpServer
from threetears.mcp.tool import McpTool, ToolProvider, ToolRegistry

#: the client's JSON-RPC error: McpError on mcp 1.x, MCPError on 2.x
_JSONRPC_ERROR: type[Exception] = getattr(mcp_exceptions, "MCPError", None) or mcp_exceptions.McpError  # type: ignore[attr-defined]


def _tool(name: str, *, input_schema: dict[str, Any] | None = None, returns: str = "ran") -> McpTool:
    """a tool named ``name`` whose handler returns ``returns``."""
    return McpTool(
        name=name,
        description=f"the {name} op",
        input_schema=input_schema if input_schema is not None else {"type": "object", "properties": {}},
        required_permission=f"testament.{name}",
        handler=AsyncMock(return_value=returns),
    )


def _server(tool_provider: ToolProvider) -> McpServer:
    """a server on ``tool_provider`` for a caller every call is allowed for."""
    identity = MagicMock()
    identity.identify = AsyncMock(return_value=Identity(principal_type="user", principal_id=uuid4()))
    authorizer = MagicMock()
    authorizer.allows = AsyncMock(return_value=True)
    authorizer.start = AsyncMock()
    authorizer.stop = AsyncMock()
    return McpServer(
        name="provider-test", identity_provider=identity, authorizer=authorizer, tool_provider=tool_provider
    )


@asynccontextmanager
async def _session(server: McpServer) -> AsyncIterator[ClientSession]:
    """an initialized client session against ``server``'s Streamable-HTTP app."""
    await server.start()
    app = server.build_http_app()
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://mcp.test")
    try:
        async with app.router.lifespan_context(app), client:
            async with streamable_http_client("http://mcp.test/mcp", http_client=client) as streams:
                async with ClientSession(streams[0], streams[1]) as session:  # 1.x yields a third value
                    await session.initialize()
                    yield session
    finally:
        await server.stop()


async def _names(session: ClientSession) -> list[str]:
    """the tool names ``session`` lists now."""
    return [tool.name for tool in (await session.list_tools()).tools]


def _wire(result: Any) -> dict[str, Any]:
    """a call result as wire JSON."""
    wire: dict[str, Any] = result.model_dump(by_alias=True)
    return wire


def _error_code(result: Any) -> str:
    """the framework error code inside a failed call's text block."""
    code: str = json.loads(result.content[0].text)["error"]["code"]
    return code


class TestTheProviderIsReadAtEachRequest:
    @pytest.mark.asyncio
    async def test_tools_list_follows_the_provider(self) -> None:
        live = [_tool("runners.list")]
        async with _session(_server(lambda: list(live))) as session:
            assert await _names(session) == ["runners.list"]
            live.append(_tool("projects.allow"))
            assert await _names(session) == ["runners.list", "projects.allow"]
            live.pop(0)
            assert await _names(session) == ["projects.allow"]

    @pytest.mark.asyncio
    async def test_tools_call_reaches_a_tool_added_after_the_session_listed(self) -> None:
        live = [_tool("runners.list")]
        async with _session(_server(lambda: list(live))) as session:
            await session.list_tools()
            live.append(_tool("projects.allow", returns="allowed"))
            result = await session.call_tool("projects.allow", {})
        assert _wire(result)["isError"] is False
        assert result.content[0].text == "allowed"  # type: ignore[union-attr]

    @pytest.mark.asyncio
    async def test_tools_call_refuses_a_tool_the_provider_no_longer_answers(self) -> None:
        live = [_tool("runners.list"), _tool("projects.allow")]
        async with _session(_server(lambda: list(live))) as session:
            await session.list_tools()
            live.pop()
            result = await session.call_tool("projects.allow", {})
        assert _wire(result)["isError"] is True
        assert _error_code(result) == "UNKNOWN_TOOL"

    @pytest.mark.asyncio
    async def test_arguments_are_checked_against_the_schema_as_it_stands_now(self) -> None:
        # mcp 1.x's decorator would check them against its cached copy from the first tools/list
        old = _tool("runners.get", input_schema={"type": "object", "properties": {"id": {"type": "integer"}}})
        new = _tool("runners.get", input_schema={"type": "object", "properties": {"id": {"type": "string"}}})
        live = [old]
        async with _session(_server(lambda: list(live))) as session:
            await session.list_tools()
            live[0] = new
            accepted = await session.call_tool("runners.get", {"id": "r1"})
            refused = await session.call_tool("runners.get", {"id": 1})
        assert _wire(accepted)["isError"] is False
        assert _wire(refused)["isError"] is True
        assert refused.content[0].text.startswith("Input validation error")  # type: ignore[union-attr]

    @pytest.mark.asyncio
    async def test_any_iterable_is_an_answer(self) -> None:
        def _generated() -> Iterable[McpTool]:
            yield _tool("runners.list")

        async with _session(_server(_generated)) as session:
            assert await _names(session) == ["runners.list"]


class TestABadProviderAnswer:
    @pytest.mark.asyncio
    async def test_two_tools_with_one_name_refuse_the_call(self) -> None:
        async with _session(_server(lambda: [_tool("runners.list"), _tool("runners.list")])) as session:
            result = await session.call_tool("runners.list", {})
        assert _wire(result)["isError"] is True
        assert _error_code(result) == "TOOLS_UNAVAILABLE"

    @pytest.mark.asyncio
    async def test_a_provider_that_raises_refuses_the_call(self) -> None:
        def _broken() -> list[McpTool]:
            raise LookupError("the op registry is reloading")

        async with _session(_server(_broken)) as session:
            result = await session.call_tool("runners.list", {})
        assert _wire(result)["isError"] is True
        assert _error_code(result) == "TOOLS_UNAVAILABLE"

    @pytest.mark.asyncio
    async def test_a_provider_that_raises_fails_tools_list(self) -> None:
        calls = 0

        def _breaks_second_time() -> list[McpTool]:
            nonlocal calls
            calls += 1
            if calls > 1:
                raise LookupError("the op registry is reloading")
            return [_tool("runners.list")]

        async with _session(_server(_breaks_second_time)) as session:
            await session.list_tools()
            with pytest.raises(_JSONRPC_ERROR):
                await session.list_tools()

    @pytest.mark.asyncio
    async def test_a_failing_provider_does_not_send_its_own_error_to_the_client(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # the SDK answers with the exception's text on either major, so the server raises a fixed one
        def _broken() -> list[McpTool]:
            raise LookupError("the op registry is reloading from /srv/ops")

        with caplog.at_level(logging.ERROR, logger="threetears.mcp"):
            async with _session(_server(_broken)) as session:
                with pytest.raises(_JSONRPC_ERROR) as refused:
                    await session.list_tools()
        assert str(refused.value) == "the server's tool set could not be read"
        logged = [record for record in caplog.records if record.name == "threetears.mcp.server"]
        assert [record.getMessage() for record in logged] == ["MCP tools/list: the tool provider failed"]
        assert logged[0].exc_info is not None
        assert str(logged[0].exc_info[1]) == "the op registry is reloading from /srv/ops"


class TestTheServerTakesOneToolSource:
    def test_neither(self) -> None:
        with pytest.raises(ValueError, match="exactly one of registry= and tool_provider="):
            McpServer(name="t", identity_provider=MagicMock(), authorizer=MagicMock())

    def test_both(self) -> None:
        registry = ToolRegistry()
        with pytest.raises(ValueError, match="exactly one of registry= and tool_provider="):
            McpServer(
                name="t",
                identity_provider=MagicMock(),
                authorizer=MagicMock(),
                registry=registry,
                tool_provider=registry.all,
            )
