"""a tool's ``prepare_arguments`` turns a looser form its schema does not state into the form it does, before the
arguments are checked; everything after sees the prepared arguments.

through a real MCP ``ClientSession`` over the server's own Streamable-HTTP app.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import httpx2
import mcp.types as mcp_types
import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from threetears.mcp import ERROR_META_KEY, Identity, McpServer, McpTool, Refusal, ToolCall, ToolRegistry

_SCHEMA = {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]}


def _integers(arguments: Mapping[str, Any]) -> dict[str, Any]:
    """a numeric string for ``n`` as the integer it writes, as an older wire accepted it."""
    prepared = dict(arguments)
    if isinstance(prepared.get("n"), str) and prepared["n"].isdigit():
        prepared["n"] = int(prepared["n"])
    return prepared


def _broken(arguments: Mapping[str, Any]) -> dict[str, Any]:
    raise ValueError("cannot prepare")


class _Seeing:
    """grants every call; records the arguments it was shown."""

    def __init__(self) -> None:
        self.seen: list[dict[str, Any]] = []

    async def allows(self, identity: Identity, permission: str) -> bool:
        return True

    async def authorize_call(self, identity: Identity, call: ToolCall) -> bool | Refusal:
        self.seen.append(dict(call.arguments))
        return True

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None


def _server(
    prepare: Callable[[Mapping[str, Any]], dict[str, Any]] | None, ran: list[dict[str, Any]], authorizer: Any = None
) -> McpServer:
    async def handler(**arguments: Any) -> str:
        ran.append(arguments)
        return "ran"

    registry = ToolRegistry()
    registry.register(
        McpTool(
            name="count",
            description="takes an integer",
            input_schema=_SCHEMA,
            required_permission="t.count",
            handler=handler,
            prepare_arguments=prepare,
        )
    )
    provider = MagicMock()
    provider.identify = AsyncMock(return_value=Identity(principal_type="user", principal_id=uuid4()))
    return McpServer(
        name="prepare",
        identity_provider=provider,
        authorizer=authorizer if authorizer is not None else _Seeing(),
        registry=registry,
    )


async def _call(server: McpServer, arguments: dict[str, Any]) -> mcp_types.CallToolResult:
    await server.start()
    app = server.build_http_app()
    client = httpx2.AsyncClient(transport=httpx2.ASGITransport(app=app), base_url="http://mcp.test")
    try:
        async with app.router.lifespan_context(app), client:
            async with streamable_http_client("http://mcp.test/mcp", http_client=client) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    return await session.call_tool("count", arguments)
    finally:
        await server.stop()


def _code(result: mcp_types.CallToolResult) -> str:
    assert result.is_error is True
    assert result.meta is not None
    code = result.meta[ERROR_META_KEY]["code"]
    assert isinstance(code, str)
    return code


@pytest.mark.asyncio
async def test_a_prepared_value_passes_the_schema_and_reaches_the_handler_and_the_authorizer() -> None:
    ran: list[dict[str, Any]] = []
    gate = _Seeing()
    result = await _call(_server(_integers, ran, gate), {"n": "5"})
    assert result.is_error is False
    assert ran == [{"n": 5}]
    assert gate.seen == [{"n": 5}]


@pytest.mark.asyncio
async def test_a_value_the_hook_leaves_is_still_refused_by_the_schema() -> None:
    ran: list[dict[str, Any]] = []
    assert _code(await _call(_server(_integers, ran), {"n": "x"})) == "INVALID_ARGUMENTS"
    assert ran == []


@pytest.mark.asyncio
async def test_a_hook_that_raises_refuses_the_call_as_invalid_arguments(caplog: pytest.LogCaptureFixture) -> None:
    ran: list[dict[str, Any]] = []
    result = await _call(_server(_broken, ran), {"n": 5})
    assert _code(result) == "INVALID_ARGUMENTS"
    assert ran == []
    assert sum("prepare_arguments raised" in record.getMessage() for record in caplog.records) == 1


@pytest.mark.asyncio
async def test_a_hook_that_answers_no_dict_refuses_the_call() -> None:
    ran: list[dict[str, Any]] = []
    result = await _call(_server(lambda arguments: arguments, ran), {"n": 5})  # type: ignore[arg-type,return-value]
    assert _code(result) == "INVALID_ARGUMENTS"
    assert ran == []


@pytest.mark.asyncio
async def test_a_tool_without_the_hook_still_refuses_a_string_for_an_integer() -> None:
    ran: list[dict[str, Any]] = []
    assert _code(await _call(_server(None, ran), {"n": "5"})) == "INVALID_ARGUMENTS"
    assert ran == []
