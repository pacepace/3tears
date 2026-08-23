"""Tests for McpClient.

The client owns a :class:`threetears.core.http_client.TracedHttpClient`
rather than a raw ``httpx.AsyncClient``, so these tests drive the real
transport with an ``httpx.MockTransport`` instead of a mock object with a
``post`` attribute. That distinction matters: the retry, tracing, and
error-normalisation the traced client performs sit BETWEEN this client and
the socket, and a mocked-out ``post`` asserts on a path production never
takes.
"""

from __future__ import annotations

from collections.abc import Callable

import httpx

from threetears.agent.tools.mcp import McpClient
from threetears.core.http_client import TracedHttpClient

_BASE_URL = "http://localhost:9000"


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> McpClient:
    """construct McpClient over a mock transport running ``handler``.

    :param handler: request handler returning a response (or raising a
        transport exception)
    :ptype handler: Callable[[httpx.Request], httpx.Response]
    :return: client whose traced transport is bound to ``handler``
    :rtype: McpClient
    """
    http = TracedHttpClient(
        upstream_base_url=_BASE_URL,
        timeout=1.0,
        max_attempts=1,
        transport=httpx.MockTransport(handler),
    )
    return McpClient(_BASE_URL, http_client=http)


def _responding(payload: dict, status_code: int = 200) -> Callable[[httpx.Request], httpx.Response]:
    """handler returning ``payload`` as JSON under ``status_code``.

    :param payload: JSON body to return
    :ptype payload: dict
    :param status_code: HTTP status to return
    :ptype status_code: int
    :return: mock-transport handler
    :rtype: Callable[[httpx.Request], httpx.Response]
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, json=payload)

    return handler


def _raising(exc: Exception) -> Callable[[httpx.Request], httpx.Response]:
    """handler raising ``exc`` instead of answering.

    :param exc: transport exception to raise
    :ptype exc: Exception
    :return: mock-transport handler
    :rtype: Callable[[httpx.Request], httpx.Response]
    """

    def handler(request: httpx.Request) -> httpx.Response:
        raise exc

    return handler


async def test_list_tools():
    client = _client(
        _responding(
            {
                "tools": [
                    {"name": "search", "description": "Search things", "inputSchema": {"type": "object"}},
                    {"name": "calc", "description": "Calculate"},
                ]
            }
        )
    )

    tools = await client.list_tools()
    assert len(tools) == 2
    assert tools[0].name == "search"
    assert tools[0].description == "Search things"
    assert tools[0].input_schema == {"type": "object"}
    assert tools[1].name == "calc"
    assert tools[1].input_schema == {}


async def test_list_tools_targets_the_spec_path():
    """The path is joined onto the traced client's base URL, not re-prefixed."""
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, json={"tools": []})

    client = _client(handler)
    await client.list_tools()
    assert seen == [f"{_BASE_URL}/mcp/v1/tools/list"]


async def test_list_tools_error():
    client = _client(_raising(httpx.ConnectError("connection refused")))

    tools = await client.list_tools()
    assert tools == []


async def test_invoke_tool_success():
    client = _client(
        _responding(
            {
                "content": [
                    {"type": "text", "text": "Hello "},
                    {"type": "text", "text": "World"},
                    {"type": "image", "data": "..."},
                ],
                "isError": False,
            }
        )
    )

    result = await client.invoke_tool("greet", {"name": "test"})
    assert result.success is True
    assert result.content == "Hello \nWorld"
    assert result.error is None
    # a server that sends only text produces exactly what it did before
    assert result.metadata is None


async def test_invoke_tool_keeps_structured_content():
    """§4.8: ``structuredContent`` reaches the caller instead of being dropped.

    Before this field existed, a structured result arrived as prose in
    ``content`` and the caller had to re-parse the rendering to recover a
    shape the server already knew.
    """
    client = _client(
        _responding(
            {
                "content": [{"type": "text", "text": '{"status": "ok"}'}],
                "structuredContent": {"status": "ok", "count": 3},
                "isError": False,
            }
        )
    )

    result = await client.invoke_tool("probe", {})
    assert result.metadata == {"status": "ok", "count": 3}
    assert result.content == '{"status": "ok"}'


async def test_invoke_tool_ignores_a_non_object_structured_field():
    """A server sending a non-object there is refused the field, not trusted.

    ``structuredContent`` is an object per the spec; taking anything else
    would hand the caller a ``metadata`` that is not a mapping, which every
    reader of that field assumes it is.
    """
    client = _client(
        _responding(
            {
                "content": [{"type": "text", "text": "[1, 2]"}],
                "structuredContent": [1, 2],
                "isError": False,
            }
        )
    )

    result = await client.invoke_tool("probe", {})
    assert result.metadata is None


async def test_invoke_tool_error_flag():
    client = _client(
        _responding(
            {
                "content": [{"type": "text", "text": "something went wrong"}],
                "isError": True,
            }
        )
    )

    result = await client.invoke_tool("failing", {})
    assert result.success is False
    assert result.content == "something went wrong"


async def test_invoke_tool_timeout():
    """A timeout still reads as a timeout after the traced client normalises it.

    ``TracedHttpClient`` collapses every transport failure into a status-less
    ``UpstreamHttpError``; the cause it chains is what keeps this message
    distinguishable from an unreachable server.
    """
    client = _client(_raising(httpx.TimeoutException("timed out")))

    result = await client.invoke_tool("slow_tool", {})
    assert result.success is False
    assert result.error == "MCP tool invocation timed out"


async def test_invoke_tool_connection_error():
    client = _client(_raising(httpx.ConnectError("refused")))

    result = await client.invoke_tool("unreachable", {})
    assert result.success is False
    assert result.error is not None
    assert "could not be reached" in result.error


async def test_invoke_tool_upstream_5xx():
    """An exhausted 5xx names the status; there is no transport cause to report."""
    client = _client(_responding({"detail": "boom"}, status_code=503))

    result = await client.invoke_tool("broken", {})
    assert result.success is False
    assert result.error is not None
    assert "503" in result.error


async def test_invoke_tool_does_not_retry_a_side_effecting_call():
    """``tools/call`` is not idempotent, so the client must send it exactly once."""
    calls: list[int] = [0]

    def handler(request: httpx.Request) -> httpx.Response:
        calls[0] += 1
        raise httpx.ConnectError("refused")

    http = TracedHttpClient(
        upstream_base_url=_BASE_URL,
        timeout=1.0,
        max_attempts=McpClient.MAX_ATTEMPTS,
        transport=httpx.MockTransport(handler),
    )
    client = McpClient(_BASE_URL, http_client=http)

    await client.invoke_tool("unreachable", {})
    assert calls[0] == 1


async def test_test_connection_success():
    client = _client(_responding({"tools": []}))

    assert await client.test_connection() is True


async def test_test_connection_failure():
    client = _client(_raising(httpx.ConnectError("refused")))

    assert await client.test_connection() is False


async def test_close_closes_the_traced_client():
    client = _client(_responding({"tools": []}))
    await client.close()
    result = await client.list_tools()
    assert result == []
