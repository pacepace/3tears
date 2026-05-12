"""Tests for McpClient."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx

from threetears.agent.tools.mcp import McpClient


def _mock_response(json_data: dict, status_code: int = 200) -> MagicMock:
    """Create a mock httpx.Response."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        resp.raise_for_status.side_effect = httpx.HTTPStatusError("error", request=MagicMock(), response=resp)
    return resp


def _client_with_post(post_mock: AsyncMock) -> McpClient:
    """construct McpClient with an injected mock http client bound to ``post``.

    :param post_mock: AsyncMock that the client's ``post`` method should invoke
    :ptype post_mock: AsyncMock
    :return: client whose internal http transport is the injected mock
    :rtype: McpClient
    """
    http = AsyncMock()
    http.post = post_mock
    return McpClient("http://localhost:9000", http_client=http)


async def test_list_tools():
    post = AsyncMock(
        return_value=_mock_response(
            {
                "tools": [
                    {"name": "search", "description": "Search things", "inputSchema": {"type": "object"}},
                    {"name": "calc", "description": "Calculate"},
                ]
            }
        )
    )
    client = _client_with_post(post)

    tools = await client.list_tools()
    assert len(tools) == 2
    assert tools[0].name == "search"
    assert tools[0].description == "Search things"
    assert tools[0].input_schema == {"type": "object"}
    assert tools[1].name == "calc"
    assert tools[1].input_schema == {}


async def test_list_tools_error():
    post = AsyncMock(side_effect=httpx.ConnectError("connection refused"))
    client = _client_with_post(post)

    tools = await client.list_tools()
    assert tools == []


async def test_invoke_tool_success():
    post = AsyncMock(
        return_value=_mock_response(
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
    client = _client_with_post(post)

    result = await client.invoke_tool("greet", {"name": "test"})
    assert result.success is True
    assert result.content == "Hello \nWorld"
    assert result.error is None


async def test_invoke_tool_error_flag():
    post = AsyncMock(
        return_value=_mock_response(
            {
                "content": [{"type": "text", "text": "something went wrong"}],
                "isError": True,
            }
        )
    )
    client = _client_with_post(post)

    result = await client.invoke_tool("failing", {})
    assert result.success is False
    assert result.content == "something went wrong"


async def test_invoke_tool_timeout():
    post = AsyncMock(side_effect=httpx.TimeoutException("timed out"))
    client = _client_with_post(post)

    result = await client.invoke_tool("slow_tool", {})
    assert result.success is False
    assert result.error == "MCP tool invocation timed out"


async def test_invoke_tool_connection_error():
    post = AsyncMock(side_effect=httpx.ConnectError("refused"))
    client = _client_with_post(post)

    result = await client.invoke_tool("unreachable", {})
    assert result.success is False
    assert result.error is not None
    assert "refused" in result.error


async def test_test_connection_success():
    post = AsyncMock(return_value=_mock_response({"tools": []}))
    client = _client_with_post(post)

    assert await client.test_connection() is True


async def test_test_connection_failure():
    post = AsyncMock(side_effect=httpx.ConnectError("refused"))
    client = _client_with_post(post)

    assert await client.test_connection() is False
