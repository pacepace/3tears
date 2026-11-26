"""a TearsTool served by 3tears-mcp's McpServer: its schema, its ToolResult and its code, and the call context.

calls go through a real MCP ``ClientSession`` over the server's own Streamable-HTTP app, so every assertion is on
what the client receives.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import httpx2
import mcp.types as mcp_types
import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from threetears.mcp import ERROR_META_KEY, CallContext, ErrorCode, ErrorCodes, Identity, McpServer, ToolRegistry
from threetears.mcp.errors import BUILTIN_ERROR_CODES
from threetears.tool_contract import TOOL_RESULT_ERROR_CODES, MCPToolDefinition, TearsTool, ToolResult
from threetears.tool_contract.mcp_face import mcp_tool

TAINTED = ErrorCode("tainted", "the session is tainted")

_INPUT = {
    "type": "object",
    "properties": {"session": {"type": "string"}, "options": {"type": "object"}},
    "required": ["session"],
}


class _Send(TearsTool):
    """answers what it was told to, and records what execute was called with."""

    face_mcp = True

    def __init__(self, answer: ToolResult | BaseException, *, output_schema: dict[str, Any] | None = None) -> None:
        self.answer = answer
        self.output_schema = output_schema
        self.calls: list[dict[str, Any]] = []

    def mcp_name(self) -> str:
        return "example.session_send"

    def mcp_version(self) -> str:
        return "1.0.0"

    def mcp_schema(self) -> MCPToolDefinition:
        return MCPToolDefinition(
            name=self.mcp_name(),
            version=self.mcp_version(),
            description="send to a session",
            input_schema=_INPUT,
            output_schema=self.output_schema,
        )

    async def execute(self, **kwargs: Any) -> ToolResult:
        self.calls.append(kwargs)
        if isinstance(self.answer, BaseException):
            raise self.answer
        return self.answer


class _WithContext(_Send):
    """an execute that asks for the call context, and keeps working without one."""

    async def execute(self, *, ctx: CallContext | None = None, **kwargs: Any) -> ToolResult:
        self.calls.append({"ctx": ctx, **kwargs})
        return ToolResult(success=True, content=ctx.tool_name if ctx is not None else "no context")


def _server(tool: TearsTool, *, error_codes: ErrorCodes | None = None, name: str | None = "session_send") -> McpServer:
    registry = ToolRegistry()
    registry.register(mcp_tool(tool, required_permission="sessions.send", name=name))
    provider = MagicMock()
    provider.identify = AsyncMock(return_value=Identity(principal_type="user", principal_id=uuid4()))
    authorizer = MagicMock()
    authorizer.allows = AsyncMock(return_value=True)
    authorizer.start = AsyncMock()
    authorizer.stop = AsyncMock()
    return McpServer(
        name="face", identity_provider=provider, authorizer=authorizer, registry=registry, error_codes=error_codes
    )


async def _session_do(server: McpServer, action: Any) -> Any:
    await server.start()
    app = server.build_http_app()
    client = httpx2.AsyncClient(transport=httpx2.ASGITransport(app=app), base_url="http://mcp.test")
    try:
        async with app.router.lifespan_context(app), client:
            async with streamable_http_client("http://mcp.test/mcp", http_client=client) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    return await action(session)
    finally:
        await server.stop()


async def _call(server: McpServer, arguments: dict[str, Any], tool: str = "session_send") -> mcp_types.CallToolResult:
    async def action(session: ClientSession) -> mcp_types.CallToolResult:
        return await session.call_tool(tool, arguments)

    result: mcp_types.CallToolResult = await _session_do(server, action)
    return result


def _text(result: mcp_types.CallToolResult) -> str:
    block = result.content[0]
    assert isinstance(block, mcp_types.TextContent)
    return block.text


def _error(result: mcp_types.CallToolResult) -> dict[str, Any]:
    assert result.is_error is True
    assert result.meta is not None
    detail = result.meta[ERROR_META_KEY]
    assert isinstance(detail, dict)
    return detail


class TestTheListing:
    @pytest.mark.asyncio
    async def test_the_tool_is_listed_with_its_schema_under_the_name_given(self) -> None:
        output = {"type": "object", "properties": {"ok": {"type": "boolean"}}}
        tool = _Send(ToolResult(success=True, content="{}"), output_schema=output)

        async def action(session: ClientSession) -> mcp_types.ListToolsResult:
            return await session.list_tools()

        listed = await _session_do(_server(tool), action)
        [entry] = listed.tools
        assert (entry.name, entry.description) == ("session_send", "send to a session")
        assert entry.input_schema == _INPUT
        assert entry.output_schema == output

    @pytest.mark.asyncio
    async def test_without_a_name_it_is_listed_under_its_canonical_name(self) -> None:
        tool = _Send(ToolResult(success=True, content="ok"))
        result = await _call(_server(tool, name=None), {"session": "s1"}, tool="example.session_send")
        assert _text(result) == "ok"


class TestTheResult:
    @pytest.mark.asyncio
    async def test_a_success_is_its_content(self) -> None:
        tool = _Send(ToolResult(success=True, content="sent"))
        result = await _call(_server(tool), {"session": "s1", "options": {"a": 1}})
        assert (result.is_error, _text(result), result.structured_content) == (False, "sent", None)
        assert tool.calls == [{"session": "s1", "options": {"a": 1}}]

    @pytest.mark.asyncio
    async def test_a_loose_value_the_schema_refuses_is_invalid_arguments_not_coerced(self) -> None:
        """the MCP face checks arguments against the input schema before run, as the protocol has them conform."""
        tool = _Send(ToolResult(success=True, content="sent"))
        detail = _error(await _call(_server(tool), {"session": "s1", "options": "{}"}))
        assert (detail["code"], detail["argument"]) == ("INVALID_ARGUMENTS", "options")
        assert tool.calls == []

    @pytest.mark.asyncio
    async def test_a_success_of_a_tool_with_an_output_schema_is_structured_too(self) -> None:
        output = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}
        tool = _Send(ToolResult(success=True, content='{"ok": true}'), output_schema=output)
        result = await _call(_server(tool), {"session": "s1"})
        assert (_text(result), result.structured_content) == ('{"ok": true}', {"ok": True})

    @pytest.mark.asyncio
    async def test_content_that_is_no_json_object_breaks_the_output_schema(self) -> None:
        tool = _Send(ToolResult(success=True, content="not json"), output_schema={"type": "object"})
        assert _error(await _call(_server(tool), {"session": "s1"}))["code"] == "OUTPUT_VALIDATION_ERROR"

    @pytest.mark.asyncio
    async def test_a_platform_code_keeps_its_code_and_retryable(self) -> None:
        envelope = json.dumps({"ok": False, "error": "lost the race"})
        tool = _Send(ToolResult(success=False, content=envelope, error="lost the race", error_code="CONFLICT"))
        result = await _call(_server(tool), {"session": "s1"})
        assert _text(result) == envelope
        assert _error(result) == {"code": "CONFLICT", "message": "lost the race", "retryable": True}

    @pytest.mark.asyncio
    async def test_a_servers_own_code_rides_metadata(self) -> None:
        envelope = json.dumps({"ok": False, "error": "tainted", "error_code": "tainted"})
        tool = _Send(ToolResult(success=False, content=envelope, metadata={"error_code": "tainted"}))
        result = await _call(_server(tool, error_codes=ErrorCodes([TAINTED])), {"session": "s1"})
        assert _text(result) == envelope
        assert _error(result)["code"] == "tainted"

    @pytest.mark.asyncio
    async def test_a_failure_naming_no_code_is_tool_error_and_an_unregistered_one_handler_error(self) -> None:
        plain = _Send(ToolResult(success=False, content="", error="it broke"))
        detail = _error(await _call(_server(plain), {"session": "s1"}))
        assert (detail["code"], detail["message"]) == ("TOOL_ERROR", "it broke")
        unknown = _Send(ToolResult(success=False, content="x", metadata={"error_code": "tainted"}))
        assert _error(await _call(_server(unknown), {"session": "s1"}))["code"] == "HANDLER_ERROR"

    @pytest.mark.asyncio
    async def test_an_exception_from_execute_is_handler_error(self) -> None:
        tool = _Send(OSError("disk"))
        detail = _error(await _call(_server(tool), {"session": "s1"}))
        assert detail["code"] == "HANDLER_ERROR"
        assert "OSError" in detail["message"]


class TestTheCallContext:
    @pytest.mark.asyncio
    async def test_an_execute_that_asks_for_it_gets_the_calls_context(self) -> None:
        tool = _WithContext(ToolResult(success=True, content=""))
        result = await _call(_server(tool), {"session": "s1"})
        assert _text(result) == "session_send"
        [call] = tool.calls
        assert isinstance(call["ctx"], CallContext)
        assert call["ctx"].request_id is not None
        assert call["session"] == "s1"

    @pytest.mark.asyncio
    async def test_the_same_tool_runs_without_one_as_the_mesh_calls_it(self) -> None:
        result = await _WithContext(ToolResult(success=True, content="")).run(session="s1")
        assert result.content == "no context"

    @pytest.mark.asyncio
    async def test_an_execute_that_does_not_ask_gets_its_arguments_alone(self) -> None:
        tool = _Send(ToolResult(success=True, content="ok"))
        await _call(_server(tool), {"session": "s1"})
        assert tool.calls == [{"session": "s1"}]


class TestRefusals:
    def test_a_tool_without_the_mcp_face_is_refused(self) -> None:
        tool = _Send(ToolResult(success=True, content=""))
        tool.face_mcp = False
        with pytest.raises(ValueError, match="face_mcp"):
            mcp_tool(tool, required_permission="p")

    def test_an_input_named_as_the_faces_context_argument_is_refused(self) -> None:
        class _Clash(_Send):
            def mcp_schema(self) -> MCPToolDefinition:
                return MCPToolDefinition(
                    name="example.clash",
                    version="1.0.0",
                    description="d",
                    input_schema={"type": "object", "properties": {"tears_call_context": {"type": "string"}}},
                )

        with pytest.raises(ValueError, match="input property"):
            mcp_tool(_Clash(ToolResult(success=True, content="")), required_permission="p")


def test_every_result_code_is_one_of_the_mcp_servers_built_in_codes_with_its_meaning() -> None:
    """the error model is aligned with the ToolResult vocabulary: each code is known, under the same spelling."""
    assert set(TOOL_RESULT_ERROR_CODES) <= set(BUILTIN_ERROR_CODES)
