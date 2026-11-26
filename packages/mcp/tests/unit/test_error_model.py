"""the error model: every failed call carries a code of the server's closed set on its protocol result.

calls go through a real MCP ``ClientSession`` over the server's own Streamable-HTTP app, so every assertion is on
the result the client receives: its ``_meta`` error (``code``, ``message``, ``retryable``, ``argument``), its
content, and its ``isError`` flag.
"""

from __future__ import annotations

import json
import logging
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import httpx2
import mcp.types as mcp_types
import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from threetears.mcp import ERROR_META_KEY, ErrorCode, ErrorCodes, McpToolError
from threetears.mcp.auth import Identity
from threetears.mcp.errors import BUILTIN_ERROR_CODES, CONFLICT, HANDLER_ERROR, TOOL_ERROR
from threetears.mcp.server import McpServer, tool_error
from threetears.mcp.tool import McpTool, ToolRegistry

NOT_FOUND = ErrorCode("not_found", "no such thing", retryable=False)
BUSY = ErrorCode("busy", "try again shortly", retryable=True)


def _server(
    handler: Any,
    *,
    input_schema: dict[str, Any] | None = None,
    error_codes: ErrorCodes | None = None,
    allows: bool = True,
    identity_error: BaseException | None = None,
    authorizer_error: BaseException | None = None,
) -> McpServer:
    registry = ToolRegistry()
    registry.register(
        McpTool(
            name="probe",
            description="probe tool",
            input_schema=input_schema or {"type": "object", "properties": {}},
            required_permission="t.probe.call",
            handler=handler,
        )
    )
    provider = MagicMock()
    if identity_error is not None:
        provider.identify = AsyncMock(side_effect=identity_error)
    else:
        provider.identify = AsyncMock(return_value=Identity(principal_type="user", principal_id=uuid4()))
    authorizer = MagicMock()
    authorizer.allows = (
        AsyncMock(side_effect=authorizer_error) if authorizer_error is not None else AsyncMock(return_value=allows)
    )
    authorizer.start = AsyncMock()
    authorizer.stop = AsyncMock()
    return McpServer(
        name="errors",
        identity_provider=provider,
        authorizer=authorizer,
        registry=registry,
        error_codes=error_codes,
    )


async def _call(server: McpServer, arguments: dict[str, Any], *, tool: str = "probe") -> mcp_types.CallToolResult:
    await server.start()
    app = server.build_http_app()
    client = httpx2.AsyncClient(transport=httpx2.ASGITransport(app=app), base_url="http://mcp.test")
    try:
        async with app.router.lifespan_context(app), client:
            async with streamable_http_client("http://mcp.test/mcp", http_client=client) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    return await session.call_tool(tool, arguments)
    finally:
        await server.stop()


def _error(result: mcp_types.CallToolResult) -> dict[str, Any]:
    assert result.is_error is True
    assert result.meta is not None
    detail = result.meta[ERROR_META_KEY]
    assert isinstance(detail, dict)
    return detail


def _text(result: mcp_types.CallToolResult) -> str:
    block = result.content[0]
    assert isinstance(block, mcp_types.TextContent)
    return block.text


async def _ok(**_: Any) -> str:
    return "ok"


class TestTheFrameworksOwnRefusals:
    @pytest.mark.asyncio
    async def test_an_unknown_tool_has_its_code_on_the_result_and_in_the_text(self) -> None:
        result = await _call(_server(_ok), {}, tool="nope")
        assert _error(result) == {
            "code": "UNKNOWN_TOOL",
            "message": "no tool named 'nope' registered",
            "retryable": False,
        }
        assert json.loads(_text(result))["error"]["code"] == "UNKNOWN_TOOL"
        assert json.loads(_text(result))["error"]["retryable"] is False

    @pytest.mark.asyncio
    async def test_a_missing_required_argument_is_invalid_arguments_naming_it(self) -> None:
        schema = {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}
        result = await _call(_server(_ok, input_schema=schema), {})
        detail = _error(result)
        assert detail["code"] == "INVALID_ARGUMENTS"
        assert detail["argument"] == "name"
        assert detail["retryable"] is False
        assert _text(result).startswith("Input validation error")

    @pytest.mark.asyncio
    async def test_a_wrong_typed_argument_is_named(self) -> None:
        schema = {"type": "object", "properties": {"count": {"type": "integer"}}}
        detail = _error(await _call(_server(_ok, input_schema=schema), {"count": "three"}))
        assert (detail["code"], detail["argument"]) == ("INVALID_ARGUMENTS", "count")

    @pytest.mark.asyncio
    async def test_an_argument_the_schema_does_not_admit_is_named(self) -> None:
        schema = {"type": "object", "properties": {}, "additionalProperties": False}
        detail = _error(await _call(_server(_ok, input_schema=schema), {"extra": 1}))
        assert (detail["code"], detail["argument"]) == ("INVALID_ARGUMENTS", "extra")

    @pytest.mark.asyncio
    async def test_identity_permission_and_authorizer_refusals_have_their_own_codes(self) -> None:
        assert _error(await _call(_server(_ok, identity_error=RuntimeError("no token")), {}))["code"] == (
            "IDENTITY_UNAVAILABLE"
        )
        assert _error(await _call(_server(_ok, allows=False), {}))["code"] == "PERMISSION_DENIED"
        detail = _error(await _call(_server(_ok, authorizer_error=ValueError("boom")), {}))
        assert (detail["code"], detail["retryable"]) == ("AUTHZ_ERROR", True)

    @pytest.mark.asyncio
    async def test_a_handler_that_raises_is_handler_error(self, caplog: pytest.LogCaptureFixture) -> None:
        async def broken(**_: Any) -> str:
            raise KeyError("x")

        with caplog.at_level(logging.WARNING):
            detail = _error(await _call(_server(broken), {}))
        assert detail["code"] == "HANDLER_ERROR"
        assert "KeyError" in detail["message"]


class TestCodesAHandlerNames:
    @pytest.mark.asyncio
    async def test_a_raised_mcp_tool_error_answers_its_registered_code(self) -> None:
        async def handler(**_: Any) -> str:
            raise McpToolError(NOT_FOUND, "no runner 'r1'", argument="runner")

        detail = _error(await _call(_server(handler, error_codes=ErrorCodes([NOT_FOUND])), {}))
        assert detail == {"code": "not_found", "message": "no runner 'r1'", "retryable": False, "argument": "runner"}

    @pytest.mark.asyncio
    async def test_a_mapped_exception_answers_its_code(self) -> None:
        codes = ErrorCodes([BUSY])
        codes.map_exception(TimeoutError, BUSY)

        async def handler(**_: Any) -> str:
            raise TimeoutError("the backend is slow")

        detail = _error(await _call(_server(handler, error_codes=codes), {}))
        assert detail == {"code": "busy", "message": "the backend is slow", "retryable": True}

    @pytest.mark.asyncio
    async def test_the_most_specific_mapping_wins(self) -> None:
        codes = ErrorCodes([BUSY, NOT_FOUND])
        codes.map_exception(LookupError, BUSY)
        codes.map_exception(KeyError, NOT_FOUND)

        async def handler(**_: Any) -> str:
            raise KeyError("k")

        assert _error(await _call(_server(handler, error_codes=codes), {}))["code"] == "not_found"

    @pytest.mark.asyncio
    async def test_a_returned_tool_error_keeps_its_content_and_carries_its_code(self) -> None:
        envelope = {"ok": False, "error": "gone", "error_code": "not_found"}

        async def handler(**_: Any) -> mcp_types.CallToolResult:
            return tool_error(envelope, code="not_found", message="gone")

        result = await _call(_server(handler, error_codes=ErrorCodes([NOT_FOUND])), {})
        assert json.loads(_text(result)) == envelope
        assert result.structured_content == envelope
        assert _error(result) == {"code": "not_found", "message": "gone", "retryable": False}

    @pytest.mark.asyncio
    async def test_a_text_tool_error_is_text_alone(self) -> None:
        async def handler(**_: Any) -> mcp_types.CallToolResult:
            return tool_error('{"ok": false}', code=CONFLICT, argument="row")

        result = await _call(_server(handler), {})
        assert _text(result) == '{"ok": false}'
        assert result.structured_content is None
        assert _error(result) == {
            "code": "CONFLICT",
            "message": '{"ok": false}',
            "retryable": True,
            "argument": "row",
        }

    @pytest.mark.asyncio
    async def test_a_failure_that_names_no_code_is_tool_error(self) -> None:
        async def handler(**_: Any) -> mcp_types.CallToolResult:
            return mcp_types.CallToolResult(
                content=[mcp_types.TextContent(type="text", text="it broke")], is_error=True
            )

        result = await _call(_server(handler), {})
        assert _error(result) == {"code": TOOL_ERROR.code, "message": "it broke", "retryable": False}
        assert _text(result) == "it broke"

    @pytest.mark.asyncio
    async def test_an_unregistered_code_is_handler_error_logged_once(self, caplog: pytest.LogCaptureFixture) -> None:
        async def handler(**_: Any) -> mcp_types.CallToolResult:
            return tool_error({"ok": False}, code="never_registered")

        server = _server(handler)
        with caplog.at_level(logging.ERROR, logger="threetears.mcp.server"):
            first = await _call(server, {})
            second = await _call(server, {})
        assert _error(first)["code"] == _error(second)["code"] == HANDLER_ERROR.code
        assert json.loads(_text(first)) == {"ok": False}
        logged = [record for record in caplog.records if "never registered" in record.getMessage()]
        assert len(logged) == 1

    @pytest.mark.asyncio
    async def test_a_raised_unregistered_code_is_handler_error(self) -> None:
        async def handler(**_: Any) -> str:
            raise McpToolError("never_registered", "x")

        assert _error(await _call(_server(handler), {}))["code"] == "HANDLER_ERROR"


class TestTheCodeSet:
    def test_the_builtin_set_holds_each_refusal_and_the_tool_result_codes(self) -> None:
        assert {
            "UNKNOWN_TOOL",
            "TOOLS_UNAVAILABLE",
            "INVALID_ARGUMENTS",
            "IDENTITY_UNAVAILABLE",
            "PERMISSION_DENIED",
            "AUTHZ_ERROR",
            "HANDLER_ERROR",
            "OUTPUT_VALIDATION_ERROR",
            "TOOL_ERROR",
            "UNKNOWN_RESOURCE",
            "CONFLICT",
            "MALFORMED_REQUEST",
            "TOOL_AUTHORIZATION_UNAVAILABLE",
            "TOOL_NOT_READY",
        } == set(BUILTIN_ERROR_CODES)

    def test_a_code_differing_only_in_case_is_refused(self) -> None:
        codes = ErrorCodes([NOT_FOUND])
        with pytest.raises(ValueError, match="already in the set"):
            codes.register(ErrorCode("NOT_FOUND", "again"))
        with pytest.raises(ValueError, match="already in the set"):
            codes.register(ErrorCode("conflict", "a builtin, lower case"))

    def test_a_malformed_code_is_refused(self) -> None:
        with pytest.raises(ValueError, match="letter"):
            ErrorCode("1bad", "starts with a digit")
        with pytest.raises(ValueError, match="description"):
            ErrorCode("ok_name", "")

    def test_mapping_an_exception_to_an_unregistered_code_is_refused(self) -> None:
        with pytest.raises(ValueError, match="register it first"):
            ErrorCodes().map_exception(KeyError, "not_found")
        codes = ErrorCodes([NOT_FOUND])
        codes.map_exception(KeyError, NOT_FOUND)
        with pytest.raises(ValueError, match="already mapped"):
            codes.map_exception(KeyError, NOT_FOUND)

    def test_the_server_lists_its_codes(self) -> None:
        server = _server(_ok, error_codes=ErrorCodes([NOT_FOUND]))
        names = [code.code for code in server.error_codes.all()]
        assert names[: len(BUILTIN_ERROR_CODES)] == list(BUILTIN_ERROR_CODES)
        assert names[-1] == "not_found"
