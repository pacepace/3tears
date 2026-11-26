"""an authorizer that asks for the call is given the tool name, the arguments and the permission, before the handler.

through a real MCP ``ClientSession`` over the server's own Streamable-HTTP app. an authorizer whose class defines
``authorize_call`` decides on the call; one that defines only ``allows`` is called as before.
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
from threetears.mcp import (
    ERROR_META_KEY,
    Authorizer,
    CallAuthorizer,
    ErrorCode,
    ErrorCodes,
    Identity,
    McpServer,
    McpTool,
    Refusal,
    ToolCall,
    ToolRegistry,
)
from threetears.mcp.auth import GrantCacheUnavailableError

TAINTED = ErrorCode("tainted", "the session is tainted")

_SCHEMA = {"type": "object", "properties": {"session": {"type": "string"}}, "required": ["session"]}


class _SessionGate:
    """refuses calls on a tainted session with a code of its own; records every call it was given."""

    def __init__(self, decision: bool | Refusal | BaseException = True) -> None:
        self.calls: list[tuple[Identity, ToolCall]] = []
        self.allows_calls = 0
        self._decision = decision

    async def allows(self, identity: Identity, permission: str) -> bool:
        self.allows_calls += 1
        return True

    async def authorize_call(self, identity: Identity, call: ToolCall) -> bool | Refusal:
        self.calls.append((identity, call))
        if isinstance(self._decision, BaseException):
            raise self._decision
        return self._decision

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None


class _PermissionOnly:
    """an authorizer of the original shape: ``allows`` alone."""

    def __init__(self) -> None:
        self.permissions: list[str] = []

    async def allows(self, identity: Identity, permission: str) -> bool:
        self.permissions.append(permission)
        return True

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None


def _server(authorizer: Any, ran: list[dict[str, Any]], *, error_codes: ErrorCodes | None = None) -> McpServer:
    async def handler(**arguments: Any) -> str:
        ran.append(arguments)
        return "ran"

    registry = ToolRegistry()
    registry.register(
        McpTool(
            name="send",
            description="send to a session",
            input_schema=_SCHEMA,
            required_permission="t.send",
            handler=handler,
        )
    )
    provider = MagicMock()
    provider.identify = AsyncMock(return_value=Identity(principal_type="user", principal_id=uuid4()))
    return McpServer(
        name="gate", identity_provider=provider, authorizer=authorizer, registry=registry, error_codes=error_codes
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
                    return await session.call_tool("send", arguments)
    finally:
        await server.stop()


def _code(result: mcp_types.CallToolResult) -> str:
    assert result.is_error is True
    assert result.meta is not None
    code = result.meta[ERROR_META_KEY]["code"]
    assert isinstance(code, str)
    return code


@pytest.mark.asyncio
async def test_the_authorizer_sees_the_tool_name_the_arguments_and_the_permission() -> None:
    gate = _SessionGate()
    ran: list[dict[str, Any]] = []
    result = await _call(_server(gate, ran), {"session": "s1"})
    assert result.is_error is False
    [(_, call)] = gate.calls
    assert (call.tool_name, dict(call.arguments), call.required_permission) == ("send", {"session": "s1"}, "t.send")
    assert gate.allows_calls == 0
    assert ran == [{"session": "s1"}]


@pytest.mark.asyncio
async def test_the_arguments_it_sees_are_read_only() -> None:
    gate = _SessionGate()
    await _call(_server(gate, []), {"session": "s1"})
    [(_, call)] = gate.calls
    with pytest.raises(TypeError):
        call.arguments["session"] = "other"  # type: ignore[index]


@pytest.mark.asyncio
async def test_a_refusal_with_its_own_code_is_answered_before_the_handler() -> None:
    ran: list[dict[str, Any]] = []
    gate = _SessionGate(Refusal(TAINTED, "session s1 is tainted", argument="session"))
    result = await _call(_server(gate, ran, error_codes=ErrorCodes([TAINTED])), {"session": "s1"})
    assert result.meta is not None
    assert result.meta[ERROR_META_KEY] == {
        "code": "tainted",
        "message": "session s1 is tainted",
        "retryable": False,
        "argument": "session",
    }
    assert ran == []


@pytest.mark.asyncio
async def test_false_is_permission_denied() -> None:
    ran: list[dict[str, Any]] = []
    assert _code(await _call(_server(_SessionGate(False), ran), {"session": "s1"})) == "PERMISSION_DENIED"
    assert ran == []


@pytest.mark.asyncio
async def test_a_refusal_naming_an_unregistered_code_is_permission_denied(caplog: pytest.LogCaptureFixture) -> None:
    ran: list[dict[str, Any]] = []
    with caplog.at_level(logging.ERROR, logger="threetears.mcp.server"):
        result = await _call(_server(_SessionGate(Refusal("tainted", "no")), ran), {"session": "s1"})
    assert _code(result) == "PERMISSION_DENIED"
    assert ran == []
    assert any("never registered" in record.getMessage() for record in caplog.records)


@pytest.mark.asyncio
async def test_an_authorizer_that_raises_denies_with_authz_error() -> None:
    ran: list[dict[str, Any]] = []
    assert _code(await _call(_server(_SessionGate(ValueError("broken")), ran), {"session": "s1"})) == "AUTHZ_ERROR"
    gate = _SessionGate(GrantCacheUnavailableError("reloading"))
    assert _code(await _call(_server(gate, ran), {"session": "s1"})) == "AUTHZ_ERROR"
    assert ran == []


@pytest.mark.asyncio
async def test_arguments_the_schema_refuses_never_reach_the_authorizer() -> None:
    gate = _SessionGate()
    assert _code(await _call(_server(gate, []), {})) == "INVALID_ARGUMENTS"
    assert gate.calls == []


@pytest.mark.asyncio
async def test_an_authorizer_with_only_allows_is_called_as_before() -> None:
    authorizer = _PermissionOnly()
    ran: list[dict[str, Any]] = []
    result = await _call(_server(authorizer, ran), {"session": "s1"})
    assert result.is_error is False
    assert authorizer.permissions == ["t.send"]


def test_the_protocols_recognise_each_shape() -> None:
    assert isinstance(_SessionGate(), CallAuthorizer)
    assert isinstance(_PermissionOnly(), Authorizer)
    assert not isinstance(_PermissionOnly(), CallAuthorizer)


_ENVELOPE = {
    "ok": False,
    "error": "session s1 is tainted",
    "error_code": "tainted",
    "data": {"tainted": {"at": "2026-11-26T10:00:00Z", "trigger": "web_fetch", "domain": "example.org"}},
}


@pytest.mark.asyncio
async def test_a_refusal_with_content_answers_exactly_that_content_and_its_code() -> None:
    ran: list[dict[str, Any]] = []
    gate = _SessionGate(Refusal(TAINTED, "session s1 is tainted", argument="session", content=_ENVELOPE))
    result = await _call(_server(gate, ran, error_codes=ErrorCodes([TAINTED])), {"session": "s1"})
    assert result.is_error is True
    [block] = result.content
    assert isinstance(block, mcp_types.TextContent)
    assert json.loads(block.text) == _ENVELOPE
    assert result.structured_content == _ENVELOPE
    assert result.meta == {
        ERROR_META_KEY: {
            "code": "tainted",
            "message": "session s1 is tainted",
            "retryable": False,
            "argument": "session",
        }
    }
    assert ran == []


@pytest.mark.asyncio
async def test_a_refusal_with_text_content_answers_that_text_alone() -> None:
    gate = _SessionGate(Refusal(TAINTED, "tainted", content='{"ok": false, "error_code": "tainted"}'))
    result = await _call(_server(gate, [], error_codes=ErrorCodes([TAINTED])), {"session": "s1"})
    [block] = result.content
    assert isinstance(block, mcp_types.TextContent)
    assert block.text == '{"ok": false, "error_code": "tainted"}'
    assert result.structured_content is None
    assert result.meta is not None
    assert result.meta[ERROR_META_KEY] == {"code": "tainted", "message": "tainted", "retryable": False}


@pytest.mark.asyncio
async def test_content_with_an_unregistered_code_keeps_the_content_and_answers_permission_denied() -> None:
    gate = _SessionGate(Refusal("never_registered", "no", content=_ENVELOPE))
    result = await _call(_server(gate, []), {"session": "s1"})
    assert result.structured_content == _ENVELOPE
    assert _code(result) == "PERMISSION_DENIED"


@pytest.mark.asyncio
async def test_a_refusal_without_content_answers_the_frameworks_refusal_text_as_before() -> None:
    gate = _SessionGate(Refusal(TAINTED, "session s1 is tainted"))
    result = await _call(_server(gate, [], error_codes=ErrorCodes([TAINTED])), {"session": "s1"})
    [block] = result.content
    assert isinstance(block, mcp_types.TextContent)
    assert json.loads(block.text) == {
        "tool_name": "send",
        "error": {"code": "tainted", "message": "session s1 is tainted", "retryable": False},
    }
    assert result.structured_content is None
