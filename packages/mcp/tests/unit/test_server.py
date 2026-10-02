"""unit tests for :class:`threetears.mcp.server.McpServer` -- dispatch + RBAC + error mapping.

constructs a minimal `McpServer` against an in-memory `ToolRegistry`,
then calls tools the way a client does: a real MCP ``ClientSession``
over the server's own Streamable-HTTP app (``build_http_app``, served
in-process through ``httpx.ASGITransport``). Every assertion is on the
``CallToolResult`` the client receives, so the tests cover what reaches
the wire -- including the SDK's wrapping of a content list into
``isError=False`` -- rather than an intermediate return value.

exercises the four error paths (unknown tool, identity failure,
authorizer raise, denied), handler failure, and the happy path.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import httpx
import mcp.types as mcp_types
import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from threetears.mcp.auth import Identity
from threetears.mcp.server import McpServer
from threetears.mcp.tool import McpTool, ToolRegistry


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------


def _registry_with(tool: McpTool) -> ToolRegistry:
    """build an isolated registry with ``tool`` registered."""
    registry = ToolRegistry()
    registry.register(tool)
    return registry


def _identity_provider(*, identity: Identity) -> Any:
    """fake IdentityProvider returning a fixed identity."""
    provider = MagicMock()
    provider.identify = AsyncMock(return_value=identity)
    return provider


def _authorizer(*, allows: bool = True) -> Any:
    """fake Authorizer returning the requested decision."""
    authz = MagicMock()
    authz.allows = AsyncMock(return_value=allows)
    authz.start = AsyncMock()
    authz.stop = AsyncMock()
    return authz


def _make_tool(
    *,
    name: str = "probe",
    permission: str = "t.probe.read",
    handler: Any | None = None,
) -> McpTool:
    """build a sample McpTool descriptor."""
    if handler is None:

        async def _h(**kwargs: Any) -> str:  # noqa: ARG001
            return "default-result"

        handler = _h
    return McpTool(
        name=name,
        description="probe tool",
        input_schema={"type": "object", "properties": {}},
        required_permission=permission,
        handler=handler,
    )


async def _call_tool(server: McpServer, tool_name: str, arguments: dict[str, Any]) -> mcp_types.CallToolResult:
    """call ``tool_name`` on ``server`` through a real MCP client session.

    starts the server, serves its Streamable-HTTP app in-process, and
    returns the ``CallToolResult`` exactly as the client receives it.

    :param server: server under test
    :ptype server: McpServer
    :param tool_name: tool to call
    :ptype tool_name: str
    :param arguments: tool arguments
    :ptype arguments: dict[str, Any]
    :return: the client-side call result
    :rtype: mcp_types.CallToolResult
    """
    await server.start()
    app = server.build_http_app()
    client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://mcp.test")
    try:
        async with app.router.lifespan_context(app), client:
            async with streamable_http_client("http://mcp.test/mcp", http_client=client) as (read, write, _sid):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool(tool_name, arguments)
    finally:
        await server.stop()
    return result


def _payload_from_result(result: mcp_types.CallToolResult) -> dict[str, Any]:
    """parse the JSON envelope inside a CallToolResult error content."""
    text_block = result.content[0]
    assert isinstance(text_block, mcp_types.TextContent)
    return json.loads(text_block.text)


def _content_text(result: mcp_types.CallToolResult) -> str:
    """extract the text of the first content block of a result."""
    block = result.content[0]
    assert isinstance(block, mcp_types.TextContent)
    return block.text


# ---------------------------------------------------------------------
# happy path
# ---------------------------------------------------------------------


class TestDispatchHappyPath:
    @pytest.mark.asyncio
    async def test_allowed_dispatch_returns_text_content_list(self) -> None:
        """authorizer-allowed call reaches the client as one text block with isError=False."""

        async def handler(**kwargs: Any) -> str:
            return f"called with {kwargs}"

        tool = _make_tool(handler=handler)
        identity = Identity(principal_type="user", principal_id=uuid4())
        server = McpServer(
            name="test",
            identity_provider=_identity_provider(identity=identity),
            authorizer=_authorizer(allows=True),
            registry=_registry_with(tool),
        )
        result = await _call_tool(server, "probe", {"foo": "bar"})
        # the server returns list[TextContent] and the SDK's call_tool decorator
        # wraps it in CallToolResult(isError=False, ...) at the protocol layer.
        assert result.isError is False
        assert len(result.content) == 1
        assert "called with {'foo': 'bar'}" in _content_text(result)
        assert result.structuredContent is None

    @pytest.mark.asyncio
    async def test_dict_handler_result_rides_both_faces(self) -> None:
        """dict result is pretty JSON for a model AND structuredContent for a program.

        It used to be the text face only. A structured payload that reaches
        its consumer as prose has to be re-parsed out of its own rendering,
        which is a reader guessing at a shape the producer already knew
        (search-spec.md §4.8).
        """

        async def handler(**_kwargs: Any) -> dict[str, Any]:
            return {"status": "ok", "count": 3}

        tool = _make_tool(handler=handler)
        identity = Identity(principal_type="user", principal_id=uuid4())
        server = McpServer(
            name="test",
            identity_provider=_identity_provider(identity=identity),
            authorizer=_authorizer(allows=True),
            registry=_registry_with(tool),
        )
        result = await _call_tool(server, "probe", {})
        assert result.isError is False
        assert json.loads(_content_text(result)) == {"status": "ok", "count": 3}
        assert result.structuredContent == {"status": "ok", "count": 3}

    @pytest.mark.asyncio
    async def test_a_non_dict_structured_result_is_wrapped_under_a_key(self) -> None:
        """``structuredContent`` is an object per the spec, so a list gets a key.

        Dropping the structure rather than wrapping it would make the field's
        presence depend on the handler's top-level type, which is exactly the
        kind of conditional a consumer cannot see from the schema.
        """

        async def handler(**_kwargs: Any) -> list[int]:
            return [1, 2, 3]

        tool = _make_tool(handler=handler)
        identity = Identity(principal_type="user", principal_id=uuid4())
        server = McpServer(
            name="test",
            identity_provider=_identity_provider(identity=identity),
            authorizer=_authorizer(allows=True),
            registry=_registry_with(tool),
        )
        result = await _call_tool(server, "probe", {})
        assert result.isError is False
        assert result.structuredContent == {"result": [1, 2, 3]}

    @pytest.mark.asyncio
    async def test_a_string_handler_result_still_returns_a_content_list(self) -> None:
        """Text-only handlers are untouched: no structured face, no shape change."""

        async def handler(**_kwargs: Any) -> str:
            return "plain prose"

        tool = _make_tool(handler=handler)
        identity = Identity(principal_type="user", principal_id=uuid4())
        server = McpServer(
            name="test",
            identity_provider=_identity_provider(identity=identity),
            authorizer=_authorizer(allows=True),
            registry=_registry_with(tool),
        )
        result = await _call_tool(server, "probe", {})
        assert result.isError is False
        assert len(result.content) == 1
        assert _content_text(result) == "plain prose"
        assert result.structuredContent is None

    @pytest.mark.asyncio
    async def test_handler_returning_text_content_list_passes_through(self) -> None:
        """handler that returns list[TextContent] (already wire-ready) is forwarded."""
        text_content = mcp_types.TextContent(type="text", text="prebuilt")

        async def handler(**_kwargs: Any) -> list[mcp_types.TextContent]:
            return [text_content]

        tool = _make_tool(handler=handler)
        identity = Identity(principal_type="user", principal_id=uuid4())
        server = McpServer(
            name="test",
            identity_provider=_identity_provider(identity=identity),
            authorizer=_authorizer(allows=True),
            registry=_registry_with(tool),
        )
        result = await _call_tool(server, "probe", {})
        assert result.isError is False
        assert result.content == [text_content]
        assert result.structuredContent is None

    @pytest.mark.asyncio
    async def test_handler_returning_call_tool_result_passes_through(self) -> None:
        """handler that returns a full CallToolResult (isError=True downstream envelope) is forwarded verbatim."""
        error_result = mcp_types.CallToolResult(
            content=[mcp_types.TextContent(type="text", text='{"error": {"code": "INSUFFICIENT_CREDITS"}}')],
            isError=True,
        )

        async def handler(**_kwargs: Any) -> mcp_types.CallToolResult:
            return error_result

        tool = _make_tool(handler=handler)
        identity = Identity(principal_type="user", principal_id=uuid4())
        server = McpServer(
            name="test",
            identity_provider=_identity_provider(identity=identity),
            authorizer=_authorizer(allows=True),
            registry=_registry_with(tool),
        )
        result = await _call_tool(server, "probe", {})
        # forwarded unchanged -- not re-wrapped into an isError=False text body
        assert result.isError is True
        assert result.content == error_result.content
        assert result.structuredContent is None


# ---------------------------------------------------------------------
# error paths -- structured envelopes, no raw exceptions
# ---------------------------------------------------------------------


class TestDispatchErrorPaths:
    """all error paths return ``CallToolResult(isError=True, ...)``.

    rationale: the SDK's ``call_tool`` decorator passes
    :class:`CallToolResult` returns through verbatim. wrapping
    errors in plain ``list[TextContent]`` and JSON-encoding
    ``isError`` inside the text body would surface to clients as
    ``isError=False`` at the protocol layer -- the failure signal
    would be invisible to clients that check
    ``CallToolResult.isError``.
    """

    @pytest.mark.asyncio
    async def test_unknown_tool_returns_call_tool_result_error(self) -> None:
        """call to a non-registered tool returns CallToolResult(isError=True)."""
        identity = Identity(principal_type="user", principal_id=uuid4())
        server = McpServer(
            name="test",
            identity_provider=_identity_provider(identity=identity),
            authorizer=_authorizer(allows=True),
            registry=ToolRegistry(),
        )
        result = await _call_tool(server, "nonesuch", {})
        assert result.isError is True
        envelope = _payload_from_result(result)
        assert envelope["error"]["code"] == "UNKNOWN_TOOL"
        assert envelope["tool_name"] == "nonesuch"

    @pytest.mark.asyncio
    async def test_identity_resolution_failure_returns_error(self) -> None:
        """IdentityProvider raise -> CallToolResult(isError=True, code=IDENTITY_UNAVAILABLE)."""
        provider = MagicMock()
        provider.identify = AsyncMock(side_effect=RuntimeError("env var missing"))
        server = McpServer(
            name="test",
            identity_provider=provider,
            authorizer=_authorizer(allows=True),
            registry=_registry_with(_make_tool()),
        )
        result = await _call_tool(server, "probe", {})
        assert result.isError is True
        assert _payload_from_result(result)["error"]["code"] == "IDENTITY_UNAVAILABLE"

    @pytest.mark.asyncio
    async def test_authorizer_raise_denies_with_authz_error(self) -> None:
        """Authorizer raise -> CallToolResult(isError=True, code=AUTHZ_ERROR)."""
        authz = MagicMock()
        authz.allows = AsyncMock(side_effect=RuntimeError("broken"))
        authz.start = AsyncMock()
        authz.stop = AsyncMock()
        identity = Identity(principal_type="user", principal_id=uuid4())
        server = McpServer(
            name="test",
            identity_provider=_identity_provider(identity=identity),
            authorizer=authz,
            registry=_registry_with(_make_tool()),
        )
        result = await _call_tool(server, "probe", {})
        assert result.isError is True
        assert _payload_from_result(result)["error"]["code"] == "AUTHZ_ERROR"

    @pytest.mark.asyncio
    async def test_denied_dispatch_returns_permission_denied(self) -> None:
        """authorizer returns False -> CallToolResult(isError=True, code=PERMISSION_DENIED)."""
        identity = Identity(principal_type="user", principal_id=uuid4())
        server = McpServer(
            name="test",
            identity_provider=_identity_provider(identity=identity),
            authorizer=_authorizer(allows=False),
            registry=_registry_with(_make_tool()),
        )
        result = await _call_tool(server, "probe", {})
        assert result.isError is True
        envelope = _payload_from_result(result)
        assert envelope["error"]["code"] == "PERMISSION_DENIED"
        assert "t.probe.read" in envelope["error"]["message"]

    @pytest.mark.asyncio
    async def test_handler_exception_returns_handler_error(self) -> None:
        """handler raise -> CallToolResult(isError=True, code=HANDLER_ERROR), no traceback leaked."""

        async def handler(**_kwargs: Any) -> str:
            raise ValueError("kaboom")

        tool = _make_tool(handler=handler)
        identity = Identity(principal_type="user", principal_id=uuid4())
        server = McpServer(
            name="test",
            identity_provider=_identity_provider(identity=identity),
            authorizer=_authorizer(allows=True),
            registry=_registry_with(tool),
        )
        result = await _call_tool(server, "probe", {})
        assert result.isError is True
        envelope = _payload_from_result(result)
        assert envelope["error"]["code"] == "HANDLER_ERROR"
        assert "ValueError" in envelope["error"]["message"]
        assert "kaboom" in envelope["error"]["message"]


# ---------------------------------------------------------------------
# server lifecycle
# ---------------------------------------------------------------------


class TestServerLifecycle:
    @pytest.mark.asyncio
    async def test_start_delegates_to_authorizer_start(self) -> None:
        """McpServer.start() forwards to Authorizer.start()."""
        authz = _authorizer()
        authz.stop = AsyncMock()
        server = McpServer(
            name="test",
            identity_provider=_identity_provider(
                identity=Identity(principal_type="user", principal_id=uuid4()),
            ),
            authorizer=authz,
            registry=ToolRegistry(),
        )
        await server.start()
        authz.start.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_stop_delegates_to_authorizer_stop(self) -> None:
        """McpServer.stop() forwards to Authorizer.stop()."""
        authz = _authorizer()
        authz.stop = AsyncMock()
        server = McpServer(
            name="test",
            identity_provider=_identity_provider(
                identity=Identity(principal_type="user", principal_id=uuid4()),
            ),
            authorizer=authz,
            registry=ToolRegistry(),
        )
        await server.start()
        await server.stop()
        authz.stop.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_serve_stdio_without_start_raises(self) -> None:
        """forgetting start() before serve_stdio is a clear error, not silent default-deny."""
        authz = _authorizer()
        server = McpServer(
            name="test",
            identity_provider=_identity_provider(
                identity=Identity(principal_type="user", principal_id=uuid4()),
            ),
            authorizer=authz,
            registry=ToolRegistry(),
        )
        with pytest.raises(RuntimeError, match="start"):
            await server.serve_stdio()

    def test_empty_name_rejected(self) -> None:
        """server name must be non-empty (matches Subjects discipline)."""
        with pytest.raises(ValueError, match="non-empty"):
            McpServer(
                name="",
                identity_provider=_identity_provider(
                    identity=Identity(principal_type="user", principal_id=uuid4()),
                ),
                authorizer=_authorizer(),
                registry=ToolRegistry(),
            )

    def test_name_property_exposes_constructor_value(self) -> None:
        """McpServer.name returns the configured server name."""
        server = McpServer(
            name="product-test",
            identity_provider=_identity_provider(
                identity=Identity(principal_type="user", principal_id=uuid4()),
            ),
            authorizer=_authorizer(),
            registry=ToolRegistry(),
        )
        assert server.name == "product-test"
