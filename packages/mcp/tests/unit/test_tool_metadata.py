"""a tool's metadata reaches ``tools/list``, and its structured results and errors reach ``tools/call``.

every assertion is on what a real MCP client receives over the server's own Streamable-HTTP app,
so the wire names (``outputSchema``, ``title``, ``annotations``, ``structuredContent``,
``isError``) are checked as the client sees them, on either SDK major
(``scripts/test-mcp-v1.sh`` runs this file on mcp 1.x).
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import httpx
import mcp.types as mcp_types
import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from threetears.mcp import McpToolAnnotations, tool_error
from threetears.mcp.auth import Identity
from threetears.mcp.server import McpServer
from threetears.mcp.tool import McpTool, ToolRegistry, register_tool

_RUNNER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"runners": {"type": "array", "items": {"type": "string"}}},
    "required": ["runners"],
    "additionalProperties": False,
}


def _server(*tools: McpTool) -> McpServer:
    """a server exposing ``tools`` to a caller every call is allowed for."""
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    identity = MagicMock()
    identity.identify = AsyncMock(return_value=Identity(principal_type="user", principal_id=uuid4()))
    authorizer = MagicMock()
    authorizer.allows = AsyncMock(return_value=True)
    authorizer.start = AsyncMock()
    authorizer.stop = AsyncMock()
    return McpServer(name="metadata-test", identity_provider=identity, authorizer=authorizer, registry=registry)


def _tool(handler: Any, **metadata: Any) -> McpTool:
    """a ``runners.list`` tool with ``handler`` and the given optional metadata."""
    return McpTool(
        name="runners.list",
        description="list the runners",
        input_schema={"type": "object", "properties": {}},
        required_permission="testament.runners.read",
        handler=handler,
        **metadata,
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


async def _listed(server: McpServer) -> dict[str, Any]:
    """the one tool ``server`` lists, as wire JSON."""
    async with _session(server) as session:
        listed = await session.list_tools()
    assert len(listed.tools) == 1
    wire: dict[str, Any] = listed.tools[0].model_dump(by_alias=True, exclude_none=True)
    return wire


async def _called(server: McpServer) -> dict[str, Any]:
    """the result of calling ``runners.list`` on ``server``, as wire JSON."""
    async with _session(server) as session:
        result = await session.call_tool("runners.list", {})
    wire: dict[str, Any] = result.model_dump(by_alias=True)
    return wire


def _text(wire: dict[str, Any]) -> str:
    """the text of a wire result's only content block."""
    assert len(wire["content"]) == 1
    text: str = wire["content"][0]["text"]
    return text


class TestToolsListCarriesTheMetadata:
    @pytest.mark.asyncio
    async def test_output_schema_title_and_annotations_are_listed(self) -> None:
        tool = _tool(
            AsyncMock(),
            output_schema=_RUNNER_SCHEMA,
            title="List runners",
            annotations=McpToolAnnotations(read_only_hint=True, idempotent_hint=True, open_world_hint=False),
        )
        listed = await _listed(_server(tool))
        assert listed["outputSchema"] == _RUNNER_SCHEMA
        assert listed["title"] == "List runners"
        # an unset hint is left out, so the client applies the spec's default for it
        assert listed["annotations"] == {"readOnlyHint": True, "idempotentHint": True, "openWorldHint": False}

    @pytest.mark.asyncio
    async def test_a_mutating_tool_lists_its_destructive_hint(self) -> None:
        tool = _tool(AsyncMock(), annotations=McpToolAnnotations(read_only_hint=False, destructive_hint=True))
        listed = await _listed(_server(tool))
        assert listed["annotations"] == {"readOnlyHint": False, "destructiveHint": True}

    @pytest.mark.asyncio
    async def test_a_tool_without_metadata_lists_none(self) -> None:
        listed = await _listed(_server(_tool(AsyncMock())))
        assert {"outputSchema", "title", "annotations"}.isdisjoint(listed)

    @pytest.mark.asyncio
    async def test_register_tool_carries_the_metadata(self) -> None:
        registry = ToolRegistry()

        @register_tool(
            name="projects.allow",
            description="allow a project",
            input_schema={"type": "object", "properties": {}},
            required_permission="testament.projects.write",
            registry=registry,
            output_schema={"type": "object"},
            title="Allow a project",
            annotations=McpToolAnnotations(destructive_hint=False),
        )
        async def _allow() -> dict[str, Any]:
            return {}

        [tool] = registry.all()
        assert tool.output_schema == {"type": "object"}
        assert tool.title == "Allow a project"
        assert tool.annotations == McpToolAnnotations(destructive_hint=False)


class TestTheToolRefusesBadMetadataAtConstruction:
    def test_an_output_schema_that_is_not_an_object_schema(self) -> None:
        with pytest.raises(ValueError, match="type must be 'object'"):
            _tool(AsyncMock(), output_schema={"type": "array"})

    def test_an_output_schema_that_is_not_a_valid_schema(self) -> None:
        with pytest.raises(ValueError, match="not a valid JSON Schema"):
            _tool(
                AsyncMock(), output_schema={"type": "object", "properties": {"n": {"type": "integer", "minimum": "0"}}}
            )

    def test_an_empty_title(self) -> None:
        with pytest.raises(ValueError, match="title must be non-empty"):
            _tool(AsyncMock(), title="")


class TestStructuredResults:
    @pytest.mark.asyncio
    async def test_a_result_that_matches_the_output_schema_goes_out_in_both_faces(self) -> None:
        payload = {"runners": ["r1", "r2"]}
        wire = await _called(_server(_tool(AsyncMock(return_value=payload), output_schema=_RUNNER_SCHEMA)))
        assert wire["isError"] is False
        assert wire["structuredContent"] == payload
        assert json.loads(_text(wire)) == payload

    @pytest.mark.asyncio
    async def test_a_result_that_breaks_the_output_schema_is_refused(self) -> None:
        handler = AsyncMock(return_value={"runners": [1, 2]})
        wire = await _called(_server(_tool(handler, output_schema=_RUNNER_SCHEMA)))
        assert wire["isError"] is True
        assert wire.get("structuredContent") is None
        assert json.loads(_text(wire))["error"]["code"] == "OUTPUT_VALIDATION_ERROR"
        handler.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_a_uuid_or_datetime_is_checked_as_the_string_the_client_receives(self) -> None:
        runner_id = uuid4()
        started_at = datetime(2026, 10, 23, 15, 10, 5, tzinfo=UTC)
        schema = {
            "type": "object",
            "properties": {"runner_id": {"type": "string"}, "started_at": {"type": "string"}},
            "required": ["runner_id", "started_at"],
        }
        handler = AsyncMock(return_value={"runner_id": runner_id, "started_at": started_at})
        wire = await _called(_server(_tool(handler, output_schema=schema)))
        assert wire["isError"] is False
        assert UUID(wire["structuredContent"]["runner_id"]) == runner_id
        assert datetime.fromisoformat(wire["structuredContent"]["started_at"]) == started_at

    @pytest.mark.asyncio
    async def test_a_refused_result_is_logged_with_what_the_tool_returned(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        handler = AsyncMock(return_value={"runners": [1, 2]})
        with caplog.at_level(logging.ERROR, logger="threetears.mcp"):
            await _called(_server(_tool(handler, output_schema=_RUNNER_SCHEMA)))
        (record,) = [record for record in caplog.records if record.name == "threetears.mcp.server"]
        logged = record.__dict__["extra_data"]
        assert logged["tool_name"] == "runners.list"
        assert json.loads(logged["structured_content"]) == {"runners": [1, 2]}

    @pytest.mark.asyncio
    async def test_a_large_refused_result_is_logged_cut_short(self, caplog: pytest.LogCaptureFixture) -> None:
        handler = AsyncMock(return_value={"runners": list(range(10_000))})
        with caplog.at_level(logging.ERROR, logger="threetears.mcp"):
            wire = await _called(_server(_tool(handler, output_schema=_RUNNER_SCHEMA)))
        assert json.loads(_text(wire))["error"]["code"] == "OUTPUT_VALIDATION_ERROR"
        (record,) = [record for record in caplog.records if record.name == "threetears.mcp.server"]
        logged: str = record.__dict__["extra_data"]["structured_content"]
        assert logged.startswith('{"runners": [0, 1, 2')
        assert logged.endswith("...[truncated]")
        assert len(logged) < len(json.dumps({"runners": list(range(10_000))}))

    @pytest.mark.asyncio
    async def test_a_tool_with_an_output_schema_that_returns_only_text_is_refused(self) -> None:
        wire = await _called(_server(_tool(AsyncMock(return_value="r1, r2"), output_schema=_RUNNER_SCHEMA)))
        assert wire["isError"] is True
        error = json.loads(_text(wire))["error"]
        assert error["code"] == "OUTPUT_VALIDATION_ERROR"
        assert "no structured content" in error["message"]

    @pytest.mark.asyncio
    async def test_a_returned_call_tool_result_is_checked_too(self) -> None:
        returned = mcp_types.CallToolResult.model_validate(
            {"content": [{"type": "text", "text": "{}"}], "structuredContent": {}, "isError": False},
        )
        wire = await _called(_server(_tool(AsyncMock(return_value=returned), output_schema=_RUNNER_SCHEMA)))
        assert wire["isError"] is True
        assert json.loads(_text(wire))["error"]["code"] == "OUTPUT_VALIDATION_ERROR"


class TestToolError:
    @pytest.mark.asyncio
    async def test_a_tool_error_goes_out_in_both_faces_as_the_handler_built_it(self) -> None:
        error = {"code": "NOT_FOUND", "message": "no runner 'r9'", "retryable": False, "field": "runner"}
        wire = await _called(_server(_tool(AsyncMock(return_value=tool_error(error)))))
        assert wire["isError"] is True
        assert wire["structuredContent"] == error
        assert json.loads(_text(wire)) == error

    @pytest.mark.asyncio
    async def test_the_output_schema_does_not_apply_to_a_tool_error(self) -> None:
        error = {"code": "CONFLICT", "message": "runner busy", "retryable": True, "field": None}
        wire = await _called(_server(_tool(AsyncMock(return_value=tool_error(error)), output_schema=_RUNNER_SCHEMA)))
        assert wire["isError"] is True
        assert wire["structuredContent"] == error

    @pytest.mark.asyncio
    async def test_a_tool_error_is_the_tools_answer_not_a_server_failure(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        error = {"code": "NOT_FOUND", "message": "no runner 'r9'", "retryable": False, "field": "runner"}
        with caplog.at_level(logging.WARNING, logger="threetears.mcp"):
            await _called(_server(_tool(AsyncMock(return_value=tool_error(error)))))
        assert [record for record in caplog.records if record.name.startswith("threetears.mcp")] == []
