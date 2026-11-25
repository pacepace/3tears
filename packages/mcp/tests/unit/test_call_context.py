"""a tool handler asks for the call's context by signature; one that does not is called as before.

the context (:class:`threetears.mcp.CallContext`) carries the client's progress token, a progress
reporter, a log channel to the client and a cancellation signal, with no MCP SDK type in sight.
the server finds the parameter annotated ``CallContext`` when the :class:`McpTool` is built and
passes the context in it; the wire tests drive a real MCP client over the server's own
Streamable-HTTP app, so what a handler sees is what a call from a client gives it.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any, Optional  # noqa: F401 -- Annotated and Optional are evaluated from annotation text
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import anyio
import httpx
import mcp
import mcp.types as mcp_types
import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from starlette.applications import Starlette
from threetears.mcp import CallContext, LogLevel
from threetears.mcp.auth import Identity
from threetears.mcp.server import McpServer
from threetears.mcp.tool import McpTool, ToolRegistry

from .conftest import LoopbackMcp

_EMPTY_SCHEMA: dict[str, Any] = {"type": "object", "properties": {}}


class _Recorder:
    """senders for a :class:`CallContext` built by hand: each call is recorded, nothing is sent."""

    def __init__(self) -> None:
        self.progress: list[tuple[float, float | None, str | None]] = []
        self.logs: list[tuple[LogLevel, Any, str | None]] = []

    async def send_progress(self, progress: float, total: float | None, message: str | None) -> None:
        self.progress.append((progress, total, message))

    async def send_log(self, level: LogLevel, data: Any, logger: str | None) -> None:
        self.logs.append((level, data, logger))


def _context(*, progress_token: str | int | None = "tok", recorder: _Recorder | None = None) -> CallContext:
    """a context for tool ``t`` whose senders record into ``recorder``."""
    recorder = recorder if recorder is not None else _Recorder()
    return CallContext(
        tool_name="t",
        request_id=7,
        progress_token=progress_token,
        progress_sender=recorder.send_progress,
        log_sender=recorder.send_log,
    )


def _tool(handler: Any, *, name: str = "probe", input_schema: dict[str, Any] | None = None) -> McpTool:
    return McpTool(
        name=name,
        description="probe tool",
        input_schema=input_schema if input_schema is not None else _EMPTY_SCHEMA,
        required_permission="t.probe.call",
        handler=handler,
    )


def _server(*tools: McpTool) -> McpServer:
    """a server exposing ``tools`` to an admin caller, every call allowed."""
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    identity_provider = MagicMock()
    identity_provider.identify = AsyncMock(
        return_value=Identity(principal_type="user", principal_id=uuid4(), is_admin=True),
    )
    authorizer = MagicMock()
    authorizer.allows = AsyncMock(return_value=True)
    authorizer.start = AsyncMock()
    authorizer.stop = AsyncMock()
    return McpServer(name="ctx-test", identity_provider=identity_provider, authorizer=authorizer, registry=registry)


@asynccontextmanager
async def _client(server: McpServer, **app_options: Any) -> AsyncIterator[ClientSession]:
    """an initialized MCP client session on ``server``'s Streamable-HTTP app, served in-process."""
    await server.start()
    app: Starlette = server.build_http_app(**app_options)
    http = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://mcp.test")
    try:
        async with app.router.lifespan_context(app), http:
            async with streamable_http_client("http://mcp.test/mcp", http_client=http) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    yield session
    finally:
        await server.stop()


def _text(result: mcp_types.CallToolResult) -> str:
    block = result.content[0]
    assert isinstance(block, mcp_types.TextContent)
    return block.text


async def _ignore_progress(_progress: float, _total: float | None, _message: str | None) -> None:
    """a progress callback whose only job is to make the client send a progress token."""


class TestCallContextOnItsOwn:
    """the context as a handler's own test builds it: senders that record."""

    @pytest.mark.asyncio
    async def test_progress_goes_to_the_sender_when_the_client_asked_for_it(self) -> None:
        recorder = _Recorder()
        ctx = _context(recorder=recorder)
        await ctx.report_progress(1, 3, "cloning")
        await ctx.report_progress(2.5)
        assert recorder.progress == [(1, 3, "cloning"), (2.5, None, None)]

    @pytest.mark.asyncio
    async def test_progress_is_not_sent_when_the_client_sent_no_token(self) -> None:
        recorder = _Recorder()
        ctx = _context(progress_token=None, recorder=recorder)
        await ctx.report_progress(1, 3, "cloning")
        assert recorder.progress == []

    @pytest.mark.asyncio
    async def test_progress_that_does_not_increase_is_refused(self) -> None:
        """the protocol requires each progress value to exceed the last; the handler hears of a bug."""
        ctx = _context()
        await ctx.report_progress(2)
        with pytest.raises(ValueError, match="increase"):
            await ctx.report_progress(2)
        with pytest.raises(ValueError, match="increase"):
            await ctx.report_progress(1)

    @pytest.mark.asyncio
    async def test_the_increase_rule_holds_without_a_token_too(self) -> None:
        """a handler's progress bug shows in a test that sends no token, not only in production."""
        ctx = _context(progress_token=None)
        await ctx.report_progress(5)
        with pytest.raises(ValueError, match="increase"):
            await ctx.report_progress(4)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(("progress", "total"), [(math.nan, None), (math.inf, None), (1.0, math.inf)])
    async def test_a_non_finite_number_is_refused(self, progress: float, total: float | None) -> None:
        with pytest.raises(ValueError, match="finite"):
            await _context().report_progress(progress, total)

    @pytest.mark.asyncio
    async def test_a_log_entry_goes_to_the_sender(self) -> None:
        recorder = _Recorder()
        ctx = _context(recorder=recorder)
        await ctx.log("info", {"phase": "clone"})
        await ctx.log("error", "failed", logger="builder.create")
        assert recorder.logs == [("info", {"phase": "clone"}, None), ("error", "failed", "builder.create")]

    @pytest.mark.asyncio
    async def test_a_context_without_senders_sends_nothing_and_does_not_fail(self) -> None:
        ctx = CallContext(tool_name="t", progress_token="tok")
        await ctx.report_progress(1)
        await ctx.log("info", "x")
        assert ctx.request_id is None

    def test_the_identifiers_are_the_ones_given(self) -> None:
        ctx = _context(progress_token=42)
        assert (ctx.tool_name, ctx.request_id, ctx.progress_token) == ("t", 7, 42)

    @pytest.mark.asyncio
    async def test_cancellation_is_seen_once_the_event_is_set(self) -> None:
        event = asyncio.Event()
        ctx = CallContext(tool_name="t", cancelled=event)
        assert ctx.cancelled is False
        event.set()
        assert ctx.cancelled is True
        with anyio.fail_after(1):
            await ctx.wait_cancelled()


def _handler_annotated(written: str) -> Any:
    """a handler whose ``ctx`` parameter is annotated with the text ``written``, defaulting to ``None``."""

    async def handler(*, ctx: Any = None) -> str:
        return "ok"

    handler.__annotations__ = {"ctx": written, "return": "str"}
    return handler


async def _plain(**kwargs: Any) -> str:
    return json.dumps(kwargs)


async def _with_context(*, value: str = "", ctx: CallContext) -> str:
    return json.dumps({"value": value, "tool": ctx.tool_name})


class TestTheHandlerDeclaresTheContext:
    """the context parameter is found from the handler's signature when the tool is built."""

    def test_a_handler_without_one_has_none(self) -> None:
        assert _tool(_plain).context_parameter is None

    def test_a_keyword_parameter_annotated_call_context_is_found(self) -> None:
        assert _tool(_with_context).context_parameter == "ctx"

    def test_a_positional_or_keyword_parameter_is_found_under_its_own_name(self) -> None:
        async def handler(call: CallContext, **kwargs: Any) -> str:
            return "ok"

        assert _tool(handler).context_parameter == "call"

    def test_an_annotation_that_cannot_be_evaluated_is_read_as_written(self) -> None:
        """a module that imports the type only for type checking still declares the context."""

        async def handler(*, ctx: threetears.mcp.CallContext) -> str:  # type: ignore[name-defined]  # noqa: F821
            return "ok"

        assert _tool(handler).context_parameter == "ctx"

    @pytest.mark.parametrize("written", ["CallContext | None", "None | CallContext", "Optional[CallContext]"])
    def test_a_context_that_may_be_none_is_found(self, written: str) -> None:
        """``CallContext | None`` keeps the handler callable without one; the server still passes it."""
        handler = _handler_annotated(written)
        assert _tool(handler).context_parameter == "ctx"

    def test_an_optional_context_that_cannot_be_evaluated_is_read_as_written(self) -> None:
        async def handler(*, ctx: threetears.mcp.CallContext | None = None) -> str:  # type: ignore[name-defined]  # noqa: F821
            return "ok"

        assert _tool(handler).context_parameter == "ctx"

    @pytest.mark.parametrize("written", ["list[CallContext]", "CallContext | int", "dict[str, CallContext | None]"])
    def test_a_context_in_any_other_form_is_refused(self, written: str) -> None:
        """the server cannot pass a context there, and the handler would otherwise run without one."""
        with pytest.raises(ValueError, match="names CallContext in a form the server cannot pass"):
            _tool(_handler_annotated(written))

    @pytest.mark.parametrize("written", ["Annotated[CallContext, 'meta']", "Annotated[CallContext | None, 'meta']"])
    def test_an_annotated_context_is_found(self, written: str) -> None:
        """``Annotated`` metadata does not hide the type it wraps."""
        assert _tool(_handler_annotated(written)).context_parameter == "ctx"

    def test_an_annotated_list_of_contexts_is_refused(self) -> None:
        with pytest.raises(ValueError, match="names CallContext in a form the server cannot pass"):
            _tool(_handler_annotated("Annotated[list[CallContext], 'meta']"))

    def test_a_failing_annotation_elsewhere_does_not_hide_the_context(self) -> None:
        """another parameter whose annotation raises when evaluated leaves this one evaluated."""

        async def handler(*, count: int, ctx: CallContext) -> str:
            return "ok"

        # `int | 'oops'` raises TypeError when evaluated
        handler.__annotations__ = {"count": "int | 'oops'", "ctx": "CallContext", "return": "str"}

        schema = {"type": "object", "properties": {"count": {"type": "integer"}}}
        assert _tool(handler, input_schema=schema).context_parameter == "ctx"

    def test_an_unevaluable_annotation_that_is_not_an_input_is_warned_about(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """an alias imported only for type checking cannot be recognised; the operator is told so."""

        async def handler(*, ctx: Ctx | None = None) -> str:  # type: ignore[name-defined]  # noqa: F821
            return "ok"

        with caplog.at_level(logging.WARNING, logger="threetears.mcp.tool"):
            tool = _tool(handler)
        assert tool.context_parameter is None
        warnings = [r.getMessage() for r in caplog.records if r.name == "threetears.mcp.tool"]
        assert len(warnings) == 1
        assert "'Ctx | None'" in warnings[0]
        assert "'ctx'" in warnings[0]

    def test_an_unevaluable_input_property_is_not_warned_about(self, caplog: pytest.LogCaptureFixture) -> None:
        async def handler(*, row: Row) -> str:  # type: ignore[name-defined]  # noqa: F821
            return "ok"

        schema = {"type": "object", "properties": {"row": {"type": "object"}}}
        with caplog.at_level(logging.WARNING, logger="threetears.mcp.tool"):
            assert _tool(handler, input_schema=schema).context_parameter is None
        assert [r for r in caplog.records if r.name == "threetears.mcp.tool"] == []

    def test_two_context_parameters_are_refused(self) -> None:
        async def handler(*, a: CallContext, b: CallContext) -> str:
            return "ok"

        with pytest.raises(ValueError, match="more than one CallContext"):
            _tool(handler)

    def test_a_positional_only_context_is_refused(self) -> None:
        async def handler(ctx: CallContext, /) -> str:
            return "ok"

        with pytest.raises(ValueError, match="by keyword"):
            _tool(handler)

    def test_a_context_named_as_an_input_property_is_refused(self) -> None:
        schema = {"type": "object", "properties": {"ctx": {"type": "string"}}}
        with pytest.raises(ValueError, match="also an input property"):
            _tool(_with_context, input_schema=schema)

    def test_tools_with_the_same_fields_compare_equal_whatever_they_derive(self) -> None:
        assert _tool(_with_context) == _tool(_with_context)


class TestOverTheWire:
    """what a handler is given when a real client calls it."""

    @pytest.mark.asyncio
    async def test_a_handler_without_the_context_gets_its_arguments_and_nothing_else(self) -> None:
        schema = {"type": "object", "properties": {"value": {"type": "string"}}}
        async with _client(_server(_tool(_plain, input_schema=schema))) as session:
            result = await session.call_tool("probe", {"value": "hi"}, progress_callback=_ignore_progress)
        assert result.is_error is False
        assert json.loads(_text(result)) == {"value": "hi"}

    @pytest.mark.asyncio
    async def test_a_handler_that_asks_gets_the_calls_identifiers(self) -> None:
        seen: list[CallContext] = []

        async def handler(*, ctx: CallContext) -> str:
            seen.append(ctx)
            return "ok"

        async with _client(_server(_tool(handler))) as session:
            with_token = await session.call_tool("probe", {}, progress_callback=_ignore_progress)
            without_token = await session.call_tool("probe", {})
        assert with_token.is_error is False
        assert without_token.is_error is False
        asked, not_asked = seen
        assert asked.tool_name == "probe"
        assert asked.request_id is not None
        assert asked.progress_token is not None
        assert not_asked.progress_token is None
        assert asked.cancelled is False

    @pytest.mark.asyncio
    async def test_a_handler_whose_context_may_be_none_is_passed_one(self) -> None:
        seen: list[CallContext | None] = []

        async def handler(*, ctx: CallContext | None = None) -> str:
            seen.append(ctx)
            return "ok"

        async with _client(_server(_tool(handler))) as session:
            result = await session.call_tool("probe", {})
        assert result.is_error is False
        assert len(seen) == 1
        assert isinstance(seen[0], CallContext)
        assert seen[0].tool_name == "probe"

    @pytest.mark.asyncio
    async def test_an_argument_named_as_the_context_is_refused_before_the_handler_runs(self) -> None:
        """a schema admitting extra properties must not let a caller supply the context."""
        ran: list[bool] = []

        async def handler(*, ctx: CallContext, **kwargs: Any) -> str:
            ran.append(True)
            return "ok"

        async with _client(_server(_tool(handler))) as session:
            result = await session.call_tool("probe", {"ctx": "forged"})
        assert result.is_error is True
        assert "'ctx' is not an argument of this tool" in _text(result)
        assert ran == []

    @pytest.mark.asyncio
    async def test_progress_and_a_log_entry_reach_the_client(self) -> None:
        """the reporter and the log channel are the call's own: the client's callbacks receive them."""
        received: list[Any] = []

        async def handler(*, ctx: CallContext) -> str:
            await ctx.report_progress(1, 2, "half")
            await ctx.log("info", {"phase": "half"}, logger="probe")
            return "ok"

        async def on_progress(progress: float, total: float | None, message: str | None) -> None:
            received.append(("progress", progress, total, message))

        async def on_log(params: mcp_types.LoggingMessageNotificationParams) -> None:
            received.append(("log", params.level, params.data, params.logger))

        server = _server(_tool(handler))
        await server.start()
        app = server.build_http_app(json_response=False)
        http = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://mcp.test")
        try:
            async with app.router.lifespan_context(app), http:
                async with streamable_http_client("http://mcp.test/mcp", http_client=http) as (read, write):
                    async with ClientSession(read, write, logging_callback=on_log) as session:
                        await session.initialize()
                        result = await session.call_tool("probe", {}, progress_callback=on_progress)
        finally:
            await server.stop()
        assert result.is_error is False
        assert received == [("progress", 1.0, 2.0, "half"), ("log", "info", {"phase": "half"}, "probe")]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("opt_in", "expected"),
        [(None, []), ("warning", [("error", "bad")]), ("debug", [("info", "routine"), ("error", "bad")])],
        ids=["no-opt-in", "warning", "debug"],
    )
    async def test_on_2026_07_28_a_log_entry_is_sent_only_at_a_level_the_request_opted_in_to(
        self,
        opt_in: str | None,
        expected: list[tuple[str, str]],
    ) -> None:
        """the protocol's per-request opt-in (the ``io.modelcontextprotocol/logLevel`` key of ``_meta``).

        ``mcp.Client`` settles on 2026-07-28; without the key the server must send no log entry.
        """
        received: list[tuple[str, Any]] = []

        async def handler(*, ctx: CallContext) -> str:
            await ctx.log("info", "routine")
            await ctx.log("error", "bad")
            return "ok"

        async def on_log(params: mcp_types.LoggingMessageNotificationParams) -> None:
            received.append((params.level, params.data))

        server = _server(_tool(handler))
        await server.start()
        app = server.build_http_app(json_response=False)
        http = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://mcp.test")
        try:
            async with app.router.lifespan_context(app), http:
                async with mcp.Client(
                    streamable_http_client("http://mcp.test/mcp", http_client=http), logging_callback=on_log
                ) as client:
                    assert client.protocol_version == "2026-07-28"
                    meta = {mcp_types.LOG_LEVEL_META_KEY: opt_in} if opt_in is not None else None
                    result = await client.call_tool("probe", {}, meta=meta)
        finally:
            await server.stop()
        assert result.is_error is False
        assert received == expected

    @pytest.mark.asyncio
    async def test_a_cancelled_call_sets_the_signal_work_outside_the_handler_waits_on(
        self,
        loopback_mcp: LoopbackMcp,
    ) -> None:
        """the client gives up on the call; a task the test owns, waiting on the context, wakes.

        over a real connection: an in-process transport never tells the app the client went away.
        """
        started = anyio.Event()
        contexts: list[CallContext] = []
        woke: list[bool] = []

        async def handler(*, ctx: CallContext) -> str:
            contexts.append(ctx)
            started.set()
            await anyio.sleep(30)
            return "never"

        async def watch() -> None:
            await started.wait()
            await contexts[0].wait_cancelled()
            woke.append(True)

        async with loopback_mcp(_server(_tool(handler)), json_response=False) as url:
            async with anyio.create_task_group() as watchers, httpx.AsyncClient() as http:
                watchers.start_soon(watch)
                async with streamable_http_client(url, http_client=http) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        with anyio.move_on_after(0.5):
                            await session.call_tool("probe", {})
                with anyio.fail_after(5):
                    await started.wait()
                    while not woke:
                        await anyio.sleep(0.01)
        assert contexts[0].cancelled is True
        assert woke == [True]
