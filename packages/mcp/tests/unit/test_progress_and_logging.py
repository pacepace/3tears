"""a handler's progress and log entries reach a real MCP client over Streamable HTTP, before the result.

the SDK's own clients over real loopback connections: ``ClientSession`` speaking the handshake-era
protocol (2025-11-25) and ``mcp.Client`` settling on 2026-07-28, against the transport in stateless
and in stateful mode. "before the result" is shown, not inferred from order: the handler waits, after
each notification, until the client's callback has received it, so a notification held back until the
result would time the call out.

a transport answering in plain JSON (``json_response=True``) has no stream to carry them; the server
says so in its log, once, instead of dropping them without a word.
"""

from __future__ import annotations

import contextlib
import logging
import warnings
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import anyio
import httpx2
import mcp
import mcp.types as mcp_types
import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp_types import LOG_LEVEL_META_KEY
from starlette.applications import Starlette
from threetears.mcp import CallContext
from threetears.mcp.auth import Identity
from threetears.mcp.server import McpServer
from threetears.mcp.tool import McpTool, ToolRegistry

from .conftest import LoopbackApp, LoopbackMcp

#: how long a handler waits for the client to have received what it just sent
_DELIVERY_DEADLINE_S = 5


class _Client:
    """the client's side of a call: its callbacks, and the order things reached it."""

    def __init__(self) -> None:
        self.received: list[tuple[Any, ...]] = []
        self.saw_progress = anyio.Event()
        self.saw_log = anyio.Event()

    async def on_progress(self, progress: float, total: float | None, message: str | None) -> None:
        self.received.append(("progress", progress, total, message))
        self.saw_progress.set()

    async def on_log(self, params: mcp_types.LoggingMessageNotificationParams) -> None:
        self.received.append(("log", params.level, params.data))
        self.saw_log.set()


def _server(handler: Any) -> McpServer:
    """a server exposing ``handler`` as tool ``build`` to an admin caller, every call allowed."""
    registry = ToolRegistry()
    registry.register(
        McpTool(
            name="build",
            description="a long tool",
            input_schema={"type": "object", "properties": {}},
            required_permission="t.build.call",
            handler=handler,
        ),
    )
    identity_provider = MagicMock()
    identity_provider.identify = AsyncMock(
        return_value=Identity(principal_type="user", principal_id=uuid4(), is_admin=True),
    )
    authorizer = MagicMock()
    authorizer.allows = AsyncMock(return_value=True)
    authorizer.start = AsyncMock()
    authorizer.stop = AsyncMock()
    return McpServer(
        name="progress-test", identity_provider=identity_provider, authorizer=authorizer, registry=registry
    )


def _phased_handler(client: _Client) -> Any:
    """a handler that reports two phases, each as progress then a log entry, waiting for each to land."""

    async def build(*, ctx: CallContext) -> str:
        for phase in (1, 2):
            client.saw_progress = anyio.Event()
            client.saw_log = anyio.Event()
            await ctx.report_progress(phase, 2, f"phase {phase}")
            with anyio.fail_after(_DELIVERY_DEADLINE_S):
                await client.saw_progress.wait()
            await ctx.log("info", {"phase": phase})
            with anyio.fail_after(_DELIVERY_DEADLINE_S):
                await client.saw_log.wait()
        return "built"

    return build


def _two_entries() -> Any:
    """a handler that logs an info and an error entry, then reports progress."""

    async def build(*, ctx: CallContext) -> str:
        await ctx.log("info", "routine")
        await ctx.log("error", "bad")
        await ctx.report_progress(1)
        return "built"

    return build


@contextlib.asynccontextmanager
async def _session(url: str, client: _Client, *, modern: bool) -> AsyncIterator[Any]:
    """an MCP client on ``url``: ``mcp.Client`` (2026-07-28) when ``modern``, else ``ClientSession``."""
    async with httpx2.AsyncClient() as http:
        if modern:
            async with mcp.Client(
                streamable_http_client(url, http_client=http), logging_callback=client.on_log
            ) as modern_client:
                assert modern_client.protocol_version == "2026-07-28"
                yield modern_client
        else:
            async with streamable_http_client(url, http_client=http) as (read, write):
                async with ClientSession(read, write, logging_callback=client.on_log) as session:
                    initialized = await session.initialize()
                    assert initialized.protocol_version == "2025-11-25"
                    yield session


def _text(result: mcp_types.CallToolResult) -> str:
    block = result.content[0]
    assert isinstance(block, mcp_types.TextContent)
    return block.text


_MODES = pytest.mark.parametrize("stateless", [True, False], ids=["stateless", "stateful"])
_PROTOCOLS = pytest.mark.parametrize("modern", [False, True], ids=["2025-11-25", "2026-07-28"])


class TestDeliveredBeforeTheResult:
    """over an SSE-answering transport each notification reaches the client while the call still runs."""

    @pytest.mark.asyncio
    @_MODES
    @_PROTOCOLS
    async def test_progress_and_log_entries_arrive_in_order_before_the_result(
        self,
        loopback_mcp: LoopbackMcp,
        stateless: bool,
        modern: bool,
    ) -> None:
        client = _Client()
        server = _server(_phased_handler(client))
        # a 2026-07-28 client receives log entries only when its request opts in (here, at debug)
        meta = {LOG_LEVEL_META_KEY: "debug"} if modern else None
        async with loopback_mcp(server, stateless=stateless, json_response=False) as url:
            async with _session(url, client, modern=modern) as session:
                result = await session.call_tool("build", {}, progress_callback=client.on_progress, meta=meta)
        client.received.append(("result", _text(result)))
        assert result.is_error is False
        assert client.received == [
            ("progress", 1.0, 2.0, "phase 1"),
            ("log", "info", {"phase": 1}),
            ("progress", 2.0, 2.0, "phase 2"),
            ("log", "info", {"phase": 2}),
            ("result", "built"),
        ]

    @pytest.mark.asyncio
    async def test_an_embedded_transport_delivers_them_too(self, loopback_app: LoopbackApp) -> None:
        """a host app serving the transport inside its own app gets the same stream."""
        client = _Client()
        server = _server(_phased_handler(client))
        await server.start()
        transport = server.build_http_transport(json_response=False)

        @contextlib.asynccontextmanager
        async def lifespan(_app: Starlette) -> AsyncIterator[None]:
            async with transport.lifespan():
                yield

        try:
            async with loopback_app(Starlette(routes=transport.routes, lifespan=lifespan)) as base:
                async with _session(f"{base}/mcp", client, modern=False) as session:
                    result = await session.call_tool("build", {}, progress_callback=client.on_progress)
        finally:
            await server.stop()
        assert result.is_error is False
        assert [entry[0] for entry in client.received] == ["progress", "log", "progress", "log"]


class TestTheLogOptIn:
    """on 2026-07-28 a log entry is sent only when the request asked for its level."""

    @pytest.mark.asyncio
    async def test_a_request_that_did_not_opt_in_gets_progress_and_no_log_entries(
        self,
        loopback_mcp: LoopbackMcp,
    ) -> None:
        client = _Client()
        async with loopback_mcp(_server(_two_entries()), json_response=False) as url:
            async with _session(url, client, modern=True) as session:
                result = await session.call_tool("build", {}, progress_callback=client.on_progress)
        assert result.is_error is False
        assert client.received == [("progress", 1.0, None, None)]

    @pytest.mark.asyncio
    async def test_a_request_that_opted_in_at_warning_gets_the_error_and_not_the_info(
        self,
        loopback_mcp: LoopbackMcp,
    ) -> None:
        client = _Client()
        async with loopback_mcp(_server(_two_entries()), json_response=False) as url:
            async with _session(url, client, modern=True) as session:
                result = await session.call_tool(
                    "build", {}, progress_callback=client.on_progress, meta={LOG_LEVEL_META_KEY: "warning"}
                )
        assert result.is_error is False
        assert client.received == [("log", "error", "bad"), ("progress", 1.0, None, None)]

    @pytest.mark.asyncio
    async def test_a_handshake_era_client_gets_every_level(self, loopback_mcp: LoopbackMcp) -> None:
        """before 2026-07-28 the protocol has no per-request opt-in."""
        client = _Client()
        async with loopback_mcp(_server(_two_entries()), json_response=False) as url:
            async with _session(url, client, modern=False) as session:
                await session.call_tool("build", {}, progress_callback=client.on_progress)
        assert client.received == [("log", "info", "routine"), ("log", "error", "bad"), ("progress", 1.0, None, None)]


# the SDK's client warns on every set_logging_level that 2026-07-28 deprecates the request; these tests speak the
# handshake-era protocol, where it is the specification's
@pytest.mark.filterwarnings("ignore::mcp.shared.exceptions.MCPDeprecationWarning")
class TestTheLoggingCapability:
    """on the handshake-era protocols the server declares ``logging`` and serves ``logging/setLevel``."""

    def test_building_the_server_raises_no_deprecation_warning(self) -> None:
        """serving the handshake-era request is a decision, not a warning on every server built."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            _server(_two_entries())

    @pytest.mark.asyncio
    @_MODES
    async def test_the_initialize_result_declares_the_logging_capability(
        self, loopback_mcp: LoopbackMcp, stateless: bool
    ) -> None:
        async with loopback_mcp(_server(_two_entries()), stateless=stateless, json_response=False) as url:
            async with httpx2.AsyncClient() as http, streamable_http_client(url, http_client=http) as (read, write):
                async with ClientSession(read, write) as session:
                    initialized = await session.initialize()
        assert initialized.protocol_version == "2025-11-25"
        assert initialized.capabilities.logging is not None

    @pytest.mark.asyncio
    async def test_a_2026_client_is_offered_no_logging_capability(self, loopback_mcp: LoopbackMcp) -> None:
        """2026-07-28 removed the capability and the request: a client there opts in per request."""
        async with loopback_mcp(_server(_two_entries()), json_response=False) as url:
            async with _session(url, _Client(), modern=True) as modern_client:
                capabilities = modern_client.server_capabilities
        assert capabilities.logging is None
        assert capabilities.tools is not None

    @pytest.mark.asyncio
    async def test_set_level_drops_the_entries_below_the_level_for_the_connections_later_calls(
        self, loopback_mcp: LoopbackMcp
    ) -> None:
        client = _Client()
        async with loopback_mcp(_server(_two_entries()), stateless=False, json_response=False) as url:
            async with _session(url, client, modern=False) as session:
                await session.set_logging_level("warning")
                result = await session.call_tool("build", {}, progress_callback=client.on_progress)
        assert result.is_error is False
        assert client.received == [("log", "error", "bad"), ("progress", 1.0, None, None)]

    @pytest.mark.asyncio
    async def test_a_later_set_level_replaces_the_earlier_one(self, loopback_mcp: LoopbackMcp) -> None:
        client = _Client()
        async with loopback_mcp(_server(_two_entries()), stateless=False, json_response=False) as url:
            async with _session(url, client, modern=False) as session:
                await session.set_logging_level("emergency")
                await session.call_tool("build", {}, progress_callback=client.on_progress)
                await session.set_logging_level("debug")
                await session.call_tool("build", {}, progress_callback=client.on_progress)
        assert client.received == [
            ("progress", 1.0, None, None),
            ("log", "info", "routine"),
            ("log", "error", "bad"),
            ("progress", 1.0, None, None),
        ]

    @pytest.mark.asyncio
    async def test_a_level_set_on_one_connection_leaves_another_at_every_level(self, loopback_mcp: LoopbackMcp) -> None:
        quiet, other = _Client(), _Client()
        async with loopback_mcp(_server(_two_entries()), stateless=False, json_response=False) as url:
            async with _session(url, quiet, modern=False) as first, _session(url, other, modern=False) as second:
                await first.set_logging_level("critical")
                await first.call_tool("build", {}, progress_callback=quiet.on_progress)
                await second.call_tool("build", {}, progress_callback=other.on_progress)
        assert quiet.received == [("progress", 1.0, None, None)]
        assert other.received == [
            ("log", "info", "routine"),
            ("log", "error", "bad"),
            ("progress", 1.0, None, None),
        ]

    @pytest.mark.asyncio
    @_MODES
    async def test_with_no_set_level_every_level_is_sent_as_before(
        self, loopback_mcp: LoopbackMcp, stateless: bool
    ) -> None:
        client = _Client()
        async with loopback_mcp(_server(_two_entries()), stateless=stateless, json_response=False) as url:
            async with _session(url, client, modern=False) as session:
                await session.call_tool("build", {}, progress_callback=client.on_progress)
        assert client.received == [("log", "info", "routine"), ("log", "error", "bad"), ("progress", 1.0, None, None)]

    @pytest.mark.asyncio
    async def test_in_stateless_mode_set_level_is_answered_and_each_call_is_its_own_connection(
        self, loopback_mcp: LoopbackMcp
    ) -> None:
        """a stateless request is a connection of its own: a level asked for has no later call on it to narrow."""
        client = _Client()
        async with loopback_mcp(_server(_two_entries()), stateless=True, json_response=False) as url:
            async with _session(url, client, modern=False) as session:
                answered = await session.set_logging_level("critical")
                await session.call_tool("build", {}, progress_callback=client.on_progress)
        assert isinstance(answered, mcp_types.EmptyResult)
        assert client.received == [("log", "info", "routine"), ("log", "error", "bad"), ("progress", 1.0, None, None)]


class TestCancellationOverTheStream:
    """a client that gives up on a call over an SSE-answering transport cancels the handler."""

    @pytest.mark.asyncio
    @_MODES
    @_PROTOCOLS
    async def test_the_handler_is_cancelled_and_the_context_says_so(
        self,
        loopback_mcp: LoopbackMcp,
        stateless: bool,
        modern: bool,
    ) -> None:
        started = anyio.Event()
        contexts: list[CallContext] = []
        cancelled_in_handler: list[bool] = []

        async def build(*, ctx: CallContext) -> str:
            contexts.append(ctx)
            started.set()
            try:
                await anyio.sleep(30)
            except anyio.get_cancelled_exc_class():
                cancelled_in_handler.append(True)
                raise
            return "never"

        client = _Client()
        async with loopback_mcp(_server(build), stateless=stateless, json_response=False) as url:
            async with _session(url, client, modern=modern) as session:
                with anyio.move_on_after(0.5):
                    await session.call_tool("build", {})
                with anyio.fail_after(_DELIVERY_DEADLINE_S):
                    await started.wait()
                    await contexts[0].wait_cancelled()
        assert cancelled_in_handler == [True]
        assert contexts[0].cancelled is True


class TestAPlainJsonTransport:
    """``json_response=True`` answers with the result alone; the server says what it could not send."""

    @pytest.mark.asyncio
    @_MODES
    async def test_nothing_reaches_the_client_and_the_server_warns_once(
        self,
        loopback_mcp: LoopbackMcp,
        caplog: pytest.LogCaptureFixture,
        stateless: bool,
    ) -> None:
        """in stateful mode too: the handler runs in the session's task, which must still see the request's transport."""

        async def build(*, ctx: CallContext) -> str:
            await ctx.report_progress(1)
            await ctx.log("info", "phase")
            return "built"

        client = _Client()
        with caplog.at_level(logging.WARNING, logger="threetears.mcp.server"):
            async with loopback_mcp(_server(build), stateless=stateless, json_response=True) as url:
                async with _session(url, client, modern=False) as session:
                    first = await session.call_tool("build", {}, progress_callback=client.on_progress)
                    second = await session.call_tool("build", {}, progress_callback=client.on_progress)
        assert (first.is_error, second.is_error) == (False, False)
        assert client.received == []
        warnings = [
            record
            for record in caplog.records
            if record.name == "threetears.mcp.server" and "json_response=False" in record.getMessage()
        ]
        assert len(warnings) == 1

    @pytest.mark.asyncio
    @_MODES
    async def test_a_handler_that_sends_nothing_raises_no_warning(
        self,
        loopback_mcp: LoopbackMcp,
        caplog: pytest.LogCaptureFixture,
        stateless: bool,
    ) -> None:
        """the warning is about what was dropped, not about the transport's choice."""

        async def build(*, ctx: CallContext) -> str:
            # no token was sent, so there is no progress to drop
            await ctx.report_progress(1)
            return "built"

        client = _Client()
        with caplog.at_level(logging.WARNING, logger="threetears.mcp.server"):
            async with loopback_mcp(_server(build), stateless=stateless, json_response=True) as url:
                async with _session(url, client, modern=False) as session:
                    result = await session.call_tool("build", {})
        assert result.is_error is False
        assert [record for record in caplog.records if "json_response=False" in record.getMessage()] == []

    @pytest.mark.asyncio
    async def test_in_stateless_mode_a_client_that_gives_up_does_not_cancel_the_handler(
        self,
        loopback_mcp: LoopbackMcp,
    ) -> None:
        """the README's table: a plain-JSON stateless call runs to its end after the client stops waiting."""
        started = anyio.Event()
        finished = anyio.Event()
        contexts: list[CallContext] = []

        async def build(*, ctx: CallContext) -> str:
            contexts.append(ctx)
            started.set()
            await anyio.sleep(1)
            finished.set()
            return "built"

        client = _Client()
        async with loopback_mcp(_server(build), stateless=True, json_response=True) as url:
            async with _session(url, client, modern=False) as session:
                with anyio.move_on_after(0.3):
                    await session.call_tool("build", {})
                with anyio.fail_after(_DELIVERY_DEADLINE_S):
                    await started.wait()
                    await finished.wait()
        assert contexts[0].cancelled is False
