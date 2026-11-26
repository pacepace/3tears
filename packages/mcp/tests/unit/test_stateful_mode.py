"""stateful mode: the bearer resolved for every message, list changes pushed, cancellation reaching the call.

the SDK's own clients over real loopback connections. ``ClientSession`` speaks the handshake-era protocol
(2025-11-25), where a stateful transport keeps a session across requests; ``mcp.Client`` settles on 2026-07-28,
where every request stands alone and a client asks for change notifications on a ``subscriptions/listen`` stream.

the bearer tests change the token the client sends between two messages of one open session, or revoke it: a
stateful session's one server task serves every message, and each must still be resolved as the caller its own
HTTP request names (the SDK runs each message's handler in that request's contextvars; these tests hold it to it).
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator
from typing import Any
from uuid import UUID, uuid4

import anyio
import httpx2
import mcp
import mcp.types as mcp_types
import pytest
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.client.subscriptions import ToolsListChanged
from threetears.mcp import (
    ERROR_META_KEY,
    BearerTokenIdentityProvider,
    CallContext,
    Identity,
    McpServer,
    McpTool,
    ToolRegistry,
    current_bearer_token,
)

from .conftest import LoopbackMcp

_DEADLINE_S = 5

_PROTOCOLS = pytest.mark.parametrize("modern", [False, True], ids=["2025-11-25", "2026-07-28"])


class _Tokens:
    """the tokens the server accepts, each naming a principal; revoking one makes it resolve no more."""

    def __init__(self) -> None:
        self.principals: dict[str, UUID] = {}

    def issue(self, token: str) -> UUID:
        self.principals[token] = uuid4()
        return self.principals[token]

    def revoke(self, token: str) -> None:
        del self.principals[token]

    async def resolve(self, token: str) -> Identity:
        principal = self.principals.get(token)
        if principal is None:
            raise RuntimeError("token revoked")
        return Identity(principal_type="user", principal_id=principal)


class _AllowAll:
    async def allows(self, identity: Identity, permission: str) -> bool:
        return True

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None


def _server(tokens: _Tokens, handler: Any) -> McpServer:
    registry = ToolRegistry()
    registry.register(
        McpTool(
            name="whoami",
            description="the caller's principal",
            input_schema={"type": "object", "properties": {}},
            required_permission="t.whoami",
            handler=handler,
        )
    )
    return McpServer(
        name="stateful",
        identity_provider=BearerTokenIdentityProvider(resolver=tokens.resolve, token_source=current_bearer_token),
        authorizer=_AllowAll(),
        registry=registry,
    )


@contextlib.asynccontextmanager
async def _client(url: str, http: httpx2.AsyncClient, *, modern: bool, **options: Any) -> AsyncIterator[Any]:
    if modern:
        async with mcp.Client(streamable_http_client(url, http_client=http), **options) as client:
            assert client.protocol_version == "2026-07-28"
            yield client
    else:
        async with streamable_http_client(url, http_client=http) as (read, write):
            async with ClientSession(read, write, **options) as session:
                await session.initialize()
                yield session


def _code(result: mcp_types.CallToolResult) -> str | None:
    if not result.is_error:
        return None
    assert result.meta is not None
    code = result.meta[ERROR_META_KEY]["code"]
    assert isinstance(code, str)
    return code


def _text(result: mcp_types.CallToolResult) -> str:
    block = result.content[0]
    assert isinstance(block, mcp_types.TextContent)
    return block.text


class TestTheBearerIsResolvedForEveryMessage:
    @pytest.mark.asyncio
    @_PROTOCOLS
    async def test_a_revoked_token_is_refused_on_the_next_message_and_no_tool_runs(
        self, loopback_mcp: LoopbackMcp, modern: bool
    ) -> None:
        tokens = _Tokens()
        tokens.issue("old")
        ran: list[str] = []

        async def whoami() -> str:
            ran.append("ran")
            return "ok"

        async with loopback_mcp(_server(tokens, whoami), stateless=False, json_response=False) as url:
            async with httpx2.AsyncClient(headers={"Authorization": "Bearer old"}) as http:
                async with _client(url, http, modern=modern) as session:
                    assert _code(await session.call_tool("whoami", {})) is None
                    tokens.revoke("old")
                    refused = await session.call_tool("whoami", {})
        assert _code(refused) == "IDENTITY_UNAVAILABLE"
        assert ran == ["ran"]

    @pytest.mark.asyncio
    @_PROTOCOLS
    async def test_each_message_is_served_as_the_caller_its_own_request_names(
        self, loopback_mcp: LoopbackMcp, modern: bool
    ) -> None:
        tokens = _Tokens()
        first = tokens.issue("first")
        second = tokens.issue("second")
        seen: list[str | None] = []

        async def whoami() -> str:
            seen.append(current_bearer_token())
            identity = await tokens.resolve(current_bearer_token() or "")
            return str(identity.principal_id)

        async with loopback_mcp(_server(tokens, whoami), stateless=False, json_response=False) as url:
            async with httpx2.AsyncClient(headers={"Authorization": "Bearer first"}) as http:
                async with _client(url, http, modern=modern) as session:
                    before = await session.call_tool("whoami", {})
                    http.headers["Authorization"] = "Bearer second"
                    after = await session.call_tool("whoami", {})
        assert (_text(before), _text(after)) == (str(first), str(second))
        assert seen == ["first", "second"]

    @pytest.mark.asyncio
    async def test_a_message_without_a_bearer_is_not_served_as_an_earlier_caller(
        self, loopback_mcp: LoopbackMcp
    ) -> None:
        tokens = _Tokens()
        tokens.issue("only")

        async def whoami() -> str:
            return "ok"

        async with loopback_mcp(_server(tokens, whoami), stateless=False, json_response=False) as url:
            async with httpx2.AsyncClient(headers={"Authorization": "Bearer only"}) as http:
                async with _client(url, http, modern=False) as session:
                    assert _code(await session.call_tool("whoami", {})) is None
                    del http.headers["Authorization"]
                    refused = await session.call_tool("whoami", {})
        assert _code(refused) == "IDENTITY_UNAVAILABLE"


class _Notifications:
    def __init__(self) -> None:
        self.received: list[Any] = []
        self.arrived = anyio.Event()

    async def handler(self, message: Any) -> None:
        self.received.append(message)
        if _is_tools_changed(message):
            self.arrived.set()


def _is_tools_changed(message: Any) -> bool:
    inner = getattr(message, "root", message)
    return isinstance(inner, mcp_types.ToolListChangedNotification)


class TestToolsListChanged:
    @pytest.mark.asyncio
    async def test_a_stateful_session_is_offered_and_sent_the_change(self, loopback_mcp: LoopbackMcp) -> None:
        tokens = _Tokens()
        tokens.issue("t")
        server = _server(tokens, _noop)
        notes = _Notifications()
        async with loopback_mcp(server, stateless=False, json_response=False) as url:
            async with httpx2.AsyncClient(headers={"Authorization": "Bearer t"}) as http:
                async with streamable_http_client(url, http_client=http) as (read, write):
                    async with ClientSession(read, write, message_handler=notes.handler) as session:
                        initialized = await session.initialize()
                        assert initialized.capabilities.tools is not None
                        assert initialized.capabilities.tools.list_changed is True
                        await session.list_tools()
                        await server.notify_tools_changed()
                        with anyio.fail_after(_DEADLINE_S):
                            await notes.arrived.wait()
        assert any(_is_tools_changed(message) for message in notes.received)

    @pytest.mark.asyncio
    async def test_a_stateless_transport_offers_no_change_notifications_and_a_push_reaches_nothing(
        self, loopback_mcp: LoopbackMcp
    ) -> None:
        tokens = _Tokens()
        tokens.issue("t")
        server = _server(tokens, _noop)
        async with loopback_mcp(server, stateless=True, json_response=False) as url:
            async with httpx2.AsyncClient(headers={"Authorization": "Bearer t"}) as http:
                async with streamable_http_client(url, http_client=http) as (read, write):
                    async with ClientSession(read, write) as session:
                        initialized = await session.initialize()
                        assert initialized.capabilities.tools is not None
                        assert not initialized.capabilities.tools.list_changed
                        await session.list_tools()
                        await server.notify_tools_changed()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("stateless", [True, False], ids=["stateless", "stateful"])
    async def test_a_2026_client_listening_gets_the_change(self, loopback_mcp: LoopbackMcp, stateless: bool) -> None:
        tokens = _Tokens()
        tokens.issue("t")
        server = _server(tokens, _noop)
        events: list[Any] = []
        async with loopback_mcp(server, stateless=stateless, json_response=False) as url:
            async with httpx2.AsyncClient(headers={"Authorization": "Bearer t"}) as http:
                async with _client(url, http, modern=True) as client:
                    async with client.listen(tools_list_changed=True) as subscription:
                        await server.notify_tools_changed()
                        with anyio.fail_after(_DEADLINE_S):
                            async for event in subscription:
                                events.append(event)
                                break
        assert events == [ToolsListChanged()]

    @pytest.mark.asyncio
    async def test_a_push_after_the_session_closed_raises_nothing(self, loopback_mcp: LoopbackMcp) -> None:
        tokens = _Tokens()
        tokens.issue("t")
        server = _server(tokens, _noop)
        async with loopback_mcp(server, stateless=False, json_response=False) as url:
            async with httpx2.AsyncClient(headers={"Authorization": "Bearer t"}) as http:
                async with _client(url, http, modern=False) as session:
                    await session.list_tools()
            await server.notify_tools_changed()


async def _noop() -> str:
    return "ok"


class TestCancellationInStatefulMode:
    @pytest.mark.asyncio
    async def test_a_cancel_notification_cancels_the_call_and_the_context_says_so(
        self, loopback_mcp: LoopbackMcp
    ) -> None:
        tokens = _Tokens()
        tokens.issue("t")
        contexts: list[CallContext] = []
        started = anyio.Event()

        async def whoami(*, ctx: CallContext) -> str:
            contexts.append(ctx)
            started.set()
            await anyio.sleep(30)
            return "never"

        async with loopback_mcp(_server(tokens, whoami), stateless=False, json_response=False) as url:
            async with httpx2.AsyncClient(headers={"Authorization": "Bearer t"}) as http:
                async with _client(url, http, modern=False) as session:
                    async with anyio.create_task_group() as group:
                        group.start_soon(session.call_tool, "whoami", {})
                        with anyio.fail_after(_DEADLINE_S):
                            await started.wait()
                        request_id = contexts[0].request_id
                        assert request_id is not None
                        await session.send_notification(
                            mcp_types.CancelledNotification(
                                params=mcp_types.CancelledNotificationParams(request_id=request_id, reason="test")
                            )
                        )
                        with anyio.fail_after(_DEADLINE_S):
                            await contexts[0].wait_cancelled()
                        group.cancel_scope.cancel()
        assert contexts[0].cancelled is True

    @pytest.mark.asyncio
    async def test_the_client_closing_its_session_cancels_a_running_call(self, loopback_mcp: LoopbackMcp) -> None:
        tokens = _Tokens()
        tokens.issue("t")
        contexts: list[CallContext] = []
        started = anyio.Event()

        async def whoami(*, ctx: CallContext) -> str:
            contexts.append(ctx)
            started.set()
            await anyio.sleep(30)
            return "never"

        async with loopback_mcp(_server(tokens, whoami), stateless=False, json_response=False) as url:
            async with httpx2.AsyncClient(headers={"Authorization": "Bearer t"}) as http:
                with anyio.move_on_after(_DEADLINE_S):
                    async with _client(url, http, modern=False) as session:
                        async with anyio.create_task_group() as group:
                            group.start_soon(session.call_tool, "whoami", {})
                            with anyio.fail_after(_DEADLINE_S):
                                await started.wait()
                            group.cancel_scope.cancel()
                    # leaving the client's block deletes the session (DELETE /mcp)
                with anyio.fail_after(_DEADLINE_S):
                    await contexts[0].wait_cancelled()
        assert contexts[0].cancelled is True
