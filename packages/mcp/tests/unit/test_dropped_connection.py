"""a client that drops the connection carrying a tools/call stops that call: the handler sees the cancellation, and
its call context says so.

a real uvicorn server and a raw HTTP client, as an orchestrator speaks: the call's POST is opened, its response
stream started, and the TCP connection closed without a word -- no ``notifications/cancelled``, no DELETE. The SDK
(2.3) cancels a handler on those two and, in stateless mode, when the request's own task ends; a stateful session's
handler ran on with nobody to answer.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import httpx2
import pytest
from threetears.mcp import CallContext, Identity, McpServer, McpTool, ToolRegistry

from .conftest import LoopbackMcp

_DEADLINE_S = 5
_HANDSHAKE = "2025-11-25"
_MODERN = "2026-07-28"


class _Calls:
    """what each call of the tool saw: its context, whether it started, ended, or was cancelled."""

    def __init__(self) -> None:
        self.started: dict[str, asyncio.Event] = {}
        self.contexts: dict[str, CallContext] = {}
        self.cancelled: dict[str, bool] = {}
        self.finished: dict[str, bool] = {}

    def started_for(self, tag: str) -> asyncio.Event:
        return self.started.setdefault(tag, asyncio.Event())


def _server(calls: _Calls) -> McpServer:
    async def work(*, tag: str, seconds: float, ctx: CallContext) -> str:
        calls.contexts[tag] = ctx
        calls.started_for(tag).set()
        try:
            await asyncio.sleep(seconds)
        except asyncio.CancelledError:
            calls.cancelled[tag] = True
            raise
        calls.finished[tag] = True
        return f"done {tag}"

    registry = ToolRegistry()
    registry.register(
        McpTool(
            name="work",
            description="sleeps for a while",
            input_schema={
                "type": "object",
                "properties": {"tag": {"type": "string"}, "seconds": {"type": "number"}},
                "required": ["tag", "seconds"],
            },
            required_permission="t.work",
            handler=work,
        )
    )
    provider = MagicMock()
    provider.identify = AsyncMock(return_value=Identity(principal_type="user", principal_id=uuid4()))
    authorizer = MagicMock()
    authorizer.allows = AsyncMock(return_value=True)
    authorizer.start = AsyncMock()
    authorizer.stop = AsyncMock()
    return McpServer(name="drops", identity_provider=provider, authorizer=authorizer, registry=registry)


def _headers(session: str | None = None, version: str = _HANDSHAKE) -> dict[str, str]:
    headers = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}
    if session is not None:
        headers["Mcp-Session-Id"] = session
    if session is not None or version == _MODERN:
        headers["Mcp-Protocol-Version"] = version
    if version == _MODERN:
        # the 2026-07-28 wire names a request's method and tool in headers, which must match its body
        headers["Mcp-Method"] = "tools/call"
        headers["Mcp-Name"] = "work"
    return headers


async def _open_session(http: httpx2.AsyncClient, url: str) -> str | None:
    """initialize; the session id a stateful transport answers, None for a stateless one."""
    response = await http.post(
        url,
        headers=_headers(),
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": _HANDSHAKE, "capabilities": {}, "clientInfo": {"name": "t", "version": "0"}},
        },
    )
    session = response.headers.get("mcp-session-id")
    await http.post(
        url,
        headers=_headers(session),
        json={"jsonrpc": "2.0", "method": "notifications/initialized"},
    )
    return session


def _call(request_id: int, tag: str, seconds: float, *, modern: bool) -> dict[str, Any]:
    params: dict[str, Any] = {"name": "work", "arguments": {"tag": tag, "seconds": seconds}}
    if modern:
        params["_meta"] = {
            "io.modelcontextprotocol/protocolVersion": _MODERN,
            "io.modelcontextprotocol/clientInfo": {"name": "t", "version": "0"},
            "io.modelcontextprotocol/clientCapabilities": {},
        }
    return {"jsonrpc": "2.0", "id": request_id, "method": "tools/call", "params": params}


async def _drop_mid_call(url: str, calls: _Calls, tag: str, *, session: str | None, modern: bool) -> None:
    """open the call's POST, wait until its handler runs, then close the connection without a word."""
    version = _MODERN if modern else _HANDSHAKE
    async with httpx2.AsyncClient(timeout=30) as http:
        async with http.stream("POST", url, headers=_headers(session, version), json=_call(7, tag, 30, modern=modern)):
            async with asyncio.timeout(_DEADLINE_S):
                await calls.started_for(tag).wait()
    # leaving both blocks closed the response and its connection


_MODES = [
    pytest.param(False, False, id="stateful-2025-11-25"),
    pytest.param(True, False, id="stateless-2025-11-25"),
    pytest.param(True, True, id="stateless-2026-07-28"),
    pytest.param(False, True, id="stateful-2026-07-28"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(("stateless", "modern"), _MODES)
async def test_a_dropped_connection_cancels_its_call_and_the_context_says_so(
    loopback_mcp: LoopbackMcp, stateless: bool, modern: bool
) -> None:
    calls = _Calls()
    async with loopback_mcp(_server(calls), stateless=stateless, json_response=False) as url:
        session = None
        if not modern:
            async with httpx2.AsyncClient(timeout=10) as http:
                session = await _open_session(http, url)
        await _drop_mid_call(url, calls, "dropped", session=session, modern=modern)
        async with asyncio.timeout(_DEADLINE_S):
            await calls.contexts["dropped"].wait_cancelled()
    assert calls.cancelled.get("dropped") is True
    assert calls.contexts["dropped"].cancelled is True


@pytest.mark.asyncio
@pytest.mark.parametrize(("stateless", "modern"), _MODES)
async def test_a_call_that_completes_is_not_cancelled(loopback_mcp: LoopbackMcp, stateless: bool, modern: bool) -> None:
    calls = _Calls()
    async with loopback_mcp(_server(calls), stateless=stateless, json_response=False) as url:
        async with httpx2.AsyncClient(timeout=10) as http:
            session = None if modern else await _open_session(http, url)
            version = _MODERN if modern else _HANDSHAKE
            response = await http.post(
                url, headers=_headers(session, version), json=_call(8, "kept", 0.05, modern=modern)
            )
        assert "done kept" in response.text
        await asyncio.sleep(0.2)
    assert calls.finished.get("kept") is True
    assert "kept" not in calls.cancelled
    assert calls.contexts["kept"].cancelled is False


@pytest.mark.asyncio
async def test_a_drop_cancels_only_the_call_its_request_carried(loopback_mcp: LoopbackMcp) -> None:
    """two calls on one stateful session, each on its own POST: dropping one leaves the other running to its end."""
    calls = _Calls()
    async with loopback_mcp(_server(calls), stateless=False, json_response=False) as url:
        async with httpx2.AsyncClient(timeout=30) as http:
            session = await _open_session(http, url)
            kept = asyncio.create_task(
                http.post(url, headers=_headers(session), json=_call(9, "kept", 1.0, modern=False))
            )
            async with asyncio.timeout(_DEADLINE_S):
                await calls.started_for("kept").wait()
            await _drop_mid_call(url, calls, "dropped", session=session, modern=False)
            async with asyncio.timeout(_DEADLINE_S):
                await calls.contexts["dropped"].wait_cancelled()
            response = await kept
        assert "done kept" in response.text
    assert calls.cancelled.get("dropped") is True
    assert "kept" not in calls.cancelled
    assert calls.finished.get("kept") is True
