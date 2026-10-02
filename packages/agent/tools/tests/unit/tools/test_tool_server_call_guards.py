"""per-call concurrency cap + hard execution timeout on :class:`ToolServer`.

Two pod-level guards protect a tool pod from a burst of heavy tools (several
scanners at once, a runaway that ignores its own budget):

* ``max_concurrent_calls`` bounds how many ``tool.run`` bodies execute at once.
* ``max_call_seconds`` is a HARD ceiling that force-ends a call running past it
  and invokes the call's registered cleanup hooks -- so a tool that spawned a
  subprocess reaps it rather than orphaning it.

These exercise both guards through the pod's front door, :meth:`ToolServer.handle_call`,
with a fully authenticated call to a fake tool, and read the outcome off the reply the
pod answers with -- plus the two properties that keep the timeout honest: a tool's OWN
``TimeoutError`` is NOT reported as the server's ceiling, and hooks fire only when the
ceiling actually trips.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from threetears.agent.tools.base_tool import MCPToolDefinition, TearsTool, ToolResult
from threetears.agent.tools.call_scope import register_call_cleanup
from threetears.agent.tools.server import CallResponse, ToolServer
from threetears.core.testing.replay_guard import FakeReplayGuard
from threetears.nats import IncomingMessage

from packages.agent.tools.tests.unit.tools.pod_auth import (
    RecordingNatsClient,
    jwks_provider,
    recording_tool_server,
    signed_call_payload,
)

_POD_ID = "guard-pod"
_HARD_LIMIT = "tool exceeded the pod hard execution limit"


class _FakeTool(TearsTool):
    """a tool whose ``execute`` body is supplied per test."""

    def __init__(self, body: Any) -> None:
        self._body = body

    async def execute(self, **kwargs: Any) -> ToolResult:
        return await self._body()

    def mcp_schema(self) -> MCPToolDefinition:
        return MCPToolDefinition(name="test.fake", version="1.0.0", description="t", input_schema={})

    def mcp_name(self) -> str:
        return "test.fake"

    def mcp_version(self) -> str:
        return "1.0.0"


def _serving(body: Any, **guards: Any) -> tuple[ToolServer, RecordingNatsClient]:
    """a tool server holding one fake tool, configured with the given guards and driven by hand.

    :param body: the coroutine function the fake tool's ``execute`` awaits
    :ptype body: Any
    :param guards: ``max_concurrent_calls`` / ``max_call_seconds``
    :ptype guards: Any
    :return: the server and the client it answers on
    :rtype: tuple[ToolServer, RecordingNatsClient]
    """
    server, rec = recording_tool_server(
        pod_id=_POD_ID,
        jwks_provider=jwks_provider,
        assertion_replay_guard=FakeReplayGuard(),
        **guards,
    )
    server.register(_FakeTool(body))
    return server, rec


async def _call(server: ToolServer) -> None:
    """deliver one authenticated call for the fake tool to the pod's call handler.

    :param server: the pod
    :ptype server: ToolServer
    :return: nothing
    :rtype: None
    """
    payload = signed_call_payload(pod_id=_POD_ID, tool_name="test.fake", tool_version="1.0.0")
    await server.handle_call(
        IncomingMessage(
            data=json.dumps(payload).encode("utf-8"),
            reply_subject="_INBOX.guards",
            subject=f"3tears.tools.internal.{_POD_ID}",
        )
    )


def _answers(rec: RecordingNatsClient) -> list[CallResponse]:
    """every reply the pod answered with, as call responses.

    :param rec: the client the pod answers on
    :ptype rec: RecordingNatsClient
    :return: the replies
    :rtype: list[CallResponse]
    """
    replies = [reply for _subject, reply in rec.replies]
    assert all(isinstance(reply, CallResponse) for reply in replies)
    return replies


async def test_concurrency_cap_bounds_simultaneous_runs() -> None:
    """with the cap at 2, only 2 of 5 delivered calls run at once; the rest queue."""
    active = 0
    peak = 0
    started = 0
    gate = asyncio.Event()

    async def body() -> ToolResult:
        nonlocal active, peak, started
        started += 1
        active += 1
        peak = max(peak, active)
        try:
            await gate.wait()
        finally:
            active -= 1
        return ToolResult(success=True, content="ok")

    server, rec = _serving(body, max_concurrent_calls=2)
    tasks = [asyncio.create_task(_call(server)) for _ in range(5)]
    # let every call clear verification and reach the semaphore; only the cap may be inside `body`.
    for _ in range(200):
        if started >= 2:
            break
        await asyncio.sleep(0.01)
    await asyncio.sleep(0.05)
    assert active == 2, "more than the cap ran at once"
    assert started == 2, "queued calls entered the tool body before a slot freed"

    gate.set()
    await asyncio.gather(*tasks)
    answers = _answers(rec)
    assert len(answers) == 5
    assert all(answer.success for answer in answers), [answer.error for answer in answers]
    assert peak == 2


async def test_hard_timeout_fails_the_call_and_runs_cleanup_hooks() -> None:
    """a call past the ceiling is answered as force-ended, after its hooks ran."""
    reaped: list[str] = []

    async def body() -> ToolResult:
        register_call_cleanup(lambda: reaped.append("killed"))
        await asyncio.sleep(10)  # far past the ceiling
        return ToolResult(success=True, content="never")

    server, rec = _serving(body, max_call_seconds=0.05)
    await _call(server)
    (answer,) = _answers(rec)
    assert answer.success is False
    assert answer.error is not None and _HARD_LIMIT in answer.error
    assert reaped == ["killed"], "cleanup hook did not run on hard timeout"


async def test_tool_own_timeout_error_is_not_reclassified() -> None:
    """a tool raising its OWN TimeoutError inside the ceiling stays a plain tool failure."""
    reaped: list[str] = []

    async def body() -> ToolResult:
        register_call_cleanup(lambda: reaped.append("killed"))
        raise TimeoutError("the tool's own per-scan budget")

    server, rec = _serving(body, max_call_seconds=5.0)
    await _call(server)
    (answer,) = _answers(rec)
    assert answer.success is False
    assert answer.error is not None
    assert "the tool's own per-scan budget" in answer.error
    assert _HARD_LIMIT not in answer.error
    assert reaped == [], "cleanup ran for a tool-owned timeout that was not the server's"


async def test_hooks_do_not_run_on_success() -> None:
    """a call that returns within the ceiling never invokes its cleanup hooks."""
    reaped: list[str] = []

    async def body() -> ToolResult:
        register_call_cleanup(lambda: reaped.append("killed"))
        return ToolResult(success=True, content="ok")

    server, rec = _serving(body, max_call_seconds=5.0)
    await _call(server)
    (answer,) = _answers(rec)
    assert answer.success is True, answer.error
    assert reaped == []


async def test_unguarded_server_is_pass_through() -> None:
    """with neither guard set the pod just runs the tool (prior behaviour)."""

    async def body() -> ToolResult:
        return ToolResult(success=True, content="ok")

    server, rec = _serving(body)
    await _call(server)
    (answer,) = _answers(rec)
    assert answer.success is True, answer.error
    assert answer.content == "ok"


def test_rejects_nonpositive_guard_values() -> None:
    """the constructor refuses guard values that could not bound anything."""
    with pytest.raises(ValueError, match="max_concurrent_calls"):
        ToolServer(nats_url="nats://stub", max_concurrent_calls=0)
    with pytest.raises(ValueError, match="max_call_seconds"):
        ToolServer(nats_url="nats://stub", max_call_seconds=0)


def test_register_call_cleanup_outside_scope_raises() -> None:
    """registering a hook with no active call scope is a programming error."""
    with pytest.raises(RuntimeError, match="outside a ToolServer call scope"):
        register_call_cleanup(lambda: None)
