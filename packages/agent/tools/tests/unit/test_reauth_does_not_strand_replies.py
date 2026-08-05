"""A connection that owes a reply must not be recycled underneath it.

NATS scopes ``allow_responses`` to the CONNECTION that received a request: the
server remembers *this* connection may answer *that* message. The tool pod also
runs a proactive re-auth loop that reconnects before its user JWT expires --
every ``ttl - leeway - buffer`` seconds, which at the platform default TTL of
150s is every 60 seconds.

Nothing related those two numbers to the tool timeout, which is 1200s for a scan
tool. So any call taking longer than about a minute ran to completion and then
lost the right to deliver its answer, discovering this only at publish:

    scanner finished {"tool": "testssl", "exit_code": 0,
                      "duration_seconds": 91.964, "timed_out": false,
                      "stdout_bytes": 67902}
    NATS error: permissions violation for publish to "_inbox...."

The scan worked. 68KB of results existed. The connection that was allowed to
send them had been rebuilt 56 seconds earlier, so nobody ever saw them -- which
reads as the tool being broken rather than the connection being recycled.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4

import pytest

from threetears.agent.tools import nats_reauth
from threetears.agent.tools.base_tool import MCPToolDefinition, TearsTool, ToolResult
from threetears.agent.tools.server import ToolServer
from threetears.nats import IncomingMessage, set_default_namespace

from unit.tools._pod_auth import StubReplayGuard as _PodReplayGuard
from unit.tools._pod_auth import jwks_provider as _pod_jwks_provider
from unit.tools._pod_auth import signed_call_payload as _signed_call_payload

pytestmark = pytest.mark.unit

_POD = "pod-under-test"


class _BlockingTool(TearsTool):
    """a tool that does not finish until the test lets it, standing in for a long scan.

    One gate PER CALL, not one shared gate: the concurrency test needs to release one dispatch while
    another is still owed, and a shared gate would release both -- which would make that test pass
    for the wrong reason (or, as it first did, fail for one).
    """

    def __init__(self) -> None:
        self.gates: list[asyncio.Event] = []

    async def execute(self, **kwargs: Any) -> ToolResult:
        del kwargs
        gate = asyncio.Event()
        self.gates.append(gate)
        await gate.wait()
        return ToolResult(success=True, content="ok")

    def mcp_schema(self) -> MCPToolDefinition:
        return MCPToolDefinition(
            name="test.blocking",
            version="1.0",
            description="blocks until released",
            input_schema={"type": "object", "properties": {}},
        )

    def mcp_name(self) -> str:
        return "test.blocking"

    def mcp_version(self) -> str:
        return "1.0"


class _SilentNats:
    """swallows every publish surface; these tests observe the obligation count, not the wire."""

    async def publish(self, **kwargs: Any) -> None:
        del kwargs

    async def jetstream_publish(self, **kwargs: Any) -> None:
        del kwargs

    async def publish_reply(self, **kwargs: Any) -> None:
        del kwargs


def _idle_server() -> tuple[ToolServer, _BlockingTool]:
    """a real :class:`ToolServer` owing nothing, plus the tool the tests use to make it owe.

    A real server rather than a stand-in: the obligation bookkeeping now spans ``handle_call``, the
    acknowledgement path and the settle helper, so a hand-rolled double could satisfy every
    assertion here while the production dispatch did something else entirely.
    """
    set_default_namespace("3tears")
    tool = _BlockingTool()
    server = ToolServer(
        namespace="3tears",
        nats_client=_SilentNats(),  # type: ignore[arg-type]
        pod_id=_POD,
        namespace_collection=None,
        jwks_provider=_pod_jwks_provider,
        assertion_replay_guard=_PodReplayGuard(),
    )
    server.register(tool)
    return server, tool


@contextlib.asynccontextmanager
async def _owed_reply(server: ToolServer, tool: _BlockingTool) -> AsyncIterator[None]:
    """hold ONE real dispatch open inside the block, so the pod genuinely owes an inbox reply."""
    payload = _signed_call_payload(
        pod_id=_POD,
        tool_name="test.blocking",
        conversation_id=uuid4(),
        user_id=uuid4(),
    )
    msg = IncomingMessage(
        data=json.dumps(payload).encode("utf-8"),
        reply_subject="_INBOX_registry_reg-1.abc",
        subject=f"3tears.tools.internal.{_POD}",
    )
    dispatch = asyncio.create_task(server.handle_call(msg))
    for _ in range(200):
        await asyncio.sleep(0.005)
        if tool.gates:
            break
    gate = tool.gates.pop()
    try:
        yield
    finally:
        gate.set()
        await asyncio.wait_for(dispatch, timeout=1.0)


class TestTheConnectionSurvivesAnUnansweredCall:
    async def test_reauth_waits_while_a_call_is_in_flight(self) -> None:
        """THE PRODUCTION BUG. The reconnect must not happen while the pod still
        owes an answer."""
        server, tool = _idle_server()
        async with _owed_reply(server, tool):
            assert server.sync_replies_in_flight == 1
            assert await server.await_sync_replies(timeout=0.05) is False

        assert await server.await_sync_replies(timeout=0.5) is True

    async def test_reauth_is_immediate_when_nothing_is_owed(self) -> None:
        """Non-vacuous: the common case must not pay for the guard. A pod with
        no work waits for nothing."""
        server, _tool = _idle_server()

        assert server.sync_replies_in_flight == 0
        assert await asyncio.wait_for(server.await_sync_replies(timeout=5.0), timeout=0.2) is True

    async def test_it_waits_for_every_outstanding_call_not_just_one(self) -> None:
        """Concurrent dispatches each own a reply; draining one proves nothing
        about the rest."""
        server, tool = _idle_server()
        async with _owed_reply(server, tool):
            assert server.sync_replies_in_flight == 1
            async with _owed_reply(server, tool):
                assert server.sync_replies_in_flight == 2
            assert await server.await_sync_replies(timeout=0.05) is False

        assert await server.await_sync_replies(timeout=0.5) is True


class TestTheWaitIsBounded:
    """Deferring forever would trade a lost reply for a dead connection.

    The JWT expires whether or not a tool is still running. Past the point where
    the reconnect must begin to beat expiry, waiting longer does not save the
    reply -- it loses the connection as well. So the wait is bounded, and the
    reply about to be discarded is named rather than dropped silently.
    """

    async def test_a_call_that_outlasts_the_grace_does_not_block_forever(self) -> None:
        server, tool = _idle_server()
        async with _owed_reply(server, tool):
            # The real grace is REAUTH_BUFFER_SECONDS; patched down so the test does
            # not sit for 30 seconds proving a timeout fires.
            original = nats_reauth.REAUTH_BUFFER_SECONDS
            try:
                nats_reauth.REAUTH_BUFFER_SECONDS = 0.05
                await asyncio.wait_for(server.drain_before_reauth(150), timeout=2.0)
            finally:
                nats_reauth.REAUTH_BUFFER_SECONDS = original


class TestTheTwoBudgetsAreRelated:
    """The mismatch that caused this must be visible before it costs a result.

    A tool timeout longer than the connection's usable life is not a runtime
    condition to detect once it has already discarded an answer -- it is a
    configuration fact knowable at startup.
    """

    def test_the_usable_connection_life_is_shorter_than_the_jwt_ttl(self) -> None:
        """The reconnect fires early by design, so the window a call can survive
        in is the TTL minus that margin -- not the TTL."""
        ttl = 150
        usable = ttl - nats_reauth.REAUTH_LEEWAY_SECONDS

        assert usable < ttl
        assert nats_reauth.seconds_until_reauth(ttl) < usable

    def test_the_platform_default_cannot_carry_a_long_tool_call(self) -> None:
        """Pins the incoherence this fix exists for: at the default TTL, a scan
        tool's 1200s budget is an order of magnitude past what the connection
        can survive. If someone raises the default, this test is where they find
        out the relationship is deliberate."""
        default_ttl = 150
        scan_tool_timeout = 1200

        assert scan_tool_timeout > default_ttl - nats_reauth.REAUTH_LEEWAY_SECONDS


class TestTheSynchronousBudgetFitsInsideTheDrainGrace:
    """The drain rescues a short call; durable delivery carries the rest.

    Draining is a mitigation, not the fix: it can only hold the connection open for the slack the
    re-auth schedule leaves, so a call longer than that grace still completes into a refused publish.
    The remedy is that such calls never take the reply-inbox path at all -- but that only holds if the
    threshold deciding which calls are "short" is no larger than the window the responder is actually
    willing to wait.

    The two numbers live in different packages and nothing else relates them, which is exactly how the
    original mismatch (a 60-second connection carrying a 1200-second tool) got in.
    """

    def test_a_call_chosen_for_the_sync_path_fits_in_the_grace(self) -> None:
        from threetears.nats import SYNC_REPLY_BUDGET_SECONDS

        assert SYNC_REPLY_BUDGET_SECONDS <= nats_reauth.REAUTH_BUFFER_SECONDS, (
            "a call the caller chose to answer synchronously can outlast the drain grace, so the "
            "responder will reconnect out from under it and refuse the reply"
        )

    def test_the_scan_tool_that_started_this_takes_the_durable_path(self) -> None:
        """Non-vacuous: the concrete call that lost 68KB of results is on the other path now."""
        from threetears.nats import requires_async_result

        assert requires_async_result(1200.0) is True
        assert requires_async_result(5.0) is False
