"""A connection that owes a reply must not be closed underneath it.

NATS scopes ``allow_responses`` to the CONNECTION that received a request: the
server remembers *this* connection may answer *that* message. The tool pod also
renews its credential before its user JWT expires.

That renewal used to be a reconnect of the one connection, and nothing related
its cadence to the tool timeout, which is 1200s for a scan tool. So any call
taking longer than the cadence ran to completion and then lost the right to
deliver its answer, discovering this only at publish:

    scanner finished {"tool": "testssl", "exit_code": 0,
                      "duration_seconds": 91.964, "timed_out": false,
                      "stdout_bytes": 67902}
    NATS error: permissions violation for publish to "_inbox...."

The scan worked. 68KB of results existed. The connection that was allowed to
send them had been rebuilt 56 seconds earlier, so nobody ever saw them -- which
reads as the tool being broken rather than the connection being recycled.

The renewal is now make-before-break: the connection that received a call stays
open for the synchronous reply budget after its successor takes over, and the
reply leaves on it (proven live in
``packages/nats/tests/integration/test_credential_renewal_live.py``). A call
longer than that budget answers on the pod's durable result subject instead.
These tests pin the pod's own half: it knows exactly what it owes, and the two
budgets stay related.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4

from threetears.agent.tools.base_tool import MCPToolDefinition, TearsTool, ToolResult
from threetears.agent.tools.server import ToolServer
from threetears.nats import (
    PLATFORM_DEFAULT_NATS_USER_JWT_TTL_SECONDS,
    SYNC_REPLY_BUDGET_SECONDS,
    IncomingMessage,
    seconds_until_retirement,
    set_default_namespace,
    unsafe_renewal_reason,
)

from threetears.core.testing.replay_guard import FakeReplayGuard
from packages.agent.tools.tests.unit.tools.pod_auth import jwks_provider as _pod_jwks_provider
from packages.agent.tools.tests.unit.tools.pod_auth import signed_call_payload as _signed_call_payload

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
        jwks_provider=_pod_jwks_provider,
        assertion_replay_guard=FakeReplayGuard(),
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


class TestThePodKnowsWhatItOwes:
    """the count a shutdown -- or any caller about to close the connection -- reads."""

    async def test_a_call_in_flight_is_owed(self) -> None:
        """THE PRODUCTION BUG'S precondition: while the call runs, the pod owes its answer."""
        server, tool = _idle_server()
        async with _owed_reply(server, tool):
            assert server.sync_replies_in_flight == 1
            assert await server.await_sync_replies(timeout=0.05) is False

        assert await server.await_sync_replies(timeout=0.5) is True

    async def test_nothing_is_owed_when_nothing_runs(self) -> None:
        """Non-vacuous: a pod with no work waits for nothing."""
        server, _tool = _idle_server()

        assert server.sync_replies_in_flight == 0
        assert await asyncio.wait_for(server.await_sync_replies(timeout=5.0), timeout=0.2) is True

    async def test_every_outstanding_call_is_counted_not_just_one(self) -> None:
        """Concurrent dispatches each own a reply; settling one proves nothing about the rest."""
        server, tool = _idle_server()
        async with _owed_reply(server, tool):
            assert server.sync_replies_in_flight == 1
            async with _owed_reply(server, tool):
                assert server.sync_replies_in_flight == 2
            assert await server.await_sync_replies(timeout=0.05) is False

        assert await server.await_sync_replies(timeout=0.5) is True


class TestTheTwoBudgetsAreRelated:
    """The mismatch that caused this must be visible before it costs a result.

    A tool timeout longer than a renewal can hold the replaced connection open is not a runtime
    condition to detect once it has already discarded an answer -- it is a configuration fact
    knowable at startup. The two numbers live in different packages and nothing else relates them,
    which is exactly how the original mismatch (a 60-second connection carrying a 1200-second tool)
    got in.
    """

    def test_a_call_chosen_for_the_sync_path_rides_a_renewal_at_the_default_ttl(self) -> None:
        """the pod declares the synchronous budget as its longest request, and the default TTL
        lets a renewal hold the replaced connection that long."""
        ttl = PLATFORM_DEFAULT_NATS_USER_JWT_TTL_SECONDS

        assert unsafe_renewal_reason(ttl, longest_request_seconds=SYNC_REPLY_BUDGET_SECONDS) is None

    def test_the_hold_covers_the_whole_synchronous_budget(self) -> None:
        """the replaced connection stays open for every reply the budget admits, on schedule."""
        from threetears.nats import seconds_until_reauth

        ttl = PLATFORM_DEFAULT_NATS_USER_JWT_TTL_SECONDS
        swap = seconds_until_reauth(ttl, longest_request_seconds=SYNC_REPLY_BUDGET_SECONDS)
        hold = seconds_until_retirement(
            ttl, connection_age_seconds=swap, longest_request_seconds=SYNC_REPLY_BUDGET_SECONDS
        )

        assert hold >= SYNC_REPLY_BUDGET_SECONDS

    def test_the_backstop_default_renews_make_before_break_even_under_a_long_tool_call(self) -> None:
        """The platform's default TTL is the day-long backstop (owner ruling Q17): access is taken
        away by a kick, and the TTL only bounds a kick that was lost. At that TTL a renewal happens
        about once a day and is still the make-before-break handover, which can hold the replaced
        connection open for a scan tool's whole 1200s budget. Asked of the renewal's own judge and
        schedule with the real default, so shortening the default is where someone finds out the
        relationship is deliberate. What sends a long call to the durable path is the synchronous
        budget, not the TTL (``test_the_scan_tool_that_started_this_takes_the_durable_path``)."""
        from threetears.nats import REAUTH_MARGIN_SECONDS, seconds_until_reauth

        scan_tool_timeout = 1200.0
        ttl = PLATFORM_DEFAULT_NATS_USER_JWT_TTL_SECONDS

        assert unsafe_renewal_reason(ttl, longest_request_seconds=scan_tool_timeout) is None
        swap = seconds_until_reauth(ttl, longest_request_seconds=scan_tool_timeout)
        assert swap == ttl - REAUTH_MARGIN_SECONDS - scan_tool_timeout, "one renewal per TTL, late in it"
        hold = seconds_until_retirement(ttl, connection_age_seconds=swap, longest_request_seconds=scan_tool_timeout)
        assert hold == scan_tool_timeout, "the replaced connection is held for the whole long call"

    def test_a_short_configured_ttl_still_cannot_carry_a_long_tool_call(self) -> None:
        """The incoherence the durable path was built for is still named when a deployment shortens
        the TTL: the renewal's judge refuses a TTL that cannot hold a scan tool's budget."""
        scan_tool_timeout = 1200.0

        assert unsafe_renewal_reason(300, longest_request_seconds=scan_tool_timeout) is not None

    def test_the_scan_tool_that_started_this_takes_the_durable_path(self) -> None:
        """Non-vacuous: the concrete call that lost 68KB of results is on the other path now."""
        from threetears.nats import requires_async_result

        assert requires_async_result(1200.0) is True
        assert requires_async_result(5.0) is False
