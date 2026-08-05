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

import pytest

from threetears.agent.tools import nats_reauth

pytestmark = pytest.mark.unit


class _Server:
    """the re-auth collaborators of :class:`ToolServer`, and nothing else.

    The real class needs a NATS connection, a registry and a tool manifest to
    construct. What is under test is one rule -- do not reconnect while a reply
    is owed -- so the two methods that implement it are exercised against the
    same attributes they use in production.
    """

    def __init__(self) -> None:
        self._pod_id = "pod-under-test"
        self._calls_in_flight = 0
        self._calls_idle = asyncio.Event()
        self._calls_idle.set()
        self.reconnects = 0

    async def _reauth_nats_once(self) -> None:
        self.reconnects += 1

    def begin_call(self) -> None:
        self._calls_in_flight += 1
        self._calls_idle.clear()

    def end_call(self) -> None:
        self._calls_in_flight -= 1
        if self._calls_in_flight <= 0:
            self._calls_in_flight = 0
            self._calls_idle.set()


def _bind(server: _Server):
    """bind the real ``_drain_before_reauth`` to the stand-in."""
    from threetears.agent.tools.server import ToolServer

    return ToolServer._drain_before_reauth.__get__(server, _Server)


class TestTheConnectionSurvivesAnUnansweredCall:
    async def test_reauth_waits_while_a_call_is_in_flight(self) -> None:
        """THE PRODUCTION BUG. The reconnect must not happen while the pod still
        owes an answer."""
        server = _Server()
        drain = _bind(server)
        server.begin_call()

        waiter = asyncio.create_task(drain(150))
        await asyncio.sleep(0.05)
        assert not waiter.done(), "re-auth proceeded while a reply was still owed"

        server.end_call()
        await asyncio.wait_for(waiter, timeout=1.0)

    async def test_reauth_is_immediate_when_nothing_is_owed(self) -> None:
        """Non-vacuous: the common case must not pay for the guard. A pod with
        no work waits for nothing."""
        server = _Server()
        drain = _bind(server)

        await asyncio.wait_for(drain(150), timeout=0.2)

    async def test_it_waits_for_every_outstanding_call_not_just_one(self) -> None:
        """Concurrent dispatches each own a reply; draining one proves nothing
        about the rest."""
        server = _Server()
        drain = _bind(server)
        server.begin_call()
        server.begin_call()

        waiter = asyncio.create_task(drain(150))
        server.end_call()
        await asyncio.sleep(0.05)
        assert not waiter.done(), "re-auth proceeded with a second call still owed"

        server.end_call()
        await asyncio.wait_for(waiter, timeout=1.0)


class TestTheWaitIsBounded:
    """Deferring forever would trade a lost reply for a dead connection.

    The JWT expires whether or not a tool is still running. Past the point where
    the reconnect must begin to beat expiry, waiting longer does not save the
    reply -- it loses the connection as well. So the wait is bounded, and the
    reply about to be discarded is named rather than dropped silently.
    """

    async def test_a_call_that_outlasts_the_grace_does_not_block_forever(self) -> None:
        server = _Server()
        drain = _bind(server)
        server.begin_call()  # never ends

        # The real grace is REAUTH_BUFFER_SECONDS; patched down so the test does
        # not sit for 30 seconds proving a timeout fires.
        original = nats_reauth.REAUTH_BUFFER_SECONDS
        try:
            nats_reauth.REAUTH_BUFFER_SECONDS = 0.05
            await asyncio.wait_for(drain(150), timeout=2.0)
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
