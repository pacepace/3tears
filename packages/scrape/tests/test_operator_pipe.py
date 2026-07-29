"""The pod side of a display stream: what the claim entitles, and what it ends.

Every test here runs the real ``serve_pipe``, the real ``attach_pipe`` and the real
``relay_stream`` over an in-memory bus, against a REAL loopback TCP server standing in for
``x11vnc``. Nothing on the path between the claim and the socket is stubbed, because the two
things worth proving are both compositions: that the bridge reaches a display at all, and that
losing the claim stops it reaching one.

The socket is real rather than a substituted transport for the same reason. A tool pod always
makes that TCP hop -- it is the one part of the path that is identical whether the operator's
WebSocket is terminated in the pod beside the display or on a hub with no route into it -- so a
test that replaced it would leave the pod's actual behaviour uncovered.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import timedelta

import pytest
from _bus_shims import FakeBus
from threetears.nats import NoOwnerError, PipeEndpoint, PipeRemoteError, Subjects, attach_pipe, open_pipe
from threetears.scrape.operator import DisplayEndpoint
from threetears.scrape.operator_pipe import serve_display
from threetears.scrape.operator_session import SessionClaim

_SESSION = "session-1"

#: The serving tool's registered namespace name, which is what the subject family is derived
#: from. A per-zone name, because that is the deployment this exists for.
_TOOL = "tools.scrape-zone_alpha.1-0-0"

_POD = "pod-a3f9c2"

#: What a real ``x11vnc`` sends the instant a client connects, and therefore the cheapest proof
#: that bytes came from the far side of a socket rather than from anything in this test.
_RFB_BANNER = b"RFB 003.008\n"


class _LoopbackDisplay:
    """A real TCP server on loopback, standing in for the ``x11vnc`` the sidecar starts.

    Not a fake of anything this package defines: it is a socket server, and what a display is
    from this side is a socket that greets whoever connects and reads what they send. It records
    the connections it accepted, which is the evidence a refused attach never reached a display,
    and when the last one closed, which is the evidence a finished stream let go of it.
    """

    def __init__(self, greeting: bytes = _RFB_BANNER) -> None:
        self._greeting = greeting
        self._server: asyncio.Server | None = None
        self._writers: list[asyncio.StreamWriter] = []
        #: Where the display is listening, once started.
        self.address: tuple[str, int] = ("127.0.0.1", 0)
        #: How many times the bridge opened a connection. Zero is what a refusal has to show.
        self.connections = 0
        self.received = bytearray()
        #: Set whenever the operator's bytes land, so a test can wait rather than sleep.
        self.wrote_to_us = asyncio.Event()
        #: Set when a connection to the display ends, from the display's own side.
        self.disconnected = asyncio.Event()

    async def start(self) -> None:
        """Listen on an ephemeral loopback port and record where."""
        self._server = await asyncio.start_server(self._serve, "127.0.0.1", 0)
        host, port, *_ = self._server.sockets[0].getsockname()
        self.address = (str(host), int(port))

    async def _serve(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """Greet whoever connected, then read them until they go away."""
        self.connections += 1
        self._writers.append(writer)
        writer.write(self._greeting)
        await writer.drain()
        try:
            while data := await reader.read(4096):
                self.received += data
                self.wrote_to_us.set()
        finally:
            writer.close()
            self.disconnected.set()

    async def stop(self) -> None:
        """Stop listening and drop whatever is still connected.

        The connections are closed from here rather than only waited on, because a bridge that
        FAILED to release one would otherwise hang this teardown forever: the server's own
        ``wait_closed`` waits for its handlers, and a handler reading a socket nobody closed never
        returns. A test that proves a release must fail on its own assertion, not by wedging the
        suite that runs it.
        """
        for writer in self._writers:
            writer.close()
        self._writers.clear()
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None


@pytest.fixture
async def display() -> AsyncIterator[_LoopbackDisplay]:
    """A running loopback display, torn down whatever the test did to it."""
    server = _LoopbackDisplay()
    await server.start()
    try:
        yield server
    finally:
        await server.stop()


def _where(server: _LoopbackDisplay) -> DisplayEndpoint:
    """A ``DisplayEndpoint`` resolving every session to *server*."""

    async def _resolve(session_id: str) -> tuple[str, int]:
        del session_id
        return server.address

    return _resolve


async def _attach(bus: FakeBus, *, family: str | None = None) -> PipeEndpoint:
    """Attach as the process holding the operator's socket does."""
    return await attach_pipe(
        bus,  # type: ignore[arg-type]
        _SESSION,
        family=Subjects.hitl_pipe_family(_TOOL) if family is None else family,
        timeout=timedelta(seconds=5),
    )


class TestTheDisplayReachesAnOperatorThroughThePipe:
    """The happy path, first: without it every refusal below would pass against a dead bridge."""

    async def test_the_displays_bytes_reach_the_caller(self, display: _LoopbackDisplay) -> None:
        """A real socket's opening bytes, carried out of the pod over the bus, unaltered."""
        bus = FakeBus()
        claim = SessionClaim(session_id=_SESSION)

        async with serve_display(bus, claim, tool=_TOOL, pod_id=_POD, display=_where(display)):  # type: ignore[arg-type]
            endpoint = await _attach(bus)
            async with open_pipe(bus, endpoint) as stream:  # type: ignore[arg-type]
                first = await asyncio.wait_for(stream.receive(), timeout=5)

        assert first == _RFB_BANNER, "the display's opening bytes did not arrive intact"
        assert display.connections == 1, "the bridge did not open exactly one connection per stream"

    async def test_the_callers_bytes_reach_the_display(self, display: _LoopbackDisplay) -> None:
        """The thin direction: an operator's keystrokes and pointer events go the other way."""
        bus = FakeBus()
        claim = SessionClaim(session_id=_SESSION)

        async with serve_display(bus, claim, tool=_TOOL, pod_id=_POD, display=_where(display)):  # type: ignore[arg-type]
            endpoint = await _attach(bus)
            async with open_pipe(bus, endpoint) as stream:  # type: ignore[arg-type]
                await asyncio.wait_for(stream.receive(), timeout=5)
                await stream.send(_RFB_BANNER)
                await asyncio.wait_for(display.wrote_to_us.wait(), timeout=5)

        assert bytes(display.received) == _RFB_BANNER, "the operator's bytes reached the display altered"

    async def test_the_stream_rides_the_serving_tools_own_family(self, display: _LoopbackDisplay) -> None:
        """A caller naming another tool's family finds no owner, which is the isolation.

        The family is a digest of the serving tool's namespace name and a pod is granted only its
        own, so a pod answering here would be answering on a subject it cannot subscribe in a
        deployment that enforces its grants -- which fails as a timeout and diagnoses as nothing.
        """
        bus = FakeBus()
        claim = SessionClaim(session_id=_SESSION)

        async with serve_display(bus, claim, tool=_TOOL, pod_id=_POD, display=_where(display)):  # type: ignore[arg-type]
            with pytest.raises(NoOwnerError):
                await _attach(bus, family=Subjects.hitl_pipe_family("tools.scrape-zone_beta.1-0-0"))

        assert display.connections == 0, "a pod served an attach addressed to another tool's family"

    async def test_the_stream_does_not_ride_the_unscoped_forward_family(self, display: _LoopbackDisplay) -> None:
        """Serving the unscoped family would be serving a subject granted to no principal.

        Separate from the test above rather than folded into it: a bridge that dropped the family
        entirely would still pass that one, because a caller naming a family the pod did not serve
        would still find nobody.
        """
        bus = FakeBus()
        claim = SessionClaim(session_id=_SESSION)

        async with serve_display(bus, claim, tool=_TOOL, pod_id=_POD, display=_where(display)):  # type: ignore[arg-type]
            with pytest.raises(NoOwnerError):
                await attach_pipe(bus, _SESSION, timeout=timedelta(seconds=5))  # type: ignore[arg-type]

        assert display.connections == 0, "a pod served an attach on the unscoped forward subject"


class TestAPodThatHoldsNoClaimNeverShowsItsDisplay:
    """The claim is what entitles this pod, so both ways of not having one have to refuse."""

    async def test_serving_refuses_to_start_on_a_claim_already_lost(self, display: _LoopbackDisplay) -> None:
        """Subscribing at all would be advertising a display this pod does not own."""
        bus = FakeBus()
        claim = SessionClaim(session_id=_SESSION)
        claim.lost.set()

        with pytest.raises(RuntimeError, match="not held by this pod"):
            async with serve_display(bus, claim, tool=_TOOL, pod_id=_POD, display=_where(display)):  # type: ignore[arg-type]
                pytest.fail("a pod served the display of a session it had already lost")

        assert not bus.registrations, "a lost session had its display subscribed anyway"
        assert display.connections == 0

    async def test_an_attach_after_the_claim_is_lost_never_reaches_the_display(self, display: _LoopbackDisplay) -> None:
        """Dropping the subscription is prompt, not instantaneous, and the gap is the hazard.

        Three assertions, and none is redundant. The caller must LEARN it reached the wrong pod,
        or it waits on a stream nobody is feeding. The display must never have been opened, which
        is what says the refusal came from the claim check rather than from any other way an
        attach can fail. And the reported type is checked for the same reason: a malformed attach
        and a version that cannot be agreed both reach the caller as this same class.
        """
        bus = FakeBus()
        claim = SessionClaim(session_id=_SESSION)

        async with serve_display(bus, claim, tool=_TOOL, pod_id=_POD, display=_where(display)):  # type: ignore[arg-type]
            claim.lost.set()
            endpoint = await _attach(bus)
            with pytest.raises(PipeRemoteError) as caught:
                async with open_pipe(bus, endpoint) as stream:  # type: ignore[arg-type]
                    await asyncio.wait_for(stream.receive(), timeout=5)

        assert caught.value.type_name == "RuntimeError"
        assert "moved to another pod" in caught.value.message
        assert display.connections == 0, "a pod that had lost the session still connected to its display"


class TestALostClaimEndsALiveStream:
    """An operator watching a display this pod no longer owns is the thing being prevented."""

    async def test_losing_the_claim_ends_the_stream(self, display: _LoopbackDisplay) -> None:
        """The display is silent after its greeting, which is what a session looks like most of
        the time -- so if the claim did not ride alongside the relay this would sit until the
        test's own timeout."""
        bus = FakeBus()
        claim = SessionClaim(session_id=_SESSION)

        async with serve_display(bus, claim, tool=_TOOL, pod_id=_POD, display=_where(display)):  # type: ignore[arg-type]
            endpoint = await _attach(bus)
            async with open_pipe(bus, endpoint) as stream:  # type: ignore[arg-type]
                assert await asyncio.wait_for(stream.receive(), timeout=5) == _RFB_BANNER
                claim.lost.set()
                ended = await asyncio.wait_for(stream.receive(), timeout=5)

        assert ended is None, "the operator's stream outlived the claim that entitled it"

    async def test_the_display_connection_is_released_when_the_claim_goes(self, display: _LoopbackDisplay) -> None:
        """A handover must not leave this pod connected to the display it stopped owning.

        A connection left open holds an ``x11vnc`` client slot for a display this pod is no longer
        entitled to, one per handover. Asserted from the DISPLAY's own side, because that is the
        end that can tell a closed socket from one that merely stopped being read.

        **What it cannot tell you**, established by breaking it rather than assumed: deleting
        ``relay_stream``'s ``writer.close()`` leaves this passing, because the writer becomes
        unreachable when the relay returns and CPython collects the transport. So this pins the
        PROPERTY -- nothing retains the connection past the stream -- and is not a guard on that
        one line. An implementation that parked the writer somewhere is what it would catch.
        """
        bus = FakeBus()
        claim = SessionClaim(session_id=_SESSION)

        async with serve_display(bus, claim, tool=_TOOL, pod_id=_POD, display=_where(display)):  # type: ignore[arg-type]
            endpoint = await _attach(bus)
            async with open_pipe(bus, endpoint) as stream:  # type: ignore[arg-type]
                assert await asyncio.wait_for(stream.receive(), timeout=5) == _RFB_BANNER
                assert not display.disconnected.is_set(), "the display was let go before the claim was lost"
                claim.lost.set()
                await asyncio.wait_for(display.disconnected.wait(), timeout=5)


async def test_the_display_stream_and_the_control_plane_never_share_a_subject() -> None:
    """One session is owner-routed twice, and the two must not derive the same subject.

    ``serve_owner`` subscribes with a queue group keyed on the subject, so if a session's
    control plane and its display stream shared a family the broker would split one pod's
    messages between the two handlers -- roughly half its control messages arriving at the
    stream's attach handler and half its attaches at the control handler. Each would look
    like a malformed request rather than a misrouted one, which is why this is asserted on
    the SUBJECTS rather than left to a round-trip test that would pass whenever both ends
    happened to agree.
    """
    from threetears.nats import Subjects

    tool = "tools.scrape-zone_alpha.1-0-0"
    session = "session-abc"
    control = Subjects.forward_scoped(Subjects.hitl_forward_family(tool), session).path
    stream = Subjects.forward_scoped(Subjects.hitl_pipe_family(tool), session).path
    assert control != stream, "a session's control plane and display stream derive one subject"
