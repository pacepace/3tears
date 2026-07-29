"""The relay's transport seam, exercised with the display replaced by two objects in memory.

Every assertion here is that substituting the transport changes NOTHING a caller can observe:
the same bytes arrive in the same order, the same stop signal ends the relay, the same
exception types are ordinary rather than faults, and the same cleanup runs on the way out. The
suite in ``test_operator.py`` proves those properties over a real TCP socket and is deliberately
untouched; this one proves the seam did not quietly become a second behaviour.

Kept out of that file rather than appended to it because the acceptance for the seam was that
its existing suite passes with no edits at all, and a file nobody opened is the cheapest way to
show that.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

import pytest
from threetears.scrape.operator import DisplayReader, DisplayTransport, DisplayWriter, relay_stream

#: An endpoint that is not an address, handed to every relay in this module.
#:
#: If the seam ever regresses to opening its own TCP connection, this cannot silently succeed:
#: there is no host called this and no port 0 to connect to, so the relay would raise instead of
#: relaying. Passing a real loopback endpoint would let a regression pass, because the default
#: transport would have connected to something.
_NOT_AN_ADDRESS = ("session-a3f9c2", 0)


class _FakeDisplayReader(DisplayReader):
    """The display's half of an in-memory connection: a script, then silence or an ending.

    *finish* is how the two halves cooperate the way a real RFB server does. The TCP test's
    server writes, reads the operator's bytes, and only then closes; with no such handshake the
    reader would report EOF before the other pump had run, ending the relay and proving only one
    direction. Left ``None`` the reader never returns at all, which is the state an idle
    operator's session is in for most of its life.
    """

    def __init__(self, chunks: Sequence[bytes], *, finish: asyncio.Event | None = None) -> None:
        self._pending = list(chunks)
        self._finish = finish

    async def read(self, n: int) -> bytes:
        """Hand over the next scripted chunk, then end or park."""
        if self._pending:
            return self._pending.pop(0)[:n]
        if self._finish is None:
            await asyncio.Event().wait()
        else:
            await self._finish.wait()
        return b""


class _FakeDisplayWriter(DisplayWriter):
    """The operator's bytes, kept where a test can compare them, plus what was done to close."""

    def __init__(self) -> None:
        self.written: list[bytes] = []
        self.drained = 0
        self.closed = False
        self.awaited_close = False
        #: Set on every write, so a reader can wait for the operator's bytes to land.
        self.wrote = asyncio.Event()

    def write(self, data: bytes) -> None:
        """Record what the operator sent, verbatim."""
        self.written.append(data)
        self.wrote.set()

    async def drain(self) -> None:
        """Count the drain rather than doing anything: nothing here is buffered."""
        self.drained += 1

    def close(self) -> None:
        """Record that the relay let go of what the transport opened."""
        self.closed = True

    async def wait_closed(self) -> None:
        """Record that the relay waited for the close, as it does for a socket."""
        self.awaited_close = True


class _FakeDisplayTransport(DisplayTransport):
    """Yields one prepared pair and remembers the endpoint it was asked for."""

    def __init__(self, reader: DisplayReader, writer: DisplayWriter) -> None:
        self._reader = reader
        self._writer = writer
        self.opened: list[tuple[str, int]] = []

    async def __call__(self, host: str, port: int) -> tuple[DisplayReader, DisplayWriter]:
        """Hand back the pair, recording what it was told the display was."""
        self.opened.append((host, port))
        return (self._reader, self._writer)


class TestASubstitutedTransportRelaysExactlyAsTheTcpDefaultDoes:
    """The seam adds a way in, not a second set of rules for what happens once you are in."""

    async def test_both_directions_carry_bytes_verbatim(self) -> None:
        """The same byte pattern the TCP test uses, over a transport that owns no socket.

        Also the evidence that the pair actually came from the transport rather than from a
        connection this function opened for itself: nothing could have been connected to
        ``_NOT_AN_ADDRESS``, so bytes arriving at all means the substitution took effect.
        """
        payload = b"\x00\x00\x01\x2c\xff\xfe\x00RFB-ish\x00"
        writer = _FakeDisplayWriter()
        reader = _FakeDisplayReader([payload], finish=writer.wrote)
        transport = _FakeDisplayTransport(reader, writer)

        client_received: list[bytes] = []
        from_client: asyncio.Queue[bytes] = asyncio.Queue()
        await from_client.put(payload)

        async def _send(data: bytes) -> None:
            client_received.append(data)

        async def _receive() -> bytes:
            return await from_client.get()

        await asyncio.wait_for(
            relay_stream(_send, _receive, *_NOT_AN_ADDRESS, transport=transport),
            timeout=5,
        )

        assert b"".join(client_received) == payload, "the display's bytes reached the client altered"
        assert writer.written == [payload], "the operator's bytes reached the display altered"
        assert transport.opened == [_NOT_AN_ADDRESS], (
            "the endpoint did not reach the transport unchanged, so something between the display "
            "lookup and the transport is interpreting it"
        )

    async def test_what_the_transport_opened_is_closed_on_the_way_out(self) -> None:
        """The relay closes the writer it was given, which a substitute depends on.

        Over TCP an unclosed writer is a socket the kernel eventually reclaims. Over anything
        else it is a stream nothing collects: one per operator session, each holding whatever
        the transport allocated to carry it.
        """
        writer = _FakeDisplayWriter()
        reader = _FakeDisplayReader([b"x"], finish=writer.wrote)
        transport = _FakeDisplayTransport(reader, writer)

        from_client: asyncio.Queue[bytes] = asyncio.Queue()
        await from_client.put(b"y")

        await asyncio.wait_for(
            relay_stream(lambda _d: asyncio.sleep(0), from_client.get, *_NOT_AN_ADDRESS, transport=transport),
            timeout=5,
        )

        assert writer.closed, "the relay ended without closing what the transport opened"
        assert writer.awaited_close, "the relay did not wait for the close it started"

    async def test_the_other_end_going_away_is_the_ordinary_end_not_a_fault(self) -> None:
        """``benign`` means the same thing here as it does over a socket.

        A client library that signals disconnection by raising is the common path, and reporting
        it as a fault is what buries the rare real one. Nothing about that judgement belongs to
        the transport, so nothing about it may change when the transport does.
        """

        class _ClientWentAway(Exception):
            """Stands in for the framework's own disconnect signal."""

        async def _receive() -> bytes:
            raise _ClientWentAway

        writer = _FakeDisplayWriter()
        transport = _FakeDisplayTransport(_FakeDisplayReader([]), writer)

        # Returns rather than raising: nothing here is a fault.
        await asyncio.wait_for(
            relay_stream(
                lambda _d: asyncio.sleep(0),
                _receive,
                *_NOT_AN_ADDRESS,
                transport=transport,
                benign=(_ClientWentAway,),
            ),
            timeout=5,
        )
        assert writer.closed, "an ordinary disconnect left the transport's writer open"

    async def test_a_pump_failing_for_any_other_reason_is_raised(self) -> None:
        """The counterpart, so a substituted transport cannot turn a fault into silence."""

        async def _receive() -> bytes:
            raise RuntimeError("the transport broke")

        transport = _FakeDisplayTransport(_FakeDisplayReader([]), _FakeDisplayWriter())

        with pytest.raises(OSError, match="failed mid-stream"):
            await asyncio.wait_for(
                relay_stream(lambda _d: asyncio.sleep(0), _receive, *_NOT_AN_ADDRESS, transport=transport),
                timeout=5,
            )

    async def test_the_relay_ends_when_the_stop_signal_completes(self) -> None:
        """A relay parked on a silent transport still lets go when the claim does.

        The reader here never returns and the client never sends, which is the shape a real
        session spends most of its life in -- so if the stop signal did not ride alongside the
        pumps this would sit until the test timed out.
        """
        lost = asyncio.Event()
        writer = _FakeDisplayWriter()
        transport = _FakeDisplayTransport(_FakeDisplayReader([]), writer)

        relay = asyncio.create_task(
            relay_stream(
                lambda _d: asyncio.sleep(0),
                asyncio.Event().wait,
                *_NOT_AN_ADDRESS,
                transport=transport,
                until=lost.wait,
            )
        )
        await asyncio.sleep(0.05)
        assert not relay.done(), "the relay ended before the claim was lost"
        lost.set()
        await asyncio.wait_for(relay, timeout=5)
        assert writer.closed, "losing the claim left the transport's writer open"

    async def test_a_relay_with_no_stop_signal_still_runs(self) -> None:
        """One pod has no handover to survive, and that is not a property of the transport."""
        transport = _FakeDisplayTransport(_FakeDisplayReader([]), _FakeDisplayWriter())

        relay = asyncio.create_task(
            relay_stream(
                lambda _d: asyncio.sleep(0),
                asyncio.Event().wait,
                *_NOT_AN_ADDRESS,
                transport=transport,
            )
        )
        await asyncio.sleep(0.05)
        assert not relay.done(), "a relay with no stop signal ended on its own"
        relay.cancel()
        with pytest.raises(asyncio.CancelledError):
            await relay

    async def test_a_transport_that_cannot_open_fails_the_relay(self) -> None:
        """Whatever the transport raises reaches the caller, who is the only one who can act.

        The route logs the session, host and port from this and closes the operator's socket
        1011. Swallowing it here would leave that operator watching a page that never connects
        and no line anywhere saying why.
        """

        async def _refuses(host: str, port: int) -> tuple[DisplayReader, DisplayWriter]:
            raise OSError(f"nothing is serving {host}:{port}")

        with pytest.raises(OSError, match="nothing is serving session-a3f9c2:0"):
            await asyncio.wait_for(
                relay_stream(lambda _d: asyncio.sleep(0), asyncio.Event().wait, *_NOT_AN_ADDRESS, transport=_refuses),
                timeout=5,
            )
