"""the server heartbeat and the per-connection credential expiry, through the real handler.

Every test drives a real :class:`~threetears.channels.websocket.WebSocketHandler` through
``handle_connection``. The peer is a socket whose inbound frames the test feeds, and whose
``close`` does NOT unblock a pending read -- a peer that has vanished never completes the close
handshake, so the handler has to end the connection itself rather than wait for a read to fail.

Time is injected, never slept: the heartbeat's wait is a manual ticker the test releases one tick
at a time, and the credential clock is a value the test sets.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from threetears.channels.protocol import ChannelMessage, ChannelResponse
from threetears.channels.websocket import UNAUTHENTICATED, WebSocketHandler

from .websocket_support import EchoRouter

#: generous guard for a step that should complete at once; a hang fails the test instead of
#: blocking the suite.
_STEP_TIMEOUT = 5.0

_PING = {"type": "ping"}
_PONG = json.dumps({"type": "pong"})


class _PeerSocket:
    """a socket whose peer the test drives frame by frame.

    ``receive_text`` waits for whatever the test feeds; :meth:`hang_up` makes the next read fail
    the way a dropped transport does. ``close`` records its code and nothing else -- it does not
    unblock a pending read, because a peer that is gone never answers the close.
    """

    def __init__(self) -> None:
        self.query_params: dict[str, str] = {"token": "valid-token"}
        self.sent: list[str] = []
        self.close_codes: list[int] = []
        self._inbound: asyncio.Queue[str | None] = asyncio.Queue()
        self._reading = asyncio.Event()

    async def accept(self) -> None:
        """accept the connection."""

    async def receive_text(self) -> str:
        """wait for the next fed frame, or fail once the peer hung up.

        :return: the frame text
        :rtype: str
        :raises ConnectionError: after :meth:`hang_up`
        """
        self._reading.set()
        text = await self._inbound.get()
        self._reading.clear()
        if text is None:
            raise ConnectionError("peer hung up")
        return text

    async def send_text(self, data: str) -> None:
        """record an outbound frame.

        :param data: the frame text
        :ptype data: str
        """
        self.sent.append(data)

    async def close(self, code: int = 1000) -> None:
        """record the close code; a gone peer never answers, so a pending read stays pending.

        :param code: the close code
        :ptype code: int
        """
        self.close_codes.append(code)

    async def feed(self, text: str) -> None:
        """deliver one frame and wait until the handler is reading again, i.e. has handled it.

        :param text: the frame text
        :ptype text: str
        """
        self._reading.clear()
        self._inbound.put_nowait(text)
        await asyncio.wait_for(self._reading.wait(), _STEP_TIMEOUT)

    def push(self, text: str) -> None:
        """deliver one frame without waiting for the handler to finish with it.

        :param text: the frame text
        :ptype text: str
        """
        self._inbound.put_nowait(text)

    def hang_up(self) -> None:
        """make the next read fail, as a dropped transport does."""
        self._inbound.put_nowait(None)

    def frames(self) -> list[dict[str, Any]]:
        """every outbound frame, decoded.

        :return: decoded frames in send order
        :rtype: list[dict[str, Any]]
        """
        return [json.loads(raw) for raw in self.sent]


class _ManualTicker:
    """the heartbeat's wait, released one tick at a time by the test."""

    def __init__(self) -> None:
        self.delays: list[float] = []
        self._entered: asyncio.Queue[None] = asyncio.Queue()
        self._release: asyncio.Queue[None] = asyncio.Queue()

    async def sleep(self, seconds: float) -> None:
        """record the requested wait, then block until the test releases a tick.

        :param seconds: the interval the heartbeat asked to wait
        :ptype seconds: float
        """
        self.delays.append(seconds)
        self._entered.put_nowait(None)
        await self._release.get()

    async def armed(self) -> None:
        """wait until the heartbeat is waiting for its next tick."""
        await asyncio.wait_for(self._entered.get(), _STEP_TIMEOUT)

    async def tick(self) -> None:
        """release one tick and wait until the heartbeat is waiting for the next one."""
        self._release.put_nowait(None)
        await self.armed()

    def release(self) -> None:
        """release one tick without waiting for another (the tick that ends the connection)."""
        self._release.put_nowait(None)


class _Clock:
    """a wall clock the test sets."""

    def __init__(self, now: float) -> None:
        self.now = now

    def __call__(self) -> float:
        """read the clock.

        :return: seconds since the epoch, as the test set them
        :rtype: float
        """
        return self.now


def _auth_with_exp(exp: object) -> Any:
    """a validator that accepts ``valid-token`` and reports ``exp`` among its claims.

    :param exp: the ``exp`` claim value to report
    :ptype exp: object
    :return: the validator
    :rtype: Any
    """

    async def validate(token: str) -> dict[str, Any]:
        del token
        return {"user_id": "user-123", "customer_id": "cust-1", "exp": exp}

    return validate


async def _no_exp_auth(token: str) -> dict[str, Any]:
    """a validator whose claims carry no ``exp``.

    :param token: the presented token
    :ptype token: str
    :return: the claims
    :rtype: dict[str, Any]
    """
    del token
    return {"user_id": "user-123", "customer_id": "cust-1"}


class _RecordingRouter:
    """records every routed message and answers each with an echo."""

    def __init__(self) -> None:
        self.routed: list[str] = []

    async def route_inbound(self, message: ChannelMessage) -> ChannelResponse | None:
        """record and echo.

        :param message: the inbound message
        :ptype message: ChannelMessage
        :return: the echo
        :rtype: ChannelResponse | None
        """
        self.routed.append(message.content)
        return ChannelResponse(content=f"echo: {message.content}")


class _HeldRouter:
    """a router whose turn stays in flight until the test lets it finish."""

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.finish = asyncio.Event()

    async def route_inbound(self, message: ChannelMessage) -> ChannelResponse | None:
        """signal the turn started, then wait to be let go.

        :param message: the inbound message
        :ptype message: ChannelMessage
        :return: the reply
        :rtype: ChannelResponse | None
        """
        self.started.set()
        await self.finish.wait()
        return ChannelResponse(content=f"done: {message.content}")


# parity-exempt: RoomState stand-in recording heartbeat/register/unregister without a presence collection
class _FakeRoomState:
    """records the presence calls the handler makes."""

    def __init__(self) -> None:
        self.registered: list[str] = []
        self.unregistered: list[str] = []
        self.heartbeats: list[str] = []

    async def register(self, connection_id: str, socket: Any) -> None:
        self.registered.append(connection_id)

    async def unregister(self, connection_id: str, socket: Any | None = None) -> None:
        self.unregistered.append(connection_id)

    async def heartbeat(self, connection_id: str) -> bool:
        self.heartbeats.append(connection_id)
        return True


# parity-exempt: fanout stand-in recording join/leave so the disconnect cleanup can be asserted
class _FakeFanout:
    """records room joins and leaves."""

    def __init__(self) -> None:
        self.joined: list[str] = []
        self.left: list[tuple[str, str]] = []

    async def join_room(self, room_id: str, connection_id: str, user_id: str, customer_id: str) -> None:
        self.joined.append(room_id)

    async def leave_room(self, room_id: str, connection_id: str) -> None:
        self.left.append((room_id, connection_id))

    async def broadcast(self, room_id: str, payload: str, *, exclude: str | None = None) -> None:
        del room_id, payload, exclude


def _handler(
    ticker: _ManualTicker,
    *,
    router: Any = None,
    auth: Any = _no_exp_auth,
    clock: _Clock | None = None,
    config: dict[str, Any] | None = None,
    **seams: Any,
) -> WebSocketHandler:
    """a real handler with the test's ticker and clock in place of real time.

    :return: the handler
    :rtype: WebSocketHandler
    """
    return WebSocketHandler(
        router=router if router is not None else EchoRouter(),
        auth_validator=auth,
        config=config,
        heartbeat_sleep=ticker.sleep,
        wall_clock=clock if clock is not None else _Clock(0.0),
        **seams,
    )


async def _finished(task: asyncio.Task[None]) -> None:
    """wait for the connection to end, failing the test if it does not.

    :param task: the ``handle_connection`` task
    :ptype task: asyncio.Task[None]
    """
    await asyncio.wait_for(task, _STEP_TIMEOUT)


class TestServerHeartbeat:
    """the handler pings every open connection and ends one whose peer stops answering."""

    @pytest.mark.asyncio
    async def test_a_peer_that_stops_answering_is_closed_and_unregistered(self) -> None:
        ticker = _ManualTicker()
        handler = _handler(ticker)
        peer = _PeerSocket()

        task = asyncio.create_task(handler.handle_connection(peer))
        await ticker.armed()
        assert handler.registry.get_connections("user-123") == [peer]

        await ticker.tick()
        assert peer.frames()[-1] == _PING
        assert peer.close_codes == []

        ticker.release()
        await _finished(task)

        assert peer.close_codes == [1011]
        assert handler.registry.get_connections("user-123") == []

    @pytest.mark.asyncio
    async def test_a_peer_that_answers_every_ping_is_kept(self) -> None:
        ticker = _ManualTicker()
        handler = _handler(ticker)
        peer = _PeerSocket()

        task = asyncio.create_task(handler.handle_connection(peer))
        await ticker.armed()
        for _ in range(3):
            await ticker.tick()
            assert peer.frames()[-1] == _PING
            await peer.feed(_PONG)

        await ticker.tick()
        assert peer.close_codes == []
        assert [frame for frame in peer.frames() if frame == _PING] == [_PING] * 4
        # a pong is the answer to a ping, never a frame to complain about
        assert not [frame for frame in peer.frames() if frame.get("type") == "error"]

        peer.hang_up()
        await _finished(task)
        assert peer.close_codes == []

    @pytest.mark.asyncio
    async def test_any_frame_from_the_peer_counts_as_an_answer(self) -> None:
        ticker = _ManualTicker()
        router = _RecordingRouter()
        handler = _handler(ticker, router=router)
        peer = _PeerSocket()

        task = asyncio.create_task(handler.handle_connection(peer))
        await ticker.armed()
        await ticker.tick()
        await peer.feed(json.dumps({"type": "message", "content": "still here"}))
        await ticker.tick()

        assert peer.close_codes == []
        assert router.routed == ["still here"]

        peer.hang_up()
        await _finished(task)

    @pytest.mark.asyncio
    async def test_the_configured_interval_is_the_wait_between_pings(self) -> None:
        ticker = _ManualTicker()
        handler = _handler(ticker, config={"heartbeat_interval": 7})
        peer = _PeerSocket()

        task = asyncio.create_task(handler.handle_connection(peer))
        await ticker.armed()
        await ticker.tick()
        await peer.feed(_PONG)
        await ticker.tick()

        assert ticker.delays == [7, 7, 7]

        peer.hang_up()
        await _finished(task)

    @pytest.mark.asyncio
    async def test_the_default_interval_is_thirty_seconds(self) -> None:
        ticker = _ManualTicker()
        handler = _handler(ticker)
        peer = _PeerSocket()

        task = asyncio.create_task(handler.handle_connection(peer))
        await ticker.armed()

        assert ticker.delays == [30]

        peer.hang_up()
        await _finished(task)

    @pytest.mark.asyncio
    async def test_a_turn_in_flight_is_not_taken_for_a_silent_peer(self) -> None:
        """while the handler is busy with a frame it is not reading, so silence proves nothing."""
        ticker = _ManualTicker()
        router = _HeldRouter()
        handler = _handler(ticker, router=router)
        peer = _PeerSocket()

        task = asyncio.create_task(handler.handle_connection(peer))
        await ticker.armed()
        peer.push(json.dumps({"type": "message", "content": "long turn"}))
        await asyncio.wait_for(router.started.wait(), _STEP_TIMEOUT)

        await ticker.tick()
        await ticker.tick()
        await ticker.tick()
        assert peer.close_codes == []

        router.finish.set()
        await peer.feed(_PONG)
        await ticker.tick()
        assert peer.close_codes == []
        assert {"type": "response", "content": "done: long turn", "metadata": {}} in peer.frames()

        peer.hang_up()
        await _finished(task)

    @pytest.mark.asyncio
    async def test_a_closed_dead_peer_leaves_its_rooms_as_a_disconnect_does(self) -> None:
        ticker = _ManualTicker()
        room_state = _FakeRoomState()
        fanout = _FakeFanout()
        handler = _handler(ticker, room_state=room_state, room_fanout=fanout)
        peer = _PeerSocket()

        task = asyncio.create_task(handler.handle_connection(peer))
        await ticker.armed()
        await peer.feed(json.dumps({"type": "join", "room": "room-1"}))
        assert fanout.joined == ["room-1"]
        connection_id = room_state.registered[0]

        await ticker.tick()
        ticker.release()
        await _finished(task)

        assert peer.close_codes == [1011]
        assert fanout.left == [("room-1", connection_id)]
        assert room_state.unregistered == [connection_id]
        assert handler.registry.get_connections("user-123") == []

    @pytest.mark.asyncio
    async def test_an_answering_member_of_a_room_keeps_its_presence_fresh(self) -> None:
        """the presence sweeper evicts a connection whose heartbeat goes stale; a live one must not go stale."""
        ticker = _ManualTicker()
        room_state = _FakeRoomState()
        fanout = _FakeFanout()
        handler = _handler(ticker, room_state=room_state, room_fanout=fanout)
        peer = _PeerSocket()

        task = asyncio.create_task(handler.handle_connection(peer))
        await ticker.armed()
        await peer.feed(json.dumps({"type": "join", "room": "room-1"}))
        connection_id = room_state.registered[0]

        await ticker.tick()
        await peer.feed(_PONG)
        await ticker.tick()

        assert room_state.heartbeats == [connection_id, connection_id]

        peer.hang_up()
        await _finished(task)

    @pytest.mark.asyncio
    async def test_a_connection_in_no_room_writes_no_presence_heartbeat(self) -> None:
        ticker = _ManualTicker()
        room_state = _FakeRoomState()
        handler = _handler(ticker, room_state=room_state, room_fanout=_FakeFanout())
        peer = _PeerSocket()

        task = asyncio.create_task(handler.handle_connection(peer))
        await ticker.armed()
        await ticker.tick()

        assert room_state.heartbeats == []

        peer.hang_up()
        await _finished(task)

    @pytest.mark.asyncio
    async def test_the_heartbeat_stops_when_the_peer_disconnects(self) -> None:
        ticker = _ManualTicker()
        handler = _handler(ticker)
        peer = _PeerSocket()

        task = asyncio.create_task(handler.handle_connection(peer))
        await ticker.armed()
        peer.hang_up()
        await _finished(task)

        # the ticker was never released: the wait in progress was cancelled with the connection
        assert ticker.delays == [30]
        assert peer.sent == [json.dumps({"type": "connected", "user_id": "user-123"})]

    @pytest.mark.parametrize("interval", [0, -5])
    def test_a_non_positive_interval_is_refused(self, interval: int) -> None:
        with pytest.raises(ValueError, match="heartbeat_interval"):
            WebSocketHandler(router=EchoRouter(), auth_validator=_no_exp_auth, config={"heartbeat_interval": interval})


class TestCredentialExpiry:
    """a connection ends when the credential it authenticated with expires."""

    @pytest.mark.asyncio
    async def test_a_frame_before_expiry_is_routed(self) -> None:
        ticker = _ManualTicker()
        router = _RecordingRouter()
        clock = _Clock(999.0)
        handler = _handler(ticker, router=router, auth=_auth_with_exp(1000), clock=clock)
        peer = _PeerSocket()

        task = asyncio.create_task(handler.handle_connection(peer))
        await ticker.armed()
        await peer.feed(json.dumps({"type": "message", "content": "in time"}))

        assert router.routed == ["in time"]
        assert peer.close_codes == []

        peer.hang_up()
        await _finished(task)

    @pytest.mark.asyncio
    async def test_the_first_frame_after_expiry_closes_the_connection_unauthenticated(self) -> None:
        ticker = _ManualTicker()
        router = _RecordingRouter()
        clock = _Clock(999.0)
        handler = _handler(ticker, router=router, auth=_auth_with_exp(1000), clock=clock)
        peer = _PeerSocket()

        task = asyncio.create_task(handler.handle_connection(peer))
        await ticker.armed()
        clock.now = 1000.0
        peer.push(json.dumps({"type": "message", "content": "too late"}))
        await _finished(task)

        assert router.routed == []
        assert peer.frames()[-1]["type"] == "error"
        assert peer.frames()[-1]["code"] == UNAUTHENTICATED
        assert peer.close_codes == [1008]
        assert handler.registry.get_connections("user-123") == []

    @pytest.mark.asyncio
    async def test_an_idle_connection_is_closed_unauthenticated_on_the_first_tick_after_expiry(self) -> None:
        ticker = _ManualTicker()
        clock = _Clock(999.0)
        handler = _handler(ticker, auth=_auth_with_exp(1000), clock=clock)
        peer = _PeerSocket()

        task = asyncio.create_task(handler.handle_connection(peer))
        await ticker.armed()
        await ticker.tick()
        await peer.feed(_PONG)
        assert peer.close_codes == []

        clock.now = 1030.0
        ticker.release()
        await _finished(task)

        last = peer.frames()[-1]
        assert last["type"] == "error"
        assert last["code"] == UNAUTHENTICATED
        assert peer.close_codes == [1008]
        assert handler.registry.get_connections("user-123") == []

    @pytest.mark.asyncio
    async def test_an_expired_connection_is_closed_even_mid_turn(self) -> None:
        ticker = _ManualTicker()
        router = _HeldRouter()
        clock = _Clock(999.0)
        handler = _handler(ticker, router=router, auth=_auth_with_exp(1000), clock=clock)
        peer = _PeerSocket()

        task = asyncio.create_task(handler.handle_connection(peer))
        await ticker.armed()
        peer.push(json.dumps({"type": "message", "content": "long turn"}))
        await asyncio.wait_for(router.started.wait(), _STEP_TIMEOUT)

        clock.now = 1001.0
        ticker.release()
        await _finished(task)

        assert peer.frames()[-1]["code"] == UNAUTHENTICATED
        assert peer.close_codes == [1008]

    @pytest.mark.asyncio
    async def test_claims_without_exp_never_expire(self) -> None:
        ticker = _ManualTicker()
        router = _RecordingRouter()
        clock = _Clock(10.0**12)
        handler = _handler(ticker, router=router, clock=clock)
        peer = _PeerSocket()

        task = asyncio.create_task(handler.handle_connection(peer))
        await ticker.armed()
        await ticker.tick()
        await peer.feed(json.dumps({"type": "message", "content": "whenever"}))
        await ticker.tick()

        assert router.routed == ["whenever"]
        assert peer.close_codes == []

        peer.hang_up()
        await _finished(task)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("exp", ["1000", True, None, [1000]])
    async def test_a_validator_reporting_a_malformed_exp_is_a_host_bug(self, exp: object) -> None:
        ticker = _ManualTicker()
        handler = _handler(ticker, auth=_auth_with_exp(exp))
        peer = _PeerSocket()

        with pytest.raises(TypeError, match="exp"):
            await asyncio.wait_for(handler.handle_connection(peer), _STEP_TIMEOUT)
