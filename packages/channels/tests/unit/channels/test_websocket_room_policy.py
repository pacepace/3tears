"""the per-room access policy, and taking a live member out of a room.

the namespace gate sees only the namespace a room resolves to, and rooms share
namespaces -- so a user with read on a shared namespace could join a colleague's
PRIVATE room and receive everything streamed to it. these tests hold the seam
that closes that: ``room_policy`` is consulted after the namespace gate on every
room action the handler gates, both must allow, and a member whose access has
since been withdrawn can be taken out of the room while their socket is live.

the sockets here stay open until the test hangs them up, and the room delivers
broadcasts to its current members only, so "received no frame" is an observation
of the room's actual membership rather than of a recording that nothing reads.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any
from uuid import UUID, uuid4

import pytest

from threetears.agent.acl import AccessDenied
from threetears.channels.frames import Frame, OpResult, RoomAccessRequest
from threetears.channels.websocket import UNAUTHENTICATED, WebSocketAuthRefused, WebSocketHandler

from .websocket_support import EchoRouter

ROOM = "cust:story:main:private-draft.md"
_HANG_UP = object()


class _LiveSocket:
    """a websocket that stays open until the test hangs it up.

    not a ``Fake<Name>``: it implements the four methods of
    :class:`~threetears.channels.websocket.WebSocketProtocol` and nothing else.
    """

    def __init__(self, token: str) -> None:
        self.query_params: dict[str, str] = {"token": token}
        self.sent: list[str] = []
        self._inbox: asyncio.Queue[object] = asyncio.Queue()

    async def accept(self) -> None:
        return None

    async def receive_text(self) -> str:
        item = await self._inbox.get()
        if item is _HANG_UP:
            raise ConnectionError("client hung up")
        assert isinstance(item, str)
        return item

    async def send_text(self, data: str) -> None:
        self.sent.append(data)

    async def close(self, code: int = 1000) -> None:
        return None

    def push(self, frame: dict[str, Any]) -> None:
        self._inbox.put_nowait(json.dumps(frame))

    def hang_up(self) -> None:
        self._inbox.put_nowait(_HANG_UP)

    def frames(self, frame_type: str) -> list[dict[str, Any]]:
        return [f for f in (json.loads(s) for s in self.sent) if f.get("type") == frame_type]


class _LoopbackRooms:
    """room state and fanout in one process: a broadcast reaches the room's CURRENT members.

    not a ``Fake<Name>``: it serves the handler's calls on
    :class:`~threetears.channels.presence.room_state.RoomState` (``register`` /
    ``unregister``) and :class:`~threetears.channels.presence.fanout.RoomFanout`
    (``join_room`` / ``leave_room`` / ``broadcast``) and delivers the way the real
    pair does -- to the live sockets of the room's members, skipping ``exclude``.
    """

    def __init__(self) -> None:
        self.sockets: dict[str, Any] = {}
        self.members: dict[str, set[str]] = {}
        self.joined: list[tuple[str, str]] = []
        self.left: list[tuple[str, str]] = []

    async def register(self, connection_id: str, socket: Any) -> None:
        self.sockets[connection_id] = socket

    async def unregister(self, connection_id: str, socket: Any | None = None) -> None:
        self.sockets.pop(connection_id, None)

    async def join_room(self, room_id: str, connection_id: str, user_id: str, customer_id: str) -> None:
        self.joined.append((room_id, connection_id))
        self.members.setdefault(room_id, set()).add(connection_id)

    async def leave_room(self, room_id: str, connection_id: str) -> None:
        self.left.append((room_id, connection_id))
        self.members.get(room_id, set()).discard(connection_id)

    async def broadcast(self, room_id: str, payload: str, *, exclude: str | None = None) -> None:
        for connection_id in sorted(self.members.get(room_id, set())):
            if connection_id != exclude and connection_id in self.sockets:
                await self.sockets[connection_id].send_text(payload)


class _Policy:
    """a room policy that refuses the ``(user_id, action)`` pairs it is told to."""

    def __init__(self) -> None:
        self.refused: set[tuple[str, str]] = set()
        self.requests: list[RoomAccessRequest] = []
        self.answer_override: object | None = None
        self.unreachable = False

    async def __call__(self, request: RoomAccessRequest) -> bool:
        self.requests.append(request)
        if self.unreachable:
            raise RuntimeError("policy store unreachable")
        if self.answer_override is not None:
            return self.answer_override  # type: ignore[return-value]
        return (request.user_id, request.action) not in self.refused


class _Ns:
    id = uuid4()
    customer_id = uuid4()
    namespace_type = "story"
    owner_agent_id = uuid4()


async def _resolve(room_id: str) -> _Ns:
    return _Ns()


class _Ops:
    def __init__(self) -> None:
        self.appended: list[str] = []

    async def __call__(self, room_id: str, user_id: str, frame: Frame) -> OpResult:
        self.appended.append(user_id)
        return OpResult(seq=len(self.appended))


class _Pod:
    """one handler, its room loopback, and the people connected to it."""

    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        with_namespace_gate: bool = True,
        replay: Any = None,
    ) -> None:
        self.rooms = _LoopbackRooms()
        self.policy = _Policy()
        self.ops = _Ops()
        self.gate_calls: list[tuple[str, UUID]] = []
        self.users: dict[str, tuple[str, str]] = {}
        self.namespace_denied: set[str] = set()

        async def _gate(
            *, ns_entity: Any, action: str, user_id: Any, agent_id: Any, cache: Any, namespace_name: Any = None
        ) -> object:
            self.gate_calls.append((action, user_id))
            if str(user_id) in self.namespace_denied:
                raise AccessDenied("denied", action=action)
            return object()

        import threetears.channels.websocket as ws_mod

        monkeypatch.setattr(ws_mod, "authorize_on_entity", _gate)

        async def _auth(token: str) -> dict[str, Any]:
            identity = self.users.get(token)
            if identity is None:
                raise WebSocketAuthRefused(UNAUTHENTICATED, "authentication required")
            return {"user_id": identity[0], "customer_id": identity[1]}

        self.handler = WebSocketHandler(
            router=EchoRouter(),
            auth_validator=_auth,
            room_state=self.rooms,  # type: ignore[arg-type]
            room_fanout=self.rooms,  # type: ignore[arg-type]
            acl_cache=object() if with_namespace_gate else None,  # type: ignore[arg-type]
            ns_resolver=_resolve if with_namespace_gate else None,
            op_handler=self.ops,
            replay_source=replay,
            room_policy=self.policy,
        )
        self._tasks: list[asyncio.Task[None]] = []

    def person(self) -> tuple[str, _LiveSocket]:
        user_id, customer_id = str(uuid4()), str(uuid4())
        token = f"token-{user_id}"
        self.users[token] = (user_id, customer_id)
        return user_id, _LiveSocket(token)

    def connect(self, socket: _LiveSocket) -> None:
        self._tasks.append(asyncio.create_task(self.handler.handle_connection(socket)))

    async def settle(self) -> None:
        for _ in range(50):
            await asyncio.sleep(0)

    async def hang_up_all(self, *sockets: _LiveSocket) -> None:
        for socket in sockets:
            socket.hang_up()
        await asyncio.gather(*self._tasks)


@pytest.fixture
def pod(monkeypatch: pytest.MonkeyPatch) -> _Pod:
    return _Pod(monkeypatch)


async def _join(pod: _Pod, socket: _LiveSocket) -> None:
    socket.push({"type": "join", "room": ROOM})
    await pod.settle()


class TestPolicyGatesEveryRoomAction:
    async def test_an_allowed_join_asks_the_policy_after_the_namespace_gate(self, pod: _Pod) -> None:
        owner, owner_socket = pod.person()
        pod.connect(owner_socket)
        await _join(pod, owner_socket)

        assert [room for room, _ in pod.rooms.joined] == [ROOM]
        assert pod.gate_calls[0][0] == "room.join"
        request = pod.policy.requests[0]
        assert request == RoomAccessRequest(
            room_id=ROOM,
            user_id=owner,
            customer_id=pod.users[owner_socket.query_params["token"]][1],
            action="room.join",
        )
        await pod.hang_up_all(owner_socket)

    async def test_a_refused_join_writes_no_membership_and_the_room_stays_silent_to_them(self, pod: _Pod) -> None:
        owner, owner_socket = pod.person()
        colleague, colleague_socket = pod.person()
        pod.policy.refused.add((colleague, "room.join"))
        pod.connect(owner_socket)
        pod.connect(colleague_socket)
        await _join(pod, owner_socket)
        await _join(pod, colleague_socket)

        owner_socket.push({"type": "typing", "room": ROOM, "payload": "private words"})
        await pod.settle()

        assert [e["message"] for e in colleague_socket.frames("error")] == ["access denied: room.join"]
        assert pod.rooms.members[ROOM] == {cid for _, cid in pod.rooms.joined}
        assert len(pod.rooms.joined) == 1
        assert colleague_socket.frames("typing") == []
        await pod.hang_up_all(owner_socket, colleague_socket)
        assert len(pod.rooms.left) == 1

    async def test_a_namespace_denial_is_final_and_the_policy_is_not_asked(self, pod: _Pod) -> None:
        stranger, socket = pod.person()
        pod.namespace_denied.add(stranger)
        pod.connect(socket)
        await _join(pod, socket)

        assert pod.rooms.joined == []
        assert pod.policy.requests == []
        await pod.hang_up_all(socket)

    async def test_a_refused_write_blocks_the_op_and_the_transient_frames(self, pod: _Pod) -> None:
        reader, reader_socket = pod.person()
        writer, writer_socket = pod.person()
        pod.policy.refused.add((reader, "entry.write"))
        pod.connect(reader_socket)
        pod.connect(writer_socket)
        await _join(pod, reader_socket)
        await _join(pod, writer_socket)

        reader_socket.push({"type": "editor.op", "room": ROOM, "payload": "delete everything"})
        reader_socket.push({"type": "cursor", "room": ROOM, "payload": "12"})
        await pod.settle()

        assert pod.ops.appended == []
        assert writer_socket.frames("editor.op") == []
        assert writer_socket.frames("cursor") == []
        assert [e["message"] for e in reader_socket.frames("error")] == [
            "access denied: entry.write",
            "access denied: entry.write",
        ]
        await pod.hang_up_all(reader_socket, writer_socket)

    async def test_the_policy_gates_rooms_when_no_namespace_gate_is_wired(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        pod = _Pod(monkeypatch, with_namespace_gate=False)
        stranger, socket = pod.person()
        pod.policy.refused.add((stranger, "room.join"))
        pod.connect(socket)
        await _join(pod, socket)

        assert pod.rooms.joined == []
        await pod.hang_up_all(socket)

    async def test_only_a_literal_true_allows(self, pod: _Pod) -> None:
        pod.policy.answer_override = "yes"
        someone, socket = pod.person()
        pod.connect(socket)
        await _join(pod, socket)

        assert pod.rooms.joined == []
        await pod.hang_up_all(socket)

    async def test_joining_a_room_twice_is_one_membership(self, pod: _Pod) -> None:
        """a second join must not take a second room reference that one disconnect cannot release."""
        someone, socket = pod.person()
        pod.connect(socket)
        await _join(pod, socket)
        await _join(pod, socket)

        assert len(pod.rooms.joined) == 1
        await pod.hang_up_all(socket)
        assert len(pod.rooms.left) == 1


class TestResumeIsGated:
    async def test_a_refused_resume_replays_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        replayed: list[str] = []

        def _replay(room_id: str, from_seq: int) -> AsyncIterator[str]:
            async def _tail() -> AsyncIterator[str]:
                replayed.append(room_id)
                yield json.dumps({"type": "editor.op", "room": room_id, "payload": "secret", "seq": 1})

            return _tail()

        pod = _Pod(monkeypatch, replay=_replay)
        stranger, socket = pod.person()
        pod.policy.refused.add((stranger, "room.join"))
        pod.connect(socket)
        socket.push({"type": "resume", "room": ROOM, "seq": 0})
        await pod.settle()

        assert replayed == []
        assert socket.frames("editor.op") == []
        assert [e["message"] for e in socket.frames("error")] == ["access denied: room.join"]
        await pod.hang_up_all(socket)

    async def test_a_refused_resume_on_connect_replays_nothing_and_stays_live(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        replayed: list[str] = []

        def _replay(room_id: str, from_seq: int) -> AsyncIterator[str]:
            async def _tail() -> AsyncIterator[str]:
                replayed.append(room_id)
                yield json.dumps({"type": "editor.op", "room": room_id, "payload": "secret", "seq": 1})

            return _tail()

        pod = _Pod(monkeypatch, replay=_replay)
        stranger, socket = pod.person()
        socket.query_params.update({"resume_room": ROOM, "resume_seq": "0"})
        pod.policy.refused.add((stranger, "room.join"))
        pod.connect(socket)
        await pod.settle()
        socket.push({"type": "message", "content": "still here", "metadata": {}})
        await pod.settle()

        assert replayed == []
        assert [r["content"] for r in socket.frames("response")] == ["echo: still here"]
        await pod.hang_up_all(socket)


class TestRevokingALiveMember:
    async def _two_members(self, pod: _Pod) -> tuple[str, _LiveSocket, str, _LiveSocket]:
        owner, owner_socket = pod.person()
        colleague, colleague_socket = pod.person()
        pod.connect(owner_socket)
        pod.connect(colleague_socket)
        await _join(pod, owner_socket)
        await _join(pod, colleague_socket)
        owner_socket.push({"type": "typing", "room": ROOM, "payload": "before"})
        await pod.settle()
        assert len(colleague_socket.frames("typing")) == 1
        return owner, owner_socket, colleague, colleague_socket

    async def test_revoke_takes_the_member_out_and_their_frames_stop(self, pod: _Pod) -> None:
        owner, owner_socket, colleague, colleague_socket = await self._two_members(pod)

        evicted = await pod.handler.revoke(ROOM, colleague)
        owner_socket.push({"type": "typing", "room": ROOM, "payload": "after the room went private"})
        colleague_socket.push({"type": "typing", "room": ROOM, "payload": "can I still write"})
        await pod.settle()

        assert evicted == 1
        assert [f["payload"] for f in colleague_socket.frames("typing")] == ["before"]
        assert owner_socket.frames("typing") == []
        errors = colleague_socket.frames("error")
        assert errors[0] == {"type": "error", "message": "access revoked", "room": ROOM}
        assert errors[1]["message"] == "not joined to room"
        await pod.hang_up_all(owner_socket, colleague_socket)
        colleague_leaves = [cid for room, cid in pod.rooms.left if room == ROOM]
        assert len(colleague_leaves) == len(set(colleague_leaves)) == 2

    async def test_revoke_keeps_other_members_and_a_rejoin_is_decided_afresh(self, pod: _Pod) -> None:
        owner, owner_socket, colleague, colleague_socket = await self._two_members(pod)

        await pod.handler.revoke(ROOM, colleague)
        colleague_socket.push({"type": "join", "room": ROOM})
        await pod.settle()
        colleague_socket.push({"type": "typing", "room": ROOM, "payload": "back"})
        await pod.settle()

        assert [f["payload"] for f in owner_socket.frames("typing")] == ["back"]
        colleague_joins = [r for r in pod.policy.requests if r.user_id == colleague and r.action == "room.join"]
        assert len(colleague_joins) == 2
        await pod.hang_up_all(owner_socket, colleague_socket)

    async def test_revoking_someone_not_in_the_room_evicts_nobody(self, pod: _Pod) -> None:
        owner, owner_socket, colleague, colleague_socket = await self._two_members(pod)

        assert await pod.handler.revoke(ROOM, str(uuid4())) == 0
        assert await pod.handler.revoke("some:other:room", colleague) == 0
        await pod.hang_up_all(owner_socket, colleague_socket)

    async def test_reevaluate_evicts_exactly_the_members_the_policy_now_refuses(self, pod: _Pod) -> None:
        owner, owner_socket, colleague, colleague_socket = await self._two_members(pod)
        pod.policy.refused.add((colleague, "room.join"))

        evicted = await pod.handler.reevaluate_room(ROOM)
        owner_socket.push({"type": "typing", "room": ROOM, "payload": "after"})
        await pod.settle()

        assert evicted == 1
        assert [f["payload"] for f in colleague_socket.frames("typing")] == ["before"]
        assert owner_socket.frames("error") == []
        await pod.hang_up_all(owner_socket, colleague_socket)

    async def test_reevaluate_evicts_when_the_policy_cannot_answer(self, pod: _Pod) -> None:
        owner, owner_socket, colleague, colleague_socket = await self._two_members(pod)
        pod.policy.unreachable = True

        assert await pod.handler.reevaluate_room(ROOM) == 2
        await pod.hang_up_all(owner_socket, colleague_socket)
