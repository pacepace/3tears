"""channels-task-03 unit proofs: typed-frame routing, authz gates, resume, no-seam==chat.

These exercise the cross-pod WebSocketHandler with **fakes** for every
injected seam (room_state / room_fanout / acl authorizer / op_handler /
replay_source). The real-NATS vertical slice + authz + resume + cleanup
proofs live in ``tests/integration/channels/``.

Contracts asserted here (design T3-D1..D5):

- with **no** room seams, a ``message`` frame routes through the existing
  chat router byte-identically, and an **unknown** frame type now gets an
  ``error`` frame (the pre-task-03 silent ``continue`` is gone).
- each typed frame hits the right path: ``join``/``leave`` → fanout (after
  ``room.join`` authz); ``editor.op`` → op_handler then broadcast carrying
  the returned seq, ``exclude`` = the author's connection id; transient
  ``cursor``/``typing``/``presence`` → broadcast with no seq.
- a denied ``join`` (authorizer raising ``AccessDenied``) → ``error`` frame,
  NO ``join_room``, NO broadcast; a denied ``editor.op`` → refused.
- enforcement: no in-process seq on the handler; unknown-not-dropped;
  no-seam dispatch == the chat path.
"""

from __future__ import annotations

import ast
import inspect
import json
from typing import Any
from uuid import UUID, uuid4

import pytest

from threetears.agent.acl import AccessDenied
from threetears.channels.frames import Frame, OpResult

from .test_websocket import MockWebSocket, _EchoRouter, _valid_auth


# -- fakes for the injected seams ------------------------------------------


class _FakeRoomState:
    """records register/unregister + the rooms left, no real collection."""

    def __init__(self) -> None:
        self.registered: list[tuple[str, Any]] = []
        self.unregistered: list[str] = []

    async def register(self, connection_id: str, socket: Any) -> None:
        self.registered.append((connection_id, socket))

    async def unregister(self, connection_id: str, socket: Any | None = None) -> None:
        self.unregistered.append(connection_id)


class _FakeFanout:
    """records join/leave/broadcast calls so dispatch can be asserted."""

    def __init__(self) -> None:
        self.joined: list[tuple[str, str, str, str]] = []
        self.left: list[tuple[str, str]] = []
        self.broadcasts: list[tuple[str, str, str | None]] = []

    async def join_room(self, room_id: str, connection_id: str, user_id: str, customer_id: str) -> None:
        self.joined.append((room_id, connection_id, user_id, customer_id))

    async def leave_room(self, room_id: str, connection_id: str) -> None:
        self.left.append((room_id, connection_id))

    async def broadcast(self, room_id: str, payload: str, *, exclude: str | None = None) -> None:
        self.broadcasts.append((room_id, payload, exclude))


class _AllowAuthorizer:
    """ns_resolver + acl stand-in: records every authz call, always allows.

    The handler is injected an ``acl_cache`` + ``ns_resolver``; this fake
    plays the resolver and, paired with ``_patch_authorize`` below, the
    allow decision. It records ``(action, user_id)`` so a test can assert
    the gate ran with the border-converted UUID.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, UUID | None]] = []
        self.ns = _Ns()

    async def __call__(self, room_id: str) -> Any:
        return self.ns


class _Ns:
    id = uuid4()
    customer_id = uuid4()
    namespace_type = "story"
    owner_agent_id = uuid4()


class _FakeOpHandler:
    """returns incrementing op-log seqs and records the frames it appended."""

    def __init__(self, start: int = 100) -> None:
        self._next = start
        self.appended: list[tuple[str, str, Frame]] = []

    async def __call__(self, room_id: str, user_id: str, frame: Frame) -> OpResult:
        self.appended.append((room_id, user_id, frame))
        seq = self._next
        self._next += 1
        return OpResult(seq=seq)


def _capture_authz(monkeypatch: pytest.MonkeyPatch, *, deny_actions: set[str] | None = None) -> list[tuple[str, Any]]:
    """patch ``authorize_on_entity`` (as imported into websocket.py) to record + allow/deny.

    Returns the recording list of ``(action, user_id)``. ``deny_actions``
    is the set of action strings that should raise ``AccessDenied``.
    """
    deny = deny_actions or set()
    recorded: list[tuple[str, Any]] = []

    async def _fake(
        *, ns_entity: Any, action: str, user_id: Any, agent_id: Any, cache: Any, namespace_name: Any = None
    ):  # noqa: ANN202
        recorded.append((action, user_id))
        if action in deny:
            raise AccessDenied("denied", action=action)
        return object()

    import threetears.channels.websocket as ws_mod

    monkeypatch.setattr(ws_mod, "authorize_on_entity", _fake)
    return recorded


def _room_seam_handler(
    monkeypatch: pytest.MonkeyPatch,
    *,
    deny_actions: set[str] | None = None,
    op_handler: _FakeOpHandler | None = None,
) -> tuple[Any, _FakeRoomState, _FakeFanout, list[tuple[str, Any]]]:
    """build a WebSocketHandler with all room seams injected + patched authz."""
    from threetears.channels.websocket import WebSocketHandler

    state = _FakeRoomState()
    fanout = _FakeFanout()
    recorded = _capture_authz(monkeypatch, deny_actions=deny_actions)
    handler = WebSocketHandler(
        router=_EchoRouter(),
        auth_validator=_valid_auth,
        room_state=state,
        room_fanout=fanout,
        acl_cache=object(),
        ns_resolver=_AllowAuthorizer(),
        op_handler=op_handler,
    )
    return handler, state, fanout, recorded


def _auth_with_customer(token: str) -> Any:
    """auth validator returning a user_id + customer_id as UUID strings."""

    async def _v(tok: str) -> dict[str, Any] | None:
        if tok == "valid-token":
            return {"user_id": str(uuid4()), "customer_id": str(uuid4())}
        return None

    return _v


# ============================================================
# no-seam == chat path (regression)
# ============================================================


class TestNoSeamIsChatPath:
    """with no room seams, message routing is byte-identical to pre-task-03."""

    @pytest.mark.asyncio
    async def test_message_routes_through_chat_router(self) -> None:
        """a ``message`` frame hits the chat router and echoes, unchanged."""
        from threetears.channels.websocket import WebSocketHandler

        handler = WebSocketHandler(router=_EchoRouter(), auth_validator=_valid_auth)
        user_msg = json.dumps({"type": "message", "content": "hello", "metadata": {}})
        ws = MockWebSocket(messages=[user_msg], query_params={"token": "valid-token"})
        await handler.handle_connection(ws)

        response = json.loads(ws.sent[1])
        assert response["type"] == "response"
        assert response["content"] == "echo: hello"

    @pytest.mark.asyncio
    async def test_unknown_frame_type_gets_error_not_silent_drop(self) -> None:
        """an unknown type now yields an ``error`` frame (old silent continue is gone)."""
        from threetears.channels.websocket import WebSocketHandler

        handler = WebSocketHandler(router=_EchoRouter(), auth_validator=_valid_auth)
        weird = json.dumps({"type": "totally-unknown", "content": "x"})
        ws = MockWebSocket(messages=[weird], query_params={"token": "valid-token"})
        await handler.handle_connection(ws)

        errors = [json.loads(m) for m in ws.sent if json.loads(m).get("type") == "error"]
        assert len(errors) == 1
        assert "totally-unknown" in errors[0]["message"] or "unknown" in errors[0]["message"].lower()

    @pytest.mark.asyncio
    async def test_frame_missing_type_errors_does_not_crash(self) -> None:
        """valid JSON with no ``type`` → an ``error`` frame, never a crash/silent drop.

        Pre-task-03 a typeless frame defaulted to ``""`` and silently
        continued. Now ``Frame`` requires ``type``; the loop must surface
        the malformed frame as an error and keep serving (the next frame
        still routes).
        """
        from threetears.channels.websocket import WebSocketHandler

        handler = WebSocketHandler(router=_EchoRouter(), auth_validator=_valid_auth)
        typeless = json.dumps({"content": "no type here"})
        good = json.dumps({"type": "message", "content": "hi", "metadata": {}})
        ws = MockWebSocket(messages=[typeless, good], query_params={"token": "valid-token"})
        await handler.handle_connection(ws)

        errors = [json.loads(m) for m in ws.sent if json.loads(m).get("type") == "error"]
        responses = [json.loads(m) for m in ws.sent if json.loads(m).get("type") == "response"]
        assert len(errors) == 1  # the typeless frame errored
        assert len(responses) == 1  # the subsequent good frame still routed
        assert responses[0]["content"] == "echo: hi"


# ============================================================
# typed routing
# ============================================================


class TestTypedRouting:
    """each frame type reaches the correct path."""

    @pytest.mark.asyncio
    async def test_join_authorizes_then_joins_room(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``join`` runs ``room.join`` authz then calls fanout.join_room."""
        handler, state, fanout, recorded = _room_seam_handler(monkeypatch)
        handler._auth_validator = _auth_with_customer("valid-token")  # noqa: SLF001

        room = "cust:story:main:scene.md"
        join_msg = json.dumps({"type": "join", "room": room})
        ws = MockWebSocket(messages=[join_msg], query_params={"token": "valid-token"})
        await handler.handle_connection(ws)

        assert [a for a, _ in recorded] == ["room.join"]
        assert len(fanout.joined) == 1
        assert fanout.joined[0][0] == room

    @pytest.mark.asyncio
    async def test_leave_leaves_room(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``leave`` calls fanout.leave_room for the joined room."""
        handler, state, fanout, recorded = _room_seam_handler(monkeypatch)
        handler._auth_validator = _auth_with_customer("valid-token")  # noqa: SLF001

        room = "cust:story:main:scene.md"
        msgs = [json.dumps({"type": "join", "room": room}), json.dumps({"type": "leave", "room": room})]
        ws = MockWebSocket(messages=msgs, query_params={"token": "valid-token"})
        await handler.handle_connection(ws)

        assert len(fanout.left) == 1
        assert fanout.left[0][0] == room

    @pytest.mark.asyncio
    async def test_editor_op_appends_then_broadcasts_with_seq(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``editor.op`` → op_handler (durable) → broadcast carrying its seq, excluding author."""
        op = _FakeOpHandler(start=100)
        handler, state, fanout, recorded = _room_seam_handler(monkeypatch, op_handler=op)
        handler._auth_validator = _auth_with_customer("valid-token")  # noqa: SLF001

        room = "cust:story:main:scene.md"
        msgs = [
            json.dumps({"type": "join", "room": room}),
            json.dumps({"type": "editor.op", "room": room, "payload": "replace(1,2,'x')"}),
        ]
        ws = MockWebSocket(messages=msgs, query_params={"token": "valid-token"})
        await handler.handle_connection(ws)

        # op handler was called once
        assert len(op.appended) == 1
        # exactly one broadcast, carrying the op-log seq, excluding the author's connection id
        assert len(fanout.broadcasts) == 1
        b_room, b_payload, b_exclude = fanout.broadcasts[0]
        assert b_room == room
        assert b_exclude is not None  # the author's connection id
        broadcast_frame = json.loads(b_payload)
        assert broadcast_frame["seq"] == 100
        # write authz ran for the op
        assert "entry.write" in [a for a, _ in recorded]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("ttype", ["cursor", "typing", "presence"])
    async def test_transient_frames_broadcast_without_seq(self, monkeypatch: pytest.MonkeyPatch, ttype: str) -> None:
        """``cursor``/``typing``/``presence`` transient-broadcast with NO seq, no op_handler."""
        handler, state, fanout, recorded = _room_seam_handler(monkeypatch)
        handler._auth_validator = _auth_with_customer("valid-token")  # noqa: SLF001

        room = "cust:story:main:scene.md"
        msgs = [
            json.dumps({"type": "join", "room": room}),
            json.dumps({"type": ttype, "room": room, "payload": "blink"}),
        ]
        ws = MockWebSocket(messages=msgs, query_params={"token": "valid-token"})
        await handler.handle_connection(ws)

        assert len(fanout.broadcasts) == 1
        _, b_payload, b_exclude = fanout.broadcasts[0]
        assert b_exclude is not None
        frame = json.loads(b_payload)
        assert frame.get("seq") is None
        assert "entry.write" in [a for a, _ in recorded]


# ============================================================
# authorization gates
# ============================================================


class TestAuthzGates:
    """a denial yields an error frame and stops the side effect."""

    @pytest.mark.asyncio
    async def test_denied_join_errors_no_join_no_broadcast(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``room.join`` deny → error frame, NO join_room, NO broadcast."""
        handler, state, fanout, recorded = _room_seam_handler(monkeypatch, deny_actions={"room.join"})
        handler._auth_validator = _auth_with_customer("valid-token")  # noqa: SLF001

        room = "cust:story:main:scene.md"
        ws = MockWebSocket(messages=[json.dumps({"type": "join", "room": room})], query_params={"token": "valid-token"})
        await handler.handle_connection(ws)

        errors = [json.loads(m) for m in ws.sent if json.loads(m).get("type") == "error"]
        assert len(errors) == 1
        assert fanout.joined == []
        assert fanout.broadcasts == []

    @pytest.mark.asyncio
    async def test_denied_editor_op_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``entry.write`` deny on ``editor.op`` → error, op_handler NOT called, no broadcast."""
        op = _FakeOpHandler()
        handler, state, fanout, recorded = _room_seam_handler(monkeypatch, deny_actions={"entry.write"}, op_handler=op)
        handler._auth_validator = _auth_with_customer("valid-token")  # noqa: SLF001

        room = "cust:story:main:scene.md"
        msgs = [
            json.dumps({"type": "join", "room": room}),
            json.dumps({"type": "editor.op", "room": room, "payload": "op"}),
        ]
        ws = MockWebSocket(messages=msgs, query_params={"token": "valid-token"})
        await handler.handle_connection(ws)

        errors = [json.loads(m) for m in ws.sent if json.loads(m).get("type") == "error"]
        assert len(errors) >= 1
        assert op.appended == []
        assert fanout.broadcasts == []

    @pytest.mark.asyncio
    async def test_authz_borders_user_id_to_uuid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """the authz gate receives a real ``UUID`` (str→UUID border-conversion)."""
        handler, state, fanout, recorded = _room_seam_handler(monkeypatch)
        handler._auth_validator = _auth_with_customer("valid-token")  # noqa: SLF001

        room = "cust:story:main:scene.md"
        ws = MockWebSocket(messages=[json.dumps({"type": "join", "room": room})], query_params={"token": "valid-token"})
        await handler.handle_connection(ws)

        assert recorded
        _, passed_user_id = recorded[0]
        assert isinstance(passed_user_id, UUID)


# ============================================================
# disconnect cleanup
# ============================================================


class TestDisconnectCleanup:
    """dropping the socket leaves joined rooms + unregisters."""

    @pytest.mark.asyncio
    async def test_disconnect_leaves_joined_rooms_and_unregisters(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """after the loop ends, every joined room is left and the handle unregistered."""
        handler, state, fanout, recorded = _room_seam_handler(monkeypatch)
        handler._auth_validator = _auth_with_customer("valid-token")  # noqa: SLF001

        room = "cust:story:main:scene.md"
        # join, then the socket drops (no leave frame sent)
        ws = MockWebSocket(messages=[json.dumps({"type": "join", "room": room})], query_params={"token": "valid-token"})
        await handler.handle_connection(ws)

        # the connection's joined room was left on disconnect
        assert [r for r, _ in fanout.left] == [room]
        # and the live handle was unregistered
        assert len(state.unregistered) == 1
        assert len(state.registered) == 1
        assert state.unregistered[0] == state.registered[0][0]


# ============================================================
# enforcement
# ============================================================


class TestEnforcement:
    """structural guards: no in-process seq, unknown-not-dropped."""

    def test_handler_keeps_no_in_process_seq_counter(self) -> None:
        """no ``seq`` counter field/attr is assigned anywhere in websocket.py.

        The resume cursor is the durable op-log seq (design T3-D4); the
        handler must never mint or increment a sequence. Asserts no
        ``self.<name>seq...`` counter assignment exists in the source.
        """
        from threetears.channels import websocket as ws_mod

        source = inspect.getsource(ws_mod)
        tree = ast.parse(source)
        offenders: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store):
                name = node.attr.lower()
                if "seq" in name or name in {"_counter", "cursor"}:
                    offenders.append(node.attr)
        assert offenders == [], f"websocket.py assigns an in-process seq/cursor counter: {offenders}"

    @pytest.mark.asyncio
    async def test_unknown_frame_not_dropped_even_with_seams(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """an unknown type errors (never silently continues) even with seams wired."""
        handler, state, fanout, recorded = _room_seam_handler(monkeypatch)
        handler._auth_validator = _auth_with_customer("valid-token")  # noqa: SLF001

        ws = MockWebSocket(
            messages=[json.dumps({"type": "no-such-type", "room": "r"})],
            query_params={"token": "valid-token"},
        )
        await handler.handle_connection(ws)

        errors = [json.loads(m) for m in ws.sent if json.loads(m).get("type") == "error"]
        assert len(errors) == 1
        assert fanout.joined == []
        assert fanout.broadcasts == []


# ============================================================
# constructor invariant: authz is all-or-nothing
# ============================================================


class TestAuthzWiringInvariant:
    """a half-wired authz config must be rejected, not silently allow-all."""

    @pytest.mark.parametrize(
        ("acl_cache", "ns_resolver"),
        [(object(), None), (None, object())],
    )
    def test_half_wired_authz_raises(self, acl_cache: Any, ns_resolver: Any) -> None:
        """exactly one of acl_cache / ns_resolver wired → ValueError (no allow-all hole)."""
        from threetears.channels.websocket import WebSocketHandler

        with pytest.raises(ValueError, match="acl_cache and ns_resolver"):
            WebSocketHandler(
                router=_EchoRouter(),
                auth_validator=_valid_auth,
                acl_cache=acl_cache,
                ns_resolver=ns_resolver,  # type: ignore[arg-type]
            )

    def test_both_absent_is_allowed_chat_config(self) -> None:
        """no authz seams → the deliberate chat config, constructs fine."""
        from threetears.channels.websocket import WebSocketHandler

        WebSocketHandler(router=_EchoRouter(), auth_validator=_valid_auth)

    def test_both_present_is_allowed(self) -> None:
        """both authz seams → an authorized config, constructs fine."""
        from threetears.channels.websocket import WebSocketHandler

        WebSocketHandler(
            router=_EchoRouter(),
            auth_validator=_valid_auth,
            acl_cache=object(),  # type: ignore[arg-type]
            ns_resolver=_AllowAuthorizer(),  # type: ignore[arg-type]
        )


# ============================================================
# chat path stays byte-identical (strict typing applies only to typed frames)
# ============================================================


class TestChatPathUnaffectedByTypedFields:
    """a ``message`` frame is read loosely — typed-frame field types do not constrain it."""

    @pytest.mark.asyncio
    async def test_message_with_typed_fields_of_wrong_type_still_routes(self) -> None:
        """a ``message`` carrying room/seq/payload of 'wrong' types still echoes (not rejected).

        the typed ``Frame`` is strict about ``room: str|None`` / ``seq: int|None`` /
        ``payload: str|None``, but the chat ``message`` path must NOT be — it reads
        ``content`` loosely and never validates those fields. a legacy chat frame
        that happens to carry those keys with other shapes is handled exactly as
        pre-task-03.
        """
        from threetears.channels.websocket import WebSocketHandler

        handler = WebSocketHandler(router=_EchoRouter(), auth_validator=_valid_auth)
        # room as int, payload as object, seq as a non-numeric string — all would
        # fail strict Frame validation, but this is a chat ``message``.
        msg = json.dumps(
            {"type": "message", "content": "hi", "room": 123, "payload": {"k": 1}, "seq": "x", "metadata": {}}
        )
        ws = MockWebSocket(messages=[msg], query_params={"token": "valid-token"})
        await handler.handle_connection(ws)

        errors = [json.loads(m) for m in ws.sent if json.loads(m).get("type") == "error"]
        responses = [json.loads(m) for m in ws.sent if json.loads(m).get("type") == "response"]
        assert errors == [], "a chat message was rejected by typed-frame validation"
        assert len(responses) == 1
        assert responses[0]["content"] == "echo: hi"


# ============================================================
# malformed principal must deny, never crash the socket
# ============================================================


class TestMalformedPrincipal:
    """a non-UUID user_id at an authz gate denies gracefully (no uncaught crash)."""

    @pytest.mark.asyncio
    async def test_non_uuid_user_id_denies_and_does_not_crash(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """auth payload with a non-UUID ``user_id`` → join denied via error frame, loop survives.

        ``user_id`` comes from the host ``auth_validator``; a malformed value
        must surface as a denial (the str→UUID border conversion is defended),
        NOT raise ``ValueError`` out of the message loop and kill the socket.
        """
        handler, state, fanout, recorded = _room_seam_handler(monkeypatch)

        async def _bad_auth(token: str) -> dict[str, Any] | None:
            if token == "valid-token":
                return {"user_id": "not-a-uuid", "customer_id": str(uuid4())}
            return None

        handler._auth_validator = _bad_auth  # noqa: SLF001

        room = "cust:story:main:scene.md"
        # a join (which would crash via UUID()) then a second frame that must still be served.
        msgs = [json.dumps({"type": "join", "room": room}), json.dumps({"type": "unknown-x", "room": room})]
        ws = MockWebSocket(messages=msgs, query_params={"token": "valid-token"})
        await handler.handle_connection(ws)  # must NOT raise

        errors = [json.loads(m) for m in ws.sent if json.loads(m).get("type") == "error"]
        assert len(errors) == 2, "expected a denial for the bad-principal join AND the later unknown frame"
        assert fanout.joined == [], "a malformed-principal join must not write membership"
        assert recorded == [], "authorize_on_entity must not be reached with a bad principal"


# ============================================================
# in-loop resume frame
# ============================================================


class TestResumeFrame:
    """a ``resume`` frame streams the durable replay tail to the socket."""

    @pytest.mark.asyncio
    async def test_resume_frame_streams_replay_tail(self) -> None:
        """a ``resume`` frame replays ``replay_source(room, seq)`` to the socket, in order."""
        from threetears.channels.websocket import WebSocketHandler

        async def _replay(room_id: str, from_seq: int) -> Any:
            for seq in range(from_seq + 1, from_seq + 4):  # N+1, N+2, N+3
                yield json.dumps({"type": "editor.op", "room": room_id, "seq": seq})

        handler = WebSocketHandler(
            router=_EchoRouter(),
            auth_validator=_valid_auth,
            replay_source=_replay,  # type: ignore[arg-type]
        )
        room = "cust:story:main:scene.md"
        resume = json.dumps({"type": "resume", "room": room, "seq": 7})
        ws = MockWebSocket(messages=[resume], query_params={"token": "valid-token"})
        await handler.handle_connection(ws)

        replayed = [json.loads(m) for m in ws.sent if json.loads(m).get("type") == "editor.op"]
        assert [f["seq"] for f in replayed] == [8, 9, 10], "resume frame did not replay the tail in order"
