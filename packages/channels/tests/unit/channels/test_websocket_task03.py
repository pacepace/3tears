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
  the returned seq to ALL members (the author needs its own op echoed back
  with the seq — the OT ack); an ``OpRejected`` → ``error`` frame, no
  broadcast; transient ``cursor``/``typing``/``presence`` → broadcast with
  no seq, author-excluded; app-registered ``frame_handlers`` extend the
  router with the app's own types.
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
    async def test_editor_op_appends_then_broadcasts_with_seq_to_all(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``editor.op`` → op_handler (durable) → broadcast carrying its seq to ALL (author incl.).

        server-authoritative OT: the author needs its own op echoed back with
        the assigned seq (the ack), so the broadcast is NOT author-excluded.
        """
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
        # exactly one broadcast, carrying the op-log seq, to ALL members (no exclude)
        assert len(fanout.broadcasts) == 1
        b_room, b_payload, b_exclude = fanout.broadcasts[0]
        assert b_room == room
        assert b_exclude is None  # broadcast to everyone incl. the author (the OT ack)
        broadcast_frame = json.loads(b_payload)
        assert broadcast_frame["seq"] == 100
        # write authz ran for the op
        assert "entry.write" in [a for a, _ in recorded]

    @pytest.mark.asyncio
    async def test_editor_op_rejected_sends_error_and_does_not_broadcast(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """an ``OpRejected`` from the op_handler → ``error`` frame, NO broadcast, socket survives."""
        from threetears.channels.frames import OpRejected

        class _RejectingOpHandler:
            async def __call__(self, room_id: str, user_id: str, frame: Frame) -> OpResult:
                raise OpRejected("sequence-conflict")

        handler, state, fanout, recorded = _room_seam_handler(monkeypatch, op_handler=_RejectingOpHandler())
        handler._auth_validator = _auth_with_customer("valid-token")  # noqa: SLF001

        room = "cust:story:main:scene.md"
        msgs = [
            json.dumps({"type": "join", "room": room}),
            json.dumps({"type": "editor.op", "room": room, "payload": "op"}),
            json.dumps({"type": "editor.op", "room": room, "payload": "op2"}),  # socket still serving
        ]
        ws = MockWebSocket(messages=msgs, query_params={"token": "valid-token"})
        await handler.handle_connection(ws)  # must NOT raise

        errors = [json.loads(m) for m in ws.sent if json.loads(m).get("type") == "error"]
        assert len(errors) == 2, "each rejected op should error; the socket must keep serving"
        assert any("sequence-conflict" in e.get("message", "") for e in errors)
        assert fanout.broadcasts == [], "a rejected op must not broadcast"

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


# ============================================================
# app-registered frame handlers (extensibility seam)
# ============================================================


class TestAppFrameHandlers:
    """apps register handlers for their own frame types (e.g. scriob ``commit``)."""

    @pytest.mark.asyncio
    async def test_registered_app_frame_dispatched_with_context_and_can_reply(self) -> None:
        """a registered app frame reaches its handler with identity + a reply ``send``."""
        from threetears.channels.websocket import WebSocketHandler

        calls: list[tuple[str, str | None, str]] = []

        async def _commit(frame: Frame, *, user_id: str, customer_id: str, connection_id: str, send: Any) -> None:
            calls.append((frame.type, frame.room, user_id))
            await send(json.dumps({"type": "committed", "op_seq": 42}))

        handler = WebSocketHandler(
            router=_EchoRouter(),
            auth_validator=_valid_auth,
            frame_handlers={"commit": _commit},
        )
        ws = MockWebSocket(
            messages=[json.dumps({"type": "commit", "room": "cust:s:main:f.md"})],
            query_params={"token": "valid-token"},
        )
        await handler.handle_connection(ws)

        assert len(calls) == 1
        assert calls[0][0] == "commit"
        assert calls[0][1] == "cust:s:main:f.md"
        committed = [json.loads(m) for m in ws.sent if json.loads(m).get("type") == "committed"]
        assert committed and committed[0]["op_seq"] == 42

    @pytest.mark.asyncio
    async def test_unregistered_type_still_errors_not_dropped(self) -> None:
        """a type covered by NO app handler still yields an ``error`` (never a silent drop)."""
        from threetears.channels.websocket import WebSocketHandler

        async def _noop(frame: Frame, **_: Any) -> None: ...

        handler = WebSocketHandler(router=_EchoRouter(), auth_validator=_valid_auth, frame_handlers={"commit": _noop})
        ws = MockWebSocket(messages=[json.dumps({"type": "no-such"})], query_params={"token": "valid-token"})
        await handler.handle_connection(ws)

        errors = [json.loads(m) for m in ws.sent if json.loads(m).get("type") == "error"]
        assert len(errors) == 1

    def test_app_handler_cannot_shadow_a_builtin_frame_type(self) -> None:
        """registering a built-in type (e.g. ``editor.op``) is rejected — apps can't break core routing."""
        from threetears.channels.websocket import WebSocketHandler

        async def _h(frame: Frame, **_: Any) -> None: ...

        with pytest.raises(ValueError, match="reserved"):
            WebSocketHandler(
                router=_EchoRouter(),
                auth_validator=_valid_auth,
                frame_handlers={"editor.op": _h},
            )


# ============================================================
# the per-frame safety net: a handler's stray exception must NOT crash the socket
# ============================================================


class TestFrameDispatchIsCrashSafe:
    """an unexpected exception from any typed-frame handler → error frame, socket survives."""

    @pytest.mark.asyncio
    async def test_unexpected_exception_in_app_handler_does_not_crash_socket(self) -> None:
        """an app frame handler that raises an UNEXPECTED error → error frame, loop keeps serving."""
        from threetears.channels.websocket import WebSocketHandler

        async def _boom(frame: Frame, **_: Any) -> None:
            raise RuntimeError("kaboom from the app handler")

        handler = WebSocketHandler(router=_EchoRouter(), auth_validator=_valid_auth, frame_handlers={"commit": _boom})
        msgs = [json.dumps({"type": "commit", "room": "r"}), json.dumps({"type": "unknown-x"})]
        ws = MockWebSocket(messages=msgs, query_params={"token": "valid-token"})
        await handler.handle_connection(ws)  # must NOT raise

        errors = [json.loads(m) for m in ws.sent if json.loads(m).get("type") == "error"]
        assert len(errors) == 2, "the boom should error AND the later frame still serve (socket alive)"

    @pytest.mark.asyncio
    async def test_unexpected_exception_in_op_handler_does_not_crash_socket(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """a non-``OpRejected`` raise from the op_handler → error frame, no broadcast, socket survives."""

        class _BoomOpHandler:
            async def __call__(self, room_id: str, user_id: str, frame: Frame) -> OpResult:
                raise RuntimeError("unexpected op_handler fault")

        handler, state, fanout, recorded = _room_seam_handler(monkeypatch, op_handler=_BoomOpHandler())
        handler._auth_validator = _auth_with_customer("valid-token")  # noqa: SLF001

        room = "cust:story:main:scene.md"
        msgs = [
            json.dumps({"type": "join", "room": room}),
            json.dumps({"type": "editor.op", "room": room, "payload": "op"}),
            json.dumps({"type": "editor.op", "room": room, "payload": "op2"}),
        ]
        ws = MockWebSocket(messages=msgs, query_params={"token": "valid-token"})
        await handler.handle_connection(ws)  # must NOT raise

        errors = [json.loads(m) for m in ws.sent if json.loads(m).get("type") == "error"]
        assert len(errors) == 2, "each faulting op errors; the socket keeps serving"
        assert fanout.broadcasts == [], "a faulting op must not broadcast"


# ============================================================
# the chat-message safety net: a router failure on the plain (non-typed)
# ``message`` path must NOT crash the socket either
# ============================================================


class _BoomRouter:
    """router whose route_inbound always raises, to prove the chat path survives it."""

    async def route_inbound(self, message: ChannelMessage) -> ChannelResponse | None:
        raise RuntimeError("kaboom from the chat router")


class TestChatMessageDispatchIsCrashSafe:
    """an unexpected exception from the chat router -> error frame, socket survives.

    regression test: before this fix, the ``msg_type == "message"`` branch of
    ``_message_loop`` had no equivalent of ``TestFrameDispatchIsCrashSafe``'s
    per-frame safety net -- ``router.route_inbound`` raising (e.g. an unknown
    target agent) propagated all the way out of the message loop and crashed
    the whole ASGI connection (1011) instead of degrading gracefully like every
    typed frame already does.
    """

    @pytest.mark.asyncio
    async def test_router_exception_on_chat_message_does_not_crash_socket(self) -> None:
        """a chat router that raises -> one error frame, loop keeps serving the next message."""
        from threetears.channels.websocket import WebSocketHandler

        handler = WebSocketHandler(router=_BoomRouter(), auth_validator=_valid_auth)
        msgs = [
            json.dumps({"type": "message", "content": "hi"}),
            json.dumps({"type": "unknown-x"}),
        ]
        ws = MockWebSocket(messages=msgs, query_params={"token": "valid-token"})
        await handler.handle_connection(ws)  # must NOT raise

        errors = [json.loads(m) for m in ws.sent if json.loads(m).get("type") == "error"]
        assert len(errors) == 2, "the chat router's raise should error AND the later frame still serve"
        assert ws.close_code != 1011, "an unhandled router exception must not surface as an ASGI internal error"


# ============================================================
# the safety net's OWN send must survive a dead socket: every best-effort
# notification/error send in the message loop and its dispatch tree must
# degrade quietly, not crash, when the socket died between the triggering
# read and the reply (the exact disconnect-mid-turn window a long-running
# frame handler runs in). regression for the prod crash: 8950bae's own new
# error-frame send (now one of many sites here) had no such guard.
# ============================================================


class _DeadSendWebSocket(MockWebSocket):
    """a socket whose ``send_text`` raises on every call AFTER the initial ``connected`` frame.

    Lets ``handle_connection`` proceed past accept/auth normally (so a test can drive real
    message-loop behavior), then simulates the client having vanished for every reply the
    handler tries to send afterward -- the exact shape of a dead-socket-mid-turn disconnect.
    Tracks every call attempted (``send_attempts``), even though each one raises, so a test can
    assert how many replies were ATTEMPTED, not just that none crashed the loop.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.send_attempts = 0

    async def send_text(self, data: str) -> None:
        self.send_attempts += 1
        if not self.sent:
            await super().send_text(data)
            return
        raise RuntimeError("simulated dead socket: send failed")


class TestMessageLoopSurvivesDeadSocketOnReply:
    """every early-loop guard reply must degrade quietly against an already-dead socket."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "bad_message",
        [
            pytest.param("x" * 100_000, id="message-too-large"),
            pytest.param("not json", id="invalid-json"),
            pytest.param(json.dumps({"no-type": "here"}), id="invalid-frame"),
        ],
    )
    async def test_guard_reply_failure_does_not_crash_the_loop(self, bad_message: str) -> None:
        """a guard-rejected message whose error reply fails to send -> loop still serves the next one."""
        from threetears.channels.websocket import WebSocketHandler

        handler = WebSocketHandler(router=_EchoRouter(), auth_validator=_valid_auth)
        msgs = [bad_message, json.dumps({"type": "message", "content": "hi"})]
        ws = _DeadSendWebSocket(messages=msgs, query_params={"token": "valid-token"})
        await handler.handle_connection(ws)  # must NOT raise

        assert ws.send_attempts >= 2, "both the failed guard reply and the later message reply must be ATTEMPTED"

    @pytest.mark.asyncio
    async def test_rate_limit_reply_failure_does_not_crash_the_loop(self) -> None:
        """the rate-limit guard's reply failing must not crash the loop either (separate window/counter path)."""
        from threetears.channels.websocket import WebSocketHandler

        handler = WebSocketHandler(router=_EchoRouter(), auth_validator=_valid_auth, config={"rate_limit_messages": 1})
        msgs = [json.dumps({"type": "message", "content": "one"})] * 3
        ws = _DeadSendWebSocket(messages=msgs, query_params={"token": "valid-token"})
        await handler.handle_connection(ws)  # must NOT raise

        assert ws.send_attempts >= 3


class TestFrameDispatchSurvivesDeadSocketOnErrorNotify:
    """the typed-frame safety net's OWN error-frame send must survive a dead socket."""

    @pytest.mark.asyncio
    async def test_app_handler_exception_notify_failure_does_not_crash_socket(self) -> None:
        """an app handler raises, AND the resulting error-frame send also fails -> loop still survives."""
        from threetears.channels.websocket import WebSocketHandler

        async def _boom(frame: Frame, **_: Any) -> None:
            raise RuntimeError("kaboom from the app handler")

        handler = WebSocketHandler(router=_EchoRouter(), auth_validator=_valid_auth, frame_handlers={"commit": _boom})
        msgs = [json.dumps({"type": "commit", "room": "r"}), json.dumps({"type": "message", "content": "hi"})]
        ws = _DeadSendWebSocket(messages=msgs, query_params={"token": "valid-token"})
        await handler.handle_connection(ws)  # must NOT raise -- this is the exact prod crash shape

        assert ws.send_attempts >= 2

    @pytest.mark.asyncio
    async def test_unknown_frame_type_notify_failure_does_not_crash_socket(self) -> None:
        """the ``_route_frame`` fallback's own error send failing must not crash the socket."""
        from threetears.channels.websocket import WebSocketHandler

        handler = WebSocketHandler(router=_EchoRouter(), auth_validator=_valid_auth)
        msgs = [json.dumps({"type": "no-such-type"}), json.dumps({"type": "message", "content": "hi"})]
        ws = _DeadSendWebSocket(messages=msgs, query_params={"token": "valid-token"})
        await handler.handle_connection(ws)  # must NOT raise

        assert ws.send_attempts >= 2

    @pytest.mark.asyncio
    async def test_app_handler_raw_send_callback_failure_propagates_and_does_not_crash_socket(self) -> None:
        """an app handler's OWN ``send(...)`` call against a dead socket, uncaught by the handler
        itself, must still degrade gracefully via the outer per-frame safety net -- proving the
        raw (unwrapped) ``send`` callback handed to app handlers is connection-safe even when the
        handler does not guard its own send calls (design note on the app-handler dispatch site:
        the callback is deliberately raw so a handler CAN observe a failed send itself, e.g. to
        stop its own work early; this proves the worst case -- a handler that does not -- still
        can't crash the connection)."""
        from threetears.channels.websocket import WebSocketHandler

        async def _uses_raw_send(frame: Frame, *, send: Any, **_: Any) -> None:
            await send(json.dumps({"type": "app-reply"}))  # not wrapped in its own try/except

        handler = WebSocketHandler(
            router=_EchoRouter(), auth_validator=_valid_auth, frame_handlers={"commit": _uses_raw_send}
        )
        msgs = [json.dumps({"type": "commit", "room": "r"}), json.dumps({"type": "message", "content": "hi"})]
        ws = _DeadSendWebSocket(messages=msgs, query_params={"token": "valid-token"})
        await handler.handle_connection(ws)  # must NOT raise

        assert ws.send_attempts >= 2


class TestChatMessageDispatchSurvivesDeadSocketOnErrorNotify:
    """regression for the actual observed prod crash: 8950bae's own new error-frame send
    (the chat-message safety net) had no guard against an already-dead socket -- this is the
    exact unhandled exception seen in the.ranch's logs during a disconnect-mid-turn window."""

    @pytest.mark.asyncio
    async def test_chat_router_exception_notify_failure_does_not_crash_socket(self) -> None:
        """the chat router raises, AND the resulting error-frame send ALSO fails -> socket survives."""
        from threetears.channels.websocket import WebSocketHandler

        handler = WebSocketHandler(router=_BoomRouter(), auth_validator=_valid_auth)
        msgs = [json.dumps({"type": "message", "content": "hi"}), json.dumps({"type": "message", "content": "again"})]
        ws = _DeadSendWebSocket(messages=msgs, query_params={"token": "valid-token"})
        await handler.handle_connection(ws)  # must NOT raise -- this is THE prod crash shape

        assert ws.send_attempts >= 2


class TestNestedRoomHandlerSendsSurviveDeadSocket:
    """error replies deep in the typed-frame dispatch tree (join/editor.op/transient/authz)
    must all degrade quietly against a dead socket -- not just the two outer safety nets."""

    @pytest.mark.asyncio
    async def test_join_without_room_reply_failure_does_not_crash_socket(self, monkeypatch: pytest.MonkeyPatch) -> None:
        handler, _state, _fanout, _recorded = _room_seam_handler(monkeypatch)
        msgs = [json.dumps({"type": "join"}), json.dumps({"type": "message", "content": "hi"})]
        ws = _DeadSendWebSocket(messages=msgs, query_params={"token": "valid-token"})
        await handler.handle_connection(ws)  # must NOT raise

        assert ws.send_attempts >= 2

    @pytest.mark.asyncio
    async def test_editor_op_without_join_reply_failure_does_not_crash_socket(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        handler, _state, _fanout, _recorded = _room_seam_handler(monkeypatch, op_handler=_FakeOpHandler(start=0))
        handler._auth_validator = _auth_with_customer("valid-token")  # noqa: SLF001
        room = "cust:story:main:scene.md"
        msgs = [
            json.dumps({"type": "editor.op", "room": room, "payload": "op"}),
            json.dumps({"type": "message", "content": "hi"}),
        ]
        ws = _DeadSendWebSocket(messages=msgs, query_params={"token": "valid-token"})
        await handler.handle_connection(ws)  # must NOT raise

        assert ws.send_attempts >= 2

    @pytest.mark.asyncio
    async def test_transient_frame_without_join_reply_failure_does_not_crash_socket(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        handler, _state, _fanout, _recorded = _room_seam_handler(monkeypatch)
        handler._auth_validator = _auth_with_customer("valid-token")  # noqa: SLF001
        room = "cust:story:main:scene.md"
        msgs = [
            json.dumps({"type": "cursor", "room": room, "payload": "{}"}),
            json.dumps({"type": "message", "content": "hi"}),
        ]
        ws = _DeadSendWebSocket(messages=msgs, query_params={"token": "valid-token"})
        await handler.handle_connection(ws)  # must NOT raise

        assert ws.send_attempts >= 2


class TestReplaySendSurvivesDeadSocket:
    """the resume/replay tail's per-payload send must degrade quietly against a dead socket,
    and stop attempting further payloads once the socket is known dead (not send N more times
    into a wire that already failed once -- best-effort, not best-effort-repeated-forever)."""

    @pytest.mark.asyncio
    async def test_resume_replay_failure_does_not_crash_and_stops_early(self) -> None:
        from threetears.channels.websocket import WebSocketHandler

        yielded = 0

        async def _replay(room_id: str, from_seq: int) -> Any:
            nonlocal yielded
            for seq in range(from_seq + 1, from_seq + 4):  # would yield 3 payloads if fully consumed
                yielded += 1
                yield json.dumps({"type": "editor.op", "room": room_id, "seq": seq})

        handler = WebSocketHandler(
            router=_EchoRouter(),
            auth_validator=_valid_auth,
            replay_source=_replay,  # type: ignore[arg-type]
        )
        room = "cust:story:main:scene.md"
        resume = json.dumps({"type": "resume", "room": room, "seq": 7})
        msgs = [resume, json.dumps({"type": "message", "content": "hi"})]
        ws = _DeadSendWebSocket(messages=msgs, query_params={"token": "valid-token"})
        attempts_before = ws.send_attempts
        await handler.handle_connection(ws)  # must NOT raise

        assert yielded == 1, "the replay loop should stop consuming the source after its first failed send"
        # exactly one send attempt for the (failed) replay payload; the connection then keeps
        # serving the next message (proven by the later message's own send attempt(s) happening
        # too, not just the connection staying alive with no further activity).
        assert ws.send_attempts > attempts_before + 1, "the connection must keep serving after the replay send fails"


# ============================================================
# membership gate: editor.op / transient require a prior join
# ============================================================


class TestEditRequiresJoin:
    """editing/broadcasting to a room requires having joined it (no silent no-ack)."""

    @pytest.mark.asyncio
    async def test_editor_op_before_join_is_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """an ``editor.op`` for a room the connection never joined → error, no op, no broadcast."""
        op = _FakeOpHandler(start=100)
        handler, state, fanout, recorded = _room_seam_handler(monkeypatch, op_handler=op)
        handler._auth_validator = _auth_with_customer("valid-token")  # noqa: SLF001

        room = "cust:story:main:scene.md"
        # editor.op WITHOUT a preceding join.
        ws = MockWebSocket(
            messages=[json.dumps({"type": "editor.op", "room": room, "payload": "op"})],
            query_params={"token": "valid-token"},
        )
        await handler.handle_connection(ws)

        errors = [json.loads(m) for m in ws.sent if json.loads(m).get("type") == "error"]
        assert len(errors) == 1
        assert op.appended == [], "an op for an unjoined room must not be appended"
        assert fanout.broadcasts == [], "an op for an unjoined room must not broadcast"

    @pytest.mark.asyncio
    async def test_editor_op_after_join_is_accepted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """a ``join`` then ``editor.op`` for that room is accepted (the gate allows members)."""
        op = _FakeOpHandler(start=100)
        handler, state, fanout, recorded = _room_seam_handler(monkeypatch, op_handler=op)
        handler._auth_validator = _auth_with_customer("valid-token")  # noqa: SLF001

        room = "cust:story:main:scene.md"
        msgs = [
            json.dumps({"type": "join", "room": room}),
            json.dumps({"type": "editor.op", "room": room, "payload": "op"}),
        ]
        ws = MockWebSocket(messages=msgs, query_params={"token": "valid-token"})
        await handler.handle_connection(ws)

        assert len(op.appended) == 1, "a joined member's op is appended"
        assert len(fanout.broadcasts) == 1
