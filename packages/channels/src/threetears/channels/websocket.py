"""framework-agnostic websocket handler for channel adapters.

provides WebSocketHandler for managing websocket connection lifecycle
including authentication, message routing, and optional streaming.
ConnectionRegistry is the pod-local, **synchronized** map of live
``user_id → socket`` handles the handler keeps for its connections. all
websocket interaction goes through WebSocketProtocol so the handler
works with starlette, fastapi, or any conforming object.

channels-task-01 superseded the previous racy, dict-based
``ConnectionRegistry`` — two un-synchronized in-process dicts
(``_connections`` + ``_rooms``) whose ``broadcast_to_room`` iterated a
room's member list *with ``await`` inside the loop* while
``join_room`` / ``leave_room`` mutated the same dict (an
iterate-while-mutate race) and which could only see members on its own
pod. cross-pod room **membership/presence** now lives in the
concurrency-safe, L1+L2 ``PresenceCollection`` and is reshaped through
:class:`~threetears.channels.presence.room_state.RoomState` (which owns
the ``connection_id → live socket`` map and snapshot-iterates it); the
cross-pod room **message fanout** is channels-task-02. what remains here
is only the handler's own live-handle bookkeeping, kept correct under
concurrency by a lock rather than left as a bare racing dict.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable
from uuid import UUID, uuid7

from pydantic import ValidationError

from threetears.agent.acl import AccessDenied, authorize_on_entity
from threetears.channels.frames import Frame, OpRejected
from threetears.channels.protocol import ChannelMessage, ChannelResponse
from threetears.observe import get_logger

if TYPE_CHECKING:
    from threetears.agent.acl import AclCache
    from threetears.channels.frames import FrameHandler, NsResolver, OpHandler, ReplaySource
    from threetears.channels.presence.fanout import RoomFanout
    from threetears.channels.presence.room_state import RoomState

__all__ = [
    "ConnectionRegistry",
    "StreamingChannelRouter",
    "WebSocketHandler",
    "WebSocketProtocol",
]

log = get_logger(__name__)

_DEFAULT_HEARTBEAT_INTERVAL = 30
_DEFAULT_MAX_MESSAGE_SIZE = 65536  # 64KB
_DEFAULT_RATE_LIMIT_MESSAGES = 10
_DEFAULT_RATE_LIMIT_WINDOW = 1.0  # seconds
_DEFAULT_JOIN_ACTION = "room.join"
_DEFAULT_WRITE_ACTION = "entry.write"

# the typed frame vocabulary the cross-pod router dispatches on (T3-D2).
_TRANSIENT_FRAME_TYPES = frozenset({"cursor", "typing", "presence"})

# the built-in frame types the router owns; an app may NOT register a
# handler for one of these (it would shadow core routing). app-specific
# frame types (e.g. scriob ``commit``) go through ``frame_handlers``.
_BUILTIN_FRAME_TYPES = frozenset({"message", "join", "leave", "editor.op", "resume", *_TRANSIENT_FRAME_TYPES})


@runtime_checkable
class WebSocketProtocol(Protocol):
    """protocol defining websocket interface accepted by handler.

    any websocket object implementing accept, receive_text, send_text,
    and close satisfies this protocol. both starlette and fastapi
    websocket objects conform without adaptation.
    """

    async def accept(self) -> None:
        """accept incoming websocket connection.

        :return: None
        :rtype: None
        """
        ...

    async def receive_text(self) -> str:
        """receive text message from websocket client.

        :return: text message from client
        :rtype: str
        :raises Exception: on client disconnect
        """
        ...

    async def send_text(self, data: str) -> None:
        """send text message to websocket client.

        :param data: text message to send
        :ptype data: str
        :return: None
        :rtype: None
        """
        ...

    async def close(self, code: int = 1000) -> None:
        """close websocket connection.

        :param code: websocket close status code
        :ptype code: int
        :return: None
        :rtype: None
        """
        ...


@runtime_checkable
class StreamingChannelRouter(Protocol):
    """protocol for routers that support token-by-token streaming.

    the send callback allows routers to push individual tokens to the
    client as they arrive from the language model. the final return
    value is the complete response for persistence and logging.
    consumers that do not need streaming can ignore this protocol.
    """

    async def route_inbound_streaming(
        self,
        message: ChannelMessage,
        send: Callable[[str], Awaitable[None]],
    ) -> ChannelResponse | None:
        """route inbound message with streaming token callback.

        :param message: normalized inbound message from channel
        :ptype message: ChannelMessage
        :param send: callback to send individual tokens to client
        :ptype send: Callable[[str], Awaitable[None]]
        :return: complete response for persistence, or None
        :rtype: ChannelResponse | None
        """
        ...


class ConnectionRegistry:
    """synchronized pod-local map of live ``user_id → socket`` handles.

    the handler's own bookkeeping of the live, non-serializable socket
    handles it is currently servicing. **not** a store of cross-pod
    membership/presence — that lives in
    :class:`~threetears.channels.presence.collection.PresenceCollection`
    (channels-task-01) and is reshaped through
    :class:`~threetears.channels.presence.room_state.RoomState`. this
    registry holds only what genuinely cannot leave the pod: the live
    handles.

    every access is guarded by a :class:`threading.Lock` so the map is
    safe under this stack's concurrency (asyncio handler coroutines plus
    ``run_in_threadpool`` worker threads) — the previous bare-dict shape
    raced (iterate-while-mutate across awaits/threads). reads return a
    fresh snapshot list so a caller never iterates the live map.
    """

    def __init__(self) -> None:
        """initialize the empty, lock-guarded connection map."""
        self._connections: dict[str, list[Any]] = {}
        self._lock = threading.Lock()

    def register(self, user_id: str, websocket: Any) -> None:
        """add a live socket handle for a user.

        :param user_id: identifier of authenticated user
        :ptype user_id: str
        :param websocket: live socket handle to register
        :ptype websocket: Any
        :return: nothing
        :rtype: None
        """
        with self._lock:
            self._connections.setdefault(user_id, []).append(websocket)

    def unregister(self, user_id: str, websocket: Any) -> None:
        """remove a user's live socket handle.

        safe to call when ``user_id`` or ``websocket`` is not
        registered. drops the user's bucket entirely when its last
        handle leaves so the map does not accrete empty lists.

        :param user_id: identifier of authenticated user
        :ptype user_id: str
        :param websocket: live socket handle to remove
        :ptype websocket: Any
        :return: nothing
        :rtype: None
        """
        with self._lock:
            connections = self._connections.get(user_id)
            if connections is None:
                return
            try:
                connections.remove(websocket)
            except ValueError:
                return
            if not connections:
                del self._connections[user_id]

    def get_connections(self, user_id: str) -> list[Any]:
        """return a snapshot of a user's live socket handles.

        returns a fresh list (never the live one), so the caller can
        iterate/await over it without racing a concurrent register /
        unregister.

        :param user_id: identifier of authenticated user
        :ptype user_id: str
        :return: snapshot list of live socket handles (empty when none)
        :rtype: list[Any]
        """
        with self._lock:
            return list(self._connections.get(user_id, []))


class WebSocketHandler:
    """manages websocket connection lifecycle with delegated authentication.

    handles accept, authenticate, message loop, and cleanup for each
    websocket connection. authentication is fully delegated to host
    application via auth_validator callable. supports optional streaming
    when router implements StreamingChannelRouter protocol.

    :param router: channel router for processing inbound messages
    :ptype router: ChannelRouter-conforming object
    :param auth_validator: callable that validates JWT token string and
        returns decoded payload dict or None if invalid
    :ptype auth_validator: Callable[[str], Awaitable[dict | None]]
    :param config: optional handler configuration overrides
    :ptype config: dict[str, Any] | None
    """

    def __init__(
        self,
        router: Any,
        auth_validator: Callable[[str], Awaitable[dict[str, Any] | None]],
        config: dict[str, Any] | None = None,
        *,
        room_state: RoomState | None = None,
        room_fanout: RoomFanout | None = None,
        acl_cache: AclCache | None = None,
        ns_resolver: NsResolver | None = None,
        op_handler: OpHandler | None = None,
        replay_source: ReplaySource | None = None,
        frame_handlers: dict[str, FrameHandler] | None = None,
        join_action: str = _DEFAULT_JOIN_ACTION,
        write_action: str = _DEFAULT_WRITE_ACTION,
    ) -> None:
        """initialize websocket handler with router, auth validator, and config.

        config keys:
          - heartbeat_interval: seconds between heartbeat pings (default 30)
          - max_message_size: maximum inbound message bytes (default 65536)
          - rate_limit_messages: max messages per window (default 10)
          - rate_limit_window: sliding window duration in seconds (default 1.0)

        the cross-pod collaboration seams (``room_state`` … ``replay_source``)
        are all optional (design T3-D5): with **none** injected the handler
        behaves exactly as the chat handler always has — ``message`` frames
        route through ``router``, no rooms / authz / resume. A room-capable,
        authorized, resumable deployment injects them. This is a *config*,
        not a preserved legacy path: scriob injects the policy (the ACL
        cache + room→namespace resolver), the durable op append
        (``op_handler``), and the replay tail (``replay_source``); channels
        owns only the mechanism + transport.

        :param router: channel router for processing inbound messages
        :ptype router: ChannelRouter-conforming object
        :param auth_validator: callable that validates JWT token string and
            returns decoded payload dict or None if invalid
        :ptype auth_validator: Callable[[str], Awaitable[dict | None]]
        :param config: optional handler configuration overrides
        :ptype config: dict[str, Any] | None
        :param room_state: cross-pod presence/room state (task-01); enables
            the per-connection live-handle registration + room membership
        :ptype room_state: RoomState | None
        :param room_fanout: cross-pod room message backplane (task-02);
            enables join/leave + broadcast
        :ptype room_fanout: RoomFanout | None
        :param acl_cache: shared ``AclCache`` (scriob's membership/grant
            loaders) consulted on every authz gate
        :ptype acl_cache: AclCache | None
        :param ns_resolver: room id → ACL namespace entity (scriob policy)
        :ptype ns_resolver: NsResolver | None
        :param op_handler: durable ``editor.op`` append (scriob op-log);
            returns the authoritative seq channels broadcasts
        :ptype op_handler: OpHandler | None
        :param replay_source: durable op-log replay tail for resume
        :ptype replay_source: ReplaySource | None
        :param frame_handlers: app-specific frame type → handler, extending
            the router with the app's own frames (e.g. scriob ``commit``)
            without forking channels; a key naming a built-in frame type is
            rejected
        :ptype frame_handlers: dict[str, FrameHandler] | None
        :param join_action: canonical ``agent-acl`` action gating join
            (default ``room.join``)
        :ptype join_action: str
        :param write_action: canonical ``agent-acl`` action gating
            broadcast / op (default ``entry.write``)
        :ptype write_action: str
        """
        self.router = router
        self._auth_validator = auth_validator
        self.config: dict[str, Any] = config if config is not None else {}
        self.heartbeat_interval: int = self.config.get("heartbeat_interval", _DEFAULT_HEARTBEAT_INTERVAL)
        self.max_message_size: int = self.config.get("max_message_size", _DEFAULT_MAX_MESSAGE_SIZE)
        self.rate_limit_messages: int = self.config.get("rate_limit_messages", _DEFAULT_RATE_LIMIT_MESSAGES)
        self.rate_limit_window: float = self.config.get("rate_limit_window", _DEFAULT_RATE_LIMIT_WINDOW)
        self.registry = ConnectionRegistry()
        # authorization is all-or-nothing: a half-wired config (one of the two
        # authz seams set, the other not) would silently authorize NOTHING
        # (the gate's no-authz allow short-circuits), turning a deployment that
        # *intended* to authorize into an allow-all hole. require both together
        # or neither, so the only un-authorized config is an explicit one.
        if (acl_cache is None) != (ns_resolver is None):
            raise ValueError("acl_cache and ns_resolver must be provided together (both or neither)")
        # app frame handlers extend the router with app-specific types; they
        # may not shadow a built-in type (that would break core routing).
        self._frame_handlers: dict[str, FrameHandler] = dict(frame_handlers or {})
        reserved = _BUILTIN_FRAME_TYPES & self._frame_handlers.keys()
        if reserved:
            raise ValueError(f"frame_handlers may not register reserved built-in frame type(s): {sorted(reserved)}")
        self._room_state = room_state
        self._room_fanout = room_fanout
        self._acl_cache = acl_cache
        self._ns_resolver = ns_resolver
        self._op_handler = op_handler
        self._replay_source = replay_source
        self._join_action = join_action
        self._write_action = write_action

    async def handle_connection(self, websocket: Any) -> None:
        """manage full lifecycle of single websocket connection.

        accepts connection, authenticates via query param or first message,
        enters message loop on success, and cleans up on disconnect or error.

        :param websocket: websocket connection conforming to WebSocketProtocol
        :ptype websocket: Any
        """
        await websocket.accept()

        auth_payload = await self._authenticate(websocket)
        if auth_payload is None:
            return

        user_id = str(auth_payload.get("user_id", ""))
        customer_id = str(auth_payload.get("customer_id", ""))

        # one stable id per socket (design T3-D5): the presence pk (task-01)
        # and the broadcast ``exclude`` (so the author never echoes their own
        # frame). lives only here + the task-01 synchronized socket map.
        connection_id = str(uuid7())

        await websocket.send_text(json.dumps({"type": "connected", "user_id": user_id}))

        # legacy chat bookkeeping (the live ``user_id → socket`` handle map).
        self.registry.register(user_id, websocket)
        # cross-pod live-handle registration (task-01), only when wired.
        if self._room_state is not None:
            await self._room_state.register(connection_id, websocket)

        # the rooms THIS connection has joined — connection-local scope only
        # (design T3-D6: not shared/queryable state), so disconnect can leave
        # each. cross-pod membership itself lives in the task-01 collection.
        joined_rooms: set[str] = set()

        try:
            # resume-on-connect (design T3-D4): if the client carries a resume
            # cursor on the query string and a replay source is wired, stream
            # the durable tail before going live so a reconnect loses nothing.
            await self._maybe_resume_on_connect(websocket)
            await self._message_loop(websocket, user_id, customer_id, connection_id, joined_rooms)
        finally:
            self.registry.unregister(user_id, websocket)
            if self._room_fanout is not None:
                for room_id in joined_rooms:
                    await self._room_fanout.leave_room(room_id, connection_id)
            if self._room_state is not None:
                await self._room_state.unregister(connection_id, websocket)

    async def _authenticate(self, websocket: Any) -> dict[str, Any] | None:
        """authenticate websocket connection via query param or first message.

        checks query_params for token first. if not present, waits for
        first message containing auth payload. sends error and closes
        connection on authentication failure.

        :param websocket: websocket connection to authenticate
        :ptype websocket: Any
        :return: decoded auth payload dict or None on failure
        :rtype: dict[str, Any] | None
        """
        token: str | None = None

        query_params = getattr(websocket, "query_params", {})
        if "token" in query_params:
            token = query_params["token"]

        if token is None:
            try:
                raw = await websocket.receive_text()
                data = json.loads(raw)
                if data.get("type") == "auth":
                    token = data.get("token")
            except Exception:
                log.warning("websocket disconnected during authentication")
                await self._close_with_error(websocket, "authentication failed")
                return None

        if token is None:
            await self._close_with_error(websocket, "no authentication token provided")
            return None

        payload = await self._auth_validator(token)
        if payload is None:
            await self._close_with_error(websocket, "authentication failed")
            return None

        result = payload
        return result

    async def _message_loop(
        self,
        websocket: Any,
        user_id: str,
        customer_id: str = "",
        connection_id: str = "",
        joined_rooms: set[str] | None = None,
    ) -> None:
        """process inbound messages until disconnect or error.

        receives JSON messages, parses each into a typed :class:`Frame`,
        and dispatches by ``type`` (design T3-D2): ``message`` runs the
        existing chat router path **unchanged**; ``join`` / ``leave`` /
        ``editor.op`` / the transient ``cursor`` / ``typing`` / ``presence``
        / ``resume`` types drive the cross-pod room seams (when wired);
        an **unknown** type yields an ``error`` frame (never a silent drop).
        enforces message size limits and sliding-window rate limiting
        before processing each message.

        :param websocket: authenticated websocket connection
        :ptype websocket: Any
        :param user_id: identifier of authenticated user
        :ptype user_id: str
        :param customer_id: tenant id from the auth payload (for room
            membership + the authz scope); empty in the chat config
        :ptype customer_id: str
        :param connection_id: this socket's stable id (presence pk +
            broadcast ``exclude``); empty in the chat config
        :ptype connection_id: str
        :param joined_rooms: connection-local set of rooms this socket has
            joined, mutated in place so the disconnect path leaves each
        :ptype joined_rooms: set[str] | None
        """
        is_streaming = isinstance(self.router, StreamingChannelRouter)
        if joined_rooms is None:
            joined_rooms = set()

        rate_window_start = time.monotonic()
        rate_message_count = 0

        while True:
            try:
                raw = await websocket.receive_text()
            except Exception:
                log.debug(
                    "websocket disconnected for user %s",
                    user_id,
                )
                break

            if len(raw) > self.max_message_size:
                await websocket.send_text(json.dumps({"type": "error", "message": "message too large"}))
                continue

            now = time.monotonic()
            if now - rate_window_start >= self.rate_limit_window:
                rate_window_start = now
                rate_message_count = 0
            rate_message_count += 1
            if rate_message_count > self.rate_limit_messages:
                await websocket.send_text(json.dumps({"type": "error", "message": "rate limit exceeded"}))
                continue

            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                log.warning(
                    "received non-JSON websocket message from user %s",
                    user_id,
                )
                await websocket.send_text(json.dumps({"type": "error", "message": "invalid json"}))
                continue

            # the chat ``message`` path is preserved verbatim: it reads ``data``
            # loosely (``.get(...)``) and never strict-validates ``room``/``seq``/
            # ``payload``, so a legacy chat frame is handled byte-identically to
            # before task-03. only the typed cross-pod frames are parsed into the
            # strict ``Frame`` envelope.
            msg_type = data.get("type", "") if isinstance(data, dict) else ""
            if msg_type != "message":
                try:
                    frame = Frame.model_validate(data)
                except ValidationError:
                    # valid JSON but not a well-formed frame (e.g. no ``type``):
                    # an ``error`` frame, never a silent drop (design T3-D2).
                    log.warning("received malformed websocket frame from user %s", user_id)
                    await websocket.send_text(json.dumps({"type": "error", "message": "invalid frame"}))
                    continue
                try:
                    await self._route_frame(
                        websocket,
                        frame,
                        user_id=user_id,
                        customer_id=customer_id,
                        connection_id=connection_id,
                        joined_rooms=joined_rooms,
                    )
                except Exception:  # prawduct:allow prawduct/broad-except -- per-frame safety net: a single frame's handler (a built-in path, an injected op_handler/frame_handler, a bad room id) must NEVER crash the whole socket; an unanticipated error becomes one error frame + a log and the connection keeps serving (recoverable rejections are already an OpRejected/error frame upstream)
                    log.exception(
                        "frame handler raised; surfacing an error and keeping the socket alive",
                        extra={"extra_data": {"user_id": user_id, "frame_type": frame.type}},
                    )
                    await websocket.send_text(Frame.error("internal error handling frame"))
                continue

            content = data.get("content", "")
            metadata = data.get("metadata", {})

            # browser-supplied per-message locale info -- mirrors the
            # devx chat client pattern: top-level fields on the WS
            # frame, populated from
            # ``Intl.DateTimeFormat().resolvedOptions().timeZone`` and
            # ``navigator.language``. fall back to ``metadata`` keys
            # of the same names so a client that bundles the values
            # under metadata still works.
            user_tz: str | None = data.get("user_timezone") or metadata.get("user_timezone")
            user_locale: str | None = data.get("user_locale") or metadata.get("user_locale")

            channel_message = ChannelMessage(
                channel_type="websocket",
                content=content,
                sender_id=user_id,
                # customer_id is the SERVER-authenticated scope from the auth
                # payload (the access-token ``customer_id`` claim, surfaced via
                # ``_message_loop``), NOT a client-supplied ``metadata`` value:
                # the host mints identity from it, so a client must not be able
                # to spoof it. empty (chat config with no customer) normalizes
                # to None so the field reads as "absent" rather than "".
                customer_id=customer_id or None,
                metadata=metadata,
                user_timezone=user_tz if isinstance(user_tz, str) and user_tz else None,
                user_locale=user_locale if isinstance(user_locale, str) and user_locale else None,
            )

            try:
                if is_streaming:
                    await self._route_streaming(websocket, channel_message)
                else:
                    await self._route_standard(websocket, channel_message)
            except Exception:  # prawduct:allow prawduct/broad-except -- per-message safety net: the
                # chat ``message`` path lacked the same protection the typed cross-pod frame path
                # already has (see the ``_route_frame`` call above) -- a router failure (an unknown
                # target agent, a downstream dispatch error) must NEVER crash the whole socket; an
                # unanticipated error becomes one error frame + a log and the connection keeps
                # serving, matching design T3-D2's "never a silent drop, never a dead connection"
                # posture for every message shape, not just typed frames.
                log.exception(
                    "chat message routing raised; surfacing an error and keeping the socket alive",
                    extra={"extra_data": {"user_id": user_id}},
                )
                await websocket.send_text(json.dumps({"type": "error", "message": "internal error handling message"}))

    async def _route_standard(self, websocket: Any, message: ChannelMessage) -> None:
        """route message through standard (non-streaming) router.

        :param websocket: websocket connection to send response to
        :ptype websocket: Any
        :param message: normalized inbound message
        :ptype message: ChannelMessage
        """
        response = await self.router.route_inbound(message)
        if response is None:
            return
        await websocket.send_text(
            json.dumps(
                {
                    "type": "response",
                    "content": response.content,
                    "metadata": response.metadata,
                }
            )
        )

    async def _route_streaming(self, websocket: Any, message: ChannelMessage) -> None:
        """route message through streaming router with token callback.

        sends individual tokens as stream-type messages and final
        complete response as response-type message.

        :param websocket: websocket connection to send tokens and response to
        :ptype websocket: Any
        :param message: normalized inbound message
        :ptype message: ChannelMessage
        """

        async def send_token(token: str) -> None:
            """send streaming token to websocket client.

            :param token: individual token from language model
            :ptype token: str
            """
            await websocket.send_text(json.dumps({"type": "stream", "content": token}))

        response = await self.router.route_inbound_streaming(message, send_token)
        if response is None:
            return
        await websocket.send_text(
            json.dumps(
                {
                    "type": "response",
                    "content": response.content,
                    "metadata": response.metadata,
                }
            )
        )

    async def _route_frame(
        self,
        websocket: Any,
        frame: Frame,
        *,
        user_id: str,
        customer_id: str,
        connection_id: str,
        joined_rooms: set[str],
    ) -> None:
        """dispatch a non-``message`` typed frame to its room seam (design T3-D2).

        ``join`` / ``leave`` drive room membership (after the ``room.join``
        gate); ``editor.op`` appends durably then broadcasts the op carrying
        the op-log seq; the transient ``cursor`` / ``typing`` / ``presence``
        types broadcast with no seq (after the ``entry.write`` gate);
        ``resume`` streams the durable tail. an **unknown** type yields an
        ``error`` frame — never a silent drop.

        :param websocket: the live socket
        :ptype websocket: Any
        :param frame: the parsed inbound frame
        :ptype frame: Frame
        :param user_id: authenticated principal
        :ptype user_id: str
        :param customer_id: tenant id
        :ptype customer_id: str
        :param connection_id: this socket's stable id
        :ptype connection_id: str
        :param joined_rooms: connection-local joined-room set (mutated)
        :ptype joined_rooms: set[str]
        """
        if frame.type == "join":
            await self._handle_join(websocket, frame, user_id, customer_id, connection_id, joined_rooms)
        elif frame.type == "leave":
            await self._handle_leave(frame, connection_id, joined_rooms)
        elif frame.type == "editor.op":
            await self._handle_editor_op(websocket, frame, user_id, joined_rooms)
        elif frame.type in _TRANSIENT_FRAME_TYPES:
            await self._handle_transient(websocket, frame, user_id, connection_id, joined_rooms)
        elif frame.type == "resume":
            await self._handle_resume(websocket, frame)
        elif frame.type in self._frame_handlers:
            # app-registered frame type (e.g. scriob ``commit``): hand it the
            # frame + identity + a reply ``send``; the app owns its own authz.
            await self._frame_handlers[frame.type](
                frame,
                user_id=user_id,
                customer_id=customer_id,
                connection_id=connection_id,
                send=websocket.send_text,
            )
        else:
            await websocket.send_text(Frame.error(f"unknown frame type: {frame.type}"))

    async def _authorize(self, websocket: Any, room_id: str, action: str, user_id: str) -> bool:
        """run an ``agent-acl`` gate for ``action`` on ``room_id``'s namespace.

        resolves the room to its ACL namespace via the injected
        ``ns_resolver`` and calls ``authorize_on_entity`` — the ONLY
        authorization path (design T3-D1/D8). ``user_id`` is border-converted
        ``str → UUID`` at the call (the UUID-boundary rule). on
        :class:`AccessDenied` an ``error`` frame is sent and ``False``
        returned; the caller performs **no** side effect. with no resolver /
        cache wired (chat config) the gate is a no-op allow.

        :param websocket: the live socket (for the denial frame)
        :ptype websocket: Any
        :param room_id: the room whose namespace is the policy object
        :ptype room_id: str
        :param action: canonical ``agent-acl`` action string
        :ptype action: str
        :param user_id: authenticated principal (str; converted to UUID)
        :ptype user_id: str
        :return: ``True`` when allowed (or no authz wired), ``False`` on deny
        :rtype: bool
        """
        if self._ns_resolver is None or self._acl_cache is None:
            return True
        # border-convert str -> UUID defensively: ``user_id`` comes from the
        # host ``auth_validator`` payload, so a malformed value must DENY (an
        # error frame), never raise out of the message loop and crash the
        # socket. an authz-enabled gate requires a UUID principal.
        try:
            principal = UUID(user_id) if user_id else None
        except ValueError:
            await websocket.send_text(Frame.error(f"access denied: {action}"))
            return False
        ns_entity = await self._ns_resolver(room_id)
        try:
            await authorize_on_entity(
                ns_entity=ns_entity,
                action=action,
                user_id=principal,
                agent_id=None,
                cache=self._acl_cache,
            )
        except AccessDenied:
            await websocket.send_text(Frame.error(f"access denied: {action}"))
            return False
        return True

    async def _handle_join(
        self,
        websocket: Any,
        frame: Frame,
        user_id: str,
        customer_id: str,
        connection_id: str,
        joined_rooms: set[str],
    ) -> None:
        """authorize ``room.join`` then add the connection to the room (task-02).

        a denied join writes **no** presence row and triggers **no**
        broadcast (the gate returns before any membership write).

        :param websocket: the live socket
        :ptype websocket: Any
        :param frame: the inbound ``join`` frame (carries ``room``)
        :ptype frame: Frame
        :param user_id: authenticated principal
        :ptype user_id: str
        :param customer_id: tenant id
        :ptype customer_id: str
        :param connection_id: this socket's stable id
        :ptype connection_id: str
        :param joined_rooms: connection-local joined-room set (mutated)
        :ptype joined_rooms: set[str]
        """
        room_id = frame.room
        if room_id is None or self._room_fanout is None:
            await websocket.send_text(Frame.error("join requires a room"))
            return
        if not await self._authorize(websocket, room_id, self._join_action, user_id):
            return
        await self._room_fanout.join_room(room_id, connection_id, user_id, customer_id)
        joined_rooms.add(room_id)

    async def _handle_leave(self, frame: Frame, connection_id: str, joined_rooms: set[str]) -> None:
        """remove the connection from a room it joined (task-02).

        leave is not authorization-gated — a member may always drop their
        own presence row.

        :param frame: the inbound ``leave`` frame (carries ``room``)
        :ptype frame: Frame
        :param connection_id: this socket's stable id
        :ptype connection_id: str
        :param joined_rooms: connection-local joined-room set (mutated)
        :ptype joined_rooms: set[str]
        """
        room_id = frame.room
        if room_id is None or self._room_fanout is None:
            return
        await self._room_fanout.leave_room(room_id, connection_id)
        joined_rooms.discard(room_id)

    async def _handle_editor_op(self, websocket: Any, frame: Frame, user_id: str, joined_rooms: set[str]) -> None:
        """authorize ``entry.write``, append durably, then broadcast with the seq.

        the durable append is the injected ``op_handler`` (scriob's op-log,
        design T3-D3) which assigns the authoritative op-log seq (T3-D4). on
        success the op frame — carrying that seq — is broadcast to **every**
        room member **including the author**: in server-authoritative OT the
        author needs its own op echoed back with the assigned seq to advance
        its version (the broadcast is the acknowledgement), and peers apply
        and rebase. (The author's client reconciles its optimistic copy by
        the op id it embedded in ``payload``.) A recoverable rejection —
        :class:`~threetears.channels.frames.OpRejected`, e.g. an op-log
        expected-sequence CAS miss — sends the sender an ``error`` frame and
        does **not** broadcast, so an everyday optimistic-concurrency miss
        never crashes the socket.

        :param websocket: the live socket
        :ptype websocket: Any
        :param frame: the inbound ``editor.op`` frame
        :ptype frame: Frame
        :param user_id: authenticated principal
        :ptype user_id: str
        :param joined_rooms: connection-local joined-room set (membership gate)
        :ptype joined_rooms: set[str]
        """
        room_id = frame.room
        if room_id is None:
            await websocket.send_text(Frame.error("editor.op requires a room"))
            return
        if self._room_fanout is None or self._op_handler is None:
            await websocket.send_text(Frame.error("editor.op is not supported on this connection"))
            return
        if room_id not in joined_rooms:
            # editing a room requires having joined it: the author needs its
            # pod subscribed to receive its own op back (the OT ack), and a
            # member is the authorization-clean unit. fail explicitly rather
            # than silently appending an op whose ack the author never sees.
            await websocket.send_text(Frame.error("not joined to room"))
            return
        if not await self._authorize(websocket, room_id, self._write_action, user_id):
            return
        try:
            result = await self._op_handler(room_id, user_id, frame)
        except OpRejected as rejected:
            # recoverable (e.g. an op-log CAS miss — the client is behind):
            # tell the sender, do NOT broadcast, keep the socket alive.
            await websocket.send_text(Frame.error(rejected.message))
            return
        op_frame = Frame(type="editor.op", room=room_id, payload=frame.payload, seq=result.seq)
        # broadcast to ALL members (no exclude): the author needs its own op
        # back carrying the authoritative seq (the ack), peers apply + rebase.
        await self._room_fanout.broadcast(room_id, op_frame.model_dump_json())

    async def _handle_transient(
        self, websocket: Any, frame: Frame, user_id: str, connection_id: str, joined_rooms: set[str]
    ) -> None:
        """authorize ``entry.write`` then transient-broadcast (no seq, no durability).

        ``cursor`` / ``typing`` / ``presence`` are fast-notify only — there is
        no op-log append and the broadcast carries **no** seq. requires having
        joined the room (you broadcast only to rooms you are in) and excludes
        the author so they do not receive their own frame.

        :param websocket: the live socket
        :ptype websocket: Any
        :param frame: the inbound transient frame
        :ptype frame: Frame
        :param user_id: authenticated principal
        :ptype user_id: str
        :param connection_id: this socket's stable id (the broadcast exclude)
        :ptype connection_id: str
        :param joined_rooms: connection-local joined-room set (membership gate)
        :ptype joined_rooms: set[str]
        """
        room_id = frame.room
        if room_id is None or self._room_fanout is None:
            await websocket.send_text(Frame.error(f"{frame.type} requires a room"))
            return
        if room_id not in joined_rooms:
            await websocket.send_text(Frame.error("not joined to room"))
            return
        if not await self._authorize(websocket, room_id, self._write_action, user_id):
            return
        out = Frame(type=frame.type, room=room_id, payload=frame.payload)
        await self._room_fanout.broadcast(room_id, out.model_dump_json(), exclude=connection_id)

    async def _handle_resume(self, websocket: Any, frame: Frame) -> None:
        """stream the durable op-log tail for a ``resume`` frame (design T3-D4).

        replays ``replay_source(room, last_seq)`` to the socket. the resume
        cursor is the op-log ``seq`` carried on the frame — never an
        in-process counter. with no replay source wired this is a no-op.

        :param websocket: the live socket
        :ptype websocket: Any
        :param frame: the inbound ``resume`` frame (carries ``room`` + ``seq``)
        :ptype frame: Frame
        """
        room_id = frame.room
        if room_id is None or self._replay_source is None:
            return
        await self._stream_replay(websocket, room_id, frame.seq or 0)

    async def _maybe_resume_on_connect(self, websocket: Any) -> None:
        """resume from a query-string cursor on connect, before going live.

        a client reconnecting to any pod may carry ``resume_room`` +
        ``resume_seq`` on the connect query string; when a ``replay_source``
        is wired, the durable tail after that seq is streamed to the socket
        before the live message loop starts, so nothing is lost across the
        reconnect (design T3-D4).

        :param websocket: the live socket
        :ptype websocket: Any
        """
        if self._replay_source is None:
            return
        query_params = getattr(websocket, "query_params", {})
        room_id = query_params.get("resume_room")
        if not room_id:
            return
        raw_seq = query_params.get("resume_seq", "0")
        try:
            from_seq = int(raw_seq)
        except TypeError, ValueError:
            from_seq = 0
        await self._stream_replay(websocket, room_id, from_seq)

    async def _stream_replay(self, websocket: Any, room_id: str, from_seq: int) -> None:
        """stream the durable tail after ``from_seq`` to the socket, in order.

        best-effort (design note 4): if the injected ``replay_source`` raises,
        surface an ``error`` frame and continue live rather than crashing the
        socket — the client can re-resume.

        :param websocket: the live socket
        :ptype websocket: Any
        :param room_id: the room to replay
        :ptype room_id: str
        :param from_seq: replay everything after this op-log seq
        :ptype from_seq: int
        """
        if self._replay_source is None:
            return
        try:
            async for payload in self._replay_source(room_id, from_seq):
                await websocket.send_text(payload)
        except Exception:  # prawduct:allow prawduct/broad-except -- resume is best-effort: a replay-source error surfaces as one error frame and the socket stays live (the client can re-resume) rather than crashing the connection
            log.warning(
                "resume replay failed; continuing live",
                extra={"extra_data": {"room_id": room_id, "from_seq": from_seq}},
            )
            await websocket.send_text(Frame.error("resume failed"))

    async def _close_with_error(self, websocket: Any, error_message: str) -> None:
        """send error message and close websocket connection.

        :param websocket: websocket connection to close
        :ptype websocket: Any
        :param error_message: human-readable error description
        :ptype error_message: str
        """
        try:
            await websocket.send_text(json.dumps({"type": "error", "message": error_message}))
        except Exception:
            pass
        try:
            await websocket.close(code=1008)
        except Exception:
            pass
