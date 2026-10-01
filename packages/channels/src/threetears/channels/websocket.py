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

import asyncio
import json
import threading
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable
from uuid import UUID, uuid7

from pydantic import ValidationError

from threetears.agent.acl import AccessDenied, authorize_on_entity
from threetears.channels.frames import Frame, OpRejected, RoomAccessRequest
from threetears.channels.protocol import ChannelMessage, ChannelResponse
from threetears.observe import get_logger

if TYPE_CHECKING:
    from threetears.agent.acl import AclCache
    from threetears.channels.frames import FrameHandler, NsResolver, OpHandler, ReplaySource, RoomPolicy
    from threetears.channels.presence.fanout import RoomFanout
    from threetears.channels.presence.room_state import RoomState

__all__ = [
    "ConnectionRegistry",
    "StreamingChannelRouter",
    "WebSocketHandler",
    "WebSocketProtocol",
    "parse_attachment_ids",
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

# the built-in frame types that act on a room. each is handled under the
# connection's room lock, so a revocation cannot interleave with one.
_ROOM_FRAME_TYPES = frozenset({"join", "leave", "editor.op", "resume", *_TRANSIENT_FRAME_TYPES})

_DEFAULT_REVOKE_REASON = "access revoked"


@dataclass(eq=False)
class _RoomConnection:
    """one live connection's identity and the rooms it is in, as this pod holds it.

    the handler keeps one per open socket so a room's members can be taken out
    from OUTSIDE the socket's own message loop (:meth:`WebSocketHandler.revoke`,
    :meth:`WebSocketHandler.reevaluate_room`). ``lock`` serializes every room
    action on the connection with those evictions: without it an eviction could
    land between a join's authorization and its membership write, and the
    member would be back in the room it was just taken out of.

    :ivar websocket: the live socket
    :ivar user_id: authenticated principal
    :ivar customer_id: the principal's customer
    :ivar joined_rooms: rooms this connection is currently a member of
    :ivar lock: serializes room actions and evictions on this connection
    """

    websocket: Any
    user_id: str
    customer_id: str
    joined_rooms: set[str] = field(default_factory=set)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


async def _safe_send(websocket: Any, payload: str, *, context: str) -> bool:
    """best-effort send: log and drop on failure, never raise.

    Every notification/error-frame send in the message loop and its typed-frame dispatch tree is
    best-effort by design (T3-D2's "never a silent drop, never a dead connection" posture for the
    CONNECTION) -- but the notification send ITSELF can fail too, when the socket died in the
    window between the triggering read and this reply (the exact shape of a disconnect-mid-turn
    client). That failure must degrade quietly, not crash the caller -- mirrors
    :meth:`WebSocketHandler._close_with_error`'s existing swallow-and-log pattern, generalized to
    every call site that needs it instead of just the auth-failure path.

    :param websocket: the live socket
    :ptype websocket: Any
    :param payload: the JSON/frame payload to send
    :ptype payload: str
    :param context: a short label identifying the call site, for the log
    :ptype context: str
    :return: ``True`` if the send succeeded, ``False`` if it failed (logged, swallowed)
    :rtype: bool
    """
    ok = True
    try:
        await websocket.send_text(payload)
    except Exception:  # prawduct:allow prawduct/broad-except -- best-effort notification send: a dead socket must not crash the caller (the message loop or a nested frame handler), which continues or degrades on its own terms
        log.debug(
            "best-effort send failed; socket likely gone",
            extra={"extra_data": {"context": context}},
        )
        ok = False
    return ok


def parse_attachment_ids(raw: object) -> list[UUID] | None:
    """read a chat frame's ``attachment_ids`` into ids, or ``None`` when it is malformed.

    the value is client-supplied JSON: it must be a list, and every element a string
    in UUID form. anything else -- a single string, a number, one bad element among
    good ones -- is malformed as a whole, never partly read, so a frame cannot
    reach an agent missing an image its sender attached. whether the sender may
    attach each id is the host's decision, made by its router.

    :param raw: the frame's ``attachment_ids`` value (``[]`` when absent)
    :ptype raw: object
    :return: the ids in order, or ``None`` when the value is malformed
    :rtype: list[UUID] | None
    """
    result: list[UUID] | None = None
    if isinstance(raw, list) and all(isinstance(item, str) for item in raw):
        try:
            result = [UUID(item) for item in raw]
        except ValueError:
            result = None
    return result


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
                # Not in this user's list: a double-unregister, or a socket registered under a
                # different user_id. Falls through to the empty-list cleanup below rather than
                # returning early, which used to strand an empty list keyed by user_id forever.
                log.debug(
                    "unregister for a socket that was not in the user's connection list",
                    extra={"extra_data": {"user_id": user_id}},
                )
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
        room_policy: RoomPolicy | None = None,
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
        :param room_policy: the app's per-room access rule, asked AFTER the
            namespace gate on every gated room action (``join`` and
            ``resume`` with ``join_action``; ``editor.op`` and the transient
            frames with ``write_action``); both must allow. the namespace
            gate cannot tell two rooms in one namespace apart, so without a
            policy anyone who may read the namespace may enter every room in
            it. ``None`` leaves the namespace gate as the only rule
        :ptype room_policy: RoomPolicy | None
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
        self._room_policy = room_policy
        # connection_id -> the live connection's identity + joined rooms, so a
        # room's members can be evicted from outside their message loops.
        # guarded like ``ConnectionRegistry``: reads take a snapshot.
        self._connections: dict[str, _RoomConnection] = {}
        self._connections_lock = threading.Lock()

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

        # deliberately NOT routed through ``_safe_send`` (crash-safety audit note): this is the
        # very first send on a freshly-accepted, freshly-authenticated socket -- a failure here
        # means the connection is broken in some fundamental way before any bookkeeping
        # (registry, room_state) has been touched, so letting it propagate to the ASGI-level
        # caller and end connection setup outright is correct; there is nothing to degrade
        # gracefully FROM yet.
        await websocket.send_text(json.dumps({"type": "connected", "user_id": user_id}))

        # legacy chat bookkeeping (the live ``user_id → socket`` handle map).
        self.registry.register(user_id, websocket)
        # cross-pod live-handle registration (task-01), only when wired.
        if self._room_state is not None:
            await self._room_state.register(connection_id, websocket)

        # the rooms THIS connection has joined — pod-local scope only
        # (design T3-D6: not shared/queryable state), so disconnect can leave
        # each and an eviction can find them. cross-pod membership itself
        # lives in the task-01 collection.
        connection = _RoomConnection(websocket=websocket, user_id=user_id, customer_id=customer_id)
        with self._connections_lock:
            self._connections[connection_id] = connection

        try:
            # resume-on-connect (design T3-D4): if the client carries a resume
            # cursor on the query string and a replay source is wired, stream
            # the durable tail before going live so a reconnect loses nothing.
            await self._maybe_resume_on_connect(websocket, connection)
            await self._message_loop(websocket, user_id, customer_id, connection_id, connection)
        finally:
            with self._connections_lock:
                self._connections.pop(connection_id, None)
            self.registry.unregister(user_id, websocket)
            # under the room lock, so an eviction in flight finishes first and
            # a room it already left is not left a second time.
            async with connection.lock:
                rooms_to_leave = sorted(connection.joined_rooms)
                connection.joined_rooms.clear()
                if self._room_fanout is not None:
                    for room_id in rooms_to_leave:
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
        customer_id: str,
        connection_id: str,
        connection: _RoomConnection,
    ) -> None:
        """process inbound messages until disconnect or error.

        receives JSON messages, parses each into a typed :class:`Frame`,
        and dispatches by ``type`` (design T3-D2): ``message`` runs the
        chat router path, refusing with an ``error`` frame a message whose
        ``content`` is empty or not a string, or whose ``metadata`` is not an
        object, before any router sees it; ``join`` / ``leave`` /
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
            broadcast ``exclude``)
        :ptype connection_id: str
        :param connection: this socket's identity and joined rooms, shared with
            the disconnect path and with evictions
        :ptype connection: _RoomConnection
        """
        is_streaming = isinstance(self.router, StreamingChannelRouter)

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
                await _safe_send(
                    websocket,
                    json.dumps({"type": "error", "message": "message too large"}),
                    context="message-too-large",
                )
                continue

            now = time.monotonic()
            if now - rate_window_start >= self.rate_limit_window:
                rate_window_start = now
                rate_message_count = 0
            rate_message_count += 1
            if rate_message_count > self.rate_limit_messages:
                await _safe_send(
                    websocket,
                    json.dumps({"type": "error", "message": "rate limit exceeded"}),
                    context="rate-limit-exceeded",
                )
                continue

            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                log.warning(
                    "received non-JSON websocket message from user %s",
                    user_id,
                )
                await _safe_send(
                    websocket, json.dumps({"type": "error", "message": "invalid json"}), context="invalid-json"
                )
                continue

            # the chat ``message`` path reads ``data`` loosely (``.get(...)``) and
            # never strict-validates ``room``/``seq``/``payload``, so a legacy chat
            # frame is handled as before task-03; it checks only that there is a
            # string to route. only the typed cross-pod frames are parsed into the
            # strict ``Frame`` envelope.
            msg_type = data.get("type", "") if isinstance(data, dict) else ""
            if msg_type != "message":
                try:
                    frame = Frame.model_validate(data)
                except ValidationError:
                    # valid JSON but not a well-formed frame (e.g. no ``type``):
                    # an ``error`` frame, never a silent drop (design T3-D2).
                    log.warning("received malformed websocket frame from user %s", user_id)
                    await _safe_send(
                        websocket, json.dumps({"type": "error", "message": "invalid frame"}), context="invalid-frame"
                    )
                    continue
                try:
                    await self._route_frame(websocket, frame, connection_id=connection_id, connection=connection)
                except Exception:  # prawduct:allow prawduct/broad-except -- per-frame safety net: a single frame's handler (a built-in path, an injected op_handler/frame_handler, a bad room id) must NEVER crash the whole socket; an unanticipated error becomes one error frame + a log and the connection keeps serving (recoverable rejections are already an OpRejected/error frame upstream)
                    log.exception(
                        "frame handler raised; surfacing an error and keeping the socket alive",
                        extra={"extra_data": {"user_id": user_id, "frame_type": frame.type}},
                    )
                    await _safe_send(
                        websocket, Frame.error("internal error handling frame"), context="route-frame-except"
                    )
                continue

            content = data.get("content", "")
            metadata = data.get("metadata", {})
            attachment_ids = parse_attachment_ids(data.get("attachment_ids", []))
            # a chat frame with nothing an agent can use is answered here, before the
            # router: dispatching it would spend a model call on nothing (the REST chat
            # door refuses an empty message too), and a non-object ``metadata`` would
            # fail the ``.get`` reads below, outside the per-message safety net. an
            # ``attachment_ids`` that is not a list of ids is refused the same way:
            # half-reading it would send the turn without an image the person attached.
            refusal: str | None = None
            if not isinstance(content, str) or not isinstance(metadata, dict) or attachment_ids is None:
                refusal = "invalid message"
            elif not content:
                refusal = "empty message"
            if refusal is not None:
                log.warning("refused websocket chat message from user %s: %s", user_id, refusal)
                await _safe_send(
                    websocket, json.dumps({"type": "error", "message": refusal}), context="chat-message-refused"
                )
                continue

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
                attachment_ids=attachment_ids or [],
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
                await _safe_send(
                    websocket,
                    json.dumps({"type": "error", "message": "internal error handling message"}),
                    context="chat-message-route-except",
                )

    async def _route_standard(self, websocket: Any, message: ChannelMessage) -> None:
        """route message through standard (non-streaming) router.

        deliberately NOT routed through ``_safe_send`` (crash-safety audit note): this and
        ``_route_streaming``'s payload sends below are called from ``_message_loop``'s
        ``msg_type == "message"`` branch, already wrapped end-to-end in that branch's own
        ``except Exception`` safety net -- a failure here propagates to that ONE outer catch
        (now itself ``_safe_send``-guarded), which logs with full chat-router context and
        attempts exactly one error notification, rather than each of THESE sends separately
        attempting a second, less-informative notification of their own.

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
        connection_id: str,
        connection: _RoomConnection,
    ) -> None:
        """dispatch a non-``message`` typed frame to its room seam (design T3-D2).

        ``join`` / ``leave`` drive room membership (after the ``room.join``
        gate); ``editor.op`` appends durably then broadcasts the op carrying
        the op-log seq; the transient ``cursor`` / ``typing`` / ``presence``
        types broadcast with no seq (after the ``entry.write`` gate);
        ``resume`` streams the durable tail. an **unknown** type yields an
        ``error`` frame — never a silent drop.

        every room frame is handled under the connection's room lock, so an
        eviction (:meth:`revoke` / :meth:`reevaluate_room`) waits for one in
        flight and the next one sees the eviction.

        :param websocket: the live socket
        :ptype websocket: Any
        :param frame: the parsed inbound frame
        :ptype frame: Frame
        :param connection_id: this socket's stable id
        :ptype connection_id: str
        :param connection: this socket's identity and joined rooms
        :ptype connection: _RoomConnection
        """
        if frame.type in _ROOM_FRAME_TYPES:
            async with connection.lock:
                await self._route_room_frame(websocket, frame, connection_id=connection_id, connection=connection)
        elif frame.type in self._frame_handlers:
            # app-registered frame type (e.g. scriob ``commit``): hand it the
            # frame + identity + a reply ``send``; the app owns its own authz.
            #
            # deliberately the RAW ``websocket.send_text``, not ``_safe_send`` (crash-safety
            # audit note, third exception): an app handler may need to OBSERVE a send failure
            # itself -- e.g. a long-running streamed handler that treats "the transport just
            # died" as a signal to stop its own work early, not just a fire-and-forget notify.
            # Wrapping it here would silently swallow that signal before the app ever sees it.
            # Worst case (a handler that does NOT itself guard its ``send`` calls) is still
            # connection-safe: any exception the handler doesn't catch propagates to THIS
            # method's own caller in ``_message_loop`` (the per-frame safety net, already
            # ``_safe_send``-guarded), so a crash still cannot reach the connection level.
            await self._frame_handlers[frame.type](
                frame,
                user_id=connection.user_id,
                customer_id=connection.customer_id,
                connection_id=connection_id,
                send=websocket.send_text,
            )
        else:
            await _safe_send(websocket, Frame.error(f"unknown frame type: {frame.type}"), context="unknown-frame-type")

    async def _route_room_frame(
        self,
        websocket: Any,
        frame: Frame,
        *,
        connection_id: str,
        connection: _RoomConnection,
    ) -> None:
        """dispatch one built-in room frame. **caller holds ``connection.lock``.**

        :param websocket: the live socket
        :ptype websocket: Any
        :param frame: the parsed inbound room frame
        :ptype frame: Frame
        :param connection_id: this socket's stable id
        :ptype connection_id: str
        :param connection: this socket's identity and joined rooms
        :ptype connection: _RoomConnection
        """
        if frame.type == "join":
            await self._handle_join(websocket, frame, connection_id, connection)
        elif frame.type == "leave":
            await self._handle_leave(frame, connection_id, connection.joined_rooms)
        elif frame.type == "editor.op":
            await self._handle_editor_op(websocket, frame, connection)
        elif frame.type in _TRANSIENT_FRAME_TYPES:
            await self._handle_transient(websocket, frame, connection_id, connection)
        else:
            await self._handle_resume(websocket, frame, connection)

    async def _decide(self, room_id: str, action: str, user_id: str, customer_id: str) -> bool:
        """answer whether ``user_id`` may perform ``action`` in ``room_id``, sending nothing.

        two gates, in order, and both must allow. first the ``agent-acl``
        namespace gate: the room resolves to its ACL namespace via the injected
        ``ns_resolver`` and ``authorize_on_entity`` decides (design T3-D1/D8),
        with ``user_id`` border-converted ``str → UUID``; a malformed principal
        is a denial, never a crash. then the injected ``room_policy``, which is
        what tells apart two rooms in one namespace; only a literal ``True``
        from it allows. a gate that is not wired allows, so the chat config
        (neither wired) allows everything, as it always has.

        a policy that raises propagates: every caller treats that as a refusal
        -- the per-frame safety net answers it with an error frame and no side
        effect, and :meth:`reevaluate_room` evicts.

        :param room_id: the room acted in
        :ptype room_id: str
        :param action: canonical action string being gated
        :ptype action: str
        :param user_id: authenticated principal (str; converted to UUID for the
            namespace gate)
        :ptype user_id: str
        :param customer_id: the principal's customer
        :ptype customer_id: str
        :return: ``True`` when every wired gate allows
        :rtype: bool
        """
        allowed = True
        if self._ns_resolver is not None and self._acl_cache is not None:
            principal: UUID | None = None
            try:
                principal = UUID(user_id) if user_id else None
            except ValueError:
                allowed = False
            if allowed:
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
                    allowed = False
        if allowed and self._room_policy is not None:
            answer = await self._room_policy(
                RoomAccessRequest(room_id=room_id, user_id=user_id, customer_id=customer_id, action=action)
            )
            allowed = answer is True
        return allowed

    async def _authorize(self, websocket: Any, room_id: str, action: str, connection: _RoomConnection) -> bool:
        """gate ``action`` in ``room_id`` for this connection, answering a refusal with an error frame.

        the decision is :meth:`_decide`; on a refusal an ``error`` frame is sent
        and ``False`` returned, and the caller performs **no** side effect.

        :param websocket: the live socket (for the denial frame)
        :ptype websocket: Any
        :param room_id: the room acted in
        :ptype room_id: str
        :param action: canonical action string
        :ptype action: str
        :param connection: the acting connection's identity
        :ptype connection: _RoomConnection
        :return: ``True`` when allowed, ``False`` on refusal
        :rtype: bool
        """
        allowed = await self._decide(room_id, action, connection.user_id, connection.customer_id)
        if not allowed:
            await _safe_send(websocket, Frame.error(f"access denied: {action}"), context="authorize-denied")
        return allowed

    async def _handle_join(
        self,
        websocket: Any,
        frame: Frame,
        connection_id: str,
        connection: _RoomConnection,
    ) -> None:
        """authorize ``room.join`` then add the connection to the room (task-02).

        a denied join writes **no** presence row and triggers **no**
        broadcast (the gate returns before any membership write). a join for
        a room the connection is already in is authorized again and changes
        nothing: a second membership write would take a second room reference
        that the one leave on disconnect can never release.

        :param websocket: the live socket
        :ptype websocket: Any
        :param frame: the inbound ``join`` frame (carries ``room``)
        :ptype frame: Frame
        :param connection_id: this socket's stable id
        :ptype connection_id: str
        :param connection: this socket's identity and joined rooms (mutated)
        :ptype connection: _RoomConnection
        """
        room_id = frame.room
        if room_id is None or self._room_fanout is None:
            await _safe_send(websocket, Frame.error("join requires a room"), context="join-no-room")
            return
        if not await self._authorize(websocket, room_id, self._join_action, connection):
            return
        if room_id in connection.joined_rooms:
            return
        await self._room_fanout.join_room(room_id, connection_id, connection.user_id, connection.customer_id)
        connection.joined_rooms.add(room_id)

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

    async def _handle_editor_op(self, websocket: Any, frame: Frame, connection: _RoomConnection) -> None:
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
        :param connection: the authoring connection's identity and joined rooms
            (membership gate)
        :ptype connection: _RoomConnection
        """
        room_id = frame.room
        if room_id is None:
            await _safe_send(websocket, Frame.error("editor.op requires a room"), context="editor-op-no-room")
            return
        if self._room_fanout is None or self._op_handler is None:
            await _safe_send(
                websocket, Frame.error("editor.op is not supported on this connection"), context="editor-op-unsupported"
            )
            return
        if room_id not in connection.joined_rooms:
            # editing a room requires having joined it: the author needs its
            # pod subscribed to receive its own op back (the OT ack), and a
            # member is the authorization-clean unit. fail explicitly rather
            # than silently appending an op whose ack the author never sees.
            await _safe_send(websocket, Frame.error("not joined to room"), context="editor-op-not-joined")
            return
        if not await self._authorize(websocket, room_id, self._write_action, connection):
            return
        try:
            result = await self._op_handler(room_id, connection.user_id, frame)
        except OpRejected as rejected:
            # recoverable (e.g. an op-log CAS miss — the client is behind):
            # tell the sender, do NOT broadcast, keep the socket alive.
            await _safe_send(websocket, Frame.error(rejected.message), context="editor-op-rejected")
            return
        op_frame = Frame(type="editor.op", room=room_id, payload=frame.payload, seq=result.seq)
        # broadcast to ALL members (no exclude): the author needs its own op
        # back carrying the authoritative seq (the ack), peers apply + rebase.
        await self._room_fanout.broadcast(room_id, op_frame.model_dump_json())

    async def _handle_transient(
        self, websocket: Any, frame: Frame, connection_id: str, connection: _RoomConnection
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
        :param connection_id: this socket's stable id (the broadcast exclude)
        :ptype connection_id: str
        :param connection: the sending connection's identity and joined rooms
            (membership gate)
        :ptype connection: _RoomConnection
        """
        room_id = frame.room
        if room_id is None or self._room_fanout is None:
            await _safe_send(websocket, Frame.error(f"{frame.type} requires a room"), context="transient-no-room")
            return
        if room_id not in connection.joined_rooms:
            await _safe_send(websocket, Frame.error("not joined to room"), context="transient-not-joined")
            return
        if not await self._authorize(websocket, room_id, self._write_action, connection):
            return
        out = Frame(type=frame.type, room=room_id, payload=frame.payload)
        await self._room_fanout.broadcast(room_id, out.model_dump_json(), exclude=connection_id)

    async def _handle_resume(self, websocket: Any, frame: Frame, connection: _RoomConnection) -> None:
        """stream the durable op-log tail for a ``resume`` frame (design T3-D4).

        replays ``replay_source(room, last_seq)`` to the socket, after the same
        ``join_action`` gate a join passes: the tail IS the room's content, so
        reading it is what joining grants. the resume cursor is the op-log
        ``seq`` carried on the frame — never an in-process counter. with no
        replay source wired this is a no-op.

        :param websocket: the live socket
        :ptype websocket: Any
        :param frame: the inbound ``resume`` frame (carries ``room`` + ``seq``)
        :ptype frame: Frame
        :param connection: the resuming connection's identity
        :ptype connection: _RoomConnection
        """
        room_id = frame.room
        if room_id is None or self._replay_source is None:
            return
        if not await self._authorize(websocket, room_id, self._join_action, connection):
            return
        await self._stream_replay(websocket, room_id, frame.seq or 0)

    async def _maybe_resume_on_connect(self, websocket: Any, connection: _RoomConnection) -> None:
        """resume from a query-string cursor on connect, before going live.

        a client reconnecting to any pod may carry ``resume_room`` +
        ``resume_seq`` on the connect query string; when a ``replay_source``
        is wired, the durable tail after that seq is streamed to the socket
        before the live message loop starts, so nothing is lost across the
        reconnect (design T3-D4). gated like a ``resume`` frame; a refusal is
        an error frame and the connection goes live without the tail.

        :param websocket: the live socket
        :ptype websocket: Any
        :param connection: the connecting socket's identity
        :ptype connection: _RoomConnection
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
        async with connection.lock:
            if not await self._authorize(websocket, room_id, self._join_action, connection):
                return
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
                # stop replaying the moment a send fails -- the socket is dead, so attempting the
                # rest of the tail is pointless (best-effort, not best-effort-repeated-forever).
                if not await _safe_send(websocket, payload, context="replay-payload"):
                    return
        except Exception:  # prawduct:allow prawduct/broad-except -- resume is best-effort: a replay-source error surfaces as one error frame and the socket stays live (the client can re-resume) rather than crashing the connection
            log.warning(
                "resume replay failed; continuing live",
                extra={"extra_data": {"room_id": room_id, "from_seq": from_seq}},
            )
            await _safe_send(websocket, Frame.error("resume failed"), context="replay-source-except")

    async def revoke(self, room_id: str, user_id: str, *, reason: str = _DEFAULT_REVOKE_REASON) -> int:
        """Take ``user_id`` out of ``room_id`` on every connection this pod holds for them.

        For when an app withdraws someone's access to a room they are already in -- a room made
        private, a person removed from a share. The join gate cannot help with that: it ran when
        they joined, and without an eviction they keep receiving everything streamed to the room.

        Each of the user's connections that is in the room leaves it exactly as a ``leave`` frame
        would, then receives an ``error`` frame carrying ``reason`` and the ``room``, so the client
        can tell the room was closed to it rather than silently going quiet. The socket stays open
        and the user's other rooms are untouched; a later ``join`` is decided afresh.

        Pod-local by construction, like :meth:`disconnect_user`: the live connections are held
        here. A deployment running several pods revokes everywhere by having each pod call this
        from whatever it already broadcasts on.

        :param room_id: the room to take the user out of.
        :ptype room_id: str
        :param user_id: the authenticated user to take out.
        :ptype user_id: str
        :param reason: text of the ``error`` frame the evicted connection receives.
        :ptype reason: str
        :return: how many of the user's connections on this pod were taken out of the room.
        :rtype: int
        """
        evicted = 0
        for connection_id, connection in self._connection_snapshot():
            if connection.user_id != user_id:
                continue
            async with connection.lock:
                if room_id in connection.joined_rooms:
                    await self._evict_locked(room_id, connection_id, connection, reason)
                    evicted += 1
        if evicted:
            log.info(
                "revoked a user's room membership",
                extra={"extra_data": {"room_id": room_id, "user_id": user_id, "evicted": evicted}},
            )
        return evicted

    async def reevaluate_room(self, room_id: str, *, reason: str = _DEFAULT_REVOKE_REASON) -> int:
        """Re-decide every member of ``room_id`` on this pod, and take out each one now refused.

        Each connection in the room is asked the question its ``join`` was: the namespace gate and
        the ``room_policy``, with the ``join_action``. One the gates now refuse leaves the room as
        :meth:`revoke` takes it out. Call this when a room's visibility tightens and the app does
        not know, or does not want to enumerate, who was in it.

        A gate that raises for a member is a refusal for that member -- access that cannot be
        confirmed is not kept -- and is logged; the others are still decided.

        Pod-local by construction, like :meth:`revoke`.

        :param room_id: the room whose members to re-decide.
        :ptype room_id: str
        :param reason: text of the ``error`` frame an evicted connection receives.
        :ptype reason: str
        :return: how many connections on this pod were taken out of the room.
        :rtype: int
        """
        evicted = 0
        for connection_id, connection in self._connection_snapshot():
            async with connection.lock:
                if room_id not in connection.joined_rooms:
                    continue
                try:
                    allowed = await self._decide(room_id, self._join_action, connection.user_id, connection.customer_id)
                except Exception:  # prawduct:allow prawduct/broad-except -- a member whose access cannot be confirmed is evicted, not kept; logged, and the room's other members are still decided
                    log.warning(
                        "room access could not be re-decided for a member; evicting",
                        exc_info=True,
                        extra={"extra_data": {"room_id": room_id, "user_id": connection.user_id}},
                    )
                    allowed = False
                if not allowed:
                    await self._evict_locked(room_id, connection_id, connection, reason)
                    evicted += 1
        if evicted:
            log.info(
                "re-decided a room's members and evicted the refused",
                extra={"extra_data": {"room_id": room_id, "evicted": evicted}},
            )
        return evicted

    def _connection_snapshot(self) -> list[tuple[str, _RoomConnection]]:
        """return a snapshot of this pod's live connections.

        :return: ``(connection_id, connection)`` pairs, copied under the lock
        :rtype: list[tuple[str, _RoomConnection]]
        """
        with self._connections_lock:
            return list(self._connections.items())

    async def _evict_locked(self, room_id: str, connection_id: str, connection: _RoomConnection, reason: str) -> None:
        """take one connection out of a room and tell it. **caller holds ``connection.lock``.**

        the room is dropped from the connection's joined set BEFORE the
        membership write, so nothing the connection sends afterwards is
        treated as coming from a member, and the disconnect path does not
        leave the room a second time.

        :param room_id: the room to leave
        :ptype room_id: str
        :param connection_id: the connection's stable id
        :ptype connection_id: str
        :param connection: the connection's identity and joined rooms (mutated)
        :ptype connection: _RoomConnection
        :param reason: text of the ``error`` frame the connection receives
        :ptype reason: str
        :return: nothing
        :rtype: None
        """
        connection.joined_rooms.discard(room_id)
        if self._room_fanout is not None:
            await self._room_fanout.leave_room(room_id, connection_id)
        await _safe_send(
            connection.websocket,
            json.dumps({"type": "error", "message": reason, "room": room_id}),
            context="room-access-revoked",
        )

    async def disconnect_user(self, user_id: str, *, reason: str = "session ended") -> int:
        """Close every live socket this pod holds for ``user_id``.

        The counterpart to server-side session revocation. A revoked refresh token stops the NEXT
        token being minted and a status check stops the next CONNECT, but neither reaches a socket
        that is already open -- it authenticated once and then just streams. Without this, an
        account disabled mid-session keeps its live editor connection until the peer happens to
        drop it.

        Pod-local by construction: the registry holds handles, which cannot leave the process. A
        deployment running several pods disconnects everywhere by having each pod call this from
        whatever it already broadcasts on (e.g. a
        :class:`~threetears.epoch.EpochListener` reload callback carrying the user id).

        Best-effort per socket: a handle that raises on send or close is logged and skipped, so one
        wedged peer cannot leave the rest of a user's sockets open.

        :param user_id: the authenticated user whose sockets should be closed.
        :ptype user_id: str
        :param reason: human-readable text delivered as an ``error`` frame before the close, so the
            client can distinguish this from a network drop and route to sign-in rather than retry.
        :ptype reason: str
        :return: how many sockets were closed on this pod.
        :rtype: int
        """
        sockets = self.registry.get_connections(user_id)
        for socket in sockets:
            await self._close_with_error(socket, reason)
        if sockets:
            log.info(
                "disconnected a user's live sockets",
                extra={"extra_data": {"user_id": user_id, "closed": len(sockets)}},
            )
        return len(sockets)

    async def _close_with_error(self, websocket: Any, error_message: str) -> None:
        """send error message and close websocket connection.

        :param websocket: websocket connection to close
        :ptype websocket: Any
        :param error_message: human-readable error description
        :ptype error_message: str
        """
        try:
            await websocket.send_text(json.dumps({"type": "error", "message": error_message}))
        except Exception as exc:  # noqa: BLE001 -- the close below still has to happen
            # The peer never received the reason it is being disconnected, so from its side the
            # connection simply drops. Only this log connects the two.
            log.debug(
                "could not deliver websocket error message before closing",
                extra={"extra_data": {"reason": error_message, "error": str(exc)}},
            )
        try:
            await websocket.close(code=1008)
        except Exception as exc:  # noqa: BLE001 -- nothing further to try
            log.debug(
                "websocket close failed",
                extra={"extra_data": {"error": str(exc)}},
            )
