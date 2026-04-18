"""framework-agnostic websocket handler for channel adapters.

provides WebSocketHandler for managing websocket connection lifecycle
including authentication, message routing, and optional streaming.
ConnectionRegistry tracks active connections and supports optional room
broadcasting. all websocket interaction goes through WebSocketProtocol
so the handler works with starlette, fastapi, or any conforming object.
"""

from __future__ import annotations

import json
import time
from collections.abc import Awaitable, Callable
from typing import Any, Protocol, runtime_checkable

from threetears.channels.protocol import ChannelMessage, ChannelResponse
from threetears.observe import get_logger

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
    """tracks active websocket connections by user and optional rooms.

    provides user-level connection tracking for the websocket handler
    and optional room-based broadcasting for collaborative features.
    consumers that do not need rooms can ignore room methods entirely.
    """

    def __init__(self) -> None:
        """initialize empty connection and room registries."""
        self._connections: dict[str, list[Any]] = {}
        self._rooms: dict[str, list[Any]] = {}

    def register(self, user_id: str, websocket: Any) -> None:
        """add websocket connection for specified user.

        :param user_id: identifier of authenticated user
        :ptype user_id: str
        :param websocket: websocket connection to register
        :ptype websocket: Any
        """
        if user_id not in self._connections:
            self._connections[user_id] = []
        self._connections[user_id].append(websocket)

    def unregister(self, user_id: str, websocket: Any) -> None:
        """remove websocket connection for specified user.

        safe to call when user_id or websocket is not registered.

        :param user_id: identifier of authenticated user
        :ptype user_id: str
        :param websocket: websocket connection to remove
        :ptype websocket: Any
        """
        connections = self._connections.get(user_id)
        if connections is None:
            return
        try:
            connections.remove(websocket)
        except ValueError:
            pass

    def get_connections(self, user_id: str) -> list[Any]:
        """return list of active websocket connections for user.

        returns empty list if user has no active connections.

        :param user_id: identifier of authenticated user
        :ptype user_id: str
        :return: list of active websocket connections
        :rtype: list[Any]
        """
        result = list(self._connections.get(user_id, []))
        return result

    def join_room(self, room_id: str, websocket: Any) -> None:
        """add websocket connection to specified room.

        :param room_id: identifier of room to join
        :ptype room_id: str
        :param websocket: websocket connection to add to room
        :ptype websocket: Any
        """
        if room_id not in self._rooms:
            self._rooms[room_id] = []
        self._rooms[room_id].append(websocket)

    def leave_room(self, room_id: str, websocket: Any) -> None:
        """remove websocket connection from specified room.

        safe to call when room_id or websocket is not in room.

        :param room_id: identifier of room to leave
        :ptype room_id: str
        :param websocket: websocket connection to remove from room
        :ptype websocket: Any
        """
        members = self._rooms.get(room_id)
        if members is None:
            return
        try:
            members.remove(websocket)
        except ValueError:
            pass

    async def broadcast_to_room(
        self,
        room_id: str,
        message: str,
        exclude: Any | None = None,
    ) -> None:
        """send message to all websocket connections in room.

        optionally excludes one connection from the broadcast
        (typically the sender). safe to call on empty or
        nonexistent rooms.

        :param room_id: identifier of room to broadcast to
        :ptype room_id: str
        :param message: text message to broadcast
        :ptype message: str
        :param exclude: websocket connection to exclude from broadcast
        :ptype exclude: Any | None
        """
        members = self._rooms.get(room_id, [])
        for ws in members:
            if ws is exclude:
                continue
            try:
                await ws.send_text(message)
            except Exception:
                log.warning(
                    "failed to broadcast to room member in room %s",
                    room_id,
                )


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
    ) -> None:
        """initialize websocket handler with router, auth validator, and config.

        config keys:
          - heartbeat_interval: seconds between heartbeat pings (default 30)
          - max_message_size: maximum inbound message bytes (default 65536)
          - rate_limit_messages: max messages per window (default 10)
          - rate_limit_window: sliding window duration in seconds (default 1.0)

        :param router: channel router for processing inbound messages
        :ptype router: ChannelRouter-conforming object
        :param auth_validator: callable that validates JWT token string and
            returns decoded payload dict or None if invalid
        :ptype auth_validator: Callable[[str], Awaitable[dict | None]]
        :param config: optional handler configuration overrides
        :ptype config: dict[str, Any] | None
        """
        self._router = router
        self._auth_validator = auth_validator
        self._config: dict[str, Any] = config if config is not None else {}
        self._heartbeat_interval: int = self._config.get("heartbeat_interval", _DEFAULT_HEARTBEAT_INTERVAL)
        self._max_message_size: int = self._config.get("max_message_size", _DEFAULT_MAX_MESSAGE_SIZE)
        self._rate_limit_messages: int = self._config.get("rate_limit_messages", _DEFAULT_RATE_LIMIT_MESSAGES)
        self._rate_limit_window: float = self._config.get("rate_limit_window", _DEFAULT_RATE_LIMIT_WINDOW)
        self.registry = ConnectionRegistry()

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

        await websocket.send_text(json.dumps({"type": "connected", "user_id": user_id}))

        self.registry.register(user_id, websocket)
        try:
            await self._message_loop(websocket, user_id)
        finally:
            self.registry.unregister(user_id, websocket)

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

    async def _message_loop(self, websocket: Any, user_id: str) -> None:
        """process inbound messages until disconnect or error.

        receives JSON messages, creates ChannelMessage, routes through
        router (with streaming if supported), and sends responses.
        enforces message size limits and sliding-window rate limiting
        before processing each message.

        :param websocket: authenticated websocket connection
        :ptype websocket: Any
        :param user_id: identifier of authenticated user
        :ptype user_id: str
        """
        is_streaming = isinstance(self._router, StreamingChannelRouter)

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

            if len(raw) > self._max_message_size:
                await websocket.send_text(json.dumps({"type": "error", "message": "message too large"}))
                continue

            now = time.monotonic()
            if now - rate_window_start >= self._rate_limit_window:
                rate_window_start = now
                rate_message_count = 0
            rate_message_count += 1
            if rate_message_count > self._rate_limit_messages:
                await websocket.send_text(json.dumps({"type": "error", "message": "rate limit exceeded"}))
                continue

            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                log.warning(
                    "received non-JSON websocket message from user %s",
                    user_id,
                )
                continue

            msg_type = data.get("type", "")
            if msg_type != "message":
                continue

            content = data.get("content", "")
            metadata = data.get("metadata", {})

            channel_message = ChannelMessage(
                channel_type="websocket",
                content=content,
                sender_id=user_id,
                metadata=metadata,
            )

            if is_streaming:
                await self._route_streaming(websocket, channel_message)
            else:
                await self._route_standard(websocket, channel_message)

    async def _route_standard(self, websocket: Any, message: ChannelMessage) -> None:
        """route message through standard (non-streaming) router.

        :param websocket: websocket connection to send response to
        :ptype websocket: Any
        :param message: normalized inbound message
        :ptype message: ChannelMessage
        """
        response = await self._router.route_inbound(message)
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

        response = await self._router.route_inbound_streaming(message, send_token)
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
