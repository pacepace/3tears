"""tests for WebSocketHandler, ConnectionRegistry, WebSocketProtocol, and StreamingChannelRouter."""

from __future__ import annotations

import ast
import asyncio
import inspect
import json
from collections.abc import Awaitable, Callable
from typing import Any

import pytest

from threetears.channels.protocol import ChannelMessage, ChannelResponse, ChannelRouter


# -- MockWebSocket for testing --


class MockWebSocket:
    """mock websocket object conforming to WebSocketProtocol.

    :param messages: ordered list of text messages to return from receive_text
    :ptype messages: list[str] | None
    :param query_params: simulated query parameters (e.g. token)
    :ptype query_params: dict[str, str] | None
    """

    def __init__(
        self,
        messages: list[str] | None = None,
        query_params: dict[str, str] | None = None,
    ) -> None:
        self.messages: list[str] = list(messages or [])
        self.sent: list[str] = []
        self.closed: bool = False
        self.close_code: int | None = None
        self.accepted: bool = False
        self.query_params: dict[str, str] = query_params or {}

    async def accept(self) -> None:
        """accept websocket connection."""
        self.accepted = True

    async def receive_text(self) -> str:
        """return next queued message or raise to simulate disconnect."""
        if not self.messages:
            raise Exception("disconnect")
        return self.messages.pop(0)

    async def send_text(self, data: str) -> None:
        """record sent message."""
        self.sent.append(data)

    async def close(self, code: int = 1000) -> None:
        """close websocket."""
        self.closed = True
        self.close_code = code


# -- Mock routers for testing --


class _EchoRouter:
    """router that echoes message content back."""

    async def route_inbound(
        self, message: ChannelMessage
    ) -> ChannelResponse | None:
        return ChannelResponse(content=f"echo: {message.content}")


class _NullRouter:
    """router that returns None for all messages."""

    async def route_inbound(
        self, message: ChannelMessage
    ) -> ChannelResponse | None:
        return None


# -- Mock auth validators --


async def _valid_auth(token: str) -> dict[str, Any] | None:
    """auth validator that accepts 'valid-token' and returns user payload."""
    if token == "valid-token":
        return {"user_id": "user-123", "name": "Test User"}
    return None


async def _always_reject_auth(token: str) -> dict[str, Any] | None:
    """auth validator that always rejects."""
    return None


# ============================================================
# Enforcement tests
# ============================================================


class TestWebSocketEnforcement:
    """enforcement tests ensuring framework and auth library independence."""

    def test_websocket_module_does_not_import_fastapi(self) -> None:
        """websocket.py must not import fastapi."""
        from threetears.channels import websocket as ws_mod

        source_path = inspect.getfile(ws_mod)
        with open(source_path) as f:
            source = f.read()

        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith(
                        "fastapi"
                    ), f"websocket.py imports fastapi: {alias.name}"
            elif isinstance(node, ast.ImportFrom):
                if node.module is not None:
                    assert not node.module.startswith(
                        "fastapi"
                    ), f"websocket.py imports from fastapi: {node.module}"

    def test_websocket_module_does_not_import_starlette(self) -> None:
        """websocket.py must not import starlette."""
        from threetears.channels import websocket as ws_mod

        source_path = inspect.getfile(ws_mod)
        with open(source_path) as f:
            source = f.read()

        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith(
                        "starlette"
                    ), f"websocket.py imports starlette: {alias.name}"
            elif isinstance(node, ast.ImportFrom):
                if node.module is not None:
                    assert not node.module.startswith(
                        "starlette"
                    ), f"websocket.py imports from starlette: {node.module}"

    def test_websocket_module_does_not_import_jwt(self) -> None:
        """websocket.py must not import jwt or any auth library."""
        from threetears.channels import websocket as ws_mod

        source_path = inspect.getfile(ws_mod)
        with open(source_path) as f:
            source = f.read()

        banned_modules = {"jwt", "jose", "pyjwt", "python_jose", "authlib"}
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert (
                        alias.name not in banned_modules
                    ), f"websocket.py imports auth library: {alias.name}"
            elif isinstance(node, ast.ImportFrom):
                if node.module is not None:
                    root_module = node.module.split(".")[0]
                    assert (
                        root_module not in banned_modules
                    ), f"websocket.py imports from auth library: {node.module}"


# ============================================================
# WebSocketProtocol tests
# ============================================================


class TestWebSocketProtocol:
    """tests for WebSocketProtocol runtime-checkable protocol."""

    def test_protocol_is_runtime_checkable(self) -> None:
        """WebSocketProtocol is a runtime_checkable Protocol."""
        from threetears.channels.websocket import WebSocketProtocol

        assert hasattr(WebSocketProtocol, "__protocol_attrs__") or hasattr(
            WebSocketProtocol, "__abstractmethods__"
        )

    def test_conforming_mock_satisfies_protocol(self) -> None:
        """MockWebSocket satisfies isinstance check for WebSocketProtocol."""
        from threetears.channels.websocket import WebSocketProtocol

        ws = MockWebSocket()
        assert isinstance(ws, WebSocketProtocol)

    def test_non_conforming_class_fails_protocol(self) -> None:
        """class without required methods does not satisfy WebSocketProtocol."""
        from threetears.channels.websocket import WebSocketProtocol

        class _BadWebSocket:
            async def accept(self) -> None:
                pass

            # missing receive_text, send_text, close

        obj = _BadWebSocket()
        assert not isinstance(obj, WebSocketProtocol)


# ============================================================
# ConnectionRegistry tests
# ============================================================


class TestConnectionRegistry:
    """tests for ConnectionRegistry."""

    def test_register_adds_connection(self) -> None:
        """register adds websocket to user's connection list."""
        from threetears.channels.websocket import ConnectionRegistry

        registry = ConnectionRegistry()
        ws = MockWebSocket()
        registry.register("user-1", ws)
        connections = registry.get_connections("user-1")
        assert ws in connections

    def test_unregister_removes_connection(self) -> None:
        """unregister removes specific websocket from user's connection list."""
        from threetears.channels.websocket import ConnectionRegistry

        registry = ConnectionRegistry()
        ws = MockWebSocket()
        registry.register("user-1", ws)
        registry.unregister("user-1", ws)
        connections = registry.get_connections("user-1")
        assert ws not in connections

    def test_get_connections_returns_empty_for_unknown_user(self) -> None:
        """get_connections returns empty list for user with no connections."""
        from threetears.channels.websocket import ConnectionRegistry

        registry = ConnectionRegistry()
        connections = registry.get_connections("unknown-user")
        assert connections == []

    def test_multiple_connections_per_user(self) -> None:
        """user can have multiple active websocket connections."""
        from threetears.channels.websocket import ConnectionRegistry

        registry = ConnectionRegistry()
        ws_a = MockWebSocket()
        ws_b = MockWebSocket()
        registry.register("user-1", ws_a)
        registry.register("user-1", ws_b)
        connections = registry.get_connections("user-1")
        assert len(connections) == 2
        assert ws_a in connections
        assert ws_b in connections

    def test_unregister_nonexistent_connection_is_safe(self) -> None:
        """unregister for connection not in registry does not raise."""
        from threetears.channels.websocket import ConnectionRegistry

        registry = ConnectionRegistry()
        ws = MockWebSocket()
        registry.unregister("user-1", ws)

    def test_unregister_only_removes_specified_connection(self) -> None:
        """unregister removes only the specified connection, not others for same user."""
        from threetears.channels.websocket import ConnectionRegistry

        registry = ConnectionRegistry()
        ws_a = MockWebSocket()
        ws_b = MockWebSocket()
        registry.register("user-1", ws_a)
        registry.register("user-1", ws_b)
        registry.unregister("user-1", ws_a)
        connections = registry.get_connections("user-1")
        assert len(connections) == 1
        assert ws_b in connections

    def test_room_join(self) -> None:
        """join_room adds websocket to room."""
        from threetears.channels.websocket import ConnectionRegistry

        registry = ConnectionRegistry()
        ws = MockWebSocket()
        registry.join_room("room-1", ws)
        # verify by broadcasting
        # (tested more thoroughly in broadcast tests)

    def test_room_leave(self) -> None:
        """leave_room removes websocket from room."""
        from threetears.channels.websocket import ConnectionRegistry

        registry = ConnectionRegistry()
        ws = MockWebSocket()
        registry.join_room("room-1", ws)
        registry.leave_room("room-1", ws)

    def test_leave_room_nonexistent_is_safe(self) -> None:
        """leave_room for websocket not in room does not raise."""
        from threetears.channels.websocket import ConnectionRegistry

        registry = ConnectionRegistry()
        ws = MockWebSocket()
        registry.leave_room("room-1", ws)

    @pytest.mark.asyncio
    async def test_broadcast_to_room_sends_to_all(self) -> None:
        """broadcast_to_room sends message to all connections in room."""
        from threetears.channels.websocket import ConnectionRegistry

        registry = ConnectionRegistry()
        ws_a = MockWebSocket()
        ws_b = MockWebSocket()
        registry.join_room("room-1", ws_a)
        registry.join_room("room-1", ws_b)
        await registry.broadcast_to_room("room-1", "hello room")
        assert "hello room" in ws_a.sent
        assert "hello room" in ws_b.sent

    @pytest.mark.asyncio
    async def test_broadcast_to_room_excludes_specified(self) -> None:
        """broadcast_to_room excludes specified websocket from broadcast."""
        from threetears.channels.websocket import ConnectionRegistry

        registry = ConnectionRegistry()
        ws_a = MockWebSocket()
        ws_b = MockWebSocket()
        registry.join_room("room-1", ws_a)
        registry.join_room("room-1", ws_b)
        await registry.broadcast_to_room("room-1", "hello room", exclude=ws_a)
        assert "hello room" not in ws_a.sent
        assert "hello room" in ws_b.sent

    @pytest.mark.asyncio
    async def test_broadcast_to_empty_room_is_safe(self) -> None:
        """broadcast_to_room on empty or nonexistent room does not raise."""
        from threetears.channels.websocket import ConnectionRegistry

        registry = ConnectionRegistry()
        await registry.broadcast_to_room("empty-room", "hello")


# ============================================================
# WebSocketHandler lifecycle tests
# ============================================================


class TestWebSocketHandlerAuthQueryParam:
    """tests for authentication via query parameter."""

    @pytest.mark.asyncio
    async def test_auth_via_query_param_token(self) -> None:
        """handler authenticates via token query parameter."""
        from threetears.channels.websocket import WebSocketHandler

        router = _EchoRouter()
        handler = WebSocketHandler(router=router, auth_validator=_valid_auth)

        ws = MockWebSocket(
            messages=[],
            query_params={"token": "valid-token"},
        )
        await handler.handle_connection(ws)

        assert ws.accepted
        # first sent message should be connected message
        assert len(ws.sent) >= 1
        connected_msg = json.loads(ws.sent[0])
        assert connected_msg["type"] == "connected"
        assert connected_msg["user_id"] == "user-123"


class TestWebSocketHandlerAuthMessage:
    """tests for authentication via first message."""

    @pytest.mark.asyncio
    async def test_auth_via_first_message(self) -> None:
        """handler authenticates via auth message when no query param."""
        from threetears.channels.websocket import WebSocketHandler

        router = _EchoRouter()
        handler = WebSocketHandler(router=router, auth_validator=_valid_auth)

        auth_msg = json.dumps({"type": "auth", "token": "valid-token"})
        ws = MockWebSocket(messages=[auth_msg])
        await handler.handle_connection(ws)

        assert ws.accepted
        assert len(ws.sent) >= 1
        connected_msg = json.loads(ws.sent[0])
        assert connected_msg["type"] == "connected"
        assert connected_msg["user_id"] == "user-123"


class TestWebSocketHandlerAuthFailure:
    """tests for authentication failure scenarios."""

    @pytest.mark.asyncio
    async def test_auth_failure_query_param_closes_connection(self) -> None:
        """handler closes connection on invalid query param token."""
        from threetears.channels.websocket import WebSocketHandler

        router = _EchoRouter()
        handler = WebSocketHandler(
            router=router, auth_validator=_always_reject_auth
        )

        auth_msg = json.dumps({"type": "auth", "token": "bad-token"})
        ws = MockWebSocket(
            messages=[auth_msg],
            query_params={"token": "bad-token"},
        )
        await handler.handle_connection(ws)

        assert ws.accepted
        # should have sent error and closed
        error_sent = False
        for msg_text in ws.sent:
            parsed = json.loads(msg_text)
            if parsed.get("type") == "error":
                error_sent = True
                break
        assert error_sent
        assert ws.closed

    @pytest.mark.asyncio
    async def test_auth_failure_message_closes_connection(self) -> None:
        """handler closes connection on invalid auth message token."""
        from threetears.channels.websocket import WebSocketHandler

        router = _EchoRouter()
        handler = WebSocketHandler(
            router=router, auth_validator=_always_reject_auth
        )

        auth_msg = json.dumps({"type": "auth", "token": "bad-token"})
        ws = MockWebSocket(messages=[auth_msg])
        await handler.handle_connection(ws)

        assert ws.accepted
        error_sent = False
        for msg_text in ws.sent:
            parsed = json.loads(msg_text)
            if parsed.get("type") == "error":
                error_sent = True
                break
        assert error_sent
        assert ws.closed


class TestWebSocketHandlerConnectedMessage:
    """tests for connected confirmation message."""

    @pytest.mark.asyncio
    async def test_successful_auth_sends_connected_message(self) -> None:
        """successful auth sends connected message with user_id."""
        from threetears.channels.websocket import WebSocketHandler

        router = _EchoRouter()
        handler = WebSocketHandler(router=router, auth_validator=_valid_auth)

        ws = MockWebSocket(
            messages=[],
            query_params={"token": "valid-token"},
        )
        await handler.handle_connection(ws)

        assert len(ws.sent) >= 1
        connected_msg = json.loads(ws.sent[0])
        assert connected_msg["type"] == "connected"
        assert connected_msg["user_id"] == "user-123"


class TestWebSocketHandlerMessageLoop:
    """tests for the message processing loop."""

    @pytest.mark.asyncio
    async def test_message_loop_routes_and_responds(self) -> None:
        """message loop creates ChannelMessage and sends router response."""
        from threetears.channels.websocket import WebSocketHandler

        router = _EchoRouter()
        handler = WebSocketHandler(router=router, auth_validator=_valid_auth)

        user_msg = json.dumps(
            {"type": "message", "content": "hello", "metadata": {}}
        )
        ws = MockWebSocket(
            messages=[user_msg],
            query_params={"token": "valid-token"},
        )
        await handler.handle_connection(ws)

        # sent[0] = connected, sent[1] = response
        assert len(ws.sent) >= 2
        response_msg = json.loads(ws.sent[1])
        assert response_msg["type"] == "response"
        assert response_msg["content"] == "echo: hello"

    @pytest.mark.asyncio
    async def test_router_returning_none_sends_no_response(self) -> None:
        """when router returns None, no response message is sent."""
        from threetears.channels.websocket import WebSocketHandler

        router = _NullRouter()
        handler = WebSocketHandler(router=router, auth_validator=_valid_auth)

        user_msg = json.dumps(
            {"type": "message", "content": "hello", "metadata": {}}
        )
        ws = MockWebSocket(
            messages=[user_msg],
            query_params={"token": "valid-token"},
        )
        await handler.handle_connection(ws)

        # only connected message should be sent, no response
        assert len(ws.sent) == 1
        connected_msg = json.loads(ws.sent[0])
        assert connected_msg["type"] == "connected"

    @pytest.mark.asyncio
    async def test_multiple_messages_in_sequence(self) -> None:
        """handler processes multiple messages in sequence."""
        from threetears.channels.websocket import WebSocketHandler

        router = _EchoRouter()
        handler = WebSocketHandler(router=router, auth_validator=_valid_auth)

        msg_a = json.dumps(
            {"type": "message", "content": "first", "metadata": {}}
        )
        msg_b = json.dumps(
            {"type": "message", "content": "second", "metadata": {}}
        )
        ws = MockWebSocket(
            messages=[msg_a, msg_b],
            query_params={"token": "valid-token"},
        )
        await handler.handle_connection(ws)

        # connected + 2 responses
        assert len(ws.sent) >= 3
        resp_a = json.loads(ws.sent[1])
        resp_b = json.loads(ws.sent[2])
        assert resp_a["content"] == "echo: first"
        assert resp_b["content"] == "echo: second"

    @pytest.mark.asyncio
    async def test_message_with_metadata(self) -> None:
        """handler passes metadata through to ChannelMessage."""
        from threetears.channels.websocket import WebSocketHandler

        received_messages: list[ChannelMessage] = []

        class _CapturingRouter:
            async def route_inbound(
                self, message: ChannelMessage
            ) -> ChannelResponse | None:
                received_messages.append(message)
                return ChannelResponse(content="ok")

        router = _CapturingRouter()
        handler = WebSocketHandler(router=router, auth_validator=_valid_auth)

        user_msg = json.dumps(
            {
                "type": "message",
                "content": "test",
                "metadata": {"custom_key": "custom_value"},
            }
        )
        ws = MockWebSocket(
            messages=[user_msg],
            query_params={"token": "valid-token"},
        )
        await handler.handle_connection(ws)

        assert len(received_messages) == 1
        assert received_messages[0].metadata.get("custom_key") == "custom_value"


class TestWebSocketHandlerDisconnect:
    """tests for disconnect and cleanup behavior."""

    @pytest.mark.asyncio
    async def test_disconnect_cleans_up_from_registry(self) -> None:
        """disconnecting client is removed from connection registry."""
        from threetears.channels.websocket import WebSocketHandler

        router = _EchoRouter()
        handler = WebSocketHandler(router=router, auth_validator=_valid_auth)

        ws = MockWebSocket(
            messages=[],
            query_params={"token": "valid-token"},
        )
        await handler.handle_connection(ws)

        # after handle_connection returns, user should be cleaned up
        connections = handler.registry.get_connections("user-123")
        assert ws not in connections

    @pytest.mark.asyncio
    async def test_disconnect_during_message_loop_cleans_up(self) -> None:
        """unexpected disconnect during message loop still cleans up registry."""
        from threetears.channels.websocket import WebSocketHandler

        router = _EchoRouter()
        handler = WebSocketHandler(router=router, auth_validator=_valid_auth)

        user_msg = json.dumps(
            {"type": "message", "content": "hello", "metadata": {}}
        )
        ws = MockWebSocket(
            messages=[user_msg],
            query_params={"token": "valid-token"},
        )
        await handler.handle_connection(ws)

        connections = handler.registry.get_connections("user-123")
        assert ws not in connections


class TestWebSocketHandlerChannelMessage:
    """tests for ChannelMessage construction from websocket messages."""

    @pytest.mark.asyncio
    async def test_channel_message_has_websocket_channel_type(self) -> None:
        """ChannelMessage created from websocket has channel_type 'websocket'."""
        from threetears.channels.websocket import WebSocketHandler

        received_messages: list[ChannelMessage] = []

        class _CapturingRouter:
            async def route_inbound(
                self, message: ChannelMessage
            ) -> ChannelResponse | None:
                received_messages.append(message)
                return None

        router = _CapturingRouter()
        handler = WebSocketHandler(router=router, auth_validator=_valid_auth)

        user_msg = json.dumps(
            {"type": "message", "content": "hello", "metadata": {}}
        )
        ws = MockWebSocket(
            messages=[user_msg],
            query_params={"token": "valid-token"},
        )
        await handler.handle_connection(ws)

        assert len(received_messages) == 1
        assert received_messages[0].channel_type == "websocket"
        assert received_messages[0].sender_id == "user-123"
        assert received_messages[0].content == "hello"


class TestWebSocketHandlerConfig:
    """tests for handler configuration."""

    @pytest.mark.asyncio
    async def test_handler_accepts_config(self) -> None:
        """handler accepts optional config dict."""
        from threetears.channels.websocket import WebSocketHandler

        router = _EchoRouter()
        config = {"heartbeat_interval": 15}
        handler = WebSocketHandler(
            router=router, auth_validator=_valid_auth, config=config
        )
        assert handler._config["heartbeat_interval"] == 15

    @pytest.mark.asyncio
    async def test_handler_default_config(self) -> None:
        """handler uses empty dict when no config provided."""
        from threetears.channels.websocket import WebSocketHandler

        router = _EchoRouter()
        handler = WebSocketHandler(router=router, auth_validator=_valid_auth)
        assert isinstance(handler._config, dict)


# ============================================================
# StreamingChannelRouter tests
# ============================================================


class TestStreamingChannelRouter:
    """tests for StreamingChannelRouter protocol."""

    def test_protocol_is_runtime_checkable(self) -> None:
        """StreamingChannelRouter is a runtime_checkable Protocol."""
        from threetears.channels.websocket import StreamingChannelRouter

        assert hasattr(
            StreamingChannelRouter, "__protocol_attrs__"
        ) or hasattr(StreamingChannelRouter, "__abstractmethods__")

    def test_conforming_class_satisfies_protocol(self) -> None:
        """class implementing route_inbound_streaming satisfies protocol."""
        from threetears.channels.websocket import StreamingChannelRouter

        class _StreamRouter:
            async def route_inbound_streaming(
                self,
                message: ChannelMessage,
                send: Callable[[str], Awaitable[None]],
            ) -> ChannelResponse | None:
                return None

        router = _StreamRouter()
        assert isinstance(router, StreamingChannelRouter)

    def test_non_conforming_class_fails_protocol(self) -> None:
        """class without route_inbound_streaming does not satisfy protocol."""
        from threetears.channels.websocket import StreamingChannelRouter

        class _NotStreaming:
            async def some_method(self) -> None:
                pass

        obj = _NotStreaming()
        assert not isinstance(obj, StreamingChannelRouter)

    @pytest.mark.asyncio
    async def test_streaming_router_sends_tokens_via_callback(self) -> None:
        """streaming router sends individual tokens via send callback."""
        from threetears.channels.websocket import StreamingChannelRouter

        class _TokenStreamRouter:
            async def route_inbound_streaming(
                self,
                message: ChannelMessage,
                send: Callable[[str], Awaitable[None]],
            ) -> ChannelResponse | None:
                tokens = ["Hello", " ", "world", "!"]
                for token in tokens:
                    await send(token)
                return ChannelResponse(content="Hello world!")

        router = _TokenStreamRouter()
        assert isinstance(router, StreamingChannelRouter)

        sent_tokens: list[str] = []

        async def capture_send(token: str) -> None:
            sent_tokens.append(token)

        msg = ChannelMessage(
            channel_type="websocket",
            content="say hello",
            sender_id="user-1",
        )
        response = await router.route_inbound_streaming(msg, capture_send)

        assert sent_tokens == ["Hello", " ", "world", "!"]
        assert response is not None
        assert response.content == "Hello world!"

    @pytest.mark.asyncio
    async def test_streaming_router_can_return_none(self) -> None:
        """streaming router can return None when no final response needed."""
        from threetears.channels.websocket import StreamingChannelRouter

        class _NullStreamRouter:
            async def route_inbound_streaming(
                self,
                message: ChannelMessage,
                send: Callable[[str], Awaitable[None]],
            ) -> ChannelResponse | None:
                await send("partial")
                return None

        router = _NullStreamRouter()
        sent_tokens: list[str] = []

        async def capture_send(token: str) -> None:
            sent_tokens.append(token)

        msg = ChannelMessage(
            channel_type="websocket",
            content="test",
            sender_id="user-1",
        )
        response = await router.route_inbound_streaming(msg, capture_send)

        assert sent_tokens == ["partial"]
        assert response is None


# ============================================================
# WebSocketHandler with StreamingChannelRouter tests
# ============================================================


class TestWebSocketHandlerStreaming:
    """tests for handler behavior when router supports streaming."""

    @pytest.mark.asyncio
    async def test_streaming_router_sends_tokens_to_client(self) -> None:
        """handler sends streaming tokens to websocket client when router supports streaming."""
        from threetears.channels.websocket import WebSocketHandler

        class _StreamRouter:
            async def route_inbound(
                self, message: ChannelMessage
            ) -> ChannelResponse | None:
                return ChannelResponse(content="full response")

            async def route_inbound_streaming(
                self,
                message: ChannelMessage,
                send: Callable[[str], Awaitable[None]],
            ) -> ChannelResponse | None:
                await send("tok1")
                await send("tok2")
                return ChannelResponse(content="tok1tok2")

        router = _StreamRouter()
        handler = WebSocketHandler(router=router, auth_validator=_valid_auth)

        user_msg = json.dumps(
            {"type": "message", "content": "hello", "metadata": {}}
        )
        ws = MockWebSocket(
            messages=[user_msg],
            query_params={"token": "valid-token"},
        )
        await handler.handle_connection(ws)

        # connected msg + streaming tokens + final response
        # tokens are sent as type "stream" messages
        stream_messages = [
            json.loads(m)
            for m in ws.sent
            if json.loads(m).get("type") == "stream"
        ]
        assert len(stream_messages) == 2
        assert stream_messages[0]["content"] == "tok1"
        assert stream_messages[1]["content"] == "tok2"

        # final response also sent
        response_messages = [
            json.loads(m)
            for m in ws.sent
            if json.loads(m).get("type") == "response"
        ]
        assert len(response_messages) == 1
        assert response_messages[0]["content"] == "tok1tok2"


# ============================================================
# Heartbeat / keepalive tests
# ============================================================


class TestWebSocketHandlerHeartbeat:
    """tests for heartbeat/keepalive mechanism."""

    @pytest.mark.asyncio
    async def test_heartbeat_config_default_interval(self) -> None:
        """default heartbeat interval is 30 seconds."""
        from threetears.channels.websocket import WebSocketHandler

        router = _EchoRouter()
        handler = WebSocketHandler(router=router, auth_validator=_valid_auth)
        assert handler._heartbeat_interval == 30

    @pytest.mark.asyncio
    async def test_heartbeat_config_custom_interval(self) -> None:
        """heartbeat interval can be configured."""
        from threetears.channels.websocket import WebSocketHandler

        router = _EchoRouter()
        handler = WebSocketHandler(
            router=router,
            auth_validator=_valid_auth,
            config={"heartbeat_interval": 15},
        )
        assert handler._heartbeat_interval == 15


# ============================================================
# Rate limiting and message size enforcement tests
# ============================================================


class TestWebSocketHandlerMessageSizeEnforcement:
    """tests for message size enforcement in message loop."""

    @pytest.mark.asyncio
    async def test_message_size_enforcement_rejects_large_message(self) -> None:
        """message exceeding max_message_size gets error response."""
        from threetears.channels.websocket import WebSocketHandler

        router = _EchoRouter()
        handler = WebSocketHandler(
            router=router,
            auth_validator=_valid_auth,
            config={"max_message_size": 100},
        )

        large_content = "x" * 200
        large_msg = json.dumps(
            {"type": "message", "content": large_content, "metadata": {}}
        )
        ws = MockWebSocket(
            messages=[large_msg],
            query_params={"token": "valid-token"},
        )
        await handler.handle_connection(ws)

        # should have connected message and error message
        error_messages = [
            json.loads(m)
            for m in ws.sent
            if json.loads(m).get("type") == "error"
        ]
        assert len(error_messages) == 1
        assert "too large" in error_messages[0]["message"]

        # should NOT have a response message (message was rejected)
        response_messages = [
            json.loads(m)
            for m in ws.sent
            if json.loads(m).get("type") == "response"
        ]
        assert len(response_messages) == 0

    @pytest.mark.asyncio
    async def test_message_size_enforcement_accepts_normal_message(self) -> None:
        """normal size message is processed successfully."""
        from threetears.channels.websocket import WebSocketHandler

        router = _EchoRouter()
        handler = WebSocketHandler(
            router=router,
            auth_validator=_valid_auth,
            config={"max_message_size": 65536},
        )

        normal_msg = json.dumps(
            {"type": "message", "content": "hello", "metadata": {}}
        )
        ws = MockWebSocket(
            messages=[normal_msg],
            query_params={"token": "valid-token"},
        )
        await handler.handle_connection(ws)

        response_messages = [
            json.loads(m)
            for m in ws.sent
            if json.loads(m).get("type") == "response"
        ]
        assert len(response_messages) == 1
        assert response_messages[0]["content"] == "echo: hello"

    @pytest.mark.asyncio
    async def test_custom_message_size_config(self) -> None:
        """config overrides default max_message_size."""
        from threetears.channels.websocket import WebSocketHandler

        router = _EchoRouter()
        handler = WebSocketHandler(
            router=router,
            auth_validator=_valid_auth,
            config={"max_message_size": 1024},
        )
        assert handler._max_message_size == 1024


class TestWebSocketHandlerRateLimiting:
    """tests for rate limiting in message loop."""

    @pytest.mark.asyncio
    async def test_rate_limit_rejects_excess_messages(self) -> None:
        """messages exceeding rate limit get error response."""
        from threetears.channels.websocket import WebSocketHandler

        router = _EchoRouter()
        handler = WebSocketHandler(
            router=router,
            auth_validator=_valid_auth,
            config={
                "rate_limit_messages": 2,
                "rate_limit_window": 60.0,
            },
        )

        # send 4 messages -- first 2 should succeed, last 2 rate-limited
        messages = []
        for i in range(4):
            messages.append(
                json.dumps(
                    {"type": "message", "content": f"msg-{i}", "metadata": {}}
                )
            )

        ws = MockWebSocket(
            messages=messages,
            query_params={"token": "valid-token"},
        )
        await handler.handle_connection(ws)

        response_messages = [
            json.loads(m)
            for m in ws.sent
            if json.loads(m).get("type") == "response"
        ]
        error_messages = [
            json.loads(m)
            for m in ws.sent
            if json.loads(m).get("type") == "error"
        ]

        assert len(response_messages) == 2
        assert len(error_messages) == 2
        for err in error_messages:
            assert "rate limit" in err["message"]

    @pytest.mark.asyncio
    async def test_rate_limit_resets_after_window(self) -> None:
        """after rate limit window passes, messages are accepted again."""
        from threetears.channels.websocket import WebSocketHandler

        router = _EchoRouter()
        handler = WebSocketHandler(
            router=router,
            auth_validator=_valid_auth,
            config={
                "rate_limit_messages": 2,
                "rate_limit_window": 0.01,
            },
        )

        # send 2 messages (exhaust rate limit)
        msg_a = json.dumps(
            {"type": "message", "content": "first", "metadata": {}}
        )
        msg_b = json.dumps(
            {"type": "message", "content": "second", "metadata": {}}
        )
        # then wait for window to expire (handled by tiny window) and send another
        msg_c = json.dumps(
            {"type": "message", "content": "third", "metadata": {}}
        )

        # use a custom websocket that sleeps between messages
        class _SlowMockWebSocket(MockWebSocket):
            """mock websocket that sleeps before delivering third message."""

            def __init__(self, messages: list[str], query_params: dict[str, str]) -> None:
                super().__init__(messages=messages, query_params=query_params)
                self._deliver_count = 0

            async def receive_text(self) -> str:
                """return next message, sleeping before the third one."""
                if not self.messages:
                    raise Exception("disconnect")
                self._deliver_count += 1
                if self._deliver_count == 3:
                    await asyncio.sleep(0.05)
                return self.messages.pop(0)

        ws = _SlowMockWebSocket(
            messages=[msg_a, msg_b, msg_c],
            query_params={"token": "valid-token"},
        )
        await handler.handle_connection(ws)

        response_messages = [
            json.loads(m)
            for m in ws.sent
            if json.loads(m).get("type") == "response"
        ]
        # all 3 should succeed because window resets before third message
        assert len(response_messages) == 3

    @pytest.mark.asyncio
    async def test_custom_rate_limit_config(self) -> None:
        """config overrides default rate limit settings."""
        from threetears.channels.websocket import WebSocketHandler

        router = _EchoRouter()
        handler = WebSocketHandler(
            router=router,
            auth_validator=_valid_auth,
            config={
                "rate_limit_messages": 20,
                "rate_limit_window": 5.0,
            },
        )
        assert handler._rate_limit_messages == 20
        assert handler._rate_limit_window == 5.0
