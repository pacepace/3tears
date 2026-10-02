"""shared doubles for the websocket handler suites: a socket, an echo router and an auth validator.

a support module rather than a test module, so the websocket suites share these under public
names instead of importing one another's private helpers.
"""

from __future__ import annotations

from typing import Any

from threetears.channels.protocol import ChannelMessage, ChannelResponse

__all__ = ["EchoRouter", "MockWebSocket", "refuse_unauthenticated", "valid_auth"]


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


class EchoRouter:
    """router that echoes message content back."""

    async def route_inbound(self, message: ChannelMessage) -> ChannelResponse | None:
        """answer with the inbound content, prefixed.

        :param message: the inbound message
        :ptype message: ChannelMessage
        :return: the echo
        :rtype: ChannelResponse | None
        """
        return ChannelResponse(content=f"echo: {message.content}")


def refuse_unauthenticated() -> Exception:
    """the refusal a validator raises for a token it cannot verify.

    :return: an ``UNAUTHENTICATED`` refusal
    :rtype: Exception
    """
    from threetears.channels.websocket import UNAUTHENTICATED, WebSocketAuthRefused

    return WebSocketAuthRefused(UNAUTHENTICATED, "authentication required")


async def valid_auth(token: str) -> dict[str, Any]:
    """auth validator that accepts 'valid-token' and returns user payload.

    :param token: the presented token
    :ptype token: str
    :return: the authenticated user's payload
    :rtype: dict[str, Any]
    :raises WebSocketAuthRefused: for any other token
    """
    if token != "valid-token":
        raise refuse_unauthenticated()
    return {"user_id": "user-123", "name": "Test User"}
