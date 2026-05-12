"""unified message protocol for channel adapters.

defines platform-agnostic dataclasses for inbound messages, outbound responses,
file attachments, and routing protocol that channel adapters implement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "Attachment",
    "ChannelMessage",
    "ChannelResponse",
    "ChannelRouter",
]


@dataclass
class Attachment:
    """file or image attachment carried with channel message or response.

    :param filename: name of attached file
    :ptype filename: str
    :param content: raw file content as bytes
    :ptype content: bytes
    :param content_type: MIME type of file content
    :ptype content_type: str
    :param description: alt text for images or brief description
    :ptype description: str | None
    """

    filename: str
    content: bytes
    content_type: str
    description: str | None = None


@dataclass
class ChannelMessage:
    """inbound message from channel adapter to platform.

    normalizes messages from different platforms (slack, discord, websocket)
    into single unified structure for routing and processing.

    :param channel_type: platform identifier (slack, discord, websocket)
    :ptype channel_type: str
    :param content: message text content
    :ptype content: str
    :param sender_id: platform-specific user identifier
    :ptype sender_id: str
    :param sender_name: display name of sender
    :ptype sender_name: str | None
    :param conversation_id: platform thread or conversation reference
    :ptype conversation_id: str | None
    :param channel_id: platform channel or room reference
    :ptype channel_id: str | None
    :param workspace_id: platform workspace or guild reference
    :ptype workspace_id: str | None
    :param attachments: files or images attached to message
    :ptype attachments: list[Attachment]
    :param reply_to_id: identifier of message being replied to
    :ptype reply_to_id: str | None
    :param metadata: platform-specific extras not captured by standard fields
    :ptype metadata: dict[str, Any]
    :param timestamp: message creation time in UTC
    :ptype timestamp: datetime
    :param user_timezone: IANA timezone name for the sending user
        (e.g. ``America/Los_Angeles``). populated by the channel
        adapter from its native source: browser
        ``Intl.DateTimeFormat().resolvedOptions().timeZone`` for
        websocket, ``users.info.tz`` for slack, locale-based fallback
        for discord. None when the adapter cannot resolve a
        per-user timezone. consumers (e.g. the ``current_date``
        builtin) read this off the agent ``CallContext`` to render
        timestamps in the user's local time. per-message rather than
        per-channel because users in shared channels have different
        timezones and one user may travel between messages.
    :ptype user_timezone: str | None
    :param user_locale: BCP 47 locale tag for the sending user
        (e.g. ``en-US``, ``ja-JP``). same per-user, per-message
        semantics as ``user_timezone``: each adapter populates from
        its native source. consumers can use this for number /
        currency / date formatting hints in tool output.
    :ptype user_locale: str | None
    """

    channel_type: str
    content: str
    sender_id: str
    sender_name: str | None = None
    conversation_id: str | None = None
    channel_id: str | None = None
    workspace_id: str | None = None
    attachments: list[Attachment] = field(default_factory=list)
    reply_to_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    user_timezone: str | None = None
    user_locale: str | None = None


@dataclass
class ChannelResponse:
    """outbound response from platform to channel adapter.

    carries response content and routing hints back to originating channel
    for delivery to end user.

    :param content: response text in markdown format
    :ptype content: str
    :param conversation_id: thread to reply in
    :ptype conversation_id: str | None
    :param channel_id: channel to reply in
    :ptype channel_id: str | None
    :param attachments: files to send with response
    :ptype attachments: list[Attachment]
    :param format_hints: rich formatting preferences for channel adapter
    :ptype format_hints: dict[str, Any]
    :param metadata: host-application-provided extras
    :ptype metadata: dict[str, Any]
    """

    content: str
    conversation_id: str | None = None
    channel_id: str | None = None
    attachments: list[Attachment] = field(default_factory=list)
    format_hints: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class ChannelRouter(Protocol):
    """protocol for routing inbound channel messages to processing pipeline.

    implementations receive normalized channel messages and return optional
    responses for delivery back through channel adapter.
    """

    async def route_inbound(self, message: ChannelMessage) -> ChannelResponse | None:
        """route inbound message from channel adapter.

        :param message: normalized inbound message from channel
        :ptype message: ChannelMessage
        :return: response to deliver back through channel, or None if no response
        :rtype: ChannelResponse | None
        """
        ...
