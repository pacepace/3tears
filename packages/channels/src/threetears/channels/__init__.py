"""3tears-channels: unified message protocol for channel adapters."""

from __future__ import annotations

__version__ = "0.5.0"

from threetears.channels.formatting import (
    build_discord_embed,
    build_discord_payload,
    build_slack_blocks,
    build_slack_payload,
    should_use_rich_formatting,
)
from threetears.channels.protocol import (
    Attachment,
    ChannelMessage,
    ChannelResponse,
    ChannelRouter,
)
from threetears.channels.websocket import (
    ConnectionRegistry,
    StreamingChannelRouter,
    WebSocketHandler,
    WebSocketProtocol,
)

__all__ = [
    "Attachment",
    "ChannelMessage",
    "ChannelResponse",
    "ChannelRouter",
    "ConnectionRegistry",
    "StreamingChannelRouter",
    "WebSocketHandler",
    "WebSocketProtocol",
    "build_discord_embed",
    "build_discord_payload",
    "build_slack_blocks",
    "build_slack_payload",
    "should_use_rich_formatting",
]

try:
    from threetears.channels.slack import SlackAdapter  # noqa: F401

    __all__.append("SlackAdapter")
except ImportError:
    pass

try:
    from threetears.channels.discord import DiscordAdapter  # noqa: F401

    __all__.append("DiscordAdapter")
except ImportError:
    pass
