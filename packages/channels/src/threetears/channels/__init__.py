"""3tears-channels: unified message protocol for channel adapters."""

from __future__ import annotations

# Version derived from pyproject.toml so the metadata is the single
# source of truth -- a future release that bumps pyproject without
# updating ``__init__.py`` can't drift the runtime ``__version__``.
# The except guard handles the rare case where the package isn't
# installed via importlib.metadata (e.g. running directly from a
# checked-out source tree without ``uv sync``); the fallback keeps
# imports working but reports ``unknown`` rather than crashing.
from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _version

try:
    __version__ = _version("3tears-channels")
except _PackageNotFoundError:  # pragma: no cover - dev fallback
    __version__ = "unknown"

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

# The webhook receiver requires the ``webhook`` extra (fastapi +
# 3tears-agent-wake). Guarded the same way as the slack / discord
# adapters so consumers without the extra installed can still import
# the rest of the channels package.
try:
    from threetears.channels.webhook import (  # noqa: F401
        Verifier,
        WebhookReceiver,
        verify_generic_hmac_sha256,
    )

    __all__.extend(["Verifier", "WebhookReceiver", "verify_generic_hmac_sha256"])
except ImportError:
    pass
