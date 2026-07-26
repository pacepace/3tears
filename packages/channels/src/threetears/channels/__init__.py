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
from threetears.channels.frames import (
    Frame,
    FrameHandler,
    NsEntity,
    NsResolver,
    OpHandler,
    OpRejected,
    OpResult,
    ReplaySource,
)
from threetears.channels.delivery import (
    ChannelDeliveryMessage,
)
from threetears.channels.protocol import (
    Attachment,
    ChannelMessage,
    ChannelResponse,
    ChannelRouter,
)
from threetears.channels.presence import (
    PRESENCE_L1_METADATA,
    PRESENCE_L1_TABLE_NAMES,
    PresenceCollection,
    PresenceConnectionCollection,
    PresenceConnectionEntity,
    PresenceSweeper,
    RoomFanout,
    RoomFrame,
    RoomIndexCollection,
    RoomIndexEntity,
    RoomMember,
    RoomState,
    create_presence_l1_backend,
)
from threetears.channels.websocket import (
    ConnectionRegistry,
    StreamingChannelRouter,
    WebSocketHandler,
    WebSocketProtocol,
)

__all__ = [
    "PRESENCE_L1_METADATA",
    "PRESENCE_L1_TABLE_NAMES",
    "Attachment",
    "ChannelDeliveryMessage",
    "ChannelMessage",
    "ChannelResponse",
    "ChannelRouter",
    "ConnectionRegistry",
    "Frame",
    "FrameHandler",
    "NsEntity",
    "NsResolver",
    "OpHandler",
    "OpRejected",
    "OpResult",
    "PresenceCollection",
    "PresenceConnectionCollection",
    "PresenceConnectionEntity",
    "PresenceSweeper",
    "ReplaySource",
    "RoomFanout",
    "RoomFrame",
    "RoomIndexCollection",
    "RoomIndexEntity",
    "RoomMember",
    "RoomState",
    "StreamingChannelRouter",
    "WebSocketHandler",
    "WebSocketProtocol",
    "build_discord_embed",
    "build_discord_payload",
    "build_slack_blocks",
    "build_slack_payload",
    "create_presence_l1_backend",
    "should_use_rich_formatting",
]


def _log_missing_adapter(name: str, extra: str, exc: ImportError) -> None:
    """Record an optional adapter that did not load.

    A missing extra and a genuinely broken adapter module raise the same ``ImportError`` here, and
    both end with ``name`` simply absent from this package. Without the underlying error the two
    are indistinguishable: the symptom surfaces much later as an ``AttributeError`` at whatever
    call site expected the adapter to exist, pointing nowhere near the real cause.

    :param name: the symbol that did not become available
    :ptype name: str
    :param extra: the packaging extra that provides it
    :ptype extra: str
    :param exc: the import failure being recorded
    :ptype exc: ImportError
    :return: nothing
    :rtype: None
    """
    from threetears.observe import get_logger

    get_logger(__name__).debug(
        "optional channel adapter not loaded",
        extra={"extra_data": {"adapter": name, "packaging_extra": extra, "error": str(exc)}},
    )


try:
    from threetears.channels.slack import SlackAdapter  # noqa: F401

    __all__.append("SlackAdapter")
except ImportError as _exc:
    _log_missing_adapter("SlackAdapter", "slack", _exc)

try:
    from threetears.channels.discord import DiscordAdapter  # noqa: F401

    __all__.append("DiscordAdapter")
except ImportError as _exc:
    _log_missing_adapter("DiscordAdapter", "discord", _exc)

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
except ImportError as _exc:
    _log_missing_adapter("WebhookReceiver", "webhook", _exc)
