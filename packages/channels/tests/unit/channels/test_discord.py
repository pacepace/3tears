"""tests for DiscordAdapter channel adapter."""

from __future__ import annotations

import ast
import inspect
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from threetears.channels.protocol import (
    Attachment,
    ChannelMessage,
    ChannelResponse,
    ChannelRouter,
)


# ---------------------------------------------------------------------------
# helper: mock router conforming to ChannelRouter protocol
# ---------------------------------------------------------------------------


class _MockRouter:
    """mock router that records calls and returns configurable responses."""

    def __init__(self, response: ChannelResponse | None = None) -> None:
        self.last_message: ChannelMessage | None = None
        self._response = response

    async def route_inbound(self, message: ChannelMessage) -> ChannelResponse | None:
        """record inbound message and return configured response.

        :param message: normalized inbound message from channel
        :ptype message: ChannelMessage
        :return: configured response or None
        :rtype: ChannelResponse | None
        """
        self.last_message = message
        return self._response


# ---------------------------------------------------------------------------
# sentinel types used for isinstance checks when discord module is mocked
# ---------------------------------------------------------------------------


class _FakeThread:
    """sentinel type standing in for discord.Thread in mocked tests."""


class _FakeDMChannel:
    """sentinel type standing in for discord.DMChannel in mocked tests."""


_UNSET = object()
"""sentinel distinguishing 'not provided' from explicit None."""


# ---------------------------------------------------------------------------
# helper: build mock discord objects
# ---------------------------------------------------------------------------


def _make_mock_author(
    *,
    author_id: int = 123456789,
    display_name: str = "TestUser",
    bot: bool = False,
) -> MagicMock:
    """build mock discord.Member or discord.User.

    :param author_id: discord snowflake id for author
    :ptype author_id: int
    :param display_name: display name of author
    :ptype display_name: str
    :param bot: whether author is bot
    :ptype bot: bool
    :return: mock author object
    :rtype: MagicMock
    """
    author = MagicMock()
    author.id = author_id
    author.display_name = display_name
    author.bot = bot
    return author


def _make_mock_guild(*, guild_id: int = 111222333) -> MagicMock:
    """build mock discord.Guild.

    :param guild_id: discord snowflake id for guild
    :ptype guild_id: int
    :return: mock guild object
    :rtype: MagicMock
    """
    guild = MagicMock()
    guild.id = guild_id
    return guild


def _make_mock_channel(
    *,
    channel_id: int = 987654321,
    is_thread: bool = False,
) -> MagicMock:
    """build mock discord.TextChannel or discord.Thread.

    uses sentinel types (_FakeThread) so that isinstance checks work
    when discord module is mocked.

    :param channel_id: discord snowflake id for channel
    :ptype channel_id: int
    :param is_thread: whether channel should pass isinstance(ch, discord.Thread)
    :ptype is_thread: bool
    :return: mock channel object
    :rtype: MagicMock
    """
    channel = MagicMock()
    channel.id = channel_id
    channel.send = AsyncMock()

    if is_thread:
        channel.__class__ = _FakeThread

    return channel


def _make_mock_dm_channel(*, channel_id: int = 555666777) -> MagicMock:
    """build mock discord.DMChannel.

    uses sentinel type (_FakeDMChannel) so that isinstance checks work
    when discord module is mocked.

    :param channel_id: discord snowflake id for DM channel
    :ptype channel_id: int
    :return: mock DM channel object
    :rtype: MagicMock
    """
    channel = MagicMock()
    channel.id = channel_id
    channel.send = AsyncMock()
    channel.__class__ = _FakeDMChannel
    return channel


def _make_mock_attachment(
    *,
    attachment_id: int = 444555666,
    filename: str = "image.png",
    content_type: str = "image/png",
    size: int = 1024,
    url: str = "https://cdn.discordapp.com/attachments/999/image.png",
) -> MagicMock:
    """build mock discord.Attachment.

    :param attachment_id: discord snowflake id for attachment
    :ptype attachment_id: int
    :param filename: filename of attachment
    :ptype filename: str
    :param content_type: MIME type of attachment
    :ptype content_type: str
    :param size: file size in bytes
    :ptype size: int
    :param url: CDN url for attachment
    :ptype url: str
    :return: mock attachment object
    :rtype: MagicMock
    """
    att = MagicMock()
    att.id = attachment_id
    att.filename = filename
    att.content_type = content_type
    att.size = size
    att.url = url
    return att


def _make_mock_message(
    *,
    author: MagicMock | None = None,
    channel: MagicMock | None = None,
    guild: Any = _UNSET,
    content: str = "test message",
    message_id: int = 777888999,
    attachments: list[MagicMock] | None = None,
    reference: MagicMock | None = None,
) -> MagicMock:
    """build mock discord.Message.

    :param author: mock author, defaults to non-bot user
    :ptype author: MagicMock | None
    :param channel: mock channel, defaults to guild text channel
    :ptype channel: MagicMock | None
    :param guild: mock guild, defaults to standard guild; pass None for no guild (DM)
    :ptype guild: Any
    :param content: message text content
    :ptype content: str
    :param message_id: discord snowflake id for message
    :ptype message_id: int
    :param attachments: list of mock attachments
    :ptype attachments: list[MagicMock] | None
    :param reference: mock message reference for replies
    :ptype reference: MagicMock | None
    :return: mock message object
    :rtype: MagicMock
    """
    msg = MagicMock()
    msg.author = author if author is not None else _make_mock_author()
    msg.channel = channel if channel is not None else _make_mock_channel()
    msg.guild = _make_mock_guild() if guild is _UNSET else guild
    msg.content = content
    msg.id = message_id
    msg.attachments = attachments if attachments is not None else []
    msg.reference = reference
    msg.create_thread = AsyncMock()
    return msg


def _setup_mock_discord_types(mock_discord: MagicMock) -> None:
    """configure mock discord module with sentinel types for isinstance checks.

    sets Thread and DMChannel on mock discord to sentinel classes that match
    the __class__ values assigned by _make_mock_channel and _make_mock_dm_channel.

    :param mock_discord: mock of the discord module
    :ptype mock_discord: MagicMock
    """
    mock_discord.Thread = _FakeThread
    mock_discord.DMChannel = _FakeDMChannel


# ---------------------------------------------------------------------------
# enforcement tests (AST / import checks)
# ---------------------------------------------------------------------------


class TestDiscordAdapterEnforcement:
    """enforcement tests verifying structural constraints of discord module."""

    def test_discord_module_does_not_import_httpx(self) -> None:
        """discord adapter must not import httpx; discord.py handles HTTP."""
        from threetears.channels import discord as discord_mod

        source = inspect.getsource(discord_mod)
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name != "httpx", "discord module must not import httpx"
            if isinstance(node, ast.ImportFrom):
                assert node.module is None or not node.module.startswith("httpx"), (
                    "discord module must not import from httpx"
                )

    def test_discord_module_uses_discord_client(self) -> None:
        """discord adapter must reference discord.Client for gateway connection."""
        from threetears.channels import discord as discord_mod

        source = inspect.getsource(discord_mod)
        assert "discord.Client" in source

    def test_discord_module_does_not_use_client_run(self) -> None:
        """discord adapter must not use client.run() which blocks the event loop."""
        from threetears.channels import discord as discord_mod

        source = inspect.getsource(discord_mod)
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                if isinstance(node.value, ast.Attribute) and node.value.attr == "_client" and node.attr == "run":
                    pytest.fail("discord module must not use client.run(); use client.start() instead")


# ---------------------------------------------------------------------------
# constructor tests
# ---------------------------------------------------------------------------


class TestDiscordAdapterConstructor:
    """tests for DiscordAdapter initialization."""

    @patch("threetears.channels.discord.discord")
    def test_creates_discord_client_with_intents(self, mock_discord: MagicMock) -> None:
        """DiscordAdapter creates discord.Client with message_content intent."""
        from threetears.channels.discord import DiscordAdapter

        mock_intents = MagicMock()
        mock_discord.Intents.default.return_value = mock_intents

        router = _MockRouter()
        DiscordAdapter(
            bot_token="test-bot-token",
            router=router,
        )

        mock_discord.Intents.default.assert_called_once()
        assert mock_intents.messages is True
        assert mock_intents.message_content is True
        mock_discord.Client.assert_called_once_with(intents=mock_intents)

    @patch("threetears.channels.discord.discord")
    def test_stores_router(self, mock_discord: MagicMock) -> None:
        """DiscordAdapter stores router reference."""
        from threetears.channels.discord import DiscordAdapter

        router = _MockRouter()
        adapter = DiscordAdapter(
            bot_token="test-bot-token",
            router=router,
        )
        assert adapter.router is router

    @patch("threetears.channels.discord.discord")
    def test_stores_config(self, mock_discord: MagicMock) -> None:
        """DiscordAdapter stores optional config dict."""
        from threetears.channels.discord import DiscordAdapter

        router = _MockRouter()
        config = {"some_key": "some_value"}
        adapter = DiscordAdapter(
            bot_token="test-bot-token",
            router=router,
            config=config,
        )
        assert adapter.config == config

    @patch("threetears.channels.discord.discord")
    def test_config_defaults_to_empty_dict(self, mock_discord: MagicMock) -> None:
        """DiscordAdapter config defaults to empty dict when not provided."""
        from threetears.channels.discord import DiscordAdapter

        router = _MockRouter()
        adapter = DiscordAdapter(
            bot_token="test-bot-token",
            router=router,
        )
        assert adapter.config == {}

    @patch("threetears.channels.discord.discord")
    def test_stores_bot_token(self, mock_discord: MagicMock) -> None:
        """DiscordAdapter stores bot_token for start()."""
        from threetears.channels.discord import DiscordAdapter

        router = _MockRouter()
        adapter = DiscordAdapter(
            bot_token="test-bot-token",
            router=router,
        )
        assert adapter.bot_token == "test-bot-token"


# ---------------------------------------------------------------------------
# start / stop lifecycle tests
# ---------------------------------------------------------------------------


class TestDiscordAdapterLifecycle:
    """tests for DiscordAdapter start and stop methods."""

    @patch("threetears.channels.discord.discord")
    async def test_start_calls_client_start(self, mock_discord: MagicMock) -> None:
        """start() calls client.start(bot_token), not client.run()."""
        from threetears.channels.discord import DiscordAdapter

        mock_client = MagicMock()
        mock_client.start = AsyncMock()
        mock_discord.Client.return_value = mock_client

        router = _MockRouter()
        adapter = DiscordAdapter(
            bot_token="test-bot-token",
            router=router,
        )
        await adapter.start()
        mock_client.start.assert_awaited_once_with("test-bot-token")

    @patch("threetears.channels.discord.discord")
    async def test_stop_calls_client_close(self, mock_discord: MagicMock) -> None:
        """stop() calls client.close()."""
        from threetears.channels.discord import DiscordAdapter

        mock_client = MagicMock()
        mock_client.close = AsyncMock()
        mock_discord.Client.return_value = mock_client

        router = _MockRouter()
        adapter = DiscordAdapter(
            bot_token="test-bot-token",
            router=router,
        )
        await adapter.stop()
        mock_client.close.assert_awaited_once()


class _FakeMessageable:
    """real class registered as discord.abc.Messageable for the isinstance guard."""

    def __init__(self) -> None:
        self.send = AsyncMock()


def _messageable_target(mock_discord: MagicMock) -> _FakeMessageable:
    """wire mock_discord so a real messageable target passes the isinstance guard."""
    mock_discord.abc.Messageable = _FakeMessageable
    return _FakeMessageable()


class TestDiscordAdapterPostMessage:
    """tests for the out-of-band REST post_message (durable channel delivery)."""

    @patch("threetears.channels.discord.discord")
    async def test_post_message_logs_in_then_posts_to_channel(self, mock_discord: MagicMock) -> None:
        """post_message logs in ONCE (REST, no gateway) and sends to the channel."""
        from threetears.channels.discord import DiscordAdapter

        target = _messageable_target(mock_discord)
        mock_client = MagicMock()
        mock_client.login = AsyncMock()
        mock_client.start = AsyncMock()
        mock_client.fetch_channel = AsyncMock(return_value=target)
        mock_discord.Client.return_value = mock_client

        adapter = DiscordAdapter(bot_token="bot-tok", router=_MockRouter())
        await adapter.post_message(channel="12345", content="the answer")

        # REST login, NOT gateway start.
        mock_client.login.assert_awaited_once_with("bot-tok")
        mock_client.start.assert_not_awaited()
        mock_client.fetch_channel.assert_awaited_once_with(12345)
        target.send.assert_awaited_once_with(content="the answer")

    @patch("threetears.channels.discord.discord")
    async def test_post_message_targets_thread_when_given(self, mock_discord: MagicMock) -> None:
        """a thread_ref routes the post into the thread, not the parent channel."""
        from threetears.channels.discord import DiscordAdapter

        target = _messageable_target(mock_discord)
        mock_client = MagicMock()
        mock_client.login = AsyncMock()
        mock_client.fetch_channel = AsyncMock(return_value=target)
        mock_discord.Client.return_value = mock_client

        adapter = DiscordAdapter(bot_token="bot-tok", router=_MockRouter())
        await adapter.post_message(channel="999", content="hi", thread_ref="777")

        mock_client.fetch_channel.assert_awaited_once_with(777)

    @patch("threetears.channels.discord.discord")
    async def test_post_message_reuses_login_across_posts(self, mock_discord: MagicMock) -> None:
        """a second post_message reuses the authenticated session (login once)."""
        from threetears.channels.discord import DiscordAdapter

        target = _messageable_target(mock_discord)
        mock_client = MagicMock()
        mock_client.login = AsyncMock()
        mock_client.fetch_channel = AsyncMock(return_value=target)
        mock_discord.Client.return_value = mock_client

        adapter = DiscordAdapter(bot_token="bot-tok", router=_MockRouter())
        await adapter.post_message(channel="1", content="a")
        await adapter.post_message(channel="1", content="b")

        mock_client.login.assert_awaited_once()


# ---------------------------------------------------------------------------
# bot filtering tests
# ---------------------------------------------------------------------------


class TestDiscordAdapterBotFiltering:
    """tests for self-message and bot filtering."""

    @patch("threetears.channels.discord.discord")
    async def test_filters_self_messages(self, mock_discord: MagicMock) -> None:
        """messages from the bot itself are filtered out."""
        from threetears.channels.discord import DiscordAdapter

        _setup_mock_discord_types(mock_discord)
        mock_client = MagicMock()
        mock_client_user = MagicMock()
        mock_client.user = mock_client_user
        mock_discord.Client.return_value = mock_client

        router = _MockRouter()
        adapter = DiscordAdapter(
            bot_token="test-bot-token",
            router=router,
        )

        message = _make_mock_message(author=mock_client_user)
        await adapter.handle_message(message)
        assert router.last_message is None

    @patch("threetears.channels.discord.discord")
    async def test_filters_bot_messages(self, mock_discord: MagicMock) -> None:
        """messages from other bots are filtered out."""
        from threetears.channels.discord import DiscordAdapter

        _setup_mock_discord_types(mock_discord)
        mock_client = MagicMock()
        mock_client.user = _make_mock_author(author_id=999)
        mock_discord.Client.return_value = mock_client

        router = _MockRouter()
        adapter = DiscordAdapter(
            bot_token="test-bot-token",
            router=router,
        )

        bot_author = _make_mock_author(author_id=888, bot=True)
        message = _make_mock_message(author=bot_author)
        await adapter.handle_message(message)
        assert router.last_message is None

    @patch("threetears.channels.discord.discord")
    async def test_processes_non_bot_messages(self, mock_discord: MagicMock) -> None:
        """messages from regular users are processed."""
        from threetears.channels.discord import DiscordAdapter

        _setup_mock_discord_types(mock_discord)
        mock_client = MagicMock()
        mock_client.user = _make_mock_author(author_id=999)
        mock_discord.Client.return_value = mock_client

        router = _MockRouter(response=ChannelResponse(content="hello"))
        adapter = DiscordAdapter(
            bot_token="test-bot-token",
            router=router,
        )

        human_author = _make_mock_author(author_id=123, bot=False)
        channel = _make_mock_channel()
        mock_thread = MagicMock()
        mock_thread.send = AsyncMock()
        message = _make_mock_message(author=human_author, channel=channel)
        message.create_thread.return_value = mock_thread
        await adapter.handle_message(message)
        assert router.last_message is not None


# ---------------------------------------------------------------------------
# inbound normalization tests
# ---------------------------------------------------------------------------


class TestDiscordAdapterInboundNormalization:
    """tests for Discord message -> ChannelMessage normalization."""

    @patch("threetears.channels.discord.discord")
    async def test_guild_channel_message(self, mock_discord: MagicMock) -> None:
        """guild channel message normalizes to ChannelMessage with correct fields."""
        from threetears.channels.discord import DiscordAdapter

        _setup_mock_discord_types(mock_discord)
        mock_client = MagicMock()
        mock_client.user = _make_mock_author(author_id=999)
        mock_discord.Client.return_value = mock_client

        router = _MockRouter(response=ChannelResponse(content="ack"))
        adapter = DiscordAdapter(
            bot_token="test-bot-token",
            router=router,
        )

        author = _make_mock_author(
            author_id=123456789,
            display_name="TestUser",
        )
        guild = _make_mock_guild(guild_id=111222333)
        channel = _make_mock_channel(channel_id=987654321)
        mock_thread = MagicMock()
        mock_thread.send = AsyncMock()
        message = _make_mock_message(
            author=author,
            guild=guild,
            channel=channel,
            content="hello world",
        )
        message.create_thread.return_value = mock_thread

        await adapter.handle_message(message)

        msg = router.last_message
        assert msg is not None
        assert msg.channel_type == "discord"
        assert msg.sender_id == "123456789"
        assert msg.sender_name == "TestUser"
        assert msg.content == "hello world"
        assert msg.channel_id == "987654321"
        assert msg.workspace_id == "111222333"

    @patch("threetears.channels.discord.discord")
    async def test_dm_message_has_no_workspace_id(self, mock_discord: MagicMock) -> None:
        """DM message normalizes with workspace_id=None."""
        from threetears.channels.discord import DiscordAdapter

        _setup_mock_discord_types(mock_discord)
        mock_client = MagicMock()
        mock_client.user = _make_mock_author(author_id=999)
        mock_discord.Client.return_value = mock_client

        router = _MockRouter(response=ChannelResponse(content="ack"))
        adapter = DiscordAdapter(
            bot_token="test-bot-token",
            router=router,
        )

        dm_channel = _make_mock_dm_channel(channel_id=555666777)
        message = _make_mock_message(
            channel=dm_channel,
            guild=None,
            content="dm message",
        )

        await adapter.handle_message(message)

        msg = router.last_message
        assert msg is not None
        assert msg.workspace_id is None

    @patch("threetears.channels.discord.discord")
    async def test_thread_message_has_conversation_id(self, mock_discord: MagicMock) -> None:
        """thread message sets conversation_id to thread channel id."""
        from threetears.channels.discord import DiscordAdapter

        _setup_mock_discord_types(mock_discord)
        mock_client = MagicMock()
        mock_client.user = _make_mock_author(author_id=999)
        mock_discord.Client.return_value = mock_client

        router = _MockRouter(response=ChannelResponse(content="ack"))
        adapter = DiscordAdapter(
            bot_token="test-bot-token",
            router=router,
        )

        thread_channel = _make_mock_channel(channel_id=444333222, is_thread=True)
        message = _make_mock_message(
            channel=thread_channel,
            content="in thread",
        )

        await adapter.handle_message(message)

        msg = router.last_message
        assert msg is not None
        assert msg.conversation_id == "444333222"

    @patch("threetears.channels.discord.discord")
    async def test_non_thread_message_has_no_conversation_id(self, mock_discord: MagicMock) -> None:
        """non-thread guild message has conversation_id=None."""
        from threetears.channels.discord import DiscordAdapter

        _setup_mock_discord_types(mock_discord)
        mock_client = MagicMock()
        mock_client.user = _make_mock_author(author_id=999)
        mock_discord.Client.return_value = mock_client

        router = _MockRouter(response=ChannelResponse(content="ack"))
        adapter = DiscordAdapter(
            bot_token="test-bot-token",
            router=router,
        )

        channel = _make_mock_channel(channel_id=987654321)
        mock_thread = MagicMock()
        mock_thread.send = AsyncMock()
        message = _make_mock_message(channel=channel, content="top level")
        message.create_thread.return_value = mock_thread

        await adapter.handle_message(message)

        msg = router.last_message
        assert msg is not None
        assert msg.conversation_id is None

    @patch("threetears.channels.discord.discord")
    async def test_attachments_mapped(self, mock_discord: MagicMock) -> None:
        """discord attachments are mapped to Attachment dataclass instances."""
        from threetears.channels.discord import DiscordAdapter

        _setup_mock_discord_types(mock_discord)
        mock_client = MagicMock()
        mock_client.user = _make_mock_author(author_id=999)
        mock_discord.Client.return_value = mock_client

        router = _MockRouter(response=ChannelResponse(content="ack"))
        adapter = DiscordAdapter(
            bot_token="test-bot-token",
            router=router,
        )

        att1 = _make_mock_attachment(
            filename="report.pdf",
            content_type="application/pdf",
        )
        att2 = _make_mock_attachment(
            filename="photo.jpg",
            content_type="image/jpeg",
        )
        channel = _make_mock_channel()
        mock_thread = MagicMock()
        mock_thread.send = AsyncMock()
        message = _make_mock_message(
            content="see attachments",
            channel=channel,
            attachments=[att1, att2],
        )
        message.create_thread.return_value = mock_thread

        await adapter.handle_message(message)

        msg = router.last_message
        assert msg is not None
        assert len(msg.attachments) == 2
        assert msg.attachments[0].filename == "report.pdf"
        assert msg.attachments[0].content_type == "application/pdf"
        assert msg.attachments[0].content == b""
        assert msg.attachments[1].filename == "photo.jpg"
        assert msg.attachments[1].content_type == "image/jpeg"

    @patch("threetears.channels.discord.discord")
    async def test_reply_reference_sets_reply_to_id(self, mock_discord: MagicMock) -> None:
        """message.reference.message_id is captured as reply_to_id."""
        from threetears.channels.discord import DiscordAdapter

        _setup_mock_discord_types(mock_discord)
        mock_client = MagicMock()
        mock_client.user = _make_mock_author(author_id=999)
        mock_discord.Client.return_value = mock_client

        router = _MockRouter(response=ChannelResponse(content="ack"))
        adapter = DiscordAdapter(
            bot_token="test-bot-token",
            router=router,
        )

        reference = MagicMock()
        reference.message_id = 333444555
        channel = _make_mock_channel()
        mock_thread = MagicMock()
        mock_thread.send = AsyncMock()
        message = _make_mock_message(
            content="replying to something",
            channel=channel,
            reference=reference,
        )
        message.create_thread.return_value = mock_thread

        await adapter.handle_message(message)

        msg = router.last_message
        assert msg is not None
        assert msg.reply_to_id == "333444555"

    @patch("threetears.channels.discord.discord")
    async def test_no_reply_reference_sets_reply_to_id_none(self, mock_discord: MagicMock) -> None:
        """message without reference has reply_to_id=None."""
        from threetears.channels.discord import DiscordAdapter

        _setup_mock_discord_types(mock_discord)
        mock_client = MagicMock()
        mock_client.user = _make_mock_author(author_id=999)
        mock_discord.Client.return_value = mock_client

        router = _MockRouter(response=ChannelResponse(content="ack"))
        adapter = DiscordAdapter(
            bot_token="test-bot-token",
            router=router,
        )

        channel = _make_mock_channel()
        mock_thread = MagicMock()
        mock_thread.send = AsyncMock()
        message = _make_mock_message(
            content="no reply",
            channel=channel,
            reference=None,
        )
        message.create_thread.return_value = mock_thread

        await adapter.handle_message(message)

        msg = router.last_message
        assert msg is not None
        assert msg.reply_to_id is None

    @patch("threetears.channels.discord.discord")
    async def test_metadata_captures_discord_fields(self, mock_discord: MagicMock) -> None:
        """metadata includes Discord-specific fields like message id."""
        from threetears.channels.discord import DiscordAdapter

        _setup_mock_discord_types(mock_discord)
        mock_client = MagicMock()
        mock_client.user = _make_mock_author(author_id=999)
        mock_discord.Client.return_value = mock_client

        router = _MockRouter(response=ChannelResponse(content="ack"))
        adapter = DiscordAdapter(
            bot_token="test-bot-token",
            router=router,
        )

        channel = _make_mock_channel()
        mock_thread = MagicMock()
        mock_thread.send = AsyncMock()
        message = _make_mock_message(
            content="hello",
            channel=channel,
            message_id=777888999,
        )
        message.create_thread.return_value = mock_thread

        await adapter.handle_message(message)

        msg = router.last_message
        assert msg is not None
        assert msg.metadata.get("message_id") == "777888999"

    @patch("threetears.channels.discord.discord")
    async def test_timestamp_is_utc_aware(self, mock_discord: MagicMock) -> None:
        """inbound ChannelMessage timestamp is UTC-aware."""
        from datetime import UTC

        from threetears.channels.discord import DiscordAdapter

        _setup_mock_discord_types(mock_discord)
        mock_client = MagicMock()
        mock_client.user = _make_mock_author(author_id=999)
        mock_discord.Client.return_value = mock_client

        router = _MockRouter(response=ChannelResponse(content="ack"))
        adapter = DiscordAdapter(
            bot_token="test-bot-token",
            router=router,
        )

        channel = _make_mock_channel()
        mock_thread = MagicMock()
        mock_thread.send = AsyncMock()
        message = _make_mock_message(content="hello", channel=channel)
        message.create_thread.return_value = mock_thread
        await adapter.handle_message(message)

        msg = router.last_message
        assert msg is not None
        assert msg.timestamp.tzinfo is not None
        assert msg.timestamp.tzinfo == UTC


# ---------------------------------------------------------------------------
# threading model tests
# ---------------------------------------------------------------------------


class TestDiscordAdapterThreading:
    """tests for threading behavior of DiscordAdapter responses."""

    @patch("threetears.channels.discord.discord")
    async def test_dm_reply_goes_to_dm_channel_directly(self, mock_discord: MagicMock) -> None:
        """DM reply sends directly to DM channel without creating thread."""
        from threetears.channels.discord import DiscordAdapter

        _setup_mock_discord_types(mock_discord)
        mock_client = MagicMock()
        mock_client.user = _make_mock_author(author_id=999)
        mock_discord.Client.return_value = mock_client

        router = _MockRouter(response=ChannelResponse(content="dm reply"))
        adapter = DiscordAdapter(
            bot_token="test-bot-token",
            router=router,
        )

        dm_channel = _make_mock_dm_channel(channel_id=555666777)
        message = _make_mock_message(
            channel=dm_channel,
            guild=None,
            content="dm message",
        )

        await adapter.handle_message(message)

        dm_channel.send.assert_awaited()
        message.create_thread.assert_not_awaited()

    @patch("threetears.channels.discord.discord")
    async def test_existing_thread_reply_goes_to_thread(self, mock_discord: MagicMock) -> None:
        """reply in existing thread sends to the thread channel."""
        from threetears.channels.discord import DiscordAdapter

        _setup_mock_discord_types(mock_discord)
        mock_client = MagicMock()
        mock_client.user = _make_mock_author(author_id=999)
        mock_discord.Client.return_value = mock_client

        router = _MockRouter(response=ChannelResponse(content="thread reply"))
        adapter = DiscordAdapter(
            bot_token="test-bot-token",
            router=router,
        )

        thread_channel = _make_mock_channel(channel_id=444333222, is_thread=True)
        message = _make_mock_message(
            channel=thread_channel,
            content="in thread",
        )

        await adapter.handle_message(message)

        thread_channel.send.assert_awaited()
        message.create_thread.assert_not_awaited()

    @patch("threetears.channels.discord.discord")
    async def test_guild_channel_message_creates_thread(self, mock_discord: MagicMock) -> None:
        """guild channel message creates new thread, replies in thread."""
        from threetears.channels.discord import DiscordAdapter

        _setup_mock_discord_types(mock_discord)
        mock_client = MagicMock()
        mock_client.user = _make_mock_author(author_id=999)
        mock_discord.Client.return_value = mock_client

        router = _MockRouter(response=ChannelResponse(content="new thread reply"))
        adapter = DiscordAdapter(
            bot_token="test-bot-token",
            router=router,
        )

        channel = _make_mock_channel(channel_id=987654321)
        mock_thread = MagicMock()
        mock_thread.send = AsyncMock()

        message = _make_mock_message(
            channel=channel,
            content="top level",
        )
        message.create_thread.return_value = mock_thread

        await adapter.handle_message(message)

        message.create_thread.assert_awaited_once()
        mock_thread.send.assert_awaited()


# ---------------------------------------------------------------------------
# response routing tests
# ---------------------------------------------------------------------------


class TestDiscordAdapterResponseRouting:
    """tests for outbound response delivery."""

    @patch("threetears.channels.discord.discord")
    async def test_router_called_with_correct_channel_message(self, mock_discord: MagicMock) -> None:
        """route_inbound receives correctly normalized ChannelMessage."""
        from threetears.channels.discord import DiscordAdapter

        _setup_mock_discord_types(mock_discord)
        mock_client = MagicMock()
        mock_client.user = _make_mock_author(author_id=999)
        mock_discord.Client.return_value = mock_client

        router = _MockRouter(response=ChannelResponse(content="ack"))
        adapter = DiscordAdapter(
            bot_token="test-bot-token",
            router=router,
        )

        channel = _make_mock_channel()
        mock_thread = MagicMock()
        mock_thread.send = AsyncMock()
        message = _make_mock_message(content="specific content", channel=channel)
        message.create_thread.return_value = mock_thread
        await adapter.handle_message(message)

        msg = router.last_message
        assert msg is not None
        assert isinstance(msg, ChannelMessage)
        assert msg.channel_type == "discord"
        assert msg.content == "specific content"

    @patch("threetears.channels.discord.discord")
    async def test_none_response_skips_reply(self, mock_discord: MagicMock) -> None:
        """when router returns None, no message is sent."""
        from threetears.channels.discord import DiscordAdapter

        _setup_mock_discord_types(mock_discord)
        mock_client = MagicMock()
        mock_client.user = _make_mock_author(author_id=999)
        mock_discord.Client.return_value = mock_client

        router = _MockRouter(response=None)
        adapter = DiscordAdapter(
            bot_token="test-bot-token",
            router=router,
        )

        channel = _make_mock_channel()
        message = _make_mock_message(channel=channel, content="ignored")
        await adapter.handle_message(message)

        channel.send.assert_not_awaited()
        message.create_thread.assert_not_awaited()

    @patch("threetears.channels.discord.discord")
    async def test_long_message_split_at_2000_chars(self, mock_discord: MagicMock) -> None:
        """messages longer than 2000 chars are split into multiple sends."""
        from threetears.channels.discord import DiscordAdapter

        _setup_mock_discord_types(mock_discord)
        mock_client = MagicMock()
        mock_client.user = _make_mock_author(author_id=999)
        mock_discord.Client.return_value = mock_client

        long_content = "x" * 4500
        router = _MockRouter(response=ChannelResponse(content=long_content))
        adapter = DiscordAdapter(
            bot_token="test-bot-token",
            router=router,
        )

        dm_channel = _make_mock_dm_channel()
        message = _make_mock_message(
            channel=dm_channel,
            guild=None,
            content="give me long text",
        )

        await adapter.handle_message(message)

        send_calls = dm_channel.send.await_args_list
        assert len(send_calls) == 3
        total_sent = sum(len(call.kwargs.get("content", "")) for call in send_calls)
        assert total_sent == 4500

    @patch("threetears.channels.discord.discord")
    async def test_response_with_file_attachments(self, mock_discord: MagicMock) -> None:
        """response attachments are sent via discord.File."""
        from threetears.channels.discord import DiscordAdapter

        _setup_mock_discord_types(mock_discord)
        mock_client = MagicMock()
        mock_client.user = _make_mock_author(author_id=999)
        mock_discord.Client.return_value = mock_client
        mock_discord.File = MagicMock()

        attachment = Attachment(
            filename="data.csv",
            content=b"col1,col2\na,b",
            content_type="text/csv",
        )
        router = _MockRouter(
            response=ChannelResponse(
                content="here are results",
                attachments=[attachment],
            )
        )
        adapter = DiscordAdapter(
            bot_token="test-bot-token",
            router=router,
        )

        dm_channel = _make_mock_dm_channel()
        message = _make_mock_message(
            channel=dm_channel,
            guild=None,
            content="get data",
        )

        await adapter.handle_message(message)

        mock_discord.File.assert_called_once()
        call_kwargs = mock_discord.File.call_args
        assert call_kwargs.kwargs.get("filename") == "data.csv"


# ---------------------------------------------------------------------------
# message splitting unit tests
# ---------------------------------------------------------------------------


class TestSplitMessage:
    """tests for _split_message helper function."""

    def test_short_message_returns_single_item(self) -> None:
        """message under limit returns single-element list."""
        from threetears.channels.discord import _split_message

        result = _split_message("hello world")
        assert result == ["hello world"]

    def test_exact_limit_returns_single_item(self) -> None:
        """message exactly at limit returns single-element list."""
        from threetears.channels.discord import _split_message

        content = "x" * 2000
        result = _split_message(content)
        assert result == [content]

    def test_long_message_splits_correctly(self) -> None:
        """message exceeding limit is split into correct number of chunks."""
        from threetears.channels.discord import _split_message

        content = "x" * 4500
        result = _split_message(content)
        assert len(result) == 3
        assert len(result[0]) == 2000
        assert len(result[1]) == 2000
        assert len(result[2]) == 500

    def test_custom_max_length(self) -> None:
        """custom max_length parameter is respected."""
        from threetears.channels.discord import _split_message

        content = "abcdefghij"
        result = _split_message(content, max_length=3)
        assert result == ["abc", "def", "ghi", "j"]

    def test_empty_message_returns_single_empty_item(self) -> None:
        """empty string returns list with single empty string."""
        from threetears.channels.discord import _split_message

        result = _split_message("")
        assert result == [""]


# ---------------------------------------------------------------------------
# ChannelRouter protocol conformance
# ---------------------------------------------------------------------------


class TestDiscordAdapterProtocol:
    """tests verifying protocol conformance and package exports."""

    def test_mock_router_satisfies_channel_router_protocol(self) -> None:
        """_MockRouter used in tests satisfies ChannelRouter protocol."""
        router = _MockRouter()
        assert isinstance(router, ChannelRouter)

    def test_discord_adapter_importable_from_package(self) -> None:
        """DiscordAdapter is importable from threetears.channels."""
        from threetears.channels import DiscordAdapter

        assert DiscordAdapter is not None

    def test_discord_adapter_in_package_all(self) -> None:
        """DiscordAdapter appears in threetears.channels.__all__."""
        import threetears.channels as channels_pkg

        assert "DiscordAdapter" in channels_pkg.__all__


# ---------------------------------------------------------------------------
# rich formatting integration tests
# ---------------------------------------------------------------------------


class TestDiscordAdapterRichFormatting:
    """tests for rich formatting integration in _send_response."""

    @patch("threetears.channels.discord.discord")
    async def test_rich_formatting_sends_embed(self, mock_discord: MagicMock) -> None:
        """when format_hints has rich keys, send() receives embed kwarg."""
        from threetears.channels.discord import DiscordAdapter

        _setup_mock_discord_types(mock_discord)
        mock_client = MagicMock()
        mock_client.user = _make_mock_author(author_id=999)
        mock_discord.Client.return_value = mock_client

        mock_embed_cls = MagicMock()
        mock_embed_instance = MagicMock()
        mock_embed_cls.return_value = mock_embed_instance
        mock_discord.Embed = mock_embed_cls

        response = ChannelResponse(
            content="rich content here",
            format_hints={"format": "rich", "title": "Summary"},
        )
        router = _MockRouter(response=response)
        adapter = DiscordAdapter(
            bot_token="test-bot-token",
            router=router,
        )

        dm_channel = _make_mock_dm_channel()
        message = _make_mock_message(
            channel=dm_channel,
            guild=None,
            content="give me results",
        )

        await adapter.handle_message(message)

        # embed should have been created
        mock_embed_cls.assert_called_once()
        embed_kwargs = mock_embed_cls.call_args.kwargs
        assert "description" in embed_kwargs

        # send should have been called with embed, not content
        dm_channel.send.assert_awaited()
        send_kwargs = dm_channel.send.await_args.kwargs
        assert "embed" in send_kwargs

    @patch("threetears.channels.discord.discord")
    async def test_plain_formatting_sends_text_only(self, mock_discord: MagicMock) -> None:
        """when no format_hints, send() receives only content kwarg (no embed)."""
        from threetears.channels.discord import DiscordAdapter

        _setup_mock_discord_types(mock_discord)
        mock_client = MagicMock()
        mock_client.user = _make_mock_author(author_id=999)
        mock_discord.Client.return_value = mock_client

        response = ChannelResponse(content="plain text reply")
        router = _MockRouter(response=response)
        adapter = DiscordAdapter(
            bot_token="test-bot-token",
            router=router,
        )

        dm_channel = _make_mock_dm_channel()
        message = _make_mock_message(
            channel=dm_channel,
            guild=None,
            content="hello",
        )

        await adapter.handle_message(message)

        dm_channel.send.assert_awaited()
        send_kwargs = dm_channel.send.await_args.kwargs
        assert "content" in send_kwargs
        assert send_kwargs["content"] == "plain text reply"
        assert "embed" not in send_kwargs
