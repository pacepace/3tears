"""discord channel adapter using discord.py library.

bridges discord guild and DM messages to platform via unified channel protocol.
uses discord.Client with gateway intents for real-time message handling.
"""

from __future__ import annotations

import io
from datetime import UTC, datetime
from typing import Any

import discord

from threetears.channels.formatting import (
    build_discord_embed,
    should_use_rich_formatting,
)
from threetears.channels.protocol import (
    Attachment,
    ChannelMessage,
    ChannelResponse,
    ChannelRouter,
)

__all__ = [
    "DiscordAdapter",
]


class DiscordAdapter:
    """channel adapter bridging discord to platform via discord.py gateway.

    receives inbound discord messages, normalizes them to ChannelMessage,
    routes through ChannelRouter, and delivers responses back to discord.
    creates threads for guild channel replies, replies directly in DMs
    and existing threads.

    :param bot_token: discord bot token for authentication
    :ptype bot_token: str
    :param router: channel router for processing inbound messages
    :ptype router: ChannelRouter
    :param config: optional adapter configuration overrides
    :ptype config: dict[str, Any] | None
    """

    def __init__(
        self,
        bot_token: str,
        router: ChannelRouter,
        config: dict[str, Any] | None = None,
    ) -> None:
        """initialize discord adapter with token, router, and optional config.

        :param bot_token: discord bot token for authentication
        :ptype bot_token: str
        :param router: channel router for processing inbound messages
        :ptype router: ChannelRouter
        :param config: optional adapter configuration overrides
        :ptype config: dict[str, Any] | None
        """
        self._bot_token = bot_token
        self._router = router
        self._config: dict[str, Any] = config if config is not None else {}

        intents = discord.Intents.default()
        intents.messages = True
        intents.message_content = True
        self._client = discord.Client(intents=intents)

        self._client.event(self._on_message)
        self._client.event(self._on_ready)

    async def start(self) -> None:
        """start discord gateway connection.

        uses client.start() which is a coroutine, NOT client.run()
        which would block and create its own event loop.
        """
        await self._client.start(self._bot_token)

    async def stop(self) -> None:
        """stop discord gateway connection.

        closes websocket connection and cleans up resources.
        """
        await self._client.close()

    async def _on_ready(self) -> None:
        """handle discord on_ready event.

        called when the bot has connected to the gateway and is ready
        to receive messages.
        """

    async def _on_message(self, message: Any) -> None:
        """handle discord on_message event.

        delegates to _handle_message for processing. registered as
        discord.py event handler via client.event().

        :param message: discord message object
        :ptype message: Any
        """
        await self._handle_message(message)

    async def _handle_message(self, message: Any) -> None:
        """process inbound discord message.

        filters self and bot messages, normalizes to ChannelMessage,
        routes through router, and delivers response.

        :param message: discord message object
        :ptype message: Any
        """
        if message.author == self._client.user:
            return

        if message.author.bot:
            return

        channel_message = _build_channel_message(message)

        response = await self._router.route_inbound(channel_message)

        if response is None:
            return

        await _send_response(
            message=message,
            response=response,
        )


def _build_channel_message(message: Any) -> ChannelMessage:
    """normalize discord message to platform ChannelMessage.

    :param message: discord message object
    :ptype message: Any
    :return: normalized channel message
    :rtype: ChannelMessage
    """
    workspace_id: str | None = str(message.guild.id) if message.guild else None

    conversation_id: str | None = str(message.channel.id) if isinstance(message.channel, discord.Thread) else None

    reply_to_id: str | None = str(message.reference.message_id) if message.reference else None

    attachments = _map_discord_attachments(message.attachments)

    metadata: dict[str, Any] = {
        "message_id": str(message.id),
    }

    result = ChannelMessage(
        channel_type="discord",
        sender_id=str(message.author.id),
        sender_name=message.author.display_name,
        content=message.content,
        channel_id=str(message.channel.id),
        workspace_id=workspace_id,
        conversation_id=conversation_id,
        attachments=attachments,
        reply_to_id=reply_to_id,
        metadata=metadata,
        timestamp=datetime.now(UTC),
    )
    return result


def _map_discord_attachments(
    attachments: list[Any],
) -> list[Attachment]:
    """map discord attachment objects to Attachment dataclass instances.

    :param attachments: list of discord attachment objects
    :ptype attachments: list[Any]
    :return: list of Attachment dataclass instances
    :rtype: list[Attachment]
    """
    result: list[Attachment] = []
    for att in attachments:
        result.append(
            Attachment(
                filename=att.filename,
                content=b"",
                content_type=att.content_type or "application/octet-stream",
                description=None,
            )
        )
    return result


def _split_message(
    content: str,
    max_length: int = 2000,
) -> list[str]:
    """split message content at max_length boundaries.

    discord enforces 2000-character limit per message. this function
    splits content into chunks that respect that limit.

    :param content: message content to split
    :ptype content: str
    :param max_length: maximum characters per chunk
    :ptype max_length: int
    :return: list of content chunks
    :rtype: list[str]
    """
    if len(content) <= max_length:
        return [content]

    result: list[str] = []
    offset = 0
    while offset < len(content):
        result.append(content[offset : offset + max_length])
        offset += max_length
    return result


async def _send_response(
    message: Any,
    response: ChannelResponse,
) -> None:
    """deliver outbound response back to discord.

    determines reply target based on channel type: DM channels get
    direct replies, existing threads get in-thread replies, and guild
    channel messages trigger thread creation.

    :param message: original discord message for context
    :ptype message: Any
    :param response: outbound response from router
    :ptype response: ChannelResponse
    """
    target_channel = _resolve_reply_target(message)

    if target_channel is None:
        thread = await message.create_thread(
            name=f"Response to {message.author.display_name}",
            auto_archive_duration=1440,
        )
        target_channel = thread

    use_rich = should_use_rich_formatting(response.format_hints)

    if use_rich:
        embed_data = build_discord_embed(response.content, response.format_hints)
        embed = discord.Embed(description=embed_data.get("description", ""))
        if "title" in embed_data:
            embed.title = embed_data["title"]
        if "color" in embed_data:
            embed.color = embed_data["color"]
        await target_channel.send(embed=embed)
    else:
        chunks = _split_message(response.content)
        for chunk in chunks:
            await target_channel.send(content=chunk)

    for attachment in response.attachments:
        file_obj = discord.File(
            fp=io.BytesIO(attachment.content),
            filename=attachment.filename,
        )
        await target_channel.send(file=file_obj)


def _resolve_reply_target(message: Any) -> Any | None:
    """determine channel to send reply to based on message context.

    returns the channel directly for DM and thread messages.
    returns None for guild channel messages (caller must create thread).

    :param message: original discord message
    :ptype message: Any
    :return: channel to reply in, or None if thread creation needed
    :rtype: Any | None
    """
    if isinstance(message.channel, discord.DMChannel):
        return message.channel

    if isinstance(message.channel, discord.Thread):
        return message.channel

    return None
