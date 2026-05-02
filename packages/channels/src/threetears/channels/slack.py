"""slack channel adapter using slack-bolt socket mode.

bridges slack workspace messages to platform via unified channel protocol.
uses slack-bolt AsyncApp with socket mode for real-time message handling
without requiring public HTTP endpoints.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from slack_bolt.async_app import AsyncApp
from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler

from threetears.channels.formatting import (
    build_slack_blocks,
    plain_text_fallback,
    should_use_rich_formatting,
)
from threetears.channels.protocol import (
    Attachment,
    ChannelMessage,
    ChannelResponse,
    ChannelRouter,
)

__all__ = [
    "SlackAdapter",
]


class SlackAdapter:
    """channel adapter bridging slack to platform via socket mode.

    receives inbound slack messages, normalizes them to ChannelMessage,
    routes through ChannelRouter, and delivers responses back to slack.

    :param bot_token: slack bot user OAuth token (xoxb-...)
    :ptype bot_token: str
    :param app_token: slack app-level token for socket mode (xapp-...)
    :ptype app_token: str
    :param router: channel router for processing inbound messages
    :ptype router: ChannelRouter
    :param config: optional adapter configuration overrides
    :ptype config: dict[str, Any] | None
    """

    def __init__(
        self,
        bot_token: str,
        app_token: str,
        router: ChannelRouter,
        config: dict[str, Any] | None = None,
    ) -> None:
        """initialize slack adapter with tokens, router, and optional config.

        :param bot_token: slack bot user OAuth token (xoxb-...)
        :ptype bot_token: str
        :param app_token: slack app-level token for socket mode (xapp-...)
        :ptype app_token: str
        :param router: channel router for processing inbound messages
        :ptype router: ChannelRouter
        :param config: optional adapter configuration overrides
        :ptype config: dict[str, Any] | None
        """
        self._app = AsyncApp(token=bot_token)
        self.app_token = app_token
        self.router = router
        self.config: dict[str, Any] = config if config is not None else {}
        self._handler: AsyncSocketModeHandler | None = None

        self._app.event("message")(self.handle_message_event)

    async def start(self) -> None:
        """start socket mode connection to slack.

        creates AsyncSocketModeHandler and initiates websocket connection.
        """
        self._handler = AsyncSocketModeHandler(self._app, self.app_token)
        await self._handler.start_async()

    async def stop(self) -> None:
        """stop socket mode connection to slack.

        closes websocket connection if handler exists. safe to call
        before start() or after already stopped.
        """
        if self._handler is not None:
            await self._handler.close_async()

    async def handle_message_event(
        self,
        event: dict[str, Any],
        say: Any,
    ) -> None:
        """public slack-event handler for inbound messages.

        registered as the slack-bolt ``message`` event callback in
        :meth:`__init__`. tests exercise this surface directly; the
        name + ``(event, say)`` shape are part of the stability
        contract.

        filters bot messages, normalizes event to ChannelMessage,
        routes through router, and delivers response back to slack.

        :param event: raw slack event payload
        :ptype event: dict[str, Any]
        :param say: slack-bolt say function for replying
        :ptype say: Any
        """
        if event.get("bot_id") or event.get("subtype") == "bot_message":
            return

        thread_ts = event.get("thread_ts")
        channel_type_raw = event.get("channel_type", "")
        is_dm = channel_type_raw == "im"

        attachments = _map_slack_files(event.get("files", []))

        # fields consumed by standard ChannelMessage attributes
        consumed_keys = {
            "type",
            "user",
            "text",
            "channel",
            "thread_ts",
            "team",
            "files",
        }
        metadata: dict[str, Any] = {k: v for k, v in event.items() if k not in consumed_keys}

        channel_message = ChannelMessage(
            channel_type="slack",
            sender_id=event.get("user", ""),
            content=event.get("text", ""),
            channel_id=event.get("channel", ""),
            conversation_id=thread_ts if thread_ts else event.get("ts"),
            workspace_id=event.get("team"),
            attachments=attachments,
            reply_to_id=thread_ts,
            metadata=metadata,
            timestamp=datetime.now(UTC),
        )

        response = await self.router.route_inbound(channel_message)

        if response is None:
            return

        await _send_response(
            say=say,
            client=self._app.client,
            response=response,
            event=event,
            thread_ts=thread_ts,
            is_dm=is_dm,
        )


def _map_slack_files(files: list[dict[str, Any]]) -> list[Attachment]:
    """map slack file objects to Attachment dataclass instances.

    :param files: list of slack file metadata dicts
    :ptype files: list[dict[str, Any]]
    :return: list of Attachment dataclass instances
    :rtype: list[Attachment]
    """
    result: list[Attachment] = []
    for f in files:
        result.append(
            Attachment(
                filename=f.get("name", ""),
                content=b"",
                content_type=f.get("mimetype", "application/octet-stream"),
                description=f.get("title"),
            )
        )
    return result


async def _send_response(
    say: Any,
    client: Any,
    response: ChannelResponse,
    event: dict[str, Any],
    thread_ts: str | None,
    is_dm: bool,
) -> None:
    """deliver outbound response back to slack.

    sends text reply and uploads any file attachments. threading behavior
    depends on whether inbound was threaded, DM, or top-level channel message.

    :param say: slack-bolt say function for replying
    :ptype say: Any
    :param client: slack async web client for file uploads
    :ptype client: Any
    :param response: outbound response from router
    :ptype response: ChannelResponse
    :param event: original slack event for context
    :ptype event: dict[str, Any]
    :param thread_ts: thread timestamp if message was in thread
    :ptype thread_ts: str | None
    :param is_dm: whether message is direct message
    :ptype is_dm: bool
    """
    use_rich = should_use_rich_formatting(response.format_hints)

    if is_dm and not thread_ts:
        if use_rich:
            blocks = build_slack_blocks(response.content, response.format_hints)
            fallback_text = plain_text_fallback(response.content)
            await say(text=fallback_text, blocks=blocks)
        else:
            await say(text=response.content)
    else:
        reply_thread_ts = thread_ts if thread_ts else event.get("ts", "")
        if use_rich:
            blocks = build_slack_blocks(response.content, response.format_hints)
            fallback_text = plain_text_fallback(response.content)
            await say(text=fallback_text, blocks=blocks, thread_ts=reply_thread_ts)
        else:
            await say(text=response.content, thread_ts=reply_thread_ts)

    channel_id = event.get("channel", "")
    effective_thread_ts = thread_ts if thread_ts else event.get("ts", "")

    for attachment in response.attachments:
        await client.files_upload_v2(
            channel=channel_id,
            filename=attachment.filename,
            content=attachment.content,
            title=attachment.filename,
            thread_ts=effective_thread_ts,
        )
