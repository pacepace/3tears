"""slack channel adapter using slack-bolt socket mode.

bridges slack workspace messages to platform via unified channel protocol.
uses slack-bolt AsyncApp with socket mode for real-time message handling
without requiring public HTTP endpoints.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any

from slack_bolt.async_app import AsyncApp
from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler

from threetears.channels.formatting import (
    markdown_to_slack_blocks,
    plain_text_fallback,
)
from threetears.channels.protocol import (
    Attachment,
    ChannelMessage,
    ChannelResponse,
    ChannelRouter,
)
from threetears.observe import get_logger

__all__ = [
    "SlackAdapter",
]

log = get_logger(__name__)

# cache TTL for per-user locale info (timezone + BCP 47 locale tag).
# Slack ``users.info`` is Tier 4 (~100 req/min); a 5-minute TTL keeps
# fetches rare without serving stale tz to a user who just travelled
# (Slack updates ``user.tz`` from the user's profile, which a user can
# change mid-session).
_USER_LOCALE_TTL_SECONDS = 300.0


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
        # per-user locale cache: maps slack user_id -> (cached_at_monotonic, tz, locale)
        # invalidated when ``cached_at_monotonic`` is older than
        # :data:`_USER_LOCALE_TTL_SECONDS`. populated lazily on first
        # message from each user via :meth:`_resolve_user_locale`.
        self._user_locale_cache: dict[str, tuple[float, str | None, str | None]] = {}

        self._app.event("message")(self.handle_message_event)

    async def _resolve_user_locale(
        self,
        user_id: str,
    ) -> tuple[str | None, str | None]:
        """resolve ``(tz, locale)`` for a slack user via ``users.info``.

        consults the per-instance TTL cache first; on miss, calls
        ``users.info(user=user_id, include_locale=True)`` and caches
        the result. failures (network error, unknown user, missing
        fields) are cached as ``(None, None)`` so a flapping user-info
        endpoint does not hammer slack with retries inside the cache
        window. callers consume the result as ``ChannelMessage``
        ``user_timezone`` / ``user_locale`` fields.

        :param user_id: slack platform user identifier (``U…`` form)
        :ptype user_id: str
        :return: tuple of (IANA timezone name, BCP 47 locale tag);
            either may be ``None`` when slack does not surface a value
            for this user
        :rtype: tuple[str | None, str | None]
        """
        cached = self._user_locale_cache.get(user_id)
        now = time.monotonic()
        result: tuple[str | None, str | None]
        if cached is not None and (now - cached[0]) < _USER_LOCALE_TTL_SECONDS:
            result = (cached[1], cached[2])
        else:
            tz: str | None = None
            locale: str | None = None
            try:
                response = await self._app.client.users_info(
                    user=user_id,
                    include_locale=True,
                )
                user_obj: dict[str, Any] = response.get("user", {}) if response else {}
                raw_tz = user_obj.get("tz")
                raw_locale = user_obj.get("locale")
                tz = raw_tz if isinstance(raw_tz, str) and raw_tz else None
                locale = raw_locale if isinstance(raw_locale, str) and raw_locale else None
            except Exception as exc:
                log.warning(
                    "slack users.info lookup failed; caching empty locale to avoid hammering the API",
                    extra={"extra_data": {"user_id": user_id, "error": str(exc)}},
                )
            self._user_locale_cache[user_id] = (now, tz, locale)
            result = (tz, locale)
        return result

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

    async def post_message(
        self,
        *,
        channel: str,
        text: str,
        thread_ts: str | None = None,
    ) -> None:
        """post a message to a channel/thread out-of-band (no inbound event).

        used by durable channel-answer delivery: a finished agent answer can
        arrive long after the originating Slack event has closed, so it is
        posted directly through the web client (``chat_postMessage``) on the
        adapter's bot token rather than the event-bound ``say`` closure. the
        markdown body is rendered into native Slack blocks, as in
        :func:`_send_response`.

        :param channel: slack channel id to post into
        :ptype channel: str
        :param text: message text in markdown
        :ptype text: str
        :param thread_ts: thread to reply in; ``None`` posts top-level
        :ptype thread_ts: str | None
        :return: nothing
        :rtype: None
        """
        # always render the agent's markdown into native Slack blocks (tables,
        # headers, mrkdwn) -- Slack does not render GitHub markdown, so posting
        # raw text shows ``**bold**`` / ``| tables |`` literally. ``text`` is the
        # notification/fallback (markdown stripped); ``blocks`` is the rendered
        # body.
        blocks = markdown_to_slack_blocks(text)
        if blocks:
            kwargs: dict[str, Any] = {
                "channel": channel,
                "text": plain_text_fallback(text),
                "blocks": blocks,
            }
        else:
            kwargs = {"channel": channel, "text": text}
        if thread_ts:
            kwargs["thread_ts"] = thread_ts
        await self._app.client.chat_postMessage(**kwargs)

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

        # per-user locale info from slack profile -- looked up via
        # ``users.info(include_locale=True)`` and cached for
        # :data:`_USER_LOCALE_TTL_SECONDS`. populated even on cache
        # miss so consumers (agent runtime stamping
        # ``CallContext.user_timezone``) read uniformly.
        sender_id = event.get("user", "")
        user_tz: str | None = None
        user_locale: str | None = None
        if sender_id:
            user_tz, user_locale = await self._resolve_user_locale(sender_id)

        channel_message = ChannelMessage(
            channel_type="slack",
            sender_id=sender_id,
            content=event.get("text", ""),
            channel_id=event.get("channel", ""),
            conversation_id=thread_ts if thread_ts else event.get("ts"),
            workspace_id=event.get("team"),
            attachments=attachments,
            reply_to_id=thread_ts,
            metadata=metadata,
            timestamp=datetime.now(UTC),
            user_timezone=user_tz,
            user_locale=user_locale,
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
    # render the agent's markdown into native Slack blocks (tables, headers,
    # mrkdwn); ``text`` is the markdown-stripped notification fallback.
    blocks = markdown_to_slack_blocks(response.content)
    fallback_text = plain_text_fallback(response.content)

    if is_dm and not thread_ts:
        if blocks:
            await say(text=fallback_text, blocks=blocks)
        else:
            await say(text=response.content)
    else:
        reply_thread_ts = thread_ts if thread_ts else event.get("ts", "")
        if blocks:
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
