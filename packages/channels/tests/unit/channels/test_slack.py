"""tests for SlackAdapter channel adapter."""

from __future__ import annotations

import ast
import inspect
from datetime import UTC
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch


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
# enforcement tests (AST / import checks)
# ---------------------------------------------------------------------------


class TestSlackAdapterEnforcement:
    """enforcement tests verifying structural constraints of slack module."""

    def test_slack_module_does_not_import_httpx(self) -> None:
        """slack adapter must not import httpx; slack-bolt handles HTTP."""
        from threetears.channels import slack as slack_mod

        source = inspect.getsource(slack_mod)
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name != "httpx", "slack module must not import httpx"
            if isinstance(node, ast.ImportFrom):
                assert node.module is None or not node.module.startswith("httpx"), (
                    "slack module must not import from httpx"
                )

    def test_slack_adapter_uses_async_socket_mode_handler(self) -> None:
        """slack adapter must reference AsyncSocketModeHandler for socket mode."""
        from threetears.channels import slack as slack_mod

        source = inspect.getsource(slack_mod)
        assert "AsyncSocketModeHandler" in source

    def test_slack_adapter_uses_async_app(self) -> None:
        """slack adapter must use AsyncApp, not synchronous App."""
        from threetears.channels import slack as slack_mod

        source = inspect.getsource(slack_mod)
        assert "AsyncApp" in source


# ---------------------------------------------------------------------------
# constructor tests
# ---------------------------------------------------------------------------


class TestSlackAdapterConstructor:
    """tests for SlackAdapter initialization."""

    @patch("threetears.channels.slack.AsyncApp")
    def test_creates_async_app_with_token(self, mock_app_cls: MagicMock) -> None:
        """SlackAdapter passes bot_token to AsyncApp constructor."""
        from threetears.channels.slack import SlackAdapter

        router = _MockRouter()
        SlackAdapter(
            bot_token="xoxb-test-token",
            app_token="xapp-test-token",
            router=router,
        )
        mock_app_cls.assert_called_once_with(token="xoxb-test-token")

    @patch("threetears.channels.slack.AsyncApp")
    def test_stores_app_token(self, mock_app_cls: MagicMock) -> None:
        """SlackAdapter stores app_token for socket mode handler creation."""
        from threetears.channels.slack import SlackAdapter

        router = _MockRouter()
        adapter = SlackAdapter(
            bot_token="xoxb-test-token",
            app_token="xapp-test-token",
            router=router,
        )
        assert adapter.app_token == "xapp-test-token"

    @patch("threetears.channels.slack.AsyncApp")
    def test_stores_router(self, mock_app_cls: MagicMock) -> None:
        """SlackAdapter stores router reference."""
        from threetears.channels.slack import SlackAdapter

        router = _MockRouter()
        adapter = SlackAdapter(
            bot_token="xoxb-test-token",
            app_token="xapp-test-token",
            router=router,
        )
        assert adapter.router is router

    @patch("threetears.channels.slack.AsyncApp")
    def test_stores_config(self, mock_app_cls: MagicMock) -> None:
        """SlackAdapter stores optional config dict."""
        from threetears.channels.slack import SlackAdapter

        router = _MockRouter()
        config = {"some_key": "some_value"}
        adapter = SlackAdapter(
            bot_token="xoxb-test-token",
            app_token="xapp-test-token",
            router=router,
            config=config,
        )
        assert adapter.config == config

    @patch("threetears.channels.slack.AsyncApp")
    def test_config_defaults_to_empty_dict(self, mock_app_cls: MagicMock) -> None:
        """SlackAdapter config defaults to empty dict when not provided."""
        from threetears.channels.slack import SlackAdapter

        router = _MockRouter()
        adapter = SlackAdapter(
            bot_token="xoxb-test-token",
            app_token="xapp-test-token",
            router=router,
        )
        assert adapter.config == {}

    @patch("threetears.channels.slack.AsyncApp")
    def test_registers_message_event_handler(self, mock_app_cls: MagicMock) -> None:
        """SlackAdapter registers a handler for message events on the app."""
        from threetears.channels.slack import SlackAdapter

        mock_app = MagicMock()
        mock_app_cls.return_value = mock_app
        router = _MockRouter()
        SlackAdapter(
            bot_token="xoxb-test-token",
            app_token="xapp-test-token",
            router=router,
        )
        mock_app.event.assert_called_with("message")


# ---------------------------------------------------------------------------
# start / stop lifecycle tests
# ---------------------------------------------------------------------------


class TestSlackAdapterLifecycle:
    """tests for SlackAdapter start and stop methods."""

    @patch("threetears.channels.slack.AsyncSocketModeHandler")
    @patch("threetears.channels.slack.AsyncApp")
    async def test_start_creates_socket_mode_handler(
        self, mock_app_cls: MagicMock, mock_handler_cls: MagicMock
    ) -> None:
        """start() creates AsyncSocketModeHandler with app and app_token."""
        from threetears.channels.slack import SlackAdapter

        mock_app = MagicMock()
        mock_app_cls.return_value = mock_app
        mock_handler = AsyncMock()
        mock_handler_cls.return_value = mock_handler

        router = _MockRouter()
        adapter = SlackAdapter(
            bot_token="xoxb-test-token",
            app_token="xapp-test-token",
            router=router,
        )
        await adapter.start()
        mock_handler_cls.assert_called_once_with(mock_app, "xapp-test-token")

    @patch("threetears.channels.slack.AsyncSocketModeHandler")
    @patch("threetears.channels.slack.AsyncApp")
    async def test_start_calls_start_async(self, mock_app_cls: MagicMock, mock_handler_cls: MagicMock) -> None:
        """start() calls handler.start_async()."""
        from threetears.channels.slack import SlackAdapter

        mock_handler = AsyncMock()
        mock_handler_cls.return_value = mock_handler

        router = _MockRouter()
        adapter = SlackAdapter(
            bot_token="xoxb-test-token",
            app_token="xapp-test-token",
            router=router,
        )
        await adapter.start()
        mock_handler.start_async.assert_awaited_once()

    @patch("threetears.channels.slack.AsyncSocketModeHandler")
    @patch("threetears.channels.slack.AsyncApp")
    async def test_stop_calls_close_async(self, mock_app_cls: MagicMock, mock_handler_cls: MagicMock) -> None:
        """stop() calls handler.close_async()."""
        from threetears.channels.slack import SlackAdapter

        mock_handler = AsyncMock()
        mock_handler_cls.return_value = mock_handler

        router = _MockRouter()
        adapter = SlackAdapter(
            bot_token="xoxb-test-token",
            app_token="xapp-test-token",
            router=router,
        )
        await adapter.start()
        await adapter.stop()
        mock_handler.close_async.assert_awaited_once()

    @patch("threetears.channels.slack.AsyncApp")
    async def test_stop_without_start_is_safe(self, mock_app_cls: MagicMock) -> None:
        """stop() before start() does not raise."""
        from threetears.channels.slack import SlackAdapter

        router = _MockRouter()
        adapter = SlackAdapter(
            bot_token="xoxb-test-token",
            app_token="xapp-test-token",
            router=router,
        )
        await adapter.stop()


# ---------------------------------------------------------------------------
# bot message filtering tests
# ---------------------------------------------------------------------------


class TestSlackAdapterBotFiltering:
    """tests for bot self-message filtering."""

    @patch("threetears.channels.slack.AsyncApp")
    async def test_filters_event_with_bot_id(self, mock_app_cls: MagicMock) -> None:
        """events with bot_id present are filtered out."""
        from threetears.channels.slack import SlackAdapter

        router = _MockRouter()
        mock_app = MagicMock()
        mock_app_cls.return_value = mock_app

        adapter = SlackAdapter(
            bot_token="xoxb-test-token",
            app_token="xapp-test-token",
            router=router,
        )

        event: dict[str, Any] = {
            "type": "message",
            "bot_id": "B12345",
            "text": "bot says hello",
            "channel": "C123",
            "ts": "1234567890.123456",
        }
        say = AsyncMock()
        await adapter.handle_message_event(event=event, say=say)
        assert router.last_message is None
        say.assert_not_awaited()

    @patch("threetears.channels.slack.AsyncApp")
    async def test_filters_event_with_bot_message_subtype(self, mock_app_cls: MagicMock) -> None:
        """events with subtype 'bot_message' are filtered out."""
        from threetears.channels.slack import SlackAdapter

        router = _MockRouter()
        mock_app = MagicMock()
        mock_app_cls.return_value = mock_app

        adapter = SlackAdapter(
            bot_token="xoxb-test-token",
            app_token="xapp-test-token",
            router=router,
        )

        event: dict[str, Any] = {
            "type": "message",
            "subtype": "bot_message",
            "text": "bot says hello",
            "channel": "C123",
            "ts": "1234567890.123456",
        }
        say = AsyncMock()
        await adapter.handle_message_event(event=event, say=say)
        assert router.last_message is None
        say.assert_not_awaited()


# ---------------------------------------------------------------------------
# inbound message normalization tests
# ---------------------------------------------------------------------------


class TestSlackAdapterInboundNormalization:
    """tests for Slack event -> ChannelMessage normalization."""

    @patch("threetears.channels.slack.AsyncApp")
    async def test_basic_channel_message(self, mock_app_cls: MagicMock) -> None:
        """basic channel message normalizes to ChannelMessage with correct fields."""
        from threetears.channels.slack import SlackAdapter

        response = ChannelResponse(content="reply text")
        router = _MockRouter(response=response)
        mock_app = MagicMock()
        mock_app_cls.return_value = mock_app

        adapter = SlackAdapter(
            bot_token="xoxb-test-token",
            app_token="xapp-test-token",
            router=router,
        )

        event: dict[str, Any] = {
            "type": "message",
            "user": "U12345",
            "text": "hello world",
            "channel": "C98765",
            "ts": "1234567890.123456",
            "team": "T00001",
        }
        say = AsyncMock()
        await adapter.handle_message_event(event=event, say=say)

        msg = router.last_message
        assert msg is not None
        assert msg.channel_type == "slack"
        assert msg.sender_id == "U12345"
        assert msg.content == "hello world"
        assert msg.channel_id == "C98765"
        assert msg.workspace_id == "T00001"

    @patch("threetears.channels.slack.AsyncApp")
    async def test_conversation_id_from_thread_ts(self, mock_app_cls: MagicMock) -> None:
        """conversation_id is set from thread_ts when present."""
        from threetears.channels.slack import SlackAdapter

        response = ChannelResponse(content="reply")
        router = _MockRouter(response=response)
        mock_app = MagicMock()
        mock_app_cls.return_value = mock_app

        adapter = SlackAdapter(
            bot_token="xoxb-test-token",
            app_token="xapp-test-token",
            router=router,
        )

        event: dict[str, Any] = {
            "type": "message",
            "user": "U12345",
            "text": "threaded reply",
            "channel": "C98765",
            "ts": "1234567890.999999",
            "thread_ts": "1234567890.000001",
            "team": "T00001",
        }
        say = AsyncMock()
        await adapter.handle_message_event(event=event, say=say)

        msg = router.last_message
        assert msg is not None
        assert msg.conversation_id == "1234567890.000001"

    @patch("threetears.channels.slack.AsyncApp")
    async def test_conversation_id_falls_back_to_ts(self, mock_app_cls: MagicMock) -> None:
        """conversation_id falls back to ts when thread_ts is absent."""
        from threetears.channels.slack import SlackAdapter

        response = ChannelResponse(content="reply")
        router = _MockRouter(response=response)
        mock_app = MagicMock()
        mock_app_cls.return_value = mock_app

        adapter = SlackAdapter(
            bot_token="xoxb-test-token",
            app_token="xapp-test-token",
            router=router,
        )

        event: dict[str, Any] = {
            "type": "message",
            "user": "U12345",
            "text": "top-level message",
            "channel": "C98765",
            "ts": "1234567890.123456",
            "team": "T00001",
        }
        say = AsyncMock()
        await adapter.handle_message_event(event=event, say=say)

        msg = router.last_message
        assert msg is not None
        assert msg.conversation_id == "1234567890.123456"

    @patch("threetears.channels.slack.AsyncApp")
    async def test_reply_to_id_from_thread_ts(self, mock_app_cls: MagicMock) -> None:
        """reply_to_id is set from thread_ts when present."""
        from threetears.channels.slack import SlackAdapter

        response = ChannelResponse(content="reply")
        router = _MockRouter(response=response)
        mock_app = MagicMock()
        mock_app_cls.return_value = mock_app

        adapter = SlackAdapter(
            bot_token="xoxb-test-token",
            app_token="xapp-test-token",
            router=router,
        )

        event: dict[str, Any] = {
            "type": "message",
            "user": "U12345",
            "text": "threaded",
            "channel": "C98765",
            "ts": "1234567890.999999",
            "thread_ts": "1234567890.000001",
            "team": "T00001",
        }
        say = AsyncMock()
        await adapter.handle_message_event(event=event, say=say)

        msg = router.last_message
        assert msg is not None
        assert msg.reply_to_id == "1234567890.000001"

    @patch("threetears.channels.slack.AsyncApp")
    async def test_reply_to_id_none_without_thread_ts(self, mock_app_cls: MagicMock) -> None:
        """reply_to_id is None when thread_ts is absent."""
        from threetears.channels.slack import SlackAdapter

        response = ChannelResponse(content="reply")
        router = _MockRouter(response=response)
        mock_app = MagicMock()
        mock_app_cls.return_value = mock_app

        adapter = SlackAdapter(
            bot_token="xoxb-test-token",
            app_token="xapp-test-token",
            router=router,
        )

        event: dict[str, Any] = {
            "type": "message",
            "user": "U12345",
            "text": "no thread",
            "channel": "C98765",
            "ts": "1234567890.123456",
        }
        say = AsyncMock()
        await adapter.handle_message_event(event=event, say=say)

        msg = router.last_message
        assert msg is not None
        assert msg.reply_to_id is None

    @patch("threetears.channels.slack.AsyncApp")
    async def test_file_attachments_mapped(self, mock_app_cls: MagicMock) -> None:
        """Slack file objects are mapped to Attachment dataclass instances."""
        from threetears.channels.slack import SlackAdapter

        response = ChannelResponse(content="got it")
        router = _MockRouter(response=response)
        mock_app = MagicMock()
        mock_app_cls.return_value = mock_app

        adapter = SlackAdapter(
            bot_token="xoxb-test-token",
            app_token="xapp-test-token",
            router=router,
        )

        event: dict[str, Any] = {
            "type": "message",
            "user": "U12345",
            "text": "see attachment",
            "channel": "C98765",
            "ts": "1234567890.123456",
            "files": [
                {
                    "name": "report.pdf",
                    "mimetype": "application/pdf",
                    "title": "quarterly report",
                },
                {
                    "name": "image.png",
                    "mimetype": "image/png",
                },
            ],
        }
        say = AsyncMock()
        await adapter.handle_message_event(event=event, say=say)

        msg = router.last_message
        assert msg is not None
        assert len(msg.attachments) == 2
        assert msg.attachments[0].filename == "report.pdf"
        assert msg.attachments[0].content_type == "application/pdf"
        assert msg.attachments[0].description == "quarterly report"
        assert msg.attachments[0].content == b""
        assert msg.attachments[1].filename == "image.png"
        assert msg.attachments[1].content_type == "image/png"
        assert msg.attachments[1].description is None

    @patch("threetears.channels.slack.AsyncApp")
    async def test_timestamp_is_utc(self, mock_app_cls: MagicMock) -> None:
        """inbound ChannelMessage timestamp is UTC-aware."""
        from threetears.channels.slack import SlackAdapter

        response = ChannelResponse(content="reply")
        router = _MockRouter(response=response)
        mock_app = MagicMock()
        mock_app_cls.return_value = mock_app

        adapter = SlackAdapter(
            bot_token="xoxb-test-token",
            app_token="xapp-test-token",
            router=router,
        )

        event: dict[str, Any] = {
            "type": "message",
            "user": "U12345",
            "text": "hello",
            "channel": "C98765",
            "ts": "1234567890.123456",
        }
        say = AsyncMock()
        await adapter.handle_message_event(event=event, say=say)

        msg = router.last_message
        assert msg is not None
        assert msg.timestamp.tzinfo is not None
        assert msg.timestamp.tzinfo == UTC

    @patch("threetears.channels.slack.AsyncApp")
    async def test_metadata_contains_slack_specific_fields(self, mock_app_cls: MagicMock) -> None:
        """metadata captures Slack-specific fields not in standard ChannelMessage."""
        from threetears.channels.slack import SlackAdapter

        response = ChannelResponse(content="reply")
        router = _MockRouter(response=response)
        mock_app = MagicMock()
        mock_app_cls.return_value = mock_app

        adapter = SlackAdapter(
            bot_token="xoxb-test-token",
            app_token="xapp-test-token",
            router=router,
        )

        event: dict[str, Any] = {
            "type": "message",
            "user": "U12345",
            "text": "hello",
            "channel": "C98765",
            "ts": "1234567890.123456",
            "team": "T00001",
            "channel_type": "channel",
            "client_msg_id": "unique-msg-id",
        }
        say = AsyncMock()
        await adapter.handle_message_event(event=event, say=say)

        msg = router.last_message
        assert msg is not None
        assert msg.metadata.get("ts") == "1234567890.123456"
        assert msg.metadata.get("channel_type") == "channel"
        assert msg.metadata.get("client_msg_id") == "unique-msg-id"


# ---------------------------------------------------------------------------
# threading model tests
# ---------------------------------------------------------------------------


class TestSlackAdapterThreading:
    """tests for threading behavior of SlackAdapter responses."""

    @patch("threetears.channels.slack.AsyncApp")
    async def test_threaded_reply_uses_existing_thread_ts(self, mock_app_cls: MagicMock) -> None:
        """reply to threaded message uses thread_ts from event."""
        from threetears.channels.slack import SlackAdapter

        response = ChannelResponse(content="threaded reply")
        router = _MockRouter(response=response)
        mock_app = MagicMock()
        mock_app_cls.return_value = mock_app

        adapter = SlackAdapter(
            bot_token="xoxb-test-token",
            app_token="xapp-test-token",
            router=router,
        )

        event: dict[str, Any] = {
            "type": "message",
            "user": "U12345",
            "text": "in thread",
            "channel": "C98765",
            "ts": "1234567890.999999",
            "thread_ts": "1234567890.000001",
        }
        say = AsyncMock()
        await adapter.handle_message_event(event=event, say=say)
        say.assert_awaited_once()
        kwargs = say.await_args.kwargs
        assert kwargs["text"] == "threaded reply"
        assert kwargs["thread_ts"] == "1234567890.000001"

    @patch("threetears.channels.slack.AsyncApp")
    async def test_channel_message_starts_new_thread(self, mock_app_cls: MagicMock) -> None:
        """reply to top-level channel message starts new thread using event ts."""
        from threetears.channels.slack import SlackAdapter

        response = ChannelResponse(content="new thread reply")
        router = _MockRouter(response=response)
        mock_app = MagicMock()
        mock_app_cls.return_value = mock_app

        adapter = SlackAdapter(
            bot_token="xoxb-test-token",
            app_token="xapp-test-token",
            router=router,
        )

        event: dict[str, Any] = {
            "type": "message",
            "user": "U12345",
            "text": "top level",
            "channel": "C98765",
            "ts": "1234567890.123456",
            "channel_type": "channel",
        }
        say = AsyncMock()
        await adapter.handle_message_event(event=event, say=say)
        say.assert_awaited_once()
        kwargs = say.await_args.kwargs
        assert kwargs["text"] == "new thread reply"
        assert kwargs["thread_ts"] == "1234567890.123456"

    @patch("threetears.channels.slack.AsyncApp")
    async def test_dm_message_replies_without_thread(self, mock_app_cls: MagicMock) -> None:
        """reply to DM message does not use thread_ts (replies in DM channel)."""
        from threetears.channels.slack import SlackAdapter

        response = ChannelResponse(content="dm reply")
        router = _MockRouter(response=response)
        mock_app = MagicMock()
        mock_app_cls.return_value = mock_app

        adapter = SlackAdapter(
            bot_token="xoxb-test-token",
            app_token="xapp-test-token",
            router=router,
        )

        event: dict[str, Any] = {
            "type": "message",
            "user": "U12345",
            "text": "dm message",
            "channel": "D98765",
            "ts": "1234567890.123456",
            "channel_type": "im",
        }
        say = AsyncMock()
        await adapter.handle_message_event(event=event, say=say)
        say.assert_awaited_once()
        kwargs = say.await_args.kwargs
        assert kwargs["text"] == "dm reply"
        assert "thread_ts" not in kwargs


# ---------------------------------------------------------------------------
# response routing tests
# ---------------------------------------------------------------------------


class TestSlackAdapterResponseRouting:
    """tests for outbound response delivery."""

    @patch("threetears.channels.slack.AsyncApp")
    async def test_router_called_with_correct_channel_message(self, mock_app_cls: MagicMock) -> None:
        """route_inbound receives correctly normalized ChannelMessage."""
        from threetears.channels.slack import SlackAdapter

        response = ChannelResponse(content="ack")
        router = _MockRouter(response=response)
        mock_app = MagicMock()
        mock_app_cls.return_value = mock_app

        adapter = SlackAdapter(
            bot_token="xoxb-test-token",
            app_token="xapp-test-token",
            router=router,
        )

        event: dict[str, Any] = {
            "type": "message",
            "user": "U12345",
            "text": "specific content",
            "channel": "C98765",
            "ts": "1234567890.123456",
            "team": "T00001",
        }
        say = AsyncMock()
        await adapter.handle_message_event(event=event, say=say)

        msg = router.last_message
        assert msg is not None
        assert isinstance(msg, ChannelMessage)
        assert msg.channel_type == "slack"
        assert msg.content == "specific content"
        assert msg.sender_id == "U12345"
        assert msg.channel_id == "C98765"
        assert msg.workspace_id == "T00001"

    @patch("threetears.channels.slack.AsyncApp")
    async def test_no_response_from_router_skips_say(self, mock_app_cls: MagicMock) -> None:
        """when router returns None, say() is not called."""
        from threetears.channels.slack import SlackAdapter

        router = _MockRouter(response=None)
        mock_app = MagicMock()
        mock_app_cls.return_value = mock_app

        adapter = SlackAdapter(
            bot_token="xoxb-test-token",
            app_token="xapp-test-token",
            router=router,
        )

        event: dict[str, Any] = {
            "type": "message",
            "user": "U12345",
            "text": "ignored",
            "channel": "C98765",
            "ts": "1234567890.123456",
        }
        say = AsyncMock()
        await adapter.handle_message_event(event=event, say=say)
        say.assert_not_awaited()

    @patch("threetears.channels.slack.AsyncApp")
    async def test_response_with_attachments_uploads_files(self, mock_app_cls: MagicMock) -> None:
        """response attachments are uploaded via files_upload_v2."""
        from threetears.channels.slack import SlackAdapter

        attachment = Attachment(
            filename="data.csv",
            content=b"col1,col2\na,b",
            content_type="text/csv",
            description="export",
        )
        response = ChannelResponse(
            content="here are results",
            attachments=[attachment],
        )
        router = _MockRouter(response=response)
        mock_app = MagicMock()
        mock_client = AsyncMock()
        # ``_resolve_user_locale`` calls
        # ``await self._app.client.users_info(...)`` then reads
        # ``response.get("user", {})`` synchronously. without a real
        # return_value the auto-generated ``users_info`` mock returns
        # an ``AsyncMock``, whose ``.get`` is itself async -- calling
        # it produces an orphan coroutine the production code never
        # awaits. an explicit empty-dict return value collapses the
        # ``if response else {}`` branch and avoids the auto-generated
        # async child entirely.
        mock_client.users_info = AsyncMock(return_value={})
        mock_app.client = mock_client
        mock_app_cls.return_value = mock_app

        adapter = SlackAdapter(
            bot_token="xoxb-test-token",
            app_token="xapp-test-token",
            router=router,
        )

        event: dict[str, Any] = {
            "type": "message",
            "user": "U12345",
            "text": "get data",
            "channel": "C98765",
            "ts": "1234567890.123456",
            "thread_ts": "1234567890.000001",
        }
        say = AsyncMock()
        await adapter.handle_message_event(event=event, say=say)

        mock_client.files_upload_v2.assert_awaited_once_with(
            channel="C98765",
            filename="data.csv",
            content=b"col1,col2\na,b",
            title="data.csv",
            thread_ts="1234567890.000001",
        )


# ---------------------------------------------------------------------------
# ChannelRouter protocol conformance
# ---------------------------------------------------------------------------


class TestSlackAdapterProtocol:
    """tests verifying protocol conformance and package exports."""

    def test_mock_router_satisfies_channel_router_protocol(self) -> None:
        """_MockRouter used in tests satisfies ChannelRouter protocol."""
        router = _MockRouter()
        assert isinstance(router, ChannelRouter)

    def test_slack_adapter_importable_from_package(self) -> None:
        """SlackAdapter is importable from threetears.channels."""
        from threetears.channels import SlackAdapter

        assert SlackAdapter is not None

    def test_slack_adapter_in_package_all(self) -> None:
        """SlackAdapter appears in threetears.channels.__all__."""
        import threetears.channels as channels_pkg

        assert "SlackAdapter" in channels_pkg.__all__


# ---------------------------------------------------------------------------
# rich formatting integration tests
# ---------------------------------------------------------------------------


class TestSlackAdapterRichFormatting:
    """tests for rich formatting integration in _send_response."""

    @patch("threetears.channels.slack.AsyncApp")
    async def test_rich_formatting_sends_blocks(self, mock_app_cls: MagicMock) -> None:
        """when format_hints has format=rich, say() receives blocks kwarg."""
        from threetears.channels.slack import SlackAdapter

        response = ChannelResponse(
            content="**bold text** and more",
            format_hints={"format": "rich"},
        )
        router = _MockRouter(response=response)
        mock_app = MagicMock()
        mock_app_cls.return_value = mock_app

        adapter = SlackAdapter(
            bot_token="xoxb-test-token",
            app_token="xapp-test-token",
            router=router,
        )

        event: dict[str, Any] = {
            "type": "message",
            "user": "U12345",
            "text": "hello",
            "channel": "C98765",
            "ts": "1234567890.123456",
            "thread_ts": "1234567890.000001",
        }
        say = AsyncMock()
        await adapter.handle_message_event(event=event, say=say)

        say.assert_awaited_once()
        call_kwargs = say.await_args.kwargs
        assert "blocks" in call_kwargs
        assert isinstance(call_kwargs["blocks"], list)
        assert len(call_kwargs["blocks"]) > 0

    @patch("threetears.channels.slack.AsyncApp")
    async def test_plain_content_rendered_to_blocks_with_text_fallback(self, mock_app_cls: MagicMock) -> None:
        """every answer renders to blocks; plain text becomes one mrkdwn section.

        the agent's answers are markdown regardless of any format_hints, so the
        adapter always renders them into native Slack blocks (Slack does not
        render GitHub markdown in the ``text`` field). the ``text`` field carries
        the plain fallback for notifications / screen readers.
        """
        from threetears.channels.slack import SlackAdapter

        response = ChannelResponse(content="plain reply")
        router = _MockRouter(response=response)
        mock_app = MagicMock()
        mock_app_cls.return_value = mock_app

        adapter = SlackAdapter(
            bot_token="xoxb-test-token",
            app_token="xapp-test-token",
            router=router,
        )

        event: dict[str, Any] = {
            "type": "message",
            "user": "U12345",
            "text": "hello",
            "channel": "C98765",
            "ts": "1234567890.123456",
            "thread_ts": "1234567890.000001",
        }
        say = AsyncMock()
        await adapter.handle_message_event(event=event, say=say)

        say.assert_awaited_once()
        call_kwargs = say.await_args.kwargs
        assert call_kwargs.get("text") == "plain reply"
        assert call_kwargs["blocks"] == [
            {"type": "section", "text": {"type": "mrkdwn", "text": "plain reply"}},
        ]

    @patch("threetears.channels.slack.AsyncApp")
    async def test_rich_formatting_includes_text_fallback(self, mock_app_cls: MagicMock) -> None:
        """when rich formatting, say() also receives plain text fallback."""
        from threetears.channels.slack import SlackAdapter

        response = ChannelResponse(
            content="**important** message",
            format_hints={"format": "rich"},
        )
        router = _MockRouter(response=response)
        mock_app = MagicMock()
        mock_app_cls.return_value = mock_app

        adapter = SlackAdapter(
            bot_token="xoxb-test-token",
            app_token="xapp-test-token",
            router=router,
        )

        event: dict[str, Any] = {
            "type": "message",
            "user": "U12345",
            "text": "hello",
            "channel": "C98765",
            "ts": "1234567890.123456",
            "thread_ts": "1234567890.000001",
        }
        say = AsyncMock()
        await adapter.handle_message_event(event=event, say=say)

        say.assert_awaited_once()
        call_kwargs = say.await_args.kwargs
        assert "text" in call_kwargs
        assert "blocks" in call_kwargs
        # fallback text should be plain (markdown stripped)
        assert "**" not in call_kwargs["text"]


# ---------------------------------------------------------------------------
# out-of-band durable delivery (post_message)
# ---------------------------------------------------------------------------


class TestSlackPostMessage:
    """tests for SlackAdapter.post_message (durable answer delivery path)."""

    @patch("threetears.channels.slack.AsyncApp")
    async def test_renders_markdown_to_blocks_with_thread(self, mock_app_cls: MagicMock) -> None:
        """post_message renders markdown into blocks and threads the reply."""
        from threetears.channels.slack import SlackAdapter

        mock_app = MagicMock()
        mock_app.client.chat_postMessage = AsyncMock()
        mock_app_cls.return_value = mock_app

        adapter = SlackAdapter(
            bot_token="xoxb-test-token",
            app_token="xapp-test-token",
            router=_MockRouter(response=ChannelResponse(content="")),
        )

        content = "**Top result:** here it is\n\n| County | Votes |\n| --- | --- |\n| Acme | 1200 |\n"
        await adapter.post_message(
            channel="C123",
            text=content,
            thread_ts="1700000000.000100",
        )

        mock_app.client.chat_postMessage.assert_awaited_once()
        kwargs = mock_app.client.chat_postMessage.await_args.kwargs
        assert kwargs["channel"] == "C123"
        assert kwargs["thread_ts"] == "1700000000.000100"
        # rendered blocks present; the markdown table became a native table block.
        block_types = [b["type"] for b in kwargs["blocks"]]
        assert "table" in block_types
        # the notification text is the plain fallback (markdown stripped).
        assert "**" not in kwargs["text"]

    @patch("threetears.channels.slack.AsyncApp")
    async def test_top_level_post_has_no_thread_ts(self, mock_app_cls: MagicMock) -> None:
        """post_message without thread_ts posts at top level (no thread key)."""
        from threetears.channels.slack import SlackAdapter

        mock_app = MagicMock()
        mock_app.client.chat_postMessage = AsyncMock()
        mock_app_cls.return_value = mock_app

        adapter = SlackAdapter(
            bot_token="xoxb-test-token",
            app_token="xapp-test-token",
            router=_MockRouter(response=ChannelResponse(content="")),
        )

        await adapter.post_message(channel="C123", text="plain answer")

        kwargs = mock_app.client.chat_postMessage.await_args.kwargs
        assert kwargs["channel"] == "C123"
        assert "thread_ts" not in kwargs
        assert kwargs["text"] == "plain answer"
