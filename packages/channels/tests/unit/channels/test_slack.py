"""tests for SlackAdapter channel adapter."""

from __future__ import annotations

import ast
import inspect
import time
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
# mutation-subtype + pre-startup replay guard tests
# ---------------------------------------------------------------------------


class TestSlackAdapterReplayGuard:
    """tests for dropping edit/delete subtypes and pre-startup replays."""

    @patch("threetears.channels.slack.AsyncApp")
    async def test_filters_message_deleted_subtype(self, mock_app_cls: MagicMock) -> None:
        """events with subtype 'message_deleted' are filtered out."""
        from threetears.channels.slack import SlackAdapter

        router = _MockRouter()
        mock_app_cls.return_value = MagicMock()
        adapter = SlackAdapter(
            bot_token="xoxb-test-token",
            app_token="xapp-test-token",
            router=router,
        )

        event: dict[str, Any] = {
            "type": "message",
            "subtype": "message_deleted",
            "channel": "C123",
            "ts": "1234567890.123456",
        }
        say = AsyncMock()
        await adapter.handle_message_event(event=event, say=say)
        assert router.last_message is None
        say.assert_not_awaited()

    @patch("threetears.channels.slack.AsyncApp")
    async def test_filters_message_changed_subtype(self, mock_app_cls: MagicMock) -> None:
        """events with subtype 'message_changed' (edits) are filtered out."""
        from threetears.channels.slack import SlackAdapter

        router = _MockRouter()
        mock_app_cls.return_value = MagicMock()
        adapter = SlackAdapter(
            bot_token="xoxb-test-token",
            app_token="xapp-test-token",
            router=router,
        )

        event: dict[str, Any] = {
            "type": "message",
            "subtype": "message_changed",
            "channel": "C123",
            "ts": "1234567890.123456",
        }
        say = AsyncMock()
        await adapter.handle_message_event(event=event, say=say)
        assert router.last_message is None
        say.assert_not_awaited()

    @patch("threetears.channels.slack.AsyncSocketModeHandler")
    @patch("threetears.channels.slack.AsyncApp")
    async def test_drops_event_posted_before_start(
        self,
        mock_app_cls: MagicMock,
        mock_handler_cls: MagicMock,
    ) -> None:
        """a message whose ts predates adapter start is a replay, dropped.

        reproduces the ghost-answer case: a question posted while the
        adapter was down is redelivered by Slack after reconnect; its ts
        predates this process's start, so it must not be answered.
        """
        from threetears.channels.slack import SlackAdapter

        router = _MockRouter(response=ChannelResponse(content="reply"))
        mock_app_cls.return_value = MagicMock()
        mock_handler = MagicMock()
        mock_handler.start_async = AsyncMock()
        mock_handler_cls.return_value = mock_handler

        adapter = SlackAdapter(
            bot_token="xoxb-test-token",
            app_token="xapp-test-token",
            router=router,
        )
        await adapter.start()

        # posted 10 minutes before the adapter began listening
        event: dict[str, Any] = {
            "type": "message",
            "user": "U12345",
            "text": "stale ghost question",
            "channel": "C98765",
            "ts": f"{time.time() - 600:.6f}",
            "team": "T00001",
        }
        say = AsyncMock()
        await adapter.handle_message_event(event=event, say=say)
        assert router.last_message is None
        say.assert_not_awaited()

    @patch("threetears.channels.slack.AsyncSocketModeHandler")
    @patch("threetears.channels.slack.AsyncApp")
    async def test_processes_event_posted_after_start(
        self,
        mock_app_cls: MagicMock,
        mock_handler_cls: MagicMock,
    ) -> None:
        """a fresh message posted after adapter start routes normally."""
        from threetears.channels.slack import SlackAdapter

        router = _MockRouter(response=ChannelResponse(content="reply"))
        mock_app_cls.return_value = MagicMock()
        mock_handler = MagicMock()
        mock_handler.start_async = AsyncMock()
        mock_handler_cls.return_value = mock_handler

        adapter = SlackAdapter(
            bot_token="xoxb-test-token",
            app_token="xapp-test-token",
            router=router,
        )
        await adapter.start()

        event: dict[str, Any] = {
            "type": "message",
            "user": "U12345",
            "text": "fresh question",
            "channel": "C98765",
            "ts": f"{time.time():.6f}",
            "team": "T00001",
        }
        say = AsyncMock()
        await adapter.handle_message_event(event=event, say=say)
        assert router.last_message is not None
        assert router.last_message.content == "fresh question"

    @patch("threetears.channels.slack.AsyncApp")
    async def test_no_replay_guard_before_start(self, mock_app_cls: MagicMock) -> None:
        """without a start baseline the guard is inactive (old ts still routes).

        an adapter that never called start (unit-test construction) does
        not drop by age -- the guard only arms once the process baseline
        is recorded.
        """
        from threetears.channels.slack import SlackAdapter

        router = _MockRouter(response=ChannelResponse(content="reply"))
        mock_app_cls.return_value = MagicMock()
        adapter = SlackAdapter(
            bot_token="xoxb-test-token",
            app_token="xapp-test-token",
            router=router,
        )

        event: dict[str, Any] = {
            "type": "message",
            "user": "U12345",
            "text": "old ts but no baseline",
            "channel": "C98765",
            "ts": "1234567890.123456",
            "team": "T00001",
        }
        say = AsyncMock()
        await adapter.handle_message_event(event=event, say=say)
        assert router.last_message is not None


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


# ---------------------------------------------------------------------------
# sender identity: name + address off the one users.info lookup
# ---------------------------------------------------------------------------


# comfortably past ``_USER_PROFILE_TTL_SECONDS`` (300s), so a clock moved by
# this much expires any cache entry regardless of later tuning of the TTL.
_TTL_OVERSHOOT_SECONDS = 3600.0


def _users_info_response(
    *,
    profile: dict[str, Any] | None = None,
    tz: str | None = "America/Los_Angeles",
    locale: str | None = "en-US",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """build a ``users.info`` reply shaped like slack's documented user object.

    :param profile: the nested ``user.profile`` object
    :ptype profile: dict[str, Any] | None
    :param tz: ``user.tz``
    :ptype tz: str | None
    :param locale: ``user.locale``
    :ptype locale: str | None
    :param extra: additional top-level ``user`` keys
    :ptype extra: dict[str, Any] | None
    :return: a full ``users.info`` response payload
    :rtype: dict[str, Any]
    """
    user: dict[str, Any] = {"id": "U12345", "tz": tz, "locale": locale}
    user["profile"] = profile if profile is not None else {}
    if extra:
        user.update(extra)
    return {"ok": True, "user": user}


def _message_event(*, user: str = "U12345", text: str = "hello") -> dict[str, Any]:
    """build a minimal inbound slack ``message`` event.

    :param user: slack sender id
    :ptype user: str
    :param text: message body
    :ptype text: str
    :return: a slack message event payload
    :rtype: dict[str, Any]
    """
    return {
        "type": "message",
        "user": user,
        "text": text,
        "channel": "C98765",
        "team": "T00001",
        "ts": "1234567890.123456",
    }


class TestSlackSenderIdentity:
    """tests for sender name / address carried onto ChannelMessage.

    driven through ``handle_message_event`` rather than the private resolver:
    a host reconciling a chat participant with a known identity reads these
    off the routed ``ChannelMessage``, so that is the surface worth pinning.
    """

    @patch("threetears.channels.slack.AsyncApp")
    async def test_name_and_address_reach_the_routed_message(self, mock_app_cls: MagicMock) -> None:
        """users.info name/email/tz/locale all land on the ChannelMessage."""
        from threetears.channels.slack import SlackAdapter

        router = _MockRouter()
        mock_app = MagicMock()
        mock_app.client = AsyncMock()
        mock_app.client.users_info = AsyncMock(
            return_value=_users_info_response(
                profile={"display_name": "alice", "real_name": "Alice Doe", "email": "alice@acme.example"},
            ),
        )
        mock_app_cls.return_value = mock_app

        adapter = SlackAdapter(bot_token="xoxb-t", app_token="xapp-t", router=router)
        await adapter.handle_message_event(event=_message_event(), say=AsyncMock())

        msg = router.last_message
        assert msg is not None
        assert msg.sender_name == "alice"
        assert msg.sender_email == "alice@acme.example"
        assert msg.sender_email_verified is True
        assert msg.user_timezone == "America/Los_Angeles"
        assert msg.user_locale == "en-US"

    @patch("threetears.channels.slack.AsyncApp")
    async def test_one_lookup_serves_every_field(self, mock_app_cls: MagicMock) -> None:
        """a single message issues exactly ONE users.info call.

        the address must not cost a second Tier-4 call: the response that
        already carries tz and locale carries the profile too.
        """
        from threetears.channels.slack import SlackAdapter

        mock_app = MagicMock()
        mock_app.client = AsyncMock()
        mock_app.client.users_info = AsyncMock(
            return_value=_users_info_response(profile={"display_name": "alice", "email": "alice@acme.example"}),
        )
        mock_app_cls.return_value = mock_app

        adapter = SlackAdapter(bot_token="xoxb-t", app_token="xapp-t", router=_MockRouter())
        await adapter.handle_message_event(event=_message_event(), say=AsyncMock())

        assert mock_app.client.users_info.await_count == 1

    @patch("threetears.channels.slack.AsyncApp")
    async def test_second_message_from_same_user_is_served_from_cache(self, mock_app_cls: MagicMock) -> None:
        """repeat messages from one sender do not re-hit users.info."""
        from threetears.channels.slack import SlackAdapter

        router = _MockRouter()
        mock_app = MagicMock()
        mock_app.client = AsyncMock()
        mock_app.client.users_info = AsyncMock(
            return_value=_users_info_response(profile={"display_name": "alice", "email": "alice@acme.example"}),
        )
        mock_app_cls.return_value = mock_app

        adapter = SlackAdapter(bot_token="xoxb-t", app_token="xapp-t", router=router)
        await adapter.handle_message_event(event=_message_event(text="one"), say=AsyncMock())
        await adapter.handle_message_event(event=_message_event(text="two"), say=AsyncMock())

        assert mock_app.client.users_info.await_count == 1
        # the cached profile is still applied to the second message, not dropped.
        msg = router.last_message
        assert msg is not None
        assert msg.content == "two"
        assert msg.sender_email == "alice@acme.example"

    @patch("threetears.channels.slack.AsyncApp")
    async def test_missing_email_scope_yields_no_address_and_no_assertion(self, mock_app_cls: MagicMock) -> None:
        """without ``users:read.email`` slack omits the field SILENTLY.

        the response is otherwise a complete success, so the only correct
        reading is "no address", never a partial or guessed one -- and the
        assertion flag must stay False so a host does not record something
        it never received.
        """
        from threetears.channels.slack import SlackAdapter

        router = _MockRouter()
        mock_app = MagicMock()
        mock_app.client = AsyncMock()
        mock_app.client.users_info = AsyncMock(
            return_value=_users_info_response(profile={"display_name": "alice", "real_name": "Alice Doe"}),
        )
        mock_app_cls.return_value = mock_app

        adapter = SlackAdapter(bot_token="xoxb-t", app_token="xapp-t", router=router)
        await adapter.handle_message_event(event=_message_event(), say=AsyncMock())

        msg = router.last_message
        assert msg is not None
        assert msg.sender_email is None
        assert msg.sender_email_verified is False
        # the name still arrives -- the missing scope costs only the address.
        assert msg.sender_name == "alice"

    @patch("threetears.channels.slack.AsyncApp")
    async def test_undocumented_is_email_confirmed_is_not_consulted(self, mock_app_cls: MagicMock) -> None:
        """``is_email_confirmed`` is absent from slack's documented user object.

        some responses carry it; identity trust is not gated on an
        undocumented field, so a False value must NOT suppress an address
        slack actually returned.
        """
        from threetears.channels.slack import SlackAdapter

        router = _MockRouter()
        mock_app = MagicMock()
        mock_app.client = AsyncMock()
        mock_app.client.users_info = AsyncMock(
            return_value=_users_info_response(
                profile={"display_name": "alice", "email": "alice@acme.example"},
                extra={"is_email_confirmed": False},
            ),
        )
        mock_app_cls.return_value = mock_app

        adapter = SlackAdapter(bot_token="xoxb-t", app_token="xapp-t", router=router)
        await adapter.handle_message_event(event=_message_event(), say=AsyncMock())

        msg = router.last_message
        assert msg is not None
        assert msg.sender_email == "alice@acme.example"
        assert msg.sender_email_verified is True

    @patch("threetears.channels.slack.AsyncApp")
    async def test_display_name_falls_back_through_real_name_to_handle(self, mock_app_cls: MagicMock) -> None:
        """a workspace with no display names still yields a name."""
        from threetears.channels.slack import SlackAdapter

        router = _MockRouter()
        mock_app = MagicMock()
        mock_app.client = AsyncMock()
        mock_app.client.users_info = AsyncMock(
            return_value=_users_info_response(profile={"real_name": "Alice Doe"}),
        )
        mock_app_cls.return_value = mock_app

        adapter = SlackAdapter(bot_token="xoxb-t", app_token="xapp-t", router=router)
        await adapter.handle_message_event(event=_message_event(), say=AsyncMock())

        msg = router.last_message
        assert msg is not None
        assert msg.sender_name == "Alice Doe"

    @patch("threetears.channels.slack.AsyncApp")
    async def test_empty_strings_normalize_to_none(self, mock_app_cls: MagicMock) -> None:
        """slack renders absent data as "" as well as null; both mean absent."""
        from threetears.channels.slack import SlackAdapter

        router = _MockRouter()
        mock_app = MagicMock()
        mock_app.client = AsyncMock()
        mock_app.client.users_info = AsyncMock(
            return_value=_users_info_response(
                profile={"display_name": "", "real_name": "", "email": ""},
                tz="",
                locale="",
            ),
        )
        mock_app_cls.return_value = mock_app

        adapter = SlackAdapter(bot_token="xoxb-t", app_token="xapp-t", router=router)
        await adapter.handle_message_event(event=_message_event(), say=AsyncMock())

        msg = router.last_message
        assert msg is not None
        assert msg.sender_name is None
        assert msg.sender_email is None
        assert msg.sender_email_verified is False
        assert msg.user_timezone is None
        assert msg.user_locale is None

    @patch("threetears.channels.slack.AsyncApp")
    async def test_lookup_failure_still_routes_the_message(self, mock_app_cls: MagicMock) -> None:
        """a users.info outage must not swallow the user's message."""
        from threetears.channels.slack import SlackAdapter

        router = _MockRouter()
        mock_app = MagicMock()
        mock_app.client = AsyncMock()
        mock_app.client.users_info = AsyncMock(side_effect=RuntimeError("slack is down"))
        mock_app_cls.return_value = mock_app

        adapter = SlackAdapter(bot_token="xoxb-t", app_token="xapp-t", router=router)
        await adapter.handle_message_event(event=_message_event(text="still routed"), say=AsyncMock())

        msg = router.last_message
        assert msg is not None
        assert msg.content == "still routed"
        assert msg.sender_name is None
        assert msg.sender_email is None
        assert msg.sender_email_verified is False


class TestSlackProfileCacheBound:
    """tests for the per-user profile cache staying bounded."""

    @patch("threetears.channels.slack.AsyncApp")
    async def test_cache_evicts_least_recently_used_past_the_cap(self, mock_app_cls: MagicMock) -> None:
        """the cache drops its LRU entry once full, and keeps the recent ones.

        asserted through the observable consequence -- an evicted user costs
        another ``users.info`` call, a retained one does not -- rather than by
        reaching into the map, so the test pins the behaviour and not the
        choice of container.

        entries are keyed by slack user id and were previously only ever
        overwritten, so an adapter in a busy workspace grew one entry per
        person who ever spoke and released none of them.
        """
        from threetears.channels import slack as slack_mod
        from threetears.channels.slack import SlackAdapter

        mock_app = MagicMock()
        mock_app.client = AsyncMock()
        mock_app.client.users_info = AsyncMock(
            return_value=_users_info_response(profile={"display_name": "someone"}),
        )
        mock_app_cls.return_value = mock_app

        adapter = SlackAdapter(bot_token="xoxb-t", app_token="xapp-t", router=_MockRouter())
        lookups = mock_app.client.users_info

        with patch.object(slack_mod, "_USER_PROFILE_CACHE_MAX_ENTRIES", 2):
            await adapter.handle_message_event(event=_message_event(user="U1"), say=AsyncMock())
            await adapter.handle_message_event(event=_message_event(user="U2"), say=AsyncMock())
            assert lookups.await_count == 2

            # a cache HIT, which also makes U2 the least recently used.
            await adapter.handle_message_event(event=_message_event(user="U1"), say=AsyncMock())
            assert lookups.await_count == 2

            # U3 overflows the cap of 2, so something must go.
            await adapter.handle_message_event(event=_message_event(user="U3"), say=AsyncMock())
            assert lookups.await_count == 3

            # U1 survived, because USING it moved it off the LRU end. this is
            # the assertion that separates an LRU from a plain drop-the-oldest-
            # insert cache: under the latter U1 would have been the one evicted
            # above, and this line would cost a fourth lookup.
            await adapter.handle_message_event(event=_message_event(user="U1"), say=AsyncMock())
            assert lookups.await_count == 3

            # ...so U2 is the one that went, and it costs a re-fetch.
            await adapter.handle_message_event(event=_message_event(user="U2"), say=AsyncMock())
            assert lookups.await_count == 4

    @patch("threetears.channels.slack.AsyncApp")
    async def test_expired_entry_is_refetched(self, mock_app_cls: MagicMock) -> None:
        """past the TTL the profile is looked up again rather than served stale."""
        from threetears.channels.slack import SlackAdapter

        mock_app = MagicMock()
        mock_app.client = AsyncMock()
        mock_app.client.users_info = AsyncMock(
            return_value=_users_info_response(profile={"display_name": "alice", "email": "alice@acme.example"}),
        )
        mock_app_cls.return_value = mock_app

        adapter = SlackAdapter(bot_token="xoxb-t", app_token="xapp-t", router=_MockRouter())
        await adapter.handle_message_event(event=_message_event(), say=AsyncMock())
        assert mock_app.client.users_info.await_count == 1

        # jump the clock past the TTL rather than sleeping through it. the
        # adapter reads ``time.monotonic`` for this cache and nothing else, so
        # moving it cannot disturb the rest of the message path.
        far_future = time.monotonic() + _TTL_OVERSHOOT_SECONDS
        with patch("threetears.channels.slack.time.monotonic", return_value=far_future):
            await adapter.handle_message_event(event=_message_event(), say=AsyncMock())

        assert mock_app.client.users_info.await_count == 2
