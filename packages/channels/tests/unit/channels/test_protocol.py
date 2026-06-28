"""tests for ChannelMessage, ChannelResponse, Attachment, and ChannelRouter protocol."""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime, timezone

import pytest

from threetears.channels.protocol import (
    Attachment,
    ChannelMessage,
    ChannelResponse,
    ChannelRouter,
)


# -- Attachment tests --


class TestAttachment:
    """tests for Attachment dataclass."""

    def test_attachment_is_dataclass(self) -> None:
        """Attachment is a dataclass, not a Pydantic BaseModel."""
        assert dataclasses.is_dataclass(Attachment)

    def test_attachment_is_not_pydantic(self) -> None:
        """Attachment does not inherit from pydantic BaseModel."""
        assert not hasattr(Attachment, "model_fields")
        assert not hasattr(Attachment, "model_validate")

    def test_attachment_creation_all_fields(self) -> None:
        """Attachment stores all fields correctly."""
        content = b"file-content-bytes"
        attachment = Attachment(
            filename="report.pdf",
            content=content,
            content_type="application/pdf",
            description="quarterly report",
        )
        assert attachment.filename == "report.pdf"
        assert attachment.content == content
        assert attachment.content_type == "application/pdf"
        assert attachment.description == "quarterly report"

    def test_attachment_description_defaults_to_none(self) -> None:
        """Attachment description defaults to None when not provided."""
        attachment = Attachment(
            filename="image.png",
            content=b"\x89PNG",
            content_type="image/png",
        )
        assert attachment.description is None

    def test_attachment_content_is_bytes(self) -> None:
        """Attachment content field holds raw bytes."""
        raw = b"\x00\x01\x02\x03"
        attachment = Attachment(
            filename="data.bin",
            content=raw,
            content_type="application/octet-stream",
        )
        assert isinstance(attachment.content, bytes)
        assert attachment.content == raw


# -- ChannelMessage tests --


class TestChannelMessage:
    """tests for ChannelMessage dataclass."""

    def test_channel_message_is_dataclass(self) -> None:
        """ChannelMessage is a dataclass, not a Pydantic BaseModel."""
        assert dataclasses.is_dataclass(ChannelMessage)

    def test_channel_message_is_not_pydantic(self) -> None:
        """ChannelMessage does not inherit from pydantic BaseModel."""
        assert not hasattr(ChannelMessage, "model_fields")
        assert not hasattr(ChannelMessage, "model_validate")

    def test_channel_message_required_fields(self) -> None:
        """ChannelMessage requires channel_type, content, and sender_id."""
        msg = ChannelMessage(
            channel_type="slack",
            content="hello world",
            sender_id="U12345",
        )
        assert msg.channel_type == "slack"
        assert msg.content == "hello world"
        assert msg.sender_id == "U12345"

    def test_channel_message_defaults(self) -> None:
        """ChannelMessage optional fields default to None or empty."""
        msg = ChannelMessage(
            channel_type="discord",
            content="test",
            sender_id="user-1",
        )
        assert msg.sender_name is None
        assert msg.customer_id is None
        assert msg.conversation_id is None
        assert msg.channel_id is None
        assert msg.workspace_id is None
        assert msg.attachments == []
        assert msg.reply_to_id is None
        assert msg.metadata == {}

    def test_channel_message_timestamp_defaults_to_utc(self) -> None:
        """ChannelMessage timestamp defaults to UTC-aware datetime."""
        before = datetime.now(UTC)
        msg = ChannelMessage(
            channel_type="websocket",
            content="test",
            sender_id="user-1",
        )
        after = datetime.now(UTC)
        assert msg.timestamp.tzinfo is not None
        assert msg.timestamp.tzinfo == UTC or msg.timestamp.tzinfo == timezone.utc
        assert before <= msg.timestamp <= after

    def test_channel_message_all_fields(self) -> None:
        """ChannelMessage stores all fields when provided."""
        attachment = Attachment(
            filename="doc.pdf",
            content=b"pdf-bytes",
            content_type="application/pdf",
        )
        ts = datetime(2026, 1, 15, 12, 0, 0, tzinfo=UTC)
        msg = ChannelMessage(
            channel_type="slack",
            content="check this doc",
            sender_id="U99999",
            sender_name="Alice",
            customer_id="cust-001",
            conversation_id="T-123-456",
            channel_id="C-789",
            workspace_id="W-001",
            attachments=[attachment],
            reply_to_id="msg-previous",
            metadata={"thread_ts": "1234567890.123456"},
            timestamp=ts,
        )
        assert msg.sender_name == "Alice"
        assert msg.customer_id == "cust-001"
        assert msg.conversation_id == "T-123-456"
        assert msg.channel_id == "C-789"
        assert msg.workspace_id == "W-001"
        assert len(msg.attachments) == 1
        assert msg.attachments[0].filename == "doc.pdf"
        assert msg.reply_to_id == "msg-previous"
        assert msg.metadata == {"thread_ts": "1234567890.123456"}
        assert msg.timestamp == ts

    def test_channel_message_attachments_independent_per_instance(self) -> None:
        """ChannelMessage attachments list is independent per instance (no shared mutable default)."""
        msg_a = ChannelMessage(
            channel_type="slack",
            content="a",
            sender_id="user-a",
        )
        msg_b = ChannelMessage(
            channel_type="slack",
            content="b",
            sender_id="user-b",
        )
        msg_a.attachments.append(
            Attachment(
                filename="only-in-a.txt",
                content=b"data",
                content_type="text/plain",
            )
        )
        assert len(msg_a.attachments) == 1
        assert len(msg_b.attachments) == 0

    def test_channel_message_metadata_independent_per_instance(self) -> None:
        """ChannelMessage metadata dict is independent per instance (no shared mutable default)."""
        msg_a = ChannelMessage(
            channel_type="slack",
            content="a",
            sender_id="user-a",
        )
        msg_b = ChannelMessage(
            channel_type="slack",
            content="b",
            sender_id="user-b",
        )
        msg_a.metadata["key"] = "value"
        assert "key" in msg_a.metadata
        assert "key" not in msg_b.metadata


# -- ChannelResponse tests --


class TestChannelResponse:
    """tests for ChannelResponse dataclass."""

    def test_channel_response_is_dataclass(self) -> None:
        """ChannelResponse is a dataclass, not a Pydantic BaseModel."""
        assert dataclasses.is_dataclass(ChannelResponse)

    def test_channel_response_is_not_pydantic(self) -> None:
        """ChannelResponse does not inherit from pydantic BaseModel."""
        assert not hasattr(ChannelResponse, "model_fields")
        assert not hasattr(ChannelResponse, "model_validate")

    def test_channel_response_required_fields(self) -> None:
        """ChannelResponse requires content."""
        resp = ChannelResponse(content="here is the answer")
        assert resp.content == "here is the answer"

    def test_channel_response_defaults(self) -> None:
        """ChannelResponse optional fields default to None or empty."""
        resp = ChannelResponse(content="response text")
        assert resp.conversation_id is None
        assert resp.channel_id is None
        assert resp.attachments == []
        assert resp.format_hints == {}
        assert resp.metadata == {}

    def test_channel_response_all_fields(self) -> None:
        """ChannelResponse stores all fields when provided."""
        attachment = Attachment(
            filename="result.csv",
            content=b"col1,col2\na,b",
            content_type="text/csv",
            description="export results",
        )
        resp = ChannelResponse(
            content="here are your results",
            conversation_id="T-123-456",
            channel_id="C-789",
            attachments=[attachment],
            format_hints={"use_blocks": True},
            metadata={"response_type": "ephemeral"},
        )
        assert resp.conversation_id == "T-123-456"
        assert resp.channel_id == "C-789"
        assert len(resp.attachments) == 1
        assert resp.attachments[0].filename == "result.csv"
        assert resp.format_hints == {"use_blocks": True}
        assert resp.metadata == {"response_type": "ephemeral"}

    def test_channel_response_attachments_independent_per_instance(self) -> None:
        """ChannelResponse attachments list is independent per instance (no shared mutable default)."""
        resp_a = ChannelResponse(content="a")
        resp_b = ChannelResponse(content="b")
        resp_a.attachments.append(
            Attachment(
                filename="only-in-a.txt",
                content=b"data",
                content_type="text/plain",
            )
        )
        assert len(resp_a.attachments) == 1
        assert len(resp_b.attachments) == 0

    def test_channel_response_format_hints_independent_per_instance(self) -> None:
        """ChannelResponse format_hints dict is independent per instance."""
        resp_a = ChannelResponse(content="a")
        resp_b = ChannelResponse(content="b")
        resp_a.format_hints["bold"] = True
        assert "bold" in resp_a.format_hints
        assert "bold" not in resp_b.format_hints

    def test_channel_response_metadata_independent_per_instance(self) -> None:
        """ChannelResponse metadata dict is independent per instance."""
        resp_a = ChannelResponse(content="a")
        resp_b = ChannelResponse(content="b")
        resp_a.metadata["key"] = "value"
        assert "key" in resp_a.metadata
        assert "key" not in resp_b.metadata


# -- ChannelRouter tests --


class TestChannelRouter:
    """tests for ChannelRouter protocol."""

    def test_channel_router_is_runtime_checkable(self) -> None:
        """ChannelRouter is a runtime_checkable Protocol."""
        assert hasattr(ChannelRouter, "__protocol_attrs__") or hasattr(ChannelRouter, "__abstractmethods__")

    def test_conforming_class_satisfies_protocol(self) -> None:
        """class implementing route_inbound satisfies ChannelRouter isinstance check."""

        class _MockRouter:
            async def route_inbound(self, message: ChannelMessage) -> ChannelResponse | None:
                return None

        router = _MockRouter()
        assert isinstance(router, ChannelRouter)

    def test_non_conforming_class_fails_protocol(self) -> None:
        """class without route_inbound does not satisfy ChannelRouter isinstance check."""

        class _NotARouter:
            async def some_other_method(self) -> None:
                pass

        obj = _NotARouter()
        assert not isinstance(obj, ChannelRouter)

    @pytest.mark.asyncio
    async def test_conforming_router_returns_response(self) -> None:
        """conforming ChannelRouter implementation can return ChannelResponse."""

        class _EchoRouter:
            async def route_inbound(self, message: ChannelMessage) -> ChannelResponse | None:
                result = ChannelResponse(
                    content=f"echo: {message.content}",
                    conversation_id=message.conversation_id,
                    channel_id=message.channel_id,
                )
                return result

        router = _EchoRouter()
        assert isinstance(router, ChannelRouter)
        msg = ChannelMessage(
            channel_type="websocket",
            content="hello",
            sender_id="user-1",
            conversation_id="conv-1",
            channel_id="ch-1",
        )
        response = await router.route_inbound(msg)
        assert response is not None
        assert response.content == "echo: hello"
        assert response.conversation_id == "conv-1"
        assert response.channel_id == "ch-1"

    @pytest.mark.asyncio
    async def test_conforming_router_returns_none(self) -> None:
        """conforming ChannelRouter implementation can return None."""

        class _NullRouter:
            async def route_inbound(self, message: ChannelMessage) -> ChannelResponse | None:
                return None

        router = _NullRouter()
        msg = ChannelMessage(
            channel_type="slack",
            content="ignored",
            sender_id="user-1",
        )
        response = await router.route_inbound(msg)
        assert response is None
