"""tests for message preprocessing utilities (LangChain-native)."""

from __future__ import annotations

import base64
from uuid import UUID

import pytest
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from threetears.media.contracts import ObjectHandle
from threetears.models.capabilities import ModelCapabilities
from threetears.models.enums import ModelStatus, ModelTier, ModelType
from threetears.models.preprocessing import (
    OBJECT_REFERENCE_BLOCK_TYPE,
    ObjectReference,
    enforce_alternating_roles,
    format_image_block,
    format_object_reference_block,
    format_vision_content,
    format_vision_reference_content,
    is_object_reference_block,
    parse_object_reference_block,
    preprocess_messages,
)


def _make_capabilities(
    requires_alternating_roles: bool | None = None,
) -> ModelCapabilities:
    """build a ``ModelCapabilities`` with minimal required fields.

    :param requires_alternating_roles: capability flag toggling the merge
    :ptype requires_alternating_roles: bool | None
    :return: pydantic capability model for tests
    :rtype: ModelCapabilities
    """
    return ModelCapabilities(
        model_name="test-model",
        model_type=ModelType.CHAT,
        model_tier=ModelTier.SMALL,
        model_status=ModelStatus.ACTIVE,
        requires_alternating_roles=requires_alternating_roles,
    )


class TestEnforceAlternatingRoles:
    """tests for ``enforce_alternating_roles`` operating on ``BaseMessage``."""

    def test_empty_list_returns_empty(self) -> None:
        """empty input returns empty list."""
        result = enforce_alternating_roles([])
        assert result == []

    def test_single_user_message(self) -> None:
        """single user message passes through unchanged."""
        msgs: list[BaseMessage] = [HumanMessage(content="hello")]
        result = enforce_alternating_roles(msgs)
        assert len(result) == 1
        assert isinstance(result[0], HumanMessage)
        assert result[0].content == "hello"

    def test_single_assistant_appends_user(self) -> None:
        """single assistant message triggers a ``Continue.`` user message."""
        msgs: list[BaseMessage] = [AIMessage(content="hi there")]
        result = enforce_alternating_roles(msgs)
        assert len(result) == 2
        assert isinstance(result[0], AIMessage)
        assert isinstance(result[1], HumanMessage)
        assert result[1].content == "Continue."

    def test_consecutive_user_messages_merged(self) -> None:
        """consecutive user messages collapse into one with newline join."""
        msgs: list[BaseMessage] = [
            HumanMessage(content="first"),
            HumanMessage(content="second"),
            AIMessage(content="reply"),
            HumanMessage(content="third"),
        ]
        result = enforce_alternating_roles(msgs)
        assert len(result) == 3
        assert isinstance(result[0], HumanMessage)
        assert result[0].content == "first\nsecond"
        assert isinstance(result[1], AIMessage)
        assert result[1].content == "reply"

    def test_consecutive_assistant_messages_merged(self) -> None:
        """consecutive assistant messages collapse into one."""
        msgs: list[BaseMessage] = [
            HumanMessage(content="hi"),
            AIMessage(content="part1"),
            AIMessage(content="part2"),
            HumanMessage(content="thanks"),
        ]
        result = enforce_alternating_roles(msgs)
        assert len(result) == 3
        assert isinstance(result[1], AIMessage)
        assert result[1].content == "part1\npart2"

    def test_leading_system_messages_preserved(self) -> None:
        """leading ``SystemMessage`` instances are preserved unchanged."""
        msgs: list[BaseMessage] = [
            SystemMessage(content="be helpful"),
            SystemMessage(content="be brief"),
            HumanMessage(content="hi"),
        ]
        result = enforce_alternating_roles(msgs)
        assert len(result) == 3
        assert isinstance(result[0], SystemMessage)
        assert isinstance(result[1], SystemMessage)
        assert isinstance(result[2], HumanMessage)

    def test_tool_messages_pass_through(self) -> None:
        """``ToolMessage`` instances are preserved in position."""
        msgs: list[BaseMessage] = [
            HumanMessage(content="run this"),
            AIMessage(content=""),
            ToolMessage(content="result", tool_call_id="tc_1"),
            HumanMessage(content="ok"),
        ]
        result = enforce_alternating_roles(msgs)
        assert len(result) == 4
        assert isinstance(result[2], ToolMessage)
        assert result[2].tool_call_id == "tc_1"

    def test_non_string_content_not_merged(self) -> None:
        """messages with non-string content are not merged."""
        msgs: list[BaseMessage] = [
            HumanMessage(content=[{"type": "text", "text": "first"}]),
            HumanMessage(content="second"),
        ]
        result = enforce_alternating_roles(msgs)
        assert len(result) == 2


class TestPreprocessMessages:
    """tests for ``preprocess_messages`` capability dispatch."""

    def test_passthrough_when_no_alternating_required(self) -> None:
        """default capabilities leave messages untouched."""
        msgs: list[BaseMessage] = [
            HumanMessage(content="a"),
            HumanMessage(content="b"),
        ]
        result = preprocess_messages(msgs, _make_capabilities(requires_alternating_roles=False))
        assert len(result) == 2

    def test_alternating_roles_applied(self) -> None:
        """``requires_alternating_roles=True`` triggers the merge transform."""
        msgs: list[BaseMessage] = [
            HumanMessage(content="a"),
            HumanMessage(content="b"),
        ]
        result = preprocess_messages(msgs, _make_capabilities(requires_alternating_roles=True))
        assert len(result) == 1
        assert isinstance(result[0], HumanMessage)
        assert result[0].content == "a\nb"


class TestFormatVisionContent:
    """tests for ``format_vision_content`` helper."""

    def test_returns_image_url_and_text_blocks(self) -> None:
        """builds two-element multipart content with image_url and text."""
        result = format_vision_content(b"abc", "image/png", "describe")
        assert len(result) == 2
        assert result[0]["type"] == "image_url"
        assert result[1]["type"] == "text"

    def test_image_block_uses_data_uri(self) -> None:
        """the image_url block embeds a base64 data URI."""
        result = format_vision_content(b"abc", "image/png", "describe")
        block = result[0]
        assert isinstance(block, dict)
        url_block = block["image_url"]
        assert isinstance(url_block, dict)
        url = url_block["url"]
        assert url.startswith("data:image/png;base64,")
        assert base64.b64decode(url.split(",", 1)[1]) == b"abc"

    def test_image_block_alone_matches_first_block_of_vision_content(self) -> None:
        """the single-block builder is what the two-block helper composes."""
        assert format_image_block(b"abc", "image/png") == format_vision_content(b"abc", "image/png", "x")[0]


def _make_handle() -> ObjectHandle:
    """build an ``ObjectHandle`` with a fixed id for reference tests.

    :return: handle naming a catalogued PNG
    :rtype: ObjectHandle
    """
    return ObjectHandle(
        object_id=UUID("0192f3a0-0000-7000-8000-000000000001"),
        s3_key="cust/conv/image/2026/09/10/0192f3a0-0000-7000-8000-000000000001/a.png",
        mime_type="image/png",
        size_bytes=3,
    )


class TestFormatObjectReferenceBlock:
    """tests for the reference block that stands in for image bytes."""

    def test_block_carries_type_id_and_mime_only(self) -> None:
        """the block names the object and its mime type and nothing else."""
        block = format_object_reference_block(_make_handle())
        assert block == {
            "type": OBJECT_REFERENCE_BLOCK_TYPE,
            "object_id": "0192f3a0-0000-7000-8000-000000000001",
            "mime_type": "image/png",
        }

    def test_block_never_carries_the_key(self) -> None:
        """the storage key stays off the block so a caller cannot name one."""
        block = format_object_reference_block(_make_handle())
        assert "s3_key" not in block
        assert not any("cust/conv" in str(v) for v in block.values())

    def test_vision_reference_content_pairs_block_with_prompt(self) -> None:
        """reference content mirrors ``format_vision_content``: reference then text."""
        result = format_vision_reference_content(_make_handle(), "describe")
        assert len(result) == 2
        assert result[0]["type"] == OBJECT_REFERENCE_BLOCK_TYPE
        assert result[1] == {"type": "text", "text": "describe"}

    def test_is_object_reference_block_recognises_only_its_type(self) -> None:
        """the predicate matches the reference type and nothing else."""
        assert is_object_reference_block(format_object_reference_block(_make_handle()))
        assert not is_object_reference_block({"type": "text", "text": "x"})
        assert not is_object_reference_block({"type": "image_url", "image_url": {"url": "data:"}})
        assert not is_object_reference_block("plain string")

    def test_parse_round_trips_the_handle_fields(self) -> None:
        """parsing a formatted block yields the id as a UUID and the mime type."""
        ref = parse_object_reference_block(format_object_reference_block(_make_handle()))
        assert ref == ObjectReference(
            object_id=UUID("0192f3a0-0000-7000-8000-000000000001"),
            mime_type="image/png",
        )

    def test_parse_rejects_wrong_type(self) -> None:
        """a text block is not a reference, and parsing it says so."""
        with pytest.raises(ValueError, match="object_reference"):
            parse_object_reference_block({"type": "text", "text": "x"})

    def test_parse_rejects_missing_mime(self) -> None:
        """a reference without a mime type is malformed, not defaulted."""
        with pytest.raises(ValueError, match="mime_type"):
            parse_object_reference_block(
                {"type": OBJECT_REFERENCE_BLOCK_TYPE, "object_id": "0192f3a0-0000-7000-8000-000000000001"}
            )

    def test_parse_rejects_non_uuid_id(self) -> None:
        """an object id that is not a UUID is malformed, not passed through."""
        with pytest.raises(ValueError, match="object_id"):
            parse_object_reference_block(
                {"type": OBJECT_REFERENCE_BLOCK_TYPE, "object_id": "not-a-uuid", "mime_type": "image/png"}
            )
