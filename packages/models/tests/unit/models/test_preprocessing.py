"""tests for message preprocessing utilities (LangChain-native)."""

from __future__ import annotations

import base64

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from threetears.models.capabilities import ModelCapabilities
from threetears.models.enums import ModelStatus, ModelTier, ModelType
from threetears.models.preprocessing import (
    enforce_alternating_roles,
    format_vision_content,
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
