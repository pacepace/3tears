"""tests for message preprocessing utilities."""

from __future__ import annotations

import base64

from threetears.models.capabilities import ModelCapabilities
from threetears.models.enums import ModelStatus, ModelTier, ModelType
from threetears.models.messages import ChatMessage, MessageRole, ToolCallRequest
from threetears.models.preprocessing import (
    enforce_alternating_roles,
    format_vision_content,
    preprocess_messages,
)


def _make_capabilities(
    requires_alternating_roles: bool | None = None,
) -> ModelCapabilities:
    """helper to create ModelCapabilities with minimal required fields."""
    return ModelCapabilities(
        model_name="test-model",
        model_type=ModelType.CHAT,
        model_tier=ModelTier.SMALL,
        model_status=ModelStatus.ACTIVE,
        requires_alternating_roles=requires_alternating_roles,
    )


class TestEnforceAlternatingRoles:
    """tests for enforce_alternating_roles function."""

    def test_empty_list_returns_empty(self) -> None:
        """empty input returns empty list."""
        result = enforce_alternating_roles([])
        assert result == []

    def test_single_user_message(self) -> None:
        """single user message is preserved as-is."""
        msgs = [ChatMessage(role=MessageRole.USER, content="hello")]
        result = enforce_alternating_roles(msgs)
        assert len(result) == 1
        assert result[0].role == MessageRole.USER
        assert result[0].content == "hello"

    def test_single_assistant_appends_user(self) -> None:
        """single assistant message gets Continue. user message appended."""
        msgs = [ChatMessage(role=MessageRole.ASSISTANT, content="hi there")]
        result = enforce_alternating_roles(msgs)
        assert len(result) == 2
        assert result[0].role == MessageRole.ASSISTANT
        assert result[0].content == "hi there"
        assert result[1].role == MessageRole.USER
        assert result[1].content == "Continue."

    def test_single_system_message(self) -> None:
        """single system message preserved with Continue. appended."""
        msgs = [ChatMessage(role=MessageRole.SYSTEM, content="you are helpful")]
        result = enforce_alternating_roles(msgs)
        assert len(result) == 2
        assert result[0].role == MessageRole.SYSTEM
        assert result[0].content == "you are helpful"
        assert result[1].role == MessageRole.USER
        assert result[1].content == "Continue."

    def test_consecutive_user_messages_merged(self) -> None:
        """two consecutive user messages merge into one with joined content."""
        msgs = [
            ChatMessage(role=MessageRole.USER, content="hello"),
            ChatMessage(role=MessageRole.USER, content="world"),
        ]
        result = enforce_alternating_roles(msgs)
        assert len(result) == 1
        assert result[0].role == MessageRole.USER
        assert result[0].content == "hello\nworld"

    def test_consecutive_assistant_messages_merged(self) -> None:
        """two consecutive assistant messages merge into one with joined content."""
        msgs = [
            ChatMessage(role=MessageRole.USER, content="question"),
            ChatMessage(role=MessageRole.ASSISTANT, content="first part"),
            ChatMessage(role=MessageRole.ASSISTANT, content="second part"),
        ]
        result = enforce_alternating_roles(msgs)
        # USER, merged ASSISTANT, then Continue. appended
        assert len(result) == 3
        assert result[0].role == MessageRole.USER
        assert result[1].role == MessageRole.ASSISTANT
        assert result[1].content == "first part\nsecond part"
        assert result[2].role == MessageRole.USER
        assert result[2].content == "Continue."

    def test_three_consecutive_same_role_merged(self) -> None:
        """three consecutive user messages merge into one."""
        msgs = [
            ChatMessage(role=MessageRole.USER, content="a"),
            ChatMessage(role=MessageRole.USER, content="b"),
            ChatMessage(role=MessageRole.USER, content="c"),
        ]
        result = enforce_alternating_roles(msgs)
        assert len(result) == 1
        assert result[0].role == MessageRole.USER
        assert result[0].content == "a\nb\nc"

    def test_merged_message_preserves_tool_calls_from_last(self) -> None:
        """merged message preserves tool_calls from last message in run."""
        tc = ToolCallRequest(id="tc-1", name="search", args={"q": "test"})
        msgs = [
            ChatMessage(role=MessageRole.USER, content="question"),
            ChatMessage(role=MessageRole.ASSISTANT, content="thinking"),
            ChatMessage(
                role=MessageRole.ASSISTANT,
                content="decided",
                tool_calls=[tc],
            ),
        ]
        result = enforce_alternating_roles(msgs)
        merged = result[1]
        assert merged.role == MessageRole.ASSISTANT
        assert merged.tool_calls is not None
        assert len(merged.tool_calls) == 1
        assert merged.tool_calls[0].name == "search"

    def test_consecutive_with_list_content_not_merged(self) -> None:
        """consecutive same-role messages with list content are kept separate."""
        msgs = [
            ChatMessage(role=MessageRole.USER, content="describe this"),
            ChatMessage(
                role=MessageRole.USER,
                content=[
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}},
                ],
            ),
        ]
        result = enforce_alternating_roles(msgs)
        assert len(result) == 2
        assert result[0].content == "describe this"
        assert isinstance(result[1].content, list)

    def test_alternating_roles_unchanged(self) -> None:
        """already alternating user/assistant messages pass through unchanged."""
        msgs = [
            ChatMessage(role=MessageRole.USER, content="hello"),
            ChatMessage(role=MessageRole.ASSISTANT, content="hi"),
            ChatMessage(role=MessageRole.USER, content="how are you"),
        ]
        result = enforce_alternating_roles(msgs)
        assert len(result) == 3
        assert result[0].role == MessageRole.USER
        assert result[1].role == MessageRole.ASSISTANT
        assert result[2].role == MessageRole.USER

    def test_system_messages_at_start_preserved(self) -> None:
        """consecutive system messages at start are preserved."""
        msgs = [
            ChatMessage(role=MessageRole.SYSTEM, content="system 1"),
            ChatMessage(role=MessageRole.SYSTEM, content="system 2"),
            ChatMessage(role=MessageRole.USER, content="hello"),
            ChatMessage(role=MessageRole.ASSISTANT, content="hi"),
        ]
        result = enforce_alternating_roles(msgs)
        # last message is ASSISTANT, so Continue. appended
        assert len(result) == 5
        assert result[0].role == MessageRole.SYSTEM
        assert result[0].content == "system 1"
        assert result[1].role == MessageRole.SYSTEM
        assert result[1].content == "system 2"
        assert result[2].role == MessageRole.USER
        assert result[3].role == MessageRole.ASSISTANT
        assert result[4].role == MessageRole.USER
        assert result[4].content == "Continue."

    def test_tool_messages_preserved(self) -> None:
        """tool messages in assistant-tool-user sequence are preserved."""
        tc = ToolCallRequest(id="tc-1", name="search")
        msgs = [
            ChatMessage(
                role=MessageRole.ASSISTANT,
                content="let me search",
                tool_calls=[tc],
            ),
            ChatMessage(
                role=MessageRole.TOOL,
                content='{"result": "found"}',
                tool_call_id="tc-1",
            ),
            ChatMessage(role=MessageRole.USER, content="thanks"),
        ]
        result = enforce_alternating_roles(msgs)
        assert len(result) == 3
        assert result[0].role == MessageRole.ASSISTANT
        assert result[1].role == MessageRole.TOOL
        assert result[1].tool_call_id == "tc-1"
        assert result[2].role == MessageRole.USER

    def test_ends_with_assistant_appends_user(self) -> None:
        """ending with assistant message appends Continue. user message."""
        msgs = [
            ChatMessage(role=MessageRole.USER, content="hello"),
            ChatMessage(role=MessageRole.ASSISTANT, content="hi there"),
        ]
        result = enforce_alternating_roles(msgs)
        assert len(result) == 3
        assert result[-1].role == MessageRole.USER
        assert result[-1].content == "Continue."

    def test_ends_with_user_no_append(self) -> None:
        """ending with user message does not append extra message."""
        msgs = [
            ChatMessage(role=MessageRole.USER, content="hello"),
            ChatMessage(role=MessageRole.ASSISTANT, content="hi"),
            ChatMessage(role=MessageRole.USER, content="bye"),
        ]
        result = enforce_alternating_roles(msgs)
        assert len(result) == 3
        assert result[-1].role == MessageRole.USER
        assert result[-1].content == "bye"

    def test_input_not_mutated(self) -> None:
        """original input list is not mutated by processing."""
        msgs = [
            ChatMessage(role=MessageRole.USER, content="a"),
            ChatMessage(role=MessageRole.USER, content="b"),
        ]
        original_len = len(msgs)
        original_contents = [m.content for m in msgs]
        enforce_alternating_roles(msgs)
        assert len(msgs) == original_len
        assert [m.content for m in msgs] == original_contents


class TestFormatVisionContent:
    """tests for format_vision_content function."""

    def test_basic_image_formatting(self) -> None:
        """returns correct structure with base64 encoded data."""
        image_data = b"\x89PNG\r\n\x1a\nfake"
        result = format_vision_content(image_data, "image/png", "describe this")
        assert result[0]["type"] == "image_url"
        assert result[1]["type"] == "text"

    def test_mime_type_included(self) -> None:
        """mime type appears in data URI."""
        image_data = b"fake-jpeg"
        result = format_vision_content(image_data, "image/jpeg", "prompt")
        image_url = result[0]["image_url"]
        assert isinstance(image_url, dict)
        assert image_url["url"].startswith("data:image/jpeg;base64,")

    def test_prompt_included(self) -> None:
        """prompt text appears in second content block."""
        result = format_vision_content(b"img", "image/png", "what is this?")
        assert result[1]["text"] == "what is this?"

    def test_returns_list_of_two_blocks(self) -> None:
        """returns exactly two content blocks."""
        result = format_vision_content(b"img", "image/png", "prompt")
        assert len(result) == 2

    def test_base64_encoding_correct(self) -> None:
        """base64 encoded data decodes back to original bytes."""
        original = b"\x00\x01\x02\xff\xfe\xfd"
        result = format_vision_content(original, "image/png", "test")
        image_url = result[0]["image_url"]
        assert isinstance(image_url, dict)
        data_uri = image_url["url"]
        b64_part = data_uri.split(",", 1)[1]
        decoded = base64.b64decode(b64_part)
        assert decoded == original


class TestPreprocessMessages:
    """tests for preprocess_messages pipeline function."""

    def test_no_transforms_when_no_flags(self) -> None:
        """capabilities with requires_alternating_roles=None leaves messages unchanged."""
        caps = _make_capabilities(requires_alternating_roles=None)
        msgs = [
            ChatMessage(role=MessageRole.USER, content="a"),
            ChatMessage(role=MessageRole.USER, content="b"),
        ]
        result = preprocess_messages(msgs, caps)
        assert len(result) == 2
        assert result[0].content == "a"
        assert result[1].content == "b"

    def test_alternating_roles_applied(self) -> None:
        """capabilities with requires_alternating_roles=True applies enforcement."""
        caps = _make_capabilities(requires_alternating_roles=True)
        msgs = [
            ChatMessage(role=MessageRole.USER, content="a"),
            ChatMessage(role=MessageRole.USER, content="b"),
        ]
        result = preprocess_messages(msgs, caps)
        # consecutive users should be merged
        assert len(result) == 1
        assert result[0].content == "a\nb"

    def test_alternating_roles_not_applied_when_false(self) -> None:
        """capabilities with requires_alternating_roles=False does not apply enforcement."""
        caps = _make_capabilities(requires_alternating_roles=False)
        msgs = [
            ChatMessage(role=MessageRole.USER, content="a"),
            ChatMessage(role=MessageRole.USER, content="b"),
        ]
        result = preprocess_messages(msgs, caps)
        assert len(result) == 2
        assert result[0].content == "a"
        assert result[1].content == "b"
