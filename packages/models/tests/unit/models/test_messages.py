"""tests for ChatMessage, MessageRole, ToolCallRequest, and ToolDefinition."""

from __future__ import annotations

import dataclasses
from enum import StrEnum

from threetears.models.messages import (
    ChatMessage,
    MessageRole,
    ToolCallRequest,
    ToolDefinition,
)


# -- MessageRole tests --


class TestMessageRole:
    """tests for MessageRole enum."""

    def test_message_role_is_str_enum(self) -> None:
        """MessageRole inherits from StrEnum."""
        assert issubclass(MessageRole, StrEnum)

    def test_message_role_values(self) -> None:
        """MessageRole contains all expected members with correct values."""
        assert MessageRole.SYSTEM == "system"
        assert MessageRole.USER == "user"
        assert MessageRole.ASSISTANT == "assistant"
        assert MessageRole.TOOL == "tool"

    def test_message_role_member_count(self) -> None:
        """MessageRole has exactly four members."""
        assert len(MessageRole) == 4

    def test_message_role_is_string_instance(self) -> None:
        """MessageRole members are string instances."""
        assert isinstance(MessageRole.SYSTEM, str)
        assert isinstance(MessageRole.TOOL, str)


# -- ToolCallRequest tests --


class TestToolCallRequest:
    """tests for ToolCallRequest dataclass."""

    def test_tool_call_request_is_dataclass(self) -> None:
        """ToolCallRequest is a dataclass, not a Pydantic BaseModel."""
        assert dataclasses.is_dataclass(ToolCallRequest)

    def test_tool_call_request_is_not_pydantic(self) -> None:
        """ToolCallRequest does not inherit from pydantic BaseModel."""
        assert not hasattr(ToolCallRequest, "model_fields")
        assert not hasattr(ToolCallRequest, "model_validate")

    def test_tool_call_request_required_fields(self) -> None:
        """ToolCallRequest requires id and name."""
        req = ToolCallRequest(id="call-1", name="search")
        assert req.id == "call-1"
        assert req.name == "search"

    def test_tool_call_request_args_default_empty(self) -> None:
        """ToolCallRequest args defaults to empty dict."""
        req = ToolCallRequest(id="call-1", name="search")
        assert req.args == {}

    def test_tool_call_request_all_fields(self) -> None:
        """ToolCallRequest stores all fields correctly."""
        args = {"query": "test", "limit": 10}
        req = ToolCallRequest(id="call-2", name="search", args=args)
        assert req.id == "call-2"
        assert req.name == "search"
        assert req.args == {"query": "test", "limit": 10}

    def test_tool_call_request_args_independent_per_instance(self) -> None:
        """ToolCallRequest args dict is independent per instance."""
        req_a = ToolCallRequest(id="a", name="tool-a")
        req_b = ToolCallRequest(id="b", name="tool-b")
        req_a.args["key"] = "value"
        assert "key" in req_a.args
        assert "key" not in req_b.args


# -- ToolDefinition tests --


class TestToolDefinition:
    """tests for ToolDefinition dataclass."""

    def test_tool_definition_is_dataclass(self) -> None:
        """ToolDefinition is a dataclass, not a Pydantic BaseModel."""
        assert dataclasses.is_dataclass(ToolDefinition)

    def test_tool_definition_is_not_pydantic(self) -> None:
        """ToolDefinition does not inherit from pydantic BaseModel."""
        assert not hasattr(ToolDefinition, "model_fields")
        assert not hasattr(ToolDefinition, "model_validate")

    def test_tool_definition_required_fields(self) -> None:
        """ToolDefinition requires name and description."""
        defn = ToolDefinition(name="search", description="search knowledge base")
        assert defn.name == "search"
        assert defn.description == "search knowledge base"

    def test_tool_definition_parameters_default_empty(self) -> None:
        """ToolDefinition parameters defaults to empty dict."""
        defn = ToolDefinition(name="search", description="search knowledge base")
        assert defn.parameters == {}

    def test_tool_definition_all_fields(self) -> None:
        """ToolDefinition stores all fields correctly."""
        params = {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        }
        defn = ToolDefinition(
            name="search",
            description="search knowledge base",
            parameters=params,
        )
        assert defn.name == "search"
        assert defn.description == "search knowledge base"
        assert defn.parameters["type"] == "object"
        assert "query" in defn.parameters["properties"]

    def test_tool_definition_parameters_independent_per_instance(self) -> None:
        """ToolDefinition parameters dict is independent per instance."""
        defn_a = ToolDefinition(name="a", description="desc a")
        defn_b = ToolDefinition(name="b", description="desc b")
        defn_a.parameters["key"] = "value"
        assert "key" in defn_a.parameters
        assert "key" not in defn_b.parameters


# -- ChatMessage tests --


class TestChatMessage:
    """tests for ChatMessage dataclass."""

    def test_chat_message_is_dataclass(self) -> None:
        """ChatMessage is a dataclass, not a Pydantic BaseModel."""
        assert dataclasses.is_dataclass(ChatMessage)

    def test_chat_message_is_not_pydantic(self) -> None:
        """ChatMessage does not inherit from pydantic BaseModel."""
        assert not hasattr(ChatMessage, "model_fields")
        assert not hasattr(ChatMessage, "model_validate")

    def test_chat_message_required_fields(self) -> None:
        """ChatMessage requires role and content."""
        msg = ChatMessage(role=MessageRole.USER, content="hello")
        assert msg.role == MessageRole.USER
        assert msg.content == "hello"

    def test_chat_message_defaults(self) -> None:
        """ChatMessage optional fields default to None."""
        msg = ChatMessage(role=MessageRole.SYSTEM, content="you are helpful")
        assert msg.tool_calls is None
        assert msg.tool_call_id is None
        assert msg.name is None

    def test_chat_message_all_fields(self) -> None:
        """ChatMessage stores all fields correctly."""
        tool_call = ToolCallRequest(
            id="call-1", name="search", args={"q": "test"}
        )
        msg = ChatMessage(
            role=MessageRole.ASSISTANT,
            content="calling tool",
            tool_calls=[tool_call],
            tool_call_id="call-1",
            name="assistant-v2",
        )
        assert msg.role == MessageRole.ASSISTANT
        assert msg.content == "calling tool"
        assert msg.tool_calls is not None
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].name == "search"
        assert msg.tool_call_id == "call-1"
        assert msg.name == "assistant-v2"

    def test_chat_message_multipart_content(self) -> None:
        """ChatMessage content can be list of dicts for multipart/vision."""
        parts = [
            {"type": "text", "text": "describe this image"},
            {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
        ]
        msg = ChatMessage(role=MessageRole.USER, content=parts)
        assert isinstance(msg.content, list)
        assert len(msg.content) == 2
        assert msg.content[0]["type"] == "text"
        assert msg.content[1]["type"] == "image_url"

    def test_chat_message_tool_result(self) -> None:
        """ChatMessage can represent tool result with tool role and tool_call_id."""
        msg = ChatMessage(
            role=MessageRole.TOOL,
            content='{"result": "42"}',
            tool_call_id="call-99",
        )
        assert msg.role == MessageRole.TOOL
        assert msg.tool_call_id == "call-99"
