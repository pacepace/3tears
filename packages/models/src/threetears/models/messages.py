"""chat message types for AI model provider communication."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

__all__ = [
    "ChatMessage",
    "MessageRole",
    "ToolCallRequest",
    "ToolDefinition",
]


class MessageRole(StrEnum):
    """role of participant in chat conversation.

    :cvar SYSTEM: system-level instruction message
    :cvar USER: user-provided input message
    :cvar ASSISTANT: model-generated response message
    :cvar TOOL: tool execution result message
    """

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class ToolCallRequest:
    """request from model to invoke tool.

    :param id: unique identifier for tool call
    :ptype id: str
    :param name: name of tool to invoke
    :ptype name: str
    :param args: arguments to pass to tool
    :ptype args: dict[str, Any]
    """

    id: str
    name: str
    args: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolDefinition:
    """definition of tool available to model.

    :param name: unique tool name
    :ptype name: str
    :param description: human-readable description of tool purpose
    :ptype description: str
    :param parameters: JSON Schema describing tool parameters
    :ptype parameters: dict[str, Any]
    """

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChatMessage:
    """single message in chat conversation.

    :param role: role of message sender
    :ptype role: MessageRole
    :param content: message content as text string or multipart list for vision
    :ptype content: str | list[dict[str, Any]]
    :param tool_calls: tool invocation requests from model
    :ptype tool_calls: list[ToolCallRequest] | None
    :param tool_call_id: identifier linking tool result to original call
    :ptype tool_call_id: str | None
    :param name: optional sender name for disambiguation
    :ptype name: str | None
    """

    role: MessageRole
    content: str | list[dict[str, Any]]
    tool_calls: list[ToolCallRequest] | None = None
    tool_call_id: str | None = None
    name: str | None = None
