"""shared conversion helpers for LangChain message type boundaries."""

from __future__ import annotations

from typing import Any, cast

from threetears.models.messages import ChatMessage, MessageRole, ToolCallRequest, ToolDefinition
from threetears.models.results import ChatChunk, ChatResult


def messages_to_lc(messages: list[ChatMessage]) -> list[Any]:
    """converts threetears ChatMessage list to LangChain message list.

    imports LangChain message types lazily to avoid module-level
    dependency on langchain-core.

    :param messages: threetears chat messages to convert
    :ptype messages: list[ChatMessage]
    :return: list of LangChain message objects
    :rtype: list[Any]
    """
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

    # LangChain message content type: str | list[str | dict[Any, Any]]
    # threetears content type: str | list[dict[str, Any]]
    # cast is safe because list[dict[str, Any]] is a subtype at runtime
    _LcContent = str | list[str | dict[Any, Any]]

    result: list[Any] = []
    for msg in messages:
        lc_content = cast(_LcContent, msg.content)
        if msg.role == MessageRole.SYSTEM:
            result.append(SystemMessage(content=lc_content))
        elif msg.role == MessageRole.USER:
            result.append(HumanMessage(content=lc_content))
        elif msg.role == MessageRole.ASSISTANT:
            lc_tool_calls: list[dict[str, Any]] = []
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    lc_tool_calls.append({"name": tc.name, "args": tc.args, "id": tc.id})
            result.append(AIMessage(content=lc_content, tool_calls=lc_tool_calls))
        elif msg.role == MessageRole.TOOL:
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            result.append(ToolMessage(content=content, tool_call_id=msg.tool_call_id or ""))
    return result


def ai_message_to_result(response: Any) -> ChatResult:
    """converts LangChain AIMessage to threetears ChatResult.

    extracts content, tool calls, model identifier, and usage metadata
    from LangChain response object.

    :param response: LangChain AIMessage from ainvoke
    :ptype response: Any
    :return: converted chat completion result
    :rtype: ChatResult
    """
    content = response.content if isinstance(response.content, str) else str(response.content)

    tool_calls: list[ToolCallRequest] | None = None
    if response.tool_calls:
        tool_calls = [
            ToolCallRequest(id=tc["id"], name=tc["name"], args=tc["args"])
            for tc in response.tool_calls
        ]

    usage: dict[str, int] | None = None
    if response.usage_metadata:
        usage = {
            "input_tokens": response.usage_metadata.get("input_tokens", 0),
            "output_tokens": response.usage_metadata.get("output_tokens", 0),
        }

    model = response.response_metadata.get("model", "") if response.response_metadata else ""

    return ChatResult(
        content=content,
        tool_calls=tool_calls,
        model=model,
        usage=usage,
    )


def ai_chunk_to_chat_chunk(chunk: Any) -> ChatChunk:
    """converts LangChain AIMessageChunk to threetears ChatChunk.

    extracts partial content, tool calls, and finish reason from
    streaming chunk.

    :param chunk: LangChain AIMessageChunk from astream
    :ptype chunk: Any
    :return: converted chat completion chunk
    :rtype: ChatChunk
    """
    content = chunk.content if isinstance(chunk.content, str) else ""

    tool_calls: list[ToolCallRequest] | None = None
    if chunk.tool_calls:
        tool_calls = [
            ToolCallRequest(
                id=tc.get("id", ""),
                name=tc.get("name", ""),
                args=tc.get("args", {}),
            )
            for tc in chunk.tool_calls
        ]

    finish_reason: str | None = None
    if chunk.response_metadata:
        finish_reason = chunk.response_metadata.get("stop_reason")

    return ChatChunk(content=content, tool_calls=tool_calls, finish_reason=finish_reason)


def tool_def_to_lc(tool: ToolDefinition) -> dict[str, Any]:
    """converts threetears ToolDefinition to LangChain tool dict format.

    :param tool: threetears tool definition to convert
    :ptype tool: ToolDefinition
    :return: dict in LangChain bind_tools format with name, description, input_schema
    :rtype: dict[str, Any]
    """
    return {
        "name": tool.name,
        "description": tool.description,
        "input_schema": tool.parameters,
    }
