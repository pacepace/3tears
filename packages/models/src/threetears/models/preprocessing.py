"""message preprocessing utilities for AI model provider communication.

Operates on LangChain ``BaseMessage`` instances directly so the
preprocessing pipeline composes cleanly with the LangChain-native provider
factories. The alternating-roles transform is the only widely-applicable
preprocessing step: providers like DeepSeek (and some other OpenAI-
compatible APIs) reject consecutive same-role messages.
"""

from __future__ import annotations

import base64

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from threetears.models.capabilities import ModelCapabilities

__all__ = [
    "enforce_alternating_roles",
    "format_vision_content",
    "preprocess_messages",
]


def _is_system(msg: BaseMessage) -> bool:
    """returns whether ``msg`` is a system message.

    :param msg: LangChain message to inspect
    :ptype msg: BaseMessage
    :return: True for ``SystemMessage`` instances
    :rtype: bool
    """
    return isinstance(msg, SystemMessage)


def _is_tool(msg: BaseMessage) -> bool:
    """returns whether ``msg`` is a tool result message.

    :param msg: LangChain message to inspect
    :ptype msg: BaseMessage
    :return: True for ``ToolMessage`` instances
    :rtype: bool
    """
    return isinstance(msg, ToolMessage)


def _role_of(msg: BaseMessage) -> str:
    """returns a canonical role string for ``msg``.

    used to detect consecutive same-role messages without relying on
    LangChain's ``type`` attribute (which has variants for chunk classes).

    :param msg: LangChain message to inspect
    :ptype msg: BaseMessage
    :return: ``"system" | "user" | "assistant" | "tool"``
    :rtype: str
    """
    if isinstance(msg, SystemMessage):
        return "system"
    if isinstance(msg, HumanMessage):
        return "user"
    if isinstance(msg, (AIMessage, AIMessageChunk)):
        return "assistant"
    if isinstance(msg, ToolMessage):
        return "tool"
    return "unknown"


def enforce_alternating_roles(
    messages: list[BaseMessage],
) -> list[BaseMessage]:
    """ensures user/assistant messages alternate, as required by some models.

    system messages are preserved at start. tool messages are preserved in
    position. consecutive same-role user or assistant messages are merged
    by joining string content with newline. if last message is not user
    role, appends continuation user message.

    :param messages: list of LangChain messages to process
    :ptype messages: list[BaseMessage]
    :return: new list with alternating roles enforced
    :rtype: list[BaseMessage]
    """
    if not messages:
        return []

    result: list[BaseMessage] = []

    # preserve leading system messages
    idx = 0
    while idx < len(messages) and _is_system(messages[idx]):
        result.append(messages[idx])
        idx += 1

    # process remaining messages, merging consecutive same-role messages
    while idx < len(messages):
        msg = messages[idx]

        # tool messages pass through untouched
        if _is_tool(msg):
            result.append(msg)
            idx += 1
            continue

        run: list[BaseMessage] = [msg]
        current_role = _role_of(msg)
        while (
            idx + 1 < len(messages)
            and _role_of(messages[idx + 1]) == current_role
            and current_role in ("user", "assistant")
        ):
            idx += 1
            run.append(messages[idx])

        if len(run) == 1:
            result.append(run[0])
        else:
            merged = _merge_message_run(run)
            if merged is not None:
                result.append(merged)
            else:
                result.extend(run)

        idx += 1

    # ensure last message is user role
    if result and _role_of(result[-1]) != "user":
        result.append(HumanMessage(content="Continue."))

    return result


def _merge_message_run(run: list[BaseMessage]) -> BaseMessage | None:
    """merges consecutive same-role messages into single message.

    only merges string content by joining with newline. if any message
    in run has non-string content, returns None to signal caller should
    keep messages separate. preserves tool_calls from last message in
    the run for assistant-role merges.

    :param run: consecutive messages with same role to merge
    :ptype run: list[BaseMessage]
    :return: merged message, or None if content types prevent merging
    :rtype: BaseMessage | None
    """
    if not all(isinstance(m.content, str) for m in run):
        return None

    merged_content = "\n".join(str(m.content) for m in run)
    role = _role_of(run[0])

    if role == "user":
        return HumanMessage(content=merged_content)
    if role == "assistant":
        last = run[-1]
        tool_calls = getattr(last, "tool_calls", None) or []
        return AIMessage(content=merged_content, tool_calls=list(tool_calls))
    return None


def format_vision_content(
    image_bytes: bytes,
    mime_type: str,
    prompt: str,
) -> list[dict[str, str | dict[str, str]]]:
    """constructs multipart content blocks for vision messages.

    encodes image bytes as base64 data URI and pairs with text prompt
    for use as ``HumanMessage`` content in vision-capable model requests.

    :param image_bytes: raw image bytes to encode
    :ptype image_bytes: bytes
    :param mime_type: MIME type of image (e.g. "image/png")
    :ptype mime_type: str
    :param prompt: text prompt to accompany image
    :ptype prompt: str
    :return: list of two content blocks (image_url and text)
    :rtype: list[dict[str, str | dict[str, str]]]
    """
    b64_string = base64.b64encode(image_bytes).decode("utf-8")

    return [
        {
            "type": "image_url",
            "image_url": {"url": f"data:{mime_type};base64,{b64_string}"},
        },
        {
            "type": "text",
            "text": prompt,
        },
    ]


def preprocess_messages(
    messages: list[BaseMessage],
    capabilities: ModelCapabilities,
) -> list[BaseMessage]:
    """applies preprocessing pipeline based on model capabilities.

    inspects capability flags and applies relevant transforms to the
    message list. currently supports alternating-role enforcement for
    models that require it.

    :param messages: list of LangChain messages to preprocess
    :ptype messages: list[BaseMessage]
    :param capabilities: model capabilities determining which transforms apply
    :ptype capabilities: ModelCapabilities
    :return: preprocessed message list
    :rtype: list[BaseMessage]
    """
    result: list[BaseMessage] = list(messages)

    if capabilities.requires_alternating_roles is True:
        result = enforce_alternating_roles(result)

    return result
