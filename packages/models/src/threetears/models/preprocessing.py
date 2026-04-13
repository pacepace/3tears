"""message preprocessing utilities for AI model provider communication."""

from __future__ import annotations

import base64

from threetears.models.capabilities import ModelCapabilities
from threetears.models.messages import ChatMessage, MessageRole


def enforce_alternating_roles(
    messages: list[ChatMessage],
) -> list[ChatMessage]:
    """ensures user/assistant messages alternate, as required by some models.

    system messages are preserved at start. tool messages are preserved in
    position. consecutive same-role user or assistant messages are merged by
    joining string content with newline. if last message is not user role,
    appends continuation user message.

    :param messages: list of chat messages to process
    :ptype messages: list[ChatMessage]
    :return: new list with alternating roles enforced
    :rtype: list[ChatMessage]
    """
    if not messages:
        return []

    result: list[ChatMessage] = []

    # preserve leading system messages
    idx = 0
    while idx < len(messages) and messages[idx].role == MessageRole.SYSTEM:
        result.append(messages[idx])
        idx += 1

    # process remaining messages, merging consecutive same-role messages
    while idx < len(messages):
        msg = messages[idx]

        # tool messages pass through untouched
        if msg.role == MessageRole.TOOL:
            result.append(msg)
            idx += 1
            continue

        # collect consecutive messages with same role (user or assistant)
        run: list[ChatMessage] = [msg]
        while (
            idx + 1 < len(messages)
            and messages[idx + 1].role == msg.role
            and messages[idx + 1].role != MessageRole.TOOL
            and messages[idx + 1].role != MessageRole.SYSTEM
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
    if result and result[-1].role != MessageRole.USER:
        result.append(ChatMessage(role=MessageRole.USER, content="Continue."))

    return result


def _merge_message_run(run: list[ChatMessage]) -> ChatMessage | None:
    """merges consecutive same-role messages into single message.

    only merges string content by joining with newline. if any message
    in run has list content, returns None to signal caller should keep
    messages separate. preserves tool_calls from last message and name
    from first message.

    :param run: consecutive messages with same role to merge
    :ptype run: list[ChatMessage]
    :return: merged message, or None if content types prevent merging
    :rtype: ChatMessage | None
    """
    if not all(isinstance(m.content, str) for m in run):
        return None

    merged_content = "\n".join(str(m.content) for m in run)

    return ChatMessage(
        role=run[0].role,
        content=merged_content,
        tool_calls=run[-1].tool_calls,
        tool_call_id=None,
        name=run[0].name,
    )


def format_vision_content(
    image_bytes: bytes,
    mime_type: str,
    prompt: str,
) -> list[dict[str, str | dict[str, str]]]:
    """constructs multipart content blocks for vision messages.

    encodes image bytes as base64 data URI and pairs with text prompt
    for use as ChatMessage content in vision-capable model requests.

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
    messages: list[ChatMessage],
    capabilities: ModelCapabilities,
) -> list[ChatMessage]:
    """applies preprocessing pipeline based on model capabilities.

    inspects capability flags and applies relevant transforms to message
    list. currently supports alternating role enforcement for models that
    require it.

    :param messages: list of chat messages to preprocess
    :ptype messages: list[ChatMessage]
    :param capabilities: model capabilities determining which transforms apply
    :ptype capabilities: ModelCapabilities
    :return: preprocessed message list
    :rtype: list[ChatMessage]
    """
    result: list[ChatMessage] = list(messages)

    if capabilities.requires_alternating_roles is True:
        result = enforce_alternating_roles(result)

    return result
