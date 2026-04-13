"""streaming helpers for chunk merging and tool call recovery."""

from __future__ import annotations

import json
from typing import Any

from threetears.models.messages import ToolCallRequest
from threetears.models.results import ChatChunk, ChatResult


def merge_chunks(chunks: list[ChatChunk]) -> ChatResult:
    """accumulates streaming chunks into single chat result.

    concatenates content strings, merges tool_calls lists preserving order,
    and captures finish_reason from last chunk that provides one.

    :param chunks: list of streaming chat chunks to merge
    :ptype chunks: list[ChatChunk]
    :return: merged chat result with accumulated content and tool calls
    :rtype: ChatResult
    """
    if not chunks:
        return ChatResult(content="")

    content_parts: list[str] = []
    all_tool_calls: list[ToolCallRequest] = []

    for chunk in chunks:
        content_parts.append(chunk.content)

        if chunk.tool_calls is not None:
            all_tool_calls.extend(chunk.tool_calls)

    return ChatResult(
        content="".join(content_parts),
        tool_calls=all_tool_calls if all_tool_calls else None,
        model="",
        usage=None,
    )


def recover_split_tool_calls(
    tool_calls: list[ToolCallRequest],
) -> list[ToolCallRequest]:
    """detects and repairs LangChain streaming corruption in tool calls.

    LangChain streaming can split single tool call across two entries:
    entry 1 has name but empty args, entry 2 has empty name but populated
    args. when this pattern is detected in consecutive entries, merges them
    using entry 1 id and name with entry 2 args.

    :param tool_calls: list of tool call requests to scan for splits
    :ptype tool_calls: list[ToolCallRequest]
    :return: new list with split tool calls merged
    :rtype: list[ToolCallRequest]
    """
    if len(tool_calls) <= 1:
        return list(tool_calls)

    result: list[ToolCallRequest] = []
    idx = 0

    while idx < len(tool_calls):
        current = tool_calls[idx]

        # check if current + next match split pattern
        if idx + 1 < len(tool_calls):
            next_call = tool_calls[idx + 1]
            is_split = (
                current.name != ""
                and current.args == {}
                and next_call.name == ""
                and next_call.args != {}
            )

            if is_split:
                result.append(
                    ToolCallRequest(
                        id=current.id,
                        name=current.name,
                        args=next_call.args,
                    )
                )
                idx += 2
                continue

        result.append(current)
        idx += 1

    return result


def recover_invalid_tool_calls(
    invalid_calls: list[dict[str, str]],
) -> tuple[list[ToolCallRequest], list[dict[str, str]]]:
    """attempts to recover tool calls from LangChain invalid_tool_calls list.

    each dict has string fields "id", "name", "args" where args is
    stringified JSON. attempts to parse args JSON and create proper
    ToolCallRequest for each. calls with unparseable args or non-dict
    parsed results remain in invalid list.

    :param invalid_calls: list of invalid tool call dicts to attempt recovery
    :ptype invalid_calls: list[dict[str, str]]
    :return: tuple of (recovered ToolCallRequests, still-invalid dicts)
    :rtype: tuple[list[ToolCallRequest], list[dict[str, str]]]
    """
    recovered: list[ToolCallRequest] = []
    still_invalid: list[dict[str, str]] = []

    for call in invalid_calls:
        call_id = call.get("id", "")
        call_name = call.get("name", "")
        call_args = call.get("args", "")

        try:
            parsed: Any = json.loads(call_args)
        except (ValueError, json.JSONDecodeError, TypeError):
            still_invalid.append(call)
            continue

        if not isinstance(parsed, dict):
            still_invalid.append(call)
            continue

        recovered.append(
            ToolCallRequest(
                id=call_id,
                name=call_name,
                args=parsed,
            )
        )

    return recovered, still_invalid
