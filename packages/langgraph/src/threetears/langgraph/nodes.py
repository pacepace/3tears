"""shared node implementations for pre-built agent graphs.

provides reusable LangGraph nodes that integrate with 3tears
infrastructure: ToolContextManager for context injection,
tool dispatch with error handling, and conditional routing.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import MessagesState

import logging

__all__ = [
    "agent_node",
    "has_tool_calls",
    "tool_node",
]

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


async def agent_node(state: MessagesState, config: RunnableConfig) -> dict[str, Any]:
    """invoke LLM with messages and optional tool binding.

    reads chat_model, system_prompt, and tools from config["configurable"].
    prepends system prompt if not already present as first message.
    binds tools to model when tools are available.
    if context_manager is in config, injects conversation context into
    the system prompt for access to previous tool results and variables.

    :param state: current agent state containing messages
    :ptype state: MessagesState
    :param config: runnable config with configurable values
    :ptype config: RunnableConfig
    :return: dict with messages list containing model response
    :rtype: dict[str, Any]
    """
    configurable = config.get("configurable", {})
    chat_model = configurable["chat_model"]
    system_prompt = configurable.get("system_prompt", "")
    tools = configurable.get("tools", [])

    messages = list(state["messages"])

    # build context sections if context_manager available
    context_manager = configurable.get("context_manager")
    if context_manager is not None:
        ctx_prompt = context_manager.build_context_prompt()
        if ctx_prompt:
            system_prompt = system_prompt + "\n\n" + ctx_prompt

    if system_prompt and (not messages or not isinstance(messages[0], SystemMessage)):
        messages.insert(0, SystemMessage(content=system_prompt))

    model = chat_model.bind_tools(tools) if tools else chat_model

    response = await model.ainvoke(messages)
    result = {"messages": [response]}
    return result


async def tool_node(state: MessagesState, config: RunnableConfig) -> dict[str, Any]:
    """execute tool calls from last AI message.

    reads tools from config["configurable"] and dispatches each tool call
    from the last message. returns ToolMessage results for each call.

    :param state: current agent state containing messages
    :ptype state: MessagesState
    :param config: runnable config with configurable values
    :ptype config: RunnableConfig
    :return: dict with messages list containing tool results
    :rtype: dict[str, Any]
    """
    configurable = config.get("configurable", {})
    tools = configurable.get("tools", [])
    tool_map: dict[str, Any] = {t.name: t for t in tools}

    last_message = state["messages"][-1]
    tool_messages: list[ToolMessage] = []

    if not isinstance(last_message, AIMessage):
        result = {"messages": tool_messages}
        return result

    for tool_call in last_message.tool_calls:
        tool = tool_map.get(tool_call["name"])
        if tool is not None:
            tool_result = await tool.ainvoke(tool_call["args"])
        else:
            tool_result = f"tool '{tool_call['name']}' not found"
        tool_messages.append(ToolMessage(content=str(tool_result), tool_call_id=tool_call["id"]))

    result = {"messages": tool_messages}
    return result


def has_tool_calls(state: MessagesState) -> str:
    """check if last message has tool calls for conditional routing.

    returns "tools" if last message contains tool calls, "end" otherwise.

    :param state: current agent state containing messages
    :ptype state: MessagesState
    :return: routing key, either "tools" or "end"
    :rtype: str
    """
    last = state["messages"][-1]
    result = "end"
    if hasattr(last, "tool_calls") and last.tool_calls:
        result = "tools"
    return result
