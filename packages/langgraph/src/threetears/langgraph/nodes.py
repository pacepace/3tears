"""shared node implementations for pre-built agent graphs.

provides reusable LangGraph nodes that integrate with 3tears
infrastructure: ToolContextManager for context injection,
tool dispatch with error handling, and conditional routing.

``agent_node`` and ``tool_node`` accept a sequence of hooks (see
:mod:`threetears.langgraph.hooks`) so downstream callers extend
them without forking. the hook sequence is read from
``config["configurable"]["_hooks"]`` as a two-key dict:
``{"agent": Sequence[AgentNodeHook], "tool": Sequence[ToolNodeHook]}``.
missing keys default to an empty sequence (pure primitive
behavior). the primitives themselves never enumerate known hook
types -- SDK-specific concerns (streaming, audit, identity) live
on the hook, not on the node.
"""

from __future__ import annotations

import asyncio
import difflib
import time
from collections.abc import Sequence
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import MessagesState

from threetears.langgraph.hooks import (
    AgentNodeHook,
    ToolNodeHook,
    compose_agent_node_hooks,
    compose_tool_node_hooks,
)
from threetears.observe import get_logger

__all__ = [
    "agent_node",
    "has_tool_calls",
    "tool_node",
]

log = get_logger(__name__)


def _resolve_agent_hooks(config: RunnableConfig) -> AgentNodeHook:
    """read the agent-hook sequence off ``configurable`` and compose it.

    the composed adapter is always returned even when no hooks are
    installed -- an empty sequence yields a no-op pass-through so
    the agent_node body can call the adapter unconditionally without
    paying a branch cost.

    :param config: runtime config dict
    :ptype config: RunnableConfig
    :return: single :class:`AgentNodeHook` fanning out to every
        installed hook in order
    :rtype: AgentNodeHook
    """
    configurable = config.get("configurable", {})
    raw = configurable.get("_hooks", {}) or {}
    hooks: Sequence[AgentNodeHook] = raw.get("agent", ()) if isinstance(raw, dict) else ()
    return compose_agent_node_hooks(hooks)


def _resolve_tool_hooks(config: RunnableConfig) -> ToolNodeHook:
    """read the tool-hook sequence off ``configurable`` and compose it.

    analogous to :func:`_resolve_agent_hooks` but for the tool-node
    hook protocol.

    :param config: runtime config dict
    :ptype config: RunnableConfig
    :return: single :class:`ToolNodeHook` fanning out to every
        installed hook in order
    :rtype: ToolNodeHook
    """
    configurable = config.get("configurable", {})
    raw = configurable.get("_hooks", {}) or {}
    hooks: Sequence[ToolNodeHook] = raw.get("tool", ()) if isinstance(raw, dict) else ()
    return compose_tool_node_hooks(hooks)


async def agent_node(state: MessagesState, config: RunnableConfig) -> dict[str, Any]:
    """invoke LLM with messages and optional tool binding.

    reads chat_model, system_prompt, and tools from config["configurable"].
    prepends system prompt if not already present as first message.
    binds tools to model when tools are available.
    if context_manager is in config, injects conversation context into
    the system prompt for access to previous tool results and variables.

    hooks installed under ``configurable["_hooks"]["agent"]`` fire
    before_invoke (after the system prompt is prepended and tools are
    bound, just before ``ainvoke``) and after_invoke (before wrapping
    the response into the state-update dict). hooks MUST NOT mutate
    the system prompt at index 0; see
    :class:`threetears.langgraph.hooks.AgentNodeHook`.

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

    # explicit list[BaseMessage] annotation required because list is
    # invariant and MessagesState types the underlying list with a
    # narrower BaseMessage union
    messages: list[BaseMessage] = list(state["messages"])

    # build context sections if context_manager available
    context_manager = configurable.get("context_manager")
    if context_manager is not None:
        ctx_prompt = context_manager.build_context_prompt()
        if ctx_prompt:
            system_prompt = system_prompt + "\n\n" + ctx_prompt

    # browser-supplied per-message locale info -- the channel adapter
    # populated ``ChannelMessage.user_timezone`` / ``user_locale`` from
    # its native source, the runtime stamped them on
    # ``configurable["_user_timezone"]`` / ``["_user_locale"]``. surface
    # them to the LLM as a non-persisted system-prompt suffix so the
    # model can render timestamps in user-local time and pass the tz
    # along to tools that accept one (e.g. ``current_date``) without
    # asking the user. injected per-turn so a user who travels mid-
    # conversation gets fresh values. NOT persisted into ``state.messages``
    # because the value can change between turns; persisting would
    # leave stale tz lines pinned in history.
    user_tz = configurable.get("_user_timezone")
    user_locale = configurable.get("_user_locale")
    locale_lines: list[str] = []
    if user_tz:
        locale_lines.append(f"User's local timezone: {user_tz}")
    if user_locale:
        locale_lines.append(f"User's locale: {user_locale}")
    if locale_lines:
        locale_block = "\n".join(locale_lines)
        if system_prompt:
            system_prompt = system_prompt + "\n\n" + locale_block
        else:
            system_prompt = locale_block

    if system_prompt and (not messages or not isinstance(messages[0], SystemMessage)):
        messages.insert(0, SystemMessage(content=system_prompt))

    hooks = _resolve_agent_hooks(config)
    state_view: dict[str, Any] = dict(state)
    messages, config = await hooks.before_invoke(messages, config, state_view)

    # re-read chat_model / tools from the post-hook configurable so hooks
    # (notably :class:`threetears.langgraph.hooks.PromptCachingHook`) can
    # swap in a pre-bound model and an empty tool list to memoize the
    # bind across turns. a hook that does not touch these keys leaves
    # the originals in place and the behavior is unchanged.
    post_configurable = config.get("configurable", {})
    chat_model = post_configurable.get("chat_model", chat_model)
    tools = post_configurable.get("tools", tools)

    model = chat_model.bind_tools(tools) if tools else chat_model
    response = await model.ainvoke(messages)
    response = await hooks.after_invoke(response, config, state_view)

    result = {"messages": [response]}
    return result


async def tool_node(state: MessagesState, config: RunnableConfig) -> dict[str, Any]:
    """execute tool calls from last AI message.

    reads tools from config["configurable"] and dispatches each tool call
    from the last message. returns ToolMessage results for each call.
    each tool receives the full :class:`RunnableConfig` via
    ``tool.ainvoke(args, config=config)`` so tools can read
    ``configurable`` entries (conversation_id, user_id, call_context,
    etc.) the handler stamps. tools that do not declare a ``config``
    parameter silently ignore it through LangChain's RunnableConfig
    threading.

    hooks installed under ``configurable["_hooks"]["tool"]`` fire
    ``before_dispatch`` once (with the full tool_call list), then
    per-call ``on_tool_start`` / ``on_tool_end``, and periodic
    ``on_heartbeat`` ticks while a slow tool is still running. the
    heartbeat interval is read from ``configurable["_hook_heartbeat_seconds"]``
    (default 10s); a value ``<= 0`` disables the heartbeat loop.
    heartbeat emission is only set up when at least one hook
    implements a non-default ``on_heartbeat``.

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
    heartbeat_interval = float(configurable.get("_hook_heartbeat_seconds", 10.0))

    last_message = state["messages"][-1]
    tool_messages: list[ToolMessage] = []

    if not isinstance(last_message, AIMessage):
        result = {"messages": tool_messages}
        return result

    hooks = _resolve_tool_hooks(config)
    state_view: dict[str, Any] = dict(state)
    # convert ToolCall TypedDicts to plain dicts at the protocol
    # boundary; list[ToolCall] is not assignable to list[dict[str, Any]]
    # because list is invariant, even though ToolCall is structurally
    # a dict[str, Any]
    tool_calls, config = await hooks.before_dispatch(
        [dict(tc) for tc in last_message.tool_calls],
        config,
        state_view,
    )

    for tool_call in tool_calls:
        await hooks.on_tool_start(tool_call, config, state_view)
        heartbeat_task: asyncio.Task[None] | None = None
        started_monotonic = time.monotonic()
        if heartbeat_interval > 0:
            heartbeat_task = asyncio.create_task(
                _heartbeat_loop(
                    hooks,
                    tool_call,
                    heartbeat_interval,
                    started_monotonic,
                    config,
                    state_view,
                ),
            )

        success = True
        tool = tool_map.get(tool_call["name"])
        try:
            if tool is not None:
                try:
                    tool_result = await tool.ainvoke(tool_call["args"], config=config)
                except Exception as exc:
                    tool_result = f"Tool error: {tool_call['name']}: {exc}"
                    success = False
            else:
                # tool name miss: surface the closest available tool names
                # so the LLM can self-correct on the next turn rather than
                # loop on the same wrong shape. names that share an underscore
                # / dot prefix the LLM picks up routinely (e.g. it emitted
                # ``datasource.x.read`` when the registered name is
                # ``datasource_x_read`` after sanitisation); difflib's
                # SequenceMatcher catches that family naturally.
                suggestions = difflib.get_close_matches(
                    tool_call["name"],
                    list(tool_map.keys()),
                    n=3,
                    cutoff=0.5,
                )
                base = f"tool '{tool_call['name']}' not found"
                if suggestions:
                    hint = ", ".join(f"'{s}'" for s in suggestions)
                    tool_result = f"{base}. did you mean: {hint}?"
                else:
                    tool_result = base
                success = False
        finally:
            if heartbeat_task is not None:
                heartbeat_task.cancel()
                try:
                    await heartbeat_task
                except asyncio.CancelledError:
                    pass

        elapsed_ms = round((time.monotonic() - started_monotonic) * 1000)
        await hooks.on_tool_end(
            tool_call,
            tool_result,
            success,
            elapsed_ms,
            config,
            state_view,
        )
        tool_messages.append(
            ToolMessage(content=str(tool_result), tool_call_id=tool_call["id"]),
        )

    result = {"messages": tool_messages}
    return result


async def _heartbeat_loop(
    hooks: ToolNodeHook,
    tool_call: dict[str, Any],
    interval: float,
    started_monotonic: float,
    config: RunnableConfig,
    state_view: dict[str, Any],
) -> None:
    """tick ``on_heartbeat`` every ``interval`` seconds until cancelled.

    the first tick fires one full interval after start, not
    immediately -- tools that complete faster than one interval emit
    zero heartbeats and the ``on_tool_start`` / ``on_tool_end`` pair
    is sufficient. cancelled cleanly by the caller via
    :meth:`asyncio.Task.cancel`.

    :param hooks: composed tool-node hook adapter
    :ptype hooks: ToolNodeHook
    :param tool_call: tool_call dict being tracked
    :ptype tool_call: dict[str, Any]
    :param interval: seconds between ticks
    :ptype interval: float
    :param started_monotonic: ``time.monotonic()`` at dispatch time
    :ptype started_monotonic: float
    :param config: runtime config
    :ptype config: RunnableConfig
    :param state_view: read-only state view
    :ptype state_view: dict[str, Any]
    :return: nothing (runs until cancelled)
    :rtype: None
    """
    while True:
        await asyncio.sleep(interval)
        elapsed = time.monotonic() - started_monotonic
        await hooks.on_heartbeat(tool_call, elapsed, config, state_view)


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
