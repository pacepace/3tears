"""hook protocols for agent_node and tool_node customization.

the canonical :func:`threetears.langgraph.nodes.agent_node` and
:func:`threetears.langgraph.nodes.tool_node` are deliberately small.
downstream callers (the aibots-agents SDK, audience_builder, any
future agent runtime) extend them via hook objects rather than by
forking the node body. a hook is a narrow protocol -- pure
before/after transforms for agent_node, and pure start/end/heartbeat
emitters for tool_node. hooks do NOT own graph state; they reach
into ``configurable`` or receive explicit state objects.

::

    class MyHook:
        async def before_invoke(self, messages, config, state):
            return rewritten_messages, config

        async def after_invoke(self, response, config, state):
            return response

    agent_node(state, config, hooks=[MyHook()])

composition: :func:`compose_agent_node_hooks` and
:func:`compose_tool_node_hooks` return adapters that fan a node's
single hook-call site across a sequence of hooks in order. the
primitives accept a ``Sequence`` of hooks directly; the compose
helpers exist for callers that want to assemble a pipeline once and
pass it through multiple graph builds.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig

__all__ = [
    "AgentNodeHook",
    "ToolNodeHook",
    "compose_agent_node_hooks",
    "compose_tool_node_hooks",
]


@runtime_checkable
class AgentNodeHook(Protocol):
    """protocol for before/after hooks on :func:`agent_node`.

    hooks are pure transformations: :meth:`before_invoke` rewrites the
    message list and/or the config just before the model is invoked;
    :meth:`after_invoke` inspects (and optionally rewrites) the raw
    model response before it is wrapped into a state-update dict.
    neither method is required -- default implementations return their
    inputs unchanged, so concrete hooks override only what they need.

    contract:

    - :meth:`before_invoke` MUST NOT mutate ``messages[0]`` when it is a
      :class:`langchain_core.messages.SystemMessage`. the system
      prefix is the stable cache-key head and downstream
      prompt-caching integrations (langgraph-task-01) rely on it.
    - hooks MUST NOT hold graph state across invocations in closure
      variables. persistent state belongs on ``configurable`` or an
      explicit state object passed through the graph.
    """

    async def before_invoke(
        self,
        messages: list[BaseMessage],
        config: RunnableConfig,
        state: dict[str, Any],
    ) -> tuple[list[BaseMessage], RunnableConfig]:
        """rewrite messages and/or config before :meth:`chat_model.ainvoke`.

        default implementation returns inputs unchanged.

        :param messages: full message list assembled by agent_node,
            including system prompt at index 0 when present
        :ptype messages: list[BaseMessage]
        :param config: LangGraph runtime config dict
        :ptype config: RunnableConfig
        :param state: current agent state (read-only view for hooks)
        :ptype state: dict[str, Any]
        :return: ``(messages, config)`` tuple for the next hook or the
            model invocation
        :rtype: tuple[list[BaseMessage], RunnableConfig]
        """
        ...

    async def after_invoke(
        self,
        response: Any,
        config: RunnableConfig,
        state: dict[str, Any],
    ) -> Any:
        """inspect or rewrite the raw model response after ``ainvoke``.

        default implementation returns the response unchanged.

        :param response: raw return value from ``chat_model.ainvoke``
            (typically an :class:`AIMessage` or compatible chunk)
        :ptype response: Any
        :param config: LangGraph runtime config dict
        :ptype config: RunnableConfig
        :param state: current agent state (read-only view for hooks)
        :ptype state: dict[str, Any]
        :return: response object (possibly rewritten) for the next hook
            or for wrapping into the state-update dict
        :rtype: Any
        """
        ...


@runtime_checkable
class ToolNodeHook(Protocol):
    """protocol for dispatch-level hooks on :func:`tool_node`.

    :meth:`before_dispatch` rewrites or filters the tool_call list
    immediately before dispatch. :meth:`on_tool_start` and
    :meth:`on_tool_end` are pure emitters -- they observe the call
    without returning a value. :meth:`on_heartbeat` observes periodic
    liveness ticks during long tool calls; implementations that do
    not care about heartbeats leave the method as a no-op. as with
    :class:`AgentNodeHook`, hooks MUST NOT hold graph state across
    invocations in closure variables.
    """

    async def before_dispatch(
        self,
        tool_calls: list[dict[str, Any]],
        config: RunnableConfig,
        state: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], RunnableConfig]:
        """rewrite tool_calls and/or config before dispatch.

        default implementation returns inputs unchanged. rewriting
        lets hooks add synthesized tool calls, filter disallowed
        ones, or amend arguments.

        :param tool_calls: list of tool_call dicts from the last
            :class:`AIMessage`; each entry has ``id``, ``name``,
            ``args`` keys
        :ptype tool_calls: list[dict[str, Any]]
        :param config: LangGraph runtime config dict
        :ptype config: RunnableConfig
        :param state: current agent state (read-only view for hooks)
        :ptype state: dict[str, Any]
        :return: ``(tool_calls, config)`` tuple for the next hook or
            dispatch
        :rtype: tuple[list[dict[str, Any]], RunnableConfig]
        """
        ...

    async def on_tool_start(
        self,
        tool_call: dict[str, Any],
        config: RunnableConfig,
        state: dict[str, Any],
    ) -> None:
        """fired immediately before a single tool is dispatched.

        default implementation is a no-op. emitters publish
        ``tool_call_start`` envelopes, open timing spans, etc.

        :param tool_call: single tool_call dict with ``id``, ``name``,
            ``args`` keys
        :ptype tool_call: dict[str, Any]
        :param config: LangGraph runtime config dict
        :ptype config: RunnableConfig
        :param state: current agent state (read-only view for hooks)
        :ptype state: dict[str, Any]
        :return: nothing
        :rtype: None
        """
        ...

    async def on_tool_end(
        self,
        tool_call: dict[str, Any],
        result: Any,
        success: bool,
        elapsed_ms: int,
        config: RunnableConfig,
        state: dict[str, Any],
    ) -> None:
        """fired immediately after a single tool call completes.

        default implementation is a no-op. ``success=False`` fires when
        the tool raised or when the tool name did not resolve.

        :param tool_call: single tool_call dict with ``id``, ``name``,
            ``args`` keys
        :ptype tool_call: dict[str, Any]
        :param result: the value returned by the tool (or an error
            string when ``success=False``)
        :ptype result: Any
        :param success: whether tool dispatch succeeded
        :ptype success: bool
        :param elapsed_ms: wall-clock milliseconds from dispatch to
            completion
        :ptype elapsed_ms: int
        :param config: LangGraph runtime config dict
        :ptype config: RunnableConfig
        :param state: current agent state (read-only view for hooks)
        :ptype state: dict[str, Any]
        :return: nothing
        :rtype: None
        """
        ...

    async def on_heartbeat(
        self,
        tool_call: dict[str, Any],
        elapsed_seconds: float,
        config: RunnableConfig,
        state: dict[str, Any],
    ) -> None:
        """fired periodically while a long tool call is running.

        default implementation is a no-op. emitters publish
        ``tool_call_progress`` envelopes so that upstream stream
        consumers see a liveness signal and do not trip on gap
        timeouts. tool_node installs a heartbeat loop only when at
        least one hook implements a non-default ``on_heartbeat``;
        implementations that do not care pay no cost.

        :param tool_call: single tool_call dict with ``id``, ``name``,
            ``args`` keys
        :ptype tool_call: dict[str, Any]
        :param elapsed_seconds: wall-clock seconds since tool dispatch
            started; monotonic
        :ptype elapsed_seconds: float
        :param config: LangGraph runtime config dict
        :ptype config: RunnableConfig
        :param state: current agent state (read-only view for hooks)
        :ptype state: dict[str, Any]
        :return: nothing
        :rtype: None
        """
        ...


class _ComposedAgentNodeHook:
    """adapter that fans agent_node hook calls across a sequence.

    hidden implementation behind :func:`compose_agent_node_hooks`.
    calls :meth:`AgentNodeHook.before_invoke` on each wrapped hook in
    order, threading the ``(messages, config)`` tuple through. fires
    :meth:`AgentNodeHook.after_invoke` in the same order (NOT
    reversed -- hooks are peers, not wrapping decorators; the sequence
    order is the one contract).
    """

    def __init__(self, hooks: Sequence[AgentNodeHook]) -> None:
        """capture the hook sequence for later dispatch.

        :param hooks: sequence of :class:`AgentNodeHook` instances
        :ptype hooks: Sequence[AgentNodeHook]
        """
        self._hooks: tuple[AgentNodeHook, ...] = tuple(hooks)

    async def before_invoke(
        self,
        messages: list[BaseMessage],
        config: RunnableConfig,
        state: dict[str, Any],
    ) -> tuple[list[BaseMessage], RunnableConfig]:
        """sequentially invoke :meth:`before_invoke` on each hook.

        :param messages: message list to thread through hooks
        :ptype messages: list[BaseMessage]
        :param config: config dict to thread through hooks
        :ptype config: RunnableConfig
        :param state: read-only state view passed to each hook
        :ptype state: dict[str, Any]
        :return: final ``(messages, config)`` after all hooks applied
        :rtype: tuple[list[BaseMessage], RunnableConfig]
        """
        current_messages = messages
        current_config = config
        for hook in self._hooks:
            current_messages, current_config = await hook.before_invoke(
                current_messages, current_config, state,
            )
        return current_messages, current_config

    async def after_invoke(
        self,
        response: Any,
        config: RunnableConfig,
        state: dict[str, Any],
    ) -> Any:
        """sequentially invoke :meth:`after_invoke` on each hook.

        :param response: response object to thread through hooks
        :ptype response: Any
        :param config: config dict passed to each hook
        :ptype config: RunnableConfig
        :param state: read-only state view passed to each hook
        :ptype state: dict[str, Any]
        :return: final response after all hooks applied
        :rtype: Any
        """
        current_response = response
        for hook in self._hooks:
            current_response = await hook.after_invoke(
                current_response, config, state,
            )
        return current_response


class _ComposedToolNodeHook:
    """adapter that fans tool_node hook calls across a sequence.

    hidden implementation behind :func:`compose_tool_node_hooks`.
    sequences ``before_dispatch`` threads; ``on_tool_start``,
    ``on_tool_end``, ``on_heartbeat`` fan out in order and are
    independent (failure in one emitter is swallowed at the node
    level -- emitters should not crash the graph).
    """

    def __init__(self, hooks: Sequence[ToolNodeHook]) -> None:
        """capture the hook sequence for later dispatch.

        :param hooks: sequence of :class:`ToolNodeHook` instances
        :ptype hooks: Sequence[ToolNodeHook]
        """
        self._hooks: tuple[ToolNodeHook, ...] = tuple(hooks)

    async def before_dispatch(
        self,
        tool_calls: list[dict[str, Any]],
        config: RunnableConfig,
        state: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], RunnableConfig]:
        """sequentially invoke :meth:`before_dispatch` on each hook.

        :param tool_calls: tool_call list to thread through hooks
        :ptype tool_calls: list[dict[str, Any]]
        :param config: config dict to thread through hooks
        :ptype config: RunnableConfig
        :param state: read-only state view passed to each hook
        :ptype state: dict[str, Any]
        :return: final ``(tool_calls, config)`` after all hooks
        :rtype: tuple[list[dict[str, Any]], RunnableConfig]
        """
        current_tool_calls = tool_calls
        current_config = config
        for hook in self._hooks:
            current_tool_calls, current_config = await hook.before_dispatch(
                current_tool_calls, current_config, state,
            )
        return current_tool_calls, current_config

    async def on_tool_start(
        self,
        tool_call: dict[str, Any],
        config: RunnableConfig,
        state: dict[str, Any],
    ) -> None:
        """fan :meth:`on_tool_start` across every hook in order.

        :param tool_call: single tool_call dict
        :ptype tool_call: dict[str, Any]
        :param config: config dict
        :ptype config: RunnableConfig
        :param state: read-only state view
        :ptype state: dict[str, Any]
        :return: nothing
        :rtype: None
        """
        for hook in self._hooks:
            await hook.on_tool_start(tool_call, config, state)

    async def on_tool_end(
        self,
        tool_call: dict[str, Any],
        result: Any,
        success: bool,
        elapsed_ms: int,
        config: RunnableConfig,
        state: dict[str, Any],
    ) -> None:
        """fan :meth:`on_tool_end` across every hook in order.

        :param tool_call: single tool_call dict
        :ptype tool_call: dict[str, Any]
        :param result: tool result value
        :ptype result: Any
        :param success: whether tool dispatch succeeded
        :ptype success: bool
        :param elapsed_ms: wall-clock milliseconds for the call
        :ptype elapsed_ms: int
        :param config: config dict
        :ptype config: RunnableConfig
        :param state: read-only state view
        :ptype state: dict[str, Any]
        :return: nothing
        :rtype: None
        """
        for hook in self._hooks:
            await hook.on_tool_end(
                tool_call, result, success, elapsed_ms, config, state,
            )

    async def on_heartbeat(
        self,
        tool_call: dict[str, Any],
        elapsed_seconds: float,
        config: RunnableConfig,
        state: dict[str, Any],
    ) -> None:
        """fan :meth:`on_heartbeat` across every hook in order.

        :param tool_call: single tool_call dict
        :ptype tool_call: dict[str, Any]
        :param elapsed_seconds: seconds since tool dispatch started
        :ptype elapsed_seconds: float
        :param config: config dict
        :ptype config: RunnableConfig
        :param state: read-only state view
        :ptype state: dict[str, Any]
        :return: nothing
        :rtype: None
        """
        for hook in self._hooks:
            await hook.on_heartbeat(tool_call, elapsed_seconds, config, state)


def compose_agent_node_hooks(
    hooks: Sequence[AgentNodeHook],
) -> AgentNodeHook:
    """compose a sequence of :class:`AgentNodeHook` into a single hook.

    returns an adapter that calls :meth:`before_invoke` and
    :meth:`after_invoke` on each wrapped hook in sequence order.
    passing an empty sequence returns an adapter that is a pure
    no-op pass-through.

    :param hooks: sequence of :class:`AgentNodeHook` instances to
        fan across
    :ptype hooks: Sequence[AgentNodeHook]
    :return: single :class:`AgentNodeHook` that dispatches to all
        wrapped hooks in order
    :rtype: AgentNodeHook
    """
    return _ComposedAgentNodeHook(hooks)


def compose_tool_node_hooks(
    hooks: Sequence[ToolNodeHook],
) -> ToolNodeHook:
    """compose a sequence of :class:`ToolNodeHook` into a single hook.

    returns an adapter that calls :meth:`before_dispatch`,
    :meth:`on_tool_start`, :meth:`on_tool_end`, and
    :meth:`on_heartbeat` on each wrapped hook in sequence order.
    passing an empty sequence returns an adapter that is a pure
    no-op pass-through.

    :param hooks: sequence of :class:`ToolNodeHook` instances to
        fan across
    :ptype hooks: Sequence[ToolNodeHook]
    :return: single :class:`ToolNodeHook` that dispatches to all
        wrapped hooks in order
    :rtype: ToolNodeHook
    """
    return _ComposedToolNodeHook(hooks)
