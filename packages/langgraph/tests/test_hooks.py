"""unit tests for :mod:`threetears.langgraph.hooks`.

tests cover the two protocols (AgentNodeHook, ToolNodeHook) and the
two composition helpers in isolation -- no graph is compiled, no
LLM is invoked. concrete protocol implementations used here are
test doubles (not production hooks).
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from threetears.langgraph import (
    AgentNodeHook,
    ToolNodeHook,
    compose_agent_node_hooks,
    compose_tool_node_hooks,
)
from threetears.langgraph.hooks import _ComposedAgentNodeHook, _ComposedToolNodeHook


class _RecordingAgentHook:
    """test double that appends a tag on each hook phase.

    used to verify sequence order and argument threading through
    :func:`compose_agent_node_hooks`.
    """

    def __init__(self, tag: str, events: list[str]) -> None:
        """store event sink shared across hooks in a test.

        :param tag: string identifier stamped on each recorded event
        :ptype tag: str
        :param events: shared list for recording the order of calls
        :ptype events: list[str]
        """
        self._tag = tag
        self._events = events

    async def before_invoke(
        self,
        messages: list[Any],
        config: RunnableConfig,
        state: dict[str, Any],
    ) -> tuple[list[Any], RunnableConfig]:
        """record call and append a sentinel to the message list.

        :param messages: incoming messages
        :ptype messages: list[Any]
        :param config: incoming config
        :ptype config: RunnableConfig
        :param state: state dict view
        :ptype state: dict[str, Any]
        :return: ``(messages + sentinel, config)``
        :rtype: tuple[list[Any], RunnableConfig]
        """
        self._events.append(f"before:{self._tag}")
        new_messages = list(messages) + [HumanMessage(content=f"from-{self._tag}")]
        return new_messages, config

    async def after_invoke(
        self,
        response: Any,
        config: RunnableConfig,
        state: dict[str, Any],
    ) -> Any:
        """record call and mark the response with the hook tag.

        :param response: incoming response
        :ptype response: Any
        :param config: incoming config
        :ptype config: RunnableConfig
        :param state: state dict view
        :ptype state: dict[str, Any]
        :return: response with hook tag appended to content
        :rtype: Any
        """
        self._events.append(f"after:{self._tag}")
        if isinstance(response, AIMessage):
            result = AIMessage(content=f"{response.content}/{self._tag}")
        else:
            result = response
        return result


class _RecordingToolHook:
    """test double that records every tool_node hook phase.

    :param tag: tag identifying this hook in recorded events
    :ptype tag: str
    :param events: shared list for recording the order of calls
    :ptype events: list[str]
    """

    def __init__(self, tag: str, events: list[str]) -> None:
        """store event sink for assertions.

        :param tag: string identifier stamped on each recorded event
        :ptype tag: str
        :param events: shared list for recording
        :ptype events: list[str]
        """
        self._tag = tag
        self._events = events

    async def before_dispatch(
        self,
        tool_calls: list[dict[str, Any]],
        config: RunnableConfig,
        state: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], RunnableConfig]:
        """record and pass through unchanged.

        :param tool_calls: incoming tool_call list
        :ptype tool_calls: list[dict[str, Any]]
        :param config: incoming config
        :ptype config: RunnableConfig
        :param state: state dict view
        :ptype state: dict[str, Any]
        :return: ``(tool_calls, config)`` unchanged
        :rtype: tuple[list[dict[str, Any]], RunnableConfig]
        """
        self._events.append(f"before_dispatch:{self._tag}:{len(tool_calls)}")
        return tool_calls, config

    async def on_tool_start(
        self,
        tool_call: dict[str, Any],
        config: RunnableConfig,
        state: dict[str, Any],
    ) -> None:
        """record tool-start emission.

        :param tool_call: single tool_call dict
        :ptype tool_call: dict[str, Any]
        :param config: config dict
        :ptype config: RunnableConfig
        :param state: state dict view
        :ptype state: dict[str, Any]
        :return: nothing
        :rtype: None
        """
        self._events.append(f"start:{self._tag}:{tool_call.get('name')}")

    async def on_tool_end(
        self,
        tool_call: dict[str, Any],
        result: Any,
        success: bool,
        elapsed_ms: int,
        config: RunnableConfig,
        state: dict[str, Any],
    ) -> None:
        """record tool-end emission.

        :param tool_call: single tool_call dict
        :ptype tool_call: dict[str, Any]
        :param result: tool result value
        :ptype result: Any
        :param success: outcome flag
        :ptype success: bool
        :param elapsed_ms: wall-clock milliseconds
        :ptype elapsed_ms: int
        :param config: config dict
        :ptype config: RunnableConfig
        :param state: state dict view
        :ptype state: dict[str, Any]
        :return: nothing
        :rtype: None
        """
        self._events.append(
            f"end:{self._tag}:{tool_call.get('name')}:{success}",
        )

    async def on_heartbeat(
        self,
        tool_call: dict[str, Any],
        elapsed_seconds: float,
        config: RunnableConfig,
        state: dict[str, Any],
    ) -> None:
        """record heartbeat emission.

        :param tool_call: single tool_call dict
        :ptype tool_call: dict[str, Any]
        :param elapsed_seconds: elapsed seconds since dispatch
        :ptype elapsed_seconds: float
        :param config: config dict
        :ptype config: RunnableConfig
        :param state: state dict view
        :ptype state: dict[str, Any]
        :return: nothing
        :rtype: None
        """
        self._events.append(f"hb:{self._tag}:{tool_call.get('name')}")


class TestAgentNodeHookProtocol:
    """runtime-checkable protocol recognizes a conforming hook."""

    def test_recording_hook_satisfies_protocol(self) -> None:
        """minimal hook double is an instance of :class:`AgentNodeHook`.

        :raises AssertionError: when protocol runtime check fails
        """
        hook = _RecordingAgentHook("x", [])
        assert isinstance(hook, AgentNodeHook)


class TestToolNodeHookProtocol:
    """runtime-checkable protocol recognizes a conforming tool hook."""

    def test_recording_hook_satisfies_protocol(self) -> None:
        """minimal hook double is an instance of :class:`ToolNodeHook`.

        :raises AssertionError: when protocol runtime check fails
        """
        hook = _RecordingToolHook("x", [])
        assert isinstance(hook, ToolNodeHook)


class TestComposeAgentNodeHooks:
    """:func:`compose_agent_node_hooks` threads hooks in order."""

    def test_empty_returns_noop_adapter(self) -> None:
        """empty sequence returns a composed adapter instance.

        :raises AssertionError: when the return type is unexpected
        """
        composed = compose_agent_node_hooks([])
        assert isinstance(composed, _ComposedAgentNodeHook)

    @pytest.mark.asyncio
    async def test_empty_before_invoke_is_identity(self) -> None:
        """empty hook list threads messages and config unchanged.

        :raises AssertionError: when threading alters inputs
        """
        composed = compose_agent_node_hooks([])
        msgs: list[Any] = [HumanMessage(content="hi")]
        cfg: RunnableConfig = {"configurable": {"key": "val"}}  # type: ignore[typeddict-item]
        out_msgs, out_cfg = await composed.before_invoke(msgs, cfg, {})
        assert out_msgs == msgs
        assert out_cfg == cfg

    @pytest.mark.asyncio
    async def test_empty_after_invoke_is_identity(self) -> None:
        """empty hook list returns response unchanged.

        :raises AssertionError: when after_invoke alters the response
        """
        composed = compose_agent_node_hooks([])
        ai = AIMessage(content="response")
        out = await composed.after_invoke(ai, {"configurable": {}}, {})
        assert out is ai

    @pytest.mark.asyncio
    async def test_before_invoke_sequences_in_order(self) -> None:
        """multiple hooks fire before_invoke in sequence order.

        :raises AssertionError: when ordering is not preserved
        """
        events: list[str] = []
        composed = compose_agent_node_hooks(
            [
                _RecordingAgentHook("a", events),
                _RecordingAgentHook("b", events),
            ],
        )
        await composed.before_invoke([HumanMessage(content="hi")], {"configurable": {}}, {})
        assert events == ["before:a", "before:b"]

    @pytest.mark.asyncio
    async def test_before_invoke_threads_messages(self) -> None:
        """each hook sees the previous hook's rewritten messages.

        :raises AssertionError: when threading does not propagate
            modifications
        """
        events: list[str] = []
        composed = compose_agent_node_hooks(
            [
                _RecordingAgentHook("a", events),
                _RecordingAgentHook("b", events),
            ],
        )
        out_msgs, _ = await composed.before_invoke(
            [HumanMessage(content="seed")],
            {"configurable": {}},
            {},
        )
        # hooks each append one sentinel -> total 3 messages
        assert len(out_msgs) == 3
        assert isinstance(out_msgs[1], HumanMessage)
        assert out_msgs[1].content == "from-a"
        assert out_msgs[2].content == "from-b"

    @pytest.mark.asyncio
    async def test_after_invoke_sequences_in_order(self) -> None:
        """multiple hooks fire after_invoke in sequence order.

        :raises AssertionError: when ordering is not preserved
        """
        events: list[str] = []
        composed = compose_agent_node_hooks(
            [
                _RecordingAgentHook("a", events),
                _RecordingAgentHook("b", events),
            ],
        )
        out = await composed.after_invoke(
            AIMessage(content="x"),
            {"configurable": {}},
            {},
        )
        assert events == ["after:a", "after:b"]
        assert isinstance(out, AIMessage)
        assert out.content == "x/a/b"


class TestComposeToolNodeHooks:
    """:func:`compose_tool_node_hooks` fans hooks in order."""

    def test_empty_returns_composed_adapter(self) -> None:
        """empty sequence returns a composed tool-hook adapter.

        :raises AssertionError: when return type is unexpected
        """
        composed = compose_tool_node_hooks([])
        assert isinstance(composed, _ComposedToolNodeHook)

    @pytest.mark.asyncio
    async def test_empty_before_dispatch_is_identity(self) -> None:
        """empty hook list threads tool_calls and config unchanged.

        :raises AssertionError: when threading alters inputs
        """
        composed = compose_tool_node_hooks([])
        tool_calls: list[dict[str, Any]] = [{"id": "1", "name": "t", "args": {}}]
        cfg: RunnableConfig = {"configurable": {}}
        out_calls, out_cfg = await composed.before_dispatch(tool_calls, cfg, {})
        assert out_calls == tool_calls
        assert out_cfg == cfg

    @pytest.mark.asyncio
    async def test_before_dispatch_sequences_hooks(self) -> None:
        """two hooks fire before_dispatch in order.

        :raises AssertionError: when ordering is not preserved
        """
        events: list[str] = []
        composed = compose_tool_node_hooks(
            [
                _RecordingToolHook("a", events),
                _RecordingToolHook("b", events),
            ],
        )
        await composed.before_dispatch(
            [{"id": "1", "name": "t", "args": {}}],
            {"configurable": {}},
            {},
        )
        assert events == ["before_dispatch:a:1", "before_dispatch:b:1"]

    @pytest.mark.asyncio
    async def test_on_tool_start_fans_out(self) -> None:
        """all hooks receive on_tool_start.

        :raises AssertionError: when any hook is skipped
        """
        events: list[str] = []
        composed = compose_tool_node_hooks(
            [
                _RecordingToolHook("a", events),
                _RecordingToolHook("b", events),
            ],
        )
        await composed.on_tool_start(
            {"id": "1", "name": "calc", "args": {}},
            {"configurable": {}},
            {},
        )
        assert events == ["start:a:calc", "start:b:calc"]

    @pytest.mark.asyncio
    async def test_on_tool_end_fans_out(self) -> None:
        """all hooks receive on_tool_end with the same arguments.

        :raises AssertionError: when any hook is skipped
        """
        events: list[str] = []
        composed = compose_tool_node_hooks(
            [
                _RecordingToolHook("a", events),
                _RecordingToolHook("b", events),
            ],
        )
        await composed.on_tool_end(
            {"id": "1", "name": "calc", "args": {}},
            "42",
            True,
            10,
            {"configurable": {}},
            {},
        )
        assert events == ["end:a:calc:True", "end:b:calc:True"]

    @pytest.mark.asyncio
    async def test_on_heartbeat_fans_out(self) -> None:
        """all hooks receive on_heartbeat.

        :raises AssertionError: when any hook is skipped
        """
        events: list[str] = []
        composed = compose_tool_node_hooks(
            [
                _RecordingToolHook("a", events),
                _RecordingToolHook("b", events),
            ],
        )
        await composed.on_heartbeat(
            {"id": "1", "name": "calc", "args": {}},
            1.5,
            {"configurable": {}},
            {},
        )
        assert events == ["hb:a:calc", "hb:b:calc"]


class TestAgentHookSystemPrefixContract:
    """document the system-prompt-preservation contract.

    the contract is documented on :class:`AgentNodeHook`; this test
    verifies a well-behaved hook follows it when handed a message list
    with a :class:`SystemMessage` at index 0.
    """

    @pytest.mark.asyncio
    async def test_recording_hook_preserves_system_at_index_zero(self) -> None:
        """_RecordingAgentHook leaves the existing system prefix in place.

        :raises AssertionError: when index 0 mutates
        """
        events: list[str] = []
        hook = _RecordingAgentHook("tag", events)
        sys_msg = SystemMessage(content="prefix")
        messages: list[Any] = [sys_msg, HumanMessage(content="hi")]
        out_msgs, _ = await hook.before_invoke(messages, {"configurable": {}}, {})
        assert out_msgs[0] is sys_msg
