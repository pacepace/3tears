"""tests for hook wiring into :func:`agent_node` and :func:`tool_node`.

covers:

- empty-hook path is identical to the pre-refactor behavior
  (existing smoke + builder tests cover this implicitly; we pin it
  with an explicit assertion here).
- before_invoke / after_invoke are invoked on hooks stamped into
  ``configurable["_hooks"]["agent"]`` in sequence order.
- tool_node threads ``config=`` into ``tool.ainvoke`` so downstream
  tools can read ``configurable`` entries -- that is the 3tears-side
  config-threading promotion the SDK previously had to fork to.
- tool_node fires ``on_tool_start`` / ``on_tool_end`` and schedules
  a heartbeat loop when at least one hook implements it.
- the heartbeat interval is configurable via
  ``configurable["_hook_heartbeat_seconds"]`` with ``<= 0`` disabling
  the loop entirely.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
)
from langchain_core.runnables import RunnableConfig
from threetears.langgraph.nodes import agent_node, tool_node


class _SpyAgentHook:
    """agent hook that records invocations and rewrites content.

    used to assert ordering plus threading: ``before_invoke`` appends
    a sentinel message so the test can verify the model saw it;
    ``after_invoke`` rewrites the response content so the test can
    verify it replaces the return value.
    """

    def __init__(self, tag: str, events: list[str]) -> None:
        """capture per-hook tag and shared event sink.

        :param tag: hook identifier stamped on events
        :ptype tag: str
        :param events: shared list tracking call order
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
        """record + append sentinel message.

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
        return list(messages) + [HumanMessage(content=f"probe-{self._tag}")], config

    async def after_invoke(
        self,
        response: Any,
        config: RunnableConfig,
        state: dict[str, Any],
    ) -> Any:
        """record + stamp tag onto response content.

        :param response: model response
        :ptype response: Any
        :param config: config
        :ptype config: RunnableConfig
        :param state: state dict view
        :ptype state: dict[str, Any]
        :return: :class:`AIMessage` with tag appended
        :rtype: Any
        """
        self._events.append(f"after:{self._tag}")
        if isinstance(response, AIMessage):
            return AIMessage(content=f"{response.content}+{self._tag}")
        return response


class _SpyToolHook:
    """tool hook with non-default ``on_heartbeat`` and full recording."""

    def __init__(self, events: list[str]) -> None:
        """store shared event sink for assertions.

        :param events: shared list tracking call order
        :ptype events: list[str]
        """
        self._events = events

    async def before_dispatch(
        self,
        tool_calls: list[dict[str, Any]],
        config: RunnableConfig,
        state: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], RunnableConfig]:
        """record without mutating.

        :param tool_calls: incoming tool_call list
        :ptype tool_calls: list[dict[str, Any]]
        :param config: incoming config
        :ptype config: RunnableConfig
        :param state: state dict view
        :ptype state: dict[str, Any]
        :return: ``(tool_calls, config)`` unchanged
        :rtype: tuple[list[dict[str, Any]], RunnableConfig]
        """
        self._events.append(f"before_dispatch:{len(tool_calls)}")
        return tool_calls, config

    async def on_tool_start(
        self,
        tool_call: dict[str, Any],
        config: RunnableConfig,
        state: dict[str, Any],
    ) -> None:
        """record start emission.

        :param tool_call: tool_call dict
        :ptype tool_call: dict[str, Any]
        :param config: config
        :ptype config: RunnableConfig
        :param state: state dict view
        :ptype state: dict[str, Any]
        :return: nothing
        :rtype: None
        """
        self._events.append(f"start:{tool_call.get('name')}")

    async def on_tool_end(
        self,
        tool_call: dict[str, Any],
        result: Any,
        success: bool,
        elapsed_ms: int,
        config: RunnableConfig,
        state: dict[str, Any],
    ) -> None:
        """record end emission.

        :param tool_call: tool_call dict
        :ptype tool_call: dict[str, Any]
        :param result: tool result
        :ptype result: Any
        :param success: outcome flag
        :ptype success: bool
        :param elapsed_ms: wall-clock ms
        :ptype elapsed_ms: int
        :param config: config
        :ptype config: RunnableConfig
        :param state: state dict view
        :ptype state: dict[str, Any]
        :return: nothing
        :rtype: None
        """
        self._events.append(f"end:{tool_call.get('name')}:{success}")

    async def on_heartbeat(
        self,
        tool_call: dict[str, Any],
        elapsed_seconds: float,
        config: RunnableConfig,
        state: dict[str, Any],
    ) -> None:
        """record heartbeat emission.

        :param tool_call: tool_call dict
        :ptype tool_call: dict[str, Any]
        :param elapsed_seconds: elapsed seconds
        :ptype elapsed_seconds: float
        :param config: config
        :ptype config: RunnableConfig
        :param state: state dict view
        :ptype state: dict[str, Any]
        :return: nothing
        :rtype: None
        """
        self._events.append(f"hb:{tool_call.get('name')}")


class TestAgentNodeHooks:
    """hooks installed via ``configurable["_hooks"]["agent"]`` fire."""

    @pytest.mark.asyncio
    async def test_no_hooks_is_noop(self) -> None:
        """empty hook set does not alter behavior.

        :raises AssertionError: when the model is not invoked once
        """
        mock_model = AsyncMock()
        mock_model.ainvoke = AsyncMock(return_value=AIMessage(content="ok"))
        state: dict[str, Any] = {"messages": [HumanMessage(content="hi")]}
        config: RunnableConfig = {
            "configurable": {"chat_model": mock_model, "system_prompt": ""},
        }
        result = await agent_node(state, config)  # type: ignore[arg-type]
        assert result["messages"][0].content == "ok"
        mock_model.ainvoke.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_before_invoke_runs_and_threads_messages(self) -> None:
        """model sees hook-appended messages.

        :raises AssertionError: when the appended message is missing
        """
        events: list[str] = []
        mock_model = AsyncMock()
        mock_model.ainvoke = AsyncMock(return_value=AIMessage(content="ok"))
        state: dict[str, Any] = {"messages": [HumanMessage(content="hi")]}
        config: RunnableConfig = {
            "configurable": {
                "chat_model": mock_model,
                "system_prompt": "",
                "_hooks": {"agent": [_SpyAgentHook("a", events)]},
            },
        }
        await agent_node(state, config)  # type: ignore[arg-type]
        call_args = mock_model.ainvoke.call_args
        messages_sent = call_args[0][0]
        assert any(isinstance(m, HumanMessage) and m.content == "probe-a" for m in messages_sent)
        assert events == ["before:a", "after:a"]

    @pytest.mark.asyncio
    async def test_after_invoke_can_rewrite_response(self) -> None:
        """hook's after_invoke return value is used for state update.

        :raises AssertionError: when the rewrite does not land
        """
        events: list[str] = []
        mock_model = AsyncMock()
        mock_model.ainvoke = AsyncMock(return_value=AIMessage(content="ok"))
        state: dict[str, Any] = {"messages": [HumanMessage(content="hi")]}
        config: RunnableConfig = {
            "configurable": {
                "chat_model": mock_model,
                "system_prompt": "",
                "_hooks": {"agent": [_SpyAgentHook("a", events)]},
            },
        }
        result = await agent_node(state, config)  # type: ignore[arg-type]
        assert result["messages"][0].content == "ok+a"


class TestToolNodeHooks:
    """tool_node emits hooks and threads config into tool.ainvoke."""

    @pytest.mark.asyncio
    async def test_config_threaded_into_tool_ainvoke(self) -> None:
        """tool.ainvoke receives ``config=`` keyword argument.

        this is the 3tears upstream contract fix: LangGraph's public
        contract says tools get config. tools that need
        ``configurable`` entries (conversation_id, call_context) read
        them through this path instead of requiring the caller to
        fork tool_node.

        :raises AssertionError: when config is not threaded
        """
        mock_tool = AsyncMock()
        mock_tool.name = "calc"
        mock_tool.ainvoke = AsyncMock(return_value="42")
        ai_msg = AIMessage(content="")
        ai_msg.tool_calls = [{"id": "tc1", "name": "calc", "args": {"x": 1}}]
        state: dict[str, Any] = {"messages": [ai_msg]}
        config: RunnableConfig = {
            "configurable": {"tools": [mock_tool], "_hook_heartbeat_seconds": 0.0},
        }
        await tool_node(state, config)  # type: ignore[arg-type]
        call_kwargs = mock_tool.ainvoke.call_args.kwargs
        assert "config" in call_kwargs
        assert call_kwargs["config"] is config

    @pytest.mark.asyncio
    async def test_on_tool_start_and_end_fire(self) -> None:
        """tool lifecycle hooks fire around dispatch.

        :raises AssertionError: when either start or end is missing
        """
        events: list[str] = []
        mock_tool = AsyncMock()
        mock_tool.name = "calc"
        mock_tool.ainvoke = AsyncMock(return_value="42")
        ai_msg = AIMessage(content="")
        ai_msg.tool_calls = [{"id": "tc1", "name": "calc", "args": {}}]
        state: dict[str, Any] = {"messages": [ai_msg]}
        config: RunnableConfig = {
            "configurable": {
                "tools": [mock_tool],
                "_hooks": {"tool": [_SpyToolHook(events)]},
                "_hook_heartbeat_seconds": 0.0,
            },
        }
        await tool_node(state, config)  # type: ignore[arg-type]
        assert events == ["before_dispatch:1", "start:calc", "end:calc:True"]

    @pytest.mark.asyncio
    async def test_heartbeat_fires_when_tool_slower_than_interval(self) -> None:
        """heartbeat ticks while a slow tool is running.

        uses a short interval and a tool that sleeps longer than one
        tick so at least one heartbeat must fire.

        :raises AssertionError: when no heartbeat was observed
        """
        events: list[str] = []

        async def slow(args: Any, config: Any = None) -> str:
            await asyncio.sleep(0.05)
            return "done"

        mock_tool = MagicMock()
        mock_tool.name = "slow"
        mock_tool.ainvoke = slow
        ai_msg = AIMessage(content="")
        ai_msg.tool_calls = [{"id": "tc1", "name": "slow", "args": {}}]
        state: dict[str, Any] = {"messages": [ai_msg]}
        config: RunnableConfig = {
            "configurable": {
                "tools": [mock_tool],
                "_hooks": {"tool": [_SpyToolHook(events)]},
                "_hook_heartbeat_seconds": 0.01,
            },
        }
        await tool_node(state, config)  # type: ignore[arg-type]
        assert any(e == "hb:slow" for e in events)

    @pytest.mark.asyncio
    async def test_heartbeat_disabled_by_zero_interval(self) -> None:
        """``_hook_heartbeat_seconds <= 0`` disables the heartbeat loop.

        :raises AssertionError: when a heartbeat fires with disabled interval
        """
        events: list[str] = []

        async def slow(args: Any, config: Any = None) -> str:
            await asyncio.sleep(0.02)
            return "done"

        mock_tool = MagicMock()
        mock_tool.name = "slow"
        mock_tool.ainvoke = slow
        ai_msg = AIMessage(content="")
        ai_msg.tool_calls = [{"id": "tc1", "name": "slow", "args": {}}]
        state: dict[str, Any] = {"messages": [ai_msg]}
        config: RunnableConfig = {
            "configurable": {
                "tools": [mock_tool],
                "_hooks": {"tool": [_SpyToolHook(events)]},
                "_hook_heartbeat_seconds": 0.0,
            },
        }
        await tool_node(state, config)  # type: ignore[arg-type]
        assert all(not e.startswith("hb:") for e in events)


class TestToolNodeUnknownToolName:
    """tool_node returns a 'did you mean' hint when the LLM emits a
    tool name that does not match any registered tool. without the
    hint, the LLM saw a bare ``tool 'X' not found`` message that
    looked the same for every kind of miss; it would then loop
    trying minor variants of the same wrong name because nothing in
    the error pointed at the right shape. the hint uses difflib's
    close-match heuristic which catches the common families of
    confusion: dot-vs-underscore boundaries, missing namespace
    prefix, transposed characters.
    """

    @pytest.mark.asyncio
    async def test_unknown_tool_returns_did_you_mean_hint(self) -> None:
        """when the LLM calls a near-miss name, the tool message
        carries a ``did you mean`` line naming up to three closest
        registered tools so the LLM can self-correct on the next
        turn.
        """
        mock_tool = AsyncMock()
        mock_tool.name = "datasource_central-reporting_schema"
        mock_tool.ainvoke = AsyncMock(return_value="should not be called")
        ai_msg = AIMessage(content="")
        # LLM emits the dotted form (common artefact of provider-side
        # name sanitisation leaking back into the model's context)
        ai_msg.tool_calls = [
            {"id": "tc1", "name": "datasource.central-reporting.schema", "args": {}},
        ]
        state: dict[str, Any] = {"messages": [ai_msg]}
        config: RunnableConfig = {
            "configurable": {"tools": [mock_tool], "_hook_heartbeat_seconds": 0.0},
        }

        result = await tool_node(state, config)  # type: ignore[arg-type]

        tool_message_content = result["messages"][0].content
        assert "not found" in tool_message_content
        assert "did you mean" in tool_message_content
        # the close match shows up explicitly so the LLM does not
        # have to guess
        assert "datasource_central-reporting_schema" in tool_message_content
        # the wrongly-called tool is NOT actually dispatched
        mock_tool.ainvoke.assert_not_called()

    @pytest.mark.asyncio
    async def test_unknown_tool_no_hint_when_nothing_close(self) -> None:
        """when nothing is structurally close to the wrong name, the
        message omits the ``did you mean`` line rather than offering
        a misleading suggestion. the LLM gets a clean unknown-tool
        error and the available-tools list it can scan.
        """
        mock_tool = AsyncMock()
        mock_tool.name = "completely_unrelated_tool"
        mock_tool.ainvoke = AsyncMock(return_value="should not be called")
        ai_msg = AIMessage(content="")
        ai_msg.tool_calls = [{"id": "tc1", "name": "xyz", "args": {}}]
        state: dict[str, Any] = {"messages": [ai_msg]}
        config: RunnableConfig = {
            "configurable": {"tools": [mock_tool], "_hook_heartbeat_seconds": 0.0},
        }

        result = await tool_node(state, config)  # type: ignore[arg-type]

        tool_message_content = result["messages"][0].content
        assert "not found" in tool_message_content
        assert "did you mean" not in tool_message_content


class TestToolNodeInterruptPropagation:
    """A ``GraphInterrupt`` (``GraphBubbleUp`` family) raised inside a tool MUST propagate out
    of :func:`tool_node`, not be swallowed into a "Tool error" ``ToolMessage``. ``interrupt()``-
    based human-in-the-loop depends on it: the broad ``except Exception`` in tool_node would
    otherwise turn the pause signal into a tool error and run the graph to completion, silently
    defeating every interrupt-based tool. Regression guard for the GraphBubbleUp re-raise."""

    @pytest.mark.asyncio
    async def test_graph_interrupt_propagates_out_of_tool_node(self) -> None:
        from langgraph.errors import GraphBubbleUp, GraphInterrupt

        mock_tool = AsyncMock()
        mock_tool.name = "edit_object"
        mock_tool.ainvoke = AsyncMock(side_effect=GraphInterrupt(()))
        ai_msg = AIMessage(content="")
        ai_msg.tool_calls = [{"id": "tc1", "name": "edit_object", "args": {}}]
        state: dict[str, Any] = {"messages": [ai_msg]}
        config: RunnableConfig = {"configurable": {"tools": [mock_tool], "_hook_heartbeat_seconds": 0.0}}

        # It must BUBBLE OUT (so Pregel/checkpointer can pause), not be caught into a ToolMessage.
        with pytest.raises(GraphBubbleUp):
            await tool_node(state, config)  # type: ignore[arg-type]


class TestSystemMessageConsolidation:
    """regression: upstream retrieval nodes (knowledge / memory) append
    their per-turn context as a separate SystemMessage to state.messages.
    an injected SystemMessage landing AFTER a human turn reached Anthropic
    as a NON-CONSECUTIVE system message ("Received multiple non-consecutive
    system messages" -> no generations). the agent_node consolidates ALL
    SystemMessages into the single leading system prompt and removes them
    from persisted history so per-turn context never accumulates.
    """

    @pytest.mark.asyncio
    async def test_injected_system_message_folded_into_single_leading_prompt(
        self,
    ) -> None:
        """a System message after a Human turn is merged into one leading
        system message; the model never sees non-consecutive systems."""
        mock_model = AsyncMock()
        mock_model.ainvoke = AsyncMock(return_value=AIMessage(content="ok"))
        state: dict[str, Any] = {
            "messages": [
                HumanMessage(content="hi"),
                SystemMessage(content="INJECTED-KNOWLEDGE", id="sys-1"),
            ],
        }
        config: RunnableConfig = {
            "configurable": {"chat_model": mock_model, "system_prompt": "BASE"},
        }

        await agent_node(state, config)  # type: ignore[arg-type]

        messages_sent = mock_model.ainvoke.call_args[0][0]
        system_messages = [m for m in messages_sent if isinstance(m, SystemMessage)]
        # exactly one system message, at index 0, carrying base + injected
        assert len(system_messages) == 1
        assert isinstance(messages_sent[0], SystemMessage)
        assert "BASE" in messages_sent[0].content
        assert "INJECTED-KNOWLEDGE" in messages_sent[0].content
        # the human turn survives, after the consolidated system message
        assert any(isinstance(m, HumanMessage) and m.content == "hi" for m in messages_sent)

    @pytest.mark.asyncio
    async def test_injected_system_message_removed_from_history(self) -> None:
        """the consumed SystemMessage is RemoveMessage'd so per-turn context
        does not persist / accumulate across turns."""
        mock_model = AsyncMock()
        mock_model.ainvoke = AsyncMock(return_value=AIMessage(content="ok"))
        state: dict[str, Any] = {
            "messages": [
                HumanMessage(content="hi"),
                SystemMessage(content="INJECTED", id="sys-1"),
            ],
        }
        config: RunnableConfig = {
            "configurable": {"chat_model": mock_model, "system_prompt": "BASE"},
        }

        result = await agent_node(state, config)  # type: ignore[arg-type]

        removals = [m for m in result["messages"] if isinstance(m, RemoveMessage)]
        assert any(m.id == "sys-1" for m in removals)
