"""Behavioral tests for :class:`SummarizationMiddleware`.

Verifies the trigger (message-count window), the ``RemoveMessage(REMOVE_ALL_MESSAGES)``
replacement idiom, the safe cutoff (the preserved window never opens on a
``ToolMessage`` orphaned from its ``AIMessage``), and the synchronous no-op mirror.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, cast

import pytest
from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import AgentState
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, RemoveMessage, ToolMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.runtime import Runtime

from threetears.langgraph.middleware_summarize import SummarizationMiddleware


class _StubSummaryModel:
    """chat model whose ``ainvoke`` returns a fixed summary message."""

    async def ainvoke(self, messages: Any, config: Any = None) -> AIMessage:
        return AIMessage(content="SUMMARY")


def _model() -> BaseChatModel:
    return cast("BaseChatModel", _StubSummaryModel())


def _before(mw: SummarizationMiddleware, messages: list[AnyMessage]) -> dict[str, Any] | None:
    state = cast("AgentState[Any]", {"messages": messages})
    return asyncio.run(mw.abefore_model(state, cast("Runtime[Any]", SimpleNamespace())))


class TestTrigger:
    def test_below_threshold_is_noop(self) -> None:
        mw = SummarizationMiddleware(_model(), trigger_messages=4, keep_messages=2)
        out = _before(mw, [HumanMessage(content="a"), AIMessage(content="b")])
        assert out is None

    def test_above_threshold_summarizes_and_replaces(self) -> None:
        mw = SummarizationMiddleware(_model(), trigger_messages=4, keep_messages=2)
        messages: list[AnyMessage] = [HumanMessage(content=f"m{i}") for i in range(6)]
        out = _before(mw, messages)
        assert out is not None
        new = out["messages"]
        assert isinstance(new[0], RemoveMessage)
        assert new[0].id == REMOVE_ALL_MESSAGES
        assert isinstance(new[1], HumanMessage)
        assert "SUMMARY" in new[1].content
        # the most-recent keep_messages ride through verbatim
        assert [m.content for m in new[2:]] == ["m4", "m5"]


class TestSafeCutoff:
    def test_preserved_window_never_opens_on_toolmessage(self) -> None:
        mw = SummarizationMiddleware(_model(), trigger_messages=4, keep_messages=3)
        # keep=3 -> naive cutoff at index 3, which is a ToolMessage; the cutoff must
        # advance past it so the preserved window opens on the following AIMessage.
        messages: list[AnyMessage] = [
            HumanMessage(content="m0"),
            AIMessage(content="m1"),
            HumanMessage(content="m2"),
            ToolMessage(content="t", tool_call_id="x"),
            AIMessage(content="m4"),
            HumanMessage(content="m5"),
        ]
        out = _before(mw, messages)
        assert out is not None
        new = out["messages"]
        assert not isinstance(new[2], ToolMessage)
        assert new[2].content == "m4"


class _FailingSummaryModel:
    """chat model whose ``ainvoke`` raises ``error``, then answers once ``error`` is cleared."""

    def __init__(self, error: BaseException) -> None:
        self.error: BaseException | None = error

    async def ainvoke(self, messages: Any, config: Any = None) -> AIMessage:
        if self.error is not None:
            raise self.error
        return AIMessage(content="SUMMARY")


class TestFailedSummary:
    """a failed summary call leaves the window as it was; it never writes a stand-in summary."""

    def test_a_model_error_leaves_the_window_unsummarized(self) -> None:
        model = _FailingSummaryModel(ConnectionError("provider down"))
        mw = SummarizationMiddleware(cast("BaseChatModel", model), trigger_messages=4, keep_messages=2)
        messages: list[AnyMessage] = [HumanMessage(content=f"m{i}") for i in range(6)]
        assert _before(mw, messages) is None
        assert [m.content for m in messages] == [f"m{i}" for i in range(6)], "nothing removed, nothing added"

    def test_a_model_timeout_leaves_the_window_unsummarized(self) -> None:
        model = _FailingSummaryModel(TimeoutError("summary call timed out"))
        mw = SummarizationMiddleware(cast("BaseChatModel", model), trigger_messages=4, keep_messages=2)
        messages: list[AnyMessage] = [HumanMessage(content=f"m{i}") for i in range(6)]
        assert _before(mw, messages) is None

    def test_the_next_model_call_retries_the_summary(self) -> None:
        """nothing was rewritten, so the window is still over the trigger and the next call summarizes."""
        model = _FailingSummaryModel(ConnectionError("provider down"))
        mw = SummarizationMiddleware(cast("BaseChatModel", model), trigger_messages=4, keep_messages=2)
        messages: list[AnyMessage] = [HumanMessage(content=f"m{i}") for i in range(6)]
        assert _before(mw, messages) is None
        model.error = None
        out = _before(mw, messages)
        assert out is not None
        assert "SUMMARY" in out["messages"][1].content

    def test_cancellation_propagates(self) -> None:
        model = _FailingSummaryModel(asyncio.CancelledError())
        mw = SummarizationMiddleware(cast("BaseChatModel", model), trigger_messages=4, keep_messages=2)
        messages: list[AnyMessage] = [HumanMessage(content=f"m{i}") for i in range(6)]
        with pytest.raises(asyncio.CancelledError):
            _before(mw, messages)


class TestSyncMirror:
    def test_sync_before_model_is_noop(self) -> None:
        mw = SummarizationMiddleware(_model(), trigger_messages=1, keep_messages=1)
        state = cast(
            "AgentState[Any]",
            {"messages": [HumanMessage(content="a"), HumanMessage(content="b")]},
        )
        out = mw.before_model(state, cast("Runtime[Any]", SimpleNamespace()))
        assert out is None


class TestShape:
    def test_is_agent_middleware(self) -> None:
        m = SummarizationMiddleware(_model())
        assert isinstance(m, AgentMiddleware)
        assert m.name == "SummarizationMiddleware"
        assert hasattr(m, "abefore_model")
        assert hasattr(m, "before_model")
