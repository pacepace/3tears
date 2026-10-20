"""``ToolLoopBudgetMiddleware``: a model that loops on tool calls is stopped, and the turn ends with text.

Most tests run a real ``create_agent`` graph over a scripted chat model, so what is pinned is what a
consumer gets: the turn's messages, and for every model call the tools it was offered, its
``tool_choice`` and the messages it saw. The scripted model answers each call from a function of
(call index, whether the budget told it to answer, messages); a looping model calls a tool every
time it is not told to answer.
"""

from __future__ import annotations

import logging
import operator
from collections.abc import Callable, Sequence
from typing import Annotated, Any, NotRequired, cast

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain.agents.middleware.types import AgentState, ExtendedModelResponse
from langchain.agents.structured_output import ToolStrategy
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphInterrupt, GraphRecursionError
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field

from threetears.langgraph import (
    DEFAULT_MAX_TOOL_CALLS_PER_TURN,
    DEFAULT_TOOL_BUDGET_FALLBACK_TEXT,
    SummarizationMiddleware,
    ToolLoopBudgetMiddleware,
    recursion_limit_for,
)

_LOGGER = "threetears.langgraph.middleware_tool_budget"

#: (call index, told to answer, messages seen) -> the model's response
Script = Callable[[int, bool, list[BaseMessage]], AIMessage]

#: How each of the budget's instructions to answer begins (the defaults, and the tuned test's "FORCE").
_TOLD = (
    "You can make no more tool calls",
    "Your recent tool calls failed",
    "Your last reply was empty",
    "You thought it through",
    "FORCE",
)


@tool
def read_note(topic: str) -> str:
    """Read the note about a topic."""
    return f"the note about {topic}"


class Answer(BaseModel):
    """The structured answer."""

    text: str


def _told(messages: Sequence[BaseMessage]) -> bool:
    last = messages[-1] if messages else None
    return isinstance(last, (HumanMessage, SystemMessage)) and str(last.content).startswith(_TOLD)


class _ScriptedModel(BaseChatModel):
    """Answers each call from ``script`` and records what every call was offered and shown."""

    script: Any
    offered: list[str] = Field(default_factory=list)
    tool_choice: Any = None
    usage: dict[str, int] | None = None
    calls: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Sequence[Any], *, tool_choice: Any = None, **kwargs: Any) -> Any:
        names = [t["name"] if isinstance(t, dict) else t.name for t in tools]
        # A shallow copy shares ``calls``, so the bound copy records into the same list.
        return self.model_copy(update={"offered": names, "tool_choice": tool_choice})

    def _respond(self, messages: list[BaseMessage]) -> ChatResult:
        index = len(self.calls)
        told = _told(messages)
        self.calls.append(
            {"offered": list(self.offered), "tool_choice": self.tool_choice, "told": told, "messages": list(messages)}
        )
        reply = self.script(index, told, messages)
        if self.usage is not None:
            reply = reply.model_copy(
                update={
                    "usage_metadata": dict(self.usage),
                    "response_metadata": {"model_name": "scripted", "call": index},
                }
            )
        return ChatResult(generations=[ChatGeneration(message=reply)])

    def _generate(self, messages: list[BaseMessage], *args: Any, **kwargs: Any) -> ChatResult:
        return self._respond(messages)

    async def _agenerate(self, messages: list[BaseMessage], *args: Any, **kwargs: Any) -> ChatResult:
        return self._respond(messages)


def _call(index: int, name: str = "read_note", n: int = 1) -> AIMessage:
    """A response that makes ``n`` tool calls and writes no text."""
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": {"topic": f"t{index}-{k}"}, "id": f"c{index}-{k}"} for k in range(n)],
    )


def _looping(answer: str = "here is what I found") -> Script:
    """Calls a tool every time, until the budget tells it to answer."""

    def script(index: int, told: bool, messages: list[BaseMessage]) -> AIMessage:
        return AIMessage(content=answer) if told else _call(index)

    return script


def _scripted(script: Script, **fields: Any) -> _ScriptedModel:
    return _ScriptedModel(script=script, **fields)


async def _run(
    model: _ScriptedModel,
    budget: ToolLoopBudgetMiddleware | None,
    *,
    recursion_limit: int | None = None,
    extra: Sequence[AgentMiddleware] = (),
    **agent_kwargs: Any,
) -> dict[str, Any]:
    middleware: list[AgentMiddleware] = ([budget] if budget else []) + list(extra)
    agent = create_agent(model=model, tools=[read_note], middleware=middleware, **agent_kwargs)
    limit = recursion_limit if recursion_limit is not None else recursion_limit_for(middleware=middleware)
    return cast(
        "dict[str, Any]",
        await agent.ainvoke({"messages": [HumanMessage(content="Say hello.")]}, {"recursion_limit": limit}),
    )


async def _turn(
    model: _ScriptedModel,
    budget: ToolLoopBudgetMiddleware | None,
    *,
    recursion_limit: int | None = None,
    extra: Sequence[AgentMiddleware] = (),
) -> list[BaseMessage]:
    state = await _run(model, budget, recursion_limit=recursion_limit, extra=extra)
    return list(state["messages"])


def _tool_results(messages: Sequence[BaseMessage]) -> list[ToolMessage]:
    return [m for m in messages if isinstance(m, ToolMessage)]


def _last_seen(call: dict[str, Any]) -> BaseMessage:
    return cast("list[BaseMessage]", call["messages"])[-1]


def _no_orphaned_tool_results(messages: Sequence[BaseMessage]) -> bool:
    called = {c["id"] for m in messages if isinstance(m, AIMessage) for c in m.tool_calls}
    return all(m.tool_call_id in called for m in _tool_results(messages))


# --- the loop is stopped and the turn ends with text -------------------------------------------------


async def test_a_looping_model_is_stopped_by_the_budget_and_the_turn_ends_with_its_answer() -> None:
    model = _scripted(_looping())
    messages = await _turn(model, ToolLoopBudgetMiddleware(max_tool_calls=5))

    assert len(_tool_results(messages)) == 5
    last = messages[-1]
    assert isinstance(last, AIMessage) and last.content == "here is what I found" and not last.tool_calls
    assert [c["told"] for c in model.calls] == [False] * 5 + [True]


async def test_without_the_budget_the_same_loop_ends_on_the_recursion_error() -> None:
    with pytest.raises(GraphRecursionError):
        await _turn(_scripted(_looping()), None, recursion_limit=25)


async def test_at_the_default_cap_the_default_recursion_limit_is_not_enough_and_the_helper_is() -> None:
    """The scriob failure: LangGraph's default 25 ends a 16-call budget's loop before the budget does."""
    with pytest.raises(GraphRecursionError):
        await _turn(_scripted(_looping()), ToolLoopBudgetMiddleware(), recursion_limit=25)

    messages = await _turn(_scripted(_looping()), ToolLoopBudgetMiddleware())
    assert len(_tool_results(messages)) == DEFAULT_MAX_TOOL_CALLS_PER_TURN
    assert messages[-1].content == "here is what I found"


async def test_the_forced_pass_keeps_the_tools_declared_and_says_to_answer_from_what_it_has() -> None:
    """Anthropic refuses tool-use history with no tools declared, so the forced pass keeps them."""
    model = _scripted(_looping())
    await _turn(model, ToolLoopBudgetMiddleware(max_tool_calls=2))

    forced = model.calls[-1]
    assert forced["offered"] == ["read_note"]
    assert forced["tool_choice"] is None
    instruction = _last_seen(forced)
    assert isinstance(instruction, HumanMessage)
    assert "no more tool calls" in str(instruction.content)
    assert "information you already have" in str(instruction.content)


async def test_the_forced_tool_choice_reaches_the_forced_pass_and_only_it() -> None:
    model = _scripted(_looping())
    await _turn(model, ToolLoopBudgetMiddleware(max_tool_calls=2, forced_tool_choice="none"))
    assert [(c["offered"], c["tool_choice"]) for c in model.calls] == [
        (["read_note"], None),
        (["read_note"], None),
        (["read_note"], "none"),
    ]


async def test_the_forced_reply_is_rebuilt_as_text_only_with_its_metering_kept() -> None:
    """A tool call and a ``tool_use`` block in the forced reply are dropped; usage and metadata stay."""

    def script(index: int, told: bool, messages: list[BaseMessage]) -> AIMessage:
        if not told:
            return _call(index)
        return AIMessage(
            content=[
                {"type": "text", "text": "final words"},
                {"type": "tool_use", "id": "rogue", "name": "read_note", "input": {}},
            ],
            tool_calls=[{"name": "read_note", "args": {}, "id": "rogue"}],
        )

    usage = {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12}
    messages = await _turn(_scripted(script, usage=usage), ToolLoopBudgetMiddleware(max_tool_calls=2))
    last = messages[-1]
    assert isinstance(last, AIMessage) and last.tool_calls == [] and last.invalid_tool_calls == []
    assert last.content == [{"type": "text", "text": "final words"}]
    assert last.usage_metadata == usage
    assert last.response_metadata["model_name"] == "scripted" and last.response_metadata["call"] == 2
    assert len(_tool_results(messages)) == 2
    assert _no_orphaned_tool_results(messages)


async def test_a_model_that_keeps_calling_tools_when_told_to_answer_still_ends_with_text() -> None:
    """The forced pass, its retry: every reply a tool call. The calls are dropped; the fallback answers."""
    model = _scripted(lambda index, told, messages: _call(index))
    messages = await _turn(model, ToolLoopBudgetMiddleware(max_tool_calls=2))
    assert messages[-1].content == DEFAULT_TOOL_BUDGET_FALLBACK_TEXT
    assert [c["told"] for c in model.calls] == [False, False, True, True]
    assert len(_tool_results(messages)) == 2
    assert _no_orphaned_tool_results(messages)


async def test_a_round_that_would_pass_the_cap_is_discarded_and_the_answer_forced() -> None:
    """Three parallel calls a round under a cap of ten: 3, 6, 9, then a round of 3 would make 12."""

    def script(index: int, told: bool, messages: list[BaseMessage]) -> AIMessage:
        return AIMessage(content="answered") if told else _call(index, n=3)

    usage = {"input_tokens": 10, "output_tokens": 2, "total_tokens": 12}
    model = _scripted(script, usage=usage)
    messages = await _turn(model, ToolLoopBudgetMiddleware(max_tool_calls=10))
    assert len(_tool_results(messages)) == 9
    last = messages[-1]
    assert last.content == "answered"
    # the fourth call's three tool calls were discarded; the fifth call is the forced pass
    assert [c["told"] for c in model.calls] == [False, False, False, False, True]
    # the step's answer is billed for the discarded call too
    assert isinstance(last, AIMessage) and last.usage_metadata is not None
    assert last.usage_metadata["total_tokens"] == 24


async def test_a_model_that_answers_on_its_own_is_untouched() -> None:
    def script(index: int, told: bool, messages: list[BaseMessage]) -> AIMessage:
        return _call(index) if index < 2 else AIMessage(content="done")

    model = _scripted(script)
    messages = await _turn(model, ToolLoopBudgetMiddleware(max_tool_calls=5))
    assert messages[-1].content == "done"
    assert not any(c["told"] for c in model.calls)
    # no nudge below warn: after the first call, each call's last message is a tool result
    assert all(isinstance(_last_seen(c), ToolMessage) for c in model.calls[1:])


# --- the turn is the invocation, not the last user message ------------------------------------------


async def test_the_summarizer_rewriting_history_mid_turn_does_not_reset_the_count() -> None:
    """Both platform middlewares at their defaults: the summarizer replaces the history with a summary
    ``HumanMessage`` and a tail every time the window passes 20 messages. The budget still forces the
    answer at its cap and the turn ends with text, inside the helper's limit."""
    model = _scripted(_looping())
    summarizer = _scripted(lambda index, told, messages: AIMessage(content="SUMMARY"))
    messages = await _turn(model, ToolLoopBudgetMiddleware(), extra=[SummarizationMiddleware(summarizer)])

    assert summarizer.calls, "the summarizer rewrote the history mid-turn"
    assert any(isinstance(m, HumanMessage) and "SUMMARY" in str(m.content) for m in messages)
    assert [c["told"] for c in model.calls] == [False] * DEFAULT_MAX_TOOL_CALLS_PER_TURN + [True]
    assert messages[-1].content == "here is what I found"


class _Interjector(AgentMiddleware):
    """Appends a ``HumanMessage`` to state before every model call, as a reminder middleware might."""

    def before_model(self, state: Any, runtime: Any) -> dict[str, Any]:
        return {"messages": [HumanMessage(content="(reminder)")]}


async def test_any_middleware_writing_a_user_message_mid_turn_does_not_reset_the_count() -> None:
    model = _scripted(_looping())
    messages = await _turn(model, ToolLoopBudgetMiddleware(max_tool_calls=4), extra=[_Interjector()])
    assert [c["told"] for c in model.calls] == [False] * 4 + [True]
    assert messages[-1].content == "here is what I found"


async def test_a_new_invocation_is_a_new_turn() -> None:
    model = _scripted(_looping())
    budget = ToolLoopBudgetMiddleware(max_tool_calls=3)
    agent = create_agent(model=model, tools=[read_note], middleware=[budget], checkpointer=InMemorySaver())
    config: Any = {"configurable": {"thread_id": "t"}, "recursion_limit": recursion_limit_for(middleware=[budget])}
    await agent.ainvoke({"messages": [HumanMessage(content="first")]}, config)
    await agent.ainvoke({"messages": [HumanMessage(content="second")]}, config)
    assert [c["told"] for c in model.calls] == ([False] * 3 + [True]) * 2


@tool
def ask_pace(topic: str) -> str:
    """Ask Pace and wait for his answer."""
    return str(interrupt(topic))


async def test_resuming_an_interrupted_turn_continues_its_count() -> None:
    """Two calls, a third that waits on a person, the resume: the cap of four counts all of them."""

    def script(index: int, told: bool, messages: list[BaseMessage]) -> AIMessage:
        if told:
            return AIMessage(content="done")
        return _call(index, name="ask_pace" if index == 2 else "read_note")

    model = _scripted(script)
    budget = ToolLoopBudgetMiddleware(max_tool_calls=4)
    agent = create_agent(model=model, tools=[read_note, ask_pace], middleware=[budget], checkpointer=InMemorySaver())
    config: Any = {"configurable": {"thread_id": "t"}, "recursion_limit": recursion_limit_for(middleware=[budget])}
    paused = await agent.ainvoke({"messages": [HumanMessage(content="go")]}, config)
    assert "__interrupt__" in paused
    state = await agent.ainvoke(Command(resume="yes"), config)
    assert [c["told"] for c in model.calls] == [False] * 4 + [True]
    assert state["messages"][-1].content == "done"


# --- tiers -------------------------------------------------------------------------------------------


async def test_each_call_tier_nudges_at_its_count() -> None:
    """Cap 10: warn at 70% (7 calls), urgent at 85% (9 calls), force at the cap (10)."""
    model = _scripted(_looping())
    await _turn(model, ToolLoopBudgetMiddleware(max_tool_calls=10))

    def nudge(index: int) -> str | None:
        last = _last_seen(model.calls[index])
        return str(last.content) if isinstance(last, HumanMessage) and index > 0 else None

    assert [nudge(i) for i in range(7)] == [None] * 7
    for i in (7, 8):
        assert (n := nudge(i)) is not None and n.startswith("Start wrapping up") and f"{i} of the 10" in n
    urgent = nudge(9)
    assert urgent is not None and urgent.startswith("Stop calling tools") and "9 of the 10" in urgent
    assert not model.calls[9]["told"]
    assert model.calls[10]["told"]
    assert len(model.calls) == 11


async def test_the_window_tiers_nudge_and_force_by_the_prompts_share_of_the_window() -> None:
    """Ten tokens a message in a 200-token window: after round k the prompt is 5 + 10k percent."""
    model = _scripted(_looping())
    budget = ToolLoopBudgetMiddleware(
        max_tool_calls=100, context_window_tokens=200, count_tokens=lambda msgs: 10 * len(msgs)
    )
    messages = await _turn(model, budget)

    kinds = []
    for i, call in enumerate(model.calls):
        last = _last_seen(call)
        if call["told"]:
            kinds.append("force")
        elif i > 0 and isinstance(last, HumanMessage):
            kinds.append("urgent" if str(last.content).startswith("Stop calling tools") else "warn")
        else:
            kinds.append(None)
    # rounds 0..6 under 70%, round 7 at 75% (warn), 8 at 85% (urgent), 9 at 95% (force)
    assert kinds == [None] * 7 + ["warn", "urgent", "force"]
    assert messages[-1].content == "here is what I found"


async def test_the_window_comes_from_the_models_profile_when_not_configured() -> None:
    model = _scripted(_looping())
    model.profile = {"max_input_tokens": 200}
    budget = ToolLoopBudgetMiddleware(max_tool_calls=100, count_tokens=lambda msgs: 10 * len(msgs))
    await _turn(model, budget)
    assert len(model.calls) == 10
    assert model.calls[-1]["told"]


async def test_thresholds_and_texts_are_the_consumers_to_tune() -> None:
    model = _scripted(_looping())
    budget = ToolLoopBudgetMiddleware(
        max_tool_calls=4,
        warn_pct=25,
        urgent_pct=50,
        force_pct=90,
        warn_text="WARN {tool_calls}/{max_tool_calls}",
        urgent_text="URGENT {tool_calls}/{max_tool_calls}",
        force_text="FORCE",
        nudge_role="system",
    )
    await _turn(model, budget)
    seen = [_last_seen(c) for c in model.calls]
    assert [type(m).__name__ for m in seen[1:]] == ["SystemMessage"] * 4
    assert [m.content for m in seen[1:]] == ["WARN 1/4", "URGENT 2/4", "URGENT 3/4", "FORCE"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_tool_calls": 0},
        {"warn_pct": 0},
        {"warn_pct": 90, "urgent_pct": 80},
        {"urgent_pct": 95, "force_pct": 90},
        {"force_pct": 101},
        {"runaway_rounds": 0},
        {"context_window_tokens": 0},
        {"fallback_text": "  "},
    ],
)
def test_nonsense_thresholds_are_refused(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        ToolLoopBudgetMiddleware(**kwargs)


# --- the empty-answer guard --------------------------------------------------------------------------


async def test_an_empty_then_reasoning_only_forced_answer_is_retried_then_falls_back() -> None:
    def script(index: int, told: bool, messages: list[BaseMessage]) -> AIMessage:
        if not told:
            return _call(index)
        if index == 2:
            return AIMessage(content="")
        return AIMessage(content=[{"type": "reasoning", "reasoning": "maybe I should look again"}])

    model = _scripted(script)
    messages = await _turn(model, ToolLoopBudgetMiddleware(max_tool_calls=2))

    assert [c["told"] for c in model.calls] == [False, False, True, True]
    retry = _last_seen(model.calls[3])
    assert "Your last reply was empty" in str(retry.content)
    last = messages[-1]
    assert isinstance(last, AIMessage) and last.content == DEFAULT_TOOL_BUDGET_FALLBACK_TEXT
    assert not last.tool_calls


async def test_the_fallback_reaches_the_stream_and_the_history() -> None:
    """The fallback runs the model chain as a stand-in model, so a token stream carries it."""
    model = _scripted(lambda index, told, messages: AIMessage(content="") if told else _call(index))
    budget = ToolLoopBudgetMiddleware(max_tool_calls=1)
    agent = create_agent(model=model, tools=[read_note], middleware=[budget])
    streamed: list[str] = []
    final: dict[str, Any] = {}
    async for event in agent.astream_events(
        {"messages": [HumanMessage(content="Say hello.")]},
        {"recursion_limit": recursion_limit_for(middleware=[budget])},
        version="v2",
    ):
        if event["event"] == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            assert isinstance(chunk, AIMessageChunk)
            streamed.append(str(chunk.content))
        elif event["event"] == "on_chain_end" and isinstance(event["data"].get("output"), dict):
            final.update(event["data"]["output"])
    assert "".join(streamed) == DEFAULT_TOOL_BUDGET_FALLBACK_TEXT
    assert final["messages"][-1].content == DEFAULT_TOOL_BUDGET_FALLBACK_TEXT


async def test_a_reasoning_only_answer_is_retried_with_its_reasoning_and_the_retry_is_the_answer() -> None:
    def script(index: int, told: bool, messages: list[BaseMessage]) -> AIMessage:
        if index == 0:
            return _call(index)
        if index == 1:
            return AIMessage(content="", additional_kwargs={"reasoning_content": "the note says hello"})
        return AIMessage(content="Hello!")

    model = _scripted(script)
    messages = await _turn(model, ToolLoopBudgetMiddleware(max_tool_calls=1))

    retry = str(_last_seen(model.calls[2]).content)
    assert "the note says hello" in retry and "wrote no reply" in retry
    assert messages[-1].content == "Hello!"
    # the empty answer is not in the conversation; only the retry's answer is
    assert [m.content for m in messages if isinstance(m, AIMessage) and not m.tool_calls] == ["Hello!"]


async def test_an_empty_natural_answer_is_guarded_too() -> None:
    """No budget pressure at all: the model just answers with nothing."""

    def script(index: int, told: bool, messages: list[BaseMessage]) -> AIMessage:
        return AIMessage(content="") if index == 0 else AIMessage(content="Hello there.")

    model = _scripted(script)
    messages = await _turn(model, ToolLoopBudgetMiddleware())
    assert messages[-1].content == "Hello there."
    assert [c["told"] for c in model.calls] == [False, True]


async def test_the_fallback_text_is_configurable() -> None:
    model = _scripted(lambda index, told, messages: AIMessage(content=""))
    messages = await _turn(model, ToolLoopBudgetMiddleware(fallback_text="Sorry -- nothing to say."))
    assert messages[-1].content == "Sorry -- nothing to say."


async def test_a_failing_retry_still_ends_the_turn_with_the_fallback() -> None:
    def script(index: int, told: bool, messages: list[BaseMessage]) -> AIMessage:
        if index == 0:
            return AIMessage(content="")
        raise RuntimeError("provider went away")

    messages = await _turn(_scripted(script), ToolLoopBudgetMiddleware())
    assert messages[-1].content == DEFAULT_TOOL_BUDGET_FALLBACK_TEXT


def _request(state: dict[str, Any] | None = None) -> ModelRequest:
    history: list[BaseMessage] = [HumanMessage(content="hi")]
    return ModelRequest(
        model=cast("BaseChatModel", _scripted(_looping())),
        messages=history,
        tools=[read_note],
        system_message=SystemMessage(content="sys"),
        state=cast("Any", state or {"messages": history}),
    )


def test_an_interrupt_during_the_retry_is_never_swallowed_into_the_fallback() -> None:
    calls: list[ModelRequest] = []

    def handler(request: ModelRequest) -> ModelResponse:
        calls.append(request)
        if len(calls) == 1:
            return ModelResponse(result=[AIMessage(content="")])
        raise GraphInterrupt(())

    with pytest.raises(GraphInterrupt):
        ToolLoopBudgetMiddleware().wrap_model_call(_request(), handler)
    assert len(calls) == 2


# --- structured output -------------------------------------------------------------------------------


def _answering_structurally(*, valid: bool = True, text: str = "") -> Script:
    def script(index: int, told: bool, messages: list[BaseMessage]) -> AIMessage:
        if not told:
            return _call(index)
        args: dict[str, Any] = {"text": "hello"} if valid else {"wrong": 1}
        return AIMessage(content=text, tool_calls=[{"name": "Answer", "args": args, "id": f"s{index}"}])

    return script


async def test_a_structured_answer_in_the_forced_pass_is_kept() -> None:
    model = _scripted(_answering_structurally())
    state = await _run(model, ToolLoopBudgetMiddleware(max_tool_calls=2), response_format=ToolStrategy(Answer))
    assert state["structured_response"] == Answer(text="hello")
    last_ai = next(m for m in reversed(state["messages"]) if isinstance(m, AIMessage))
    assert [c["name"] for c in last_ai.tool_calls] == ["Answer"]
    assert len(model.calls) == 3


def test_a_structured_answer_is_passed_through_even_past_the_cap() -> None:
    """A structured answer ends the turn; it is never counted, discarded or forced."""
    calls: list[ModelRequest] = []
    structured = ModelResponse(
        result=[AIMessage(content="", tool_calls=[{"name": "Answer", "args": {"text": "hi"}, "id": "s"}] * 3)],
        structured_response=Answer(text="hi"),
    )

    def handler(request: ModelRequest) -> ModelResponse:
        calls.append(request)
        return structured

    state = {"messages": [HumanMessage(content="hi")], "tool_loop_budget": {"tool_calls": 15, "rounds": 15}}
    result = ToolLoopBudgetMiddleware().wrap_model_call(_request(state), handler)
    assert len(calls) == 1
    assert result.model_response is structured


async def test_a_failed_structured_parse_in_the_forced_pass_leaves_no_orphaned_tool_result() -> None:
    """A parse error comes back as the call plus an error ``ToolMessage``; dropping the call drops both."""
    model = _scripted(_answering_structurally(valid=False, text="here is my answer"))
    state = await _run(model, ToolLoopBudgetMiddleware(max_tool_calls=2), response_format=ToolStrategy(Answer))
    messages = state["messages"]
    assert messages[-1].content == "here is my answer"
    assert _no_orphaned_tool_results(messages)
    assert "structured_response" not in state


# --- another middleware's state command survives ------------------------------------------------------


class _MarkState(AgentState[Any]):
    marks: NotRequired[Annotated[list[str], operator.add]]


class _Marker(AgentMiddleware):
    """An inner middleware whose every model call returns a state command naming the call."""

    state_schema = _MarkState

    def __init__(self) -> None:
        super().__init__()
        self.count = 0

    async def awrap_model_call(self, request: ModelRequest, handler: Any) -> ExtendedModelResponse:
        response = await handler(request)
        self.count += 1
        return ExtendedModelResponse(model_response=response, command=Command(update={"marks": [f"m{self.count}"]}))


async def test_an_inner_middlewares_command_survives_a_discarded_round_and_the_fallback() -> None:
    """Each model step keeps the command of its last inner call: the forced pass's chain, not the
    discarded round; the fallback, not the empty answers before it."""

    def script(index: int, told: bool, messages: list[BaseMessage]) -> AIMessage:
        return AIMessage(content="") if told else _call(index, n=2)

    marker = _Marker()
    state = await _run(_scripted(script), ToolLoopBudgetMiddleware(max_tool_calls=3), extra=[marker])
    # step 1: m1 (two calls). step 2: m2 discarded (would make 4), m3 forced empty, m4 retry empty,
    # m5 the fallback stand-in.
    assert state["marks"] == ["m1", "m5"]
    assert state["messages"][-1].content == DEFAULT_TOOL_BUDGET_FALLBACK_TEXT


# --- the runaway guard -------------------------------------------------------------------------------


async def test_rounds_with_no_text_whose_calls_all_fail_force_the_answer_early() -> None:
    """Calls to a tool that does not exist come back as error results; three such rounds is a runaway."""

    def script(index: int, told: bool, messages: list[BaseMessage]) -> AIMessage:
        return AIMessage(content="I could not do that.") if told else _call(index, name="no_such_tool")

    model = _scripted(script)
    messages = await _turn(model, ToolLoopBudgetMiddleware(max_tool_calls=16, runaway_rounds=3))

    assert [c["told"] for c in model.calls] == [False, False, False, True]
    assert all(m.status == "error" for m in _tool_results(messages))
    assert "Your recent tool calls failed" in str(_last_seen(model.calls[3]).content)
    assert messages[-1].content == "I could not do that."


async def test_a_round_that_writes_text_or_succeeds_resets_the_runaway_count() -> None:
    def script(index: int, told: bool, messages: list[BaseMessage]) -> AIMessage:
        if index in (0, 1, 3, 4):
            return _call(index, name="no_such_tool")
        if index == 2:
            return _call(index)  # a successful round between the failing ones
        return AIMessage(content="done")

    model = _scripted(script)
    messages = await _turn(model, ToolLoopBudgetMiddleware(max_tool_calls=16, runaway_rounds=3))
    assert messages[-1].content == "done"
    assert not any(c["told"] for c in model.calls)


# --- the recursion limit -----------------------------------------------------------------------------


class _BeforeModel(AgentMiddleware):
    """A hook middleware: ``create_agent`` gives it a node on every round."""

    def before_model(self, state: Any, runtime: Any) -> None:
        return None


class _AfterModel(AgentMiddleware):
    """An ``after_model`` hook, as scriob's write-serializing middleware has: a node every round."""

    def after_model(self, state: Any, runtime: Any) -> None:
        return None


class _SecondAfterModel(_AfterModel):
    """Another ``after_model`` hook (an agent takes one instance of each middleware class)."""


class _AfterAgent(AgentMiddleware):
    """A once-per-run hook middleware."""

    def after_agent(self, state: Any, runtime: Any) -> None:
        return None


def test_the_helper_takes_the_cap_from_the_budget_and_counts_every_hook_node() -> None:
    budget = ToolLoopBudgetMiddleware(max_tool_calls=4)
    # the input step, five model calls, four tool rounds, the budget's own before_agent
    plain = recursion_limit_for(middleware=[budget], margin=0)
    assert plain == 1 + 5 + 4 + 1
    assert recursion_limit_for(4, middleware=[budget], margin=0) == plain
    assert recursion_limit_for(middleware=[budget, _BeforeModel()], margin=0) == plain + 5
    assert recursion_limit_for(middleware=[budget, _AfterModel()], margin=0) == plain + 5
    assert recursion_limit_for(middleware=[budget, _BeforeModel(), _AfterModel()], margin=0) == plain + 10
    assert recursion_limit_for(middleware=[budget, _AfterAgent()], margin=0) == plain + 1
    # with no budget in the list, the cap is the argument's (default 16)
    assert recursion_limit_for(4, margin=0) == plain - 1
    assert recursion_limit_for(margin=0) == recursion_limit_for(DEFAULT_MAX_TOOL_CALLS_PER_TURN, margin=0)


@pytest.mark.parametrize(
    ("cap", "middleware"),
    [
        (0, []),
        (5, [ToolLoopBudgetMiddleware(max_tool_calls=4)]),
        (None, [ToolLoopBudgetMiddleware(), ToolLoopBudgetMiddleware()]),
    ],
)
def test_the_helper_refuses_a_cap_it_cannot_trust(cap: int | None, middleware: list[AgentMiddleware]) -> None:
    with pytest.raises(ValueError):
        recursion_limit_for(cap, middleware=middleware)


@pytest.mark.parametrize("cap", [1, 3, 6])
@pytest.mark.parametrize(
    "extra",
    [(), (_BeforeModel(),), (_AfterModel(),), (_BeforeModel(), _AfterModel(), _SecondAfterModel(), _AfterAgent())],
)
async def test_the_helpers_limit_exceeds_the_worst_turn_the_budget_allows(
    cap: int, extra: tuple[AgentMiddleware, ...]
) -> None:
    """Worst case: every round uses the cap one call at a time, then the forced answer, its retry and
    the fallback. It runs at the helper's limit with no margin at all; the budget, not the recursion
    error, ends it. One step less is not enough, so the margin-free limit is the worst turn exactly.
    """
    model = _scripted(lambda index, told, messages: AIMessage(content="") if told else _call(index))
    budget = ToolLoopBudgetMiddleware(max_tool_calls=cap)
    limit = recursion_limit_for(middleware=[budget, *extra], margin=0)
    messages = await _turn(model, budget, recursion_limit=limit, extra=extra)
    assert messages[-1].content == DEFAULT_TOOL_BUDGET_FALLBACK_TEXT
    assert len(_tool_results(messages)) == cap

    with pytest.raises(GraphRecursionError):
        await _turn(model, budget, recursion_limit=limit - 1, extra=extra)


# --- observability -----------------------------------------------------------------------------------


async def test_tier_transitions_and_the_forced_answer_are_logged_with_the_count_and_no_prompt(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        await _turn(_scripted(_looping("a secret answer")), ToolLoopBudgetMiddleware(max_tool_calls=10))

    records = [r for r in caplog.records if r.name == _LOGGER]
    tiers = [r.__dict__["extra_data"] for r in records if r.getMessage() == "tool-loop budget tier reached"]
    # once per transition, not once per call in a tier
    assert [(t["tier"], t["tool_calls"]) for t in tiers] == [("warn", 7), ("urgent", 9), ("force", 10)]
    [forced] = [r.__dict__["extra_data"] for r in records if r.getMessage() == "tool-loop budget forced an answer"]
    assert forced == {"cause": "tool_calls", "tool_calls": 10, "max_tool_calls": 10}
    for record in records:
        rendered = record.getMessage() + repr(record.__dict__.get("extra_data"))
        assert "Say hello" not in rendered and "note about" not in rendered and "secret" not in rendered


# --- the sync seam -----------------------------------------------------------------------------------


def test_the_sync_seam_runs_the_same_plan() -> None:
    model = _scripted(_looping())
    budget = ToolLoopBudgetMiddleware(max_tool_calls=3)
    agent = create_agent(model=model, tools=[read_note], middleware=[budget])
    state = agent.invoke(
        {"messages": [HumanMessage(content="Say hello.")]},
        {"recursion_limit": recursion_limit_for(middleware=[budget])},
    )
    assert [c["told"] for c in model.calls] == [False] * 3 + [True]
    assert state["messages"][-1].content == "here is what I found"


# --- the driver: which failures it survives, and what it logs of them ---------------------------------


async def test_a_failing_main_call_surfaces_its_own_exception() -> None:
    def script(index: int, told: bool, messages: list[BaseMessage]) -> AIMessage:
        raise RuntimeError("provider went away")

    with pytest.raises(RuntimeError, match="provider went away"):
        await _turn(_scripted(script), ToolLoopBudgetMiddleware())


async def test_a_failing_forced_call_surfaces_its_own_exception() -> None:
    def script(index: int, told: bool, messages: list[BaseMessage]) -> AIMessage:
        if told:
            raise RuntimeError("provider went away while answering")
        return _call(index)

    with pytest.raises(RuntimeError, match="provider went away while answering"):
        await _turn(_scripted(script), ToolLoopBudgetMiddleware(max_tool_calls=2))


class _RefusesTheStandIn(AgentMiddleware):
    """An inner middleware that fails the fallback's stand-in model call, and passes every other call."""

    async def awrap_model_call(self, request: ModelRequest, handler: Any) -> Any:
        # the agent's own model is the scripted one; the fallback runs on the budget's stand-in
        if not isinstance(request.model, _ScriptedModel):
            raise _ProviderError("SECRET-PROMPT-TEXT echoed by the stand-in's failure")
        return await handler(request)


class _ProviderError(Exception):
    """A provider's error: it types its HTTP status, and its message may echo the prompt."""

    status_code = 429


def _rendered(record: logging.LogRecord) -> str:
    """Everything a log handler would write for ``record``: message, traceback and extra fields."""
    return logging.Formatter().format(record) + repr(record.__dict__.get("extra_data"))


async def test_a_failing_stand_in_still_answers_with_the_fallback_text(caplog: pytest.LogCaptureFixture) -> None:
    model = _scripted(lambda index, told, messages: AIMessage(content="") if told else _call(index))
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        messages = await _turn(model, ToolLoopBudgetMiddleware(max_tool_calls=1), extra=[_RefusesTheStandIn()])

    assert messages[-1].content == DEFAULT_TOOL_BUDGET_FALLBACK_TEXT
    [failed] = [r for r in caplog.records if r.getMessage().startswith("tool-loop budget fallback call failed")]
    assert failed.__dict__["extra_data"]["call"] == "fallback"
    assert "SECRET-PROMPT-TEXT" not in _rendered(failed)


async def test_a_failed_retry_logs_its_counts_class_and_status_and_none_of_the_providers_text(
    caplog: pytest.LogCaptureFixture,
) -> None:
    def script(index: int, told: bool, messages: list[BaseMessage]) -> AIMessage:
        if not told:
            return _call(index)
        if index == 2:
            return AIMessage(content="")
        raise _ProviderError("the request was: SECRET-PROMPT-TEXT Say hello.")

    with caplog.at_level(logging.WARNING):
        messages = await _turn(_scripted(script), ToolLoopBudgetMiddleware(max_tool_calls=2))

    assert messages[-1].content == DEFAULT_TOOL_BUDGET_FALLBACK_TEXT
    [failed] = [r for r in caplog.records if r.getMessage().startswith("tool-loop budget retry failed")]
    data = failed.__dict__["extra_data"]
    assert data["call"] == "retry"
    assert (data["tool_calls"], data["max_tool_calls"]) == (2, 2)
    assert (data["error_type"], data["status_code"]) == ("_ProviderError", 429)
    for record in caplog.records:
        assert "SECRET-PROMPT-TEXT" not in _rendered(record)


async def test_an_interrupt_during_the_retry_is_never_swallowed_on_the_async_seam() -> None:
    calls: list[ModelRequest] = []

    async def handler(request: ModelRequest) -> ModelResponse:
        calls.append(request)
        if len(calls) == 1:
            return ModelResponse(result=[AIMessage(content="")])
        raise GraphInterrupt(())

    with pytest.raises(GraphInterrupt):
        await ToolLoopBudgetMiddleware().awrap_model_call(_request(), handler)
    assert len(calls) == 2
